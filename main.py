#!/usr/bin/env python
"""
main.py – GPTで台本（伸びる構成）→ OpenAI TTS → 「lines.json & full.mp3」生成 →
          chunk_builder.py で動画生成 → upload_youtube.py でアップロード。
          combos.yaml の各エントリを順に処理して、複数動画をアップロード。

追加点:
- CONTENT_MODE（dialogue/howto/listicle/wisdom/fact/qa）で“伸びる構成”に最適化
- topic="AUTO" で当日トピックを自動選択（pick_by_content_type）
- seed hook を強化（_make_seed_phrase）
- 行ごとの TTS スタイル（energetic/calm/serious/neutral）
- 行間に短い無音ギャップ（聴感テンポ改善）
- タイトル/タグを中立化＋学習語に寄せてスコアリング
- 🔤（追加）日本語タイトル時は音声言語に応じて「◯◯語会話」を自然に付与
"""

import argparse, logging, re, json, subprocess, os
from datetime import datetime
from pathlib import Path
from shutil import rmtree

import yaml
from pydub import AudioSegment
from openai import OpenAI

from config         import BASE, OUTPUT, TEMP
from dialogue       import make_dialogue
from translate      import translate
from tts_openai     import speak
from audio_fx       import enhance
from bg_image       import fetch as fetch_bg
from thumbnail      import make_thumbnail
from upload_youtube import upload
from topic_picker   import pick_by_content_type

GPT = OpenAI()
MAX_SHORTS_SEC   = 59.0
CONTENT_MODE     = os.environ.get("CONTENT_MODE", "dialogue")  # dialogue/howto/listicle/wisdom/fact/qa

# ───────────────────────────────────────────────
# combos.yaml 読み込み
# ───────────────────────────────────────────────
with open(BASE / "combos.yaml", encoding="utf-8") as f:
    COMBOS = yaml.safe_load(f)["combos"]

def reset_temp():
    if TEMP.exists():
        rmtree(TEMP)
    TEMP.mkdir(exist_ok=True)

def sanitize_title(raw: str) -> str:
    title = re.sub(r"^\s*(?:\d+\s*[.)]|[-•・])\s*", "", raw)
    title = re.sub(r"[\s\u3000]+", " ", title).strip()
    return title[:97] + "…" if len(title) > 100 else title or "Auto Video"

# 中立キーワード（特定言語/国に依存しない）
TOP_KEYWORDS = ["ホテル", "空港", "レストラン", "自己紹介", "予約", "面接", "受付", "支払い", "道案内"]

# 学習ニーズを強く示すキーワード（スコアを上げる）
LEARN_KEYWORDS = [
    "フレーズ", "表現", "例文", "言い方", "言い換え", "丁寧", "自然", "敬語",
    "発音", "リスニング", "スピーキング", "語彙", "単語", "文法",
    "練習", "実践", "基礎", "初心者", "上達", "コツ", "攻略",
    "頻出", "定番", "使える", "よく使う", "テンプレ", "3選", "5選", "NG", "OK", "Pro"
]

def score_title(t: str) -> int:
    score = 0
    # ① シーン系キーワード
    if any(t.startswith(k) for k in TOP_KEYWORDS):
        score += 20
    if re.search(r"\d+|チェックイン|注文|予約|空港|ホテル|レストラン|面接|受付|道案内|支払い", t):
        score += 15
    # ② 学習系ワード
    if any(k in t for k in LEARN_KEYWORDS):
        score += 25
    if re.search(r"(で使える|でよく使う|便利|自然な|丁寧な|言い換え|言い方|フレーズ|例文|テンプレ|コツ|3選|5選|NG|OK|Pro)", t):
        score += 15
    # ③ 28文字以内を優遇
    score += max(0, 15 - max(0, len(t) - 28))
    return score

LANG_NAME = {
    "en": "English", "pt": "Portuguese", "id": "Indonesian",
    "ja": "Japanese","ko": "Korean", "es": "Spanish",
}

# 日本語タイトル用のラベル（音声言語→◯◯語会話）
JP_CONV_LABEL = {
    "en": "英会話",
    "ja": "日本語会話",
    "es": "スペイン語会話",
    "pt": "ポルトガル語会話",
    "ko": "韓国語会話",
    "id": "インドネシア語会話",
}

# ───────────────────────────────────────────────
# トピック取得: "AUTO"なら自動ピック、文字列ならそのまま
# ───────────────────────────────────────────────
def resolve_topic(arg_topic: str) -> str:
    if arg_topic and arg_topic.strip().lower() == "auto":
        first_audio_lang = COMBOS[0]["audio"]
        topic = pick_by_content_type(CONTENT_MODE, first_audio_lang)
        logging.info(f"[AUTO TOPIC] {topic}")
        return topic
    return arg_topic

# ───────────────────────────────────────────────
# ✅ HOOK 生成 (seed_phrase)
# ───────────────────────────────────────────────
def _make_seed_phrase(topic: str, lang_code: str) -> str:
    lang = LANG_NAME.get(lang_code, "the target language")
    prompt = (
        f"Write ONE short hook in {lang} that grabs attention for a 30–45s short video about: {topic}. "
        "Start with a question or a bold contrast. ≤10 words. No quotes."
    )
    try:
        rsp = GPT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.7,
        )
        return rsp.choices[0].message.content.strip()
    except Exception:
        return ""

# ───────────────────────────────────────────────
# タイトル・説明・タグ
# ───────────────────────────────────────────────
def make_title(topic, title_lang: str, audio_lang_for_label: str | None = None):
    """
    日本語タイトルのときは音声言語に応じた「◯◯語会話」ラベルを自然に付与。
    さらに学習語に寄せてスコアリングして最良の1本を返す。
    """
    if title_lang == "ja":
        prompt = (
            "You are a YouTube copywriter.\n"
            "Generate 5 concise Japanese titles (each ≤28 JP chars) for a short educational video.\n"
            "Start with a strong scenario keyword and include a clear benefit.\n"
            f"Scenario/topic: {topic}\nReturn 5 lines only."
        )
    else:
        prompt = (
            f"You are a YouTube copywriter.\n"
            f"Generate 5 concise {LANG_NAME.get(title_lang,'English')} titles (≤55 chars) for a short educational video.\n"
            f"Topic: {topic}\nEach should be clear, emotional, and benefit-driven.\nReturn 5 lines only."
        )

    rsp = GPT.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.7
    )
    cands = [sanitize_title(x) for x in rsp.choices[0].message.content.split("\n") if x.strip()]

    if title_lang == "ja":
        label = JP_CONV_LABEL.get(audio_lang_for_label or "", "")
        def add_prefix(t: str) -> str:
            # シーン先頭語が無ければ topic を前置
            if not any(t.startswith(k) for k in TOP_KEYWORDS):
                t = f"{topic} {t}"
            # ラベルが入っていなければ先頭に付与
            if label and label not in t:
                t = f"{label} {t}"
            return t
        cands = [add_prefix(t) for t in cands]
        return sorted(cands, key=score_title, reverse=True)[0][:28]
    else:
        return max(cands, key=len)[:55]

def make_desc(topic, title_lang: str):
    prompt_desc = (
        f"Write one catchy summary (≤90 chars) in {LANG_NAME.get(title_lang,'English')} "
        f"for a YouTube Shorts about \"{topic}\". End with a simple call-to-action."
    )
    rsp = GPT.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt_desc}],
        temperature=0.5
    )
    base = rsp.choices[0].message.content.strip()

    prompt_tags = (
        f"List 2 or 3 short hashtags in {LANG_NAME.get(title_lang,'English')} "
        "related to conversation or learning."
    )
    tag_rsp = GPT.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt_tags}],
        temperature=0.3
    )
    hashtags = tag_rsp.choices[0].message.content.strip().replace("\n"," ")
    return f"{base} {hashtags}"

def make_tags(topic, audio_lang, subs, title_lang):
    tags = [
        topic,
        "conversation", "speaking practice", "listening practice",
        "language learning", "subtitles",
    ]
    for code in subs:
        if code in LANG_NAME:
            tags.append(f"{LANG_NAME[code]} subtitles")
    return list(dict.fromkeys(tags))[:15]

# ───────────────────────────────────────────────
# 行ごとの TTS スタイル
# ───────────────────────────────────────────────
def _style_for_line(idx: int, total: int, mode: str) -> str:
    if idx == 0:
        return "energetic"
    if idx == total - 1:
        return "calm" if mode in ("wisdom", "fact") else "serious"
    if mode in ("howto", "listicle", "qa"):
        return "serious" if idx in (2, 3) else "neutral"
    return "neutral"

# ───────────────────────────────────────────────
# 音声結合・トリム（行間に無音ギャップを持たせ、dur にも反映）
# ───────────────────────────────────────────────
def _concat_trim_to(mp_paths, max_sec, gap_ms=120):
    max_ms = int(max_sec * 1000)
    combined = AudioSegment.silent(duration=0)
    new_durs, elapsed = [], 0

    for idx, p in enumerate(mp_paths):
        seg = AudioSegment.from_file(p)
        seg_ms = len(seg)
        extra = gap_ms if idx < len(mp_paths) - 1 else 0  # 最後以外は無音を付与
        need = seg_ms + extra

        remain = max_ms - elapsed
        if remain <= 0:
            break

        if need <= remain:
            combined += seg
            elapsed += seg_ms
            if extra:
                combined += AudioSegment.silent(duration=extra)
                elapsed += extra
            new_durs.append((seg_ms + extra) / 1000.0)
        else:
            if remain <= seg_ms:
                combined += seg[:remain]
                new_durs.append(remain / 1000.0)
                elapsed += remain
            else:
                used_gap = remain - seg_ms
                combined += seg
                if used_gap > 0:
                    combined += AudioSegment.silent(duration=used_gap)
                new_durs.append((seg_ms + used_gap) / 1000.0)
                elapsed += seg_ms + used_gap
            break

    # 🔁 中間はWAVで出力（MP3の壊れフレーム問題を回避）
    (TEMP / "full_raw.wav").unlink(missing_ok=True)
    combined.export(TEMP / "full_raw.wav", format="wav")
    return new_durs

# ───────────────────────────────────────────────
# 1コンボ処理
# ───────────────────────────────────────────────
def run_one(topic, turns, audio_lang, subs, title_lang, yt_privacy, account, do_upload, chunk_size):
    reset_temp()

    # トピック（音声言語へ翻訳してから生成すると自然）
    topic_for_dialogue = translate(topic, audio_lang) if audio_lang != "ja" else topic

    # 強めの hook
    seed_phrase = _make_seed_phrase(topic_for_dialogue, audio_lang)

    # 台本（互換: List[(spk, line)])
    dialogue = make_dialogue(
        topic_for_dialogue, audio_lang, turns,
        seed_phrase=seed_phrase, mode=CONTENT_MODE
    )

    # 音声＆字幕
    valid_dialogue = [(spk, line) for (spk, line) in dialogue if line.strip()]
    mp_parts, sub_rows = [], [[] for _ in subs]
    for i, (spk, line) in enumerate(valid_dialogue, 1):
        mp = TEMP / f"{i:02d}.mp3"
        style = _style_for_line(i-1, len(valid_dialogue), CONTENT_MODE)
        speak(audio_lang, spk, line, mp, style=style)
        mp_parts.append(mp)
        for r, lang in enumerate(subs):
            sub_rows[r].append(line if lang == audio_lang else translate(line, lang))

    # 結合・整音
    new_durs = _concat_trim_to(mp_parts, MAX_SHORTS_SEC, gap_ms=120)

    # 🔒 enhance前にサイズチェック（生成失敗の早期検知）
    raw_wav = TEMP / "full_raw.wav"
    if (not raw_wav.exists()) or (raw_wav.stat().st_size < 10000):
        size = raw_wav.stat().st_size if raw_wav.exists() else 0
        raise RuntimeError(f"生成音声が不完全です: {raw_wav} size={size}")

    enhance(raw_wav, TEMP/"full.mp3")

    # 背景
    bg_png = TEMP / "bg.png"
    fetch_bg(topic, bg_png)

    # 台本行数とオーディオ尺の整合
    valid_dialogue = valid_dialogue[:len(new_durs)]

    # lines.json 生成（[spk, sub1, sub2, ..., dur]）
    lines_data = []
    for i, ((spk, txt), dur) in enumerate(zip(valid_dialogue, new_durs)):
        row = [spk]
        for r in range(len(subs)):
            row.append(sub_rows[r][i])
        row.append(dur)
        lines_data.append(row)
    (TEMP/"lines.json").write_text(json.dumps(lines_data, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.lines_only:
        return

    # サムネ
    thumb = TEMP / "thumbnail.jpg"
    thumb_lang = subs[1] if len(subs) > 1 else audio_lang
    make_thumbnail(topic, thumb_lang, thumb)

    # 動画生成
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_mp4 = OUTPUT / f"{audio_lang}-{'_'.join(subs)}_{stamp}.mp4"
    final_mp4.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", str(BASE/"chunk_builder.py"),
        str(TEMP/"lines.json"), str(TEMP/"full.mp3"), str(bg_png),
        "--chunk", str(chunk_size),
        "--rows", str(len(subs)),
        "--out", str(final_mp4),
    ]
    logging.info("🔹 chunk_builder cmd: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if not do_upload:
        return

    # メタ生成＆アップロード
    title = make_title(topic, title_lang, audio_lang_for_label=audio_lang)
    desc  = make_desc(topic, title_lang)
    tags  = make_tags(topic, audio_lang, subs, title_lang)

    upload(video_path=final_mp4, title=title, desc=desc, tags=tags,
           privacy=yt_privacy, account=account, thumbnail=thumb, default_lang=audio_lang)

# ───────────────────────────────────────────────
def run_all(topic, turns, privacy, do_upload, chunk_size):
    for combo in COMBOS:
        audio_lang  = combo["audio"]
        subs        = combo["subs"]
        account     = combo.get("account","default")
        title_lang  = combo.get("title_lang", subs[1] if len(subs)>1 else audio_lang)
        logging.info(f"=== Combo: {audio_lang}, subs={subs}, account={account}, title_lang={title_lang}, mode={CONTENT_MODE} ===")
        run_one(topic, turns, audio_lang, subs, title_lang, privacy, account, do_upload, chunk_size)

# ───────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("topic", help='会話テーマ。自動選択する場合は "AUTO" を指定')
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--privacy", default="unlisted", choices=["public","unlisted","private"])
    ap.add_argument("--lines-only", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--chunk", type=int, default=9999, help="Shortsは分割せず1本推奨")
    args = ap.parse_args()

    topic = resolve_topic(args.topic)
    run_all(topic, args.turns, args.privacy, not args.no_upload, args.chunk)