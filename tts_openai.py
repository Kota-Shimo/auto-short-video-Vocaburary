"""OpenAI TTS wrapper – language-aware & two-speaker support with simple style control."""

import re
from pathlib import Path
from openai import OpenAI
from config import OPENAI_API_KEY, VOICE_MAP

client = OpenAI(api_key=OPENAI_API_KEY)

# フォールバック用（言語が VOICE_MAP に無い場合）
FALLBACK_VOICES = ("alloy", "echo")  # (Alice, Bob)


def _clean_for_tts(text: str, lang: str) -> str:
    """
    音声合成前にテキストを整形：
    - 話者名「Alice:」「Bob:」「N:」などを削除
    - 不要な記号や半端な句読点を除去
    - 空白や改行を整理
    - 日本語は英単語を除去して誤読を防ぐ
    """
    t = re.sub(r"^[A-Za-z]+:\s*", "", text)  # Alice:/Bob:/N: など削除
    t = re.sub(r"\s+", " ", t).strip()

    if lang == "ja":
        t = re.sub(r"[A-Za-z]+", "", t)            # 英単語/ローマ字の排除
        t = re.sub(r"[#\"'※＊*~`]", "", t)         # 記号の整理
        t = re.sub(r"\s+", " ", t).strip()

    return t or "。"


def _apply_style(text: str, lang: str, style: str) -> str:
    """
    テキスト整形だけで抑揚を“誘導”する軽量スタイル。
    - energetic: 句点→感嘆、語尾を軽く上げる
    - calm     : 句点増で落ち着き、語尾を柔らかく
    - serious  : 断定調・短文寄せ
    - neutral  : 変更なし
    ※ TTSモデルのパラメータを直接いじらないため、下位互換で安全。
    """
    s = text.strip()
    st = (style or "neutral").lower()

    if st == "energetic":
        if lang == "ja":
            s = s.replace("。", "！")
            if not s.endswith(("！", "？")):
                s += "！"
        else:
            if not s.endswith(("!", "?")):
                s += "!"
    elif st == "calm":
        if lang == "ja":
            s = s.replace("！", "。").replace("!", "。")
            if not s.endswith("。"):
                s += "。"
        else:
            s = s.replace("!", ".")
            if not s.endswith("."):
                s += "."
    elif st == "serious":
        if lang == "ja":
            # 余分な装飾を抑え、句点で締める
            s = re.sub(r"[！!？?]+$", "", s)
            if not s.endswith("。"):
                s += "。"
        else:
            s = re.sub(r"[!?.]+$", ".", s)
            if not s.endswith("."):
                s += "."
    # neutral は変更なし
    return s


def speak(lang: str, speaker: str, text: str, out_path: Path, style: str = "neutral"):
    """
    lang     : 'en', 'ja', 'pt', 'id' など
    speaker  : 'Alice' / 'Bob' / 'N'（N=ナレーション）
    text     : セリフ
    out_path : 書き出し先 .mp3
    style    : 'neutral' | 'energetic' | 'calm' | 'serious'（任意）
    """
    # 1) 事前整形
    clean_text = _clean_for_tts(text, lang)
    styled_text = _apply_style(clean_text, lang, style)

    # 2) ボイス選択（N=Alice流用）
    v_a, v_b = VOICE_MAP.get(lang, FALLBACK_VOICES)
    spk = (speaker or "").lower()
    voice_id = v_a if spk in ("alice", "n") else v_b
    # ※ Bob に合わせたい場合は以下に変更：
    # voice_id = v_b if spk in ("bob", "n") else v_a

    # 3) 合成
    resp = client.audio.speech.create(
        model="tts-1",          # 高音質は "tts-1-hd"
        voice=voice_id,
        input=styled_text,
    )
    out_path.write_bytes(resp.content)
