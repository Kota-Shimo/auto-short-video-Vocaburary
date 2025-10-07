"""Generate a two-person *discussion / roleplay* script via GPT-4o
with strict monolingual output, hook support, and a light loop-back ending.
"""

from typing import List, Tuple
import re
from openai import OpenAI
from config import OPENAI_API_KEY

openai = OpenAI(api_key=OPENAI_API_KEY)


# --------------------------- language helpers ---------------------------

def _lang_rules(lang: str, topic_hint: str) -> str:
    """
    Strong, language-specific constraints to prevent code-switching
    *without* destroying proper nouns or airline codes, etc.
    """
    if lang == "ja":
        # 日本語台本の英語混入を避けつつ、固有名詞は許可
        return (
            "This is a Japanese listening script. Use natural Japanese only. "
            "Avoid English words and romaji (Latin letters) *except* for proper nouns or codes "
            "(e.g., JAL, ANA, QRコード). Do not translate or explain such proper nouns; keep them as-is. "
            "Ignore any implication that English should appear even if the topic contains '英語'."
        )
    return f"Stay entirely in {lang}. Avoid mixing other languages."


# できるだけ“意味を削らない”軽い整形だけにする
_LATIN = re.compile(r"[A-Za-z]")

def _fallback_line(lang: str) -> str:
    return {
        "ja": "はい。",
        "ko": "네.",
        "es": "Vale.",
        "pt": "Certo.",
        "id": "Oke.",
        "en": "Okay."
    }.get(lang, "Okay.")

def _sanitize_line(lang: str, text: str) -> str:
    """
    Light post-processing for TTS/字幕安定化:
      - 改行/空白の正規化
      - 三点リーダの揺れ統一
      - 日本語のみ、末尾に句読点を補う（無音終止を避ける）
    ※ アルファベット削除は行わない（JAL, ANA などを温存）
    """
    txt = (text or "").strip()
    if not txt:
        return _fallback_line(lang)

    # 共通の軽い整形
    txt = txt.replace("…", "…").replace("...", "…")
    txt = re.sub(r"\s+", " ", txt).strip()

    if lang == "ja":
        # 末尾に句読点が無ければ「。」を補う（TTSが自然に止まる）
        if not re.search(r"[。！？!?]$", txt):
            txt += "。"

    return txt


def _extract_dialogue_lines(raw: str) -> List[str]:
    """
    GPTの出力から "Alice:" / "Bob:" 行だけを安全に抽出。
    例: "1) Alice: ..." や "- Bob: ..." も拾う。
    """
    out: List[str] = []
    for ln in (raw or "").splitlines():
        ln = ln.strip()
        m = re.match(r"^\s*(?:\d+[\).\-\s]*)?(Alice|Bob)\s*:\s*(.*)$", ln, flags=re.IGNORECASE)
        if m:
            speaker = "Alice" if m.group(1).lower() == "alice" else "Bob"
            text = m.group(2).strip()
            out.append(f"{speaker}: {text}")
    return out


# --------------------------- main entry ---------------------------

def make_dialogue(
    topic: str,
    lang: str,
    turns: int = 8,
    seed_phrase: str = ""
) -> List[Tuple[str, str]]:
    """
    - Alice/Bob が交互に話す短い自然会話を生成（厳密に 2*turns 行）
    - seed_phrase は“ムード/スタイルのヒント”として使用（逐語の繰り返しは禁止）
    - 最終行は話題やフックを軽くリフレインしてループ感を演出
    - 日本語は“英語混入を避けつつ固有名詞は許容”のプロンプトを付与
    - 出力後の整形は最小限（意味の削除は行わない）
    """
    # 日本語だけ括弧でトピックを補助表示
    topic_hint = f"「{topic}」" if lang == "ja" else topic
    lang_rules = _lang_rules(lang, topic_hint)

    prompt = (
        f"You are a native-level {lang.upper()} dialogue writer.\n"
        f"Write a short, natural conversation in {lang} between Alice and Bob.\n\n"
        f"Scene topic: {topic_hint}\n"
        f"Tone reference (seed phrase): \"{seed_phrase}\" "
        f"(use only as mood/style hint; do not repeat it literally).\n\n"
        "Rules:\n"
        "1) Alternate strictly: Alice, Bob, Alice, Bob...\n"
        f"2) Produce exactly {turns * 2} lines.\n"
        "3) Each line begins with 'Alice:' or 'Bob:' and contains one short, natural sentence.\n"
        f"4) {lang_rules}\n"
        "5) No ellipses beyond a single one (…); no emojis; no bullet points; no stage directions.\n"
        "6) Keep it friendly, realistic, and concise.\n"
        "7) Make the final line subtly echo the main topic or hook to feel loopable.\n"
        "8) Output ONLY the dialogue lines (no explanations).\n"
    )

    rsp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.45,
    )

    # 抽出（余分な見出し/番号は無視）
    raw_lines = _extract_dialogue_lines(rsp.choices[0].message.content)

    # 行数調整：過剰カット & 不足を交互に補完
    raw_lines = raw_lines[: turns * 2]
    while len(raw_lines) < turns * 2:
        # スピーカーは交互に補完
        spk = "Alice" if len(raw_lines) % 2 == 0 else "Bob"
        raw_lines.append(f"{spk}: {_fallback_line(lang)}")

    # "Alice: こんにちは" → ("Alice", "こんにちは") に整形＋軽いクリーンアップ
    parsed: List[Tuple[str, str]] = []
    for ln in raw_lines:
        spk, txt = ln.split(":", 1)
        parsed.append((spk.strip(), _sanitize_line(lang, txt)))

    return parsed