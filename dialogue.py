# dialogue.py
"""Generate a two-person roleplay script via GPT-4o with a growth-oriented structure.
   - Backward compatible: returns List[(speaker, text)] with 'Alice'/'Bob' alternating.
   - Adds `mode` to control patterns: dialogue/howto/listicle/wisdom/fact/qa
   - Enforces hook → 3 beats → closing, short lines, no code-switching.
"""

from typing import List, Tuple
import re
import os
from openai import OpenAI
from config import OPENAI_API_KEY

openai = OpenAI(api_key=OPENAI_API_KEY)

# ─────────────────────────────────────────
# Mode guides (内部で“伸びる構成”を強制)
# ─────────────────────────────────────────
_MODE_GUIDE = {
    "dialogue": "Real-life roleplay. Hook(0-2s) -> Turn1 -> Turn2 -> Turn3 -> Closing(<=8s left). Keep universal.",
    "howto":    "Actionable 3 steps. Hook -> Step1 -> Step2 -> Step3 -> Closing.",
    "listicle": "3 points. Hook -> Point1 -> Point2 -> Point3 -> Closing.",
    "wisdom":   "Motivational. Hook -> Key1 -> Key2 -> Key3 -> Closing.",
    "fact":     "Micro-knowledge. Hook -> Fact1 -> Fact2 -> Fact3 -> Closing.",
    "qa":       "NG/OK/Pro. Hook -> NG -> OK -> Pro -> Closing.",
}

def _lang_rules(lang: str) -> str:
    """Language-specific constraints to avoid code-switching."""
    if lang == "ja":
        # 日本語台本の英語・ローマ字混入を強く禁止
        return (
            "This is a Japanese listening script. "
            "Use pure Japanese only. "
            "Do NOT include any English words, romaji (Latin letters), or code-switching. "
            "Ignore any implication that English should appear even if the topic contains '英語'. "
            "Natural Japanese only."
        )
    return f"Stay entirely in {lang}. Avoid mixing other languages."

def _sanitize_line(lang: str, text: str) -> str:
    """TTSが詰まりやすい要素を軽減。"""
    txt = text.strip()
    if lang == "ja":
        txt = re.sub(r"[A-Za-z]+", "", txt)           # ローマ字/英単語除去（数字は保持）
        txt = txt.replace("...", "。").replace("…", "。")
        txt = re.sub(r"\s*:\s*", ": ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
    else:
        txt = txt.replace("…", "...").strip()
    return txt

def _fallback_line(lang: str) -> str:
    return "はい。" if lang == "ja" else "Okay."

def make_dialogue(
    topic: str,
    lang: str,
    turns: int = 8,
    seed_phrase: str = "",
    mode: str = "dialogue",
) -> List[Tuple[str, str]]:
    """
    Returns: List[(speaker, text)] with strict alternation 'Alice'/'Bob'.
    - first line acts as Hook, last line as Closing（軽くループ感）
    - 中間は 3ビート（数字/具体例/言い換えで変化を付ける）
    - `mode` によって内容の型を変えるが、出力フォーマットは常に同じ
    """
    topic_hint = f"「{topic}」" if lang == "ja" else topic
    lang_rules = _lang_rules(lang)
    mode_guide = _MODE_GUIDE.get(mode, _MODE_GUIDE["dialogue"])

    # GPT へのプロンプト（“伸びる構成”を交互会話にマッピング）
    # 1行は短く：EN<=12 words / JAは~20モーラ目安
    user = f"""
You are a native-level {lang.upper()} dialogue writer.

Write a short, natural 2-person conversation in {lang} between Alice and Bob.
Scene topic: {topic_hint}
Tone reference (seed phrase): "{seed_phrase}" (style hint only; do not repeat literally)

STRUCTURE (map to alternating lines):
- Line1 (Hook, 0–2s): bold claim or question that pulls attention
- Lines2–4 (Beats 1–2): add a change of pattern (numbers, contrast, example)
- Lines5–6 (Beat 3): one concrete, visual tip/example
- Final line (Closing, <=8s left): one clear action; subtly echo the topic for loop feel

Rules:
1) Alternate strictly: Alice:, Bob:, Alice:, Bob: ... until exactly {turns * 2} lines.
2) Each line = one short sentence; no lists, no stage directions, no emojis.
3) {lang_rules}
4) EN: <=12 words/line. JA: keep concise (~<=20 mora feel).
5) Avoid repetitive endings; vary rhythm/phrasing every ~8 seconds.
6) Output ONLY the dialogue lines. No explanations.
"""

    rsp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user}],
        temperature=0.6,   # 伸びる言い回し＋安定性のバランス
        timeout=45,
    )

    raw_lines = (rsp.choices[0].message.content or "").strip().splitlines()
    # "Alice:" / "Bob:" で始まる行のみ抽出
    lines = [l.strip() for l in raw_lines if l.strip().startswith(("Alice:", "Bob:"))]

    # 余剰カット・不足は交互で補完（中身が空なら最短の埋め草）
    lines = lines[: turns * 2]
    while len(lines) < turns * 2:
        lines.append("Alice:" if len(lines) % 2 == 0 else "Bob:")

    parsed: List[Tuple[str, str]] = []
    for idx, ln in enumerate(lines):
        if ":" in ln:
            spk, txt = ln.split(":", 1)
            txt = txt.strip()
        else:
            spk = "Alice" if idx % 2 == 0 else "Bob"
            txt = ""

        txt = _sanitize_line(lang, txt) or _fallback_line(lang)
        parsed.append((spk.strip(), txt))

    # 念のため先頭/末尾の“役割”は最低限守られているように軽整形（内容はAIに任せる）
    # ここでは書き換えすぎない。空で来たら埋め草のみ。
    if not parsed[0][1]:
        parsed[0] = (parsed[0][0], _fallback_line(lang))
    if not parsed[-1][1]:
        parsed[-1] = (parsed[-1][0], _fallback_line(lang))

    return parsed
