# dialogue.py
"""Generate short scripts via GPT-4o.
- Backward compatible: returns List[(speaker, text)] with 'Alice'/'Bob' alternating in dialogue modes.
- Monologue-first in specific modes (wisdom, fact), using speaker 'N' (Narrator).
- Enforces growth structure: Hook → 3 beats → Closing, short lines, no code-switching.
"""

from typing import List, Tuple
import re
from openai import OpenAI
from config import OPENAI_API_KEY

openai = OpenAI(api_key=OPENAI_API_KEY)

# ─────────────────────────────────────────
# モード定義
# ─────────────────────────────────────────
MODE_GUIDE = {
    "dialogue": "Real-life roleplay. Hook(0-2s) -> Turn1 -> Turn2 -> Turn3 -> Closing(<=8s left). Keep universal.",
    "howto":    "Actionable 3 steps. Hook -> Step1 -> Step2 -> Step3 -> Closing.",
    "listicle": "3 points. Hook -> Point1 -> Point2 -> Point3 -> Closing.",
    "wisdom":   "Motivational. Hook -> Key1 -> Key2 -> Key3 -> Closing.",
    "fact":     "Micro-knowledge. Hook -> Fact1 -> Fact2 -> Fact3 -> Closing.",
    "qa":       "NG/OK/Pro. Hook -> NG -> OK -> Pro -> Closing.",
}

# ここで「ナレーション中心」にしたいモードを指定
MONOLOGUE_MODES = {"wisdom", "fact"}

def _lang_rules(lang: str) -> str:
    """Language-specific constraints to avoid code-switching."""
    if lang == "ja":
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
    Returns: List[(speaker, text)]
    - dialogue/howto/listicle/qa → Alice/Bob 交互の会話中心
    - wisdom/fact → N（Narrator）中心のモノローグ（必要に応じて A/B を少しだけ使う可）
    - Hook → 3ビート → Closing（ループ感）を強制
    """
    is_monologue = mode in MONOLOGUE_MODES

    topic_hint = f"「{topic}」" if lang == "ja" else topic
    lang_rules = _lang_rules(lang)
    mode_guide = MODE_GUIDE.get(mode, MODE_GUIDE["dialogue"])

    if is_monologue:
        # ── モノローグ優先プロンプト ───────────────────────────────
        user = f"""
You are a native-level {lang.upper()} narration writer.

Write a short, natural monologue in {lang} by a narrator 'N'.
Topic: {topic_hint}
Tone ref (seed): "{seed_phrase}" (style hint only; do not repeat literally)

STRUCTURE:
- Line1 (Hook, 0–2s): bold claim or question to pull attention
- Lines2–4 (Beats 1–2): add pattern change (numbers, contrast, concrete example)
- Lines5–6 (Beat 3): one visual tip/example
- Final line (Closing, <=8s left): one clear action; subtly echo topic for loop feel

Rules:
1) Produce exactly {turns} lines (all spoken by 'N'), concise one sentence each.
2) Prefix each line with 'N:'.
3) {lang_rules}
4) EN: <=12 words/line. JA: keep concise (~<=20 mora).
5) Avoid lists, stage directions, emojis.
6) Output ONLY the lines (no explanations).
"""
    else:
        # ── 会話優先プロンプト（Alice/Bob 交互） ─────────────────────
        user = f"""
You are a native-level {lang.upper()} dialogue writer.

Write a short, natural 2-person conversation in {lang} between Alice and Bob.
Scene topic: {topic_hint}
Tone ref (seed): "{seed_phrase}" (style hint only; do not repeat literally)
Mode: {mode} ({mode_guide})

STRUCTURE (map to alternating lines):
- Line1 (Hook, 0–2s): bold claim or question
- Lines2–4 (Beats 1–2): pattern change (numbers, contrast, example)
- Lines5–6 (Beat 3): one concrete, visual tip/example
- Final line (Closing, <=8s left): one clear action; subtly echo topic for loop feel

Rules:
1) Alternate strictly: Alice:, Bob:, Alice:, Bob: ... until exactly {turns * 2} lines.
2) Each line = one short sentence; no lists, no stage directions, no emojis.
3) {lang_rules}
4) EN: <=12 words/line. JA: keep concise (~<=20 mora).
5) Avoid repetitive endings; vary rhythm every ~8 seconds.
6) Output ONLY the dialogue lines. No explanations.
"""

    rsp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user}],
        temperature=0.6,
        timeout=45,
    )

    raw_lines = (rsp.choices[0].message.content or "").strip().splitlines()
    result: List[Tuple[str, str]] = []

    if is_monologue:
        # "N:" 行のみを採用。足りなければ埋め草、超過はカット。
        lines = [l.strip() for l in raw_lines if l.strip().startswith("N:")]
        lines = lines[:turns]
        while len(lines) < turns:
            lines.append("N:")  # 空でも後段でフォールバック
        for ln in lines:
            spk, txt = ("N", ln.split(":", 1)[1].strip()) if ":" in ln else ("N", "")
            txt = _sanitize_line(lang, txt) or _fallback_line(lang)
            result.append((spk, txt))
        return result

    # 会話モード："Alice:" / "Bob:" のみ採用
    lines = [l.strip() for l in raw_lines if l.strip().startswith(("Alice:", "Bob:"))]
    lines = lines[: turns * 2]
    while len(lines) < turns * 2:
        lines.append("Alice:" if len(lines) % 2 == 0 else "Bob:")

    for idx, ln in enumerate(lines):
        if ":" in ln:
            spk, txt = ln.split(":", 1)
            txt = txt.strip()
        else:
            spk = "Alice" if idx % 2 == 0 else "Bob"
            txt = ""
        txt = _sanitize_line(lang, txt) or _fallback_line(lang)
        result.append((spk.strip(), txt))

    return result
