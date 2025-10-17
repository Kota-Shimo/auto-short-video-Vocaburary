# dialogue.py
"""Generate short scripts via GPT-4o.
- Backward compatible: returns List[(speaker, text)] with 'Alice'/'Bob' alternating in dialogue modes.
- Monologue-first in specific modes (wisdom, fact), using speaker 'N' (Narrator).
- Growth structure: Hook → 3 beats → Closing. Short lines. Strictly monolingual & neutral (no language/country mentions).
"""

from typing import List, Tuple
import re
from openai import OpenAI
from config import OPENAI_API_KEY

openai = OpenAI(api_key=OPENAI_API_KEY)

# ─────────────────────────────────────────
# モード定義（中立的ガイド）※記号や英字ラベルの例示は避ける
# ─────────────────────────────────────────
MODE_GUIDE = {
    "dialogue": "Real-life roleplay. Hook, then three short turns, then a concise closing. Keep universal.",
    "howto":    "Actionable three-step guidance. Hook, then step one, step two, and a closing.",
    "listicle": "Three concise points. Hook, point one, point two, point three, closing.",
    "wisdom":   "Motivational. Hook, three short insights, closing.",
    "fact":     "Micro-knowledge. Hook, three small facts/examples, closing.",
    "qa":       "Three-step improvement: problem, better, best. Hook first, then closing.",
}

# ナレーション中心にしたいモード
MONOLOGUE_MODES = {"wisdom", "fact"}

# ─────────────────────────────────────────
# 言語・国名を出さない前提で、厳密モノリンガル化（ASCII記号の例示も出力禁止）
# ─────────────────────────────────────────
def _lang_rules(lang: str) -> str:
    """
    出力言語を厳密に単一化し、他言語/他文字体系や翻訳注釈を禁止。
    出力内に言語名・国名・学習者呼称を出さない。
    ASCII記号の使用も避ける（/, -, →, (), [], <>, | など）。
    """
    if lang == "ja":
        return (
            "Write entirely in Japanese. "
            "Do not include Latin letters, non-Japanese words, or code-switching. "
            "Avoid ASCII symbols such as '/', '-', '→', '()', '[]', '<>', and '|'. "
            "No translation glosses or bracketed meanings. "
            "Do not mention any language names, nationalities, or countries."
        )
    return (
        f"Write entirely in {lang}. "
        "Do not code-switch or include other languages/writing systems. "
        "Avoid ASCII symbols like '/', '-', '→', '()', '[]', '<>', and '|'. "
        "No translation glosses or bracketed meanings. "
        "Do not mention any language names, nationalities, or countries."
    )

# 学習イントロを促すヒント（各言語で自然に）
def _learning_intro_hint(lang: str) -> str:
    if lang == "ja":
        return "Make the first line clearly say it's a learning intro, e.g.,『今日は〜を学ぼう』『今日の学び: 〜』. Keep it short and natural."
    else:
        return "Make the first line clearly invite learning, e.g., 'Let's learn … today.' or 'Today, we learn …'. Keep it short and natural."

# ─────────────────────────────────────────
# TTS 安定のための軽い整形（JAのみ特別処理）
# ─────────────────────────────────────────
def _sanitize_line(lang: str, text: str) -> str:
    txt = text.strip()
    if lang == "ja":
        # ローマ字/英単語除去（数字は保持）
        txt = re.sub(r"[A-Za-z]+", "", txt)
        # 三点リーダなどを句点へ
        txt = txt.replace("...", "。").replace("…", "。")
        # ラベルのコロン周り整形
        txt = re.sub(r"\s*:\s*", ": ", txt)
        # 余分な空白圧縮
        txt = re.sub(r"\s+", " ", txt).strip()
        # 末尾が中途半端なら軽く締める
        if txt and txt[-1] not in "。！？…!?":
            txt += "。"
    else:
        txt = txt.replace("…", "...").strip()
    return txt

def _fallback_line(lang: str) -> str:
    return "はい。" if lang == "ja" else "Okay."

# ─────────────────────────────────────────
# 本体
# ─────────────────────────────────────────
def make_dialogue(
    topic: str,
    lang: str,
    turns: int = 8,
    seed_phrase: str = "",
    mode: str = "dialogue",
) -> List[Tuple[str, str]]:
    """
    Returns: List[(speaker, text)]
    - dialogue/howto/listicle/qa → Alice/Bob 交互の会話中心（合計 2*turns 行）
    - wisdom/fact → N（Narrator）中心のモノローグ（合計 turns 行）
    - Hook → 3ビート → Closing（ループ感）を強制
    - 出力は中立：特定の言語名・国名・学習者呼称を出さない
    """
    is_monologue = mode in MONOLOGUE_MODES

    topic_hint = f"「{topic}」" if lang == "ja" else topic
    lang_rules = _lang_rules(lang)
    mode_guide = MODE_GUIDE.get(mode, MODE_GUIDE["dialogue"])

    # 行長ヒント（文字体系ベース）
    length_hint = (
        "For alphabetic scripts: <= 12 words per line. "
        "For CJK or similar: keep lines concise (~<=20 characters)."
    )

    # モード別追加ルール＋学習イントロ要求（記号・英字ラベルを使わない）
    intro_hint = _learning_intro_hint(lang)
    if mode == "dialogue":
        extra_rule = (
            "Include exactly one short learning tip within the dialogue "
            "as plain sentences (no labels). "
            "Avoid slashes, arrows, or letter labels. "
            + intro_hint
        )
    elif mode == "fact":
        extra_rule = (
            "Include one short, surprising point about communication or cultural nuance, "
            "plus one concise example expression that fits the scene, as plain sentences. "
            "Avoid slashes, arrows, or letter labels. "
            + intro_hint
        )
    elif mode == "howto":
        extra_rule = (
            "Structure as a quick reason, then two short steps, then a simple nudge. "
            "Express them as plain sentences; do not use numbered lists, slashes, or arrows. "
            + intro_hint
        )
    elif mode == "listicle":
        extra_rule = (
            "Present three parallel points with a clear rhythm as plain sentences "
            "(use natural connectors; avoid slashes and arrows). "
            + intro_hint
        )
    elif mode == "wisdom":
        extra_rule = (
            "Keep it reflective and encouraging: one key idea, a tiny example, and a gentle takeaway. "
            "Use plain sentences; no slashes or arrows. "
            + intro_hint
        )
    elif mode == "qa":
        extra_rule = (
            "Show a three-step improvement pattern as plain sentences: problem, better, best. "
            "Do not use arrows, slashes, or letter labels. "
            + intro_hint
        )
    else:
        extra_rule = intro_hint

    if is_monologue:
        # ── モノローグ（N のみ） ───────────────────────────────
        user = f"""
You are a native-level narration writer.

Write a short, natural monologue in {lang} by a narrator 'N'.
Topic: {topic_hint}
Tone ref (seed): "{seed_phrase}" (style hint only; do not repeat literally)
Mode: {mode} ({mode_guide})
{extra_rule}

STRUCTURE:
- Line1 (Learning-intro Hook, 0–2s): clearly invite learning for today's topic
- Lines2–4 (Beats 1–2): add pattern change (numbers, contrast, concrete example)
- Lines5–6 (Beat 3): one visual tip/example
- Final line (Closing, <=8s left): one clear action; subtly echo topic for loop feel

Rules:
1) Produce exactly {turns} lines (all spoken by 'N'), concise one sentence each.
2) Prefix each line with 'N:'.
3) {lang_rules}
4) {length_hint}
5) Do not mention any language names, nationalities, or countries.
6) Avoid lists, stage directions, emojis.
7) Output ONLY the lines (no explanations).
""".strip()
    else:
        # ── 会話（Alice/Bob 交互） ─────────────────────────────
        user = f"""
You are a native-level dialogue writer.

Write a short, natural 2-person conversation in {lang} between Alice and Bob.
Scene topic: {topic_hint}
Tone ref (seed): "{seed_phrase}" (style hint only; do not repeat literally)
Mode: {mode} ({mode_guide})
{extra_rule}

STRUCTURE (map to alternating lines):
- Line1 (Learning-intro Hook, 0–2s): clearly invite learning for today's topic
- Lines2–4 (Beats 1–2): pattern change (numbers, contrast, example)
- Lines5–6 (Beat 3): one concrete, visual tip/example
- Final line (Closing, <=8s left): one clear action; subtly echo topic for loop feel

Rules:
1) Alternate strictly: Alice:, Bob:, Alice:, Bob: ... until exactly {turns * 2} lines.
2) Each line = one short sentence; no lists, no stage directions, no emojis.
3) {lang_rules}
4) {length_hint}
5) Do not mention any language names, nationalities, or countries.
6) Avoid repetitive endings; vary rhythm every ~8 seconds.
7) Output ONLY the dialogue lines. No explanations.
""".strip()

    rsp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user}],
        temperature=0.6,
        timeout=45,
    )

    raw = rsp.choices[0].message.content or ""
    raw_lines = raw.strip().splitlines()
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