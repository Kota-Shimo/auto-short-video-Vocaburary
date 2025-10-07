# topic_picker.py
"""
Pick TODAY’s podcast/video topic.

- まず GPT-4o に “自然な日本語の会話シーン名” を 1 行だけリクエスト
- 応答が旧形式（「<大テーマ> - <具体シーン>」や「英語/英会話」を含む）でも
  自動的に “ホテルでのチェックイン会話” のような自然表現へ正規化
- API 呼び出しが失敗したら SEED_TOPICS からランダムでフォールバック
"""

import random
import datetime
import os
import openai
import re

openai.api_key = os.getenv("OPENAI_API_KEY")

# ── フォールバック用プリセット（最初から自然な日本語） ───────────────
SEED_TOPICS: list[str] = [
    "ホテルでのチェックイン会話",
    "ホテルでの朝食案内",
    "ホテルでの部屋設備の説明",
    "ホテルでのチェックアウト会話",

    "空港でのチェックイン会話",
    "空港での保安検査のやりとり",
    "空港での搭乗口アナウンスの受け答え",
    "機内でのやりとり",

    "レストランでの入店と席案内",
    "レストランでの注文会話",
    "料理の説明を聞く会話",
    "レストランでの会計会話",
]

# ────────────────────────────────────────
# 正規化ユーティリティ
# ────────────────────────────────────────

# 大テーマの置換表（「◯◯英語/英会話」→「◯◯での」等）
THEME_MAP = {
    "ホテル英語": "ホテルでの",
    "空港英会話": "空港での",
    "空港英語": "空港での",
    "レストラン英語": "レストランでの",
    "旅行英会話": "旅行中の",
    "接客英語": "接客の",
    "仕事で使う英語": "仕事での",
    "ビジネス英語": "ビジネスでの",
}

# 「の会話」を付けずに自然な名詞句で終わらせたいキーワード
NO_SUFFIX_KEYWORDS = [
    "会話", "やりとり", "案内", "説明", "確認", "手続き", "質問", "受け答え",
    "問い合わせ", "オーダー", "予約", "対応", "注意点", "ポイント"
]

def _clean_line(raw: str) -> str:
    """先頭行を取り、両端の引用符や記号を除去"""
    first = raw.strip().splitlines()[0]
    t = re.sub(r'^[\s"“”\'\-•・]+', "", first)
    t = re.sub(r'[\s"“”\']+$', "", t)
    t = re.sub(r'\s+', " ", t).strip()
    return t

def _needs_suffix(scene: str) -> bool:
    return not any(k in scene for k in NO_SUFFIX_KEYWORDS)

def _normalize_hyphen_form(s: str) -> str:
    """
    「<大テーマ>(英語|英会話)? - <具体シーン>」→「<場所/文脈> <具体シーン>(の会話)」
    例: "ホテル英語 - チェックイン" → "ホテルでの チェックイン会話"
    """
    m = re.match(r'^\s*(.+?)\s*(?:英語|英会話)?\s*-\s*(.+?)\s*$', s)
    if not m:
        return s  # ハイフン形式でない → そのまま

    theme, scene = m.group(1), m.group(2)
    for k, v in THEME_MAP.items():
        if k in theme:
            theme = theme.replace(k, v)
            break
    if not re.search(r'(での|中の|の)$', theme):
        theme = theme + "での"
    theme = re.sub(r'(での)+', 'での', theme)
    theme = re.sub(r'(のの)+', 'の', theme)

    if _needs_suffix(scene):
        scene_out = f"{scene}の会話"
    else:
        scene_out = scene

    topic = f"{theme} {scene_out}"
    return re.sub(r'\s+', " ", topic).strip()

def _normalize(topic: str) -> str:
    """自然な会話トピックに正規化"""
    t = topic
    if " - " in t or "-" in t:
        t = _normalize_hyphen_form(t)
    for k, v in THEME_MAP.items():
        t = t.replace(k, v)
    if re.search(r'(での|中の|の)\s*$', t):
        t = t.rstrip(" の")
    if not any(k in t for k in NO_SUFFIX_KEYWORDS):
        m2 = re.match(r'^(.*での)\s+(.+)$', t)
        if m2:
            prefix, scene = m2.group(1), m2.group(2)
            if _needs_suffix(scene):
                t = f"{prefix} {scene}の会話"
    t = re.sub(r'\s+', " ", t).strip()
    return t

# ────────────────────────────────────────
def pick() -> str:
    """自然な日本語の会話シーントピックを返す。"""
    today = datetime.date.today().isoformat()
    prompt = (
        f"Today is {today}. "
        "日本語で、語学学習向けの“自然な会話シーン名”を1つだけ提案してください。"
        "例:『ホテルでのチェックイン会話』『空港での保安検査のやりとり』『レストランでの注文会話』。"
        "15〜20文字程度を目安。句読点や引用符は不要。"
        "返答はそのフレーズ1行のみ。"
    )

    try:
        rsp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            timeout=20,
        )
        raw = _clean_line(rsp.choices[0].message.content)
        topic = _normalize(raw)
        if not topic or re.search(r'[A-Za-z]', topic):
            return random.choice(SEED_TOPICS)
        return topic
    except Exception:
        return random.choice(SEED_TOPICS)

# ────────────────────────────────────────
# ✅ 追加: コンテンツタイプ別トピック生成
# ────────────────────────────────────────
def pick_by_content_type(content_type: str, audio_lang: str) -> str:
    """
    コンテンツ種別に応じて、誰でも刺さる“伸びる”ショート動画トピックを1行返す。
    dialogue のときは既存の正規化ロジックを活かして自然な会話シーンに整える。
    """
    import datetime, random, re, os
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    today = datetime.date.today().isoformat()
    GUIDE = {
        "dialogue": "誰でも遭遇する生活/仕事シーン名（例: ホテルでのチェックイン会話、レストランでの注文会話など）。",
        "howto": "30秒で実践できるハウツー（例: 英語が自然に聞こえる3ステップ、集中力を上げるコツ3つ）。",
        "listicle": "3ポイントで話せるテーマ（例: モチベーションが上がる習慣3つ、印象が良くなるフレーズ3選）。",
        "wisdom": "短い知恵・名言（例: 先延ばしを防ぐ一言、小さく始める力）。",
        "fact": "豆知識・雑学（例: 海外で驚かれる日本の文化、言葉の由来など）。",
        "qa": "誤解されやすいQ&A（例: 発音のNG→OK→Proパターン、英会話の間違いあるある）。",
    }

    prompt = (
        f"Today is {today}. "
        f"日本語で、{GUIDE.get(content_type, GUIDE['dialogue'])} "
        "句読点や引用符なしで自然な1行だけ返してください。"
    )

    try:
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            timeout=20,
        )
        raw = _clean_line(rsp.choices[0].message.content)
        if content_type == "dialogue":
            raw = _normalize(raw)
        if not raw or re.search(r"[A-Za-z]", raw):
            return random.choice(SEED_TOPICS)
        return raw
    except Exception:
        return random.choice(SEED_TOPICS)

# ────────────────────────────────────────
if __name__ == "__main__":
    print(pick())
