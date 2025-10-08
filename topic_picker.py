# topic_picker.py
"""
Pick TODAY’s podcast/video topic.

- GPT-4o に “自然な日本語の会話シーン名 or 汎用ショート向けトピック” を 1 行だけリクエスト
- 旧形式（「<大テーマ> - <具体シーン>」や「◯◯英語/英会話」）でも
  自動的に “ホテルでのチェックイン会話” のような自然表現へ正規化（dialogue時）
- API 失敗時は SEED_TOPICS からフォールバック
- 生成は常に「日本語」→ main.py 側で各音声言語に翻訳（多言語コンボとの整合を保つ）

※ 学習寄り強化：
  dialogue / qa / howto / listicle のときは
  『<場面>で使える<ターゲット><終端>』の人気フォーマットで即時生成（ノーAPI）。
"""

import os
import re
import random
import datetime
from typing import List
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── フォールバック用プリセット（最初から自然な日本語） ───────────────
SEED_TOPICS: List[str] = [
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
# 正規化ユーティリティ（dialogue 用）
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
    first = (raw or "").strip().splitlines()[0] if raw else ""
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

    scene_out = f"{scene}の会話" if _needs_suffix(scene) else scene
    topic = f"{theme} {scene_out}"
    return re.sub(r'\s+', " ", topic).strip()

def _normalize(topic: str) -> str:
    """自然な会話トピックに正規化（dialogue 限定）"""
    t = topic or ""
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
    """自然な日本語の会話シーントピックを 1 行返す（dialogue 相当）。"""
    today = datetime.date.today().isoformat()
    prompt = (
        f"Today is {today}. "
        "日本語で、語学学習向けの“自然な会話シーン名”を1つだけ提案してください。"
        "例:『ホテルでのチェックイン会話』『空港での保安検査のやりとり』『レストランでの注文会話』。"
        "15〜20文字程度。句読点や引用符は不要。"
        "返答はそのフレーズ1行のみ。"
    )

    try:
        rsp = client.chat.completions.create(
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
# ✅ コンテンツタイプ別トピック生成（学習寄せ：「〜で使える〇〇」優先）
# ────────────────────────────────────────

# 学習フォーマット用の候補（国/文化に依存しない汎用シーンのみ）
_SCENES = [
    "自己紹介", "面接", "予約", "受付", "支払い", "道案内", "電話対応",
    "オンライン会議", "確認のやりとり", "依頼のやりとり", "謝罪", "予定変更",
    "レストランの注文", "空港のチェックイン", "ホテルのチェックイン",
]
_TARGETS = [
    "丁寧フレーズ", "基本表現", "依頼の言い方", "断り方", "確認フレーズ",
    "相槌", "クッション言葉", "時間の聞き方", "理由の伝え方",
]
_ENDINGS = ["3選", "一言", "言い方", "基本", "厳選"]

def _mk_learning_topic() -> str:
    """『<場面>で使える<ターゲット><終端>』で短く返す。"""
    scene  = random.choice(_SCENES)
    target = random.choice(_TARGETS)
    end    = random.choice(_ENDINGS)
    s = f"{scene}で使える{target}{end}"
    return s[:24]  # 日本語は長くなりがちなので軽く丸める

GUIDE = {
    "dialogue": "誰でも遭遇する生活/仕事シーン名（例: チェックイン会話、注文会話、道を尋ねるやりとり）。",
    "howto":    "30秒で実践できるコツ（例: 伝わりやすく話す3ステップ、集中力を上げるコツ3つ）。",
    "listicle": "3ポイントで学べるテーマ（例: 心を掴むコツ3つ、印象が良くなる言い回し3選）。",
    "wisdom":   "短い知恵・名言（例: 先延ばしを防ぐ一言、続けるための小さな仕組み）。",
    "fact":     "文化やコミュニケーションの豆知識（例: 相づちの違い、言葉の由来など）。",
    "qa":       "誤解されやすいQ&A（例: NG→OK→Proの言い換え、あるあるの勘違い）。",
}

def pick_by_content_type(content_type: str, audio_lang: str) -> str:
    """
    コンテンツ種別に応じて、誰でも刺さる“伸びる”ショート動画トピックを1行返す。
    - 生成は常に日本語（後段で各音声言語に翻訳するため）
    - dialogue / qa / howto / listicle → 『〜で使える〜』の学習フォーマットで即時生成（API不要）
    - wisdom / fact → GUIDE を使い GPT で1行生成（従来系）
    - dialogue のみに旧式トピックが来た場合の正規化（_normalize）
    """
    ct = (content_type or "dialogue").lower()

    # 学習フォーマットを優先（安定・短時間・低コスト）
    if ct in {"dialogue", "qa", "howto", "listicle"}:
        return _mk_learning_topic()

    # wisdom / fact は GPT で生成（短い1行）
    today = datetime.date.today().isoformat()
    guide = GUIDE.get(ct, GUIDE["dialogue"])
    prompt = (
        f"Today is {today}. "
        f"日本語で、{guide} "
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
        if ct == "dialogue":
            raw = _normalize(raw)
        if not raw or re.search(r"[A-Za-z]", raw):
            return random.choice(SEED_TOPICS)
        return raw[:24]
    except Exception:
        return random.choice(SEED_TOPICS)

# ────────────────────────────────────────
if __name__ == "__main__":
    print(pick())