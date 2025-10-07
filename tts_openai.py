"""OpenAI TTS wrapper – language-aware & two-speaker support."""

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


def speak(lang: str, speaker: str, text: str, out_path: Path):
    """
    lang     : 'en', 'ja', 'pt', 'id' など
    speaker  : 'Alice' / 'Bob' / 'N'（N=ナレーション）
    text     : セリフ
    out_path : 書き出し先 .mp3
    """
    clean_text = _clean_for_tts(text, lang)

    v_a, v_b = VOICE_MAP.get(lang, FALLBACK_VOICES)
    spk = (speaker or "").lower()

    # ✅ N を既存ボイスに割り当て（ここだけが今回の追加）
    # N を Alice と同じボイスにしたい場合：
    voice_id = v_a if spk in ("alice", "n") else v_b

    # ※ Bob に合わせたい場合は ↓ のようにする
    # voice_id = v_b if spk in ("bob", "n") else v_a

    resp = client.audio.speech.create(
        model="tts-1",          # 高音質は "tts-1-hd"
        voice=voice_id,
        input=clean_text,
    )
    out_path.write_bytes(resp.content)
