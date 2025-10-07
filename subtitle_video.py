# ================= subtitle_video.py =================
from moviepy import (
    ImageClip, TextClip, AudioFileClip, ColorClip, concatenate_videoclips
)
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
import os, unicodedata as ud, re, textwrap
from pathlib import Path
FONT_DIR  = Path(__file__).parent / "fonts"
FONT_LATN = str(FONT_DIR / "RobotoSerif_36pt-Bold.ttf")
FONT_JP   = str(FONT_DIR / "NotoSansJP-Bold.ttf")
FONT_KO   = str(FONT_DIR / "malgunbd.ttf")

# ── X 位置ずらし用 ───────────────────────────────
SHIFT_X = -45
def xpos(w: int) -> int:
    return (SCREEN_W - w) // 2 + SHIFT_X

# 中央寄せ用（モノローグ時のみ使用）
def xpos_center(w: int) -> int:
    return (SCREEN_W - w) // 2
# ────────────────────────────────────────────────

# ── CJK 折り返し ────────────────────────────────
def wrap_cjk(text: str, width: int = 16) -> str:
    """日本語や漢字のみの文を width 文字で手動改行"""
    if re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", text):
        return "\n".join(textwrap.wrap(text, width, break_long_words=True))
    return text
# ────────────────────────────────────────────────

for f in (FONT_LATN, FONT_JP, FONT_KO):
    if not os.path.isfile(f):
        raise FileNotFoundError(f"Font not found: {f}")

def pick_font(text: str) -> str:
    """文字種を見て適切なフォントパスを返す"""
    for ch in text:
        name = ud.name(ch, "")
        if "HANGUL" in name:
            return FONT_KO
        if any(tag in name for tag in ("CJK", "HIRAGANA", "KATAKANA")):
            return FONT_JP
    return FONT_LATN
# =============================

# ===== レイアウト定数 =====
FONT_SIZE_TOP      = 50   # 上段（音声言語）
FONT_SIZE_BOT      = 45   # 下段（翻訳字幕）
LINE_GAP           = 28
POS_Y              = 920
TEXT_W             = 880
SCREEN_W, SCREEN_H = 1080, 1920
BOTTOM_MARGIN      = 40
PAD_X, PAD_Y       = 22, 16
# ===========================

# ── 半透明黒帯 ──────────────────────────────────
def _bg(txt: TextClip) -> ColorClip:
    return (
        ColorClip((txt.w + PAD_X * 2, txt.h + PAD_Y * 2), (0, 0, 0))
        .with_opacity(0.55)
    )
# ────────────────────────────────────────────────

def build_video(
    lines,
    bg_path,
    voice_mp3,
    out_mp4,
    rows: int = 2,
    # ▼ 受け口だけ追加（最小変更）
    fsize_top=None,
    fsize_bot=None,
    hide_n_label: bool = True,
    monologue_center: bool = False
):
    """
    lines : [(speaker, row1_text, row2_text, duration_sec), ...]
    rows  : 1 = 上段のみ / 2 = 上段+下段
    hide_n_label    : True のとき N のラベルを表示しない
    monologue_center: True のとき N の本文ブロックを中央寄せ配置
    """
    bg_base = ImageClip(bg_path).resized((SCREEN_W, SCREEN_H))
    clips = []

    # フォントサイズ（指定があれば優先）
    FS_TOP = fsize_top or FONT_SIZE_TOP
    FS_BOT = fsize_bot or FONT_SIZE_BOT

    for speaker, *row_texts, dur in lines:
        is_n = (speaker == "N")
        posf = xpos_center if (is_n and monologue_center) else xpos

        # ---------- 上段 ----------
        top_body = wrap_cjk(row_texts[0])
        # N のときラベルを隠す（テキストはそのまま）
        top_txt  = top_body if (is_n and hide_n_label) else f"{speaker}: {top_body}"
        top_clip = TextClip(
            text=top_txt,
            font=pick_font(top_body),                  # ← 言語別フォント
            font_size=FS_TOP,
            color="white", stroke_color="black", stroke_width=4,
            method="caption", size=(TEXT_W, None),
        )
        top_bg   = _bg(top_clip)

        elem = [
            top_bg  .with_position((posf(top_bg.w),  POS_Y - PAD_Y)),
            top_clip.with_position((posf(top_clip.w), POS_Y)),
        ]
        block_h = top_bg.h

        # ---------- 下段 ----------
        if rows >= 2:
            bot_body = wrap_cjk(row_texts[1]) + "\n "
            bot_clip = TextClip(
                text=bot_body,
                font=pick_font(bot_body),              # ← 言語別フォント
                font_size=FS_BOT,
                color="white", stroke_color="black", stroke_width=4,
                method="caption", size=(TEXT_W, None),
            )
            bot_bg = _bg(bot_clip)
            y_bot  = POS_Y + top_bg.h + LINE_GAP
            elem += [
                bot_bg  .with_position((posf(bot_bg.w),  y_bot - PAD_Y)),
                bot_clip.with_position((posf(bot_clip.w), y_bot)),
            ]
            block_h += LINE_GAP + bot_bg.h

        # ---------- はみ出し補正 ----------
        overflow = POS_Y + block_h + BOTTOM_MARGIN - SCREEN_H
        if overflow > 0:
            elem = [c.with_position((c.pos(0)[0], c.pos(0)[1] - overflow)) for c in elem]

        # ---------- 合成 ----------
        comp = CompositeVideoClip([bg_base, *elem]).with_duration(dur)
        clips.append(comp)

    video = concatenate_videoclips(clips, method="compose") \
              .with_audio(AudioFileClip(voice_mp3))
    video.write_videofile(out_mp4, fps=30, codec="libx264", audio_codec="aac")
# =====================================================