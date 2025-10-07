# ================= subtitle_video.py =================
from moviepy import (
    ImageClip, TextClip, AudioFileClip, ColorClip, concatenate_videoclips
)
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
import os, unicodedata as ud, re, textwrap
from pathlib import Path

# ---- フォントパス ------------------------------------------------------------
FONT_DIR  = Path(__file__).parent / "fonts"
FONT_LATN = str(FONT_DIR / "RobotoSerif_36pt-Bold.ttf")
FONT_JP   = str(FONT_DIR / "NotoSansJP-Bold.ttf")
FONT_KO   = str(FONT_DIR / "malgunbd.ttf")

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

# ---- 画面レイアウト ----------------------------------------------------------
SCREEN_W, SCREEN_H = 1080, 1920
TEXT_W_DEFAULT     = 880
TEXT_W_MONO        = 980   # モノローグ時は気持ち広め
FONT_SIZE_TOP_DEF  = 50    # 上段（音声言語）
FONT_SIZE_BOT_DEF  = 45    # 下段（翻訳字幕）
LINE_GAP           = 28
POS_Y              = 920
BOTTOM_MARGIN      = 40
PAD_X, PAD_Y       = 22, 16

# 既存の左寄せ寄りセンタリング用シフト（会話時にラベル分だけ少し左へ）
SHIFT_X_DIALOGUE   = -45

def xpos(width: int, centered: bool) -> int:
    """中央 or ラベル考慮の軽い左寄せ寄りセンター"""
    shift = 0 if centered else SHIFT_X_DIALOGUE
    return (SCREEN_W - width) // 2 + shift

# ---- CJK 折り返し ------------------------------------------------------------
def wrap_cjk(text: str, width: int = 16) -> str:
    """日本語や漢字のみの文を width 文字で手動改行"""
    if re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", text):
        return "\n".join(textwrap.wrap(text, width, break_long_words=True))
    return text

# ---- 半透明黒帯 --------------------------------------------------------------
def _bg(txt: TextClip) -> ColorClip:
    return (
        ColorClip((txt.w + PAD_X * 2, txt.h + PAD_Y * 2), (0, 0, 0))
        .with_opacity(0.55)
    )

# ---- メイン：字幕付き動画をビルド -------------------------------------------
def build_video(
    lines,
    bg_path,
    voice_mp3,
    out_mp4,
    rows: int = 2,
    fsize_top: int | None = None,
    fsize_bot: int | None = None,
    hide_n_label: bool = True,
    monologue_center: bool = False
):
    """
    lines : [(speaker, row1_text, row2_text, duration_sec), ...]
    rows  : 1 = 上段のみ / 2 = 上段+下段
    hide_n_label    : True のとき N(ナレーション)のラベルを描画しない
    monologue_center: True のとき N の本文ブロックを完全センター＆幅広に
    """
    # 背景
    bg_base = ImageClip(bg_path).resized((SCREEN_W, SCREEN_H))
    clips = []

    # フォントサイズ確定
    FS_TOP = fsize_top or FONT_SIZE_TOP_DEF
    FS_BOT = fsize_bot or FONT_SIZE_BOT_DEF

    for row in lines:
        # 行の展開（[(spk, sub1, sub2, dur)]）
        if rows >= 2:
            speaker, row1, row2, dur = row[0], row[1], row[2], row[-1]
        else:
            speaker, row1, dur = row[0], row[1], row[-1]
            row2 = None

        is_narration = (speaker == "N")
        centered = (is_narration and monologue_center)

        # ---- 上段テキスト（話者ラベルの有無）---------------------------------
        body_top  = wrap_cjk(row1)
        if is_narration and hide_n_label:
            top_txt = body_top
        else:
            top_txt = f"{speaker}: {body_top}"

        text_w = TEXT_W_MONO if centered else TEXT_W_DEFAULT

        top_clip = TextClip(
            text=top_txt,
            font=pick_font(body_top),
            font_size=FS_TOP,
            color="white", stroke_color="black", stroke_width=4,
            method="caption",
            size=(text_w, None),
            align="center" if centered else "West"
        )
        top_bg   = _bg(top_clip)

        elem = [
            top_bg  .with_position((xpos(top_bg.w, centered),  POS_Y - PAD_Y)),
            top_clip.with_position((xpos(top_clip.w, centered), POS_Y)),
        ]
        block_h = top_bg.h

        # ---- 下段テキスト -----------------------------------------------------
        if rows >= 2 and row2 is not None and str(row2).strip():
            body_bot = wrap_cjk(str(row2)) + "\n "
            bot_clip = TextClip(
                text=body_bot,
                font=pick_font(body_bot),
                font_size=FS_BOT,
                color="white", stroke_color="black", stroke_width=4,
                method="caption",
                size=(text_w, None),
                align="center" if centered else "West"
            )
            bot_bg = _bg(bot_clip)
            y_bot  = POS_Y + top_bg.h + LINE_GAP
            elem += [
                bot_bg  .with_position((xpos(bot_bg.w, centered),  y_bot - PAD_Y)),
                bot_clip.with_position((xpos(bot_clip.w, centered), y_bot)),
            ]
            block_h += LINE_GAP + bot_bg.h

        # ---- はみ出し補正（下端にかからないように全体を上へ）---------------
        overflow = POS_Y + block_h + BOTTOM_MARGIN - SCREEN_H
        if overflow > 0:
            elem = [c.with_position((c.pos(0)[0], c.pos(0)[1] - overflow)) for c in elem]

        # ---- 合成 --------------------------------------------------------------
        comp = CompositeVideoClip([bg_base, *elem]).with_duration(dur)
        clips.append(comp)

    # ---- 連結 & オーディオ合成 ------------------------------------------------
    video = concatenate_videoclips(clips, method="compose") \
              .with_audio(AudioFileClip(voice_mp3))
    video.write_videofile(out_mp4, fps=30, codec="libx264", audio_codec="aac")
# =====================================================