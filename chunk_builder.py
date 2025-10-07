#!/usr/bin/env python3
"""
長尺 lines.json + full.mp3 + 背景 → チャンク分割して mp4 を作成し
最後に ffmpeg concat で 1 本に結合する。

usage:
  python chunk_builder.py temp/lines.json temp/full.mp3 temp/bg.png \
        --chunk 60 --rows 2 --fsize-top 65 --fsize-bot 60 \
        --out output/final_long.mp4
"""
import argparse
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from os import makedirs

from subtitle_video import build_video  # 既存の字幕つき動画生成関数

# ───────────────────── CLI ─────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("lines_json",  help="lines.json: [[spk, line1, line2, dur], ...]")
ap.add_argument("full_mp3",    help="通し音声ファイル (mp3)")
ap.add_argument("bg_png",      help="背景画像 (1920x1080 など)")
ap.add_argument("--out",       default="output/final.mp4", help="最終出力先 mp4")
ap.add_argument("--chunk",     type=int, default=40, help="1 チャンクあたりの行数")
ap.add_argument("--rows",      type=int, default=2,  help="字幕段数 (上段=音声言語, 下段=翻訳など)")
ap.add_argument("--fsize-top", type=int, default=None, help="上段字幕フォントサイズ")
ap.add_argument("--fsize-bot", type=int, default=None, help="下段字幕フォントサイズ")
# 追加: モノローグ(N)のラベル表示/配置オプション
ap.add_argument("--show-n-label", action="store_true",
                help="N(ナレーション)のラベルを表示したい場合に指定（デフォルトは非表示）")
ap.add_argument("--center-n", action="store_true",
                help="N(ナレーション)の字幕を中央寄せにする（推奨）")
args = ap.parse_args()

SCRIPT     = Path(args.lines_json)
FULL_MP3   = Path(args.full_mp3)
BG_PNG     = Path(args.bg_png)
FINAL_MP4  = Path(args.out)

LINES_PER  = args.chunk   # 分割チャンクサイズ
ROWS       = args.rows

if not (SCRIPT.exists() and FULL_MP3.exists() and BG_PNG.exists()):
    raise SystemExit("❌ 必要なファイルが見つかりません。引数を確認してください。")

# 出力先ディレクトリを用意
makedirs(FINAL_MP4.parent, exist_ok=True)

# ───────────────────── 処理開始 ─────────────────────
TEMP = Path(tempfile.mkdtemp(prefix="chunks_"))
print("🗂️  Temp dir =", TEMP)

# lines.json 読み込み: [[spk, line1, line2, dur], ...] の形
lines = json.loads(SCRIPT.read_text())

# lines.json を chunk ごとに分割
parts = [lines[i:i+LINES_PER] for i in range(0, len(lines), LINES_PER)]

# durations: 各行の秒数を読み取って累積和を作る
durations  = [row[-1] for row in lines]  # row[-1] は dur
cumulative = [0]
for d in durations:
    cumulative.append(cumulative[-1] + d)  # 累積

part_files = []

# ここで N 表示制御用のフラグをまとめる（subtitle_video へ渡す）
hide_n_label = not args.show_n_label
monologue_center = bool(args.center_n)

for idx, chunk in enumerate(parts):
    # start〜end の秒数を計算
    t_start = cumulative[idx * LINES_PER]
    t_end   = cumulative[idx * LINES_PER + len(chunk)]
    t_len   = t_end - t_start

    # チャンク用の音声 mp3
    audio_part = TEMP / f"audio_{idx}.mp3"
    # 出力 mp4
    mp4_part   = TEMP / f"part_{idx:02d}.mp4"

    # ffmpeg で通し音声(full.mp3)から必要部分だけ切り出し
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", f"{t_start}", "-t", f"{t_len}",
        "-i", str(FULL_MP3),
        "-acodec", "copy", str(audio_part)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    print(f"▶️ part {idx+1}/{len(parts)} | 行数={len(chunk)}"
          f" | start={t_start:.1f}s len={t_len:.1f}s")

    # フォントサイズなど可変指定をまとめる
    extra_args = {}
    if args.fsize_top:
        extra_args["fsize_top"] = args.fsize_top
    if args.fsize_bot:
        extra_args["fsize_bot"] = args.fsize_bot

    # 追加: N の表示制御オプションを渡す
    extra_args["hide_n_label"] = hide_n_label
    extra_args["monologue_center"] = monologue_center

    # 字幕つき動画を生成
    build_video(
        lines=chunk,
        bg_path=BG_PNG,
        voice_mp3=audio_part,
        out_mp4=mp4_part,
        rows=ROWS,
        **extra_args  # fsize_top, fsize_bot, hide_n_label, monologue_center
    )

    part_files.append(mp4_part)

# ───────────────────── concat ─────────────────────
concat_txt = TEMP / "concat.txt"
concat_txt.write_text("\n".join(f"file '{p.resolve()}'" for p in part_files))

subprocess.run([
    "ffmpeg", "-y",
    "-f", "concat", "-safe", "0",
    "-i", str(concat_txt),
    "-c", "copy", str(FINAL_MP4)
], check=True)

print("✅ 完了:", FINAL_MP4)

# 後始末（不要ならコメントアウトして残しても良い）
shutil.rmtree(TEMP)
print("🧹 Temp dir removed →", TEMP)
