# audio_fx.py – “良いマイク風” (deesser 非依存バージョン)
import subprocess, shutil
from pathlib import Path

# -----------------------------------------------------------
# FILTER chain
#   1) highpass 60 Hz         : 空調/机振動カット
#   2) lowpass  10.5 kHz      : モスキートノイズ抑制
#   3) presence EQ 4 kHz +3dB : 明瞭度
#   4) soft de-ess  8 kHz −2dB: 歯擦音をやや抑える (simple EQ)
#   5) soft compressor        : ratio 2:1 で自然に
#   6) loudnorm (-16 LUFS)    : ポッドキャスト標準ラウドネス
FILTER = (
    "highpass=f=60,"
    "lowpass=f=10500,"
    "equalizer=f=4000:width_type=h:width=150:g=3,"
    "equalizer=f=8000:width_type=h:width=300:g=-2,"
    "acompressor=threshold=-18dB:ratio=2:knee=2:attack=15:release=200,"
    "loudnorm=I=-16:TP=-1.5:LRA=11"
)
# -----------------------------------------------------------

def enhance(in_mp3: Path, out_mp3: Path):
    """
    in_mp3  : 入力 mp3
    out_mp3 : 整音後 mp3
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg が見つかりません。PATH を確認してください。")

    cmd = [
        "ffmpeg", "-y", "-i", str(in_mp3),
        "-af", FILTER,
        "-ar", "48000",                # 48 kHz に統一（必要に応じて 44100）
        str(out_mp3)
    ]

    # 実行＆フォールバック処理
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        print("⚠️ deesser や一部フィルタが使えない可能性があります。loudnorm のみにフォールバックします。")
        subprocess.run([
            "ffmpeg", "-y", "-i", str(in_mp3),
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ar", "48000",
            str(out_mp3)
        ], check=True)