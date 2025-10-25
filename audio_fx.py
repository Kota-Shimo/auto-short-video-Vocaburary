# audio_fx.py – フィルタ自動フォールバック版（FFmpeg最小構成でも落ちない）
import subprocess, shutil
from pathlib import Path

# ベースEQ/コンプ（軽量で互換性高いものだけ使用）
BASE_CHAIN = (
    "highpass=f=60,"                                  # 低域ノイズ除去
    "lowpass=f=10500,"                                # 高域ノイズ抑制
    "equalizer=f=4000:width_type=h:width=150:g=3,"    # プレゼンス付与
    "equalizer=f=8000:width_type=h:width=300:g=-2,"   # 軽いデエッシング相当
    "acompressor=threshold=-18dB:ratio=2:knee=2:attack=15:release=200"  # ソフト圧縮
)
BASE_CHAIN = "".join(BASE_CHAIN)

# 試行候補のフィルタチェーン（上から順に試す）
TRY_CHAINS = [
    f"{BASE_CHAIN},loudnorm=I=-16:TP=-1.5:LRA=11",   # 1) 標準
    f"{BASE_CHAIN},dynaudnorm=f=150:g=7",            # 2) 代替ノーマライズ
    f"{BASE_CHAIN},volume=-1.5dB"                    # 3) 最低限の音量調整
]

def _run_ffmpeg_filter(inp: Path, out: Path, chain: str) -> subprocess.CompletedProcess:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(inp),
        "-af", chain,
        "-ar", "48000",          # 48kHzに統一
        "-c:a", "libmp3lame", "-q:a", "2",
        str(out)
    ]
    # stderrを捕まえてデバッグしやすく
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def enhance(in_mp3: Path, out_mp3: Path):
    """
    in_mp3  : 入力 mp3
    out_mp3 : 整音後 mp3（必ず作る）
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg が見つかりません。PATH を確認してください。")

    in_mp3 = Path(in_mp3)
    out_mp3 = Path(out_mp3)
    out_mp3.parent.mkdir(parents=True, exist_ok=True)

    # 1)～3) を順に試す
    last_err = None
    for chain in TRY_CHAINS:
        proc = _run_ffmpeg_filter(in_mp3, out_mp3, chain)
        if proc.returncode == 0:
            return  # 成功
        last_err = proc
        # 失敗理由をログ出力（Actionsのログで確認可能）
        print(f"[audio_fx] FFmpeg failed with chain: {chain}")
        print(proc.stderr)

    # すべて失敗 → フォールバック：再エンコードのみ（フィルタ無し）
    print("[audio_fx] All filter chains failed. Falling back to plain re-encode.")
    plain = [
        "ffmpeg", "-y",
        "-i", str(in_mp3),
        "-ar", "48000",
        "-c:a", "libmp3lame", "-q:a", "2",
        str(out_mp3)
    ]
    proc2 = subprocess.run(plain, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc2.returncode != 0:
        # ここまで来て失敗なら、stderrを添えて例外化
        err_msg = proc2.stderr or (last_err.stderr if last_err else "")
        raise RuntimeError(f"ffmpeg 最終フォールバックも失敗しました（exit={proc2.returncode}）。\n{err_msg}")