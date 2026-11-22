"""Kiem chung ban port tren FineGYM, chay lien tiep (chay lai file nay = chay tiep, buoc da xong chi in lai):
  1. luong b, jm, bm: 1 clip / video        2. gop 4 luong (1 clip)
  3. cho 5-fold (runs/cv/.../summary.txt) xong roi moi chay luong j voi 10 clip / video (nang, de khong tranh GPU)
"""
import subprocess
import sys
import time
from pathlib import Path

CK = "../debug/ckpt/protogcn_finegym_{}.pth"


def run(*a):
    print("\n>>> python " + " ".join(a), flush=True); subprocess.run([sys.executable, *a], check=True)


for m in ("j", "b", "jm", "bm"):
    run("eval_finegym.py", "--ckpt", CK.format(m))
run("ensemble_finegym.py")
cv, t0 = Path("runs/cv/yolo26n-pose_my_botsort/summary.txt"), time.time()
while not cv.exists() and time.time() - t0 < 3 * 3600:
    print(f"cho 5-fold xong roi chay ban 10 clip... ({(time.time() - t0) / 60:.0f} phut)", flush=True); time.sleep(120)
run("eval_finegym.py", "--ckpt", CK.format("j"), "--clips", "10")
print("\nXONG.", flush=True)
