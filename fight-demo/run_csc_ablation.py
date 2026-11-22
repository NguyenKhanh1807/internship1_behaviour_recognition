"""Ablation loss CSC cua ProtoGCN tren RLVS (luong bone, cung seed / du lieu / so epoch voi lan chay goc lambda = 0.2):
  1. lambda = 0   : tat CSC, chi con cross-entropy (nhanh PRN khong nhan gradient) -> co che prototype co giup khong?
  2. lambda = 0.3 : gia tri toi uu bai bao bao cao tren NTU-120
Ket qua: runs/protogcn_ft/yolo26n-pose_my_botsort_protogcn_finegym_b_csc0/ va ..._csc0.3/ (summary.txt co san bang so sanh)
"""
import subprocess
import sys

for w in ("0", "0.3"):
    a = ["finetune_protogcn.py", "--init", "../debug/ckpt/protogcn_finegym_b.pth", "--csc-weight", w]
    print("\n>>> python " + " ".join(a), flush=True); subprocess.run([sys.executable] + a, check=True)
print("\nXONG.", flush=True)
