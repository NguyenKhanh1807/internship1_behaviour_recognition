"""5-fold cho them 2 luong ProtoGCN (joint-motion, bone-motion), roi bao cao ca 5 mo hinh:
protogcn_b, protogcn_j, protogcn_jm, protogcn_bm, stgcn_1p (cac fold da xong cua b, j, stgcn duoc bo qua).
Ket qua 3 mo hinh truoc do da duoc luu lai o runs/cv/<cache>/summary_3mohinh.txt va oof_pred_3mohinh.csv.
Chay lai file nay = chay tiep tu fold dang do.
"""
import subprocess
import sys

a = ["cv_models.py", "--models", "protogcn_b,protogcn_j,protogcn_jm,protogcn_bm,stgcn_1p"]
print(">>> python " + " ".join(a), flush=True); subprocess.run([sys.executable] + a, check=True)
print("\nXONG.", flush=True)
