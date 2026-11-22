"""Chay lien tiep 3 buoc (moi buoc tu chay tiep neu bi ngat, chay lai file nay la duoc):
  1. ProtoGCN luong bone-motion (bm), train tu dau vi chua co checkpoint FineGYM cho luong bm
  2. Gop 2 luong train tu dau (j + bm)
  3. Danh gia 5-fold + validation: protogcn_b, protogcn_j, stgcn_1p (cv_models.py)
Ket qua: runs/protogcn_ft/yolo26n-pose_my_botsort_scratch_p100_bm/, runs/protogcn_ft/ensemble_scratch_p100.txt,
         runs/cv/yolo26n-pose_my_botsort/summary.txt
"""
import subprocess
import sys

STEPS = [
    ["finetune_protogcn.py", "--init", "scratch", "--modality", "bm"],
    ["ensemble_streams.py", "--prefix", "scratch_p100"],
    ["cv_models.py"],
]

for i, s in enumerate(STEPS, 1):
    print(f"\n>>> buoc {i}/{len(STEPS)}: python {' '.join(s)}", flush=True)
    subprocess.run([sys.executable] + s, check=True)
print("\nXONG ca 3 buoc.", flush=True)
