"""Chay lien tiep 2 buoc:
  1. Fine-tune ProtoGCN luong bone-motion tu checkpoint FineGYM (debug/ckpt/protogcn_finegym_bm.pth)
  2. Gop 4 luong fine-tune tu FineGYM (j + b + jm + bm)
Ket qua: runs/protogcn_ft/yolo26n-pose_my_botsort_protogcn_finegym_bm/, runs/protogcn_ft/ensemble_protogcn_finegym.txt
"""
import subprocess
import sys

STEPS = [
    ["finetune_protogcn.py", "--init", "../debug/ckpt/protogcn_finegym_bm.pth"],
    ["ensemble_streams.py"],
]

for i, s in enumerate(STEPS, 1):
    print(f"\n>>> buoc {i}/{len(STEPS)}: python {' '.join(s)}", flush=True)
    subprocess.run([sys.executable] + s, check=True)
print("\nXONG ca 2 buoc.", flush=True)
