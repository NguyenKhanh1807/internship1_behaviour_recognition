"""Gop cac luong ProtoGCN tren FineGYM val (trung binh xac suat, trong so bang nhau) tu ket qua cua eval_finegym.py.
  python ensemble_finegym.py [--clips 1]      -> runs/finegym_eval/ensemble_clips<n>.txt
Bai bao bao cao 95.9% (mean class acc) cho ban gop nhieu luong hon (co ca cac luong k ma ban port chua ho tro) -> chi de tham khao."""
import argparse
import itertools
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--clips", type=int, default=1); args = ap.parse_args()
out = Path("runs/finegym_eval"); y = np.load(out / "labels.npy")
S = {m: np.load(f).astype(np.float32) for m in ("j", "b", "jm", "bm")
     for f in [out / f"protogcn_finegym_{m}_clips{args.clips}_prob.npy"] if f.exists()}


def met(p):
    pred = p.argmax(1)
    return float((pred == y).mean()), float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


lines = [f"Gop luong ProtoGCN tren FineGYM val ({len(y)} video, {args.clips} clip/video), trong so bang nhau", "",
         f"{'luong / to hop':<16}{'top-1':>8}{'mean class acc':>16}"]
for k in range(1, len(S) + 1):
    for c in itertools.combinations(S, k):
        t, m = met(np.mean([S[x] for x in c], 0)); lines.append(f"{'+'.join(c):<16}{t:>8.4f}{m:>16.4f}")
lines += ["", "Bai bao (Table 2): 95.9% cho ban gop nhieu luong hon, gom ca cac luong chua co trong ban port."]
text = "\n".join(lines); print(text); (out / f"ensemble_clips{args.clips}.txt").write_text(text + "\n")
