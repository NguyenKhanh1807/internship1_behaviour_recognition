"""Gop (ensemble) cac luong ProtoGCN j / b / jm / bm, giong cach bai bao cong diem softmax nhieu luong.

Doc test_prob.npy cua cac lan finetune_protogcn.py (cung 210 video test, cung thu tu), trung binh xac suat,
bao cao P/R/F1/Acc/AUC + CI95 bootstrap, va McNemar so voi ST-GCN 1p.
Trong so bang nhau, nguong 0.5 (chua dung tap validation -> khong chon trong so/nguong tren tap test).

  python ensemble_streams.py                     # mac dinh: cac luong fine-tune tu FineGYM
  python ensemble_streams.py --prefix scratch_p100
Ket qua: runs/protogcn_ft/ensemble_<prefix>.txt
"""
import argparse
import itertools
from pathlib import Path

import numpy as np

from finetune_protogcn import bootstrap


def metrics(yt, prob):
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
    pred = (prob >= 0.5).astype(int)
    cf1, cauc = bootstrap(yt, pred, prob)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    return (f"{precision_score(yt, pred, zero_division=0):>6.2f}{recall_score(yt, pred):>6.2f}{f1_score(yt, pred):>6.2f} "
            f"{cf1:<12}{(pred == yt).mean():>6.2f}{roc_auc_score(yt, prob):>7.3f} {cauc:<12} {fp:>3}/{fn:<3}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="yolo26n-pose_my_botsort")
    ap.add_argument("--prefix", default="protogcn_finegym", help="ten lan chay truoc hau to luong (_j, _b, _jm, _bm)")
    args = ap.parse_args()

    base = Path("runs/protogcn_ft")
    y = None
    streams = {}
    for m in ("j", "b", "jm", "bm"):
        d = base / f"{args.cache}_{args.prefix}_{m}"
        if m == "j" and not d.exists() and args.prefix.startswith("scratch"):
            d = base / f"{args.cache}_{args.prefix}"            # lan scratch luong j khong co hau to
        if (d / "test_prob.npy").exists():
            streams[m] = np.load(d / "test_prob.npy")
            if y is None:
                import csv
                y = np.array([int(r["label"]) for r in csv.DictReader(open(d / "test_pred.csv"))])
    if len(streams) < 1:
        raise SystemExit(f"khong thay luong nao trong {base}/{args.cache}_{args.prefix}_*")
    other = {}
    for g, d in (("ST-GCN 1p", Path("runs/stgcn_ft") / args.cache), ("ST-GCN 2p", Path("runs/stgcn_ft") / (args.cache + "_2p"))):
        if (d / "test_prob.npy").exists():
            other[g] = np.load(d / "test_prob.npy")

    head = f"{'luong / to hop (cung 210 video test)':<36}{'P':>6}{'R':>6}{'F1':>6} {'CI95 F1':<12}{'Acc':>6}{'AUC':>7} {'CI95 AUC':<12} FP/FN"
    lines = [f"Ensemble ProtoGCN ({args.prefix}), trung binh xac suat, trong so bang nhau, nguong 0.5", "", head]
    for m, p in streams.items():
        lines.append(f"{'ProtoGCN ' + m:<36}" + metrics(y, p))
    combos = []
    for k in range(2, len(streams) + 1):
        for c in itertools.combinations(streams, k):
            p = np.mean([streams[m] for m in c], 0)
            combos.append(("+".join(c), p))
            lines.append(f"{'ProtoGCN ' + '+'.join(c):<36}" + metrics(y, p))
    for g, p in other.items():
        lines.append(f"{g:<36}" + metrics(y, p))
    if streams and other:
        allp = np.mean(list(streams.values()) + list(other.values()), 0)
        lines.append(f"{'ProtoGCN ' + '+'.join(streams) + ' + ST-GCN 1p/2p':<36}" + metrics(y, allp))
        combos.append(("+".join(streams) + "+stgcn", allp))

    # tuong quan giua cac luong: luong cang khac nhau thi gop cang co loi
    if len(streams) > 1:
        ks = list(streams)
        C = np.corrcoef([streams[k] for k in ks])
        lines += ["", "tuong quan xac suat giua cac luong: " + ", ".join(f"{a}-{b} {C[i, j]:.2f}"
                  for (i, a), (j, b) in itertools.combinations(enumerate(ks), 2))]
    if "ST-GCN 1p" in other and combos:
        from scipy.stats import binomtest
        best = max(combos, key=lambda c: ((c[1] >= .5) == y).mean())
        a = (best[1] >= .5) == y; b = (other["ST-GCN 1p"] >= .5) == y
        n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
        pv = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        lines += [f"McNemar {best[0]} (Acc cao nhat) vs ST-GCN 1p: dung/sai = {n01}, nguoc lai = {n10}, p = {pv:.3f}",
                  "(chon to hop tot nhat tren tap test -> lac quan; can validation hoac 5-fold de ket luan)"]

    text = "\n".join(lines)
    print(text)
    (base / f"ensemble_{args.prefix}.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
