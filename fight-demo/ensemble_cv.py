"""Gop mo hinh tren du doan out-of-fold cua 5-fold (runs/cv/<cache>/oof_pred.csv), TRUNG BINH XAC SUAT, NGUONG CO DINH 0.5.
Khac cv_models.py --report (hoc nguong tren validation ~126 video, khong on dinh), o day khong hoc gi them nen khong can validation.
McNemar: so voi stgcn_1p (nguong 0.5) tren ca 1049 video. Chi dung thu vien chuan.
  python ensemble_cv.py      -> runs/cv/<cache>/ensemble_nguong05.txt
Luu y: day la phan tich them sau khi da co ket qua (nhieu to hop duoc thu) -> p-value chua hieu chinh so sanh nhieu lan.
"""
import csv
import math
import statistics as S
from pathlib import Path

out = Path("runs/cv/yolo26n-pose_my_botsort")
R = list(csv.DictReader(open(out / "oof_pred.csv")))
y = [int(r["label"]) for r in R]; fold = [int(r["fold"]) for r in R]
NAMES = {"b": "protogcn_b", "j": "protogcn_j", "jm": "protogcn_jm", "bm": "protogcn_bm", "st": "stgcn_1p"}
P = {k: [float(r[f"p:{v} (nguong 0.5)"]) for r in R] for k, v in NAMES.items() if f"p:{v} (nguong 0.5)" in R[0]}


def f1(pred, yy):
    tp = sum(a and b for a, b in zip(pred, yy)); fp = sum(a and not b for a, b in zip(pred, yy)); fn = sum((not a) and b for a, b in zip(pred, yy))
    return 2 * tp / max(2 * tp + fp + fn, 1), fp, fn


def auc(p):
    pos = [a for a, b in zip(p, y) if b]; neg = [a for a, b in zip(p, y) if not b]
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def binom_p(a, b):
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


st = [x >= .5 for x in P["st"]]
COMBOS = [("stgcn_1p", ("st",)), ("protogcn_b", ("b",)), ("protogcn_j", ("j",)), ("protogcn_jm", ("jm",)), ("protogcn_bm", ("bm",)),
          ("b+j", ("b", "j")), ("b+st", ("b", "st")), ("j+st", ("j", "st")), ("b+j+st", ("b", "j", "st")),
          ("4 luong ProtoGCN (b+j+jm+bm)", ("b", "j", "jm", "bm")), ("5 mo hinh (b+j+jm+bm+st)", ("b", "j", "jm", "bm", "st"))]
lines = [f"Gop mo hinh tren 5-fold OOF ({len(y)} video), trung binh xac suat, nguong 0.5", "",
         f"{'mo hinh / to hop':<30}{'F1 tb+-std':>14}{'F1 OOF':>8}{'AUC OOF':>9}{'FP/FN':>8}   McNemar vs stgcn_1p"]
for name, ks in COMBOS:
    if any(k not in P for k in ks):
        continue
    p = [sum(P[k][i] for k in ks) / len(ks) for i in range(len(y))]; pred = [x >= .5 for x in p]
    per = [f1([pred[i] for i in range(len(y)) if fold[i] == f], [y[i] for i in range(len(y)) if fold[i] == f])[0] for f in sorted(set(fold))]
    F, fp, fn = f1(pred, y)
    a = sum((pr == bool(t)) and (s != bool(t)) for pr, s, t in zip(pred, st, y)); b = sum((pr != bool(t)) and (s == bool(t)) for pr, s, t in zip(pred, st, y))
    mc = "-" if ks == ("st",) else f"dung/ST-GCN sai {a}, nguoc lai {b}, p = {binom_p(a, b):.4f}"
    lines.append(f"{name:<30}{S.mean(per):>8.3f}+-{S.pstdev(per):.3f}{F:>8.3f}{auc(p):>9.3f}{f'{fp}/{fn}':>8}   {mc}")
text = "\n".join(lines); print(text); (out / "ensemble_nguong05.txt").write_text(text + "\n")
