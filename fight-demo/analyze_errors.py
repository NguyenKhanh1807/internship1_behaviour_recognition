"""Phan tich loi tren 210 video test: mo hinh nao sai video nao, loi chung / rieng, dac trung cua nhom FP / FN.

  python analyze_errors.py            # in bang + ghi runs/error_analysis/errors.csv
  python analyze_errors.py --sheets   # them anh ghep 3 frame / video cho FP va FN cua ProtoGCN bone (can cv2)
Chi doc test_pred.csv da co (khong chay lai mo hinh).
"""
import argparse
import csv
import statistics as S
from pathlib import Path

CACHE = "yolo26n-pose_my_botsort"
RUNS = {"proto_b": f"runs/protogcn_ft/{CACHE}_protogcn_finegym_b", "proto_j": f"runs/protogcn_ft/{CACHE}_protogcn_finegym_j",
        "proto_jm": f"runs/protogcn_ft/{CACHE}_protogcn_finegym_jm", "proto_bm": f"runs/protogcn_ft/{CACHE}_protogcn_finegym_bm",
        "stgcn_1p": f"runs/stgcn_ft/{CACHE}", "stgcn_2p": f"runs/stgcn_ft/{CACHE}_2p"}
COLS = ["n_frames", "persons_mean", "frac_2plus", "id_churn", "min_dist", "speed_p90", "motion_all_p90", "approach_p90"]


def load(d):
    return {r["video"]: (int(r["label"]), float(r["p_fight"])) for r in csv.DictReader(open(Path(d) / "test_pred.csv"))}


def sheet(names, out, data="datasets"):
    import cv2
    import numpy as np
    rows = []
    for n in names:
        vid = next((Path(data) / ("NonViolence" if n.startswith("NV") else "Violence")).glob(n + ".*"))
        cap = cv2.VideoCapture(str(vid)); T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); tiles = []
        for q in (0.2, 0.5, 0.8):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(T * q)); ok, f = cap.read()
            if not ok:
                f = np.zeros((180, 320, 3), np.uint8)
            h, w = f.shape[:2]; s = min(320 / w, 200 / h); f = cv2.resize(f, (int(w * s), int(h * s)))
            c = np.zeros((200, 320, 3), np.uint8); y = (200 - f.shape[0]) // 2; x = (320 - f.shape[1]) // 2
            c[y:y + f.shape[0], x:x + f.shape[1]] = f; tiles.append(c)
        r = np.hstack(tiles); cv2.putText(r, n, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2); rows.append(r)
    cv2.imwrite(str(out), np.vstack(rows))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sheets", action="store_true"); args = ap.parse_args()
    M = {k: load(d) for k, d in RUNS.items() if (Path(d) / "test_pred.csv").exists()}
    vids = list(M["proto_b"])
    F = {r["video"]: r for r in csv.DictReader(open(f"runs/rf/{CACHE}/features.csv"))}
    wrong = {k: {v for v in vids if (m[v][1] >= .5) != bool(m[v][0])} for k, m in M.items()}
    out = Path("runs/error_analysis"); out.mkdir(parents=True, exist_ok=True)
    b, s = wrong["proto_b"], wrong["stgcn_1p"]
    lines = [f"{len(vids)} video test. So video sai: " + ", ".join(f"{k} {len(w)}" for k, w in wrong.items()),
             f"proto_b va stgcn_1p cung sai: {len(b & s)} | chi proto_b sai: {len(b - s)} | chi stgcn_1p sai: {len(s - b)}",
             f"sai o MOI mo hinh: {sorted(set.intersection(*wrong.values()))}",
             f"sai o it nhat 1 mo hinh: {len(set.union(*wrong.values()))}", "", "trung vi dac trung theo nhom (ProtoGCN bone):"]
    lab = lambda v: M["proto_b"][v][0]
    groups = {"danh nhau, doan dung": [v for v in vids if v not in b and lab(v)], "danh nhau, bo sot (FN)": [v for v in b if lab(v)],
              "binh thuong, doan dung": [v for v in vids if v not in b and not lab(v)], "binh thuong, bao sai (FP)": [v for v in b if not lab(v)]}
    for g, vs in groups.items():
        lines.append(f"  {g:<26} n={len(vs):>3}  " + "  ".join(f"{c}={S.median([float(F[v][c]) for v in vs if v in F]):.2f}" for c in COLS))
    with open(out / "errors.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["video", "label"] + [f"p:{k}" for k in M] + ["so_mo_hinh_sai"] + COLS)
        for v in sorted(set.union(*wrong.values()), key=lambda v: (-sum(v in x for x in wrong.values()), v)):
            w.writerow([v, lab(v)] + [M[k][v][1] for k in M] + [sum(v in x for x in wrong.values())] + [round(float(F[v][c]), 3) if v in F else "" for c in COLS])
    text = "\n".join(lines); print(text); (out / "summary.txt").write_text(text + "\n")
    if args.sheets:
        sheet(sorted(v for v in b if not lab(v)), out / "proto_b_FP.jpg"); sheet(sorted(v for v in b if lab(v)), out / "proto_b_FN.jpg")
    print(f"\nDa luu {out}/")


if __name__ == "__main__":
    main()
