"""Giai doan 4a: hoc phan loai danh nhau tu SKELETON bang Random Forest (thay cho nguong dat tay).

Moi video -> 1 vector dac trung (tong hop tu cac cap nguoi qua cac frame), dung lai keypoints.csv trong cache
cua eval_videos.py (khong chay lai YOLO). Danh gia bang cross-validation 5-fold lap lai, so voi heuristic.

Cach chay (trong fight-demo, venv da bat; can:  pip install scikit-learn):
    python train_rf.py                                   # cache mac dinh YOLO26n + BoT-SORT ReID
    python train_rf.py --cache runs/eval_cache/yolo11n-pose_tracktrack
Ket qua: runs/rf/<ten cache>/ (summary.txt, features.csv, importance.csv)
"""

import argparse
import csv
from collections import defaultdict, deque
from itertools import combinations
from pathlib import Path

import numpy as np

from fight_demo import approach_speed, body_center, center_speed, gap_ok, limb_speed, update_pairs
from replay_heuristic import heuristic_parser, load_frames

FEATURES = [
    "n_frames", "persons_mean", "frac_2plus", "id_churn",          # muc do nhin thay / theo vet nguoi
    "frac_close", "min_dist",                                     # khoang cach giua cac cap
    "speed_p90", "speed_max", "frac_close_fast",                  # toc do tay/chan
    "approach_p90", "approach_max",                               # tay/chan lao ve phia nguoi kia
    "impact_p90", "impact_max",                                   # than nguoi kia bi dich chuyen
    "longest_streak_s", "events_base", "events_impact",           # chinh heuristic, dung lam dac trung
    "motion_all_p90",                                             # chuyen dong chung cua ca canh
]


def video_features(kp_csv: Path, fps: float) -> list:
    frames, _ = load_frames(str(kp_csv))
    hist = defaultdict(lambda: deque(maxlen=2))
    ids_all, n_pers, dists, speeds, apps, imps, motion = set(), [], [], [], [], [], []
    close_frames = close_fast = streak_best = 0
    streak = defaultdict(int)
    order = sorted(frames)
    for f in order:
        ids, kp, hs = frames[f]
        hmap = dict(zip(ids, hs))
        ids_all.update(ids)
        n_pers.append(len(ids))
        for t, k in zip(ids, kp):
            hist[t].append((f, k))
            if gap_ok(hist[t], 1):
                motion.append(limb_speed(hist[t][-2][1], k, hmap[t]))
        any_close = any_fast = False
        active = set()
        for a, b in combinations(sorted(ids), 2):
            if not (gap_ok(hist[a], 1) and gap_ok(hist[b], 1)):
                continue
            ca, cb = body_center(hist[a][-1][1]), body_center(hist[b][-1][1])
            if ca is None or cb is None:
                continue
            h = (hmap[a] + hmap[b]) / 2
            d = float(np.linalg.norm(ca - cb) / h)
            dists.append(d)
            if d >= 1.2:
                continue
            any_close = True
            sp = max(limb_speed(hist[a][-2][1], hist[a][-1][1], hmap[a]), limb_speed(hist[b][-2][1], hist[b][-1][1], hmap[b]))
            if sp > 3.0:  # nhay keypoint / doi ID, khong phai chuyen dong that
                continue
            speeds.append(sp)
            apps.append(max(approach_speed(hist[a], cb, hmap[a]), approach_speed(hist[b], ca, hmap[b])))
            imps.append(max(center_speed(hist[a], hmap[a]), center_speed(hist[b], hmap[b])))
            if sp > 0.08:
                any_fast = True
                streak[(a, b)] += 1
                active.add((a, b))
                streak_best = max(streak_best, streak[(a, b)])
        for p in list(streak):
            if p not in active:
                streak[p] = 0
        close_frames += any_close
        close_fast += any_fast

    def stat(x, q):
        return float(np.percentile(x, q)) if len(x) else 0.0

    def events(**kw):
        h = argparse.Namespace(**{**vars(heuristic_parser().parse_args([])), **kw})
        hh, st = defaultdict(lambda: deque(maxlen=2)), defaultdict(int)
        return sum(len(update_pairs(f, frames[f][0], np.stack(frames[f][1]), np.array(frames[f][2]), hh, st, h, 5)) for f in order)

    nf = max(len(order), 1)
    return [
        len(order), float(np.mean(n_pers)) if n_pers else 0.0, float(np.mean([n >= 2 for n in n_pers])) if n_pers else 0.0,
        len(ids_all) / max(np.mean(n_pers) if n_pers else 1, 1),
        close_frames / nf, float(min(dists)) if dists else 9.0,
        stat(speeds, 90), stat(speeds, 100), close_fast / nf,
        stat(apps, 90), stat(apps, 100),
        stat(imps, 90), stat(imps, 100),
        streak_best / fps, events(), events(impact=0.02),
        stat(motion, 90),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="runs/eval_cache/yolo26n-pose_my_botsort")
    ap.add_argument("--data", default="datasets", help="thu muc chua Violence/ NonViolence (de lay fps)")
    ap.add_argument("--repeats", type=int, default=10, help="so lan lap lai 5-fold CV voi seed khac nhau")
    args = ap.parse_args()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.model_selection import StratifiedKFold

    cache = Path(args.cache)
    rows, X, y = [], [], []
    for sub, lab, folder in (("pos", 1, "Violence"), ("neg", 0, "NonViolence")):
        for d in sorted((cache / sub).iterdir()):
            kp = d / "keypoints.csv"
            if not kp.exists() or not open(kp).readline().strip().endswith("box_h"):
                continue
            vid = next((Path(args.data) / folder).glob(d.name + ".*"), None)
            fps = 30.0
            if vid is not None:
                import cv2
                cap = cv2.VideoCapture(str(vid)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; cap.release()
            feats = video_features(kp, fps)
            X.append(feats); y.append(lab); rows.append([d.name, lab, *feats])
    X, y = np.array(X, dtype=float), np.array(y)
    print(f"{len(y)} video ({y.sum()} danh nhau, {len(y) - y.sum()} binh thuong), {X.shape[1]} dac trung")

    out = Path("runs/rf") / cache.name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "features.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["video", "label", *FEATURES]); w.writerows(rows)

    def scores(pred):
        return precision_score(y, pred, zero_division=0), recall_score(y, pred), f1_score(y, pred), float((pred == y).mean())

    i_base, i_imp = FEATURES.index("events_base"), FEATURES.index("events_impact")
    res = {"heuristic base": [scores((X[:, i_base] >= 1).astype(int))],
           "heuristic impact0.02": [scores((X[:, i_imp] >= 1).astype(int))], "Random Forest": []}
    imps = []
    for r in range(args.repeats):
        pred = np.zeros_like(y)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=r)
            clf.fit(X[tr], y[tr])
            pred[te] = clf.predict(X[te])
            imps.append(clf.feature_importances_)
        res["Random Forest"].append(scores(pred))

    lines = [f"{len(y)} video, Random Forest: 5-fold CV lap {args.repeats} lan (trung binh +- do lech chuan)",
             f"{'phuong phap':<22}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Accuracy':>12}"]
    for name, s in res.items():
        s = np.array(s); m, sd = s.mean(0), s.std(0)
        lines.append(f"{name:<22}" + "".join(f"{a:>7.2f}+-{b:.2f}" for a, b in zip(m, sd)))
    imp = np.mean(imps, 0)
    order = np.argsort(-imp)
    lines += ["", "Do quan trong dac trung (Random Forest):"]
    lines += [f"  {FEATURES[i]:<18} {imp[i]:.3f}" for i in order]
    text = "\n".join(lines)
    print(text)
    (out / "summary.txt").write_text(text + "\n")
    with open(out / "importance.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["feature", "importance"]); w.writerows([[FEATURES[i], round(imp[i], 4)] for i in order])
    print(f"\nDa luu: {out}/")


if __name__ == "__main__":
    main()
