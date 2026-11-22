"""Giai doan 2-3: do heuristic danh nhau o MUC VIDEO tren bo du lieu co nhan (RLVS, RWF-2000, ...).

Hai buoc:
  1) Chay YOLO-pose + tracking MOT LAN cho moi video (cham), luu keypoints.csv vao runs/eval_cache/<model>_<tracker>/
  2) Chay lai heuristic tren keypoints.csv voi NHIEU bo nguong / dieu kien (nhanh, vai giay) -> bang so sanh.

Mot video bi coi la "bao danh nhau" neu co >= --min-events su kien. So voi nhan (ten thu muc) -> P / R / F1.
Thu muc dataset: <root>/.../Violence + NonViolence  (RLVS)  hoac  Fight + NonFight (RWF-2000).

Cach chay (trong fight-demo, venv da bat):
    python eval_videos.py --root datasets --n 20                                   # YOLO11n + TrackTrack
    python eval_videos.py --root datasets --n 20 --model yolo26n-pose.pt --tracker my_botsort.yaml
    python eval_videos.py --root datasets --n 20 --seed 1                          # bo video KHAC de kiem tra lai
    python eval_videos.py --root datasets --n 20 --presets base,combo               # chi mot so preset
    python eval_videos.py --root datasets --n 20 --presets custom --direction 0.2   # tu dat tham so
"""

import argparse
import csv
import random
import subprocess
import sys
from pathlib import Path

from replay_heuristic import heuristic_parser, replay

POS_NAMES = {"violence", "fight", "fights", "violent"}
NEG_NAMES = {"nonviolence", "nonfight", "nonfights", "non-violence", "non_violence", "normal"}
EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# Cac bo dieu kien de so sanh (khoa = ten tham so heuristic). "base" = heuristic goc.
PRESETS = {
    "base":        {},
    "full_body":   {"full_body": True},
    "max_gap3":    {"max_gap": 3},
    "min_sec0.3":  {"min_sec": 0.3},
    "no_glitch":   {"max_speed": 1.0},
    "dir0.03":     {"direction": 0.03},
    "dir0.05":     {"direction": 0.05},
    "dir0.1":      {"direction": 0.1},
    "impact0.01":  {"impact": 0.01},
    "impact0.02":  {"impact": 0.02},
    "impact0.05":  {"impact": 0.05},
    "combo":       {"full_body": True, "max_speed": 1.0, "max_gap": 3, "direction": 0.05},
    # chon tren bo seed 0 (YOLO26n + BoT-SORT ReID): it bao nham nhat ma van giu recall -> PHAI kiem tra lai tren bo khac
    "tuned":       {"direction": 0.02, "impact": 0.02, "max_speed": 1.0, "speed": 0.06},
}

ap = argparse.ArgumentParser(parents=[heuristic_parser()])
ap.add_argument("--root", required=True, help="thu muc dataset")
ap.add_argument("--n", type=int, default=20, help="so video moi lop (ngau nhien theo --seed)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--exclude-seed", type=int, default=None, help="loai cac video da dung de chon nguong (bo seed nay, cung --n) -> bo kiem tra doc lap")
ap.add_argument("--model", default="yolo11n-pose.pt")
ap.add_argument("--tracker", default="tracktrack.yaml")
ap.add_argument("--device", default="mps")
ap.add_argument("--min-events", type=int, default=1)
ap.add_argument("--presets", default="all", help="'all' hoac danh sach ten, vd base,combo ; 'custom' = dung cac tham so dong lenh")
args = ap.parse_args()


def find_dirs(root: Path):
    pos = next((d for d in [root, *root.rglob("*")] if d.is_dir() and d.name.lower() in POS_NAMES), None)
    neg = next((d for d in [root, *root.rglob("*")] if d.is_dir() and d.name.lower() in NEG_NAMES), None)
    if pos is None or neg is None:
        sys.exit(f"Khong tim thay thu muc Violence/NonViolence (hoac Fight/NonFight) trong {root}")
    return pos, neg


def video_fps(path: Path) -> float:
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


pos_dir, neg_dir = find_dirs(Path(args.root))
rng = random.Random(args.seed)
rng_ex = random.Random(args.exclude_seed) if args.exclude_seed is not None else None
videos = []
for label, d in ((1, pos_dir), (0, neg_dir)):
    files = sorted(f for f in d.iterdir() if f.suffix.lower() in EXTS)
    if rng_ex is not None:  # bo dung y het cac video ma lan chay --seed <exclude-seed> da chon
        used = set(rng_ex.sample(files, min(args.n, len(files))))
        files = [f for f in files if f not in used]
    videos += [(label, f) for f in rng.sample(files, min(args.n, len(files)))]
print(f"Danh nhau: {pos_dir}  |  Binh thuong: {neg_dir}  |  {len(videos)} video (seed {args.seed})")

# ---------------- Buoc 1: YOLO + tracking (co cache) ----------------
cache = Path("runs/eval_cache") / f"{Path(args.model).stem}_{Path(args.tracker).stem}"
items = []
for i, (label, vid) in enumerate(videos, 1):
    out = cache / ("pos" if label else "neg") / vid.stem
    kp = out / "keypoints.csv"
    fresh = kp.exists() and open(kp).readline().strip().endswith("box_h")
    if not fresh:
        print(f"  [{i}/{len(videos)}] chay YOLO + tracking: {vid.name}", flush=True)
        cmd = [sys.executable, "fight_demo.py", "--source", str(vid), "--model", args.model,
               "--tracker", args.tracker, "--device", args.device, "--out", str(out), "--no-video"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not kp.exists():
            print(f"      LOI: {(r.stderr.strip().splitlines() or ['?'])[-1]}")
            continue
    items.append((label, vid, kp, video_fps(vid)))
print(f"Cache tracking: {cache}  ({len(items)} video san sang)\n")

# ---------------- Buoc 2: chay lai heuristic voi tung preset ----------------
if args.presets == "custom":
    chosen = {"custom": {}}
else:
    names = list(PRESETS) if args.presets == "all" else args.presets.split(",")
    chosen = {n: PRESETS[n] for n in names}

defaults = vars(heuristic_parser().parse_args([]))
user = {k: v for k, v in vars(args).items() if k in defaults}
per_video = {vid.name: {"video": vid.name, "label": label} for label, vid, _, _ in items}
summary = []
for name, overrides in chosen.items():
    h = argparse.Namespace(**{**(user if name == "custom" else defaults), **overrides})
    tp = fp = fn = tn = 0
    for label, vid, kp, fps in items:
        n_ev = replay(str(kp), h, fps)
        pred = int(n_ev >= args.min_events)
        per_video[vid.name][name] = n_ev
        tp += label and pred
        fn += label and not pred
        fp += (not label) and pred
        tn += (not label) and not pred
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / max(len(items), 1)
    summary.append((name, tp, fn, fp, tn, prec, rec, f1, acc))

lines = [f"{'preset':<12}{'TP':>4}{'FN':>4}{'FP':>4}{'TN':>4}{'Prec':>7}{'Rec':>7}{'F1':>7}{'Acc':>7}   dieu kien"]
for name, tp, fn, fp, tn, prec, rec, f1, acc in summary:
    cond = PRESETS.get(name, {k: user[k] for k in user if user[k] != defaults[k]})
    lines.append(f"{name:<12}{tp:>4}{fn:>4}{fp:>4}{tn:>4}{prec:>7.2f}{rec:>7.2f}{f1:>7.2f}{acc:>7.2f}   {cond}")
table = "\n".join(lines)
print(table)

out_dir = Path("runs/eval") / (f"{cache.name}_seed{args.seed}_n{args.n}" + (f"_excl{args.exclude_seed}" if args.exclude_seed is not None else ""))
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "summary.txt").write_text(table + "\n")
with open(out_dir / "per_video.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["video", "label", *chosen])
    w.writeheader()
    w.writerows(per_video.values())
print(f"\nDa luu: {out_dir / 'summary.txt'}  va  per_video.csv (so su kien cua tung video theo tung preset)")
