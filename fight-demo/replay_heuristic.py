"""Chay lai heuristic danh nhau tren keypoints.csv da co
"""

import argparse
import csv
from collections import defaultdict, deque

import numpy as np

from fight_demo import update_pairs


def heuristic_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--dist", type=float, default=1.2)
    ap.add_argument("--speed", type=float, default=0.08)
    ap.add_argument("--min-frames", type=int, default=5)
    ap.add_argument("--direction", type=float, default=0.0)
    ap.add_argument("--impact", type=float, default=0.0)
    ap.add_argument("--full-body", action="store_true")
    ap.add_argument("--min-sec", type=float, default=0.0)
    ap.add_argument("--max-gap", type=int, default=1)
    ap.add_argument("--max-speed", type=float, default=0.0)
    return ap


def load_frames(kp_csv: str):
    """frame -> (ids, kpts (N,17,3), heights). Chi frame co nguoi moi xuat hien (giong luc chay that)."""
    frames = defaultdict(lambda: ([], [], []))
    approx = False
    with open(kp_csv, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        has_h = header[-1] == "box_h"
        for row in rd:
            f, tid = int(row[0]), int(row[1])
            k = np.array(row[2:53], dtype=np.float32).reshape(17, 3)
            if has_h:
                h = float(row[53])
            else:  # uoc luong: chieu cao bao quanh cac keypoint nhin thay
                vis = k[k[:, 2] > 0.3]
                h = float(vis[:, 1].max() - vis[:, 1].min()) * 1.15 if len(vis) >= 2 else 1.0
                approx = True
            ids, kp, hs = frames[f]
            ids.append(tid), kp.append(k), hs.append(max(h, 1.0))
    return frames, approx


def replay(kp_csv: str, args, fps: float) -> int:
    """Tra ve tong so dong su kien (giong so dong events.csv neu chay fight_demo.py voi cung tham so)."""
    frames, _ = load_frames(kp_csv)
    history = defaultdict(lambda: deque(maxlen=2))
    streak = defaultdict(int)
    min_frames = max(1, round(args.min_sec * fps)) if args.min_sec > 0 else args.min_frames
    n = 0
    for f in sorted(frames):
        ids, kp, hs = frames[f]
        n += len(update_pairs(f, ids, np.stack(kp), np.array(hs), history, streak, args, min_frames))
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(parents=[heuristic_parser()])
    ap.add_argument("kp_csv")
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args()
    _, approx = load_frames(a.kp_csv)
    print(f"{replay(a.kp_csv, a, a.fps)} su kien" + ("  (chieu cao UOC LUONG - file cu khong co box_h)" if approx else ""))
