"""So sanh cac lan chay fight_demo.py: dem so ID, ID ngan, so nguoi trung binh moi frame.
Cach dung:  python compare_runs.py runs/sparring_tracktrack runs/sparring_bytetrack ...
Chi dung thu vien chuan cua Python (khong can pandas).
"""
import csv
import sys
from collections import defaultdict

SHORT = 30  # ID xuat hien < 30 frame (~1 giay) coi la "ID ngan" (thuong do doi ID / mat track)

print(f"{'run':<34}{'so ID':>8}{'ID <30f':>10}{'nguoi/frame':>13}{'events':>9}")
for run in sys.argv[1:]:
    life = defaultdict(set)      # track_id -> cac frame xuat hien
    per_frame = defaultdict(set)  # frame -> cac track_id
    with open(f"{run}/keypoints.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            life[row["track_id"]].add(row["frame"])
            per_frame[row["frame"]].add(row["track_id"])
    n_short = sum(len(v) < SHORT for v in life.values())
    avg = sum(len(v) for v in per_frame.values()) / max(len(per_frame), 1)
    try:
        with open(f"{run}/events.csv") as fh:
            ev = sum(1 for _ in fh) - 1
    except FileNotFoundError:
        ev = 0
    print(f"{run:<34}{len(life):>8}{n_short:>10}{avg:>13.1f}{ev:>9}")
