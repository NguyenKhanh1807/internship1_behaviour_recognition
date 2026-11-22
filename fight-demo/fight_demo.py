from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
from itertools import combinations
from pathlib import Path

import numpy as np

# Chi so keypoint COCO-17
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16
LIMBS = [L_WRIST, R_WRIST, L_ANKLE, R_ANKLE] 
KPT_CONF = 0.3 


def body_center(k: np.ndarray) -> np.ndarray | None:
    """Tam than = trung binh vai + hong co confidence du cao. k: (17, 3)."""
    idx = [i for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP) if k[i, 2] >= KPT_CONF]
    return k[idx, :2].mean(axis=0) if idx else None


def limb_speed(prev: np.ndarray, cur: np.ndarray, scale: float) -> float:
    """Toc do lon nhat cua co tay / co chan giua 2 frame, chia cho chieu cao nguoi (khong phu thuoc do phan giai)."""
    ok = (prev[LIMBS, 2] >= KPT_CONF) & (cur[LIMBS, 2] >= KPT_CONF)
    if not ok.any() or scale <= 0:
        return 0.0
    d = np.linalg.norm(cur[LIMBS, :2] - prev[LIMBS, :2], axis=1)[ok]
    return float(d.max() / scale)


def consecutive(hist: deque) -> bool:
    """2 skeleton gan nhat co o 2 frame lien nhau khong. hist: deque cac (frame, kpts).

    Neu ID bi mat vai frame roi xuat hien lai, khoang dich chuyen tich luy qua nhieu frame
    se bi tinh thanh toc do 1 frame -> bao dong sai luc 2 nguoi che nhau.
    """
    return len(hist) == 2 and hist[1][0] - hist[0][0] == 1


def pair_score(hist_a, hist_b, h_a: float, h_b: float):
    """Tra ve (khoang cach chuan hoa, toc do chuan hoa lon nhat) cho mot cap ID tai frame hien tai."""
    ca, cb = body_center(hist_a[-1][1]), body_center(hist_b[-1][1])
    if ca is None or cb is None:
        return None
    h = (h_a + h_b) / 2
    dist = float(np.linalg.norm(ca - cb) / h)
    dt_a, dt_b = max(hist_a[-1][0] - hist_a[-2][0], 1), max(hist_b[-1][0] - hist_b[-2][0], 1)
    speed = max(limb_speed(hist_a[-2][1], hist_a[-1][1], h_a) / dt_a, limb_speed(hist_b[-2][1], hist_b[-1][1], h_b) / dt_b)
    return dist, speed


# ----------------------------- Giai doan 3: dieu kien bo sung (mac dinh TAT) -----------------------------
def gap_ok(hist: deque, max_gap: int) -> bool:
    """2 skeleton gan nhat cach nhau <= max_gap frame (max_gap=1 giong consecutive cu)."""
    return len(hist) == 2 and 1 <= hist[1][0] - hist[0][0] <= max_gap


def full_body(k: np.ndarray) -> bool:
    """Thay it nhat 1 vai VA 1 hong -> nguoi khong bi cat khung qua nhieu (loai can mat)."""
    return bool((k[[L_SHOULDER, R_SHOULDER], 2] >= KPT_CONF).any() and (k[[L_HIP, R_HIP], 2] >= KPT_CONF).any())


def approach_speed(hist_x, target_center: np.ndarray, h_x: float) -> float:
    """Toc do lon nhat cua co tay / co chan cua X HUONG VE tam than nguoi kia (chieu van toc len vector toi muc tieu),
    chuan hoa theo chieu cao X va so frame bi khuyet. Vung tay len troi / sang ngang -> gan 0."""
    (f0, prev), (f1, cur) = hist_x[-2], hist_x[-1]
    dt = max(f1 - f0, 1)
    best = 0.0
    for j in LIMBS:
        if prev[j, 2] < KPT_CONF or cur[j, 2] < KPT_CONF:
            continue
        v = (cur[j, :2] - prev[j, :2]) / dt
        to_t = target_center - prev[j, :2]
        n = np.linalg.norm(to_t)
        if n > 1e-6:
            best = max(best, float(v @ (to_t / n)) / h_x)
    return best


def center_speed(hist_y, h_y: float) -> float:
    """Toc do dich chuyen tam than cua Y (phan ung khi bi danh / bi xo day), chuan hoa theo chieu cao."""
    (f0, prev), (f1, cur) = hist_y[-2], hist_y[-1]
    c0, c1 = body_center(prev), body_center(cur)
    if c0 is None or c1 is None:
        return 0.0
    return float(np.linalg.norm(c1 - c0) / max(f1 - f0, 1) / h_y)


def extra_ok(hist_a, hist_b, h_a, h_b, args) -> bool:
    """Ton tai X (A hoac B) tan cong: tay/chan X lao ve phia Y (--direction) VA Y co phan ung (--impact)."""
    if args.direction <= 0 and args.impact <= 0:
        return True
    ca, cb = body_center(hist_a[-1][1]), body_center(hist_b[-1][1])
    for hx, hy, cy, hxh, hyh in ((hist_a, hist_b, cb, h_a, h_b), (hist_b, hist_a, ca, h_b, h_a)):
        if args.direction > 0 and approach_speed(hx, cy, hxh) <= args.direction:
            continue
        if args.impact > 0 and center_speed(hy, hyh) <= args.impact:
            continue
        return True
    return False


def update_pairs(f, ids, kpts, heights, history, streak, args, min_frames):
    """Cap nhat heuristic cho 1 frame. Tra ve list (a, b, dist, speed, streak) cac cap dang bao dong.
    history: dict id -> deque(maxlen=2) cac (frame, kpts); streak: dict (a, b) -> so frame lien tiep thoa dieu kien."""
    hmap = dict(zip(ids, heights))
    for tid, k in zip(ids, kpts):
        history[tid].append((f, k))
    active, alerts, present = set(), [], set(ids)
    for a, b in combinations(sorted(ids), 2):
        if not gap_ok(history[a], args.max_gap) or not gap_ok(history[b], args.max_gap):
            continue
        if args.full_body and not (full_body(history[a][-1][1]) and full_body(history[b][-1][1])):
            continue
        s = pair_score(history[a], history[b], hmap[a], hmap[b])
        if s is None:
            continue
        dist, speed = s
        if getattr(args, "max_speed", 0) > 0 and speed > args.max_speed:
            continue
        if dist < args.dist and speed > args.speed and extra_ok(history[a], history[b], hmap[a], hmap[b], args):
            streak[(a, b)] += 1
            active.add((a, b))
            if streak[(a, b)] >= min_frames:
                alerts.append((a, b, dist, speed, streak[(a, b)]))
    for pair in list(streak):
        # max-gap > 1: neu 1 trong 2 ID tam mat o frame nay thi giu streak, khong reset
        if pair not in active and (args.max_gap <= 1 or (pair[0] in present and pair[1] in present)):
            streak[pair] = 0
    return alerts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="duong dan video, hoac 0 cho webcam")
    ap.add_argument("--model", default="yolo11n-pose.pt", help="yolo11n-pose.pt / yolo26n-pose.pt / ...")
    ap.add_argument("--tracker", default="botsort.yaml", help="botsort.yaml / bytetrack.yaml / tracktrack.yaml")
    ap.add_argument("--device", default=None, help="mps (Mac Apple Silicon), cpu, hoac 0 (GPU NVIDIA)")
    ap.add_argument("--out", default="runs/fight_demo")
    ap.add_argument("--dist", type=float, default=1.2, help="2 nguoi 'gan nhau' neu khoang cach tam < dist x chieu cao")
    ap.add_argument("--speed", type=float, default=0.08, help="toc do dau chi > speed x chieu cao / frame la 'nhanh'")
    ap.add_argument("--min-frames", type=int, default=5, help="so frame lien tiep thoa dieu kien moi bao dong")
    ap.add_argument("--no-video", action="store_true", help="khong ghi annotated.mp4 (nhanh hon khi chay hang loat)")
    # Giai doan 3 (mac dinh tat -> ket qua giong het ban cu)
    ap.add_argument("--direction", type=float, default=0.0, help="toc do tay/chan HUONG VE nguoi kia > nguong nay (0 = tat)")
    ap.add_argument("--impact", type=float, default=0.0, help="nguoi bi tan cong phai dich chuyen than > nguong nay (0 = tat)")
    ap.add_argument("--full-body", action="store_true", help="chi xet nguoi thay duoc vai + hong")
    ap.add_argument("--min-sec", type=float, default=0.0, help="thoi gian toi thieu (giay) thay cho --min-frames (0 = tat)")
    ap.add_argument("--max-gap", type=int, default=1, help="cho phep ID mat toi da N-1 frame giua 2 lan thay (1 = giong cu)")
    ap.add_argument("--max-speed", type=float, default=0.0, help="bo qua cap co toc do > nguong (nhay keypoint / doi ID, khong phai don danh that). 0 = tat")
    args = ap.parse_args()

    import cv2
    try:
        YOLO = __import__("ultralytics", fromlist=["YOLO"]).YOLO
    except ImportError as exc:
        raise SystemExit("Missing dependency: install it with `python -m pip install ultralytics`.") from exc

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # giu dung toc do video goc
    cap.release()
    min_frames = max(1, round(args.min_sec * fps)) if args.min_sec > 0 else args.min_frames

    history = defaultdict(lambda: deque(maxlen=2))  # 2 (frame, kpts) gan nhat cho heuristic
    full = defaultdict(list)                         # toan bo chuoi skeleton de luu .npz
    full_frames = defaultdict(list)                  # so frame tuong ung, de biet cho nao ID bi mat
    streak = defaultdict(int)                        # so frame lien tiep mot cap ID thoa dieu kien
    writer = None

    kp_file = open(out / "keypoints.csv", "w", newline="")
    ev_file = open(out / "events.csv", "w", newline="")
    kp_csv, ev_csv = csv.writer(kp_file), csv.writer(ev_file)
    kp_csv.writerow(["frame", "track_id"] + [f"k{i}_{c}" for i in range(17) for c in ("x", "y", "conf")] + ["box_h"])
    ev_csv.writerow(["frame", "id_a", "id_b", "dist_norm", "speed_norm", "streak"])

    results = model.track(source=source, tracker=args.tracker, device=args.device,
                          persist=True, stream=True, verbose=False)
    for f, r in enumerate(results):
        frame = None if args.no_video else r.plot()  # ve san box + ID + skeleton
        if writer is None and frame is not None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        alerts = []
        if r.boxes.id is not None and r.keypoints is not None:
            ids = r.boxes.id.int().tolist()
            kpts = r.keypoints.data.cpu().numpy()        # (N, 17, 3)
            heights = r.boxes.xywh[:, 3].cpu().numpy()    # chieu cao box, dung de chuan hoa
            for tid, k, hb in zip(ids, kpts, heights):
                full[tid].append(k)
                full_frames[tid].append(f)
                kp_csv.writerow([f, tid] + np.round(k.reshape(-1), 2).tolist() + [round(float(hb), 1)])
            for a, b, dist, speed, st in update_pairs(f, ids, kpts, heights, history, streak, args, min_frames):
                alerts.append((a, b))
                ev_csv.writerow([f, a, b, round(dist, 3), round(speed, 3), st])

        if frame is not None:
            for i, (a, b) in enumerate(alerts):
                cv2.putText(frame, f"NGHI DANH NHAU: ID {a} - ID {b}", (20, 40 + 35 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            writer.write(frame)

    if writer is not None:
        writer.release()
    kp_file.close()
    ev_file.close()
    arrays = {}
    for k, v in full.items():
        arrays[f"id_{k}"] = np.stack(v)                          # (T, 17, 3)
        arrays[f"id_{k}_frames"] = np.array(full_frames[k])      # (T,) so frame cua tung skeleton
    np.savez(out / "skeletons.npz", **arrays)
    print(f"Xong. {f + 1} frame, {len(full)} ID. Ket qua trong: {out.resolve()}")


if __name__ == "__main__":
    main()
