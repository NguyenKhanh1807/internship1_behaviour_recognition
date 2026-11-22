"""Chay TUNG BUOC luong YOLO-pose + tracking, in shape o moi buoc.

Cach chay (tu thu muc "Internship 1", da activate venv):
    python debug/debug_flow.py                                   # mac dinh: video sparring, frame 500-502
    python debug/debug_flow.py --start 567                       # doi frame (vd pha clinch 182 / 567 / 611)
    python debug/debug_flow.py --source path/to/anh.jpg          # 1 anh bat ky (bo qua phase tracking)
    python debug/debug_flow.py --model yolo11s-pose.pt           # doi model (tu tai neu chua co)
    python debug/debug_flow.py --imgsz 320                       # doi kich thuoc dau vao
    PAUSE=1 python debug/debug_flow.py    # dung lai sau moi buoc (pdb): go  p <bien>  de xem, c  de di tiep
Hoac mo trong VS Code, dat breakpoint o cac dong "# >>> BREAKPOINT" roi bam F5.

Code goc tuong ung ghi o dau moi buoc (duong dan tinh tu source/ultralytics/ultralytics/).
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml

ROOT = Path(__file__).resolve().parents[1]

# ============================ INPUT: DOI O DAY ================================
# Sua gia tri mac dinh (default=...) hoac truyen tham so khi chay / trong .vscode/launch.json "args"
ap = argparse.ArgumentParser()
ap.add_argument("--source", default=str(ROOT / "fight-demo" / "videos" / "sparring.mp4"),
                help="video (.mp4) hoac anh (.jpg/.png)")
ap.add_argument("--start", type=int, default=500, help="frame bat dau (chi dung cho video)")
ap.add_argument("--frames", type=int, default=3, help="so frame lien tiep de xem tracking (chi dung cho video)")
ap.add_argument("--model", default=str(ROOT / "fight-demo" / "yolo11n-pose.pt"), help="ho YOLO11-pose: yolo11n/s/m-pose.pt (YOLO26 co head khac, buoc 4 se khong khop)")
ap.add_argument("--imgsz", type=int, default=640, help="kich thuoc canh dai cua anh dua vao model")
args = ap.parse_args()
# ==============================================================================

WEIGHTS = args.model
VIDEO = Path(args.source)
IS_VIDEO = VIDEO.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
START_FRAME, N_FRAMES = (args.start, args.frames) if IS_VIDEO else (0, 1)
PAUSE = os.environ.get("PAUSE") == "1"
torch.set_printoptions(precision=2, sci_mode=False)


def step(title):
    print("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80)


def pause():
    if PAUSE:
        breakpoint()  # go 'c' de chay tiep, 'p ten_bien' de xem bien


def read_frames():
    if not IS_VIDEO:
        img = cv2.imread(str(VIDEO))
        assert img is not None, f"Khong doc duoc anh {VIDEO}"
        return [img]
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    frames = []
    for _ in range(N_FRAMES):
        ok, f = cap.read()
        assert ok, f"Khong doc duoc frame tu {VIDEO}"
        frames.append(f)
    cap.release()
    return frames


frames = read_frames()
img0 = frames[0]

# -----------------------------------------------------------------------------
step("BUOC 1. Load model  (engine/model.py -> nn/tasks.py PoseModel, parse_model doc yolo11-pose.yaml)")
print(f"Input: {VIDEO.name}" + (f", frame {START_FRAME}..{START_FRAME + N_FRAMES - 1}" if IS_VIDEO else " (anh)"))
model = YOLO(str(WEIGHTS))
model.predict(img0, imgsz=args.imgsz, verbose=False)  # goi 1 lan de Ultralytics dung predictor (setup_model, fuse Conv+BN, warmup)
p = model.predictor                 # engine/predictor.py BasePredictor (o day la PosePredictor)
net = p.model.model                 # PoseModel that su (p.model la AutoBackend boc ngoai)
head = net.model[-1]                # layer cuoi = Pose head (nn/modules/head.py:572)
print(f"Model: {type(net).__name__}, so layer: {len(net.model)}")
print(f"Head: {type(head).__name__}, stride={head.stride.tolist()}, kpt_shape={head.kpt_shape}, nc={head.nc}, reg_max={head.reg_max}")
print("\n i   from        layer   (khop voi so 0..23 trong hinh kien truc)")
for m in net.model:
    print(f"{m.i:>2}   {str(m.f):<10}  {m.type.split('.')[-1]}")
pause()  # >>> BREAKPOINT 1: xem net.model, head

# -----------------------------------------------------------------------------
step("BUOC 2. Tien xu ly  (engine/predictor.py:183 preprocess, LetterBox trong data/augment.py)")
im = p.preprocess([img0])  # BGR uint8 HxWx3  ->  RGB float 1x3xH'xW', 0..1
print(f"Anh goc: {img0.shape} (H, W, C) dtype={img0.dtype}")
print(f"Tensor : {tuple(im.shape)} (B, C, H, W) dtype={im.dtype} min={im.min():.2f} max={im.max():.2f}")
print("Chu y: khi predict, LetterBox dung auto=True -> chi pad toi boi so cua 32, nen anh 16:9 thanh 384x640, khong phai 640x640")
pause()  # >>> BREAKPOINT 2: xem im

# -----------------------------------------------------------------------------
step("BUOC 3. Backbone + Neck, chay TUNG LAYER  (nn/tasks.py:181 _predict_once)")
y, x = [], im
with torch.no_grad():
    for m in net.model[:-1]:  # tat ca tru head
        if m.f != -1:  # layer lay input tu layer cu (skip / Concat)
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x if m.i in net.save else None)
        print(f"layer {m.i:>2} {m.type.split('.')[-1]:<10} from={str(m.f):<9} -> {tuple(x.shape)}")
    feats = [y[j] for j in head.f]  # 3 feature map dua vao head (P3, P4, P5)
print("\nInput cua head (from =", head.f, "):")
for name, f in zip(("P3", "P4", "P5"), feats):
    print(f"  {name}: {tuple(f.shape)}  -> {f.shape[2]}x{f.shape[3]} = {f.shape[2] * f.shape[3]} o luoi")
pause()  # >>> BREAKPOINT 3: xem y (output tung layer), feats

# -----------------------------------------------------------------------------
step("BUOC 4a. Head - output THO cua 3 nhanh  (nn/modules/head.py forward_head, dong 176-178 va 641)")
with torch.no_grad():
    raw = head.forward_head(feats, **head.one2many)  # cv2 (box), cv3 (class), cv4 (keypoint)
A = raw["boxes"].shape[-1]
print(f"boxes  (cv2, hoi quy box)    : {tuple(raw['boxes'].shape)}  = 4 canh x {head.reg_max} bin DFL")
print(f"scores (cv3, phan loai)      : {tuple(raw['scores'].shape)}")
print(f"kpts   (cv4, hoi quy kpt)    : {tuple(raw['kpts'].shape)}  = 17 x (x, y, conf)")
print(f"Tong so o (anchor) A = {A}")
pause()  # >>> BREAKPOINT 4a: xem raw

# -----------------------------------------------------------------------------
step("BUOC 4b. Head - DECODE  (_inference dong 629-632, _get_decode_boxes 219, DFL block.py:78, kpts_decode 644)")
with torch.no_grad():
    dec = head._inference(raw)  # (1, 4 + 1 + 51, A)
print(f"Sau decode: {tuple(dec.shape)}  = 4 box (xywh pixel) + 1 score + 51 keypoint")
print(f"anchors {tuple(head.anchors.shape)} (tam o luoi), strides {tuple(head.strides.shape)}")

best = int(dec[0, 4].argmax())  # o co score cao nhat
ax, ay = head.anchors[:, best].tolist()
s = float(head.strides[0, best])
print(f"\nO tot nhat: index={best}, tam o (theo don vi o luoi)=({ax:.1f}, {ay:.1f}), stride={s:.0f}, score={dec[0, 4, best]:.3f}")
print(f"  box xywh = {dec[0, :4, best].tolist()}")
rx, ry = raw["kpts"][0, 0, best].item(), raw["kpts"][0, 1, best].item()  # mui (keypoint 0), gia tri tho
manual_x = (rx * 2.0 + (ax - 0.5)) * s
manual_y = (ry * 2.0 + (ay - 0.5)) * s
print(f"  Mui (kpt 0): tho=({rx:.3f}, {ry:.3f})")
print(f"     tinh tay (x_tho*2 + (anchor-0.5)) * stride = ({manual_x:.1f}, {manual_y:.1f})")
print(f"     kpts_decode cho ra                         = ({dec[0, 5, best]:.1f}, {dec[0, 6, best]:.1f})")

with torch.no_grad():
    full = p.inference(im)  # chay ca model mot lan nhu Ultralytics that (engine/predictor.py:208)
print(f"\nKiem tra: tu chay tung buoc == model chay mot lan? {torch.allclose(dec, full[0], atol=1e-4)}")
pause()  # >>> BREAKPOINT 4b: xem dec, best

# -----------------------------------------------------------------------------
step("BUOC 5. Hau xu ly: NMS + scale ve anh goc  (models/yolo/detect/predict.py:55, pose/predict.py:62-65)")
res = p.postprocess(full, im, [img0])[0]
print(f"Truoc NMS: {A} o  ->  sau NMS: {len(res)} nguoi (conf >= {p.args.conf}, iou = {p.args.iou})")
print(f"boxes.data     : {tuple(res.boxes.data.shape)}   (x1, y1, x2, y2, conf, cls)")
print(f"keypoints.data : {tuple(res.keypoints.data.shape)} (nguoi, 17 khop, x/y/conf) - toa do tren anh GOC {img0.shape[1]}x{img0.shape[0]}")
if len(res):
    print(f"Nguoi 0: box={res.boxes.xyxy[0].tolist()}\n         mui={res.keypoints.data[0, 0].tolist()}")
pause()  # >>> BREAKPOINT 5: xem res

# -----------------------------------------------------------------------------
step(f"BUOC 6. Tracking qua {N_FRAMES} frame  (trackers/track.py:132-138, byte_tracker.py:274 update)")
if not IS_VIDEO:
    print("Input la anh -> khong co frame tiep theo de tracking. Dung video de xem phase nay.")
    raise SystemExit
cfg = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
tracker = BYTETracker(args=cfg)
for t, img in enumerate(frames):
    with torch.no_grad():
        im_t = p.preprocess([img])
        res_t = p.postprocess(p.inference(im_t), im_t, [img])[0]
    det = res_t.boxes.cpu().numpy()             # tracker chi nhan BOX, khong nhan keypoint
    tracks = tracker.update(det, img)           # Kalman predict -> IoU -> Hungarian -> cap nhat vong doi
    print(f"\nFrame {START_FRAME + t}: {len(det)} detection -> {len(tracks)} track")
    if len(tracks) == 0:
        print("  (frame dau: track moi o trang thai New, chua duoc xac nhan nen chua tra ve)")
        continue
    print("  tracks[:, cot] = x1 y1 x2 y2 | track_id | score | cls | idx (vi tri detection goc)")
    idx = tracks[:, -1].astype(int)             # track.py:137
    res_t = res_t[idx]                          # track.py:138 -> keypoint di theo box, nhan ID
    for row, kp in zip(tracks, res_t.keypoints.data):
        print(f"  ID {int(row[4]):>3}  idx={int(row[7])}  score={row[5]:.2f}  mui=({kp[0, 0]:.0f}, {kp[0, 1]:.0f})")
    st = tracker.tracked_stracks[0]
    print(f"  Kalman state cua track ID {st.track_id} (x, y, a, h, vx, vy, va, vh): {np.round(st.mean, 2).tolist()}")
    pause()  # >>> BREAKPOINT 6: xem tracks, tracker.tracked_stracks, tracker.lost_stracks

step("XONG. Luong: anh -> preprocess -> 23 layer -> head (3 nhanh) -> decode -> NMS -> tracker -> ID + 17 keypoint")
