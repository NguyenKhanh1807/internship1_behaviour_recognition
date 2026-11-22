"""Chay model.track() THAT cua Ultralytics tren vai frame - de dat breakpoint thang trong source YOLO.

Khong tu chia buoc nhu debug_flow.py: o day goi dung API that, nen breakpoint dat trong
source/ultralytics/ultralytics/... (predictor.py, tasks.py, head.py, track.py, byte_tracker.py) se dung lai.
Yeu cau: venv da cai ban clone o che do editable:  pip install -e source/ultralytics --no-deps
"""

from pathlib import Path

import cv2

import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "fight-demo" / "videos" / "sparring.mp4"
START, N = 567, 3  # doi frame o day (500 = binh thuong, 567 = pha clinch)

print("Ultralytics dang chay tu:", Path(ultralytics.__file__).parent)  # phai la .../source/ultralytics/ultralytics

model = YOLO(str(ROOT / "fight-demo" / "yolo11n-pose.pt"))
cap = cv2.VideoCapture(str(VIDEO))
cap.set(cv2.CAP_PROP_POS_FRAMES, START)
for t in range(N):
    ok, frame = cap.read()
    if not ok:
        break
    # persist=True: giu tracker giua cac lan goi -> ID duoc noi qua cac frame
    r = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
    ids = r.boxes.id.int().tolist() if r.boxes.id is not None else []
    print(f"frame {START + t}: {len(r)} nguoi, ID = {ids}, keypoints {tuple(r.keypoints.data.shape)}")
cap.release()
