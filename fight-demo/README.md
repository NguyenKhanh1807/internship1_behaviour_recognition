# Fight demo: YOLO-pose + tracking + heuristic theo cặp ID

## Môi trường đã chạy
- MacBook Apple M5, macOS 25.6, GPU qua MPS
- Python 3.12 (venv `.venv`), Ultralytics 8.4.189, PyTorch 2.14.1
- Model `yolo11n-pose.pt` (tự tải lần đầu)

## Cài đặt (làm lại từ đầu)
```bash
cd ~/"Internship 1/fight-demo"
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install ultralytics yt-dlp imageio-ffmpeg
```

## Video thử
`videos/sparring.mp4`: đoạn 0:30–1:15 (45 s, 1348 frame, 1280x720) của
"BRUTAL STRIKING! Amateur Boxer RAGES FORWARD In Sparring!" — https://www.youtube.com/watch?v=AjJON5D9Qjg
Chỉ dùng để thử nghiệm học tập. Lệnh tải:
```bash
FF=$(.venv/bin/python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
.venv/bin/yt-dlp --ffmpeg-location "$FF" -f "bv*[height<=720][ext=mp4]+ba[ext=m4a]" \
  --merge-output-format mp4 --download-sections "*0:30-1:15" \
  -o videos/sparring.mp4 "https://www.youtube.com/watch?v=AjJON5D9Qjg"
```

## Chạy
```bash
.venv/bin/python fight_demo.py --source videos/sparring.mp4 --device mps --tracker botsort.yaml   --out runs/sparring_botsort
.venv/bin/python fight_demo.py --source videos/sparring.mp4 --device mps --tracker bytetrack.yaml --out runs/sparring_bytetrack
.venv/bin/python fight_demo.py --source videos/sparring.mp4 --device mps --tracker tracktrack.yaml --out runs/sparring_tracktrack
```
Mỗi lần chạy khoảng 44 s cho 1348 frame (~30 FPS).

## Kết quả (2026-10-26)
Trung bình ~10 người/frame (2 võ sĩ, HLV, khán giả quanh võ đài).

| Tracker | Số ID | ID sống < 30 frame | ID dài nhất (frame) | Frame có cảnh báo | Số cặp ID bị cảnh báo |
|---|---|---|---|---|---|
| BoT-SORT | 236 | 127 | 1186 | 483 | 287 |
| ByteTrack | 269 | 170 | 1344 | 460 | 258 |
| TrackTrack (mặc định 8.4.189) | 99 | 31 | 1342 | 326 | 117 |

Quan sát (ảnh trong `runs/`):
- `runs/sparring_tracktrack/alert_1_49_f611.jpg`: cảnh báo đúng — 2 võ sĩ ID 1 và ID 49 đang ra đòn.
- `runs/sparring_tracktrack/alert_1_46_f567.jpg`: ngoài cặp đúng (1–20) còn cảnh báo sai 1–31, 1–40, 1–46
  (khán giả, HLV). Lý do: khoảng cách đo trên ảnh 2D nên người đứng sau/trước trông như "gần";
  tốc độ lấy max của 2 người nên chỉ cần võ sĩ ra đòn là mọi người "gần" võ sĩ đều bị kéo vào.
- Đổi ID: ở frame 567 võ sĩ đội mũ xanh là ID 20, đến frame 611 lại là ID 49 và ID 20 chuyển sang người phía sau.
- `runs/sparring_botsort/frames/f1250.jpg`: lúc 2 võ sĩ ôm ghì, box chồng nhau và BoT-SORT sinh ra loạt ID mới (961, 982, 1020).

## Thay đổi so với bản script ban đầu
1. Chỉ tính tốc độ khi 2 skeleton gần nhất ở 2 frame liền nhau (trước đây ID mất vài frame rồi quay lại sẽ bị tính thành tốc độ rất lớn).
2. `skeletons.npz` lưu thêm `id_<k>_frames` (số frame của từng skeleton) để biết chỗ ID bị mất.
