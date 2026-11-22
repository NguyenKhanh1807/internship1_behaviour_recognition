#!/bin/bash
# Giai doan 1: chay 3 thi nghiem tracking tren video sparring roi so sanh voi ket qua cu.
# Cach chay (trong thu muc fight-demo, venv da bat):  bash run_tracking_exps.sh
# Moi lan chay ~1-2 phut tren MacBook. Lan dau se tu tai yolo26n-pose.pt.
set -e
VIDEO=videos/sparring.mp4
DEV=${DEV:-mps}

echo "[1/3] BoT-SORT + ReID, YOLO11n"
python fight_demo.py --source $VIDEO --tracker my_botsort.yaml --device $DEV --out runs/sparring_botsort_reid

echo "[2/3] TrackTrack, YOLO26n"
python fight_demo.py --source $VIDEO --model yolo26n-pose.pt --tracker tracktrack.yaml --device $DEV --out runs/sparring_tracktrack_y26

echo "[3/3] BoT-SORT + ReID, YOLO26n"
python fight_demo.py --source $VIDEO --model yolo26n-pose.pt --tracker my_botsort.yaml --device $DEV --out runs/sparring_botsort_reid_y26

echo
python compare_runs.py runs/sparring_tracktrack runs/sparring_botsort runs/sparring_bytetrack \
  runs/sparring_botsort_reid runs/sparring_tracktrack_y26 runs/sparring_botsort_reid_y26 | tee runs/compare_tracking.txt
echo "Da luu bang so sanh: runs/compare_tracking.txt"
