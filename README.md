# Báo cáo: nhận diện đánh nhau từ skeleton

Mở file này ở chế độ preview (Cmd+Shift+V) rồi bấm vào link để nhảy thẳng tới dòng code / file kết quả.
Chạy thử: tab **Run and Debug** (Cmd+Shift+D) → chọn cấu hình có số thứ tự `[1]`…`[8]` → F5. Mục 8–9 (ProtoGCN, 5-fold): cấu hình `[9]`, `[10]` hoặc chạy bằng terminal.

> Breakpoint: VS Code không cho đặt sẵn breakpoint bằng file, nên mỗi dòng cần dừng có ghi **(BP)**.
> Bấm link → con trỏ tới đúng dòng → nhấn **F9** để đặt breakpoint, rồi chạy cấu hình tương ứng.

---

## 1. YOLO-pose: keypoint được phát hiện thế nào
Chạy: **[1] YOLO flow - frame 567** (dừng sau từng bước, Enter để đi tiếp)

| Bước | Code | Ghi chú |
|---|---|---|
| Letterbox 1280×720 → 384×640 | [predictor.py:234](source/ultralytics/ultralytics/engine/predictor.py#L234) **(BP)** | 5040 ô thay vì 8400 |
| Backbone + neck | [tasks.py:181 `_predict_once`](source/ultralytics/ultralytics/nn/tasks.py#L181) | P3 48×80, P4 24×40, P5 12×20 |
| Head Pose (box 64 + class 1 + kpt 51) | [head.py:572 `class Pose`](source/ultralytics/ultralytics/nn/modules/head.py#L572), [forward_head:634](source/ultralytics/ultralytics/nn/modules/head.py#L634) | |
| **Decode keypoint ra pixel** | [head.py:644 `kpts_decode`](source/ultralytics/ultralytics/nn/modules/head.py#L644) **(BP)** | x = (2x̂ + anchor − 0.5) · stride |
| NMS | [detect/predict.py:33 `postprocess`](source/ultralytics/ultralytics/models/yolo/detect/predict.py#L33) | |
| Đưa về ảnh gốc | [pose/predict.py:64 `scale_coords`](source/ultralytics/ultralytics/models/yolo/pose/predict.py#L64) **(BP)** | bỏ pad, chia scale |

Script debug từng bước: [debug/debug_flow.py](debug/debug_flow.py) · Train 1 batch: [debug/debug_train.py](debug/debug_train.py)

Khi train: [TaskAlignedAssigner (tal.py:14)](source/ultralytics/ultralytics/utils/tal.py#L14) · [KeypointLoss = 1 − OKS (loss.py:335)](source/ultralytics/ultralytics/utils/loss.py#L335), [dòng tính loss:351](source/ultralytics/ultralytics/utils/loss.py#L351)

## 2. Tracking: keypoint nhận ID thế nào
Chạy: **[2] YOLO track - code goc (frame 567-569)**

| Bước | Code |
|---|---|
| Kalman dự đoán | [byte_tracker.py:89 `multi_predict`](source/ultralytics/ultralytics/trackers/byte_tracker.py#L89) |
| Vòng update | [byte_tracker.py:274 `update`](source/ultralytics/ultralytics/trackers/byte_tracker.py#L274) **(BP)** |
| Chi phí IoU + ReID (lấy min) | [bot_sort.py:197–209](source/ultralytics/ultralytics/trackers/bot_sort.py#L197) |
| Hungarian ghép cặp | [byte_tracker.py:397](source/ultralytics/ultralytics/trackers/byte_tracker.py#L397) **(BP)** |
| **Keypoint nhận ID theo box** | [track.py:137–138 `result[idx]`](source/ultralytics/ultralytics/trackers/track.py#L137) **(BP)** |

Cấu hình ReID: [my_botsort.yaml](fight-demo/my_botsort.yaml)

Kết quả:
- So sánh 6 tracker: [runs/compare_tracking.txt](fight-demo/runs/compare_tracking.txt)
- Ảnh frame 500 (skeleton + ID): [f500.jpg](fight-demo/runs/sparring_botsort/frames/f500.jpg)
- Đoạn ôm ghì (đổi ID): [_preview_clinch.jpg](fight-demo/runs/_preview_clinch.jpg) · video [clinch_compare_botsort_vs_TN3.mp4](fight-demo/runs/clinch_compare_botsort_vs_TN3.mp4)

## 3. fight_demo.py: luật phát hiện đánh nhau (6 tầng)
Chạy: **[3] fight_demo - sparring (BoT-SORT ReID + YOLO26n)**

| Tầng | Code |
|---|---|
| 1. YOLO + tracking | [fight_demo.py:184 `model.track`](fight-demo/fight_demo.py#L184) |
| 3. Đặc trưng | [body_center:20](fight-demo/fight_demo.py#L20), [limb_speed:26](fight-demo/fight_demo.py#L26) |
| 4. Luật theo cặp | [pair_score:44](fight-demo/fight_demo.py#L44), [**update_pairs:107**](fight-demo/fight_demo.py#L107) **(BP)** |
| 5. Lọc thêm | [approach_speed:67](fight-demo/fight_demo.py#L67), [center_speed:84](fight-demo/fight_demo.py#L84), [extra_ok:93](fight-demo/fight_demo.py#L93) |

Kết quả: [alert f567](fight-demo/runs/sparring_tracktrack/alert_1_46_f567.jpg) · [events.csv](fight-demo/runs/sparring_tracktrack/events.csv) · [video](fight-demo/runs/sparring_tracktrack/annotated_h264.mp4)

## 4. Đánh giá luật trên RLVS
Code: [eval_videos.py](fight-demo/eval_videos.py) · [replay_heuristic.py](fight-demo/replay_heuristic.py)

| Tập | Kết quả |
|---|---|
| 20+20 video, seed 0 (chỉnh ngưỡng) | [summary.txt](fight-demo/runs/eval/yolo26n-pose_my_botsort_seed0_n20/summary.txt) |
| 20+20 video, seed 1 (kiểm tra lại) | [summary.txt](fight-demo/runs/eval/yolo26n-pose_my_botsort_seed1_n20_excl0/summary.txt) |
| **500+500 video, seed 2** | [summary.txt](fight-demo/runs/eval/yolo26n-pose_my_botsort_seed2_n500/summary.txt) · [per_video.csv](fight-demo/runs/eval/yolo26n-pose_my_botsort_seed2_n500/per_video.csv) |
| Ảnh các ca sai | [_preview_errors.jpg](fight-demo/runs/eval/_preview_errors.jpg) |

## 5. Random Forest
Chạy: **[4] Random Forest (1061 video)** · Code: [train_rf.py, video_features:34](fight-demo/train_rf.py#L34)

Kết quả: [summary.txt](fight-demo/runs/rf/yolo26n-pose_my_botsort/summary.txt) · [importance.csv](fight-demo/runs/rf/yolo26n-pose_my_botsort/importance.csv)

## 6. ST-GCN
Chạy: **[5] ST-GCN tung buoc (debug_stgcn.py)**

| Phần | Code |
|---|---|
| Đồ thị A (3, 17, 17) | [debug_stgcn.py:178 `stgcn_spatial`](debug/debug_stgcn.py#L178) |
| GCN: trộn khớp kề nhau | [debug_stgcn.py:221 einsum](debug/debug_stgcn.py#L221) **(BP)** |
| Head: trung bình 2 người | [debug_stgcn.py:298](debug/debug_stgcn.py#L298) **(BP)** |
| Zero-shot + probe | [eval_stgcn.py:45 `run_video`](fight-demo/eval_stgcn.py#L45) |

Kết quả zero-shot / probe: [summary.txt](fight-demo/runs/stgcn/yolo26n-pose_my_botsort/summary.txt)

## 7. Fine-tune ST-GCN và đồ thị 2 người
Chạy: **[6] Fine-tune ST-GCN 1p - xem ket qua** · **[7] Fine-tune ST-GCN 2p - xem ket qua**
(đã có checkpoint nên chạy chỉ in lại bảng kết quả, vài giây)

| Phần | Code |
|---|---|
| Đồ thị 34 đỉnh, cạnh giữa 2 người | [finetune_stgcn.py:26 `EDGES_2P`](fight-demo/finetune_stgcn.py#L26) |
| Nạp trọng số NTU, nhân đôi cho 34 khớp | [finetune_stgcn.py:35 `build`](fight-demo/finetune_stgcn.py#L35) **(BP)** |
| Lấy mẫu 100 frame + lật + đổi người | [finetune_stgcn.py:88 `train_clip`](fight-demo/finetune_stgcn.py#L88) |

| Kết quả | 1 người riêng (gốc) | Đồ thị 2 người |
|---|---|---|
| Bảng so sánh (cùng 210 video test) | [summary.txt](fight-demo/runs/stgcn_ft/yolo26n-pose_my_botsort/summary.txt) | [summary.txt](fight-demo/runs/stgcn_ft/yolo26n-pose_my_botsort_2p/summary.txt) |
| Loss theo epoch | [train_log.csv](fight-demo/runs/stgcn_ft/yolo26n-pose_my_botsort/train_log.csv) | [train_log.csv](fight-demo/runs/stgcn_ft/yolo26n-pose_my_botsort_2p/train_log.csv) |
| Dự đoán từng video | [test_pred.csv](fight-demo/runs/stgcn_ft/yolo26n-pose_my_botsort/test_pred.csv) | [test_pred.csv](fight-demo/runs/stgcn_ft/yolo26n-pose_my_botsort_2p/test_pred.csv) |

## 8. ProtoGCN (Liu et al., CVPR 2025)
Bài báo: [2411.18941v2.pdf](report/papers/2411.18941v2.pdf) · Bản port thuần PyTorch: [protogcn_model.py](fight-demo/protogcn_model.py)

Chạy: **[10] ProtoGCN bone-motion tu FineGYM + ensemble 4 luong** ([run_bm_finegym.py](fight-demo/run_bm_finegym.py)), hoặc từng lệnh trong terminal (thư mục `fight-demo`, venv đã bật):
```bash
python finetune_protogcn.py --init ../debug/ckpt/protogcn_finegym_j.pth    # luồng joint
python finetune_protogcn.py --init ../debug/ckpt/protogcn_finegym_b.pth    # luồng bone
python finetune_protogcn.py --init ../debug/ckpt/protogcn_finegym_jm.pth   # luồng joint-motion
python finetune_protogcn.py --init ../debug/ckpt/protogcn_finegym_bm.pth   # luồng bone-motion
python finetune_protogcn.py --init scratch                                 # train từ đầu, 100 prototype
python finetune_protogcn.py --init scratch --num-prototype 400             # train từ đầu, 400 prototype
python finetune_protogcn.py --init scratch --modality bm                   # bone-motion train từ đầu
python ensemble_streams.py                                                 # gộp các luồng
```
(đã có checkpoint và `test_prob.npy` nên chạy lại chỉ in bảng kết quả)

| Phần | Code | Ghi chú |
|---|---|---|
| 17 khớp COCO → 20 khớp (thêm giữa hông, giữa cột sống, giữa vai) | [protogcn_model.py:28 `kinetics_transform`](fight-demo/protogcn_model.py#L28) | |
| Luồng j / b / jm / bm | [protogcn_model.py:41 `gen_feat`](fight-demo/protogcn_model.py#L41) | b = khớp − khớp cha, jm = hiệu 2 frame |
| Đồ thị A0 học từ dữ liệu | [protogcn_model.py:58 `random_graph`](fight-demo/protogcn_model.py#L58) | 8 ma trận 20×20, khởi tạo ngẫu nhiên |
| **MTE: A = A0 + α·tanh(Qi − Kj) + β·softmax(Q·Kᵀ)** | [protogcn_model.py:146 `unit_gcn.forward`](fight-demo/protogcn_model.py#L146) **(BP)** | A_inter và A_intra riêng cho từng mẫu |
| TCN đa tỉ lệ, 6 nhánh | [protogcn_model.py:77 `mstcn`](fight-demo/protogcn_model.py#L77) | |
| Backbone 10 khối, 96 → 192 → 384 kênh | [protogcn_model.py:222 `ProtoGCN.forward`](fight-demo/protogcn_model.py#L222) **(BP)** | T: 100 → 50 → 25 |
| **PRN: R = softmax(X·W_queryᵀ), Z = R·W_memory** | [protogcn_model.py:186](fight-demo/protogcn_model.py#L186) **(BP)** | X: 400 cạnh × 384 kênh của khối cuối |
| Loss tương phản theo lớp (CSC) | [protogcn_model.py:249](fight-demo/protogcn_model.py#L249) | tâm lớp EMA 0.9, nhiệt độ 0.125 |
| Head + loss tổng L = CE + 0.2·CSC | [protogcn_model.py:283 `SimpleHead.loss`](fight-demo/protogcn_model.py#L283) | logit chỉ dùng đặc trưng backbone; PRN + CSC chỉ tác động lúc train |
| Nạp checkpoint FineGYM, bỏ FC 99 lớp, reset `data_bn` | [protogcn_model.py:300 `load_pretrained`](fight-demo/protogcn_model.py#L300) | |
| Vòng train | [finetune_protogcn.py:137](fight-demo/finetune_protogcn.py#L137) **(BP)** | 12 epoch, lr 0.01 cosine, bs 16 |
| Test 10 clip / video | [finetune_protogcn.py:157](fight-demo/finetune_protogcn.py#L157) | trung bình softmax |

**Kiểm chứng bản port** (cấu hình **[12]** hoặc `python run_finegym_check.py`; code: [eval_finegym.py](fight-demo/eval_finegym.py), [ensemble_finegym.py](fight-demo/ensemble_finegym.py); dữ liệu `debug/data/gym_hrnet.pkl` của PYSKL):
4 checkpoint chính thức, giữ fc 99 lớp, không fine-tune, pipeline giống config gốc [j_1.py](debug/ckpt/protogcn_finegym_j1/j_1.py), 8521 video val FineGYM, 1 clip/video. Mốc lấy từ log của tác giả ([j_1](debug/ckpt/protogcn_finegym_j1/20231211_012214.log), [b_1, jm, bm](debug/ckpt/protogcn_finegym_logs/)).

| Luồng | Top-1 bản port | Top-1 log tác giả | Mean class bản port | Mean class log tác giả |
|---|---|---|---|---|
| joint | 0.9309 | 0.9308 | 0.9046 | 0.9058 |
| bone | 0.9481 | 0.9486 | 0.9286 | 0.9287 |
| joint-motion | 0.9378 | 0.9389 | 0.9154 | 0.9163 |
| bone-motion | 0.9304 | 0.9309 | 0.9071 | 0.9083 |
| gộp 4 luồng ([chi tiết](fight-demo/runs/finegym_eval/ensemble_clips1.txt)) | 0.9569 | | 0.9380 | |

→ cả 4 luồng lệch không quá 0.0012: bản port tính đúng như code gốc. Bài báo ghi 95.9% cho bản gộp nhiều luồng hơn (có cả luồng k chưa port) nên chỉ để tham khảo.
Luồng joint với 10 clip/video (đúng cách bài báo test): bản port top-1 0.9326, mean class 0.9094; log tác giả 0.9328 / 0.9092 ([kết quả](fight-demo/runs/finegym_eval/protogcn_finegym_j_clips10.txt)). Số liệu RLVS bên dưới KHÔNG so trực tiếp được với bài báo (khác dữ liệu, số lớp, chỉ số).

Kết quả trên cùng 210 video test (4.02M tham số, ~50 s/epoch, 138 ms/video test trên MPS):

| Cấu hình | P | R | F1 | CI95 F1 | AUC | FP/FN | Kết quả |
|---|---|---|---|---|---|---|---|
| **bone (FineGYM)** | 0.93 | 0.93 | **0.93** | 0.89–0.96 | 0.973 | 8/7 | [summary.txt](fight-demo/runs/protogcn_ft/yolo26n-pose_my_botsort_protogcn_finegym_b/summary.txt) |
| joint (FineGYM) | 0.87 | 0.88 | 0.87 | 0.82–0.92 | 0.963 | 14/13 | [summary.txt](fight-demo/runs/protogcn_ft/yolo26n-pose_my_botsort_protogcn_finegym_j/summary.txt) |
| joint-motion (FineGYM) | 0.85 | 0.93 | 0.89 | 0.84–0.93 | 0.958 | 18/7 | [summary.txt](fight-demo/runs/protogcn_ft/yolo26n-pose_my_botsort_protogcn_finegym_jm/summary.txt) |
| bone-motion (FineGYM) | 0.89 | 0.88 | 0.88 | 0.84–0.93 | 0.949 | 12/13 | [summary.txt](fight-demo/runs/protogcn_ft/yolo26n-pose_my_botsort_protogcn_finegym_bm/summary.txt) |
| joint, train từ đầu, 100 prototype | 0.85 | 0.89 | 0.87 | 0.82–0.91 | 0.918 | 17/12 | [summary.txt](fight-demo/runs/protogcn_ft/yolo26n-pose_my_botsort_scratch_p100/summary.txt) |
| joint, train từ đầu, 400 prototype | 0.88 | 0.88 | 0.88 | 0.83–0.92 | 0.931 | 13/13 | [summary.txt](fight-demo/runs/protogcn_ft/yolo26n-pose_my_botsort_scratch_p400/summary.txt) |
| bone-motion, train từ đầu, 100 prototype | 0.84 | 0.86 | 0.85 | 0.80–0.90 | 0.929 | 17/15 | [summary.txt](fight-demo/runs/protogcn_ft/yolo26n-pose_my_botsort_scratch_p100_bm/summary.txt) |
| ensemble j+b | 0.92 | 0.92 | 0.92 | 0.88–0.95 | 0.979 | 9/8 | [ensemble_protogcn_finegym.txt](fight-demo/runs/protogcn_ft/ensemble_protogcn_finegym.txt) |
| ensemble j+b+jm | 0.90 | 0.92 | 0.91 | 0.87–0.95 | 0.977 | 11/8 | [ensemble_protogcn_finegym_3luong.txt](fight-demo/runs/protogcn_ft/ensemble_protogcn_finegym_3luong.txt) |
| ensemble 4 luồng j+b+jm+bm | 0.90 | 0.92 | 0.91 | 0.86–0.94 | 0.974 | 11/9 | [ensemble_protogcn_finegym.txt](fight-demo/runs/protogcn_ft/ensemble_protogcn_finegym.txt) |

- Tương quan xác suất giữa các luồng 0.88–0.94 nên gộp luồng không hơn bone riêng (kể cả đủ 4 luồng).
- Hai lần chạy bm (ngày 18/11) chạy song song trên cùng GPU nên s/epoch và ms/video của chúng không so được với các lần khác.
- McNemar bone vs ST-GCN 1p: bone đúng / ST-GCN sai = 13, ngược lại = 5, p = 0.096 → chưa có ý nghĩa thống kê.
- Luồng tốt nhất được chọn trên chính tập test nên số liệu lạc quan; mục 9 (5-fold) cho thấy bone thực tế ngang joint và ST-GCN.

**Ablation loss CSC** (cấu hình **[13]** hoặc `python run_csc_ablation.py`; luồng bone, cùng seed / dữ liệu / 12 epoch, chỉ đổi λ qua `--csc-weight`):

| λ (trọng số CSC) | P | R | F1 | AUC | FP/FN | Kết quả |
|---|---|---|---|---|---|---|
| 0 (tắt CSC, chỉ còn CE; PRN không được học) | 0.90 | 0.92 | 0.91 | 0.972 | 11/9 | [summary.txt](fight-demo/runs/protogcn_ft/yolo26n-pose_my_botsort_protogcn_finegym_b_csc0/summary.txt) |
| 0.2 (code gốc) | 0.93 | 0.93 | 0.93 | 0.973 | 8/7 | [summary.txt](fight-demo/runs/protogcn_ft/yolo26n-pose_my_botsort_protogcn_finegym_b/summary.txt) |
| 0.3 (tối ưu theo bài báo) | 0.90 | 0.89 | 0.89 | 0.970 | 11/12 | [summary.txt](fight-demo/runs/protogcn_ft/yolo26n-pose_my_botsort_protogcn_finegym_b_csc0.3/summary.txt) |

- AUC gần như không đổi (0.970–0.973); F1 lên xuống không theo thứ tự của λ (0.91 → 0.93 → 0.89), chỉ khác nhau 7–8 video trên 210 → nằm trong nhiễu.
- Chưa có bằng chứng loss CSC / cơ chế prototype giúp ích trên RLVS (2 lớp); lợi ích đến từ backbone + pretrain FineGYM. Muốn kết luận chắc phải chạy 5-fold cho từng λ.

## 9. Đánh giá 5-fold + validation
Chạy: **[9]** ([run_bm_and_cv.py](fight-demo/run_bm_and_cv.py): bone, joint, ST-GCN) rồi **[14]** ([run_cv_all_streams.py](fight-demo/run_cv_all_streams.py): thêm joint-motion, bone-motion) · Code: [cv_models.py](fight-demo/cv_models.py), [splits:35](fight-demo/cv_models.py#L35), gộp ngưỡng 0.5: [ensemble_cv.py](fight-demo/ensemble_cv.py)

```bash
python cv_models.py --models protogcn_b,protogcn_j,protogcn_jm,protogcn_bm,stgcn_1p   # fold đã xong được bỏ qua
python cv_models.py --report   # chỉ in bảng từ kết quả đã có
python ensemble_cv.py          # gộp mô hình trên dự đoán out-of-fold, ngưỡng cố định 0.5
```
Mỗi fold: 1/5 video làm test (~210), phần còn lại tách 15% làm validation (~126), train ~713. McNemar tính trên dự đoán out-of-fold của cả 1049 video, so với ST-GCN.
Kết quả: [ensemble_nguong05.txt](fight-demo/runs/cv/yolo26n-pose_my_botsort/ensemble_nguong05.txt) · [summary.txt](fight-demo/runs/cv/yolo26n-pose_my_botsort/summary.txt) (5 mô hình, có ngưỡng học trên val và stacking) · [summary_3mohinh.txt](fight-demo/runs/cv/yolo26n-pose_my_botsort/summary_3mohinh.txt) · [oof_pred.csv](fight-demo/runs/cv/yolo26n-pose_my_botsort/oof_pred.csv)

| Mô hình (trung bình xác suất, ngưỡng 0.5) | F1 tb ± std | AUC OOF | FP/FN (1049 video) | McNemar so với ST-GCN |
|---|---|---|---|---|
| **Gộp 5 mô hình** (4 luồng ProtoGCN + ST-GCN) | **0.934 ± 0.003** | 0.980 | 38/32 | 39 / 13, **p = 0.0004** |
| **Gộp 4 luồng ProtoGCN** (b + j + jm + bm) | **0.933 ± 0.008** | 0.980 | 36/35 | 47 / 22, **p = 0.004** |
| ProtoGCN bone + joint | 0.930 ± 0.014 | 0.979 | 37/37 | 48 / 26, p = 0.014 |
| ProtoGCN joint | 0.916 ± 0.015 | 0.974 | 42/47 | 43 / 36, p = 0.50 |
| ProtoGCN bone-motion | 0.915 ± 0.012 | 0.969 | 48/42 | p = 0.61 |
| ProtoGCN joint-motion | 0.914 ± 0.007 | 0.969 | 48/43 | 39 / 34, p = 0.64 |
| ST-GCN 1p | 0.911 ± 0.017 | 0.965 | 59/37 | – |
| ProtoGCN bone | 0.909 ± 0.008 | 0.974 | 50/47 | 48 / 49, p = 1.00 |

- **Từng luồng ProtoGCN riêng lẻ ngang ST-GCN** (F1 0.909–0.916 so với 0.911; p ≥ 0.50). F1 0.93 của bone ở mục 8 là do lần tách 210 video thuận lợi.
- **Gộp 4 luồng ProtoGCN (đúng cách bài báo dùng) hơn ST-GCN có ý nghĩa thống kê**: F1 0.933, cảnh báo sai 59 → 36. Thêm ST-GCN vào được 0.934 và ổn định nhất giữa các fold (± 0.003).
- AUC từng luồng bone và joint cao hơn ST-GCN ở cả 5/5 fold (0.974 so với 0.965).
- **Ngưỡng học trên validation không ổn định** (val chỉ ~126 video): gộp 5 mô hình chỉ còn 0.919 (p = 0.32), gộp 3 mô hình (b + j + ST-GCN) 0.932 (p = 0.004), ST-GCN một mình kém đi (0.893). Stacking 5 mô hình bằng hồi quy logistic: 0.927 (p = 0.013). → dùng ngưỡng 0.5.
- Bảng trên là phân tích thêm sau khi có kết quả, thử nhiều tổ hợp nên p chưa hiệu chỉnh so sánh nhiều lần; với Bonferroni cho 6 tổ hợp (ngưỡng 0.0083) thì gộp 5 mô hình và gộp 4 luồng vẫn có ý nghĩa.

## 10. Phân tích lỗi (210 video test)
Chạy trong terminal: `python analyze_errors.py` (thêm `--sheets` để tạo ảnh ghép) · Code: [analyze_errors.py](fight-demo/analyze_errors.py)

Kết quả: [summary.txt](fight-demo/runs/error_analysis/summary.txt) · [errors.csv](fight-demo/runs/error_analysis/errors.csv) (47 video có ít nhất 1 mô hình sai) · ảnh [cảnh báo sai](fight-demo/runs/error_analysis/proto_b_FP.jpg), [bỏ sót](fight-demo/runs/error_analysis/proto_b_FN.jpg)

| Nhóm video | Số video | Gồm |
|---|---|---|
| ProtoGCN bone và ST-GCN 1p cùng sai | 10 | 7 FP, 3 FN |
| Chỉ ST-GCN 1p sai | 13 | 9 FP, 4 FN |
| Chỉ ProtoGCN bone sai | 5 | 1 FP, 4 FN |

- Bone hơn ST-GCN ở cảnh báo sai (sửa 9, thêm 1); số bỏ sót bằng nhau (7) nhưng chỉ trùng 3 video.
- 6 video sai ở cả 6 mô hình GCN: NV_233, NV_258, NV_568, NV_590, NV_687, V_843.
- FP của bone (8): thể thao đối kháng (NV_118, NV_566, NV_568), nhảy múa (NV_233, NV_258), đám đông (NV_454, NV_590, NV_687). Trung vị 3.62 người/frame, tốc độ tay chân p90 = 0.49 (video bình thường đoán đúng: 0.07).
- FN của bone (7): vật lộn dưới đất / người chồng lên nhau (V_254, V_255, V_842, V_843), cận cảnh bị che (V_43, V_721), chỉ thấy 1 người (V_967). Trung vị 1.35 người/frame → lỗi từ khâu phát hiện khớp.

## 11. Bảng tổng hợp

Cột "F1 riêng" là số đo ở từng mục (tập đánh giá khác nhau); "F1 (210 test)" đo trên cùng 210 video của một lần tách; "F1 5-fold" là trung bình 5 fold trên 1049 video (đáng tin nhất).

| Phương pháp | F1 riêng | F1 (210 test) | F1 5-fold | Ghi chú |
|---|---|---|---|---|
| Luật heuristic | 0.76 | 0.72 | | + impact 0.02 → 0.77 (500+500 video) |
| Random Forest | 0.87 | 0.84 | | chuyển động cả cảnh một mình đã 0.78; AUC 0.92 |
| ST-GCN zero-shot | 0.67 | | | AUC 0.40 |
| ST-GCN probe 256 | 0.88 | 0.88 | | AUC 0.92 |
| ST-GCN fine-tune | | 0.90 | 0.911 | AUC 5-fold 0.966 |
| ST-GCN đồ thị 2 người | | 0.88 | | AUC 0.93 |
| ProtoGCN joint | | 0.87 | 0.916 | AUC 5-fold 0.974 |
| ProtoGCN joint-motion | | 0.89 | 0.914 | AUC 5-fold 0.969 |
| ProtoGCN bone-motion | | 0.88 | 0.915 | AUC 5-fold 0.969 |
| ProtoGCN bone | | 0.93 | 0.909 | AUC 5-fold 0.975; ngang ST-GCN (McNemar p = 1.00) |
| ProtoGCN ensemble j+b | | 0.92 | | AUC 0.98 |
| **Gộp 4 luồng ProtoGCN** | | 0.91 | **0.933** | AUC 0.980; hơn ST-GCN có ý nghĩa thống kê (p = 0.004) |
| **Gộp 4 luồng ProtoGCN + ST-GCN** | | | **0.934** | AUC 0.980; p = 0.0004; ngưỡng 0.5 |

**Kết luận:** bản port ProtoGCN đúng với code gốc (mục 8, kiểm chứng FineGYM: 4 luồng 1 clip và joint 10 clip đều khớp log tác giả). Trên RLVS, từng luồng ProtoGCN riêng lẻ ngang ST-GCN về F1 (≈ 0.91), nhỉnh hơn về AUC; gộp 4 luồng ProtoGCN đạt F1 0.933 và hơn ST-GCN có ý nghĩa thống kê (p = 0.004), thêm ST-GCN được 0.934 (p = 0.0004). Loss CSC chưa cho thấy lợi ích riêng trên bài toán 2 lớp (mục 8).
