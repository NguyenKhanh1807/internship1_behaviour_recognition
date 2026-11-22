"""Debug TUNG BUOC 1 batch training YOLO-pose tren coco8-pose (4 anh). Khong train that, chi 1 buoc.

Cach chay (tu thu muc "Internship 1", da activate venv):
    python debug/debug_train.py              # co augmentation (Mosaic, flip...)
    python debug/debug_train.py --no-aug     # tat augmentation de nhan de doi chieu voi file .txt
    PAUSE=1 python debug/debug_train.py      # dung lai sau moi buoc (pdb)
Hoac F5 trong VS Code, dat breakpoint o cac dong "# >>> BREAKPOINT" va trong source (xem huong dan).
Lan dau chay se tu tai dataset coco8-pose (~1 MB).
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset, img2label_paths
from ultralytics.utils import DEFAULT_CFG

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "debug"

# ============================ INPUT: DOI O DAY ================================
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(ROOT / "fight-demo" / "yolo11n-pose.pt"))
ap.add_argument("--data", default="coco8-pose.yaml", help="dataset yaml")
ap.add_argument("--no-aug", action="store_true", help="tat Mosaic / flip / HSV / affine")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
# ==============================================================================

PAUSE = os.environ.get("PAUSE") == "1"
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.set_printoptions(precision=3, sci_mode=False)

KPT_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
             "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]


def step(title):
    print("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80)


def pause():
    if PAUSE:
        breakpoint()


def draw(img_rgb_chw, bboxes, kpts, path):
    """Ve box + keypoint (toa do chuan hoa 0..1) len anh tensor RGB (3,H,W) roi luu ra file."""
    img = np.ascontiguousarray(img_rgb_chw.permute(1, 2, 0).numpy()[..., ::-1])  # RGB -> BGR cho cv2
    h, w = img.shape[:2]
    for (cx, cy, bw, bh), kp in zip(bboxes.tolist(), kpts.tolist()):
        cv2.rectangle(img, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)), (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), (0, 255, 0), 2)
        for j, (x, y, v) in enumerate(kp):
            if v > 0:
                color = (0, 0, 255) if j % 2 == 1 else (255, 0, 0)  # trai = do, phai = xanh (mui = xanh)
                cv2.circle(img, (int(x * w), int(y * h)), 4, color, -1)
    cv2.imwrite(str(path), img)


# -----------------------------------------------------------------------------
step("BUOC 1. Dataset yaml + file nhan goc  (data/utils.py check_det_dataset, cfg/datasets/coco8-pose.yaml)")
data = check_det_dataset(args.data)  # doc yaml, tu tai dataset neu chua co
print(f"Thu muc dataset: {data['path']}")
print(f"train: {data['train']}")
print(f"kpt_shape: {data['kpt_shape']}   flip_idx: {data['flip_idx']}")

img_files = sorted(Path(data["train"]).glob("*.jpg"))
img0 = img_files[0]
label0 = Path(img2label_paths([str(img0)])[0])
lines = label0.read_text().strip().splitlines()
print(f"\nAnh: {img0.name}   ->   file nhan: {label0.relative_to(Path(data['path']))}   ({len(lines)} nguoi)")
vals = [float(v) for v in lines[0].split()]
print(f"Dong 1 co {len(vals)} so = 1 class + 4 box + 17 x 3 keypoint")
print(f"  class={int(vals[0])}  box (cx, cy, w, h, chuan hoa 0..1) = {vals[1:5]}")
print("  keypoint (x, y chuan hoa 0..1; v: 0 = khong gan nhan, 1 = bi che, 2 = thay ro):")
kp_raw = np.array(vals[5:]).reshape(17, 3)
for j, (x, y, v) in enumerate(kp_raw):
    print(f"    {j:>2} {KPT_NAMES[j]:<11} x={x:.3f} y={y:.3f} v={int(v)}")
pause()  # >>> BREAKPOINT 1: xem data, lines, kp_raw

# -----------------------------------------------------------------------------
step("BUOC 2. flip_idx - vi sao lat anh phai doi trai <-> phai  (data/augment.py:1614-1632 RandomFlip)")
flip_idx = data["flip_idx"]
kp_flip_sai = kp_raw.copy()
kp_flip_sai[:, 0] = np.where(kp_flip_sai[:, 2] > 0, 1 - kp_flip_sai[:, 0], 0)  # chi lat toa do x
kp_flip_dung = kp_flip_sai[flip_idx]  # lat x + doi chi so theo flip_idx (dong 1632)
for j in (5, 6, 9, 10):
    print(f"  {KPT_NAMES[j]:<11} goc x={kp_raw[j, 0]:.3f} | chi lat x: {kp_flip_sai[j, 0]:.3f} | lat x + flip_idx: {kp_flip_dung[j, 0]:.3f}")
print("-> Lat guong thi vai TRAI that cua nguoi xuat hien o vi tri cua vai PHAI. Neu chi lat x ma khong doi chi so,")
print("   nhan 'l_shoulder' se nam o vi tri trong giong vai phai -> model hoc sai. flip_idx hoan doi cac cap trai/phai.")
pause()  # >>> BREAKPOINT 2

# -----------------------------------------------------------------------------
step("BUOC 3. Dataset + augmentation -> 1 mau  (data/base.py:439 __getitem__, data/augment.py v8_transforms)")
cfg = get_cfg(DEFAULT_CFG, dict(task="pose", imgsz=640))
if args.no_aug:
    for k in ("mosaic", "fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v", "translate", "scale", "degrees", "mixup"):
        setattr(cfg, k, 0.0)
print(f"Augment: mosaic={cfg.mosaic} fliplr={cfg.fliplr} scale={cfg.scale} translate={cfg.translate} hsv_v={cfg.hsv_v}")
ds = build_yolo_dataset(cfg, data["train"], batch=4, data=data, mode="train", stride=32)
print(f"Cac buoc augment: {[type(t).__name__ for t in ds.transforms.transforms]}")
sample = ds[0]  # <- dat breakpoint o data/base.py:441 de buoc vao tung transform
for k in ("img", "cls", "bboxes", "keypoints", "batch_idx"):
    v = sample[k]
    print(f"  {k:<10} {tuple(v.shape)}  dtype={v.dtype}")
print("  keypoints da chuan hoa 0..1 theo kich thuoc anh 640x640 (Format, augment.py:2482-2483)")
draw(sample["img"], sample["bboxes"], sample["keypoints"], OUT / "train_sample0.jpg")
print(f"Da luu anh sau augment: {OUT / 'train_sample0.jpg'}  (cham do = khop trai, xanh = khop phai)")
pause()  # >>> BREAKPOINT 3: xem sample

# -----------------------------------------------------------------------------
step("BUOC 4. Gom batch  (data/dataset.py:434 collate_fn) + chuan hoa anh (models/yolo/detect/train.py:119)")
batch = ds.collate_fn([ds[i] for i in range(len(ds))])
batch["img"] = batch["img"].float() / 255
for k in ("img", "cls", "bboxes", "keypoints", "batch_idx"):
    print(f"  {k:<10} {tuple(batch[k].shape)}")
print(f"  batch_idx = {batch['batch_idx'].int().tolist()}   <- nguoi nao thuoc anh nao trong batch")
print("  Moi anh co so nguoi khac nhau, nen nhan cua ca batch duoc NOI DAI ra, dung batch_idx de biet thuoc anh nao")
pause()  # >>> BREAKPOINT 4: xem batch

# -----------------------------------------------------------------------------
step("BUOC 5. Forward o che do TRAIN  (nn/tasks.py _predict_once -> head.py Detect.forward: if self.training: return preds)")
net = YOLO(args.model).model  # PoseModel
net.args = cfg  # loss doc he so tu day: box, cls, dfl, pose, kobj
for p in net.parameters():
    p.requires_grad_(True)
net.train()
preds = net(batch["img"])
print("Khi train, head tra ve gia tri THO (chua decode, chua NMS):")
for k in ("boxes", "scores", "kpts"):
    print(f"  {k:<7} {tuple(preds[k].shape)}")
A = preds["scores"].shape[-1]
print(f"  feats   {[tuple(f.shape) for f in preds['feats']]}")
print(f"  So o A = {A} (anh train la 640x640 vuong -> 80x80 + 40x40 + 20x20)")
pause()  # >>> BREAKPOINT 5: xem preds

# -----------------------------------------------------------------------------
step("BUOC 6. TaskAlignedAssigner - o nao chiu trach nhiem cho nguoi nao  (utils/loss.py:422, utils/tal.py:142)")
crit = net.init_criterion()  # v8PoseLoss (utils/loss.py:679)
(fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = crit.get_assigned_targets_and_loss(preds, batch)
print(f"Assigner: topk={crit.assigner.topk}, alpha={crit.assigner.alpha}, beta={crit.assigner.beta}  ->  align = score^0.5 * IoU^6")
print(f"fg_mask {tuple(fg_mask.shape)}: True = o duoc gan cho 1 nguoi (positive), False = nen (negative)")
levels = {"P3 (stride 8)": (0, 6400), "P4 (stride 16)": (6400, 8000), "P5 (stride 32)": (8000, 8400)}
for b in range(fg_mask.shape[0]):
    n_gt = int((batch["batch_idx"] == b).sum())
    pos = fg_mask[b].nonzero().flatten()
    per_level = {name: int(((pos >= s) & (pos < e)).sum()) for name, (s, e) in levels.items()}
    per_gt = [int((target_gt_idx[b, pos] == g).sum()) for g in range(n_gt)]
    print(f"  Anh {b}: {n_gt} nguoi | {len(pos)}/{A} o positive | theo tang {per_level} | so o moi nguoi {per_gt}")
print("-> Phan lon o la nen. Moi nguoi co toi da 10 o (topk). Nguoi to duoc gan o P5, nguoi nho o P3.")
pause()  # >>> BREAKPOINT 6: xem fg_mask, target_gt_idx

# -----------------------------------------------------------------------------
step("BUOC 7. Loss  (utils/loss.py:705 v8PoseLoss.loss, KeypointLoss 335, calculate_keypoints_loss 764)")
loss, items = net.loss(batch, preds)  # criterion(preds, batch)
gains = {"box_loss": cfg.box, "pose_loss": cfg.pose, "kobj_loss": cfg.kobj, "cls_loss": cfg.cls, "dfl_loss": cfg.dfl}
print(f"{'thanh phan':<10} {'gia tri (da nhan gain)':>24} {'gain':>6}  y nghia")
desc = {"box_loss": "CIoU giua box du doan va box that", "pose_loss": "1 - OKS cua 17 keypoint",
        "kobj_loss": "BCE: khop co duoc gan nhan khong", "cls_loss": "BCE: o co phai nguoi khong",
        "dfl_loss": "phan bo 16 bin cua 4 canh box"}
for k, v in items.items():
    print(f"{k:<10} {float(v):>24.4f} {gains.get(k, ''):>6}  {desc.get(k, '')}")
total = loss.sum()
print(f"Tong loss (x batch_size {batch['img'].shape[0]}) = {total.item():.4f}")
pause()  # >>> BREAKPOINT 7: xem loss, items

# -----------------------------------------------------------------------------
step("BUOC 8. Backward + 1 buoc cap nhat trong so  (engine/trainer.py: backward -> clip_grad_norm_ dong 913 -> optimizer.step)")


def eval_loss():
    """Tinh loss o che do eval (BatchNorm khong cap nhat) de so sanh truoc / sau cong bang."""
    net.eval()
    with torch.no_grad():
        l, it = net.loss(batch, net(batch["img"]))  # eval tra (y, preds) -> criterion tu lay preds tho
    net.train()
    return l.sum().item(), it


before, items_b = eval_loss()
net.zero_grad()
total.backward()
head = net.model[-1]
print("Gradient trung binh |grad| cua lop conv cuoi moi nhanh, theo tung tang:")
print(f"  {'':<14}{'P3 (stride 8)':>15}{'P4 (stride 16)':>16}{'P5 (stride 32)':>16}")
for name, branch in (("cv2 box", head.cv2), ("cv3 class", head.cv3), ("cv4 keypoint", head.cv4)):
    g = [branch[i][-1].weight.grad.abs().mean().item() for i in range(3)]
    print(f"  {name:<14}{g[0]:>15.2e}{g[1]:>16.2e}{g[2]:>16.2e}")
print(f"  layer 0 (conv dau tien): {net.model[0].conv.weight.grad.abs().mean().item():.2e}")
print("-> Tang khong co o positive nao (o day thuong la P3) thi box/keypoint KHONG co gradient: loss box va keypoint")
print("   chi tinh o o positive. Nhanh class van co gradient o moi tang vi BCE tinh tren TAT CA 8400 o.")

gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)  # giong trainer.py:913
print(f"\nDo lon gradient (norm) truoc khi cat = {gnorm:.1f} -> cat ve toi da 10 (trainer.py:913)")
opt = torch.optim.SGD(net.parameters(), lr=1e-4)
opt.step()
after, items_a = eval_loss()
print(f"Tong loss (eval) truoc = {before:.4f}  ->  sau 1 buoc = {after:.4f}")
print(f"   pose_loss truoc / sau: {float(items_b['pose_loss']):.4f} / {float(items_a['pose_loss']):.4f}")
print("   (model da train san nen thay doi rat nho; buoc qua lon hoac khong cat gradient se lam loss tang vot)")
pause()  # >>> BREAKPOINT 8

step("XONG. Training = nhan -> augment -> batch -> forward (tho) -> assigner -> loss -> backward -> cap nhat")
