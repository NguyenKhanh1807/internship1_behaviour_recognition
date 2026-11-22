"""Mo xe ST-GCN (MMAction2) TUNG BUOC tren skeleton cua chinh ban - KHONG can cai MMAction2 / mmcv.

Code o day CHEP SAT code goc MMAction2 v1.2.0 (moi buoc ghi file + dong goc), dat ten module GIONG HET
de nap duoc checkpoint chinh thuc. Chi can numpy + torch + cv2 (da co trong fight-demo/.venv).

1) Tai checkpoint ST-GCN joint, NTU-60 X-Sub, skeleton 2D COCO-17 (12 MB, 1 lan):
   curl -L -o debug/ckpt/stgcn_ntu60_2d_joint.pth "https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth"
2) Chay (tu thu muc "Internship 1", venv da bat):
   python debug/debug_stgcn.py                                        # mac dinh: 1 video danh nhau trong cache RLVS
   python debug/debug_stgcn.py --kp fight-demo/runs/sparring_botsort_reid_y26/keypoints.csv --video fight-demo/videos/sparring.mp4
   PAUSE=1 python debug/debug_stgcn.py                                # dung lai sau moi buoc (pdb)
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FD = ROOT / "fight-demo"
PAUSE = os.environ.get("PAUSE") == "1"
np.set_printoptions(precision=3, suppress=True, linewidth=140)

NTU60 = ["drink water", "eat meal", "brush teeth", "brush hair", "drop", "pick up", "throw", "sit down",
         "stand up", "clapping", "reading", "writing", "tear up paper", "wear jacket", "take off jacket",
         "wear a shoe", "take off a shoe", "wear glasses", "take off glasses", "put on a hat", "take off a hat",
         "cheer up", "hand waving", "kicking something", "reach into pocket", "hopping", "jump up",
         "phone call", "play with phone", "typing", "point to something", "taking a selfie", "check time",
         "rub two hands", "nod head / bow", "shake head", "wipe face", "salute", "put palms together",
         "cross hands in front", "sneeze / cough", "staggering", "falling", "touch head", "touch chest",
         "touch back", "touch neck", "nausea / vomiting", "use a fan",
         "A50 punching/slapping other person", "A51 kicking other person", "A52 pushing other person",
         "A53 pat on back", "A54 point finger at other", "A55 hugging", "A56 giving object", "A57 touch other's pocket",
         "A58 handshaking", "A59 walking towards", "A60 walking apart"]
FIGHT = [49, 50, 51]  # chi so 0-based cua A50, A51, A52


def step(t):
    print("\n" + "=" * 90 + f"\n{t}\n" + "=" * 90)


def pause():
    if PAUSE:
        breakpoint()


# =============================== PHAN 1: DU LIEU (numpy) ===============================
def load_skeleton(kp_csv: Path, num_person: int = 2, select: str = "score"):
    """keypoints.csv (fight_demo.py) -> keypoint (M, T, V, 2), keypoint_score (M, T, V).
    Giong dinh dang annotation cua MMAction2 PoseDataset (moi nguoi 1 hang, frame thieu = 0)."""
    rows = defaultdict(dict)  # track_id -> frame -> (17, 3)
    frames = set()
    with open(kp_csv, newline="") as fh:
        rd = csv.reader(fh)
        next(rd)
        for r in rd:
            f, tid = int(r[0]), int(r[1])
            rows[tid][f] = np.array(r[2:53], dtype=np.float32).reshape(17, 3)
            frames.add(f)
    T = max(frames) + 1
    # chon num_person track "noi bat" nhat: tong do tin cay keypoint lon nhat (xuat hien lau + ro)
    score = {tid: sum(k[:, 2].sum() for k in fr.values()) for tid, fr in rows.items()}
    chosen = sorted(score, key=score.get, reverse=True)[:num_person]
    if select == "pair" and num_person == 2:
        pair = best_pair(rows)
        if pair is not None:
            chosen = list(pair)
    kp = np.zeros((len(chosen), T, 17, 2), np.float32)
    sc = np.zeros((len(chosen), T, 17), np.float32)
    for m, tid in enumerate(chosen):
        for f, k in rows[tid].items():
            kp[m, f], sc[m, f] = k[:, :2], k[:, 2]
    return kp, sc, chosen, len(rows)


def _center_height(k):
    """Tam than (vai + hong) va chieu cao (khoang y cua cac khop thay ro) cua 1 skeleton (17, 3)."""
    torso = [i for i in (5, 6, 11, 12) if k[i, 2] >= 0.3]
    vis = k[k[:, 2] >= 0.3]
    if not torso or len(vis) < 4:
        return None, None
    return k[torso, :2].mean(0), max(np.ptp(vis[:, 1]), 1.0)


def best_pair(rows):
    """Chon CAP track tuong tac nhieu nhat: so frame ca hai cung co mat VA o gan nhau (< 1.5 x chieu cao).
    Khac cach mac dinh (2 track ro nhat), cap nay moi la 2 nguoi co kha nang dang danh nhau."""
    info = {tid: {f: _center_height(k) for f, k in fr.items()} for tid, fr in rows.items()}
    best, best_key = None, (0, 0)
    tids = sorted(rows)
    for i, a in enumerate(tids):
        for b in tids[i + 1:]:
            close = co = 0
            for f in rows[a].keys() & rows[b].keys():
                (ca, ha), (cb, hb) = info[a][f], info[b][f]
                if ca is None or cb is None:
                    continue
                co += 1
                close += np.linalg.norm(ca - cb) < 1.5 * (ha + hb) / 2
            if (close, co) > best_key:
                best, best_key = (a, b), (close, co)
    return best


def pre_normalize_2d(kp, img_shape):
    """pose_transforms.py:710 PreNormalize2D - dua toa do pixel ve [-1, 1], tam anh = 0."""
    h, w = img_shape
    kp = kp.copy()
    kp[..., 0] = (kp[..., 0] - w / 2) / (w / 2)
    kp[..., 1] = (kp[..., 1] - h / 2) / (h / 2)
    return kp


def gen_ske_feat_joint(kp, sc):
    """pose_transforms.py:932 GenSkeFeat(feats=['j']) - ghep (x, y) voi score -> C = 3."""
    return np.concatenate([kp, sc[..., None]], -1)


def uniform_sample_test(num_frames, clip_len=100, num_clips=1, seed=255):
    """pose_transforms.py:995 UniformSampleFrames._get_test_clips (test_mode=True). Tra ve chi so frame."""
    np.random.seed(seed)
    all_inds = []
    for i in range(num_clips):
        if num_frames < clip_len:                       # video ngan hon 100 frame -> lap vong
            start = i if num_frames < num_clips else i * num_frames // num_clips
            inds = np.arange(start, start + clip_len)
        elif num_frames < 2 * clip_len:                 # 100..199 frame -> chen ngau nhien cac frame
            basic = np.arange(clip_len)
            pick = np.random.choice(clip_len + 1, num_frames - clip_len, replace=False)
            offset = np.zeros(clip_len + 1, np.int64)
            offset[pick] = 1
            inds = basic + np.cumsum(offset)[:-1]
        else:                                           # >= 200 frame -> chia 100 doan, moi doan lay 1 frame
            bids = np.array([j * num_frames // clip_len for j in range(clip_len + 1)])
            inds = bids[:clip_len] + np.random.randint(np.diff(bids))
        all_inds.append(inds)
    return np.mod(np.concatenate(all_inds), num_frames)


def format_gcn_input(x, num_person=2, num_clips=1):
    """formatting.py:388 FormatGCNInput - dem/cat ve dung num_person, tach clip: (nc, M, T, V, C)."""
    M = x.shape[0]
    if M < num_person:
        x = np.concatenate([x, np.zeros((num_person - M,) + x.shape[1:], x.dtype)], 0)
    x = x[:num_person]
    M, T, V, C = x.shape
    return np.ascontiguousarray(x.reshape(M, num_clips, T // num_clips, V, C).transpose(1, 0, 2, 3, 4))


# =============================== PHAN 2: DO THI (numpy) ===============================
COCO_INWARD = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6), (9, 7), (7, 5), (10, 8), (8, 6),
               (5, 0), (6, 0), (1, 0), (3, 1), (2, 0), (4, 2)]   # graph.py:169-171
CENTER = 0                                                       # graph.py:172 (mui)


def hop_distance(num_node, edges, max_hop=1):
    """graph.py:81 get_hop_distance - so canh ngan nhat giua 2 khop (chi tinh toi max_hop, con lai = inf)."""
    A = np.eye(num_node)
    for i, j in edges:
        A[i, j] = A[j, i] = 1
    hop = np.full((num_node, num_node), np.inf)
    mats = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    for d in range(max_hop, -1, -1):
        hop[np.stack(mats) [d] > 0] = d
    return hop


def normalize_digraph(A):
    """graph.py:57 - chia moi cot cho bac cua dinh -> lay TRUNG BINH hang xom thay vi tong."""
    deg = A.sum(0)
    Dn = np.diag([1 / d if d > 0 else 0 for d in deg])
    return A @ Dn


def stgcn_spatial(num_node=17, edges=COCO_INWARD, center=CENTER, max_hop=1):
    """graph.py:183 Graph.stgcn_spatial - chia hang xom thanh 3 nhom: A0 goc, A1 ly tam, A2 huong tam."""
    hop = hop_distance(num_node, edges, max_hop)
    adj = (hop <= max_hop).astype(float)
    nadj = normalize_digraph(adj)
    A = []
    for h in range(max_hop + 1):
        close, further = np.zeros((num_node, num_node)), np.zeros((num_node, num_node))
        for i in range(num_node):
            for j in range(num_node):
                if hop[j, i] == h:
                    if hop[j, center] >= hop[i, center]:
                        close[j, i] = nadj[j, i]
                    else:
                        further[j, i] = nadj[j, i]
        A.append(close)
        if h > 0:
            A.append(further)
    return np.stack(A).astype(np.float32)


# =============================== PHAN 3: MODEL (torch) ===============================
def build_model():
    import torch
    import torch.nn as nn

    class unit_gcn(nn.Module):  # gcn_utils.py: unit_gcn (adaptive='importance', conv_pos='pre')
        def __init__(self, cin, cout, A):
            super().__init__()
            self.num_subsets = A.size(0)
            self.register_buffer("A", A)                       # do thi co dinh (3, 17, 17)
            self.PA = nn.Parameter(torch.ones_like(A))         # trong so tam quan trong cua canh, hoc duoc
            self.conv = nn.Conv2d(cin, cout * A.size(0), 1)    # W cho ca 3 nhom cung luc
            self.bn = nn.BatchNorm2d(cout)
            self.act = nn.ReLU()

        def forward(self, x, verbose=False):
            n, c, t, v = x.shape
            A = self.A * self.PA
            x = self.conv(x)                                   # (n, 3*Cout, T, V)
            x = x.view(n, self.num_subsets, -1, t, v)          # (n, 3, Cout, T, V)
            if verbose:
                print(f"      gcn: conv 1x1 -> {tuple(x.shape)}  (n, K=3 nhom, Cout, T, V)")
            x = torch.einsum("nkctv,kvw->nctw", (x, A)).contiguous()
            if verbose:
                print(f"      gcn: einsum voi A (3,17,17) -> {tuple(x.shape)}  (cong 3 nhom, tron khop ke nhau)")
            return self.act(self.bn(x))

    class unit_tcn(nn.Module):  # gcn_utils.py: unit_tcn
        def __init__(self, cin, cout, kernel_size=9, stride=1):
            super().__init__()
            pad = (kernel_size - 1) // 2
            self.conv = nn.Conv2d(cin, cout, (kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
            self.bn = nn.BatchNorm2d(cout)
            self.drop = nn.Dropout(0)

        def forward(self, x):
            return self.drop(self.bn(self.conv(x)))

    class STGCNBlock(nn.Module):  # stgcn.py:16
        def __init__(self, cin, cout, A, stride=1, residual=True):
            super().__init__()
            self.gcn = unit_gcn(cin, cout, A)
            self.tcn = unit_tcn(cout, cout, 9, stride)
            self.relu = nn.ReLU()
            if not residual:
                self.residual = lambda x: 0
            elif cin == cout and stride == 1:
                self.residual = lambda x: x
            else:
                self.residual = unit_tcn(cin, cout, kernel_size=1, stride=stride)

        def forward(self, x, verbose=False):
            res = self.residual(x)
            y = self.gcn(x, verbose)
            y = self.tcn(y)
            if verbose:
                print(f"      tcn: conv (9 x 1) theo thoi gian -> {tuple(y.shape)}")
            return self.relu(y + res)

    class STGCN(nn.Module):  # stgcn.py: STGCN (base 64, 10 stage, inflate/down o stage 5 va 8)
        def __init__(self, A, in_channels=3, base=64):
            super().__init__()
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))   # 'VC': chuan hoa 51 = 17 khop x 3
            mods = [STGCNBlock(in_channels, base, A.clone(), 1, residual=False)]
            inflate = 0
            for i in range(2, 11):
                stride = 1 + (i in (5, 8))
                cin = base
                if i in (5, 8):
                    inflate += 1
                cout = 64 * 2 ** inflate
                base = cout
                mods.append(STGCNBlock(cin, cout, A.clone(), stride))
            self.gcn = nn.ModuleList(mods)

        def forward(self, x, verbose=False):
            N, M, T, V, C = x.size()
            x = x.permute(0, 1, 3, 4, 2).contiguous()                 # (N, M, V, C, T)
            x = self.data_bn(x.view(N * M, V * C, T))                  # BN tren 51 kenh
            x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
            if verbose:
                print(f"  vao backbone: {tuple(x.shape)}  (N*M, C, T, V): 2 nguoi thanh 2 mau DOC LAP")
            for i, blk in enumerate(self.gcn):
                x = blk(x, verbose and i in (0, 4))
                if verbose:
                    print(f"  block {i + 1:>2}: {tuple(x.shape)}")
            return x.reshape((N, M) + x.shape[1:])

    class GCNHead(nn.Module):  # heads/gcn_head.py
        def __init__(self, num_classes=60, in_channels=256):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(in_channels, num_classes)

        def forward(self, x, verbose=False):
            N, M, C, T, V = x.shape
            x = self.pool(x.view(N * M, C, T, V)).view(N, M, C)        # trung binh theo T va V
            if verbose:
                print(f"  pool T, V: {tuple(x.shape)}  (N, M, C) - moi nguoi con 1 vector 256 so")
            x = x.mean(dim=1)                                          # gcn_head.py:64 TRUNG BINH 2 NGUOI
            if verbose:
                print(f"  mean theo M (gcn_head.py:64): {tuple(x.shape)}  <- tuong tac 2 nguoi bi gop mat o day")
            return self.fc(x)

    class Recognizer(nn.Module):  # recognizers/recognizer_gcn.py
        def __init__(self, A):
            super().__init__()
            self.backbone = STGCN(A)
            self.cls_head = GCNHead()

    return Recognizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp", default=None, help="keypoints.csv (mac dinh: video V_ dau tien trong cache RLVS)")
    ap.add_argument("--video", default=None, help="video goc (de lay kich thuoc anh)")
    ap.add_argument("--ckpt", default=str(ROOT / "debug" / "ckpt" / "stgcn_ntu60_2d_joint.pth"))
    ap.add_argument("--num-clips", type=int, default=10, help="test protocol MMAction2: 10 clip, lay trung binh")
    args = ap.parse_args()

    cache = FD / "runs" / "eval_cache" / "yolo26n-pose_my_botsort"
    kp_csv = Path(args.kp) if args.kp else sorted((cache / "pos").glob("*/keypoints.csv"))[0]
    video = Path(args.video) if args.video else next((FD / "datasets" / "Violence").glob(kp_csv.parent.name + ".*"), None)

    # ---------------------------------------------------------------------------------
    step("BUOC 1. keypoints.csv -> annotation kieu MMAction2 (PoseDataset)")
    kp, sc, chosen, n_ids = load_skeleton(kp_csv)
    img_shape = (1080, 1920)
    if video is not None:
        import cv2
        cap = cv2.VideoCapture(str(video))
        img_shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        cap.release()
    print(f"File: {kp_csv}\nVideo: {video}  img_shape (H, W) = {img_shape}")
    print(f"Co {n_ids} track ID, chon 2 track noi bat nhat: {chosen}")
    print(f"keypoint {kp.shape} (M, T, V, 2)   keypoint_score {sc.shape} (M, T, V)")
    print(f"So frame co mat: nguoi 0 = {(sc[0].sum(-1) > 0).sum()}, nguoi 1 = {(sc[1].sum(-1) > 0).sum() if len(sc) > 1 else 0} / {kp.shape[1]}")
    pause()

    step("BUOC 2. PreNormalize2D (pose_transforms.py:710): pixel -> [-1, 1]")
    f0 = int(np.argmax(sc[0].sum(-1) > 0))
    print(f"Mui nguoi 0, frame {f0}: pixel {kp[0, f0, 0]}")
    kpn = pre_normalize_2d(kp, img_shape)
    print(f"  -> chuan hoa {kpn[0, f0, 0]}   (x - W/2)/(W/2), (y - H/2)/(H/2): tam anh = (0, 0)")
    pause()

    step("BUOC 3. GenSkeFeat feats=['j'] (pose_transforms.py:932): ghep score -> C = 3")
    x = gen_ske_feat_joint(kpn, sc)
    print(f"{x.shape} (M, T, V, C=3: x, y, score)")
    pause()

    step(f"BUOC 4. UniformSampleFrames clip_len=100, num_clips={args.num_clips}, test_mode (pose_transforms.py:995)")
    T = x.shape[1]
    branch = "< 100: lap vong" if T < 100 else ("100..199: chen ngau nhien frame lap" if T < 200 else ">= 200: 100 doan, moi doan 1 frame")
    inds = uniform_sample_test(T, 100, args.num_clips)
    print(f"Video co {T} frame -> nhanh '{branch}'")
    print(f"Chi so frame clip 1 (20 so dau): {inds[:20]}")
    print(f"Tong {len(inds)} chi so = {args.num_clips} clip x 100 frame")
    pause()

    step("BUOC 5. PoseDecode + FormatGCNInput (formatting.py:388): -> (num_clips, M, T, V, C)")
    x = x[:, inds]                                              # PoseDecode: lay dung cac frame da chon
    x = format_gcn_input(x, 2, args.num_clips)
    print(f"{x.shape}  = (num_clips, M=2, T=100, V=17, C=3) - day la 'inputs' cua RecognizerGCN")
    pause()

    step("BUOC 6. Do thi khung xuong: Graph(layout='coco', mode='stgcn_spatial') (graph.py:183)")
    A = stgcn_spatial()
    print(f"A {A.shape}: 3 nhom (A0 goc, A1 ly tam, A2 huong tam) x 17 x 17")
    names = ["mui", "mat T", "mat P", "tai T", "tai P", "vai T", "vai P", "khuyu T", "khuyu P", "co tay T",
             "co tay P", "hong T", "hong P", "goi T", "goi P", "co chan T", "co chan P"]
    for j in (8, 6, 0):
        groups = [[names[i] for i in range(17) if A[k, i, j] > 0] for k in range(3)]
        print(f"  {names[j]:<8}: A0 goc {groups[0]} | A1 ly tam (xa mui hon) {groups[1]} | A2 huong tam (gan mui hon) {groups[2]}")
    print("  Cot j cua A_k = khop j NHAN thong tin tu nhung khop nao. Trong code: a_close = hang xom KHONG gan tam hon")
    print("  khop dang xet (tuc ly tam), a_further = hang xom gan tam hon (huong tam). Tam (center) = mui.")
    pause()

    # ---------------------------------------------------------------------------------
    try:
        import torch
    except ImportError:
        print("\nChua co torch -> dung o day (phan model can venv fight-demo/.venv).")
        return
    torch.set_printoptions(precision=3, sci_mode=False)
    model = build_model()(torch.from_numpy(A)).eval()
    n_params = sum(p.numel() for p in model.parameters())

    step(f"BUOC 7. Nap checkpoint ST-GCN NTU-60 2D joint ({n_params / 1e6:.2f} trieu tham so)")
    if Path(args.ckpt).exists():
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Nap {len(sd)} tensor tu {Path(args.ckpt).name}. Thieu: {missing or 'khong'}  Thua: {unexpected or 'khong'}")
        loaded = not missing
    else:
        print(f"KHONG thay {args.ckpt} -> dung trong so NGAU NHIEN (shape dung, ket qua du doan VO NGHIA).")
        print("Tai checkpoint bang lenh curl o dau file nay.")
        loaded = False
    pause()

    step("BUOC 8. Backbone STGCN (stgcn.py: forward) - shape qua tung block")
    inp = torch.from_numpy(x[:1])                               # clip 1: (1, M, T, V, C)
    with torch.no_grad():
        feat = model.backbone(inp, verbose=True)
    print(f"Ra backbone: {tuple(feat.shape)}  (N, M, C=256, T=25, V=17)")
    pause()

    step("BUOC 9. GCNHead (gcn_head.py): pool -> trung binh 2 nguoi -> fc 60 lop")
    with torch.no_grad():
        logit = model.cls_head(feat, verbose=True)
        print(f"  fc: {tuple(logit.shape)} (N, 60 lop NTU)")
        allc = model.cls_head(model.backbone(torch.from_numpy(x)))       # ca num_clips clip
        prob = allc.softmax(-1).mean(0)                                   # trung binh xac suat cac clip
    pause()

    step(f"BUOC 10. Ket qua (trung binh {args.num_clips} clip){'' if loaded else '  [TRONG SO NGAU NHIEN - KHONG CO Y NGHIA]'}")
    top = prob.argsort(descending=True)[:5]
    for r, i in enumerate(top.tolist(), 1):
        print(f"  top{r}: {NTU60[i]:<36} {prob[i]:.3f}")
    fight = prob[FIGHT].sum().item()
    print(f"\nDiem 'danh nhau' = P(A50) + P(A51) + P(A52) = {fight:.3f}")
    print("  " + ", ".join(f"{NTU60[i].split()[0]}={prob[i]:.3f}" for i in FIGHT))


if __name__ == "__main__":
    main()
