"""Buoi 5: fine-tune ST-GCN (pretrained NTU-60 2D joint) thanh bo phan loai 2 lop: danh nhau / binh thuong.

Tach train/test 80/20 theo ten video (co dinh seed), so sanh voi luat, RF va probe256 tren CUNG tap test.
Chay nhieu lan cung lenh duoc: moi lan chay tiep tu checkpoint cho toi khi du --epochs, roi danh gia.

  python finetune_stgcn.py                 # prep -> train (tiep tuc) -> test
  python finetune_stgcn.py --budget 160    # dung sau ~160 giay (may ao gioi han thoi gian), chay lai de tiep
Ket qua: runs/stgcn_ft/<cache>/summary.txt, test_pred.csv, train_log.csv
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "debug"))
import debug_stgcn as D  # noqa: E402

FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]  # doi trai <-> phai (COCO 17)
# Do thi 2 nguoi (y tuong 2P-GCN): 34 dinh = nguoi A (0-16) + nguoi B (17-33).
# Canh trong than giu nguyen cho ca 2 nguoi; canh GIUA 2 nguoi: moi khop noi voi khop cung ten cua nguoi kia.
EDGES_2P = (D.COCO_INWARD + [(i + 17, j + 17) for i, j in D.COCO_INWARD] + [(i + 17, i) for i in range(17)])


def to_2p(c):
    """(nc, M=2, T, 17, C) -> (nc, 1, T, 34, C): ghep 2 nguoi thanh 1 do thi."""
    nc, M, T, V, C = c.shape
    return np.ascontiguousarray(c.transpose(0, 2, 1, 3, 4).reshape(nc, T, M * V, C)[:, None])


def build(torch, nn, ckpt, graph):
    """ST-GCN nap trong so NTU-60. graph='2p': A (3, 34, 34); data_bn va PA duoc nhan doi tu trong so 17 khop."""
    if graph == "1p":
        model = D.build_model()(torch.from_numpy(D.stgcn_spatial()))
    else:
        model = D.build_model()(torch.from_numpy(D.stgcn_spatial(34, EDGES_2P, 0)))
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    if graph == "2p":
        new = {}
        for k, v in sd.items():
            if k.startswith("backbone.data_bn.") and v.ndim == 1:
                v = torch.cat([v, v])                                   # 51 -> 102 kenh (17 khop x 3, 2 lan)
            elif k.endswith(".gcn.A"):
                continue                                                # dung do thi moi
            elif k.endswith(".gcn.PA"):
                pa = torch.ones(3, 34, 34)
                pa[:, :17, :17] = v; pa[:, 17:, 17:] = v                 # khoi trong than: giu, canh giua 2 nguoi: 1
                v = pa
            new[k] = v
        sd = new
    miss, unexp = model.load_state_dict(sd, strict=False)
    assert all(m.endswith(".gcn.A") for m in miss) and not unexp, (miss, unexp)
    model.cls_head.fc = nn.Linear(256, 2)                    # 60 lop NTU -> 2 lop
    return model


def prepare(cache, out):
    """Doc keypoints.csv -> (M=2, T, V, 3) da PreNormalize2D, luu 1 file npz (noi theo truc T)."""
    f = out / "data.npz"
    if f.exists():
        z = np.load(f)
        return z["x"], z["off"], z["y"], z["names"]
    import cv2
    xs, ys, names = [], [], []
    for sub, lab, folder in (("pos", 1, "Violence"), ("neg", 0, "NonViolence")):
        for d in sorted((cache / sub).iterdir()):
            kp_csv = d / "keypoints.csv"
            vid = next((Path("datasets") / folder).glob(d.name + ".*"), None)
            if not kp_csv.exists() or vid is None or sum(1 for _ in open(kp_csv)) <= 1:
                continue
            kp, sc, _, _ = D.load_skeleton(kp_csv, 2, "score")
            cap = cv2.VideoCapture(str(vid))
            shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            cap.release()
            x = D.gen_ske_feat_joint(D.pre_normalize_2d(kp, shape), sc)       # (M, T, 17, 3)
            if x.shape[0] < 2:
                x = np.concatenate([x, np.zeros((2 - x.shape[0],) + x.shape[1:], np.float32)], 0)
            xs.append(x.astype(np.float32)), ys.append(lab), names.append(d.name)
    off = np.cumsum([0] + [x.shape[1] for x in xs])
    np.savez(f, x=np.concatenate(xs, 1), off=off, y=np.array(ys), names=np.array(names))
    return prepare(cache, out)


def train_clip(x, rng, clip_len=100):
    """UniformSampleFrames che do train: chia 100 doan, moi doan lay 1 frame ngau nhien; + lat ngang 50%."""
    T = x.shape[1]
    if T < clip_len:
        inds = (rng.integers(T) + np.arange(clip_len)) % T
    else:
        bids = np.array([j * T // clip_len for j in range(clip_len + 1)])
        inds = bids[:-1] + rng.integers(np.maximum(np.diff(bids), 1))
    c = x[:, inds].copy()
    if rng.random() < 0.5:
        c = c[:, :, FLIP]
        vis = c[..., 2] > 0
        c[..., 0][vis] *= -1
    if rng.random() < 0.5:                                  # doi thu tu 2 nguoi (do thi 2p khong doi xung)
        c = c[::-1].copy()
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="runs/eval_cache/yolo26n-pose_my_botsort")
    ap.add_argument("--ckpt", default=str(D.ROOT / "debug" / "ckpt" / "stgcn_ntu60_2d_joint.pth"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graph", default="1p", choices=["1p", "2p"], help="1p: ST-GCN goc (2 nguoi rieng), 2p: do thi 2 nguoi 34 dinh")
    ap.add_argument("--budget", type=float, default=1e9, help="giay; het thi luu checkpoint va thoat")
    args = ap.parse_args()
    t0 = time.time()

    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    torch.manual_seed(args.seed)

    cache = Path(args.cache)
    base = Path("runs/stgcn_ft") / cache.name
    base.mkdir(parents=True, exist_ok=True)
    X, off, y, names = prepare(cache, base)                 # du lieu dung chung cho 1p va 2p
    out = base if args.graph == "1p" else base.parent / (cache.name + "_2p")
    out.mkdir(parents=True, exist_ok=True)
    conv = (lambda c: c) if args.graph == "1p" else to_2p
    vids = [X[:, off[i]:off[i + 1]] for i in range(len(y))]
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=args.test_size, stratify=y, random_state=args.seed)
    print(f"{len(y)} video | train {len(tr)} | test {len(te)}  (tach theo ten video, seed {args.seed})", flush=True)

    model = build(torch, nn, args.ckpt, args.graph)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    steps_per_ep = math.ceil(len(tr) / args.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / (args.epochs * steps_per_ep))))

    ck = out / "ckpt.pt"
    ep, log = 0, []
    if ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]), opt.load_state_dict(st["opt"]), sched.load_state_dict(st["sched"])
        ep, log = st["epoch"], st["log"]
        print(f"tiep tuc tu epoch {ep}", flush=True)

    lossf = nn.CrossEntropyLoss()
    last = 0.0
    while ep < args.epochs:
        if time.time() - t0 + last * 1.1 > args.budget:
            print(f"het budget sau epoch {ep}/{args.epochs} -> chay lai lenh de tiep tuc")
            return
        te0 = time.time()
        model.train()
        rng = np.random.default_rng(args.seed * 1000 + ep)
        perm = rng.permutation(tr)
        tot, correct, n = 0.0, 0, 0
        for b in range(0, len(perm), args.bs):
            bi = perm[b:b + args.bs]
            xb = torch.from_numpy(conv(np.stack([train_clip(vids[i], rng) for i in bi])))   # (B, M, 100, 17, 3)
            yb = torch.from_numpy(y[bi]).long()
            logit = model.cls_head(model.backbone(xb))
            loss = lossf(logit, yb)
            opt.zero_grad(), loss.backward(), opt.step(), sched.step()
            tot += loss.item() * len(bi); correct += (logit.argmax(1) == yb).sum().item(); n += len(bi)
        ep += 1
        last = time.time() - te0
        log.append((ep, round(tot / n, 4), round(correct / n, 4), round(last, 1)))
        print(f"epoch {ep:>2}/{args.epochs}  loss {tot / n:.4f}  train acc {correct / n:.3f}  ({last:.0f}s)", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep, "log": log}, ck)

    # ---------- danh gia tren tap test: 10 clip / video, trung binh softmax ----------
    pf = out / "test_prob.npy"
    if pf.exists():
        prob = np.load(pf)
    else:
        model.eval()
        prob = []
        with torch.no_grad():
            for k, i in enumerate(te):
                x = vids[i]
                c = conv(D.format_gcn_input(x[:, D.uniform_sample_test(x.shape[1], 100, 10)], 2, 10))
                prob.append(model.cls_head(model.backbone(torch.from_numpy(c))).softmax(-1).mean(0)[1].item())
                if time.time() - t0 > args.budget:
                    print(f"het budget khi test ({k + 1}/{len(te)}) -> chay lai lenh")
                    return
        prob = np.array(prob)
        np.save(pf, prob)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    yt = y[te]
    test_names = set(names[te])

    def row(name, pred, score=None):
        auc = f"{roc_auc_score(yt_, score):.3f}" if score is not None else "  -  "
        return (f"{name:<30}{precision_score(yt_, pred, zero_division=0):>8.2f}{recall_score(yt_, pred):>8.2f}"
                f"{f1_score(yt_, pred):>8.2f}{(pred == yt_).mean():>8.2f}{auc:>8}")

    lines = [f"do thi {args.graph}. {len(y)} video, train {len(tr)}, test {len(te)} (ti le {args.test_size}, seed {args.seed}), {args.epochs} epoch, lr {args.lr}, bs {args.bs}",
             "", f"{'phuong phap (cung tap test)':<30}{'P':>8}{'R':>8}{'F1':>8}{'Acc':>8}{'AUC':>8}"]
    yt_ = yt
    lines.append(row(f"ST-GCN fine-tune ({args.graph})", (prob >= 0.5).astype(int), prob))
    other = (base.parent / (cache.name + "_2p") if args.graph == "1p" else base) / "test_prob.npy"
    if other.exists():
        po = np.load(other)
        lines.append(row(f"ST-GCN fine-tune ({'2p' if args.graph == '1p' else '1p'})", (po >= 0.5).astype(int), po))

    # RF + luat: dung features.csv cua train_rf.py
    fr = Path("runs/rf") / cache.name / "features.csv"
    if fr.exists():
        with open(fr) as fh:
            rd = csv.reader(fh); head = next(rd); R = {r[0]: r for r in rd}
        Fn = head[2:]
        trn = [n for n in names[tr] if n in R]; ten = [n for n in names[te] if n in R]
        Xtr = np.array([[float(v) for v in R[n][2:]] for n in trn]); ytr = np.array([int(R[n][1]) for n in trn])
        Xte = np.array([[float(v) for v in R[n][2:]] for n in ten]); yt_ = np.array([int(R[n][1]) for n in ten])
        lines.append(row("luat base", (Xte[:, Fn.index("events_base")] >= 1).astype(int)))
        lines.append(row("luat impact0.02", (Xte[:, Fn.index("events_impact")] >= 1).astype(int)))
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=0).fit(Xtr, ytr)
        lines.append(row("Random Forest", rf.predict(Xte), rf.predict_proba(Xte)[:, 1]))
        if len(ten) != len(te):
            lines.append(f"  (luat/RF: {len(ten)}/{len(te)} video test co trong features.csv)")
    # probe256: dac trung backbone pretrained (eval_stgcn.py)
    fo = Path("runs/stgcn") / cache.name / "outputs_img.npz"
    if fo.exists():
        z = np.load(fo); pos = {n: i for i, n in enumerate(z["names"])}
        a = [pos[n] for n in names[tr] if n in pos]; b = [pos[n] for n in names[te] if n in pos]
        yt_ = z["y"][b]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")).fit(z["emb"][a], z["y"][a])
        lines.append(row("ST-GCN probe256 (dong bang)", clf.predict(z["emb"][b]), clf.predict_proba(z["emb"][b])[:, 1]))

    text = "\n".join(lines)
    print("\n" + text)
    (out / "summary.txt").write_text(text + "\n")
    with open(out / "train_log.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["epoch", "loss", "train_acc", "sec"]); w.writerows(log)
    with open(out / "test_pred.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["video", "label", "p_fight"])
        w.writerows([[names[i], int(y[i]), round(float(p), 4)] for i, p in zip(te, prob)])
    print(f"\nDa luu {out}/")


if __name__ == "__main__":
    main()
