"""Buoi 6: tai hien ProtoGCN (CVPR 2025) va fine-tune cho bai toan danh nhau / binh thuong (RLVS).

Dung CHUNG du lieu + CHUNG cach tach train/test voi finetune_stgcn.py (data.npz, seed 0, 839/210),
nen so sanh truc tiep duoc voi ST-GCN 1p/2p, Random Forest, luat va probe256 tren cung 210 video test.

  python finetune_protogcn.py --init ../debug/ckpt/protogcn_finegym_j.pth   # fine-tune tu checkpoint FineGYM (COCO 2D)
  python finetune_protogcn.py --init ../debug/ckpt/protogcn_finegym_b.pth   # luong bone (doan tu ten file)
  python finetune_protogcn.py --init ../debug/ckpt/protogcn_finegym_jm.pth  # luong joint-motion
  python ensemble_streams.py                                               # gop cac luong
  python finetune_protogcn.py --init scratch --epochs 30                   # train tu dau (ablation)
  python finetune_protogcn.py ... --budget 160                             # dung som, chay lai de tiep
Ket qua: runs/protogcn_ft/<cache>_<init>/summary.txt, train_log.csv, test_pred.csv, test_prob.npy
"""

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np

import finetune_stgcn as F                       # prepare(), train_clip(), D (debug_stgcn)
import protogcn_model as P

D = F.D


MOD = "j"


def to20(c):
    """(..., 17, 3) -> (..., 20, 3) theo Kinetics_Transform, roi GenSkeFeat theo luong MOD (j/b/jm/bm)."""
    return P.gen_feat(P.kinetics_transform(c), MOD)


def bootstrap(yt, pred, score, n=1000, seed=0):
    """Khoang tin cay 95% cho F1 va AUC bang bootstrap theo video."""
    from sklearn.metrics import f1_score, roc_auc_score
    rng = np.random.default_rng(seed)
    f1s, aucs = [], []
    for _ in range(n):
        i = rng.integers(0, len(yt), len(yt))
        if yt[i].min() == yt[i].max():
            continue
        f1s.append(f1_score(yt[i], pred[i]))
        if score is not None:
            aucs.append(roc_auc_score(yt[i], score[i]))
    ci = lambda a: f"[{np.percentile(a, 2.5):.2f}-{np.percentile(a, 97.5):.2f}]" if a else ""
    return ci(f1s), ci(aucs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="runs/eval_cache/yolo26n-pose_my_botsort")
    ap.add_argument("--init", default="scratch", help="'scratch' hoac duong dan checkpoint ProtoGCN (.pth)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-prototype", type=int, default=100, help="chi dung khi --init scratch (checkpoint thi tu doc)")
    ap.add_argument("--modality", default=None, choices=["j", "b", "jm", "bm"],
                    help="luong dau vao; bo trong = doan tu ten checkpoint (..._b.pth -> b), mac dinh j")
    ap.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    ap.add_argument("--budget", type=float, default=1e9)
    ap.add_argument("--csc-weight", type=float, default=0.2,
                    help="trong so loss CSC (lambda). 0.2 = code goc, 0.3 = bai bao, 0 = tat CSC (chi con CE; PRN khong duoc hoc)")
    args = ap.parse_args()
    t0 = time.time()
    global MOD
    if args.modality is None:
        stem = Path(args.init).stem if args.init != "scratch" else ""
        args.modality = next((m for m in ("bm", "jm", "b", "j") if stem.endswith("_" + m)), "j")
    MOD = args.modality

    import torch
    from sklearn.model_selection import train_test_split
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    cache = Path(args.cache)
    X, off, y, names = F.prepare(cache, Path("runs/stgcn_ft") / cache.name)      # cung data.npz voi ST-GCN
    tag = (f"scratch_p{args.num_prototype}" + ("" if MOD == "j" else f"_{MOD}")) if args.init == "scratch" else Path(args.init).stem
    if args.init != "scratch" and not tag.endswith("_" + MOD):
        print(f"CANH BAO: checkpoint {tag} co the khong phai luong '{MOD}'", flush=True)
    if args.csc_weight != 0.2:                               # ablation: ket qua luu rieng, khong de len lan chay goc
        tag += f"_csc{args.csc_weight:g}"
    out = Path("runs/protogcn_ft") / f"{cache.name}_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    vids = [X[:, off[i]:off[i + 1]] for i in range(len(y))]
    tr, te = train_test_split(np.arange(len(y)), test_size=args.test_size, stratify=y, random_state=args.seed)
    print(f"{len(y)} video | train {len(tr)} | test {len(te)} | device {dev} | init {tag} | luong {MOD}", flush=True)

    n_proto = args.num_prototype                             # mac dinh 100 = giong checkpoint de so sanh cong bang
    if args.init != "scratch":                               # checkpoint FineGYM j_1 tren Drive dung 100 prototype
        sd0 = torch.load(args.init, map_location="cpu", weights_only=False)
        n_proto = sd0.get("state_dict", sd0)["backbone.prn.query_matrix.weight"].shape[0]
        del sd0
    model = P.Recognizer(num_classes=2, num_prototype=n_proto)
    model.cls_head.weight = args.csc_weight                  # L = CE + weight * CSC
    print(f"so prototype: {n_proto} | trong so CSC: {args.csc_weight:g}", flush=True)
    if args.init != "scratch":
        n_ok, skip, miss = P.load_pretrained(model, args.init)
        print(f"nap {n_ok} tensor tu {args.init}; bo qua {len(skip)} ({[s for s in skip if 'fc_cls' not in s][:5]}...); "
              f"khoi tao moi {len(miss)}: {miss}", flush=True)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"so tham so: {n_param / 1e6:.2f}M", flush=True)
    model.to(dev)

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
            if len(bi) < 2:                                   # BatchNorm can >= 2 mau
                continue
            xb = torch.from_numpy(to20(np.stack([F.train_clip(vids[i], rng) for i in bi]))).to(dev)  # (B,2,100,20,3)
            yb = torch.from_numpy(y[bi]).long().to(dev)
            logit, graph = model(xb)
            loss = model.cls_head.loss(logit, graph, yb)      # CE + csc_weight * CSC
            opt.zero_grad(), loss.backward(), opt.step(), sched.step()
            tot += loss.item() * len(bi); correct += (logit.argmax(1) == yb).sum().item(); n += len(bi)
        ep += 1
        last = time.time() - te0
        log.append((ep, round(tot / n, 4), round(correct / n, 4), round(last, 1)))
        print(f"epoch {ep:>2}/{args.epochs}  loss {tot / n:.4f}  train acc {correct / n:.3f}  ({last:.0f}s)", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep, "log": log}, ck)

    # ---------- test: 10 clip / video, trung binh softmax (giong test_pipeline num_clips=10) ----------
    pf = out / "test_prob.npy"
    if pf.exists():
        prob = np.load(pf)
        ms = float((out / "ms_per_video.txt").read_text()) if (out / "ms_per_video.txt").exists() else float("nan")
    else:
        model.eval()
        prob, tt = [], 0.0
        with torch.no_grad():
            for i in te:
                x = vids[i]
                c = to20(D.format_gcn_input(x[:, D.uniform_sample_test(x.shape[1], 100, 10)], 2, 10))
                s = time.time()
                logit, _ = model(torch.from_numpy(c).to(dev))
                prob.append(logit.softmax(-1).mean(0)[1].item())
                tt += time.time() - s
        prob = np.array(prob)
        ms = 1000 * tt / len(te)
        np.save(pf, prob)
        (out / "ms_per_video.txt").write_text(f"{ms:.1f}")

    # ---------- bang so sanh tren CUNG tap test ----------
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def row(name, yt, pred, score=None):
        auc = f"{roc_auc_score(yt, score):.3f}" if score is not None else "  -  "
        cf1, cauc = bootstrap(yt, pred, score)
        tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
        return (f"{name:<32}{precision_score(yt, pred, zero_division=0):>6.2f}{recall_score(yt, pred):>6.2f}"
                f"{f1_score(yt, pred):>6.2f} {cf1:<12}{(pred == yt).mean():>6.2f}{auc:>7} {cauc:<12} {fp:>3}/{fn:<3}")

    yt = y[te]
    lines = [f"ProtoGCN init={tag}. {len(y)} video, train {len(tr)}, test {len(te)} (seed {args.seed}), "
             f"{args.epochs} epoch, lr {args.lr}, bs {args.bs}, {n_param / 1e6:.2f}M tham so, "
             f"{np.mean([l[3] for l in log]) if log else float('nan'):.0f}s/epoch, {ms:.0f} ms/video test (10 clip, {dev})",
             "", f"{'phuong phap (cung 210 video test)':<32}{'P':>6}{'R':>6}{'F1':>6} {'CI95 F1':<12}{'Acc':>6}{'AUC':>7} {'CI95 AUC':<12} FP/FN"]
    lines.append(row(f"ProtoGCN ({tag})", yt, (prob >= 0.5).astype(int), prob))
    for d in sorted(Path("runs/protogcn_ft").glob(cache.name + "_*")):     # cac lan ProtoGCN khac
        if d != out and (d / "test_prob.npy").exists():
            po = np.load(d / "test_prob.npy")
            lines.append(row(f"ProtoGCN ({d.name[len(cache.name) + 1:]})", yt, (po >= 0.5).astype(int), po))
    for g, d in (("1p", Path("runs/stgcn_ft") / cache.name), ("2p", Path("runs/stgcn_ft") / (cache.name + "_2p"))):
        if (d / "test_prob.npy").exists():
            po = np.load(d / "test_prob.npy")
            lines.append(row(f"ST-GCN fine-tune ({g})", yt, (po >= 0.5).astype(int), po))

    fr = Path("runs/rf") / cache.name / "features.csv"
    if fr.exists():
        with open(fr) as fh:
            rd = csv.reader(fh); head = next(rd); R = {r[0]: r for r in rd}
        Fn = head[2:]
        trn = [n for n in names[tr] if n in R]; ten = [n for n in names[te] if n in R]
        Xtr = np.array([[float(v) for v in R[n][2:]] for n in trn]); ytr = np.array([int(R[n][1]) for n in trn])
        Xte = np.array([[float(v) for v in R[n][2:]] for n in ten]); yr = np.array([int(R[n][1]) for n in ten])
        lines.append(row("luat base", yr, (Xte[:, Fn.index("events_base")] >= 1).astype(int)))
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=0).fit(Xtr, ytr)
        lines.append(row("Random Forest", yr, rf.predict(Xte), rf.predict_proba(Xte)[:, 1]))
    fo = Path("runs/stgcn") / cache.name / "outputs_img.npz"
    if fo.exists():
        z = np.load(fo); pos = {n: i for i, n in enumerate(z["names"])}
        a = [pos[n] for n in names[tr] if n in pos]; b = [pos[n] for n in names[te] if n in pos]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")).fit(z["emb"][a], z["y"][a])
        lines.append(row("ST-GCN probe256 (dong bang)", z["y"][b], clf.predict(z["emb"][b]), clf.predict_proba(z["emb"][b])[:, 1]))

    # McNemar: ProtoGCN vs ST-GCN 1p (cung video) - khac biet co y nghia thong ke khong?
    d1 = Path("runs/stgcn_ft") / cache.name / "test_prob.npy"
    if d1.exists():
        from scipy.stats import binomtest
        a = (prob >= 0.5).astype(int) == yt; b = (np.load(d1) >= 0.5).astype(int) == yt
        n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
        p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        lines += ["", f"McNemar ProtoGCN({tag}) vs ST-GCN 1p: ProtoGCN dung/ST-GCN sai = {n01}, nguoc lai = {n10}, p = {p:.3f}"]

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
