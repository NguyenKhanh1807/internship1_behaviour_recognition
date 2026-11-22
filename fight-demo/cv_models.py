"""Buoi 7: danh gia 5-fold + tap validation cho ProtoGCN (bone, joint, ...) va ST-GCN 1p.

Moi fold: 1/5 video lam TEST; phan con lai tach tiep 15% lam VALIDATION, 85% de TRAIN.
  - moi mo hinh fine-tune tren TRAIN (cung so epoch / lr nhu truoc), du doan VAL va TEST
  - nguong va trong so ensemble chi hoc tren VAL -> ap len TEST (khong nhin tap test)
Gop du doan TEST cua 5 fold = du doan "out-of-fold" cho CA 1049 video -> McNemar manh hon nhieu so voi 210 video.

  python cv_models.py                                  # mac dinh: protogcn_b, protogcn_j, stgcn_1p
  python cv_models.py --models protogcn_b,stgcn_1p     # chi chay mot so mo hinh
  python cv_models.py --report                         # chi in bang tu ket qua da co
Chay lai cung lenh = chay tiep (moi epoch luu checkpoint, fold xong thi bo qua).
Ket qua: runs/cv/<cache>/summary.txt, oof_pred.csv
"""
import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np

import finetune_stgcn as F
import protogcn_model as P

D = F.D
CK = D.ROOT / "debug" / "ckpt"
MODELS = {   # ten: (loai, checkpoint, luong)
    "protogcn_b": ("proto", CK / "protogcn_finegym_b.pth", "b"),
    "protogcn_j": ("proto", CK / "protogcn_finegym_j.pth", "j"),
    "protogcn_jm": ("proto", CK / "protogcn_finegym_jm.pth", "jm"),
    "protogcn_bm": ("proto", CK / "protogcn_finegym_bm.pth", "bm"),
    "stgcn_1p": ("stgcn", CK / "stgcn_ntu60_2d_joint.pth", "j"),
}


def splits(y, k, seed, val_size=0.15):
    from sklearn.model_selection import StratifiedKFold, train_test_split
    out = []
    for f, (trv, te) in enumerate(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
        tr, va = train_test_split(trv, test_size=val_size, stratify=y[trv], random_state=seed + f)
        out.append((tr, va, te))
    return out


def make(kind, ckpt, mod, torch, nn):
    """Tra ve (model, ham bien doi clip, ham forward -> logit, ham loss)."""
    if kind == "proto":
        n_proto = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]["backbone.prn.query_matrix.weight"].shape[0]
        m = P.Recognizer(2, n_proto)
        P.load_pretrained(m, ckpt)
        conv = lambda c: P.gen_feat(P.kinetics_transform(c), mod)
        fwd = lambda x: m(x)
        loss = lambda out, yb: m.cls_head.loss(out[0], out[1], yb)
        logit = lambda out: out[0]
    else:
        m = F.build(torch, nn, ckpt, "1p")
        conv = lambda c: c
        fwd = lambda x: m.cls_head(m.backbone(x))
        ce = nn.CrossEntropyLoss()
        loss = lambda out, yb: ce(out, yb)
        logit = lambda out: out
    return m, conv, fwd, loss, logit


def predict(vids, idx, conv, fwd, logit, torch, dev):
    out = []
    with torch.no_grad():
        for i in idx:
            x = vids[i]
            c = conv(D.format_gcn_input(x[:, D.uniform_sample_test(x.shape[1], 100, 10)], 2, 10))
            out.append(logit(fwd(torch.from_numpy(c).to(dev))).softmax(-1).mean(0)[1].item())
    return np.array(out)


def run_fold(name, fold, tr, va, te, vids, y, args, torch, nn, dev, t0):
    d = Path(args.out) / name / f"fold{fold}"
    d.mkdir(parents=True, exist_ok=True)
    if (d / "test_prob.npy").exists():
        return True
    kind, ckpt, mod = MODELS[name]
    torch.manual_seed(args.seed + fold)
    model, conv, fwd, lossf, logit = make(kind, ckpt, mod, torch, nn)
    model.to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    spe = math.ceil(len(tr) / args.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / (args.epochs * spe))))
    ep, log = 0, []
    ck = d / "ckpt.pt"
    if ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]), opt.load_state_dict(st["opt"]), sched.load_state_dict(st["sched"])
        ep, log = st["epoch"], st["log"]
    last = 0.0
    while ep < args.epochs:
        if time.time() - t0 + last * 1.1 > args.budget:
            print("  het budget -> chay lai lenh de tiep tuc", flush=True)
            return False
        e0 = time.time()
        model.train()
        rng = np.random.default_rng(args.seed * 1000 + fold * 100 + ep)
        perm = rng.permutation(tr)
        tot, cor, n = 0.0, 0, 0
        for b in range(0, len(perm), args.bs):
            bi = perm[b:b + args.bs]
            if len(bi) < 2:
                continue
            xb = torch.from_numpy(conv(np.stack([F.train_clip(vids[i], rng) for i in bi]))).to(dev)
            yb = torch.from_numpy(y[bi]).long().to(dev)
            out = fwd(xb)
            loss = lossf(out, yb)
            opt.zero_grad(), loss.backward(), opt.step(), sched.step()
            tot += loss.item() * len(bi); cor += (logit(out).argmax(1) == yb).sum().item(); n += len(bi)
        ep += 1
        last = time.time() - e0
        log.append((ep, round(tot / n, 4), round(cor / n, 4), round(last, 1)))
        print(f"  {name} fold {fold} epoch {ep:>2}/{args.epochs} loss {tot / n:.4f} train acc {cor / n:.3f} ({last:.0f}s)", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep, "log": log}, ck)
    model.eval()
    np.save(d / "val_prob.npy", predict(vids, va, conv, fwd, logit, torch, dev))
    np.save(d / "test_prob.npy", predict(vids, te, conv, fwd, logit, torch, dev))
    np.savez(d / "idx.npz", tr=tr, va=va, te=te)
    ck.unlink()                                             # bo checkpoint trung gian (~40 MB/fold)
    return True


def best_threshold(yv, pv):
    from sklearn.metrics import f1_score
    ts = np.arange(0.05, 0.96, 0.01)
    return float(ts[np.argmax([f1_score(yv, (pv >= t).astype(int)) for t in ts])])


def report(names, y, sp, args):
    from scipy.stats import binomtest
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    base = Path(args.out)
    have = [n for n in names if all((base / n / f"fold{f}" / "test_prob.npy").exists() for f in range(args.folds))]
    if not have:
        print("chua co mo hinh nao chay xong du 5 fold"); return
    V = {n: [np.load(base / n / f"fold{f}" / "val_prob.npy") for f in range(args.folds)] for n in have}
    T = {n: [np.load(base / n / f"fold{f}" / "test_prob.npy") for f in range(args.folds)] for n in have}

    # cac "phuong phap": tung mo hinh (nguong 0.5 / nguong hoc tren val) + ensemble trung binh + stacking hoc tren val
    meth = {}
    for n in have:
        meth[f"{n} (nguong 0.5)"] = [(T[n][f], 0.5) for f in range(args.folds)]
        meth[f"{n} (nguong theo val)"] = [(T[n][f], best_threshold(y[sp[f][1]], V[n][f])) for f in range(args.folds)]
    if len(have) > 1:
        tag = "+".join(have)
        meth[f"TB {tag} (nguong theo val)"] = []
        meth[f"stacking {tag} (LR tren val)"] = []
        for f in range(args.folds):
            yv = y[sp[f][1]]
            pv = np.mean([V[n][f] for n in have], 0); pt = np.mean([T[n][f] for n in have], 0)
            meth[f"TB {tag} (nguong theo val)"].append((pt, best_threshold(yv, pv)))
            Xv = np.stack([V[n][f] for n in have], 1); Xt = np.stack([T[n][f] for n in have], 1)
            lr = LogisticRegression(C=1.0).fit(Xv, yv)
            meth[f"stacking {tag} (LR tren val)"].append((lr.predict_proba(Xt)[:, 1], 0.5))

    lines = [f"{args.folds}-fold, seed {args.seed}, {len(y)} video; moi fold: train ~{len(sp[0][0])}, val ~{len(sp[0][1])}, test ~{len(sp[0][2])}; "
             f"{args.epochs} epoch, lr {args.lr}", "",
             f"{'phuong phap':<52}{'F1 tb+-std':>14}{'AUC tb+-std':>16}{'Acc tb':>8} | {'F1 OOF':>7}{'AUC OOF':>8}{'FP/FN':>9}"]
    oof = {}
    for k, lst in meth.items():
        f1s, aucs, accs = [], [], []
        P_ = np.zeros(len(y)); Q = np.zeros(len(y), int)
        for f, (p, t) in enumerate(lst):
            te = sp[f][2]; yt = y[te]; pr = (p >= t).astype(int)
            f1s.append(f1_score(yt, pr)); aucs.append(roc_auc_score(yt, p)); accs.append((pr == yt).mean())
            P_[te] = p; Q[te] = pr
        oof[k] = (P_, Q)
        fp = int(((Q == 1) & (y == 0)).sum()); fn = int(((Q == 0) & (y == 1)).sum())
        lines.append(f"{k:<52}{np.mean(f1s):>8.3f}+-{np.std(f1s):.3f}{np.mean(aucs):>10.3f}+-{np.std(aucs):.3f}{np.mean(accs):>8.3f} | "
                     f"{f1_score(y, Q):>7.3f}{roc_auc_score(y, P_):>8.3f}{fp:>5}/{fn:<4}")

    ref = next((k for k in oof if k.startswith("stgcn_1p (nguong 0.5")), None)
    if ref:
        lines += ["", f"McNemar tren ca {len(y)} video (du doan out-of-fold), so voi {ref}:"]
        rb = oof[ref][1] == y
        for k, (_, Q) in oof.items():
            if k == ref:
                continue
            ra = Q == y
            n01, n10 = int((ra & ~rb).sum()), int((~ra & rb).sum())
            pv = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            lines.append(f"  {k:<52} dung/ST-GCN sai {n01:>3}, nguoc lai {n10:>3}, p = {pv:.4f}")
    text = "\n".join(lines)
    print("\n" + text)
    (base / "summary.txt").write_text(text + "\n")
    with open(base / "oof_pred.csv", "w", newline="") as fh:
        w = csv.writer(fh); ks = list(oof)
        w.writerow(["video", "label", "fold"] + [f"p:{k}" for k in ks])
        fold_of = np.zeros(len(y), int)
        for f in range(args.folds):
            fold_of[sp[f][2]] = f
        for i in range(len(y)):
            w.writerow([args.names[i], int(y[i]), fold_of[i]] + [round(float(oof[k][0][i]), 4) for k in ks])
    print(f"\nDa luu {base}/summary.txt, oof_pred.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="runs/eval_cache/yolo26n-pose_my_botsort")
    ap.add_argument("--models", default="protogcn_b,protogcn_j,stgcn_1p")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--budget", type=float, default=1e9)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    names = [m.strip() for m in args.models.split(",") if m.strip()]
    for n in names:
        assert n in MODELS, f"{n} khong co trong {list(MODELS)}"
        assert MODELS[n][1].exists(), f"thieu checkpoint {MODELS[n][1]}"

    cache = Path(args.cache)
    X, off, y, vnames = F.prepare(cache, Path("runs/stgcn_ft") / cache.name)
    args.names = vnames
    args.out = str(Path("runs/cv") / cache.name)
    sp = splits(y, args.folds, args.seed)
    if not args.report:
        import torch
        import torch.nn as nn
        dev = args.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        vids = [X[:, off[i]:off[i + 1]] for i in range(len(y))]
        print(f"{len(y)} video, {args.folds} fold, mo hinh {names}, device {dev}", flush=True)
        for n in names:
            for f, (tr, va, te) in enumerate(sp):
                if not run_fold(n, f, tr, va, te, vids, y, args, torch, nn, dev, t0):
                    return
                print(f"xong {n} fold {f} ({(time.time() - t0) / 60:.0f} phut)", flush=True)
    report(names, y, sp, args)


if __name__ == "__main__":
    main()
