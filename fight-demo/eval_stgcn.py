"""Buoi 3-4: danh gia ST-GCN (pretrained NTU-60, 2D joint) tren cac video RLVS da co skeleton trong cache.

Dung lai pipeline + model trong debug/debug_stgcn.py (chep sat MMAction2, nap checkpoint chinh thuc).
3 cach dung model:
  zero-shot : diem danh nhau = P(A50 dam) + P(A51 da) + P(A52 xo day), nguong chon bang CV
  probe60   : Logistic Regression tren vector 60 xac suat NTU (hoc lai lop cuoi, 5-fold CV)
  probe256  : Logistic Regression tren dac trung 256 chieu cua backbone (truoc lop fc)
2 cach chuan hoa toa do:
  img    : PreNormalize2D goc (chia theo kich thuoc anh) - dung y MMAction2
  person : dua tam 2 nguoi ve (0, 0) va scale chieu cao nguoi ~1.2 (giong ti le nguoi trong NTU)

Chay (trong fight-demo):  python eval_stgcn.py      ->  runs/stgcn/<cache>/summary.txt, per_video.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "debug"))
import debug_stgcn as D  # noqa: E402


def person_normalize(kp, sc, target_h=1.2):
    """Tam = trung binh moi keypoint nhin thay; chieu cao nguoi (median) -> target_h."""
    vis = sc > 0.3
    if vis.sum() < 4:
        return np.zeros_like(kp)
    pts = kp[vis]
    center = pts.mean(0)
    heights = []
    for m in range(kp.shape[0]):
        for t in range(kp.shape[1]):
            v = vis[m, t]
            if v.sum() >= 4:
                heights.append(np.ptp(kp[m, t, v, 1]))
    h = np.median(heights) if heights else 1.0
    out = (kp - center) * (target_h / max(h, 1e-3))
    out[sc <= 0] = 0
    return out


def run_video(model, torch, kp_csv, video, norm, num_clips=10, select="score"):
    kp, sc, _, _ = D.load_skeleton(kp_csv, 2, select)
    if norm == "img":
        import cv2
        cap = cv2.VideoCapture(str(video))
        shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        cap.release()
        kpn = D.pre_normalize_2d(kp, shape)
    else:
        kpn = person_normalize(kp, sc)
    x = D.gen_ske_feat_joint(kpn, sc)
    inds = D.uniform_sample_test(x.shape[1], 100, num_clips)
    x = D.format_gcn_input(x[:, inds], 2, num_clips)
    with torch.no_grad():
        feat = model.backbone(torch.from_numpy(x))                      # (nc, M, 256, 25, 17)
        N, M, C, T, V = feat.shape
        emb = model.cls_head.pool(feat.view(N * M, C, T, V)).view(N, M, C).mean(1)   # (nc, 256)
        prob = model.cls_head.fc(emb).softmax(-1).mean(0).numpy()      # (60,)
    return prob, emb.mean(0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="runs/eval_cache/yolo26n-pose_my_botsort")
    ap.add_argument("--ckpt", default=str(D.ROOT / "debug" / "ckpt" / "stgcn_ntu60_2d_joint.pth"))
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--select", default="score", choices=["score", "pair"], help="chon 2 nguoi: score = 2 track ro nhat (mac dinh), pair = cap gan nhau lau nhat")
    ap.add_argument("--norms", default="img,person")
    args = ap.parse_args()

    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = D.build_model()(torch.from_numpy(D.stgcn_spatial())).eval()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    miss, unexp = model.load_state_dict(sd, strict=False)
    assert not miss and not unexp, (miss, unexp)

    cache = Path(args.cache)
    items = []
    for sub, lab, folder in (("pos", 1, "Violence"), ("neg", 0, "NonViolence")):
        for d in sorted((cache / sub).iterdir()):
            kp = d / "keypoints.csv"
            vid = next((Path("datasets") / folder).glob(d.name + ".*"), None)
            if kp.exists() and vid is not None and sum(1 for _ in open(kp)) > 1:
                items.append((d.name, lab, kp, vid))
    y = np.array([it[1] for it in items])
    print(f"{len(items)} video ({y.sum()} danh nhau, {len(y) - y.sum()} binh thuong)")

    out = Path("runs/stgcn") / (cache.name + ("" if args.select == "score" else "_" + args.select))
    out.mkdir(parents=True, exist_ok=True)
    lines, rows = [], {it[0]: {"video": it[0], "label": it[1]} for it in items}
    for norm in args.norms.split(","):
        P, E = [], []
        for i, (name, lab, kp, vid) in enumerate(items, 1):
            fc = out / f"feat_{norm}" / f"{lab}_{name}.npz"   # cache theo video de chay tiep duoc
            if fc.exists():
                z = np.load(fc); p, e = z["p"], z["e"]
            else:
                p, e = run_video(model, torch, kp, vid, norm, select=args.select)
                fc.parent.mkdir(parents=True, exist_ok=True)
                np.savez(fc, p=p, e=e)
            P.append(p), E.append(e)
            rows[name][f"{norm}_fight"] = round(float(p[D.FIGHT].sum()), 4)
            rows[name][f"{norm}_top1"] = D.NTU60[int(p.argmax())]
            if i % 20 == 0:
                print(f"  [{norm}] {i}/{len(items)}", flush=True)
        P, E = np.array(P), np.array(E)
        fight = P[:, D.FIGHT].sum(1)
        top1 = [D.NTU60[i] for i in P.argmax(1)]
        common = {c: top1.count(c) for c in set(top1)}
        lines.append(f"\n=== chuan hoa '{norm}' ===")
        lines.append("top-1 pho bien: " + ", ".join(f"{k} ({v})" for k, v in sorted(common.items(), key=lambda z: -z[1])[:5]))
        lines.append(f"diem danh nhau trung binh: danh nhau {fight[y == 1].mean():.3f} | binh thuong {fight[y == 0].mean():.3f}"
                     f" | ROC-AUC {roc_auc_score(y, fight):.3f}")

        def cv(fit_predict):
            s = []
            for r in range(args.repeats):
                pred = np.zeros_like(y)
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(y, y):
                    pred[te] = fit_predict(tr, te)
                s.append((precision_score(y, pred, zero_division=0), recall_score(y, pred), f1_score(y, pred)))
            s = np.array(s)
            return s.mean(0), s.std(0)

        def zero_shot(tr, te):  # chon nguong tot nhat (F1) tren tap train, ap len tap test
            cands = np.unique(fight[tr])
            best = max(cands, key=lambda t: f1_score(y[tr], (fight[tr] >= t).astype(int)))
            return (fight[te] >= best).astype(int)

        def probe(X):
            def f(tr, te):
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
                return clf.fit(X[tr], y[tr]).predict(X[te])
            return f

        lines.append(f"{'cach dung':<34}{'Precision':>14}{'Recall':>14}{'F1':>14}")
        for name, fn in (("zero-shot (A50+A51+A52, nguong CV)", zero_shot), ("probe60 (LogReg tren 60 xac suat)", probe(np.log(P + 1e-6))),
                         ("probe256 (LogReg tren dac trung)", probe(E))):
            m, s = cv(fn)
            lines.append(f"{name:<34}" + "".join(f"{a:>8.2f}+-{b:.2f}" for a, b in zip(m, s)))
        print("\n".join(lines[-6:]), flush=True)
        np.savez(out / f"outputs_{norm}.npz", prob=P, emb=E, y=y, names=np.array([it[0] for it in items]))

    text = f"chon nguoi: {args.select}. {len(items)} video RLVS, ST-GCN NTU-60 2D joint (pretrained), 5-fold CV lap {args.repeats} lan" + "\n".join([""] + lines)
    (out / "summary.txt").write_text(text + "\n")
    keys = ["video", "label"] + [f"{n}_{c}" for n in args.norms.split(",") for c in ("fight", "top1")]
    with open(out / "per_video.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows.values())
    print(f"\nDa luu {out}/")


if __name__ == "__main__":
    main()
