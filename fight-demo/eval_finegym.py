"""Kiem chung ban port ProtoGCN: chay checkpoint FineGYM CHINH THUC (giu nguyen fc 99 lop, khong fine-tune)
tren tap val cua FineGYM va so voi so trong log cua tac gia (debug/ckpt/protogcn_finegym_j1/*.log).

Neu ban port tinh dung nhu code goc thi do chinh xac phai xap xi so trong log:
  luong j, 1 clip  (val trong luc train, epoch 141): top-1 0.9308, mean class acc 0.9058
  luong j, 10 clip (test cuoi cung)                : top-1 0.9328, mean class acc 0.9092

Du lieu: gym_hrnet.pkl cua PYSKL (https://download.openmmlab.com/mmaction/pyskl/data/gym/gym_hrnet.pkl)
  python eval_finegym.py                       # luong j, 1 clip / video
  python eval_finegym.py --clips 10            # giong test_pipeline cua bai bao
  python eval_finegym.py --ckpt ../debug/ckpt/protogcn_finegym_b.pth
Pipeline giong config goc (j_1.py): UniformSampleFrames -> PoseDecode -> Kinetics_Transform -> GenSkeFeat -> FormatGCNInput
(toa do pixel, KHONG PreNormalize2D). Ket qua: runs/finegym_eval/<checkpoint>_clips<n>.txt
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np

import finetune_stgcn as F
import protogcn_model as P

D = F.D
# (top-1, mean class acc) trong log cua tac gia: 1 clip = val o epoch tot nhat, 10 clip = test cuoi cung
# log: debug/ckpt/protogcn_finegym_j1/*.log (j_1, epoch 141), debug/ckpt/protogcn_finegym_logs/{b_1,jm,bm}.log (epoch 137, 133, 144)
REF = {("j", 1): (0.9308, 0.9058), ("j", 10): (0.9328, 0.9092), ("b", 1): (0.9486, 0.9287), ("b", 10): (0.9484, 0.9304),
       ("jm", 1): (0.9389, 0.9163), ("jm", 10): (0.9407, 0.9178), ("bm", 1): (0.9309, 0.9083), ("bm", 10): (0.9331, 0.9099)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=str(D.ROOT / "debug" / "data" / "gym_hrnet.pkl"))
    ap.add_argument("--ckpt", default=str(D.ROOT / "debug" / "ckpt" / "protogcn_finegym_j.pth"))
    ap.add_argument("--modality", default=None, choices=["j", "b", "jm", "bm"])
    ap.add_argument("--clips", type=int, default=1)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="chi chay N video dau (thu nhanh)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    done = Path("runs/finegym_eval") / f"{Path(args.ckpt).stem}_clips{args.clips}.txt"
    if done.exists() and (Path("runs/finegym_eval") / "labels.npy").exists() and not args.limit:
        print(done.read_text()); return                                # da chay roi -> chi in lai
    import torch
    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    stem = Path(args.ckpt).stem
    mod = args.modality or next((m for m in ("bm", "jm", "b", "j") if stem.endswith("_" + m)), "j")

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    n_cls = sd["cls_head.fc_cls.weight"].shape[0]
    n_proto = sd["backbone.prn.query_matrix.weight"].shape[0]
    model = P.Recognizer(n_cls, n_proto)
    miss, unexp = model.load_state_dict(sd, strict=False)             # KHONG reset data_bn, KHONG bo fc_cls
    print(f"checkpoint {stem}: {len(sd)} tensor, {n_cls} lop, {n_proto} prototype | thieu {list(miss)} | thua {list(unexp)}", flush=True)
    assert not miss and not unexp, "state_dict khong khop hoan toan voi ban port"
    model.to(dev).eval()

    with open(args.pkl, "rb") as fh:
        data = pickle.load(fh)
    val = set(data["split"]["val"])
    anns = [a for a in data["annotations"] if a["frame_dir"] in val]
    if args.limit:
        anns = anns[:args.limit]
    print(f"{len(anns)} video val | luong {mod} | {args.clips} clip/video | device {dev}", flush=True)

    def clip(a):
        x = np.concatenate([a["keypoint"].astype(np.float32), a["keypoint_score"].astype(np.float32)[..., None]], -1)   # (M, T, 17, 3)
        x = x[:, D.uniform_sample_test(x.shape[1], 100, args.clips)]
        return P.gen_feat(P.kinetics_transform(D.format_gcn_input(x, 2, args.clips)), mod)                           # (nc, 2, 100, 20, 3)

    labels, probs, t0 = [], [], time.time()
    step = args.bs if args.clips == 1 else 1
    with torch.no_grad():
        for b in range(0, len(anns), step):
            batch = anns[b:b + step]
            x = np.concatenate([clip(a) for a in batch], 0)
            p = model(torch.from_numpy(x).to(dev))[0].softmax(-1).cpu().numpy()
            probs.append(p.reshape(len(batch), args.clips, -1).mean(1)); labels += [a["label"] for a in batch]
            if (b // step) % 20 == 0:
                print(f"  {b + len(batch)}/{len(anns)}  ({time.time() - t0:.0f}s)", flush=True)
    probs, labels = np.concatenate(probs), np.array(labels)
    pred = probs.argmax(1)
    top1 = float((pred == labels).mean())
    top5 = float((np.argsort(-probs, 1)[:, :5] == labels[:, None]).any(1).mean())
    mca = float(np.mean([(pred[labels == c] == c).mean() for c in np.unique(labels)]))
    ref = REF.get((mod, args.clips)) if not args.limit else None
    lines = [f"Kiem chung ban port ProtoGCN tren FineGYM val: checkpoint {stem}, luong {mod}, {args.clips} clip/video, {len(anns)} video, {dev}, {time.time() - t0:.0f}s",
             f"  ban port (protogcn_model.py): top-1 {top1:.4f} | top-5 {top5:.4f} | mean class acc {mca:.4f}"]
    if ref:
        lines += [f"  log cua tac gia              : top-1 {ref[0]:.4f} |              | mean class acc {ref[1]:.4f}",
                  f"  chenh lech                   : top-1 {top1 - ref[0]:+.4f} |              | mean class acc {mca - ref[1]:+.4f}"]
    text = "\n".join(lines); print("\n" + text)
    out = Path("runs/finegym_eval"); out.mkdir(parents=True, exist_ok=True)
    if not args.limit:
        (out / f"{stem}_clips{args.clips}.txt").write_text(text + "\n")
        np.save(out / f"{stem}_clips{args.clips}_prob.npy", probs.astype(np.float16))
        np.save(out / "labels.npy", labels)


if __name__ == "__main__":
    main()
