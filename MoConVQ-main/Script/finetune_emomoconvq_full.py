"""
finetune_emomoconvq_full.py
============================
Full-scale EmoMoconvq fine-tuning on the distillation dataset of 463
(stripped_caption, emotion, target_tokens, target_latents) triples produced
by Script/build_distillation_dataset.py.

Compared with the controlled overfit of Script/finetune_emomoconvq.py:
  - hundreds of distinct motions instead of 5 clips from one walking sequence
  - per-sample train/val split for monitoring generalisation
  - per-emotion loss tracking (the dataset is class-imbalanced)
  - best-by-val-loss checkpoint of just the trainable parameters

Run from MoConVQ-main:
    python Script/finetune_emomoconvq_full.py
Optional:
    --epochs N        (default 8)
    --lr R            (default 1e-4)
    --val-frac F      (default 0.1)
    --seed S          (default 0)
"""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from MoConVQCore.Model.emo_cross_trans import (
    EmoText2Motion_Transformer, EMOTION_TO_IDX, IDX_TO_EMOTION)


class gpt_config:
    num_vq = 512; embed_dim = 768; clip_dim = 512
    block_size = 52; num_layers = 9; n_head = 8
    drop_out_rate = 0.1; fc_rate = 2


DEPTH = 4


def text2bert_batch(texts, tok, enc, device):
    """Pre-compute T5 features for a list of texts; return list of (feat, mask)."""
    out = []
    for t in texts:
        enc_in = tok(t, return_tensors="pt", padding=True,
                     truncation=True, max_length=256)
        enc_in = {k: v.to(device) for k, v in enc_in.items()}
        with torch.no_grad():
            r = enc(**enc_in)
        out.append((r.last_hidden_state.detach(),
                    (~enc_in["attention_mask"].bool()).detach()))
    return out


def load_distill(h5_path):
    samples = []
    with h5py.File(h5_path, "r") as f:
        keys = sorted(k for k in f.keys() if k.startswith("sample_"))
        for k in keys:
            g = f[k]
            # base_idx is only present in the v2 (5-variant) datasets.
            # In v1 we fall back to using a unique per-sample id.
            base_idx = (int(g.attrs["base_idx"])
                        if "base_idx" in g.attrs else -1)
            samples.append({
                "caption_stripped": str(g.attrs["caption_stripped"]),
                "caption_full":     str(g.attrs["caption_full"]),
                "emotion":          str(g.attrs["emotion"]),
                "emotion_id":       int(g.attrs["emotion_id"]),
                "base_idx":         base_idx,
                "tokens":           np.array(g["tokens"][:], dtype=np.int64),
                "latents":          np.array(g["latents"][:], dtype=np.float32),
            })
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default=os.path.join(
        "out", "distillation_dataset", "distill_dataset.h5"))
    parser.add_argument("--out-dir",
        default=os.path.join("out", "finetune_full"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tune-top-k", type=int, default=4)
    parser.add_argument("--keep-emotions", type=str, default="",
                        help="comma-separated list of emotion names to keep "
                             "(e.g. neutral,happy,fearful). default = all.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---------- data ----------------------------------------------------
    samples = load_distill(args.h5)
    print(f"[data] {len(samples)} samples loaded from {args.h5}")

    # optional emotion-class filter
    if args.keep_emotions:
        keep = {e.strip() for e in args.keep_emotions.split(",") if e.strip()}
        before = len(samples)
        samples = [s for s in samples if s["emotion"] in keep]
        print(f"[data] kept {len(samples)} / {before} samples "
              f"matching emotions {sorted(keep)}")

    cls_count = {}
    for s in samples:
        cls_count[s["emotion"]] = cls_count.get(s["emotion"], 0) + 1
    print(f"[data] class counts: {cls_count}")

    rng = np.random.RandomState(args.seed)
    # Detect v2 dataset (base_idx present): split by base so val captions
    # are entirely held out from train -- a fair compositional test.
    has_base = all(s["base_idx"] >= 0 for s in samples)
    if has_base:
        all_bases = sorted({s["base_idx"] for s in samples})
        rng.shuffle(all_bases)
        n_val_b = max(1, int(len(all_bases) * args.val_frac))
        val_bases = set(all_bases[:n_val_b])
        train_idx = sorted(i for i, s in enumerate(samples)
                           if s["base_idx"] not in val_bases)
        val_idx = sorted(i for i, s in enumerate(samples)
                         if s["base_idx"] in val_bases)
        print(f"[data] base-stratified split  "
              f"train={len(train_idx)} ({len(all_bases)-n_val_b} bases)  "
              f"val={len(val_idx)} ({n_val_b} bases)")
    else:
        idx_all = np.arange(len(samples))
        rng.shuffle(idx_all)
        n_val = max(1, int(len(samples) * args.val_frac))
        val_idx = sorted(idx_all[:n_val].tolist())
        train_idx = sorted(idx_all[n_val:].tolist())
        print(f"[data] random split  train={len(train_idx)}  val={len(val_idx)}")

    # ---------- model ---------------------------------------------------
    cfg = {k: getattr(gpt_config, k)
           for k in vars(gpt_config) if not k.startswith("_")}
    dummy_emb = [torch.zeros(cfg["num_vq"] + 2, cfg["embed_dim"])
                 for _ in range(8)]
    model = EmoText2Motion_Transformer(embeddings=dummy_emb, **cfg)

    sd = torch.load("text_generation_GPT.pth", map_location="cpu")
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    assert not [k for k in res.missing_keys if "emotion" not in k]
    print(f"[model] pretrained checkpoint loaded "
          f"(missing only {len(res.missing_keys)} emotion params)")

    model = model.to(device).eval()
    model.configure_finetuning(num_temporal_layers_to_tune=args.tune_top_k)

    # ---------- T5 features (pre-computed once, then T5 is freed) ------
    from transformers import T5Tokenizer, T5EncoderModel
    print("[t5] loading t5-large ...")
    tok = T5Tokenizer.from_pretrained("t5-large")
    enc = T5EncoderModel.from_pretrained("t5-large").to(device).eval()

    print("[t5] pre-computing features for all samples ...")
    feats = []
    for i, s in enumerate(samples):
        e_in = tok(s["caption_stripped"], return_tensors="pt", padding=True,
                   truncation=True, max_length=256)
        e_in = {k: v.to(device) for k, v in e_in.items()}
        with torch.no_grad():
            r = enc(**e_in)
        feats.append((r.last_hidden_state.detach(),
                      (~e_in["attention_mask"].bool()).detach()))
    del enc
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"[t5] cached features for {len(feats)} captions; T5 freed")

    clip_feature = torch.zeros(1, cfg["clip_dim"], device=device)

    # ---------- pre-tensorize per-sample (latents, idxs, emo) ----------
    prepared = []
    for s, (bf, bm) in zip(samples, feats):
        T = s["tokens"].shape[0]
        L = s["latents"].shape[0]
        # forward expects latents length = idxs length - 1
        # h5 has T tokens and either T-1 or T-2 latents -> align to len min
        keep_T = min(T, L + 1)
        toks = torch.tensor(s["tokens"][:keep_T], dtype=torch.long,
                            device=device).unsqueeze(0)              # (1,T,4)
        lats = torch.tensor(s["latents"][:keep_T - 1], dtype=torch.float32,
                            device=device).unsqueeze(0)              # (1,T-1,768)
        emo_id = torch.tensor([s["emotion_id"]], device=device)
        prepared.append((lats, toks, bf, bm, emo_id, s["emotion"]))

    # ---------- optimizer ----------------------------------------------
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    # save copy of initial emotion embedding to track shift
    init_emo_w = model.trans_temporal.emotion_embedding.weight.detach().clone()

    # ---------- train loop ---------------------------------------------
    def epoch_pass(indices, train=True):
        per_emo_loss = {}
        per_emo_count = {}
        total_loss = 0.0
        total_n = 0
        for j, idx in enumerate(indices):
            lats, toks, bf, bm, emo_id, emo_name = prepared[idx]
            if train:
                logits, _ = model(lats, toks, clip_feature, bf, bm,
                                  emotion=emo_id)
                pred = logits[:, :, :DEPTH, :]
                loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]),
                                       toks.reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    logits, _ = model(lats, toks, clip_feature, bf, bm,
                                      emotion=emo_id)
                    pred = logits[:, :, :DEPTH, :]
                    loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]),
                                           toks.reshape(-1))
            li = float(loss.item())
            total_loss += li
            total_n += 1
            per_emo_loss[emo_name] = per_emo_loss.get(emo_name, 0.0) + li
            per_emo_count[emo_name] = per_emo_count.get(emo_name, 0) + 1
        mean = total_loss / max(total_n, 1)
        per_emo_mean = {k: per_emo_loss[k] / per_emo_count[k]
                        for k in per_emo_loss}
        return mean, per_emo_mean

    history = {"train": [], "val": [], "per_emo_train": [], "per_emo_val": []}
    best_val = float("inf")
    best_path = os.path.join(args.out_dir, "emomoconvq_best.pt")

    print(f"[train] epochs={args.epochs} lr={args.lr} "
          f"tune-top-k={args.tune_top_k}")
    t0 = time.time()
    for ep in range(args.epochs):
        rng.shuffle(train_idx)
        tr, tr_per = epoch_pass(train_idx, train=True)
        va, va_per = epoch_pass(val_idx, train=False)
        history["train"].append(tr)
        history["val"].append(va)
        history["per_emo_train"].append(tr_per)
        history["per_emo_val"].append(va_per)

        emo_shift = float(
            (model.trans_temporal.emotion_embedding.weight.detach()
             - init_emo_w).norm().item())
        print(f"  epoch {ep:2d}  train {tr:.4f}  val {va:.4f}  "
              f"emo-shift {emo_shift:.3f}  "
              f"({(time.time()-t0)/60:.1f} min elapsed)")
        # per-emotion val for visibility
        emo_str = "  ".join(
            f"{k}:{va_per[k]:.3f}" for k in sorted(va_per))
        print(f"      val by emotion: {emo_str}")

        if va < best_val:
            best_val = va
            # save only trainable parameters (small)
            tunable_state = {
                k: v for k, v in model.state_dict().items()
                if (k.startswith("trans_temporal.emotion_") or
                    any(k.startswith(f"trans_temporal.blocks.{12 - i - 1}.")
                        for i in range(args.tune_top_k)))
            }
            torch.save(tunable_state, best_path)
            print(f"      best-so-far val={va:.4f} -> {best_path}")

    # ---------- save summary + plot ------------------------------------
    summary = {
        "n_train": len(train_idx), "n_val": len(val_idx),
        "epochs": args.epochs, "lr": args.lr, "tune_top_k": args.tune_top_k,
        "history": history,
        "best_val_loss": best_val,
        "emotion_embedding_shift": float(
            (model.trans_temporal.emotion_embedding.weight.detach()
             - init_emo_w).norm().item()),
        "class_counts": cls_count,
    }
    with open(os.path.join(args.out_dir, "finetune_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(history["train"], "-o", color="#d9534f",
                label="train", markersize=4)
        ax.plot(history["val"], "-s", color="#4a90d9",
                label="val", markersize=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("NLL loss")
        ax.set_title("EmoMoconvq full fine-tuning (distillation dataset)")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "finetune_loss_curve_full.png"),
                    dpi=200)
        print(f"[ok] loss curve -> {os.path.join(args.out_dir, 'finetune_loss_curve_full.png')}")
    except ImportError:
        pass

    print(f"\n[done] best val NLL = {best_val:.4f}  ({(time.time()-t0)/60:.1f} min)")
    print(f"[ok] summary    -> {os.path.join(args.out_dir, 'finetune_summary.json')}")
    print(f"[ok] best model -> {best_path}")


if __name__ == "__main__":
    main()
