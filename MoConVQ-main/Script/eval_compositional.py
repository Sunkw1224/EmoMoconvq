"""
eval_compositional.py
======================
Compositional-generalization evaluation for EmoMoconvq -- the core scientific
test of whether the emotion module adds controllable variation BEYOND what
the baseline T2M-MoConGPT already does through emotion adverbs in text.

Test set
--------
N neutral captions sampled from HumanML3D's training split that
(a) contain NO emotion keywords, and
(b) are NOT in our 463-pair fine-tuning set.
For each test caption c we then generate:

  * EmoMoconvq:  5 outputs, one per emotion label (seed fixed across them
                  so any difference comes from the emotion knob alone)
  * Baseline  :  5 outputs, varying the random seed (no emotion knob)

Metrics
-------
1. Token-disagreement matrix  D[c, i, j] = fraction of positions where the
   two outputs disagree. We report:
     sep_emo  = mean over c,i!=j of the EmoMoconvq matrix
     sep_base = mean over c,s!=t of the baseline matrix
     ratio    = sep_emo / sep_base   (> 1 means emotion is more than noise)

2. Token classifier accuracy.  We train a small bi-GRU on the 463-pair
   distillation dataset (tokens -> emotion), hold out 10% for sanity, then
   run it on EmoMoconvq's generations and report classification accuracy
   per emotion (does the classifier recognise the intended emotion?).

3. Confusion matrix figure for the classifier on EmoMoconvq generations.

Pre-requisites:
    Script/build_emotion_dataset.py     (the 463-pair csv)
    Script/build_distillation_dataset.py (the .h5)
    Script/finetune_emomoconvq_full.py  (the best.pt)

Run from MoConVQ-main:
    python Script/eval_compositional.py
"""

import argparse
import csv
import json
import os
import re
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import MoConVQCore.Utils.pytorch_utils as ptu
from MoConVQCore.Model.cross_trans_ori_fixsum import Text2Motion_Transformer
from MoConVQCore.Model.emo_cross_trans import (
    EmoText2Motion_Transformer, EMOTION_TO_IDX, IDX_TO_EMOTION)


# Emotion lexicon (must stay in sync with build_emotion_dataset.py)
EMOTION_KEYWORDS = {
    "happy": ["happy", "happily", "joyful", "joyfully", "joyously", "cheerful",
              "cheerfully", "gleeful", "gleefully", "merrily", "jolly",
              "delighted", "elated", "excitedly"],
    "sad":   ["sad", "sadly", "sorrowful", "sorrowfully", "gloomy", "gloomily",
              "depressed", "miserable", "miserably", "unhappy", "unhappily",
              "dejected", "mournful", "mournfully", "glum", "downcast"],
    "angry": ["angry", "angrily", "furious", "furiously", "irritated",
              "irritably", "enraged", "annoyed", "irate", "aggressive",
              "aggressively"],
    "fearful": ["fearful", "fearfully", "scared", "afraid", "frightened",
                "terrified", "timid", "timidly", "nervous", "nervously",
                "anxious", "anxiously", "cautious", "cautiously",
                "hesitant", "hesitantly"],
}
_ALL_EMO_WORDS = sorted(
    {w for ws in EMOTION_KEYWORDS.values() for w in ws}, key=len, reverse=True)
_EMO_RE = re.compile(r"\b(" + "|".join(_ALL_EMO_WORDS) + r")\b", re.IGNORECASE)
EMO_LIST_ALL = ["neutral", "happy", "sad", "angry", "fearful"]
DEPTH = 4


class gpt_config:
    num_vq = 512; embed_dim = 768; clip_dim = 512
    block_size = 52; num_layers = 9; n_head = 8
    drop_out_rate = 0.1; fc_rate = 2


# ============================================================
# 1. Neutral caption pool from HumanML3D
# ============================================================
def sample_neutral_captions(texts_dir, split_file, exclude_ids,
                            n_captions, seed=0):
    """Return list of (motion_id, caption) for captions that:
       - belong to the train split,
       - contain no emotion keyword,
       - whose motion_id is not in exclude_ids.
    """
    with open(split_file, "r", encoding="utf-8") as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)

    pool = []
    for mid in ids:
        if mid in exclude_ids:
            continue
        fn = os.path.join(texts_dir, mid + ".txt")
        if not os.path.isfile(fn):
            continue
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cap = line.split("#")[0].strip()
                if not cap:
                    continue
                if _EMO_RE.search(cap):
                    continue  # contains emotion keyword
                pool.append((mid, cap))
                break  # take only the first caption per motion
        if len(pool) >= n_captions:
            break
    return pool[:n_captions]


# ============================================================
# 2. Model construction
# ============================================================
def build_models(device, tune_top_k, ft_checkpoint=None):
    print("[setup] building agent ...")
    from Script.moconvq_builder import build_agent
    agent, env = build_agent(gpu=0)
    ptu.init_gpu(True, gpu_id=0)
    agent.simple_load("moconvq_base.data", strict=True)
    agent.eval()

    embed_torch = [
        torch.cat([bn.embedding, torch.zeros_like(bn.embedding[:2])], dim=0)
        for bn in agent.posterior.bottle_neck_list
    ]
    cfg = {k: getattr(gpt_config, k)
           for k in vars(gpt_config) if not k.startswith("_")}

    # --- baseline (no emotion knob) ---
    base = Text2Motion_Transformer(embeddings=embed_torch, **cfg).to(device)
    sd = torch.load("text_generation_GPT.pth", map_location=device)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    base.load_state_dict(sd, strict=True); base.eval()

    # --- EmoMoconvq ---
    emo = EmoText2Motion_Transformer(embeddings=embed_torch, **cfg).to(device)
    emo.load_state_dict(sd, strict=False)
    # overlay best fine-tuned tunable subset
    best_path = ft_checkpoint or os.path.join(
        "out", "finetune_full", "emomoconvq_best.pt")
    if os.path.isfile(best_path):
        tunable = torch.load(best_path, map_location=device)
        res = emo.load_state_dict(tunable, strict=False)
        print(f"[setup] loaded fine-tuned checkpoint: {len(tunable)} tensors "
              f"(missing={len(res.missing_keys)})")
    else:
        print(f"[WARN] no best fine-tuned checkpoint at {best_path}; "
              f"running on zero-init emotion module")
    emo.eval(); emo.configure_finetuning(num_temporal_layers_to_tune=tune_top_k,
                                          verbose=False)
    return agent, base, emo


def text2bert(text, tok, enc, device):
    e = tok(text, return_tensors="pt", padding=True, truncation=True,
            max_length=256)
    e = {k: v.to(device) for k, v in e.items()}
    with torch.no_grad():
        r = enc(**e)
    return r.last_hidden_state.detach(), (~e["attention_mask"].bool()).detach()


def sample_tokens(model, clip_feat, bf, bm, device,
                  emotion=None, seed=0, depth=DEPTH):
    """Sample once and return (tokens [T,depth], latents [T-1 or T-2, 768])."""
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        if emotion is None:
            cur, idxs = model.sample(clip_feat, bf, bm)
        else:
            emo_id = torch.tensor([emotion], device=device)
            cur, idxs = model.sample(clip_feat, bf, bm, emotion=emo_id)
    idxs_np = idxs.detach().cpu().numpy().reshape(-1)
    T = len(idxs_np) // depth
    return (idxs_np[:T * depth].reshape(T, depth).astype(np.int64),
            cur.detach().cpu().numpy()[0].astype(np.float32))


# ============================================================
# 3. Token-disagreement metric
# ============================================================
def pairwise_token_disagreement(token_seqs):
    """token_seqs: list of (T_i, depth) arrays. Return n x n disagreement matrix."""
    n = len(token_seqs)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = token_seqs[i], token_seqs[j]
            T = min(len(a), len(b))
            if T == 0:
                M[i, j] = 1.0; continue
            M[i, j] = float((a[:T] != b[:T]).any(axis=1).mean())
    return M


# ============================================================
# 4. Tiny GRU classifier  tokens -> emotion
# ============================================================
class TokenClassifier(nn.Module):
    def __init__(self, num_vq=514, embed_dim=64, hid=128, n_class=5,
                 _unused=None):
        super().__init__()
        self.embeds = nn.ModuleList(
            [nn.Embedding(num_vq, embed_dim) for _ in range(DEPTH)])
        self.gru = nn.GRU(embed_dim * DEPTH, hid, batch_first=True,
                          bidirectional=True)
        self.head = nn.Linear(hid * 2, n_class)

    def forward(self, tokens):           # tokens: (B, T, DEPTH)
        e = torch.cat([self.embeds[d](tokens[:, :, d])
                       for d in range(DEPTH)], dim=-1)  # (B, T, embed*depth)
        o, _ = self.gru(e)                              # (B, T, 2*hid)
        return self.head(o.mean(dim=1))                 # (B, n_class)


def train_classifier(distill_h5, device, n_epochs=30, lr=1e-3, val_frac=0.1,
                     seed=0, keep_orig_ids=None):
    """Train TokenClassifier on distill dataset.  Return (model, val_acc).

    If `keep_orig_ids` is given, only samples whose emotion_id is in that
    list are kept, and labels are remapped to a dense 0..n_class-1 space
    in the same order as `keep_orig_ids`.
    """
    with h5py.File(distill_h5, "r") as f:
        keys = sorted(k for k in f.keys() if k.startswith("sample_"))
        raw = []
        for k in keys:
            g = f[k]
            raw.append((np.array(g["tokens"][:], dtype=np.int64),
                        int(g.attrs["emotion_id"])))

    if keep_orig_ids is None:
        items = raw
        n_class = 5
    else:
        remap = {orig: new for new, orig in enumerate(keep_orig_ids)}
        items = [(t, remap[oid]) for t, oid in raw if oid in remap]
        n_class = len(keep_orig_ids)
    print(f"  [cls] dataset filtered: {len(items)} samples, {n_class} classes")

    rng = np.random.RandomState(seed)
    idx = np.arange(len(items)); rng.shuffle(idx)
    n_val = max(5, int(len(items) * val_frac))
    val_idx = idx[:n_val]; train_idx = idx[n_val:]

    model = TokenClassifier(n_class=n_class).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def step(indices, train):
        loss_sum, correct, total = 0.0, 0, 0
        for i in indices:
            toks, lab = items[i]
            x = torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor([lab], dtype=torch.long, device=device)
            if train:
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                opt.zero_grad(); loss.backward(); opt.step()
            else:
                with torch.no_grad():
                    logits = model(x); loss = F.cross_entropy(logits, y)
            loss_sum += float(loss.item())
            correct += int((logits.argmax(-1) == y).item())
            total += 1
        return loss_sum / total, correct / total

    best_val = -1.0
    for ep in range(n_epochs):
        rng.shuffle(train_idx)
        tr_loss, tr_acc = step(train_idx, train=True)
        model.eval()
        va_loss, va_acc = step(val_idx, train=False)
        model.train()
        if va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        if ep == 0 or (ep + 1) % 5 == 0 or ep == n_epochs - 1:
            print(f"  [cls] ep {ep:2d}  train {tr_loss:.3f}/{tr_acc:.2%}  "
                  f"val {va_loss:.3f}/{va_acc:.2%}")
    model.load_state_dict(best_state); model.eval()
    return model, best_val


# ============================================================
# 5. Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-captions", type=int, default=50)
    parser.add_argument("--texts-dir", default=os.path.join(
        "..", "HumanML3D", "texts"))
    parser.add_argument("--split-file", default=os.path.join(
        "..", "HumanML3D", "texts", "train.txt"))
    parser.add_argument("--emotion-csv", default=os.path.join(
        "out", "emotion_dataset", "emotion_dataset.csv"))
    parser.add_argument("--distill-h5", default=os.path.join(
        "out", "distillation_dataset", "distill_dataset.h5"))
    parser.add_argument("--out-dir", default=os.path.join(
        "out", "eval_compositional"))
    parser.add_argument("--checkpoint", default=os.path.join(
        "out", "finetune_full", "emomoconvq_best.pt"),
        help="path to fine-tuned tunable-subset checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tune-top-k", type=int, default=4)
    parser.add_argument("--keep-emotions", type=str, default="",
                        help="comma-separated list of emotion names to keep "
                             "for both classifier training and EmoMoconvq "
                             "generation evaluation. default = all 5.")
    args = parser.parse_args()

    # hide our flags from build_agent's internal argparse
    sys.argv = [sys.argv[0]]

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- which emotions to evaluate ------------------------------------
    if args.keep_emotions:
        keep_set = {e.strip() for e in args.keep_emotions.split(",") if e.strip()}
        # canonicalise order by original EMOTION_TO_IDX index for consistency
        emo_list = [e for e in EMO_LIST_ALL if e in keep_set]
    else:
        emo_list = list(EMO_LIST_ALL)
    keep_orig_ids = [EMOTION_TO_IDX[e] for e in emo_list]
    n_emo = len(emo_list)
    print(f"[setup] evaluating {n_emo} emotion classes: {emo_list}")

    # --- exclude motion-ids that already appear in our 463-pair set ----
    exclude_ids = set()
    with open(args.emotion_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            mid = r.get("motion_id", "")
            if mid:
                exclude_ids.add(mid)
    print(f"[data] excluding {len(exclude_ids)} motion-ids from train pool")

    test_caps = sample_neutral_captions(
        args.texts_dir, args.split_file, exclude_ids,
        n_captions=args.n_captions, seed=args.seed)
    print(f"[data] curated {len(test_caps)} neutral test captions")
    with open(os.path.join(args.out_dir, "test_captions.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["idx", "motion_id", "caption"])
        for i, (mid, cap) in enumerate(test_caps):
            w.writerow([i, mid, cap])

    # --- models ---------------------------------------------------------
    agent, base, emo = build_models(device, args.tune_top_k, args.checkpoint)

    from transformers import T5Tokenizer, T5EncoderModel
    print("[t5] loading t5-large ...")
    tok = T5Tokenizer.from_pretrained("t5-large")
    enc = T5EncoderModel.from_pretrained("t5-large").to(device).eval()
    clip_feature = torch.zeros(1, gpt_config.clip_dim, device=device)

    # --- generation ----------------------------------------------------
    print(f"[gen] generating {len(test_caps)} captions × "
          f"({len(emo_list)} emotions + 5 baseline seeds) ...")
    emo_outputs = []     # [c][emotion_idx] -> (tokens, latents)
    base_outputs = []    # [c][seed_idx]    -> (tokens, latents)
    t0 = time.time()
    for ci, (mid, cap) in enumerate(test_caps):
        bf, bm = text2bert(cap, tok, enc, device)

        # EmoMoconvq: same seed, varying emotion
        emo_row = []
        for ei, ename in enumerate(emo_list):
            toks, lats = sample_tokens(emo, clip_feature, bf, bm, device,
                                       emotion=EMOTION_TO_IDX[ename],
                                       seed=args.seed)
            emo_row.append((toks, lats))
        emo_outputs.append(emo_row)

        # Baseline: varying seed
        base_row = []
        for s in range(len(emo_list)):
            toks, lats = sample_tokens(base, clip_feature, bf, bm, device,
                                       emotion=None, seed=args.seed + s)
            base_row.append((toks, lats))
        base_outputs.append(base_row)

        if (ci + 1) % 10 == 0 or ci == len(test_caps) - 1:
            print(f"  [{ci+1:3d}/{len(test_caps)}] "
                  f"({(ci+1)/(time.time()-t0):.2f} caps/s)")

    # free T5
    del enc
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- metric 1: pairwise token disagreement -------------------------
    sep_emo_per_cap, sep_base_per_cap = [], []
    for ci in range(len(test_caps)):
        Me = pairwise_token_disagreement([t for t, _ in emo_outputs[ci]])
        Mb = pairwise_token_disagreement([t for t, _ in base_outputs[ci]])
        # off-diagonal mean
        n = Me.shape[0]
        sep_emo_per_cap.append(float(Me[~np.eye(n, dtype=bool)].mean()))
        sep_base_per_cap.append(float(Mb[~np.eye(n, dtype=bool)].mean()))
    sep_emo  = float(np.mean(sep_emo_per_cap))
    sep_base = float(np.mean(sep_base_per_cap))
    ratio    = sep_emo / max(sep_base, 1e-6)
    print("\n" + "=" * 62)
    print(" Compositional generalization metrics")
    print("=" * 62)
    print(f"  EmoMoconvq emotion separation  (mean disagreement) : {sep_emo:.4f}")
    print(f"  Baseline   stochastic separation (mean disagreement): {sep_base:.4f}")
    print(f"  ratio (emo / base) : {ratio:.3f}    (> 1.0 means knob > noise)")

    # --- metric 2: token classifier ------------------------------------
    print(f"\n[cls] training {n_emo}-class classifier on distillation ...")
    cls, cls_val_acc = train_classifier(args.distill_h5, device,
                                         keep_orig_ids=keep_orig_ids)
    print(f"  [cls] classifier val acc on distill held-out: {cls_val_acc:.2%}")

    # --- metric 3: classifier on EmoMoconvq generations ----------------
    print("[cls] evaluating classifier on EmoMoconvq generations ...")
    correct, total = 0, 0
    n_emo = len(emo_list)
    confusion = np.zeros((n_emo, n_emo), dtype=int)
    per_emo_correct = {e: 0 for e in emo_list}
    per_emo_total   = {e: 0 for e in emo_list}
    for ci in range(len(test_caps)):
        for ei, ename in enumerate(emo_list):
            toks, _ = emo_outputs[ci][ei]
            x = torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                pred = int(cls(x).argmax(-1).item())
            confusion[ei, pred] += 1
            per_emo_total[ename] += 1
            if pred == ei:
                correct += 1
                per_emo_correct[ename] += 1
            total += 1
    acc = correct / max(total, 1)
    print(f"  classifier accuracy on EmoMoconvq generations: {acc:.2%} "
          f"(chance = {1.0/n_emo:.2%})")
    for e in emo_list:
        if per_emo_total[e]:
            print(f"    {e:8s}  {per_emo_correct[e]}/{per_emo_total[e]} = "
                  f"{per_emo_correct[e]/per_emo_total[e]:.2%}")

    # --- baseline classifier sanity: check classifier on baseline outputs
    base_correct = 0; base_total = 0
    for ci in range(len(test_caps)):
        for s in range(n_emo):
            toks, _ = base_outputs[ci][s]
            x = torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                pred = int(cls(x).argmax(-1).item())
            # baseline has no intended emotion -- we just record predictions
            base_total += 1
    print(f"  (baseline outputs have no intended emotion; we don't score them)")

    # --- save -----------------------------------------------------------
    np.savez(os.path.join(args.out_dir, "generations.npz"),
             emo_tokens=np.array(
                 [[emo_outputs[c][e][0] for e in range(n_emo)]
                  for c in range(len(test_caps))], dtype=object),
             base_tokens=np.array(
                 [[base_outputs[c][s][0] for s in range(n_emo)]
                  for c in range(len(test_caps))], dtype=object))

    results = {
        "n_captions": len(test_caps),
        "emotions": emo_list,
        "metrics": {
            "sep_emo_mean": sep_emo,
            "sep_base_mean": sep_base,
            "separation_ratio": ratio,
            "classifier_val_acc_on_distill": cls_val_acc,
            "classifier_acc_on_emomoconvq_generations": acc,
            "per_emotion_acc": {
                e: (per_emo_correct[e] / per_emo_total[e]
                    if per_emo_total[e] else None) for e in emo_list},
        },
        "confusion_matrix": confusion.tolist(),
    }
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- confusion-matrix figure ---------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        im = ax.imshow(confusion, cmap="Blues")
        ax.set_xticks(range(n_emo)); ax.set_yticks(range(n_emo))
        ax.set_xticklabels(emo_list, rotation=30, ha="right")
        ax.set_yticklabels(emo_list)
        ax.set_xlabel("classifier prediction")
        ax.set_ylabel("intended emotion (conditioning label)")
        ax.set_title(f"Classifier on EmoMoconvq generations  "
                     f"(acc={acc:.1%})")
        for i in range(n_emo):
            for j in range(n_emo):
                ax.text(j, i, str(confusion[i, j]),
                        ha="center", va="center",
                        color="white" if confusion[i, j] > confusion.max() / 2
                        else "black", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "confusion.png"), dpi=200)
        print(f"\n[ok] confusion -> {os.path.join(args.out_dir, 'confusion.png')}")
    except ImportError:
        pass

    print(f"[ok] results   -> {os.path.join(args.out_dir, 'results.json')}")
    print(f"[ok] captions  -> {os.path.join(args.out_dir, 'test_captions.csv')}")


if __name__ == "__main__":
    main()
