"""
build_distillation_dataset_v2.py
=================================
Five-variant distillation dataset for EmoMoconvq fine-tuning.

Why v2 (vs v1):
---------------
v1 paired every (caption -> emotion) one-to-one (463 unique captions, one
emotion each). Result: the model could memorise (caption -> tokens) purely
from text and IGNORE the emotion label. Compositional eval confirmed this:
the emotion knob had no effect on held-out neutral captions.

v2 fixes the experimental design: each base caption is paired with ALL
five emotion labels (and five distinct target token sequences). The base
TEXT is identical across the five variants, so the model can only fit
them by routing through the discrete emotion label -- the emotion module
is forced to do work.

Data construction:
  1. Pick N base captions from HumanML3D's train split that
       * contain no emotion keyword
       * contain a clear motion verb (walk / run / sit / ...)
       * are NOT in our v1 (463) set
  2. For every base caption B and every emotion e in
     {neutral, happy, sad, angry, fearful}:
       * compose `full = B [+ ' ' + adverb(e)]`
       * feed `full` to the baseline T2M-MoConGPT -> target tokens
  3. Save (base_caption, emotion, target_tokens, target_latents)
     with caption_stripped = base so the existing fine-tuning script
     reads it unchanged.

Run from the MoConVQ-main directory:
    python Script/build_distillation_dataset_v2.py
Optional:
    --n-base N        number of base captions (default 100)
"""

import argparse
import csv
import os
import re
import sys
import time

import h5py
import numpy as np
import torch

import MoConVQCore.Utils.pytorch_utils as ptu
from MoConVQCore.Model.cross_trans_ori_fixsum import Text2Motion_Transformer


EMOTION_ADVERB = {
    "neutral": "",            # baseline default behaviour
    "happy":   "happily",
    "sad":     "sadly",
    "angry":   "angrily",
    "fearful": "fearfully",
}
EMOTION_TO_IDX = {"neutral": 0, "happy": 1, "sad": 2, "angry": 3, "fearful": 4}
EMOTIONS = list(EMOTION_ADVERB.keys())


# Reject captions that contain any emotion keyword from v1
EMOTION_KEYWORDS = {
    "happy":   ["happy","happily","joyful","joyfully","joyously","cheerful",
                "cheerfully","gleeful","gleefully","merrily","jolly",
                "delighted","elated","excitedly"],
    "sad":     ["sad","sadly","sorrowful","sorrowfully","gloomy","gloomily",
                "depressed","miserable","miserably","unhappy","unhappily",
                "dejected","mournful","mournfully","glum","downcast"],
    "angry":   ["angry","angrily","furious","furiously","irritated",
                "irritably","enraged","annoyed","irate","aggressive",
                "aggressively"],
    "fearful": ["fearful","fearfully","scared","afraid","frightened",
                "terrified","timid","timidly","nervous","nervously",
                "anxious","anxiously","cautious","cautiously",
                "hesitant","hesitantly"],
}
_ALL_EMO_WORDS = sorted(
    {w for ws in EMOTION_KEYWORDS.values() for w in ws}, key=len, reverse=True)
_EMO_RE = re.compile(r"\b(" + "|".join(_ALL_EMO_WORDS) + r")\b", re.IGNORECASE)

# require at least one motion verb
_ACTION_RE = re.compile(
    r"\b(walk|run|jump|sit|stand|turn|step|move|march|stride|stroll|jog|"
    r"pick|throw|kick|punch|wave|dance|squat|crouch|hop|leap|skip|"
    r"raise|lower|reach|swing|crawl|climb|push|pull|spin|rotate|"
    r"bow|nod|shake|stretch|lean|bend|twist|lift|drop|catch)s?\b",
    re.IGNORECASE)


class gpt_config:
    num_vq = 512; embed_dim = 768; clip_dim = 512
    block_size = 52; num_layers = 9; n_head = 8
    drop_out_rate = 0.1; fc_rate = 2


def select_base_captions(texts_dir, split_file, exclude_ids,
                         n_base, seed=0):
    """Pick `n_base` diverse base captions."""
    with open(split_file, "r", encoding="utf-8") as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    out = []
    for mid in ids:
        if mid in exclude_ids:
            continue
        fn = os.path.join(texts_dir, mid + ".txt")
        if not os.path.isfile(fn):
            continue
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                cap = line.strip().split("#")[0].strip()
                if not cap:
                    continue
                if _EMO_RE.search(cap):
                    continue
                if not _ACTION_RE.search(cap):
                    continue
                wc = len(cap.split())
                if wc < 5 or wc > 25:
                    continue
                out.append((mid, cap))
                break
        if len(out) >= n_base:
            break
    return out[:n_base]


def augment_emotion(base, emotion):
    """Inject the emotion adverb at the sentence end."""
    adv = EMOTION_ADVERB.get(emotion, "")
    if not adv:
        return base
    s = base.rstrip(" .,!?;:")
    return f"{s} {adv}."


def text2bert(text, tok, enc, device):
    e = tok(text, return_tensors="pt", padding=True,
            truncation=True, max_length=256)
    e = {k: v.to(device) for k, v in e.items()}
    with torch.no_grad():
        r = enc(**e)
    return r.last_hidden_state, ~e["attention_mask"].bool()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-base", type=int, default=100)
    parser.add_argument("--texts-dir", default=os.path.join(
        "..", "HumanML3D", "texts"))
    parser.add_argument("--split-file", default=os.path.join(
        "..", "HumanML3D", "texts", "train.txt"))
    parser.add_argument("--exclude-csv", default=os.path.join(
        "out", "emotion_dataset", "emotion_dataset.csv"))
    parser.add_argument("--out-dir", default=os.path.join(
        "out", "distillation_dataset_v2"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]            # hide our flags from build_agent

    os.makedirs(args.out_dir, exist_ok=True)

    # --- exclude v1 motion ids -----------------------------------------
    exclude_ids = set()
    if os.path.isfile(args.exclude_csv):
        with open(args.exclude_csv, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                mid = r.get("motion_id", "")
                if mid:
                    exclude_ids.add(mid)
    print(f"[data] excluding {len(exclude_ids)} motion ids from v1")

    bases = select_base_captions(args.texts_dir, args.split_file,
                                  exclude_ids, args.n_base, seed=args.seed)
    print(f"[data] selected {len(bases)} base captions")
    if not bases:
        raise SystemExit("[ERROR] no base captions found -- check paths.")

    with open(os.path.join(args.out_dir, "base_captions.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["base_idx", "motion_id", "caption"])
        for i, (mid, cap) in enumerate(bases):
            w.writerow([i, mid, cap])

    # --- build agent + baseline GPT + T5 -------------------------------
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
    gpt = Text2Motion_Transformer(embeddings=embed_torch, **cfg).to(ptu.device)
    sd = torch.load("text_generation_GPT.pth", map_location=ptu.device)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    gpt.load_state_dict(sd, strict=True)
    gpt.eval()

    from transformers import T5Tokenizer, T5EncoderModel
    print("[t5] loading t5-large ...")
    tok = T5Tokenizer.from_pretrained("t5-large")
    enc = T5EncoderModel.from_pretrained("t5-large").to(ptu.device).eval()

    clip_feature = torch.zeros((1, 512), device=ptu.device)
    depth = 4

    h5_path = os.path.join(args.out_dir, "distill_dataset.h5")
    h5 = h5py.File(h5_path, "w")
    meta = h5.create_group("meta")

    summary = open(os.path.join(args.out_dir, "summary.csv"), "w",
                   newline="", encoding="utf-8")
    sw = csv.writer(summary)
    sw.writerow(["sample_idx", "base_idx", "motion_id", "emotion",
                 "emotion_id", "caption_base", "caption_full", "T", "ok"])

    Ts = []; sample_idx = 0; n_ok = 0; n_short = 0
    t0 = time.time()

    for bi, (mid, base) in enumerate(bases):
        for emotion in EMOTIONS:
            full = augment_emotion(base, emotion)
            # fix seed for each emotion variant so any difference between
            # variants is driven by the input text alone (not sampling noise)
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            try:
                bf, bm = text2bert(full, tok, enc, ptu.device)
                with torch.no_grad():
                    cur, idxs = gpt.sample(clip_feature, bf, bm)
                idxs_np = idxs.detach().cpu().numpy().reshape(-1)
                T = len(idxs_np) // depth
                idxs_np = idxs_np[:T * depth].reshape(T, depth).astype(np.int64)
                latents = cur.detach().cpu().numpy()[0].astype(np.float32)
                ok = T >= 5
                if not ok:
                    n_short += 1
                    sw.writerow([sample_idx, bi, mid, emotion,
                                 EMOTION_TO_IDX[emotion], base, full, T, 0])
                    sample_idx += 1
                    continue
            except Exception as e:
                print(f"  [skip] base {bi} / {emotion}: {e}")
                sample_idx += 1
                continue

            g = h5.create_group(f"sample_{sample_idx:04d}")
            g.create_dataset("tokens", data=idxs_np)
            g.create_dataset("latents", data=latents)
            g.attrs["motion_id"] = mid
            g.attrs["base_idx"] = bi
            g.attrs["emotion"] = emotion
            g.attrs["emotion_id"] = EMOTION_TO_IDX[emotion]
            g.attrs["caption_base"] = base
            g.attrs["caption_full"] = full
            # alias so the existing finetune script reads it as the input text
            g.attrs["caption_stripped"] = base
            g.attrs["T"] = T
            sw.writerow([sample_idx, bi, mid, emotion,
                         EMOTION_TO_IDX[emotion], base, full, T, 1])
            Ts.append(T); n_ok += 1; sample_idx += 1

        if (bi + 1) % 10 == 0 or bi == len(bases) - 1:
            elapsed = time.time() - t0
            rate = (bi + 1) / max(elapsed, 1e-3)
            eta = (len(bases) - bi - 1) / max(rate, 1e-3)
            print(f"  [{bi+1:3d}/{len(bases)}]  "
                  f"{rate:.2f} bases/s  ETA {eta/60:.1f} min")

    meta.attrs["n_samples"] = n_ok
    meta.attrs["n_base"] = len(bases)
    meta.attrs["n_emotions"] = len(EMOTIONS)
    meta.attrs["depth"] = depth
    meta.create_dataset("emotion_keys",
        data=np.array(list(EMOTION_TO_IDX.keys()), dtype=h5py.string_dtype()))
    meta.create_dataset("emotion_vals",
        data=np.array(list(EMOTION_TO_IDX.values()), dtype=np.int64))
    h5.close(); summary.close()

    elapsed = time.time() - t0
    print("\n" + "=" * 62)
    print(" Distillation dataset v2 built")
    print("=" * 62)
    print(f"  successful samples : {n_ok} / {len(bases)*len(EMOTIONS)}")
    print(f"  short / discarded  : {n_short}")
    if Ts:
        Ts = np.asarray(Ts)
        print(f"  sequence length T  : min {Ts.min()} / med {int(np.median(Ts))} "
              f"/ max {Ts.max()} / mean {Ts.mean():.1f}")
    print(f"  elapsed            : {elapsed/60:.1f} min")
    print(f"\n[ok] h5 dataset -> {h5_path}")
    print(f"[ok] summary    -> {os.path.join(args.out_dir, 'summary.csv')}")
    print(f"[ok] base list  -> {os.path.join(args.out_dir, 'base_captions.csv')}")


if __name__ == "__main__":
    main()
