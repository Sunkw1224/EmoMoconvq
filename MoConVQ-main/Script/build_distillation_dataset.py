"""
build_distillation_dataset.py
==============================
Build the EmoMoconvq fine-tuning dataset by *distilling* the pretrained
baseline T2M-MoConGPT.

Pre-check (Script/investigate_baseline_emotion.py) confirmed that the
baseline produces very different motion tokens for "walks happily" vs
"walks sadly" etc. We exploit that: for each of the 463 emotion-labelled
captions from build_emotion_dataset.py, we use the baseline (text only)
to generate motion tokens, and save the triple

    (stripped_caption, emotion_label, target_tokens)

The stripped caption has the emotion-bearing word removed, so during
EmoMoconvq fine-tuning the discrete emotion label is the ONLY source of
the emotional information that the baseline used. This forces the
emotion module to learn the (label -> motion-style) mapping that was
implicit in the baseline's text training.

Pre-requisite:  python Script/build_emotion_dataset.py  (creates the CSV)

Run from the MoConVQ-main directory:
    python Script/build_distillation_dataset.py
Optional:
    --limit N     # only process the first N rows (for a smaller debug run)
    --max-length M  # cap the sample length per motion (default 50)
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


# Keep in sync with build_emotion_dataset.py
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
EMOTION_TO_IDX = {"neutral": 0, "happy": 1, "sad": 2, "angry": 3, "fearful": 4}

_ALL_WORDS = sorted(
    {w for ws in EMOTION_KEYWORDS.values() for w in ws}, key=len, reverse=True)
_STRIP_RE = re.compile(r"\b(" + "|".join(_ALL_WORDS) + r")\b", re.IGNORECASE)
_PUNCT_FIX_RE = re.compile(r"\s+([,.;:!?])")
_MULTI_WS_RE = re.compile(r"\s+")


def strip_emotion_words(text):
    """Remove every emotion keyword (word-bounded) from `text`."""
    out = _STRIP_RE.sub("", text)
    out = _PUNCT_FIX_RE.sub(r"\1", out)
    out = _MULTI_WS_RE.sub(" ", out).strip().strip(",")
    if not out or out.split() in ([], ["a"], ["the"]):
        return "a person moves"
    return out


class gpt_config:
    num_vq = 512; embed_dim = 768; clip_dim = 512
    block_size = 52; num_layers = 9; n_head = 8
    drop_out_rate = 0.1; fc_rate = 2


def text2bert(text, tok, enc, device):
    enc_in = tok(text, return_tensors="pt", padding=True,
                 truncation=True, max_length=256)
    enc_in = {k: v.to(device) for k, v in enc_in.items()}
    with torch.no_grad():
        out = enc(**enc_in)
    return out.last_hidden_state, ~enc_in["attention_mask"].bool()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-in", default=os.path.join(
        "out", "emotion_dataset", "emotion_dataset.csv"),
        help="emotion-labelled caption csv from build_emotion_dataset.py")
    parser.add_argument("--out-dir", default=os.path.join(
        "out", "distillation_dataset"))
    parser.add_argument("--limit", type=int, default=0,
                        help="if >0, only process the first N rows")
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # build_agent's own argparse reads sys.argv -- hide our flags from it
    sys.argv = [sys.argv[0]]

    os.makedirs(args.out_dir, exist_ok=True)

    # --- read 463 (motion_id, emotion, caption, matched_keywords) -------
    rows = []
    with open(args.csv_in, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if args.limit > 0:
        rows = rows[:args.limit]
    print(f"[data] {len(rows)} emotion-labelled captions to distill")

    # --- build pretrained agent + GPT + T5 ------------------------------
    print("[setup] building agent ...")
    from Script.moconvq_builder import build_agent
    device = 0
    agent, env = build_agent(gpu=device)
    ptu.init_gpu(True, gpu_id=device)
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
    depth = 4   # MoConGPT max_depth

    # --- generate ---------------------------------------------------------
    h5_path = os.path.join(args.out_dir, "distill_dataset.h5")
    h5 = h5py.File(h5_path, "w")
    grp_meta = h5.create_group("meta")

    # also keep a flat csv summary for inspection
    summary_csv = open(os.path.join(args.out_dir, "summary.csv"),
                       "w", newline="", encoding="utf-8")
    sw = csv.writer(summary_csv)
    sw.writerow(["idx", "motion_id", "emotion", "emotion_id",
                 "caption_full", "caption_stripped", "T", "ok"])

    Ts = []
    n_ok = 0
    n_short = 0  # captions that generated very short sequences (T<5)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    t0 = time.time()
    for i, r in enumerate(rows):
        caption_full = r["caption"].strip()
        emotion = r["emotion"]
        emotion_id = EMOTION_TO_IDX[emotion]
        caption_stripped = strip_emotion_words(caption_full)

        try:
            bert_feature, bert_mask = text2bert(caption_full, tok, enc, ptu.device)
            with torch.no_grad():
                cur_embedding, idxs = gpt.sample(clip_feature, bert_feature, bert_mask)
            # tokens come out as a flat (T*depth, 1) tensor
            idxs_np = idxs.detach().cpu().numpy().reshape(-1)
            T = len(idxs_np) // depth
            idxs_np = idxs_np[:T * depth].reshape(T, depth).astype(np.int64)
            latents = cur_embedding.detach().cpu().numpy()[0].astype(np.float32)
            # We need at least T>=5 tokens and >=1 latent for teacher-forcing
            ok = T >= 5 and latents.shape[0] >= 1
            if not ok:
                n_short += 1
                sw.writerow([i, r.get("motion_id", ""), emotion, emotion_id,
                             caption_full, caption_stripped, T, 0])
                continue
        except Exception as e:
            print(f"  [skip] row {i} ({caption_full[:40]}...) -> {e}")
            sw.writerow([i, r.get("motion_id", ""), emotion, emotion_id,
                         caption_full, caption_stripped, 0, 0])
            continue

        g = h5.create_group(f"sample_{i:04d}")
        g.create_dataset("tokens", data=idxs_np)
        g.create_dataset("latents", data=latents)
        g.attrs["motion_id"] = r.get("motion_id", "")
        g.attrs["emotion"] = emotion
        g.attrs["emotion_id"] = emotion_id
        g.attrs["caption_full"] = caption_full
        g.attrs["caption_stripped"] = caption_stripped
        g.attrs["T"] = T
        n_ok += 1
        Ts.append(T)
        sw.writerow([i, r.get("motion_id", ""), emotion, emotion_id,
                     caption_full, caption_stripped, T, 1])

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            eta = (len(rows) - i - 1) / max(rate, 1e-3)
            print(f"  [{i+1:4d}/{len(rows)}] T={T:2d}  "
                  f"({rate:.2f} samples/s, ETA {eta/60:.1f} min)")

    grp_meta.attrs["n_samples"] = n_ok
    grp_meta.attrs["depth"] = depth
    grp_meta.attrs["max_length"] = args.max_length
    grp_meta.create_dataset("emotion_to_idx_keys",
        data=np.array(list(EMOTION_TO_IDX.keys()), dtype=h5py.string_dtype()))
    grp_meta.create_dataset("emotion_to_idx_vals",
        data=np.array(list(EMOTION_TO_IDX.values()), dtype=np.int64))
    h5.close()
    summary_csv.close()

    # --- final report ---------------------------------------------------
    elapsed = time.time() - t0
    print("\n" + "=" * 62)
    print(" Distillation dataset built")
    print("=" * 62)
    print(f"  successful samples : {n_ok} / {len(rows)}")
    print(f"  short / discarded  : {n_short}")
    if Ts:
        Ts = np.asarray(Ts)
        print(f"  sequence length T  : min {Ts.min()} / med {int(np.median(Ts))}"
              f" / max {Ts.max()} / mean {Ts.mean():.1f}")
    print(f"  elapsed            : {elapsed/60:.1f} min")
    print(f"\n[ok] h5 dataset -> {h5_path}")
    print(f"[ok] summary    -> {os.path.join(args.out_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
