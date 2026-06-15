"""
investigate_baseline_emotion.py
================================
Quick pre-check for the distillation data path:
does the pretrained baseline T2M-MoConGPT produce DIFFERENT motion tokens
for different emotion variants of the same base caption?

If YES -> distillation is viable: generate (text, emotion, tokens) triples
         from the baseline for the 463-pair emotion corpus.
If NO  -> the baseline ignores emotion words and we need a different
         data strategy.

We sample tokens for 5 emotion variants of "a person walks":
    neutral / happy / sad / angry / fearful
and report token agreement (per-position match rate over all RVQ layers).

Pure pre-check: skips physics decoding, only inspects tokens (fast).

Run from MoConVQ-main:
    python Script/investigate_baseline_emotion.py
"""

import os
import numpy as np
import torch

import MoConVQCore.Utils.pytorch_utils as ptu
from MoConVQCore.Model.cross_trans_ori_fixsum import Text2Motion_Transformer


CAPTIONS = [
    ("neutral", "a person walks"),
    ("happy",   "a person walks happily"),
    ("sad",     "a person walks sadly"),
    ("angry",   "a person walks angrily"),
    ("fearful", "a person walks fearfully"),
]

SEED = 0  # fix seed so any difference is from the input, not sampling noise


class gpt_config:
    num_vq = 512; embed_dim = 768; clip_dim = 512
    block_size = 52; num_layers = 9; n_head = 8
    drop_out_rate = 0.1; fc_rate = 2


def text2bert(text, tok, enc, device):
    enc_in = tok(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
    enc_in = {k: v.to(device) for k, v in enc_in.items()}
    with torch.no_grad():
        out = enc(**enc_in)
    return out.last_hidden_state, ~enc_in['attention_mask'].bool()


def main():
    out_dir = os.path.join("out", "investigate_emotion")
    os.makedirs(out_dir, exist_ok=True)

    from Script.moconvq_builder import build_agent
    device = 0
    agent, env = build_agent(gpu=device)
    ptu.init_gpu(True, gpu_id=device)
    agent.simple_load(r'moconvq_base.data', strict=True)
    agent.eval()

    # build the text-conditioned GPT and load pretrained weights
    embed_torch = [
        torch.cat([bn.embedding, torch.zeros_like(bn.embedding[:2])], dim=0)
        for bn in agent.posterior.bottle_neck_list
    ]
    cfg = {k: getattr(gpt_config, k) for k in vars(gpt_config) if not k.startswith('_')}
    gpt = Text2Motion_Transformer(embeddings=embed_torch, **cfg).to(ptu.device)
    sd = torch.load('text_generation_GPT.pth', map_location=ptu.device)
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    # checkpoint has DataParallel-wrapped extra 'module.' from the original
    # text2motion training code; strip once more if needed
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    gpt.load_state_dict(sd, strict=True)
    gpt.eval()

    from transformers import T5Tokenizer, T5EncoderModel
    print("[t5] loading t5-large ...")
    tok = T5Tokenizer.from_pretrained('t5-large')
    enc = T5EncoderModel.from_pretrained('t5-large').to(ptu.device).eval()

    clip_feature = torch.zeros((1, 512), device=ptu.device)

    results = {}
    for label, caption in CAPTIONS:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        bert_feature, bert_mask = text2bert(caption, tok, enc, ptu.device)
        print(f"[gen] '{caption}' ...")
        with torch.no_grad():
            cur_embedding, idxs = gpt.sample(clip_feature, bert_feature, bert_mask)
        # idxs is (T*depth, 1) per the sample loop -- reshape to (T, depth)
        idxs_np = idxs.detach().cpu().numpy().reshape(-1)
        # determine depth: max_depth = 4 in the model
        depth = 4
        T = len(idxs_np) // depth
        idxs_np = idxs_np[:T * depth].reshape(T, depth)
        results[label] = {
            "caption": caption,
            "tokens": idxs_np,
            "n_steps": T,
            "latents": cur_embedding.detach().cpu().numpy()[0],  # (T, 768)
        }
        print(f"      -> {idxs_np.shape} tokens, {results[label]['latents'].shape} latents")

    # ----- comparison report --------------------------------------------
    print("\n" + "=" * 64)
    print(" Baseline emotion-sensitivity check  (seed fixed)")
    print("=" * 64)
    print("\nToken sequence length per condition:")
    for lab in results:
        print(f"  {lab:8s}  T = {results[lab]['n_steps']}")

    print("\nPer-pair per-position TOKEN match rate (1.0 = identical):")
    labels = list(results.keys())
    n = len(labels)
    M = np.full((n, n), -1.0)
    for i in range(n):
        for j in range(n):
            a = results[labels[i]]["tokens"]
            b = results[labels[j]]["tokens"]
            T = min(len(a), len(b))
            if T == 0:
                continue
            same = float((a[:T] == b[:T]).all(axis=1).mean())
            M[i, j] = same
    print(f"  {'':8s} " + " ".join(f"{l:>8s}" for l in labels))
    for i, lab in enumerate(labels):
        row = " ".join(f"{M[i,j]:>8.3f}" for j in range(n))
        print(f"  {lab:8s} {row}")

    print("\nLatent L2 distance (mean over time, normalized by feature std):")
    for i in range(n):
        for j in range(i + 1, n):
            a = results[labels[i]]["latents"]
            b = results[labels[j]]["latents"]
            T = min(len(a), len(b))
            d = float(np.linalg.norm(a[:T] - b[:T], axis=-1).mean())
            print(f"  {labels[i]:>8s} vs {labels[j]:<8s}  L2 = {d:.4f}")

    # save raw for downstream
    np.savez(os.path.join(out_dir, "baseline_emotion_tokens.npz"),
             **{f"tokens_{l}": results[l]["tokens"] for l in labels},
             **{f"latents_{l}": results[l]["latents"] for l in labels},
             captions=np.array([results[l]["caption"] for l in labels]),
             labels=np.array(labels))
    print(f"\n[ok] saved -> {os.path.join(out_dir, 'baseline_emotion_tokens.npz')}")


if __name__ == "__main__":
    main()
