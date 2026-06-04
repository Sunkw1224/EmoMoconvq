"""
finetune_emomoconvq.py
======================
Initial fine-tuning experiment for EmoMoconvq.

The official MoConVQ release does NOT ship MoConGPT training code, so the
negative-log-likelihood (NLL) fine-tuning loop is implemented here, following
the teacher-forcing structure of Text2Motion_Transformer.forward and the
proposal's Eq. (3).

Because the HumanML3D -> MoConVQ motion-tokenisation pipeline is not yet built
(the project's main remaining work), this script runs a *controlled
small-scale overfitting experiment*: a handful of motion clips taken from
`walk1_subject5` are paired with emotion-labelled captions, and the
emotion-conditioned model is fine-tuned on them. A decreasing NLL curve and
non-zero emotion-module gradients confirm that the EmoMoconvq module and the
fine-tuning pipeline are functional end-to-end.

Prerequisite:  python Script/build_emo_finetune_data.py   (creates the .npz)

Run from the MoConVQ-main directory:
    python Script/finetune_emomoconvq.py
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F

from MoConVQCore.Model.emo_cross_trans import (
    EmoText2Motion_Transformer, EMOTION_TO_IDX)


class gpt_config:
    """Same configuration as Script/text2motion_generation.py."""
    num_vq = 512
    embed_dim = 768
    clip_dim = 512
    block_size = 52
    num_layers = 9
    n_head = 8
    drop_out_rate = 0.1
    fc_rate = 2


# 5 clips: constant action, one emotion each. The captions carry the emotion
# adverb (as in the real EmoMoconvq use-case) and the discrete label is fed
# through the emotion-embedding module.
CLIP_SPEC = [
    ("a person walks",            "neutral"),
    ("a person walks happily",    "happy"),
    ("a person walks sadly",      "sad"),
    ("a person walks angrily",    "angry"),
    ("a person walks fearfully",  "fearful"),
]

CLIP_LEN = 50          # latent timesteps per clip (<= block_size)
DEPTH = 4              # RVQ layers used by MoConGPT (max_depth)
N_STEPS = 60
LR = 1e-4


def text2bert(text, tokenizer, encoder, device):
    enc = tokenizer(text, return_tensors="pt", padding=True,
                    truncation=True, max_length=256)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = encoder(**enc)
    return out.last_hidden_state, ~enc["attention_mask"].bool()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print(f"[setup] device = {device}")
    out_dir = os.path.join("out", "emo_finetune")
    os.makedirs(out_dir, exist_ok=True)

    # --- load the tokenised motion --------------------------------------
    npz_path = os.path.join(out_dir, "walk1_subject5_tokens.npz")
    if not os.path.isfile(npz_path):
        raise SystemExit(f"[ERROR] {npz_path} not found -- run "
                         f"Script/build_emo_finetune_data.py first")
    data = np.load(npz_path)
    idxs_full = data["indexs"][:DEPTH].T          # (L, DEPTH)
    latent_full = data["latent_vq"][0]            # (L, 768)
    L = idxs_full.shape[0]
    print(f"[data] motion length = {L} latents; using {len(CLIP_SPEC)} "
          f"clips of {CLIP_LEN}")

    # --- build the emotion-conditioned GPT ------------------------------
    cfg = {k: getattr(gpt_config, k)
           for k in vars(gpt_config) if not k.startswith("_")}
    dummy_emb = [torch.zeros(cfg["num_vq"] + 2, cfg["embed_dim"])
                 for _ in range(8)]              # overwritten by checkpoint
    model = EmoText2Motion_Transformer(embeddings=dummy_emb, **cfg)

    sd = torch.load("text_generation_GPT.pth", map_location="cpu")
    sd = {(k[7:] if k.startswith("module.") else k): v
          for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    assert not [k for k in res.missing_keys if "emotion" not in k]
    print(f"[model] loaded pretrained GPT (missing only "
          f"{len(res.missing_keys)} emotion params)")

    model = model.to(device).eval()              # eval() -> no dropout noise
    model.configure_finetuning(num_temporal_layers_to_tune=4)

    # --- T5 text features (computed once, then T5 is freed) -------------
    from transformers import T5Tokenizer, T5EncoderModel
    print("[t5] loading t5-large ...")
    tokenizer = T5Tokenizer.from_pretrained("t5-large")
    encoder = T5EncoderModel.from_pretrained("t5-large").to(device).eval()

    clips = []
    for i, (caption, emotion) in enumerate(CLIP_SPEC):
        o = i * CLIP_LEN
        idxs_c = idxs_full[o:o + CLIP_LEN]                  # (S, DEPTH)
        lat_c = latent_full[o:o + CLIP_LEN - 1]            # (S-1, 768)
        bert_feature, bert_mask = text2bert(caption, tokenizer, encoder, device)
        clips.append({
            "caption": caption, "emotion": emotion,
            "latents": torch.tensor(lat_c, dtype=torch.float32,
                                    device=device).unsqueeze(0),
            "idxs": torch.tensor(idxs_c, dtype=torch.long,
                                 device=device).unsqueeze(0),
            "bert_feature": bert_feature, "bert_mask": bert_mask,
            "emotion_id": torch.tensor([EMOTION_TO_IDX[emotion]],
                                       device=device),
        })
    del encoder
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"[t5] extracted features for {len(clips)} captions; T5 freed")

    clip_feature = torch.zeros(1, cfg["clip_dim"], device=device)
    emo_params = list(model.trans_temporal.emotion_embedding.parameters()) + \
        list(model.trans_temporal.emotion_proj.parameters())
    emo_w0 = model.trans_temporal.emotion_embedding.weight.detach().clone()

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR)

    # --- fine-tuning loop ----------------------------------------------
    print(f"[train] {N_STEPS} steps, lr={LR}")
    loss_curve, gradnorm_curve = [], []
    for step in range(N_STEPS):
        step_losses, step_grad = [], []
        for clip in clips:
            logits, _ = model(clip["latents"], clip["idxs"], clip_feature,
                              clip["bert_feature"], clip["bert_mask"],
                              emotion=clip["emotion_id"])
            # logits: (1, S, DEPTH+1, num_vq+1) -> keep the DEPTH real layers
            pred = logits[:, :, :DEPTH, :]
            loss = F.cross_entropy(pred.reshape(-1, pred.shape[-1]),
                                   clip["idxs"].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            gnorm = torch.norm(torch.stack(
                [p.grad.norm() for p in emo_params if p.grad is not None]))
            optimizer.step()
            step_losses.append(loss.item())
            step_grad.append(gnorm.item())
        loss_curve.append(float(np.mean(step_losses)))
        gradnorm_curve.append(float(np.mean(step_grad)))
        if step % 25 == 0 or step == N_STEPS - 1:
            print(f"  step {step:4d}  NLL = {loss_curve[-1]:.4f}  "
                  f"emo-grad = {gradnorm_curve[-1]:.4f}")

    # --- emotion-swap diagnostic ---------------------------------------
    # after training, measure NLL with the CORRECT vs a WRONG emotion label.
    print("\n[diagnostic] emotion-swap test (lower NLL = better fit)")
    with torch.no_grad():
        correct, swapped = [], []
        for i, clip in enumerate(clips):
            wrong_id = torch.tensor(
                [EMOTION_TO_IDX[CLIP_SPEC[(i + 1) % len(CLIP_SPEC)][1]]],
                device=device)
            for emo_id, bucket in [(clip["emotion_id"], correct),
                                   (wrong_id, swapped)]:
                logits, _ = model(clip["latents"], clip["idxs"], clip_feature,
                                  clip["bert_feature"], clip["bert_mask"],
                                  emotion=emo_id)
                pred = logits[:, :, :DEPTH, :]
                nll = F.cross_entropy(pred.reshape(-1, pred.shape[-1]),
                                      clip["idxs"].reshape(-1))
                bucket.append(nll.item())
            print(f"  {clip['emotion']:8s}  correct={correct[-1]:.4f}  "
                  f"wrong-label={swapped[-1]:.4f}")

    emo_w1 = model.trans_temporal.emotion_embedding.weight.detach()
    emo_shift = (emo_w1 - emo_w0).norm().item()

    # --- save results ---------------------------------------------------
    summary = {
        "n_steps": N_STEPS, "lr": LR, "n_clips": len(clips),
        "clip_len": CLIP_LEN, "depth": DEPTH,
        "nll_start": loss_curve[0], "nll_end": loss_curve[-1],
        "emotion_embedding_shift": emo_shift,
        "diag_correct_mean": float(np.mean(correct)),
        "diag_wronglabel_mean": float(np.mean(swapped)),
        "loss_curve": loss_curve, "gradnorm_curve": gradnorm_curve,
    }
    with open(os.path.join(out_dir, "finetune_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[result] NLL  {loss_curve[0]:.4f} -> {loss_curve[-1]:.4f}")
    print(f"[result] emotion-embedding moved by L2 = {emo_shift:.4f}")
    print(f"[result] emotion-swap: correct {np.mean(correct):.4f} "
          f"vs wrong-label {np.mean(swapped):.4f}")

    # --- plot -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(6, 3.6))
        ax1.plot(loss_curve, color="#d9534f", label="NLL loss")
        ax1.set_xlabel("fine-tuning step")
        ax1.set_ylabel("NLL loss", color="#d9534f")
        ax1.tick_params(axis="y", labelcolor="#d9534f")
        ax2 = ax1.twinx()
        ax2.plot(gradnorm_curve, color="#4a90d9", alpha=0.7,
                 label="emotion-module grad norm")
        ax2.set_ylabel("emotion-module grad norm", color="#4a90d9")
        ax2.tick_params(axis="y", labelcolor="#4a90d9")
        ax1.set_title("EmoMoconvq fine-tuning: NLL loss curve")
        fig.tight_layout()
        fig_path = os.path.join(out_dir, "finetune_loss_curve.png")
        fig.savefig(fig_path, dpi=200)
        print(f"[ok] loss curve -> {fig_path}")
    except ImportError:
        print("[warn] matplotlib unavailable -- skipped plot")

    print(f"[ok] summary    -> {os.path.join(out_dir, 'finetune_summary.json')}")


if __name__ == "__main__":
    main()
