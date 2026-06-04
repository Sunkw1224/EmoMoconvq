"""
test_emo_module.py -- sanity checks for the EmoMoconvq emotion module.

Pure-CPU structural test (no GPU / ODE / mpi needed).
Run from the MoConVQ-main directory:
    python Script/test_emo_module.py

Checks
------
  1. EmoText2Motion_Transformer constructs.
  2. It is a clean SUPERSET of the pretrained checkpoint: loading
     text_generation_GPT.pth with strict=False leaves *only* the new
     emotion parameters missing, and produces NO unexpected keys.
  3. configure_finetuning() freezes the pretrained model and unfreezes
     only the parameter-efficient subset.
  4. Zero-init: at step 0 emotion fusion is a no-op -- the temporal
     feature with / without an emotion label is numerically identical
     (so the model degrades gracefully to the baseline).
  5. After perturbing MLP_proj, the emotion label DOES change the output.
"""

import os
import torch

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


def build_model():
    cfg = {k: getattr(gpt_config, k)
           for k in vars(gpt_config) if not k.startswith("_")}
    # dummy RVQ codebook embeddings: 8 layers, (num_vq + 2) codes, embed_dim-d.
    # real values are overwritten when the pretrained checkpoint is loaded.
    embeddings = [torch.randn(cfg["num_vq"] + 2, cfg["embed_dim"])
                  for _ in range(8)]
    return EmoText2Motion_Transformer(embeddings=embeddings, **cfg)


def main():
    torch.manual_seed(0)
    print("=" * 62)
    print(" EmoMoconvq emotion module -- sanity test")
    print("=" * 62)

    # --- 1. construction -------------------------------------------------
    model = build_model().eval()
    print("[1] EmoText2Motion_Transformer constructed OK")

    # --- 2. checkpoint compatibility ------------------------------------
    ckpt_path = "text_generation_GPT.pth"
    if os.path.isfile(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
        # the checkpoint was saved from a DataParallel wrapper -> strip prefix
        sd = {(k[7:] if k.startswith("module.") else k): v
              for k, v in sd.items()}
        result = model.load_state_dict(sd, strict=False)
        missing = list(result.missing_keys)
        unexpected = list(result.unexpected_keys)
        emo_missing = [k for k in missing if "emotion" in k]
        non_emo_missing = [k for k in missing if "emotion" not in k]
        print("[2] loaded pretrained checkpoint with strict=False")
        print(f"    missing emotion params (expected) : {len(emo_missing)} -> "
              f"{emo_missing}")
        print(f"    missing NON-emotion params (BAD)  : {len(non_emo_missing)}")
        print(f"    unexpected keys (BAD)             : {len(unexpected)}")
        assert not non_emo_missing, non_emo_missing[:10]
        assert not unexpected, unexpected[:10]
        print("    OK: emotion module is a clean superset of the baseline")
    else:
        print(f"[2] SKIP -- {ckpt_path} not found in current directory")

    # --- 3. parameter-efficient fine-tuning setup ----------------------
    stats = model.configure_finetuning(num_temporal_layers_to_tune=4)
    assert 0 < stats["trainable"] < stats["total"]
    assert stats["emotion"] > 0
    print("[3] configure_finetuning() OK")

    # --- 4 & 5. emotion-fusion behaviour -------------------------------
    B, L = 1, 12
    clip_feature = torch.zeros(B, gpt_config.clip_dim)
    bert_feature = torch.randn(B, L, 1024)
    bert_mask = torch.zeros(B, L, dtype=torch.bool)
    emo = model.trans_temporal

    with torch.no_grad():
        emo.current_emotion = None
        out_none = emo([], clip_feature, bert_feature, bert_mask)
        emo.current_emotion = torch.tensor([EMOTION_TO_IDX["happy"]])
        out_happy0 = emo([], clip_feature, bert_feature, bert_mask)
        diff0 = (out_none - out_happy0).abs().max().item()
    print(f"[4] zero-init  : max|out(no-emo) - out(happy)| = {diff0:.2e} "
          f"(expect ~0)")
    assert diff0 < 1e-5, "emotion fusion is not a no-op at init!"

    with torch.no_grad():
        # perturb MLP_proj's last layer to mimic a fine-tuned state
        for p in emo.emotion_proj[-1].parameters():
            p.normal_(0.0, 0.5)
        emo.current_emotion = torch.tensor([EMOTION_TO_IDX["happy"]])
        out_happy = emo([], clip_feature, bert_feature, bert_mask)
        emo.current_emotion = torch.tensor([EMOTION_TO_IDX["sad"]])
        out_sad = emo([], clip_feature, bert_feature, bert_mask)
        diff_hs = (out_happy - out_sad).abs().max().item()
    emo.current_emotion = None
    print(f"[5] trained-like: max|out(happy) - out(sad)|   = {diff_hs:.3e} "
          f"(expect > 0)")
    assert diff_hs > 0, "emotion label has no effect after training!"

    print("=" * 62)
    print(" ALL CHECKS PASSED")
    print("=" * 62)


if __name__ == "__main__":
    main()
