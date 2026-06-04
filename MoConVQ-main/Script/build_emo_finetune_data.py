"""
build_emo_finetune_data.py
==========================
Encode an available MoConVQ-format motion into RVQ token sequences, so the
EmoMoconvq fine-tuning pipeline can be validated end-to-end.

Background
----------
A proper emotion fine-tuning set needs (text, emotion, motion-RVQ-tokens)
triples. We have (text, emotion) from build_emotion_dataset.py, but the
HumanML3D motions are SMPL-based and cannot be fed to the MoConVQ encoder
without a retargeting pipeline (this is the project's main remaining work).

To still validate the fine-tuning code + emotion module, this script encodes
the one motion shipped with the pretrained data (`simple_motion_data.h5` ->
`walk1_subject5`, a CMU walking clip) into RVQ tokens and quantised latents.
The resulting tokens are later chunked + paired with emotion-labelled
captions for a controlled overfitting experiment.

Run from the MoConVQ-main directory:
    python Script/build_emo_finetune_data.py
"""

import os
import numpy as np
import torch
import h5py

import tokenize_motion as tk            # reuse build_args / get_model
import MoConVQCore.Utils.pytorch_utils as ptu


def main():
    out_dir = os.path.join("out", "emo_finetune")
    os.makedirs(out_dir, exist_ok=True)

    # --- build the pretrained MoConVQ agent ------------------------------
    model_args = tk.build_args(args_in=[])
    ptu.init_gpu(True, gpu_id=0)
    agent, env = tk.get_model(model_args)

    # --- load the one MoConVQ-format motion we have ----------------------
    with h5py.File("simple_motion_data.h5", "r") as f:
        obs = f["walk1_subject5/observation"][:]      # (T_frames, 323)
    print(f"[data] walk1_subject5 observation: {obs.shape} {obs.dtype}")

    # --- encode -> RVQ tokens + quantised latents ------------------------
    info = agent.encode_seq_all(None, np.asarray(obs))
    print("[encode] info keys and shapes:")
    for k, v in info.items():
        if torch.is_tensor(v):
            print(f"    {k:16s} {tuple(v.shape)}  {v.dtype}")
        else:
            print(f"    {k:16s} {type(v)}")

    idxs = info["indexs"]            # RVQ code indices
    latent_vq = info["latent_vq"]    # quantised latent (GPT operates here)

    idxs = idxs.detach().cpu().numpy()
    latent_vq = latent_vq.detach().cpu().numpy()
    print(f"[encode] indexs   -> {idxs.shape}")
    print(f"[encode] latent_vq-> {latent_vq.shape}")

    out_path = os.path.join(out_dir, "walk1_subject5_tokens.npz")
    np.savez_compressed(out_path, indexs=idxs, latent_vq=latent_vq)
    print(f"[ok] saved -> {out_path}")


if __name__ == "__main__":
    main()
