"""
gradio_demo.py
===============
Phase H: interactive web demo of the EmoMoconvq emotion-conditioned model.

UI:
  * Text box  : motion description (e.g. "a person walks forward")
  * Dropdown  : emotion label (neutral / happy / fearful)
  * Seed      : random seed
  * Button    : generate -> physics simulation -> BVH download

After launch the demo is available at http://127.0.0.1:7860.

Run from the MoConVQ-main directory:
    python Script/gradio_demo.py
Optional:
    --checkpoint PATH   tuned model checkpoint
    --port N            listen port (default 7860)
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import torch

import MoConVQCore.Utils.pytorch_utils as ptu
from MoConVQCore.Model.emo_cross_trans import (
    EmoText2Motion_Transformer, EMOTION_TO_IDX)


EMO_LIST_ALL = ["neutral", "happy", "sad", "angry", "fearful"]


class gpt_config:
    num_vq = 512; embed_dim = 768; clip_dim = 512
    block_size = 52; num_layers = 9; n_head = 8
    drop_out_rate = 0.1; fc_rate = 2


# globals -- loaded once on first request
_state = {"loaded": False}


def load_all(checkpoint_path, keep_emotions, device):
    if _state.get("loaded"):
        return _state

    print("[load] building MoConVQ agent (slow first time) ...")
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
    emo = EmoText2Motion_Transformer(embeddings=embed_torch, **cfg).to(device)

    sd = torch.load("text_generation_GPT.pth", map_location=device)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    emo.load_state_dict(sd, strict=False)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        tunable = torch.load(checkpoint_path, map_location=device)
        emo.load_state_dict(tunable, strict=False)
        print(f"[load] loaded fine-tuned checkpoint: {len(tunable)} tensors")
    else:
        print(f"[warn] no fine-tuned checkpoint at {checkpoint_path}; "
              f"running pretrained baseline + zero-init emotion module")
    emo.eval()

    from transformers import T5Tokenizer, T5EncoderModel
    print("[load] loading t5-large ...")
    tok = T5Tokenizer.from_pretrained("t5-large")
    enc = T5EncoderModel.from_pretrained("t5-large").to(device).eval()

    _state.update({
        "loaded": True,
        "agent": agent, "env": env, "emo": emo,
        "tok": tok, "enc": enc,
        "clip_feature": torch.zeros(1, cfg["clip_dim"], device=device),
        "device": device,
        "keep_emotions": keep_emotions,
    })
    print("[load] ready.")
    return _state


def generate_motion(text, emotion, seed, progress=None):
    """Run text+emotion -> BVH file path."""
    st = _state
    if not st.get("loaded"):
        return None, "Model still loading; please wait and retry."
    device = st["device"]
    agent = st["agent"]
    emo_model = st["emo"]
    tok, enc = st["tok"], st["enc"]
    clip_feature = st["clip_feature"]

    if not text or not text.strip():
        return None, "Please enter a motion description."
    if emotion not in EMOTION_TO_IDX:
        return None, f"Unknown emotion: {emotion}"

    if progress is not None:
        progress(0.05, desc="encoding text")
    enc_in = tok(text, return_tensors="pt", padding=True,
                 truncation=True, max_length=256)
    enc_in = {k: v.to(device) for k, v in enc_in.items()}
    with torch.no_grad():
        r = enc(**enc_in)
    bf = r.last_hidden_state
    bm = ~enc_in["attention_mask"].bool()

    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    if progress is not None:
        progress(0.20, desc="sampling motion tokens")
    emo_id = torch.tensor([EMOTION_TO_IDX[emotion]], device=device)
    with torch.no_grad():
        cur_embedding, _ = emo_model.sample(clip_feature, bf, bm,
                                            emotion=emo_id)

    if progress is not None:
        progress(0.35, desc="decoding -> physics simulation")
    dconv = agent.posterior.decoder.decode_dynamic(cur_embedding)

    import VclSimuBackend
    CharacterToBVH = VclSimuBackend.ODESim.CharacterTOBVH
    saver = CharacterToBVH(agent.env.sim_character, 120)
    saver.bvh_hierarchy_no_root()

    observation, _ = agent.env.reset(0)
    total_steps = int(dconv.shape[1])
    for i in range(total_steps):
        obs = observation["observation"]
        action, _ = agent.act_tracking(
            obs_history=[obs.reshape(1, 323)],
            target_latent=dconv[:, i])
        action = ptu.to_numpy(action).flatten()
        for k in range(6):
            saver.append_no_root_to_buffer()
            if k == 0:
                step_gen = agent.env.step_core(action, using_yield=True)
            _ = next(step_gen)
        try:
            info_ = next(step_gen)
        except StopIteration as e:
            info_ = e.value
        observation = info_[0]
        if progress is not None and i % 4 == 0:
            progress(0.35 + 0.60 * (i + 1) / total_steps,
                     desc=f"physics step {i+1}/{total_steps}")

    out_path = tempfile.NamedTemporaryFile(suffix=".bvh", delete=False).name
    saver.to_file(out_path)
    if progress is not None:
        progress(1.0, desc="done")
    msg = (f"Generated {total_steps} latents (~{total_steps*6/120:.1f}s of motion).\n"
           f"Text: '{text}'   Emotion: {emotion}   Seed: {seed}")
    return out_path, msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(
        "out", "finetune_full_v2_3class", "emomoconvq_best.pt"))
    parser.add_argument("--keep-emotions", default="neutral,happy,fearful")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--server", default="127.0.0.1")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]    # hide our flags from build_agent

    device = "cuda" if torch.cuda.is_available() else "cpu"
    keep_set = {e.strip() for e in args.keep_emotions.split(",") if e.strip()}
    emo_choices = [e for e in EMO_LIST_ALL if e in keep_set]
    if not emo_choices:
        emo_choices = list(EMO_LIST_ALL)

    # eager-load so first request is fast
    load_all(args.checkpoint, args.keep_emotions, device)

    import gradio as gr

    with gr.Blocks(title="EmoMoconvq Demo",
                   theme=gr.themes.Soft()) as demo:
        gr.Markdown("# EmoMoconvq")
        gr.Markdown(
            "**Emotion-conditioned physics-based motion generation.**  \n"
            "Describe a motion in text and pick an emotion label. The model "
            "generates RVQ tokens via the fine-tuned T2M-MoConGPT, then "
            "decodes them through MoConVQ's physics simulator. The output "
            "is a BVH file you can drop into Blender / Maya / Three.js.")
        with gr.Row():
            with gr.Column(scale=2):
                text_in = gr.Textbox(
                    value="a person walks forward",
                    label="Motion description",
                    lines=2)
                emo_in = gr.Dropdown(
                    emo_choices, value=emo_choices[0],
                    label="Emotion label")
                seed_in = gr.Number(value=0, precision=0,
                                    label="Random seed")
                with gr.Row():
                    gen_btn = gr.Button("Generate", variant="primary")
                    clear_btn = gr.Button("Reset", variant="secondary")
                gr.Examples(
                    examples=[
                        ["a person walks forward",       "happy",   0],
                        ["a person walks forward",       "fearful", 0],
                        ["a person walks forward",       "neutral", 0],
                        ["a person sits down on a chair","happy",   1],
                        ["a person jumps and turns",     "fearful", 2],
                    ],
                    inputs=[text_in, emo_in, seed_in])
            with gr.Column(scale=3):
                file_out = gr.File(
                    label="Generated motion (BVH)",
                    file_count="single")
                info_out = gr.Textbox(label="Log", lines=3, max_lines=8)

        def _gen(t, e, s, progress=gr.Progress()):
            path, msg = generate_motion(t, e, s, progress)
            return path, msg

        gen_btn.click(_gen,
                      inputs=[text_in, emo_in, seed_in],
                      outputs=[file_out, info_out])
        clear_btn.click(lambda: ("a person walks forward",
                                  emo_choices[0], 0, None, ""),
                        inputs=[],
                        outputs=[text_in, emo_in, seed_in,
                                  file_out, info_out])

    demo.launch(server_name=args.server, server_port=args.port,
                share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
