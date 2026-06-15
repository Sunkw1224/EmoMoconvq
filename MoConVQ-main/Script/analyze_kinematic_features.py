"""
analyze_kinematic_features.py
==============================
Phase D: kinematic feature analysis of EmoMoconvq's emotion-conditioned outputs.

We bypass the learned token classifier (which can only score patterns that
already exist in the distillation targets) and ask the more direct question:

    Does the discrete emotion label produce statistically different motion
    KINEMATICS across the test captions?

Six features from the affective-motion literature are computed from the
physics-simulated character trajectory:

  1. vertical_energy : variance of root height (high = bouncy/happy)
  2. avg_height      : mean root height (low = slumped/sad)
  3. avg_speed       : mean horizontal root velocity
  4. step_freq       : foot-strike rate per second (fast = angry/happy)
  5. lean_angle      : mean trunk angle from vertical (forward = angry)
  6. arm_swing       : std of hand/wrist excursion (large = happy)

We run a one-way ANOVA across the active emotion classes for each feature
and report effect sizes (eta^2) and p-values.

As a side product, every generated motion is saved as a BVH file, which
Phase G (video rendering) reuses directly.

Run from the MoConVQ-main directory:
    python Script/analyze_kinematic_features.py \\
        --checkpoint out/finetune_full_v2_3class/emomoconvq_best.pt \\
        --keep-emotions neutral,happy,fearful
"""

import argparse
import csv
import json
import os
import re
import sys
import time

import numpy as np
import torch

import MoConVQCore.Utils.pytorch_utils as ptu
from MoConVQCore.Model.emo_cross_trans import (
    EmoText2Motion_Transformer, EMOTION_TO_IDX)


EMO_LIST_ALL = ["neutral", "happy", "sad", "angry", "fearful"]


# Emotion lexicon (used to exclude HumanML3D captions that already carry
# emotion words from the test pool).
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


class gpt_config:
    num_vq = 512; embed_dim = 768; clip_dim = 512
    block_size = 52; num_layers = 9; n_head = 8
    drop_out_rate = 0.1; fc_rate = 2


# ============================================================
# Test caption pool (same construction as eval_compositional.py)
# ============================================================
def sample_neutral_captions(texts_dir, split_file, exclude_ids,
                            n_captions, seed=0):
    with open(split_file, "r", encoding="utf-8") as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    rng = np.random.RandomState(seed); rng.shuffle(ids)
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
                if not cap or _EMO_RE.search(cap):
                    continue
                out.append((mid, cap)); break
        if len(out) >= n_captions:
            break
    return out[:n_captions]


# ============================================================
# Run physics simulation, collect per-frame body positions
# ============================================================
def simulate_and_capture(agent, env, cur_embedding):
    """Decode embedding -> physics step -> return (positions, saver_for_bvh)."""
    dconv = agent.posterior.decoder.decode_dynamic(cur_embedding)

    import VclSimuBackend
    CharacterToBVH = VclSimuBackend.ODESim.CharacterTOBVH
    saver = CharacterToBVH(agent.env.sim_character, 120)
    saver.bvh_hierarchy_no_root()

    observation, _ = agent.env.reset(0)

    pos_traj = []     # list of (n_bodies, 3) per simulation frame

    for i in range(dconv.shape[1]):
        obs = observation["observation"]
        action, _ = agent.act_tracking(
            obs_history=[obs.reshape(1, 323)],
            target_latent=dconv[:, i])
        action = ptu.to_numpy(action).flatten()
        step_generator = None
        for k in range(6):
            saver.append_no_root_to_buffer()
            if k == 0:
                step_generator = agent.env.step_core(action, using_yield=True)
            _ = next(step_generator)
            # capture body positions every physics substep
            try:
                bp = agent.env.sim_character.body_info.get_body_pos()
                pos_traj.append(np.asarray(bp).copy())
            except Exception:
                # fallback: try alternative accessor
                bp = agent.env.sim_character.body_info.body_pos
                pos_traj.append(np.asarray(bp).copy())
        try:
            info_ = next(step_generator)
        except StopIteration as e:
            info_ = e.value
        new_observation, _, _, _ = info_
        observation = new_observation

    pos_traj = np.stack(pos_traj, axis=0)        # (T, n_bodies, 3)
    return pos_traj, saver


# ============================================================
# Body-index helpers   (MoConVQ skeleton joint layout)
# ============================================================
# These indices are the order in which bodies are added by the ODE
# character builder; the first body is the root (pelvis-area), then
# they follow the BVH hierarchy of base.bvh:
#   0  root / pelvis
#   1  pelvis_lowerback
#   2  lowerback_torso
#   3  torso_head        <- head
#   4  rTorso_Clavicle
#   5  rShoulder
#   6  rElbow
#   7  rWrist            <- right hand
#   8  lTorso_Clavicle
#   9  lShoulder
#   10 lElbow
#   11 lWrist            <- left hand
#   12 rHip
#   13 rKnee
#   14 rAnkle
#   15 rToeJoint         <- right foot
#   16 lHip
#   17 lKnee
#   18 lAnkle
#   19 lToeJoint         <- left foot
ROOT, HEAD, RWRIST, LWRIST, RFOOT, LFOOT = 0, 3, 7, 11, 15, 19


def compute_features(pos_traj, fps=120 * 1):
    """pos_traj: (T, n_bodies, 3). Y is up.  Returns dict of 6 features."""
    n_frames = pos_traj.shape[0]
    if n_frames < 4:
        return None

    n_bodies = pos_traj.shape[1]
    head_idx = HEAD if HEAD < n_bodies else 0
    rwr_idx  = RWRIST if RWRIST < n_bodies else 0
    lwr_idx  = LWRIST if LWRIST < n_bodies else 0
    rft_idx  = RFOOT  if RFOOT  < n_bodies else 0
    lft_idx  = LFOOT  if LFOOT  < n_bodies else 0

    root = pos_traj[:, ROOT, :]           # (T, 3)
    head = pos_traj[:, head_idx, :]
    rwr  = pos_traj[:, rwr_idx, :]
    lwr  = pos_traj[:, lwr_idx, :]
    rft  = pos_traj[:, rft_idx, :]
    lft  = pos_traj[:, lft_idx, :]

    # ---- 1. vertical_energy : variance of root height (Y axis) ----------
    vertical_energy = float(np.var(root[:, 1]))

    # ---- 2. avg_height : mean root height -------------------------------
    avg_height = float(np.mean(root[:, 1]))

    # ---- 3. avg_speed : mean horizontal speed of root (XZ plane) -------
    horiz = root[:, [0, 2]]
    dt = 1.0 / fps
    speed = np.linalg.norm(np.diff(horiz, axis=0), axis=1) / dt
    avg_speed = float(np.mean(speed))

    # ---- 4. step_freq : foot-strike rate --------------------------------
    # detect a strike as a local minimum of foot Y-height
    try:
        from scipy.signal import find_peaks
        avg_foot_y = 0.5 * (rft[:, 1] + lft[:, 1])
        peaks, _ = find_peaks(-avg_foot_y, distance=max(2, fps // 8))
        duration = n_frames / fps
        step_freq = float(len(peaks) / max(duration, 1e-3))
    except Exception:
        step_freq = float("nan")

    # ---- 5. lean_angle : mean trunk angle from vertical (deg) ----------
    spine = head - root           # (T, 3)
    spine_norm = np.linalg.norm(spine, axis=1) + 1e-8
    cos_up = np.clip(spine[:, 1] / spine_norm, -1.0, 1.0)
    lean_rad = np.arccos(cos_up)
    lean_angle = float(np.degrees(np.mean(lean_rad)))

    # ---- 6. arm_swing : range of hand position (mean over both arms) ---
    rwr_range = float(np.linalg.norm(rwr.max(axis=0) - rwr.min(axis=0)))
    lwr_range = float(np.linalg.norm(lwr.max(axis=0) - lwr.min(axis=0)))
    arm_swing = 0.5 * (rwr_range + lwr_range)

    return {
        "vertical_energy": vertical_energy,
        "avg_height":      avg_height,
        "avg_speed":       avg_speed,
        "step_freq":       step_freq,
        "lean_angle":      lean_angle,
        "arm_swing":       arm_swing,
    }


# ============================================================
# Main
# ============================================================
def text2bert(text, tok, enc, device):
    e = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    e = {k: v.to(device) for k, v in e.items()}
    with torch.no_grad():
        r = enc(**e)
    return r.last_hidden_state.detach(), (~e["attention_mask"].bool()).detach()


def build_models(device, checkpoint_path):
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
        print(f"[setup] loaded fine-tuned checkpoint: {len(tunable)} tensors")
    else:
        print(f"[warn] no fine-tuned checkpoint loaded (path={checkpoint_path})")
    emo.eval()
    return agent, env, emo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(
        "out", "finetune_full_v2_3class", "emomoconvq_best.pt"))
    parser.add_argument("--keep-emotions", default="neutral,happy,fearful",
                        help="emotion subset to evaluate")
    parser.add_argument("--n-captions", type=int, default=20,
                        help="number of test captions (smaller than for "
                             "compositional eval because physics sim is slow)")
    parser.add_argument("--texts-dir", default=os.path.join(
        "..", "HumanML3D", "texts"))
    parser.add_argument("--split-file", default=os.path.join(
        "..", "HumanML3D", "texts", "train.txt"))
    parser.add_argument("--emotion-csv", default=os.path.join(
        "out", "emotion_dataset", "emotion_dataset.csv"))
    parser.add_argument("--out-dir", default=os.path.join(
        "out", "kinematic_features"))
    parser.add_argument("--save-bvh", action="store_true", default=True,
                        help="save BVH for each generated motion (used by "
                             "Phase G video rendering)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.argv = [sys.argv[0]]    # hide our flags from build_agent argparse
    os.makedirs(args.out_dir, exist_ok=True)
    bvh_dir = os.path.join(args.out_dir, "bvh")
    os.makedirs(bvh_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- emotion subset -----------------------------------------------
    keep_set = {e.strip() for e in args.keep_emotions.split(",") if e.strip()}
    emo_list = [e for e in EMO_LIST_ALL if e in keep_set]
    print(f"[setup] evaluating emotions: {emo_list}")
    keep_orig_ids = [EMOTION_TO_IDX[e] for e in emo_list]

    # --- test captions -------------------------------------------------
    exclude_ids = set()
    if os.path.isfile(args.emotion_csv):
        with open(args.emotion_csv, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("motion_id", ""):
                    exclude_ids.add(r["motion_id"])
    test_caps = sample_neutral_captions(
        args.texts_dir, args.split_file, exclude_ids,
        n_captions=args.n_captions, seed=args.seed)
    print(f"[data] curated {len(test_caps)} neutral test captions")

    with open(os.path.join(args.out_dir, "test_captions.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["idx", "motion_id", "caption"])
        for i, (mid, cap) in enumerate(test_caps):
            w.writerow([i, mid, cap])

    # --- models --------------------------------------------------------
    agent, env, emo = build_models(device, args.checkpoint)

    from transformers import T5Tokenizer, T5EncoderModel
    print("[t5] loading t5-large ...")
    tok = T5Tokenizer.from_pretrained("t5-large")
    enc = T5EncoderModel.from_pretrained("t5-large").to(device).eval()
    clip_feature = torch.zeros(1, gpt_config.clip_dim, device=device)

    # --- generate + simulate + capture features -----------------------
    feat_names = ["vertical_energy", "avg_height", "avg_speed",
                  "step_freq", "lean_angle", "arm_swing"]
    rows = []   # (caption_idx, emotion, feat_dict)

    t0 = time.time()
    n_total = len(test_caps) * len(emo_list)
    n_done  = 0
    for ci, (mid, cap) in enumerate(test_caps):
        bf, bm = text2bert(cap, tok, enc, device)
        for ename in emo_list:
            torch.manual_seed(args.seed); np.random.seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            emo_id = torch.tensor([EMOTION_TO_IDX[ename]], device=device)
            try:
                with torch.no_grad():
                    cur_embedding, _ = emo.sample(clip_feature, bf, bm,
                                                  emotion=emo_id)
            except Exception as e:
                print(f"  [skip-gen] cap {ci} {ename}: {e}")
                n_done += 1; continue

            try:
                pos_traj, saver = simulate_and_capture(agent, env, cur_embedding)
            except Exception as e:
                print(f"  [skip-sim] cap {ci} {ename}: {e}")
                n_done += 1; continue

            feats = compute_features(pos_traj)
            if feats is None:
                print(f"  [skip-feat] cap {ci} {ename}: too short")
                n_done += 1; continue
            row = {"caption_idx": ci, "motion_id": mid,
                   "caption": cap, "emotion": ename, **feats}
            rows.append(row)

            if args.save_bvh:
                bvh_name = f"cap{ci:03d}_{ename}.bvh"
                try:
                    saver.to_file(os.path.join(bvh_dir, bvh_name))
                except Exception as e:
                    print(f"  [warn-bvh] {bvh_name}: {e}")

            n_done += 1
            if n_done % 5 == 0 or n_done == n_total:
                rate = n_done / max(time.time() - t0, 1e-3)
                eta = (n_total - n_done) / max(rate, 1e-3)
                print(f"  [{n_done:4d}/{n_total}]  "
                      f"{rate:.2f} samples/s  ETA {eta/60:.1f} min")

    # --- write features.csv -------------------------------------------
    feat_csv = os.path.join(args.out_dir, "features.csv")
    with open(feat_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["caption_idx", "motion_id", "caption", "emotion"] + feat_names)
        for r in rows:
            w.writerow([r["caption_idx"], r["motion_id"], r["caption"],
                        r["emotion"]] + [r[k] for k in feat_names])
    print(f"\n[ok] features -> {feat_csv}  ({len(rows)} rows)")

    # --- ANOVA per feature --------------------------------------------
    try:
        from scipy import stats
    except ImportError:
        print("[warn] scipy not available -- skipping ANOVA")
        return

    print("\n" + "=" * 76)
    print(" One-way ANOVA: kinematic feature ~ emotion label")
    print("=" * 76)
    print(f"{'feature':<18s}{'F':>10s}{'p':>14s}{'eta^2':>10s}  per-emotion mean")
    print("-" * 76)
    anova = {}
    for feat in feat_names:
        groups = [[r[feat] for r in rows if r["emotion"] == e and not
                   (isinstance(r[feat], float) and np.isnan(r[feat]))]
                  for e in emo_list]
        if any(len(g) < 2 for g in groups):
            print(f"{feat:<18s}  (insufficient data per group)")
            continue
        F, p = stats.f_oneway(*groups)
        # eta-squared: SS_between / SS_total
        all_vals = np.concatenate([np.asarray(g) for g in groups])
        grand_mean = all_vals.mean()
        ss_total = ((all_vals - grand_mean) ** 2).sum()
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        eta2 = float(ss_between / max(ss_total, 1e-12))
        means = "  ".join(
            f"{e}:{np.mean(g):.3f}" for e, g in zip(emo_list, groups))
        sig = "*" if p < 0.05 else " "
        print(f"{feat:<18s}{F:>10.3f}{p:>13.3e}{sig}{eta2:>10.3f}  {means}")
        anova[feat] = {"F": float(F), "p": float(p), "eta_sq": eta2,
                        "means_by_emotion": {e: float(np.mean(g))
                                              for e, g in zip(emo_list, groups)},
                        "n_per_emotion": {e: len(g)
                                          for e, g in zip(emo_list, groups)}}

    with open(os.path.join(args.out_dir, "anova_results.json"), "w") as f:
        json.dump({"emotions": emo_list, "anova": anova,
                   "n_total_rows": len(rows)}, f, indent=2)
    print(f"\n[ok] anova    -> {os.path.join(args.out_dir, 'anova_results.json')}")

    # --- per-feature distribution figure ------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
        for ax, feat in zip(axes.flat, feat_names):
            groups = [[r[feat] for r in rows if r["emotion"] == e and not
                       (isinstance(r[feat], float) and np.isnan(r[feat]))]
                      for e in emo_list]
            if all(g for g in groups):
                bp = ax.boxplot(groups, labels=emo_list, patch_artist=True)
                colors = ["#9aa0a6", "#f2b134", "#4a90d9", "#d9534f", "#6f42c1"]
                for patch, c in zip(bp["boxes"], colors):
                    patch.set_facecolor(c); patch.set_alpha(0.6)
            ax.set_title(feat)
            if feat in anova:
                p = anova[feat]["p"]; e2 = anova[feat]["eta_sq"]
                ax.text(0.02, 0.98,
                        f"p={p:.2e}\nη²={e2:.3f}",
                        transform=ax.transAxes, va="top", ha="left",
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3",
                                  fc="white", alpha=0.7, ec="gray"))
            ax.tick_params(axis="x", labelsize=8)
        fig.suptitle("Kinematic features per conditioning emotion "
                     f"(n={len(rows)} samples)", fontsize=11)
        fig.tight_layout()
        fig_path = os.path.join(args.out_dir, "feature_distributions.png")
        fig.savefig(fig_path, dpi=200)
        print(f"[ok] figure   -> {fig_path}")
    except Exception as e:
        print(f"[warn] could not draw figure: {e}")


if __name__ == "__main__":
    main()
