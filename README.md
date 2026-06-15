# EmoMoconvq

**On the Decoder Bottleneck in Physics-Based Emotion-Conditioned Motion Generation**

SJTU 2025-2026 Robotics course final project — Kangwei Sun (524531910006).

This project asks a single question:

> *Can a frozen physics-based text-to-motion model (MoConVQ) be retrofitted with
> discrete emotion control via parameter-efficient fine-tuning?*

The answer turns out to be **yes at the token level, no at the motion level** —
and the reason why is the contribution of this report.

---

## TL;DR — the result

|                              | Result |
|------------------------------|--------|
| Token-level emotion signal   | ✅ Detectable. Separation ratio **1.36 ×** baseline noise; 3-class classifier reaches **39.3 %** accuracy (happy alone 50 %). |
| Motion-level emotion signal  | ❌ Not detectable. 6 standard kinematic features (vertical energy, step frequency, lean angle, …) show **one-way ANOVA p > 0.7** for *every* feature; paired analysis p > 0.26. |
| Diagnosis                    | **Decoder bottleneck.** MoConVQ's frozen physics tracker is optimised for kinematic fidelity to a single training distribution, so it normalises emotion-induced token differences *out* of the rendered motion. |

Full write-up: [paper-template-latex/final-524531910006.pdf](paper-template-latex/final-524531910006.pdf).

---

## Repository layout

```
EmoMoconvq/
├── MoConVQ-main/                  upstream MoConVQ (SIGGRAPH 2024) + our additions
│   ├── MoConVQCore/Model/
│   │   └── emo_cross_trans.py     EmoMoconvq emotion-embedding module
│   └── Script/
│       ├── build_emotion_dataset.py            v1 — 463 emotion-bearing captions
│       ├── build_distillation_dataset.py       v1 distillation pairs
│       ├── build_distillation_dataset_v2.py    v2 — 100 base × 5 emotions
│       ├── finetune_emomoconvq_full.py         full fine-tune (--keep-emotions for Path B / D')
│       ├── eval_compositional.py               compositional generalisation eval
│       ├── analyze_kinematic_features.py       Phase D — ANOVA on 6 affective features
│       ├── analyze_kinematic_paired.py         paired-difference analysis
│       ├── render_qualitative_videos.py        Phase G — 15-video comparison grid
│       ├── render_bvh_blender.py               Blender 5.x per-bone renderer
│       └── gradio_demo.py                      Phase H — interactive web demo
├── paper-template-latex/
│   ├── final-524531910006.pdf     final 5-page IEEEtran report
│   ├── final-524531910006.tex     source
│   └── fig_*.png                  progression / confusion / ANOVA / paired figures
├── ppt_build/
│   ├── EmoMoconvq_spotlight.pptx  9-slide minimalist spotlight deck
│   ├── 演讲稿.docx                 ~5 min Chinese speech script
│   ├── build_pptx.js              pptxgenjs generator
│   └── build_script.js            docx-js speech generator
├── HumanML3D/                     subset of HumanML3D used for filtering
├── proposal-524531910006.pdf      original project proposal
└── Midterm_Report.md              midterm summary
```

## Setup

Environment is essentially MoConVQ's environment (see
[MoConVQ-main/README.md](MoConVQ-main/README.md)) plus a few extras for our
analysis scripts.

```bash
# 1. Conda env (same as upstream)
cd MoConVQ-main
conda env create -f requirements.yml
conda activate moconvq

# 2. Build the C++ physics simulator (Windows, MSVC)
setup.cmd

# 3. Pretrained weights — download from MoConVQ release page and drop into MoConVQ-main/:
#    moconvq_base.data, text_generation_GPT.pth, unconditional_GPT.pth, simple_motion_data.h5
#    (URLs in MoConVQ-main/README.md; >100 MB so not tracked in git)

# 4. Extra deps for our scripts
pip install gradio matplotlib scipy pandas

# 5. (Optional) Blender 5.x for qualitative video rendering
```

## Reproducing the experiments

The three progressive variants in the report:

```bash
# --- v1: emotion-augmented captions, top-4 fine-tune ---
python Script/build_distillation_dataset.py
python Script/finetune_emomoconvq_full.py            # default top-4

# --- v2: 5-variant grid (100 base × 5 emotions) ---
python Script/build_distillation_dataset_v2.py
python Script/finetune_emomoconvq_full.py --dataset v2

# --- Path B / 3-class D': emotion-only conditioning ---
python Script/finetune_emomoconvq_full.py --dataset v2 --keep-emotions happy,sad,neutral

# Evaluation
python Script/eval_compositional.py --checkpoint <path>
python Script/analyze_kinematic_features.py          # Phase D ANOVA → fig_kinematic_anova.png
python Script/analyze_kinematic_paired.py            # paired-difference
```

Qualitative videos (Phase G):

```bash
python Script/render_qualitative_videos.py           # 5 captions × 3 emotions = 15 clips
```

## Interactive demo

```bash
cd MoConVQ-main
python Script/gradio_demo.py            # local: http://127.0.0.1:7860
python Script/gradio_demo.py --share    # public *.gradio.live URL (72 h)
```

Type a caption, pick an emotion, generate a physics-driven BVH and an MP4
preview in the browser.

## Final deliverables

| Item                    | Location |
|-------------------------|----------|
| 5-page final report     | [paper-template-latex/final-524531910006.pdf](paper-template-latex/final-524531910006.pdf) |
| Spotlight presentation  | [ppt_build/EmoMoconvq_spotlight.pptx](ppt_build/EmoMoconvq_spotlight.pptx) (video embedded) |
| Chinese speech script   | [ppt_build/演讲稿.docx](ppt_build/演讲稿.docx) |
| Interactive demo        | `Script/gradio_demo.py` |
| Qualitative comparison  | `ppt_build/cap008_compare.mp4` (3 emotions side-by-side) |

## Honest limitations

This project deliberately reports a **negative motion-level result** rather
than claiming success on a metric that does not reflect physical expressivity.
The decoder-bottleneck diagnosis is the contribution.

Things that **don't** work and why:

- **Path C (real BVH distillation)** is blocked: 100STYLE BVH uses a generic
  skeleton; MoConVQ uses a custom 22-joint skeleton with fused joints like
  `pelvis_lowerback`, so retargeting fails at the BVH parser.
- **3-class classifier on distillation val set: 27 %** (below chance) — an
  honest finding that the distillation targets carry a weak emotion signal.
- The Gradio demo runs **only on the host machine** that has the C++ simulator
  + 1.2 GB of weights + a GPU. `--share` exposes it as a public URL but the
  computation still happens on the host.

## Acknowledgments

Built on **MoConVQ** (Yao et al., SIGGRAPH 2024 Journal Track) —
[moconvq.github.io](https://moconvq.github.io/). The frozen tracker and the
T2M-MoConGPT baseline are unmodified upstream components; EmoMoconvq adds only
an emotion embedding + projection on top of T5 features.

Dataset filtering uses **HumanML3D** captions.

