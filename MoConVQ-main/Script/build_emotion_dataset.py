"""
build_emotion_dataset.py
========================
Filter the HumanML3D training-set captions for emotion-related keywords and
build a small emotion-labelled (motion_id, caption, emotion) dataset that will
later be used to fine-tune EmoMoconvq.

This is a pure CPU text-processing step -- it does NOT need the motion data,
only the HumanML3D `texts/` annotations.

Run from the MoConVQ-main directory:
    python Script/build_emotion_dataset.py

Optional arguments:
    --texts-dir   path to the HumanML3D 'texts' folder
    --split-file  path to train.txt (the training-split id list)
    --out-dir     output directory for the generated dataset / stats / figure
"""

import os
import re
import csv
import argparse
from collections import defaultdict

# ---------------------------------------------------------------------------
# Emotion keyword lexicons.
# These are intentionally kept as an editable dict so the dataset size and
# precision can be tuned by hand after inspecting the first run.
# ---------------------------------------------------------------------------
EMOTION_KEYWORDS = {
    "happy": [
        "happy", "happily", "joyful", "joyfully", "joyously", "cheerful",
        "cheerfully", "gleeful", "gleefully", "merrily", "jolly",
        "delighted", "elated", "excitedly",
    ],
    "sad": [
        "sad", "sadly", "sorrowful", "sorrowfully", "gloomy", "gloomily",
        "depressed", "miserable", "miserably", "unhappy", "unhappily",
        "dejected", "mournful", "mournfully", "glum", "downcast",
    ],
    "angry": [
        "angry", "angrily", "furious", "furiously", "irritated", "irritably",
        "enraged", "annoyed", "irate", "aggressive", "aggressively",
    ],
    "fearful": [
        "fearful", "fearfully", "scared", "afraid", "frightened", "terrified",
        "timid", "timidly", "nervous", "nervously", "anxious", "anxiously",
        "cautious", "cautiously", "hesitant", "hesitantly",
    ],
}

# Pre-compile one word-boundary regex per emotion (whole-word, case-insensitive).
EMOTION_REGEX = {
    emo: re.compile(r"\b(" + "|".join(words) + r")\b", re.IGNORECASE)
    for emo, words in EMOTION_KEYWORDS.items()
}


def classify_caption(caption):
    """Classify one caption.

    Returns (label, matched) where label is one of:
        'neutral'  -> no emotion keyword found
        'conflict' -> keywords from 2+ emotion groups (ambiguous, dropped)
        <emotion>  -> exactly one emotion group matched
    `matched` is the list/dict of matched keywords (for inspection).
    """
    hits = {}
    for emo, rgx in EMOTION_REGEX.items():
        found = rgx.findall(caption)
        if found:
            hits[emo] = sorted({w.lower() for w in found})
    if len(hits) == 0:
        return "neutral", []
    if len(hits) == 1:
        emo = next(iter(hits))
        return emo, hits[emo]
    return "conflict", hits


def parse_captions(text_file):
    """Yield each caption (text before the first '#') in a HumanML3D txt file."""
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            caption = line.split("#")[0].strip()
            if caption:
                yield caption


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_texts = os.path.normpath(
        os.path.join(script_dir, "..", "..", "HumanML3D", "texts"))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts-dir", default=default_texts,
                        help="HumanML3D 'texts' folder")
    parser.add_argument("--split-file", default=None,
                        help="training split id list (default: <texts-dir>/train.txt)")
    parser.add_argument("--out-dir", default=os.path.join(script_dir, "..", "out", "emotion_dataset"),
                        help="output directory")
    args = parser.parse_args()

    texts_dir = args.texts_dir
    split_file = args.split_file or os.path.join(texts_dir, "train.txt")
    out_dir = os.path.normpath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(texts_dir):
        raise SystemExit(f"[ERROR] texts dir not found: {texts_dir}")
    if not os.path.isfile(split_file):
        raise SystemExit(f"[ERROR] split file not found: {split_file}")

    # --- read the training-split id list ---------------------------------
    with open(split_file, "r", encoding="utf-8") as f:
        split_ids = [ln.strip() for ln in f if ln.strip()]
    print(f"[info] training-split ids: {len(split_ids)}")

    # --- scan every caption ----------------------------------------------
    rows = []                       # emotional (non-neutral, non-conflict) pairs
    n_neutral = 0
    n_conflict = 0
    n_total_caps = 0
    n_missing = 0
    motions_per_emotion = defaultdict(set)

    for mid in split_ids:
        tf = os.path.join(texts_dir, mid + ".txt")
        if not os.path.isfile(tf):
            n_missing += 1
            continue
        for caption in parse_captions(tf):
            n_total_caps += 1
            label, matched = classify_caption(caption)
            if label == "neutral":
                n_neutral += 1
            elif label == "conflict":
                n_conflict += 1
            else:
                rows.append((mid, label, caption, "|".join(matched)))
                motions_per_emotion[label].add(mid)

    # --- write the emotion dataset csv -----------------------------------
    dataset_csv = os.path.join(out_dir, "emotion_dataset.csv")
    with open(dataset_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["motion_id", "emotion", "caption", "matched_keywords"])
        w.writerows(rows)

    # --- write the per-class statistics csv ------------------------------
    stats_csv = os.path.join(out_dir, "emotion_stats.csv")
    emotions = list(EMOTION_KEYWORDS.keys())
    with open(stats_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["emotion", "n_caption_pairs", "n_unique_motions"])
        for emo in emotions:
            n_cap = sum(1 for r in rows if r[1] == emo)
            w.writerow([emo, n_cap, len(motions_per_emotion[emo])])

    # --- console summary table -------------------------------------------
    print("\n" + "=" * 58)
    print("  EmoMoconvq fine-tuning dataset -- HumanML3D keyword filter")
    print("=" * 58)
    print(f"  {'emotion':<10}{'caption pairs':>16}{'unique motions':>18}")
    print("  " + "-" * 54)
    total_pairs = 0
    for emo in emotions:
        n_cap = sum(1 for r in rows if r[1] == emo)
        total_pairs += n_cap
        print(f"  {emo:<10}{n_cap:>16}{len(motions_per_emotion[emo]):>18}")
    print("  " + "-" * 54)
    print(f"  {'TOTAL':<10}{total_pairs:>16}")
    print("=" * 58)
    print(f"  neutral captions (no keyword) : {n_neutral}")
    print(f"  conflict captions (dropped)   : {n_conflict}")
    print(f"  total captions scanned        : {n_total_caps}")
    print(f"  missing text files            : {n_missing}")
    print("=" * 58)
    print(f"[ok] dataset  -> {dataset_csv}")
    print(f"[ok] stats    -> {stats_csv}")

    # --- optional bar-chart figure for the report ------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        counts = [sum(1 for r in rows if r[1] == emo) for emo in emotions]
        colors = ["#f2b134", "#4a90d9", "#d9534f", "#6f42c1"]
        fig, ax = plt.subplots(figsize=(5, 3.2))
        bars = ax.bar(emotions, counts, color=colors[:len(emotions)])
        ax.set_ylabel("number of caption-text pairs")
        ax.set_title("EmoMoconvq dataset: emotion class distribution")
        for b, c in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    str(c), ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig_path = os.path.join(out_dir, "emotion_distribution.png")
        fig.savefig(fig_path, dpi=200)
        print(f"[ok] figure   -> {fig_path}")
    except ImportError:
        print("[warn] matplotlib not available -- skipped bar chart")


if __name__ == "__main__":
    main()
