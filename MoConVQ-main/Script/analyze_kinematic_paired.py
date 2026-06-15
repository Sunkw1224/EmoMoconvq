"""
analyze_kinematic_paired.py
============================
Paired (within-caption) statistical analysis of EmoMoconvq's kinematic
features. This is the correct statistical test for our design: each
caption appears under all evaluated emotions, so caption-level variance
must be removed before testing for emotion effects.

We do two things per feature:

  1. Paired t-test: for each pair (happy - neutral), (fearful - neutral),
     test whether the within-caption difference is significantly non-zero
     across all captions.

  2. Repeated-measures one-way ANOVA equivalent (within-subjects F):
     decompose total SS into between-emotion, between-caption (subject),
     and residual; test the emotion effect against residual.

A "sign agreement" rate is also reported per feature: the fraction of
captions where (happy - neutral) and (fearful - neutral) have the
expected sign (happy lighter, fearful tenser).

Input:  out/kinematic_features/features.csv  (from analyze_kinematic_features)
Output: out/kinematic_features/paired_stats.json
        out/kinematic_features/paired_differences.png
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


FEAT_NAMES = ["vertical_energy", "avg_height", "avg_speed",
              "step_freq", "lean_angle", "arm_swing"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", default=os.path.join(
        "out", "kinematic_features", "features.csv"))
    parser.add_argument("--out-dir", default=os.path.join(
        "out", "kinematic_features"))
    parser.add_argument("--baseline", default="neutral",
                        help="emotion to use as the within-caption reference")
    args = parser.parse_args()

    # ---- load -----------------------------------------------------------
    by_cap = defaultdict(dict)        # cap_idx -> emotion -> {feat: val}
    emo_set = set()
    with open(args.features_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ci = int(r["caption_idx"])
            e = r["emotion"]
            emo_set.add(e)
            row = {fn: float(r[fn]) for fn in FEAT_NAMES if r.get(fn) not in
                   (None, "", "nan")}
            by_cap[ci][e] = row

    emo_list = sorted(emo_set)
    if args.baseline not in emo_list:
        raise SystemExit(f"[ERROR] baseline emotion '{args.baseline}' not in "
                         f"data ({emo_list})")
    others = [e for e in emo_list if e != args.baseline]
    print(f"[data] {len(by_cap)} captions × {emo_list} emotions")
    print(f"[data] baseline emotion: {args.baseline};  comparing: {others}")

    # ---- build paired matrices  (n_caption, n_feature) per emotion -----
    captions = sorted(c for c in by_cap if
                      all(e in by_cap[c] for e in emo_list))
    if len(captions) < 3:
        raise SystemExit(f"[ERROR] only {len(captions)} captions have all "
                         f"emotions; need at least 3")
    print(f"[data] {len(captions)} captions have full emotion coverage")

    F = {e: np.array([[by_cap[c][e][fn] for fn in FEAT_NAMES]
                      for c in captions])
         for e in emo_list}   # F[e]: (n_cap, n_feat)

    # ---- paired t-tests vs baseline ------------------------------------
    try:
        from scipy import stats
        have_scipy = True
    except ImportError:
        have_scipy = False
        print("[warn] scipy not available -- only basic stats reported")

    results = {"baseline": args.baseline,
               "emotions": emo_list,
               "n_captions": len(captions),
               "per_feature": {}}

    print("\n" + "=" * 90)
    print(" Paired (within-caption) analysis vs baseline = "
          f"'{args.baseline}'")
    print("=" * 90)
    print(f"{'feature':<18s}" + "".join(
        f"{e+' diff':>14s}{'(p)':>10s}{'sign':>7s}" for e in others))
    print("-" * 90)

    for fi, fn in enumerate(FEAT_NAMES):
        per_feat = {"baseline_mean": float(F[args.baseline][:, fi].mean()),
                    "baseline_std":  float(F[args.baseline][:, fi].std(ddof=1))}
        line = f"{fn:<18s}"
        for e in others:
            diff = F[e][:, fi] - F[args.baseline][:, fi]
            mean_diff = float(diff.mean())
            sign_agreement = float((np.sign(diff) ==
                                    np.sign(mean_diff)).mean()) \
                if abs(mean_diff) > 1e-12 else 0.5
            if have_scipy:
                tstat, pval = stats.ttest_1samp(diff, popmean=0.0)
                pval = float(pval)
            else:
                pval = float("nan"); tstat = float("nan")
            sig = "*" if pval < 0.05 else " "
            line += f"{mean_diff:>+13.4f}{sig}{pval:>10.2e}{sign_agreement:>6.0%} "
            per_feat[f"{e}_vs_{args.baseline}"] = {
                "mean_diff": mean_diff,
                "p_value": pval,
                "t_stat": float(tstat) if have_scipy else None,
                "sign_consistency": sign_agreement,
            }
        results["per_feature"][fn] = per_feat
        print(line)

    print()
    print("  '*' marks p < 0.05.  'sign' = fraction of captions where diff "
          "matches its mean direction;")
    print("  >70% means the effect is consistent across captions.")

    # ---- repeated-measures ANOVA equivalent: within-subjects F test ----
    if have_scipy:
        print("\n" + "=" * 76)
        print(" Within-subjects (repeated measures) one-way ANOVA per feature")
        print("=" * 76)
        print(f"{'feature':<18s}{'F_within':>12s}{'p':>14s}{'eta_p^2':>12s}")
        print("-" * 76)
        for fi, fn in enumerate(FEAT_NAMES):
            # data matrix (subjects=captions, conditions=emotions)
            X = np.array([F[e][:, fi] for e in emo_list]).T  # (n_cap, n_emo)
            n_subj, n_cond = X.shape
            grand_mean = X.mean()
            subj_means = X.mean(axis=1)         # per-caption baseline
            cond_means = X.mean(axis=0)         # per-emotion mean

            SS_total   = ((X - grand_mean) ** 2).sum()
            SS_subj    = n_cond * ((subj_means - grand_mean) ** 2).sum()
            SS_cond    = n_subj * ((cond_means - grand_mean) ** 2).sum()
            SS_error   = SS_total - SS_subj - SS_cond
            df_cond    = n_cond - 1
            df_error   = (n_cond - 1) * (n_subj - 1)
            MS_cond    = SS_cond  / max(df_cond, 1)
            MS_error   = SS_error / max(df_error, 1)
            F_within   = MS_cond / max(MS_error, 1e-20)
            p_within   = float(1.0 - stats.f.cdf(F_within, df_cond, df_error))
            eta_p_sq   = float(SS_cond / max(SS_cond + SS_error, 1e-12))
            sig = "*" if p_within < 0.05 else " "
            print(f"{fn:<18s}{F_within:>12.3f}{p_within:>13.2e}{sig}{eta_p_sq:>12.3f}")
            results["per_feature"][fn]["within_F"] = float(F_within)
            results["per_feature"][fn]["within_p"] = p_within
            results["per_feature"][fn]["within_eta_p_sq"] = eta_p_sq

    # ---- save -----------------------------------------------------------
    out_json = os.path.join(args.out_dir, "paired_stats.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[ok] paired_stats -> {out_json}")

    # ---- figure: per-caption difference distributions ------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
        colors = {"happy": "#f2b134", "fearful": "#6f42c1",
                  "neutral": "#9aa0a6", "sad": "#4a90d9",
                  "angry": "#d9534f"}
        for ax, fn in zip(axes.flat, FEAT_NAMES):
            data = []
            labels = []
            cs = []
            for e in others:
                fi = FEAT_NAMES.index(fn)
                diff = F[e][:, fi] - F[args.baseline][:, fi]
                data.append(diff)
                labels.append(f"{e}-{args.baseline}")
                cs.append(colors.get(e, "gray"))
            bp = ax.boxplot(data, labels=labels, patch_artist=True,
                            widths=0.55, showmeans=True)
            for patch, c in zip(bp["boxes"], cs):
                patch.set_facecolor(c); patch.set_alpha(0.55)
            ax.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.6)
            ax.set_title(fn)
            p_str = ""
            pf = results["per_feature"][fn]
            for e in others:
                key = f"{e}_vs_{args.baseline}"
                if key in pf:
                    p = pf[key]["p_value"]
                    p_str += f"{e}: p={p:.2e}\n"
            ax.text(0.02, 0.98, p_str.strip(),
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.25",
                              fc="white", alpha=0.8, ec="gray"))
            ax.tick_params(axis="x", labelsize=8)
        fig.suptitle(f"Within-caption kinematic differences "
                     f"({len(captions)} paired captions, vs '{args.baseline}')",
                     fontsize=11)
        fig.tight_layout()
        fig_path = os.path.join(args.out_dir, "paired_differences.png")
        fig.savefig(fig_path, dpi=200)
        print(f"[ok] figure       -> {fig_path}")
    except Exception as e:
        print(f"[warn] could not draw figure: {e}")


if __name__ == "__main__":
    main()
