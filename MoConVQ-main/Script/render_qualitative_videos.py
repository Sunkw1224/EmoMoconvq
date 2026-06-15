"""
render_qualitative_videos.py
=============================
Phase G orchestrator: render representative BVH motions from Phase D into
MP4 clips, then composite a side-by-side comparison MP4 per caption using
ffmpeg's hstack filter.

Prerequisite: Phase D done -> BVHs at out/kinematic_features/bvh/

Run from the MoConVQ-main directory:
    python Script/render_qualitative_videos.py
Optional:
    --blender PATH    explicit path to blender.exe (default: 'blender' on PATH)
    --ffmpeg  PATH    explicit path to ffmpeg.exe  (default: 'ffmpeg' on PATH)
    --n-captions N    how many captions to render (default 5)
    --emotions LIST   comma-separated emotions to render (must match the BVHs)
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh-dir", default=os.path.join(
        "out", "kinematic_features", "bvh"))
    parser.add_argument("--captions-csv", default=os.path.join(
        "out", "kinematic_features", "test_captions.csv"))
    parser.add_argument("--out-dir", default=os.path.join(
        "out", "qualitative_videos"))
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--ffmpeg",  default="ffmpeg")
    parser.add_argument("--n-captions", type=int, default=5)
    parser.add_argument("--emotions", default="neutral,happy,fearful")
    parser.add_argument("--engine", default="eevee",
                        choices=["eevee", "workbench", "cycles"])
    parser.add_argument("--width",  type=int, default=480)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    blender_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "render_bvh_blender.py")
    if not os.path.isfile(blender_script):
        raise SystemExit(f"[ERROR] missing {blender_script}")

    mp4_dir = os.path.join(args.out_dir, "mp4_per_clip")
    grid_dir = os.path.join(args.out_dir, "mp4_per_caption_grid")
    os.makedirs(mp4_dir, exist_ok=True)
    os.makedirs(grid_dir, exist_ok=True)

    emos = [e.strip() for e in args.emotions.split(",") if e.strip()]
    print(f"[setup] emotions to render: {emos}")

    # ----- read caption list --------------------------------------------
    caps = []
    if os.path.isfile(args.captions_csv):
        with open(args.captions_csv, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                caps.append((int(r["idx"]), r["caption"]))
    else:
        # fall back: scan bvh dir for unique caption indices
        seen = set()
        for fn in sorted(os.listdir(args.bvh_dir)):
            if fn.startswith("cap") and fn.endswith(".bvh"):
                ci = int(fn[3:6])
                if ci not in seen:
                    seen.add(ci); caps.append((ci, ""))

    if not caps:
        raise SystemExit(f"[ERROR] no captions found via {args.captions_csv} "
                         f"or BVH listing of {args.bvh_dir}")
    caps = caps[:args.n_captions]
    print(f"[setup] {len(caps)} captions, "
          f"{len(caps) * len(emos)} total clips to render")

    # check executables exist
    if shutil.which(args.blender) is None and not os.path.isfile(args.blender):
        print(f"[WARN] blender '{args.blender}' not found on PATH. "
              f"If installed but not on PATH, re-run with --blender "
              f"\"C:\\Program Files\\Blender Foundation\\Blender 4.x\\blender.exe\".")
    has_ffmpeg = (shutil.which(args.ffmpeg) is not None
                  or os.path.isfile(args.ffmpeg))
    if not has_ffmpeg:
        print(f"[WARN] ffmpeg '{args.ffmpeg}' not found. "
              f"Per-clip MP4s will still be produced (by Blender), "
              f"but the side-by-side comparison grid will be skipped.")

    # ----- render per-clip MP4 ------------------------------------------
    rendered = {}        # caption_idx -> {emotion: mp4_path}
    for ci, cap_text in caps:
        rendered[ci] = {}
        for e in emos:
            bvh = os.path.join(args.bvh_dir, f"cap{ci:03d}_{e}.bvh")
            mp4 = os.path.join(mp4_dir, f"cap{ci:03d}_{e}.mp4")
            if not os.path.isfile(bvh):
                print(f"  [skip] missing BVH {bvh}")
                continue
            if os.path.isfile(mp4):
                print(f"  [cached] {mp4}")
                rendered[ci][e] = mp4
                continue
            cmd = [args.blender, "--background", "--python", blender_script,
                   "--", "--input", bvh, "--output", mp4,
                   "--label", e,
                   "--engine", args.engine,
                   "--width", str(args.width),
                   "--height", str(args.height)]
            print(f"  rendering cap{ci:03d} / {e} ...")
            try:
                subprocess.run(cmd, check=True, timeout=600,
                               capture_output=True, text=True)
                rendered[ci][e] = mp4
            except subprocess.CalledProcessError as ex:
                print(f"  [error] cap{ci:03d}/{e}: blender returned "
                      f"{ex.returncode}\n      stderr tail: "
                      f"{(ex.stderr or '').splitlines()[-3:]}")
            except subprocess.TimeoutExpired:
                print(f"  [error] cap{ci:03d}/{e}: blender timeout")

    # ----- compose side-by-side grid per caption with ffmpeg -----------
    if has_ffmpeg:
        print("\n[grid] composing side-by-side comparison MP4s ...")
        for ci, cap_text in caps:
            files = [rendered.get(ci, {}).get(e) for e in emos]
            if any(p is None for p in files):
                print(f"  [skip] caption {ci}: missing one or more emotion mp4s")
                continue
            out_grid = os.path.join(grid_dir, f"cap{ci:03d}_compare.mp4")
            # build hstack filter
            inputs = []
            for p in files:
                inputs += ["-i", p]
            filt = ("[0:v][1:v]" if len(files) == 2
                    else "".join(f"[{i}:v]" for i in range(len(files))))
            filt += f"hstack=inputs={len(files)}[v]"
            cmd = [args.ffmpeg, "-y", *inputs, "-filter_complex", filt,
                   "-map", "[v]", "-c:v", "libx264", "-crf", "20",
                   out_grid]
            try:
                r = subprocess.run(cmd, check=True, capture_output=True,
                                   text=True)
                print(f"  -> {out_grid}")
            except subprocess.CalledProcessError as ex:
                err_tail = (ex.stderr or "").splitlines()[-8:]
                print(f"  [error] grid for caption {ci}: ffmpeg returned "
                      f"{ex.returncode}")
                print(f"      cmd: {' '.join(cmd)}")
                print(f"      stderr tail:")
                for line in err_tail:
                    print(f"        {line}")
    else:
        print("\n[grid] skipped (ffmpeg not available).")

    print(f"\n[done] per-clip MP4s in {mp4_dir}")
    if has_ffmpeg:
        print(f"[done] comparison grids in {grid_dir}")


if __name__ == "__main__":
    main()
