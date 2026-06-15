"""
render_bvh_blender.py
======================
Blender headless renderer for a single BVH motion. Used by Phase G to turn
the BVHs produced by analyze_kinematic_features.py into MP4 clips for the
qualitative emotion comparison.

Invocation:
    blender --background --python Script/render_bvh_blender.py -- \\
        --input  path/to/motion.bvh \\
        --output path/to/motion.mp4 \\
        --label  "happy" \\
        [--fps 30] [--width 480] [--height 480] [--engine eevee]

Notes:
  * The script clears the default scene, imports the BVH, places a ground
    plane, sun and area lights, and a fixed camera. EEVEE is used by default
    (orders of magnitude faster than Cycles for skeletal previews).
  * Designed for Blender 3.x / 4.x; the engine name is auto-detected.
"""

import math
import os
import shutil
import subprocess
import sys
import tempfile

import bpy


# ----------------------------------------------------------------------
def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--label",  default="")
    p.add_argument("--fps",    type=int, default=30)
    p.add_argument("--width",  type=int, default=480)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--frame-step", type=int, default=4,
                   help="render every Nth source frame (BVH is 120fps); "
                        "default 4 -> ~30 effective fps")
    p.add_argument("--engine", default="eevee",
                   choices=["eevee", "workbench", "cycles"])
    return p.parse_args(argv)


def list_engine_ids(scn):
    """Query the legal enum identifiers for scn.render.engine."""
    try:
        prop = scn.render.bl_rna.properties["engine"]
        return [item.identifier for item in prop.enum_items]
    except Exception:
        return []


def pick_engine(scn, candidates):
    """Set scn.render.engine to the first available candidate; return name."""
    available = list_engine_ids(scn)
    for name in candidates:
        if name in available:
            try:
                scn.render.engine = name
                return name
            except Exception:
                continue
    # final fallback: try setting whatever the user passed and let Blender
    # raise -- gives a useful error message
    if candidates:
        try:
            scn.render.engine = candidates[0]
            return candidates[0]
        except Exception:
            pass
    return scn.render.engine


def main():
    args = parse_args()

    # --- clean slate ---------------------------------------------------
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene

    # --- import BVH ----------------------------------------------------
    bpy.ops.import_anim.bvh(
        filepath=args.input,
        global_scale=1.0,
        rotate_mode="NATIVE",
        axis_forward="-Z",
        axis_up="Y",
        use_fps_scale=False,
        update_scene_duration=True,
    )
    arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    if arm is None:
        raise RuntimeError(f"no armature imported from {args.input}")

    # --- per-bone visible geometry (sphere at head + cylinder along the
    #     bone). We DON'T use Armature/Skin modifiers in Blender 5.x --
    #     they have caching issues we couldn't resolve. Instead we keep
    #     one solid sphere and one cylinder per bone and just update
    #     their object-level location/rotation/scale every frame, which
    #     reliably triggers a depsgraph evaluation and re-render.
    from mathutils import Vector, Quaternion

    skel_mat = bpy.data.materials.new("skel_mat")
    skel_mat.use_nodes = True
    bsdf_s = skel_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf_s:
        bsdf_s.inputs["Base Color"].default_value = (0.22, 0.24, 0.32, 1.0)
        if "Roughness" in bsdf_s.inputs:
            bsdf_s.inputs["Roughness"].default_value = 0.45

    bones = list(arm.data.bones)
    print(f"[debug] armature has {len(bones)} bones")

    bone_geometry = []     # list of (bone_name, sphere_obj, cyl_obj)
    SPHERE_R = 0.07
    CYL_R = 0.035
    Z_AXIS = Vector((0.0, 0.0, 1.0))

    for b in bones:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=SPHERE_R, segments=14,
                                              ring_count=10,
                                              location=(0.0, 0.0, 0.0))
        sph = bpy.context.active_object
        sph.name = f"skel_sph_{b.name}"
        sph.data.materials.append(skel_mat)
        # Unit-depth cylinder, scaled along Z each frame to match bone length.
        bpy.ops.mesh.primitive_cylinder_add(radius=CYL_R, depth=1.0,
                                            location=(0.0, 0.0, 0.0))
        cyl = bpy.context.active_object
        cyl.name = f"skel_cyl_{b.name}"
        cyl.data.materials.append(skel_mat)
        cyl.rotation_mode = "QUATERNION"
        bone_geometry.append((b.name, sph, cyl))

    _SKEL_ARM = arm
    _SKEL_GEOM = bone_geometry

    # (debug red sphere removed -- the pipeline is verified working)

    def bake_skeleton_for_current_frame():
        """Move each per-bone sphere/cylinder to follow the current pose."""
        n_ok = 0
        first3_world = []
        for i, (bname, sph, cyl) in enumerate(_SKEL_GEOM):
            pb = _SKEL_ARM.pose.bones.get(bname)
            if pb is None:
                continue
            head_w = _SKEL_ARM.matrix_world @ pb.head
            tail_w = _SKEL_ARM.matrix_world @ pb.tail
            sph.location = head_w
            vec = tail_w - head_w
            L = vec.length
            if L < 1e-4:
                cyl.location = head_w
                cyl.scale = (1.0, 1.0, 1e-4)
            else:
                cyl.location = (head_w + tail_w) * 0.5
                cyl.scale = (1.0, 1.0, L)
                cyl.rotation_quaternion = Z_AXIS.rotation_difference(vec / L)
            n_ok += 1
            if i < 3:
                first3_world.append((bname, tuple(round(c, 3) for c in head_w)))
        # force a depsgraph evaluation so the new transforms take effect
        bpy.context.view_layer.update()

    print(f"[debug] spawned {len(bone_geometry)} (sphere, cylinder) "
          f"pairs as per-bone visible geometry")

    # frame range from imported action
    if arm.animation_data and arm.animation_data.action:
        f0, f1 = arm.animation_data.action.frame_range
        scn.frame_start = max(1, int(f0))
        scn.frame_end   = max(scn.frame_start + 1, int(f1))

    # --- ground plane (large, light grey, receives shadows) ------------
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane_mat = bpy.data.materials.new("plane_mat")
    plane_mat.use_nodes = True
    bsdf = plane_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.85, 0.85, 0.88, 1.0)
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.7
    plane.data.materials.append(plane_mat)

    # --- lights --------------------------------------------------------
    bpy.ops.object.light_add(type="SUN", location=(4, -3, 6))
    bpy.context.object.data.energy = 3.0
    bpy.ops.object.light_add(type="AREA", location=(-3, -4, 4))
    bpy.context.object.data.energy = 200
    bpy.context.object.data.size = 5

    # --- camera --------------------------------------------------------
    # IMPORTANT: in MoConVQ's BVH the character is NOT centred at world
    # origin; it can be tens of metres away from origin. We track the
    # RootJoint *sphere* (which we re-position every frame in the bake),
    # so the camera always frames the actual character position.
    root_target = None
    for bname, sph, cyl in bone_geometry:
        if bname.lower().endswith("rootjoint") or bname.lower() == "root":
            root_target = sph
            break
    if root_target is None and bone_geometry:
        root_target = bone_geometry[0][1]   # fall back to first sphere
    print(f"[debug] camera track target: "
          f"{root_target.name if root_target else 'NONE -- will fall back to armature'}")

    bpy.ops.object.camera_add(location=(2.8, -6.0, 2.2))
    cam = bpy.context.object
    cam.data.lens = 32
    scn.camera = cam
    track = cam.constraints.new("TRACK_TO")
    track.target = root_target if root_target is not None else arm
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    # The camera is positioned relative to world origin, but the character
    # can walk anywhere. Parent the camera to the root sphere so it follows
    # the character with a fixed offset (5 m back, 2 m up).
    if root_target is not None:
        cam.parent = root_target
        cam.location = (2.8, -6.0, 1.3)

    # Log the armature's resting bounding box so we can sanity-check
    # camera framing if something still looks off.
    try:
        verts = [arm.matrix_world @ b.head_local for b in arm.data.bones]
        verts += [arm.matrix_world @ b.tail_local for b in arm.data.bones]
        xs = [v.x for v in verts]; ys = [v.y for v in verts]; zs = [v.z for v in verts]
        print(f"[debug] armature world bbox: "
              f"x=[{min(xs):.2f},{max(xs):.2f}]  "
              f"y=[{min(ys):.2f},{max(ys):.2f}]  "
              f"z=[{min(zs):.2f},{max(zs):.2f}]")
    except Exception:
        pass

    # (on-screen label removed -- it's now drawn by ffmpeg in the
    # orchestrator script after rendering, which is more robust to camera
    # changes and gives crisper text.)

    # --- render settings ----------------------------------------------
    scn.render.resolution_x = args.width
    scn.render.resolution_y = args.height
    scn.render.fps = args.fps
    scn.frame_step = max(1, args.frame_step)

    # Blender 5.x removed FFMPEG output. We render PNG sequence to a tmp
    # directory and call ffmpeg afterwards to mux into MP4.
    tmp_png_dir = tempfile.mkdtemp(prefix="emomoconvq_render_")
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    scn.render.image_settings.color_depth = "8"
    scn.render.filepath = os.path.join(tmp_png_dir, "frame_")

    available = list_engine_ids(scn)
    print(f"[engines] available: {available}")
    if args.engine == "eevee":
        eng = pick_engine(scn, ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE",
                                  "BLENDER_WORKBENCH"])
    elif args.engine == "workbench":
        eng = pick_engine(scn, ["BLENDER_WORKBENCH"])
    elif args.engine == "cycles":
        eng = pick_engine(scn, ["CYCLES"])
        try:
            scn.cycles.samples = 32
        except Exception:
            pass
    else:
        eng = scn.render.engine
    print(f"[engines] using: {eng}")

    # --- render PNG sequence (per-frame bake of skeleton vertices) -----
    print(f"[render] engine={scn.render.engine}  "
          f"frames={scn.frame_start}..{scn.frame_end}  "
          f"step={scn.frame_step}  png_dir={tmp_png_dir}")
    frames = list(range(scn.frame_start, scn.frame_end + 1, scn.frame_step))
    for i, f in enumerate(frames):
        scn.frame_set(f)
        bake_skeleton_for_current_frame()
        scn.render.filepath = os.path.join(tmp_png_dir, f"frame_{f:05d}")
        bpy.ops.render.render(write_still=True)
        if (i + 1) % 20 == 0 or i == len(frames) - 1:
            print(f"  rendered {i+1}/{len(frames)} frames")

    # --- mux PNGs -> MP4 via ffmpeg -----------------------------------
    pngs = sorted(p for p in os.listdir(tmp_png_dir) if p.endswith(".png"))
    if not pngs:
        print(f"[error] no PNGs produced in {tmp_png_dir}")
        sys.exit(2)
    print(f"[mux] {len(pngs)} PNG frames -> {args.output}")

    # Blender names frames by their absolute frame number with step skipped:
    # e.g. with frame_step=4 we get frame_0001.png, frame_0005.png, ...
    # Rename to a dense sequence first so ffmpeg can read with %04d.
    for i, p in enumerate(pngs, start=1):
        src = os.path.join(tmp_png_dir, p)
        dst = os.path.join(tmp_png_dir, f"seq_{i:05d}.png")
        os.rename(src, dst)

    # find ffmpeg (assume on PATH, since the orchestrator already required it)
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    ffmpeg_cmd = [
        ffmpeg, "-y",
        "-framerate", str(args.fps),
        "-i", os.path.join(tmp_png_dir, "seq_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        "-movflags", "+faststart",
        args.output,
    ]
    try:
        r = subprocess.run(ffmpeg_cmd, check=True,
                           capture_output=True, text=True)
        print(f"[ok] -> {args.output}")
    except subprocess.CalledProcessError as ex:
        print(f"[error] ffmpeg mux failed (rc={ex.returncode}):")
        for line in (ex.stderr or "").splitlines()[-10:]:
            print(f"   {line}")
        sys.exit(3)
    finally:
        shutil.rmtree(tmp_png_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
