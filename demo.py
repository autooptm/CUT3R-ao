#!/usr/bin/env python3
"""
3D Point Cloud Inference and Visualization Script

This script performs inference using the ARCroco3DStereo model and visualizes the
resulting 3D point clouds with the PointCloudViewer. Use the command-line arguments
to adjust parameters such as the model checkpoint path, image sequence directory,
image size, device, etc.

Usage:
    python demo.py [--model_path MODEL_PATH] [--seq_path SEQ_PATH] [--size IMG_SIZE]
                            [--device DEVICE] [--vis_threshold VIS_THRESHOLD] [--output_dir OUT_DIR]

Example:
    python demo.py --model_path src/cut3r_512_dpt_4_64.pth \
        --seq_path examples/001 --device cuda --size 512
"""

import os
import numpy as np
import torch
import time
import glob
import random
import cv2
import argparse
import tempfile
import shutil
from copy import deepcopy
from add_ckpt_path import add_path_to_dust3r
import imageio.v2 as iio
from concurrent.futures import ThreadPoolExecutor

_AO_OPT_2 = os.environ.get("CUT3R_OPT_2", "1") != "0"
_AO_OPT_3 = int(os.environ.get("CUT3R_OPT_3", "24"))
_AO_PNG_LEVEL = int(os.environ.get("CUT3R_PNG_LEVEL", "1"))

# Set random seed for reproducibility.
random.seed(42)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud inference and visualization using ARCroco3DStereo."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="src/cut3r_512_dpt_4_64.pth",
        help="Path to the pretrained model checkpoint.",
    )
    parser.add_argument(
        "--seq_path",
        type=str,
        default="",
        help="Path to the directory containing the image sequence.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--size",
        type=int,
        default="512",
        help="Shape that input images will be rescaled to; if using 224+linear model, choose 224 otherwise 512",
    )
    parser.add_argument(
        "--vis_threshold",
        type=float,
        default=1.5,
        help="Visualization threshold for the point cloud viewer. Ranging from 1 to INF",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./demo_tmp",
        help="value for tempfile.tempdir",
    )

    parser.add_argument(
        "--seq_paths",
        type=str,
        nargs="+",
        default=None,
        help="Optional: process SEVERAL sequences in one process, reusing the one "
             "loaded model, writing each into <output_dir>/<sequence name>. The "
             "single --seq_path invocation is unchanged.",
    )
    parser.add_argument(
        "--no_viewer",
        action="store_true",
        help="Optional: skip the interactive viser viewer (implied by --seq_paths).",
    )

    return parser.parse_args()


def prepare_input(
    img_paths, img_mask, size, raymaps=None, raymap_mask=None, revisit=1, update=True
):
    """
    Prepare input views for inference from a list of image paths.

    Args:
        img_paths (list): List of image file paths.
        img_mask (list of bool): Flags indicating valid images.
        size (int): Target image size.
        raymaps (list, optional): List of ray maps.
        raymap_mask (list, optional): Flags indicating valid ray maps.
        revisit (int): How many times to revisit each view.
        update (bool): Whether to update the state on revisits.

    Returns:
        list: A list of view dictionaries.
    """
    # Import image loader (delayed import needed after adding ckpt path).
    from src.dust3r.utils.image import load_images

    images = load_images(img_paths, size=size)
    views = []

    if raymaps is None and raymap_mask is None:
        # Only images are provided.
        for i in range(len(images)):
            view = {
                "img": images[i]["img"],
                "ray_map": torch.full(
                    (images[i]["img"].shape[0], 6, 1, 1),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(images[i]["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                    0
                ),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
            views.append(view)
    else:
        # Combine images and raymaps.
        num_views = len(images) + len(raymaps)
        assert len(img_mask) == len(raymap_mask) == num_views
        assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

        j = 0
        k = 0
        for i in range(num_views):
            view = {
                "img": (
                    images[j]["img"]
                    if img_mask[i]
                    else torch.full_like(images[0]["img"], torch.nan)
                ),
                "ray_map": (
                    raymaps[k]
                    if raymap_mask[i]
                    else torch.full_like(raymaps[0], torch.nan)
                ),
                "true_shape": (
                    torch.from_numpy(images[j]["true_shape"])
                    if img_mask[i]
                    else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                ),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                    0
                ),
                "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                "update": torch.tensor(img_mask[i]).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
            if img_mask[i]:
                j += 1
            if raymap_mask[i]:
                k += 1
            views.append(view)
        assert j == len(images) and k == len(raymaps)

    if revisit > 1:
        new_views = []
        for r in range(revisit):
            for i, view in enumerate(views):
                new_view = deepcopy(view)
                new_view["idx"] = r * len(views) + i
                new_view["instance"] = str(r * len(views) + i)
                if r > 0 and not update:
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                new_views.append(new_view)
        return new_views

    return views


def prepare_output(outputs, outdir, revisit=1, use_pose=True):
    """
    Process inference outputs to generate point clouds and camera parameters for visualization.

    Args:
        outputs (dict): Inference outputs.
        revisit (int): Number of revisits per view.
        use_pose (bool): Whether to transform points using camera pose.

    Returns:
        tuple: (points, colors, confidence, camera parameters dictionary)
    """
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf

    # Only keep the outputs corresponding to one full pass.
    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    def _host(t):
        return t if _AO_OPT_2 else t.cpu()

    pts3ds_self_ls = [_host(output["pts3d_in_self_view"]) for output in outputs["pred"]]
    pts3ds_other = [_host(output["pts3d_in_other_view"]) for output in outputs["pred"]]
    conf_self = [_host(output["conf_self"]) for output in outputs["pred"]]
    conf_other = [_host(output["conf"]) for output in outputs["pred"]]
    pts3ds_self = torch.cat(pts3ds_self_ls, 0)

    # Recover camera poses.
    pr_poses = [
        _host(pose_encoding_to_camera(pred["camera_pose"].clone()))
        for pred in outputs["pred"]
    ]
    R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
    t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)

    if use_pose:
        transformed_pts3ds_other = []
        for pose, pself in zip(pr_poses, pts3ds_self):
            transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
        pts3ds_other = transformed_pts3ds_other
        conf_other = conf_self

    # Estimate focal length based on depth.
    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    colors = [
        0.5 * (_host(output["img"]).permute(0, 2, 3, 1) + 1.0)
        for output in outputs["views"]
    ]

    cam_dict = {
        "focal": focal.cpu().numpy(),
        "pp": pp.cpu().numpy(),
        "R": R_c2w.cpu().numpy(),
        "t": t_c2w.cpu().numpy(),
    }

    pts3ds_self_tosave = pts3ds_self  # B, H, W, 3
    depths_tosave = pts3ds_self_tosave[..., 2]
    pts3ds_other_tosave = torch.cat(pts3ds_other)  # B, H, W, 3
    conf_self_tosave = torch.cat(conf_self)  # B, H, W
    conf_other_tosave = torch.cat(conf_other)  # B, H, W
    colors_tosave = torch.cat(
        [
            0.5 * (_host(output["img"]).permute(0, 2, 3, 1) + 1.0)
            for output in outputs["views"]
        ]
    )  # [B, H, W, 3]
    cam2world_tosave = torch.cat(pr_poses)  # B, 4, 4
    intrinsics_tosave = (
        torch.eye(3, device=cam2world_tosave.device)
        .unsqueeze(0)
        .repeat(cam2world_tosave.shape[0], 1, 1)
    )  # B, 3, 3
    intrinsics_tosave[:, 0, 0] = focal.detach().to(intrinsics_tosave.device)
    intrinsics_tosave[:, 1, 1] = focal.detach().to(intrinsics_tosave.device)
    intrinsics_tosave[:, 0, 2] = pp[:, 0]
    intrinsics_tosave[:, 1, 2] = pp[:, 1]

    os.makedirs(os.path.join(outdir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "conf"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "color"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "camera"), exist_ok=True)
    depths_np = depths_tosave.cpu().numpy()
    conf_np = conf_self_tosave.cpu().numpy()
    colors_np = (colors_tosave * 255).to(torch.uint8).cpu().numpy()
    c2w_np = cam2world_tosave.cpu().numpy()
    intrins_np = intrinsics_tosave.cpu().numpy()

    def _write_frame(f_id):
        np.save(os.path.join(outdir, "depth", f"{f_id:06d}.npy"), depths_np[f_id])
        np.save(os.path.join(outdir, "conf", f"{f_id:06d}.npy"), conf_np[f_id])
        iio.imwrite(
            os.path.join(outdir, "color", f"{f_id:06d}.png"),
            colors_np[f_id],
            compress_level=_AO_PNG_LEVEL,
        )
        np.savez(
            os.path.join(outdir, "camera", f"{f_id:06d}.npz"),
            pose=c2w_np[f_id],
            intrinsics=intrins_np[f_id],
        )

    n_frames = len(pts3ds_self)
    if _AO_OPT_3 > 1:
        with ThreadPoolExecutor(max_workers=_AO_OPT_3) as pool:
            list(pool.map(_write_frame, range(n_frames)))
    else:
        for f_id in range(n_frames):
            _write_frame(f_id)

    return pts3ds_other, colors, conf_other, cam_dict


def parse_seq_path(p):
    if os.path.isdir(p):
        img_paths = sorted(glob.glob(f"{p}/*"))
        tmpdirname = None
    else:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {p}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Error: Video FPS is 0 for {p}")
        frame_interval = 1
        frame_indices = list(range(0, total_frames, frame_interval))
        print(
            f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}"
        )
        img_paths = []
        tmpdirname = tempfile.mkdtemp()
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(tmpdirname, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            img_paths.append(frame_path)
        cap.release()
    return img_paths, tmpdirname


def run_inference(args, model=None):
    """
    Execute the full inference and visualization pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    # Set up the computation device.
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    # Add the checkpoint path (required for model imports in the dust3r package).
    add_path_to_dust3r(args.model_path)

    # Import model and inference functions after adding the ckpt path.
    from src.dust3r.inference import inference, inference_recurrent
    from src.dust3r.model import ARCroco3DStereo
    from viser_utils import PointCloudViewer

    # Prepare image file paths.
    img_paths, tmpdirname = parse_seq_path(args.seq_path)
    if not img_paths:
        print(f"No images found in {args.seq_path}. Please verify the path.")
        return

    print(f"Found {len(img_paths)} images in {args.seq_path}.")
    img_mask = [True] * len(img_paths)

    # Prepare input views.
    print("Preparing input views...")
    views = prepare_input(
        img_paths=img_paths,
        img_mask=img_mask,
        size=args.size,
        revisit=1,
        update=True,
    )
    if tmpdirname is not None:
        shutil.rmtree(tmpdirname)

    if model is None:
        print(f"Loading model from {args.model_path}...")
        model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
        model.eval()

    # Run inference.
    print("Running inference...")
    _ao_head = getattr(model, "downstream_head", None)
    _ao_skip_was = getattr(_ao_head, "skip_unread_outputs", None)
    if _ao_skip_was is not None:
        _ao_head.skip_unread_outputs = True
    start_time = time.time()
    outputs, state_args = inference(
        views, model, device, to_host=not _AO_OPT_2
    )
    total_time = time.time() - start_time
    per_frame_time = total_time / len(views)
    print(
        f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame)."
    )

    # Process outputs for visualization.
    print("Preparing output for visualization...")
    pts3ds_other, colors, conf, cam_dict = prepare_output(
        outputs, args.output_dir, 1, True
    )

    if _ao_skip_was is not None:
        _ao_head.skip_unread_outputs = _ao_skip_was

    # Convert tensors to numpy arrays for visualization.
    pts3ds_to_vis = [p.cpu().numpy() for p in pts3ds_other]
    colors_to_vis = [c.cpu().numpy() for c in colors]
    conf = [c.cpu() for c in conf]
    edge_colors = [None] * len(pts3ds_to_vis)

    if getattr(args, "no_viewer", False):
        return

    # Create and run the point cloud viewer.
    print("Launching point cloud viewer...")
    viewer = PointCloudViewer(
        model,
        state_args,
        pts3ds_to_vis,
        colors_to_vis,
        conf,
        cam_dict,
        device=device,
        edge_color_list=edge_colors,
        show_camera=True,
        vis_threshold=args.vis_threshold,
        size = args.size
    )
    viewer.run()


def run_sequences(args):
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    add_path_to_dust3r(args.model_path)
    from src.dust3r.model import ARCroco3DStereo

    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()

    base_output_dir = args.output_dir
    for _ao_seq_path in args.seq_paths:
        seq_args = deepcopy(args)
        seq_args.device = device
        seq_args.seq_path = _ao_seq_path
        seq_args.no_viewer = True
        seq_args.output_dir = os.path.join(
            base_output_dir, os.path.basename(_ao_seq_path.rstrip("/")) or "seq")
        run_inference(seq_args, model=model)


def main():
    args = parse_args()
    if args.seq_paths:
        run_sequences(args)
        return
    if not args.seq_path:
        print(
            "No inputs found! Please use our gradio demo if you would like to iteractively upload inputs."
        )
        return
    else:
        run_inference(args)


if __name__ == "__main__":
    main()
