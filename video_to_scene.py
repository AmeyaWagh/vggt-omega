#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Convert a video file to a 3D GLB scene using VGGT-Omega."""

import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from rich.console import Console
from torchvision import transforms as TF

from visual_util import predictions_to_glb
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.pose_enc import encoding_to_camera

console = Console()


def extract_frames(video_path: str, sample_fps: float) -> list[Image.Image]:
    """Read frames from a video stream into memory as RGB PIL images."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        native_fps = 1.0
    frame_interval = max(1, int(round(native_fps / sample_fps)))

    frames = []
    frame_idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        frame_idx += 1
    cap.release()

    if not frames:
        raise ValueError(f"No frames could be extracted from {video_path}")
    return frames


def preprocess_frames(
    frames: list[Image.Image],
    image_resolution: int = 512,
    patch_size: int = 16,
    mode: str = "balanced",
) -> torch.Tensor:
    """Apply the same crop/resize/pad pipeline as load_and_preprocess_images."""
    to_tensor = TF.ToTensor()
    images = []
    shapes = set()

    for frame in frames:
        frame = _crop_aspect_ratio(frame)
        w, h = frame.size
        aspect = h / max(w, 1)
        if mode == "balanced":
            th, tw = _balanced_shape(aspect, image_resolution, patch_size)
        else:
            th, tw = _max_size_shape(aspect, image_resolution, patch_size)
        frame = frame.resize((tw, th), Image.Resampling.BICUBIC)
        t = to_tensor(frame)
        shapes.add((t.shape[1], t.shape[2]))
        images.append(t)

    if len(shapes) > 1:
        images = _pad_to_common_size(images, shapes)

    return torch.stack(images)


def _crop_aspect_ratio(image: Image.Image, min_ar: float = 0.5, max_ar: float = 2.0) -> Image.Image:
    w, h = image.size
    ar = h / max(w, 1)
    if ar < min_ar:
        cw = min(w, max(1, int(round(h / min_ar))))
        left = max((w - cw) // 2, 0)
        return image.crop((left, 0, left + cw, h))
    if ar > max_ar:
        ch = min(h, max(1, int(round(w * max_ar))))
        top = max((h - ch) // 2, 0)
        return image.crop((0, top, w, top + ch))
    return image


def _balanced_shape(aspect: float, resolution: int, patch_size: int) -> tuple[int, int]:
    n = (resolution // patch_size) ** 2
    wp = np.sqrt(n / aspect)
    hp = n / wp
    wp = max(1, int(np.round(wp)))
    hp = max(1, int(np.round(hp)))
    return hp * patch_size, wp * patch_size


def _max_size_shape(aspect: float, resolution: int, patch_size: int) -> tuple[int, int]:
    def snap(v):
        return max(patch_size, int(np.round(float(v) / patch_size)) * patch_size)

    if aspect >= 1.0:
        return resolution, snap(resolution / aspect)
    return snap(resolution * aspect), resolution


def _pad_to_common_size(images: list[torch.Tensor], shapes: set) -> list[torch.Tensor]:
    mh = max(s[0] for s in shapes)
    mw = max(s[1] for s in shapes)
    padded = []
    for img in images:
        ph = mh - img.shape[1]
        pw = mw - img.shape[2]
        if ph > 0 or pw > 0:
            pt, pb = ph // 2, ph - ph // 2
            pl, pr = pw // 2, pw - pw // 2
            img = torch.nn.functional.pad(img, (pl, pr, pt, pb), value=1.0)
        padded.append(img)
    return padded


def load_model(checkpoint_path: str, device: str) -> VGGTOmega:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = VGGTOmega().eval()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    return model.to(device)


def run_inference(frames: list[Image.Image], model: VGGTOmega, image_resolution: int, device: str) -> dict:
    images = preprocess_frames(frames, image_resolution=image_resolution).to(device)
    console.log(f"Input tensor: {tuple(images.shape)}")

    with torch.inference_mode():
        predictions = model(images)

    extrinsic, intrinsic = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    predictions_np = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
            if value.shape[0] == 1:
                value = value[0]
            predictions_np[key] = value

    predictions_np["world_points_from_depth"] = _unproject_depth(
        predictions_np["depth"],
        predictions_np["extrinsic"],
        predictions_np["intrinsic"],
    )
    return predictions_np


def _unproject_depth(depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [(x - cx) / fx * depth, (y - cy) / fy * depth, depth],
        axis=-1,
    )
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a video to a 3D GLB scene using VGGT-Omega.")
    parser.add_argument("--video", required=True, help="Path to input video file.")
    parser.add_argument("--checkpoint", required=True, help="Path to VGGT-Omega checkpoint (.pt).")
    parser.add_argument("--output", default=None, help="Output GLB path. Defaults to <video_stem>.glb.")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame sampling rate in fps (default: 1.0).")
    parser.add_argument("--image-resolution", type=int, default=512, help="Input image resolution (default: 512).")
    parser.add_argument("--conf-thres", type=float, default=20.0, help="Confidence threshold percentile (default: 20.0).")
    parser.add_argument("--max-points", type=int, default=1_000_000, help="Max point cloud points (default: 1000000).")
    parser.add_argument("--no-cameras", action="store_true", help="Omit camera frustums from the scene.")
    parser.add_argument("--mask-sky", action="store_true", help="Filter sky points using skyseg.onnx.")
    parser.add_argument("--mask-black-bg", action="store_true", help="Filter near-black background points.")
    parser.add_argument("--mask-white-bg", action="store_true", help="Filter near-white background points.")
    parser.add_argument("--save-predictions", action="store_true", help="Save raw predictions as .npz alongside the GLB.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    output_path = args.output or (os.path.splitext(args.video)[0] + ".glb")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    console.log(f"Loading model from [bold]{args.checkpoint}[/bold] on [bold]{args.device}[/bold]")
    model = load_model(args.checkpoint, args.device)

    console.log(f"Extracting frames at {args.fps} fps...")
    frames = extract_frames(args.video, args.fps)
    console.log(f"Extracted [bold]{len(frames)}[/bold] frames")

    console.log("Running inference...")
    predictions = run_inference(frames, model, args.image_resolution, args.device)

    if args.save_predictions:
        npz_path = os.path.splitext(output_path)[0] + "_predictions.npz"
        np.savez(npz_path, **predictions)
        console.log(f"Predictions saved to [bold]{npz_path}[/bold]")

    console.log("Building GLB scene...")
    scene = predictions_to_glb(
        predictions,
        conf_thres=args.conf_thres,
        show_cam=not args.no_cameras,
        mask_sky=args.mask_sky,
        mask_black_bg=args.mask_black_bg,
        mask_white_bg=args.mask_white_bg,
        max_points=args.max_points,
    )
    scene.export(file_obj=output_path)
    console.log(f"Scene saved to [bold green]{output_path}[/bold green]")


if __name__ == "__main__":
    main()
