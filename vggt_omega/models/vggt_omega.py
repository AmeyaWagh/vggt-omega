# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
import torch.nn as nn

from vggt_omega.models.aggregator import Aggregator
from vggt_omega.models.heads import CameraHead, DenseHead, TextAlignmentHead
from vggt_omega.models.constants import ModelOutputKeys

class VGGTOmega(nn.Module):
    """Minimal VGGT-Omega inference model for camera and depth prediction."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        enable_camera: bool = True,
        enable_depth: bool = True,
        enable_alignment: bool = False,
    ) -> None:
        super().__init__()

        self.aggregator = Aggregator(patch_size=patch_size, embed_dim=embed_dim)
        _warn_if_rope_not_max(self.aggregator)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.dense_head = DenseHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_depth else None
        self.text_alignment_head = TextAlignmentHead(dim_in=2 * embed_dim) if enable_alignment else None

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        # images: (N, C, H, W) or (B, N, C, H, W)
        if len(images.shape) == 4:
            images = images.unsqueeze(0) # Add batch dimension if missing, resulting in shape (1, N, C, H, W)

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            aggregated_tokens_list, patch_token_start = self.aggregator(images)

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which VGGTOmega needs.")

        predictions: dict[str, torch.Tensor] = {}
        predictions[ModelOutputKeys.CAMERA_AND_REGISTER_TOKENS] = final_tokens[:, :, :patch_token_start].contiguous()
        with torch.autocast(device_type="cuda", enabled=False):

            # Predict camera parameters from the camera/register tokens using the camera head, if it is enabled.
            camera_params = self.predict_camera_parameters(aggregated_tokens_list, patch_token_start=patch_token_start)
            predictions[ModelOutputKeys.POSE_ENC] = camera_params

            # Predict depth maps from the patch tokens using the dense head, if it is enabled.
            depth, depth_conf = self.predict_depth(aggregated_tokens_list, images=images, patch_token_start=patch_token_start)
            predictions[ModelOutputKeys.DEPTH] = depth
            predictions[ModelOutputKeys.DEPTH_CONFIDENCE] = depth_conf

            # Predict text-alignment from the camera/register tokens using the text alignment head, if it is enabled.
            text_alignment_outputs = self.predict_text_alignment(aggregated_tokens_list, patch_token_start=patch_token_start)
            predictions[ModelOutputKeys.TEXT_ALIGNMENT] = text_alignment_outputs

        if not self.training:
            predictions[ModelOutputKeys.IMAGES] = images # (B, N, C, H, W)
        return predictions
    
    def predict_camera_parameters(self, aggregated_tokens_list: list[torch.Tensor], patch_token_start: int) -> torch.Tensor | None:
        if self.camera_head is None:
            return None
        return self.camera_head(aggregated_tokens_list, patch_token_start=patch_token_start)
    
    def predict_depth(self, aggregated_tokens_list: list[torch.Tensor], images: torch.Tensor, patch_token_start: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if self.dense_head is None:
            return None, None
        return self.dense_head(aggregated_tokens_list, images=images, patch_token_start=patch_token_start)
    
    def predict_text_alignment(self, aggregated_tokens_list: list[torch.Tensor], patch_token_start: int) -> dict[str, torch.Tensor] | None:
        if self.text_alignment_head is None:
            return None
        return self.text_alignment_head(aggregated_tokens_list, patch_token_start=patch_token_start)




def _warn_if_rope_not_max(aggregator: nn.Module) -> None:
    for name, module in (("aggregator.patch_embed", aggregator.patch_embed), ("aggregator", aggregator)):
        rope_embed = getattr(module, "rope_embed", None)
        normalize_coords = getattr(rope_embed, "normalize_coords", None)
        if normalize_coords != "max":
            warnings.warn(
                f"{name} RoPE normalize_coords is {normalize_coords!r}; "
                "the released VGGT-Omega checkpoint was trained with 'max'.",
                stacklevel=2,
            )
