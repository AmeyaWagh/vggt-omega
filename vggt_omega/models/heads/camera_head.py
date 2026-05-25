# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.models.layers import SelfAttentionBlock


class CameraHead(nn.Module):
    """Camera head used by the released VGGT-Omega checkpoints."""

    def __init__(self, dim_in: int = 2048) -> None:
        super().__init__()

        self.token_norm = nn.LayerNorm(dim_in, eps=1e-5)
        # Head-local transformer blocks that mix camera and register tokens across frames.
        self.trunk = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=dim_in,
                    num_heads=16,
                    ffn_ratio=4.0,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    init_values=1e-5,
                    use_qk_norm=False,
                    mask_k_bias=True,
                )
                for _ in range(4)
            ]
        )
        self.trunk_norm = nn.LayerNorm(dim_in, eps=1e-5)
        self.camera_branch = nn.Sequential(
            nn.Linear(dim_in, dim_in // 2, bias=True),
            nn.GELU(),
            nn.Linear(dim_in // 2, 9, bias=True),
        )

    def forward(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        patch_token_start: int,
    ) -> torch.Tensor:
        """ Predict camera parameters from the camera/register tokens in the aggregated tokens list.
         The camera/register tokens are expected to be in the first part of the token dimension,
         before the patch tokens which start at patch_token_start. The camera head applies several transformer blocks
         to allow information exchange across frames, and then predicts the camera parameters from the first token (index 0) 
         of the camera/register tokens, which is designated as the camera token.
        """
        tokens = aggregated_tokens_list[-1]
        if tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which CameraHead needs.")
        batch_size, num_frames, num_tokens, _ = tokens.shape

        if patch_token_start is None:
            raise ValueError("patch_token_start is required for CameraHead")
        if patch_token_start > num_tokens:
            raise ValueError(f"patch_token_start ({patch_token_start}) exceeds token length ({num_tokens})")

        if tokens.dtype != torch.float32:
            tokens = tokens.float()

        # Access the camera/register tokens, which are expected to be in the first part of the token dimension before the patch tokens.
        camera_and_register_tokens = tokens[:, :, :patch_token_start]

        # Normalize the camera/register tokens before feeding into the transformer blocks, 
        # which empirically improves training stability.
        camera_and_register_tokens = self.token_norm(camera_and_register_tokens)

        # (B, F, T, D) -> (B, F*T, D) where
        # F - number of frames, T - number of camera/register tokens, D - token dimension.  
        camera_and_register_tokens = camera_and_register_tokens.reshape(batch_size, num_frames * patch_token_start, -1)
        rope_sincos = None # The released VGGT-Omega checkpoints do not use ROPE in the camera head, so we pass None here.
        for block in self.trunk:
            camera_and_register_tokens = block(camera_and_register_tokens, rope_sincos)

        # (B, F*T, D) -> (B, F, T, D)
        camera_and_register_tokens = camera_and_register_tokens.reshape(batch_size, num_frames, patch_token_start, -1)

        # Normalize the camera tokens before the final prediction layer.
        camera_tokens = self.trunk_norm(camera_and_register_tokens[:, :, 0])
        
        # Predict camera parameters from the camera token (the first token of the camera/register tokens).
        # (translation (3), quaternion (4), fov (1)) = 8 parameters in total.
        camera_parameters = _apply_camera_activation(self.camera_branch(camera_tokens))

        # Final shape is (B, F, 8) where the 8 parameters are (translation (3), quaternion (4), fov (2)).
        return camera_parameters


def _apply_camera_activation(raw_camera: torch.Tensor) -> torch.Tensor:
    translation = raw_camera[..., :3]
    quaternion = raw_camera[..., 3:7]
    fov = F.relu(raw_camera[..., 7:]) + 0.01
    return torch.cat([translation, quaternion, fov], dim=-1)
