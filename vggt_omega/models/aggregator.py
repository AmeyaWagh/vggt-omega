# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from vggt_omega.models.layers import Mlp, RopePositionEmbedding, SelfAttentionBlock
from vggt_omega.models.layers.vision_transformer import DinoVisionTransformer


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """Alternating-attention encoder over video frames."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 16,
        register_attention_block_indices: list[int] = [2, 6, 9, 14, 20],
        cached_layer_indices: tuple[int, ...] = (4, 11, 17, 23),
    ) -> None:
        super().__init__()

        self.patch_embed = _build_vision_transformer(patch_size=patch_size, embed_dim=embed_dim)
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=100,
            normalize_coords="max",
            dtype=torch.float32,
        )

        self.frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                )
                for _ in range(depth)
            ]
        )
        self.inter_frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.patch_size = patch_size
        self.cached_layer_indices = set(cached_layer_indices)
        self.camera_token = nn.Parameter(torch.empty(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.empty(1, 2, num_register_tokens, embed_dim))
        self.patch_token_start = 1 + num_register_tokens

        self.inter_frame_attention_types = ["global"] * depth
        for idx in register_attention_block_indices:
            if idx < 0 or idx >= depth:
                raise ValueError(f"register_attention_block_indices contains invalid block index {idx}")
            self.inter_frame_attention_types[idx] = "register"

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.camera_token, std=1e-3)
        nn.init.normal_(self.register_token, std=1e-3)

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[list[torch.Tensor | None], int]:
        """
        Args:
            images: (B, N, C, H, W)  tensor of input frames.
        Returns:
            A tuple of (aggregated_tokens_list, patch_token_start), where:
            - aggregated_tokens_list is a list of length `depth`, where each element is either a (B, N, T, D) tensor of aggregated tokens at that layer (if the layer index is in `cached_layer_indices`) or None (if the layer index is not cached).
            - patch_token_start is the index at which patch tokens start in the token sequence, which is needed by the heads to separate camera/register tokens from patch tokens.  
        """

        batch_size, num_frames, num_channels, height, width = images.shape
        if num_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {num_channels}")

        # Pre-process images.
        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(batch_size * num_frames, num_channels, height, width) # (B*N, C, H, W)

        # Reshape and expand the camera and register tokens to match the batch size and number of frames.
        # The first token in camera_tokens is a shared token that is expanded across all frames, 
        # while the second token is a per-frame token that is repeated for each frame.
        # camera_tokens: (1, 2, 1, D) -> (B*N, 2, D) where D is the embedding dimension.
        # register_tokens: (1, 2, R, D) -> (B*N, 2, R, D) -> (B*N, 2*R, D) where R is the number of register tokens.
        camera_token = slice_expand_and_flatten(self.camera_token, batch_size, num_frames)
        register_token = slice_expand_and_flatten(self.register_token, batch_size, num_frames)

        # Encode the images into patch tokens using DinoViT. 
        # The patch embedding module will normalize the patch tokens, 
        # so we don't need to do any additional normalization here.
        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]


        # Concatenate the camera token, register tokens, and patch tokens to form the input token sequence for the frame blocks.
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        # scene_tokens: (B*N, 1 + R + P, D) where R is the number of register tokens and P is the number of patch tokens.
        _, num_tokens, embed_dim = tokens.shape


        # Compute RoPE embeddings for the frame blocks. 
        # The RoPE embeddings are computed based on the spatial dimensions of the input frames, 
        # which are derived from the height and width of the input images and the patch size. 
        # The same RoPE embeddings are used for all frame blocks, 
        # since they all operate on the same spatial dimensions.
        patch_grid_size = (height // self.patch_size, width // self.patch_size)
        with torch.no_grad():
            rope_sin, rope_cos = self.rope_embed(H=patch_grid_size[0], W=patch_grid_size[1])
            frame_rope = (
                rope_sin.to(device=patch_tokens.device, dtype=torch.float32),
                rope_cos.to(device=patch_tokens.device, dtype=torch.float32),
            )

        outputs: list[torch.Tensor | None] = []
        for block_idx in range(self.depth):
            # Run the frame block, which applies self-attention and MLP to each frame independently.
            tokens, frame_tokens = self._run_frame_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                frame_rope,
            )

            # Run the inter-frame attention block, which applies self-attention across frames.
            tokens = self._run_inter_frame_attention_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                self.inter_frame_attention_types[block_idx],
            )
            if block_idx in self.cached_layer_indices:
                outputs.append(torch.cat([frame_tokens, tokens], dim=-1))
            else:
                outputs.append(None)

        return outputs, self.patch_token_start

    def _run_frame_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        rope_sincos: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs a single frame block, which applies self-attention and MLP to each frame independently.
        
        Args:
            tokens: (B*N, T, D) tensor of input tokens for the block, 
                where T is the total number of tokens (camera + register + patch) and D is the embedding dimension.
            batch_size: The batch size (B) of the input frames.
            num_frames: The number of frames (N) in the input.
            num_tokens: The total number of tokens (T) in the input token sequence.
            embed_dim: The embedding dimension (D) of the input tokens.
            block_idx: The index of the current block, used to select the appropriate frame block module.
            rope_sincos: A tuple of (rope_sin, rope_cos) tensors containing the pre-computed RoPE embeddings for 
                the spatial dimensions of the input frames, which are passed to the frame block for use in 
                the attention mechanism.
        Returns:
            A tuple of (tokens, frame_tokens), where:
            - tokens is a (B*N, T, D) tensor of output tokens from the frame block, which will be passed to 
                the inter-frame attention block.
            - frame_tokens is a (B, N, T, D) tensor of the same tokens reshaped to separate the batch and frame dimensions, 
                which will be cached for output if the block index is in `cached_layer_indices`.
        """
        tokens = tokens.view(batch_size * num_frames, num_tokens, embed_dim)
        tokens = self.frame_blocks[block_idx](tokens, rope_sincos)
        return tokens, tokens.view(batch_size, num_frames, num_tokens, embed_dim)

    def _run_inter_frame_attention_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        attention_type: str,
    ) -> torch.Tensor:
        """Runs a single inter-frame attention block, which applies self-attention across frames.
        Args:
            tokens: (B*N, T, D) tensor of input tokens for the block, 
                where T is the total number of tokens (camera + register + patch) and D is the embedding dimension.
            batch_size: The batch size (B) of the input frames.
            num_frames: The number of frames (N) in the input.
            num_tokens: The total number of tokens (T) in the input token sequence.
            embed_dim: The embedding dimension (D) of the input tokens.
            block_idx: The index of the current block, used to select the appropriate inter-frame attention block module.
            attention_type: The type of inter-frame attention to apply, 
                which can be either "global" for attention across all tokens, or "register" for attention only across the camera and
                register tokens. The attention type is determined by the `inter_frame_attention_types` list, 
                which is set in the constructor based on the `register_attention_block_indices` parameter.
        Returns:
            A (B*N, T, D) tensor of output tokens from the inter-frame attention block.
        """
        tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type == "global":
            tokens = tokens.view(batch_size, num_frames * num_tokens, embed_dim)
            tokens = self.inter_frame_blocks[block_idx](tokens, None)
            return tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type != "register":
            raise ValueError(f"Unknown inter-frame attention type: {attention_type}")

        patch_token_start = self.patch_token_start
        camera_and_register_tokens = tokens[:, :, :patch_token_start].reshape(
            batch_size,
            num_frames * patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, :, patch_token_start:].reshape(
            batch_size,
            num_frames * (num_tokens - patch_token_start),
            embed_dim,
        )

        camera_and_register_tokens = self.inter_frame_blocks[block_idx](camera_and_register_tokens, None)
        tokens = torch.cat([camera_and_register_tokens, patch_tokens], dim=1)

        camera_and_register_tokens = tokens[:, : num_frames * patch_token_start].view(
            batch_size,
            num_frames,
            patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, num_frames * patch_token_start :].view(
            batch_size,
            num_frames,
            num_tokens - patch_token_start,
            embed_dim,
        )
        return torch.cat([camera_and_register_tokens, patch_tokens], dim=2)


def _build_vision_transformer(patch_size: int, embed_dim: int) -> DinoVisionTransformer:
    model = DinoVisionTransformer(
        img_size=224,
        patch_size=patch_size,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="max",
        pos_embed_rope_dtype="fp32",
        embed_dim=embed_dim,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-5,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    )
    model.init_weights()
    return model


def slice_expand_and_flatten(token_tensor: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
    """ Helper function to slice and expand tokens.

    Slices the input token tensor to select the first token, expands it across the frame dimension, 
    and flattens it back to (B*N, T, D) shape.
    Args:
        token_tensor: (1, 2, T, D) tensor of tokens to be sliced and expanded.
        batch_size: The batch size (B) of the input frames.
        num_frames: The number of frames (N) in the input.
    Returns:
        A (B*N, T, D) tensor where the first token from token_tensor is expanded across the frame dimension 
        for each item in the batch, and the rest of the tokens are repeated for each frame.
    """
    first_frame_token = token_tensor[:, 0:1].expand(batch_size, 1, *token_tensor.shape[2:])
    other_frame_tokens = token_tensor[:, 1:].expand(batch_size, num_frames - 1, *token_tensor.shape[2:])
    tokens = torch.cat([first_frame_token, other_frame_tokens], dim=1)
    return tokens.view(batch_size * num_frames, *tokens.shape[2:])
