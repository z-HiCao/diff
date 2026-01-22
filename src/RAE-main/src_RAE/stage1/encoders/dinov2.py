from transformers import Dinov2WithRegistersModel
from torch import nn
import torch
from math import *
from . import register_encoder


@register_encoder()
class Dinov2withNorm(nn.Module):
    def __init__(
        self,
        dinov2_path: str,
        normalize: bool = True,
    ):
        super().__init__()
        # Support both local paths and HuggingFace model IDs
        try:
            self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path, local_files_only=True)
        except (OSError, ValueError, AttributeError):
            self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path, local_files_only=False)
        self.encoder.requires_grad_(False)
        if normalize:
            self.encoder.layernorm.elementwise_affine = False
            self.encoder.layernorm.weight = None
            self.encoder.layernorm.bias = None
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size
         
    def dinov2_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x, output_hidden_states=True)
        unused_token_num = 5  # 1 CLS + 4 register tokens
        image_features = x.last_hidden_state[:, unused_token_num:]
        return image_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 12, 224, 224]
        B = x.size(0)

        # 1. Conv2d patch embedding（这是关键）
        x = self.encoder.embeddings.patch_embeddings.projection(x)
        # x: [B, 768, 16, 16]

        # 2. flatten → tokens
        x = x.flatten(2).transpose(1, 2)
        # x: [B, 256, 768]

        # 3. CLS + register + pos embed
        cls = self.encoder.embeddings.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        if self.encoder.embeddings.register_tokens is not None:
            reg = self.encoder.embeddings.register_tokens.expand(B, -1, -1)
            x = torch.cat([x, reg], dim=1)

        x = x + self.encoder.embeddings.position_embeddings[:, : x.size(1)]
        x = self.encoder.embeddings.dropout(x)

        # 4. Transformer
        out = self.encoder.encoder(x, output_hidden_states=True)

        image_features = out.last_hidden_state[:, 5:]
        return image_features

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     # x: (B, 12, H, W)
    #     B, C, H, W = x.shape
    #     patch_size = self.patch_size
    #     H_patch, W_patch = H // patch_size, W // patch_size
    #     # 1. 手工 patchify + projection
    #     x = x.reshape(B, C, H_patch, patch_size, W_patch, patch_size)
    #     x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, H_patch * W_patch, C * patch_size * patch_size)
    #     # 2. 线性映射（和你前面换好的 12->768 Conv2d 等价）
    #     x = self.encoder.embeddings.patch_embeddings.projection(x)   # (B, N, 768)
    #     # 3. 加位置编码、cls/reg token
    #     x = self.encoder.embeddings.forward_with_pos_embed(x)        # 该函数已帮你拼好 cls/reg
    #     # 4. 扔给 Transformer
    #     out = self.encoder.encoder(x, output_hidden_states=True)
    #     unused_token_num = 5
    #     image_features = out.last_hidden_state[:, unused_token_num:]
    #     return image_features
    #     # return self.dinov2_forward(x)

