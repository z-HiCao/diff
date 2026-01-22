import torch
import torch.nn as nn
from .decoders import GeneralDecoder
from .encoders import ARCHS
from transformers import AutoConfig, AutoImageProcessor
from typing import Optional
from math import sqrt
from typing import Protocol
import torch.nn.functional as F

class Stage1Protocal(Protocol):
    # must have patch size attribute
    patch_size: int
    hidden_size: int 
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ...

class RAE(nn.Module):
    def __init__(self, 
        # ---- encoder configs ----
        encoder_cls: str = 'Dinov2withNorm',
        encoder_config_path: str = 'facebook/dinov2-base',
        encoder_input_size: int = 224,
        encoder_params: dict = {},
        # ---- decoder configs ----
        decoder_config_path: str = 'vit_mae-base',
        decoder_patch_size: int = 16,
        pretrained_decoder_path: Optional[str] = None,
        # ---- noising, reshaping and normalization-----
        noise_tau: float = 0.8,
        reshape_to_2d: bool = True,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        encoder_cls = ARCHS[encoder_cls]
        self.encoder: Stage1Protocal = encoder_cls(**encoder_params)
        print(f"encoder_config_path: {encoder_config_path}")
         # ===== 2. 原地扩成 12 通道 =====
        self._expand_input_channels(num_groups=4)

        proc = AutoImageProcessor.from_pretrained(encoder_config_path)
       
        self.encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
        self.encoder_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
        encoder_config = AutoConfig.from_pretrained(encoder_config_path)
        # see if the encoder has patch size attribute            
        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size
        assert self.encoder_input_size % self.encoder_patch_size == 0, f"encoder_input_size {self.encoder_input_size} must be divisible by encoder_patch_size {self.encoder_patch_size}"
        self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2 # number of patches of the latent
        
        # decoder
        decoder_config = AutoConfig.from_pretrained('/opt/data/private/wjy/LRY/DiffSplat-main/src/RAE-main/configs/config.json') #RESET
        decoder_config.hidden_size = self.latent_dim # set the hidden size of the decoder to be the same as the encoder's output
        decoder_config.patch_size = decoder_patch_size
        decoder_config.image_size = int(decoder_patch_size * sqrt(self.base_patches)) 
        self.decoder = GeneralDecoder(decoder_config, num_patches=self.base_patches)

        #ADD
        # # ---------------- CNN Refinement Head ----------------
        # self.refine_cnn = nn.Sequential(
        #     nn.Conv2d(12, 64, kernel_size=3, padding=1), #12 是output channel数
        #     nn.GELU(),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.GELU(),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.GELU(),
        # )

        # # ------------ Multi-Head Prediction ------------
        # self.rgb_head = nn.Conv2d(64, 3, kernel_size=1)        # (0–2)
        # self.scale_head = nn.Conv2d(64, 3, kernel_size=1)      # (3–5)
        # self.rot_head = nn.Conv2d(64, 4, kernel_size=1)        # (6–9)
        # self.opacity_head = nn.Conv2d(64, 1, kernel_size=1)    # (10)
        # self.depth_head = nn.Conv2d(64, 1, kernel_size=1)      # (11)

        # load pretrained decoder weights
        if pretrained_decoder_path is not None:
            print(f"Loading pretrained decoder from {pretrained_decoder_path}")
            state_dict = torch.load(pretrained_decoder_path, map_location='cpu')
            keys = self.decoder.load_state_dict(state_dict, strict=False)
            if len(keys.missing_keys) > 0:
                print(f"Missing keys when loading pretrained decoder: {keys.missing_keys}")
        self.noise_tau = noise_tau
        self.reshape_to_2d = reshape_to_2d
        if normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location='cpu')
            self.latent_mean = stats.get('mean', None)
            self.latent_var = stats.get('var', None)
            self.do_normalization = True
            self.eps = eps
            print(f"Loaded normalization stats from {normalization_stat_path}")
        else:
            self.do_normalization = False
    def noising(self, x: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand((x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device)
        noise = noise_sigma * torch.randn_like(x)
        return x + noise

    def encode(self, x: torch.Tensor) -> torch.Tensor:#x[b*v,12,h,w]
        # normalize input
         #应当只对RGB维度，用mean/std归一化
        # _, _, h, w = x.shape
        rgb = x[:, :3]
        other = x[:, 3:]

        rgb = (rgb - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)
        x = torch.cat([rgb, other], dim=1)
        #注释掉的为源代码
        # if h != self.encoder_input_size or w != self.encoder_input_size:
        #     x = nn.functional.interpolate(x, size=(self.encoder_input_size, self.encoder_input_size), mode='bicubic', align_corners=False)
        # x = (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)
        z = self.encoder(x)
        if self.training and self.noise_tau > 0:
            z = self.noising(z)
        if self.reshape_to_2d:
            b, n, c = z.shape
            h = w = int(sqrt(n))
            assert h * h == n, f"Token number {n} is not a square"
            z = z.transpose(1, 2).view(b, c, h, w)
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = z * torch.sqrt(latent_var + self.eps) + latent_mean
        if self.reshape_to_2d:
            b, c, h, w = z.shape
            n = h * w
            z = z.view(b, c, n).transpose(1, 2)
        output = self.decoder(z, drop_cls_token=False).logits
        x_rec = self.decoder.unpatchify(output) #输出12维
        # x_rec = x_rec * self.encoder_std.to(x_rec.device) + self.encoder_mean.to(x_rec.device)
        # 只对RGB通道做std标准化
        rgb = x_rec[:, :3]
        other = x_rec[:, 3:]  # shape [B, 9, H, W]
        rgb = rgb * self.encoder_std.to(rgb.device) + self.encoder_mean.to(rgb.device)
        x_rec = torch.cat([rgb, other], dim=1)

        # #=======CNN refinement + Multi MLP head======
        # feat = self.refine_cnn(x_rec) #[B,64,H,W]
        #  # ---- Multi-head parameter prediction ----
        rgb = torch.sigmoid(rgb)              # (B,3,H,W)
        scale = F.softplus(other[:,:3])             # (B,3,H,W), >0
        rotation = other[:,3:7]                       # (B,4,H,W), raw quat
        opacity = torch.sigmoid(other[:,7:8])      # (B,1,H,W) in (0,1)
        depth = other[:,8:9]                        # (B,1,H,W), free range
        final = torch.cat([rgb, scale, rotation, opacity, depth], dim=1)

        return final
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec
    
    # 复制权重的函数
    def _expand_input_channels(self, num_groups: int = 4):
        # 1. 找对层
        proj = self.encoder.encoder.embeddings.patch_embeddings.projection
        # assert isinstance(proj, nn.Conv2d) and proj.in_channels == 3

        # 2. 新建 12 通道卷积
        new_proj = nn.Conv2d(
            in_channels=3 * num_groups,
            out_channels=proj.out_channels,
            kernel_size=proj.kernel_size,
            stride=proj.stride,
            padding=proj.padding,
            bias=(proj.bias is not None),
        )

        # 3. 复制权重
        with torch.no_grad():
            new_weight = torch.cat([proj.weight.data] * num_groups, dim=1)
            new_proj.weight.copy_(new_weight)
            if proj.bias is not None:
                new_proj.bias.copy_(proj.bias.data)

        # 4. 原位替换
        self.encoder.encoder.embeddings.patch_embeddings.projection = new_proj