#!/usr/bin/env python3
"""
Run a stage-1 RAE reconstruction from a config file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs
from stage1 import RAE

import sys
sys.path.append("/opt/data/private/wjy/LRY/DiffSplat-main")
from src.data import GObjaverseParquetDataset, ParquetChunkDataSource, MultiEpochsChunkedDataLoader, yield_forever
from src.models import GSAutoencoderKL, GSRecon, get_optimizer, get_lr_scheduler
import src.utils.util as util
import src.utils.vis_util as vis_util
from src.options import opt_dict
import accelerate
from accelerate import Accelerator
from accelerate import DataLoaderConfiguration, DeepSpeedPlugin
import torch.nn.functional as F
import os
from accelerate.logging import get_logger as get_accelerate_logger
import logging
import src.utils.geo_util as geo_util
import src.utils.vis_util as vis_util
from einops import rearrange, repeat
DEFAULT_IMAGE = Path("assets/pixabay_cat.png")


def get_device(explicit: str | None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    tensor = transforms.ToTensor()(image).unsqueeze(0)  # (1, C, H, W)
    return tensor


def reconstruct(rae: RAE, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        latent = rae.encode(image)
        recon = rae.decode(latent)
    return latent, recon



def reconstruct_splat(rae, frames_flat: torch.Tensor, original_splat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回 latent, recon, diff, mean_diff
    """
    rae.eval()
    device = frames_flat.device
    B, V, C, H, W = original_splat.shape
    F = C // 3
    assert C == F * 3, f"Expected C to be divisible by 3, got {C}"

    with torch.no_grad():
        latent = rae.encode(frames_flat)
        recon_flat = rae.decode(latent)

    recon = recon_flat.view(B, V, F, 3, H, W).reshape(B, V, C, H, W)

    recon = recon*2.0 -1.0 #变回（-1，1）
    # L2 norm over channel dimension
    diff = torch.norm(original_splat - recon, p=2, dim=2)  # (B, V, H, W)

    mse_per_view = ((original_splat - recon) ** 2).mean(dim=[2,3,4])  # (B,V)
    print("MSE per view:", mse_per_view)
    print("Mean MSE over batch:", mse_per_view.mean().item())
    mean_mse = mse_per_view.mean()

     # ================================
    # ⭐ 新增：RGB MSE（只比较前 3 channel）
    # ================================
    recon_rgb = recon[:, :, :3]              # (B, V, 3, H, W)
    original_rgb = original_splat[:, :, :3]  # (B, V, 3, H, W)

    mse_rgb_per_view = ((original_rgb - recon_rgb) ** 2).mean(dim=[2,3,4])  # (B,V)
    print("MSE per view (RGB only):", mse_rgb_per_view)
    print("Mean MSE over batch (RGB only):", mse_rgb_per_view.mean().item())
    mean_mse_rgb = mse_rgb_per_view.mean()

    return latent, recon, diff,  mean_mse, mean_mse_rgb

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct an input image using a Stage-1 RAE loaded from config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config with a stage_1 section.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"Input image to reconstruct (default: {DEFAULT_IMAGE}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("recon.png"),
        help="Where to save the reconstructed image (default: recon.png).",
    )
    parser.add_argument(
        "--device",
        help="Torch device to use (e.g. cuda, cuda:1, cpu). Auto-detect if omitted.",
    )
    args = parser.parse_args()

    device = get_device(args.device)

    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    rae_config, *_ = parse_configs(args.config)
    if rae_config is None:
        raise ValueError(
            f"No stage_1 section found in config {args.config}. "
            "Please supply a config with a stage_1 target."
        )

    torch.set_grad_enabled(False)
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()

    image = load_image(args.image).to(device)
    latent, recon = reconstruct(rae, image)

    image_resized = F.interpolate(image, size=(256, 256), mode="bilinear", align_corners=False)


    recon = recon.clamp(0.0, 1.0)
    mse = ((image_resized - recon) ** 2).mean()
    print("MSE:", mse.item())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(recon, args.output)

    print(f"Saved reconstruction to {args.output.resolve()}")
    print(f"Input shape: {tuple(image.shape)}, latent shape: {tuple(latent.shape)}, recon shape: {tuple(recon.shape)}")

def save_render_images(render_images, save_dir="renders", start_idx=1):
    """
    保存渲染图像为顺序编号 PNG 文件。
    
    Args:
        render_images: torch.Tensor, shape (B, V, 3, H, W)
        save_dir: 保存目录
        start_idx: 起始编号
    Returns:
        next_idx: 下次保存的起始编号
    """
    os.makedirs(save_dir, exist_ok=True)

    B, V, C, H, W = render_images.shape
    imgs = render_images.clamp(0, 1).cpu()
    imgs = imgs.reshape(-1, C, H, W)  # flatten to (B*V, 3, H, W)

    idx = start_idx
    for img in imgs:
        img = img.permute(1, 2, 0)          # -> (H, W, 3)
        img = (img * 255).byte().numpy()
        filename = os.path.join(save_dir, f"{idx:06d}.png")
        Image.fromarray(img).save(filename)
        idx += 1

    print(f"保存完成: {save_dir}, 从 {start_idx} 到 {idx-1}")
    return idx  # 返回下次起始编号

def test_splat_method1_singleImage():
    '''测试splat输入'''
    parser = argparse.ArgumentParser(
        description="Reconstruct an input image using a Stage-1 RAE loaded from config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config with a stage_1 section.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"Input image to reconstruct (default: {DEFAULT_IMAGE}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("recon.png"),
        help="Where to save the reconstructed image (default: recon.png).",
    )
    parser.add_argument(
        "--device",
        help="Torch device to use (e.g. cuda, cuda:1, cpu). Auto-detect if omitted.",
    )
    args = parser.parse_args()

    device = get_device(args.device)

    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    rae_config, *_ = parse_configs(args.config)
    if rae_config is None:
        raise ValueError(
            f"No stage_1 section found in config {args.config}. "
            "Please supply a config with a stage_1 target."
        )

    torch.set_grad_enabled(False)
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()
    opt = opt_dict["gsvae"]
    
    gsrecon = GSRecon(opt).to(device)
    gsrecon = gsrecon.requires_grad_(False)
    gsrecon = gsrecon.eval()
    
    image = load_image(args.image).to(device)
    gsrecon_outputs = gsrecon.forward_gaussians(
                image,
                input_C2W,
                input_fxfycxcy,
            )
    splat = torch.cat([
            gsrecon_outputs["rgb"],
            gsrecon_outputs["scale"],
            gsrecon_outputs["rotation"],
            gsrecon_outputs["opacity"],
            gsrecon_outputs["depth"],
        ], dim=2)  # (1, V_in, C=12, H, W)
    B, V,C, H, W = splat.shape
    F = 4
    frames = splat.view(B, V, F, 3, H, W)
    # flatten batch → (B*V*4, 3, H, W)
    frames_flat = frames.reshape(B * V * F, 3, H, W)

    latent, recon = reconstruct(rae, image)

    recon = recon.clamp(0.0, 1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(recon, args.output)

    print(f"Saved reconstruction to {args.output.resolve()}")
    print(f"Input shape: {tuple(image.shape)}, latent shape: {tuple(latent.shape)}, recon shape: {tuple(recon.shape)}")

def test_splat_method1(dtype: torch.dtype = torch.float32):
    '''测试splat输入'''
    parser = argparse.ArgumentParser(
        description="Reconstruct an input image using a Stage-1 RAE loaded from config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config with a stage_1 section.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"Input image to reconstruct (default: {DEFAULT_IMAGE}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("recon.png"),
        help="Where to save the reconstructed image (default: recon.png).",
    )
    parser.add_argument(
        "--device",
        default=0,
        help="Torch device to use (e.g. cuda, cuda:1, cpu). Auto-detect if omitted.",
    )
    parser.add_argument(
        "--load_pretrained_gsrecon",
        type=str,
        default="gsrecon_gobj265k_cnp_even4",
        # default=None,
        help="Tag of a pretrained GSRecon in this project"
    )
    parser.add_argument(
        "--load_pretrained_gsrecon_ckpt",
        type=int,
        default=-1,
        help="Iteration of the pretrained GSRecon checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--hdfs_dir",
        type=str,
        default=None,
        help="Path to the HDFS directory to save checkpoints"
    )
    args = parser.parse_args()

    device = get_device(args.device)


    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    rae_config, *_ = parse_configs(args.config)
    if rae_config is None:
        raise ValueError(
            f"No stage_1 section found in config {args.config}. "
            "Please supply a config with a stage_1 target."
        )

    torch.set_grad_enabled(False)
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()
    opt = opt_dict["gsvae"]
    # GSRecon
    gsrecon = GSRecon(opt)
    gsrecon = gsrecon.requires_grad_(False)
    gsrecon = gsrecon.eval()
    V_in = opt.num_input_views

    if args.load_pretrained_gsrecon is not None:
        print(f"Load GSRecon checkpoint from [{args.load_pretrained_gsrecon}] iteration [{args.load_pretrained_gsrecon_ckpt:06d}]\n")
        gsrecon = util.load_ckpt(
            os.path.join(args.output_dir, args.load_pretrained_gsrecon, "checkpoints"),
            args.load_pretrained_gsrecon_ckpt,
            None if args.hdfs_dir is None else os.path.join(args.project_hdfs_dir, args.load_pretrained_gsrecon),
            gsrecon, None
        ).to(device)
        
    
    val_dataset = GObjaverseParquetDataset(
        data_source=ParquetChunkDataSource("./dataset/val", "test"),
        shuffle=True,  # shuffle for various visualization
        shuffle_buffer_size=-1,  # `-1`: not shuffle actually
        chunks_queue_max_size=1,  # number of preloading chunks
        # GObjaverse
        opt=opt,
        training=False,
    )
    val_loader = MultiEpochsChunkedDataLoader(
        val_dataset,
        batch_size=4,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )

    # 初始化保存编号
    next_render_idx = 1
    next_renderbefore_idx = 1
    for batch in yield_forever(val_loader):
        image = batch["image"].to(device=device, dtype=dtype)
        C2W = batch["C2W"].to(device=device, dtype=dtype)
        fxfycxcy = batch["fxfycxcy"].to(device=device, dtype=dtype)

        V_in, V_cond, V = opt.num_input_views, opt.num_cond_views, opt.num_views

        # ------------------- Condition Views -------------------
        cond_idx = [0]  # 第一个视角必须在输入中
        if V_cond > 1:
            cond_idx += np.random.choice(range(1, V), V_cond-1, replace=False).tolist()

        imgs_out = image
        imgs_cond = image[:, :, cond_idx, ...].squeeze(1)  # (B, V_cond, 3, H, W)
        B = imgs_cond.shape[0]

        # ------------------- Squeeze 多余维度 -------------------
        if C2W.dim() == 5 and C2W.shape[1] == 1:
            C2W = C2W.squeeze(1)
            fxfycxcy = fxfycxcy.squeeze(1)
            imgs_out = imgs_out.squeeze(1)

        # ------------------- 切分输入 -------------------
        input_image = imgs_out[:, :V_in, ...].to(device=device, dtype=dtype)
        input_C2W = C2W[:, :V_in, ...].to(device=device, dtype=dtype)
        input_fxfycxcy = fxfycxcy[:, :V_in, ...].to(device=device, dtype=dtype)
        cond_C2W = C2W[:, cond_idx, ...].to(device=device, dtype=dtype)
        cond_fxfycxcy = fxfycxcy[:, cond_idx, ...].to(device=device, dtype=dtype)

        # ------------------- Normal / Coord -------------------
        if opt.input_normal:
            normal_map = batch["normal"][:, :,:V_in, ...].to(device=device, dtype=dtype).contiguous()
            normal_map = normal_map.squeeze(1)
            input_image = torch.cat([input_image, normal_map], dim=2)
        if opt.input_coord:
            coord_map = batch["coord"][:,:, :V_in, ...].to(device=device, dtype=dtype).contiguous()
            coord_map = coord_map.squeeze(1)
            input_image = torch.cat([input_image, coord_map], dim=2)

        # ------------------- Plucker Embeddings -------------------
        if opt.input_concat_plucker:
            H = W = opt.input_res
            plucker, _ = geo_util.plucker_ray(H, W, input_C2W, input_fxfycxcy)
            if opt.view_concat_condition:
                cond_plucker, _ = geo_util.plucker_ray(H, W, cond_C2W, cond_fxfycxcy)
                plucker = torch.cat([cond_plucker, plucker], dim=1)
            plucker = rearrange(plucker, "b v c h w -> (b v) c h w")
        else:
            plucker = None

        # ------------------- Forward -------------------
        with torch.no_grad():
            gsrecon_outputs = gsrecon.forward_gaussians(
                input_image,
                input_C2W,
                input_fxfycxcy
            )

            splat = torch.cat([
                    gsrecon_outputs["rgb"],
                    gsrecon_outputs["scale"],
                    gsrecon_outputs["rotation"],
                    gsrecon_outputs["opacity"],
                    gsrecon_outputs["depth"],
                ], dim=2)  # (B, V_in, C=12, H, W)
            B, V,C, H, W = splat.shape
            F = 4
            frames = splat.view(B, V, F, 3, H, W)
            # flatten batch → (B*V*4, 3, H, W)
            frames_flat = frames.reshape(B * V * F, 3, H, W)
            #归一化
            frames_flat = (frames_flat + 1.0) / 2.0

            latent,recon,diff,mean_diff,mean_rgb = reconstruct_splat(rae, frames_flat, original_splat=splat)
            recon_model_outputs = {
                "rgb": recon[:, :, :3, ...],
                "scale": recon[:, :, 3:6, ...],
                "rotation": recon[:, :, 6:10, ...],
                "opacity": recon[:, :, 10:11, ...],
                "depth": recon[:, :, 11:12, ...],
            }
            splat_output = {
                "rgb": splat[:, :, :3, ...],
                "scale":splat[:, :, 3:6, ...],
                "rotation": splat[:, :, 6:10, ...],
                "opacity": splat[:, :, 10:11, ...],
                "depth": splat[:, :, 11:12, ...],
            }
            input_C2W = input_C2W.squeeze(1)
            input_fxfycxcy = input_fxfycxcy.squeeze(1)
            C2W = C2W.squeeze(1)
            fxfycxcy = fxfycxcy.squeeze(1)
            render_outputs = gsrecon.gs_renderer.render(recon_model_outputs,input_C2W, input_fxfycxcy, C2W, fxfycxcy)
            render_images = render_outputs["image"]  # (B, V, 3, H, W)
            # 保存渲染后图片
            next_render_idx = save_render_images(render_images, save_dir="renders", start_idx=next_render_idx)



            before_outputs = gsrecon.gs_renderer.render(gsrecon_outputs,input_C2W, input_fxfycxcy, C2W, fxfycxcy)
            before_image = before_outputs["image"]
            # 保存渲染前图片
            next_renderbefore_idx = save_render_images(before_image, save_dir="render_before", start_idx=next_renderbefore_idx)

            #latent:[B,768,16,16]
            print(f"recon loss [per-pixel L2]: {diff.shape}")
            print(f"mean recon loss [scalar]: {mean_diff.item():.6f}")  

    

if __name__ == "__main__":
    test_splat_method1()
