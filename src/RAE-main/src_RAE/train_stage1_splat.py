"""
Stage-1 RAE training script with reconstruction, LPIPS, and GAN losses.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate # 引入 default_collate
from torchvision import transforms
from torchvision.datasets import ImageFolder
from glob import glob

from omegaconf import OmegaConf

from disc import (
    DiffAug,
    LPIPS,
    build_discriminator,
    hinge_d_loss,
    vanilla_d_loss,
    vanilla_g_loss,
)
from stage1 import RAE
from utils import wandb_utils
from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs
from utils.optim_utils import build_optimizer, build_scheduler

import sys
sys.path.append("/opt/data/private/wjy/LRY/DiffSplat-main")

# diffsplat
from src.data import GObjaverseParquetDataset, ParquetChunkDataSource, MultiEpochsChunkedDataLoader, yield_forever
from src.models import GSAutoencoderKL, GSRecon, get_optimizer, get_lr_scheduler
import src.utils.util as util
import src.utils.vis_util as vis_util
from src.options import opt_dict
import accelerate
from accelerate import Accelerator
from accelerate import DataLoaderConfiguration, DeepSpeedPlugin
from skimage.metrics import structural_similarity as calculate_ssim
from einops import rearrange
import numpy as np
# --- [CRITICAL FIX] 自定义安全 Collate 函数 ---
def safe_collate_fn(batch):
    """
    自定义 Collate。
    核心目标：无论发生什么，必须返回一个包含 Tensors 的有效字典。
    绝对不能返回 None，也不能返回空字典 {}。
    """
    # 1. 过滤掉 None 和 空字典
    valid_batch = [b for b in batch if b is not None and len(b.keys()) > 0]
    
    # [Failsafe] 如果整个 Batch 的数据都坏了/空了
    if len(valid_batch) == 0:
        print("[SAFE COLLATE] CRITICAL: Entire batch is empty/invalid. Creating a dummy batch.")
        # 我们必须手动构造一个假 Batch，让 accelerate 有东西可以搬运
        # 假设图像大小为 256 (根据 args.image_size，这里写死或者从外部获取)
        # 为了安全，我们生成最小可行 Tensor
        dummy_batch = {
            "image": torch.zeros(1, 3, 256, 256),
            "C2W": torch.eye(4).unsqueeze(0),
            "fxfycxcy": torch.tensor([512.0, 512.0, 0.5, 0.5]).unsqueeze(0),
            "cam_pose": torch.zeros(1, 3),
            "mask": torch.zeros(1, 1, 256, 256),
            "prompt_embed": torch.zeros(1, 77, 768),
            # 如果有 input_C2W 逻辑，这里其实不需要补，因为下面的 default_collate 也没法补
        }
        return dummy_batch

    # 2. 正常拼接
    try:
        collated_batch = default_collate(valid_batch)
    except Exception as e:
        print(f"[SAFE COLLATE] Standard collate failed: {e}. Fallback to dummy.")
        # 如果因为形状不匹配导致拼接失败，取第一个有效样本扩充
        first_sample = valid_batch[0]
        # 简单地把第一个样本变成 Batch=1 返回
        collated_batch = {}
        for k, v in first_sample.items():
            if isinstance(v, torch.Tensor):
                collated_batch[k] = v.unsqueeze(0)
            else:
                collated_batch[k] = torch.tensor([0]) # 甚至可以忽略

    # 3. [终极清洗] 再次检查拼接结果
    if isinstance(collated_batch, dict):
        clean_batch = {}
        for k, v in collated_batch.items():
            if v is None:
                print(f"[SAFE COLLATE] Key '{k}' became None! Filling with zeros.")
                clean_batch[k] = torch.zeros(1)
            else:
                clean_batch[k] = v
        return clean_batch
    
    return collated_batch

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-1 RAE with GAN and LPIPS losses.")
    parser.add_argument("--config", type=str, required=True, help="YAML config containing a stage_1 section.")
    parser.add_argument("--data-path", type=Path, required=True, help="Directory with ImageFolder structure.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to store training outputs.")
    parser.add_argument("--image-size", type=int, default=256, help="Image resolution (assumes square images).")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--global-seed", type=int, default=None, help="Override training.global_seed from the config.")    
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path to resume training.")
    parser.add_argument("--wandb", action='store_true', help='Use Weights & Biases for logging if set.')
    parser.add_argument("--wandb_token_path",type=str,default=None,help="path for wandb token")
    parser.add_argument("--gradient_accumulation_steps",type=int,default=8,help="set gradient_accumulation_steps")
    return parser.parse_args()

def create_logger(logging_dir):
    if not dist.is_available() or not dist.is_initialized():
        rank = 0
    else:
        rank = dist.get_rank()

    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def setup_distributed() -> Tuple[int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, current_model: torch.nn.Module, decay: float) -> None:
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(current_model.named_parameters())
    for name, param in model_params.items():
        if name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def calculate_adaptive_weight(
    recon_loss: torch.Tensor,
    gan_loss: torch.Tensor,
    layer: torch.nn.Parameter,
    max_d_weight: float = 1e4,
) -> torch.Tensor:
    recon_grads = torch.autograd.grad(recon_loss, layer, retain_graph=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, layer, retain_graph=True)[0]
    d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
    d_weight = torch.clamp(d_weight, 0.0, max_d_weight)
    return d_weight.detach()

def select_gan_losses(disc_kind: str, gen_kind: str):
    if disc_kind == "hinge":
        disc_loss_fn = hinge_d_loss
    elif disc_kind == "vanilla":
        disc_loss_fn = vanilla_d_loss
    else:
        raise ValueError(f"Unsupported discriminator loss '{disc_kind}'")

    if gen_kind == "vanilla":
        gen_loss_fn = vanilla_g_loss
    else:
        raise ValueError(f"Unsupported generator loss '{gen_kind}'")
    return disc_loss_fn, gen_loss_fn

# def save_checkpoint(path, step, epoch, model, ema_model, optimizer, scheduler, disc, disc_optimizer, disc_scheduler):
#     state = {
#         "step": step,
#         "epoch": epoch,
#         "model": model.module.state_dict(),
#         "ema": ema_model.state_dict(),
#         "optimizer": optimizer.state_dict(),
#         "scheduler": scheduler.state_dict() if scheduler is not None else None,
#         "disc": disc.state_dict(),
#         "disc_optimizer": disc_optimizer.state_dict(),
#         "disc_scheduler": disc_scheduler.state_dict() if disc_scheduler is not None else None,
#     }
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     torch.save(state, path)
def save_checkpoint(
    path,
    step,
    epoch,
    model,
    ema_model,
    optimizer,
    scheduler=None,
    disc=None,
    disc_optimizer=None,
    disc_scheduler=None,
    accelerator=None,
):
    """
    Universal checkpoint saver.
    Works for:
      - single GPU
      - DDP
      - HuggingFace Accelerate
    """

    # -------- unwrap model safely --------
    if accelerator is not None:
        model_to_save = accelerator.unwrap_model(model)
    else:
        model_to_save = model.module if hasattr(model, "module") else model

    state = {
        "step": step,
        "epoch": epoch,
        "model": model_to_save.state_dict(),
        "ema": ema_model.state_dict() if ema_model is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }

    # -------- discriminator (optional) --------
    if disc is not None:
        state["disc"] = disc.state_dict()
    if disc_optimizer is not None:
        state["disc_optimizer"] = disc_optimizer.state_dict()
    if disc_scheduler is not None:
        state["disc_scheduler"] = disc_scheduler.state_dict()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path, model, ema_model, optimizer, scheduler, disc, disc_optimizer, disc_scheduler):
    checkpoint = torch.load(path, map_location="cpu")
    model.module.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    disc.load_state_dict(checkpoint["disc"])
    disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
    if disc_scheduler is not None and checkpoint.get("disc_scheduler") is not None:
        disc_scheduler.load_state_dict(checkpoint["disc_scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)

def unpack_splat_tensor(splat_tensor):
    """
    将 RAE 输出的 Tensor (B, C, H, W) 拆分为 DiffSplat Renderer 需要的字典。
    假设通道顺序为: Offset(3), Opacity(1), Scale(3), Rotation(4), RGB(3) -> 总共 14 通道
    如果不符合 14 通道，根据实际情况调整切片。
    """
    B, C, H, W = splat_tensor.shape
    
    # DiffSplat 标准通道布局推测
    # 0-3: Offset
    # 3-4: Opacity
    # 4-7: Scale
    # 7-11: Rotation
    # 11-14: RGB
    
    params = {}
    
    if C == 14:
        params["offset"] = splat_tensor[:, 0:3, :, :]
        params["opacity"] = splat_tensor[:, 3:4, :, :]
        params["scale"] = splat_tensor[:, 4:7, :, :]
        params["rotation"] = splat_tensor[:, 7:11, :, :]
        params["rgb"] = splat_tensor[:, 11:14, :, :]
    elif C == 11: # 如果没有 offset
        params["opacity"] = splat_tensor[:, 0:1, :, :]
        params["scale"] = splat_tensor[:, 1:4, :, :]
        params["rotation"] = splat_tensor[:, 4:8, :, :]
        params["rgb"] = splat_tensor[:, 8:11, :, :]
    else:
        # Fallback: 打印错误并尝试强行切分防止 Crash，需要根据日志调整
        print(f"[WARN] Unexpected channel count {C} in splat tensor! Trying generic split.")
        params["rgb"] = splat_tensor[:, :3, :, :]
        params["scale"] = splat_tensor[:, 3:6, :, :]
        params["rotation"] = splat_tensor[:, 6:10, :, :]
        params["opacity"] = splat_tensor[:, 10:11, :, :]
        
    return params

def main():
    gradient_accumulation_steps = 8

    args = parse_args()
    with open(args.wandb_token_path, "r") as f:
        os.environ["WANDB_API_KEY"] = f.read().strip()

    rank, world_size, device = setup_distributed()
    (rae_config, *_) = parse_configs(args.config)
    full_cfg = OmegaConf.load(args.config)
    training_section = full_cfg.get("training", None)
    training_cfg = OmegaConf.to_container(training_section, resolve=True) if training_section is not None else {}
    training_cfg = dict(training_cfg) if isinstance(training_cfg, dict) else {}

    gan_section = full_cfg.get("gan", None)
    gan_cfg = OmegaConf.to_container(gan_section, resolve=True) if gan_section is not None else {}
    if not gan_cfg:
        raise ValueError("Config must define a top-level 'gan' section for stage-1 training.")
    disc_cfg = gan_cfg.get("disc", {})
    loss_cfg = gan_cfg.get("loss", {})
    perceptual_weight = float(loss_cfg.get("perceptual_weight", 0.0))
    disc_weight = float(loss_cfg.get("disc_weight", 0.0))
    gan_start_epoch = int(loss_cfg.get("disc_start", 0))
    disc_update_epoch = int(loss_cfg.get("disc_upd_start", gan_start_epoch))
    lpips_start_epoch = int(loss_cfg.get("lpips_start", 0))
    
    disc_updates = int(loss_cfg.get("disc_updates", 1))
    max_d_weight = float(loss_cfg.get("max_d_weight", 1e4))
    disc_loss_type = loss_cfg.get("disc_loss", "hinge")
    gen_loss_type = loss_cfg.get("gen_loss", "vanilla")

    batch_size = int(training_cfg.get("batch_size", 16))
    log_interval = int(training_cfg.get("log_interval", 100))
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 1000))
    ema_decay = float(training_cfg.get("ema_decay", 0.9999))
    num_epochs = int(training_cfg.get("epochs", 200))
    default_seed = int(training_cfg.get("global_seed", 0))
    global_seed = args.global_seed if args.global_seed is not None else default_seed
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    clip_grad_val = training_cfg.get("clip_grad", 1.0)
    clip_grad = float(clip_grad_val) if clip_grad_val is not None else None
    if clip_grad is not None and clip_grad <= 0:
        clip_grad = None

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*")) - 1
        model_target = str(rae_config.get("target", "stage1"))
        model_string_name = model_target.split(".")[-1]
        precision_suffix = f"-{args.precision}" if args.precision == "bf16" else ""
        experiment_name = f"{experiment_index:03d}-{model_string_name}{precision_suffix}"
        experiment_dir = os.path.join(args.results_dir, experiment_name)
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        if args.wandb:
            entity = os.environ.get("ENTITY", "")
            project = os.environ.get("PROJECT", "")
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        experiment_dir = None
        checkpoint_dir = None
        logger = create_logger(None)
    
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.encoder.eval()
    rae.decoder.train()
    ema_model = deepcopy(rae).to(device).eval()
    ema_model.requires_grad_(False)
    rae.encoder.requires_grad_(False)
    rae.decoder.requires_grad_(True)
    decoder = rae.decoder
    optimizer, optim_msg = build_optimizer(decoder.parameters(), training_cfg)

    discriminator, disc_aug = build_discriminator(disc_cfg, device)
    disc_params = [p for p in discriminator.parameters() if p.requires_grad]
    disc_optimizer, disc_optim_msg = build_optimizer(disc_params, disc_cfg)
    disc_scheduler: LambdaLR | None = None
    disc_sched_msg: Optional[str] = None

    discriminator.train()
    disc_loss_fn, gen_loss_fn = select_gan_losses(disc_loss_type, gen_loss_type)

    lpips = LPIPS().to(device)
    lpips.eval()

    scaler: GradScaler | None
    if args.precision == "fp16":
        scaler = GradScaler()
        autocast_kwargs = dict(enabled=True, dtype=torch.float16)
    elif args.precision == "bf16":
        scaler = None
        autocast_kwargs = dict(enabled=True, dtype=torch.bfloat16)
    else:
        scaler = None
        autocast_kwargs = dict(enabled=False)

    opt = opt_dict[training_cfg.get("opt_type")]
    #提取需要的参数
    V_in = opt.num_input_views

    train_dataset = GObjaverseParquetDataset(
        data_source=ParquetChunkDataSource("./dataset/train", training_cfg.get("file_name_train")),
        shuffle=True,
        shuffle_buffer_size=-1,
        chunks_queue_max_size=1,
        opt=opt,
        training=True,
    )
    
    # [FIX] 使用 safe_collate_fn
    train_loader = MultiEpochsChunkedDataLoader(
        train_dataset,
        batch_size=training_cfg.get("batch_size"),
        num_workers=0,
        drop_last=True,
        pin_memory=True,
        shuffle=False,
        collate_fn=safe_collate_fn # <--- 注入安全 Collate
    )

    # #测试：未经过accelerate包装，能否取batch
    # batch = next(iter(train_loader))

    # print("=== Batch Structure ===")
    # for k, v in batch.items():
    #     print(k, type(v), isinstance(v, torch.Tensor))


    steps_per_epoch = len(train_loader)

    if rank == 0:
        print(f"=== TRAINING PLAN ===")
        print(f"Total Epochs: {num_epochs}")
        print(f"Steps per Epoch: {steps_per_epoch}")
        print(f"Total Steps to run: {num_epochs * steps_per_epoch}")
        print(f"=====================")
    
    scheduler: LambdaLR | None = None
    sched_msg: Optional[str] = None
    if training_cfg.get("scheduler"):
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, training_cfg)

    if disc_cfg.get("scheduler"):
        disc_scheduler, disc_sched_msg = build_scheduler(disc_optimizer, steps_per_epoch, disc_cfg)
    start_epoch = 0
    global_step = 0

    # ... (Sample Checking Code Skipped for brevity) ...
    
    if args.ckpt and Path(args.ckpt).is_file():
        pass # Resume logic handled later after prepare

    accelerator = Accelerator(split_batches=False,gradient_accumulation_steps=args.gradient_accumulation_steps,
                            dataloader_config=DataLoaderConfiguration(non_blocking=True))
    device = accelerator.device
    
    rae, optimizer, train_loader, discriminator, disc_optimizer = accelerator.prepare(
        rae, optimizer, train_loader, discriminator, disc_optimizer
    )

    ddp_model = rae
    model_woddp = accelerator.unwrap_model(ddp_model)
    
    if args.ckpt and Path(args.ckpt).is_file():
        start_epoch, global_step = load_checkpoint(
            Path(args.ckpt),
            ddp_model,
            ema_model,
            optimizer,
            scheduler,
            accelerator.unwrap_model(discriminator),
            disc_optimizer,
            disc_scheduler,
        )
        logger.info(f"[Rank {rank}] Resumed from {args.ckpt} (epoch={start_epoch}, step={global_step}).")

    last_layer = model_woddp.decoder.decoder_pred.weight
    gan_start_step = gan_start_epoch * steps_per_epoch
    disc_update_step = disc_update_epoch * steps_per_epoch
    lpips_start_step = lpips_start_epoch * steps_per_epoch

    gsrecon = GSRecon(opt).to(device)
    gsrecon = gsrecon.requires_grad_(False)
    gsrecon = gsrecon.eval()

    
    print(f"Load GSRecon checkpoint \n")
    gsrecon = util.load_ckpt(
        os.path.join("out", "gsrecon_gobj265k_cnp_even4", "checkpoints"),-1,
        model = gsrecon
    ).to(device)

    for epoch in range(start_epoch, num_epochs):
        ddp_model.train()
        epoch_metrics: Dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(1, device=device))
        num_batches = 0
        
        # for batch in yield_forever(train_loader):
        for batch in train_loader:
            # [FIX] 空 Batch 检查
            if not batch:
                continue
                
            use_gan = global_step >= gan_start_step and disc_weight > 0.0
            # train_disc = global_step >= disc_update_step and disc_weight > 0.0
            # 先不用train
            # use_gan = False
            train_disc = False
            use_lpips = global_step >= lpips_start_step and perceptual_weight > 0.0
            
            images = batch["image"].to(device, non_blocking=True).squeeze(1)

            current_bs = images.shape[0]
            if current_bs != batch_size:
                print(f"[WARN] Skipping weird batch with size {current_bs} (Expected {batch_size}). Likely a dummy batch from safe_collate.")
                continue

            C2W = batch["C2W"].to(device, non_blocking=True).squeeze(1)# (B, V, 4, 4)   （4，8，4，4）
            fxfycxcy = batch["fxfycxcy"].to(device, non_blocking=True).squeeze(1) #(B,V,4)  (4,8,4)

            input_image = images[:,:V_in,...] # (4,4,3,256,256)
            img_useForRecon = input_image  #recon还需要在channel维拼接normal\coord
            input_C2W = C2W[:, :V_in, ...]# (B, Vin, 4, 4)  (4,4,4,4)
            input_fxfycxcy = fxfycxcy[:, :V_in, ...] # (4,4,4)
            
            if opt.input_normal:
                normal_map = batch["normal"][:, :,:V_in, ...].to(device=device, dtype=torch.float32).contiguous() #(4,1,4,3,256,256)
                normal_map = normal_map.squeeze(1) #(4,4,3,256,256)
                img_useForRecon = torch.cat([img_useForRecon, normal_map], dim=2) #(4,4,6,256,256)
            if opt.input_coord:
                coord_map = batch["coord"][:,:, :V_in, ...].to(device=device, dtype=torch.float32).contiguous() #(4,1,4,3,256,256)
                coord_map = coord_map.squeeze(1) #(4,4,3,256,256)
                img_useForRecon = torch.cat([img_useForRecon, coord_map], dim=2) #(4,4,9,256,256)
            
            V,C, H, W = images.shape[-4:] #(8,3,256,256)
            #gaussian gt
            gs_output = gsrecon.forward_gaussians(img_useForRecon,input_C2W, input_fxfycxcy)
            gs = torch.cat([
                gs_output["rgb"],
                gs_output["scale"],
                gs_output["rotation"],
                gs_output["opacity"],
                gs_output["depth"],
            ], dim=2)#[b,vin,12,h,w],(4,4,12,256,256)
            gs_flat = gs.reshape(-1,12,H,W)#展为[b*vin,12,h,w],(16,12,256.256)
            gs_flat_224 = F.interpolate(
                gs_flat,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
                antialias=True
            ) #(16,12,224,224)
            # input_image_flat = input_image.reshape(-1, C, H, W)#展开后输入encoder，[B,V_in,C,H,W]=>[B*V_IN,C,H,W]

            # real_normed = input_image_flat * 2.0 - 1.0
            optimizer.zero_grad(set_to_none=True)
            discriminator.eval()

            with autocast(**autocast_kwargs):
                with torch.no_grad():
                    z = model_woddp.encode(gs_flat_224) #(16,768,16,16)
                
                recon_splat = model_woddp.decode(z)  #(16,12,256,256)
                recon_splat = recon_splat.view(batch_size, V_in, 12,256,256) #[B,V_in,12,H,W],#(4,4,12,256,256)
                # splat_params_dict = unpack_splat_tensor(recon_splat)
                model_outputs = {
                "rgb": recon_splat[:, :,:3, ...],
                "scale": recon_splat[:, :, 3:6, ...],
                "rotation": recon_splat[:, :, 6:10, ...],
                "opacity": recon_splat[:, :, 10:11, ...],
                "depth": recon_splat[:,:, 11:12, ...],
            }
                outputs = {}
                recon_img = gsrecon.gs_renderer.render(model_outputs, input_C2W, input_fxfycxcy, C2W, fxfycxcy) #会输出image、coord、normal，也许可以用这些训练。
                
                
                rec_loss = F.mse_loss(recon_splat, gs)
                outputs["mseloss"] = rec_loss

                if use_lpips:
                    #recon_img 有新视角。
                    lpips_loss = lpips(images.view(-1, C, H, W), recon_img["image"].view(-1, C, H, W)).mean() #recon_img原本为[B,V,C,H,W]=>[B*V,C,H,W],LPIPS只接受3D/4D
                else:
                    lpips_loss = rec_loss.new_zeros(())

                outputs["lpips"] = lpips_loss
                recon_total = rec_loss + perceptual_weight * lpips_loss

                with torch.no_grad():
                    # PSNR
                    outputs["psnr"] = -10.0 * torch.log10(
                        torch.mean((images - recon_img["image"].detach()) ** 2)
                    )

                    # SSIM
                    outputs["ssim"] = torch.tensor(
                        calculate_ssim(
                            (rearrange(images, "b v c h w -> (b v c) h w")
                                .cpu().float().numpy() * 255.).astype(np.uint8),
                            (rearrange(recon_img["image"].detach(), "b v c h w -> (b v c) h w")
                                .cpu().float().numpy() * 255.).astype(np.uint8),
                            channel_axis=0,
                        ),
                        device=images.device,
                    )


                if use_gan:
                    recon_normed = recon_img.clamp(0, 1) * 2.0 - 1.0
                    fake_aug = disc_aug.aug(recon_normed) 
                    logits_fake, _ = discriminator(fake_aug, None)
                    gan_loss = gen_loss_fn(logits_fake)
                    adaptive_weight = calculate_adaptive_weight(
                        recon_total, gan_loss, last_layer, max_d_weight
                    )
                    total_loss = recon_total + disc_weight * adaptive_weight * gan_loss
                else:
                    gan_loss = torch.zeros_like(recon_total)
                    adaptive_weight = torch.zeros_like(recon_total)
                    total_loss = recon_total

            if scaler:
                scaler.scale(total_loss).backward()
                if clip_grad is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                accelerator.backward(total_loss)
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # update_ema(ema_model, ddp_model.module, ema_decay)
            update_ema(ema_model,accelerator.unwrap_model(ddp_model), ema_decay)


            disc_metrics: Dict[str, torch.Tensor] = {}
            if train_disc:
                discriminator.train()
                recon_normed_detached = recon_img.detach().clamp(0, 1) * 2.0 - 1.0
                for _ in range(disc_updates):
                    disc_optimizer.zero_grad(set_to_none=True)
                    with autocast(**autocast_kwargs):
                        # fake_detached = real_normed.detach() # Placeholder logic
                        # fake_detached = fake_detached.clamp(-1.0, 1.0)
                        # fake_detached = torch.round((fake_detached + 1.0) * 127.5) / 127.5 - 1.0
                        fake_detached = recon_normed_detached
                        fake_input = disc_aug.aug(fake_detached)
                        real_input = disc_aug.aug(real_normed)
                        logits_fake, logits_real = discriminator(fake_input, real_input)
                        d_loss = disc_loss_fn(logits_real, logits_fake)
                    if scaler:
                        scaler.scale(d_loss).backward()
                        scaler.step(disc_optimizer)
                        scaler.update()
                    else:
                        d_loss.backward()
                        disc_optimizer.step()
                    disc_metrics = {
                        "disc_loss": d_loss.detach(),
                        "logits_real": logits_real.detach().mean(),
                        "logits_fake": logits_fake.detach().mean(),
                    }
                    if disc_scheduler is not None:
                        disc_scheduler.step()
                discriminator.eval()

            epoch_metrics["recon"] += rec_loss.detach()
            epoch_metrics["lpips"] += lpips_loss.detach()
            epoch_metrics["gan"] += gan_loss.detach()
            epoch_metrics["total"] += total_loss.detach()
            num_batches += 1

            if log_interval > 0 and global_step % log_interval == 0 and rank == 0:
                stats = {
                    "loss/total": total_loss.detach().item(),
                    "loss/recon": rec_loss.detach().item(),
                    "loss/lpips": lpips_loss.detach().item(),
                    "loss/gan": gan_loss.detach().item(),
                    "gan/weight": adaptive_weight.detach().item(),
                    "lr/generator": optimizer.param_groups[0]["lr"],
                }
                if disc_metrics:
                    stats.update(
                        {
                            "loss/disc": disc_metrics["disc_loss"].item(),
                            "disc/logits_real": disc_metrics["logits_real"].item(),
                            "disc/logits_fake": disc_metrics["logits_fake"].item(),
                            "lr/discriminator": disc_optimizer.param_groups[0]["lr"],
                        }
                    )
                logger.info(
                    f"[Epoch {epoch} | Step {global_step}] "
                    + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items())
                )
                if args.wandb:
                    wandb_utils.log(stats, step=global_step)

            if checkpoint_interval > 0 and global_step % checkpoint_interval == 0 and rank == 0:
                ckpt_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                save_checkpoint(
                    ckpt_path,
                    global_step,
                    epoch,
                    ddp_model,
                    ema_model,
                    optimizer,
                    scheduler,
                    accelerator.unwrap_model(discriminator),
                    disc_optimizer,
                    disc_scheduler,
                )

            global_step += 1

        if rank == 0 and num_batches > 0:
            avg_recon = (epoch_metrics["recon"] / num_batches).item()
            avg_lpips = (epoch_metrics["lpips"] / num_batches).item()
            avg_gan = (epoch_metrics["gan"] / num_batches).item()
            avg_total = (epoch_metrics["total"] / num_batches).item()
            epoch_stats = {
                "epoch/loss_total": avg_total,
                "epoch/loss_recon": avg_recon,
                "epoch/loss_lpips": avg_lpips,
                "epoch/loss_gan": avg_gan,
            }
            logger.info(
                f"[Epoch {epoch}] "
                + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items())
            )
            if args.wandb:
                wandb_utils.log(epoch_stats, step=global_step)
    cleanup_distributed()

if __name__ == "__main__":
    main()