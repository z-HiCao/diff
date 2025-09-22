from typing import *
from numpy import ndarray
from torch import Tensor

import os
import json
from collections import defaultdict

import imageio
import imageio.v3 as iio
import cv2
import numpy as np
import torch
import torch.nn.functional as tF
from kiui.cam import orbit_camera, undo_orbit_camera

from src.data.utils.chunk_dataset import ChunkedDataset
from src.options import Options
from src.utils import normalize_normals, unproject_depth

import OpenEXR
import Imath


def stack_in_chunks(tensor_list, chunk_size=8):
    """分chunk拼接，避免一次性占用过大内存"""
    chunks = []
    for i in range(0, len(tensor_list), chunk_size):
        sub = tensor_list[i:i + chunk_size]
        chunks.append(torch.stack(sub, dim=0))
    return torch.cat(chunks, dim=0)

class GObjaverseParquetDataset(ChunkedDataset):
    def __init__(
        self,
        data_source,
        shuffle=True,
        shuffle_buffer_size=-1,
        chunks_queue_max_size=1,
        opt=None,
        training=True,
        root_dir=None,
        *args, **kwargs
    ):
        super().__init__(data_source, shuffle, shuffle_buffer_size, chunks_queue_max_size)
        self.opt = opt
        self.file_dir_train = opt.file_dir_train if opt is not None else None
        self.file_dir_test = opt.file_dir_test if opt is not None else None
        self.training = training
        self.root_dir = root_dir

        # Default camera intrinsics
        self.fxfycxcy = torch.tensor([opt.fxfy, opt.fxfy, 0.5, 0.5], dtype=torch.float32)  # (4,)

        if opt.prompt_embed_dir is not None:
            try:
                self.negative_prompt_embed = torch.from_numpy(np.load(f"{opt.prompt_embed_dir}/null.npy")).float()
            except FileNotFoundError:
                self.negative_prompt_embed = None
            try:
                self.negative_pooled_prompt_embed = torch.from_numpy(np.load(f"{opt.prompt_embed_dir}/null_pooled.npy")).float()
            except FileNotFoundError:
                self.negative_pooled_prompt_embed = None
            try:
                self.negative_prompt_attention_mask = torch.from_numpy(np.load(f"{opt.prompt_embed_dir}/null_attention_mask.npy")).float()
            except FileNotFoundError:
                self.negative_prompt_attention_mask = None

            if "xl" in opt.pretrained_model_name_or_path:  # SDXL: zero out negative prompt embedding
                if self.negative_prompt_embed is not None and self.negative_pooled_prompt_embed is not None:
                    self.negative_prompt_embed = torch.zeros_like(self.negative_prompt_embed)
                    self.negative_pooled_prompt_embed = torch.zeros_like(self.negative_pooled_prompt_embed)

        super().__init__(data_source,*args, **kwargs)

    def __len__(self):
        return self.opt.dataset_size



    def get_trainable_data_from_raw_data(self, raw_data_list) -> Dict[str, Tensor]:
        all_data_lists = defaultdict(list)

        for sample in raw_data_list:
            V, V_in = self.opt.num_views, self.opt.num_input_views
            assert V >= V_in

            _pick_func = self._pick_even_view_indices if self.opt.load_even_views or not self.training else self._pick_random_view_indices
            
            random_idxs = _pick_func(V_in)
            _num_tries = 0
            while not self._check_views_exist(sample, random_idxs):
                random_idxs = _pick_func(V_in)
                _num_tries += 1
                if _num_tries > 100:
                    print(f"[WARNING] Only {len(random_idxs)} views found for {sample['__key__']}, repeating views")
                    while len(random_idxs) < V_in:
                        random_idxs.append(np.random.choice(random_idxs))
                    break

            except_idxs = random_idxs + [24, 39]
            if self.opt.exclude_topdown_views:
                except_idxs += [25, 26]

            for i in np.random.permutation(40):
                if len(random_idxs) >= V:
                    break
                if f"{i:05d}.png" in sample and i not in except_idxs:
                    try:
                        val = sample[f"{i:05d}.png"]
                        if isinstance(val, str):
                            with open(val, "rb") as f:
                                val = f.read()
                        _ = np.frombuffer(val, np.uint8)

                        # _ = np.frombuffer(sample[f"{i:05d}.png"], np.uint8)
                        assert sample[f"{i:05d}.json"] is not None
                        random_idxs.append(i)
                    except Exception as e:
                        print(f"[WARN] Skipped view {i:05d} from {sample['__key__']} due to error: {e}")
                        pass
            
            while len(random_idxs) < V:
                random_idxs.append(np.random.choice(random_idxs))

            sample_data_dict = defaultdict(list)
            init_azi = None
            
            # --- Load and process data for each view ---
            for vid in random_idxs:
                png_key = f"{vid:05d}.png"
                json_key = f"{vid:05d}.json"

                if (png_key not in sample or sample[png_key] is None or
                    json_key not in sample or sample[json_key] is None):
                    print(f"[WARNING] Missing essential files for view {vid} of {sample['__key__']}, skipping this view.")
                    continue

                try:
                    image = self._load_png(sample[png_key])
                    mask = image[3:4]
                    image = image[:3] * mask + (1. - mask)
                    # print(f"mask size start: {mask.shape}")
                    
                    sample_data_dict["fxfycxcy"].append(self.fxfycxcy)
                    sample_data_dict["image"].append(image)
                    sample_data_dict["mask"].append(mask)

                    c2w = self._load_camera_from_json(sample[json_key])
                    c2w[1] *= -1
                    c2w[[1, 2]] = c2w[[2, 1]]
                    c2w[:3, 1:3] *= -1
                    sample_data_dict["original_C2W"].append(torch.from_numpy(c2w).float())
                    
                    ele, azi, dis = undo_orbit_camera(c2w)
                    if init_azi is None:
                        init_azi = azi
                    azi = (azi - init_azi) % 360.
                    ele = abs(ele) - 1e-8 if ele >= 0 else -(abs(ele) - 1e-8)
                    new_c2w = torch.from_numpy(orbit_camera(ele, azi, dis)).float()
                    sample_data_dict["C2W"].append(new_c2w)
                    sample_data_dict["cam_pose"].append(torch.tensor([np.deg2rad(ele), np.deg2rad(azi), dis], dtype=torch.float32))

                    # Optional files
                    if self.opt.load_canny:
                        gray = cv2.cvtColor(image.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY)
                        canny = cv2.Canny((gray * 255.).astype(np.uint8), 100., 200.)
                        canny = torch.from_numpy(canny).unsqueeze(0).float().repeat(3, 1, 1) / 255.
                        canny = -canny + 1.
                        sample_data_dict["canny"].append(canny)

                    if self.opt.load_albedo:
                        albedo_key = f"{vid:05d}_albedo.png"
                        if albedo_key in sample and sample[albedo_key] is not None:
                            albedo = self._load_png(sample[albedo_key])
                            albedo = albedo * mask + (1. - mask)
                            sample_data_dict["albedo"].append(albedo)

                    if self.opt.load_normal or self.opt.load_coord:
                        nd_key = f"{vid:05d}_nd.exr"
                        # if nd_key not in sample:
                        #     print(f"[DEBUG] {nd_key} not in sample for {sample['__key__']}")
                        # elif sample[nd_key] is None:
                        #     print(f"[DEBUG] {nd_key} exists but value is None for {sample['__key__']}")
                        # else:
                        #     print(f"[DEBUG] loading {nd_key} for {sample['__key__']}")

                        if nd_key in sample and sample[nd_key] is not None:
                            nd = self._load_png(sample[nd_key])
                            if self.opt.load_normal:
                                normal = nd[:3] * 2. - 1.
                                normal[0, ...] *= -1
                                sample_data_dict["normal"].append(normal)
                            if self.opt.load_coord or self.opt.load_depth:
                                depth = nd[3] * 5.
                                sample_data_dict["depth"].append(depth)
                    
                    if self.opt.load_mr:
                        mr_key = f"{vid:05d}_mr.png"
                        if mr_key in sample and sample[mr_key] is not None:
                            mr = self._load_png(sample[mr_key])
                            mr = mr * mask + (1. - mask)
                            sample_data_dict["mr"].append(mr)

                except Exception as e:
                    print(f"[WARNING] Failed to process view {vid} for {sample['__key__']} with error: {e}, skipping this view.")
                    continue
            
            if not sample_data_dict:
                print(f"[WARNING] No views were successfully loaded for sample {sample['__key__']}, skipping.")
                continue

            # for key, tensor_list in sample_data_dict.items():
            #     single_sample_stack = torch.stack(tensor_list, dim=0)
            #     all_data_lists[key].append(single_sample_stack)
            # === 改造这里：每个 sample 内部也用分chunk stack ===
            for key, tensor_list in sample_data_dict.items():
                single_sample_stack = stack_in_chunks(tensor_list, chunk_size=8)
                all_data_lists[key].append(single_sample_stack)

        if not all_data_lists:
            print(f"[ERROR] No valid samples were loaded for the entire batch. Returning empty dict.")
            return {}

        final_return_dict = {}
        for key, sample_stacks in all_data_lists.items():
            final_return_dict[key] = stack_in_chunks(sample_stacks, chunk_size=4)
        # for key, sample_stacks in all_data_lists.items():
        #     final_return_dict[key] = torch.stack(sample_stacks, dim=0)

        # --- Post-processing and data transformation ---
        
        # 调整 C2W 矩阵的 Y 和 Z 轴，使其符合特定惯例
        if "C2W" in final_return_dict:
            final_return_dict["C2W"][:, :, :3, 1:3] *= -1

        # 根据配置对相机进行归一化
        if self.opt.norm_camera and "C2W" in final_return_dict:
            # final_return_dict["C2W"] 的形状是 (B, V, 4, 4)
            # 我们只用第一个视角来计算归一化 scale，形状是 (B, 3)
            # torch.norm(..., dim=-1) 后形状变为 (B,)
            scale = self.opt.norm_radius / (torch.norm(final_return_dict["C2W"][:, 0, :3, 3], dim=-1) + 1e-8)
            
            # 将 scale 的形状从 (B,) 调整为 (B, 1, 1)，以便与 (B, V, 3) 进行广播
            scale_broadcast = scale.unsqueeze(-1).unsqueeze(-1)
            
            # 对所有视角的位置向量进行归一化
            final_return_dict["C2W"][:, :, :3, 3] *= scale_broadcast
            
            # 对所有视角的相机距离进行归一化
            # final_return_dict["cam_pose"][:, :, 2] 的形状是 (B, V)
            # scale.unsqueeze(-1) 的形状是 (B, 1)，可以广播
            final_return_dict["cam_pose"][:, :, 2] *= scale.unsqueeze(-1)

        # if "normal" not in final_return_dict:
        #     print("[DEBUG]normal not in here")
        # 处理法线数据
        if self.opt.load_normal and "normal" in final_return_dict:
            normals = normalize_normals(final_return_dict["normal"], final_return_dict["original_C2W"], i=0)
            normals = torch.einsum("bnrc,bvrhw->bvrhw", final_return_dict["C2W"][:, :, :3, :3], normals).contiguous()
            normals = normals * 0.5 + 0.5
            normals = normals * final_return_dict["mask"] + (1. - final_return_dict["mask"])
            final_return_dict["normal"] = normals
            # print("[DEBUG]get normals chuli")
            final_return_dict.pop("original_C2W")
        
        # 处理坐标和深度数据
        if self.opt.load_coord and "depth" in final_return_dict:
            #ADD
            final_return_dict["mask"] = final_return_dict["mask"].squeeze(2)

            coords = unproject_depth(final_return_dict["depth"] * final_return_dict["mask"],
                final_return_dict["C2W"], final_return_dict["fxfycxcy"])

            #ADD,因为后面coords与mask做乘法，需要五维
            final_return_dict["mask"] = final_return_dict["mask"].unsqueeze(2)
            coords = coords * 0.5 + 0.5
            coords = coords * final_return_dict["mask"] + (1. - final_return_dict["mask"])
            final_return_dict["coord"] = coords
            if not self.opt.load_depth:
                final_return_dict.pop("depth")
                
        if self.opt.load_depth and "depth" in final_return_dict:
            depths = final_return_dict["depth"].unsqueeze(2) * final_return_dict["mask"].unsqueeze(2)
            assert depths.min() >= 0.
            if self.opt.normalize_depth:
                depths_reshaped = depths.view(depths.shape[0], depths.shape[1], -1)
                depths_max = depths_reshaped.max(dim=-1, keepdim=True).values
                depths = depths_reshaped / depths_max.clamp(min=1e-6)
                depths = depths.view(final_return_dict["depth"].shape)
            depths = -depths + 1.
            final_return_dict["depth"] = depths.repeat(1, 1, 3, 1, 1)

        # Resize to the input resolution
        for key in ["image", "mask", "albedo", "normal", "coord", "depth", "mr", "canny"]:
            if key in final_return_dict:
                batch_size, num_views, C, H, W = final_return_dict[key].shape
                final_return_dict[key] = tF.interpolate(
                    final_return_dict[key].view(-1, C, H, W),
                    size=(self.opt.input_res, self.opt.input_res),
                    mode="bilinear", align_corners=False, antialias=True
                ).view(batch_size, num_views, C, self.opt.input_res, self.opt.input_res)
                # print(f"[DEBUG] {key}.shape: {final_return_dict[key].shape}")


        # Handle anti-aliased normal, coord and depth
        if self.opt.load_normal and "normal" in final_return_dict:
            final_return_dict["normal"] = final_return_dict["normal"] * final_return_dict["mask"] + (1. - final_return_dict["mask"])
        if self.opt.load_coord and "coord" in final_return_dict:
            final_return_dict["coord"] = final_return_dict["coord"] * final_return_dict["mask"] + (1. - final_return_dict["mask"])
        if self.opt.load_depth and "depth" in final_return_dict:
            final_return_dict["depth"] = final_return_dict["depth"] * final_return_dict["mask"] + (1. - final_return_dict["mask"])

        # Load precomputed caption embeddings
        for sample_uid in raw_data_list:
            if "uid" in sample_uid:
                uid = sample_uid["uid"].decode("utf-8").split("/")[-1].split(".")[0]
                prompt_embed_path = f"{self.opt.prompt_embed_dir}/{uid}.npy"
                if os.path.exists(prompt_embed_path):
                    final_return_dict["prompt_embed"] = torch.from_numpy(np.load(prompt_embed_path))
                else:
                    final_return_dict["prompt_embed"] = self.negative_prompt_embed.repeat(len(raw_data_list), 1)

        for key in final_return_dict.keys():
            if not isinstance(final_return_dict[key], Tensor):
                print(f"[WARN] Converting non-tensor value for key [{key}] to tensor.")
                final_return_dict[key] = torch.tensor(final_return_dict[key])

        return dict(final_return_dict)        

    def _load_png(self, path_or_bytes: str | bytes, uint16=False) -> torch.Tensor:
        '''支持 EXR 和 PNG 的读取'''
        if isinstance(path_or_bytes, str):
            ext = os.path.splitext(path_or_bytes)[1].lower()
            if ext == ".exr":
                # 用 OpenEXR + Imath 读取
                try:
                    exr_file = OpenEXR.InputFile(path_or_bytes)
                    dw = exr_file.header()['dataWindow']
                    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
                    pt = Imath.PixelType(Imath.PixelType.FLOAT)
                    R = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32).reshape(size[1], size[0])
                    G = np.frombuffer(exr_file.channel('G', pt), dtype=np.float32).reshape(size[1], size[0])
                    B = np.frombuffer(exr_file.channel('B', pt), dtype=np.float32).reshape(size[1], size[0])
                    img = np.stack([R, G, B], axis=-1)
                except Exception as e:
                    print(f"[WARN] EXR decode failed: {path_or_bytes}, error: {e}")
                    img = np.zeros((512, 512, 3), dtype=np.float32)
            else:
                # 读取普通 PNG
                with open(path_or_bytes, "rb") as f:
                    png_bytes = f.read()
                img = np.frombuffer(png_bytes, np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"[WARN] PNG decode failed: {path_or_bytes}")
                    img = np.zeros((512, 512, 4), dtype=np.float32)
                img = img.astype(np.float32) / (65535. if uint16 else 255.)
        else:
            # bytes 输入暂时只处理 PNG
            img = np.frombuffer(path_or_bytes, np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
            img = img.astype(np.float32) / (65535. if uint16 else 255.)

        if img.shape[2] == 3:
            # BGR -> RGB
            depth_channel = np.zeros((img.shape[0], img.shape[1], 1), dtype=img.dtype)
            img = np.concatenate([img, depth_channel], axis=2)
        elif img.shape[2] > 3:
            # 前3个通道是颜色，其他通道保持不变
            img[:, :, :3] = img[:, :, :3][..., ::-1]
            img = img.copy()

        return torch.from_numpy(img.copy()).nan_to_num_(0.).permute(2, 0, 1)

    def _load_camera_from_json(self, json_path_or_bytes: str | bytes) -> np.ndarray:
        # 如果是路径，先读成 bytes
        if isinstance(json_path_or_bytes, str):
            if not os.path.exists(json_path_or_bytes):
                print(f"[WARN] File does not exist: {json_path_or_bytes}")
                return np.eye(4)  # Return identity matrix as fallback
            with open(json_path_or_bytes, "rb") as f:
                json_bytes = f.read()
        else:
            json_bytes = json_path_or_bytes

        #json_dict = json.loads(json_bytes)
        try:
            json_dict = json.loads(json_bytes)
        except json.JSONDecodeError:
            print(f"[WARN] Invalid or empty JSON file: {json_path_or_bytes}, skipping...")
            # 返回一个默认相机矩阵，避免报错
            return np.eye(4)

        c2w = np.eye(4)
        c2w[:3, 0] = np.array(json_dict["x"])
        c2w[:3, 1] = np.array(json_dict["y"])
        c2w[:3, 2] = np.array(json_dict["z"])
        c2w[:3, 3] = np.array(json_dict["origin"])
        return c2w

    def _pick_even_view_indices(self, num_views: int = 4) -> List[int]:
        assert 12 % num_views == 0  # `12` for even-view sampling in GObjaverse

        if np.random.rand() < 2/3:
            index0 = np.random.choice(range(24))  # 0~23: 24 views in ele from [5, 30]; hard-coded for GObjaverse
            return [(index0 + (24 // num_views)*i) % 24 for i in range(num_views)]
        else:
            index0 = np.random.choice(range(12))  # 27~38: 12 views in ele from [-5, 5]; hard-coded for GObjaverse
            return [((index0 + (12 // num_views)*i) % 12 + 27) for i in range(num_views)]

    def _pick_random_view_indices(self, num_views: int = 4) -> List[int]:
        assert num_views <= 40  # `40` is hard-coded for GObjaverse

        indices = (set(range(40)) - set([25, 26])) if self.opt.exclude_topdown_views else (set(range(40)))  # `40` is hard-coded for GObjaverse
        return np.random.choice(list(indices), num_views, replace=False).tolist()

    def _check_views_exist(self, sample: Dict[str, Union[str, bytes]], vids: List[int]) -> bool:
        for vid in vids:
            if f"{vid:05d}.png" not in sample:
                return False
            try:
                assert sample[f"{vid:05d}.png"] is not None and sample[f"{vid:05d}.json"] is not None
            except:  # TypeError: a bytes-like object is required, not 'NoneType'; KeyError: '00001.json'
                return False
        return True
