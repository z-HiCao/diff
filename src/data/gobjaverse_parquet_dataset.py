# from typing import *
# from numpy import ndarray
# from torch import Tensor

# import os
# import json
# from collections import defaultdict

# import imageio
# import imageio.v3 as iio
# import cv2
# import numpy as np
# import torch
# import torch.nn.functional as tF
# from kiui.cam import orbit_camera, undo_orbit_camera

# from src.data.utils.chunk_dataset import ChunkedDataset
# from src.options import Options
# from src.utils import normalize_normals, unproject_depth

# import OpenEXR
# import Imath


# def stack_in_chunks(tensor_list, chunk_size=8):
#     """分chunk拼接，避免一次性占用过大内存"""
#     chunks = []
#     for i in range(0, len(tensor_list), chunk_size):
#         sub = tensor_list[i:i + chunk_size]
#         chunks.append(torch.stack(sub, dim=0))
#     return torch.cat(chunks, dim=0)

# class GObjaverseParquetDataset(ChunkedDataset):
#     def __init__(
#         self,
#         data_source,
#         shuffle=True,
#         shuffle_buffer_size=-1,
#         chunks_queue_max_size=1,
#         opt=None,
#         training=True,
#         root_dir=None,
#         *args, **kwargs
#     ):
#         super().__init__(data_source, shuffle, shuffle_buffer_size, chunks_queue_max_size)
#         self.opt = opt
#         self.file_dir_train = opt.file_dir_train if opt is not None else None
#         self.file_dir_test = opt.file_dir_test if opt is not None else None
#         self.training = training
#         self.root_dir = root_dir

#         # Default camera intrinsics
#         self.fxfycxcy = torch.tensor([opt.fxfy, opt.fxfy, 0.5, 0.5], dtype=torch.float32)  # (4,)

#         if opt.prompt_embed_dir is not None:
#             try:
#                 self.negative_prompt_embed = torch.from_numpy(np.load(f"{opt.prompt_embed_dir}/null.npy")).float()
#             except FileNotFoundError:
#                 print(f"[WARN] null.npy not found. Using zero tensor for negative_prompt_embed.")
#                 self.negative_prompt_embed = torch.zeros(77, 768).float()
#             try:
#                 self.negative_pooled_prompt_embed = torch.from_numpy(np.load(f"{opt.prompt_embed_dir}/null_pooled.npy")).float()
#             except FileNotFoundError:
#                 self.negative_pooled_prompt_embed = torch.zeros(1, 1280).float()
#             try:
#                 self.negative_prompt_attention_mask = torch.from_numpy(np.load(f"{opt.prompt_embed_dir}/null_attention_mask.npy")).float()
#             except FileNotFoundError:
#                 self.negative_prompt_attention_mask = torch.ones(1, 77).float()

#             if "xl" in opt.pretrained_model_name_or_path:  # SDXL: zero out negative prompt embedding
#                 if self.negative_prompt_embed is not None and self.negative_pooled_prompt_embed is not None:
#                     self.negative_prompt_embed = torch.zeros_like(self.negative_prompt_embed)
#                     self.negative_pooled_prompt_embed = torch.zeros_like(self.negative_pooled_prompt_embed)
#         else:
#             # 如果 opt.prompt_embed_dir 为 None，仍需定义这些属性以防后续代码报错
#             self.negative_prompt_embed = torch.zeros(77, 768).float()
#             self.negative_pooled_prompt_embed = torch.zeros(1, 1280).float()
#             self.negative_prompt_attention_mask = torch.ones(1, 77).float()

#         # super().__init__(data_source,*args, **kwargs)

#     def __len__(self):
#         return self.opt.dataset_size

#     def __getitem__(self, index: int) -> Dict[str, Tensor]:
        
#         try:
#             # 1. 获取原始数据
#             raw_row_data = self.data_source.get_row_data(index) 
#             if raw_row_data is None:
#                 print(f"[ERROR] get_row_data returned None for index {index}")
#                 return self._get_placeholder_single_sample()

#             # 2. 获取可训练数据
#             raw_data_list = [raw_row_data]
#             final_data_dict = self.get_trainable_data_from_raw_data(raw_data_list)
            
#             # 3. 检查是否为空
#             if not final_data_dict:
#                 print(f"[ERROR] Sample {index} filtered. Returning placeholder.")
#                 return self._get_placeholder_single_sample()

#             # 4. 去掉 Batch 维度 (B=1) 并构建 single_sample_dict
#             single_sample_dict = {}
#             for k, v in final_data_dict.items():
#                 if v is None: # [Security Check]
#                     continue
#                 if isinstance(v, torch.Tensor) and v.shape[0] == 1:
#                     single_sample_dict[k] = v.squeeze(0) 
#                 else:
#                     single_sample_dict[k] = v
            
#             # 5. 强制一致性检查 (Force Consistency)
#             # 确保返回的字典包含所有预期的 Key。如果某些可选 Key 缺失（因为文件缺失），手动补零。
#             # 以 placeholder 为模板检查缺失的 Key
#             placeholder_template = self._get_placeholder_single_sample()
#             for key, val in placeholder_template.items():
#                 if key not in single_sample_dict:
#                     # print(f"[WARN] Key {key} missing in sample {index}, filling with zeros.")
#                     single_sample_dict[key] = val # 使用占位符的零 Tensor 填充
#                 elif single_sample_dict[key] is None:
#                     single_sample_dict[key] = val

#             return single_sample_dict
        
#         except Exception as e:
#             print(f"[FATAL ERROR] Index {index}: {e}. Returning placeholder.")
#             return self._get_placeholder_single_sample()

#     def get_trainable_data_from_raw_data(self, raw_data_list) -> Dict[str, Tensor]:
#         all_data_lists = defaultdict(list)

#         for sample in raw_data_list:
#             V, V_in = self.opt.num_views, self.opt.num_input_views
#             assert V >= V_in

#             _pick_func = self._pick_even_view_indices if self.opt.load_even_views or not self.training else self._pick_random_view_indices
            
#             random_idxs = _pick_func(V_in)
#             _num_tries = 0
#             while not self._check_views_exist(sample, random_idxs):
#                 random_idxs = _pick_func(V_in)
#                 _num_tries += 1
#                 if _num_tries > 100:
#                     print(f"[WARNING] Only {len(random_idxs)} views found for {sample['__key__']}, repeating views")
#                     while len(random_idxs) < V_in:
#                         random_idxs.append(np.random.choice(random_idxs))
#                     break

#             except_idxs = random_idxs + [24, 39]
#             if self.opt.exclude_topdown_views:
#                 except_idxs += [25, 26]

#             for i in np.random.permutation(40):
#                 if len(random_idxs) >= V:
#                     break
#                 if f"{i:05d}.png" in sample and i not in except_idxs:
#                     try:
#                         val = sample[f"{i:05d}.png"]
#                         if isinstance(val, str):
#                             with open(val, "rb") as f:
#                                 val = f.read()
#                         _ = np.frombuffer(val, np.uint8)

#                         # _ = np.frombuffer(sample[f"{i:05d}.png"], np.uint8)
#                         assert sample[f"{i:05d}.json"] is not None
#                         random_idxs.append(i)
#                     except Exception as e:
#                         print(f"[WARN] Skipped view {i:05d} from {sample['__key__']} due to error: {e}")
#                         pass
            
#             while len(random_idxs) < V:
#                 random_idxs.append(np.random.choice(random_idxs))

#             sample_data_dict = defaultdict(list)
#             init_azi = None
            
#             # --- Load and process data for each view ---
#             for vid in random_idxs:
#                 png_key = f"{vid:05d}.png"
#                 json_key = f"{vid:05d}.json"

#                 if (png_key not in sample or sample[png_key] is None or
#                     json_key not in sample or sample[json_key] is None):
#                     print(f"[WARNING] Missing essential files for view {vid} of {sample['__key__']}, skipping this view.")
#                     continue

#                 try:
#                     image = self._load_png(sample[png_key])
#                     mask = image[3:4]
#                     image = image[:3] * mask + (1. - mask)
#                     # print(f"mask size start: {mask.shape}")
                    
#                     sample_data_dict["fxfycxcy"].append(self.fxfycxcy)
#                     sample_data_dict["image"].append(image)
#                     sample_data_dict["mask"].append(mask)

#                     c2w = self._load_camera_from_json(sample[json_key])
#                     c2w[1] *= -1
#                     c2w[[1, 2]] = c2w[[2, 1]]
#                     c2w[:3, 1:3] *= -1
#                     sample_data_dict["original_C2W"].append(torch.from_numpy(c2w).float())
                    
#                     ele, azi, dis = undo_orbit_camera(c2w)
#                     if init_azi is None:
#                         init_azi = azi
#                     azi = (azi - init_azi) % 360.
#                     ele = abs(ele) - 1e-8 if ele >= 0 else -(abs(ele) - 1e-8)
#                     new_c2w = torch.from_numpy(orbit_camera(ele, azi, dis)).float()
#                     sample_data_dict["C2W"].append(new_c2w)
#                     sample_data_dict["cam_pose"].append(torch.tensor([np.deg2rad(ele), np.deg2rad(azi), dis], dtype=torch.float32))

#                     # Optional files
#                     if self.opt.load_canny:
#                         gray = cv2.cvtColor(image.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY)
#                         canny = cv2.Canny((gray * 255.).astype(np.uint8), 100., 200.)
#                         canny = torch.from_numpy(canny).unsqueeze(0).float().repeat(3, 1, 1) / 255.
#                         canny = -canny + 1.
#                         sample_data_dict["canny"].append(canny)

#                     if self.opt.load_albedo:
#                         albedo_key = f"{vid:05d}_albedo.png"
#                         if albedo_key in sample and sample[albedo_key] is not None:
#                             albedo = self._load_png(sample[albedo_key])
#                             albedo = albedo * mask + (1. - mask)
#                             sample_data_dict["albedo"].append(albedo)

#                     if self.opt.load_normal or self.opt.load_coord:
#                         nd_key = f"{vid:05d}_nd.exr"
#                         # if nd_key not in sample:
#                         #     print(f"[DEBUG] {nd_key} not in sample for {sample['__key__']}")
#                         # elif sample[nd_key] is None:
#                         #     print(f"[DEBUG] {nd_key} exists but value is None for {sample['__key__']}")
#                         # else:
#                         #     print(f"[DEBUG] loading {nd_key} for {sample['__key__']}")

#                         if nd_key in sample and sample[nd_key] is not None:
#                             nd = self._load_png(sample[nd_key])
#                             if self.opt.load_normal:
#                                 normal = nd[:3] * 2. - 1.
#                                 normal[0, ...] *= -1
#                                 sample_data_dict["normal"].append(normal)
#                             if self.opt.load_coord or self.opt.load_depth:
#                                 depth = nd[3] * 5.
#                                 sample_data_dict["depth"].append(depth)
                    
#                     if self.opt.load_mr:
#                         mr_key = f"{vid:05d}_mr.png"
#                         if mr_key in sample and sample[mr_key] is not None:
#                             mr = self._load_png(sample[mr_key])
#                             mr = mr * mask + (1. - mask)
#                             sample_data_dict["mr"].append(mr)

#                 except Exception as e:
#                     print(f"[WARNING] Failed to process view {vid} for {sample['__key__']} with error: {e}, skipping this view.")
#                     continue
            
#             if not sample_data_dict:
#                 print(f"[WARNING] No views were successfully loaded for sample {sample['__key__']}, skipping.")
#                 continue

#             # for key, tensor_list in sample_data_dict.items():
#             #     single_sample_stack = torch.stack(tensor_list, dim=0)
#             #     all_data_lists[key].append(single_sample_stack)
#             # === 改造这里：每个 sample 内部也用分chunk stack ===
#             for key, tensor_list in sample_data_dict.items():
#                 single_sample_stack = stack_in_chunks(tensor_list, chunk_size=8)
#                 all_data_lists[key].append(single_sample_stack)

#         if not all_data_lists:
#             print(f"[ERROR] No valid samples were loaded for the entire batch. Returning placeholder batch.")
#             # 使用 raw_data_list 的长度作为 Batch size B
#             return self._get_placeholder_batch(len(raw_data_list))

#         final_return_dict = {}
#         for key, sample_stacks in all_data_lists.items():
#             final_return_dict[key] = stack_in_chunks(sample_stacks, chunk_size=4)
#         # for key, sample_stacks in all_data_lists.items():
#         #     final_return_dict[key] = torch.stack(sample_stacks, dim=0)

#         # --- Post-processing and data transformation ---
        
#         # 调整 C2W 矩阵的 Y 和 Z 轴，使其符合特定惯例
#         if "C2W" in final_return_dict:
#             final_return_dict["C2W"][:, :, :3, 1:3] *= -1

#         # 根据配置对相机进行归一化
#         if self.opt.norm_camera and "C2W" in final_return_dict:
#             # final_return_dict["C2W"] 的形状是 (B, V, 4, 4)
#             # 我们只用第一个视角来计算归一化 scale，形状是 (B, 3)
#             # torch.norm(..., dim=-1) 后形状变为 (B,)
#             scale = self.opt.norm_radius / (torch.norm(final_return_dict["C2W"][:, 0, :3, 3], dim=-1) + 1e-8)
            
#             # 将 scale 的形状从 (B,) 调整为 (B, 1, 1)，以便与 (B, V, 3) 进行广播
#             scale_broadcast = scale.unsqueeze(-1).unsqueeze(-1)
            
#             # 对所有视角的位置向量进行归一化
#             final_return_dict["C2W"][:, :, :3, 3] *= scale_broadcast
            
#             # 对所有视角的相机距离进行归一化
#             # final_return_dict["cam_pose"][:, :, 2] 的形状是 (B, V)
#             # scale.unsqueeze(-1) 的形状是 (B, 1)，可以广播
#             final_return_dict["cam_pose"][:, :, 2] *= scale.unsqueeze(-1)

#         # if "normal" not in final_return_dict:
#         #     print("[DEBUG]normal not in here")
#         # 处理法线数据
#         if self.opt.load_normal and "normal" in final_return_dict:
#             normals = normalize_normals(final_return_dict["normal"], final_return_dict["original_C2W"], i=0)
#             normals = torch.einsum("bnrc,bvrhw->bvrhw", final_return_dict["C2W"][:, :, :3, :3], normals).contiguous()
#             normals = normals * 0.5 + 0.5
#             normals = normals * final_return_dict["mask"] + (1. - final_return_dict["mask"])
#             final_return_dict["normal"] = normals
#             # print("[DEBUG]get normals chuli")
#             final_return_dict.pop("original_C2W")
        
#         # 处理坐标和深度数据
#         if self.opt.load_coord and "depth" in final_return_dict:
#             # 在进行 unproject_depth 之前，depth 应该是 (B, V, H, W)
#             # final_return_dict["depth"] 的形状是 (B, V, H, W)
#             # final_return_dict["mask"] 的形状是 (B, V, 1, H, W)
#             mask_2d = final_return_dict["mask"].squeeze(2)

#             coords = unproject_depth(final_return_dict["depth"] * mask_2d, final_return_dict["C2W"], final_return_dict["fxfycxcy"])
#             #ADD,因为后面coords与mask做乘法，需要五维
#             final_return_dict["mask"] = final_return_dict["mask"]
#             coords = coords * 0.5 + 0.5
#             coords = coords * final_return_dict["mask"] + (1. - final_return_dict["mask"])
#             final_return_dict["coord"] = coords
#             if not self.opt.load_depth:
#                 final_return_dict.pop("depth")
                
#         if self.opt.load_depth and "depth" in final_return_dict:
#             depths = final_return_dict["depth"].unsqueeze(2) * final_return_dict["mask"]
#             assert depths.min() >= 0.
#             if self.opt.normalize_depth:
#                 depths_reshaped = depths.view(depths.shape[0], depths.shape[1], -1)
#                 depths_max = depths_reshaped.max(dim=-1, keepdim=True).values
#                 depths = depths_reshaped / depths_max.clamp(min=1e-6)
#                 depths = depths.view(final_return_dict["depth"].shape[0], final_return_dict["depth"].shape[1], 1, final_return_dict["depth"].shape[2], final_return_dict["depth"].shape[3])

#             depths = -depths + 1.
#             final_return_dict["depth"] = depths.repeat(1, 1, 3, 1, 1)

#         # Resize to the input resolution
#         for key in ["image", "mask", "albedo", "normal", "coord", "depth", "mr", "canny"]:
#             if key in final_return_dict:
#                 batch_size, num_views, C, H, W = final_return_dict[key].shape
#                 final_return_dict[key] = tF.interpolate(
#                     final_return_dict[key].view(-1, C, H, W),
#                     size=(self.opt.input_res, self.opt.input_res),
#                     mode="bilinear", align_corners=False, antialias=True
#                 ).view(batch_size, num_views, C, self.opt.input_res, self.opt.input_res)
#                 # print(f"[DEBUG] {key}.shape: {final_return_dict[key].shape}")


#         # Handle anti-aliased normal, coord and depth
#         if self.opt.load_normal and "normal" in final_return_dict:
#             final_return_dict["normal"] = final_return_dict["normal"] * final_return_dict["mask"] + (1. - final_return_dict["mask"])
#         if self.opt.load_coord and "coord" in final_return_dict:
#             final_return_dict["coord"] = final_return_dict["coord"] * final_return_dict["mask"] + (1. - final_return_dict["mask"])
#         if self.opt.load_depth and "depth" in final_return_dict:
#             final_return_dict["depth"] = final_return_dict["depth"] * final_return_dict["mask"] + (1. - final_return_dict["mask"])

#         # Load precomputed caption embeddings
#         for sample_uid in raw_data_list:
#             if "uid" in sample_uid:
#                 uid = sample_uid["uid"].decode("utf-8").split("/")[-1].split(".")[0]
#                 prompt_embed_path = f"{self.opt.prompt_embed_dir}/{uid}.npy"
#                 if os.path.exists(prompt_embed_path):
#                     embed = torch.from_numpy(np.load(prompt_embed_path)).float()
#                 else:
#                     embed = self.negative_prompt_embed

#                 if embed.dim() == 2: # e.g., (77, 768) -> (1, 77, 768)
#                     embed = embed.unsqueeze(0)
                    
#                 # final_return_dict["prompt_embed"] = embed.repeat(len(raw_data_list), 1, 1)
#                 final_return_dict["prompt_embed"] = embed

#                 if "xl" in self.opt.pretrained_model_name_or_path:
#                     pooled_embed_path = f"{self.opt.prompt_embed_dir}/{uid}_pooled.npy"
                    
#                     if os.path.exists(pooled_embed_path):
#                         pooled_embed = torch.from_numpy(np.load(pooled_embed_path)).float()
#                     else:
#                         pooled_embed = self.negative_pooled_prompt_embed
                        
#                     if pooled_embed.dim() == 1: # e.g., (1280) -> (1, 1280)
#                         pooled_embed = pooled_embed.unsqueeze(0)
                        
#                     final_return_dict["pooled_prompt_embed"] = pooled_embed
                    
#                     mask_path = f"{self.opt.prompt_embed_dir}/{uid}_attention_mask.npy"
                    
#                     if os.path.exists(mask_path):
#                         mask = torch.from_numpy(np.load(mask_path)).float()
#                     else:
#                         mask = self.negative_prompt_attention_mask
                        
#                     if mask.dim() == 1: # e.g., (77) -> (1, 77)
#                         mask = mask.unsqueeze(0)

#                     final_return_dict["prompt_attention_mask"] = mask

#         for key in final_return_dict.keys():
#             if not isinstance(final_return_dict[key], Tensor):
#                 try:
#                     print(f"[WARN] Converting non-tensor value for key [{key}] to tensor.")
#                     final_return_dict[key] = torch.tensor(final_return_dict[key])
#                 except Exception as e:
#                     print(f"[FATAL] Failed to convert non-tensor value for key [{key}] to tensor. Error: {e}")
#                     # 关键：如果无法转换，将其设置为一个零 Tensor 占位符，防止 NoneType 错误
#                     final_return_dict[key] = torch.zeros(1)

#         return dict(final_return_dict)        

#     def _load_png(self, path_or_bytes: str | bytes, uint16=False) -> torch.Tensor:
#         '''支持 EXR 和 PNG 的读取'''
#         if isinstance(path_or_bytes, str):
#             ext = os.path.splitext(path_or_bytes)[1].lower()
#             if ext == ".exr":
#                 # 用 OpenEXR + Imath 读取
#                 try:
#                     exr_file = OpenEXR.InputFile(path_or_bytes)
#                     dw = exr_file.header()['dataWindow']
#                     size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
#                     pt = Imath.PixelType(Imath.PixelType.FLOAT)
#                     R = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32).reshape(size[1], size[0])
#                     G = np.frombuffer(exr_file.channel('G', pt), dtype=np.float32).reshape(size[1], size[0])
#                     B = np.frombuffer(exr_file.channel('B', pt), dtype=np.float32).reshape(size[1], size[0])
#                     try:
#                         A = np.frombuffer(exr_file.channel('A', pt), dtype=np.float32).reshape(size[1], size[0])
#                     except:
#                         try: # 有些 EXR 可能命名为 Z (深度)
#                             A = np.frombuffer(exr_file.channel('Z', pt), dtype=np.float32).reshape(size[1], size[0])
#                         except:
#                             A = np.zeros_like(R)
#                     img = np.stack([R, G, B], axis=-1)
#                 except Exception as e:
#                     print(f"[WARN] EXR decode failed: {path_or_bytes}, error: {e}")
#                     img = np.zeros((512, 512, 3), dtype=np.float32)
#             else:
#                 # 读取普通 PNG
#                 with open(path_or_bytes, "rb") as f:
#                     png_bytes = f.read()
#                 img = np.frombuffer(png_bytes, np.uint8)
#                 img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
#                 if img is None:
#                     print(f"[WARN] PNG decode failed: {path_or_bytes}")
#                     img = np.zeros((512, 512, 4), dtype=np.float32)
#                 img = img.astype(np.float32) / (65535. if uint16 else 255.)
#         else:
#             # bytes 输入暂时只处理 PNG
#             img = np.frombuffer(path_or_bytes, np.uint8)
#             img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
#             if img is None:
#                 print(f"[WARN] PNG decode failed from bytes. Returning zero placeholder.")
#                 img = np.zeros((512, 512, 4), dtype=np.float32)
#             img = img.astype(np.float32) / (65535. if uint16 else 255.)

#         if img.ndim == 2:
#             # 灰度图，添加颜色通道和 Alpha/Depth 通道
#             img = np.stack([img, img, img, np.ones_like(img)], axis=-1)
#         elif img.shape[2] == 3:
#             # BGR -> RGB
#             rgb = img[:, :, :3][..., ::-1].copy()
#             alpha_channel = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype)
#             img = np.concatenate([rgb, alpha_channel], axis=2)
#         elif img.shape[2] == 4:
#             # RGBA/BGRA，确保前三通道是 RGB
#             img[:, :, :3] = img[:, :, :3][..., ::-1]
#             img = img.copy()
#         elif img.shape[2] > 4:
#              # 多通道，只保留前4个通道并确保前三通道是 RGB
#             img = img[:, :, :4]
#             img[:, :, :3] = img[:, :, :3][..., ::-1]
#             img = img.copy()

#         return torch.from_numpy(img.copy()).nan_to_num_(0.).permute(2, 0, 1)

#     def _get_placeholder_single_sample(self) -> Dict[str, Tensor]:
#         """为单个样本创建零 Tensor 占位符 (不含 Batch 维度)"""
#         V = self.opt.num_views
#         R = self.opt.input_res # 最终分辨率
        
#         # --- 占位符数据 ---
#         placeholder_images = torch.zeros(V, 3, R, R, dtype=torch.float32)
#         placeholder_c2w = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(V, 1, 1) # (V, 4, 4)
#         placeholder_cam_pose = torch.zeros(V, 3, dtype=torch.float32) # (V, 3)

#         # 填充所有必需的键
#         placeholder_dict = {
#             # 图像和基础数据
#             "images": placeholder_images, # (V, 3, R, R)
#             "C2W": placeholder_c2w,       # (V, 4, 4)
#             "fxfycxcy": self.fxfycxcy,    # (4,)
#             "cam_pose": placeholder_cam_pose, # (V, 3)
#             "mask": torch.zeros(V, 1, R, R, dtype=torch.float32), # (V, 1, R, R)
#             # 提示词嵌入 (B=1, 所以没有 Batch 维度)
#             "prompt_embed": self.negative_prompt_embed, 
#         }
        
#         # --- SDXL 占位符 ---
#         if "xl" in self.opt.pretrained_model_name_or_path:
#              placeholder_dict["pooled_prompt_embed"] = self.negative_pooled_prompt_embed.squeeze(0)
#              placeholder_dict["prompt_attention_mask"] = self.negative_prompt_attention_mask.squeeze(0)

#         if self.opt.load_canny:
#              placeholder_dict["canny"] = torch.zeros(V, 3, R, R, dtype=torch.float32)
#         if self.opt.load_albedo:
#              placeholder_dict["albedo"] = torch.zeros(V, 3, R, R, dtype=torch.float32)
#         if self.opt.load_normal:
#              placeholder_dict["normal"] = torch.zeros(V, 3, R, R, dtype=torch.float32)
#         if self.opt.load_coord:
#              placeholder_dict["coord"] = torch.zeros(V, 3, R, R, dtype=torch.float32)
#         if self.opt.load_depth:
#              placeholder_dict["depth"] = torch.zeros(V, 3, R, R, dtype=torch.float32) 
#         if self.opt.load_mr:
#              placeholder_dict["mr"] = torch.zeros(V, 3, R, R, dtype=torch.float32)
#         return placeholder_dict

#     def _load_camera_from_json(self, json_path_or_bytes: str | bytes) -> np.ndarray:
#         # 如果是路径，先读成 bytes
#         if isinstance(json_path_or_bytes, str):
#             if not os.path.exists(json_path_or_bytes):
#                 print(f"[WARN] File does not exist: {json_path_or_bytes}")
#                 return np.eye(4)  # Return identity matrix as fallback
#             with open(json_path_or_bytes, "rb") as f:
#                 json_bytes = f.read()
#         else:
#             json_bytes = json_path_or_bytes

#         #json_dict = json.loads(json_bytes)
#         try:
#             json_dict = json.loads(json_bytes)
#         except json.JSONDecodeError:
#             print(f"[WARN] Invalid or empty JSON file: {json_path_or_bytes}, skipping...")
#             # 返回一个默认相机矩阵，避免报错
#             return np.eye(4)

#         c2w = np.eye(4)
#         c2w[:3, 0] = np.array(json_dict["x"])
#         c2w[:3, 1] = np.array(json_dict["y"])
#         c2w[:3, 2] = np.array(json_dict["z"])
#         c2w[:3, 3] = np.array(json_dict["origin"])
#         return c2w

#     def _pick_even_view_indices(self, num_views: int = 4) -> List[int]:
#         assert 12 % num_views == 0  # `12` for even-view sampling in GObjaverse

#         if np.random.rand() < 2/3:
#             index0 = np.random.choice(range(24))  # 0~23: 24 views in ele from [5, 30]; hard-coded for GObjaverse
#             return [(index0 + (24 // num_views)*i) % 24 for i in range(num_views)]
#         else:
#             index0 = np.random.choice(range(12))  # 27~38: 12 views in ele from [-5, 5]; hard-coded for GObjaverse
#             return [((index0 + (12 // num_views)*i) % 12 + 27) for i in range(num_views)]

#     def _pick_random_view_indices(self, num_views: int = 4) -> List[int]:
#         assert num_views <= 40  # `40` is hard-coded for GObjaverse

#         indices = (set(range(40)) - set([25, 26])) if self.opt.exclude_topdown_views else (set(range(40)))  # `40` is hard-coded for GObjaverse
#         return np.random.choice(list(indices), num_views, replace=False).tolist()

#     def _check_views_exist(self, sample: Dict[str, Union[str, bytes]], vids: List[int]) -> bool:
#         for vid in vids:
#             if f"{vid:05d}.png" not in sample:
#                 return False
#             try:
#                 assert sample[f"{vid:05d}.png"] is not None and sample[f"{vid:05d}.json"] is not None
#             except:  # TypeError: a bytes-like object is required, not 'NoneType'; KeyError: '00001.json'
#                 return False
#         return True
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
    """分chunk拼接，避免一次性占用过大内存。永远返回 Tensor，不返回 None。"""
    if not tensor_list:
        return torch.tensor([])
    
    valid_tensors = [t for t in tensor_list if t is not None]
    if not valid_tensors:
        return torch.tensor([])

    chunks = []
    try:
        for i in range(0, len(valid_tensors), chunk_size):
            sub = valid_tensors[i:i + chunk_size]
            if len(sub) > 0:
                chunks.append(torch.stack(sub, dim=0))
    except Exception:
        pass 
    
    if not chunks:
        return torch.tensor([])
    
    try:
        return torch.cat(chunks, dim=0)
    except Exception:
        return torch.tensor([])

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
        self.data_source = data_source
        self.training = training
        self.root_dir = root_dir

        fxfy = getattr(opt, 'fxfy', 512.0)
        if fxfy is None: fxfy = 512.0
        self.fxfycxcy = torch.tensor([fxfy, fxfy, 0.5, 0.5], dtype=torch.float32)

        self.negative_prompt_embed = torch.zeros(77, 768).float()
        self.negative_pooled_prompt_embed = torch.zeros(1, 1280).float()
        self.negative_prompt_attention_mask = torch.ones(1, 77).float()

        embed_dir = getattr(opt, 'prompt_embed_dir', None)
        if embed_dir is not None:
            try:
                path = f"{embed_dir}/null.npy"
                if os.path.exists(path):
                    self.negative_prompt_embed = torch.from_numpy(np.load(path)).float()
            except: pass
            try:
                path = f"{embed_dir}/null_pooled.npy"
                if os.path.exists(path):
                    self.negative_pooled_prompt_embed = torch.from_numpy(np.load(path)).float()
            except: pass
            try:
                path = f"{embed_dir}/null_attention_mask.npy"
                if os.path.exists(path):
                    self.negative_prompt_attention_mask = torch.from_numpy(np.load(path)).float()
            except: pass

            if "xl" in getattr(opt, 'pretrained_model_name_or_path', ""):
                 self.negative_prompt_embed = torch.zeros_like(self.negative_prompt_embed)
                 self.negative_pooled_prompt_embed = torch.zeros_like(self.negative_pooled_prompt_embed)

    def __len__(self):
        return getattr(self.opt, 'dataset_size', 1000)

    def get_trainable_data_from_raw_data(self, raw_data_list) -> Dict[str, Tensor]:
        single_template = self._get_placeholder_single_sample()
        # 默认 batch size 至少为 1，防止 raw_data_list 为空
        batch_size_in = len(raw_data_list) if raw_data_list else 1
        
        if not raw_data_list:
            return self._create_safe_batch_from_template(single_template, 1)

        all_data_lists = defaultdict(list)

        for sample in raw_data_list:
            if sample is None: continue
            
            V = getattr(self.opt, 'num_views', 4)
            V_in = getattr(self.opt, 'num_input_views', 1)
            load_even = getattr(self.opt, 'load_even_views', False)
            _pick_func = self._pick_even_view_indices if load_even or not self.training else self._pick_random_view_indices
            
            random_idxs = _pick_func(V_in)
            for _ in range(20):
                if all(f"{vid:05d}.png" in sample for vid in random_idxs): break
                random_idxs = _pick_func(V_in)
            
            while len(random_idxs) < V:
                random_idxs.append(random_idxs[-1])

            sample_data_dict = defaultdict(list)
            init_azi = None
            
            for vid in random_idxs:
                png_key = f"{vid:05d}.png"
                json_key = f"{vid:05d}.json"
                if png_key not in sample or json_key not in sample: continue

                try:
                    image = self._load_png(sample[png_key])
                    mask = image[3:4]
                    image = image[:3] * mask + (1. - mask)
                    
                    sample_data_dict["fxfycxcy"].append(self.fxfycxcy)
                    sample_data_dict["image"].append(image)
                    sample_data_dict["mask"].append(mask)

                    c2w = self._load_camera_from_json(sample[json_key])
                    c2w[1] *= -1; c2w[[1, 2]] = c2w[[2, 1]]; c2w[:3, 1:3] *= -1
                    sample_data_dict["original_C2W"].append(torch.from_numpy(c2w).float())
                    
                    ele, azi, dis = undo_orbit_camera(c2w)
                    if init_azi is None: init_azi = azi
                    azi = (azi - init_azi) % 360.
                    ele = abs(ele) - 1e-8 if ele >= 0 else -(abs(ele) - 1e-8)
                    new_c2w = torch.from_numpy(orbit_camera(ele, azi, dis)).float()
                    sample_data_dict["C2W"].append(new_c2w)
                    sample_data_dict["cam_pose"].append(torch.tensor([np.deg2rad(ele), np.deg2rad(azi), dis], dtype=torch.float32))

                    if getattr(self.opt, 'load_canny', False):
                        gray = cv2.cvtColor(image.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2GRAY)
                        canny = cv2.Canny((gray * 255.).astype(np.uint8), 100., 200.)
                        canny = torch.from_numpy(canny).unsqueeze(0).float().repeat(3, 1, 1) / 255.
                        canny = -canny + 1.
                        sample_data_dict["canny"].append(canny)

                    if getattr(self.opt, 'load_albedo', False):
                        k = f"{vid:05d}_albedo.png"
                        if k in sample: sample_data_dict["albedo"].append(self._load_png(sample[k]) * mask + (1. - mask))

                    if getattr(self.opt, 'load_normal', False) or getattr(self.opt, 'load_coord', False):
                        k = f"{vid:05d}_nd.exr"
                        if k in sample:
                            nd = self._load_png(sample[k])
                            if getattr(self.opt, 'load_normal', False):
                                normal = nd[:3] * 2. - 1.
                                normal[0, ...] *= -1
                                sample_data_dict["normal"].append(normal)
                            if getattr(self.opt, 'load_coord', False) or getattr(self.opt, 'load_depth', False):
                                sample_data_dict["depth"].append(nd[3] * 5.)
                    
                    if getattr(self.opt, 'load_mr', False):
                        k = f"{vid:05d}_mr.png"
                        if k in sample: sample_data_dict["mr"].append(self._load_png(sample[k]) * mask + (1. - mask))

                except Exception: continue
            
            if not sample_data_dict or len(sample_data_dict["image"]) == 0: continue

            temp_sample_stack = {}
            valid_view_stack = True
            for key, tensor_list in sample_data_dict.items():
                stack = stack_in_chunks(tensor_list, chunk_size=8)
                if stack.numel() == 0: # Check empty tensor
                    valid_view_stack = False; break
                temp_sample_stack[key] = stack
            
            if valid_view_stack:
                for k, v in temp_sample_stack.items():
                    all_data_lists[k].append(v)

        # --- 核心修正点 ---
        # 即使 all_data_lists 为空，我们也不能返回空字典。
        # 必须返回一个合法的、全是 0 的 Batch。
        if not all_data_lists or not all_data_lists.get('image'):
             return self._create_safe_batch_from_template(single_template, max(1, batch_size_in))

        final_return_dict = {}
        for key, sample_stacks in all_data_lists.items():
            res = stack_in_chunks(sample_stacks, chunk_size=4)
            # 只有当 Tensor 非空时才加入
            if res.numel() > 0:
                final_return_dict[key] = res
        
        if 'image' not in final_return_dict:
             return self._create_safe_batch_from_template(single_template, max(1, batch_size_in))
             
        actual_batch_size = final_return_dict['image'].shape[0]

        # Post-processing
        if "C2W" in final_return_dict:
            final_return_dict["C2W"][:, :, :3, 1:3] *= -1

        if getattr(self.opt, 'norm_camera', False) and "C2W" in final_return_dict:
            scale = getattr(self.opt, 'norm_radius', 1.0) / (torch.norm(final_return_dict["C2W"][:, 0, :3, 3], dim=-1) + 1e-8)
            final_return_dict["C2W"][:, :, :3, 3] *= scale.reshape(-1, 1, 1)
            final_return_dict["cam_pose"][:, :, 2] *= scale.reshape(-1, 1)

        if getattr(self.opt, 'load_normal', False) and "normal" in final_return_dict:
            if "original_C2W" in final_return_dict:
                normals = normalize_normals(final_return_dict["normal"], final_return_dict["original_C2W"], i=0)
                normals = torch.einsum("bnrc,bvrhw->bvrhw", final_return_dict["C2W"][:, :, :3, :3], normals).contiguous()
                normals = normals * 0.5 + 0.5
                normals = normals * final_return_dict["mask"] + (1. - final_return_dict["mask"])
                final_return_dict["normal"] = normals
        
        if "original_C2W" in final_return_dict: final_return_dict.pop("original_C2W")

        if getattr(self.opt, 'load_coord', False) and "depth" in final_return_dict:
            mask_2d = final_return_dict["mask"].squeeze(2)
            coords = unproject_depth(final_return_dict["depth"] * mask_2d, final_return_dict["C2W"], final_return_dict["fxfycxcy"])
            coords = coords * 0.5 + 0.5
            coords = coords * final_return_dict["mask"] + (1. - final_return_dict["mask"])
            final_return_dict["coord"] = coords
            if not getattr(self.opt, 'load_depth', False): final_return_dict.pop("depth")

        if getattr(self.opt, 'load_depth', False) and "depth" in final_return_dict:
            depths = final_return_dict["depth"].unsqueeze(2) * final_return_dict["mask"]
            if getattr(self.opt, 'normalize_depth', False):
                depths_reshaped = depths.view(depths.shape[0], depths.shape[1], -1)
                depths_max = depths_reshaped.max(dim=-1, keepdim=True).values
                depths = depths_reshaped / depths_max.clamp(min=1e-6)
                depths = depths.view(final_return_dict["depth"].shape)
            final_return_dict["depth"] = (-depths + 1.).repeat(1, 1, 3, 1, 1)

        input_res = getattr(self.opt, 'input_res', 256)
        for key in ["image", "mask", "albedo", "normal", "coord", "depth", "mr", "canny"]:
            if key in final_return_dict:
                B_sz, num_views, C, H, W = final_return_dict[key].shape
                final_return_dict[key] = tF.interpolate(
                    final_return_dict[key].view(-1, C, H, W),
                    size=(input_res, input_res),
                    mode="bilinear", align_corners=False, antialias=True
                ).view(B_sz, num_views, C, input_res, input_res)

        for key in ["normal", "coord", "depth"]:
            if key in final_return_dict and getattr(self.opt, f"load_{key}", False):
                final_return_dict[key] = final_return_dict[key] * final_return_dict["mask"] + (1. - final_return_dict["mask"])

        if "prompt_embed" not in final_return_dict:
            final_return_dict["prompt_embed"] = self.negative_prompt_embed.unsqueeze(0).repeat(actual_batch_size, 1, 1)
        
        if "xl" in getattr(self.opt, 'pretrained_model_name_or_path', ""):
            if "pooled_prompt_embed" not in final_return_dict:
                final_return_dict["pooled_prompt_embed"] = self.negative_pooled_prompt_embed.unsqueeze(0).repeat(actual_batch_size, 1, 1)
            if "prompt_attention_mask" not in final_return_dict:
                final_return_dict["prompt_attention_mask"] = self.negative_prompt_attention_mask.unsqueeze(0).repeat(actual_batch_size, 1, 1)

        return self._sanitize_batch(final_return_dict, single_template, actual_batch_size)

    def _sanitize_batch(self, batch, template, batch_size):
        """强制清洗 Batch"""
        clean_batch = {}
        for key, template_tensor in template.items():
            val = batch.get(key, None)
            
            if val is None or not isinstance(val, torch.Tensor):
                # print(f"[DEBUG] Filling missing key {key} with zeros.")
                clean_batch[key] = template_tensor.unsqueeze(0).repeat(batch_size, *([1]*template_tensor.ndim))
            else:
                clean_batch[key] = val
        return clean_batch

    def _create_safe_batch_from_template(self, template, batch_size):
        safe_batch = {}
        for k, v in template.items():
            safe_batch[k] = v.unsqueeze(0).repeat(batch_size, *([1]*v.ndim))
        return safe_batch

    def _get_placeholder_single_sample(self) -> Dict[str, Tensor]:
        V = getattr(self.opt, 'num_views', 4)
        R = getattr(self.opt, 'input_res', 256)
        
        placeholder_fxfycxcy = self.fxfycxcy.unsqueeze(0).repeat(V, 1)

        placeholder_dict = {
            "image": torch.zeros(V, 3, R, R, dtype=torch.float32), 
            "C2W": torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(V, 1, 1),       
            "fxfycxcy": placeholder_fxfycxcy,    
            "cam_pose": torch.zeros(V, 3, dtype=torch.float32), 
            "mask": torch.zeros(V, 1, R, R, dtype=torch.float32), 
            "prompt_embed": self.negative_prompt_embed, 
        }
        
        if "xl" in getattr(self.opt, 'pretrained_model_name_or_path', ""):
             placeholder_dict["pooled_prompt_embed"] = self.negative_pooled_prompt_embed.squeeze(0)
             placeholder_dict["prompt_attention_mask"] = self.negative_prompt_attention_mask.squeeze(0)

        if getattr(self.opt, 'load_canny', False): placeholder_dict["canny"] = torch.zeros(V, 3, R, R, dtype=torch.float32)
        if getattr(self.opt, 'load_albedo', False): placeholder_dict["albedo"] = torch.zeros(V, 3, R, R, dtype=torch.float32)
        if getattr(self.opt, 'load_normal', False): placeholder_dict["normal"] = torch.zeros(V, 3, R, R, dtype=torch.float32)
        if getattr(self.opt, 'load_coord', False): placeholder_dict["coord"] = torch.zeros(V, 3, R, R, dtype=torch.float32)
        if getattr(self.opt, 'load_depth', False): placeholder_dict["depth"] = torch.zeros(V, 3, R, R, dtype=torch.float32) 
        if getattr(self.opt, 'load_mr', False): placeholder_dict["mr"] = torch.zeros(V, 3, R, R, dtype=torch.float32)

        for k, v in placeholder_dict.items():
            if v is None: placeholder_dict[k] = torch.zeros(1)

        return placeholder_dict

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        raw = None
        try: raw = self.data_source[index]
        except: pass
        if raw is None: 
            batch = self._create_safe_batch_from_template(self._get_placeholder_single_sample(), 1)
        else:
            batch = self.get_trainable_data_from_raw_data([raw])
        single = {}
        for k, v in batch.items():
            if v.shape[0] == 1: single[k] = v.squeeze(0)
            else: single[k] = v
        return single

    def _load_png(self, path_or_bytes: str | bytes, uint16=False) -> torch.Tensor:
        try:
            if isinstance(path_or_bytes, str):
                ext = os.path.splitext(path_or_bytes)[1].lower()
                if ext == ".exr":
                    exr_file = OpenEXR.InputFile(path_or_bytes)
                    pt = Imath.PixelType(Imath.PixelType.FLOAT)
                    dw = exr_file.header()['dataWindow']
                    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
                    R = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32).reshape(size[1], size[0])
                    G = np.frombuffer(exr_file.channel('G', pt), dtype=np.float32).reshape(size[1], size[0])
                    B = np.frombuffer(exr_file.channel('B', pt), dtype=np.float32).reshape(size[1], size[0])
                    try: A = np.frombuffer(exr_file.channel('A', pt), dtype=np.float32).reshape(size[1], size[0])
                    except: A = np.zeros_like(R)
                    img = np.stack([R, G, B], axis=-1)
                else:
                    with open(path_or_bytes, "rb") as f: b = f.read()
                    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
            else:
                img = cv2.imdecode(np.frombuffer(path_or_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
            
            if img is None: return torch.zeros(3, 512, 512, dtype=torch.float32)
            img = img.astype(np.float32) / (65535. if uint16 else 255.)
            if img.ndim == 2: img = np.stack([img, img, img, np.ones_like(img)], axis=-1)
            elif img.shape[2] == 3: img = np.concatenate([img[..., ::-1], np.ones_like(img[..., :1])], axis=2)
            elif img.shape[2] == 4: img = img.copy(); img[..., :3] = img[..., :3][..., ::-1]
            return torch.from_numpy(img).nan_to_num_(0.).permute(2, 0, 1)
        except:
            return torch.zeros(3, 512, 512, dtype=torch.float32)

    def _load_camera_from_json(self, json_path_or_bytes) -> np.ndarray:
        try:
            if isinstance(json_path_or_bytes, str):
                if not os.path.exists(json_path_or_bytes): return np.eye(4)
                with open(json_path_or_bytes, "rb") as f: b = f.read()
            else: b = json_path_or_bytes
            d = json.loads(b)
            c2w = np.eye(4); c2w[:3, 0] = d["x"]; c2w[:3, 1] = d["y"]; c2w[:3, 2] = d["z"]; c2w[:3, 3] = d["origin"]
            return c2w
        except: return np.eye(4)

    def _pick_even_view_indices(self, num_views=4):
        if np.random.rand() < 2/3: i0 = np.random.choice(range(24)); return [(i0 + (24//num_views)*i)%24 for i in range(num_views)]
        else: i0 = np.random.choice(range(12)); return [((i0 + (12//num_views)*i)%12 + 27) for i in range(num_views)]

    def _pick_random_view_indices(self, num_views=4):
        ex = getattr(self.opt, 'exclude_topdown_views', False)
        s = (set(range(40)) - set([25, 26])) if ex else set(range(40))
        return np.random.choice(list(s), num_views, replace=False).tolist()
    
    def _check_views_exist(self, sample, vids): return True