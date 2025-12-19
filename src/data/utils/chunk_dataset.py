from torch.utils.data.dataset import IterableDataset
from torch.utils.data.dataloader import DataLoader
import torch
import os
import cv2
import numpy as np
from torchvision import transforms
from typing import List, Dict, Any
from typing import Iterator, Optional



# This is a pseudo class that constructs a dataset from chunks
class ChunkedDataset(IterableDataset):
    """
        通用本地数据集读取 Dataset.
        会在初始化时扫描本地目录，构建 data_list。
        """

    # def __init__(self, root_dir: str, *args, **kwargs):
    #     """
    #     Args:
    #         root_dir: 数据集的根目录, e.g. "/path/to/G-Objaverse"
    #     """
    #     super().__init__()
    #     self.root_dir = root_dir
    #     self.data_list: List[Dict[str, Any]] = self._scan_dataset()
    def __init__(self, data_source, *args, **kwargs):
        super().__init__()
        if isinstance(data_source, str):
            self.root_dir = data_source
            self.data_list: List[Dict[str, Any]] = self._scan_dataset()
        else:
            # print("[DEBUG]INTO HERE")
            self.root_dir = None  # 不使用路径
            self.data_list = data_source.samples
            print("type(data_source):", type(data_source))
            if hasattr(data_source, "samples"):
                print("len(data_source.samples):", len(data_source.samples))
                # self.data_list = self.data_list[:50] # 只保留前 50 个样本
                # print("DEBUG: Temporarily reduced samples to:", len(self.data_list))

    def _scan_dataset(self) -> List[Dict[str, Any]]:
        """
        遍历 root_dir，构建每个样本的字典.
        返回一个 data_list: List[Dict]
        """
        data_list = []
        for dir_id in os.listdir(self.root_dir):
            dir_path = os.path.join(self.root_dir, dir_id)
            if not os.path.isdir(dir_path):
                continue
            for object_id in os.listdir(dir_path):
                # object_dir = os.path.join(dir_path, object_id, "campos_512_v4")
                object_dir = os.path.join(dir_path, object_id)
                if not os.path.isdir(object_dir):
                    continue

                sample = {"__key__": f"{dir_id}/{object_id}", "uid": f"{dir_id}/{object_id}".encode("utf-8")}

                for i in range(40):  # 假设固定 40 views
                    view_dir = os.path.join(object_dir, f"{i:05}")
                    image_path = os.path.join(view_dir, f"{i:05}.png")
                    albedo_path = os.path.join(view_dir, f"{i:05}_albedo.png")
                    mr_path = os.path.join(view_dir, f"{i:05}_mr.png")
                    nd_path = os.path.join(view_dir, f"{i:05}_nd.exr")
                    ng_path = os.path.join(view_dir, f"{i:05}_ng.exr")
                    transform_path = os.path.join(view_dir, f"{i:05}.json")

                    # print(f"[DEBUG]nd_path:{nd_path}")

                    if os.path.exists(image_path):
                        sample[f"{i:05}.png"] = image_path
                    if os.path.exists(albedo_path):
                        sample[f"{i:05}_albedo.png"] = albedo_path
                    if os.path.exists(mr_path):
                        sample[f"{i:05}_mr.png"] = mr_path
                    if os.path.exists(nd_path):
                        # print("[DEBUG]success save nd")
                        sample[f"{i:05}_nd.exr"] = nd_path
                    if os.path.exists(ng_path):
                        sample[f"{i:05}_ng.exr"] = ng_path
                    if os.path.exists(transform_path):
                        sample[f"{i:05}.json"] = transform_path


                data_list.append(sample)
        return data_list

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start, iter_end = 0, len(self.data_list)
        else:
            per_worker = int(len(self.data_list) / worker_info.num_workers)
            iter_start = worker_info.id * per_worker
            iter_end = iter_start + per_worker if worker_info.id != worker_info.num_workers - 1 else len(self.data_list)

        for i in range(iter_start, iter_end):
            raw_data = [self.data_list[i]]
            # print(f"[DEBUG]raw_data_size:{len(raw_data)}")
            batch = self.get_trainable_data_from_raw_data(raw_data)
            if batch is None:
                print(f"[INFO] Skipping sample {i} because batch is None")
                continue
            yield batch

    def get_trainable_data_from_raw_data(self, raw_data_list: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    
        pass

# This is a pseudo class that loads data in chunks from HDFS
class ChunkedDataLoader(DataLoader):
    """
       通用 DataLoader, 未来可扩展成 HDFS 或其他存储
       """

    def __init__(self, dataset: ChunkedDataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)

    def __iter__(self) -> Iterator:
        for batch in super().__iter__():
            # 如果 batch 为 None 或包含 None，跳过
            if self.skip_none:
                if batch is None:
                    continue
                if isinstance(batch, (list, tuple)) and any(x is None for x in batch):
                    continue
            yield batch
    # raise NotImplementedError("Please implement your own dataloading logic")

class ChunkedDataLoader(DataLoader):
    """
    通用 DataLoader，自动跳过 None
    """

    def __init__(self, dataset: ChunkedDataset, *args, skip_none=True, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.skip_none = skip_none

    def __iter__(self) -> Iterator:
        for batch in super().__iter__():
            if self.skip_none:
                if batch is None:
                    continue
                if isinstance(batch, (list, tuple)) and any(x is None for x in batch):
                    continue
            yield batch

#test
# dataset = ChunkedDataset("/opt/data/private/wjy/LRY/DiffSplat-main/dataset/train")
#
# for i, sample in enumerate(dataset):
#     print(f"[CHECK] {i}: keys={list(sample.keys())}")