# This is a pseudo class that collect data in chunks from HDFS
import os
from typing import List, Dict, Any

class ParquetChunkDataSource:
    """
    本地文件夹数据源，按 sample 迭代。
    每个样本对应一个 object 文件夹，包含多 view 文件。
    """

    def __init__(self, root_dir: str, file_name: str):
        """
        Args:
            root_dir: 数据集根目录，例如 "/path/to/G-Objaverse"
        """
        self.root_dir = root_dir
        self.samples = self._scan_dataset()  # list[Dict[str, str]]，保存每个 sample 的文件路径
        # self.samples = self.samples[:100]
        # print(f"DEBUG: Temporarily reduced samples to: {len(self.samples)}")        

    def _scan_dataset(self) -> list[dict]:
        data_list = []
        for dir_id in os.listdir(self.root_dir):
            dir_path = os.path.join(self.root_dir, dir_id)
            if not os.path.isdir(dir_path):
                continue
            for object_id in os.listdir(dir_path):
                object_dir = os.path.join(dir_path, object_id)
                if not os.path.isdir(object_dir):
                    continue

                sample = {"__key__": f"{dir_id}/{object_id}", "uid": f"{dir_id}/{object_id}".encode("utf-8")}

                for i in range(40):  # 每个 object 假设 40 views
                    view_dir = os.path.join(object_dir, f"{i:05}")
                    paths = {
                        "png": os.path.join(view_dir, f"{i:05}.png"),
                        "albedo": os.path.join(view_dir, f"{i:05}_albedo.png"),
                        "mr": os.path.join(view_dir, f"{i:05}_mr.png"),
                        "nd": os.path.join(view_dir, f"{i:05}_nd.exr"),
                        "json": os.path.join(view_dir, f"{i:05}.json")
                    }
                    # 只保存存在的文件路径
                    for k, p in paths.items():
                        if os.path.exists(p):
                            sample[f"{i:05d}" + ("_nd.exr" if k == "nd" else f"_{k}.png" if k in ["mr", "albedo"] else ".png" if k == "png" else ".json")] = p

                data_list.append(sample)
        return data_list

    def __iter__(self):
        """
        逐个返回 sample，不按 chunk
        """
        for sample in self.samples:
            yield sample

    def __len__(self):
        return len(self.samples)