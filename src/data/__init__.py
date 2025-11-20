from typing import *

from src.data.utils.chunk_data_source import ParquetChunkDataSource
from src.data.utils.chunk_dataset import ChunkedDataLoader
from src.data.gobjaverse_parquet_dataset import GObjaverseParquetDataset


# Copied from https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/loader.py
class MultiEpochsChunkedDataLoader(ChunkedDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._DataLoader__initialized = False
        if self.batch_sampler is None:
            self.sampler = _RepeatSampler(self.sampler)
        else:
            self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    # def __len__(self):
    #     return len(self.sampler) if self.batch_sampler is None else len(self.batch_sampler.sampler)

    # def __iter__(self):
    #     for i in range(len(self)):
    #         yield next(self.iterator)
        
    def __iter__(self):
        while True:
            try:
                batch = next(self.iterator)
            except StopIteration:
                # 当 iterator 正常耗尽时，重新初始化它
                self.iterator = super().__iter__()
                batch = next(self.iterator)

            yield batch
        


class _RepeatSampler:
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """
    def __init__(self, sampler):
        self.sampler = sampler

    # def __len__(self):
    #     return len(self.sampler)

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


# def yield_forever(iterator: Iterator[Any]):
#     while True:
#         for x in iterator:
#             yield x

# 修改 src/data/__init__.py 中的 yield_forever 函数

def debug_check_none(data, path="batch"):
    """递归检查数据结构中是否存在 None"""
    if data is None:
        print(f"\n[DEBUG FOUND NONE!] Found NoneType at: {path}")
        return True
    
    if isinstance(data, dict):
        for k, v in data.items():
            if debug_check_none(v, path=f"{path}['{k}']"):
                return True
    elif isinstance(data, (list, tuple)):
        for i, v in enumerate(data):
            if debug_check_none(v, path=f"{path}[{i}]"):
                return True
    # 如果是 Tensor 或其他类型，通常认为安全，除非它是包含 None 的 Object Tensor（罕见）
    return False

def yield_forever(iterator: Iterator[Any]):
    while True:
        for i, x in enumerate(iterator):
            # --- [新增] 调试代码开始 ---
            # 在交给 accelerate/GPU 之前，先检查一遍
            if debug_check_none(x):
                print(f"[DEBUG INFO] The batch (index {i}) contains None! Stopping execution to prevent crash.")
                # 可以在这里打印 keys 帮助定位
                if isinstance(x, dict):
                    print(f"[DEBUG INFO] Top-level keys: {list(x.keys())}")
                raise ValueError("Batch contains None (see log above for location)")
            # --- [新增] 调试代码结束 ---
            
            yield x
