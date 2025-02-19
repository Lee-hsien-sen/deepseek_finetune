
from datasets import load_dataset, DatasetDict

class DatasetLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_raw_data(self) -> DatasetDict:
        """加载原始数据集"""
        return load_dataset("json", data_files=self.data_path, split="train")

    def split_dataset(
        self,
        dataset: DatasetDict,
        test_size: float = 0.1
    ) -> DatasetDict:
        """数据集划分"""
        return dataset.train_test_split(test_size=test_size)