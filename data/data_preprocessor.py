from datasets import DatasetDict
from transformers import AutoTokenizer

class DataPreprocessor:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def format_text(self, example: dict) -> dict:
        """格式化文本"""
        text = f"Instruction: {example['instruction']}\n"
        if example.get("input"):
            text += f"Input: {example['input']}\n"
        text += f"Output: {example['output']}"
        return {"text": text}

    def tokenize(
        self,
        dataset: DatasetDict,
        max_length: int
    ) -> DatasetDict:
        """分词处理"""
        return dataset.map(
            lambda x: self.tokenizer(
                x["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            ),
            batched=True,
            remove_columns=["text"]
        )