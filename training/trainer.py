
from safetensors import torch
from transformers import TrainingArguments, Trainer


class DeepSeekTrainer:
    def __init__(self, model, tokenizer, training_config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = training_config

    def _get_training_args(self) -> TrainingArguments:
        """获取训练参数"""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.per_device_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            fp16=self.config.fp16,
            optim="adamw_torch",
            report_to="none"
        )

    def train(self, train_dataset):
        """执行训练"""
        training_args = self._get_training_args()

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=self._data_collator
        )

        trainer.train()
        return trainer

    def _data_collator(self, data):
        """数据整理函数"""
        return {
            "input_ids": torch.stack([d["input_ids"] for d in data]),
            "attention_mask": torch.stack([d["attention_mask"] for d in data]),
            "labels": torch.stack([d["input_ids"] for d in data])
        }