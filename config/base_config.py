
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    output_dir: str = "./output"
    learning_rate: float = 2e-5
    per_device_batch_size: int = 2
    num_train_epochs: int = 3
    max_seq_length: int = 1024
    gradient_accumulation_steps: int = 4
    logging_steps: int = 50
    save_steps: int = 500
    fp16: bool = True

@dataclass
class ModelConfig:
    model_name: str = "deepseek-ai/deepseek-r1"
    device_map: str = "auto"
    quantization: bool = False
    torch_dtype: str = "bfloat16"