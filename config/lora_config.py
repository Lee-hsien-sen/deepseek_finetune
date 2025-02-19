
from dataclasses import dataclass
from peft import LoraConfig

@dataclass
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 32
    target_modules: list = None
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def create(self) -> LoraConfig:
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules or ["q_proj", "v_proj"],
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=self.task_type
        )