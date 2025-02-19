
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch
from deepseek_finetune.config.base_config import ModelConfig


class ModelLoader:
    def __init__(self, config: ModelConfig):
        self.config = config

    def load_model(self):
        """加载基础模型"""
        quantization_config = self._get_quantization_config()

        return AutoModelForCausalLM.from_pretrained(
self.config.model_name,
            device_map=self.config.device_map,
            quantization_config=quantization_config,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            trust_remote_code=True
        )

    def _get_quantization_config(self) -> BitsAndBytesConfig:
        """获取量化配置"""
        if not self.config.quantization:
            return None

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )