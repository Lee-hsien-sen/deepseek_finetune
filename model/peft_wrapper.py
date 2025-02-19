
from peft import prepare_model_for_kbit_training, get_peft_model


class PEFTWrapper:
    def __init__(self, model, peft_config):
        self.model = model
        self.peft_config = peft_config

    def apply_peft(self):
        """应用参数高效微调"""
        if self.peft_config.quantization:
            self.model = prepare_model_for_kbit_training(self.model)

        if self.peft_config.use_lora:
            lora_config = self.peft_config.lora_config.create()
            self.model = get_peft_model(self.model, lora_config)

        return self.model