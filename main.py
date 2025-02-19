from config.base_config import ModelConfig, TrainingConfig
from config.lora_config import LoRAConfig
from data.dataset_loader import DatasetLoader
from data.data_preprocessor import DataPreprocessor
from deepseek_finetune.inference import DeepSeekInference
from model.model_loader import ModelLoader
from model.peft_wrapper import PEFTWrapper
from training.trainer import DeepSeekTrainer
from transformers import AutoTokenizer



def doMain():
    # 初始化配置
    model_config = ModelConfig(quantization=True)
    training_config = TrainingConfig()
    lora_config = LoRAConfig()

    # 加载模型
    model_loader = ModelLoader(model_config)
    model = model_loader.load_model()
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)

    # 应用PEFT
    peft_wrapper = PEFTWrapper(model, lora_config)
    model = peft_wrapper.apply_peft()

    # 加载数据
    data_loader = DatasetLoader("data.json")
    raw_data = data_loader.load_raw_data()

    # 预处理数据
    preprocessor = DataPreprocessor(tokenizer)
    formatted_data = raw_data.map(preprocessor.format_text)
    tokenized_data = preprocessor.tokenize(formatted_data, training_config.max_seq_length)

    # 开始训练
    trainer = DeepSeekTrainer(model, tokenizer, training_config)
    trainer.train(tokenized_data)

    # 保存模型
    model.save_pretrained("fine-tuned-model")
    tokenizer.save_pretrained("fine-tuned-model")

    # 推理示例
    inference = DeepSeekInference("fine-tuned-model", "fine-tuned-model")
    inference.load_model()
    print(inference.generate("解释深度学习的基本原理"))


if __name__ == '__main__':
    doMain()