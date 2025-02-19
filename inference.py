from transformers import pipeline


class DeepSeekInference:
    def __init__(self, model_path, tokenizer_path):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.pipeline = None

    def load_model(self):
        """加载推理模型"""
        self.pipeline = pipeline(
            "text-generation",
            model=self.model_path,
            tokenizer=self.tokenizer_path,
            device=0
        )

    def generate(self, instruction: str, input_text: str = "", **kwargs):
        """生成文本"""
        prompt = f"Instruction: {instruction}\n"
        if input_text:
            prompt += f"Input: {input_text}\n"
        prompt += "Output:"

        return self.pipeline(
            prompt,
            max_new_tokens=kwargs.get("max_new_tokens", 256),
            temperature=kwargs.get("temperature", 0.7),
            do_sample=True
        )[0]["generated_text"]