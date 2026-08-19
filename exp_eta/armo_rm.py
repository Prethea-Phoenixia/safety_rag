import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig


class ArmoRMPipeline:
    def __init__(self, model_id, device_map="cuda:0", truncation=True, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            low_cpu_mem_usage=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages: list[dict[str, str]]) -> dict[str, float]:
        """
        messages: OpenAI chat messages to be scored
        Note: no batching since due to length differences, the model will have to pad to the max length which is not efficient
        Returns: a dictionary with the score between 0 and 1
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", padding=True, truncation=self.truncation, max_length=self.max_length
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            score = output.score.float().item()
        return {"score": score}
