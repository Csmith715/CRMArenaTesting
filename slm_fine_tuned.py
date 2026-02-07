from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

class LoadFineTunedModel:
    def __init__(self, model_name: str, saved_model_path: str):
        self.model_name = model_name
        self.model_save_location = saved_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True
        )
        self.fine_tuned_model = None

    def load_fine_tune(self):
        base = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=quant_config,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16
        )
        self.fine_tuned_model = PeftModel.from_pretrained(base, self.model_save_location)

    def generate_test_response(self, message_list):
        prompt = self.tokenizer.apply_chat_template(message_list, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.fine_tuned_model.device)

        with torch.no_grad():
            out = self.fine_tuned_model.generate(
                **inputs,
                max_new_tokens=48,
                do_sample=False
            )
        gen_tokens = out[0, inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        return answer
