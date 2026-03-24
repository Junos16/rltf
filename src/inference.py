import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from .config import ExperimentConfig
from .env import DummyEnv, GSM8KEnv, MATH500Env

class InferenceEngine:
    def __init__(self, config: ExperimentConfig, adapter_path: str | None = None):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name, 
            quantization_config=quant_config,
            device_map="cuda"
        )
        
        print(f"Loaded Model: {self.config.model_name}")

        if adapter_path:
            if os.path.exists(adapter_path):
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
                print(f"Loaded adapter: {adapter_path}")
            else:
                print("No adapter found in path")
        else:
            print("Running infernce on base unaligned model")

    def _generate(self, prompts: list):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=self.config.hyperparameters.max_tokens,
                temperature=self.config.hyperparameters.temperature,
                top_p=self.config.hyperparameters.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True
            )
        
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_length:]
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
    def chat(self, prompt: str):
        print(f"User Prompt: {prompt}")
        responses = self._generate([prompt])
        print(f"Response: {responses[0]}")
        return responses[0]
        
    def evaluate_env(self, num_samples: int = 10):
        print(f"Env: {self.config.env}")
        if self.config.env == "dummy":
            env = DummyEnv(split="test")
        elif self.config.env == "gsm8k":
            env = GSM8KEnv(split="test")
        elif self.config.env == "math500":
            env = MATH500Env(split="test")
        else:
            raise ValueError(f"Unknown env: {self.config.env}")
            
        print(f"Evaluating {num_samples} prompts...")
        prompt_dicts = env.generate_prompts(batch_size=num_samples)
        
        batch_size = 2
        total_reward = 0
        valid_samples = 0
        
        for i in range(0, len(prompt_dicts), batch_size):
            batch = prompt_dicts[i:i+batch_size]
            prompts = [p["prompt"] for p in batch]
            truths = [p["truth"] for p in batch]
            
            responses = self._generate(prompts)
            
            for prompt, y, truth in zip(prompts, responses, truths):
                r = env.evaluate(y, truth)
                print(f"\n{'='*40}\n[Prompt]\n{prompt}\n[Response]\n{y}\n[Reward]: {r}")
                
                if r is not None:
                    total_reward += r
                    valid_samples += 1
                
        avg_reward = total_reward / valid_samples if valid_samples > 0 else 0
        print(f"Average Reward over {valid_samples} samples: {avg_reward:.4f}")
        return avg_reward
