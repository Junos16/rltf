import os
import tempfile
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from typing import List, Tuple
from .config import Hyperparameters
from .utils import log_system_usage

class PolicyModel:
    def __init__(self, model_name: str, hyperparams: Hyperparameters):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            quantization_config=quant_config, 
            device_map="cuda"
        )
        self.peft_config = LoraConfig(r=hyperparams.lora_rank, target_modules=hyperparams.lora_target_modules)
        self.model = get_peft_model(self.model, self.peft_config)
        self.model.config.use_cache = False
        self.model.to(dtype=torch.bfloat16)
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            enable_lora=True,
            max_lora_rank=hyperparams.lora_rank,
            dtype="bfloat16",
            max_model_len=hyperparams.max_tokens * 2,
            gpu_memory_utilization=hyperparams.gpu_memory_utilization,
            enforce_eager=True,
        )
        self.sampling_params = SamplingParams(
            temperature=hyperparams.temperature,
            top_p=hyperparams.top_p,
            max_tokens=hyperparams.max_tokens,
            ignore_eos=True,
            logprobs=1,
        )
        self._lora_dir = tempfile.mkdtemp()
        self._lora_request_id = 0

    def sync_weights(self):
        self.model.save_pretrained(self._lora_dir)
        self.tokenizer.save_pretrained(self._lora_dir)
        self._lora_request_id += 1

    def generate(self, prompts: List[str], max_tokens: int = 512) -> Tuple[List[str], torch.Tensor]:
        lora_request = LoRARequest(
            lora_name=f"adapter_{self._lora_request_id}",
            lora_int_id=self._lora_request_id,
            lora_path=self._lora_dir,
        )
        outputs = self.llm.generate(prompts, self.sampling_params, lora_request=lora_request)
        generated_texts = [output.outputs[0].text for output in outputs]
        logprobs = [output.outputs[0].logprobs for output in outputs]
        return generated_texts, logprobs
        
    def forward_train(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.logits

class JudgeModel:
    def __init__(self, judge_model_name: str, hyperparams: Hyperparameters):
        self.llm = LLM(
            model=judge_model_name, 
            tensor_parallel_size=1, 
            dtype="bfloat16", 
            max_model_len=hyperparams.max_tokens * 2, 
            gpu_memory_utilization=hyperparams.gpu_memory_utilization,
            enforce_eager=True,
        )
        self.sampling_params = SamplingParams(
            temperature=hyperparams.temperature,
            top_p=hyperparams.top_p,
            max_tokens=hyperparams.max_tokens,
            ignore_eos=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(judge_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token        

    def get_critique(self, prompt: str, initial_answer: str, r0: float = None, correctness_only: bool = False) -> str:
        if correctness_only and r0 is not None:
            return "Your previous answer was correct." if r0 > 0.0 else "Your previous answer was incorrect."
            
        query = f"""You are given a question and your previous attempt below. Your task is provide a critique of the attempt.\n\n[Prompt]\n{prompt}\n[Attempt]\n{initial_answer}\nCritique: """
        outputs = self.llm.generate([query], self.sampling_params)
        critique = outputs[0].outputs[0].text
        return critique
