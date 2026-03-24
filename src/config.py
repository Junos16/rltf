import json
from dataclasses import dataclass, asdict
from typing import List, Optional

@dataclass
class Hyperparameters:
    num_iterations: int = 100
    group_size: int = 8
    vllm_batch_size: int = 4
    learning_rate: float = 1e-5
    max_tokens: int = 512
    lora_rank: int = 16
    lora_target_modules: Optional[List[str]] = None
    rl_coef: float = 1.0
    clip_ratio: float = 0.2
    temperature: float = 1.0
    top_p: float = 0.9
    gpu_memory_utilization: float = 0.3

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj"]

    @classmethod
    def from_json(cls, file_path: str):
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(**data)

@dataclass
class ExperimentConfig:
    model_name: str
    judge_model: str
    env: str
    algo: str
    log_dir: str
    hyperparameters: Hyperparameters
    use_correctness_only: bool = False
    sft_coef: float = 0.1

