from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    model_name: str
    judge_model: str
    env: str
    algo: str
    group_size: int
    vllm_batch_size: int
    learning_rate: float
    lora_rank: int
    rl_coef: float
    log_dir: str
    num_iterations: int = 100
    max_tokens: int = 512
