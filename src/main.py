import argparse
import logging
from .config import ExperimentConfig
from .trainer import Trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="RLTF: Reinforcement Learning from Text Feedback")
    
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", 
                        help="HuggingFace model to align.")
    parser.add_argument("--judge_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", 
                        help="Model used for generating critiques.")
    parser.add_argument("--env", type=str, choices=["dummy", "gsm8k", "math500"], 
                        default="gsm8k", help="Task environment.")
    parser.add_argument("--algo", type=str, choices=["grpo", "rltf_sd", "rltf_fm"], 
                        default="rltf_sd", help="Alignment algorithm.")
    parser.add_argument("--group_size", type=int, default=8, 
                        help="Rollouts per prompt for GRPO advantage centering.")
    parser.add_argument("--vllm_batch_size", type=int, default=4, 
                        help="Prompts per iteration.")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--rl_coef", type=float, default=1.0)
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--log_dir", type=str, default="./logs")
    
    args = parser.parse_args()
    
    logger.info(f"Model: {args.model_name} | Judge: {args.judge_model}")
    logger.info(f"Env: {args.env} | Algo: {args.algo} | Iterations: {args.num_iterations}")
    
    config = ExperimentConfig(
        model_name=args.model_name,
        judge_model=args.judge_model,
        env=args.env,
        algo=args.algo,
        group_size=args.group_size,
        vllm_batch_size=args.vllm_batch_size,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        rl_coef=args.rl_coef,
        log_dir=args.log_dir,
        num_iterations=args.num_iterations,
        max_tokens=args.max_tokens,
    )
    
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
