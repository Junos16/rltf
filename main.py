import argparse
import logging
import os
from src.config import ExperimentConfig, Hyperparameters
from src.trainer import Trainer
from src.inference import InferenceEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="RLTF: Reinforcement Learning from Text Feedback")
    
    parser.add_argument("--action", type=str, choices=["train", "eval"], default="train",
                        help="Action to perform: run training or evaluate a saved model.")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to saved LoRA adapter weights (for eval).")
    parser.add_argument("--prompt", type=str, default=None,
                        help="User input prompt to generate a response for (for eval).")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of environment samples to evaluate (for eval).")
    
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", 
                        help="HuggingFace model to align.")
    parser.add_argument("--judge_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", 
                        help="Model used for generating critiques.")
    parser.add_argument("--env", type=str, choices=["dummy", "gsm8k", "math500"], 
                        default="gsm8k", help="Task environment.")
    parser.add_argument("--algo", type=str, choices=["grpo", "rltf_sd", "rltf_fm"], 
                        default="rltf_sd", help="Alignment algorithm.")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--config_file", type=str, default="hyperparams.json", 
                        help="Path to the JSON hyperparameters configuration file.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"Configuration file not found: {args.config_file}. Please create one with hyperparameters.")
    
    hyperparams = Hyperparameters.from_json(args.config_file)
    
    logger.info(f"Model: {args.model_name} | Judge: {args.judge_model}")
    logger.info(f"Env: {args.env} | Algo: {args.algo} | Config: {args.config_file}")
    
    config = ExperimentConfig(
        model_name=args.model_name,
        judge_model=args.judge_model,
        env=args.env,
        algo=args.algo,
        log_dir=args.log_dir,
        hyperparameters=hyperparams
    )
    
    if args.action == "train":
        trainer = Trainer(config)
        trainer.train()
    elif args.action == "eval":
        engine = InferenceEngine(config, adapter_path=args.adapter_path)
        if args.prompt:
            engine.chat(args.prompt)
        else:
            engine.evaluate_env(num_samples=args.num_samples)

if __name__ == "__main__":
    main()
