import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from .config import ExperimentConfig
from .models import PolicyModel, JudgeModel
from .env import DummyEnv, GSM8KEnv, MATH500Env
from .datatypes import Transition, Trajectory, TrajectoryGroup
from .data_processing import compute_advantages, trajectory_to_data, apply_distillation_mask, create_feedback_modeling_target

def clipped_surrogate_loss(logprobs, old_logprobs, advantages, mask, clip_ratio: float) -> torch.Tensor:
    # Calculate clipped surrogate loss
    ratio = torch.exp(logprobs - old_logprobs)
    surrogate1 = ratio * advantages
    surrogate2 = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * advantages
    token_losses = -torch.min(surrogate1, surrogate2)
    loss = (token_losses * mask).sum() / (mask.sum() + 1e-8)
    return loss

def get_env(env_name: str):
    # Get the environment
    if env_name == "dummy":
        return DummyEnv()
    elif env_name == "gsm8k":
        return GSM8KEnv()
    elif env_name == "math500":
        return MATH500Env()
    else:
        raise ValueError(f"Unknown env: {env_name}")

class Trainer:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Initializing Environment...")
        self.env = get_env(self.config.env)
        
        print("Initializing Models...")
        self.policy = PolicyModel(config.model_name, hyperparams=config.hyperparameters)
        
        if config.algo != "rltf_fm":
            self.judge = JudgeModel(config.judge_model, hyperparams=config.hyperparameters)
            
        self.optimizer = AdamW(self.policy.model.parameters(), lr=self.config.hyperparameters.learning_rate)
        
    def train(self):
        # Train the model
        for iteration in range(self.config.hyperparameters.num_iterations):
            print(f"--- Iteration {iteration} ---")
            
            prompt_dicts = self.env.generate_prompts(batch_size=self.config.hyperparameters.vllm_batch_size)
            
            all_trajectory_groups = []
            
            for p_dict in prompt_dicts:
                prompt = p_dict["prompt"]
                truth = p_dict["truth"]
                
                trajectories = []
                y0_rewards = []
                
                for _ in range(self.config.hyperparameters.group_size):
                    y0_list, _ = self.policy.generate([prompt])
                    y0 = y0_list[0]
                    
                    r0 = self.env.evaluate(y0, truth)
                    y0_rewards.append(r0)
                    
                    c0 = self.judge.get_critique(prompt, y0)
                    
                    y1_list, logprobs_list = self.policy.generate([prompt + y0 + c0])
                    y1 = y1_list[0]
                    dummy_logprobs_tensor = torch.zeros(1024, dtype=torch.float32) 
                    
                    reward = self.env.evaluate(y1, truth)
                    
                    t0 = Transition(observation=prompt, action=y0, action_logprobs=torch.tensor([0]), reward=r0)
                    t1 = Transition(observation=prompt+y0, action=c0, action_logprobs=torch.tensor([0]), reward=None)
                    t2 = Transition(observation=prompt+y0+c0, action=y1, action_logprobs=dummy_logprobs_tensor, reward=reward)
                    
                    traj = Trajectory(transitions=[t0, t1, t2])
                    trajectories.append(traj)
                
                group = TrajectoryGroup(prompt=prompt, trajectories=trajectories, y0_rewards=y0_rewards)
                all_trajectory_groups.append(group)
            
            advantages = compute_advantages(all_trajectory_groups)
            
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            self.policy.model.train() 
            
            flat_trajectories = [traj for group in all_trajectory_groups for traj in group.trajectories]
            
            for traj, adv in zip(flat_trajectories, advantages):
                if self.config.algo == 'rltf_sd':
                    data = apply_distillation_mask(traj, adv, self.policy.tokenizer)
                elif self.config.algo == 'rltf_fm':
                    data = create_feedback_modeling_target(traj, self.policy.tokenizer)
                else:
                    data = trajectory_to_data(traj, adv, self.policy.tokenizer)
                
                input_ids = data["input_ids"].unsqueeze(0).to(self.device)
                attention_mask = data["attention_mask"].unsqueeze(0).to(self.device)
                mask = data["loss_mask"].unsqueeze(0).to(self.device)
                old_logprobs = data["old_logprobs"].unsqueeze(0).to(self.device)
                adv_tensor = data["advantages"].unsqueeze(0).to(self.device)
                
                logits = self.policy.forward_train(input_ids, attention_mask)
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                shift_mask = mask[..., 1:].contiguous()
                shift_old_logprobs = old_logprobs[..., 1:shift_mask.shape[1]+1].contiguous() 
                
                logprobs = F.log_softmax(shift_logits, dim=-1)
                action_logprobs = torch.gather(logprobs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
                
                if self.config.algo == 'rltf_fm':
                    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none')
                    loss = loss.view(shift_labels.size())
                    loss = (loss * shift_mask).sum() / (shift_mask.sum() + 1e-8)
                else:
                    loss = clipped_surrogate_loss(action_logprobs, shift_old_logprobs, adv_tensor, shift_mask, self.config.hyperparameters.clip_ratio)
                
                total_loss += loss
            
            self.optimizer.zero_grad()
            batch_loss = total_loss / len(flat_trajectories)
            batch_loss.backward()
            self.optimizer.step()
            
            print(f"Loss: {batch_loss.item():.4f}")

        print("Training complete. Saving LoRA adapter...")
        os.makedirs(self.config.log_dir, exist_ok=True)
        self.policy.model.save_pretrained(self.config.log_dir)
        self.policy.tokenizer.save_pretrained(self.config.log_dir)
        print(f"LoRA adapter saved to {self.config.log_dir}")
        