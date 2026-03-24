import torch
from typing import List, Dict, Any

def compute_advantages(trajectory_groups, algo: str = "grpo") -> List[torch.Tensor]:
    # Calculate advantages of the trajectories
    advantages = []
    for group in trajectory_groups:
        advantages.extend(group.get_advantages(algo))
    return advantages

def trajectory_to_data(trajectory, advantage: float, tokenizer, max_length: int = 1024, include_y0: bool = False) -> Dict[str, torch.Tensor]:
    # Convert trajectory to data for training

    prompt_text = trajectory.get_prompt()
    y0_text = trajectory.get_y0()
    c0_text = trajectory.get_c0()
    
    context_text = prompt_text + y0_text + c0_text
    y1_text = trajectory.get_y1()
    full_text = context_text + y1_text
    
    encoded = tokenizer(full_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    
    input_ids = encoded["input_ids"].squeeze(0)
    attention_mask = encoded["attention_mask"].squeeze(0)
    loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    
    sequence_length = attention_mask.sum().item()
    
    if include_y0:
        # Multi-turn GRPO: train on both y0 and y1 (mask only the prompt)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_length = len(prompt_ids)
        if sequence_length > prompt_length:
            loss_mask[prompt_length:sequence_length] = 1.0
    else:
        # Default: train only on y1 (mask prompt+y0+c0)
        context_ids = tokenizer(context_text, add_special_tokens=False)["input_ids"]
        context_length = len(context_ids)
        if sequence_length > context_length:
            loss_mask[context_length:sequence_length] = 1.0
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "advantages": torch.tensor([advantage], dtype=torch.float32),
        "old_logprobs": trajectory.get_action_logprobs()
    }

def apply_distillation_mask(trajectory, advantage: float, tokenizer, max_length: int = 1024) -> Dict[str, torch.Tensor]:
    # Apply distillation mask to the trajectory

    context_text = trajectory.get_prompt()
    y1_text = trajectory.get_y1()
    full_text = context_text + y1_text
    
    encoded = tokenizer(full_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    input_ids = encoded["input_ids"].squeeze(0)
    attention_mask = encoded["attention_mask"].squeeze(0)
    
    loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    context_ids = tokenizer(context_text, add_special_tokens=False)["input_ids"]
    context_length = len(context_ids)
    sequence_length = attention_mask.sum().item()
    
    if sequence_length > context_length:
        loss_mask[context_length:sequence_length] = 1.0
        
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "advantages": torch.tensor([advantage], dtype=torch.float32),
        "old_logprobs": trajectory.get_action_logprobs()
    }

def create_feedback_modeling_target(trajectory, tokenizer, max_length: int = 1024) -> Dict[str, torch.Tensor]:
    # Create feedback modeling target for the trajectory
    
    prompt_text = trajectory.get_prompt()
    y0_text = trajectory.get_y0()
    context_text = prompt_text + y0_text
    c0_text = trajectory.get_c0()
    
    full_text = context_text + c0_text
    
    encoded = tokenizer(full_text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    input_ids = encoded["input_ids"].squeeze(0)
    attention_mask = encoded["attention_mask"].squeeze(0)
    
    loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    context_ids = tokenizer(context_text, add_special_tokens=False)["input_ids"]
    context_length = len(context_ids)
    sequence_length = attention_mask.sum().item()
    
    if sequence_length > context_length:
        loss_mask[context_length:sequence_length] = 1.0
        
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "advantages": torch.tensor([0.0], dtype=torch.float32), 
        "old_logprobs": torch.zeros_like(input_ids, dtype=torch.float32)
    }
