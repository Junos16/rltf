"""
Checkpointing Utilities
-----------------------
This file provides helper functions to save and load PyTorch model weights and optimizer states.

Why subclassing or custom scripts?
When training Large Language Models (LLMs), training jobs frequently crash due to Out-Of-Memory (OOM) errors, API rate limits (like our Judge), or cluster timeouts. 
If a 20-hour run crashes at hour 19, you lose everything unless you frequently "checkpoint" (save) the exact state of the model's weights and the optimizer's momentum to disk.

While HuggingFace's Trainer automatically saves checkpoints during `trainer.train()`, writing a custom checkpointing utility gives you full control. You can use these functions to:
1. Manually save the model before a risky operation.
2. Load a specific historical checkpoint to resume a crashed run precisely where it left off.
3. Save only the adapter weights (if using LoRA/PEFT) to save disk space.
"""

import os
import torch
from transformers import PreTrainedModel
from torch.optim import Optimizer

def save_checkpoint(model: PreTrainedModel, optimizer: Optimizer, epoch: int, step: int, save_dir: str = "checkpoints"):
    """
    Saves the model state dict and optimizer state dict to a specific directory.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a unique filename for this checkpoint
    checkpoint_name = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    
    # Pack the states into a dictionary
    checkpoint_dict = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    # Save to disk
    torch.save(checkpoint_dict, checkpoint_path)
    print(f"✅ Successfully saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(model: PreTrainedModel, optimizer: Optimizer, checkpoint_path: str):
    """
    Loads model weights and optimizer momentum from a saved .pt file,
    mutating the provided model and optimizer in-place.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    # Load the dictionary from disk (map_location='cpu' prevents VRAM spikes during loading)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Restore the states
    model.load_state_dict(checkpoint_dict['model_state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
    
    epoch = checkpoint_dict.get('epoch', 0)
    step = checkpoint_dict.get('step', 0)
    
    print(f"🔄 Successfully restored model and optimizer from {checkpoint_path} (Epoch {epoch}, Step {step})")
    
    return epoch, step
