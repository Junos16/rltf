"""
RLTF-SD Custom Trainer
----------------------
This file contains our custom Self-Distillation trainer for RLTF-SD.

Why subclass GRPOTrainer?
Implementing stable PPO/GRPO clipping, KL-divergence penalties, and log-probability tracking from scratch in raw PyTorch takes hundreds of lines of complex math. TRL's `GRPOTrainer` has already perfected this.

How does it work?
TRL's `GRPOTrainer` calls a massive internal method named `_generate_and_score_completions` for every training step. It expects that method to generate tokens, compute rewards, and calculate the Advantage (how much better a generation was compared to average). 

We override that exact method to inject our 2-turn logic:
1. We let TRL generate the first turn (`y_0`) exactly as it normally would.
2. We evaluate the reward (`r_0`) using our programmable `reward.py` function.
3. We call the Gemini LLM Judge (`c_0`) asynchronously.
4. We build the Turn 2 prompt using the paper string templates.
5. We let TRL generate the second turn (`y_1`).
6. We evaluate the new reward (`r_1`).
7. We calculate the Self Distillation Advantage: A_SD = r_1 - mean(r_0).
8. Finally, we pack `y_1` and `A_SD` into the exact dictionary shape TRL originally expected and return it.

The rest of the trainer computes the standard Proximal Policy Optimization loss perfectly on Turn 2!
"""

import math
import torch
import asyncio
from trl import GRPOTrainer
from typing import Dict, Any, Union

from src.judge import RLTFJudge
from src.reward import parse_answer, get_reward

class RLTF_SD_Trainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the asynchronous Judge to generate critiques
        self.judge = RLTFJudge(model_name="gemini-2.5-flash")
        
    def _prepare_turn_2_prompts(self, original_prompts: list[str], y_0: list[str], critiques: list[str]) -> list[str]:
        """
        Construct the Turn 2 prompts, x_1, exactly as dictated by the paper's Appendix D.1.
        """
        new_prompts = []
        for prompt, ans, critique in zip(original_prompts, y_0, critiques):
            # This follows the structure defined in the paper for Feedback Conditioning
            x_1 = f"Problem: {prompt}\n\nYour Previous Solution:\n{ans}\n\nExpert Critique:\n{critique}\n\nPlease revise your solution and output the final answer within \\boxed{{}}.\\nRemember to use LaTeX."
            new_prompts.append(x_1)
        return new_prompts

    def _generate_and_score_completions(self, model, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        This is the method TRL calls during `training_step()` before computing loss.
        We intercept it to run our 2-turn pipeline.
        """
        device = self.accelerator.device
        
        # 1. We extract the original x_0 text from the inputs (assuming they are stored in the dataset)
        # Since TRL tokenizes everything, we will decode the prompts back to text for the judge.
        prompts_text = [self.processing_class.decode(ids, skip_special_tokens=True) for ids in inputs["prompt_ids"]]
        
        # In TRL, GRPO expands the batch by num_generations (e.g. 4 or 8) so we get multiple samples for standard GRPO.
        num_generations = self.args.num_generations
        
        # 2. RUN TURN 1 -> Generate y_0
        # For simplicity, we can let the superclass baseline generate_and_score to handle Turn 1.
        # But we actually want to generate Turn 1 and Turn 2. The cleanest approach is to use the unwrapped model directly.
        unwrapped_model = self.accelerator.unwrap_model(model)
        
        # Generate Turn 1
        with torch.no_grad():
            outputs_0 = unwrapped_model.generate(
                inputs["prompt_ids"],
                attention_mask=inputs.get("prompt_mask"),
                max_new_tokens=self.args.max_completion_length,
                do_sample=True,
                temperature=self.args.temperature,
                pad_token_id=self.processing_class.pad_token_id,
            )
            # Slice off the prompt to get just the generated completion
            completion_ids_0 = outputs_0[:, inputs["prompt_ids"].shape[1]:]
            y_0_text = [self.processing_class.decode(ids, skip_special_tokens=True) for ids in completion_ids_0]
            
        # 3. GET TURN 1 REWARDS (r_0)
        # Assuming the ground_truth logic was mapped into the original dataset inputs.
        # If your dataset has "ground_truth", TRL usually passes it via kwargs if defined in the dataset collator.
        ground_truths = inputs.get("ground_truth", [""] * len(prompts_text)) # Fallback if not injected 
        
        r_0 = []
        for ans, truth in zip(y_0_text, ground_truths):
            parsed = parse_answer(ans)
            r_0.append(get_reward(parsed, truth))
        r_0_tensor = torch.tensor(r_0, dtype=torch.float32, device=device)
        
        # 4. GET CRITIQUES FROM JUDGE
        # We run the async batched api call synchronously in this Python thread
        # Because API rate limits and waiting can block the GPU, in a massive training run you'd pre-compute this.
        # For this implementation, we block and wait!
        c_0_text = asyncio.run(self.judge.batch_get_critiques(prompts_text, y_0_text))
        
        # 5. PREPARE TURN 2 PROMPTS
        x_1_text = self._prepare_turn_2_prompts(prompts_text, y_0_text, c_0_text)
        
        # Re-tokenize the new x_1 prompts
        x_1_inputs = self.processing_class(
            x_1_text, 
            return_tensors="pt", 
            padding=True, 
            padding_side="left"
        ).to(device)
        
        # 6. RUN TURN 2 -> Generate y_1 (This is what we actually train on!)
        with torch.no_grad():
            outputs_1 = unwrapped_model.generate(
                x_1_inputs["input_ids"],
                attention_mask=x_1_inputs["attention_mask"],
                max_new_tokens=self.args.max_completion_length,
                do_sample=True,
                temperature=self.args.temperature,
                pad_token_id=self.processing_class.pad_token_id,
            )
            completion_ids_1 = outputs_1[:, x_1_inputs["input_ids"].shape[1]:]
            y_1_text = [self.processing_class.decode(ids, skip_special_tokens=True) for ids in completion_ids_1]
            
            # We must mask pad tokens for the loss so we don't train the model to output pad tokens.
            completion_mask = (completion_ids_1 != self.processing_class.pad_token_id).int()

        # 7. GET TURN 2 REWARDS (r_1)
        r_1 = []
        for ans, truth in zip(y_1_text, ground_truths):
            parsed = parse_answer(ans)
            r_1.append(get_reward(parsed, truth))
        r_1_tensor = torch.tensor(r_1, dtype=torch.float32, device=device)
        
        # 8. SELF DISTILLATION ADVANTAGE
        # A_SD = r_1 - mean(r_0)
        # We group by the original prompts (as num_generations expands the batch)
        # To do this safely, we reshape to (batch_size, num_generations) and calc mean r_0
        batch_size = len(prompts_text) // num_generations
        r_0_grouped = r_0_tensor.view(batch_size, num_generations)
        mean_r_0 = r_0_grouped.mean(dim=1, keepdim=True).repeat_interleave(num_generations, dim=0).flatten()
        
        advantages = r_1_tensor - mean_r_0
        
        # 9. RETURN DICT TO TRL
        # We give TRL the new prompts (x_1), new completions (y_1), and our A_SD advantage!
        # TRL will automatically compute the log_probs of y_1 under x_1, compare to the reference model,
        # apply the advantage, and compute Proximal Policy Optimization loss!
        return {
            "prompt_ids": x_1_inputs["input_ids"],
            "prompt_mask": x_1_inputs["attention_mask"],
            "completion_ids": completion_ids_1,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": inputs.get("num_items_in_batch"),
        }