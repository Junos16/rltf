"""
RLTF-SD Trainer (Custom PyTorch Training Loop)
===============================================

This file implements Algorithm 1 from the RLTF-SD paper. We use a plain
PyTorch training loop instead of subclassing TRL's GRPOTrainer because the
paper's self-distillation loss requires computing log pi(y1 | x0), i.e. the
probability of the SECOND turn answer conditioned on the ORIGINAL prompt.
TRL always conditions on the prompt that was used during generation (x1),
so we can't shoehorn this into GRPOTrainer without hacking its internals.

The training loop for every batch does the following:
    For each prompt x0 in the batch:
        1. Generate N first-turn answers: y0 ~ pi(. | x0)
        2. Score them: r0 = reward(y0)
        3. Get judge critiques: c0 = judge(x0, y0)
        4. Build the feedback prompt: x1 = f(x0, y0, c0)
        5. Generate N second-turn answers: y1 ~ pi(. | x1)
        6. Score them: r1 = reward(y1)

    Then compute three loss terms (all AWR-style, no importance weighting):
        - SD loss:      A_sd  * log pi(y1 | x0)   where A_sd  = r1 - mean(r0)
        - RL turn-0:    A_rl0 * log pi(y0 | x0)   where A_rl0 = R  - mean(R), R = r0 + gamma*r1
        - RL turn-1:    A_rl1 * log pi(y1 | x1)   where A_rl1 = r1 - mean(r1)

    Total gradient = sd_gradient + rl_coeff * rl_gradient

    The key insight is the SD loss: by training log pi(y1 | x0) with the
    advantage A_sd, we teach the model to produce the corrected answer
    WITHOUT needing the feedback at test time. This "compiles away" the
    judge into the model's single-turn ability.

How you could have implemented this yourself:
    1. Start with a standard PyTorch training loop (model, optimizer, for loop).
    2. Add the generation step using model.generate() inside torch.no_grad().
    3. Compute log-probabilities by doing a forward pass with the generated
       tokens and extracting logits -> log_softmax -> gather.
    4. Multiply log-probs by advantages (just scalars from reward differences).
    5. That's it — the rest is bookkeeping.
"""

import torch
import torch.nn.functional as F
import asyncio
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig, TaskType

from src.judge import RLTFJudge
from src.reward import parse_answer, get_reward


"""
get_per_token_logps: Given a model, token ids, and an attention mask,
runs a forward pass and returns the log-probability of each token
in the sequence (shifted by 1, since language models predict the next token).
This is the building block for all three loss terms.
"""
def get_per_token_logps(model, input_ids, attention_mask):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    logits = logits[:, :-1, :]
    target_ids = input_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

    mask = attention_mask[:, 1:]
    token_logps = token_logps * mask

    return token_logps, mask


"""
build_turn_2_prompt: Creates the x1 prompt exactly matching the paper's
Appendix D.1 "Policy Prompt". Takes the original question, the model's
first attempt, and the judge critique, and formats them into the string
the model sees for its second attempt.
"""
def build_turn_2_prompt(question, previous_response, critique):
    return f"""Question: {question}

You are given your previous attempt and an expert critique of it below. Your task is to produce an improved solution using the critique.

Your Previous Solution:
{previous_response}

Expert Critique:
{critique}

Instructions:
- Write your answer as a fresh solution to the original problem. Do not refer to your previous attempt.
- Do not mention or refer to the critique or the revision process.
- Use the critique only to improve correctness, clarity, and reasoning.
- Avoid using phrases like "Correctly applying the critique..." or "Reexamining my earlier solution...", etc., as the final answer should stand alone.
Let's think step by step and output the final answer within \\boxed{{}}."""


"""
tokenize_and_pad: Takes a list of text strings and returns left-padded
input_ids and attention_mask tensors on the given device.
Left-padding is required for generation with decoder-only models.
"""
def tokenize_and_pad(tokenizer, texts, device, max_length=None):
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True if max_length else False,
        max_length=max_length,
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


"""
generate_and_decode: Generates completions from the model given
tokenized prompts. Returns the generated text strings and the
full sequence (prompt + completion) token ids.
"""
@torch.no_grad()
def generate_and_decode(model, tokenizer, prompt_ids, prompt_mask, max_new_tokens, temperature):
    outputs = model.generate(
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.pad_token_id,
    )
    completion_ids = outputs[:, prompt_ids.shape[1]:]
    texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    return texts, outputs


"""
compute_rewards: Takes a list of generated text answers and ground truth
strings, parses the boxed answer out of each generation, and returns
a tensor of 0/1 reward values.
"""
def compute_rewards(generated_texts, ground_truths, device):
    rewards = []
    for text, truth in zip(generated_texts, ground_truths):
        parsed = parse_answer(text)
        rewards.append(get_reward(parsed, truth))
    return torch.tensor(rewards, dtype=torch.float32, device=device)


"""
sequence_logprob: Computes the total log-probability of a completion
conditioned on a prompt. We concatenate [prompt, completion], run a
forward pass, and sum the per-token log-probs over just the completion
portion. This is used for the SD loss where we need log pi(y1 | x0).
"""
def sequence_logprob(model, tokenizer, prompt_texts, completion_texts, device, max_length=2048):
    combined_texts = [p + c for p, c in zip(prompt_texts, completion_texts)]

    tokenizer.padding_side = "right"
    combined = tokenizer(combined_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    combined_ids = combined["input_ids"].to(device)
    combined_mask = combined["attention_mask"].to(device)

    tokenizer.padding_side = "left"
    prompt_only = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    prompt_lengths = prompt_only["attention_mask"].sum(dim=1)

    token_logps, mask = get_per_token_logps(model, combined_ids, combined_mask)

    seq_logps = []
    for i in range(len(prompt_texts)):
        start = prompt_lengths[i] - 1
        end = mask[i].sum()
        if start < end:
            seq_logps.append(token_logps[i, start:end].sum())
        else:
            seq_logps.append(torch.tensor(0.0, device=device))

    return torch.stack(seq_logps)


class RLTF_SD_Trainer:
    """
    __init__: Sets up the model (with LoRA adapters), optimizer, judge,
    and all hyperparameters. Uses the paper's defaults:
    group_size=8, gamma=1.0, rl_coeff=0.1, lr=2e-5, KL=0.0.
    """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset,
        max_completion_length: int = 1024,
        group_size: int = 8,
        gamma: float = 1.0,
        rl_coeff: float = 0.1,
        lr: float = 2e-5,
        temperature: float = 0.7,
        lora_rank: int = 32,
    ):
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.max_completion_length = max_completion_length
        self.group_size = group_size
        self.gamma = gamma
        self.rl_coeff = rl_coeff
        self.temperature = temperature

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
        self.model = get_peft_model(model, lora_config)
        self.model.print_trainable_parameters()

        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.judge = RLTFJudge(model_name="gemini-2.5-flash")
        self.device = next(self.model.parameters()).device

    """
    _rollout_one_prompt: The core 2-turn rollout for a single prompt.
    Generates N first-turn answers, gets judge critiques, builds
    the feedback-conditioned prompt, generates N second-turn answers,
    and returns all the pieces needed for the loss calculation.
    """
    def _rollout_one_prompt(self, prompt_text, ground_truth):
        N = self.group_size

        prompt_repeated = [prompt_text] * N
        prompt_ids, prompt_mask = tokenize_and_pad(self.tokenizer, prompt_repeated, self.device)

        y0_texts, _ = generate_and_decode(
            self.model, self.tokenizer, prompt_ids, prompt_mask,
            self.max_completion_length, self.temperature,
        )

        r0 = compute_rewards(y0_texts, [ground_truth] * N, self.device)

        critiques = asyncio.run(
            self.judge.batch_get_critiques(prompt_repeated, y0_texts)
        )

        x1_texts = [
            build_turn_2_prompt(prompt_text, y0, c)
            for y0, c in zip(y0_texts, critiques)
        ]

        x1_ids, x1_mask = tokenize_and_pad(self.tokenizer, x1_texts, self.device)

        y1_texts, _ = generate_and_decode(
            self.model, self.tokenizer, x1_ids, x1_mask,
            self.max_completion_length, self.temperature,
        )

        r1 = compute_rewards(y1_texts, [ground_truth] * N, self.device)

        return {
            "prompt_text": prompt_text,
            "y0_texts": y0_texts,
            "y1_texts": y1_texts,
            "x1_texts": x1_texts,
            "r0": r0,
            "r1": r1,
        }

    """
    _compute_loss: Implements the three loss terms from Algorithm 1.
    Takes the rollout data for one prompt and returns the combined
    scalar loss to backpropagate.
    """
    def _compute_loss(self, rollout):
        prompt_text = rollout["prompt_text"]
        y0_texts = rollout["y0_texts"]
        y1_texts = rollout["y1_texts"]
        x1_texts = rollout["x1_texts"]
        r0 = rollout["r0"]
        r1 = rollout["r1"]
        N = self.group_size

        R = r0 + self.gamma * r1

        b0 = r0.mean()
        bR = R.mean()
        b1 = r1.mean()

        A_sd = r1 - b0
        A_rl0 = R - bR
        A_rl1 = r1 - b1

        prompt_repeated = [prompt_text] * N

        sd_logps = sequence_logprob(
            self.model, self.tokenizer,
            prompt_repeated, y1_texts,
            self.device,
        )
        sd_loss = -(A_sd.detach() * sd_logps).mean()

        rl0_logps = sequence_logprob(
            self.model, self.tokenizer,
            prompt_repeated, y0_texts,
            self.device,
        )
        rl0_loss = -(A_rl0.detach() * rl0_logps).mean()

        rl1_logps = sequence_logprob(
            self.model, self.tokenizer,
            x1_texts, y1_texts,
            self.device,
        )
        rl1_loss = -(A_rl1.detach() * rl1_logps).mean()

        rl_loss = rl0_loss + rl1_loss
        total_loss = sd_loss + self.rl_coeff * rl_loss

        return total_loss, {
            "sd_loss": sd_loss.item(),
            "rl0_loss": rl0_loss.item(),
            "rl1_loss": rl1_loss.item(),
            "mean_r0": r0.mean().item(),
            "mean_r1": r1.mean().item(),
        }

    """
    train: The main training loop. Iterates over the dataset for the
    given number of epochs, does a 2-turn rollout for each prompt,
    computes the combined loss, and takes an optimizer step.
    Logs metrics every log_every steps.
    """
    def train(self, num_epochs=1, log_every=5, save_every=50, save_dir="checkpoints"):
        self.model.train()
        global_step = 0

        for epoch in range(num_epochs):
            for idx in range(len(self.train_dataset)):
                example = self.train_dataset[idx]
                prompt_text = example["prompt"]
                ground_truth = example["answer"]

                rollout = self._rollout_one_prompt(prompt_text, ground_truth)

                loss, metrics = self._compute_loss(rollout)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                global_step += 1

                if global_step % log_every == 0:
                    print(
                        f"[Step {global_step}] loss={loss.item():.4f} "
                        f"sd={metrics['sd_loss']:.4f} "
                        f"rl0={metrics['rl0_loss']:.4f} "
                        f"rl1={metrics['rl1_loss']:.4f} "
                        f"r0={metrics['mean_r0']:.2f} "
                        f"r1={metrics['mean_r1']:.2f}"
                    )

                if save_dir and global_step % save_every == 0:
                    self.model.save_pretrained(f"{save_dir}/step_{global_step}")
                    print(f"Saved checkpoint at step {global_step}")

        return self.model