import re
from datasets import load_dataset
from typing import List, Dict, Any

class BaseEnv:
    def __init__(self, dataset_name: str, split: str = "train"):
        self.dataset = load_dataset(dataset_name, split=split)
        self.idx = 0
        self.dataset_size = len(self.dataset)

    def generate_prompts(self, batch_size: int) -> List[Dict[str, Any]]:
        # Randomly sample batch_size prompts from the dataset
        pass
        

    def evaluate(self, answer: str, ground_truth: str) -> float:
        # Parse the LM's generation and check against the ground truth.
        pass

class DummyEnv(BaseEnv):
    def __init__(self, split: str = "train"):
        pass

    def generate_prompts(self, batch_size: int) -> List[Dict[str, Any]]:
        # Randomly sample batch_size prompts from the dataset
        return [{"prompt": "Repeat the word HELLO exactly.", "truth": "HELLO"} for _ in range(batch_size)]

    def evaluate(self, answer: str, ground_truth: str) -> float:
        # Parse the LM's generation and check against the ground truth.
        return 1.0 if ground_truth in answer else 0.0


class GSM8KEnv(BaseEnv):
    def __init__(self, split: str = "train"):
        self.dataset = load_dataset("openai/gsm8k", name="main", split=split)
        self.dataset = self.dataset.shuffle(seed=42)
        self.idx = 0
        self.dataset_size = len(self.dataset)

    def generate_prompts(self, batch_size: int) -> List[Dict[str, Any]]:
        # Randomly sample batch_size prompts from the dataset
        batch = []
        for _ in range(batch_size):
            if self.idx >= self.dataset_size:
                self.idx = 0
                self.dataset = self.dataset.shuffle(seed=42)
            
            item = self.dataset[self.idx]
            self.idx += 1
            
            prompt = item["question"] + " Let's think step by step and output the final answer within \\boxed{}."
            batch.append({"prompt": prompt, "truth": item["answer"]})
            
        return batch

    def evaluate(self, answer: str, ground_truth: str) -> float:
        # Parse the LM's generation and check against the ground truth.
        truth_match = re.search(r"####\s*(.+)", ground_truth)
        model_match = re.search(r"\\boxed\{([^}]*)\}", answer)
        
        if not truth_match or not model_match:
            return 0.0
            
        return 1.0 if model_match.group(1).strip() == truth_match.group(1).strip() else 0.0


class MATH500Env(BaseEnv):
    def __init__(self, split: str = "test"):
        self.dataset = load_dataset("HuggingFaceH4/MATH-500", name="default", split=split)
        self.dataset = self.dataset.shuffle(seed=42)
        self.idx = 0
        self.dataset_size = len(self.dataset)

    def generate_prompts(self, batch_size: int) -> List[Dict[str, Any]]:
        # Randomly sample batch_size prompts from the dataset
        batch = []
        for _ in range(batch_size):
            if self.idx >= self.dataset_size:
                self.idx = 0
                self.dataset = self.dataset.shuffle(seed=42)
            
            item = self.dataset[self.idx]
            self.idx += 1
            
            prompt = item["problem"] + " Let's think step by step and output the final answer within \\boxed{}."
            batch.append({"prompt": prompt, "truth": item["solution"]})
            
        return batch

    def evaluate(self, answer: str, ground_truth: str) -> float:
        # Parse the LM's generation and check against the ground truth.
        truth_match = re.search(r"\\boxed\{([^}]*)\}", ground_truth)
        model_match = re.search(r"\\boxed\{([^}]*)\}", answer)
        
        if not truth_match or not model_match:
            return 0.0
            
        return 1.0 if model_match.group(1).strip() == truth_match.group(1).strip() else 0.0
