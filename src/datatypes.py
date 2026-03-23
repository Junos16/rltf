from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch

@dataclass
class Transition:
    observation: str
    action: str
    action_logprobs: torch.Tensor
    reward: Optional[float] = None

@dataclass
class Trajectory:
    transitions: List[Transition]

    @property
    def total_reward(self) -> float:
        return sum(t.reward for t in self.transitions)
    
    def get_y0(self) -> str:
        return self.transitions[0].action
    
    def get_c0(self) -> str:
        return self.transitions[1].action
    
    def get_y1(self) -> str:
        return self.transitions[2].action

    def get_prompt(self) -> str:
        return self.transitions[0].observation

    def get_action_logprobs(self) -> torch.Tensor:
        return torch.cat([t.action_logprobs for t in self.transitions])

@dataclass
class TrajectoryGroup:
    prompt: str
    trajectories: List[Trajectory]

    @property
    def advantages(self) -> List[float]:
        rewards = [t.total_reward for t in self.trajectories]
        mean = torch.mean(rewards)
        std = torch.std(rewards)
        return [(reward - mean) / (std + 1e-4) for reward in rewards]
