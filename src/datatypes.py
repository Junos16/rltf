from dataclasses import dataclass, field
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
        return sum(t.reward for t in self.transitions if t.reward is not None)
    
    @property
    def y0_reward(self) -> float:
        return self.transitions[0].reward if self.transitions[0].reward is not None else 0.0
    
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
    y0_rewards: List[float] = field(default_factory=list)

    def get_advantages(self, algo: str = "grpo") -> List[float]:
        # Calculate advantages using algo-dependent baseline
        # RLTF-SD/FM: first-turn baseline to avoid gradient-signal collapse (Section 3.1)
        # GRPO: second-turn baseline (standard multi-turn GRPO)
        y1_rewards = [t.total_reward for t in self.trajectories]
        if algo in ('rltf_sd', 'rltf_fm') and self.y0_rewards:
            baseline = sum(self.y0_rewards) / len(self.y0_rewards)
        else:
            baseline = sum(y1_rewards) / len(y1_rewards)
        return [r - baseline for r in y1_rewards]
