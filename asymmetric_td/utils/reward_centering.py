"""
Reward Centering
================
Reduces variance in TD targets by subtracting running mean.
Reference: "Reward Centering" arXiv 2024
"""

import torch
from typing import Union


class RewardCentering:
    """
    Adaptive reward centering using exponential moving average.
    
    Subtracts the running mean of rewards to reduce variance in TD targets.
    Particularly useful for environments with non-zero mean rewards.
    
    Args:
        beta: Smoothing factor for EMA (default: 0.999)
        initial_mean: Initial estimate of reward mean (default: 0.0)
    
    Example:
        >>> centering = RewardCentering(beta=0.999)
        >>> centered_reward = centering(raw_reward)
    """
    
    def __init__(self, beta: float = 0.999, initial_mean: float = 0.0):
        self.beta = beta
        self.running_mean = initial_mean
        self.count = 0
    
    def __call__(self, reward: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Center a reward by subtracting running mean."""
        return self.center(reward)
    
    def center(self, reward: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """
        Center reward by subtracting running mean.
        
        Args:
            reward: Raw reward value
            
        Returns:
            Centered reward (reward - running_mean)
        """
        # Convert tensor to float for running mean calculation
        if isinstance(reward, torch.Tensor):
            reward_val = reward.item() if reward.numel() == 1 else reward.mean().item()
        else:
            reward_val = reward
        
        # Update running mean
        self.running_mean = self.beta * self.running_mean + (1 - self.beta) * reward_val
        self.count += 1
        
        # Return centered reward
        return reward - self.running_mean
    
    def center_batch(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Center a batch of rewards.
        
        Args:
            rewards: Tensor of rewards [batch_size]
            
        Returns:
            Centered rewards
        """
        # Update running mean with batch mean
        batch_mean = rewards.mean().item()
        self.running_mean = self.beta * self.running_mean + (1 - self.beta) * batch_mean
        self.count += len(rewards)
        
        return rewards - self.running_mean
    
    def reset(self):
        """Reset running mean to zero."""
        self.running_mean = 0.0
        self.count = 0
    
    @property
    def mean(self) -> float:
        """Get current running mean."""
        return self.running_mean
    
    def state_dict(self) -> dict:
        """Get state for saving."""
        return {
            "running_mean": self.running_mean,
            "count": self.count,
            "beta": self.beta
        }
    
    def load_state_dict(self, state: dict):
        """Load state from dict."""
        self.running_mean = state["running_mean"]
        self.count = state["count"]
        self.beta = state.get("beta", self.beta)
