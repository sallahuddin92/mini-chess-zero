"""
Soft Update (Polyak Averaging)
==============================
Smooth target network updates for stable training.
Reference: Lillicrap et al., 2015 (DDPG)
"""

import torch.nn as nn
from typing import Iterator, Tuple


def soft_update(
    target_net: nn.Module,
    source_net: nn.Module,
    tau: float = 0.005
) -> None:
    """
    Perform soft update of target network parameters.
    
    θ_target = τ * θ_source + (1 - τ) * θ_target
    
    Args:
        target_net: Target network to update
        source_net: Source network (usually policy network)
        tau: Interpolation factor (0 = no update, 1 = hard copy)
    
    Example:
        >>> soft_update(target_net, policy_net, tau=0.005)
    """
    for target_param, source_param in zip(
        target_net.parameters(), 
        source_net.parameters()
    ):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )


def hard_update(target_net: nn.Module, source_net: nn.Module) -> None:
    """
    Perform hard update (full copy) of target network.
    
    Args:
        target_net: Target network to update
        source_net: Source network
    """
    target_net.load_state_dict(source_net.state_dict())


class PolyakAverager:
    """
    Manages soft target updates with automatic scheduling.
    
    Args:
        target_net: Target network
        source_net: Source network  
        tau: Soft update coefficient
        update_every: Only update every N calls (default: 1)
    """
    
    def __init__(
        self,
        target_net: nn.Module,
        source_net: nn.Module,
        tau: float = 0.005,
        update_every: int = 1
    ):
        self.target_net = target_net
        self.source_net = source_net
        self.tau = tau
        self.update_every = update_every
        self.call_count = 0
    
    def update(self) -> bool:
        """
        Perform update if schedule allows.
        
        Returns:
            True if update was performed
        """
        self.call_count += 1
        
        if self.call_count % self.update_every == 0:
            soft_update(self.target_net, self.source_net, self.tau)
            return True
        return False
    
    def force_update(self):
        """Force immediate hard update."""
        hard_update(self.target_net, self.source_net)
