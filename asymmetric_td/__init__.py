"""
Asymmetric TD Learning
======================
A PyTorch library for stable deep reinforcement learning.

Features:
- Asymmetric TD Loss (novel)
- Gradient clipping utilities
- Soft target updates (Polyak averaging)
- Q-value clipping
- Reward centering

Usage:
    from asymmetric_td import StableDQN
    from asymmetric_td.losses import AsymmetricTDLoss
    
    agent = StableDQN(state_dim=25, action_dim=625)
    loss_fn = AsymmetricTDLoss(pos_weight=0.5, neg_weight=1.5)
"""

__version__ = "0.1.0"
__author__ = "Mini-Chess Zero Research Team"

from .agents.stable_dqn import StableDQN
from .losses.asymmetric_td import AsymmetricTDLoss
from .utils.reward_centering import RewardCentering
from .utils.soft_update import soft_update

__all__ = [
    "StableDQN",
    "AsymmetricTDLoss",
    "RewardCentering",
    "soft_update",
]
