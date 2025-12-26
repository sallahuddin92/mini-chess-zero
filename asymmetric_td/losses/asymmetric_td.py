"""
Asymmetric Temporal Difference Loss
====================================
A novel loss function for stable reinforcement learning.

Biological inspiration: Dopamine neurons show asymmetric response to 
prediction errors (Schultz, 1997). Positive errors (better than expected)
trigger moderate response, while negative errors trigger strong inhibition.

This asymmetry naturally prevents overoptimistic value estimates while
enabling rapid learning from mistakes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AsymmetricTDLoss(nn.Module):
    """
    Asymmetric Temporal Difference Loss.
    
    Weights positive and negative TD errors differently:
    - Positive TD errors (Q > target): overestimation, learn cautiously
    - Negative TD errors (Q < target): underestimation, learn aggressively
    
    Args:
        pos_weight: Weight for positive TD errors (default: 0.5)
        neg_weight: Weight for negative TD errors (default: 1.5)
        reduction: Reduction method ('mean', 'sum', 'none')
        use_huber: Use Huber loss instead of MSE (default: True)
        huber_delta: Delta for Huber loss (default: 1.0)
    
    Example:
        >>> loss_fn = AsymmetricTDLoss(pos_weight=0.5, neg_weight=1.5)
        >>> predicted = torch.tensor([1.0, 2.0, -1.0])
        >>> target = torch.tensor([0.0, 0.0, 0.0])
        >>> loss = loss_fn(predicted, target)
    """
    
    def __init__(
        self,
        pos_weight: float = 0.5,
        neg_weight: float = 1.5,
        reduction: str = 'mean',
        use_huber: bool = True,
        huber_delta: float = 1.0
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.reduction = reduction
        self.use_huber = use_huber
        self.huber_delta = huber_delta
    
    def forward(
        self, 
        predicted: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute asymmetric TD loss.
        
        Args:
            predicted: Predicted Q-values
            target: Target Q-values (should be detached)
            
        Returns:
            Asymmetrically weighted loss
        """
        # Compute TD errors
        td_errors = predicted - target
        
        # Compute asymmetric weights
        weights = torch.where(
            td_errors > 0,
            self.pos_weight * torch.ones_like(td_errors),
            self.neg_weight * torch.ones_like(td_errors)
        )
        
        # Compute base loss
        if self.use_huber:
            base_loss = F.smooth_l1_loss(
                predicted, target, 
                reduction='none',
                beta=self.huber_delta
            )
        else:
            base_loss = F.mse_loss(predicted, target, reduction='none')
        
        # Apply asymmetric weighting
        weighted_loss = weights * base_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss
    
    def extra_repr(self) -> str:
        return f"pos_weight={self.pos_weight}, neg_weight={self.neg_weight}"


class QuantileAsymmetricLoss(nn.Module):
    """
    Quantile regression with asymmetric weighting.
    
    Combines asymmetric TD with distributional RL concepts.
    Useful for risk-sensitive applications (e.g., trading).
    
    Args:
        tau: Quantile to estimate (0.5 = median)
        pos_weight: Additional weight for positive errors
        neg_weight: Additional weight for negative errors
    """
    
    def __init__(
        self,
        tau: float = 0.5,
        pos_weight: float = 0.5,
        neg_weight: float = 1.5
    ):
        super().__init__()
        self.tau = tau
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
    
    def forward(
        self, 
        predicted: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        errors = target - predicted
        
        # Quantile weights
        quantile_weights = torch.where(
            errors > 0,
            self.tau,
            1 - self.tau
        )
        
        # Asymmetric weights
        asym_weights = torch.where(
            errors < 0,  # Underestimation of target
            self.neg_weight,
            self.pos_weight
        )
        
        # Combined loss
        loss = quantile_weights * asym_weights * errors.abs()
        return loss.mean()
