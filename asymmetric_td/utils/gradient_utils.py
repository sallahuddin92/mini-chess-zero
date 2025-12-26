"""
Gradient Utilities
==================
Gradient clipping and monitoring tools.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict


def clip_gradients(
    model: nn.Module,
    max_norm: float = 10.0,
    norm_type: float = 2.0
) -> float:
    """
    Clip gradients by global L2 norm.
    
    Args:
        model: Neural network with gradients
        max_norm: Maximum allowed gradient norm
        norm_type: Type of norm (default: 2.0 for L2)
        
    Returns:
        Total gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm,
        norm_type=norm_type
    ).item()


def clip_gradients_value(
    model: nn.Module,
    clip_value: float = 1.0
) -> None:
    """
    Clip gradients by value (element-wise).
    
    Args:
        model: Neural network with gradients
        clip_value: Maximum absolute value for any gradient element
    """
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)


def get_gradient_stats(model: nn.Module) -> Dict[str, float]:
    """
    Get gradient statistics for monitoring.
    
    Args:
        model: Neural network with gradients
        
    Returns:
        Dictionary with gradient statistics
    """
    total_norm = 0.0
    max_grad = 0.0
    min_grad = float('inf')
    num_params = 0
    
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            
            param_max = param.grad.data.abs().max().item()
            param_min = param.grad.data.abs().min().item()
            
            max_grad = max(max_grad, param_max)
            min_grad = min(min_grad, param_min)
            num_params += param.numel()
    
    total_norm = total_norm ** 0.5
    
    return {
        "total_norm": total_norm,
        "max": max_grad,
        "min": min_grad if min_grad != float('inf') else 0.0,
        "num_params": num_params
    }


class GradientMonitor:
    """
    Monitors gradient health during training.
    
    Tracks gradient statistics and detects anomalies
    (explosion, vanishing, NaN).
    """
    
    def __init__(
        self,
        explosion_threshold: float = 1000.0,
        vanishing_threshold: float = 1e-7
    ):
        self.explosion_threshold = explosion_threshold
        self.vanishing_threshold = vanishing_threshold
        self.history = []
    
    def check(self, model: nn.Module) -> Dict[str, any]:
        """
        Check gradient health.
        
        Returns:
            Dictionary with health status and stats
        """
        stats = get_gradient_stats(model)
        
        # Check for NaN
        has_nan = False
        for param in model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                has_nan = True
                break
        
        status = "healthy"
        if has_nan:
            status = "nan"
        elif stats["total_norm"] > self.explosion_threshold:
            status = "exploding"
        elif stats["total_norm"] < self.vanishing_threshold:
            status = "vanishing"
        
        result = {
            "status": status,
            **stats
        }
        
        self.history.append(result)
        return result
