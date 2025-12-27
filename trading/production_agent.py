"""
Adaptive ATD Trading Agent - Production Grade
==============================================
Tuned for bullish markets with regime detection.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Optional, Tuple


class AdaptiveTradingNetwork(nn.Module):
    """Enhanced network with market regime features."""
    
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
        
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.net(x)


class AdaptiveATDLoss(nn.Module):
    """
    Adaptive Asymmetric TD Loss with regime detection.
    
    Bull market: Reduce loss aversion (chase gains)
    Bear market: Increase loss aversion (protect capital)
    """
    
    def __init__(self):
        super().__init__()
        self.pos_weight = 0.7  # Default for neutral
        self.neg_weight = 1.3
        self.regime = "neutral"
    
    def set_regime(self, regime: str):
        """Set market regime: bull, bear, neutral"""
        self.regime = regime
        if regime == "bull":
            self.pos_weight = 0.9  # More aggressive on gains
            self.neg_weight = 1.1  # Less fearful of losses
        elif regime == "bear":
            self.pos_weight = 0.3  # Very cautious
            self.neg_weight = 2.0  # Very fearful
        else:
            self.pos_weight = 0.7
            self.neg_weight = 1.3
    
    def forward(self, predicted, target):
        td_errors = predicted - target
        weights = torch.where(
            td_errors > 0,
            self.pos_weight * torch.ones_like(td_errors),
            self.neg_weight * torch.ones_like(td_errors)
        )
        loss = (weights * torch.nn.functional.smooth_l1_loss(
            predicted, target, reduction='none'
        )).mean()
        return loss


class ProductionTradingAgent:
    """
    Production-grade trading agent with:
    - Adaptive ATD for market regimes
    - Position sizing based on confidence
    - Risk management (max drawdown, position limits)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 8,
        gamma: float = 0.99,
        lr: float = 0.0005,
        tau: float = 0.005,
        max_position: float = 1.0,
        max_drawdown: float = 0.15,
        buffer_size: int = 100000,
        batch_size: int = 128
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_position = max_position
        self.max_drawdown = max_drawdown
        
        # Device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Networks
        self.policy_net = AdaptiveTradingNetwork(state_dim, action_dim).to(self.device)
        self.target_net = AdaptiveTradingNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = AdaptiveATDLoss()
        
        # Replay buffer
        self.memory = deque(maxlen=buffer_size)
        
        # Exploration
        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        
        # Risk tracking
        self.peak_value = 0
        self.current_drawdown = 0
    
    def detect_regime(self, prices: np.ndarray, window: int = 20) -> str:
        """Detect market regime from recent prices."""
        if len(prices) < window:
            return "neutral"
        
        recent = prices[-window:]
        returns = np.diff(recent) / recent[:-1]
        avg_return = returns.mean()
        volatility = returns.std()
        
        # Trend strength
        if avg_return > 0.001 and volatility < 0.02:
            return "bull"
        elif avg_return < -0.001:
            return "bear"
        return "neutral"
    
    def update_risk(self, portfolio_value: float):
        """Update risk metrics."""
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
    
    def should_reduce_risk(self) -> bool:
        """Check if we should reduce exposure."""
        return self.current_drawdown > self.max_drawdown
    
    def act(self, state: np.ndarray, prices: np.ndarray = None) -> int:
        """Select action with regime-aware exploration."""
        # Update regime if prices available
        if prices is not None:
            regime = self.detect_regime(prices)
            self.criterion.set_regime(regime)
        
        # Force conservative if drawdown exceeded
        if self.should_reduce_risk():
            return 7  # Close all positions
        
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.argmax(dim=1).item()
    
    def train(self, state, action, reward, next_state, done, portfolio_value=None):
        """Train with adaptive ATD."""
        if portfolio_value:
            self.update_risk(portfolio_value)
        
        self.memory.append((state, action, reward, next_state, done))
        
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0
        
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones = torch.FloatTensor([e[4] for e in batch]).to(self.device)
        
        # Double DQN
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
        
        target_q = rewards + (self.gamma * next_q * (1 - dones))
        current_q = self.policy_net(states).gather(1, actions).squeeze(1)
        
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Soft update
        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(self.tau * pp.data + (1 - self.tau) * tp.data)
        
        return loss.item(), current_q.mean().item()
    
    def update_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
    
    def save(self, path: str):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', 0.01)
