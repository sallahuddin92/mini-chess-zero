"""
Trading Agent with Asymmetric TD Learning
==========================================
Risk-aware trading agent using ATD for loss aversion.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Optional, Tuple, Dict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asymmetric_td.losses import AsymmetricTDLoss
from asymmetric_td.utils import RewardCentering, soft_update, clip_gradients


class TradingNetwork(nn.Module):
    """
    Neural network for trading decisions.
    Uses LSTM for temporal patterns + Dense for decision.
    """
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lstm_layers: int = 1
    ):
        super().__init__()
        
        # Feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Decision network
        self.decision_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Initialize with small weights
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_net(x)
        return self.decision_net(features)


class TradingAsymmetricTDLoss(nn.Module):
    """
    Trading-specific Asymmetric TD Loss.
    
    More aggressive asymmetry for trading:
    - Gains: weight = 0.3 (very cautious)
    - Losses: weight = 2.0 (very aggressive correction)
    
    This reflects behavioral economics: losses hurt ~2x more than gains.
    """
    
    def __init__(
        self,
        gain_weight: float = 0.3,
        loss_weight: float = 2.0,
        drawdown_penalty: float = 0.5
    ):
        super().__init__()
        self.gain_weight = gain_weight
        self.loss_weight = loss_weight
        self.drawdown_penalty = drawdown_penalty
        self.max_value = 0.0
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        portfolio_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        td_errors = predicted - target
        
        # Asymmetric weights
        weights = torch.where(
            td_errors > 0,
            self.gain_weight * torch.ones_like(td_errors),
            self.loss_weight * torch.ones_like(td_errors)
        )
        
        # Base loss
        base_loss = torch.nn.functional.smooth_l1_loss(
            predicted, target, reduction='none'
        )
        
        loss = (weights * base_loss).mean()
        
        # Add drawdown penalty if portfolio values provided
        if portfolio_values is not None:
            current_max = portfolio_values.max()
            if current_max > self.max_value:
                self.max_value = current_max.item()
            
            drawdown = (self.max_value - portfolio_values) / (self.max_value + 1e-10)
            drawdown_loss = self.drawdown_penalty * drawdown.mean()
            loss = loss + drawdown_loss
        
        return loss


class TradingAgent:
    """
    Trading agent with Asymmetric TD Learning.
    
    Features:
    - Risk-aware loss function (2x loss aversion)
    - Drawdown penalty
    - Position sizing based on confidence
    - Experience replay with prioritization for losses
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 8,
        gamma: float = 0.99,
        lr: float = 0.0003,
        tau: float = 0.005,
        grad_clip: float = 1.0,
        buffer_size: int = 100000,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,
        device: Optional[str] = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        
        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Networks
        self.policy_net = TradingNetwork(state_dim, action_dim).to(self.device)
        self.target_net = TradingNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Loss (trading-specific ATD)
        self.criterion = TradingAsymmetricTDLoss(
            gain_weight=0.3,
            loss_weight=2.0,
            drawdown_penalty=0.5
        )
        
        # Reward centering
        self.reward_centering = RewardCentering(beta=0.999)
        
        # Replay buffer with priority for losses
        self.memory = deque(maxlen=buffer_size)
        self.loss_memory = deque(maxlen=buffer_size // 4)  # Priority buffer for losses
        
        # Stats
        self.train_step = 0
        self.total_reward = 0.0
    
    def act(self, state: np.ndarray) -> int:
        """Select action with epsilon-greedy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.argmax(dim=1).item()
    
    def train(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        portfolio_value: float = None
    ) -> Tuple[float, float]:
        """Perform one training step."""
        # Center reward
        centered_reward = self.reward_centering(reward)
        
        # Store experience
        experience = (state, action, centered_reward, next_state, done, portfolio_value or 0)
        self.memory.append(experience)
        
        # Prioritize negative experiences
        if reward < 0:
            self.loss_memory.append(experience)
        
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0
        
        # Sample with priority: 30% from loss buffer, 70% regular
        n_loss = min(len(self.loss_memory), self.batch_size // 3)
        n_regular = self.batch_size - n_loss
        
        batch = random.sample(self.memory, n_regular)
        if n_loss > 0 and len(self.loss_memory) >= n_loss:
            batch += random.sample(self.loss_memory, n_loss)
        
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        actions = torch.LongTensor(np.array([e[1] for e in batch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([e[4] for e in batch])).to(self.device)
        portfolio_values = torch.FloatTensor(np.array([e[5] for e in batch])).to(self.device)
        
        # Double DQN
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
        
        target_q = rewards + (self.gamma * next_q * (1 - dones))
        current_q = self.policy_net(states).gather(1, actions).squeeze(1)
        
        # Trading-specific ATD loss
        loss = self.criterion(current_q, target_q, portfolio_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        clip_gradients(self.policy_net, self.grad_clip)
        self.optimizer.step()
        
        # Soft update
        soft_update(self.target_net, self.policy_net, self.tau)
        
        self.train_step += 1
        
        return loss.item(), current_q.mean().item()
    
    def update_epsilon(self):
        """Decay exploration."""
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath: str):
        """Save agent."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.train_step = checkpoint.get('train_step', 0)
