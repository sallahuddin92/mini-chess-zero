"""
Stable DQN Agent
================
A research-grade DQN implementation with all stabilization techniques.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Optional, Tuple, Dict, Any

from ..losses import AsymmetricTDLoss
from ..utils import RewardCentering, soft_update, clip_gradients


class DQNNetwork(nn.Module):
    """
    Deep Q-Network with optional LayerNorm.
    
    Args:
        input_dim: State dimension
        output_dim: Action dimension
        hidden_dims: Hidden layer sizes (default: [256, 256])
        use_layernorm: Whether to use LayerNorm (default: True)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        use_layernorm: bool = True
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Orthogonal initialization
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StableDQN:
    """
    Stable Deep Q-Network with all stabilization techniques.
    
    Features:
    - Double DQN (reduces overestimation)
    - Asymmetric TD Loss (novel)
    - Gradient clipping
    - Polyak soft target updates
    - Q-value clipping
    - Reward centering
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        gamma: Discount factor (default: 0.95)
        lr: Learning rate (default: 0.0001)
        tau: Soft update coefficient (default: 0.005)
        grad_clip: Gradient clipping norm (default: 10.0)
        q_clip: Q-value clipping range (default: (-10, 10))
        atd_weights: (pos_weight, neg_weight) for ATD (default: (0.5, 1.5))
        buffer_size: Replay buffer size (default: 50000)
        batch_size: Training batch size (default: 64)
        epsilon_start: Initial exploration rate (default: 1.0)
        epsilon_end: Final exploration rate (default: 0.05)
        epsilon_decay: Exploration decay rate (default: 0.995)
        device: Torch device (default: auto-detect)
    
    Example:
        >>> agent = StableDQN(state_dim=25, action_dim=625)
        >>> action = agent.act(state, action_mask=legal_moves)
        >>> loss = agent.train(state, action, reward, next_state, done)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.95,
        lr: float = 0.0001,
        tau: float = 0.005,
        grad_clip: float = 10.0,
        q_clip: Tuple[float, float] = (-10.0, 10.0),
        atd_weights: Tuple[float, float] = (0.5, 1.5),
        buffer_size: int = 50000,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        hidden_dims: Tuple[int, ...] = (256, 256),
        use_layernorm: bool = True,
        use_reward_centering: bool = True,
        device: Optional[str] = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.grad_clip = grad_clip
        self.q_clip_min, self.q_clip_max = q_clip
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
        self.policy_net = DQNNetwork(
            state_dim, action_dim, hidden_dims, use_layernorm
        ).to(self.device)
        
        self.target_net = DQNNetwork(
            state_dim, action_dim, hidden_dims, use_layernorm
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Loss
        self.criterion = AsymmetricTDLoss(
            pos_weight=atd_weights[0],
            neg_weight=atd_weights[1]
        )
        
        # Reward centering
        self.use_reward_centering = use_reward_centering
        self.reward_centering = RewardCentering() if use_reward_centering else None
        
        # Replay buffer
        self.memory = deque(maxlen=buffer_size)
        
        # Training stats
        self.train_step = 0
    
    def act(
        self,
        state: np.ndarray,
        action_mask: Optional[np.ndarray] = None
    ) -> int:
        """
        Select action using epsilon-greedy with optional action masking.
        
        Args:
            state: Current state
            action_mask: Boolean array where True = legal action
            
        Returns:
            Selected action index
        """
        if np.random.rand() <= self.epsilon:
            # Random action
            if action_mask is not None:
                legal_indices = np.where(action_mask)[0]
                return np.random.choice(legal_indices) if len(legal_indices) > 0 else 0
            return random.randrange(self.action_dim)
        
        # Greedy action
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
            
            if action_mask is not None:
                mask_t = torch.BoolTensor(action_mask).to(self.device)
                q_values[0][~mask_t] = float('-inf')
        
        return torch.argmax(q_values[0]).item()
    
    def train(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Tuple[float, float]:
        """
        Perform one training step.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            
        Returns:
            (loss, avg_q_value)
        """
        # Apply reward centering
        if self.reward_centering:
            reward = self.reward_centering(reward)
        
        # Store experience
        self.memory.append((state, action, reward, next_state, done))
        
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0
        
        # Sample minibatch
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        actions = torch.LongTensor(np.array([e[1] for e in batch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([e[4] for e in batch])).to(self.device)
        
        # Double DQN
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            next_q = torch.clamp(next_q, self.q_clip_min, self.q_clip_max)
        
        target_q = rewards + (self.gamma * next_q * (1 - dones))
        target_q = torch.clamp(target_q, self.q_clip_min, self.q_clip_max)
        
        current_q = self.policy_net(states).gather(1, actions).squeeze(1)
        
        # Asymmetric TD Loss
        loss = self.criterion(current_q, target_q)
        
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
        """Decay exploration rate."""
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'reward_mean': self.reward_centering.mean if self.reward_centering else 0.0
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.train_step = checkpoint.get('train_step', 0)
        if self.reward_centering and 'reward_mean' in checkpoint:
            self.reward_centering.running_mean = checkpoint['reward_mean']
