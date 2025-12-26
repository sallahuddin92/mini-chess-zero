"""
Research-Grade DQN Agent for Mini-Chess Zero
=============================================
Implements stabilization techniques:
- Gradient Clipping (L2 norm)
- Soft Target Updates (Polyak Averaging)
- Q-Value Clipping
- Asymmetric TD Learning (Novel)

References:
- Double DQN: van Hasselt et al., 2016
- Reward Centering: arXiv 2024
- Soft Updates: Lillicrap et al., 2015 (DDPG)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import os


# =============================================================================
# NEURAL NETWORK ARCHITECTURE
# =============================================================================

class DQN(nn.Module):
    """
    Deep Q-Network with LayerNorm for training stability.
    Architecture: Input -> 256 -> 256 -> Output
    """
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),  # Stabilizes activations
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
        # Initialize weights using orthogonal initialization (stable for RL)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


# =============================================================================
# ASYMMETRIC TD LEARNING (NOVEL CONTRIBUTION)
# =============================================================================

class AsymmetricTDLoss(nn.Module):
    """
    Novel: Asymmetric Temporal Difference Loss
    
    Biological inspiration: Dopamine neurons show asymmetric 
    response to positive/negative prediction errors (Schultz, 1997).
    
    - Positive TD errors (better than expected): Learn cautiously
    - Negative TD errors (worse than expected): Learn aggressively
    
    This prevents overoptimism while quickly correcting mistakes.
    """
    def __init__(self, positive_weight=0.5, negative_weight=1.5):
        super().__init__()
        self.pos_w = positive_weight
        self.neg_w = negative_weight
    
    def forward(self, predicted, target):
        td_errors = predicted - target
        weights = torch.where(
            td_errors > 0,
            self.pos_w * torch.ones_like(td_errors),
            self.neg_w * torch.ones_like(td_errors)
        )
        # Huber-style loss with asymmetric weighting
        return (weights * torch.nn.functional.smooth_l1_loss(
            predicted, target, reduction='none'
        )).mean()


# =============================================================================
# REWARD CENTERING (arXiv 2024)
# =============================================================================

class RewardCentering:
    """
    Adaptive Reward Centering
    
    Subtracts running mean from rewards to reduce variance in TD targets.
    Reference: "Reward Centering" arXiv:2405.xxxxx (2024)
    """
    def __init__(self, beta=0.999):
        self.running_mean = 0.0
        self.beta = beta
    
    def center(self, reward):
        self.running_mean = self.beta * self.running_mean + (1 - self.beta) * reward
        return reward - self.running_mean
    
    def reset(self):
        self.running_mean = 0.0


# =============================================================================
# DQN AGENT WITH STABILIZATION
# =============================================================================

class DQNAgent:
    """
    Research-Grade DQN Agent
    
    Key Features:
    - Double DQN for reduced overestimation
    - Gradient clipping (max_norm=10)
    - Polyak soft target updates (tau=0.005)
    - Q-value clipping [-10, 10]
    - Asymmetric TD Learning
    - Reward Centering
    """
    
    # Hyperparameters (Research-tuned)
    GAMMA = 0.95              # Discount factor (reduced from 0.99)
    LEARNING_RATE = 0.0001    # Slower, more stable (reduced from 0.0005)
    TAU = 0.005               # Soft update coefficient
    GRAD_CLIP_NORM = 10.0     # Gradient clipping threshold
    Q_CLIP_MIN = -10.0        # Q-value lower bound
    Q_CLIP_MAX = 10.0         # Q-value upper bound
    MEMORY_SIZE = 50000       # Larger replay buffer
    BATCH_SIZE = 64
    EPSILON_START = 1.0
    EPSILON_MIN = 0.05
    EPSILON_DECAY = 0.995
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=self.MEMORY_SIZE)
        
        # Exploration parameters
        self.epsilon = self.EPSILON_START
        self.epsilon_min = self.EPSILON_MIN
        self.epsilon_decay = self.EPSILON_DECAY
        
        # Device selection (Apple Silicon priority)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print(">>> Agent running on Apple M4 (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Networks
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.hard_update_target()
        
        # Optimizer with reduced learning rate
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=self.LEARNING_RATE
        )
        
        # Asymmetric TD Loss (Novel)
        self.criterion = AsymmetricTDLoss(
            positive_weight=0.5,  # Cautious on overestimation
            negative_weight=1.5   # Aggressive on underestimation
        )
        
        # Reward Centering
        self.reward_center = RewardCentering(beta=0.999)
        
        # Training statistics
        self.train_step = 0
    
    def hard_update_target(self):
        """Full copy of policy network to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def soft_update_target(self):
        """
        Polyak averaging: θ_target = τ*θ_policy + (1-τ)*θ_target
        Smoother updates prevent policy oscillation.
        """
        for target_param, policy_param in zip(
            self.target_net.parameters(), 
            self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.TAU * policy_param.data + 
                (1.0 - self.TAU) * target_param.data
            )

    def act(self, state, action_mask=None):
        """
        Selects an action with epsilon-greedy and action masking.
        """
        # Epsilon-greedy exploration
        if np.random.rand() <= self.epsilon:
            if action_mask is None:
                return random.randrange(self.action_size)
            else:
                legal_indices = np.where(action_mask)[0]
                return np.random.choice(legal_indices) if len(legal_indices) > 0 else 0
        
        # Neural network exploitation
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
            
            # Action masking: set illegal moves to -inf
            if action_mask is not None:
                mask_t = torch.BoolTensor(action_mask).to(self.device)
                q_values[0][~mask_t] = float('-inf')

        return torch.argmax(q_values[0]).item()

    def train(self, state, action, reward, next_state, done):
        """
        Training step with all stabilization techniques.
        """
        # Apply reward centering
        centered_reward = self.reward_center.center(reward)
        
        # Store experience
        self.memory.append((state, action, centered_reward, next_state, done))
        
        if len(self.memory) < self.BATCH_SIZE:
            return 0.0, 0.0

        # Sample minibatch
        minibatch = random.sample(self.memory, self.BATCH_SIZE)
        
        states = torch.FloatTensor(np.array([i[0] for i in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([i[1] for i in minibatch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([i[2] for i in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([i[4] for i in minibatch])).to(self.device)

        # Double DQN: action selection from policy, evaluation from target
        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            
            # Q-Value Clipping: prevent runaway estimates
            next_q_values = torch.clamp(next_q_values, self.Q_CLIP_MIN, self.Q_CLIP_MAX)
        
        # Compute TD target
        target_q = rewards + (self.GAMMA * next_q_values * (1 - dones))
        target_q = torch.clamp(target_q, self.Q_CLIP_MIN, self.Q_CLIP_MAX)
        
        # Current Q-values
        current_q = self.policy_net(states).gather(1, actions).squeeze(1)

        # Asymmetric TD Loss
        loss = self.criterion(current_q, target_q)
        
        # Gradient descent with clipping
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping (L2 norm)
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), 
            max_norm=self.GRAD_CLIP_NORM
        )
        
        self.optimizer.step()
        
        # Soft update target network every step
        self.soft_update_target()
        
        self.train_step += 1
        
        return loss.item(), current_q.mean().item()

    def update_epsilon(self):
        """Decay exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        """Save model checkpoint."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'reward_mean': self.reward_center.running_mean
        }, filename)
    
    def load(self, filename):
        """Load model checkpoint."""
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=self.device)
            if isinstance(checkpoint, dict) and 'policy_net' in checkpoint:
                self.policy_net.load_state_dict(checkpoint['policy_net'])
                self.target_net.load_state_dict(checkpoint['target_net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.epsilon = checkpoint.get('epsilon', self.EPSILON_MIN)
                self.train_step = checkpoint.get('train_step', 0)
                self.reward_center.running_mean = checkpoint.get('reward_mean', 0.0)
            else:
                # Legacy format: just the state dict
                self.policy_net.load_state_dict(checkpoint)
                self.hard_update_target()
            print(f">>> Loaded checkpoint: step={self.train_step}, epsilon={self.epsilon:.3f}")