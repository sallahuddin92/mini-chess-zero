"""
Ablation Study Framework for Asymmetric TD Learning
====================================================
Runs controlled experiments to validate each component's contribution.

Experiments:
1. FULL: All stabilization (baseline - what we already have)
2. NO_ATD: Remove Asymmetric TD, use standard Huber loss
3. NO_GRAD_CLIP: Remove gradient clipping
4. NO_SOFT_UPDATE: Use hard updates instead of Polyak
5. NO_Q_CLIP: Remove Q-value clipping
6. VANILLA: No stabilization at all (standard DDQN)
"""

import sys
import os
import csv
import copy
import random
import numpy as np
from collections import deque
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.environment import MiniChessEnv


# =============================================================================
# CONFIGURABLE AGENT FOR ABLATION
# =============================================================================

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, use_layernorm=True):
        super().__init__()
        if use_layernorm:
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, output_dim)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, output_dim)
            )
        
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


class AblationAgent:
    """
    Configurable agent for ablation studies.
    Each flag enables/disables a specific stabilization technique.
    """
    
    def __init__(
        self,
        state_size,
        action_size,
        # Ablation flags
        use_asymmetric_td=True,
        use_gradient_clipping=True,
        use_soft_update=True,
        use_q_clipping=True,
        use_reward_centering=True,
        use_layernorm=True,
        # Hyperparameters
        gamma=0.95,
        lr=0.0001,
        tau=0.005,
        grad_clip_norm=10.0,
        q_clip_range=(-10.0, 10.0),
        atd_weights=(0.5, 1.5),
    ):
        self.state_size = state_size
        self.action_size = action_size
        
        # Ablation flags
        self.use_asymmetric_td = use_asymmetric_td
        self.use_gradient_clipping = use_gradient_clipping
        self.use_soft_update = use_soft_update
        self.use_q_clipping = use_q_clipping
        self.use_reward_centering = use_reward_centering
        
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.grad_clip_norm = grad_clip_norm
        self.q_clip_min, self.q_clip_max = q_clip_range
        self.atd_pos_w, self.atd_neg_w = atd_weights
        
        # Training state
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.memory = deque(maxlen=50000)
        
        # Reward centering
        self.reward_mean = 0.0
        self.reward_beta = 0.999
        
        # Device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Networks
        self.policy_net = DQN(state_size, action_size, use_layernorm).to(self.device)
        self.target_net = DQN(state_size, action_size, use_layernorm).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss(reduction='none')

    def act(self, state, action_mask=None):
        if np.random.rand() <= self.epsilon:
            if action_mask is not None:
                legal_indices = np.where(action_mask)[0]
                return np.random.choice(legal_indices) if len(legal_indices) > 0 else 0
            return random.randrange(self.action_size)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
            if action_mask is not None:
                mask_t = torch.BoolTensor(action_mask).to(self.device)
                q_values[0][~mask_t] = float('-inf')
        return torch.argmax(q_values[0]).item()

    def train(self, state, action, reward, next_state, done):
        # Reward centering (optional)
        if self.use_reward_centering:
            self.reward_mean = self.reward_beta * self.reward_mean + (1 - self.reward_beta) * reward
            reward = reward - self.reward_mean
        
        self.memory.append((state, action, reward, next_state, done))
        
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([i[0] for i in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([i[1] for i in minibatch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([i[2] for i in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([i[4] for i in minibatch])).to(self.device)

        # Double DQN
        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            
            # Q-value clipping (optional)
            if self.use_q_clipping:
                next_q_values = torch.clamp(next_q_values, self.q_clip_min, self.q_clip_max)
        
        target_q = rewards + (self.gamma * next_q_values * (1 - dones))
        if self.use_q_clipping:
            target_q = torch.clamp(target_q, self.q_clip_min, self.q_clip_max)
        
        current_q = self.policy_net(states).gather(1, actions).squeeze(1)
        
        # Loss computation
        td_errors = current_q - target_q
        
        if self.use_asymmetric_td:
            # Asymmetric weighting
            weights = torch.where(
                td_errors > 0,
                self.atd_pos_w * torch.ones_like(td_errors),
                self.atd_neg_w * torch.ones_like(td_errors)
            )
            loss = (weights * self.criterion(current_q, target_q)).mean()
        else:
            # Standard Huber loss
            loss = self.criterion(current_q, target_q).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (optional)
        if self.use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip_norm)
        
        self.optimizer.step()
        
        # Target update
        if self.use_soft_update:
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
        
        return loss.item(), current_q.mean().item()

    def hard_update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

EXPERIMENT_CONFIGS = {
    "FULL": {
        "use_asymmetric_td": True,
        "use_gradient_clipping": True,
        "use_soft_update": True,
        "use_q_clipping": True,
        "use_reward_centering": True,
        "use_layernorm": True,
    },
    "NO_ATD": {
        "use_asymmetric_td": False,  # KEY ABLATION
        "use_gradient_clipping": True,
        "use_soft_update": True,
        "use_q_clipping": True,
        "use_reward_centering": True,
        "use_layernorm": True,
    },
    "NO_GRAD_CLIP": {
        "use_asymmetric_td": True,
        "use_gradient_clipping": False,  # KEY ABLATION
        "use_soft_update": True,
        "use_q_clipping": True,
        "use_reward_centering": True,
        "use_layernorm": True,
    },
    "NO_SOFT_UPDATE": {
        "use_asymmetric_td": True,
        "use_gradient_clipping": True,
        "use_soft_update": False,  # KEY ABLATION
        "use_q_clipping": True,
        "use_reward_centering": True,
        "use_layernorm": True,
    },
    "NO_Q_CLIP": {
        "use_asymmetric_td": True,
        "use_gradient_clipping": True,
        "use_soft_update": True,
        "use_q_clipping": False,  # KEY ABLATION
        "use_reward_centering": True,
        "use_layernorm": True,
    },
    "VANILLA": {
        "use_asymmetric_td": False,
        "use_gradient_clipping": False,
        "use_soft_update": False,
        "use_q_clipping": False,
        "use_reward_centering": False,
        "use_layernorm": False,
    },
}


# =============================================================================
# TRAINING LOOP
# =============================================================================

def run_experiment(config_name, num_games=500, seed=42):
    """Run a single experiment with given configuration."""
    
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    config = EXPERIMENT_CONFIGS[config_name]
    
    env = MiniChessEnv()
    agent = AblationAgent(env.state_size, env.action_size, **config)
    
    results = {
        "config": config_name,
        "seed": seed,
        "games": [],
        "wins": 0,
        "losses": 0,
    }
    
    print(f"\n{'='*50}")
    print(f"Running: {config_name} (seed={seed})")
    print(f"{'='*50}")
    
    for game in range(1, num_games + 1):
        state = env.reset()
        done = False
        moves = 0
        total_q = 0
        total_loss = 0
        steps = 0
        game_reward = 0
        
        while not done and moves < 40:
            moves += 1
            
            # Agent move
            mask = env.get_action_mask()
            if not np.any(mask):
                done = True
                break
            
            action = agent.act(state, action_mask=mask)
            next_state, reward, done, _ = env.step(action)
            
            loss, q = agent.train(state, action, reward, next_state, done)
            total_loss += loss
            total_q += q
            steps += 1
            game_reward += reward
            state = next_state
            
            if done:
                break
            
            # Opponent move (random for simplicity in ablation)
            opp_mask = env.get_action_mask()
            if not np.any(opp_mask):
                game_reward += 1.0
                done = True
                break
            
            opp_action = np.random.choice(np.where(opp_mask)[0])
            next_state, opp_reward, done, _ = env.step(opp_action)
            
            if done:
                game_reward -= 1.0
            
            state = next_state
        
        # Hard update for non-soft-update experiments
        if not config["use_soft_update"] and game % 10 == 0:
            agent.hard_update_target()
        
        agent.update_epsilon()
        
        outcome = "Win" if game_reward > 0 else "Loss/Draw"
        if game_reward > 0:
            results["wins"] += 1
        else:
            results["losses"] += 1
        
        results["games"].append({
            "game": game,
            "outcome": outcome,
            "moves": moves,
            "avg_loss": total_loss / max(steps, 1),
            "avg_q": total_q / max(steps, 1),
        })
        
        if game % 100 == 0:
            win_rate = results["wins"] / game * 100
            print(f"Game {game}/{num_games} | Win Rate: {win_rate:.1f}% | Q: {total_q/max(steps,1):.2f}")
    
    results["final_win_rate"] = results["wins"] / num_games * 100
    results["final_avg_q"] = np.mean([g["avg_q"] for g in results["games"][-50:]])
    results["final_avg_loss"] = np.mean([g["avg_loss"] for g in results["games"][-50:]])
    
    return results


def run_all_ablations(num_games=500, seeds=[42, 123, 456]):
    """Run all ablation experiments with multiple seeds."""
    
    all_results = {}
    
    for config_name in EXPERIMENT_CONFIGS.keys():
        all_results[config_name] = []
        
        for seed in seeds:
            results = run_experiment(config_name, num_games=num_games, seed=seed)
            all_results[config_name].append(results)
            
            # Save intermediate results
            save_results(all_results, "research/experiments/ablation_results.json")
    
    return all_results


def save_results(results, filepath):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {filepath}")


def print_summary(all_results):
    """Print summary table of ablation results."""
    
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)
    print(f"{'Config':<20} {'Win Rate':<15} {'Avg Q':<15} {'Avg Loss':<15}")
    print("-"*70)
    
    for config_name, runs in all_results.items():
        win_rates = [r["final_win_rate"] for r in runs]
        avg_qs = [r["final_avg_q"] for r in runs]
        avg_losses = [r["final_avg_loss"] for r in runs]
        
        wr_mean = np.mean(win_rates)
        wr_std = np.std(win_rates)
        q_mean = np.mean(avg_qs)
        loss_mean = np.mean(avg_losses)
        
        print(f"{config_name:<20} {wr_mean:.1f}% ± {wr_std:.1f} {q_mean:<15.2f} {loss_mean:<15.4f}")
    
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--config", type=str, default="all", 
                       help="Config name or 'all' for full ablation")
    parser.add_argument("--games", type=int, default=500,
                       help="Number of games per experiment")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                       help="Random seeds for reproducibility")
    
    args = parser.parse_args()
    
    if args.config == "all":
        results = run_all_ablations(num_games=args.games, seeds=args.seeds)
        print_summary(results)
    else:
        if args.config not in EXPERIMENT_CONFIGS:
            print(f"Unknown config: {args.config}")
            print(f"Available: {list(EXPERIMENT_CONFIGS.keys())}")
        else:
            results = run_experiment(args.config, num_games=args.games, seed=args.seeds[0])
            print(f"\nFinal Win Rate: {results['final_win_rate']:.1f}%")
            print(f"Final Avg Q: {results['final_avg_q']:.2f}")
            print(f"Final Avg Loss: {results['final_avg_loss']:.4f}")
