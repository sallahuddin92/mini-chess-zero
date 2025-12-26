"""
Mini-Chess Zero Training Monitor
================================
Research-grade self-play training with:
- Phantom League (past-self opponents)
- Real-time Rich dashboard
- Divergence detection and recovery
"""

import sys
import os
import csv
import copy
import random
import numpy as np
from collections import deque

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

# Optional Sparkline
try:
    from rich.sparkline import Sparkline
    HAS_SPARKLINE = True
except ImportError:
    HAS_SPARKLINE = False

# Import Project Modules
sys.path.append(os.getcwd())
try:
    from src.environment import MiniChessEnv
    from src.agent import DQNAgent
except ImportError:
    print("Please ensure src/environment.py and src/agent.py exist.")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "total_games": 1000,          # More games for proper training
    "max_moves": 40,              # Max moves per game
    "metrics_file": "models/self_play_metrics.csv",
    "checkpoint_dir": "models/checkpoints",
    "checkpoint_every": 50,       # Save checkpoint frequency
    "snapshot_every": 25,         # Phantom snapshot frequency
    "divergence_threshold": 100,  # Q-value panic threshold
}

os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)


# =============================================================================
# PHANTOM LEAGUE (Self-Play Opponent Pool)
# =============================================================================

class PhantomLeague:
    """
    Maintains a pool of past policy snapshots for diverse self-play.
    Prevents policy collapse by training against varied opponents.
    """
    def __init__(self, agent_class, state_size, action_size):
        self.phantoms = []
        self.agent_class = agent_class
        self.state_size = state_size
        self.action_size = action_size
        self.opponent = agent_class(state_size, action_size)
        self.opponent.epsilon = 0.05  # Low exploration for opponent

    def save_snapshot(self, active_agent):
        """Store a copy of current policy weights."""
        brain = copy.deepcopy(active_agent.policy_net.state_dict())
        self.phantoms.append(brain)
        if len(self.phantoms) > 10:
            self.phantoms.pop(0)

    def get_move(self, state, env, active_agent):
        """Select move using varied opponent strategies."""
        mask = env.get_action_mask()
        
        if not np.any(mask):
            return 0  # No moves possible

        roll = random.random()
        
        if roll < 0.1:  # 10% Random
            legal_indices = np.where(mask)[0]
            return np.random.choice(legal_indices)
            
        elif roll < 0.5 or not self.phantoms:  # 40% Current self
            return active_agent.act(state, action_mask=mask)
            
        else:  # 50% Phantom (Past self)
            past_brain = random.choice(self.phantoms)
            self.opponent.policy_net.load_state_dict(past_brain)
            return self.opponent.act(state, action_mask=mask)


# =============================================================================
# DASHBOARD
# =============================================================================

def generate_dashboard(data, monitor):
    """Generate Rich dashboard layout."""
    # Color based on panic state
    color = "red" if monitor["is_panic"] else "green"
    
    header = Panel(
        Text(
            f" MINI-CHESS ZERO (M4) | Game {data['game']}/{CONFIG['total_games']} | {data['opp_type']} ",
            style=f"bold white on {color}",
            justify="center"
        )
    )
    
    # Stats grid
    grid = Table.grid(padding=1)
    grid.add_column(style="cyan", justify="right")
    grid.add_column(style="yellow", justify="left")
    grid.add_row("Epsilon:", f"{data['epsilon']:.3f}")
    grid.add_row("Avg Q-Value:", f"{data['q_val']:.4f}")
    grid.add_row("Last Loss:", f"{data['loss']:.4f}")
    grid.add_row("Win Rate:", f"{data['win_rate']:.1f}%")
    
    stats = Panel(grid, title="Telemetry", border_style="blue")
    
    # Q-trend graph
    if HAS_SPARKLINE and len(monitor["q_hist"]) > 5:
        graph = Panel(
            Sparkline(monitor["q_hist"], summary_function=max),
            title="Q-Trend",
            border_style="green"
        )
    else:
        graph = Panel(f"Games: {data['wins']}W / {data['losses']}L", border_style="white")
    
    # Layout
    layout = Layout()
    layout.split_column(
        Layout(header, size=3),
        Layout(name="body")
    )
    layout["body"].split_row(Layout(stats), Layout(graph))
    
    return layout


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train():
    """Main training loop with all research-grade techniques."""
    env = MiniChessEnv()
    agent = DQNAgent(env.state_size, env.action_size)
    league = PhantomLeague(DQNAgent, env.state_size, env.action_size)
    
    # Statistics tracking
    q_history = deque(maxlen=100)
    wins = 0
    losses = 0
    
    # Reset metrics log
    with open(CONFIG["metrics_file"], 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Game", "Outcome", "Moves", "FinalReward", "AvgLoss", "AvgQ"])
    
    print("=" * 60)
    print("   MINI-CHESS ZERO - Research Grade Training")
    print("   Stabilization: Gradient Clip | Soft Updates | ATD Loss")
    print("=" * 60)
    
    with Live(refresh_per_second=10, screen=False) as live:
        for game in range(1, CONFIG["total_games"] + 1):
            state = env.reset()
            moves = 0
            done = False
            total_loss = 0
            total_q = 0
            train_steps = 0
            game_reward = 0
            
            opp_type = "Phantom" if league.phantoms else "Self"
            
            while not done and moves < CONFIG["max_moves"]:
                moves += 1
                
                # === WHITE (AGENT) ===
                mask = env.get_action_mask()
                if not np.any(mask):
                    done = True
                    break
                
                action = agent.act(state, action_mask=mask)
                next_state, reward, done, info = env.step(action)
                
                loss, q = agent.train(state, action, reward, next_state, done)
                total_loss += loss
                total_q += q
                train_steps += 1
                game_reward += reward
                
                # Track Q-values (clamped for display)
                q_history.append(max(min(int(q * 10), 100), -100))
                
                state = next_state
                
                if done:
                    break
                
                # === BLACK (OPPONENT) ===
                opp_mask = env.get_action_mask()
                if not np.any(opp_mask):
                    # Opponent has no moves = Agent wins
                    agent.train(state, action, 1.0, next_state, True)
                    game_reward += 1.0
                    done = True
                    break
                
                opp_move = league.get_move(state, env, agent)
                next_state_opp, r_opp, done_opp, _ = env.step(opp_move)
                
                state = next_state_opp
                
                if done_opp:
                    # Agent lost
                    agent.train(state, action, -1.0, next_state_opp, True)
                    game_reward -= 1.0
                    done = True
                
                # Update dashboard
                avg_loss = total_loss / max(train_steps, 1)
                avg_q = total_q / max(train_steps, 1)
                win_rate = (wins / max(game - 1, 1)) * 100
                
                is_panic = avg_loss > CONFIG["divergence_threshold"] or abs(avg_q) > 50
                
                data = {
                    "game": game,
                    "epsilon": agent.epsilon,
                    "q_val": avg_q,
                    "loss": avg_loss,
                    "opp_type": opp_type,
                    "win_rate": win_rate,
                    "wins": wins,
                    "losses": losses
                }
                monitor = {
                    "q_hist": list(q_history),
                    "is_panic": is_panic
                }
                live.update(generate_dashboard(data, monitor))
            
            # End of game
            agent.update_epsilon()
            
            # Determine outcome
            if game_reward > 0:
                outcome = "Win"
                wins += 1
            else:
                outcome = "Loss/Draw"
                losses += 1
            
            # Periodic snapshots
            if game % CONFIG["snapshot_every"] == 0:
                league.save_snapshot(agent)
            
            # Periodic checkpoints
            if game % CONFIG["checkpoint_every"] == 0:
                agent.save(f"{CONFIG['checkpoint_dir']}/checkpoint_{game}.pth")
            
            # Log metrics
            avg_loss = total_loss / max(train_steps, 1)
            avg_q = total_q / max(train_steps, 1)
            
            with open(CONFIG["metrics_file"], 'a', newline='') as f:
                csv.writer(f).writerow([
                    game, outcome, moves, 
                    f"{game_reward:.3f}", 
                    f"{avg_loss:.4f}", 
                    f"{avg_q:.4f}"
                ])
    
    # Final summary
    print("\n" + "=" * 60)
    print("   TRAINING COMPLETE")
    print(f"   Total Games: {CONFIG['total_games']}")
    print(f"   Final Win Rate: {(wins / CONFIG['total_games']) * 100:.1f}%")
    print(f"   Final Epsilon: {agent.epsilon:.4f}")
    print("=" * 60)
    
    # Save final model
    agent.save("models/final_model.pth")
    print("Saved final model to models/final_model.pth")


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\nTraining interrupted safely.")