# src/evaluate_metrics.py
import os
import torch
import numpy as np
from src.chess_env import ChessEnv
from src.train_dqn_chess import DQN, select_action

# -------------------
# Config
# -------------------
MODEL_FILE = "models/dqn_chess.pth"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
NUM_GAMES = 50  # Number of self-play games to evaluate

print(f"Using device: {DEVICE}")

# -------------------
# Load Environment and Model
# -------------------
env = ChessEnv()
input_dim = len(env.get_state())
output_dim = 64*64  # 4096 possible moves

policy_net = DQN(input_dim, output_dim).to(DEVICE)
if os.path.exists(MODEL_FILE):
    policy_net.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    policy_net.eval()
    print("[loaded] Existing model checkpoint")
else:
    raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

# -------------------
# Evaluation Loop
# -------------------
results = {"win": 0, "loss": 0, "draw": 0}
trajectory_lengths = []

for game_num in range(1, NUM_GAMES+1):
    state = env.reset()
    done = False
    moves = 0
    final_reward = 0

    while not done:
        move, _ = select_action(state, env, policy_net, steps_done=0)  # Deterministic
        state, reward, done = env.step(move)
        moves += 1
        if done:
            final_reward = reward  # Capture last reward for game outcome

    trajectory_lengths.append(moves)
    
    # Determine result from final reward
    if final_reward > 0:
        game_result = "win"
    elif final_reward < 0:
        game_result = "loss"
    else:
        game_result = "draw"

    results[game_result] += 1
    print(f"[game {game_num}] Moves: {moves} | Result: {game_result}")

# -------------------
# Summary Metrics
# -------------------
total_games = NUM_GAMES
print("\n=== Evaluation Summary ===")
print(f"Total games: {total_games}")
print(f"Wins: {results['win']} ({results['win']/total_games*100:.2f}%)")
print(f"Losses: {results['loss']} ({results['loss']/total_games*100:.2f}%)")
print(f"Draws: {results['draw']} ({results['draw']/total_games*100:.2f}%)")
print(f"Average moves per game: {np.mean(trajectory_lengths):.2f}")
print(f"Max moves in a game: {np.max(trajectory_lengths)}")
print(f"Min moves in a game: {np.min(trajectory_lengths)}")
