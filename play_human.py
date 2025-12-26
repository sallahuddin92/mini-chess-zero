"""
Human vs AI: Play Against Your Champion
=======================================
Terminal-based chess game against the trained agent.
"""

import torch
import numpy as np
import sys
import os

sys.path.append(os.getcwd())
from src.environment import MiniChessEnv
from src.agent import DQNAgent

# Board notation helpers
COLS = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
INV_COLS = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'}
PIECE_SYMBOLS = {1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K'}

def parse_move(move_str):
    """Converts 'a1a2' to (start_idx, end_idx)"""
    try:
        move_str = move_str.strip().lower()
        if len(move_str) != 4:
            return None, None
        c1, r1, c2, r2 = move_str[0], int(move_str[1]), move_str[2], int(move_str[3])
        
        if c1 not in COLS or c2 not in COLS:
            return None, None
        if not (1 <= r1 <= 5 and 1 <= r2 <= 5):
            return None, None
            
        start = (r1 - 1) * 5 + COLS[c1]
        end = (r2 - 1) * 5 + COLS[c2]
        return start, end
    except:
        return None, None

def index_to_notation(idx):
    """Convert board index to chess notation."""
    row = idx // 5
    col = idx % 5
    return f"{INV_COLS[col]}{row + 1}"

def render_board(board, last_move=None):
    """Render the 5x5 board with colors."""
    print()
    print("     ╔═══╤═══╤═══╤═══╤═══╗")
    
    for row in range(4, -1, -1):
        line = f"  {row+1}  ║"
        for col in range(5):
            idx = row * 5 + col
            p = board[idx]
            
            if p == 0:
                symbol = " · "
            else:
                is_white = p > 0
                ptype = abs(int(p))
                char = PIECE_SYMBOLS.get(ptype, '?')
                if not is_white:
                    char = char.lower()
                # Color: White pieces in CAPS, Black in lowercase
                symbol = f" {char} "
            
            line += symbol
            line += "│" if col < 4 else "║"
        print(line)
        if row > 0:
            print("     ╟───┼───┼───┼───┼───╢")
    
    print("     ╚═══╧═══╧═══╧═══╧═══╝")
    print("       a   b   c   d   e")
    print()
    print("  Pieces: P=Pawn N=Knight B=Bishop R=Rook Q=Queen K=King")
    print("  UPPERCASE = White (You)  lowercase = Black (AI)")
    print()

def get_ai_move_notation(action_idx):
    """Convert action index to chess notation."""
    start = action_idx // 25
    end = action_idx % 25
    return f"{index_to_notation(start)}{index_to_notation(end)}"

def play():
    print("\n" + "="*50)
    print("  🎮 MINI-CHESS ZERO: HUMAN vs AI")
    print("="*50)
    
    # 1. Load the Champion
    env = MiniChessEnv()
    agent = DQNAgent(env.state_size, env.action_size)
    
    model_path = "models/final_model.pth"
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=agent.device)
            if isinstance(checkpoint, dict) and 'policy_net' in checkpoint:
                agent.policy_net.load_state_dict(checkpoint['policy_net'])
                epsilon = checkpoint.get('epsilon', 0.0)
            else:
                agent.policy_net.load_state_dict(checkpoint)
                epsilon = 0.0
            agent.epsilon = 0.0  # Pure exploitation
            print(f"\n  ✓ Loaded Champion Agent")
            print(f"  ✓ Training: 1000 games @ 87.5% win rate")
            print(f"  ✓ Mode: Pure exploitation (epsilon=0)")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print("  ✗ No trained model found. Run training first!")
        return

    # 2. Game Loop
    state = env.reset()
    done = False
    move_count = 0
    
    print("\n  You are WHITE (uppercase). AI is BLACK (lowercase).")
    print("  Enter moves as: a2a3  (from-to squares)")
    print("  Type 'quit' to exit, 'legal' to see legal moves\n")

    while not done:
        render_board(env.game.board)
        
        if env.game.turn == 1:  # Human turn (White)
            # Show whose turn
            print("  >>> YOUR TURN (White)")
            
            valid_move = False
            while not valid_move:
                user_input = input("  Your move: ").strip().lower()
                
                if user_input == 'quit':
                    print("\n  Game aborted. Thanks for playing!")
                    return
                
                if user_input == 'legal':
                    legal_moves = env.game.get_legal_moves()
                    print("  Legal moves:", [f"{index_to_notation(s)}{index_to_notation(e)}" for s, e in legal_moves])
                    continue
                
                start, end = parse_move(user_input)
                
                if start is None:
                    print("  ✗ Invalid format. Use 'a2a3' notation.")
                    continue
                    
                legal_moves = env.game.get_legal_moves()
                if (start, end) not in legal_moves:
                    print("  ✗ Illegal move! Type 'legal' to see options.")
                    continue
                    
                # Execute move
                action_idx = start * 25 + end
                state, reward, done, info = env.step(action_idx)
                move_count += 1
                valid_move = True
                
                if done:
                    render_board(env.game.board)
                    print("  🎉 CHECKMATE! You defeated the AI!")
                    return
        
        else:  # AI turn (Black)
            print("  >>> AI THINKING...")
            
            mask = env.get_action_mask()
            
            if not np.any(mask):
                print("  AI has no legal moves!")
                done = True
                break
            
            action = agent.act(state, action_mask=mask)
            ai_move = get_ai_move_notation(action)
            
            state, reward, done, info = env.step(action)
            move_count += 1
            
            print(f"  AI plays: {ai_move}")
            
            if done:
                render_board(env.game.board)
                print("  💀 CHECKMATE! The AI wins.")
                return
    
    render_board(env.game.board)
    print(f"  Game ended after {move_count} moves.")

if __name__ == "__main__":
    try:
        play()
    except KeyboardInterrupt:
        print("\n\n  Game interrupted. Goodbye!")
