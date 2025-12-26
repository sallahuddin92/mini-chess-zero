"""
Neural Network Weight Analysis
==============================
Analyze which pieces and positions the agent values most.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.getcwd())
from src.agent import DQNAgent
from src.environment import MiniChessEnv

def analyze_weights():
    """Analyze the neural network weights to understand agent preferences."""
    
    print("\n" + "="*60)
    print("  NEURAL NETWORK WEIGHT ANALYSIS")
    print("  What has the AI learned to value?")
    print("="*60)
    
    # Load model
    env = MiniChessEnv()
    agent = DQNAgent(env.state_size, env.action_size)
    
    model_path = "models/final_model.pth"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=agent.device)
        if isinstance(checkpoint, dict) and 'policy_net' in checkpoint:
            agent.policy_net.load_state_dict(checkpoint['policy_net'])
        else:
            agent.policy_net.load_state_dict(checkpoint)
        print("  ✓ Model loaded successfully")
    else:
        print("  ✗ No model found!")
        return
    
    # Get first layer weights (input to first hidden layer)
    first_layer = None
    for name, param in agent.policy_net.named_parameters():
        if 'weight' in name and param.dim() == 2:
            first_layer = param.data.cpu().numpy()
            print(f"  ✓ First layer shape: {first_layer.shape}")
            break
    
    if first_layer is None:
        print("  Could not find first layer weights")
        return
    
    # The input is 25 squares, normalized board state
    # Each neuron in hidden layer learns to detect features
    
    # Weight magnitude per input position
    input_importance = np.abs(first_layer).sum(axis=0)
    
    # Reshape to 5x5 board
    if len(input_importance) >= 25:
        board_importance = input_importance[:25].reshape(5, 5)
    else:
        print(f"  Input size mismatch: {len(input_importance)}")
        return
    
    print("\n  Board Position Importance (Higher = More Attention):")
    print("  +" + "-"*25 + "+")
    for row in range(4, -1, -1):
        line = f"  |"
        for col in range(5):
            val = board_importance[row, col]
            line += f" {val:5.1f}"
        line += " |"
        print(line)
    print("  +" + "-"*25 + "+")
    print("      a     b     c     d     e")
    
    # Analyze piece value learning
    # Create synthetic inputs for each piece type
    print("\n  Piece Value Analysis (Q-value for having each piece):")
    print("  " + "-"*40)
    
    piece_names = {1: 'Pawn', 2: 'Knight', 3: 'Bishop', 4: 'Rook', 5: 'Queen', 6: 'King'}
    piece_values = {}
    
    for piece_type in range(1, 7):
        # Create a state with just this piece at center
        test_state = np.zeros(25, dtype=np.float32)
        test_state[12] = piece_type / 6.0  # Center square, normalized
        
        state_t = torch.FloatTensor(test_state).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            q_values = agent.policy_net(state_t)
            avg_q = q_values.mean().item()
        
        piece_values[piece_type] = avg_q
        print(f"  {piece_names[piece_type]:8s}: Q = {avg_q:+.2f}")
    
    # Rank pieces by learned value
    print("\n  Learned Piece Ranking (by Q-value):")
    sorted_pieces = sorted(piece_values.items(), key=lambda x: x[1], reverse=True)
    for rank, (piece_type, q_val) in enumerate(sorted_pieces, 1):
        print(f"    {rank}. {piece_names[piece_type]:8s} (Q = {q_val:+.2f})")
    
    # Visualize board importance
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heatmap of position importance
    ax1 = axes[0]
    im1 = ax1.imshow(board_importance, cmap='YlOrRd', aspect='equal')
    ax1.set_xticks(range(5))
    ax1.set_yticks(range(5))
    ax1.set_xticklabels(['a', 'b', 'c', 'd', 'e'])
    ax1.set_yticklabels(['1', '2', '3', '4', '5'])
    ax1.set_title('Board Position Importance\n(First Layer Weight Magnitude)', fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Weight Magnitude')
    
    # Add annotations
    for i in range(5):
        for j in range(5):
            ax1.text(j, i, f'{board_importance[i,j]:.1f}', 
                    ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Bar chart of piece values
    ax2 = axes[1]
    pieces = [piece_names[p] for p in range(1, 7)]
    values = [piece_values[p] for p in range(1, 7)]
    colors = ['#3498db' if v > 0 else '#e74c3c' for v in values]
    
    bars = ax2.bar(pieces, values, color=colors, edgecolor='white', linewidth=2)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_ylabel('Average Q-Value', fontweight='bold')
    ax2.set_title('Learned Piece Values\n(Q-value with single piece at center)', fontweight='bold')
    ax2.set_ylim(min(values) - 1, max(values) + 1)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:+.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    save_path = "models/weight_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n  ✓ Visualization saved: {save_path}")
    plt.close()
    
    print("\n" + "="*60)
    print("  ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    analyze_weights()
