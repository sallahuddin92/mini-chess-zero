"""
Mini-Chess Zero Training Visualization
=======================================
Generates publication-quality charts for research analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

def load_data(filepath='models/self_play_metrics.csv'):
    """Load and preprocess training metrics."""
    df = pd.read_csv(filepath)
    df['WinBinary'] = (df['Outcome'] == 'Win').astype(int)
    df['RollingWinRate'] = df['WinBinary'].rolling(window=50, min_periods=1).mean() * 100
    df['RollingLoss'] = df['AvgLoss'].rolling(window=50, min_periods=1).mean()
    df['RollingQ'] = df['AvgQ'].rolling(window=50, min_periods=1).mean()
    return df

def create_training_dashboard(df, save_path='models/training_analysis.png'):
    """Create comprehensive 2x2 training dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Mini-Chess Zero: Training Analysis\nAsymmetric TD Learning with Stabilization', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    games = df['Game']
    
    # 1. Win Rate Over Time
    ax1 = axes[0, 0]
    ax1.fill_between(games, df['RollingWinRate'], alpha=0.3, color='#2ecc71')
    ax1.plot(games, df['RollingWinRate'], color='#27ae60', linewidth=2, label='Rolling Win Rate (50 games)')
    ax1.axhline(y=87.5, color='#e74c3c', linestyle='--', linewidth=1.5, label='Final: 87.5%')
    ax1.set_xlabel('Game')
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_title('Win Rate Progression', fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Q-Value Stability
    ax2 = axes[0, 1]
    ax2.fill_between(games, df['RollingQ'], alpha=0.3, color='#3498db')
    ax2.plot(games, df['RollingQ'], color='#2980b9', linewidth=2, label='Rolling Avg Q-Value')
    ax2.axhline(y=10, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.7, label='Clip Bound (±10)')
    ax2.axhline(y=-10, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_xlabel('Game')
    ax2.set_ylabel('Q-Value')
    ax2.set_title('Q-Value Stability (Bounded)', fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Loss Convergence
    ax3 = axes[1, 0]
    ax3.fill_between(games, df['RollingLoss'], alpha=0.3, color='#9b59b6')
    ax3.plot(games, df['RollingLoss'], color='#8e44ad', linewidth=2, label='Rolling Avg Loss')
    ax3.set_xlabel('Game')
    ax3.set_ylabel('Loss')
    ax3.set_title('Loss Convergence', fontweight='bold')
    ax3.set_ylim(0, max(df['RollingLoss']) * 1.2)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Game Length Distribution
    ax4 = axes[1, 1]
    wins = df[df['Outcome'] == 'Win']['Moves']
    losses = df[df['Outcome'] == 'Loss/Draw']['Moves']
    
    ax4.hist(wins, bins=20, alpha=0.7, color='#2ecc71', label=f'Wins (n={len(wins)})', edgecolor='white')
    ax4.hist(losses, bins=20, alpha=0.7, color='#e74c3c', label=f'Loss/Draw (n={len(losses)})', edgecolor='white')
    ax4.set_xlabel('Moves per Game')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Game Length Distribution', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    return save_path

def create_comparison_chart(save_path='models/before_after_comparison.png'):
    """Create before/after comparison chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Win Rate\n(%)', 'Q-Value\n(log scale)', 'Loss\n(log scale)']
    before = [3, np.log10(70615864), np.log10(5643622)]
    after = [87.5, np.log10(9.2), np.log10(0.05)]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, before, width, label='Before (Baseline)', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, after, width, label='After (Stabilized)', color='#2ecc71', alpha=0.8)
    
    ax.set_ylabel('Value (log scale for Q/Loss)')
    ax.set_title('Training Stabilization: Before vs After\nMini-Chess Zero with Asymmetric TD Learning', 
                 fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Add value labels
    for bar, val in zip(bars1, [3, 70615864, 5643622]):
        ax.annotate(f'{val:,.0f}' if val > 100 else f'{val}%',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    for bar, val in zip(bars2, [87.5, 9.2, 0.05]):
        ax.annotate(f'{val}' if val < 100 else f'{val}%',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()
    
    return save_path

def print_summary_stats(df):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("  MINI-CHESS ZERO: TRAINING SUMMARY")
    print("="*60)
    
    total_wins = (df['Outcome'] == 'Win').sum()
    total_games = len(df)
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total Games: {total_games}")
    print(f"   Total Wins: {total_wins} ({total_wins/total_games*100:.1f}%)")
    print(f"   Total Losses/Draws: {total_games - total_wins}")
    
    print(f"\n📈 Q-Value Analysis:")
    print(f"   Initial Q: {df['AvgQ'].iloc[:10].mean():.4f}")
    print(f"   Final Q: {df['AvgQ'].iloc[-50:].mean():.4f}")
    print(f"   Max Q: {df['AvgQ'].max():.4f}")
    print(f"   Min Q: {df['AvgQ'].min():.4f}")
    
    print(f"\n📉 Loss Analysis:")
    print(f"   Initial Loss: {df['AvgLoss'].iloc[:10].mean():.4f}")
    print(f"   Final Loss: {df['AvgLoss'].iloc[-50:].mean():.4f}")
    print(f"   Max Loss: {df['AvgLoss'].max():.4f}")
    
    print(f"\n🎮 Game Length:")
    print(f"   Average Moves: {df['Moves'].mean():.1f}")
    print(f"   Fastest Win: {df[df['Outcome']=='Win']['Moves'].min()} moves")
    print(f"   Longest Game: {df['Moves'].max()} moves")
    
    # Win rate by segment
    print(f"\n📊 Win Rate by Training Phase:")
    for i in range(0, 1000, 200):
        segment = df[(df['Game'] > i) & (df['Game'] <= i+200)]
        wr = (segment['Outcome'] == 'Win').mean() * 100
        print(f"   Games {i+1}-{i+200}: {wr:.1f}%")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    # Load data
    df = load_data()
    
    # Print statistics
    print_summary_stats(df)
    
    # Create visualizations
    print("\n🎨 Generating visualizations...")
    create_training_dashboard(df)
    create_comparison_chart()
    
    print("\n✅ All visualizations saved to models/ directory")
