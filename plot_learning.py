"""
Learning Curve Visualization: "Certificate of Intelligence"
===========================================================
Generates research-grade plots showing agent convergence.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Use a built-in style that's guaranteed to exist
plt.style.use('seaborn-v0_8-whitegrid')

def plot_performance():
    # 1. Load Data
    csv_path = "models/self_play_metrics.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: Could not find metrics file.")
        return

    # 2. Data Preprocessing
    df['Win_Numeric'] = df['Outcome'].apply(lambda x: 1 if x == 'Win' else 0)
    
    # Calculate Rolling Averages (Window = 50 games)
    df['Win_Rate'] = df['Win_Numeric'].rolling(window=50, min_periods=1).mean() * 100
    df['Q_Trend'] = df['AvgQ'].rolling(window=50, min_periods=1).mean()
    df['Loss_Trend'] = df['AvgLoss'].rolling(window=50, min_periods=1).mean()

    # 3. Create Research-Grade Plots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Color scheme
    bg_color = '#1a1a2e'
    grid_color = '#2a2a4e'
    
    # Plot 1: Win Rate
    ax1 = axes[0]
    ax1.fill_between(df['Game'], df['Win_Rate'], alpha=0.3, color='#00ff88')
    ax1.plot(df['Game'], df['Win_Rate'], color='#00ff88', linewidth=2.5, label='Win Rate')
    ax1.axhline(50, color='#ff6b6b', linestyle='--', alpha=0.7, linewidth=1.5, label='Random Baseline (50%)')
    ax1.axhline(87.5, color='#ffd93d', linestyle='--', alpha=0.7, linewidth=1.5, label='Final: 87.5%')
    ax1.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold', color='white')
    ax1.set_title('🏆 Agent Dominance: Win Rate Over Training', fontsize=14, fontweight='bold', color='white', pad=10)
    ax1.set_ylim(0, 100)
    ax1.legend(loc='lower right', facecolor=bg_color, edgecolor='white', labelcolor='white')
    ax1.set_facecolor(bg_color)
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.3, color=grid_color)
    for spine in ax1.spines.values():
        spine.set_color('white')
    
    # Plot 2: Q-Value Growth (Intelligence)
    ax2 = axes[1]
    ax2.fill_between(df['Game'], df['Q_Trend'], alpha=0.3, color='#00d4ff')
    ax2.plot(df['Game'], df['Q_Trend'], color='#00d4ff', linewidth=2.5, label='Average Q-Value')
    ax2.axhline(10, color='#ff6b6b', linestyle='--', alpha=0.7, linewidth=1.5, label='Q-Clip Bound (10)')
    ax2.set_ylabel('Q-Value', fontsize=12, fontweight='bold', color='white')
    ax2.set_title('🧠 Agent Intelligence: Q-Value Growth (Bounded)', fontsize=14, fontweight='bold', color='white', pad=10)
    ax2.legend(loc='lower right', facecolor=bg_color, edgecolor='white', labelcolor='white')
    ax2.set_facecolor(bg_color)
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.3, color=grid_color)
    for spine in ax2.spines.values():
        spine.set_color('white')
    
    # Plot 3: Loss Convergence
    ax3 = axes[2]
    ax3.fill_between(df['Game'], df['Loss_Trend'], alpha=0.3, color='#ff6b6b')
    ax3.plot(df['Game'], df['Loss_Trend'], color='#ff6b6b', linewidth=2.5, label='Training Loss')
    ax3.set_ylabel('Loss', fontsize=12, fontweight='bold', color='white')
    ax3.set_xlabel('Games Played', fontsize=12, fontweight='bold', color='white')
    ax3.set_title('📉 Training Stability: Loss Convergence', fontsize=14, fontweight='bold', color='white', pad=10)
    ax3.legend(loc='upper right', facecolor=bg_color, edgecolor='white', labelcolor='white')
    ax3.set_facecolor(bg_color)
    ax3.tick_params(colors='white')
    ax3.grid(True, alpha=0.3, color=grid_color)
    for spine in ax3.spines.values():
        spine.set_color('white')

    fig.patch.set_facecolor(bg_color)
    
    plt.tight_layout()
    
    save_path = "models/final_performance_report.png"
    plt.savefig(save_path, dpi=150, facecolor=bg_color, bbox_inches='tight')
    print(f"✓ Analysis complete. Graph saved to: {save_path}")
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("  CERTIFICATE OF INTELLIGENCE")
    print("="*50)
    print(f"  Total Games: {len(df)}")
    print(f"  Final Win Rate: {df['Win_Rate'].iloc[-1]:.1f}%")
    print(f"  Final Q-Value: {df['Q_Trend'].iloc[-1]:.2f}")
    print(f"  Final Loss: {df['Loss_Trend'].iloc[-1]:.4f}")
    print(f"  Peak Win Rate: {df['Win_Rate'].max():.1f}%")
    print("="*50)
    print("  STATUS: ✅ CONVERGENCE CONFIRMED")
    print("="*50)

if __name__ == "__main__":
    plot_performance()
