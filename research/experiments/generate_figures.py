"""
Generate publication-quality figures from ablation study results.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

def load_results(filepath='research/experiments/ablation_results.json'):
    with open(filepath, 'r') as f:
        return json.load(f)

def create_bar_chart(results, save_path='research/figures/ablation_winrate.png'):
    """Bar chart comparing win rates across conditions."""
    
    configs = list(results.keys())
    
    # Calculate means and stds
    means = []
    stds = []
    for config in configs:
        win_rates = [r['final_win_rate'] for r in results[config]]
        means.append(np.mean(win_rates))
        stds.append(np.std(win_rates))
    
    # Colors: green for stable, red for vanilla
    colors = ['#2ecc71' if c != 'VANILLA' else '#e74c3c' for c in configs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(configs))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='white')
    
    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('Win Rate (%)', fontweight='bold')
    ax.set_title('Ablation Study: Win Rate Comparison\n(500 games × 3 seeds per condition)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.annotate(f'{mean:.1f}%',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1),
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Legend
    stable_patch = mpatches.Patch(color='#2ecc71', label='Stabilized')
    vanilla_patch = mpatches.Patch(color='#e74c3c', label='No Stabilization')
    ax.legend(handles=[stable_patch, vanilla_patch], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()

def create_qvalue_comparison(results, save_path='research/figures/ablation_qvalue.png'):
    """Bar chart comparing Q-values (log scale for VANILLA)."""
    
    configs = list(results.keys())
    
    means = []
    for config in configs:
        avg_qs = [r['final_avg_q'] for r in results[config]]
        means.append(np.mean(avg_qs))
    
    colors = ['#3498db' if c != 'VANILLA' else '#e74c3c' for c in configs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(configs))
    bars = ax.bar(x, means, color=colors, alpha=0.8, edgecolor='white')
    
    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('Average Q-Value (log scale)', fontweight='bold')
    ax.set_title('Q-Value Stability Across Ablation Conditions\n(VANILLA explodes without stabilization)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_yscale('log')
    
    # Add value labels
    for bar, mean in zip(bars, means):
        label = f'{mean:.0f}' if mean > 100 else f'{mean:.1f}'
        ax.annotate(label,
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add horizontal line for "acceptable" range
    ax.axhline(y=10, color='green', linestyle='--', linewidth=2, label='Target Q-range (≤10)')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()

def create_loss_comparison(results, save_path='research/figures/ablation_loss.png'):
    """Bar chart comparing loss values."""
    
    configs = list(results.keys())
    
    means = []
    for config in configs:
        avg_losses = [r['final_avg_loss'] for r in results[config]]
        means.append(np.mean(avg_losses))
    
    colors = ['#9b59b6' if c != 'VANILLA' else '#e74c3c' for c in configs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(configs))
    bars = ax.bar(x, means, color=colors, alpha=0.8, edgecolor='white')
    
    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('Average Loss (log scale)', fontweight='bold')
    ax.set_title('Training Loss Across Ablation Conditions', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_yscale('log')
    
    for bar, mean in zip(bars, means):
        label = f'{mean:.0f}' if mean > 1 else f'{mean:.3f}'
        ax.annotate(label,
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()

def create_summary_table(results, save_path='research/figures/ablation_table.png'):
    """Create a clean summary table as an image."""
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    # Prepare data
    headers = ['Configuration', 'Win Rate', 'Q-Value', 'Loss', 'Status']
    
    rows = []
    for config, runs in results.items():
        wr_mean = np.mean([r['final_win_rate'] for r in runs])
        wr_std = np.std([r['final_win_rate'] for r in runs])
        q_mean = np.mean([r['final_avg_q'] for r in runs])
        loss_mean = np.mean([r['final_avg_loss'] for r in runs])
        
        status = '✓ Stable' if q_mean < 100 else '✗ Exploded'
        
        rows.append([
            config,
            f'{wr_mean:.1f}% ± {wr_std:.1f}',
            f'{q_mean:.2f}' if q_mean < 100 else f'{q_mean:,.0f}',
            f'{loss_mean:.4f}' if loss_mean < 1 else f'{loss_mean:,.1f}',
            status
        ])
    
    # Create table
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colColours=['#3498db']*5
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color VANILLA row red
    for i, config in enumerate(results.keys()):
        if config == 'VANILLA':
            for j in range(len(headers)):
                table[(i+1, j)].set_facecolor('#ffe6e6')
    
    plt.title('Ablation Study Summary\n(500 games × 3 seeds per condition)', 
              fontweight='bold', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()

def print_latex_table(results):
    """Generate LaTeX table for paper."""
    
    print("\n% LaTeX Table for Paper")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Ablation Study Results}")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("Configuration & Win Rate (\\%) & Avg Q-Value & Avg Loss \\\\")
    print("\\midrule")
    
    for config, runs in results.items():
        wr_mean = np.mean([r['final_win_rate'] for r in runs])
        wr_std = np.std([r['final_win_rate'] for r in runs])
        q_mean = np.mean([r['final_avg_q'] for r in runs])
        loss_mean = np.mean([r['final_avg_loss'] for r in runs])
        
        if config == 'VANILLA':
            print(f"\\textbf{{{config}}} & {wr_mean:.1f} $\\pm$ {wr_std:.1f} & \\textbf{{{q_mean:,.0f}}} & \\textbf{{{loss_mean:,.1f}}} \\\\")
        else:
            print(f"{config} & {wr_mean:.1f} $\\pm$ {wr_std:.1f} & {q_mean:.2f} & {loss_mean:.4f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()

if __name__ == "__main__":
    os.makedirs('research/figures', exist_ok=True)
    
    results = load_results()
    
    print("\n🎨 Generating publication figures...")
    create_bar_chart(results)
    create_qvalue_comparison(results)
    create_loss_comparison(results)
    create_summary_table(results)
    
    print_latex_table(results)
    
    print("\n✅ All figures saved to research/figures/")
