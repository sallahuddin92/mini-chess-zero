"""
Backtesting Framework
=====================
Complete backtesting system for evaluating trading strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from trading_env import TradingEnv, generate_synthetic_data
from trading_agent import TradingAgent


@dataclass
class BacktestResult:
    """Container for backtest results."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_pnl: float
    portfolio_values: np.ndarray
    trades: List[Dict]
    daily_returns: np.ndarray


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / (returns.std() + 1e-10)


def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
    """Calculate maximum drawdown."""
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    return drawdown.max() * 100


def calculate_profit_factor(trades: List[Dict]) -> float:
    """Calculate profit factor (gross profit / gross loss)."""
    profits = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
    losses = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
    return profits / (losses + 1e-10)


def backtest(
    agent: TradingAgent,
    env: TradingEnv,
    episodes: int = 1,
    train: bool = False,
    verbose: bool = True
) -> BacktestResult:
    """
    Run backtest on trading environment.
    
    Args:
        agent: Trading agent
        env: Trading environment
        episodes: Number of episodes to run
        train: Whether to train during backtest
        verbose: Print progress
    
    Returns:
        BacktestResult with performance metrics
    """
    all_portfolio_values = []
    all_trades = []
    all_returns = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        episode_values = [env.initial_capital]
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if train:
                agent.train(
                    state, action, reward, next_state, done,
                    info['portfolio_value']
                )
            
            state = next_state
            episode_values.append(info['portfolio_value'])
        
        if train:
            agent.update_epsilon()
        
        all_portfolio_values.extend(episode_values)
        all_trades.extend(env.trades)
        
        # Calculate returns
        values = np.array(episode_values)
        returns = np.diff(values) / values[:-1]
        all_returns.extend(returns.tolist())
        
        if verbose and episode % 10 == 0:
            final_value = episode_values[-1]
            ret = (final_value / env.initial_capital - 1) * 100
            print(f"Episode {episode+1}/{episodes}: Return={ret:.2f}%, "
                  f"Trades={len(env.trades)}, Epsilon={agent.epsilon:.4f}")
    
    # Calculate metrics
    portfolio_values = np.array(all_portfolio_values)
    daily_returns = np.array(all_returns)
    
    total_return = (portfolio_values[-1] / env.initial_capital - 1) * 100
    sharpe = calculate_sharpe_ratio(daily_returns)
    max_dd = calculate_max_drawdown(portfolio_values)
    
    # Trade analysis
    trade_pnls = []
    for i, trade in enumerate(all_trades):
        if i > 0 and trade['action'] in ['sell', 'close']:
            # Approximate P&L
            pnl = (trade['price'] - all_trades[i-1]['price']) * trade['shares']
            trade_pnls.append(pnl)
    
    winning_trades = sum(1 for p in trade_pnls if p > 0)
    win_rate = winning_trades / (len(trade_pnls) + 1e-10) * 100
    profit_factor = calculate_profit_factor([{'pnl': p} for p in trade_pnls])
    
    return BacktestResult(
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_trades=len(all_trades),
        avg_trade_pnl=np.mean(trade_pnls) if trade_pnls else 0,
        portfolio_values=portfolio_values,
        trades=all_trades,
        daily_returns=daily_returns
    )


def plot_backtest_results(
    result: BacktestResult,
    benchmark_values: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """Generate backtest visualization."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Portfolio value
    ax1 = axes[0]
    ax1.plot(result.portfolio_values, label='Strategy', color='#00ff88', linewidth=2)
    if benchmark_values is not None:
        ax1.plot(benchmark_values, label='Buy & Hold', color='#ff6b6b', linewidth=1.5, alpha=0.7)
    ax1.set_title('Portfolio Value', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#1a1a2e')
    
    # Drawdown
    ax2 = axes[1]
    peak = np.maximum.accumulate(result.portfolio_values)
    drawdown = (peak - result.portfolio_values) / peak * 100
    ax2.fill_between(range(len(drawdown)), 0, -drawdown, color='#ff6b6b', alpha=0.5)
    ax2.set_title('Drawdown', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#1a1a2e')
    
    # Daily returns distribution
    ax3 = axes[2]
    ax3.hist(result.daily_returns * 100, bins=50, color='#00d4ff', alpha=0.7, edgecolor='white')
    ax3.axvline(x=0, color='white', linestyle='--', linewidth=1)
    ax3.set_title('Daily Returns Distribution', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Return (%)')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#1a1a2e')
    
    fig.patch.set_facecolor('#0f0f1a')
    for ax in axes:
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#0f0f1a', bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def print_backtest_summary(result: BacktestResult):
    """Print formatted backtest summary."""
    print("\n" + "="*60)
    print("  BACKTEST RESULTS")
    print("="*60)
    print(f"  Total Return:     {result.total_return:+.2f}%")
    print(f"  Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:     {result.max_drawdown:.2f}%")
    print(f"  Win Rate:         {result.win_rate:.1f}%")
    print(f"  Profit Factor:    {result.profit_factor:.2f}")
    print(f"  Total Trades:     {result.total_trades}")
    print(f"  Avg Trade P&L:    ${result.avg_trade_pnl:.2f}")
    print("="*60)
    
    # Risk-adjusted assessment
    if result.sharpe_ratio > 1.5:
        assessment = "🌟 Excellent"
    elif result.sharpe_ratio > 1.0:
        assessment = "✅ Good"
    elif result.sharpe_ratio > 0.5:
        assessment = "⚠️ Acceptable"
    else:
        assessment = "❌ Poor"
    
    print(f"  Risk-Adjusted:    {assessment}")
    print("="*60)


if __name__ == "__main__":
    print("\n🚀 Trading Backtest with Asymmetric TD Learning")
    print("-" * 50)
    
    # Generate synthetic data
    print("Generating market data...")
    prices, volumes = generate_synthetic_data(n_days=500, volatility=0.02, trend=0.0003)
    
    # Create environment
    env = TradingEnv(
        prices=prices,
        volumes=volumes,
        window_size=20,
        initial_capital=10000,
        transaction_cost=0.001
    )
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    agent = TradingAgent(
        state_dim=state_dim,
        action_dim=8,
        epsilon_decay=0.999
    )
    
    print(f"State dimension: {state_dim}")
    print(f"Device: {agent.device}")
    
    # Training phase
    print("\n📚 Training (100 episodes)...")
    train_result = backtest(agent, env, episodes=100, train=True, verbose=True)
    
    print_backtest_summary(train_result)
    
    # Testing phase (no exploration)
    agent.epsilon = 0.0
    print("\n🎯 Testing (no exploration)...")
    test_result = backtest(agent, env, episodes=1, train=False, verbose=True)
    
    print_backtest_summary(test_result)
    
    # Generate visualization
    # Buy & hold benchmark
    benchmark = 10000 * prices / prices[0]
    plot_backtest_results(
        test_result,
        benchmark_values=benchmark[:len(test_result.portfolio_values)],
        save_path="backtest_results.png"
    )
    
    print("\n✅ Backtest complete!")
