"""
Production Backtest - Malaysia KLCI
====================================
Adaptive ATD with regime detection for real trading.
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('..')

from trading_env import TradingEnv
from production_agent import ProductionTradingAgent


def run_production_backtest():
    print("\n" + "="*60)
    print("  🇲🇾 PRODUCTION KLCI BACKTEST")
    print("  Adaptive ATD with Regime Detection")
    print("="*60)
    
    # Load data
    df = pd.read_csv('data/klci_real_2024.csv')
    prices = df['Close'].values.astype(np.float32)
    volumes = df['Volume'].values.astype(np.float32)
    
    print(f"\n  Data: KLCI 2024 ({len(prices)} days)")
    print(f"  Market Return: {(prices[-1]/prices[0]-1)*100:+.2f}%")
    
    # 80/20 split
    split = int(len(prices) * 0.8)
    train_p, train_v = prices[:split], volumes[:split]
    test_p, test_v = prices[split:], volumes[split:]
    
    # Create env and agent
    train_env = TradingEnv(train_p, train_v, window_size=10, initial_capital=10000)
    state_dim = train_env.observation_space.shape[0]
    
    agent = ProductionTradingAgent(
        state_dim=state_dim,
        action_dim=8,
        lr=0.001,
        max_drawdown=0.10  # 10% max drawdown
    )
    
    # Extended training
    print("\n📚 Training (300 episodes)...")
    for ep in range(300):
        state, _ = train_env.reset()
        done = False
        
        while not done:
            action = agent.act(state, train_p[:train_env.current_step+1])
            ns, r, t, tr, info = train_env.step(action)
            done = t or tr
            agent.train(state, action, r, ns, done, info['portfolio_value'])
            state = ns
        
        agent.update_epsilon()
        
        if ep % 50 == 0:
            ret = (info['portfolio_value']/10000-1)*100
            print(f"  Ep {ep:3d}/300 | Return: {ret:+.2f}% | Regime: {agent.criterion.regime}")
    
    # Test
    print("\n🎯 Testing...")
    test_env = TradingEnv(test_p, test_v, window_size=10, initial_capital=10000)
    agent.epsilon = 0
    agent.peak_value = 10000
    
    state, _ = test_env.reset()
    done = False
    
    while not done:
        action = agent.act(state, test_p[:test_env.current_step+1])
        ns, r, t, tr, info = test_env.step(action)
        done = t or tr
        state = ns
    
    # Results
    final = info['portfolio_value']
    agent_ret = (final/10000-1)*100
    market_ret = (test_p[-1]/test_p[0]-1)*100
    
    print("\n" + "="*60)
    print("  PRODUCTION RESULTS")
    print("="*60)
    print(f"  Agent Return:  {agent_ret:+.2f}%")
    print(f"  Market Return: {market_ret:+.2f}%")
    print(f"  Alpha:         {agent_ret - market_ret:+.2f}%")
    print(f"  Final Value:   RM {final:,.2f}")
    print("="*60)
    
    if agent_ret > market_ret:
        print("  ✅ OUTPERFORMED MARKET!")
    
    # Save model
    agent.save('production_model.pth')
    print("  ✓ Model saved to production_model.pth")
    
    return agent_ret, market_ret


if __name__ == "__main__":
    run_production_backtest()
