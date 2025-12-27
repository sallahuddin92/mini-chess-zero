"""
Malaysia Market Backtest with Asymmetric TD Learning
=====================================================
Real KLCI data backtest - No bias, no assumptions
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_env import TradingEnv
from trading_agent import TradingAgent


def run_malaysia_backtest():
    print("\n" + "="*60)
    print("  🇲🇾 MALAYSIA KLCI BACKTEST")
    print("  Asymmetric TD Learning on Real Market Data")
    print("="*60)
    
    # Load real KLCI data
    df = pd.read_csv('data/klci_real_2024.csv')
    prices = df['Close'].values.astype(np.float32)
    volumes = df['Volume'].values.astype(np.float32)
    
    print(f"\n  Data: KLCI Jan-Jun 2024")
    print(f"  Days: {len(prices)}")
    print(f"  Price: RM {prices[0]:.2f} → RM {prices[-1]:.2f}")
    print(f"  Market Return: {(prices[-1]/prices[0]-1)*100:+.2f}%")
    
    # Split 80/20
    split = int(len(prices) * 0.8)
    train_prices, train_volumes = prices[:split], volumes[:split]
    test_prices, test_volumes = prices[split:], volumes[split:]
    
    print(f"\n  Train: {len(train_prices)} days")
    print(f"  Test:  {len(test_prices)} days")
    
    # Create environment
    train_env = TradingEnv(
        prices=train_prices,
        volumes=train_volumes,
        window_size=10,
        initial_capital=10000,
        transaction_cost=0.001
    )
    
    # Create agent
    state_dim = train_env.observation_space.shape[0]
    agent = TradingAgent(
        state_dim=state_dim,
        action_dim=8,
        lr=0.0003,
        epsilon_decay=0.995
    )
    
    print(f"\n📚 Training on KLCI data...")
    
    # Training
    for episode in range(50):
        state, _ = train_env.reset()
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, term, trunc, info = train_env.step(action)
            done = term or trunc
            agent.train(state, action, reward, next_state, done, info['portfolio_value'])
            state = next_state
        
        agent.update_epsilon()
        
        if episode % 10 == 0:
            ret = (info['portfolio_value'] / 10000 - 1) * 100
            print(f"  Episode {episode:3d}/50 | Return: {ret:+.2f}%")
    
    # Testing
    print(f"\n🎯 Testing on unseen KLCI data...")
    
    test_env = TradingEnv(
        prices=test_prices,
        volumes=test_volumes,
        window_size=10,
        initial_capital=10000,
        transaction_cost=0.001
    )
    
    agent.epsilon = 0.0  # No exploration
    state, _ = test_env.reset()
    done = False
    
    while not done:
        action = agent.act(state)
        next_state, reward, term, trunc, info = test_env.step(action)
        done = term or trunc
        state = next_state
    
    # Results
    final_value = info['portfolio_value']
    agent_return = (final_value / 10000 - 1) * 100
    market_return = (test_prices[-1] / test_prices[0] - 1) * 100
    
    print("\n" + "="*60)
    print("  BACKTEST RESULTS")
    print("="*60)
    print(f"  Initial Capital: RM 10,000")
    print(f"  Final Value:     RM {final_value:,.2f}")
    print(f"  Agent Return:    {agent_return:+.2f}%")
    print(f"  Market Return:   {market_return:+.2f}%")
    print(f"  Alpha:           {agent_return - market_return:+.2f}%")
    print(f"  Total Trades:    {len(test_env.trades)}")
    print("="*60)
    
    if agent_return > market_return:
        print("  ✅ Agent OUTPERFORMED the market!")
    else:
        print("  ⚠️ Agent underperformed the market")
    print("="*60)
    
    return agent_return, market_return


if __name__ == "__main__":
    run_malaysia_backtest()
