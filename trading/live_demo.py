"""
Live Trading Demo
=================
Demonstrates the trading system with simulated broker.
"""

import pandas as pd
import numpy as np
from trading_env import TradingEnv
from production_agent import ProductionTradingAgent
from live_broker import SimulatedBroker, LiveTrader


def demo_live_trading():
    print("="*60)
    print("  🚀 LIVE TRADING DEMO")
    print("  SimulatedBroker + Adaptive ATD Agent")
    print("="*60)
    
    # Load data
    df = pd.read_csv('data/klci_real_2024.csv')
    prices = df['Close'].values.astype(np.float32)
    
    # Train agent first
    print("\n📚 Training agent...")
    env = TradingEnv(prices[:100], np.ones(100)*1e8, window_size=5, initial_capital=10000)
    agent = ProductionTradingAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=8,
        lr=0.01
    )
    agent.criterion.pos_weight = 1.5
    agent.criterion.neg_weight = 0.5
    
    for ep in range(100):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            ns, r, t, tr, info = env.step(action)
            done = t or tr
            agent.train(state, action, r, ns, done, info['portfolio_value'])
            state = ns
        agent.update_epsilon()
    
    print("  ✓ Agent trained")
    
    # Setup broker
    broker = SimulatedBroker(initial_balance=10000)
    trader = LiveTrader(broker, agent, symbol='KLCI')
    
    print(f"\n💰 Initial Balance: RM {broker.get_balance():,.2f}")
    print("\n📈 Simulating live trading...\n")
    
    # Simulate live trading on remaining data
    test_prices = prices[100:]
    
    for i, price in enumerate(test_prices):
        broker.set_price('KLCI', price)
        
        # Create simple state
        if i >= 5:
            recent = test_prices[max(0,i-20):i+1]
            state = np.zeros(env.observation_space.shape[0])
            norm = (recent - recent[0]) / (recent[0] + 1e-10)
            state[:len(norm)] = norm[:env.observation_space.shape[0]]
            
            order = trader.run_once(state)
            
            if i % 5 == 0:
                stats = trader.get_stats()
                pv = broker.get_portfolio_value()
                ret = (pv / 10000 - 1) * 100
                print(f"  Day {i:3d} | Price: RM {price:.2f} | Portfolio: RM {pv:,.2f} ({ret:+.2f}%)")
    
    # Final stats
    final_value = broker.get_portfolio_value()
    final_ret = (final_value / 10000 - 1) * 100
    market_ret = (test_prices[-1] / test_prices[0] - 1) * 100
    
    print("\n" + "="*60)
    print("  FINAL RESULTS")
    print("="*60)
    print(f"  Final Portfolio: RM {final_value:,.2f}")
    print(f"  Agent Return:    {final_ret:+.2f}%")
    print(f"  Market Return:   {market_ret:+.2f}%")
    print(f"  Alpha:           {final_ret - market_ret:+.2f}%")
    print(f"  Total Trades:    {len(trader.trade_log)}")
    print("="*60)


if __name__ == "__main__":
    demo_live_trading()
