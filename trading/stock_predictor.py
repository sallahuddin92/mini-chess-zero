"""
Malaysia Stock Predictor
=========================
Uses ATD learning to predict individual stock movements.
Uses KLCI as market context for better predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from production_agent import ProductionTradingAgent
import os


# Top Malaysia Stocks (download from Investing.com)
MALAYSIA_STOCKS = {
    "MAYBANK": "Malayan Banking Berhad",
    "CIMB": "CIMB Group Holdings",
    "PBBANK": "Public Bank Berhad", 
    "TENAGA": "Tenaga Nasional",
    "PCHEM": "Petronas Chemicals",
    "TM": "Telekom Malaysia",
    "IHH": "IHH Healthcare",
    "SIME": "Sime Darby Plantation",
    "AXIATA": "Axiata Group",
    "MISC": "MISC Berhad"
}


class MalaysiaStockPredictor:
    """
    Predicts individual Malaysia stock movements.
    
    Features:
    - Uses KLCI as market context
    - Trained per-stock agents
    - Outputs buy/sell/hold signals
    """
    
    def __init__(self, klci_data: np.ndarray):
        """
        Initialize with KLCI market data as context.
        
        Args:
            klci_data: KLCI price history
        """
        self.klci = klci_data
        self.agents: Dict[str, ProductionTradingAgent] = {}
        self.predictions: Dict[str, str] = {}
    
    def add_stock(self, symbol: str, prices: np.ndarray):
        """Add a stock and train its agent."""
        print(f"Training agent for {symbol}...")
        
        # Create agent with market-aware features
        # State: [stock_features (10), klci_features (10)]
        agent = ProductionTradingAgent(
            state_dim=20,
            action_dim=3,  # 0=hold, 1=buy, 2=sell
            lr=0.005
        )
        
        # Trend-following for stocks
        agent.criterion.pos_weight = 1.5
        agent.criterion.neg_weight = 0.5
        
        # Train
        split = int(len(prices) * 0.8)
        for ep in range(50):
            for i in range(20, split):
                state = self._create_state(prices, i)
                reward = (prices[i] - prices[i-1]) / prices[i-1]
                next_state = self._create_state(prices, i+1) if i+1 < split else state
                agent.train(state, agent.act(state), reward, next_state, i==split-1, 10000)
            agent.update_epsilon()
        
        agent.epsilon = 0
        self.agents[symbol] = agent
        print(f"  ✓ {symbol} agent trained")
    
    def _create_state(self, prices: np.ndarray, idx: int) -> np.ndarray:
        """Create state with stock + market features."""
        state = np.zeros(20)
        
        # Stock features (normalized returns)
        if idx >= 10:
            stock_window = prices[idx-10:idx]
            state[:10] = (stock_window - stock_window[0]) / (stock_window[0] + 1e-10)
        
        # KLCI features (market context)
        if idx < len(self.klci) and idx >= 10:
            klci_window = self.klci[idx-10:idx]
            state[10:20] = (klci_window - klci_window[0]) / (klci_window[0] + 1e-10)
        
        return state.astype(np.float32)
    
    def predict(self, symbol: str, prices: np.ndarray) -> Tuple[str, float]:
        """
        Predict action for a stock.
        
        Returns:
            (signal, confidence): e.g., ("BUY", 0.75)
        """
        if symbol not in self.agents:
            return "HOLD", 0.0
        
        agent = self.agents[symbol]
        state = self._create_state(prices, len(prices)-1)
        
        # Get Q-values for all actions
        import torch
        state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            q_values = agent.policy_net(state_t).cpu().numpy()[0]
        
        # Softmax for confidence
        exp_q = np.exp(q_values - q_values.max())
        probs = exp_q / exp_q.sum()
        
        action = q_values.argmax()
        confidence = probs[action]
        
        signals = {0: "HOLD", 1: "BUY", 2: "SELL"}
        return signals[action], float(confidence)
    
    def scan_all_stocks(self, stock_data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Scan all stocks and return predictions.
        
        Returns:
            DataFrame with predictions for all stocks
        """
        results = []
        
        for symbol, prices in stock_data.items():
            signal, confidence = self.predict(symbol, prices)
            
            # Calculate additional metrics
            if len(prices) >= 5:
                momentum = (prices[-1] / prices[-5] - 1) * 100
                volatility = np.std(np.diff(prices[-20:]) / prices[-20:-1]) * 100
            else:
                momentum = 0
                volatility = 0
            
            results.append({
                "Symbol": symbol,
                "Company": MALAYSIA_STOCKS.get(symbol, "Unknown"),
                "Signal": signal,
                "Confidence": f"{confidence:.1%}",
                "Price": f"RM {prices[-1]:.2f}",
                "5D Momentum": f"{momentum:+.1f}%",
                "Volatility": f"{volatility:.1f}%"
            })
        
        return pd.DataFrame(results)
    
    def backtest_stock(self, symbol: str, prices: np.ndarray) -> Dict:
        """
        Backtest a single stock.
        
        Returns:
            Performance metrics
        """
        agent = self.agents.get(symbol)
        if not agent:
            return {"error": "No agent for this stock"}
        
        split = int(len(prices) * 0.8)
        test_prices = prices[split:]
        
        capital = 10000.0
        shares = 0.0
        trades = []
        
        for i in range(20, len(test_prices)):
            state = self._create_state(test_prices, i)
            action = agent.act(state)
            
            if action == 1 and capital > 100:  # BUY
                buy_amt = capital * 0.5
                shares += buy_amt / test_prices[i]
                capital -= buy_amt
                trades.append(("BUY", test_prices[i]))
            elif action == 2 and shares > 0:  # SELL
                capital += shares * test_prices[i]
                trades.append(("SELL", test_prices[i]))
                shares = 0
        
        if shares > 0:
            capital += shares * test_prices[-1]
        
        agent_return = (capital / 10000 - 1) * 100
        market_return = (test_prices[-1] / test_prices[0] - 1) * 100
        
        return {
            "symbol": symbol,
            "agent_return": agent_return,
            "market_return": market_return,
            "alpha": agent_return - market_return,
            "trades": len(trades)
        }


def demo():
    """Demo with KLCI data."""
    print("="*60)
    print("  🇲🇾 MALAYSIA STOCK PREDICTOR")
    print("="*60)
    
    # Load KLCI
    print("\nLoading KLCI data...")
    klci_df = pd.read_csv("FTSE Malaysia KLCI Historical Data-2.csv")
    klci_df['Close'] = klci_df['Price'].str.replace(',', '').astype(float)
    klci_prices = klci_df['Close'].values[::-1].astype(np.float32)
    
    print(f"  KLCI: {len(klci_prices)} days")
    
    # Create predictor
    predictor = MalaysiaStockPredictor(klci_prices)
    
    # For demo, we'll use KLCI as a "stock" too
    predictor.add_stock("KLCI", klci_prices)
    
    # Get prediction
    signal, confidence = predictor.predict("KLCI", klci_prices)
    print(f"\n📊 KLCI Prediction: {signal} ({confidence:.1%} confidence)")
    
    # Backtest
    result = predictor.backtest_stock("KLCI", klci_prices)
    print(f"\n📈 Backtest Results:")
    print(f"  Agent Return: {result['agent_return']:+.2f}%")
    print(f"  Market Return: {result['market_return']:+.2f}%")
    print(f"  Alpha: {result['alpha']:+.2f}%")
    print(f"  Trades: {result['trades']}")
    
    print("\n" + "="*60)
    print("  To add individual stocks:")
    print("  1. Download from Investing.com (MAYBANK, CIMB, etc.)")
    print("  2. Call predictor.add_stock('MAYBANK', prices)")
    print("  3. Get signal with predictor.predict('MAYBANK', prices)")
    print("="*60)


if __name__ == "__main__":
    demo()
