"""
Fast Stock Predictor - Universal Model
=======================================
Pre-trained on KLCI, instant predictions for any stock.
"""

import numpy as np
import pandas as pd
import torch
import os
from production_agent import ProductionTradingAgent


class FastPredictor:
    """Instant stock predictions using pre-trained universal model."""
    
    MODEL_PATH = "models/universal_predictor.pth"
    
    def __init__(self):
        self.agent = ProductionTradingAgent(state_dim=20, action_dim=3, lr=0.01)
        self.agent.criterion.pos_weight = 1.5
        self.agent.criterion.neg_weight = 0.5
        
        if os.path.exists(self.MODEL_PATH):
            self.agent.load(self.MODEL_PATH)
            print("✓ Loaded pre-trained model")
        else:
            print("⚠ No model found. Run train_universal() first.")
    
    def train_universal(self, klci_csv: str):
        """Train once on KLCI, use for all stocks."""
        print("Training universal model on KLCI...")
        
        df = pd.read_csv(klci_csv)
        df['Close'] = df['Price'].str.replace(',', '').astype(float)
        prices = df['Close'].values[::-1].astype(np.float32)
        
        # Quality training (50 epochs)
        for ep in range(50):
            for i in range(20, len(prices)-1):
                state = self._make_state(prices, i)
                reward = (prices[i+1] - prices[i]) / prices[i]
                next_state = self._make_state(prices, i+1)
                self.agent.train(state, self.agent.act(state), reward, next_state, False, 10000)
            self.agent.update_epsilon()
            if ep % 10 == 0:
                print(f"  Epoch {ep+1}/50")
        
        # Save
        os.makedirs("models", exist_ok=True)
        self.agent.save(self.MODEL_PATH)
        self.agent.epsilon = 0
        print(f"✓ Saved to {self.MODEL_PATH}")
    
    def train_multi_stock(self, folder: str = "training_stocks", epochs: int = 50):
        """
        Train on multiple stocks for better generalization.
        
        Args:
            folder: Folder containing stock CSV files
            epochs: Training epochs per stock
        """
        import glob
        
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        
        if not csv_files:
            print(f"⚠ No CSV files found in {folder}/")
            print("  Add stock CSVs from Investing.com")
            return
        
        print(f"Training on {len(csv_files)} stocks ({epochs} epochs each)...")
        
        all_prices = []
        
        # Load all stock data
        for csv_file in csv_files:
            name = os.path.basename(csv_file).replace('.csv', '').split()[0]
            try:
                df = pd.read_csv(csv_file)
                if 'Price' in df.columns:
                    df['Close'] = df['Price'].astype(str).str.replace(',', '').astype(float)
                else:
                    df['Close'] = df['Close'].astype(str).str.replace(',', '').astype(float)
                prices = df['Close'].dropna().values[::-1].astype(np.float32)
                if len(prices) > 100:
                    all_prices.append((name, prices))
                    print(f"  ✓ {name}: {len(prices)} days")
            except Exception as e:
                print(f"  ✗ {name}: {e}")
        
        if not all_prices:
            print("No valid stock data found")
            return
        
        # Train on all stocks
        print(f"\nTraining universal model on {len(all_prices)} stocks...")
        
        for ep in range(epochs):
            for name, prices in all_prices:
                for i in range(20, len(prices)-1):
                    state = self._make_state(prices, i)
                    reward = (prices[i+1] - prices[i]) / prices[i]
                    next_state = self._make_state(prices, i+1)
                    self.agent.train(state, self.agent.act(state), reward, next_state, False, 10000)
            self.agent.update_epsilon()
            if ep % 10 == 0:
                print(f"  Epoch {ep+1}/{epochs}")
        
        # Save
        os.makedirs("models", exist_ok=True)
        self.agent.save(self.MODEL_PATH)
        self.agent.epsilon = 0
        print(f"\n✓ Model trained on {len(all_prices)} stocks")
        print(f"✓ Saved to {self.MODEL_PATH}")
    
    def _make_state(self, prices: np.ndarray, idx: int) -> np.ndarray:
        """Create normalized state vector."""
        if idx < 20:
            return np.zeros(20, dtype=np.float32)
        window = prices[idx-20:idx]
        return ((window - window.mean()) / (window.std() + 1e-10)).astype(np.float32)
    
    def predict(self, prices: np.ndarray) -> dict:
        """Instant prediction for any stock."""
        state = self._make_state(prices, len(prices)-1)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
        with torch.no_grad():
            q_values = self.agent.policy_net(state_t).cpu().numpy()[0]
        
        exp_q = np.exp(q_values - q_values.max())
        probs = exp_q / exp_q.sum()
        
        action = q_values.argmax()
        signals = {0: "HOLD", 1: "BUY", 2: "SELL"}
        
        # Calculate momentum
        if len(prices) >= 5:
            momentum = float((prices[-1] / prices[-5] - 1) * 100)
        else:
            momentum = 0.0
        
        return {
            "signal": signals[int(action)],
            "confidence": float(probs[action]),
            "price": float(prices[-1]),
            "momentum_5d": float(momentum)
        }
    
    def scan_stocks(self, stock_files: list) -> pd.DataFrame:
        """Scan multiple stocks instantly."""
        results = []
        
        for filepath in stock_files:
            name = os.path.basename(filepath).replace('.csv', '').split()[0]
            
            df = pd.read_csv(filepath)
            if 'Price' in df.columns:
                df['Close'] = df['Price'].str.replace(',', '').astype(float)
            prices = df['Close'].values[::-1].astype(np.float32)
            
            pred = self.predict(prices)
            results.append({
                "Stock": name,
                "Signal": pred["signal"],
                "Confidence": f"{pred['confidence']:.0%}",
                "Price": f"RM {pred['price']:.2f}",
                "5D Mom": f"{pred['momentum_5d']:+.1f}%"
            })
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    import sys
    
    predictor = FastPredictor()
    
    # Train if needed
    if not os.path.exists(FastPredictor.MODEL_PATH):
        predictor.train_universal("FTSE Malaysia KLCI Historical Data-2.csv")
    
    # Test on Inari
    print("\n📊 INARI Prediction:")
    df = pd.read_csv("Inari Amertron Stock Price History-2.csv")
    prices = df['Price'].astype(float).values[::-1].astype(np.float32)
    
    result = predictor.predict(prices)
    print(f"   Signal: {result['signal']}")
    print(f"   Confidence: {result['confidence']:.0%}")
    print(f"   Price: RM {result['price']:.2f}")
    print(f"   5D Momentum: {result['momentum_5d']:+.1f}%")
