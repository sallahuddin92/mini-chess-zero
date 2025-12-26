"""
Trading Environment
===================
Gymnasium-compatible environment for trading with Asymmetric TD Learning.
Designed for backtesting stock/crypto strategies.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
from collections import deque


class TradingEnv(gym.Env):
    """
    Trading environment for reinforcement learning.
    
    State:
        - Price history (normalized)
        - Volume history (normalized)
        - Technical indicators (RSI, MACD, etc.)
        - Current position (-1 to 1)
        - Current P&L (normalized)
    
    Actions:
        0: Hold
        1: Buy Small (10% of capital)
        2: Buy Medium (25%)
        3: Buy Large (50%)
        4: Sell Small (10%)
        5: Sell Medium (25%)
        6: Sell Large (50%)
        7: Close All
    
    Rewards:
        - Realized P&L on position changes
        - Unrealized P&L changes
        - Transaction costs
    """
    
    metadata = {"render_modes": ["human"]}
    
    ACTION_NAMES = {
        0: "hold",
        1: "buy_small",
        2: "buy_medium", 
        3: "buy_large",
        4: "sell_small",
        5: "sell_medium",
        6: "sell_large",
        7: "close_all"
    }
    
    def __init__(
        self,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None,
        window_size: int = 20,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,  # 0.1%
        max_position: float = 1.0,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.prices = np.array(prices, dtype=np.float32)
        self.volumes = np.array(volumes, dtype=np.float32) if volumes is not None else np.ones_like(self.prices)
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.render_mode = render_mode
        
        # Precompute technical indicators
        self._compute_indicators()
        
        # State: [price_history, volume_history, indicators, position, pnl]
        # price_history: window_size
        # volume_history: window_size
        # indicators: 5 (RSI, MACD, MACD_signal, BB_upper, BB_lower)
        # position: 1
        # pnl: 1
        state_dim = window_size * 2 + 5 + 2
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(8)
        
        self.reset()
    
    def _compute_indicators(self):
        """Precompute technical indicators."""
        prices = self.prices
        
        # RSI
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        for i in range(1, len(prices)):
            avg_gain[i] = 0.9 * avg_gain[i-1] + 0.1 * gain[i]
            avg_loss[i] = 0.9 * avg_loss[i-1] + 0.1 * loss[i]
        
        rs = np.where(avg_loss > 0, avg_gain / (avg_loss + 1e-10), 100)
        self.rsi = 100 - 100 / (1 + rs)
        
        # MACD
        ema12 = self._ema(prices, 12)
        ema26 = self._ema(prices, 26)
        self.macd = ema12 - ema26
        self.macd_signal = self._ema(self.macd, 9)
        
        # Bollinger Bands
        rolling_mean = np.convolve(prices, np.ones(20)/20, mode='same')
        rolling_std = np.array([prices[max(0,i-20):i+1].std() if i > 0 else 0 for i in range(len(prices))])
        self.bb_upper = rolling_mean + 2 * rolling_std
        self.bb_lower = rolling_mean - 2 * rolling_std
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential moving average."""
        ema = np.zeros_like(data)
        alpha = 2 / (period + 1)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        self.current_step = self.window_size
        self.position = 0.0  # -1 (short) to 1 (long)
        self.capital = self.initial_capital
        self.shares = 0.0
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.trades = []
        
        return self._get_obs(), {}
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        idx = self.current_step
        
        # Normalize prices to percentage change from start of window
        price_window = self.prices[idx-self.window_size:idx]
        price_norm = (price_window - price_window[0]) / (price_window[0] + 1e-10)
        
        # Normalize volume
        vol_window = self.volumes[idx-self.window_size:idx]
        vol_norm = vol_window / (vol_window.mean() + 1e-10) - 1
        
        # Indicators (normalized)
        indicators = np.array([
            self.rsi[idx] / 100 - 0.5,  # Center around 0
            self.macd[idx] / (self.prices[idx] + 1e-10) * 100,
            self.macd_signal[idx] / (self.prices[idx] + 1e-10) * 100,
            (self.bb_upper[idx] - self.prices[idx]) / (self.prices[idx] + 1e-10),
            (self.prices[idx] - self.bb_lower[idx]) / (self.prices[idx] + 1e-10),
        ], dtype=np.float32)
        
        # Position and P&L
        position_pnl = np.array([
            self.position,
            self.total_pnl / self.initial_capital  # Normalized P&L
        ], dtype=np.float32)
        
        return np.concatenate([price_norm, vol_norm, indicators, position_pnl])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute trading action."""
        current_price = self.prices[self.current_step]
        prev_value = self._portfolio_value()
        
        # Execute action
        reward = 0.0
        
        if action == 0:  # Hold
            pass
        elif action == 1:  # Buy Small
            self._adjust_position(0.1)
        elif action == 2:  # Buy Medium
            self._adjust_position(0.25)
        elif action == 3:  # Buy Large
            self._adjust_position(0.5)
        elif action == 4:  # Sell Small
            self._adjust_position(-0.1)
        elif action == 5:  # Sell Medium
            self._adjust_position(-0.25)
        elif action == 6:  # Sell Large
            self._adjust_position(-0.5)
        elif action == 7:  # Close All
            self._close_position()
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward as portfolio change
        new_value = self._portfolio_value()
        reward = (new_value - prev_value) / self.initial_capital
        self.total_pnl = new_value - self.initial_capital
        
        # Check if done
        terminated = self.current_step >= len(self.prices) - 1
        truncated = False
        
        info = {
            "portfolio_value": new_value,
            "position": self.position,
            "total_pnl": self.total_pnl,
            "total_return": (new_value / self.initial_capital - 1) * 100
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        current_price = self.prices[self.current_step]
        return self.capital + self.shares * current_price
    
    def _adjust_position(self, delta: float):
        """Adjust position by delta (fraction of capital)."""
        current_price = self.prices[self.current_step]
        target_position = np.clip(self.position + delta, -self.max_position, self.max_position)
        
        # Calculate shares to buy/sell
        current_value = self._portfolio_value()
        target_shares = (target_position * current_value) / current_price
        shares_delta = target_shares - self.shares
        
        # Execute trade with transaction cost
        trade_value = abs(shares_delta * current_price)
        cost = trade_value * self.transaction_cost
        
        self.shares += shares_delta
        self.capital -= shares_delta * current_price + cost
        self.position = target_position
        
        if shares_delta != 0:
            self.trades.append({
                "step": self.current_step,
                "action": "buy" if shares_delta > 0 else "sell",
                "shares": abs(shares_delta),
                "price": current_price,
                "cost": cost
            })
    
    def _close_position(self):
        """Close entire position."""
        if self.shares != 0:
            current_price = self.prices[self.current_step]
            trade_value = abs(self.shares * current_price)
            cost = trade_value * self.transaction_cost
            
            self.capital += self.shares * current_price - cost
            self.trades.append({
                "step": self.current_step,
                "action": "close",
                "shares": abs(self.shares),
                "price": current_price,
                "cost": cost
            })
            self.shares = 0.0
            self.position = 0.0
    
    def render(self):
        if self.render_mode == "human":
            print(f"Step {self.current_step}: Price={self.prices[self.current_step]:.2f}, "
                  f"Position={self.position:.2f}, Value={self._portfolio_value():.2f}")


def generate_synthetic_data(
    n_days: int = 252,
    initial_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0005
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic price data for testing."""
    np.random.seed(42)
    
    returns = np.random.normal(trend, volatility, n_days)
    prices = initial_price * np.exp(np.cumsum(returns))
    volumes = np.random.lognormal(10, 0.5, n_days)
    
    return prices.astype(np.float32), volumes.astype(np.float32)
