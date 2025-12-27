"""
Extended KLCI Dataset Generator
===============================
Generates 3+ years of realistic KLCI data based on actual market patterns.
Uses real statistical properties from historical data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


def generate_extended_klci(
    start_date: str = "2021-01-01",
    end_date: str = "2024-12-26",
    initial_price: float = 1500.0,
    annual_return: float = 0.05,
    annual_volatility: float = 0.15,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic KLCI data with proper statistical properties.
    
    Based on actual KLCI characteristics:
    - Average annual return: ~5-8%
    - Annual volatility: ~12-18%
    - Occasional crashes: -15% to -30%
    - Recovery patterns
    """
    np.random.seed(seed)
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Generate business days
    dates = pd.bdate_range(start, end)
    n_days = len(dates)
    
    print(f"Generating {n_days} trading days...")
    
    # Daily parameters
    daily_return = annual_return / 252
    daily_vol = annual_volatility / np.sqrt(252)
    
    # Generate returns with regime switches
    returns = np.zeros(n_days)
    regime = "normal"
    
    for i in range(n_days):
        # Random regime changes
        if np.random.random() < 0.005:  # 0.5% chance of crash
            regime = "crash"
        elif np.random.random() < 0.02:  # 2% chance of rally
            regime = "rally"
        elif np.random.random() < 0.05:  # 5% chance of returning to normal
            regime = "normal"
        
        if regime == "crash":
            returns[i] = np.random.normal(-0.01, daily_vol * 2)
        elif regime == "rally":
            returns[i] = np.random.normal(0.005, daily_vol * 0.8)
        else:
            returns[i] = np.random.normal(daily_return, daily_vol)
    
    # Generate prices
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC
    highs = prices * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
    opens = np.roll(prices, 1)
    opens[0] = initial_price
    
    # Add some intraday variation
    opens = opens + np.random.normal(0, prices * 0.002)
    
    # Ensure OHLC consistency
    highs = np.maximum(highs, np.maximum(opens, prices))
    lows = np.minimum(lows, np.minimum(opens, prices))
    
    # Generate volume (correlated with volatility)
    base_volume = 200_000_000
    volume = base_volume * (1 + np.abs(returns) * 10) * np.random.uniform(0.8, 1.2, n_days)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Volume': volume.astype(int)
    })
    
    return df


def get_market_statistics(df: pd.DataFrame) -> dict:
    """Calculate key market statistics."""
    returns = df['Close'].pct_change().dropna()
    
    return {
        'trading_days': len(df),
        'start_date': str(df['Date'].iloc[0].date()),
        'end_date': str(df['Date'].iloc[-1].date()),
        'start_price': df['Close'].iloc[0],
        'end_price': df['Close'].iloc[-1],
        'total_return': (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100,
        'annual_return': returns.mean() * 252 * 100,
        'annual_volatility': returns.std() * np.sqrt(252) * 100,
        'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
        'max_drawdown': calculate_max_drawdown(df['Close'].values),
        'best_day': returns.max() * 100,
        'worst_day': returns.min() * 100
    }


def calculate_max_drawdown(prices):
    peak = np.maximum.accumulate(prices)
    drawdown = (peak - prices) / peak * 100
    return drawdown.max()


if __name__ == "__main__":
    print("="*60)
    print("  EXTENDED KLCI DATA GENERATOR")
    print("="*60)
    
    # Generate 3+ years of data
    df = generate_extended_klci(
        start_date="2021-01-01",
        end_date="2024-12-26",
        initial_price=1520.0,
        annual_return=0.06,
        annual_volatility=0.14
    )
    
    # Save
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/klci_extended_3years.csv', index=False)
    
    # Statistics
    stats = get_market_statistics(df)
    
    print(f"\n  Period: {stats['start_date']} to {stats['end_date']}")
    print(f"  Trading Days: {stats['trading_days']}")
    print(f"  Price: RM {stats['start_price']:.2f} -> RM {stats['end_price']:.2f}")
    print(f"  Total Return: {stats['total_return']:+.2f}%")
    print(f"  Annual Return: {stats['annual_return']:+.2f}%")
    print(f"  Volatility: {stats['annual_volatility']:.2f}%")
    print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {stats['max_drawdown']:.2f}%")
    print(f"  Best Day: {stats['best_day']:+.2f}%")
    print(f"  Worst Day: {stats['worst_day']:+.2f}%")
    print("="*60)
    print(f"\n✓ Saved to: data/klci_extended_3years.csv")
