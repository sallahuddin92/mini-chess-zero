"""
Live Trading Connector
======================
Connects to free brokers for paper/live trading.
Supports: Alpaca (US stocks), Binance (Crypto)
"""

import os
import time
import json
from datetime import datetime
from typing import Dict, Optional
from abc import ABC, abstractmethod


class BaseBroker(ABC):
    """Base class for broker connections."""
    
    @abstractmethod
    def get_balance(self) -> float:
        pass
    
    @abstractmethod
    def get_position(self, symbol: str) -> float:
        pass
    
    @abstractmethod
    def place_order(self, symbol: str, side: str, qty: float) -> Dict:
        pass
    
    @abstractmethod
    def get_price(self, symbol: str) -> float:
        pass


class AlpacaBroker(BaseBroker):
    """
    Alpaca Paper Trading (FREE)
    Sign up: https://alpaca.markets
    
    Features:
    - Free paper trading
    - US stocks + crypto
    - Real-time data
    """
    
    def __init__(self, api_key: str = None, secret_key: str = None, paper: bool = True):
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        self.paper = paper
        
        if paper:
            self.base_url = 'https://paper-api.alpaca.markets'
        else:
            self.base_url = 'https://api.alpaca.markets'
        
        self.headers = {
            'APCA-API-KEY-ID': self.api_key or '',
            'APCA-API-SECRET-KEY': self.secret_key or ''
        }
    
    def _request(self, method: str, endpoint: str, data: Dict = None):
        import requests
        url = f"{self.base_url}/v2{endpoint}"
        resp = requests.request(method, url, headers=self.headers, json=data)
        return resp.json()
    
    def get_balance(self) -> float:
        account = self._request('GET', '/account')
        return float(account.get('cash', 0))
    
    def get_position(self, symbol: str) -> float:
        try:
            pos = self._request('GET', f'/positions/{symbol}')
            return float(pos.get('qty', 0))
        except:
            return 0.0
    
    def place_order(self, symbol: str, side: str, qty: float) -> Dict:
        order = {
            'symbol': symbol,
            'qty': qty,
            'side': side,
            'type': 'market',
            'time_in_force': 'gtc'
        }
        return self._request('POST', '/orders', order)
    
    def get_price(self, symbol: str) -> float:
        # Use latest trade
        import requests
        url = f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest"
        resp = requests.get(url, headers=self.headers)
        data = resp.json()
        return float(data.get('trade', {}).get('p', 0))


class BinanceBroker(BaseBroker):
    """
    Binance Paper Trading (FREE)
    Sign up: https://testnet.binance.vision
    
    Features:
    - Free testnet
    - Crypto only
    - Real-time prices
    """
    
    def __init__(self, api_key: str = None, secret_key: str = None, testnet: bool = True):
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.secret_key = secret_key or os.getenv('BINANCE_SECRET_KEY')
        
        if testnet:
            self.base_url = 'https://testnet.binance.vision'
        else:
            self.base_url = 'https://api.binance.com'
    
    def get_balance(self) -> float:
        # Simplified - full implementation needs HMAC signing
        return 10000.0  # Testnet default
    
    def get_position(self, symbol: str) -> float:
        return 0.0
    
    def place_order(self, symbol: str, side: str, qty: float) -> Dict:
        return {'status': 'simulated', 'symbol': symbol, 'side': side, 'qty': qty}
    
    def get_price(self, symbol: str) -> float:
        import requests
        url = f"{self.base_url}/api/v3/ticker/price?symbol={symbol}"
        resp = requests.get(url)
        data = resp.json()
        return float(data.get('price', 0))


class SimulatedBroker(BaseBroker):
    """
    Local Simulator (No API needed)
    
    Features:
    - Works offline
    - Simulates fills
    - Good for testing
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        self.balance = initial_balance
        self.positions = {}
        self.prices = {}
        self.orders = []
    
    def set_price(self, symbol: str, price: float):
        self.prices[symbol] = price
    
    def get_balance(self) -> float:
        return self.balance
    
    def get_position(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)
    
    def place_order(self, symbol: str, side: str, qty: float) -> Dict:
        price = self.prices.get(symbol, 100.0)
        value = price * qty
        
        if side == 'buy':
            if self.balance >= value:
                self.balance -= value
                self.positions[symbol] = self.positions.get(symbol, 0) + qty
                status = 'filled'
            else:
                status = 'rejected'
        else:  # sell
            if self.positions.get(symbol, 0) >= qty:
                self.balance += value
                self.positions[symbol] -= qty
                status = 'filled'
            else:
                status = 'rejected'
        
        order = {
            'id': len(self.orders),
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'price': price,
            'status': status,
            'time': datetime.now().isoformat()
        }
        self.orders.append(order)
        return order
    
    def get_price(self, symbol: str) -> float:
        return self.prices.get(symbol, 100.0)
    
    def get_portfolio_value(self) -> float:
        value = self.balance
        for symbol, qty in self.positions.items():
            value += qty * self.prices.get(symbol, 0)
        return value


class LiveTrader:
    """
    Live trading manager with ATD agent.
    """
    
    def __init__(self, broker: BaseBroker, agent, symbol: str = 'SPY'):
        self.broker = broker
        self.agent = agent
        self.symbol = symbol
        self.running = False
        self.trade_log = []
    
    def map_action_to_trade(self, action: int, current_price: float):
        """Map agent action to trade order."""
        balance = self.broker.get_balance()
        position = self.broker.get_position(self.symbol)
        
        # Actions: 0=hold, 1-3=buy, 4-6=sell, 7=close
        if action == 0:
            return None
        elif action in [1, 2, 3]:
            # Buy: 10%, 25%, 50% of balance
            pcts = {1: 0.1, 2: 0.25, 3: 0.5}
            amount = (balance * pcts[action]) / current_price
            if amount > 0:
                return ('buy', amount)
        elif action in [4, 5, 6]:
            # Sell: 10%, 25%, 50% of position
            pcts = {4: 0.1, 5: 0.25, 6: 0.5}
            amount = position * pcts[action]
            if amount > 0:
                return ('sell', amount)
        elif action == 7:
            # Close all
            if position > 0:
                return ('sell', position)
        
        return None
    
    def run_once(self, state):
        """Execute one trading step."""
        price = self.broker.get_price(self.symbol)
        action = self.agent.act(state)
        
        trade = self.map_action_to_trade(action, price)
        
        if trade:
            side, qty = trade
            order = self.broker.place_order(self.symbol, side, qty)
            self.trade_log.append(order)
            print(f"  {side.upper()} {qty:.4f} {self.symbol} @ ${price:.2f}")
            return order
        
        return None
    
    def get_stats(self):
        """Get trading statistics."""
        return {
            'balance': self.broker.get_balance(),
            'position': self.broker.get_position(self.symbol),
            'trades': len(self.trade_log),
            'portfolio': self.broker.get_portfolio_value() if hasattr(self.broker, 'get_portfolio_value') else None
        }


# Quick test
if __name__ == "__main__":
    print("Testing SimulatedBroker...")
    broker = SimulatedBroker(10000)
    broker.set_price('TEST', 100)
    
    print(f"Initial balance: ${broker.get_balance():.2f}")
    
    order = broker.place_order('TEST', 'buy', 10)
    print(f"Buy order: {order}")
    
    broker.set_price('TEST', 110)  # Price goes up
    print(f"Portfolio value: ${broker.get_portfolio_value():.2f}")
    
    order = broker.place_order('TEST', 'sell', 10)
    print(f"Sell order: {order}")
    
    print(f"Final balance: ${broker.get_balance():.2f}")
    print(f"Profit: ${broker.get_balance() - 10000:.2f}")
