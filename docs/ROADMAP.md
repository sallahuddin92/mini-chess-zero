# Mini-Chess Zero: Strategic Roadmap

## Project Vision

**Mini-Chess Zero** demonstrates a novel **Asymmetric Temporal Difference Learning** technique that solved critical training instability in deep reinforcement learning. The core innovation—weighting TD errors asymmetrically based on sign—has broad applications beyond chess.

**Core Achievement**: 87.5% win rate (up from 3%) with Q-values stabilized at 9.2 (down from 70 million).

---

# Part 1: Research Publication

## 1.1 Original Research Thesis

**Title**: "Asymmetric Temporal Difference Learning: Biologically-Inspired Stabilization for Deep Reinforcement Learning"

**Novel Contributions**:
1. Asymmetric TD weighting inspired by dopamine response asymmetry
2. Combined stabilization stack for value-based RL
3. Empirical validation in adversarial game domain

## 1.2 Methodology

### Experimental Design
```
Baseline: Standard DDQN
Treatment: DDQN + ATD + Stabilization Stack
Domain: 5×5 Gardner Mini-Chess
Metrics: Win rate, Q-value stability, loss convergence
Games: 1000 self-play episodes
```

### Ablation Studies Required
| Experiment | Remove | Measure Impact |
|------------|--------|----------------|
| A1 | Asymmetric TD only | Q-value explosion? |
| A2 | Gradient clipping only | Training stability? |
| A3 | Soft updates only | Policy oscillation? |
| A4 | All stabilization | Full baseline comparison |

## 1.3 Paper Structure

```
1. Abstract (150 words)
2. Introduction
   - RL instability problem
   - Biological motivation
3. Related Work
   - DDQN, gradient clipping, Polyak averaging
4. Method: Asymmetric TD Learning
   - Mathematical formulation
   - Integration with DQN
5. Experiments
   - Mini-Chess domain
   - Ablation studies
   - Comparison to baselines
6. Results
   - Training curves
   - Statistical significance
7. Discussion
   - Limitations
   - Future work
8. Conclusion
```

## 1.4 Deliverables Checklist

- [ ] Run ablation experiments (4 conditions × 3 seeds)
- [ ] Statistical significance tests (paired t-test)
- [ ] Generate publication-quality figures
- [ ] Write paper draft (LaTeX, 8 pages)
- [ ] Submit to arXiv
- [ ] Target venue: NeurIPS Workshop / AAAI / IJCAI

---

# Part 2: Python Package

## 2.1 Package Design

**Name**: `stable-baselines-atd` or `asymmetric-td`

```
asymmetric_td/
├── __init__.py
├── losses/
│   ├── __init__.py
│   ├── asymmetric_td.py      # Core ATD loss
│   └── huber_atd.py          # Huber variant
├── agents/
│   ├── __init__.py
│   ├── dqn_stable.py         # Stabilized DQN
│   └── ddqn_stable.py        # Stabilized DDQN
├── utils/
│   ├── reward_centering.py
│   ├── gradient_utils.py
│   └── soft_update.py
├── envs/
│   └── mini_chess.py         # Example environment
└── examples/
    ├── cartpole.py
    ├── mini_chess.py
    └── custom_env.py
```

## 2.2 API Design

```python
# Installation
# pip install asymmetric-td

from asymmetric_td import StableDQN, AsymmetricTDLoss
from asymmetric_td.utils import RewardCentering

# Create stabilized agent
agent = StableDQN(
    state_dim=25,
    action_dim=625,
    gamma=0.95,
    tau=0.005,                    # Soft update rate
    grad_clip=10.0,               # Gradient clipping
    q_clip=(-10, 10),             # Q-value bounds
    atd_weights=(0.5, 1.5),       # (positive, negative) TD weights
    reward_centering=True
)

# Train
for episode in range(1000):
    state = env.reset()
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        loss = agent.train(state, action, reward, next_state, done)
        state = next_state
```

## 2.3 Dependencies

```
torch>=1.9.0
numpy>=1.19.0
gymnasium>=0.26.0  # Optional for env compatibility
```

## 2.4 Deliverables Checklist

- [ ] Create package structure
- [ ] Implement core modules
- [ ] Write unit tests (pytest)
- [ ] Create documentation (mkdocs)
- [ ] Set up CI/CD (GitHub Actions)
- [ ] Publish to PyPI
- [ ] Write usage examples

---

# Part 3: Chess App

## 3.1 Architecture

```
┌─────────────────────────────────────────────────┐
│                   Frontend                       │
│  (React/Next.js or React Native for mobile)     │
├─────────────────────────────────────────────────┤
│                   API Layer                      │
│  (FastAPI - REST endpoints for moves)           │
├─────────────────────────────────────────────────┤
│                   AI Engine                      │
│  (PyTorch model + difficulty scaling)           │
├─────────────────────────────────────────────────┤
│                   Game Logic                     │
│  (Mini-Chess rules, state management)           │
└─────────────────────────────────────────────────┘
```

## 3.2 AI Difficulty Scaling

```python
class AdaptiveAI:
    DIFFICULTY_LEVELS = {
        "beginner": {"epsilon": 0.5, "noise": 0.3},
        "intermediate": {"epsilon": 0.2, "noise": 0.1},
        "advanced": {"epsilon": 0.05, "noise": 0.02},
        "expert": {"epsilon": 0.0, "noise": 0.0}
    }
    
    def get_move(self, state, difficulty="intermediate"):
        config = self.DIFFICULTY_LEVELS[difficulty]
        
        # Add controlled randomness for easier levels
        if random.random() < config["epsilon"]:
            return random.choice(legal_moves)
        
        q_values = self.model.predict(state)
        q_values += np.random.normal(0, config["noise"], q_values.shape)
        return argmax(q_values)
```

## 3.3 Unique Features

1. **Q-Value Explanation**: Show why AI chose a move
2. **Move Analysis**: Rate player moves against AI evaluation
3. **Learning Mode**: Suggest better alternatives
4. **Progressive Difficulty**: Auto-adjust based on player performance

## 3.4 Deliverables Checklist

- [ ] FastAPI backend with endpoints
- [ ] React frontend with chessboard
- [ ] AI difficulty scaling system
- [ ] Q-value visualization
- [ ] Mobile-responsive design
- [ ] Deploy to Vercel/Railway

---

# Part 4: Trading Adaptation

## 4.1 Domain Translation

| Chess Concept | Trading Equivalent |
|---------------|-------------------|
| Board state | Portfolio + market indicators |
| Move | Buy/Sell/Hold action |
| Win/Loss | Profit/Loss |
| Piece value | Position size |
| King capture | Stop-loss trigger |

## 4.2 Asymmetric TD for Trading

**Key Insight**: Losses hurt more than equivalent gains (loss aversion)

```python
class TradingATDLoss:
    def __init__(self, gain_weight=0.5, loss_weight=2.0):
        """
        Trading-specific: penalize losses even more heavily
        than in game domains (2.0 vs 1.5)
        """
        self.gain_w = gain_weight
        self.loss_w = loss_weight
    
    def forward(self, predicted_returns, actual_returns):
        td_errors = predicted_returns - actual_returns
        
        # Asymmetric: penalize underestimation of risk
        weights = torch.where(td_errors > 0, self.gain_w, self.loss_w)
        return (weights * td_errors.pow(2)).mean()
```

## 4.3 State Representation

```python
state = {
    "price_history": last_20_prices,      # Normalized
    "volume": last_20_volumes,             # Normalized
    "position": current_holdings,          # -1 to 1
    "pnl": current_profit_loss,           # Normalized
    "indicators": [rsi, macd, bollinger]  # Technical
}
# Flatten to vector: ~100 features
```

## 4.4 Action Space

```python
actions = {
    0: "hold",
    1: "buy_small",   # 10% of capital
    2: "buy_medium",  # 25% of capital
    3: "buy_large",   # 50% of capital
    4: "sell_small",
    5: "sell_medium",
    6: "sell_large",
    7: "close_all"    # Exit position
}
```

## 4.5 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Sharpe Ratio | Risk-adjusted returns |
| Max Drawdown | Worst peak-to-trough loss |
| Win Rate | % profitable trades |
| Profit Factor | Gross profit / Gross loss |
| Sortino Ratio | Downside-risk adjusted returns |

## 4.6 Backtesting Framework

```python
def backtest(agent, historical_data, initial_capital=10000):
    portfolio = Portfolio(initial_capital)
    
    for day in historical_data:
        state = get_state(portfolio, day)
        action = agent.act(state)
        
        execute_trade(portfolio, action, day.price)
        
        reward = calculate_pnl(portfolio)
        agent.train(state, action, reward, next_state, done)
    
    return {
        "final_value": portfolio.value,
        "sharpe": calculate_sharpe(portfolio.returns),
        "max_drawdown": calculate_max_drawdown(portfolio.history)
    }
```

## 4.7 Deliverables Checklist

- [ ] Create TradingEnv gymnasium wrapper
- [ ] Implement price data loader (yfinance)
- [ ] Adapt ATD loss for trading
- [ ] Build backtesting framework
- [ ] Run experiments on SPY/BTC
- [ ] Generate performance reports

---

# Evaluation & Iteration

## Testing Strategy

| Direction | Test Type | Metrics |
|-----------|-----------|---------|
| Research | Ablation | Statistical significance |
| Package | Unit/Integration | Coverage, CI passing |
| Chess App | E2E, Load | Latency, user satisfaction |
| Trading | Backtest | Sharpe, drawdown |

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Overfitting to training data | Cross-validation, holdout sets |
| Q-value instability | Already solved with ATD |
| Trading: look-ahead bias | Strict temporal separation |
| App: slow inference | Model quantization, caching |

---

# Execution Order

**Recommended sequence based on effort vs impact:**

```
Step 1: Python Package (foundation for all others)
    ↓
Step 2: Research Publication (validates the method)
    ↓
Step 3A: Chess App (consumer product)
   OR
Step 3B: Trading Adaptation (B2B/fintech)
```

---

## Next Step

**Which direction would you like to start with?**

1. **Research Publication** - Run ablation experiments
2. **Python Package** - Create package structure
3. **Chess App** - Build FastAPI backend
4. **Trading Adaptation** - Create TradingEnv

I will proceed step-by-step, prompting you before each major milestone.
