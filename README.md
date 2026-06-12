# Regime-Adaptive Portfolio Modeling via Hybrid Machine Learning

## Motivation

Traditional portfolios maintain fixed allocations regardless of market conditions.

However, financial markets exhibit distinct regimes characterized by different levels of risk, volatility, and return behavior. During market stress, investors often seek safe-haven assets such as Gold, while during expansionary periods equities tend to outperform.

This project investigates whether identifying latent market regimes and adapting portfolio allocations accordingly can improve risk-adjusted returns relative to traditional static portfolios.

---

## Problem Statement

Can machine learning identify changing market conditions and dynamically allocate between equities and Gold to improve portfolio performance?

To answer this question, I built a complete regime-detection and portfolio-construction pipeline using:

- Hidden Markov Models (HMMs)
- Random Forest Classifiers
- Logistic Regression
- Historical Backtesting

using over 20 years of market data.

---

## Data

Daily market data was downloaded using Yahoo Finance.

Assets used:

| Asset | Ticker |
|---------|---------|
| S&P 500 Index | ^GSPC |
| Gold Futures | GC=F |

Date Range:

2005 – 2025

---

## Project Workflow

### 1. Data Processing

Daily adjusted closing prices were collected and transformed into returns.

```text
Price Data
    ↓
Daily Returns
    ↓
Feature Engineering
```

Example return series:

| Date | Gold Return | S&P500 Return |
|--------|--------|--------|
| 2005-01-04 | -0.00047 | -0.01167 |
| 2005-01-05 | -0.00443 | -0.00363 |
| 2005-01-06 | -0.01313 | 0.00351 |

---

### 2. Feature Engineering

To capture market dynamics, multiple momentum, trend and volatility indicators were constructed.

Features:

```text
spx_ret_1d
spx_ret_5d
spx_ret_20d
gold_ret_1d
gold_ret_5d
vol_5d
vol_20d
ma20_ratio
ma50_ratio
```

These features capture:

- Short-term momentum
- Medium-term momentum
- Market volatility
- Trend persistence
- Equity-Gold interaction

---

### 3. Hidden Markov Model

A 2-state Gaussian Hidden Markov Model was trained on market returns.

The model automatically inferred:

- Bull Market Regime
- Bear Market Regime

without requiring any manual labeling.

The inferred regime determines portfolio allocation.

---

### 4. Regime-Based Portfolio Construction

Monthly portfolio weights were adjusted according to the detected market state.

#### Bull Regime

```text
80% S&P500
20% Gold
```

#### Bear Regime

```text
20% S&P500
80% Gold
```

This allows the portfolio to reduce equity exposure during periods of elevated market risk.

---

### 5. Supervised Regime Prediction

The HMM-generated regimes were treated as pseudo-labels.

A next-period prediction task was formulated:

```python
target = regime.shift(-1)
```

Two supervised learning models were trained:

- Random Forest
- Logistic Regression

using engineered market features.

This enables forecasting future market regimes rather than repeatedly fitting an HMM.

---

## Results

### Baseline Portfolio Performance

Comparison of a static 60/40 Stock-Gold portfolio against Buy-and-Hold equities.

| Metric | Static 60/40 | Buy & Hold |
|----------|----------|----------|
| CAGR | 10.67% | 8.87% |
| Volatility | 13.81% | 19.07% |
| Sharpe Ratio | 0.80 | 0.54 |
| Maximum Drawdown | -35.3% | -56.8% |

Key observation:

The addition of Gold significantly reduced drawdowns and volatility while improving long-term returns.

---

### HMM Regime Strategy

| Metric | HMM Regime |
|----------|----------|
| CAGR | 12.40% |
| Volatility | 13.60% |
| Sharpe Ratio | 0.93 |
| Maximum Drawdown | -29.7% |

Relative to the static 60/40 benchmark:

- CAGR improved by approximately 16%
- Sharpe Ratio improved by approximately 15%
- Maximum Drawdown reduced by approximately 16%

The regime-aware strategy achieved higher returns without increasing portfolio risk.

---

### Regime Prediction Performance

#### Random Forest

| Metric | Score |
|----------|----------|
| Accuracy | 89.18% |
| F1 Score | 92.30% |

#### Logistic Regression

| Metric | Score |
|----------|----------|
| Accuracy | 71.35% |
| F1 Score | 82.69% |

The Random Forest substantially outperformed Logistic Regression, indicating that nonlinear relationships are important for regime identification.

---

### Portfolio Growth

Final cumulative wealth growth:

| Strategy | Approximate Growth |
|----------|----------|
| Buy & Hold Stocks | 6.3× |
| Static 60/40 | 9.2× |
| HMM Regime Strategy | 12.5× |

The regime-aware strategy delivered the strongest long-term performance while maintaining lower drawdowns.

---

## Tech Stack

### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Machine Learning

- scikit-learn
- hmmlearn

### Financial Data

- yfinance

### Visualization

- Matplotlib

---

## Key Takeaways

- Hidden Markov Models can effectively identify latent market regimes.
- Dynamic allocation between equities and Gold improves risk-adjusted performance.
- Random Forest models successfully learn HMM-generated regime structure with 89% classification accuracy.
- Regime-aware investing outperformed both Buy-and-Hold equities and a traditional 60/40 allocation over the backtest period.

---

## Author

Aanvi Jha  
B.Tech, Computer Science and Engineering  
Finance Club, IIT Roorkee
