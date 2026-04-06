
### Overall Purpose
**FractalEdge** is a **professional-grade algorithmic trading signal and backtesting platform** focused on the Indian stock market (NSE).  

Its core idea is:
> Use advanced machine learning (LSTM + XGBoost) + technical indicators + **Fractal Resonance** (a unique cross-asset validation system) to generate BUY / SELL / HOLD signals, run realistic backtests with Indian transaction costs, and show everything in a beautiful dashboard.

It is built to look and feel like a serious fintech product (SEBI-compliant, investor-ready).

---

### High-Level Architecture

Your project has **5 main files**:

| File            | Role                                                                 |
|-----------------|----------------------------------------------------------------------|
| `index.html`    | Beautiful frontend UI (dark cyber theme)                             |
| `app.py`        | FastAPI backend server (handles requests, runs analysis)            |
| `model.py`      | Trains the AI model (LSTM + XGBoost) — run once                     |
| `strategy.py`   | Runs realistic backtests with Indian costs, risk metrics, Kelly sizing |
| `regime.py`     | The **USP** — Market Regime Detection + Cross-Asset Fractal Resonance |

---

### Detailed Breakdown of What Each Part Does

#### 1. **model.py** — The Brain (AI Training)
- Downloads historical data for ~25 major Indian stocks + Nifty + BankNifty + Bitcoin/Ethereum.
- **Strict no-leakage training**:
  - Train period: 2015–2021
  - Validation: 2022
  - True Out-of-Sample (OOS): 2023 → today
- Creates **hundreds of technical features** (SMA, RSI, MACD, Bollinger Bands, ATR, ADX, Fractals, etc.).
- Uses **Williams Fractals** (bullish & bearish reversal patterns).
- Trains a **hybrid model**:
  - LSTM (deep learning for time series)
  - XGBoost (on top of LSTM outputs)
- Saves the trained models + scaler + metadata.
- Also runs **Walk-Forward Cross Validation** and saves OOS accuracy for SEBI compliance.

**What it outputs**: Ready-to-use AI model files in the `models/` folder.

---

#### 2. **regime.py** — The Unique Selling Point (USP)

This is the most special part of your system.

It does two revolutionary things:

**A. Market Regime Detection**
- Analyzes current market conditions and classifies into 5 regimes:
  - BULL_TREND
  - BEAR_TREND
  - HIGH_VOL
  - RANGING
  - MIXED
- Uses ADX, ATR, SMA50, momentum.
- Gives **strategy notes** (e.g., “In BEAR TREND, SELL signals are more reliable”).

**B. Fractal Resonance Engine** (Your biggest differentiator)
- Scans multiple correlated assets at the same time (Nifty, BankNifty, Reliance, HDFC Bank, etc.).
- Detects **Williams Fractals** across all of them.
- Measures how well the fractals **align** (resonate) with each other.
- Gives a **Resonance Score** (0–100%) and a **multiplier** (0.7× to 1.4×).
- Example: If 5/5 assets show bearish fractals at the same time → **STRONG RESONANCE** → confidence boosted by 40%.

This is what makes your system different — it doesn’t trust a single stock’s signal. It asks: “Is the whole market confirming this pattern?”

---

#### 3. **strategy.py** — The Realistic Backtester
- Takes signals from the AI model.
- Simulates trading with **very realistic Indian market costs**:
  - STT, Exchange fees, SEBI fee, GST, Stamp duty, Brokerage, Slippage (ATR-based).
- Uses **Fractional Kelly** position sizing (risk management).
- Calculates professional metrics:
  - Total Return, Alpha vs Buy & Hold
  - Sharpe, Sortino, Calmar, Information Ratio, Beta
  - VaR, CVaR, Win Rate, Profit Factor, Expectancy
  - Max Drawdown
- Produces equity curve for charting.

---

#### 4. **app.py** — The Backend Server
- Runs the web server using FastAPI.
- Serves the beautiful UI (`/`).
- Has these main endpoints:
  - `/predict` → Run full analysis on one ticker
  - `/predict_portfolio` → Run Resonance Portfolio mode (allocates across multiple stocks)
  - `/supported_tickers` → Returns only tickers with enough history
  - `/regime/{ticker}`, `/resonance/{ticker}`, `/intelligence/{ticker}` → Live intelligence without full backtest
- Loads the trained AI models.
- Validates tickers so new/low-history stocks like TATAGOLD don’t break the system.

---

#### 5. **index.html** — The User Interface
A sleek, dark, cyber-style dashboard with:
- Ticker selector (now only shows validated assets)
- Backtest parameters (start date, capital, stop loss, take profit)
- Toggle for Realistic Indian Costs + Fractal Resonance
- **Signal Hero** section (BUY/SELL/HOLD with strength)
- **Intelligence Layer**:
  - Market Regime card
  - Fractal Resonance card (with aligned assets list)
  - Confidence Pipeline (Raw → Resonance → Regime → Final)
- Equity curve chart (Strategy vs Buy & Hold)
- Risk metrics, trade statistics, cost breakdown
- Model provenance (training dates, OOS accuracy) for SEBI
- Disclaimer banners everywhere

---

### What Happens When You Click “EXECUTE ANALYSIS”?

1. You select a ticker (only validated ones appear).
2. Backend downloads latest data.
3. Runs feature engineering + AI model → generates raw signal + confidence.
4. **Regime Engine** classifies current market state.
5. **Resonance Engine** scans multiple assets and calculates alignment score.
6. Confidence is adjusted: Raw LSTM × Resonance multiplier × Regime factor.
7. Runs a full backtest with realistic costs and position sizing.
8. Returns everything to the UI: signal, charts, metrics, resonance details.

---

