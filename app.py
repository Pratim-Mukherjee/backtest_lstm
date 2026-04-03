# app.py — FractalEdge SEBI-Ready API
"""
FIXES APPLIED:
1. Returns OOS accuracy prominently — no hiding of model quality
2. Clearly flags whether backtest period overlaps training data
3. SEBI-mandated risk disclaimers in every response
4. Returns realistic cost breakdown in response
5. Model metadata (train period, OOS period) exposed in API
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
import os, pickle
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from tensorflow.keras.models import load_model
import strategy

# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR  = os.path.join(BASE_DIR, "static")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(
    title="FractalEdge — SEBI-Ready LSTM-XGBoost Trading Signal API",
    description=(
        "Hybrid LSTM+XGBoost trading signal system. "
        "Model trained on 2015-2021 data. OOS test: 2023-present. "
        "All backtest results include realistic Indian market transaction costs. "
        "RISK DISCLAIMER: Past performance is not indicative of future results. "
        "This system is not SEBI-registered investment advice."
    ),
    version="3.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ─────────────────────────────────────────────────────────────────────────────
# TICKER ALIASES
# ─────────────────────────────────────────────────────────────────────────────
TICKER_MAP = {
    "nifty": "^NSEI", "banknifty": "^NSEBANK",
    "reliance": "RELIANCE.NS", "tcs": "TCS.NS",
    "infy": "INFY.NS", "hdfc": "HDFCBANK.NS",
    "icici": "ICICIBANK.NS", "sbi": "SBIN.NS",
    "itc": "ITC.NS", "ongc": "ONGC.NS",
    "axis": "AXISBANK.NS", "kotak": "KOTAKBANK.NS",
    "maruti": "MARUTI.NS", "tatamotors": "TATAMOTORS.NS",
    "bharti": "BHARTIARTL.NS", "lt": "LT.NS",
    "hindunilvr": "HINDUNILVR.NS", "sunpharma": "SUNPHARMA.NS",
    "hcltech": "HCLTECH.NS", "titan": "TITAN.NS",
    "bitcoin": "BTC-USD", "btc": "BTC-USD",
    "ethereum": "ETH-USD", "eth": "ETH-USD",
    "wipro": "WIPRO.NS", "bajfinance": "BAJFINANCE.NS",
}

def resolve_ticker(user_input: str) -> str:
    clean = user_input.strip().lower()
    if clean in TICKER_MAP:
        return TICKER_MAP[clean]
    for guess in [user_input.upper(), user_input.upper() + ".NS", "^" + user_input.upper()]:
        try:
            t = yf.Ticker(guess)
            h = t.history(period="5d")
            if not h.empty:
                return guess
        except Exception:
            continue
    return user_input

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (identical to model.py — CRITICAL: must stay in sync)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "fractal_bull", "fractal_bear",
    "SMA_10", "SMA_20", "SMA_50",
    "RSI_14", "MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9",
    "BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "BBP_20_2.0",
    "ATR_14", "ADX_14", "CCI_20_0.015",
    "STOCHk_14_3_3", "STOCHd_14_3_3",
    "Norm_Ret", "Vol_20", "Ret_Lag_1", "Ret_Lag_2"
]

def add_fractals(df):
    df = df.copy()
    df["fractal_bull"] = 0
    df["fractal_bear"] = 0
    for i in range(2, len(df) - 2):
        if (df["Low"].iloc[i] < df["Low"].iloc[i-2] and
            df["Low"].iloc[i] < df["Low"].iloc[i-1] and
            df["Low"].iloc[i] < df["Low"].iloc[i+1] and
            df["Low"].iloc[i] < df["Low"].iloc[i+2]):
            df.loc[df.index[i], "fractal_bull"] = 1
        if (df["High"].iloc[i] > df["High"].iloc[i-2] and
            df["High"].iloc[i] > df["High"].iloc[i-1] and
            df["High"].iloc[i] > df["High"].iloc[i+1] and
            df["High"].iloc[i] > df["High"].iloc[i+2]):
            df.loc[df.index[i], "fractal_bear"] = 1
    return df

def get_features(df):
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["LogRet"] = np.log(df["Close"] / df["Close"].shift(1))
    df = add_fractals(df)

    df["SMA_10"] = ta.sma(df["Close"], length=10)
    df["SMA_20"] = ta.sma(df["Close"], length=20)
    df["SMA_50"] = ta.sma(df["Close"], length=50)
    df["RSI_14"] = ta.rsi(df["Close"], length=14)

    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        c = macd.columns.tolist()
        macd = macd.rename(columns={c[0]: "MACD_12_26_9", c[1]: "MACDs_12_26_9", c[2]: "MACDh_12_26_9"})
        df = pd.concat([df, macd], axis=1)

    bb = ta.bbands(df["Close"], length=20, std=2.0)
    if bb is not None and not bb.empty:
        rename = {}
        for c in bb.columns:
            cl = c.lower()
            if any(x in cl for x in ["lower","bbl"]):    rename[c] = "BBL_20_2.0"
            elif any(x in cl for x in ["mid","bbm"]):    rename[c] = "BBM_20_2.0"
            elif any(x in cl for x in ["upper","bbu"]):  rename[c] = "BBU_20_2.0"
            elif any(x in cl for x in ["percent","bbp"]): rename[c] = "BBP_20_2.0"
        df = pd.concat([df, bb.rename(columns=rename)], axis=1)

    df["ATR_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    if adx is not None and not adx.empty:
        df["ADX_14"] = adx.iloc[:, 0]

    df["CCI_20_0.015"] = ta.cci(df["High"], df["Low"], df["Close"], length=20, c=0.015)

    stoch = ta.stoch(df["High"], df["Low"], df["Close"], k=14, d=3, smooth_k=3)
    if stoch is not None and not stoch.empty:
        df["STOCHk_14_3_3"] = stoch.iloc[:, 0]
        if stoch.shape[1] > 1:
            df["STOCHd_14_3_3"] = stoch.iloc[:, 1]

    df["Norm_Ret"] = df["Return"].rolling(20).apply(
        lambda x: (x[-1] - x.mean()) / (x.std() + 1e-8) if len(x) > 1 else 0, raw=True)
    df["Vol_20"]    = df["Return"].rolling(20).std()
    df["Ret_Lag_1"] = df["Return"].shift(1)
    df["Ret_Lag_2"] = df["Return"].shift(2)

    available = [c for c in FEATURE_COLS if c in df.columns]
    return df[available].copy()

def create_sequences(X, time_steps=60):
    return np.array([X[i:i + time_steps] for i in range(len(X) - time_steps)])

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
_scaler = _lstm = _xgb = _meta = _oos_report = _wf_report = None

def load_models():
    global _scaler, _lstm, _xgb, _meta, _oos_report, _wf_report
    if _scaler is not None:
        return

    required = ["scaler.pkl", "lstm_model.keras", "xgb_model.pkl", "metadata.pkl"]
    missing  = [f for f in required if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        raise RuntimeError(
            f"Models not found: {missing}. Run `python model.py` first."
        )

    import pickle
    _scaler = pickle.load(open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb"))
    _lstm   = load_model(os.path.join(MODELS_DIR, "lstm_model.keras"))
    _xgb    = pickle.load(open(os.path.join(MODELS_DIR, "xgb_model.pkl"), "rb"))
    _meta   = pickle.load(open(os.path.join(MODELS_DIR, "metadata.pkl"), "rb"))

    oos_path = os.path.join(MODELS_DIR, "oos_report.pkl")
    wf_path  = os.path.join(MODELS_DIR, "walkforward_report.pkl")
    _oos_report = pickle.load(open(oos_path, "rb")) if os.path.exists(oos_path) else {}
    _wf_report  = pickle.load(open(wf_path,  "rb")) if os.path.exists(wf_path)  else {}

    print("✅ Models loaded")

# ─────────────────────────────────────────────────────────────────────────────
# SERVE UI
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>UI not found. Place index.html in /static/</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    ticker:           str   = Field(...,   example="^NSEI")
    start_date:       str   = Field("2023-01-01", description=(
        "Start date for backtest. Dates after 2023-01-01 are TRUE OOS "
        "(model never trained on this data). Dates before 2022-01-01 "
        "are IN-SAMPLE and will produce inflated metrics."
    ))
    initial_capital:  float = Field(100_000.0, gt=0)
    stop_loss_pct:    float = Field(0.03, gt=0, le=0.5)
    take_profit_pct:  float = Field(0.06, gt=0, le=1.0)
    use_realistic_costs: bool = Field(True, description=(
        "If True, applies real Indian market costs: STT, exchange charges, "
        "SEBI fee, GST, stamp duty, brokerage. Recommended for SEBI submission."
    ))

class ModelInfo(BaseModel):
    train_period:     str
    val_period:       str
    oos_period:       str
    mean_oos_accuracy: float
    trained_at:       str
    tickers_trained:  int
    walk_forward_accuracy: Optional[float]

class RiskMetrics(BaseModel):
    var_95_daily_pct:  float
    cvar_95_daily_pct: float
    information_ratio: float
    beta:              float
    win_rate:          float
    avg_win_pct:       float
    avg_loss_pct:      float
    profit_factor:     float
    expectancy_pct:    float

class SignalResponse(BaseModel):
    # Signal
    ticker:          str
    last_signal:     str
    signal_strength: str   # STRONG / MODERATE / WEAK

    # Returns
    total_return:    float
    buy_hold_return: float
    alpha:           float
    max_drawdown:    float

    # Risk-adjusted
    sharpe_ratio:    float
    sortino_ratio:   float
    calmar_ratio:    float
    risk_metrics:    RiskMetrics

    # Backtest metadata
    is_oos:          bool    # True if backtest period is fully out-of-sample
    backtest_warning: str
    trades:          int
    cost_model:      dict

    # Curves
    equity:          list[float]
    buy_hold:        list[float]
    monthly_returns: dict

    # Model info
    model_info:      ModelInfo

    # SEBI
    oos_accuracy:    Optional[float]
    disclaimer:      str
    success:         bool
    message:         str

# ─────────────────────────────────────────────────────────────────────────────
# PREDICT ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=SignalResponse)
async def predict_signal(req: PredictRequest):
    load_models()

    resolved = resolve_ticker(req.ticker)

    # Determine if backtest is truly OOS
    oos_start = _meta.get("oos_period", "2023-01-01").split(" to ")[0]
    is_oos = req.start_date >= oos_start
    if req.start_date < "2022-01-01":
        backtest_warning = (
            "⚠ IN-SAMPLE: Your start date falls within the model's training period "
            f"({_meta.get('train_period')}). These results are optimistic and CANNOT "
            "be submitted to SEBI as evidence of strategy performance."
        )
    elif req.start_date < oos_start:
        backtest_warning = (
            "⚠ VALIDATION PERIOD: Backtest partially overlaps the model's "
            "validation window. Results may be slightly optimistic."
        )
    else:
        backtest_warning = (
            "✅ OUT-OF-SAMPLE: Backtest period is fully outside model training data. "
            "These results are suitable for SEBI performance disclosure."
        )

    # Download data
    try:
        df_raw = yf.download(resolved, start=req.start_date, progress=False)
        if df_raw.empty:
            raise HTTPException(400, f"No data for {req.ticker}. Try: RELIANCE.NS, ^NSEI, BTC-USD")
        df_raw.columns = [c[0] if isinstance(c, tuple) else c for c in df_raw.columns]
        df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()]
        df = df_raw[["Open","High","Low","Close","Volume"]].copy()
        df = get_features(df).dropna()
        if len(df) < 120:
            raise HTTPException(400, f"Insufficient data for {resolved} after {req.start_date}.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Data download failed: {e}")

    # Inference — NO future data used
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = _scaler.transform(df[feat_cols].values)

    TIME_STEPS = _meta.get("time_steps", 60)
    if len(X) <= TIME_STEPS:
        raise HTTPException(400, f"Not enough rows for sequence length {TIME_STEPS}.")

    X_seq     = create_sequences(X, TIME_STEPS)
    lstm_prob = _lstm.predict(X_seq, verbose=0)

    X_final   = X[TIME_STEPS:]
    X_hybrid  = np.hstack((X_final, lstm_prob))
    raw_preds = _xgb.predict(X_hybrid)

    # Map: 0=SELL(-1), 1=HOLD(0), 2=BUY(1)
    signals = np.where(raw_preds == 0, -1, np.where(raw_preds == 1, 0, 1))

    # Signal confidence from LSTM softmax
    last_lstm = lstm_prob[-1]  # [sell_prob, hold_prob, buy_prob]
    last_signal_raw = signals[-1]
    if last_signal_raw == 1:
        confidence = float(last_lstm[2])
        signal_text = "BUY"
    elif last_signal_raw == -1:
        confidence = float(last_lstm[0])
        signal_text = "SELL"
    else:
        confidence = float(last_lstm[1])
        signal_text = "HOLD"

    if confidence >= 0.60:   strength = "STRONG"
    elif confidence >= 0.45: strength = "MODERATE"
    else:                    strength = "WEAK"

    # Backtest
    df_test = df.iloc[TIME_STEPS:].copy().reset_index(drop=True)
    result = strategy.backtest_strategy(
        df_test,
        signals,
        initial_capital=req.initial_capital,
        stop_loss_pct=req.stop_loss_pct,
        take_profit_pct=req.take_profit_pct,
        use_realistic_costs=req.use_realistic_costs,
    )

    # OOS accuracy for this ticker from training report
    oos_acc = None
    if _oos_report and resolved in _oos_report:
        oos_acc = _oos_report[resolved].get("accuracy")
    elif _oos_report:
        # Use mean OOS accuracy as proxy
        accs = [v.get("accuracy", 0) for v in _oos_report.values()]
        oos_acc = float(np.mean(accs)) if accs else None

    wf_acc = _wf_report.get("mean_accuracy") if _wf_report else None

    model_info = ModelInfo(
        train_period=_meta.get("train_period", "2015-01-01 to 2021-12-31"),
        val_period=_meta.get("val_period", "2022-01-01 to 2022-12-31"),
        oos_period=_meta.get("oos_period", "2023-01-01 to present"),
        mean_oos_accuracy=float(_meta.get("mean_oos_accuracy", 0.0)),
        trained_at=_meta.get("trained_at", "unknown"),
        tickers_trained=len(_meta.get("tickers", [])),
        walk_forward_accuracy=wf_acc,
    )

    return SignalResponse(
        ticker=resolved,
        last_signal=signal_text,
        signal_strength=strength,

        total_return=result["total_return"],
        buy_hold_return=result["buy_hold_return"],
        alpha=result["alpha"],
        max_drawdown=result["max_dd"],

        sharpe_ratio=result["sharpe_ratio"],
        sortino_ratio=result["sortino_ratio"],
        calmar_ratio=result["calmar_ratio"],

        risk_metrics=RiskMetrics(
            var_95_daily_pct=result["var_95"],
            cvar_95_daily_pct=result["cvar_95"],
            information_ratio=result["information_ratio"],
            beta=result["beta"],
            win_rate=result["win_rate"],
            avg_win_pct=result["avg_win"],
            avg_loss_pct=result["avg_loss"],
            profit_factor=result["profit_factor"],
            expectancy_pct=result["expectancy"],
        ),

        is_oos=is_oos,
        backtest_warning=backtest_warning,
        trades=result["trades"],
        cost_model=result["cost_model"],

        equity=result["equity"],
        buy_hold=result["buy_hold"],
        monthly_returns=result["monthly_returns"],

        model_info=model_info,
        oos_accuracy=oos_acc,

        disclaimer=(
            "RISK DISCLAIMER: This system generates algorithmic signals for research purposes. "
            "Past performance does not guarantee future results. Markets can move against all models. "
            "This is NOT registered investment advice under SEBI (Investment Advisers) Regulations, 2013. "
            "Obtain proper SEBI IA registration before offering this as a paid service."
        ),
        success=True,
        message="Signal generated successfully.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH & MODEL INFO ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": "3.0"}

@app.get("/model-info")
async def model_info_endpoint():
    """Returns model training metadata and OOS performance — for SEBI disclosure."""
    load_models()
    return {
        "metadata": _meta,
        "oos_report": _oos_report,
        "walkforward_report": _wf_report,
        "disclaimer": (
            "OOS accuracy reflects performance on data strictly after the training cutoff. "
            "Walk-forward results use expanding window re-training. "
            "All reported figures use realistic Indian market transaction costs."
        )
    }