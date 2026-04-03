# app.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import strategy

# --- 1. Setup paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

app = FastAPI(
    title="FractalEdge - LSTM-XGBoost Trading Signal",
    description="Professional hybrid LSTM-XGBoost trading signals with backtesting for multiple assets.",
    version="2.0"
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- 2. Ticker resolution ---
TICKER_ALIAS_MAP = {
    "nifty": "^NSEI",
    "banknifty": "^NSEBANK",
    "reliance": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "infy": "INFY.NS",
    "hdfc": "HDFCBANK.NS",
    "icici": "ICICIBANK.NS",
    "sbi": "SBIN.NS",
    "itc": "ITC.NS",
    "ongc": "ONGC.NS",
    "axis": "AXISBANK.NS",
    "kotak": "KOTAKBANK.NS",
    "maruti": "MARUTI.NS",
    "tatamotors": "TATAMOTORS.NS",
    "bharti": "BHARTIARTL.NS",
    "lt": "LT.NS",
    "hindunilvr": "HINDUNILVR.NS",
    "sunpharma": "SUNPHARMA.NS",
    "hcltech": "HCLTECH.NS",
    "titan": "TITAN.NS",
    "ultracemco": "ULTRACEMCO.NS",
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
}

def resolve_ticker(user_input: str) -> str:
    clean = user_input.strip().lower()
    if clean in TICKER_ALIAS_MAP:
        return TICKER_ALIAS_MAP[clean]

    # Try direct ticker
    try:
        ticker = yf.Ticker(user_input)
        info = ticker.info or {}
        if info.get("symbol"):
            return user_input
    except Exception:
        pass

    # Common guesses
    guesses = [
        user_input.upper(),
        user_input.upper() + ".NS",
        "^" + user_input.upper()
    ]
    for guess in guesses:
        try:
            ticker = yf.Ticker(guess)
            hist = ticker.history(period="1mo")
            if not hist.empty and len(hist) >= 10:
                return guess
        except Exception:
            continue

    return user_input

# --- 3. Serve UI ---
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)

# --- 4. Model loading ---
scaler = None
lstm_model = None
xgb_model = None
metadata = None

def load_models():
    global scaler, lstm_model, xgb_model, metadata
    if scaler is not None:
        return

    try:
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        lstm_path = os.path.join(MODELS_DIR, "lstm_model.keras")
        xgb_path = os.path.join(MODELS_DIR, "xgb_model.pkl")
        meta_path = os.path.join(MODELS_DIR, "metadata.pkl")

        scaler = pickle.load(open(scaler_path, "rb"))
        lstm_model = load_model(lstm_path)
        xgb_model = pickle.load(open(xgb_path, "rb"))
        metadata = pickle.load(open(meta_path, "rb"))

        print("✅ Models loaded successfully (multi-ticker trained)")
    except Exception as e:
        raise RuntimeError(f"Failed to load models: {e}")

# --- 5. Feature engineering (kept same as before) ---
def add_fractals(df):
    df = df.copy()
    df['fractal_bull'] = 0
    df['fractal_bear'] = 0
    for i in range(2, len(df) - 2):
        if (df['Low'].iloc[i] < df['Low'].iloc[i-2] and
            df['Low'].iloc[i] < df['Low'].iloc[i-1] and
            df['Low'].iloc[i] < df['Low'].iloc[i+1] and
            df['Low'].iloc[i] < df['Low'].iloc[i+2]):
            df.loc[df.index[i], 'fractal_bull'] = 1
        if (df['High'].iloc[i] > df['High'].iloc[i-2] and
            df['High'].iloc[i] > df['High'].iloc[i-1] and
            df['High'].iloc[i] > df['High'].iloc[i+1] and
            df['High'].iloc[i] > df['High'].iloc[i+2]):
            df.loc[df.index[i], 'fractal_bear'] = 1
    return df

def get_features(df):
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
    df = add_fractals(df)

    df['SMA_10'] = ta.sma(df['Close'], length=10)
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['RSI_14'] = ta.rsi(df['Close'], length=14)

    # MACD
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if not macd.empty:
        macd_cols = macd.columns.tolist()
        if len(macd_cols) >= 3:
            macd = macd.rename(columns={
                macd_cols[0]: "MACD_12_26_9",
                macd_cols[1]: "MACDs_12_26_9",
                macd_cols[2]: "MACDh_12_26_9"
            })
        df = pd.concat([df, macd], axis=1)

    # Bollinger Bands - robust
    bb = ta.bbands(df['Close'], length=20, std=2.0)
    if not bb.empty:
        bb_cols = bb.columns.tolist()
        rename = {}
        for c in bb_cols:
            if any(x in c.lower() for x in ['lower', 'bbl']): rename[c] = "BBL_20_2.0"
            elif any(x in c.lower() for x in ['mid', 'bbm', 'basis']): rename[c] = "BBM_20_2.0"
            elif any(x in c.lower() for x in ['upper', 'bbu']): rename[c] = "BBU_20_2.0"
            elif any(x in c.lower() for x in ['percent', 'bbp']): rename[c] = "BBP_20_2.0"
        bb = bb.rename(columns=rename)
        df = pd.concat([df, bb], axis=1)

    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    if not adx.empty and adx.shape[1] > 0:
        df['ADX_14'] = adx.iloc[:, 0]

    df['CCI_20_0.015'] = ta.cci(df['High'], df['Low'], df['Close'], length=20, c=0.015)

    stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3, smooth_k=3)
    if not stoch.empty:
        df['STOCHk_14_3_3'] = stoch.iloc[:, 0]
        if stoch.shape[1] > 1:
            df['STOCHd_14_3_3'] = stoch.iloc[:, 1]

    df['Norm_Ret'] = df['Return'].rolling(20).apply(
        lambda x: (x[-1] - x.mean()) / (x.std() + 1e-8) if len(x) > 1 else 0, raw=True)
    df['Vol_20'] = df['Return'].rolling(20).std()
    df['Ret_Lag_1'] = df['Return'].shift(1)
    df['Ret_Lag_2'] = df['Return'].shift(2)

    # CRITICAL: Use exactly these 27 feature columns (matching your scaler)
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

    # Only keep columns that actually exist (prevents feature mismatch)
    available = [c for c in FEATURE_COLS if c in df.columns]
    result_df = df[available].copy()

    print(f"Debug: Generated {len(available)} features: {available}")  # temporary debug line

    return result_df
    
def create_sequences(X, time_steps=60):
    Xs = []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
    return np.array(Xs)

# --- 6. API Schemas ---
class PredictRequest(BaseModel):
    ticker: str
    start_date: str = "2018-01-01"
    initial_capital: float = 100000.0
    fee: float = 0.001
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.04
    slippage_pct: float = 0.001

class SignalResponse(BaseModel):
    ticker: str
    last_signal: str
    total_return: float
    max_drawdown: float
    buy_hold_return: float
    trades: int
    equity: list[float]
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    success: bool
    message: str

# --- 7. Main Prediction Endpoint ---
@app.post("/predict", response_model=SignalResponse)
async def predict_signal(request: PredictRequest):
    load_models()

    resolved = resolve_ticker(request.ticker)

    try:
        df_raw = yf.download(
            resolved,
            start=request.start_date,
            progress=False
        )
        if df_raw.empty:
            raise HTTPException(
                400,
                f"No data found for ticker: {request.ticker}. Try RELIANCE.NS, ^NSEI, AAPL, etc."
            )

        df_raw.columns = [col[0] if isinstance(col, tuple) else col for col in df_raw.columns]
        df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()].copy()

        df = df_raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df = get_features(df)
        df = df.dropna()

        if len(df) < 120:
            raise HTTPException(
                400,
                f"Insufficient data for {resolved} after {request.start_date}."
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            400,
            f"Data download failed for {request.ticker}: {str(e)}"
        )

    # Feature preparation
    feature_cols = [c for c in df.columns if c not in ['Future_Return', 'Target']]
    X = df[feature_cols].values
    X = scaler.transform(X)

    TIME_STEPS = 60
    if len(X) <= TIME_STEPS:
        raise HTTPException(400, "Not enough data for sequence length 60.")

    X_seq = create_sequences(X, TIME_STEPS)
    lstm_probs = lstm_model.predict(X_seq, verbose=0)

    X_final = X[TIME_STEPS:]
    X_hybrid = np.hstack((X_final, lstm_probs))
    xgb_preds = xgb_model.predict(X_hybrid)

    signals = np.where(xgb_preds == 0, -1,
                       np.where(xgb_preds == 1, 0, 1))

    df_test = df.iloc[TIME_STEPS:].copy().reset_index(drop=True)

    # Backtest with improved strategy
    result = strategy.backtest_strategy(
        df_test,
        signals,
        initial_capital=request.initial_capital,
        fee=request.fee,
        stop_loss_pct=request.stop_loss_pct,
        take_profit_pct=request.take_profit_pct,
        slippage_pct=request.slippage_pct
    )

    last_signal = signals[-1]
    signal_text = "BUY" if last_signal == 1 else "SELL" if last_signal == -1 else "HOLD"

    return SignalResponse(
        ticker=resolved,
        last_signal=signal_text,
        total_return=result["total_return"],
        max_drawdown=result["max_dd"],
        buy_hold_return=result["buy_hold_return"],
        trades=result["trades"],
        equity=result["equity"],
        sharpe_ratio=result["sharpe_ratio"],
        sortino_ratio=result["sortino_ratio"],
        calmar_ratio=result["calmar_ratio"],
        success=True,
        message="Prediction and backtest completed successfully."
    )