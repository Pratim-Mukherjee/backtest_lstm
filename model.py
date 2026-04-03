# model.py  –  FINAL FIXED multi-ticker training
"""
Run this:
    python model.py
"""

import os, pickle
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ── Fractals ─────────────────────────────────────────────────────────────
def add_fractals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fractal_bull"] = 0
    df["fractal_bear"] = 0
    for i in range(2, len(df) - 2):
        if (df["Low"].iloc[i] < df["Low"].iloc[i-2] and df["Low"].iloc[i] < df["Low"].iloc[i-1] and
            df["Low"].iloc[i] < df["Low"].iloc[i+1] and df["Low"].iloc[i] < df["Low"].iloc[i+2]):
            df.loc[df.index[i], "fractal_bull"] = 1
        if (df["High"].iloc[i] > df["High"].iloc[i-2] and df["High"].iloc[i] > df["High"].iloc[i-1] and
            df["High"].iloc[i] > df["High"].iloc[i+1] and df["High"].iloc[i] > df["High"].iloc[i+2]):
            df.loc[df.index[i], "fractal_bear"] = 1
    return df

# ── Robust Feature Engineering ───────────────────────────────────────────
def download_and_enrich(ticker: str, start_date: str = "2018-01-01") -> pd.DataFrame:
    print(f"📥 Downloading {ticker} ...")
    try:
        df_raw = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
        if df_raw.empty:
            print(f"⚠️  No data for {ticker}")
            return pd.DataFrame()

        df_raw.columns = [col[0] if isinstance(col, tuple) else col for col in df_raw.columns]
        df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()].copy()

        df = df_raw[["Open", "High", "Low", "Close", "Volume"]].copy()

        df["Return"] = df["Close"].pct_change()
        df["LogRet"] = np.log(df["Close"] / df["Close"].shift(1))
        df = add_fractals(df)

        # SMA & RSI
        df["SMA_10"] = ta.sma(df["Close"], length=10)
        df["SMA_20"] = ta.sma(df["Close"], length=20)
        df["SMA_50"] = ta.sma(df["Close"], length=50)
        df["RSI_14"] = ta.rsi(df["Close"], length=14)

        # MACD - safe handling
        macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        if not macd.empty:
            macd_cols = macd.columns.tolist()
            rename_map = {macd_cols[0]: "MACD_12_26_9",
                          macd_cols[1]: "MACDs_12_26_9",
                          macd_cols[2]: "MACDh_12_26_9"}
            macd = macd.rename(columns=rename_map)
            df = pd.concat([df, macd], axis=1)

        # Bollinger Bands - very robust handling
        bb = ta.bbands(df["Close"], length=20, std=2.0)
        if not bb.empty:
            bb_cols = bb.columns.tolist()
            rename = {}
            for c in bb_cols:
                if any(x in c.lower() for x in ["lower", "bbl"]):
                    rename[c] = "BBL_20_2.0"
                elif any(x in c.lower() for x in ["mid", "bbm", "basis"]):
                    rename[c] = "BBM_20_2.0"
                elif any(x in c.lower() for x in ["upper", "bbu"]):
                    rename[c] = "BBU_20_2.0"
                elif any(x in c.lower() for x in ["percent", "bbp"]):
                    rename[c] = "BBP_20_2.0"
            bb = bb.rename(columns=rename)
            df = pd.concat([df, bb], axis=1)

        # ATR, ADX, CCI, Stoch
        df["ATR_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

        adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
        if not adx.empty:
            df["ADX_14"] = adx.iloc[:, 0]   # First column is usually ADX

        df["CCI_20_0.015"] = ta.cci(df["High"], df["Low"], df["Close"], length=20, c=0.015)

        stoch = ta.stoch(df["High"], df["Low"], df["Close"], k=14, d=3, smooth_k=3)
        if not stoch.empty:
            stoch_cols = stoch.columns.tolist()
            df["STOCHk_14_3_3"] = stoch.iloc[:, 0]
            if len(stoch_cols) > 1:
                df["STOCHd_14_3_3"] = stoch.iloc[:, 1]

        # Additional features
        df["Norm_Ret"] = df["Return"].rolling(20).apply(
            lambda x: (x[-1] - x.mean()) / (x.std() + 1e-8) if len(x) > 1 else 0, raw=True)
        df["Vol_20"] = df["Return"].rolling(20).std()
        df["Ret_Lag_1"] = df["Return"].shift(1)
        df["Ret_Lag_2"] = df["Return"].shift(2)

        # Select only available features
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

        available = [c for c in FEATURE_COLS if c in df.columns]
        df_final = df[available].dropna()

        print(f"✅ {ticker} → {len(df_final):,} clean rows")
        return df_final

    except Exception as e:
        print(f"❌ {ticker} failed: {str(e)[:120]}")
        return pd.DataFrame()

# ── Labels & Sequences ───────────────────────────────────────────────────
def create_labels(df: pd.DataFrame, future_period=5, pos_thresh=0.020, neg_thresh=-0.020):
    df = df.copy()
    df["Future_Return"] = df["Close"].shift(-future_period) / df["Close"] - 1
    df["Target"] = df["Future_Return"].apply(
        lambda r: 1 if r >= pos_thresh else (-1 if r <= neg_thresh else 0)
    )
    return df.dropna()

def create_sequences(X: np.ndarray, y: np.ndarray, time_steps: int):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32, return_sequences=True),
        Dropout(0.3),
        LSTM(16),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(32, activation="relu"),
        Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# ── Main Training ────────────────────────────────────────────────────────
def train_and_save(output_dir: str = "models"):
    os.makedirs(output_dir, exist_ok=True)

    tickers = [
        "^NSEI", "^NSEBANK", "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
        "ICICIBANK.NS", "SBIN.NS", "ITC.NS", "ONGC.NS", "AXISBANK.NS", "KOTAKBANK.NS",
        "MARUTI.NS", "BHARTIARTL.NS", "LT.NS", "HINDUNILVR.NS", "SUNPHARMA.NS",
        "HCLTECH.NS", "TITAN.NS", "ULTRACEMCO.NS", "BTC-USD", "ETH-USD"
    ]

    print(f"📂 Starting training on {len(tickers)} tickers...\n")

    FEATURE_COLS = [
        "Open", "High", "Low", "Close", "Volume", "fractal_bull", "fractal_bear",
        "SMA_10", "SMA_20", "SMA_50", "RSI_14", "MACD_12_26_9", "MACDs_12_26_9",
        "MACDh_12_26_9", "BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "BBP_20_2.0",
        "ATR_14", "ADX_14", "CCI_20_0.015", "STOCHk_14_3_3", "STOCHd_14_3_3",
        "Norm_Ret", "Vol_20", "Ret_Lag_1", "Ret_Lag_2"
    ]

    raw_train_list = []
    for ticker in tickers:
        df = download_and_enrich(ticker)
        if len(df) < 300:
            continue
        df = create_labels(df)
        if len(df) < 200:
            continue
        split = int(0.8 * len(df))
        X_raw = df[FEATURE_COLS].values[:split]
        raw_train_list.append(X_raw)

    if not raw_train_list:
        raise RuntimeError("No usable data from any ticker. Check internet / yfinance.")

    X_train_raw = np.vstack(raw_train_list)
    scaler = StandardScaler().fit(X_train_raw)
    print(f"\n✅ Scaler fitted on {X_train_raw.shape[0]:,} samples")

    # Sequences
    TIME_STEPS = 60
    X_tr_seq_list, y_tr_seq_list = [], []

    for ticker in tickers:
        df = download_and_enrich(ticker)
        if len(df) < 300: 
            continue
        df = create_labels(df)
        if len(df) < 200: 
            continue

        X = df[FEATURE_COLS].values
        y = df["Target"].values
        split = int(0.8 * len(df))

        X_train = scaler.transform(X[:split])
        X_tr_seq, y_tr_seq = create_sequences(X_train, y[:split], TIME_STEPS)

        if len(X_tr_seq) > 10:
            X_tr_seq_list.append(X_tr_seq)
            y_tr_seq_list.append(y_tr_seq)

    if not X_tr_seq_list:
        raise RuntimeError("Not enough sequences generated.")

    X_tr_seq_all = np.vstack(X_tr_seq_list)
    y_tr_seq_all = np.concatenate(y_tr_seq_list)

    print(f"✅ Total training sequences: {X_tr_seq_all.shape[0]:,}")

    # LSTM
    print("\n🧠 Training LSTM...")
    lstm_model = build_lstm((TIME_STEPS, X_tr_seq_all.shape[2]))
    early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

    lstm_model.fit(
        X_tr_seq_all, to_categorical(y_tr_seq_all + 1, num_classes=3),
        epochs=40, batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1, shuffle=False
    )

    lstm_tr_pred = lstm_model.predict(X_tr_seq_all, verbose=0)

    # XGBoost
    print("\n🌲 Training XGBoost...")
    y_tr_hyb = np.where(y_tr_seq_all == -1, 0, np.where(y_tr_seq_all == 0, 1, 2))
    X_tr_hyb = np.hstack((X_tr_seq_all[:, -1, :], lstm_tr_pred))

    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective="multi:softprob", num_class=3,
        random_state=42, verbosity=0
    )
    xgb_model.fit(X_tr_hyb, y_tr_hyb)

    # Save
    print("\n💾 Saving models to ./models/")
    pickle.dump(scaler, open(os.path.join(output_dir, "scaler.pkl"), "wb"))
    lstm_model.save(os.path.join(output_dir, "lstm_model.keras"))
    pickle.dump(xgb_model, open(os.path.join(output_dir, "xgb_model.pkl"), "wb"))
    pickle.dump({
        "tickers": tickers,
        "feature_cols": FEATURE_COLS,
        "time_steps": TIME_STEPS
    }, open(os.path.join(output_dir, "metadata.pkl"), "wb"))

    print("🎉 Training completed successfully! You can now run the FastAPI app.")

if __name__ == "__main__":
    train_and_save()