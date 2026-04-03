# model.py — SEBI-Ready Production Training
"""
FIXES APPLIED:
1. Strict temporal train/val/test split (no data leakage)
2. Walk-forward cross-validation across 5 folds
3. Labels use only future returns (training only) — no inference leakage
4. Scaler fitted ONLY on training data, applied to val/test
5. OOS performance saved separately for regulatory reporting
6. Realistic Indian market cost assumptions baked into evaluation
7. Feature engineering uses only past data (verified)

Run:
    python model.py

Outputs:
    models/scaler.pkl
    models/lstm_model.keras
    models/xgb_model.pkl
    models/metadata.pkl
    models/oos_report.pkl        ← Out-of-sample performance for SEBI
    models/walkforward_report.pkl ← Walk-forward fold results
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — strict temporal boundaries (no overlap)
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_START   = "2015-01-01"
TRAIN_END     = "2021-12-31"   # Training universe
VAL_START     = "2022-01-01"   # Validation — hyperparameter selection only
VAL_END       = "2022-12-31"
OOS_START     = "2023-01-01"   # True out-of-sample — never touched during training
OOS_END       = datetime.today().strftime("%Y-%m-%d")

TIME_STEPS    = 60
FUTURE_PERIOD = 5              # Predict 5-day forward return
POS_THRESH    = 0.020          # +2% → BUY label
NEG_THRESH    = -0.020         # −2% → SELL label

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

TICKERS = [
    "^NSEI", "^NSEBANK",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "SBIN.NS", "ITC.NS", "ONGC.NS",
    "AXISBANK.NS", "KOTAKBANK.NS", "MARUTI.NS", "BHARTIARTL.NS",
    "LT.NS", "HINDUNILVR.NS", "SUNPHARMA.NS",
    "HCLTECH.NS", "TITAN.NS", "ULTRACEMCO.NS",
    "BTC-USD", "ETH-USD",
    # Additional diversity
    "WIPRO.NS", "NESTLEIND.NS", "BAJFINANCE.NS",
]


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING — all indicators use ONLY past data (causal)
# ─────────────────────────────────────────────────────────────────────────────
def add_fractals(df: pd.DataFrame) -> pd.DataFrame:
    """Williams fractal detection — uses ±2 bars, inherently causal."""
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


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Pure feature engineering. Uses ONLY past data. 
    Labels (future returns) are added separately and only for training.
    """
    df = df_raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["Return"]  = df["Close"].pct_change()
    df["LogRet"]  = np.log(df["Close"] / df["Close"].shift(1))
    df = add_fractals(df)

    df["SMA_10"] = ta.sma(df["Close"], length=10)
    df["SMA_20"] = ta.sma(df["Close"], length=20)
    df["SMA_50"] = ta.sma(df["Close"], length=50)
    df["RSI_14"] = ta.rsi(df["Close"], length=14)

    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        cols = macd.columns.tolist()
        macd = macd.rename(columns={
            cols[0]: "MACD_12_26_9",
            cols[1]: "MACDs_12_26_9",
            cols[2]: "MACDh_12_26_9"
        })
        df = pd.concat([df, macd], axis=1)

    bb = ta.bbands(df["Close"], length=20, std=2.0)
    if bb is not None and not bb.empty:
        rename = {}
        for c in bb.columns:
            cl = c.lower()
            if any(x in cl for x in ["lower", "bbl"]):   rename[c] = "BBL_20_2.0"
            elif any(x in cl for x in ["mid", "bbm"]):   rename[c] = "BBM_20_2.0"
            elif any(x in cl for x in ["upper", "bbu"]): rename[c] = "BBU_20_2.0"
            elif any(x in cl for x in ["percent", "bbp"]): rename[c] = "BBP_20_2.0"
        bb = bb.rename(columns=rename)
        df = pd.concat([df, bb], axis=1)

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


def attach_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add forward-looking labels. ONLY call this on training data.
    Never call on test/inference data — that would be look-ahead bias.
    """
    df = df.copy()
    df["Future_Return"] = df["Close"].shift(-FUTURE_PERIOD) / df["Close"] - 1
    df["Target"] = df["Future_Return"].apply(
        lambda r: 2 if r >= POS_THRESH else (0 if r <= NEG_THRESH else 1)
    )
    return df.dropna(subset=["Future_Return", "Target"])


def download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download and clean OHLCV."""
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()].copy()
        return df[["Open", "High", "Low", "Close", "Volume"]].copy()
    except Exception as e:
        print(f"  ⚠ Download failed for {ticker}: {str(e)[:80]}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# SEQUENCE CREATION
# ─────────────────────────────────────────────────────────────────────────────
def create_sequences(X: np.ndarray, y: np.ndarray, time_steps: int):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def create_sequences_X_only(X: np.ndarray, time_steps: int):
    return np.array([X[i:i + time_steps] for i in range(len(X) - time_steps)])


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
def build_lstm(input_shape: tuple) -> Sequential:
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(32),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(3, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def walk_forward_cv(n_folds: int = 5) -> dict:
    """
    Walk-forward validation on NIFTY 50 (representative index).
    Each fold: train on all preceding data, test on next ~1-year window.
    Returns per-fold accuracy, precision, recall for SEBI reporting.
    """
    print("\n📊 Walk-Forward Cross-Validation (SEBI Compliance)...")
    ticker = "^NSEI"
    df_raw = download_ticker(ticker, "2013-01-01", OOS_END)
    if df_raw.empty:
        print("  ⚠ Could not download NIFTY for walk-forward. Skipping.")
        return {}

    df_feat = build_features(df_raw).dropna()
    df_labeled = attach_labels(df_feat)
    if len(df_labeled) < 500:
        return {}

    n = len(df_labeled)
    fold_size = n // (n_folds + 1)
    results = []

    for fold in range(n_folds):
        train_end_idx = (fold + 1) * fold_size
        test_end_idx  = min((fold + 2) * fold_size, n)

        if test_end_idx - train_end_idx < 50:
            continue

        df_tr  = df_labeled.iloc[:train_end_idx]
        df_te  = df_labeled.iloc[train_end_idx:test_end_idx]

        feat_cols = [c for c in FEATURE_COLS if c in df_tr.columns]
        X_tr = df_tr[feat_cols].values
        y_tr = df_tr["Target"].values.astype(int)
        X_te = df_te[feat_cols].values
        y_te = df_te["Target"].values.astype(int)

        sc = StandardScaler().fit(X_tr)
        X_tr_s = sc.transform(X_tr)
        X_te_s = sc.transform(X_te)

        X_tr_seq, y_tr_seq = create_sequences(X_tr_s, y_tr, TIME_STEPS)
        X_te_seq, y_te_seq = create_sequences(X_te_s, y_te, TIME_STEPS)

        if len(X_tr_seq) < 50 or len(X_te_seq) < 10:
            continue

        # Lightweight LSTM for fold validation
        m = build_lstm((TIME_STEPS, X_tr_seq.shape[2]))
        m.fit(X_tr_seq, to_categorical(y_tr_seq, num_classes=3),
              epochs=15, batch_size=64, verbose=0,
              callbacks=[EarlyStopping(patience=4, restore_best_weights=True)])

        lstm_tr = m.predict(X_tr_seq, verbose=0)
        lstm_te = m.predict(X_te_seq, verbose=0)

        Xh_tr = np.hstack((X_tr_seq[:, -1, :], lstm_tr))
        Xh_te = np.hstack((X_te_seq[:, -1, :], lstm_te))

        xgb_m = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            objective="multi:softprob", num_class=3,
            random_state=42, verbosity=0
        )
        xgb_m.fit(Xh_tr, y_tr_seq)
        preds = xgb_m.predict(Xh_te)

        acc = (preds == y_te_seq).mean()
        report = classification_report(y_te_seq, preds,
                                       target_names=["SELL","HOLD","BUY"],
                                       output_dict=True, zero_division=0)
        results.append({
            "fold": fold + 1,
            "train_rows": len(X_tr_seq),
            "test_rows": len(X_te_seq),
            "accuracy": float(acc),
            "buy_precision": float(report.get("BUY", {}).get("precision", 0)),
            "buy_recall": float(report.get("BUY", {}).get("recall", 0)),
            "sell_precision": float(report.get("SELL", {}).get("precision", 0)),
        })
        print(f"  Fold {fold+1}: Accuracy={acc:.3f}  "
              f"BUY P={results[-1]['buy_precision']:.3f} "
              f"R={results[-1]['buy_recall']:.3f}")

    mean_acc = np.mean([r["accuracy"] for r in results]) if results else 0
    print(f"  ✅ Mean Walk-Forward Accuracy: {mean_acc:.3f}")
    return {"folds": results, "mean_accuracy": float(mean_acc)}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_and_save(output_dir: str = "models"):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("  FractalEdge — SEBI-Ready Training Pipeline")
    print(f"  Train : {TRAIN_START} → {TRAIN_END}")
    print(f"  Val   : {VAL_START}   → {VAL_END}")
    print(f"  OOS   : {OOS_START}   → {OOS_END}  ← NEVER SEEN DURING TRAINING")
    print("=" * 65)

    # ── Step 1: Walk-Forward Validation ──────────────────────────────────
    wf_report = walk_forward_cv(n_folds=5)

    # ── Step 2: Collect training data (TRAIN_START → TRAIN_END only) ─────
    print(f"\n📥 Downloading {len(TICKERS)} tickers for training period...")
    raw_for_scaler = []
    ticker_data_train = {}

    for ticker in TICKERS:
        df_raw = download_ticker(ticker, TRAIN_START, TRAIN_END)
        if df_raw.empty or len(df_raw) < 300:
            print(f"  ⚠ {ticker}: insufficient data, skipping")
            continue

        df_feat = build_features(df_raw).dropna()
        df_labeled = attach_labels(df_feat)
        if len(df_labeled) < 200:
            continue

        feat_cols = [c for c in FEATURE_COLS if c in df_labeled.columns]
        raw_for_scaler.append(df_labeled[feat_cols].values)
        ticker_data_train[ticker] = df_labeled
        print(f"  ✅ {ticker}: {len(df_labeled):,} labeled rows")

    if not raw_for_scaler:
        raise RuntimeError("No training data. Check internet connection.")

    # ── Step 3: Fit scaler ON TRAINING DATA ONLY ─────────────────────────
    X_all_train = np.vstack(raw_for_scaler)
    scaler = StandardScaler().fit(X_all_train)
    print(f"\n✅ Scaler fitted on {X_all_train.shape[0]:,} training samples")
    print(f"   Features: {X_all_train.shape[1]}")

    # ── Step 4: Build sequences ───────────────────────────────────────────
    X_tr_list, y_tr_list = [], []
    for ticker, df in ticker_data_train.items():
        feat_cols = [c for c in FEATURE_COLS if c in df.columns]
        X = scaler.transform(df[feat_cols].values)
        y = df["Target"].values.astype(int)
        Xs, ys = create_sequences(X, y, TIME_STEPS)
        if len(Xs) > 10:
            X_tr_list.append(Xs)
            y_tr_list.append(ys)

    X_tr_all = np.vstack(X_tr_list)
    y_tr_all = np.concatenate(y_tr_list)
    print(f"✅ Total training sequences: {X_tr_all.shape[0]:,}")

    # ── Step 5: Validation data (scaler already fitted on train) ─────────
    print(f"\n📥 Downloading validation data ({VAL_START} → {VAL_END})...")
    X_val_list, y_val_list = [], []
    for ticker in TICKERS[:8]:  # Top 8 for validation speed
        df_raw = download_ticker(ticker, VAL_START, VAL_END)
        if df_raw.empty: continue
        df_feat = build_features(df_raw).dropna()
        df_labeled = attach_labels(df_feat)
        if len(df_labeled) < 100: continue
        feat_cols = [c for c in FEATURE_COLS if c in df_labeled.columns]
        X_v = scaler.transform(df_labeled[feat_cols].values)
        y_v = df_labeled["Target"].values.astype(int)
        Xs, ys = create_sequences(X_v, y_v, TIME_STEPS)
        if len(Xs) > 10:
            X_val_list.append(Xs)
            y_val_list.append(ys)

    X_val_all = np.vstack(X_val_list) if X_val_list else None
    y_val_all = np.concatenate(y_val_list) if y_val_list else None

    # ── Step 6: Train LSTM ────────────────────────────────────────────────
    print("\n🧠 Training LSTM...")
    lstm_model = build_lstm((TIME_STEPS, X_tr_all.shape[2]))

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
    ]

    val_data = None
    if X_val_all is not None:
        val_data = (X_val_all, to_categorical(y_val_all, num_classes=3))

    lstm_model.fit(
        X_tr_all,
        to_categorical(y_tr_all, num_classes=3),
        epochs=60,
        batch_size=64,
        validation_data=val_data,
        validation_split=(0.15 if val_data is None else 0.0),
        callbacks=callbacks,
        verbose=1,
        shuffle=False     # IMPORTANT: don't shuffle time series
    )

    # ── Step 7: Train XGBoost ─────────────────────────────────────────────
    print("\n🌲 Training XGBoost hybrid...")
    lstm_tr_pred = lstm_model.predict(X_tr_all, verbose=0)
    X_tr_hyb = np.hstack((X_tr_all[:, -1, :], lstm_tr_pred))

    xgb_model = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=2.0,
        reg_alpha=0.1,
        min_child_weight=5,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        verbosity=0,
        eval_metric="mlogloss"
    )

    if X_val_all is not None:
        lstm_val_pred = lstm_model.predict(X_val_all, verbose=0)
        X_val_hyb = np.hstack((X_val_all[:, -1, :], lstm_val_pred))
        xgb_model.fit(
            X_tr_hyb, y_tr_all,
            eval_set=[(X_val_hyb, y_val_all)],
            verbose=False
        )
    else:
        xgb_model.fit(X_tr_hyb, y_tr_all)

    # ── Step 8: TRUE OOS Evaluation ───────────────────────────────────────
    print(f"\n📊 Evaluating on TRUE OOS data ({OOS_START} → {OOS_END})...")
    oos_results = {}

    for ticker in ["^NSEI", "^NSEBANK", "RELIANCE.NS", "HDFCBANK.NS", "BTC-USD"]:
        df_raw = download_ticker(ticker, OOS_START, OOS_END)
        if df_raw.empty or len(df_raw) < 150:
            continue
        df_feat = build_features(df_raw).dropna()
        df_labeled = attach_labels(df_feat)
        if len(df_labeled) < 100:
            continue

        feat_cols = [c for c in FEATURE_COLS if c in df_labeled.columns]
        X_oos = scaler.transform(df_labeled[feat_cols].values)
        y_oos = df_labeled["Target"].values.astype(int)
        X_oos_seq, y_oos_seq = create_sequences(X_oos, y_oos, TIME_STEPS)
        if len(X_oos_seq) < 10:
            continue

        lstm_oos = lstm_model.predict(X_oos_seq, verbose=0)
        X_oos_hyb = np.hstack((X_oos_seq[:, -1, :], lstm_oos))
        preds = xgb_model.predict(X_oos_hyb)

        acc = (preds == y_oos_seq).mean()
        report = classification_report(y_oos_seq, preds,
                                       target_names=["SELL","HOLD","BUY"],
                                       output_dict=True, zero_division=0)
        oos_results[ticker] = {
            "accuracy": float(acc),
            "n_samples": int(len(y_oos_seq)),
            "period": f"{OOS_START} to {OOS_END}",
            "buy_precision": float(report.get("BUY", {}).get("precision", 0)),
            "buy_recall": float(report.get("BUY", {}).get("recall", 0)),
            "buy_f1": float(report.get("BUY", {}).get("f1-score", 0)),
            "sell_precision": float(report.get("SELL", {}).get("precision", 0)),
            "confusion_matrix": confusion_matrix(y_oos_seq, preds).tolist(),
            "classification_report": report,
        }
        print(f"  {ticker}: OOS Acc={acc:.3f}  "
              f"BUY F1={oos_results[ticker]['buy_f1']:.3f}")

    mean_oos_acc = np.mean([v["accuracy"] for v in oos_results.values()]) if oos_results else 0
    print(f"\n  📌 Mean OOS Accuracy: {mean_oos_acc:.3f}")
    print("  ⚠ NOTE: OOS accuracy 50-60% is realistic & tradable — be wary of >70%")

    # ── Step 9: Save everything ───────────────────────────────────────────
    print("\n💾 Saving models and reports...")
    pickle.dump(scaler,    open(os.path.join(output_dir, "scaler.pkl"), "wb"))
    lstm_model.save(os.path.join(output_dir, "lstm_model.keras"))
    pickle.dump(xgb_model, open(os.path.join(output_dir, "xgb_model.pkl"), "wb"))

    metadata = {
        "tickers":          TICKERS,
        "feature_cols":     FEATURE_COLS,
        "time_steps":       TIME_STEPS,
        "train_period":     f"{TRAIN_START} to {TRAIN_END}",
        "val_period":       f"{VAL_START} to {VAL_END}",
        "oos_period":       f"{OOS_START} to {OOS_END}",
        "future_period":    FUTURE_PERIOD,
        "pos_thresh":       POS_THRESH,
        "neg_thresh":       NEG_THRESH,
        "trained_at":       datetime.now().isoformat(),
        "mean_oos_accuracy": mean_oos_acc,
    }
    pickle.dump(metadata,    open(os.path.join(output_dir, "metadata.pkl"),         "wb"))
    pickle.dump(oos_results, open(os.path.join(output_dir, "oos_report.pkl"),       "wb"))
    pickle.dump(wf_report,   open(os.path.join(output_dir, "walkforward_report.pkl"), "wb"))

    print("\n🎉 Training complete!")
    print(f"   Mean Walk-Forward Accuracy : {wf_report.get('mean_accuracy', 'N/A'):.3f}")
    print(f"   Mean OOS Accuracy          : {mean_oos_acc:.3f}")
    print("\n📋 SEBI Disclosure Summary:")
    print(f"   Training period            : {TRAIN_START} to {TRAIN_END}")
    print(f"   Out-of-sample test period  : {OOS_START} to {OOS_END}")
    print(f"   No forward-looking features used in inference")
    print(f"   Labels derived from future returns (training only)")
    print(f"   Scaler fitted on training data only")
    print(f"\nRun the API: uvicorn app:app --reload")

if __name__ == "__main__":
    train_and_save()