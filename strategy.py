import numpy as np
import pandas as pd

def backtest_strategy(df_test, signals, initial_capital=100000.0, fee=0.001,
                      stop_loss_pct=0.03, take_profit_pct=0.04, slippage_pct=0.001):
    df = df_test.copy().reset_index(drop=True)
    price = df['Close'].values
    atr = df['ATR_14'].values if 'ATR_14' in df.columns else np.full(len(df), price.mean()*0.01)
    n = len(price)
    signals = np.asarray(signals).astype(int)

    equity = np.full(n, initial_capital, dtype=float)
    in_position = False
    entry_price = 0.0
    position_size = 1.0  # dynamic sizing

    for i in range(1, n):
        equity[i] = equity[i-1]
        signal = signals[i]
        curr_price = price[i]
        curr_atr = atr[i] if i < len(atr) else atr[-1]

        # Regime-aware position sizing (Kelly-inspired)
        vol_adjust = min(1.0, 0.02 / (curr_atr / price[i-1] + 1e-8))  # risk max 2% of capital per trade
        position_size = max(0.2, vol_adjust)

        if in_position:
            daily_ret = (curr_price - price[i-1]) / price[i-1]
            equity[i] *= (1 + daily_ret * position_size)

            ret_since_entry = (curr_price / entry_price) - 1
            if ret_since_entry <= -stop_loss_pct or ret_since_entry >= take_profit_pct:
                equity[i] *= (1 - slippage_pct) * (1 - fee)
                in_position = False

        if not in_position and signal == 1:
            entry_price = curr_price * (1 + slippage_pct)
            equity[i] *= (1 - fee)
            in_position = True
        elif in_position and signal == -1:
            equity[i] *= (1 - slippage_pct) * (1 - fee)
            in_position = False

    equity = equity.tolist()

    # Buy & Hold
    buy_hold = np.full(n, initial_capital, dtype=float)
    for i in range(1, n):
        ret = (price[i] - price[i-1]) / price[i-1]
        buy_hold[i] = buy_hold[i-1] * (1 + ret)
    buy_hold = buy_hold.tolist()

    # Core stats
    start, end = equity[0], equity[-1]
    total_return = 100 * (end - start) / start
    buy_hold_return = 100 * (buy_hold[-1] - start) / start

    cummax = np.maximum.accumulate(equity)
    dd = (np.array(equity) - cummax) / cummax
    max_dd = -100 * np.min(dd) if np.min(dd) < 0 else 0.0

    trades = int((signals != 0).sum())

    # Risk metrics
    daily_rets = np.diff(equity) / np.array(equity[:-1])
    mean_daily = np.mean(daily_rets)
    std_daily = np.std(daily_rets)
    sharpe = (mean_daily / std_daily * np.sqrt(252)) if std_daily > 1e-8 else 0.0
    downside = daily_rets[daily_rets < 0]
    sortino = (mean_daily / np.std(downside) * np.sqrt(252)) if len(downside) > 0 else 0.0
    calmar = total_return / max_dd if max_dd > 0 else 0.0

    # Extra Insights
    trade_returns = []
    in_trade = False
    entry = 0
    for i in range(1, n):
        if signals[i] == 1 and not in_trade:
            entry = price[i]
            in_trade = True
        elif (signals[i] == -1 or i == n-1) and in_trade:
            ret = (price[i] / entry) - 1
            trade_returns.append(ret)
            in_trade = False

    win_rate = np.mean(np.array(trade_returns) > 0) * 100 if trade_returns else 0
    profit_factor = abs(sum([r for r in trade_returns if r > 0]) / sum([abs(r) for r in trade_returns if r < 0])) if any(r < 0 for r in trade_returns) else float('inf')
    expectancy = np.mean(trade_returns) if trade_returns else 0

    # Regime (simple ADX-based)
    adx = df['ADX_14'].values if 'ADX_14' in df.columns else np.zeros(n)
    trending_days = (adx > 25).sum() / n * 100

    return {
        "total_return": float(total_return),
        "max_dd": float(max_dd),
        "buy_hold_return": float(buy_hold_return),
        "trades": trades,
        "equity": equity,
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "calmar_ratio": float(calmar),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "expectancy": float(expectancy),
        "trending_regime_pct": float(trending_days),
        "buy_hold": buy_hold
    }