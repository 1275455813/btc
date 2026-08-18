#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大周期 EMA 趋势过滤模块

独立于回测主代码。输入 K线 DataFrame, 输出逐 bar 的开仓方向信号(整数数组):
    0 = 不可开仓(横盘 / 数据预热期)
    1 = 允许开多(上升趋势)
    2 = 允许开空(下降趋势)

用法(回测):
    from trend_filter import compute_trend_filter

    tf = compute_trend_filter(
        df,                  # 必须含 ts(时间) 和 close(收盘价) 列; 若 df 以时间为索引则 ts 可省略
        bar="4h",            # 大周期K线级别: 1h/4h/1D ...
        ema_period=50,       # 大周期 EMA 周期
        require_slope=False, # True: 额外要求 EMA 斜率同向, 减少震荡期反复开仓
        band_pct=0.0,        # 缓冲带: 0.003 = 价格须高于/低于 EMA 0.3% 才算趋势确立
    )
    # tf[i]: 第 i 根 bar 收盘后的方向信号, 取值 0 / 1 / 2

用法(实盘):
    from trend_filter import live_signal

    sig = live_signal(recent_df)                  # 只返回当前可开仓方向: 0 / 1 / 2
    info = live_signal(recent_df, return_details=True)  # dict, 便于记日志
"""

import numpy as np
import pandas as pd

# 大周期K线级别 -> 分钟数(用于估算所需最少数据量)
_BAR_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30,
                "1H": 60, "2H": 120, "4H": 240, "6H": 360, "12H": 720,
                "1D": 1440, "1W": 10080}


def _as_time_index(df):
    """返回 datetime 索引; 优先用 ts 列, 其次用 df 自带索引。"""
    if "ts" in df.columns:
        return pd.to_datetime(df["ts"])
    return pd.to_datetime(df.index)


def _infer_bar_minutes(ts):
    """根据时间索引相邻间隔中位数推断小周期K线分钟数(先统一转纳秒, 兼容 ns/us 单位)。"""
    ns = ts.astype("datetime64[ns]").astype(np.int64)
    diffs = np.diff(ns) // 60_000_000_000  # 纳秒 -> 分钟
    diffs = diffs[diffs > 0]
    return int(np.median(diffs)) if len(diffs) else 1


def _compute_raw(df, bar, ema_period):
    """核心计算: 返回 (close, ts, last_ema, rising)。

    last_ema / rising 都是已用 shift(1) 处理过的"已收盘大周期bar"值,
    并按小周期 ffill 对齐, 长度 = len(df), 无未来函数。
    """
    close = df["close"].to_numpy(dtype=float)
    ts = _as_time_index(df)

    s = pd.Series(close, index=ts)
    # pandas resample 要求小写单位(如 4h/1d), OKX 传入的是大写(4H/1D),
    # 这里统一转小写后再重采样。
    big = s.resample(bar.lower(), label="left", closed="left").last()
    ema = big.ewm(span=ema_period, min_periods=ema_period, adjust=False).mean()

    last_ema = ema.shift(1).reindex(ts, method="ffill").to_numpy()
    rising = (ema.shift(1) > ema.shift(4)).reindex(ts, method="ffill").to_numpy()
    return close, ts, last_ema, rising


def compute_trend_filter(df, bar="4h", ema_period=50, require_slope=False, band_pct=0.0):
    """
    计算逐 bar 趋势过滤结果(回测用)。

    返回:
        np.ndarray[int], 长度 = len(df), 每根 bar 取值:
            0 = 不可开仓(横盘 / 数据预热期)
            1 = 允许开多(上升趋势)
            2 = 允许开空(下降趋势)

    无未来函数保证:
        - 大周期 EMA 只取"已收盘"的 bar(shift(1)) 后再 ffill 到当前小周期 bar;
        - 斜率同理只比较已收盘的大周期 bar。
    """
    n = len(df)
    close, ts, last_ema, rising = _compute_raw(df, bar, ema_period)

    # 数据预热期(EMA 未形成)一律视为 0(不可开仓), 宁缺毋滥
    allow_long = np.where(np.isnan(last_ema), False, close > last_ema * (1 + band_pct))
    allow_short = np.where(np.isnan(last_ema), False, close < last_ema * (1 - band_pct))

    if require_slope:
        allow_long = allow_long & rising
        allow_short = allow_short & ~rising

    # 编码: 0=不可开仓 1=开多 2=开空 (long/short 天然互斥, 不会同时为 True)
    signal = np.zeros(n, dtype=int)
    signal[allow_long] = 1
    signal[allow_short] = 2
    return signal


def live_signal(df, bar="4h", ema_period=50, require_slope=False, band_pct=0.0,
                min_history=None, return_details=False):
    """
    实盘函数: 判断当前(最后一根bar)可开什么仓位。

    参数:
        df           : 最近的K线(需含 ts/close 列), 建议传入 >= min_history 根;
                       通常每次K线收盘后, 把最近 N 根K线传进来即可。
        min_history  : 最少需要多少根小周期K线才能算(默认按大周期EMA自动估算);
                       不足时返回 0(数据不足, 不可开仓)。
        return_details: False(默认)只返回 int: 0=不可开仓 1=开多 2=开空;
                        True 返回 dict: {signal, reason, ts, close, ema, rising, ...}, 便于记日志。

    返回:
        int 或 dict, 见 return_details 说明。
    """
    close, ts, last_ema, rising = _compute_raw(df, bar, ema_period)
    n = len(df)

    if min_history is None:
        span = _BAR_MINUTES.get(bar.upper(), 240) / _infer_bar_minutes(ts)
        min_history = int((ema_period + 2) * span) + 5

    if n < min_history or np.isnan(last_ema[-1]):
        sig = 0
        reason = f"数据不足或EMA未形成({n}根 < 建议{min_history}根)"
    else:
        ema_last = float(last_ema[-1])
        rising_last = bool(rising[-1])
        above = close[-1] > ema_last * (1 + band_pct)
        below = close[-1] < ema_last * (1 - band_pct)
        if require_slope:
            above = above and rising_last
            below = below and not rising_last
        if above:
            sig, reason = 1, "上升趋势, 可开多"
        elif below:
            sig, reason = 2, "下降趋势, 可开空"
        else:
            sig, reason = 0, "横盘或缓冲带内, 不可开仓"

    if not return_details:
        return sig

    return dict(
        signal=sig, reason=reason, ts=str(ts.iloc[-1]), close=float(close[-1]),
        ema=float(last_ema[-1]) if not np.isnan(last_ema[-1]) else None,
        rising=bool(rising[-1]) if not np.isnan(rising[-1]) else None,
        bar=bar, ema_period=ema_period,
    )


def latest_state(df, **kwargs):
    """便捷函数: 返回最后一根 bar 的方向信号: 0=不可开仓, 1=开多, 2=开空。"""
    return int(compute_trend_filter(df, **kwargs)[-1])


if __name__ == "__main__":
    import os

    sample = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "BTC_USDT_15m_6m.csv")
    data = pd.read_csv(sample)
    tf = compute_trend_filter(data)
    n0 = int((tf == 0).sum())
    n1 = int((tf == 1).sum())
    n2 = int((tf == 2).sum())
    print(f"共 {len(data)} 根 15m K线")
    print(f"不可开仓(0): {n0} bar ({n0 / len(data):.1%})")
    print(f"允许开多(1): {n1} bar ({n1 / len(data):.1%})")
    print(f"允许开空(2): {n2} bar ({n2 / len(data):.1%})")

    # 实盘函数演示: 取最近 1000 根(约 10 天, 满足 4h EMA50 预热)
    recent = data.tail(1000)
    print("\n[实盘演示] 最近 1000 根K线:")
    print(f"  live_signal()              -> {live_signal(recent)}")
    print(f"  live_signal(details=True)  -> {live_signal(recent, return_details=True)}")
    print(f"  live_signal(数据不足演示)   -> {live_signal(recent.head(20))}")
