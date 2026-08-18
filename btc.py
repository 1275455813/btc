#!/usr/bin/env python3
"""
BTC-USDT 永续合约 RSI(14) 均值回归策略 — 实盘机器人
=========================================================================
品种:     BTC-USDT-SWAP (OKX 永续)
K线:      15 分钟
费率:     全流程仅挂限价单 (入场 post-only, 出场限价/条件限价)

策略逻辑 (回测验证: 2026-07-12 ~ 2026-08-11, 胜率 65.7%, 回报 +1.43%)
  做多: RSI(14) 从下方上穿 30
  做空: RSI(14) 从上方下穿 70

出场 (预测价格提前挂单, 全部限价成交):
  多头:
    - RSI 止盈: RSI ≥ (exit_long - tp_arm_margin) 时, 二分反推 RSI=exit_long 的价格,
      在该价预挂卖出限价单; 每轮询预测价漂移 ≥ replace_pct 即撤单重挂。
    - 移动止损: 从最高点回落 ≥ sl_arm_ratio*trailing_pct 时, 预挂
      peak*(1-trailing_pct) 的条件限价卖单; 回落 < sl_arm_ratio*trailing_pct 撤单。
  空头: 对称 (反弹 / trough*(1+trailing_pct) 买单)。
"""

import asyncio
import json
import logging
import math
import os
import signal
import subprocess
import sys
from typing import Optional, Dict, Any, List

# ── OKX SDK ──────────────────────────────────────────────────────────
try:
    import okx.Account as AccountAPI
    import okx.Trade as TradeAPI
    import okx.MarketData as MarketDataAPI
    import okx.PublicData as PublicDataAPI
except ImportError:
    print("请先安装 python-okx: pip install python-okx")
    sys.exit(1)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    ★ 配置区 — 参数全部放在 config.json ★            ║
# ╚══════════════════════════════════════════════════════════════════════╝

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

# 配置字段说明 (与 config.json 一一对应):
#   api_key / api_secret / passphrase : OKX API 密钥
#   proxy                            : 代理地址, 如 "http://127.0.0.1:7890", 留空 "" 表示不使用代理
#   demo                             : True=模拟盘, False=实盘
#   symbol                           : 交易品种
#   order_size / max_position        : 每单张数 / 最大持仓张数
#   rsi_period / entry_long / entry_short / exit_long / exit_short / trailing_pct : 策略参数
#   tp_arm_margin                    : 提前预挂止盈的 RSI 余量 (RSI 点), 如 5 表示距阈值 5 点内开始预挂
#   sl_arm_ratio                     : 移动止损预挂启动比例 (占 trailing_pct 的比例), 如 0.5
#   replace_pct                      : 预测挂单价漂移超过该比例时撤单重挂 (如 0.001 = 0.1%)
#   sl_guard_pct                     : 止损条件单触发后的限价偏移 (确保成交), 如 0.001 = 0.1%
#   limit_offset_bps / order_timeout / max_retry : 挂单参数
#   poll_interval / candle_limit / max_fetch_errors : 运行参数


def load_config(path: str = CONFIG_PATH) -> dict:
    """从 JSON 文件加载配置; 文件不存在或格式错误时给出明确提示并退出。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到配置文件 {path}, 请从 config.json 创建并填写参数。")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"错误: 配置文件 {path} 不是合法的 JSON: {e}")
        sys.exit(1)
    if not isinstance(cfg, dict):
        print("错误: 配置文件格式错误, 顶层必须是 JSON 对象。")
        sys.exit(1)
    return cfg


CONFIG = load_config()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         策略核心引擎                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝

class RSIStrategy:
    """RSI(14) 均值回归 — 信号生成 & 出场/预挂单计算"""

    def __init__(self, cfg: dict):
        self.period    = cfg["rsi_period"]
        self.long_in   = cfg["entry_long"]
        self.short_in  = cfg["entry_short"]
        self.long_out  = cfg["exit_long"]
        self.short_out = cfg["exit_short"]
        self.trail_pct = cfg["trailing_pct"]

        self.tp_arm_margin = float(cfg.get("tp_arm_margin", 5))
        self.sl_arm_ratio  = float(cfg.get("sl_arm_ratio", 0.5))

        self._rsi_history: List[float] = []       # 最近两条 RSI 用于检测交叉
        self._current_rsi: Optional[float] = None

        # 持仓追踪 (由 Bot 在开仓时设置, 平仓时清除)
        self.in_position:  Optional[str] = None     # "long" / "short" / None
        self.entry_price:  float = 0.0
        self.peak_price:   float = 0.0              # 多头持仓期间最高价
        self.trough_price: float = 0.0              # 空头持仓期间最低价

    # ── RSI 计算 (Wilder smoothing) ────────────────────────────────
    def _wilders_rsi(self, closes: List[float]) -> float:
        """给定足够长的收盘价序列, 返回最新 RSI 值"""
        if len(closes) < self.period + 1:
            return 50.0  # 数据不足返回中性

        gains, losses = [], []
        for i in range(1, self.period + 1):
            diff = closes[i] - closes[i - 1]
            gains.append(diff if diff > 0 else 0)
            losses.append(-diff if diff < 0 else 0)

        avg_gain = sum(gains) / self.period
        avg_loss = sum(losses) / self.period

        for i in range(self.period + 1, len(closes)):
            diff = closes[i] - closes[i - 1]
            gain = diff if diff > 0 else 0
            loss = -diff if diff < 0 else 0
            avg_gain = (avg_gain * (self.period - 1) + gain) / self.period
            avg_loss = (avg_loss * (self.period - 1) + loss) / self.period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _rsi_at_price(self, closes: List[float], p: float) -> float:
        """假设下一收盘价为 p 时的 RSI 值"""
        return self._wilders_rsi(list(closes) + [float(p)])

    # ── 更新 K 线 ──────────────────────────────────────────────────
    def update(self, closes: List[float]):
        """传入完整收盘价序列, 更新最新 RSI 并检测交叉"""
        if len(closes) < self.period + 1:
            return None
        rsi = self._wilders_rsi(closes)
        self._current_rsi = rsi

        self._rsi_history.append(rsi)
        if len(self._rsi_history) > 2:
            self._rsi_history = self._rsi_history[-2:]

    # ── 入场信号 ───────────────────────────────────────────────────
    def entry_signal(self) -> Optional[str]:
        """返回 'long' / 'short' / None"""
        if len(self._rsi_history) < 2:
            return None
        prev_rsi, curr_rsi = self._rsi_history[-2], self._rsi_history[-1]

        if prev_rsi <= self.long_in and curr_rsi > self.long_in:
            return "long"
        if prev_rsi >= self.short_in and curr_rsi < self.short_in:
            return "short"
        return None

    # ── 止盈预挂: 是否已进入预挂区间 ──────────────────────────────
    @property
    def tp_armed(self) -> bool:
        if self.in_position is None or self._current_rsi is None:
            return False
        if self.in_position == "long":
            return self._current_rsi >= self.long_out - self.tp_arm_margin
        return self._current_rsi <= self.short_out + self.tp_arm_margin

    # ── 止盈预挂: 二分反解目标价格 ────────────────────────────────
    def predict_exit_price(self, closes: List[float]) -> Optional[float]:
        """
        返回"下一根收盘价 p 使 RSI 恰等于出场阈值"的价格。
        多头返回 exit_long 对应价 (上方), 空头返回 exit_short 对应价 (下方)。
        """
        if self.in_position is None or len(closes) < self.period + 1:
            return None

        target = self.long_out if self.in_position == "long" else self.short_out
        cur = self._wilders_rsi(closes)
        ref = closes[-1]

        # 已经越过阈值 → 返回当前价, 由调用方夹取到 maker 一侧
        if self.in_position == "long":
            if cur >= target:
                return ref
            lo, hi = ref, ref * 1.20
        else:
            if cur <= target:
                return ref
            lo, hi = ref * 0.80, ref

        # 扩展区间, 保证 target 落在 [RSI(lo), RSI(hi)] 内
        for _ in range(10):
            if self._rsi_at_price(closes, lo) <= target <= self._rsi_at_price(closes, hi):
                break
            if self.in_position == "long":
                hi *= 1.3
            else:
                lo *= 0.7

        for _ in range(120):
            mid = (lo + hi) / 2.0
            if self._rsi_at_price(closes, mid) < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    # ── 移动止损预挂信息 ──────────────────────────────────────────
    def stop_info(self, current_price: float):
        """返回 (armed, stop_price)。armed=True 表示接近止损, 应预挂条件单。"""
        if self.in_position is None:
            return False, 0.0

        arm = self.sl_arm_ratio * self.trail_pct

        if self.in_position == "long":
            if self.peak_price <= 0:
                return False, 0.0
            drawdown = (self.peak_price - current_price) / self.peak_price
            stop_price = self.peak_price * (1 - self.trail_pct)
            return drawdown >= arm, stop_price

        if self.trough_price <= 0:
            return False, 0.0
        rebound = (current_price - self.trough_price) / self.trough_price
        stop_price = self.trough_price * (1 + self.trail_pct)
        return rebound >= arm, stop_price

    # ── 更新持仓峰值/谷值 ──────────────────────────────────────────
    def update_peak_trough(self, current_price: float):
        if self.in_position == "long":
            if current_price > self.peak_price:
                self.peak_price = current_price
        elif self.in_position == "short":
            if self.trough_price == 0 or current_price < self.trough_price:
                self.trough_price = current_price

    # ── 重置持仓状态 ───────────────────────────────────────────────
    def clear_position(self):
        self.in_position  = None
        self.entry_price  = 0.0
        self.peak_price   = 0.0
        self.trough_price = 0.0

    @property
    def rsi(self) -> Optional[float]:
        return self._current_rsi


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                       OKX 交易执行层                                ║
# ╚══════════════════════════════════════════════════════════════════════╝

class OKXTradingBot:
    """OKX 永续合约交易执行 — 全部挂限价单 (入场 post-only, 出场限价/条件限价)"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.demo = cfg["demo"]

        flag = "1" if self.demo else "0"  # 1=模拟盘, 0=实盘
        proxy = cfg.get("proxy") or None
        timeout = cfg.get("timeout", 30)
        self._account_api = AccountAPI.AccountAPI(
            cfg["api_key"], cfg["api_secret"], cfg["passphrase"], False, flag,
            proxy=proxy)
        self._trade_api = TradeAPI.TradeAPI(
            cfg["api_key"], cfg["api_secret"], cfg["passphrase"], False, flag,
            proxy=proxy)
        self._market_api = MarketDataAPI.MarketAPI(
            cfg["api_key"], cfg["api_secret"], cfg["passphrase"], False, flag,
            proxy=proxy)
        self._public_api = PublicDataAPI.PublicAPI(
            cfg["api_key"], cfg["api_secret"], cfg["passphrase"], False, flag,
            proxy=proxy)
        for _api in (self._account_api, self._trade_api,
                     self._market_api, self._public_api):
            _api.timeout = timeout

        self.symbol    = cfg["symbol"]
        self.order_sz  = str(cfg["order_size"])
        self.max_pos   = cfg["max_position"]

        self.strategy = RSIStrategy(cfg)

        # 入场挂单追踪
        self._entry_order_id: Optional[str] = None
        self._entry_side:     Optional[str] = None
        self._entry_retry:    int = 0

        # 止盈挂单追踪 (普通 post-only 限价单)
        self._tp_order_id: Optional[str] = None
        self._tp_order_px: float = 0.0

        # 移动止损条件单追踪 (OKX algo order)
        self._sl_algo_id:  Optional[str] = None
        self._sl_stop_px:  float = 0.0

        # 合约信息 (运行时填充)
        self._tick_size: float = 0.1

        # 运行控制
        self._running = False
        self._closes: List[float] = []
        self._fetch_error_count = 0

    # ── 工具: 价格对齐到 tick ─────────────────────────────────────
    def _round_tick(self, px: float) -> float:
        tick = self._tick_size
        if tick <= 0:
            return px
        decimals = max(0, -int(round(math.log10(tick))))
        return round(round(px / tick) * tick, decimals)

    # ── 初始化 ─────────────────────────────────────────────────────
    async def initialize(self):
        while True:
            try:
                log.info("正在获取合约信息...")
                resp = self._public_api.get_instruments(
                    instType="SWAP", instId=self.symbol)
                if resp.get("code") != "0":
                    raise RuntimeError(f"获取合约信息失败: {resp}")
                inst = resp["data"][0]
                self._tick_size = float(inst["tickSz"])
                log.info(f"合约: {self.symbol}  |  tick: {self._tick_size}  |  "
                         f"ctVal: {inst.get('ctVal', '?')}")
                await self._fetch_and_update_closes()
                return
            except Exception as e:
                log.error(f"初始化异常: {e}  |  30 秒后重试初始化...")
                await asyncio.sleep(30)

    async def _fetch_and_update_closes(self):
        try:
            resp = self._market_api.get_candlesticks(
                instId=self.symbol, bar="15m", limit=str(self.cfg["candle_limit"]))
            if resp.get("code") != "0":
                log.warning(f"获取K线失败: {resp}")
                return
            candles = resp["data"]
            closes = [float(c[4]) for c in reversed(candles)]  # oldest → newest
            self._closes = closes
            self.strategy.update(closes)
            if self._fetch_error_count:
                log.info(f"K线获取恢复正常, 连续异常计数清零 (此前 {self._fetch_error_count} 次)")
            self._fetch_error_count = 0
            log.info(f"K 线更新完成, {len(closes)} 根 | 最新 RSI={self.strategy.rsi:.1f}")
        except Exception as e:
            self._fetch_error_count += 1
            log.error(f"获取K线异常: {e}  (连续第 {self._fetch_error_count} 次)")
            if self._fetch_error_count >= self.cfg.get("max_fetch_errors", 10):
                log.error(f"连续 {self._fetch_error_count} 次获取K线异常, 达到阈值, "
                          f"即将重新运行程序主体...")
                restart_program()

    # ── 行情 / 持仓 / 挂单查询 ────────────────────────────────────
    def _get_current_price(self) -> Optional[float]:
        try:
            resp = self._market_api.get_ticker(instId=self.symbol)
            if resp.get("code") == "0" and resp["data"]:
                return float(resp["data"][0]["last"])
        except Exception as e:
            log.error(f"获取行情异常: {e}")
        return None

    def _get_position(self) -> Optional[Dict[str, Any]]:
        try:
            resp = self._account_api.get_positions(instId=self.symbol)
            if resp.get("code") != "0":
                return None
            for pos in resp.get("data", []):
                qty = float(pos.get("pos", 0))
                if qty != 0:
                    return {"side": "long" if qty > 0 else "short",
                            "qty": abs(qty),
                            "avg_px": float(pos["avgPx"]),
                            "upl": float(pos.get("upl", 0))}
        except Exception as e:
            log.error(f"查询持仓异常: {e}")
        return None

    def _get_pending_orders(self) -> List[Dict]:
        try:
            resp = self._trade_api.get_order_list(instId=self.symbol, state="live")
            if resp.get("code") != "0":
                return []
            return resp.get("data", [])
        except Exception as e:
            log.error(f"查询挂单异常: {e}")
            return []

    # ── 下单 / 撤单 / 检查成交 ────────────────────────────────────
    def _place_limit(self, side: str, price: float, sz=None,
                     reduce_only: bool = False) -> Optional[str]:
        """post-only 限价单。sz 缺省用 order_size。"""
        px = self._round_tick(price)
        sz_s = str(sz) if sz is not None else self.order_sz
        ro = "true" if reduce_only else ""
        try:
            resp = self._trade_api.place_order(
                instId=self.symbol,
                tdMode="cross",
                side=side,
                ordType="post_only",
                sz=sz_s,
                px=str(px),
                reduceOnly=ro)
            if resp.get("code") == "0":
                ord_id = resp["data"][0]["ordId"]
                log.info(f"挂限价单: {side.upper()} {sz_s}张 @ {px} id={ord_id}")
                return ord_id
            log.error(f"挂限价单失败: {resp}")
            return None
        except Exception as e:
            log.error(f"挂限价单异常: {e}")
            return None

    def _place_stop_algo(self, side: str, trigger_px: float, order_px: float,
                         qty: float) -> Optional[str]:
        """移动止损条件单: 价格触及 trigger_px 时挂出 order_px 的限价单 (reduceOnly)。"""
        try:
            resp = self._trade_api.place_algo_order(
                instId=self.symbol,
                tdMode="cross",
                side=side,
                ordType="conditional",
                sz=str(qty),
                reduceOnly="true",
                triggerPx=str(trigger_px),
                orderPx=str(order_px))
            if resp.get("code") == "0":
                algo_id = resp["data"][0]["algoId"]
                log.info(f"挂止损条件单: {side.upper()} {qty}张 trigger={trigger_px} "
                         f"order={order_px} algoId={algo_id}")
                return algo_id
            log.error(f"挂止损条件单失败: {resp}")
            return None
        except Exception as e:
            log.error(f"挂止损条件单异常: {e}")
            return None

    def _cancel_order_id(self, ord_id: str) -> bool:
        try:
            resp = self._trade_api.cancel_order(instId=self.symbol, ordId=ord_id)
            ok = resp.get("code") == "0"
            if ok:
                log.info(f"撤单成功: ordId={ord_id}")
            else:
                log.warning(f"撤单失败 ordId={ord_id}, resp={resp}")
            return ok
        except Exception as e:
            log.error(f"撤单异常 ordId={ord_id}: {e}")
            return False

    def _cancel_algo_id(self, algo_id: str) -> bool:
        try:
            resp = self._trade_api.cancel_algo_order(
                [{"instId": self.symbol, "algoId": algo_id}])
            ok = resp.get("code") == "0"
            if ok:
                log.info(f"撤条件单成功: algoId={algo_id}")
            else:
                log.warning(f"撤条件单失败 algoId={algo_id}, resp={resp}")
            return ok
        except Exception as e:
            log.error(f"撤条件单异常 algoId={algo_id}: {e}")
            return False

    def _check_order_filled(self, ord_id: str) -> bool:
        try:
            resp = self._trade_api.get_order(instId=self.symbol, ordId=ord_id)
            if resp.get("code") != "0":
                return False
            return resp["data"][0]["state"] == "filled"
        except Exception:
            return False

    # ── 入场下单 ─────────────────────────────────────────────────────
    def _place_entry(self, direction: str):
        price = self._get_current_price()
        if price is None:
            log.error("无法获取价格, 跳过入场")
            return

        offset = price * self.cfg["limit_offset_bps"] / 10000.0
        if direction == "long":
            px = price - offset
            side = "buy"
        else:
            px = price + offset
            side = "sell"

        ord_id = self._place_limit(side, px)
        if ord_id:
            self._entry_order_id = ord_id
            self._entry_side = direction
            self._entry_retry = 0

    # ── 止盈挂单 ─────────────────────────────────────────────────────
    def _place_tp(self, side: str, px: float, qty: float):
        close_side = "sell" if side == "long" else "buy"
        ord_id = self._place_limit(close_side, px, sz=qty, reduce_only=True)
        if ord_id:
            self._tp_order_id = ord_id
            self._tp_order_px = px

    def _cancel_tp(self):
        if self._tp_order_id:
            self._cancel_order_id(self._tp_order_id)
            self._tp_order_id = None
            self._tp_order_px = 0.0

    # ── 移动止损挂单 ─────────────────────────────────────────────────
    def _place_sl(self, side: str, trigger: float, order_px: float, qty: float):
        close_side = "sell" if side == "long" else "buy"
        algo_id = self._place_stop_algo(close_side, trigger, order_px, qty)
        if algo_id:
            self._sl_algo_id = algo_id
            self._sl_stop_px = trigger

    def _cancel_sl(self):
        if self._sl_algo_id:
            self._cancel_algo_id(self._sl_algo_id)
            self._sl_algo_id = None
            self._sl_stop_px = 0.0

    # ── 管理出场挂单 (止盈预挂 + 移动止损预挂) ─────────────────────
    def _manage_exit_orders(self, pos: Dict[str, Any], price: float):
        side = pos["side"]
        qty = pos["qty"]
        closes = self._closes
        replace_pct = self.cfg.get("replace_pct", 0.001)
        guard_pct = self.cfg.get("sl_guard_pct", 0.001)

        # ── 1) RSI 止盈预挂 ──────────────────────────────────────
        tp_px: Optional[float] = None
        if self.strategy.tp_armed:
            tp_px = self.strategy.predict_exit_price(closes)

        if tp_px is not None:
            # 保证 post-only 挂在对己方有利一侧 (多头卖单 ≥ 当前价, 空头买单 ≤ 当前价)
            if side == "long":
                tp_px = max(tp_px, price)
            else:
                tp_px = min(tp_px, price)
            tp_px = self._round_tick(tp_px)

            if self._tp_order_id is None:
                self._place_tp(side, tp_px, qty)
            elif self._tp_order_px and \
                    abs(tp_px - self._tp_order_px) / self._tp_order_px >= replace_pct:
                log.info(f"止盈预测价漂移 {abs(tp_px - self._tp_order_px) / self._tp_order_px:.4%}, 撤单重挂")
                self._cancel_tp()
                self._place_tp(side, tp_px, qty)
        else:
            self._cancel_tp()

        # ── 2) 移动止损预挂 ──────────────────────────────────────
        sl_armed, sl_stop_px = self.strategy.stop_info(price)
        if sl_armed and sl_stop_px > 0:
            sl_stop_px = self._round_tick(sl_stop_px)
            order_px = sl_stop_px * (1 - guard_pct) if side == "long" \
                else sl_stop_px * (1 + guard_pct)
            order_px = self._round_tick(order_px)

            if self._sl_algo_id is None:
                self._place_sl(side, sl_stop_px, order_px, qty)
            elif self._sl_stop_px and \
                    abs(sl_stop_px - self._sl_stop_px) / self._sl_stop_px >= replace_pct:
                log.info(f"止损线漂移 {abs(sl_stop_px - self._sl_stop_px) / self._sl_stop_px:.4%}, 撤单重挂")
                self._cancel_sl()
                self._place_sl(side, sl_stop_px, order_px, qty)
        else:
            self._cancel_sl()

    # ── 主循环的一步 ─────────────────────────────────────────────────
    async def _tick(self):
        await self._fetch_and_update_closes()
        if len(self._closes) < self.cfg["rsi_period"] + 1:
            log.warning("K 线数据不足, 跳过本周期")
            return

        price = self._get_current_price()
        if price is None:
            return

        pos = self._get_position()
        log.info(f"持仓状态: {pos} | 最新价: {price}")

        # ── 情况 A: 有入场挂单待成交 ──────────────────────────────
        if self._entry_order_id:
            if self._check_order_filled(self._entry_order_id):
                log.info(f"入场单成交: {self._entry_order_id}")
                self._entry_order_id = None
                self._entry_side = None
                self._entry_retry = 0
                pos = self._get_position()
                if pos:
                    self.strategy.in_position  = pos["side"]
                    self.strategy.entry_price  = pos["avg_px"]
                    self.strategy.peak_price   = pos["avg_px"]
                    self.strategy.trough_price = pos["avg_px"]
                    log.info(f"开仓成功: {pos['side']} @ {pos['avg_px']:.2f}")
            else:
                elapsed = self._entry_retry * self.cfg["poll_interval"]
                if elapsed >= self.cfg["order_timeout"]:
                    if self._entry_retry >= self.cfg["max_retry"]:
                        log.warning(f"入场挂单重试 {self._entry_retry} 次仍未成交, 放弃信号")
                        self._cancel_order_id(self._entry_order_id)
                        self._entry_order_id = None
                        self._entry_side = None
                        self._entry_retry = 0
                    else:
                        log.info(f"入场挂单超时 ({elapsed}s), 撤单重挂 "
                                 f"({self._entry_retry+1}/{self.cfg['max_retry']})")
                        side = self._entry_side
                        self._cancel_order_id(self._entry_order_id)
                        self._entry_order_id = None
                        if side:
                            self._place_entry(side)
                            self._entry_retry += 1
                else:
                    self._entry_retry += 1
            return

        # ── 情况 B: 有持仓 → 管理出场挂单 ────────────────────────
        if pos:
            side = pos["side"]
            self.strategy.in_position = side
            self.strategy.update_peak_trough(price)
            self._manage_exit_orders(pos, price)
            return

        # ── 持仓刚被平掉 (TP / SL / 外部) → 清理残留挂单 ────────
        if self.strategy.in_position is not None:
            log.info("检测到持仓已平, 清理残留出场挂单")
            self._cancel_tp()
            self._cancel_sl()
            self.strategy.clear_position()

        # ── 情况 C: 空仓, 检查入场信号 ────────────────────────────
        signal = self.strategy.entry_signal()
        if signal:
            log.info(f"入场信号: {signal.upper()} | RSI={self.strategy.rsi:.1f} | "
                     f"price={price:.2f}")
            self._place_entry(signal)

    # ── 主循环 ──────────────────────────────────────────────────────
    async def run(self):
        await self.initialize()

        log.info("=" * 60)
        log.info(f"策略启动: RSI({self.cfg['rsi_period']}) 均值回归")
        log.info(f"品种: {self.symbol}  |  K线: 15m  |  模式: "
                 f"{'模拟盘' if self.demo else '★实盘★'}")
        log.info(f"每单: {self.order_sz} 张  |  最大持仓: {self.max_pos} 张")
        log.info(f"多头入场: RSI 上穿 {self.cfg['entry_long']} | "
                 f"止盈预挂 RSI≥{self.cfg['exit_long']-self.cfg.get('tp_arm_margin', 5)} "
                 f"(目标 RSI={self.cfg['exit_long']})")
        log.info(f"空头入场: RSI 下穿 {self.cfg['entry_short']} | "
                 f"止盈预挂 RSI≤{self.cfg['exit_short']+self.cfg.get('tp_arm_margin', 5)} "
                 f"(目标 RSI={self.cfg['exit_short']})")
        log.info(f"移动止损: 涨跌幅 {self.cfg['trailing_pct']*100:.2f}% | "
                 f"预挂启动 {self.cfg.get('sl_arm_ratio', 0.5)*self.cfg['trailing_pct']*100:.2f}%")
        log.info(f"重挂漂移阈值: {self.cfg.get('replace_pct', 0.001)*100:.1f}%")
        log.info(f"费率: 全流程仅挂单 (入场 post-only)")
        log.info("=" * 60)

        self._running = True
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                log.error(f"主循环异常: {e}", exc_info=True)
            await asyncio.sleep(self.cfg["poll_interval"])

    def stop(self):
        log.info("收到停止信号, 正在退出...")
        self._running = False
        if self._entry_order_id:
            self._cancel_order_id(self._entry_order_id)
        self._cancel_tp()
        self._cancel_sl()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                           入口                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

log = logging.getLogger("rsi_bot")
log.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s  [%(levelname)-5s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"))
log.handlers.clear()
log.addHandler(_handler)


def validate_config(cfg: dict):
    errors = []
    if not cfg["api_key"]:
        errors.append("api_key 为空 — 请在 config.json 中填写 OKX API Key")
    if not cfg["api_secret"]:
        errors.append("api_secret 为空 — 请在 config.json 中填写 OKX Secret Key")
    if not cfg["passphrase"]:
        errors.append("passphrase 为空 — 请在 config.json 中填写 OKX Passphrase")
    if cfg["order_size"] < 0.01:
        errors.append(f"order_size ({cfg['order_size']}) < 最小下单量")
    if errors:
        for e in errors:
            log.error(e)
        return False
    return True


def restart_program():
    log.info("正在重新启动程序...")
    try:
        cmd = [sys.executable, os.path.abspath(__file__)]
        cmd.extend(sys.argv[1:])
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            creationflags = 0
        subprocess.Popen(cmd, creationflags=creationflags)
    except Exception as e:
        log.error(f"重新启动程序失败: {e}")
        return
    os._exit(0)


async def main():
    if not validate_config(CONFIG):
        log.warning("配置校验未通过, 请修改后重试。")
        return

    bot = OKXTradingBot(CONFIG)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, bot.stop)
        except NotImplementedError:
            pass

    try:
        await bot.run()
    except KeyboardInterrupt:
        bot.stop()
    finally:
        log.info("机器人已停止。")


if __name__ == "__main__":
    asyncio.run(main())