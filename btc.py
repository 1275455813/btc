#!/usr/bin/env python3
"""
BTC-USDT 永续合约 RSI(14) 均值回归策略 — 纯做市商（Post-Only）实盘机器人
=========================================================================
品种:     BTC-USDT-SWAP (OKX 永续)
K线:      15 分钟
费率:     做市商 0.02% (post-only 限价单保证)
下单方式: 仅挂单 (post-only limit) — 永不主动吃单

策略逻辑 (回测验证: 2026-07-12 ~ 2026-08-11, 胜率 65.7%, 回报 +1.43%)
  做多: RSI(14) 从下方上穿 30
  做空: RSI(14) 从上方下穿 70
  多头平仓: RSI(14) ≥ 65  或  从持仓最高点回落 ≥ 2.0%
  空头平仓: RSI(14) ≤ 35  或  从持仓最低点反弹 ≥ 2.0%

依赖: pip install python-okx
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_DOWN, ROUND_UP
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
# ║                    ★ 配置区 — 请填写你的密钥 ★                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

CONFIG = {
    # ── OKX API 密钥 (在 https://www.okx.com/account/my-api 创建) ──
    "api_key":     "230a0e34-a550-4e50-ad66-4d19b55630c6",           # ← 填写你的 API Key
    "api_secret":  "3316946829E8A838C5E103A770EC1F96",           # ← 填写你的 Secret Key
    "passphrase":  "577320179aA.",           # ← 填写你的 Passphrase

    # ── 交易模式 ──
    "demo":        False,         # True=模拟盘(需要 demo API key), False=实盘
    "symbol":      "BTC-USDT-SWAP",

    # ── 仓位与风控 ──
    "order_size":  0.05,            # 每单合约张数 (1张 = 0.01 BTC,  100张 = 1 BTC)
    "max_position": 0.1,           # 最大持仓张数 (同向)

    # ── 策略参数 (与回测一致) ──
    "rsi_period":  14,           # RSI 周期
    "entry_long":  30,           # 做多入场阈值 (RSI 上穿此值)
    "entry_short": 70,           # 做空入场阈值 (RSI 下穿此值)
    "exit_long":   65,           # 多头 RSI 出场阈值
    "exit_short":  35,           # 空头 RSI 出场阈值
    "trailing_pct": 0.02,        # 移动止损 2.0%

    # ── 挂单参数 ──
    "limit_offset_bps": 1.0,     # 限价单偏移 (bps, 1.0 = 万分之一, 挂在我们有利的方向)
    "order_timeout":   120,      # 挂单超时秒数, 超时未成交自动撤单重挂
    "max_retry":       5,        # 同方向最大重挂次数, 超过则放弃本信号

    # ── 运行 ──
    "poll_interval": 15,         # 轮询间隔 (秒)
    "candle_limit":  200,        # 初始加载 K 线条数
}

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                         策略核心引擎                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝

class RSIStrategy:
    """RSI(14) 均值回归 — 信号生成 & 出场判断"""

    def __init__(self, cfg: dict):
        self.period    = cfg["rsi_period"]
        self.long_in   = cfg["entry_long"]
        self.short_in  = cfg["entry_short"]
        self.long_out  = cfg["exit_long"]
        self.short_out = cfg["exit_short"]
        self.trail_pct = cfg["trailing_pct"]

        self._gains: List[float] = []
        self._losses: List[float] = []
        self._prev_close: Optional[float] = None
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

    # ── 更新 K 线 ──────────────────────────────────────────────────
    def update(self, closes: List[float]):
        """传入完整收盘价序列, 检查信号"""
        if len(closes) < self.period + 1:
            return None

        rsi = self._wilders_rsi(closes)
        self._current_rsi = rsi

        # 保留最近 2 个 RSI 用于交叉检测
        self._rsi_history.append(rsi)
        if len(self._rsi_history) > 2:
            self._rsi_history = self._rsi_history[-2:]

    # ── 入场信号 ───────────────────────────────────────────────────
    def entry_signal(self) -> Optional[str]:
        """返回 'long' / 'short' / None"""
        if len(self._rsi_history) < 2:
            return None
        prev_rsi, curr_rsi = self._rsi_history[-2], self._rsi_history[-1]

        # 做多: RSI 从 ≤30 上穿到 >30
        if prev_rsi <= self.long_in and curr_rsi > self.long_in:
            return "long"
        # 做空: RSI 从 ≥70 下穿到 <70
        if prev_rsi >= self.short_in and curr_rsi < self.short_in:
            return "short"
        return None
        #return "long"

    # ── 出场信号 ───────────────────────────────────────────────────
    def exit_signal(self, current_price: float) -> bool:
        """
        返回 True = 需要平仓。
        调用方需先检查 self.in_position。
        """
        if self.in_position is None or self._current_rsi is None:
            return False

        if self.in_position == "long":
            # 条件 1: RSI ≥ 65
            if self._current_rsi >= self.long_out:
                return True
            # 条件 2: 从最高点回落 ≥ 2.0%
            if self.peak_price > 0:
                drawdown = (self.peak_price - current_price) / self.peak_price
                if drawdown >= self.trail_pct:
                    return True

        elif self.in_position == "short":
            # 条件 1: RSI ≤ 35
            if self._current_rsi <= self.short_out:
                return True
            # 条件 2: 从最低点反弹 ≥ 2.0%
            if self.trough_price > 0:
                rebound = (current_price - self.trough_price) / self.trough_price
                if rebound >= self.trail_pct:
                    return True

        return False

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
    """OKX 永续合约交易执行 — 纯 Post-Only 限价单"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.demo = cfg["demo"]

        # 初始化 OKX API 客户端
        flag = "1" if self.demo else "0"  # 1=模拟盘, 0=实盘
        self._account_api    = AccountAPI.AccountAPI(
            cfg["api_key"], cfg["api_secret"], cfg["passphrase"], False, flag)
        self._trade_api      = TradeAPI.TradeAPI(
            cfg["api_key"], cfg["api_secret"], cfg["passphrase"], False, flag)
        self._market_api     = MarketDataAPI.MarketAPI(
            cfg["api_key"], cfg["api_secret"], cfg["passphrase"], False, flag)
        self._public_api     = PublicDataAPI.PublicAPI(
            cfg["api_key"], cfg["api_secret"], cfg["passphrase"], False, flag)

        self.symbol    = cfg["symbol"]
        self.order_sz  = str(cfg["order_size"])
        self.max_pos   = cfg["max_position"]

        # 策略引擎
        self.strategy = RSIStrategy(cfg)

        # 订单管理
        self._pending_order_id: Optional[str] = None
        self._pending_side:     Optional[str] = None
        self._retry_count:      int = 0

        # 合约信息 (运行时填充)
        self._tick_size:  float = 0.1
        self._lot_size:   int   = 1
        self._min_size:   int   = 1

        # 运行控制
        self._running = False
        self._closes: List[float] = []

    # ── 初始化: 获取合约信息 + 加载历史K线 ────────────────────────
    async def initialize(self):
        """获取合约规格并预热 RSI"""
        log.info("正在获取合约信息...")
        try:
            resp = self._public_api.get_instruments(
                instType="SWAP", instId=self.symbol)
            if resp.get("code") != "0":
                raise RuntimeError(f"获取合约信息失败: {resp}")
            inst = resp["data"][0]
            self._tick_size = float(inst["tickSz"])
            self._lot_size  = int(float(inst["lotSz"]))
            self._min_size  = int(float(inst["minSz"]))
            log.info(f"合约: {self.symbol}  |  tick: {self._tick_size}  |  "
                     f"lot: {self._lot_size}  |  ctVal: {inst.get('ctVal', '?')}")
        except Exception as e:
            log.error(f"获取合约信息异常: {e}")
            raise

        # 预热 K 线
        await self._fetch_and_update_closes()

    async def _fetch_and_update_closes(self):
        """从 REST 拉取历史 K 线并更新策略"""
        try:
            resp = self._market_api.get_candlesticks(
                instId=self.symbol,
                bar="15m",
                limit=str(self.cfg["candle_limit"]))
            if resp.get("code") != "0":
                log.warning(f"获取K线失败: {resp}")
                return
            # OKX 返回顺序: [ts, o, h, l, c, vol, ...] 最新在前
            candles = resp["data"]
            closes = [float(c[4]) for c in reversed(candles)]
            self._closes = closes
            self.strategy.update(closes)
            log.info(f"K 线预热完成, {len(closes)} 根 | 最新 RSI={self.strategy.rsi:.1f}")
        except Exception as e:
            log.error(f"获取K线异常: {e}")

    # ── 获取最新价格 ────────────────────────────────────────────────
    def _get_current_price(self) -> Optional[float]:
        """获取最新成交价"""
        try:
            resp = self._market_api.get_ticker(instId=self.symbol)
            if resp.get("code") == "0" and resp["data"]:
                return float(resp["data"][0]["last"])
        except Exception as e:
            log.error(f"获取行情异常: {e}")
        return None

    # ── 查询持仓 ─────────────────────────────────────────────────────
    def _get_position(self) -> Optional[Dict[str, Any]]:
        """返回当前 BTC-USDT-SWAP 持仓, 无持仓返回 None"""
        try:
            resp = self._account_api.get_positions(instId=self.symbol)
            if resp.get("code") != "0":
                return None
            for pos in resp.get("data", []):
                qty = float(pos.get("pos", 0))
                log.info(f"debug1, {qty}")
                dbug1 = pos["posSide"]
                log.info(f"debug2, {dbug1}")
                if qty != 0:
                    return {"side": "long" if qty > 0 else "short",
                            "qty": abs(qty),
                            "avg_px": float(pos["avgPx"]),
                            "upl": float(pos.get("upl", 0))}
        except Exception as e:
            log.error(f"查询持仓异常: {e}")
        return None

    # ── 查询挂单 ─────────────────────────────────────────────────────
    def _get_pending_orders(self) -> List[Dict]:
        try:
            resp = self._trade_api.get_order_list(instId=self.symbol, state="live")
            if resp.get("code") != "0":
                return []
            return resp.get("data", [])
        except Exception as e:
            log.error(f"查询挂单异常: {e}")
            return []

    # ── 撤单 ─────────────────────────────────────────────────────────
    def _cancel_all_orders(self) -> bool:
        """取消所有挂单。返回 True 表示全部取消成功; False 表示有订单取消失败。
           仅当全部成功时才清除内部 tracking 状态, 防止取消失败后重复下单。"""
        orders = self._get_pending_orders()
        if not orders:
            self._pending_order_id = None
            self._pending_side = None
            return True
        all_cancelled = True
        for o in orders:
            try:
                resp = self._trade_api.cancel_order(
                    instId=self.symbol, ordId=o["ordId"])
                if resp.get("code") == "0":
                    log.info(f"撤单成功: ordId={o['ordId']} side={o.get('side')} sz={o.get('sz')}张")
                else:
                    log.warning(f"撤单业务失败 ordId={o['ordId']}, resp={resp}")
                    all_cancelled = False
            except Exception as e:
                log.error(f"撤单异常 ordId={o['ordId']}: {e}")
                all_cancelled = False
        if all_cancelled:
            self._pending_order_id = None
            self._pending_side = None
        return all_cancelled

    # ── 下限价单 (Post-Only) ────────────────────────────────────────
    def _place_limit(self, side: str, price: float) -> Optional[str]:
        """
        下限价单 (post-only)。
        side: 'buy' or 'sell'
        price: 限价
        返回 orderId or None
        """
        # 规范化价格到 tick
        tick = self._tick_size
        px = round(round(price / tick) * tick, max(0, -int(round(__import__('math').log10(tick)))))
        sz = str(self.order_sz)

        try:
            resp = self._trade_api.place_order(
                instId=self.symbol,
                tdMode="cross",           # 全仓
                side=side,
                ordType="post_only",      # ← 关键: 纯做市商
                sz=sz,
                px=str(px))
            if resp.get("code") == "0":
                ord_id = resp["data"][0]["ordId"]
                log.info(f"挂单: {side.upper()} {sz}张 @ {px}  (post-only)  id={ord_id}")
                return ord_id
            else:
                log.error(f"下单失败: {resp}")
                return None
        except Exception as e:
            log.error(f"下单异常: {e}")
            return None

    # ── 市价平仓 (紧急情况也允许 taker) ────────────────────────────
    def _close_position(self, side: str):
        """平掉当前持仓 (使用限价单以保持 maker 费率)"""
        price = self._get_current_price()
        if price is None:
            log.error("无法获取价格, 平仓失败")
            return False

        # 平仓方向: long 持仓 → sell 平仓; short 持仓 → buy 平仓
        close_side = "sell" if side == "long" else "buy"
        # Post-only 平仓: 挂对己方有利的价格
        offset = price * self.cfg["limit_offset_bps"] / 10000.0
        if close_side == "sell":
            px = price + offset  # 卖单挂高一点
        else:
            px = price - offset  # 买单挂低一点

        ord_id = self._place_limit(close_side, px)
        if ord_id:
            self._pending_order_id = ord_id
            self._pending_side = close_side
            self._retry_count = 0
            return True
        return False

    # ── 入场下单 ─────────────────────────────────────────────────────
    def _place_entry(self, direction: str):
        """direction: 'long' 或 'short'"""
        price = self._get_current_price()
        if price is None:
            log.error("无法获取价格, 跳过入场")
            return

        # 偏移量让我们成为 maker
        offset = price * self.cfg["limit_offset_bps"] / 10000.0

        if direction == "long":
            px = price - offset  # 买入价低于当前价 → maker
        else:
            px = price + offset  # 卖出价高于当前价 → maker

        ord_id = self._place_limit("buy" if direction == "long" else "sell", px)
        if ord_id:
            self._pending_order_id = ord_id
            self._pending_side = direction
            self._retry_count = 0

    # ── 检查挂单是否成交 ────────────────────────────────────────────
    def _check_order_filled(self, ord_id: str) -> bool:
        try:
            resp = self._trade_api.get_order(instId=self.symbol, ordId=ord_id)
            if resp.get("code") != "0":
                return False
            state = resp["data"][0]["state"]
            return state == "filled"
        except Exception:
            return False

    # ── 主循环的一步 ─────────────────────────────────────────────────
    async def _tick(self):
        """每个轮询周期执行一次"""
        # 1. 刷新 K 线 & RSI
        await self._fetch_and_update_closes()
        if len(self._closes) < self.cfg["rsi_period"] + 1:
            log.warning("K 线数据不足, 跳过本周期")
            return

        # 2. 获取当前价格
        price = self._get_current_price()
        if price is None:
            return

        # 3. 同步持仓状态
        pos = self._get_position()
        log.info(f"持仓状态: {pos}")
        # ── 情况 A: 有挂单待成交 ──────────────────────────────────
        if self._pending_order_id:
            if self._check_order_filled(self._pending_order_id):
                log.info(f"订单成交: {self._pending_order_id}")
                self._pending_order_id = None
                # 检查新持仓
                pos = self._get_position()
                if pos:
                    self.strategy.in_position  = pos["side"]
                    self.strategy.entry_price  = pos["avg_px"]
                    self.strategy.peak_price   = pos["avg_px"]
                    self.strategy.trough_price = pos["avg_px"]
                    log.info(f"开仓成功: {pos['side']} @ {pos['avg_px']:.2f}")
                self._retry_count = 0
                self._pending_side = None
            else:
                # 挂单超时处理
                elapsed = self._retry_count * self.cfg["poll_interval"]
                if elapsed >= self.cfg["order_timeout"]:
                    if self._retry_count >= self.cfg["max_retry"]:
                        log.warning(f"挂单重试 {self._retry_count} 次仍未成交, 放弃信号")
                        if not self._cancel_all_orders():
                            log.warning("撤单未完全成功, 但已放弃信号 — 请手动检查挂单")
                        self._retry_count = 0
                    else:
                        log.info(f"挂单超时 ({elapsed}s), 撤单重挂 ({self._retry_count+1}/{self.cfg['max_retry']})")
                        if self._cancel_all_orders():
                            side = self._pending_side
                            self._pending_side = None
                            self._place_entry(side)
                            self._retry_count += 1
                        else:
                            log.warning("撤单未完全成功, 跳过重挂以避免重复下单")
                else:
                    self._retry_count += 1
            return  # 有挂单时不处理新信号

        # ── 情况 B: 有持仓 ─────────────────────────────────────────
        if pos:
            side = pos["side"]
            self.strategy.in_position = side

            # 更新峰值/谷值
            self.strategy.update_peak_trough(price)

            # 检查出场条件
            if self.strategy.exit_signal(price):
                rsi_val = self.strategy.rsi
                log.info(f"出场信号触发: {side} | RSI={rsi_val:.1f} | price={price:.2f} | "
                         f"peak={self.strategy.peak_price:.2f} trough={self.strategy.trough_price:.2f}")
                if not self._cancel_all_orders():
                    log.warning("撤单未完全成功, 跳过平仓下单以避免重复委托")
                else:
                    self._close_position(side)
                    self.strategy.clear_position()
            return

        # ── 情况 C: 空仓, 检查入场信号 ────────────────────────────
        signal = self.strategy.entry_signal()
        if signal:
            log.info(f"入场信号: {signal.upper()} | RSI={self.strategy.rsi:.1f} | price={price:.2f}")
            if not self._cancel_all_orders():
                log.warning("撤单未完全成功, 跳过入场以避免重复委托")
            else:
                self._place_entry(signal)

    # ── 主循环 ──────────────────────────────────────────────────────
    async def run(self):
        await self.initialize()

        log.info("=" * 60)
        log.info(f"策略启动: RSI({self.cfg['rsi_period']}) 均值回归")
        log.info(f"品种: {self.symbol}  |  K线: 15m  |  模式: {'模拟盘' if self.demo else '★实盘★'}")
        log.info(f"每单: {self.order_sz} 张  |  最大持仓: {self.max_pos} 张")
        log.info(f"做多: RSI<{self.cfg['entry_long']} → 平仓 RSI>{self.cfg['exit_long']} 或 -{self.cfg['trailing_pct']*100:.0f}%")
        log.info(f"做空: RSI>{self.cfg['entry_short']} → 平仓 RSI<{self.cfg['exit_short']} 或 +{self.cfg['trailing_pct']*100:.0f}%")
        log.info(f"费率: 做市商 0.02% (post-only)")
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
        # 不强制平仓 — 用户可以手动管理剩余仓位
        self._cancel_all_orders()


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                           入口                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

# ── 日志 ──────────────────────────────────────────────────────────────
log = logging.getLogger("rsi_bot")
log.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s  [%(levelname)-5s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"))
log.handlers.clear()
log.addHandler(_handler)


def validate_config(cfg: dict):
    """启动前校验配置"""
    errors = []
    if not cfg["api_key"]:
        errors.append("api_key 为空 — 请在脚本顶部 CONFIG 中填写 OKX API Key")
    if not cfg["api_secret"]:
        errors.append("api_secret 为空 — 请在脚本顶部 CONFIG 中填写 OKX Secret Key")
    if not cfg["passphrase"]:
        errors.append("passphrase 为空 — 请在脚本顶部 CONFIG 中填写 OKX Passphrase")
    if cfg["order_size"] < 0.01:
        errors.append(f"order_size ({cfg['order_size']}) < 最小下单量")
    if errors:
        for e in errors:
            log.error(e)
        return False
    return True


async def main():
    if not validate_config(CONFIG):
        log.warning("配置校验未通过, 请修改后重试。")
        return

    bot = OKXTradingBot(CONFIG)

    # 信号处理: Ctrl+C 优雅退出
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, bot.stop)
        except NotImplementedError:
            # Windows 不支持 add_signal_handler
            pass

    try:
        await bot.run()
    except KeyboardInterrupt:
        bot.stop()
    finally:
        log.info("机器人已停止。")


if __name__ == "__main__":
    asyncio.run(main())
