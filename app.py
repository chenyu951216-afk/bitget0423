from __future__ import annotations

import math
import os
import threading
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List

import ccxt
import pandas as pd
import pandas_ta as ta
import requests
from flask import Flask, jsonify, render_template, request

from api_state_routes import (
    build_ai_panel_payload,
    build_positions_payload,
    build_state_lite_payload,
)
from dashboard_state import ai_panel_cache, positions_cache, state_lite_cache
from openai_trade_decision import (
    build_candidate_payload as build_openai_trade_candidate,
    build_dashboard_payload as build_openai_trade_dashboard,
    consult_trade_decision,
    default_trade_config,
    load_trade_state,
    save_trade_state,
)
from state_service import DEFAULT_RUNTIME_STATE, env_or_blank


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OPENAI_TRADE_STATE_PATH = os.path.join(DATA_DIR, "openai_trade_state.json")
SNAPSHOT_STATE_PATH = os.path.join(DATA_DIR, "runtime_snapshot.json")

SCAN_INTERVAL_SEC = max(45, int(float(env_or_blank("SCAN_INTERVAL_SEC", "75") or 75)))
POSITION_INTERVAL_SEC = max(15, int(float(env_or_blank("POSITION_INTERVAL_SEC", "25") or 25)))
OPENAI_SYNC_INTERVAL_SEC = max(20, int(float(env_or_blank("OPENAI_SYNC_INTERVAL_SEC", "30") or 30)))
SCAN_SYMBOL_LIMIT = max(4, min(12, int(float(env_or_blank("SCAN_SYMBOL_LIMIT", "6") or 6))))
TOP_SIGNAL_LIMIT = max(4, min(12, int(float(env_or_blank("TOP_SIGNAL_LIMIT", "8") or 8))))
TIMEFRAME_BAR_LIMIT = max(100, min(240, int(float(env_or_blank("SCAN_BAR_LIMIT", "120") or 120))))
ACTIVE_HISTORY_LIMIT = 40

TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
DEFAULT_PARAMS = {
    "sl_mult": 1.8,
    "tp_mult": 2.6,
    "breakeven_atr": 1.0,
    "trail_trigger_atr": 1.4,
    "trail_pct": 0.45,
}

app = Flask(__name__)
os.makedirs(DATA_DIR, exist_ok=True)

bitget_config = {
    "apiKey": env_or_blank("BITGET_API_KEY"),
    "secret": env_or_blank("BITGET_SECRET"),
    "password": env_or_blank("BITGET_PASSWORD"),
    "enableRateLimit": True,
    "options": {"defaultType": "swap", "defaultMarginMode": "cross"},
}
exchange = ccxt.bitget(bitget_config)
exchange.enableRateLimit = True
exchange.timeout = 15000

STATE_LOCK = threading.RLock()
OPENAI_LOCK = threading.RLock()
WORKER_LOCK = threading.RLock()

WORKERS_STARTED = False
OPENAI_API_KEY = env_or_blank("OPENAI_API_KEY")
OPENAI_TRADE_CONFIG = default_trade_config(lambda name, default="": env_or_blank(name, default))
OPENAI_TRADE_STATE = load_trade_state(OPENAI_TRADE_STATE_PATH)
MARKET_CAP_CACHE: Dict[str, Dict[str, Any]] = {}
MARKET_CAP_CACHE_TS: Dict[str, float] = {}
DERIVATIVES_CACHE: Dict[str, Dict[str, Any]] = {}
DERIVATIVES_CACHE_TS: Dict[str, float] = {}


def tw_now_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    return datetime.now().strftime(fmt)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value or 0.0)
        return number if math.isfinite(number) else float(default)
    except Exception:
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value or 0))
    except Exception:
        return int(default)


def clamp(value: Any, low: float, high: float) -> float:
    return max(float(low), min(float(high), safe_float(value, low)))


def compact_symbol(symbol: str) -> str:
    return str(symbol or "").replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")


def default_backend_threads() -> Dict[str, Dict[str, Any]]:
    now = tw_now_str()
    return {
        "scan": {
            "name": "scan",
            "label": "Market Scan",
            "status": "starting",
            "detail": "Waiting for first scan cycle.",
            "note": "Collects market data, multi-timeframe candles, indicators, liquidity, and derivative context.",
            "updated_at": now,
            "updated_ts": time.time(),
            "starts": 0,
            "restart_count": 0,
            "last_error": "",
        },
        "positions": {
            "name": "positions",
            "label": "Position Sync",
            "status": "starting",
            "detail": "Waiting for first position sync.",
            "note": "Keeps active positions, equity, unrealized PnL, and close-all controls in sync.",
            "updated_at": now,
            "updated_ts": time.time(),
            "starts": 0,
            "restart_count": 0,
            "last_error": "",
        },
        "openai": {
            "name": "openai",
            "label": "OpenAI Control",
            "status": "starting",
            "detail": "Waiting for first OpenAI dashboard sync.",
            "note": "Maintains OpenAI budget, recent decisions, pending watch plans, and empty-response fallback handling.",
            "updated_at": now,
            "updated_ts": time.time(),
            "starts": 0,
            "restart_count": 0,
            "last_error": "",
        },
    }


STATE: Dict[str, Any] = {
    "last_update": "--",
    "scan_progress": "Starting scanner...",
    "equity": 0.0,
    "total_pnl": 0.0,
    "threshold_info": {"current": 48, "phase": "OpenAI+Scan"},
    "risk_status": {"trading_ok": True, "halt_reason": "", "consecutive_loss": 0, "daily_loss_pct": 0.0},
    "market_info": {"pattern": "neutral", "direction": "neutral", "btc_price": 0.0, "prediction": "Waiting for first scan."},
    "latest_news_title": "Local AI / news modules are disabled in this deployment.",
    "learn_summary": {"total_trades": 0, "win_rate": 0.0, "avg_pnl": 0.0, "current_sl_mult": DEFAULT_PARAMS["sl_mult"], "current_tp_mult": DEFAULT_PARAMS["tp_mult"]},
    "lt_info": {"position": None, "entry_price": 0, "pnl": 0, "pattern": "disabled", "prediction": "Local long-term AI is disabled."},
    "fvg_orders": {},
    "trailing_info": {},
    "protection_state": {},
    "top_signals": [],
    "active_positions": [],
    "trade_history": [],
    "backend_threads": default_backend_threads(),
}

AI_PANEL: Dict[str, Any] = {
    "params": dict(DEFAULT_PARAMS),
    "best_strategies": [],
    "last_learning": "disabled",
    "last_backtest": "disabled",
    "market_db_info": {"symbols": [], "timeframes": list(TIMEFRAMES), "last_update": "--"},
    "openai_trade": build_openai_trade_dashboard(
        OPENAI_TRADE_STATE,
        OPENAI_TRADE_CONFIG,
        api_key_present=bool(OPENAI_API_KEY),
    ),
}

AUTO_BACKTEST_STATE: Dict[str, Any] = {
    "running": False,
    "summary": "Backtest is disabled in this deployment. Scan, positions, OpenAI, and UI stay active.",
    "results": [],
    "errors": [],
    "last_run": "--",
    "last_duration_sec": 0.0,
    "scanned_markets": 0,
    "target_count": 0,
    "db_symbols": [],
    "db_last_update": "--",
    "data_timeframes": list(TIMEFRAMES),
}

TREND_DASHBOARD = {
    "trend_mode": "learning",
    "hold_reason": "normal_manage",
    "trend_confidence": 0.0,
    "trend_learning_count": 0,
    "trend_continuation_rate": 0.0,
    "trend_avg_run_pct": 0.0,
    "trend_avg_pullback_pct": 0.0,
    "trend_note": "Local AI learning and replay modules are disabled. Decisions rely on live scan data and OpenAI trade control.",
}


def persist_runtime_snapshot() -> None:
    payload = {
        "state": {
            "trade_history": list(STATE.get("trade_history", []))[-ACTIVE_HISTORY_LIMIT:],
            "latest_news_title": STATE.get("latest_news_title", ""),
        }
    }
    try:
        import json

        with open(SNAPSHOT_STATE_PATH, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_runtime_snapshot() -> None:
    if not os.path.exists(SNAPSHOT_STATE_PATH):
        return
    try:
        import json

        with open(SNAPSHOT_STATE_PATH, "r", encoding="utf-8") as fh:
            payload = json.load(fh) or {}
        base = dict(payload.get("state") or {})
        with STATE_LOCK:
            if isinstance(base.get("trade_history"), list):
                STATE["trade_history"] = list(base["trade_history"])[-ACTIVE_HISTORY_LIMIT:]
            if base.get("latest_news_title"):
                STATE["latest_news_title"] = str(base["latest_news_title"])
    except Exception:
        pass


def sync_runtime_views() -> None:
    DEFAULT_RUNTIME_STATE.update(
        threshold=dict(STATE.get("threshold_info") or {}),
        ai_panel=dict(AI_PANEL),
        auto_backtest=dict(AUTO_BACKTEST_STATE),
        risk_status=dict(STATE.get("risk_status") or {}),
        market_state=dict(STATE.get("market_info") or {}),
    )


def update_state(**kwargs: Any) -> Dict[str, Any]:
    with STATE_LOCK:
        for key, value in kwargs.items():
            STATE[key] = value
        STATE["last_update"] = tw_now_str()
        snapshot = dict(STATE)
    sync_runtime_views()
    persist_runtime_snapshot()
    return snapshot


def set_backend_thread(name: str, status: str, detail: str, *, error: str = "") -> None:
    with STATE_LOCK:
        threads = dict(STATE.get("backend_threads") or default_backend_threads())
        row = dict(threads.get(name) or {})
        row["status"] = status
        row["detail"] = detail
        row["updated_at"] = tw_now_str()
        row["updated_ts"] = time.time()
        if error:
            row["last_error"] = error[:300]
        threads[name] = row
        STATE["backend_threads"] = threads
    sync_runtime_views()


def mark_thread_started(name: str) -> None:
    with STATE_LOCK:
        threads = dict(STATE.get("backend_threads") or default_backend_threads())
        row = dict(threads.get(name) or {})
        row["starts"] = safe_int(row.get("starts"), 0) + 1
        row["status"] = "running"
        row["detail"] = "Started."
        row["updated_at"] = tw_now_str()
        row["updated_ts"] = time.time()
        threads[name] = row
        STATE["backend_threads"] = threads


def append_trade_history(item: Dict[str, Any]) -> None:
    with STATE_LOCK:
        history = list(STATE.get("trade_history") or [])
        history.insert(0, dict(item or {}))
        STATE["trade_history"] = history[:ACTIVE_HISTORY_LIMIT]
    persist_runtime_snapshot()


def market_type_label(market: Dict[str, Any]) -> str:
    if bool(market.get("swap")):
        return "swap"
    if bool(market.get("future")):
        return "future"
    return "spot"


def is_usdt_symbol(market: Dict[str, Any]) -> bool:
    symbol = str(market.get("symbol") or "")
    settle = str(market.get("settle") or "").upper()
    quote = str(market.get("quote") or "").upper()
    return symbol.endswith("/USDT:USDT") or settle == "USDT" or quote == "USDT"


def safe_request_json(url: str, *, params: Dict[str, Any] | None = None, timeout: float = 6.0) -> Any:
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def fetch_market_cap_snapshot(base_asset: str) -> Dict[str, Any]:
    base = str(base_asset or "").strip().lower()
    now = time.time()
    if base in MARKET_CAP_CACHE and (now - MARKET_CAP_CACHE_TS.get(base, 0)) < 6 * 3600:
        return dict(MARKET_CAP_CACHE[base])
    data = safe_request_json(
        "https://api.coingecko.com/api/v3/coins/markets",
        params={"vs_currency": "usd", "symbols": base, "sparkline": "false"},
        timeout=7.0,
    )
    if isinstance(data, list) and data:
        first = dict(data[0] or {})
        row = {
            "market_cap_usd": safe_float(first.get("market_cap"), 0.0),
            "fdv_usd": safe_float(first.get("fully_diluted_valuation"), 0.0),
            "circulating_supply": safe_float(first.get("circulating_supply"), 0.0),
            "total_supply": safe_float(first.get("total_supply"), 0.0),
        }
    else:
        row = {
            "market_cap_usd": 0.0,
            "fdv_usd": 0.0,
            "circulating_supply": 0.0,
            "total_supply": 0.0,
        }
    MARKET_CAP_CACHE[base] = row
    MARKET_CAP_CACHE_TS[base] = now
    return dict(row)


def safe_fetch_ohlcv_df(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not rows:
        return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v"])
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"])
    for col in ["o", "h", "l", "c", "v"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna().reset_index(drop=True)


def last_value(series: Any, default: float = 0.0) -> float:
    try:
        if series is None or len(series) == 0:
            return float(default)
        return safe_float(series.iloc[-1], default)
    except Exception:
        return float(default)


def serialize_bars(df: pd.DataFrame, limit: int = 120) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if df.empty:
        return rows
    for _, row in df.tail(limit).iterrows():
        rows.append(
            {
                "time": safe_int(row.get("t"), 0),
                "open": safe_float(row.get("o"), 0.0),
                "high": safe_float(row.get("h"), 0.0),
                "low": safe_float(row.get("l"), 0.0),
                "close": safe_float(row.get("c"), 0.0),
                "volume": safe_float(row.get("v"), 0.0),
            }
        )
    return rows


def build_timeframe_stats(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty or len(df) < 35:
        return {"bars": len(df)}
    c = df["c"]
    h = df["h"]
    l = df["l"]
    v = df["v"]

    rsi = ta.rsi(c, length=14)
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    atr = ta.atr(h, l, c, length=14)
    adx = ta.adx(h, l, c, length=14)
    bb = ta.bbands(c, length=20, std=2.0)
    stoch = ta.stoch(h, l, c, k=14, d=3, smooth_k=3)
    ema20 = ta.ema(c, length=20)
    ema50 = ta.ema(c, length=50)
    ema100 = ta.ema(c, length=100)
    ema200 = ta.ema(c, length=200)
    ma20 = ta.sma(c, length=20)
    ma60 = ta.sma(c, length=60)
    vwap = ta.vwap(h, l, c, v)
    avg_vol20 = ta.sma(v, length=20)
    avg_vol60 = ta.sma(v, length=60)

    last_close = safe_float(c.iloc[-1], 0.0)
    prev_close = safe_float(c.iloc[-2], last_close)
    atr_last = max(last_value(atr, 0.0), last_close * 0.002 if last_close > 0 else 0.0)
    adx_last = last_value(adx["ADX_14"], 0.0) if isinstance(adx, pd.DataFrame) and "ADX_14" in adx else 0.0
    plus_di = last_value(adx["DMP_14"], 0.0) if isinstance(adx, pd.DataFrame) and "DMP_14" in adx else 0.0
    minus_di = last_value(adx["DMN_14"], 0.0) if isinstance(adx, pd.DataFrame) and "DMN_14" in adx else 0.0
    macd_last = last_value(macd.iloc[:, 0], 0.0) if isinstance(macd, pd.DataFrame) else 0.0
    macd_signal = last_value(macd.iloc[:, 1], 0.0) if isinstance(macd, pd.DataFrame) else 0.0
    macd_hist = last_value(macd.iloc[:, 2], 0.0) if isinstance(macd, pd.DataFrame) else 0.0
    bb_upper = last_value(bb.iloc[:, 0], 0.0) if isinstance(bb, pd.DataFrame) else 0.0
    bb_mid = last_value(bb.iloc[:, 1], 0.0) if isinstance(bb, pd.DataFrame) else 0.0
    bb_lower = last_value(bb.iloc[:, 2], 0.0) if isinstance(bb, pd.DataFrame) else 0.0
    stoch_k = last_value(stoch.iloc[:, 0], 0.0) if isinstance(stoch, pd.DataFrame) else 0.0
    stoch_d = last_value(stoch.iloc[:, 1], 0.0) if isinstance(stoch, pd.DataFrame) else 0.0

    trend_label = "neutral"
    if last_value(ema20, last_close) > last_value(ema50, last_close) > last_value(ema200, last_close):
        trend_label = "uptrend"
    elif last_value(ema20, last_close) < last_value(ema50, last_close) < last_value(ema200, last_close):
        trend_label = "downtrend"

    recent_high = safe_float(h.tail(20).max(), last_close)
    recent_low = safe_float(l.tail(20).min(), last_close)
    support_levels = [safe_float(l.tail(10).min(), recent_low), recent_low]
    resistance_levels = [safe_float(h.tail(10).max(), recent_high), recent_high]
    avg_volume_20 = last_value(avg_vol20, 0.0)
    avg_volume_60 = last_value(avg_vol60, avg_volume_20)
    last_volume = safe_float(v.iloc[-1], 0.0)

    return {
        "bars": len(df),
        "last_close": last_close,
        "prev_close": prev_close,
        "atr": round(atr_last, 6),
        "atr_pct": round((atr_last / last_close) * 100 if last_close > 0 else 0.0, 4),
        "rsi": round(last_value(rsi, 50.0), 4),
        "adx": round(adx_last, 4),
        "plus_di": round(plus_di, 4),
        "minus_di": round(minus_di, 4),
        "ema9": round(last_value(ta.ema(c, length=9), last_close), 6),
        "ema20": round(last_value(ema20, last_close), 6),
        "ema50": round(last_value(ema50, last_close), 6),
        "ema100": round(last_value(ema100, last_close), 6),
        "ema200": round(last_value(ema200, last_close), 6),
        "ma20": round(last_value(ma20, last_close), 6),
        "ma60": round(last_value(ma60, last_close), 6),
        "vwap": round(last_value(vwap, last_close), 6),
        "trend_label": trend_label,
        "macd": round(macd_last, 6),
        "macd_signal": round(macd_signal, 6),
        "macd_hist": round(macd_hist, 6),
        "bb_upper": round(bb_upper, 6),
        "bb_mid": round(bb_mid, 6),
        "bb_lower": round(bb_lower, 6),
        "bb_width_pct": round(((bb_upper - bb_lower) / last_close) * 100 if last_close > 0 else 0.0, 4),
        "bb_position_pct": round(((last_close - bb_lower) / max(bb_upper - bb_lower, 1e-9)) * 100, 4),
        "stoch_k": round(stoch_k, 4),
        "stoch_d": round(stoch_d, 4),
        "kdj_j": round((3 * stoch_k) - (2 * stoch_d), 4),
        "ret_3bars_pct": round(((last_close / safe_float(c.iloc[-4], last_close)) - 1) * 100 if len(c) >= 4 else 0.0, 4),
        "ret_12bars_pct": round(((last_close / safe_float(c.iloc[-13], last_close)) - 1) * 100 if len(c) >= 13 else 0.0, 4),
        "ret_24bars_pct": round(((last_close / safe_float(c.iloc[-25], last_close)) - 1) * 100 if len(c) >= 25 else 0.0, 4),
        "vol_ratio": round((last_volume / max(avg_volume_20, 1e-9)), 4),
        "avg_volume_20": round(avg_volume_20, 4),
        "avg_volume_60": round(avg_volume_60, 4),
        "last_volume": round(last_volume, 4),
        "swing_high_20": round(recent_high, 6),
        "swing_low_20": round(recent_low, 6),
        "distance_to_swing_high_pct": round(((recent_high - last_close) / max(last_close, 1e-9)) * 100, 4),
        "distance_to_swing_low_pct": round(((last_close - recent_low) / max(last_close, 1e-9)) * 100, 4),
        "support_levels": [round(x, 6) for x in support_levels if x > 0],
        "resistance_levels": [round(x, 6) for x in resistance_levels if x > 0],
        "recent_structure_high": round(recent_high, 6),
        "recent_structure_low": round(recent_low, 6),
    }


def build_liquidity_context(symbol: str, ticker: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "spread_pct": 0.0,
        "bid_depth_5": 0.0,
        "ask_depth_5": 0.0,
        "bid_depth_10": 0.0,
        "ask_depth_10": 0.0,
        "depth_imbalance_10": 0.0,
        "largest_bid_wall_price": 0.0,
        "largest_bid_wall_size": 0.0,
        "largest_ask_wall_price": 0.0,
        "largest_ask_wall_size": 0.0,
        "recent_trades_count": 0,
        "aggressive_buy_volume": 0.0,
        "aggressive_sell_volume": 0.0,
        "aggressive_buy_notional": 0.0,
        "aggressive_sell_notional": 0.0,
        "buy_sell_notional_ratio": 1.0,
        "cvd_notional": 0.0,
        "cvd_bias": "neutral",
        "volume_anomaly_5m": 0.0,
        "volume_anomaly_15m": 0.0,
        "errors": [],
    }
    bid = safe_float(ticker.get("bid"), 0.0)
    ask = safe_float(ticker.get("ask"), 0.0)
    last = safe_float(ticker.get("last"), 0.0)
    if bid > 0 and ask > 0 and last > 0:
        result["spread_pct"] = round(((ask - bid) / last) * 100, 6)
    try:
        book = exchange.fetch_order_book(symbol, limit=10)
        bids = list(book.get("bids") or [])
        asks = list(book.get("asks") or [])
        result["bid_depth_5"] = round(sum(safe_float(row[1], 0.0) for row in bids[:5]), 4)
        result["ask_depth_5"] = round(sum(safe_float(row[1], 0.0) for row in asks[:5]), 4)
        result["bid_depth_10"] = round(sum(safe_float(row[1], 0.0) for row in bids[:10]), 4)
        result["ask_depth_10"] = round(sum(safe_float(row[1], 0.0) for row in asks[:10]), 4)
        denom = max(result["bid_depth_10"] + result["ask_depth_10"], 1e-9)
        result["depth_imbalance_10"] = round((result["bid_depth_10"] - result["ask_depth_10"]) / denom, 6)
    except Exception as exc:
        result["errors"].append("order_book:{}".format(str(exc)[:120]))
    try:
        trades = list(exchange.fetch_trades(symbol, limit=120) or [])
        result["recent_trades_count"] = len(trades)
        now_ms = int(time.time() * 1000)
        buy_notional = 0.0
        sell_notional = 0.0
        vol_5m = 0.0
        vol_prev_5m = 0.0
        vol_15m = 0.0
        vol_prev_15m = 0.0
        for trade in trades:
            amount = safe_float(trade.get("amount"), 0.0)
            price = safe_float(trade.get("price"), 0.0)
            notional = amount * price
            side = str(trade.get("side") or "").lower()
            ts = safe_int(trade.get("timestamp"), 0)
            age_ms = max(now_ms - ts, 0)
            if side == "buy":
                buy_notional += notional
                result["aggressive_buy_volume"] += amount
            elif side == "sell":
                sell_notional += notional
                result["aggressive_sell_volume"] += amount
            if age_ms <= 5 * 60 * 1000:
                vol_5m += notional
            elif age_ms <= 10 * 60 * 1000:
                vol_prev_5m += notional
            if age_ms <= 15 * 60 * 1000:
                vol_15m += notional
            elif age_ms <= 30 * 60 * 1000:
                vol_prev_15m += notional
        result["aggressive_buy_notional"] = round(buy_notional, 4)
        result["aggressive_sell_notional"] = round(sell_notional, 4)
        result["buy_sell_notional_ratio"] = round(buy_notional / max(sell_notional, 1e-9), 4)
        result["cvd_notional"] = round(buy_notional - sell_notional, 4)
        if buy_notional > sell_notional * 1.15:
            result["cvd_bias"] = "buying"
        elif sell_notional > buy_notional * 1.15:
            result["cvd_bias"] = "selling"
        result["volume_anomaly_5m"] = round(vol_5m / max(vol_prev_5m, 1e-9), 4)
        result["volume_anomaly_15m"] = round(vol_15m / max(vol_prev_15m, 1e-9), 4)
    except Exception as exc:
        result["errors"].append("trades:{}".format(str(exc)[:120]))
    return result


def build_derivatives_context(symbol: str, ticker: Dict[str, Any]) -> Dict[str, Any]:
    now = time.time()
    if symbol in DERIVATIVES_CACHE and (now - DERIVATIVES_CACHE_TS.get(symbol, 0)) < 45:
        return dict(DERIVATIVES_CACHE[symbol])
    result = {
        "funding_rate": 0.0,
        "next_funding_time": "",
        "open_interest": 0.0,
        "open_interest_value_usdt": 0.0,
        "open_interest_change_pct_5m": 0.0,
        "long_short_ratio": 1.0,
        "top_trader_long_short_ratio": 1.0,
        "whale_position_change_pct": 0.0,
        "basis_pct": 0.0,
        "mark_price": safe_float((ticker.get("info") or {}).get("markPrice"), safe_float(ticker.get("last"), 0.0)),
        "index_price": safe_float((ticker.get("info") or {}).get("indexPrice"), safe_float(ticker.get("last"), 0.0)),
        "liquidation_volume_24h": 0.0,
        "liquidation_map_status": "unavailable",
        "leverage_heat": "normal",
        "leverage_heat_score": 0.0,
        "errors": [],
    }
    try:
        fetch_funding_rate = getattr(exchange, "fetch_funding_rate", None)
        if callable(fetch_funding_rate):
            funding = dict(fetch_funding_rate(symbol) or {})
            result["funding_rate"] = round(safe_float(funding.get("fundingRate", funding.get("funding_rate")), 0.0), 8)
            result["next_funding_time"] = str(funding.get("fundingDatetime") or funding.get("nextFundingTime") or "")[:40]
    except Exception as exc:
        result["errors"].append("funding:{}".format(str(exc)[:120]))
    try:
        fetch_open_interest = getattr(exchange, "fetch_open_interest", None)
        if callable(fetch_open_interest):
            oi = dict(fetch_open_interest(symbol) or {})
            result["open_interest"] = round(safe_float(oi.get("openInterestAmount", oi.get("openInterest", oi.get("amount"))), 0.0), 4)
            result["open_interest_value_usdt"] = round(safe_float(oi.get("openInterestValue", oi.get("value")), 0.0), 4)
    except Exception as exc:
        result["errors"].append("open_interest:{}".format(str(exc)[:120]))
    info = dict(ticker.get("info") or {})
    buy_vol = safe_float(info.get("buyVolume"), 0.0)
    sell_vol = safe_float(info.get("sellVolume"), 0.0)
    if buy_vol > 0 or sell_vol > 0:
        result["long_short_ratio"] = round((buy_vol + 1e-9) / max(sell_vol, 1e-9), 4)
        result["top_trader_long_short_ratio"] = result["long_short_ratio"]
        result["whale_position_change_pct"] = round(((buy_vol - sell_vol) / max(buy_vol + sell_vol, 1e-9)) * 100, 4)
    if result["mark_price"] > 0 and result["index_price"] > 0:
        result["basis_pct"] = round(((result["mark_price"] - result["index_price"]) / result["index_price"]) * 100, 6)
    heat = abs(result["funding_rate"]) * 10000.0
    if result["open_interest_value_usdt"] > 0:
        heat += min(result["open_interest_value_usdt"] / 1_000_000.0, 50.0)
    result["leverage_heat_score"] = round(heat, 4)
    if heat >= 35:
        result["leverage_heat"] = "high"
    elif heat >= 18:
        result["leverage_heat"] = "elevated"
    DERIVATIVES_CACHE[symbol] = result
    DERIVATIVES_CACHE_TS[symbol] = now
    return dict(result)


def infer_setup(tf15: Dict[str, Any], tf1h: Dict[str, Any], tf4h: Dict[str, Any]) -> str:
    if tf15.get("trend_label") == "uptrend" and tf1h.get("trend_label") == "uptrend":
        return "trend_pullback_long"
    if tf15.get("trend_label") == "downtrend" and tf1h.get("trend_label") == "downtrend":
        return "trend_pullback_short"
    if safe_float(tf15.get("bb_width_pct"), 0.0) < 3 and safe_float(tf15.get("adx"), 0.0) < 20:
        return "compression_watch"
    if tf4h.get("trend_label") == "uptrend":
        return "higher_timeframe_long"
    if tf4h.get("trend_label") == "downtrend":
        return "higher_timeframe_short"
    return "range_watch"


def build_signal_from_context(symbol: str, market: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    tf15 = dict((context.get("multi_timeframe") or {}).get("15m") or {})
    tf1h = dict((context.get("multi_timeframe") or {}).get("1h") or {})
    tf4h = dict((context.get("multi_timeframe") or {}).get("4h") or {})
    tf1d = dict((context.get("multi_timeframe") or {}).get("1d") or {})

    long_score = 0.0
    short_score = 0.0
    notes: List[str] = []

    if tf15.get("ema20", 0) > tf15.get("ema50", 0):
        long_score += 12
        notes.append("15m EMA20 above EMA50")
    else:
        short_score += 12
    if tf1h.get("ema20", 0) > tf1h.get("ema50", 0):
        long_score += 10
        notes.append("1h trend aligned long")
    else:
        short_score += 10
    if tf4h.get("ema50", 0) > tf4h.get("ema200", 0):
        long_score += 8
    elif tf4h.get("ema50", 0) < tf4h.get("ema200", 0):
        short_score += 8
    if safe_float(tf15.get("macd_hist"), 0.0) > 0:
        long_score += 8
    else:
        short_score += 8
    if safe_float(tf15.get("rsi"), 50.0) >= 55:
        long_score += 7
    elif safe_float(tf15.get("rsi"), 50.0) <= 45:
        short_score += 7
    if safe_float(tf15.get("adx"), 0.0) >= 18:
        if safe_float(tf15.get("plus_di"), 0.0) > safe_float(tf15.get("minus_di"), 0.0):
            long_score += 6
        else:
            short_score += 6
    if safe_float(tf15.get("last_close"), 0.0) > safe_float(tf15.get("vwap"), 0.0):
        long_score += 4
    else:
        short_score += 4
    if safe_float(tf15.get("vol_ratio"), 0.0) >= 1.15:
        if safe_float(tf15.get("macd_hist"), 0.0) >= 0:
            long_score += 4
        else:
            short_score += 4

    score = round(long_score - short_score, 2)
    price = safe_float((context.get("basic_market_data") or {}).get("current_price"), 0.0)
    atr = max(safe_float(tf15.get("atr"), 0.0), price * 0.002 if price > 0 else 0.0)
    side = "long" if score >= 0 else "short"
    confidence = clamp((abs(score) / 55.0) * 100.0, 18.0, 96.0)
    entry_quality = clamp((abs(score) / 8.0), 1.0, 10.0)
    rr_ratio = 2.2 if abs(score) >= 18 else 1.8
    if side == "long":
        stop_loss = max(safe_float(tf15.get("recent_structure_low"), price - atr * 1.8), price - atr * 1.8)
        take_profit = max(price + atr * rr_ratio, safe_float(tf15.get("recent_structure_high"), 0.0))
    else:
        stop_loss = min(max(price + atr * 1.8, price * 1.002), max(safe_float(tf15.get("recent_structure_high"), price + atr * 1.8), price + atr * 1.8))
        take_profit = min(price - atr * rr_ratio, safe_float(tf15.get("recent_structure_low"), price - atr * rr_ratio))
    market_dir = "bullish" if tf1d.get("trend_label") == "uptrend" else "bearish" if tf1d.get("trend_label") == "downtrend" else "neutral"
    breakdown = {
        "Setup": infer_setup(tf15, tf1h, tf4h),
        "Regime": market_dir,
        "RegimeConfidence": round(clamp((abs(score) / 40.0) * 100.0, 20.0, 92.0), 1),
        "RegimeDir": market_dir,
        "TrendConfidence": round(confidence, 1),
        "SignalQuality": round(entry_quality, 1),
        "RR": round(rr_ratio, 2),
        "VWAPDistanceATR": round((price - safe_float(tf15.get("vwap"), price)) / max(atr, 1e-9), 4),
        "EMA20DistanceATR": round((price - safe_float(tf15.get("ema20"), price)) / max(atr, 1e-9), 4),
        "SRDistanceATR": round(min(abs(price - safe_float(tf15.get("recent_structure_high"), price)), abs(price - safe_float(tf15.get("recent_structure_low"), price))) / max(atr, 1e-9), 4),
    }
    return {
        "symbol": symbol,
        "direction": "long" if side == "long" else "short",
        "side": side,
        "score": score,
        "raw_score": score,
        "priority_score": abs(score),
        "direction_confidence": round(confidence, 1),
        "trend_confidence": round(confidence, 1),
        "entry_quality": round(entry_quality, 1),
        "signal_grade": "A" if abs(score) >= 24 else "B" if abs(score) >= 14 else "C",
        "setup_label": breakdown["Setup"],
        "price": round(price, 6),
        "stop_loss": round(stop_loss, 6),
        "take_profit": round(take_profit, 6),
        "rr_ratio": round(abs((take_profit - price) / max(abs(price - stop_loss), 1e-9)), 2),
        "margin_pct": round(clamp(0.03 + (abs(score) / 900.0), 0.03, 0.08), 4),
        "est_pnl": round(((take_profit - price) / max(price, 1e-9)) * 100 if side == "long" else ((price - take_profit) / max(price, 1e-9)) * 100, 2),
        "breakdown": breakdown,
        "desc": " | ".join(notes[:4]) or "Live scan signal",
        "trend_mode": "learning",
        "hold_reason": "normal_manage",
        "trend_note": "Local AI learning removed. Signal uses live market structure, indicators, liquidity, and derivative context.",
        "openai_market_context": context,
        "market": market,
    }


def build_market_context(symbol: str, ticker: Dict[str, Any], market: Dict[str, Any]) -> Dict[str, Any]:
    multi_timeframe: Dict[str, Any] = {}
    timeframe_bars: Dict[str, Any] = {}
    errors: List[str] = []
    for tf in TIMEFRAMES:
        try:
            df = safe_fetch_ohlcv_df(symbol, tf, TIMEFRAME_BAR_LIMIT)
            multi_timeframe[tf] = build_timeframe_stats(df)
            timeframe_bars[tf] = serialize_bars(df, TIMEFRAME_BAR_LIMIT)
        except Exception as exc:
            multi_timeframe[tf] = {"bars": 0}
            timeframe_bars[tf] = []
            errors.append("{}:{}".format(tf, str(exc)[:120]))
    base_asset = str(market.get("base") or compact_symbol(symbol).replace("USDT", "")).lower()
    supply = fetch_market_cap_snapshot(base_asset)
    derivatives = build_derivatives_context(symbol, ticker)
    liquidity = build_liquidity_context(symbol, ticker)

    basic_market = {
        "symbol": compact_symbol(symbol),
        "exchange": "Bitget",
        "market_type": market_type_label(market),
        "current_price": round(safe_float(ticker.get("last"), 0.0), 6),
        "change_24h_pct": round(safe_float(ticker.get("percentage"), 0.0), 4),
        "quote_volume_24h": round(safe_float(ticker.get("quoteVolume"), ticker.get("baseVolume")), 4),
        "base_volume_24h": round(safe_float(ticker.get("baseVolume"), 0.0), 4),
        "market_cap_usd": round(safe_float(supply.get("market_cap_usd"), 0.0), 2),
        "fdv_usd": round(safe_float(supply.get("fdv_usd"), 0.0), 2),
        "circulating_supply": round(safe_float(supply.get("circulating_supply"), 0.0), 2),
        "total_supply": round(safe_float(supply.get("total_supply"), 0.0), 2),
        "funding_rate": derivatives.get("funding_rate", 0.0),
        "open_interest": derivatives.get("open_interest", 0.0),
        "open_interest_value_usdt": derivatives.get("open_interest_value_usdt", 0.0),
        "long_short_ratio": derivatives.get("long_short_ratio", 1.0),
        "top_trader_long_short_ratio": derivatives.get("top_trader_long_short_ratio", 1.0),
        "whale_position_change_pct": derivatives.get("whale_position_change_pct", 0.0),
    }
    tf15 = dict(multi_timeframe.get("15m") or {})
    tf1h = dict(multi_timeframe.get("1h") or {})
    tf4h = dict(multi_timeframe.get("4h") or {})
    bias = "long" if tf15.get("trend_label") == "uptrend" else "short" if tf15.get("trend_label") == "downtrend" else "neutral"
    aligned = [tf for tf, row in multi_timeframe.items() if row.get("trend_label") == ("uptrend" if bias == "long" else "downtrend")]
    opposing = [tf for tf, row in multi_timeframe.items() if bias != "neutral" and row.get("trend_label") not in ("neutral", "uptrend" if bias == "long" else "downtrend")]
    pressure_summary = {
        "side": bias,
        "aligned_timeframes": aligned,
        "opposing_timeframes": opposing,
        "nearest_blocking_timeframe": "1h" if bias == "long" else "15m",
        "nearest_blocking_price": tf1h.get("recent_structure_high", 0.0) if bias == "long" else tf15.get("recent_structure_low", 0.0),
        "nearest_backing_timeframe": "15m",
        "nearest_backing_price": tf15.get("recent_structure_low", 0.0) if bias == "long" else tf15.get("recent_structure_high", 0.0),
        "nearest_blocking_distance_atr": 0.0,
        "nearest_backing_distance_atr": 0.0,
        "stacked_blocking_within_1atr": len([tf for tf in opposing if tf in ("15m", "1h")]),
        "stacked_blocking_within_2atr": len(opposing),
    }
    return {
        "style": {"holding_period": "intraday", "trade_goal": "Use live scan plus OpenAI.", "decision_priority": "market_context>multi_timeframe>liquidity>derivatives>OpenAI"},
        "signal_context": {"side": bias, "current_price": basic_market["current_price"], "atr_15m": tf15.get("atr", 0.0), "atr_4h": tf4h.get("atr", 0.0)},
        "latest_closed_candle": {"direction": "up" if safe_float(tf15.get("last_close"), 0.0) >= safe_float(tf15.get("prev_close"), 0.0) else "down", "shape": "wide" if safe_float(tf15.get("atr_pct"), 0.0) >= 1.0 else "normal"},
        "momentum": {"long_score": round(sum(1 for row in [tf15, tf1h, tf4h] if row.get("trend_label") == "uptrend"), 2), "short_score": round(sum(1 for row in [tf15, tf1h, tf4h] if row.get("trend_label") == "downtrend"), 2), "signals": [tf15.get("trend_label"), tf1h.get("trend_label"), tf4h.get("trend_label")], "volume_build": safe_float(tf15.get("vol_ratio"), 0.0) >= 1.15, "compression": safe_float(tf15.get("bb_width_pct"), 0.0) <= 3.0},
        "levels": {"nearest_support": tf15.get("recent_structure_low", 0.0), "nearest_resistance": tf15.get("recent_structure_high", 0.0), "support_levels": tf15.get("support_levels", []), "resistance_levels": tf15.get("resistance_levels", []), "recent_high": tf15.get("recent_structure_high", 0.0), "recent_low": tf15.get("recent_structure_low", 0.0)},
        "market_state": {"ticker": {"last": basic_market["current_price"], "bid": round(safe_float(ticker.get("bid"), 0.0), 6), "ask": round(safe_float(ticker.get("ask"), 0.0), 6), "spread_pct": liquidity.get("spread_pct", 0.0), "mark_price": derivatives.get("mark_price", basic_market["current_price"]), "index_price": derivatives.get("index_price", basic_market["current_price"])}},
        "basic_market_data": basic_market,
        "liquidity_context": liquidity,
        "derivatives_context": derivatives,
        "news_context": {"items": [], "summary": "News and local AI modules are disabled for stability in this deployment."},
        "multi_timeframe": multi_timeframe,
        "timeframe_bars": timeframe_bars,
        "pre_breakout_radar": {"ready": len(aligned) >= 2, "phase": "watch", "direction": bias, "score": round(safe_float(tf15.get("adx"), 0.0), 2), "summary": "Multi-timeframe scan without local AI learning.", "note": "Uses live candles, derivatives, and liquidity only."},
        "execution_context": {"spread_pct": liquidity.get("spread_pct", 0.0), "top_depth_ratio": liquidity.get("depth_imbalance_10", 0.0), "api_error_streak": len(errors) + len(liquidity.get("errors", [])) + len(derivatives.get("errors", [])), "status": "ok" if not errors else "degraded", "notes": errors + list(liquidity.get("errors", [])) + list(derivatives.get("errors", []))},
        "multi_timeframe_pressure_summary": pressure_summary,
        "multi_timeframe_pressure": {},
        "reference_context": {"summary": "Live scan built from Bitget market, multi-timeframe candles, liquidity, and derivative data. Local AI and replay modules are disabled."},
    }


def flatten_openai_result(result: Dict[str, Any]) -> Dict[str, Any]:
    decision = dict(result.get("decision") or {})
    return {
        "ai_enabled": True,
        "openai_enabled": True,
        "openai_status": str(result.get("status") or ""),
        "openai_status_label": str(result.get("status") or "").replace("_", " "),
        "openai_model": str(result.get("model") or ""),
        "openai_order_type": decision.get("order_type", ""),
        "openai_margin_pct": safe_float(decision.get("margin_pct"), 0.0),
        "openai_leverage": safe_float(decision.get("leverage"), 0.0),
        "openai_thesis": str(decision.get("thesis") or ""),
        "openai_reason_to_skip": str(decision.get("reason_to_skip") or ""),
        "openai_market_read": str(decision.get("market_read") or ""),
        "openai_entry_plan": str(decision.get("entry_plan") or ""),
        "openai_entry_reason": str(decision.get("entry_reason") or ""),
        "openai_take_profit_plan": str(decision.get("take_profit_plan") or ""),
        "openai_stop_loss_reason": str(decision.get("stop_loss_reason") or ""),
        "openai_if_missed_plan": str(decision.get("if_missed_plan") or ""),
        "openai_reference_summary": str(decision.get("reference_summary") or ""),
        "openai_chase_if_triggered": bool(decision.get("chase_if_triggered", False)),
        "openai_chase_trigger_price": safe_float(decision.get("chase_trigger_price"), 0.0),
        "openai_chase_limit_price": safe_float(decision.get("chase_limit_price"), 0.0),
        "openai_risk_notes": list(decision.get("risk_notes") or []),
        "openai_aggressive_note": str(decision.get("aggressive_note") or ""),
        "openai_watch_trigger_type": str(decision.get("watch_trigger_type") or ""),
        "openai_watch_trigger_price": safe_float(decision.get("watch_trigger_price"), 0.0),
        "openai_watch_invalidation_price": safe_float(decision.get("watch_invalidation_price"), 0.0),
        "openai_watch_note": str(decision.get("watch_note") or ""),
        "openai_recheck_reason": str(decision.get("recheck_reason") or ""),
        "openai_watch_structure_condition": str(decision.get("watch_structure_condition") or ""),
        "openai_watch_volume_condition": str(decision.get("watch_volume_condition") or ""),
        "openai_watch_confirmations": list(decision.get("watch_confirmations") or []),
        "openai_watch_invalidations": list(decision.get("watch_invalidations") or []),
        "will_order": bool(decision.get("should_trade", False)),
        "threshold": 48.0,
        "reasons": list(result.get("reasons") or []),
    }


def refresh_openai_dashboard() -> None:
    global OPENAI_TRADE_STATE
    with OPENAI_LOCK:
        OPENAI_TRADE_STATE = load_trade_state(OPENAI_TRADE_STATE_PATH)
        AI_PANEL["openai_trade"] = build_openai_trade_dashboard(
            OPENAI_TRADE_STATE,
            OPENAI_TRADE_CONFIG,
            api_key_present=bool(OPENAI_API_KEY),
        )
        save_trade_state(OPENAI_TRADE_STATE_PATH, OPENAI_TRADE_STATE)
    sync_runtime_views()


def build_portfolio_snapshot() -> Dict[str, Any]:
    with STATE_LOCK:
        active = list(STATE.get("active_positions") or [])
        equity = safe_float(STATE.get("equity"), 0.0)
    long_count = sum(1 for row in active if str(row.get("side") or "").lower() == "long")
    short_count = sum(1 for row in active if str(row.get("side") or "").lower() == "short")
    return {
        "equity": equity,
        "active_position_count": len(active),
        "long_count": long_count,
        "short_count": short_count,
        "position_symbols": [row.get("symbol") for row in active if row.get("symbol")],
    }


def sync_positions_once() -> None:
    set_backend_thread("positions", "running", "Syncing balance and active positions.")
    try:
        equity = 0.0
        positions_payload: List[Dict[str, Any]] = []
        try:
            balance = exchange.fetch_balance()
            total = dict(balance.get("total") or {})
            equity = safe_float(total.get("USDT"), 0.0)
        except Exception:
            equity = safe_float(STATE.get("equity"), 0.0)
        try:
            raw_positions = list(exchange.fetch_positions() or [])
        except Exception:
            raw_positions = []
        for row in raw_positions:
            contracts = abs(safe_float(row.get("contracts"), 0.0))
            if contracts <= 0:
                continue
            side = str(row.get("side") or "").lower()
            entry = safe_float(row.get("entryPrice"), 0.0)
            mark = safe_float(row.get("markPrice"), safe_float(row.get("lastPrice"), 0.0))
            unrealized = safe_float(row.get("unrealizedPnl"), 0.0)
            leverage = safe_float(row.get("leverage"), 1.0)
            pct = safe_float(row.get("percentage"), 0.0)
            drawdown = 0.0
            if entry > 0 and mark > 0:
                if side == "long":
                    drawdown = ((mark - entry) / entry) * 100
                else:
                    drawdown = ((entry - mark) / entry) * 100
            positions_payload.append(
                {
                    "symbol": str(row.get("symbol") or ""),
                    "side": side,
                    "entryPrice": entry,
                    "markPrice": mark,
                    "contracts": contracts,
                    "unrealizedPnl": round(unrealized, 4),
                    "leverage": leverage,
                    "percentage": pct,
                    "drawdown_pct": round(drawdown, 4),
                    "leveraged_pnl_pct": round(pct, 4),
                    "trend_mode": "learning",
                    "hold_reason": "normal_manage",
                    "trend_confidence": 0.0,
                }
            )
        total_pnl = round(sum(safe_float(row.get("unrealizedPnl"), 0.0) for row in positions_payload), 4)
        update_state(active_positions=positions_payload, equity=round(equity, 4), total_pnl=total_pnl)
        set_backend_thread("positions", "running", "Synced {} active positions.".format(len(positions_payload)))
    except Exception as exc:
        set_backend_thread("positions", "crashed", "Position sync failed.", error=str(exc))


def update_market_overview(top_signals: List[Dict[str, Any]]) -> None:
    btc_signal = next((row for row in top_signals if row.get("symbol") == "BTC/USDT:USDT"), None)
    if not btc_signal and top_signals:
        btc_signal = top_signals[0]
    market_info = {
        "pattern": str((btc_signal or {}).get("setup_label") or "neutral"),
        "direction": str((btc_signal or {}).get("direction") or "neutral"),
        "btc_price": safe_float((btc_signal or {}).get("price"), 0.0),
        "prediction": str((btc_signal or {}).get("desc") or "Waiting for signal."),
    }
    update_state(market_info=market_info)


def refresh_learning_summary() -> None:
    with STATE_LOCK:
        history = list(STATE.get("trade_history") or [])
    closed = [row for row in history if row.get("pnl_pct") is not None]
    total = len(closed)
    wins = sum(1 for row in closed if safe_float(row.get("pnl_pct"), 0.0) > 0)
    avg = round(sum(safe_float(row.get("pnl_pct"), 0.0) for row in closed) / max(total, 1), 4) if total else 0.0
    update_state(
        learn_summary={
            "total_trades": total,
            "win_rate": round((wins / max(total, 1)) * 100, 1) if total else 0.0,
            "avg_pnl": avg,
            "current_sl_mult": AI_PANEL["params"]["sl_mult"],
            "current_tp_mult": AI_PANEL["params"]["tp_mult"],
        }
    )


def maybe_run_openai(top_signals: List[Dict[str, Any]]) -> None:
    global OPENAI_TRADE_STATE
    if not top_signals:
        refresh_openai_dashboard()
        return
    best = dict(top_signals[0])
    candidate = build_openai_trade_candidate(
        signal=best,
        market=best.get("market") or {},
        risk_status=STATE.get("risk_status") or {},
        portfolio=build_portfolio_snapshot(),
        top_candidates=top_signals[: min(len(top_signals), 5)],
        constraints={
            "fixed_leverage": 20,
            "min_leverage": 4,
            "max_leverage": 25,
            "min_margin_pct": 0.03,
            "max_margin_pct": 0.08,
        },
        rank_index=1,
    )
    try:
        with OPENAI_LOCK:
            OPENAI_TRADE_STATE, result = consult_trade_decision(
                state=OPENAI_TRADE_STATE,
                state_path=OPENAI_TRADE_STATE_PATH,
                api_key=OPENAI_API_KEY,
                config=OPENAI_TRADE_CONFIG,
                candidate=candidate,
                logger=lambda msg: set_backend_thread("openai", "running", msg[:180]),
            )
            AI_PANEL["openai_trade"] = build_openai_trade_dashboard(
                OPENAI_TRADE_STATE,
                OPENAI_TRADE_CONFIG,
                api_key_present=bool(OPENAI_API_KEY),
            )
        best["auto_order"] = flatten_openai_result(result)
        top_signals[0] = best
        set_backend_thread("openai", "running", "OpenAI status: {}".format(result.get("status", "unknown")))
    except Exception as exc:
        refresh_openai_dashboard()
        set_backend_thread("openai", "crashed", "OpenAI sync failed.", error=str(exc))


def perform_scan_cycle() -> None:
    set_backend_thread("scan", "running", "Fetching markets and scan universe.")
    update_state(scan_progress="Loading Bitget tickers...")
    try:
        markets = exchange.load_markets()
        tickers = exchange.fetch_tickers()
        tradeable = []
        for symbol, market in markets.items():
            if not isinstance(market, dict):
                continue
            if not is_usdt_symbol(market):
                continue
            if not bool(market.get("active", True)):
                continue
            if market_type_label(market) == "spot":
                continue
            ticker = dict(tickers.get(symbol) or {})
            if safe_float(ticker.get("quoteVolume"), 0.0) <= 0:
                continue
            tradeable.append((symbol, market, ticker))
        tradeable.sort(key=lambda row: safe_float(row[2].get("quoteVolume"), 0.0), reverse=True)
        selected = tradeable[:SCAN_SYMBOL_LIMIT]
        top_signals: List[Dict[str, Any]] = []
        for idx, (symbol, market, ticker) in enumerate(selected, start=1):
            update_state(scan_progress="Scanning {}/{} {}".format(idx, len(selected), compact_symbol(symbol)))
            try:
                context = build_market_context(symbol, ticker, market)
                signal = build_signal_from_context(symbol, {"symbol": symbol, "pattern": "live_scan"}, context)
                top_signals.append(signal)
            except Exception as exc:
                top_signals.append(
                    {
                        "symbol": symbol,
                        "direction": "neutral",
                        "side": "long",
                        "score": 0.0,
                        "raw_score": 0.0,
                        "priority_score": 0.0,
                        "direction_confidence": 0.0,
                        "trend_confidence": 0.0,
                        "entry_quality": 0.0,
                        "signal_grade": "C",
                        "setup_label": "scan_error",
                        "price": safe_float(ticker.get("last"), 0.0),
                        "stop_loss": 0.0,
                        "take_profit": 0.0,
                        "rr_ratio": 0.0,
                        "margin_pct": 0.03,
                        "est_pnl": 0.0,
                        "breakdown": {"Setup": "scan_error"},
                        "desc": "Scan failed: {}".format(str(exc)[:120]),
                        "trend_mode": "learning",
                        "hold_reason": "normal_manage",
                        "trend_note": "Scan failed, payload degraded.",
                        "openai_market_context": {},
                        "market": {"symbol": symbol},
                    }
                )
        top_signals.sort(key=lambda row: abs(safe_float(row.get("priority_score"), 0.0)), reverse=True)
        top_signals = top_signals[:TOP_SIGNAL_LIMIT]
        for idx, row in enumerate(top_signals, start=1):
            row["rank"] = idx
        maybe_run_openai(top_signals)
        AI_PANEL["best_strategies"] = [
            {
                "strategy": row.get("setup_label"),
                "count": 1,
                "win_rate": 0.0,
                "avg_pnl": row.get("est_pnl", 0.0),
                "score": row.get("score", 0.0),
                "confidence": safe_float(row.get("direction_confidence"), 0.0) / 100.0,
                "timeframes": "/".join(TIMEFRAMES),
                "updated_at": tw_now_str(),
            }
            for row in top_signals[:5]
        ]
        AI_PANEL["market_db_info"] = {
            "symbols": [row.get("symbol") for row in top_signals],
            "timeframes": list(TIMEFRAMES),
            "last_update": tw_now_str(),
        }
        AUTO_BACKTEST_STATE["scanned_markets"] = len(top_signals)
        AUTO_BACKTEST_STATE["target_count"] = len(selected)
        AUTO_BACKTEST_STATE["db_symbols"] = [row.get("symbol") for row in top_signals]
        AUTO_BACKTEST_STATE["db_last_update"] = tw_now_str()
        update_state(top_signals=top_signals, scan_progress="Scan complete. {} symbols updated.".format(len(top_signals)))
        update_market_overview(top_signals)
        refresh_learning_summary()
        set_backend_thread("scan", "running", "Updated {} symbols with live scan context.".format(len(top_signals)))
    except Exception as exc:
        update_state(scan_progress="Scan failed: {}".format(str(exc)[:120]))
        set_backend_thread("scan", "crashed", "Scan cycle failed.", error="{}\n{}".format(exc, traceback.format_exc()[:500]))


def scan_thread() -> None:
    mark_thread_started("scan")
    while True:
        perform_scan_cycle()
        time.sleep(SCAN_INTERVAL_SEC)


def positions_thread() -> None:
    mark_thread_started("positions")
    while True:
        sync_positions_once()
        time.sleep(POSITION_INTERVAL_SEC)


def openai_thread() -> None:
    mark_thread_started("openai")
    while True:
        try:
            refresh_openai_dashboard()
            set_backend_thread("openai", "running", "OpenAI dashboard synchronized.")
        except Exception as exc:
            set_backend_thread("openai", "crashed", "OpenAI background refresh failed.", error=str(exc))
        time.sleep(OPENAI_SYNC_INTERVAL_SEC)


def start_background_workers() -> None:
    global WORKERS_STARTED
    with WORKER_LOCK:
        if WORKERS_STARTED:
            return
        load_runtime_snapshot()
        threads = [
            threading.Thread(target=scan_thread, name="scan-thread", daemon=True),
            threading.Thread(target=positions_thread, name="positions-thread", daemon=True),
            threading.Thread(target=openai_thread, name="openai-thread", daemon=True),
        ]
        for thread in threads:
            thread.start()
        WORKERS_STARTED = True
        sync_runtime_views()


def post_fork(server, worker) -> None:
    start_background_workers()


@app.route("/")
def index():
    start_background_workers()
    return render_template("index.html")


@app.route("/api/state_lite")
def api_state_lite():
    start_background_workers()
    payload = state_lite_cache.get_or_build(
        lambda: build_state_lite_payload(
            {
                **STATE,
                "trend_dashboard": dict(TREND_DASHBOARD),
                "ai_panel": dict(AI_PANEL),
                "auto_backtest": dict(AUTO_BACKTEST_STATE),
            }
        ),
        force=bool(request.args.get("force")),
    )
    return jsonify(payload)


@app.route("/api/positions_state")
def api_positions_state():
    start_background_workers()
    payload = positions_cache.get_or_build(
        lambda: build_positions_payload(dict(STATE)),
        force=bool(request.args.get("force")),
    )
    return jsonify(payload)


@app.route("/api/ai_panel_state")
def api_ai_panel_state():
    start_background_workers()
    payload = ai_panel_cache.get_or_build(
        lambda: build_ai_panel_payload(
            {
                **STATE,
                "ai_panel": dict(AI_PANEL),
                "auto_backtest": dict(AUTO_BACKTEST_STATE),
                "trend_dashboard": dict(TREND_DASHBOARD),
            }
        ),
        force=bool(request.args.get("force")),
    )
    return jsonify(payload)


@app.route("/api/ai_status")
def api_ai_status():
    start_background_workers()
    return jsonify(
        {
            "ok": True,
            "ai_panel": dict(AI_PANEL),
            "auto_backtest": dict(AUTO_BACKTEST_STATE),
            "backend_threads": dict(STATE.get("backend_threads") or {}),
        }
    )


@app.route("/api/ai_db_stats")
def api_ai_db_stats():
    start_background_workers()
    live_total = safe_int((STATE.get("learn_summary") or {}).get("total_trades"), 0)
    return jsonify(
        {
            "ok": True,
            "mode": "openai_scan_only",
            "ai_ready": bool((AI_PANEL.get("openai_trade") or {}).get("api_key_present")),
            "total_live": live_total,
            "last_learning": AI_PANEL.get("last_learning", "disabled"),
            "last_backtest": AI_PANEL.get("last_backtest", "disabled"),
        }
    )


@app.route("/api/backtest")
def api_backtest():
    return jsonify(
        {
            "ok": False,
            "error": "Backtest is disabled in this deployment. Scan, positions, OpenAI, and UI remain active.",
        }
    )


@app.route("/api/force_backtest", methods=["POST"])
def api_force_backtest():
    return jsonify(
        {
            "ok": False,
            "error": "Backtest is disabled in this deployment.",
            "summary": AUTO_BACKTEST_STATE.get("summary", ""),
        }
    )


@app.route("/api/cancel_fvg_order", methods=["POST"])
def api_cancel_fvg_order():
    payload = request.get_json(silent=True) or {}
    symbol = str(payload.get("symbol") or "")
    with STATE_LOCK:
        fvg_orders = dict(STATE.get("fvg_orders") or {})
        if symbol in fvg_orders:
            del fvg_orders[symbol]
            STATE["fvg_orders"] = fvg_orders
    return jsonify({"ok": True, "message": "No active FVG order for {}.".format(symbol or "symbol")})


@app.route("/api/risk_override", methods=["POST"])
def api_risk_override():
    payload = request.get_json(silent=True) or {}
    action = str(payload.get("action") or "").lower()
    if action == "release":
        update_state(
            risk_status={
                "trading_ok": True,
                "halt_reason": "",
                "consecutive_loss": 0,
                "daily_loss_pct": 0.0,
            }
        )
        return jsonify({"ok": True, "message": "Risk hold released."})
    return jsonify({"ok": False, "message": "Unsupported action."}), 400


def close_position_market(position: Dict[str, Any]) -> Dict[str, Any]:
    symbol = str(position.get("symbol") or "")
    side = str(position.get("side") or "").lower()
    contracts = abs(safe_float(position.get("contracts"), 0.0))
    if not symbol or contracts <= 0:
        return {"symbol": symbol, "ok": False, "error": "Invalid position payload."}
    close_side = "sell" if side == "long" else "buy"
    params: Dict[str, Any] = {"reduceOnly": True}
    if side in ("long", "short"):
        params["posSide"] = side
    try:
        exchange.create_order(symbol, "market", close_side, contracts, None, params)
        append_trade_history(
            {
                "time": tw_now_str(),
                "symbol": symbol,
                "side": "close_{}".format(side),
                "price": safe_float(position.get("markPrice"), position.get("entryPrice")),
                "score": 0,
                "stop_loss": 0,
                "take_profit": 0,
                "pnl_pct": safe_float(position.get("percentage"), 0.0),
            }
        )
        return {"symbol": symbol, "ok": True}
    except Exception as exc:
        return {"symbol": symbol, "ok": False, "error": str(exc)[:220]}


@app.route("/api/close_all", methods=["POST"])
def api_close_all():
    start_background_workers()
    with STATE_LOCK:
        active = list(STATE.get("active_positions") or [])
    results = [close_position_market(row) for row in active]
    sync_positions_once()
    ok_count = sum(1 for row in results if row.get("ok"))
    return jsonify({"ok": True, "closed": ok_count, "results": results})


if __name__ == "__main__":
    start_background_workers()
    app.run(host="0.0.0.0", port=int(env_or_blank("PORT", "5000") or 5000))
