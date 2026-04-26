from __future__ import annotations

import math
import os
import threading
import time
import traceback
import json
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
MAX_ACTIVE_POSITION_SCAN_INTERVAL_SEC = max(5, int(float(env_or_blank("MAX_ACTIVE_POSITION_SCAN_INTERVAL_SEC", "12") or 12)))
ACTIVE_POSITION_PRICE_POLL_SEC = max(2, int(float(env_or_blank("ACTIVE_POSITION_PRICE_POLL_SEC", "3") or 3)))
OPENAI_SYNC_INTERVAL_SEC = max(20, int(float(env_or_blank("OPENAI_SYNC_INTERVAL_SEC", "30") or 30)))
SCAN_SYMBOL_LIMIT = max(8, min(36, int(float(env_or_blank("SCAN_SYMBOL_LIMIT", "18") or 18))))
TOP_SIGNAL_LIMIT = max(5, min(15, int(float(env_or_blank("TOP_SIGNAL_LIMIT", "10") or 10))))
TIMEFRAME_BAR_LIMIT = max(100, min(240, int(float(env_or_blank("SCAN_BAR_LIMIT", "120") or 120))))
ACTIVE_HISTORY_LIMIT = 40
GENERAL_TOP_PICK = 5
SHORT_GAINER_TOP_PICK = 3
SHORT_GAINER_MIN_24H_PCT = max(20.0, float(env_or_blank("SHORT_GAINER_MIN_24H_PCT", "40") or 40))
MAX_OPEN_POSITIONS = 5
AI_SKIP_COOLDOWN_SEC = 60 * 60
FIXED_ORDER_NOTIONAL_USDT = max(5.0, float(env_or_blank("FIXED_ORDER_NOTIONAL_USDT", "40") or 40))
BTC_ETH_MIN_ORDER_NOTIONAL_USDT = max(FIXED_ORDER_NOTIONAL_USDT, float(env_or_blank("BTC_ETH_MIN_ORDER_NOTIONAL_USDT", "150") or 150))
HIGH_NOTIONAL_SYMBOLS = {"BTC", "ETH", "XAU", "XAG", "SOL"}
MIN_SYMBOL_QUOTE_VOLUME = max(100_000.0, float(env_or_blank("SCAN_MIN_QUOTE_VOLUME", "300000") or 300000))

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
ORDER_LOCK = threading.RLock()
REVIEW_LOCK = threading.RLock()

WORKERS_STARTED = False
OPENAI_API_KEY = env_or_blank("OPENAI_API_KEY")
OPENAI_TRADE_CONFIG = default_trade_config(lambda name, default="": env_or_blank(name, default))
OPENAI_TRADE_CONFIG["cooldown_minutes"] = min(int(OPENAI_TRADE_CONFIG.get("cooldown_minutes", 60) or 60), 60)
OPENAI_TRADE_CONFIG["same_payload_reuse_minutes"] = min(int(OPENAI_TRADE_CONFIG.get("same_payload_reuse_minutes", 60) or 60), 60)
OPENAI_TRADE_CONFIG["global_min_interval_minutes"] = 0
OPENAI_TRADE_CONFIG["top_k_per_scan"] = GENERAL_TOP_PICK
OPENAI_TRADE_CONFIG["sends_per_scan"] = 1
OPENAI_TRADE_CONFIG["min_score_abs"] = 0.0
OPENAI_TRADE_STATE = load_trade_state(OPENAI_TRADE_STATE_PATH)
MARKET_CAP_CACHE: Dict[str, Dict[str, Any]] = {}
MARKET_CAP_CACHE_TS: Dict[str, float] = {}
DERIVATIVES_CACHE: Dict[str, Dict[str, Any]] = {}
DERIVATIVES_CACHE_TS: Dict[str, float] = {}
OHLCV_CACHE: Dict[str, pd.DataFrame] = {}
OHLCV_CACHE_TS: Dict[str, float] = {}
OPEN_INTEREST_SNAPSHOT: Dict[str, Dict[str, float]] = {}
REVIEW_TRACKER: Dict[str, Dict[str, Any]] = {}
POSITION_RULES: Dict[str, Dict[str, Any]] = {}


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


def base_asset(symbol: str) -> str:
    token = compact_symbol(symbol)
    if token.endswith("USDT"):
        token = token[:-4]
    return token.upper()


def fixed_order_notional_usdt(symbol: str) -> float:
    return BTC_ETH_MIN_ORDER_NOTIONAL_USDT if base_asset(symbol) in HIGH_NOTIONAL_SYMBOLS else FIXED_ORDER_NOTIONAL_USDT


def now_ts() -> float:
    return time.time()


def pct_change(current: float, previous: float) -> float:
    current = safe_float(current, 0.0)
    previous = safe_float(previous, 0.0)
    if previous <= 0:
        return 0.0
    return ((current / previous) - 1.0) * 100.0


def linear_score(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp(((safe_float(value, low) - low) / (high - low)) * 100.0, 0.0, 100.0)


def extract_recent_bars(context: Dict[str, Any], timeframe: str, limit: int = 12) -> List[Dict[str, Any]]:
    rows = list((context.get("timeframe_bars") or {}).get(timeframe) or [])
    return [dict(row or {}) for row in rows[-max(limit, 1):]]


def candle_shape_metrics(bar: Dict[str, Any]) -> Dict[str, float]:
    high = safe_float(bar.get("high"), 0.0)
    low = safe_float(bar.get("low"), 0.0)
    open_price = safe_float(bar.get("open"), 0.0)
    close_price = safe_float(bar.get("close"), 0.0)
    candle_range = max(high - low, 1e-9)
    body = abs(close_price - open_price)
    upper_wick = max(high - max(open_price, close_price), 0.0)
    lower_wick = max(min(open_price, close_price) - low, 0.0)
    return {
        "range": candle_range,
        "body_pct": (body / candle_range) * 100.0,
        "upper_wick_pct": (upper_wick / candle_range) * 100.0,
        "lower_wick_pct": (lower_wick / candle_range) * 100.0,
    }


def count_monotonic(values: List[float], direction: str) -> int:
    total = 0
    for idx in range(1, len(values)):
        if direction == "up" and values[idx] > values[idx - 1]:
            total += 1
        if direction == "down" and values[idx] < values[idx - 1]:
            total += 1
    return total


def structure_profile(context: Dict[str, Any], timeframe: str = "15m") -> Dict[str, Any]:
    bars = extract_recent_bars(context, timeframe, 8)
    highs = [safe_float(row.get("high"), 0.0) for row in bars]
    lows = [safe_float(row.get("low"), 0.0) for row in bars]
    closes = [safe_float(row.get("close"), 0.0) for row in bars]
    hh = count_monotonic(highs[-5:], "up")
    hl = count_monotonic(lows[-5:], "up")
    lh = count_monotonic(highs[-5:], "down")
    ll = count_monotonic(lows[-5:], "down")
    upper_wicks = sum(1 for row in bars[-3:] if candle_shape_metrics(row)["upper_wick_pct"] >= 40.0)
    lower_wicks = sum(1 for row in bars[-3:] if candle_shape_metrics(row)["lower_wick_pct"] >= 40.0)
    last_close = closes[-1] if closes else 0.0
    prior_high = max(highs[:-1], default=last_close)
    prior_low = min(lows[:-1], default=last_close)
    return {
        "hh_count": hh,
        "hl_count": hl,
        "lh_count": lh,
        "ll_count": ll,
        "upper_wick_count": upper_wicks,
        "lower_wick_count": lower_wicks,
        "close_above_prior_high": bool(last_close > prior_high and prior_high > 0),
        "close_below_prior_low": bool(last_close < prior_low and prior_low > 0),
        "prior_high": prior_high,
        "prior_low": prior_low,
    }


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
    "general_top_signals": [],
    "short_gainer_signals": [],
    "watchlist": [],
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
            "watchlist": list(STATE.get("watchlist", []))[:20],
            "fvg_orders": dict(STATE.get("fvg_orders") or {}),
            "general_top_signals": list(STATE.get("general_top_signals", []))[:GENERAL_TOP_PICK],
            "short_gainer_signals": list(STATE.get("short_gainer_signals", []))[:SHORT_GAINER_TOP_PICK],
            "trailing_info": dict(STATE.get("trailing_info") or {}),
        },
        "review_tracker": {
            key: value
            for key, value in list(REVIEW_TRACKER.items())[:80]
            if isinstance(value, dict)
        },
        "position_rules": {
            key: value
            for key, value in list(POSITION_RULES.items())[:80]
            if isinstance(value, dict)
        },
    }
    try:
        with open(SNAPSHOT_STATE_PATH, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_runtime_snapshot() -> None:
    global REVIEW_TRACKER, POSITION_RULES
    if not os.path.exists(SNAPSHOT_STATE_PATH):
        return
    try:
        with open(SNAPSHOT_STATE_PATH, "r", encoding="utf-8") as fh:
            payload = json.load(fh) or {}
        base = dict(payload.get("state") or {})
        with STATE_LOCK:
            if isinstance(base.get("trade_history"), list):
                STATE["trade_history"] = list(base["trade_history"])[-ACTIVE_HISTORY_LIMIT:]
            if base.get("latest_news_title"):
                STATE["latest_news_title"] = str(base["latest_news_title"])
            if isinstance(base.get("watchlist"), list):
                STATE["watchlist"] = list(base["watchlist"])[:20]
            if isinstance(base.get("fvg_orders"), dict):
                STATE["fvg_orders"] = dict(base["fvg_orders"])
            if isinstance(base.get("general_top_signals"), list):
                STATE["general_top_signals"] = list(base["general_top_signals"])[:GENERAL_TOP_PICK]
            if isinstance(base.get("short_gainer_signals"), list):
                STATE["short_gainer_signals"] = list(base["short_gainer_signals"])[:SHORT_GAINER_TOP_PICK]
            if isinstance(base.get("trailing_info"), dict):
                STATE["trailing_info"] = dict(base["trailing_info"])
        if isinstance(payload.get("review_tracker"), dict):
            REVIEW_TRACKER = dict(payload["review_tracker"])
        if isinstance(payload.get("position_rules"), dict):
            POSITION_RULES = dict(payload["position_rules"])
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


def is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return "429" in text or "too many requests" in text or "rate limit" in text


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
            "available": True,
            "market_cap_usd": safe_float(first.get("market_cap"), 0.0),
            "fdv_usd": safe_float(first.get("fully_diluted_valuation"), 0.0),
            "circulating_supply": safe_float(first.get("circulating_supply"), 0.0),
            "total_supply": safe_float(first.get("total_supply"), 0.0),
        }
    else:
        row = {
            "available": False,
            "market_cap_usd": None,
            "fdv_usd": None,
            "circulating_supply": None,
            "total_supply": None,
        }
    MARKET_CAP_CACHE[base] = row
    MARKET_CAP_CACHE_TS[base] = now
    return dict(row)


def safe_fetch_ohlcv_df(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    cache_key = "{}|{}|{}".format(symbol, timeframe, limit)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not rows:
                break
            df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"])
            for col in ["o", "h", "l", "c", "v"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna().sort_values("t").reset_index(drop=True)
            try:
                df["dt"] = pd.to_datetime(df["t"], unit="ms", utc=True, errors="coerce")
                df = df.dropna(subset=["dt"]).set_index("dt", drop=False)
            except Exception:
                pass
            OHLCV_CACHE[cache_key] = df.copy()
            OHLCV_CACHE_TS[cache_key] = time.time()
            return df
        except Exception as exc:
            last_error = exc
            if is_rate_limit_error(exc) and attempt < 2:
                time.sleep(0.6 * (attempt + 1))
                continue
            break
    cached = OHLCV_CACHE.get(cache_key)
    if isinstance(cached, pd.DataFrame) and not cached.empty and (time.time() - OHLCV_CACHE_TS.get(cache_key, 0.0)) <= 20 * 60:
        return cached.copy()
    if last_error:
        raise last_error
    return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v"])


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
    try:
        vwap = ta.vwap(h, l, c, v)
    except Exception:
        typical = (h + l + c) / 3.0
        vwap = (typical * v).cumsum() / v.cumsum().replace(0, pd.NA)
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
    bb_upper = 0.0
    bb_mid = 0.0
    bb_lower = 0.0
    if isinstance(bb, pd.DataFrame) and not bb.empty:
        cols = [str(col) for col in bb.columns]
        upper_col = next((col for col in bb.columns if str(col).upper().startswith("BBU")), bb.columns[-1])
        mid_col = next((col for col in bb.columns if str(col).upper().startswith("BBM")), bb.columns[min(1, len(cols) - 1)])
        lower_col = next((col for col in bb.columns if str(col).upper().startswith("BBL")), bb.columns[0])
        bb_upper = last_value(bb[upper_col], 0.0)
        bb_mid = last_value(bb[mid_col], 0.0)
        bb_lower = last_value(bb[lower_col], 0.0)
    stoch_k = last_value(stoch.iloc[:, 0], 0.0) if isinstance(stoch, pd.DataFrame) else 0.0
    stoch_d = last_value(stoch.iloc[:, 1], 0.0) if isinstance(stoch, pd.DataFrame) else 0.0
    bb_width = max(bb_upper - bb_lower, 0.0)
    bb_position = ((last_close - bb_lower) / bb_width) * 100 if bb_width > 1e-9 else 50.0

    trend_label = "neutral"
    if last_value(ema20, last_close) > last_value(ema50, last_close) > last_value(ema200, last_close):
        trend_label = "uptrend"
    elif last_value(ema20, last_close) < last_value(ema50, last_close) < last_value(ema200, last_close):
        trend_label = "downtrend"

    recent_high = safe_float(h.tail(20).max(), last_close)
    recent_low = safe_float(l.tail(20).min(), last_close)
    prior_window_high = h.iloc[-7:-1] if len(h) >= 7 else h.iloc[:-1]
    prior_window_low = l.iloc[-7:-1] if len(l) >= 7 else l.iloc[:-1]
    prior_structure_high = safe_float(prior_window_high.max(), recent_high) if len(prior_window_high) > 0 else recent_high
    prior_structure_low = safe_float(prior_window_low.min(), recent_low) if len(prior_window_low) > 0 else recent_low
    current_bar_high = safe_float(h.iloc[-1], last_close)
    current_bar_low = safe_float(l.iloc[-1], last_close)
    current_bar_range = max(current_bar_high - current_bar_low, 0.0)
    current_bar_range_atr = current_bar_range / max(atr_last, 1e-9)
    support_levels = [safe_float(l.tail(10).min(), recent_low), recent_low]
    resistance_levels = [safe_float(h.tail(10).max(), recent_high), recent_high]
    avg_volume_20 = last_value(avg_vol20, 0.0)
    avg_volume_60 = last_value(avg_vol60, avg_volume_20)
    last_volume = safe_float(v.iloc[-1], 0.0)
    ret_3bars_pct = round(((last_close / safe_float(c.iloc[-4], last_close)) - 1) * 100 if len(c) >= 4 else 0.0, 4)
    ret_12bars_pct = round(((last_close / safe_float(c.iloc[-13], last_close)) - 1) * 100 if len(c) >= 13 else 0.0, 4)
    ret_24bars_pct = round(((last_close / safe_float(c.iloc[-25], last_close)) - 1) * 100 if len(c) >= 25 else 0.0, 4)
    vol_ratio = round((last_volume / max(avg_volume_20, 1e-9)), 4)
    explosive_move = abs(ret_3bars_pct) >= 5.5 or (abs(ret_12bars_pct) >= 9.0 and current_bar_range_atr >= 1.35 and vol_ratio >= 1.5)

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
        "bb_width_pct": round((bb_width / last_close) * 100 if last_close > 0 else 0.0, 4),
        "bb_position_pct": round(clamp(bb_position, 0.0, 100.0), 4),
        "stoch_k": round(stoch_k, 4),
        "stoch_d": round(stoch_d, 4),
        "kdj_j": round((3 * stoch_k) - (2 * stoch_d), 4),
        "ret_3bars_pct": ret_3bars_pct,
        "ret_12bars_pct": ret_12bars_pct,
        "ret_24bars_pct": ret_24bars_pct,
        "vol_ratio": vol_ratio,
        "avg_volume_20": round(avg_volume_20, 4),
        "avg_volume_60": round(avg_volume_60, 4),
        "last_volume": round(last_volume, 4),
        "swing_high_20": round(recent_high, 6),
        "swing_low_20": round(recent_low, 6),
        "prior_structure_high_6": round(prior_structure_high, 6),
        "prior_structure_low_6": round(prior_structure_low, 6),
        "current_bar_high": round(current_bar_high, 6),
        "current_bar_low": round(current_bar_low, 6),
        "current_bar_range_atr": round(current_bar_range_atr, 4),
        "explosive_move": bool(explosive_move),
        "distance_to_swing_high_pct": round(((recent_high - last_close) / max(last_close, 1e-9)) * 100, 4),
        "distance_to_swing_low_pct": round(((last_close - recent_low) / max(last_close, 1e-9)) * 100, 4),
        "support_levels": [round(x, 6) for x in support_levels if x > 0],
        "resistance_levels": [round(x, 6) for x in resistance_levels if x > 0],
        "recent_structure_high": round(recent_high, 6),
        "recent_structure_low": round(recent_low, 6),
    }


def build_liquidity_context(symbol: str, ticker: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "spread_pct": None,
        "bid_depth_5": None,
        "ask_depth_5": None,
        "bid_depth_10": None,
        "ask_depth_10": None,
        "depth_imbalance_10": None,
        "largest_bid_wall_price": None,
        "largest_bid_wall_size": None,
        "largest_ask_wall_price": None,
        "largest_ask_wall_size": None,
        "recent_trades_count": 0,
        "aggressive_buy_volume": None,
        "aggressive_sell_volume": None,
        "aggressive_buy_notional": None,
        "aggressive_sell_notional": None,
        "buy_sell_notional_ratio": None,
        "cvd_notional": None,
        "cvd_bias": "neutral",
        "volume_anomaly_5m": None,
        "volume_anomaly_15m": None,
        "order_book_available": False,
        "recent_trades_available": False,
        "errors": [],
    }
    bid = safe_float(ticker.get("bid"), 0.0)
    ask = safe_float(ticker.get("ask"), 0.0)
    last = safe_float(ticker.get("last"), 0.0)
    if bid > 0 and ask > 0 and last > 0:
        result["spread_pct"] = round(((ask - bid) / last) * 100, 6)
    try:
        book = None
        for attempt in range(3):
            try:
                book = exchange.fetch_order_book(symbol, limit=10)
                break
            except Exception as exc:
                if is_rate_limit_error(exc) and attempt < 2:
                    time.sleep(0.35 * (attempt + 1))
                    continue
                raise
        bids = list(book.get("bids") or [])
        asks = list(book.get("asks") or [])
        result["bid_depth_5"] = round(sum(safe_float(row[1], 0.0) for row in bids[:5]), 4)
        result["ask_depth_5"] = round(sum(safe_float(row[1], 0.0) for row in asks[:5]), 4)
        result["bid_depth_10"] = round(sum(safe_float(row[1], 0.0) for row in bids[:10]), 4)
        result["ask_depth_10"] = round(sum(safe_float(row[1], 0.0) for row in asks[:10]), 4)
        denom = max(result["bid_depth_10"] + result["ask_depth_10"], 1e-9)
        result["depth_imbalance_10"] = round((result["bid_depth_10"] - result["ask_depth_10"]) / denom, 6)
        if bids:
            largest_bid = max(bids[:10], key=lambda row: safe_float(row[1], 0.0))
            result["largest_bid_wall_price"] = round(safe_float(largest_bid[0], 0.0), 6)
            result["largest_bid_wall_size"] = round(safe_float(largest_bid[1], 0.0), 4)
        if asks:
            largest_ask = max(asks[:10], key=lambda row: safe_float(row[1], 0.0))
            result["largest_ask_wall_price"] = round(safe_float(largest_ask[0], 0.0), 6)
            result["largest_ask_wall_size"] = round(safe_float(largest_ask[1], 0.0), 4)
        result["order_book_available"] = True
    except Exception as exc:
        result["errors"].append("order_book:{}".format(str(exc)[:120]))
    try:
        trades = None
        for attempt in range(3):
            try:
                trades = list(exchange.fetch_trades(symbol, limit=120) or [])
                break
            except Exception as exc:
                if is_rate_limit_error(exc) and attempt < 2:
                    time.sleep(0.35 * (attempt + 1))
                    continue
                raise
        trades = list(trades or [])
        result["recent_trades_count"] = len(trades)
        now_ms = int(time.time() * 1000)
        buy_notional = 0.0
        sell_notional = 0.0
        buy_volume = 0.0
        sell_volume = 0.0
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
                buy_volume += amount
            elif side == "sell":
                sell_notional += notional
                sell_volume += amount
            if age_ms <= 5 * 60 * 1000:
                vol_5m += notional
            elif age_ms <= 10 * 60 * 1000:
                vol_prev_5m += notional
            if age_ms <= 15 * 60 * 1000:
                vol_15m += notional
            elif age_ms <= 30 * 60 * 1000:
                vol_prev_15m += notional
        result["aggressive_buy_volume"] = round(buy_volume, 4)
        result["aggressive_sell_volume"] = round(sell_volume, 4)
        result["aggressive_buy_notional"] = round(buy_notional, 4)
        result["aggressive_sell_notional"] = round(sell_notional, 4)
        result["buy_sell_notional_ratio"] = round(buy_notional / max(sell_notional, 1e-9), 4) if (buy_notional > 0 or sell_notional > 0) else None
        result["cvd_notional"] = round(buy_notional - sell_notional, 4)
        if buy_notional > sell_notional * 1.15:
            result["cvd_bias"] = "buying"
        elif sell_notional > buy_notional * 1.15:
            result["cvd_bias"] = "selling"
        result["volume_anomaly_5m"] = round(vol_5m / vol_prev_5m, 4) if vol_prev_5m > 0 else None
        result["volume_anomaly_15m"] = round(vol_15m / vol_prev_15m, 4) if vol_prev_15m > 0 else None
        result["recent_trades_available"] = True
    except Exception as exc:
        result["errors"].append("trades:{}".format(str(exc)[:120]))
    return result


def build_derivatives_context(symbol: str, ticker: Dict[str, Any]) -> Dict[str, Any]:
    now = time.time()
    if symbol in DERIVATIVES_CACHE and (now - DERIVATIVES_CACHE_TS.get(symbol, 0)) < 45:
        return dict(DERIVATIVES_CACHE[symbol])
    result = {
        "funding_rate": None,
        "next_funding_time": "",
        "open_interest": None,
        "open_interest_value_usdt": None,
        "open_interest_change_pct_5m": None,
        "long_short_ratio": None,
        "top_trader_long_short_ratio": None,
        "whale_position_change_pct": None,
        "basis_pct": None,
        "mark_price": safe_float((ticker.get("info") or {}).get("markPrice"), safe_float(ticker.get("last"), 0.0)),
        "index_price": safe_float((ticker.get("info") or {}).get("indexPrice"), safe_float(ticker.get("last"), 0.0)),
        "liquidation_volume_24h": None,
        "liquidation_map_status": "unavailable",
        "leverage_heat": "unavailable",
        "leverage_heat_score": None,
        "funding_available": False,
        "open_interest_available": False,
        "errors": [],
    }
    try:
        fetch_funding_rate = getattr(exchange, "fetch_funding_rate", None)
        if callable(fetch_funding_rate):
            funding = None
            for attempt in range(3):
                try:
                    funding = dict(fetch_funding_rate(symbol) or {})
                    break
                except Exception as exc:
                    if is_rate_limit_error(exc) and attempt < 2:
                        time.sleep(0.35 * (attempt + 1))
                        continue
                    raise
            funding = dict(funding or {})
            result["funding_rate"] = round(safe_float(funding.get("fundingRate", funding.get("funding_rate")), 0.0), 8)
            result["next_funding_time"] = str(funding.get("fundingDatetime") or funding.get("nextFundingTime") or "")[:40]
            result["funding_available"] = True
    except Exception as exc:
        result["errors"].append("funding:{}".format(str(exc)[:120]))
    try:
        fetch_open_interest = getattr(exchange, "fetch_open_interest", None)
        if callable(fetch_open_interest):
            oi = None
            for attempt in range(3):
                try:
                    oi = dict(fetch_open_interest(symbol) or {})
                    break
                except Exception as exc:
                    if is_rate_limit_error(exc) and attempt < 2:
                        time.sleep(0.35 * (attempt + 1))
                        continue
                    raise
            oi = dict(oi or {})
            result["open_interest"] = round(safe_float(oi.get("openInterestAmount", oi.get("openInterest", oi.get("amount"))), 0.0), 4)
            result["open_interest_value_usdt"] = round(safe_float(oi.get("openInterestValue", oi.get("value")), 0.0), 4)
            result["open_interest_available"] = True
    except Exception as exc:
        result["errors"].append("open_interest:{}".format(str(exc)[:120]))
    if safe_float(result.get("open_interest_value_usdt"), 0.0) <= 0 and safe_float(result.get("open_interest"), 0.0) > 0 and safe_float(result.get("mark_price"), 0.0) > 0:
        result["open_interest_value_usdt"] = round(safe_float(result.get("open_interest"), 0.0) * safe_float(result.get("mark_price"), 0.0), 4)
    prev_snapshot = dict(OPEN_INTEREST_SNAPSHOT.get(symbol) or {})
    prev_oi = safe_float(prev_snapshot.get("open_interest"), 0.0)
    prev_ts = safe_float(prev_snapshot.get("ts"), 0.0)
    curr_oi = safe_float(result.get("open_interest"), 0.0)
    now_ts_val = time.time()
    if curr_oi > 0 and prev_oi > 0 and prev_ts > 0 and (now_ts_val - prev_ts) <= 15 * 60:
        result["open_interest_change_pct_5m"] = round(((curr_oi / prev_oi) - 1.0) * 100.0, 4)
    OPEN_INTEREST_SNAPSHOT[symbol] = {"open_interest": curr_oi, "ts": now_ts_val}
    info = dict(ticker.get("info") or {})
    buy_vol = safe_float(info.get("buyVolume"), 0.0)
    sell_vol = safe_float(info.get("sellVolume"), 0.0)
    if buy_vol > 0 or sell_vol > 0:
        result["long_short_ratio"] = round((buy_vol + 1e-9) / max(sell_vol, 1e-9), 4)
        result["top_trader_long_short_ratio"] = result["long_short_ratio"]
        result["whale_position_change_pct"] = round(((buy_vol - sell_vol) / max(buy_vol + sell_vol, 1e-9)) * 100, 4)
    if result["mark_price"] > 0 and result["index_price"] > 0:
        result["basis_pct"] = round(((result["mark_price"] - result["index_price"]) / result["index_price"]) * 100, 6)
    heat = abs(safe_float(result["funding_rate"], 0.0)) * 10000.0
    if safe_float(result["open_interest_value_usdt"], 0.0) > 0:
        heat += min(result["open_interest_value_usdt"] / 1_000_000.0, 50.0)
    if result["funding_available"] or result["open_interest_available"]:
        result["leverage_heat_score"] = round(heat, 4)
        if heat >= 35:
            result["leverage_heat"] = "high"
        elif heat >= 18:
            result["leverage_heat"] = "elevated"
        else:
            result["leverage_heat"] = "normal"
    DERIVATIVES_CACHE[symbol] = result
    DERIVATIVES_CACHE_TS[symbol] = now
    return dict(result)


def infer_setup(tf15: Dict[str, Any], tf1h: Dict[str, Any], tf4h: Dict[str, Any], side: str, structure: Dict[str, Any], risk_score: float) -> str:
    if side == "short" and risk_score >= 50:
        return "reversal_short"
    if side == "long" and tf15.get("ema20", 0) > tf15.get("ema50", 0) > tf15.get("ema200", 0) and tf1h.get("ema20", 0) > tf1h.get("ema50", 0):
        return "trend_continuation_long"
    if side == "short" and tf15.get("ema20", 0) < tf15.get("ema50", 0) < tf15.get("ema200", 0) and tf1h.get("ema20", 0) < tf1h.get("ema50", 0):
        return "trend_continuation_short"
    if side == "short" and structure.get("close_above_prior_high"):
        return "reversal_short" if risk_score >= 28 else "countertrend_short_watch"
    if side == "long" and structure.get("close_below_prior_low"):
        return "pullback_long_watch"
    if structure.get("close_above_prior_high"):
        return "breakout_long"
    if structure.get("close_below_prior_low"):
        return "breakdown_short"
    if safe_float(tf15.get("bb_width_pct"), 0.0) < 3.2 and safe_float(tf15.get("adx"), 0.0) < 20:
        return "compression_watch"
    if tf4h.get("trend_label") == "uptrend":
        return "higher_timeframe_long"
    if tf4h.get("trend_label") == "downtrend":
        return "higher_timeframe_short"
    return "range_watch"


def score_trend_component(side: str, context: Dict[str, Any], structure: Dict[str, Any]) -> Dict[str, Any]:
    tf15 = dict((context.get("multi_timeframe") or {}).get("15m") or {})
    tf1h = dict((context.get("multi_timeframe") or {}).get("1h") or {})
    price = safe_float((context.get("basic_market_data") or {}).get("current_price"), 0.0)
    score = 0.0
    notes: List[str] = []
    if side == "long":
        if tf15.get("ema20", 0) > tf15.get("ema50", 0) > tf15.get("ema200", 0):
            score += 28
            notes.append("15m EMA stack bullish")
        if tf1h.get("ema20", 0) > tf1h.get("ema50", 0):
            score += 18
            notes.append("1h trend aligned")
        if safe_float(tf15.get("adx"), 0.0) > 20 and safe_float(tf15.get("plus_di"), 0.0) >= safe_float(tf15.get("minus_di"), 0.0):
            score += 20
        score += min(structure.get("hh_count", 0) * 8 + structure.get("hl_count", 0) * 8, 24)
        if price > safe_float(tf15.get("vwap"), price):
            score += 10
    else:
        if tf15.get("ema20", 0) < tf15.get("ema50", 0) < tf15.get("ema200", 0):
            score += 28
            notes.append("15m EMA stack bearish")
        if tf1h.get("ema20", 0) < tf1h.get("ema50", 0):
            score += 18
            notes.append("1h trend aligned short")
        if safe_float(tf15.get("adx"), 0.0) > 20 and safe_float(tf15.get("minus_di"), 0.0) >= safe_float(tf15.get("plus_di"), 0.0):
            score += 20
        score += min(structure.get("lh_count", 0) * 8 + structure.get("ll_count", 0) * 8, 24)
        if price < safe_float(tf15.get("vwap"), price):
            score += 10
    return {"score": clamp(score, 0, 100), "notes": notes}


def score_momentum_component(side: str, context: Dict[str, Any], structure: Dict[str, Any]) -> Dict[str, Any]:
    tf5 = dict((context.get("multi_timeframe") or {}).get("5m") or {})
    tf15 = dict((context.get("multi_timeframe") or {}).get("15m") or {})
    score = 0.0
    notes: List[str] = []
    if max(safe_float(tf5.get("vol_ratio"), 0.0), safe_float(tf15.get("vol_ratio"), 0.0)) >= 1.8:
        score += 24
        notes.append("volume expansion")
    if side == "long":
        if safe_float(tf5.get("macd_hist"), 0.0) > 0 and safe_float(tf5.get("macd"), 0.0) >= safe_float(tf5.get("macd_signal"), 0.0):
            score += 18
        if 55 <= safe_float(tf15.get("rsi"), 50.0) <= 68:
            score += 16
        if structure.get("close_above_prior_high"):
            score += 24
            notes.append("breakout confirmed")
    else:
        if safe_float(tf5.get("macd_hist"), 0.0) < 0 and safe_float(tf5.get("macd"), 0.0) <= safe_float(tf5.get("macd_signal"), 0.0):
            score += 18
        if 32 <= safe_float(tf15.get("rsi"), 50.0) <= 45:
            score += 16
        if structure.get("close_below_prior_low"):
            score += 24
            notes.append("breakdown confirmed")
    if safe_float(tf15.get("bb_width_pct"), 0.0) <= 3.2 and max(safe_float(tf5.get("vol_ratio"), 0.0), safe_float(tf15.get("vol_ratio"), 0.0)) >= 1.3:
        score += 18
        notes.append("squeeze release")
    return {"score": clamp(score, 0, 100), "notes": notes}


def score_positioning_component(side: str, context: Dict[str, Any]) -> Dict[str, Any]:
    tf15 = dict((context.get("multi_timeframe") or {}).get("15m") or {})
    basic = dict(context.get("basic_market_data") or {})
    derivatives = dict(context.get("derivatives_context") or {})
    liquidity = dict(context.get("liquidity_context") or {})
    price_change = safe_float(tf15.get("ret_12bars_pct"), 0.0)
    oi_change = safe_float(derivatives.get("open_interest_change_pct_5m"), 0.0)
    funding = safe_float(derivatives.get("funding_rate"), 0.0)
    ratio = safe_float(derivatives.get("long_short_ratio"), 1.0)
    whale = safe_float(derivatives.get("whale_position_change_pct"), 0.0)
    cvd_bias = str(liquidity.get("cvd_bias") or "neutral")
    score = 0.0
    notes: List[str] = []
    if side == "long":
        if price_change > 0 and oi_change >= 0:
            score += 28
            notes.append("price + OI rising")
        if cvd_bias == "buying":
            score += 18
        if 0 <= funding <= 0.0008:
            score += 18
        elif funding > 0.0015:
            score -= 12
        if 0.85 <= ratio <= 1.85:
            score += 16
        if whale >= 0:
            score += min(20, abs(whale) * 0.5)
    else:
        if price_change < 0 and oi_change >= 0:
            score += 28
            notes.append("price down + OI rising")
        if cvd_bias == "selling":
            score += 18
        if -0.0008 <= funding <= 0.0008:
            score += 18
        elif funding > 0.002:
            score += 10
        if 0.85 <= ratio <= 1.85:
            score += 12
        elif ratio > 2.2:
            score += 18
        if whale <= 0:
            score += min(20, abs(whale) * 0.5)
    if safe_float(basic.get("open_interest_value_usdt"), 0.0) <= 0:
        score *= 0.72
    return {"score": clamp(score, 0, 100), "notes": notes}


def score_volatility_efficiency(context: Dict[str, Any]) -> Dict[str, Any]:
    tf15 = dict((context.get("multi_timeframe") or {}).get("15m") or {})
    tf1d = dict((context.get("multi_timeframe") or {}).get("1d") or {})
    liquidity = dict(context.get("liquidity_context") or {})
    price = safe_float((context.get("basic_market_data") or {}).get("current_price"), 0.0)
    daily_range_pct = pct_change(safe_float(tf1d.get("recent_structure_high"), price), safe_float(tf1d.get("recent_structure_low"), price))
    atr_pct = safe_float(tf15.get("atr_pct"), 0.0)
    spread = safe_float(liquidity.get("spread_pct"), 0.0)
    depth = safe_float(liquidity.get("bid_depth_10"), 0.0) + safe_float(liquidity.get("ask_depth_10"), 0.0)
    score = 0.0
    if atr_pct >= 0.45:
        score += 34
    if daily_range_pct >= 4.0:
        score += 30
    if 0 < spread <= 0.08:
        score += 18
    if depth >= 1500:
        score += 18
    return {"score": clamp(score, 0, 100), "notes": ["tradable volatility"] if score >= 45 else []}


def score_risk_component(side: str, context: Dict[str, Any], structure: Dict[str, Any]) -> Dict[str, Any]:
    tf5 = dict((context.get("multi_timeframe") or {}).get("5m") or {})
    tf15 = dict((context.get("multi_timeframe") or {}).get("15m") or {})
    liquidity = dict(context.get("liquidity_context") or {})
    derivatives = dict(context.get("derivatives_context") or {})
    score = 0.0
    notes: List[str] = []
    if max(safe_float(tf5.get("vol_ratio"), 0.0), safe_float(tf15.get("vol_ratio"), 0.0)) < 1.0 and (structure.get("close_above_prior_high") or structure.get("close_below_prior_low")):
        score += 24
        notes.append("breakout without follow-through volume")
    funding = abs(safe_float(derivatives.get("funding_rate"), 0.0))
    if funding >= 0.002:
        score += 20
        notes.append("funding overheated")
    if side == "long" and structure.get("upper_wick_count", 0) >= 2:
        score += 18
    if side == "short" and structure.get("lower_wick_count", 0) >= 2:
        score += 18
    if safe_float(liquidity.get("spread_pct"), 0.0) > 0.15:
        score += 18
    if safe_float(liquidity.get("bid_depth_10"), 0.0) + safe_float(liquidity.get("ask_depth_10"), 0.0) < 500:
        score += 16
    if len(list((context.get("news_context") or {}).get("items") or [])) > 0:
        score += 8
    return {"score": clamp(score, 0, 100), "notes": notes}


def score_relative_strength(side: str, context: Dict[str, Any], btc_context: Dict[str, Any] | None) -> Dict[str, Any]:
    tf15 = dict((context.get("multi_timeframe") or {}).get("15m") or {})
    tf1h = dict((context.get("multi_timeframe") or {}).get("1h") or {})
    btc15 = dict(((btc_context or {}).get("multi_timeframe") or {}).get("15m") or {})
    btc1h = dict(((btc_context or {}).get("multi_timeframe") or {}).get("1h") or {})
    if not btc_context:
        return {"score": 50.0, "notes": ["BTC benchmark unavailable"]}
    alt15 = safe_float(tf15.get("ret_12bars_pct"), 0.0)
    alt1h = safe_float(tf1h.get("ret_12bars_pct"), 0.0)
    btc15_ret = safe_float(btc15.get("ret_12bars_pct"), 0.0)
    btc1h_ret = safe_float(btc1h.get("ret_12bars_pct"), 0.0)
    score = 0.0
    notes: List[str] = []
    if side == "long":
        if abs(btc15_ret) <= 0.8 and alt15 > btc15_ret:
            score += 34
        if btc15_ret < 0 and alt15 >= btc15_ret:
            score += 30
        if btc15_ret > 0 and alt15 > btc15_ret * 1.2:
            score += 36
            notes.append("outperforming BTC")
    else:
        if btc15_ret >= 0 and alt15 < btc15_ret - 0.7:
            score += 34
        if btc15_ret < 0 and alt15 < btc15_ret:
            score += 30
        if btc1h_ret > 0 and alt1h < 0:
            score += 36
            notes.append("weak vs BTC")
    return {"score": clamp(score, 0, 100), "notes": notes}


def derive_trade_levels(side: str, context: Dict[str, Any], total_score: float) -> Dict[str, float]:
    tf15 = dict((context.get("multi_timeframe") or {}).get("15m") or {})
    price = safe_float((context.get("basic_market_data") or {}).get("current_price"), 0.0)
    atr = max(safe_float(tf15.get("atr"), 0.0), price * 0.003 if price > 0 else 0.0)
    support = safe_float(tf15.get("recent_structure_low"), price - atr * 1.6)
    resistance = safe_float(tf15.get("recent_structure_high"), price + atr * 1.6)
    prior_support = safe_float(tf15.get("prior_structure_low_6"), support)
    prior_resistance = safe_float(tf15.get("prior_structure_high_6"), resistance)
    current_bar_low = safe_float(tf15.get("current_bar_low"), 0.0)
    current_bar_high = safe_float(tf15.get("current_bar_high"), 0.0)
    explosive_move = bool(tf15.get("explosive_move", False))
    rr_target = 2.4 if total_score >= 72 else 2.0 if total_score >= 58 else 1.7
    stop_buffer_atr = 0.18
    min_stop_gap_atr = 0.95
    max_stop_gap_atr = 1.75
    if side == "long":
        entry = price
        stop_anchor = prior_support if 0 < prior_support < entry else support
        if explosive_move and 0 < current_bar_low < entry and (stop_anchor <= 0 or current_bar_low < stop_anchor):
            stop_anchor = current_bar_low
        structure_stop = stop_anchor - atr * stop_buffer_atr if 0 < stop_anchor < entry else entry - atr * min_stop_gap_atr
        stop = max(min(structure_stop, entry - atr * min_stop_gap_atr), entry - atr * max_stop_gap_atr, entry * 0.7)
        if stop >= entry:
            stop = entry - atr * min_stop_gap_atr
        take_profit = max(resistance, entry + max(entry - stop, atr) * rr_target)
        stop_anchor_used = stop_anchor
    else:
        entry = price
        stop_anchor = prior_resistance if prior_resistance > entry else resistance
        if explosive_move and current_bar_high > entry and current_bar_high > stop_anchor:
            stop_anchor = current_bar_high
        structure_stop = stop_anchor + atr * stop_buffer_atr if stop_anchor > entry else entry + atr * min_stop_gap_atr
        stop = min(max(structure_stop, entry + atr * min_stop_gap_atr), entry + atr * max_stop_gap_atr, entry * 1.3)
        if stop <= entry:
            stop = entry + atr * min_stop_gap_atr
        take_profit = min(support, entry - max(stop - entry, atr) * rr_target) if support > 0 else entry - max(stop - entry, atr) * rr_target
        stop_anchor_used = stop_anchor
    rr_ratio = abs((take_profit - entry) / max(abs(entry - stop), 1e-9))
    return {
        "price": round(entry, 6),
        "stop_loss": round(stop, 6),
        "take_profit": round(take_profit, 6),
        "rr_ratio": round(rr_ratio, 2),
        "atr": round(atr, 6),
        "stop_anchor_price": round(safe_float(stop_anchor_used, 0.0), 6),
        "stop_anchor_source": "current_bar_exception" if explosive_move and (
            (side == "long" and stop_anchor_used == current_bar_low)
            or (side == "short" and stop_anchor_used == current_bar_high)
        ) else "prior_swing_6",
        "stop_same_bar_exception": bool(explosive_move),
    }


def detect_short_reversal_signal(context: Dict[str, Any]) -> Dict[str, Any]:
    tf5 = dict((context.get("multi_timeframe") or {}).get("5m") or {})
    tf15 = dict((context.get("multi_timeframe") or {}).get("15m") or {})
    structure = structure_profile(context, "5m")
    liquidity = dict(context.get("liquidity_context") or {})
    triggers: List[str] = []
    if safe_float(tf5.get("rsi"), 50.0) >= 72 and safe_float(tf15.get("rsi"), 50.0) >= 65:
        triggers.append("RSI overextended")
    if safe_float(tf5.get("macd_hist"), 0.0) < 0 or safe_float(tf5.get("macd"), 0.0) < safe_float(tf5.get("macd_signal"), 0.0):
        triggers.append("5m MACD rolled over")
    if safe_float(tf5.get("last_close"), 0.0) < safe_float(tf5.get("vwap"), 0.0):
        triggers.append("lost VWAP")
    if structure.get("upper_wick_count", 0) >= 2:
        triggers.append("exhaustion wicks")
    if str(liquidity.get("cvd_bias") or "") == "selling":
        triggers.append("sell-side CVD")
    if safe_float(tf15.get("vol_ratio"), 0.0) >= 1.25:
        triggers.append("distribution volume")
    return {"ready": len(triggers) >= 3, "triggers": triggers[:5]}


def build_signal_from_context(symbol: str, market: Dict[str, Any], context: Dict[str, Any], btc_context: Dict[str, Any] | None = None, *, candidate_source: str = "general", forced_side: str | None = None) -> Dict[str, Any]:
    tf15 = dict((context.get("multi_timeframe") or {}).get("15m") or {})
    tf1h = dict((context.get("multi_timeframe") or {}).get("1h") or {})
    tf4h = dict((context.get("multi_timeframe") or {}).get("4h") or {})
    tf1d = dict((context.get("multi_timeframe") or {}).get("1d") or {})
    structure = structure_profile(context, "15m")
    long_trend = score_trend_component("long", context, structure)
    short_trend = score_trend_component("short", context, structure)
    long_momentum = score_momentum_component("long", context, structure)
    short_momentum = score_momentum_component("short", context, structure)
    long_positioning = score_positioning_component("long", context)
    short_positioning = score_positioning_component("short", context)
    volatility = score_volatility_efficiency(context)
    long_relative = score_relative_strength("long", context, btc_context)
    short_relative = score_relative_strength("short", context, btc_context)
    long_risk = score_risk_component("long", context, structure)
    short_risk = score_risk_component("short", context, structure)

    long_total = (
        long_trend["score"] * 0.30
        + long_momentum["score"] * 0.25
        + long_positioning["score"] * 0.20
        + volatility["score"] * 0.10
        + long_relative["score"] * 0.10
        - long_risk["score"] * 0.15
    )
    short_total = (
        short_trend["score"] * 0.30
        + short_momentum["score"] * 0.25
        + short_positioning["score"] * 0.20
        + volatility["score"] * 0.10
        + short_relative["score"] * 0.10
        - short_risk["score"] * 0.15
    )
    side = forced_side or ("long" if long_total >= short_total else "short")
    total_score = max(long_total, short_total) if forced_side is None else (long_total if forced_side == "long" else short_total)
    trend = long_trend if side == "long" else short_trend
    momentum = long_momentum if side == "long" else short_momentum
    positioning = long_positioning if side == "long" else short_positioning
    relative = long_relative if side == "long" else short_relative
    risk = long_risk if side == "long" else short_risk
    levels = derive_trade_levels(side, context, total_score)
    confidence = clamp(total_score, 12.0, 98.0)
    entry_quality = clamp((trend["score"] * 0.35 + momentum["score"] * 0.35 + volatility["score"] * 0.15 + relative["score"] * 0.15) / 10.0, 1.0, 10.0)
    market_dir = "bullish" if tf1d.get("trend_label") == "uptrend" else "bearish" if tf1d.get("trend_label") == "downtrend" else "neutral"
    breakdown = {
        "Setup": infer_setup(tf15, tf1h, tf4h, side, structure, risk["score"]),
        "Regime": market_dir,
        "TrendStrength": round(trend["score"], 2),
        "MomentumBurst": round(momentum["score"], 2),
        "Positioning": round(positioning["score"], 2),
        "VolatilityEfficiency": round(volatility["score"], 2),
        "RelativeStrength": round(relative["score"], 2),
        "RiskDeduction": round(risk["score"], 2),
        "RegimeConfidence": round(confidence, 1),
        "TrendConfidence": round(confidence, 1),
        "SignalQuality": round(entry_quality, 1),
        "RR": round(levels["rr_ratio"], 2),
        "VWAPDistanceATR": round((levels["price"] - safe_float(tf15.get("vwap"), levels["price"])) / max(levels["atr"], 1e-9), 4),
        "EMA20DistanceATR": round((levels["price"] - safe_float(tf15.get("ema20"), levels["price"])) / max(levels["atr"], 1e-9), 4),
        "SRDistanceATR": round(min(abs(levels["price"] - safe_float(tf15.get("recent_structure_high"), levels["price"])), abs(levels["price"] - safe_float(tf15.get("recent_structure_low"), levels["price"]))) / max(levels["atr"], 1e-9), 4),
    }
    notes = trend["notes"] + momentum["notes"] + positioning["notes"] + relative["notes"] + risk["notes"]
    signed_score = round(total_score if side == "long" else -total_score, 2)
    return {
        "symbol": symbol,
        "direction": side,
        "side": side,
        "score": signed_score,
        "raw_score": signed_score,
        "priority_score": round(total_score, 2),
        "direction_confidence": round(confidence, 1),
        "trend_confidence": round(confidence, 1),
        "entry_quality": round(entry_quality, 1),
        "signal_grade": "A" if total_score >= 72 else "B" if total_score >= 58 else "C",
        "setup_label": breakdown["Setup"],
        "price": levels["price"],
        "stop_loss": levels["stop_loss"],
        "take_profit": levels["take_profit"],
        "rr_ratio": levels["rr_ratio"],
        "margin_pct": round(clamp((fixed_order_notional_usdt(symbol) / max(_get_symbol_max_leverage(symbol), 1)) / max(safe_float(STATE.get("equity"), 100.0), 100.0), 0.01, 0.25), 4),
        "est_pnl": round(((levels["take_profit"] - levels["price"]) / max(levels["price"], 1e-9)) * 100 if side == "long" else ((levels["price"] - levels["take_profit"]) / max(levels["price"], 1e-9)) * 100, 2),
        "breakdown": breakdown,
        "desc": " | ".join(notes[:5]) or "Live scan signal",
        "trend_mode": "learning",
        "hold_reason": "normal_manage",
        "trend_note": "Local AI learning removed. Ranking uses trend, momentum, positioning, volatility, BTC relative strength, and risk deduction.",
        "candidate_source": candidate_source,
        "scanner_intent": "short_reversal_review" if candidate_source == "short_gainers" else "general_rank_review",
        "atr": levels["atr"],
        "atr15": levels["atr"],
        "reference_trade_plan": {
            "machine_entry_hint": levels["price"],
            "machine_stop_loss_hint": levels["stop_loss"],
            "machine_take_profit_hint": levels["take_profit"],
            "machine_rr_hint": levels["rr_ratio"],
            "machine_est_pnl_pct_hint": round(abs(safe_float(((levels["take_profit"] - levels["price"]) / max(levels["price"], 1e-9)) * 100 if side == "long" else ((levels["price"] - levels["take_profit"]) / max(levels["price"], 1e-9)) * 100, 0.0)), 4),
            "machine_stop_anchor_price": levels["stop_anchor_price"],
            "machine_stop_anchor_source": levels["stop_anchor_source"],
            "machine_stop_same_bar_exception": levels["stop_same_bar_exception"],
            "note": "Signal-specific execution hints derived from the selected side, live structure, ATR, and prior-swing anti-sweep stop logic.",
        },
        "openai_market_context": context,
        "market": market,
    }


def build_pressure_snapshot(tf_stats: Dict[str, Any], context: Dict[str, Any], timeframe: str) -> Dict[str, Any]:
    price = safe_float((context.get("basic_market_data") or {}).get("current_price"), safe_float(tf_stats.get("last_close"), 0.0))
    atr = max(safe_float(tf_stats.get("atr"), 0.0), price * 0.002 if price > 0 else 0.0)
    structure = structure_profile({"timeframe_bars": {timeframe: list((context.get("timeframe_bars") or {}).get(timeframe) or [])}}, timeframe)
    close_price = safe_float(tf_stats.get("last_close"), price)
    trend_stack = "bullish" if close_price >= safe_float(tf_stats.get("ema20"), close_price) >= safe_float(tf_stats.get("ema50"), close_price) else "bearish" if close_price <= safe_float(tf_stats.get("ema20"), close_price) <= safe_float(tf_stats.get("ema50"), close_price) else "mixed"
    swing_bias = "bullish" if structure.get("hh_count", 0) >= 3 and structure.get("hl_count", 0) >= 3 else "bearish" if structure.get("lh_count", 0) >= 3 and structure.get("ll_count", 0) >= 3 else "mixed"
    pressure_price = safe_float(tf_stats.get("recent_structure_high"), close_price)
    support_price = safe_float(tf_stats.get("recent_structure_low"), close_price)
    return {
        "structure_bias": swing_bias,
        "trend_stack": trend_stack,
        "swing_bias": swing_bias,
        "recent_break": "breakout" if structure.get("close_above_prior_high") else "breakdown" if structure.get("close_below_prior_low") else "inside",
        "pressure_price": pressure_price,
        "support_price": support_price,
        "pressure_distance_pct": round(max((pressure_price - close_price) / max(close_price, 1e-9) * 100.0, 0.0), 4),
        "support_distance_pct": round(max((close_price - support_price) / max(close_price, 1e-9) * 100.0, 0.0), 4),
        "pressure_distance_atr": round(max((pressure_price - close_price) / max(atr, 1e-9), 0.0), 4),
        "support_distance_atr": round(max((close_price - support_price) / max(atr, 1e-9), 0.0), 4),
        "close_vs_ema20_pct": round(pct_change(close_price, safe_float(tf_stats.get("ema20"), close_price)), 4),
        "close_vs_ema50_pct": round(pct_change(close_price, safe_float(tf_stats.get("ema50"), close_price)), 4),
        "volume_ratio": round(safe_float(tf_stats.get("vol_ratio"), 0.0), 4),
        "hh_count": structure.get("hh_count", 0),
        "hl_count": structure.get("hl_count", 0),
        "lh_count": structure.get("lh_count", 0),
        "ll_count": structure.get("ll_count", 0),
    }


def build_market_context(symbol: str, ticker: Dict[str, Any], market: Dict[str, Any]) -> Dict[str, Any]:
    multi_timeframe: Dict[str, Any] = {}
    timeframe_bars: Dict[str, Any] = {}
    pressure_map: Dict[str, Any] = {}
    raw_frames: Dict[str, pd.DataFrame] = {}
    errors: List[str] = []
    for tf in TIMEFRAMES:
        try:
            df = safe_fetch_ohlcv_df(symbol, tf, TIMEFRAME_BAR_LIMIT)
            raw_frames[tf] = df.copy()
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
    total_error_count = len(errors) + len(liquidity.get("errors", [])) + len(derivatives.get("errors", []))

    basic_market = {
        "symbol": compact_symbol(symbol),
        "exchange": "Bitget",
        "market_type": market_type_label(market),
        "current_price": round(safe_float(ticker.get("last"), 0.0), 6),
        "change_24h_pct": round(safe_float(ticker.get("percentage"), 0.0), 4),
        "quote_volume_24h": round(safe_float(ticker.get("quoteVolume"), ticker.get("baseVolume")), 4),
        "base_volume_24h": round(safe_float(ticker.get("baseVolume"), 0.0), 4),
        "market_cap_available": bool(supply.get("available", False)),
        "market_cap_usd": round(safe_float(supply.get("market_cap_usd"), 0.0), 2) if supply.get("market_cap_usd") is not None else None,
        "fdv_usd": round(safe_float(supply.get("fdv_usd"), 0.0), 2) if supply.get("fdv_usd") is not None else None,
        "circulating_supply": round(safe_float(supply.get("circulating_supply"), 0.0), 2) if supply.get("circulating_supply") is not None else None,
        "total_supply": round(safe_float(supply.get("total_supply"), 0.0), 2) if supply.get("total_supply") is not None else None,
        "funding_rate": derivatives.get("funding_rate"),
        "open_interest": derivatives.get("open_interest"),
        "open_interest_value_usdt": derivatives.get("open_interest_value_usdt"),
        "long_short_ratio": derivatives.get("long_short_ratio"),
        "top_trader_long_short_ratio": derivatives.get("top_trader_long_short_ratio"),
        "whale_position_change_pct": derivatives.get("whale_position_change_pct"),
    }
    tf15 = dict(multi_timeframe.get("15m") or {})
    tf1h = dict(multi_timeframe.get("1h") or {})
    tf4h = dict(multi_timeframe.get("4h") or {})
    bias = "long" if tf15.get("trend_label") == "uptrend" else "short" if tf15.get("trend_label") == "downtrend" else "neutral"
    aligned_label = "uptrend" if bias == "long" else "downtrend" if bias == "short" else ""
    aligned = [tf for tf, row in multi_timeframe.items() if aligned_label and row.get("trend_label") == aligned_label]
    opposing = [tf for tf, row in multi_timeframe.items() if aligned_label and row.get("trend_label") not in ("neutral", aligned_label)]
    nearest_blocking_timeframe = "1h" if bias == "long" else "15m"
    nearest_blocking_price = tf1h.get("recent_structure_high", 0.0) if bias == "long" else tf15.get("recent_structure_low", 0.0)
    nearest_backing_timeframe = "15m"
    nearest_backing_price = tf15.get("recent_structure_low", 0.0) if bias == "long" else tf15.get("recent_structure_high", 0.0)
    if bias == "neutral":
        nearest_blocking_timeframe = "15m"
        nearest_blocking_price = tf15.get("recent_structure_high", 0.0)
        nearest_backing_timeframe = "15m"
        nearest_backing_price = tf15.get("recent_structure_low", 0.0)
    pressure_summary = {
        "side": bias,
        "aligned_timeframes": aligned,
        "opposing_timeframes": opposing,
        "nearest_blocking_timeframe": nearest_blocking_timeframe,
        "nearest_blocking_price": nearest_blocking_price,
        "nearest_backing_timeframe": nearest_backing_timeframe,
        "nearest_backing_price": nearest_backing_price,
        "nearest_blocking_distance_atr": 0.0,
        "nearest_backing_distance_atr": 0.0,
        "stacked_blocking_within_1atr": len([tf for tf in opposing if tf in ("15m", "1h")]),
        "stacked_blocking_within_2atr": len(opposing),
    }
    for tf, stats in multi_timeframe.items():
        pressure_map[tf] = build_pressure_snapshot(stats, {"basic_market_data": basic_market, "timeframe_bars": timeframe_bars}, tf)
    reference_trade_plan = {
        "machine_entry_hint": basic_market["current_price"],
        "machine_stop_loss_hint": tf15.get("recent_structure_low", 0.0) if bias in ("long", "neutral") else tf15.get("recent_structure_high", 0.0),
        "machine_take_profit_hint": tf15.get("recent_structure_high", 0.0) if bias in ("long", "neutral") else tf15.get("recent_structure_low", 0.0),
        "machine_rr_hint": 2.0,
        "machine_est_pnl_pct_hint": abs(safe_float(tf15.get("ret_12bars_pct"), 0.0)),
        "note": "Context-only structure hints from live market context; candidate-side execution uses the signal-specific trade plan.",
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
        "news_context": {"available": False, "note": "News feed is currently unavailable in this deployment.", "items": []},
        "multi_timeframe": multi_timeframe,
        "timeframe_bars": timeframe_bars,
        "pre_breakout_radar": {"ready": len(aligned) >= 2, "phase": "watch", "direction": bias, "score": round(safe_float(tf15.get("adx"), 0.0), 2), "summary": "Multi-timeframe scan built from live candles, liquidity, and derivatives.", "note": "Live market structure monitor."},
        "execution_context": {"spread_pct": liquidity.get("spread_pct", 0.0), "top_depth_ratio": liquidity.get("depth_imbalance_10", 0.0), "api_error_streak": total_error_count, "status": "ok" if total_error_count == 0 else "degraded", "notes": errors + list(liquidity.get("errors", [])) + list(derivatives.get("errors", []))},
        "execution_policy": {
            "fixed_leverage": int(max(_get_symbol_max_leverage(symbol), 1)),
            "leverage_mode": "cross_max",
            "min_order_margin_usdt": round(fixed_order_notional_usdt(symbol) / max(_get_symbol_max_leverage(symbol), 1), 4),
            "fixed_order_notional_usdt": fixed_order_notional_usdt(symbol),
            "margin_pct_range": [0.01, 0.20],
        },
        "multi_timeframe_pressure_summary": pressure_summary,
        "multi_timeframe_pressure": pressure_map,
        "reference_trade_plan": reference_trade_plan,
        "reference_context": {"summary": "Live scan built from Bitget market, multi-timeframe candles, liquidity, and derivatives."},
    }


def flatten_openai_result(result: Dict[str, Any]) -> Dict[str, Any]:
    decision = dict(result.get("decision") or {})
    action = review_action(decision) if decision else ""
    return {
        "ai_enabled": True,
        "openai_enabled": True,
        "openai_status": str(result.get("status") or ""),
        "openai_status_label": str(result.get("status") or "").replace("_", " "),
        "openai_action": action,
        "openai_model": str(result.get("model") or ""),
        "openai_trend_state": str(decision.get("trend_state") or ""),
        "openai_timing_state": str(decision.get("timing_state") or ""),
        "openai_breakout_assessment": str(decision.get("breakout_assessment") or ""),
        "openai_trade_side": str(decision.get("trade_side") or ""),
        "openai_rr_ratio": safe_float(decision.get("rr_ratio"), 0.0),
        "openai_scale_in_recommended": bool(decision.get("scale_in_recommended", False)),
        "openai_scale_in_price": safe_float(decision.get("scale_in_price"), 0.0),
        "openai_scale_in_qty_pct": safe_float(decision.get("scale_in_qty_pct"), 0.0),
        "openai_scale_in_condition": str(decision.get("scale_in_condition") or ""),
        "openai_scale_in_note": str(decision.get("scale_in_note") or ""),
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
        dashboard = build_openai_trade_dashboard(
            OPENAI_TRADE_STATE,
            OPENAI_TRADE_CONFIG,
            api_key_present=bool(OPENAI_API_KEY),
        )
        dashboard["gate_debug"] = dict(AI_PANEL.get("openai_gate_debug") or {})
        AI_PANEL["openai_trade"] = dashboard
        save_trade_state(OPENAI_TRADE_STATE_PATH, OPENAI_TRADE_STATE)
    sync_openai_pending_advice()
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
        "same_direction_count": max(long_count, short_count),
        "open_symbols": [row.get("symbol") for row in active if row.get("symbol")],
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
        process_position_rules(positions_payload)
        set_backend_thread("positions", "running", "Synced {} active positions.".format(len(positions_payload)))
    except Exception as exc:
        set_backend_thread("positions", "crashed", "Position sync failed.", error=str(exc))


def _fetch_live_position_prices(symbols: List[str]) -> Dict[str, float]:
    clean_symbols = [str(sym or "") for sym in symbols if str(sym or "")]
    if not clean_symbols:
        return {}
    prices: Dict[str, float] = {}
    try:
        if len(clean_symbols) == 1:
            ticker = exchange.fetch_ticker(clean_symbols[0]) or {}
            price = safe_float(ticker.get("last"), 0.0)
            if price > 0:
                prices[clean_symbols[0]] = price
            return prices
        tickers = exchange.fetch_tickers(clean_symbols) or {}
        for symbol in clean_symbols:
            price = safe_float((dict(tickers.get(symbol) or {})).get("last"), 0.0)
            if price > 0:
                prices[symbol] = price
        if prices:
            return prices
    except Exception:
        prices = {}
    for symbol in clean_symbols:
        try:
            ticker = exchange.fetch_ticker(symbol) or {}
            price = safe_float(ticker.get("last"), 0.0)
            if price > 0:
                prices[symbol] = price
        except Exception:
            continue
    return prices


def sync_active_position_prices_once() -> bool:
    with STATE_LOCK:
        active_positions = [dict(row or {}) for row in list(STATE.get("active_positions") or []) if row.get("symbol")]
        equity = safe_float(STATE.get("equity"), 0.0)
    if not active_positions:
        return False
    live_prices = _fetch_live_position_prices([str(row.get("symbol") or "") for row in active_positions])
    if not live_prices:
        return False
    changed = False
    updated_positions: List[Dict[str, Any]] = []
    for row in active_positions:
        symbol = str(row.get("symbol") or "")
        entry = safe_float(row.get("entryPrice"), 0.0)
        leverage = safe_float(row.get("leverage"), 1.0)
        mark = safe_float(live_prices.get(symbol), safe_float(row.get("markPrice"), 0.0))
        if mark <= 0:
            updated_positions.append(row)
            continue
        raw_move_pct = 0.0
        if entry > 0:
            if str(row.get("side") or "").lower() == "long":
                raw_move_pct = ((mark - entry) / entry) * 100
            else:
                raw_move_pct = ((entry - mark) / entry) * 100
        next_row = dict(row)
        changed = changed or abs(mark - safe_float(row.get("markPrice"), 0.0)) > 1e-12
        next_row["markPrice"] = round(mark, 8)
        next_row["drawdown_pct"] = round(raw_move_pct, 4)
        next_row["leveraged_pnl_pct"] = round(raw_move_pct * leverage, 4)
        updated_positions.append(next_row)
    if not changed:
        return False
    total_pnl = round(sum(safe_float(row.get("unrealizedPnl"), 0.0) for row in updated_positions), 4)
    with STATE_LOCK:
        STATE["active_positions"] = updated_positions
        STATE["equity"] = round(equity, 4)
        STATE["total_pnl"] = total_pnl
        STATE["last_update"] = tw_now_str()
    process_position_rules(updated_positions)
    sync_runtime_views()
    set_backend_thread("positions", "running", "Tracking live prices for {} held symbols.".format(len(updated_positions)))
    return True


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


def diversified_selection(signals: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    ranked = sorted(signals, key=lambda row: safe_float(row.get("priority_score"), 0.0), reverse=True)
    selected: List[Dict[str, Any]] = []
    used_assets = set()
    used_setups = set()
    for row in ranked:
        if len(selected) >= limit:
            break
        asset = base_asset(row.get("symbol", ""))
        setup = "{}|{}|{}".format(row.get("candidate_source"), row.get("side"), row.get("setup_label"))
        same_side_count = sum(1 for item in selected if item.get("side") == row.get("side"))
        if asset in used_assets:
            continue
        if setup in used_setups and len(ranked) > limit:
            continue
        if same_side_count >= 3:
            continue
        selected.append(dict(row))
        used_assets.add(asset)
        used_setups.add(setup)
    if len(selected) < limit:
        existing = {row.get("symbol") for row in selected}
        for row in ranked:
            if len(selected) >= limit:
                break
            if row.get("symbol") in existing:
                continue
            selected.append(dict(row))
            existing.add(row.get("symbol"))
    return selected[:limit]


def build_scan_error_signal(symbol: str, ticker: Dict[str, Any], error: Exception, candidate_source: str = "general") -> Dict[str, Any]:
    return {
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
        "desc": "Scan failed: {}".format(str(error)[:120]),
        "trend_mode": "learning",
        "hold_reason": "normal_manage",
        "trend_note": "Scan failed, payload degraded.",
        "candidate_source": candidate_source,
        "scanner_intent": "scan_error",
        "openai_market_context": {},
        "market": {"symbol": symbol},
    }


def build_openai_constraints(symbol: str = "") -> Dict[str, Any]:
    max_symbol_leverage = max(_get_symbol_max_leverage(symbol), 1) if symbol else max(int(OPENAI_TRADE_CONFIG.get("max_leverage", 25) or 25), 1)
    order_notional_usdt = fixed_order_notional_usdt(symbol) if symbol else FIXED_ORDER_NOTIONAL_USDT
    return {
        "fixed_leverage": int(max_symbol_leverage),
        "min_leverage": int(OPENAI_TRADE_CONFIG.get("min_leverage", 4) or 4),
        "max_leverage": int(max_symbol_leverage),
        "min_margin_pct": 0.01,
        "max_margin_pct": 0.20,
        "fixed_order_notional_usdt": order_notional_usdt,
        "min_order_margin_usdt": round(order_notional_usdt / max_symbol_leverage, 4),
        "trade_style": "short_term_intraday",
        "max_open_positions": MAX_OPEN_POSITIONS,
        "max_same_direction": 3,
        "leverage_policy": "always_use_symbol_max",
    }


def open_position_symbols() -> set[str]:
    with STATE_LOCK:
        return {str(row.get("symbol") or "") for row in list(STATE.get("active_positions") or []) if row.get("symbol")}


def pending_order_symbols() -> set[str]:
    with STATE_LOCK:
        return set((STATE.get("fvg_orders") or {}).keys())


def watched_observe_symbols() -> set[str]:
    with REVIEW_LOCK:
        return {
            str(symbol or "")
            for symbol, row in REVIEW_TRACKER.items()
            if str((row or {}).get("status") or "") == "observe" and str(symbol or "")
        }


def review_action(decision: Dict[str, Any]) -> str:
    decision = dict(decision or {})
    if bool(decision.get("should_trade", False)):
        return "enter"
    if str(decision.get("watch_trigger_type") or "none").lower() != "none":
        return "observe"
    return "skip"


def update_watchlist_state() -> None:
    watchlist = []
    with REVIEW_LOCK:
        for symbol, row in REVIEW_TRACKER.items():
            if str(row.get("status") or "") != "observe":
                continue
            watchlist.append(
                {
                    "symbol": symbol,
                    "status": "observe",
                    "side": row.get("side"),
                    "watch_trigger_type": row.get("watch_trigger_type"),
                    "watch_trigger_price": row.get("watch_trigger_price"),
                    "watch_invalidation_price": row.get("watch_invalidation_price"),
                    "recheck_reason": row.get("recheck_reason"),
                    "candidate_source": row.get("candidate_source"),
                    "updated_at": row.get("updated_at"),
                }
            )
    update_state(watchlist=watchlist[:20])


def watch_condition_met(signal: Dict[str, Any], tracker: Dict[str, Any]) -> bool:
    return bool(evaluate_watch_condition(signal, tracker).get("ready"))


def _watch_timeframe_key(raw_timeframe: str) -> str:
    text = str(raw_timeframe or "").lower()
    for key in ("1m", "5m", "15m", "1h", "4h", "1d"):
        if key in text:
            return key
    return "15m"


def evaluate_watch_condition(signal: Dict[str, Any], tracker: Dict[str, Any]) -> Dict[str, Any]:
    decision = dict(tracker or {})
    if not decision:
        return {"ready": False, "stage": "missing_plan", "reason": "缺少觀察計畫"}
    trigger_type = str(decision.get("watch_trigger_type") or "none").lower()
    if trigger_type in ("none", "manual_review"):
        return {"ready": False, "stage": "manual_review_needed", "reason": "缺少可機器追蹤的主觸發條件"}
    price = safe_float(signal.get("price"), 0.0)
    trigger = safe_float(decision.get("watch_trigger_price"), 0.0)
    invalidation = safe_float(decision.get("watch_invalidation_price"), 0.0)
    side = str(decision.get("side") or signal.get("side") or "long").lower()
    tf_key = _watch_timeframe_key(decision.get("watch_timeframe"))
    context = dict(signal.get("openai_market_context") or {})
    multi = dict(context.get("multi_timeframe") or {})
    tf_stats = dict(multi.get(tf_key) or {})
    micro_stats = dict(multi.get("1m") or {})
    liquidity = dict((signal.get("openai_market_context") or {}).get("liquidity_context") or {})
    tf_close = safe_float(tf_stats.get("last_close"), price)
    trigger_candle = str(decision.get("watch_trigger_candle") or "none").lower()
    retest_rule = str(decision.get("watch_retest_rule") or "none").lower()
    micro_vwap_rule = str(decision.get("watch_micro_vwap_rule") or "none").lower()
    micro_ema20_rule = str(decision.get("watch_micro_ema20_rule") or "none").lower()
    volume_ratio_min = max(safe_float(decision.get("watch_volume_ratio_min"), 0.0), 0.0)
    anomaly = safe_float(liquidity.get("volume_anomaly_5m" if tf_key == "5m" else "volume_anomaly_15m"), 0.0)
    tf_vol_ratio = safe_float(tf_stats.get("vol_ratio"), 0.0)
    vol_proxy = max(tf_vol_ratio, anomaly)
    micro_close = safe_float(micro_stats.get("last_close"), price)
    micro_vwap = safe_float(micro_stats.get("vwap"), micro_close)
    micro_ema20 = safe_float(micro_stats.get("ema20"), micro_close)

    if side == "long":
        if invalidation > 0 and price < invalidation:
            return {"ready": False, "stage": "invalidated", "reason": "價格已跌破失效價"}
    else:
        if invalidation > 0 and price > invalidation:
            return {"ready": False, "stage": "invalidated", "reason": "價格已站回失效價上方"}

    base_trigger_ready = False
    if side == "long":
        if trigger_type == "pullback_to_entry":
            base_trigger_ready = trigger > 0 and price <= trigger * 1.004
        if trigger_type == "breakout_reclaim":
            base_trigger_ready = trigger > 0 and price >= trigger
        if trigger_type == "volume_confirmation":
            base_trigger_ready = trigger > 0 and price >= trigger
    else:
        if trigger_type == "pullback_to_entry":
            base_trigger_ready = trigger > 0 and price >= trigger * 0.996
        if trigger_type == "breakdown_confirm":
            base_trigger_ready = trigger > 0 and price <= trigger
        if trigger_type == "volume_confirmation":
            base_trigger_ready = trigger > 0 and price <= trigger
        if trigger_type == "breakout_reclaim":
            base_trigger_ready = trigger > 0 and price <= trigger
    if not base_trigger_ready:
        return {"ready": False, "stage": "waiting_price_trigger", "reason": "主觸發價尚未達成"}

    if trigger_candle == "close_above" and not (trigger > 0 and tf_close >= trigger):
        return {"ready": False, "stage": "waiting_candle_close", "reason": f"{tf_key} 收盤尚未站上主觸發價"}
    if trigger_candle == "close_below" and not (trigger > 0 and tf_close <= trigger):
        return {"ready": False, "stage": "waiting_candle_close", "reason": f"{tf_key} 收盤尚未跌破主觸發價"}

    if retest_rule == "hold_above" and not (trigger > 0 and price >= trigger and tf_close >= trigger):
        return {"ready": False, "stage": "waiting_retest", "reason": "回測後尚未確認站穩"}
    if retest_rule == "hold_below" and not (trigger > 0 and price <= trigger and tf_close <= trigger):
        return {"ready": False, "stage": "waiting_retest", "reason": "回測後尚未確認站不上"}
    if retest_rule == "fail_above" and not (trigger > 0 and price < trigger and tf_close < trigger):
        return {"ready": False, "stage": "waiting_retest", "reason": "回測失敗條件尚未成立"}
    if retest_rule == "fail_below" and not (trigger > 0 and price > trigger and tf_close > trigger):
        return {"ready": False, "stage": "waiting_retest", "reason": "回測失敗條件尚未成立"}

    if micro_vwap_rule == "above" and not (micro_close >= micro_vwap):
        return {"ready": False, "stage": "waiting_micro_structure", "reason": "1m 尚未站回 VWAP 上方"}
    if micro_vwap_rule == "below" and not (micro_close <= micro_vwap):
        return {"ready": False, "stage": "waiting_micro_structure", "reason": "1m 尚未落在 VWAP 下方"}
    if micro_ema20_rule == "above" and not (micro_close >= micro_ema20):
        return {"ready": False, "stage": "waiting_micro_structure", "reason": "1m 尚未站回 EMA20 上方"}
    if micro_ema20_rule == "below" and not (micro_close <= micro_ema20):
        return {"ready": False, "stage": "waiting_micro_structure", "reason": "1m 尚未落在 EMA20 下方"}

    if volume_ratio_min > 0 and vol_proxy < volume_ratio_min:
        return {"ready": False, "stage": "waiting_volume_confirmation", "reason": f"量能條件未達標，目前僅 {round(vol_proxy, 3)}"}

    return {"ready": True, "stage": "ready_to_reask_ai", "reason": "AI 指定的主觸發、結構與量能條件已成立"}


def candidate_review_gate(signal: Dict[str, Any]) -> Dict[str, Any]:
    symbol = str(signal.get("symbol") or "")
    if not symbol:
        return {"ok": False, "reason": "missing_symbol", "force_recheck": False}
    if symbol in open_position_symbols() or symbol in pending_order_symbols():
        return {"ok": False, "reason": "already_active", "force_recheck": False}
    with REVIEW_LOCK:
        tracker = dict(REVIEW_TRACKER.get(symbol) or {})
    if not tracker:
        return {"ok": True, "reason": "fresh", "force_recheck": False}
    status = str(tracker.get("status") or "")
    now = now_ts()
    if status == "skip" and now < safe_float(tracker.get("next_allowed_ts"), 0.0):
        return {"ok": False, "reason": "skip_cooldown", "force_recheck": False}
    if status == "observe":
        eval_result = evaluate_watch_condition(signal, tracker)
        with REVIEW_LOCK:
            if symbol in REVIEW_TRACKER:
                REVIEW_TRACKER[symbol]["current_price"] = safe_float(signal.get("price"), 0.0)
                REVIEW_TRACKER[symbol]["last_checked_ts"] = now
                REVIEW_TRACKER[symbol]["updated_at"] = tw_now_str()
                REVIEW_TRACKER[symbol]["tracking_status"] = str(eval_result.get("stage") or "bot_tracking_active")
                REVIEW_TRACKER[symbol]["tracking_reason"] = str(eval_result.get("reason") or "")
                REVIEW_TRACKER[symbol]["trigger_ready"] = bool(eval_result.get("ready", False))
        ready = bool(eval_result.get("ready", False))
        return {"ok": ready, "reason": "watch_triggered" if ready else "watch_waiting", "force_recheck": ready}
    return {"ok": True, "reason": "reopen", "force_recheck": False}


def apply_review_tracker(signal: Dict[str, Any], result: Dict[str, Any]) -> str:
    decision = dict(result.get("decision") or {})
    if not decision:
        return "none"
    action = review_action(decision)
    symbol = str(signal.get("symbol") or "")
    with REVIEW_LOCK:
        row = {
            "status": action,
            "side": signal.get("side"),
            "candidate_source": signal.get("candidate_source"),
            "updated_at": tw_now_str(),
            "tracking_status": "bot_tracking_active" if action == "observe" else "inactive",
            "tracking_reason": "",
            "trigger_ready": False,
            "current_price": safe_float(signal.get("price"), 0.0),
            "watch_trigger_type": str(decision.get("watch_trigger_type") or "none"),
            "watch_trigger_price": safe_float(decision.get("watch_trigger_price"), 0.0),
            "watch_invalidation_price": safe_float(decision.get("watch_invalidation_price"), 0.0),
            "recheck_reason": str(decision.get("recheck_reason") or ""),
            "decision": decision,
            "next_allowed_ts": 0.0,
        }
        if action == "skip":
            row["next_allowed_ts"] = now_ts() + AI_SKIP_COOLDOWN_SEC
        if action == "observe":
            row["created_ts"] = now_ts()
        REVIEW_TRACKER[symbol] = row
    update_watchlist_state()
    sync_openai_pending_advice()
    persist_runtime_snapshot()
    return action


def sync_openai_pending_advice() -> None:
    global OPENAI_TRADE_STATE
    pending: Dict[str, Any] = {}
    with REVIEW_LOCK:
        for symbol, row in REVIEW_TRACKER.items():
            if str(row.get("status") or "") != "observe":
                continue
            decision = dict(row.get("decision") or {})
            pending[symbol] = {
                "symbol": symbol,
                "side": row.get("side"),
                "status": "watching",
                "bot_tracking_enabled": True,
                "tracking_status": str(row.get("tracking_status") or "bot_tracking_active"),
                "tracking_reason": str(row.get("tracking_reason") or ""),
                "trigger_ready": bool(row.get("trigger_ready", False)),
                "candidate_source": row.get("candidate_source"),
                "created_ts": safe_float(row.get("created_ts"), now_ts()),
                "last_checked_ts": now_ts(),
                "last_checked_at": tw_now_str(),
                "current_price": safe_float(row.get("current_price"), 0.0),
                "watch_note": str(decision.get("watch_note") or ""),
                "entry_plan": str(decision.get("entry_plan") or ""),
                "watch_timeframe": str(decision.get("watch_timeframe") or ""),
                "watch_price_zone_low": safe_float(decision.get("watch_price_zone_low"), 0.0),
                "watch_price_zone_high": safe_float(decision.get("watch_price_zone_high"), 0.0),
                "watch_trigger_type": str(decision.get("watch_trigger_type") or "none"),
                "watch_trigger_price": safe_float(decision.get("watch_trigger_price"), 0.0),
                "watch_invalidation_price": safe_float(decision.get("watch_invalidation_price"), 0.0),
                "watch_structure_condition": str(decision.get("watch_structure_condition") or ""),
                "watch_volume_condition": str(decision.get("watch_volume_condition") or ""),
                "watch_trigger_candle": str(decision.get("watch_trigger_candle") or "none"),
                "watch_retest_rule": str(decision.get("watch_retest_rule") or "none"),
                "watch_volume_ratio_min": safe_float(decision.get("watch_volume_ratio_min"), 0.0),
                "watch_micro_vwap_rule": str(decision.get("watch_micro_vwap_rule") or "none"),
                "watch_micro_ema20_rule": str(decision.get("watch_micro_ema20_rule") or "none"),
                "watch_checklist": list(decision.get("watch_checklist") or []),
                "watch_confirmations": list(decision.get("watch_confirmations") or []),
                "watch_invalidations": list(decision.get("watch_invalidations") or []),
                "watch_recheck_priority": safe_float(decision.get("watch_recheck_priority"), 0.0),
                "recheck_reason": str(decision.get("recheck_reason") or ""),
            }
            trigger = safe_float(decision.get("watch_trigger_price"), 0.0)
            current = safe_float(row.get("current_price"), 0.0)
            pending[symbol]["distance_pct"] = round((((current / trigger) - 1.0) * 100.0), 4) if trigger > 0 and current > 0 else None
    with OPENAI_LOCK:
        OPENAI_TRADE_STATE = load_trade_state(OPENAI_TRADE_STATE_PATH) | dict(OPENAI_TRADE_STATE or {})
        OPENAI_TRADE_STATE["pending_advice"] = pending
        save_trade_state(OPENAI_TRADE_STATE_PATH, OPENAI_TRADE_STATE)
        dashboard = build_openai_trade_dashboard(
            OPENAI_TRADE_STATE,
            OPENAI_TRADE_CONFIG,
            api_key_present=bool(OPENAI_API_KEY),
        )
        dashboard["gate_debug"] = dict(AI_PANEL.get("openai_gate_debug") or {})
        AI_PANEL["openai_trade"] = dashboard


def risk_unit_from_rule(rule: Dict[str, Any]) -> float:
    return abs(
        safe_float(rule.get("initial_entry_price"), 0.0)
        - safe_float(rule.get("risk_reference_stop_loss"), safe_float(rule.get("initial_stop_loss"), 0.0))
    )


def build_trailing_snapshot(rule: Dict[str, Any]) -> Dict[str, Any]:
    partials = list(rule.get("partials") or [])
    done = [row for row in partials if row.get("done")]
    next_partial = next((row for row in partials if not row.get("done")), None)
    return {
        "active_stop_loss": safe_float(rule.get("active_stop_loss"), 0.0),
        "stop_stage": str(rule.get("stop_stage") or "initial"),
        "trail_pct": safe_float(rule.get("trail_pct"), 0.0),
        "trailing_active": bool(rule.get("trailing_active", False)),
        "trailing_stop": safe_float(rule.get("trailing_stop"), 0.0),
        "partials_done": len(done),
        "partials_total": len(partials),
        "partial_progress": "{}/{}".format(len(done), len(partials)),
        "next_partial_r": safe_float((next_partial or {}).get("r_multiple"), 0.0),
        "next_partial_fraction": safe_float((next_partial or {}).get("fraction"), 0.0),
        "scale_in_enabled": bool(rule.get("scale_in_enabled", False)),
        "scale_in_done": bool(rule.get("scale_in_done", False)),
        "scale_in_price": safe_float(rule.get("scale_in_price"), 0.0),
        "scale_in_note": str(rule.get("scale_in_note") or ""),
        "updated_at": rule.get("updated_at"),
    }


def refresh_trailing_state() -> None:
    with STATE_LOCK:
        STATE["trailing_info"] = {symbol: build_trailing_snapshot(rule) for symbol, rule in POSITION_RULES.items() if isinstance(rule, dict)}


def default_position_partials() -> List[Dict[str, Any]]:
    return [
        {"r_multiple": 1.0, "fraction": 0.25, "done": False},
        {"r_multiple": 1.5, "fraction": 0.25, "done": False},
        {"r_multiple": 2.0, "fraction": 0.30, "done": False},
    ]


def _copy_position_partials(partials: Any) -> List[Dict[str, Any]]:
    source = list(partials or [])
    if not source:
        source = default_position_partials()
    normalized: List[Dict[str, Any]] = []
    for row in source:
        item = dict(row or {})
        normalized.append(
            {
                "r_multiple": safe_float(item.get("r_multiple"), 0.0),
                "fraction": clamp(item.get("fraction"), 0.0, 1.0),
                "done": bool(item.get("done", False)),
            }
        )
    return normalized


def _position_side_to_order_side(side: str) -> str:
    return "buy" if str(side or "").lower() == "long" else "sell"


def _stop_improved(side: str, candidate_stop: float, reference_stop: float, min_delta: float) -> bool:
    if candidate_stop <= 0:
        return False
    if reference_stop <= 0:
        return True
    if str(side or "").lower() == "long":
        return candidate_stop > (reference_stop + min_delta)
    return candidate_stop < (reference_stop - min_delta)


def _build_position_rule_from_live_position(position: Dict[str, Any]) -> Dict[str, Any] | None:
    symbol = str(position.get("symbol") or "")
    side = str(position.get("side") or "").lower()
    entry = safe_float(position.get("entryPrice"), 0.0)
    mark_price = safe_float(position.get("markPrice"), entry)
    qty = safe_float(position.get("contracts"), 0.0)
    leverage = safe_float(position.get("leverage"), 1.0)
    if not symbol or side not in ("long", "short") or entry <= 0 or qty <= 0:
        return None
    with STATE_LOCK:
        protection = dict((STATE.get("protection_state") or {}).get(symbol) or {})
        history = list(STATE.get("trade_history") or [])
    stop_loss = safe_float(protection.get("sl"), 0.0)
    take_profit = safe_float(protection.get("tp"), 0.0)
    if stop_loss <= 0:
        return None
    if side == "long" and stop_loss >= entry:
        return None
    if side == "short" and stop_loss <= entry:
        return None
    risk_reference_stop = 0.0
    for row in reversed(history):
        if str(row.get("symbol") or "") != symbol:
            continue
        if str(row.get("side") or "").lower() != side:
            continue
        candidate_stop = safe_float(row.get("stop_loss"), 0.0)
        if candidate_stop <= 0:
            continue
        if side == "long" and candidate_stop >= entry:
            continue
        if side == "short" and candidate_stop <= entry:
            continue
        risk_reference_stop = candidate_stop
        break
    if risk_reference_stop <= 0:
        risk_reference_stop = stop_loss
    highest_price = max(entry, mark_price) if side == "long" else entry
    lowest_price = min(entry, mark_price) if side == "short" else entry
    return {
        "symbol": symbol,
        "side": side,
        "initial_entry_price": entry,
        "initial_stop_loss": stop_loss,
        "risk_reference_stop_loss": risk_reference_stop,
        "initial_take_profit": take_profit,
        "active_stop_loss": stop_loss,
        "stop_stage": "initial",
        "initial_qty": qty,
        "remaining_qty": qty,
        "leverage": leverage,
        "partials": default_position_partials(),
        "highest_price": highest_price,
        "lowest_price": lowest_price,
        "trailing_active": False,
        "trailing_stop": 0.0,
        "trail_pct": 0.0,
        "scale_in_enabled": False,
        "scale_in_price": 0.0,
        "scale_in_qty_pct": 0.0,
        "scale_in_done": False,
        "scale_in_condition": "",
        "scale_in_note": "",
        "recovered_from_position": True,
        "last_exchange_stop_sync_price": stop_loss,
        "last_exchange_stop_sync_stage": "initial",
        "last_exchange_stop_sync_qty": qty,
        "last_exchange_stop_sync_ts": time.time(),
        "last_exchange_stop_sync_ok": bool(protection.get("sl_ok", False)),
        "created_at": tw_now_str(),
        "updated_at": tw_now_str(),
    }


def _sync_rule_with_live_position(rule: Dict[str, Any], position: Dict[str, Any]) -> Dict[str, Any]:
    synced = dict(rule or {})
    actual_entry = safe_float(position.get("entryPrice"), 0.0)
    actual_mark = safe_float(position.get("markPrice"), actual_entry)
    actual_qty = safe_float(position.get("contracts"), 0.0)
    actual_leverage = safe_float(position.get("leverage"), safe_float(synced.get("leverage"), 1.0))
    previous_entry = safe_float(synced.get("initial_entry_price"), 0.0)
    if actual_entry > 0 and not math.isclose(previous_entry, actual_entry, rel_tol=1e-4, abs_tol=max(actual_entry * 1e-5, 1e-8)):
        synced["initial_entry_price"] = actual_entry
        if safe_float(synced.get("highest_price"), 0.0) <= 0:
            synced["highest_price"] = max(actual_entry, actual_mark)
        if safe_float(synced.get("lowest_price"), 0.0) <= 0:
            synced["lowest_price"] = min(actual_entry, actual_mark)
    if actual_qty > 0 and safe_float(synced.get("initial_qty"), 0.0) <= 0:
        synced["initial_qty"] = actual_qty
    if actual_leverage > 0:
        synced["leverage"] = actual_leverage
    if safe_float(synced.get("risk_reference_stop_loss"), 0.0) <= 0:
        synced["risk_reference_stop_loss"] = safe_float(synced.get("initial_stop_loss"), 0.0)
    synced["partials"] = _copy_position_partials(synced.get("partials"))
    if "remaining_qty" not in synced:
        synced["remaining_qty"] = actual_qty
    return synced


def _sync_managed_stop_to_exchange(symbol: str, side: str, qty: float, stop_loss: float, take_profit: float, stop_stage: str, rule: Dict[str, Any], risk: float) -> None:
    stage = str(stop_stage or "initial")
    qty = safe_float(qty, 0.0)
    stop_loss = safe_float(stop_loss, 0.0)
    take_profit = safe_float(take_profit, 0.0)
    if qty <= 0 or stop_loss <= 0 or stage == "initial":
        return
    last_stage = str(rule.get("last_exchange_stop_sync_stage") or "")
    last_stop = safe_float(rule.get("last_exchange_stop_sync_price"), 0.0)
    last_qty = safe_float(rule.get("last_exchange_stop_sync_qty"), 0.0)
    last_ts = safe_float(rule.get("last_exchange_stop_sync_ts"), 0.0)
    qty_delta = max(last_qty * 0.15, qty * 0.15, 1e-8)
    stop_delta = max(risk * (0.20 if stage == "trailing" else 0.05), abs(stop_loss) * 0.0005, 1e-8)
    stage_changed = stage != last_stage
    qty_changed = abs(qty - last_qty) > qty_delta
    stop_improved = _stop_improved(side, stop_loss, last_stop, stop_delta)
    if not stage_changed and not qty_changed and not stop_improved:
        return
    if stage == "trailing" and not stage_changed and (time.time() - last_ts) < 15:
        return
    result = ensure_exchange_protection(
        symbol,
        _position_side_to_order_side(side),
        qty,
        stop_loss,
        take_profit,
        manage_take_profit_locally=True,
    )
    rule["last_exchange_stop_sync_price"] = stop_loss
    rule["last_exchange_stop_sync_stage"] = stage
    rule["last_exchange_stop_sync_qty"] = qty
    rule["last_exchange_stop_sync_ts"] = time.time()
    rule["last_exchange_stop_sync_ok"] = bool(result.get("sl_ok", False))


def initialize_position_rule(signal: Dict[str, Any], qty: float, leverage: float, decision: Dict[str, Any]) -> None:
    symbol = str(signal.get("symbol") or "")
    if not symbol:
        return
    scale_in_price = safe_float(decision.get("scale_in_price"), 0.0)
    scale_in_qty_pct = clamp(decision.get("scale_in_qty_pct"), 0.0, 1.0)
    partials = default_position_partials()
    POSITION_RULES[symbol] = {
        "symbol": symbol,
        "side": str(signal.get("side") or ""),
        "initial_entry_price": safe_float(signal.get("price"), 0.0),
        "initial_stop_loss": safe_float(signal.get("stop_loss"), 0.0),
        "risk_reference_stop_loss": safe_float(signal.get("stop_loss"), 0.0),
        "initial_take_profit": safe_float(signal.get("take_profit"), 0.0),
        "active_stop_loss": safe_float(signal.get("stop_loss"), 0.0),
        "stop_stage": "initial",
        "initial_qty": safe_float(qty, 0.0),
        "remaining_qty": safe_float(qty, 0.0),
        "leverage": leverage,
        "partials": partials,
        "highest_price": safe_float(signal.get("price"), 0.0),
        "lowest_price": safe_float(signal.get("price"), 0.0),
        "trailing_active": False,
        "trailing_stop": 0.0,
        "trail_pct": 0.0,
        "scale_in_enabled": scale_in_price > 0 and scale_in_qty_pct > 0,
        "scale_in_price": scale_in_price,
        "scale_in_qty_pct": scale_in_qty_pct,
        "scale_in_done": False,
        "scale_in_condition": str(decision.get("scale_in_condition") or ""),
        "scale_in_note": str(decision.get("scale_in_note") or ""),
        "last_exchange_stop_sync_price": safe_float(signal.get("stop_loss"), 0.0),
        "last_exchange_stop_sync_stage": "initial",
        "last_exchange_stop_sync_qty": safe_float(qty, 0.0),
        "last_exchange_stop_sync_ts": time.time(),
        "last_exchange_stop_sync_ok": True,
        "created_at": tw_now_str(),
        "updated_at": tw_now_str(),
    }
    refresh_trailing_state()
    persist_runtime_snapshot()


def remove_position_rule(symbol: str) -> None:
    if symbol in POSITION_RULES:
        del POSITION_RULES[symbol]
        refresh_trailing_state()
        persist_runtime_snapshot()


def reduce_position(symbol: str, side: str, qty: float, reason: str) -> Dict[str, Any]:
    qty = safe_float(qty, 0.0)
    if qty <= 0:
        return {"ok": False, "error": "qty_zero"}
    try:
        qty = safe_float(exchange.amount_to_precision(symbol, qty), qty)
    except Exception:
        qty = safe_float(qty, 0.0)
    if qty <= 0:
        return {"ok": False, "error": "qty_precision_zero"}
    close_side = "sell" if str(side).lower() == "long" else "buy"
    params = {
        "reduceOnly": True,
        "tdMode": "cross",
        "marginMode": "cross",
        "posSide": "long" if str(side).lower() == "long" else "short",
    }
    try:
        order = exchange.create_order(symbol, "market", close_side, qty, None, params)
        append_trade_history(
            {
                "time": tw_now_str(),
                "symbol": symbol,
                "side": "partial_close_{}".format(side),
                "price": 0.0,
                "score": 0.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "pnl_pct": None,
                "contracts": round(qty, 8),
                "decision_source": reason,
            }
        )
        return {"ok": True, "order_id": order.get("id")}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:220]}


def scale_in_position(symbol: str, side: str, rule: Dict[str, Any]) -> Dict[str, Any]:
    add_qty = safe_float(rule.get("initial_qty"), 0.0) * clamp(rule.get("scale_in_qty_pct"), 0.0, 1.0)
    if add_qty <= 0:
        return {"ok": False, "error": "scale_qty_zero"}
    order_side = "buy" if str(side).lower() == "long" else "sell"
    params = {"tdMode": "cross", "marginMode": "cross", "posSide": "long" if str(side).lower() == "long" else "short"}
    try:
        order = exchange.create_order(symbol, "market", order_side, add_qty, None, params)
        append_trade_history(
            {
                "time": tw_now_str(),
                "symbol": symbol,
                "side": "scale_in_{}".format(side),
                "price": safe_float(rule.get("scale_in_price"), 0.0),
                "score": 0.0,
                "stop_loss": safe_float(rule.get("initial_stop_loss"), 0.0),
                "take_profit": safe_float(rule.get("initial_take_profit"), 0.0),
                "pnl_pct": None,
                "contracts": round(add_qty, 8),
                "decision_source": "ai_scale_in",
            }
        )
        return {"ok": True, "order_id": order.get("id"), "qty": add_qty}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:220]}


def process_position_rules(positions_payload: List[Dict[str, Any]]) -> None:
    active_map = {str(row.get("symbol") or ""): dict(row) for row in positions_payload if row.get("symbol")}
    stale_symbols = [sym for sym in list(POSITION_RULES.keys()) if sym not in active_map]
    for sym in stale_symbols:
        remove_position_rule(sym)
    for symbol, position in active_map.items():
        rule = POSITION_RULES.get(symbol)
        if not isinstance(rule, dict):
            bootstrapped = _build_position_rule_from_live_position(position)
            if not isinstance(bootstrapped, dict):
                continue
            POSITION_RULES[symbol] = bootstrapped
            rule = bootstrapped
        rule = _sync_rule_with_live_position(rule, position)
        side = str(rule.get("side") or position.get("side") or "").lower()
        mark_price = safe_float(position.get("markPrice"), 0.0)
        entry = safe_float(rule.get("initial_entry_price"), safe_float(position.get("entryPrice"), 0.0))
        stop = safe_float(rule.get("initial_stop_loss"), 0.0)
        take_profit = safe_float(rule.get("initial_take_profit"), 0.0)
        current_qty = safe_float(position.get("contracts"), 0.0)
        if mark_price <= 0 or entry <= 0 or stop <= 0 or current_qty <= 0:
            continue
        risk = max(risk_unit_from_rule(rule), 1e-9)
        if side == "long":
            current_r = (mark_price - entry) / risk
            rule["highest_price"] = max(safe_float(rule.get("highest_price"), entry), mark_price)
            best_r = (safe_float(rule.get("highest_price"), mark_price) - entry) / risk
        else:
            current_r = (entry - mark_price) / risk
            low_ref = safe_float(rule.get("lowest_price"), entry) or entry
            rule["lowest_price"] = min(low_ref, mark_price)
            best_r = (entry - safe_float(rule.get("lowest_price"), mark_price)) / risk
        rule["remaining_qty"] = current_qty
        managed_stop = stop
        stop_stage = "initial"
        if side == "long":
            if best_r >= 1.0:
                managed_stop = max(managed_stop, entry)
                stop_stage = "breakeven"
            if best_r >= 2.0:
                managed_stop = max(managed_stop, entry + risk)
                stop_stage = "lock_1r"
                rule["trailing_active"] = True
            if rule.get("trailing_active"):
                trail_stop = max(managed_stop, safe_float(rule.get("highest_price"), mark_price) - risk)
                managed_stop = max(managed_stop, trail_stop)
                rule["trailing_stop"] = trail_stop
                if trail_stop > (entry + risk + 1e-12):
                    stop_stage = "trailing"
        else:
            if best_r >= 1.0:
                managed_stop = min(managed_stop, entry)
                stop_stage = "breakeven"
            if best_r >= 2.0:
                managed_stop = min(managed_stop, entry - risk)
                stop_stage = "lock_1r"
                rule["trailing_active"] = True
            if rule.get("trailing_active"):
                trail_stop = min(managed_stop, safe_float(rule.get("lowest_price"), mark_price) + risk)
                managed_stop = min(managed_stop, trail_stop)
                rule["trailing_stop"] = trail_stop
                if trail_stop < (entry - risk - 1e-12):
                    stop_stage = "trailing"
        rule["active_stop_loss"] = managed_stop
        rule["stop_stage"] = stop_stage
        hard_stop_hit = (side == "long" and mark_price <= managed_stop) or (side == "short" and mark_price >= managed_stop)
        if hard_stop_hit:
            result = reduce_position(symbol, side, current_qty, "hard_stop_loss_{}".format(stop_stage))
            if result.get("ok"):
                remove_position_rule(symbol)
                continue
        hard_take_profit_hit = take_profit > 0 and (
            (side == "long" and mark_price >= take_profit)
            or (side == "short" and mark_price <= take_profit)
        )
        if hard_take_profit_hit:
            result = reduce_position(symbol, side, current_qty, "hard_take_profit")
            if result.get("ok"):
                remove_position_rule(symbol)
                continue
        for partial in list(rule.get("partials") or []):
            if partial.get("done"):
                continue
            trigger_r = safe_float(partial.get("r_multiple"), 0.0)
            if current_r < trigger_r:
                continue
            close_qty = min(current_qty, safe_float(rule.get("initial_qty"), current_qty) * safe_float(partial.get("fraction"), 0.0))
            result = reduce_position(symbol, side, close_qty, "tp_{}R".format(trigger_r))
            if result.get("ok"):
                partial["done"] = True
                current_qty = max(current_qty - close_qty, 0.0)
                rule["remaining_qty"] = current_qty
                if trigger_r >= 2.0:
                    rule["trailing_active"] = True
            break
        if rule.get("scale_in_enabled") and not rule.get("scale_in_done"):
            scale_trigger = safe_float(rule.get("scale_in_price"), 0.0)
            should_scale = False
            if scale_trigger > 0:
                if side == "long" and mark_price <= scale_trigger:
                    should_scale = True
                if side == "short" and mark_price >= scale_trigger:
                    should_scale = True
            if should_scale:
                result = scale_in_position(symbol, side, rule)
                if result.get("ok"):
                    rule["scale_in_done"] = True
                    current_qty = current_qty + safe_float(result.get("qty"), 0.0)
                    rule["remaining_qty"] = current_qty
        effective_stop = safe_float(rule.get("active_stop_loss"), stop)
        rule["trail_pct"] = abs((mark_price - effective_stop) / max(mark_price, 1e-9))
        _sync_managed_stop_to_exchange(symbol, side, current_qty, effective_stop, take_profit, stop_stage, rule, risk)
        rule["updated_at"] = tw_now_str()
        POSITION_RULES[symbol] = rule
    refresh_trailing_state()
    persist_runtime_snapshot()


def _apply_openai_trade_plan_to_signal(sig: Dict[str, Any], decision: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    plan = dict(decision or {})
    if not plan:
        return sig
    sig["openai_trade_plan"] = plan
    sig["openai_trade_meta"] = {
        "model": str((result or {}).get("model") or ""),
        "status": str((result or {}).get("status") or ""),
        "payload_hash": str((result or {}).get("payload_hash") or ""),
    }
    sig["decision_source"] = "openai"
    sig["planned_entry_price"] = float(plan.get("entry_price", sig.get("price", 0)) or sig.get("price", 0))
    sig["stop_loss"] = float(plan.get("stop_loss", sig.get("stop_loss", 0)) or sig.get("stop_loss", 0))
    sig["take_profit"] = float(plan.get("take_profit", sig.get("take_profit", 0)) or sig.get("take_profit", 0))
    sig["scale_in_price"] = float(plan.get("scale_in_price", 0) or 0)
    sig["scale_in_qty_pct"] = clamp(plan.get("scale_in_qty_pct"), 0.0, 1.0)
    sig["scale_in_condition"] = str(plan.get("scale_in_condition") or "")
    sig["scale_in_note"] = str(plan.get("scale_in_note") or "")
    if str(plan.get("order_type") or "").lower() != "limit" and sig["planned_entry_price"] > 0:
        sig["price"] = sig["planned_entry_price"]
    return sig


def _get_symbol_max_leverage(symbol: str) -> int:
    lev = 0
    try:
        market = exchange.market(symbol)
        info = dict(market.get("info") or {})
        for field in ("maxLeverage", "maxLev", "leverageMax"):
            value = info.get(field)
            if value in (None, ""):
                continue
            lev = max(lev, int(float(value)))
        lev = max(lev, int(float((market.get("limits") or {}).get("leverage", {}).get("max", 0) or 0)))
    except Exception:
        lev = 0
    if lev <= 1:
        lev = max(1, int(OPENAI_TRADE_CONFIG.get("max_leverage", 25) or 25))
    return lev


def _force_set_symbol_max_leverage(symbol: str, side: str) -> tuple[int, Dict[str, Any], str, bool]:
    pos_side = "long" if str(side or "").lower() in ("buy", "long") else "short"
    lev = _get_symbol_max_leverage(symbol)
    attempts = [
        {},
        {"tdMode": "cross", "holdSide": pos_side},
        {"marginMode": "cross", "holdSide": pos_side},
        {"tdMode": "cross", "posSide": pos_side},
        {"marginMode": "cross", "posSide": pos_side},
    ]
    errors: List[str] = []
    try:
        if hasattr(exchange, "set_margin_mode"):
            for params in attempts:
                try:
                    exchange.set_margin_mode("cross", symbol, params or None)
                except Exception:
                    continue
    except Exception:
        pass
    for params in attempts:
        try:
            if params:
                exchange.set_leverage(lev, symbol, params)
            else:
                exchange.set_leverage(lev, symbol)
            return lev, params, "", True
        except Exception as exc:
            errors.append(str(exc)[:180])
    return lev, {}, " | ".join(errors[:3]), False


def compute_order_size(symbol: str, entry_price: float, leverage: float) -> Dict[str, float]:
    entry_price = max(safe_float(entry_price, 0.0), 1e-9)
    leverage = max(safe_float(leverage, 1.0), 1.0)
    raw_qty = fixed_order_notional_usdt(symbol) / entry_price
    try:
        market = exchange.market(symbol)
        min_amt = safe_float(((market.get("limits") or {}).get("amount") or {}).get("min"), 0.0)
        if min_amt > 0:
            raw_qty = max(raw_qty, min_amt)
    except Exception:
        pass
    qty = safe_float(exchange.amount_to_precision(symbol, raw_qty), raw_qty)
    return {
        "qty": qty,
        "order_usdt": round(qty * entry_price, 4),
        "margin_usdt": round((qty * entry_price) / leverage, 4),
    }


def ensure_exchange_protection(symbol: str, side: str, qty: float, stop_loss: float, take_profit: float, *, manage_take_profit_locally: bool = True) -> Dict[str, bool]:
    close_side = "sell" if str(side).lower() == "buy" else "buy"
    pos_side = "long" if str(side).lower() == "buy" else "short"
    outcome = {"sl_ok": False, "tp_ok": False}
    sl_attempts = [
        {"reduceOnly": True, "stopPrice": str(stop_loss), "orderType": "stop", "posSide": pos_side, "tdMode": "cross", "marginMode": "cross"},
        {"reduceOnly": True, "stopLossPrice": str(stop_loss), "posSide": pos_side, "tdMode": "cross", "marginMode": "cross"},
        {"reduceOnly": True, "triggerPrice": str(stop_loss), "triggerType": "mark_price", "posSide": pos_side, "tdMode": "cross", "marginMode": "cross"},
    ]
    tp_attempts = [] if manage_take_profit_locally else [
        {"reduceOnly": True, "stopPrice": str(take_profit), "orderType": "takeProfit", "posSide": pos_side, "tdMode": "cross", "marginMode": "cross"},
        {"reduceOnly": True, "takeProfitPrice": str(take_profit), "triggerPrice": str(take_profit), "posSide": pos_side, "tdMode": "cross", "marginMode": "cross"},
        {"reduceOnly": True, "triggerPrice": str(take_profit), "triggerType": "mark_price", "orderType": "takeProfit", "posSide": pos_side, "tdMode": "cross", "marginMode": "cross"},
    ]
    for params in sl_attempts:
        try:
            exchange.create_order(symbol, "market", close_side, qty, None, params)
            outcome["sl_ok"] = True
            break
        except Exception:
            continue
    if manage_take_profit_locally:
        outcome["tp_ok"] = True
    else:
        for params in tp_attempts:
            try:
                exchange.create_order(symbol, "market", close_side, qty, None, params)
                outcome["tp_ok"] = True
                break
            except Exception:
                continue
    with STATE_LOCK:
        protection_state = dict(STATE.get("protection_state") or {})
        protection_state[symbol] = {
            "sl_ok": outcome["sl_ok"],
            "tp_ok": outcome["tp_ok"],
            "sl": round(stop_loss, 8),
            "tp": round(take_profit, 8),
            "tp_mode": "local_bot_managed" if manage_take_profit_locally else "exchange_take_profit",
            "updated_at": tw_now_str(),
        }
        STATE["protection_state"] = protection_state
    return outcome


def record_open_trade(signal: Dict[str, Any], qty: float, leverage: float, order_usdt: float, decision: Dict[str, Any]) -> None:
    append_trade_history(
        {
            "time": tw_now_str(),
            "symbol": signal.get("symbol"),
            "side": signal.get("side"),
            "price": safe_float(signal.get("price"), 0.0),
            "score": safe_float(signal.get("score"), 0.0),
            "stop_loss": safe_float(signal.get("stop_loss"), 0.0),
            "take_profit": safe_float(signal.get("take_profit"), 0.0),
            "pnl_pct": None,
            "contracts": round(qty, 8),
            "order_usdt": round(order_usdt, 4),
            "leverage": leverage,
            "decision_source": "openai",
            "openai_order_type": str(decision.get("order_type") or ""),
            "openai_action": review_action(decision),
        }
    )


def clear_pending_order(symbol: str) -> None:
    with STATE_LOCK:
        pending = dict(STATE.get("fvg_orders") or {})
        if symbol in pending:
            del pending[symbol]
            STATE["fvg_orders"] = pending


def register_pending_order(symbol: str, order: Dict[str, Any], signal: Dict[str, Any], leverage: int, size_info: Dict[str, float]) -> None:
    with STATE_LOCK:
        pending = dict(STATE.get("fvg_orders") or {})
        pending[symbol] = {
            "symbol": symbol,
            "order_id": str(order.get("id") or ""),
            "price": safe_float(signal.get("planned_entry_price", signal.get("price")), 0.0),
            "stop_loss": safe_float(signal.get("stop_loss"), 0.0),
            "take_profit": safe_float(signal.get("take_profit"), 0.0),
            "side": signal.get("side"),
            "leverage": leverage,
            "qty": size_info.get("qty", 0.0),
            "order_usdt": size_info.get("order_usdt", 0.0),
            "decision": dict(signal.get("openai_trade_plan") or {}),
            "updated_at": tw_now_str(),
        }
        STATE["fvg_orders"] = pending


def place_order_from_signal(signal: Dict[str, Any]) -> Dict[str, Any]:
    symbol = str(signal.get("symbol") or "")
    decision = dict(signal.get("openai_trade_plan") or {})
    if not symbol or not decision:
        return {"ok": False, "error": "missing_plan"}
    with ORDER_LOCK:
        if len(open_position_symbols()) >= MAX_OPEN_POSITIONS:
            return {"ok": False, "error": "max_positions"}
        if symbol in open_position_symbols() or symbol in pending_order_symbols():
            return {"ok": False, "error": "already_active"}
        order_side = "buy" if str(signal.get("side") or "").lower() == "long" else "sell"
        leverage, _, lev_error, lev_ok = _force_set_symbol_max_leverage(symbol, order_side)
        if not lev_ok:
            return {"ok": False, "error": lev_error or "leverage_failed"}
        planned_entry = safe_float(decision.get("entry_price"), safe_float(signal.get("price"), 0.0))
        signal["planned_entry_price"] = planned_entry
        signal["price"] = planned_entry if planned_entry > 0 else safe_float(signal.get("price"), 0.0)
        signal["stop_loss"] = safe_float(decision.get("stop_loss"), safe_float(signal.get("stop_loss"), 0.0))
        signal["take_profit"] = safe_float(decision.get("take_profit"), safe_float(signal.get("take_profit"), 0.0))
        size_info = compute_order_size(symbol, signal["price"], leverage)
        qty = safe_float(size_info.get("qty"), 0.0)
        if qty <= 0:
            return {"ok": False, "error": "qty_zero"}
        pos_side = "long" if order_side == "buy" else "short"
        params = {"tdMode": "cross", "marginMode": "cross", "posSide": pos_side}
        order_type = "limit" if str(decision.get("order_type") or "").lower() == "limit" else "market"
        try:
            if order_type == "limit":
                order = exchange.create_order(symbol, "limit", order_side, qty, signal["price"], params)
                register_pending_order(symbol, order, signal, leverage, size_info)
                return {"ok": True, "pending": True, "order_id": order.get("id")}
            order = exchange.create_order(symbol, "market", order_side, qty, None, params)
            fill_price = safe_float(order.get("average"), safe_float(order.get("price"), safe_float(signal.get("price"), 0.0)))
            if fill_price > 0:
                signal["price"] = fill_price
            ensure_exchange_protection(symbol, order_side, qty, signal["stop_loss"], signal["take_profit"], manage_take_profit_locally=True)
            record_open_trade(signal, qty, leverage, size_info.get("order_usdt", 0.0), decision)
            initialize_position_rule(signal, qty, leverage, decision)
            return {"ok": True, "pending": False, "order_id": order.get("id")}
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:220]}


def manage_pending_limit_orders() -> None:
    with STATE_LOCK:
        pending = dict(STATE.get("fvg_orders") or {})
    for symbol, row in pending.items():
        order_id = str(row.get("order_id") or "")
        if not order_id:
            clear_pending_order(symbol)
            continue
        try:
            order = exchange.fetch_order(order_id, symbol)
        except Exception:
            order = {}
        status = str(order.get("status") or "").lower()
        decision = dict(row.get("decision") or {})
        if status in ("closed", "filled"):
            fill_price = safe_float(order.get("average"), safe_float(order.get("price"), safe_float(row.get("price"), 0.0)))
            ensure_exchange_protection(
                symbol,
                "buy" if str(row.get("side") or "").lower() == "long" else "sell",
                safe_float(row.get("qty"), 0.0),
                safe_float(row.get("stop_loss"), 0.0),
                safe_float(row.get("take_profit"), 0.0),
                manage_take_profit_locally=True,
            )
            initialize_position_rule(
                {
                    "symbol": symbol,
                    "side": row.get("side"),
                    "price": fill_price if fill_price > 0 else safe_float(row.get("price"), 0.0),
                    "stop_loss": safe_float(row.get("stop_loss"), 0.0),
                    "take_profit": safe_float(row.get("take_profit"), 0.0),
                },
                safe_float(row.get("qty"), 0.0),
                safe_float(row.get("leverage"), 0.0),
                decision,
            )
            append_trade_history(
                {
                    "time": tw_now_str(),
                    "symbol": symbol,
                    "side": row.get("side"),
                    "price": safe_float(row.get("price"), 0.0),
                    "score": 0.0,
                    "stop_loss": safe_float(row.get("stop_loss"), 0.0),
                    "take_profit": safe_float(row.get("take_profit"), 0.0),
                    "pnl_pct": None,
                    "order_usdt": safe_float(row.get("order_usdt"), 0.0),
                    "decision_source": "openai_limit_fill",
                }
            )
            clear_pending_order(symbol)
            continue
        cancel_price = safe_float(decision.get("limit_cancel_price"), 0.0)
        current_price = 0.0
        try:
            current_price = safe_float((exchange.fetch_ticker(symbol) or {}).get("last"), 0.0)
        except Exception:
            current_price = safe_float(row.get("price"), 0.0)
        side = str(row.get("side") or "").lower()
        should_cancel = False
        if cancel_price > 0:
            if side == "long" and current_price < cancel_price:
                should_cancel = True
            if side == "short" and current_price > cancel_price:
                should_cancel = True
        if should_cancel:
            try:
                exchange.cancel_order(order_id, symbol)
            except Exception:
                pass
            clear_pending_order(symbol)


def build_scan_universe(markets: Dict[str, Any], tickers: Dict[str, Any]) -> tuple[List[tuple[str, Dict[str, Any], Dict[str, Any]]], List[tuple[str, Dict[str, Any], Dict[str, Any]]], List[tuple[str, Dict[str, Any], Dict[str, Any]]], List[tuple[str, Dict[str, Any], Dict[str, Any]]]]:
    tradeable = []
    for symbol, market in markets.items():
        if not isinstance(market, dict):
            continue
        if not is_usdt_symbol(market) or not bool(market.get("active", True)) or market_type_label(market) == "spot":
            continue
        ticker = dict(tickers.get(symbol) or {})
        if safe_float(ticker.get("quoteVolume"), 0.0) < MIN_SYMBOL_QUOTE_VOLUME:
            continue
        tradeable.append((symbol, market, ticker))
    by_volume = sorted(tradeable, key=lambda row: safe_float(row[2].get("quoteVolume"), 0.0), reverse=True)
    by_change = sorted(tradeable, key=lambda row: safe_float(row[2].get("percentage"), 0.0), reverse=True)
    volume_universe = by_volume[:SCAN_SYMBOL_LIMIT]
    short_gainer_universe = [row for row in by_change if safe_float(row[2].get("percentage"), 0.0) >= SHORT_GAINER_MIN_24H_PCT][: max(SHORT_GAINER_TOP_PICK * 2, 6)]
    watch_universe = []
    watched = watched_observe_symbols()
    for symbol in watched:
        market = dict(markets.get(symbol) or {})
        if not market:
            continue
        ticker = dict(tickers.get(symbol) or {})
        watch_universe.append((symbol, market, ticker))
    union: Dict[str, tuple[str, Dict[str, Any], Dict[str, Any]]] = {}
    for symbol, market, ticker in volume_universe + short_gainer_universe + watch_universe:
        union[symbol] = (symbol, market, ticker)
    if "BTC/USDT:USDT" in markets:
        union["BTC/USDT:USDT"] = ("BTC/USDT:USDT", dict(markets.get("BTC/USDT:USDT") or {}), dict(tickers.get("BTC/USDT:USDT") or {}))
    return list(union.values()), volume_universe, short_gainer_universe, watch_universe


def choose_review_candidates(general_top: List[Dict[str, Any]], short_gainers: List[Dict[str, Any]], pending_advice_signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    if len(open_position_symbols()) >= MAX_OPEN_POSITIONS:
        return candidates
    for pool in (pending_advice_signals, short_gainers, general_top):
        for row in pool:
            gate = candidate_review_gate(row)
            picked = dict(row)
            picked["force_openai_recheck"] = bool(gate.get("force_recheck", False))
            picked["local_gate_ok"] = bool(gate.get("ok", False))
            picked["local_gate_reason"] = str(gate.get("reason") or "")
            candidates.append(picked)
    return candidates


def maybe_run_openai(general_top: List[Dict[str, Any]], short_gainers: List[Dict[str, Any]], pending_advice_signals: List[Dict[str, Any]]) -> None:
    global OPENAI_TRADE_STATE
    reviewed_candidates = choose_review_candidates(general_top, short_gainers, pending_advice_signals)
    if not reviewed_candidates:
        AI_PANEL["openai_gate_debug"] = {"attempted": "", "status": "no_ranked_candidates", "blockers": [], "forced_first_send": False}
        refresh_openai_dashboard()
        return
    blockers: List[str] = []
    for reviewed in reviewed_candidates:
        if not reviewed.get("local_gate_ok"):
            blockers.append("{}:{}".format(compact_symbol(reviewed.get("symbol")), reviewed.get("local_gate_reason")))
            continue
        candidate = build_openai_trade_candidate(
            signal=reviewed,
            market=reviewed.get("market") or {},
            risk_status=STATE.get("risk_status") or {},
            portfolio=build_portfolio_snapshot(),
            top_candidates=(short_gainers if reviewed.get("candidate_source") == "short_gainers" else general_top)[:5],
            constraints=build_openai_constraints(str(reviewed.get("symbol") or "")),
            rank_index=max(safe_int(reviewed.get("rank"), 1) - 1, 0),
        )
        candidate["score"] = safe_float(reviewed.get("priority_score"), safe_float(reviewed.get("score"), 0.0))
        if int((OPENAI_TRADE_STATE or {}).get("api_calls", 0) or 0) == 0:
            candidate["force_recheck"] = True
        if reviewed.get("candidate_source") == "short_gainers":
            candidate["short_gainer_context"] = {
                "change_24h_pct": safe_float(((reviewed.get("openai_market_context") or {}).get("basic_market_data") or {}).get("change_24h_pct"), 0.0),
                "scanner_reason": "top_24h_gainer_reversal",
            }
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
                dashboard = build_openai_trade_dashboard(
                    OPENAI_TRADE_STATE,
                    OPENAI_TRADE_CONFIG,
                    api_key_present=bool(OPENAI_API_KEY),
                )
                dashboard["gate_debug"] = dict(AI_PANEL.get("openai_gate_debug") or {})
                AI_PANEL["openai_trade"] = dashboard
            status = str(result.get("status") or "")
            decision = dict(result.get("decision") or {})
            if status in ("below_min_score", "cooldown_active", "not_ranked", "global_interval_active", "cached_reuse"):
                blockers.append("{}:{}".format(compact_symbol(reviewed.get("symbol")), status))
                continue
            if decision:
                _apply_openai_trade_plan_to_signal(reviewed, decision, result)
                reviewed["auto_order"] = flatten_openai_result(result)
                action = apply_review_tracker(reviewed, result)
                if action == "enter":
                    place_order_from_signal(reviewed)
            for pool in (general_top, short_gainers):
                for idx, row in enumerate(pool):
                    if row.get("symbol") == reviewed.get("symbol"):
                        pool[idx] = reviewed
                        break
            set_backend_thread("openai", "running", "OpenAI status: {}".format(status or "unknown"))
            AI_PANEL["openai_gate_debug"] = {"attempted": compact_symbol(reviewed.get("symbol")), "status": status, "blockers": blockers[:10], "forced_first_send": bool(candidate.get("force_recheck", False))}
            refresh_openai_dashboard()
            return
        except Exception as exc:
            refresh_openai_dashboard()
            set_backend_thread("openai", "crashed", "OpenAI sync failed.", error=str(exc))
            return
    AI_PANEL["openai_gate_debug"] = {"attempted": "", "status": "no_candidate_sent", "blockers": blockers[:10], "forced_first_send": False}
    set_backend_thread("openai", "running", "No eligible OpenAI candidate this cycle. {}".format(" | ".join(blockers[:3]) if blockers else ""))
    refresh_openai_dashboard()


def perform_scan_cycle() -> None:
    set_backend_thread("scan", "running", "Fetching markets and scan universe.")
    update_state(scan_progress="Loading Bitget tickers...")
    try:
        markets = exchange.load_markets()
        tickers = exchange.fetch_tickers()
        selected_union, volume_universe, short_gainer_universe, watch_universe = build_scan_universe(markets, tickers)
        context_map: Dict[str, Dict[str, Any]] = {}
        for idx, (symbol, market, ticker) in enumerate(selected_union, start=1):
            update_state(scan_progress="Scanning {}/{} {}".format(idx, len(selected_union), compact_symbol(symbol)))
            try:
                context_map[symbol] = build_market_context(symbol, ticker, market)
            except Exception as exc:
                context_map[symbol] = {"basic_market_data": {"current_price": safe_float(ticker.get("last"), 0.0)}, "multi_timeframe": {}, "timeframe_bars": {}}
                update_state(scan_progress="Context degraded for {}: {}".format(compact_symbol(symbol), str(exc)[:90]))
        btc_context = context_map.get("BTC/USDT:USDT")
        general_signals: List[Dict[str, Any]] = []
        for symbol, market, ticker in volume_universe:
            try:
                general_signals.append(build_signal_from_context(symbol, {"symbol": symbol, "pattern": "general_scan"}, context_map[symbol], btc_context, candidate_source="general"))
            except Exception as exc:
                general_signals.append(build_scan_error_signal(symbol, ticker, exc, "general"))
        short_gainer_signals: List[Dict[str, Any]] = []
        for symbol, market, ticker in short_gainer_universe:
            try:
                context = context_map[symbol]
                reversal = detect_short_reversal_signal(context)
                if not reversal.get("ready"):
                    continue
                signal = build_signal_from_context(symbol, {"symbol": symbol, "pattern": "short_gainers"}, context, btc_context, candidate_source="short_gainers", forced_side="short")
                signal["desc"] = " | ".join(reversal.get("triggers") or []) or signal.get("desc", "")
                signal["short_gainer_reversal"] = reversal
                short_gainer_signals.append(signal)
            except Exception:
                continue
        pending_advice_signals: List[Dict[str, Any]] = []
        for symbol, market, ticker in watch_universe:
            try:
                tracker = dict(REVIEW_TRACKER.get(symbol) or {})
                if str(tracker.get("status") or "") != "observe":
                    continue
                signal = build_signal_from_context(
                    symbol,
                    {"symbol": symbol, "pattern": "pending_advice"},
                    context_map[symbol],
                    btc_context,
                    candidate_source="pending_advice",
                    forced_side=str(tracker.get("side") or "").lower() or None,
                )
                signal["rank"] = 1
                signal["desc"] = str(((tracker.get("decision") or {}).get("watch_note")) or signal.get("desc", ""))
                pending_advice_signals.append(signal)
            except Exception as exc:
                update_state(scan_progress="Pending advice tracking degraded for {}: {}".format(compact_symbol(symbol), str(exc)[:90]))
        general_top = diversified_selection(general_signals, GENERAL_TOP_PICK)
        short_gainer_top = diversified_selection(
            sorted(
                short_gainer_signals,
                key=lambda row: safe_float(((row.get("openai_market_context") or {}).get("basic_market_data") or {}).get("change_24h_pct"), 0.0),
                reverse=True,
            ),
            SHORT_GAINER_TOP_PICK,
        )
        for idx, row in enumerate(general_top, start=1):
            row["rank"] = idx
        for idx, row in enumerate(short_gainer_top, start=1):
            row["rank"] = idx
        manage_pending_limit_orders()
        maybe_run_openai(general_top, short_gainer_top, pending_advice_signals)
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
            for row in general_top[:5]
        ]
        AI_PANEL["market_db_info"] = {
            "symbols": [row.get("symbol") for row in general_top],
            "short_gainers": [row.get("symbol") for row in short_gainer_top],
            "timeframes": list(TIMEFRAMES),
            "last_update": tw_now_str(),
        }
        AUTO_BACKTEST_STATE["scanned_markets"] = len(general_top) + len(short_gainer_top)
        AUTO_BACKTEST_STATE["target_count"] = len(selected_union)
        AUTO_BACKTEST_STATE["db_symbols"] = [row.get("symbol") for row in general_top] + [row.get("symbol") for row in short_gainer_top]
        AUTO_BACKTEST_STATE["db_last_update"] = tw_now_str()
        update_state(
            top_signals=general_top,
            general_top_signals=general_top,
            short_gainer_signals=short_gainer_top,
            pending_advice_signals=pending_advice_signals[:10],
            scan_progress="Scan complete. {} general / {} short-gainer symbols ready.".format(len(general_top), len(short_gainer_top)),
        )
        update_market_overview(general_top or short_gainer_top)
        refresh_learning_summary()
        update_watchlist_state()
        set_backend_thread("scan", "running", "Updated {} ranked symbols and {} short-gainer reversal candidates.".format(len(general_top), len(short_gainer_top)))
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
    last_full_sync_ts = 0.0
    while True:
        active_count = len(open_position_symbols())
        now = time.time()
        full_sync_interval = POSITION_INTERVAL_SEC if active_count == 0 else MAX_ACTIVE_POSITION_SCAN_INTERVAL_SEC
        should_full_sync = last_full_sync_ts <= 0 or (now - last_full_sync_ts) >= full_sync_interval
        if should_full_sync:
            sync_positions_once()
            last_full_sync_ts = time.time()
            time.sleep(max(ACTIVE_POSITION_PRICE_POLL_SEC if len(open_position_symbols()) > 0 else POSITION_INTERVAL_SEC, 1))
            continue
        sync_active_position_prices_once()
        time.sleep(max(ACTIVE_POSITION_PRICE_POLL_SEC, 1))


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
        row = dict(fvg_orders.get(symbol) or {})
    if row.get("order_id"):
        try:
            exchange.cancel_order(str(row.get("order_id")), symbol)
        except Exception:
            pass
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
    params: Dict[str, Any] = {"reduceOnly": True, "tdMode": "cross", "marginMode": "cross"}
    if side in ("long", "short"):
        params["posSide"] = side
    try:
        exchange.create_order(symbol, "market", close_side, contracts, None, params)
        remove_position_rule(symbol)
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
