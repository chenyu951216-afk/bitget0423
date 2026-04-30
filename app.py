from __future__ import annotations

import math
import os
import threading
import time
import traceback
import json
import hmac
import base64
import hashlib
from datetime import datetime
from typing import Any, Dict, List
from urllib.parse import urlencode

import ccxt
import pandas as pd
import pandas_ta as ta
import requests
from flask import Flask, jsonify, render_template, request as flask_request

from api_state_routes import (
    build_ai_panel_payload,
    build_positions_payload,
    build_state_lite_payload,
    compact_ai_panel_payload,
    compact_state_lite_payload,
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
from pre_breakout_candidate_scanner import PreBreakoutCandidateScanner
from state_service import DEFAULT_RUNTIME_STATE, env_or_blank


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OPENAI_TRADE_STATE_PATH = os.path.join(DATA_DIR, "openai_trade_state.json")
SNAPSHOT_STATE_PATH = os.path.join(DATA_DIR, "runtime_snapshot.json")

SCAN_INTERVAL_SEC = max(30, int(float(env_or_blank("SCAN_INTERVAL_SEC", "30") or 30)))
POSITION_INTERVAL_SEC = max(15, int(float(env_or_blank("POSITION_INTERVAL_SEC", "25") or 25)))
MAX_ACTIVE_POSITION_SCAN_INTERVAL_SEC = max(5, int(float(env_or_blank("MAX_ACTIVE_POSITION_SCAN_INTERVAL_SEC", "12") or 12)))
ACTIVE_POSITION_PRICE_POLL_SEC = max(1, int(float(env_or_blank("ACTIVE_POSITION_PRICE_POLL_SEC", "1") or 1)))
OPENAI_SYNC_INTERVAL_SEC = max(20, int(float(env_or_blank("OPENAI_SYNC_INTERVAL_SEC", "30") or 30)))
SCAN_SYMBOL_LIMIT = max(8, min(120, int(float(env_or_blank("SCAN_SYMBOL_LIMIT", "50") or 50))))
TOP_SIGNAL_LIMIT = max(5, min(15, int(float(env_or_blank("TOP_SIGNAL_LIMIT", "10") or 10))))
TIMEFRAME_BAR_LIMIT = max(100, min(240, int(float(env_or_blank("SCAN_BAR_LIMIT", "120") or 120))))
ACTIVE_HISTORY_LIMIT = 40
GENERAL_TOP_PICK = max(5, min(SCAN_SYMBOL_LIMIT, int(float(env_or_blank("GENERAL_TOP_PICK", str(SCAN_SYMBOL_LIMIT)) or SCAN_SYMBOL_LIMIT))))
SHORT_GAINER_TOP_PICK = 3
PREBREAKOUT_TOP_PICK = max(1, min(5, int(float(env_or_blank("PREBREAKOUT_TOP_PICK", "3") or 3))))
PREBREAKOUT_SYMBOL_LIMIT = max(PREBREAKOUT_TOP_PICK * 2, min(60, int(float(env_or_blank("PREBREAKOUT_SYMBOL_LIMIT", "24") or 24))))
SHORT_GAINER_MIN_24H_PCT = max(5.0, float(env_or_blank("SHORT_GAINER_MIN_24H_PCT", "15") or 15))
MAX_OPEN_POSITIONS = 5
MAX_SAME_DIRECTION_POSITIONS = max(1, min(MAX_OPEN_POSITIONS, int(float(env_or_blank("MAX_SAME_DIRECTION_POSITIONS", "4") or 4))))
PREBREAKOUT_MAX_OPEN_POSITIONS = max(1, min(3, int(float(env_or_blank("PREBREAKOUT_MAX_OPEN_POSITIONS", "1") or 1))))
PREBREAKOUT_PRICE_CHANGE_PCT = max(0.1, float(env_or_blank("PREBREAKOUT_RECHECK_PRICE_CHANGE_PCT", "0.35") or 0.35))
PREBREAKOUT_CVD_CHANGE_USDT = max(1_000.0, float(env_or_blank("PREBREAKOUT_RECHECK_CVD_CHANGE_USDT", "50000") or 50000))
PREBREAKOUT_DEPTH_IMBALANCE_CHANGE = max(0.02, float(env_or_blank("PREBREAKOUT_RECHECK_DEPTH_IMBALANCE_CHANGE", "0.08") or 0.08))
PREBREAKOUT_TRIGGER_DIST_CHANGE_ATR = max(0.05, float(env_or_blank("PREBREAKOUT_RECHECK_TRIGGER_CHANGE_ATR", "0.22") or 0.22))
ENABLE_ADVANCED_POSITION_MANAGEMENT = str(env_or_blank("ENABLE_ADVANCED_POSITION_MANAGEMENT", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}
AI_SKIP_COOLDOWN_SEC = 60 * 60
AI_OBSERVE_RECHECK_COOLDOWN_SEC = max(90, int(float(env_or_blank("AI_OBSERVE_RECHECK_COOLDOWN_SEC", "300") or 300)))
OBSERVE_MANUAL_WINDOW_SEC = min(3600, max(300, int(float(env_or_blank("OBSERVE_MANUAL_WINDOW_SEC", "3600") or 3600))))
ORDER_BLOCK_EXPIRY_SEC = min(3600, max(300, int(float(env_or_blank("ORDER_BLOCK_EXPIRY_SEC", "3600") or 3600))))
FIXED_ORDER_NOTIONAL_USDT = max(5.0, float(env_or_blank("FIXED_ORDER_NOTIONAL_USDT", "100") or 100))
BTC_ETH_MIN_ORDER_NOTIONAL_USDT = max(FIXED_ORDER_NOTIONAL_USDT, float(env_or_blank("BTC_ETH_MIN_ORDER_NOTIONAL_USDT", "150") or 150))
HIGH_PRICE_ORDER_THRESHOLD_USDT = max(1.0, float(env_or_blank("HIGH_PRICE_ORDER_THRESHOLD_USDT", "30") or 30))
HIGH_PRICE_ORDER_NOTIONAL_USDT = max(BTC_ETH_MIN_ORDER_NOTIONAL_USDT, float(env_or_blank("HIGH_PRICE_ORDER_NOTIONAL_USDT", "450") or 450))
HIGH_NOTIONAL_SYMBOLS = {"BTC", "ETH", "XAU", "XAG", "SOL"}
SPECIAL_NOTIONAL_SYMBOLS = {"XRP", "DOGE"}
SPECIAL_NOTIONAL_USDT = max(FIXED_ORDER_NOTIONAL_USDT, float(env_or_blank("SPECIAL_NOTIONAL_USDT", "150") or 150))
BTC_BETA_SYMBOLS = {
    "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT", "BNB", "LTC",
    "BCH", "APT", "ARB", "OP", "SUI", "NEAR", "INJ", "FIL", "TRX", "TON",
}
MIN_SYMBOL_QUOTE_VOLUME = max(100_000.0, float(env_or_blank("SCAN_MIN_QUOTE_VOLUME", "300000") or 300000))
EXCHANGE_PROTECTION_MODE = str(env_or_blank("EXCHANGE_PROTECTION_MODE", "hybrid") or "hybrid").strip().lower()
ENABLE_EXCHANGE_TRIGGER_PROTECTION = EXCHANGE_PROTECTION_MODE in {"exchange", "hybrid"}
STATE_SNAPSHOT_MIN_INTERVAL_SEC = max(0.5, float(env_or_blank("STATE_SNAPSHOT_MIN_INTERVAL_SEC", "2.0") or 2.0))
EXCHANGE_SNAPSHOT_CACHE_TTL_SEC = max(0.0, float(env_or_blank("EXCHANGE_SNAPSHOT_CACHE_TTL_SEC", "0") or 0))
OPENAI_SOURCE_ROTATION = ["short_gainers", "general", "prebreakout_scanner"]
SCAN_SYMBOLS_PER_CYCLE = max(2, int(float(env_or_blank("SCAN_SYMBOLS_PER_CYCLE", "2") or 2)))
BITGET_PRODUCT_TYPE = str(env_or_blank("BITGET_PRODUCT_TYPE", "USDT-FUTURES") or "USDT-FUTURES").strip().upper()
BITGET_COPY_PRODUCT_TYPE = BITGET_PRODUCT_TYPE
BITGET_MARGIN_COIN = str(env_or_blank("BITGET_MARGIN_COIN", "USDT") or "USDT").strip().upper()
BITGET_MARGIN_MODE = str(env_or_blank("BITGET_MARGIN_MODE", "crossed") or "crossed").strip().lower()
if BITGET_MARGIN_MODE == "cross":
    BITGET_MARGIN_MODE = "crossed"
if BITGET_MARGIN_MODE not in {"crossed", "isolated"}:
    BITGET_MARGIN_MODE = "crossed"
BITGET_API_MODE = str(env_or_blank("BITGET_API_MODE", "copy_trader") or "copy_trader").strip().lower()
BITGET_USE_COPY_TRADER_API = BITGET_API_MODE in {"copy", "copy_trade", "copy_trader", "trader_copy"}
BASE_URL = str(env_or_blank("BITGET_API_BASE_URL", "https://api.bitget.com") or "https://api.bitget.com").strip().rstrip("/")
BITGET_COPY_PERMISSION_FALLBACK_TO_SWAP = str(env_or_blank("BITGET_COPY_PERMISSION_FALLBACK_TO_SWAP", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
BITGET_COPY_STRICT_ONLY = str(env_or_blank("BITGET_COPY_STRICT_ONLY", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}
BITGET_ACCOUNT_NAME_HINT = str(env_or_blank("BITGET_ACCOUNT_NAME_HINT", "") or "").strip()
COPY_PERMISSION_COOLDOWN_SEC = max(60, int(float(env_or_blank("COPY_PERMISSION_COOLDOWN_SEC", "600") or 600)))
COPY_SYMBOLS_CACHE_TTL_SEC = max(15, int(float(env_or_blank("COPY_SYMBOLS_CACHE_TTL_SEC", "120") or 120)))
try:
    COPY_TRADE_ORDER_MIN_INTERVAL_SEC = max(
        1.2,
        float(env_or_blank("COPY_TRADE_ORDER_MIN_INTERVAL_SEC", "1.2") or 1.2),
    )
except Exception:
    COPY_TRADE_ORDER_MIN_INTERVAL_SEC = 1.2

BITGET_ENDPOINTS: Dict[str, str] = {
    # Bitget official behavior for futures copy-trader keys:
    # open position -> mix place-order, close position -> copy trader close-tracking.
    "copy_trader_open_order": "/api/v2/mix/order/place-order",
    "mix_place_order": "/api/v2/mix/order/place-order",
    "mix_order_detail": "/api/v2/mix/order/detail",
    "mix_cancel_order": "/api/v2/mix/order/cancel-order",
    "copy_current_track": "/api/v2/copy/mix-trader/order-current-track",
    "copy_close_positions": "/api/v2/copy/mix-trader/order-close-positions",
    "copy_modify_tpsl": "/api/v2/copy/mix-trader/order-modify-tpsl",
    "copy_symbol_settings": "/api/v2/copy/mix-trader/config-query-symbols",
    "copy_total_detail": "/api/v2/copy/mix-trader/order-total-detail",
    "spot_account_info": "/api/v2/spot/account/info",
}

TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
DEFAULT_PARAMS = {
    "sl_mult": 1.8,
    "tp_mult": 2.6,
    "breakeven_atr": 1.0,
    "trail_trigger_atr": 1.4,
    "trail_pct": 0.45,
}
LOCAL_TP_MIN_R = max(0.8, float(env_or_blank("LOCAL_TP_MIN_R", "1.4") or 1.4))
LOCAL_TP_DEFAULT_R = max(LOCAL_TP_MIN_R, float(env_or_blank("LOCAL_TP_DEFAULT_R", "1.8") or 1.8))
LOCAL_TP_MAX_R = max(LOCAL_TP_DEFAULT_R, float(env_or_blank("LOCAL_TP_MAX_R", "3.5") or 3.5))
STRUCTURE_SWING_LOOKBACK_15M = max(12, min(48, int(float(env_or_blank("STRUCTURE_SWING_LOOKBACK_15M", "24") or 24))))
OPENAI_MIN_RR_FOR_ENTRY = max(0.8, min(3.0, float(env_or_blank("OPENAI_MIN_RR_FOR_ENTRY", "1.35") or 1.35)))
try:
    OPENAI_EXECUTION_MIN_SCORE = max(
        0.0,
        float(
            env_or_blank(
                "OPENAI_EXECUTION_MIN_SCORE",
                env_or_blank("OPENAI_STRICT_MIN_SCORE", env_or_blank("OPENAI_TRADE_MIN_SCORE", "48")),
            ) or "48"
        ),
    )
except Exception:
    OPENAI_EXECUTION_MIN_SCORE = 48.0

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
PREBREAKOUT_SCANNER = PreBreakoutCandidateScanner(exchange)

STATE_LOCK = threading.RLock()
OPENAI_LOCK = threading.RLock()
WORKER_LOCK = threading.RLock()
ORDER_LOCK = threading.RLock()
REVIEW_LOCK = threading.RLock()

WORKERS_STARTED = False
OPENAI_API_KEY = env_or_blank("OPENAI_API_KEY")
OPENAI_TRADE_CONFIG = default_trade_config(lambda name, default="": env_or_blank(name, default))
if str(OPENAI_TRADE_CONFIG.get("model") or "").strip().lower().startswith("gpt-4.1-mini"):
    OPENAI_TRADE_CONFIG["reasoning_effort"] = "medium"
    OPENAI_TRADE_CONFIG["retry_reasoning_effort"] = "medium"
OPENAI_TRADE_CONFIG["cooldown_minutes"] = 20
OPENAI_TRADE_CONFIG["same_payload_reuse_minutes"] = 20
OPENAI_TRADE_CONFIG["global_min_interval_minutes"] = 0
OPENAI_TRADE_CONFIG["top_k_per_scan"] = GENERAL_TOP_PICK
OPENAI_TRADE_CONFIG["sends_per_scan"] = 2
try:
    _strict_min_score = float(
        env_or_blank(
            "OPENAI_STRICT_MIN_SCORE",
            str(OPENAI_TRADE_CONFIG.get("min_score_abs", 48.0)),
        ) or OPENAI_TRADE_CONFIG.get("min_score_abs", 48.0)
    )
except Exception:
    _strict_min_score = float(OPENAI_TRADE_CONFIG.get("min_score_abs", 48.0) or 48.0)
OPENAI_TRADE_CONFIG["min_score_abs"] = max(_strict_min_score, 0.0)
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
LAST_MARKETS_CACHE: Dict[str, Any] = {}
LAST_TICKERS_CACHE: Dict[str, Any] = {}
LAST_MARKETS_TS = 0.0
LAST_TICKERS_TS = 0.0
LAST_STATE_SNAPSHOT_TS = 0.0
OPENAI_REVIEW_ROTATION_SIGNATURE = ""
OPENAI_REVIEW_ROTATION_CURSOR = 0
COPY_ALLOWED_SYMBOL_KEYS: set[str] = set()
COPY_ALLOWED_SYMBOL_KEYS_TS = 0.0
BITGET_AUTHORITY_CACHE: Dict[str, Any] = {"ts": 0.0, "authorities": set(), "trader_type": "", "error": ""}
COPY_TRADE_ORDER_LOCK = threading.RLock()
LAST_COPY_TRADE_ORDER_TS = 0.0


def tw_now_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    return datetime.now().strftime(fmt)


def tw_from_ts(ts: Any, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    value = safe_float(ts, 0.0)
    if value <= 0:
        return "--"
    try:
        return datetime.fromtimestamp(value).strftime(fmt)
    except Exception:
        return "--"


def _is_copy_permission_error(err: Any) -> bool:
    text = str(err or "")
    text_lower = text.lower()
    probes = (
        "40014",
        "Incorrect permissions",
        "trace read",
        "trace write",
        "future order read",
        "future order write",
        "copy-equity",
    )
    return any(str(token).lower() in text_lower for token in probes)


def _is_trace_permission_error(err: Any) -> bool:
    text = str(err or "").lower()
    probes = (
        "40014",
        "trace read",
        "trace write",
        "future order read",
        "future order write",
    )
    return any(token in text for token in probes)


def _bitget_account_hint_tokens() -> List[str]:
    raw = str(BITGET_ACCOUNT_NAME_HINT or "").strip()
    if not raw:
        return []
    normalized = raw.replace("，", ",").replace(";", ",")
    return [token.strip().lower() for token in normalized.split(",") if str(token).strip()]


BITGET_ACCOUNT_HINT_TOKENS = _bitget_account_hint_tokens()


def _bitget_account_row_label(row: Dict[str, Any]) -> str:
    item = dict(row or {})
    probes = (
        "accountName",
        "account_name",
        "name",
        "label",
        "remark",
        "tag",
        "accountTypeName",
        "account_type_name",
        "accountType",
        "account_type",
        "subAccountName",
        "sub_account_name",
        "businessType",
        "business_type",
        "account",
    )
    parts: List[str] = []
    for key in probes:
        text = str(item.get(key) or "").strip()
        if text:
            parts.append(text)
    return " | ".join(parts)


def _bitget_account_row_matches_hint(row: Dict[str, Any]) -> bool:
    if not BITGET_ACCOUNT_HINT_TOKENS:
        return True
    label = _bitget_account_row_label(row).lower()
    if not label:
        return False
    return all(token in label for token in BITGET_ACCOUNT_HINT_TOKENS)


def _bitget_order_product_types() -> List[str]:
    ordered: List[str] = [str(BITGET_PRODUCT_TYPE or "").strip()]
    out: List[str] = []
    for value in ordered:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out or [str(BITGET_PRODUCT_TYPE or "USDT-FUTURES").strip()]


def _mask_secret_tail(value: Any, *, keep: int = 6) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= keep:
        return "*" * len(text)
    return ("*" * max(0, len(text) - keep)) + text[-keep:]


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


def symbol_key(symbol: str) -> str:
    token = compact_symbol(symbol).upper().strip()
    return token.replace("/", "").replace(":", "").replace("-", "").replace("_", "")


def symbol_contract_id(symbol: str) -> str:
    token = symbol_key(symbol)
    if token:
        return token
    return compact_symbol(symbol).upper().strip()


def contract_symbol_to_ccxt_symbol(symbol: str) -> str:
    raw = str(symbol or "").strip()
    if not raw:
        return ""
    if "/" in raw:
        return raw
    token = symbol_key(raw)
    if not token:
        return raw
    settle_candidates = [str(BITGET_MARGIN_COIN or "").strip().upper(), "USDT", "USDC"]
    seen_settle: set[str] = set()
    for settle in settle_candidates:
        settle = str(settle or "").strip().upper()
        if not settle or settle in seen_settle:
            continue
        seen_settle.add(settle)
        if token.endswith(settle) and len(token) > len(settle):
            base = token[: -len(settle)]
            return "{}/{}:{}".format(base, settle, settle)
    return raw


class BitgetAPIError(RuntimeError):
    def __init__(self, details: Dict[str, Any]):
        self.details = dict(details or {})
        super().__init__(json.dumps(self.details, ensure_ascii=False, separators=(",", ":")))


def sign(timestamp: str, method: str, request_path: str, query_string: str = "", body_text: str = "") -> str:
    query = str(query_string or "")
    if query and not query.startswith("?"):
        query = "?" + query
    prehash = "{}{}{}{}{}".format(
        str(timestamp or ""),
        str(method or "GET").upper(),
        str(request_path or ""),
        query,
        str(body_text or ""),
    )
    secret = str(env_or_blank("BITGET_SECRET", "") or "")
    return base64.b64encode(
        hmac.new(secret.encode("utf-8"), prehash.encode("utf-8"), hashlib.sha256).digest()
    ).decode("utf-8")


def _bitget_auth_headers(timestamp: str, method: str, request_path: str, query_string: str = "", body_text: str = "") -> Dict[str, str]:
    passphrase = str(
        env_or_blank("BITGET_PASSWORD", "")
        or env_or_blank("BITGET_PASSPHRASE", "")
        or ""
    )
    return {
        "ACCESS-KEY": str(env_or_blank("BITGET_API_KEY", "") or ""),
        "ACCESS-SIGN": sign(timestamp, method, request_path, query_string, body_text),
        "ACCESS-PASSPHRASE": passphrase,
        "ACCESS-TIMESTAMP": str(timestamp or ""),
        "locale": "en-US",
        "Content-Type": "application/json",
    }


def request(method: str, endpoint: str, *, params: Dict[str, Any] | None = None, payload: Dict[str, Any] | None = None) -> Any:
    method_upper = str(method or "GET").upper()
    request_path = str(endpoint or "").strip()
    clean_params = {str(k): v for k, v in dict(params or {}).items() if v not in (None, "")}
    query_text = urlencode(clean_params)
    query_for_sign = "?{}".format(query_text) if query_text else ""
    clean_payload = dict(payload or {})
    body_text = json.dumps(clean_payload, separators=(",", ":"), ensure_ascii=False) if clean_payload else ""
    timestamp = str(int(time.time() * 1000))
    headers = _bitget_auth_headers(timestamp, method_upper, request_path, query_for_sign, body_text)
    url = "{}{}".format(BASE_URL, request_path)
    if query_text:
        url = "{}?{}".format(url, query_text)
    response = requests.request(
        method_upper,
        url,
        headers=headers,
        data=body_text if body_text else None,
        timeout=15,
    )
    raw_text = str(getattr(response, "text", "") or "")
    try:
        data = response.json()
    except Exception:
        data = {}
    status_code = int(getattr(response, "status_code", 0) or 0)
    code = str((data or {}).get("code") or "")
    msg = str((data or {}).get("msg") or (data or {}).get("message") or raw_text[:240] or "request_failed")
    response_data = (data or {}).get("data")
    if status_code >= 400 or code != "00000":
        raise BitgetAPIError(
            {
                "endpoint": request_path,
                "params": clean_params,
                "payload": clean_payload,
                "status_code": status_code,
                "code": code,
                "msg": msg[:260],
                "data": response_data,
            }
        )
    return response_data


def _bitget_private_request(method: str, request_path: str, *, query: Dict[str, Any] | None = None, body: Dict[str, Any] | None = None) -> Any:
    return request(method, request_path, params=query, payload=body)


def _bitget_spot_account_info() -> Dict[str, Any]:
    data = _bitget_private_request("GET", BITGET_ENDPOINTS["spot_account_info"])
    return dict(data or {})


def _bitget_authorities_snapshot(force: bool = False) -> Dict[str, Any]:
    global BITGET_AUTHORITY_CACHE
    now = now_ts()
    cache = dict(BITGET_AUTHORITY_CACHE or {})
    if (not force) and (now - safe_float(cache.get("ts"), 0.0) < 300.0):
        return {
            "authorities": set(cache.get("authorities") or set()),
            "trader_type": str(cache.get("trader_type") or ""),
            "error": str(cache.get("error") or ""),
        }
    try:
        info = _bitget_spot_account_info()
        authorities = {str(x or "").strip().lower() for x in list(info.get("authorities") or []) if str(x or "").strip()}
        trader_type = str(info.get("traderType") or "").strip().lower()
        BITGET_AUTHORITY_CACHE = {"ts": now, "authorities": set(authorities), "trader_type": trader_type, "error": ""}
    except Exception as exc:
        BITGET_AUTHORITY_CACHE = {"ts": now, "authorities": set(), "trader_type": "", "error": str(exc)[:220]}
    return {
        "authorities": set(BITGET_AUTHORITY_CACHE.get("authorities") or set()),
        "trader_type": str(BITGET_AUTHORITY_CACHE.get("trader_type") or ""),
        "error": str(BITGET_AUTHORITY_CACHE.get("error") or ""),
    }


def _copy_authority_ready() -> tuple[bool, str]:
    snap = _bitget_authorities_snapshot()
    authorities = set(snap.get("authorities") or set())
    if not authorities:
        err = str(snap.get("error") or "").strip()
        if err:
            return False, "authority_probe_failed:{}".format(err[:180])
        return False, "authority_probe_empty"
    has_copy_read = bool({"ttor", "ttow"} & authorities)
    if not has_copy_read:
        return False, "copy_authority_missing:{}".format(",".join(sorted(list(authorities))[:12]))
    return True, "ok"


def _build_contract_order_params(side: str, *, reduce_only: bool = False) -> Dict[str, Any]:
    pos_side = "long" if str(side or "").lower() in ("buy", "long") else "short"
    params: Dict[str, Any] = {
        "productType": BITGET_PRODUCT_TYPE,
        "marginCoin": BITGET_MARGIN_COIN,
        "marginMode": BITGET_MARGIN_MODE,
        "posSide": pos_side,
        "tradeSide": "close" if reduce_only else "open",
        "reduceOnly": "YES" if reduce_only else "NO",
    }
    if BITGET_MARGIN_MODE == "crossed":
        params["tdMode"] = "cross"
    else:
        params["tdMode"] = "isolated"
    return params


def _enforce_copy_order_rate_limit() -> None:
    global LAST_COPY_TRADE_ORDER_TS
    with COPY_TRADE_ORDER_LOCK:
        now = time.time()
        wait_sec = COPY_TRADE_ORDER_MIN_INTERVAL_SEC - (now - LAST_COPY_TRADE_ORDER_TS)
        if wait_sec > 0:
            time.sleep(wait_sec)
        LAST_COPY_TRADE_ORDER_TS = time.time()


def _require_copy_trader_mode(caller: str) -> None:
    if BITGET_USE_COPY_TRADER_API:
        return
    raise RuntimeError(
        "{}_requires_copy_trader_mode:BITGET_API_MODE={}".format(
            str(caller or "api_call"),
            BITGET_API_MODE,
        )
    )


def _resolve_hedge_fields(side: str, trade_side: str, hold_side: str = "") -> tuple[str, str, str]:
    side_clean = str(side or "").strip().lower()
    if side_clean in {"long", "open_long", "close_short"}:
        side_clean = "buy"
    if side_clean in {"short", "open_short", "close_long"}:
        side_clean = "sell"
    if side_clean not in {"buy", "sell"}:
        raise ValueError("invalid_side:{}".format(side))
    trade_side_clean = str(trade_side or "").strip().lower()
    if trade_side_clean not in {"open", "close"}:
        raise ValueError("invalid_tradeSide:{}".format(trade_side))
    hold_side_clean = str(hold_side or "").strip().lower()
    if hold_side_clean not in {"long", "short"}:
        if trade_side_clean == "open":
            hold_side_clean = "long" if side_clean == "buy" else "short"
        else:
            hold_side_clean = "long" if side_clean == "sell" else "short"
    return side_clean, trade_side_clean, hold_side_clean


def get_balance() -> Dict[str, float]:
    _require_copy_trader_mode("get_balance")
    copy_summary = dict(_bitget_private_request("GET", BITGET_ENDPOINTS["copy_total_detail"]) or {})
    tracks = _bitget_copy_fetch_current_tracks()
    equity = safe_float(copy_summary.get("totalEquity"), 0.0)
    margin = 0.0
    unrealized = 0.0
    for row in tracks:
        margin += abs(safe_float(row.get("openMarginSz"), 0.0))
        unrealized += safe_float(row.get("unrealizedPL"), safe_float(row.get("unrealizedPnl"), 0.0))
    available = max(equity - margin, 0.0)
    return {
        "available": available,
        "equity": equity,
        "margin": margin,
        "unrealized_pnl": unrealized,
    }


def get_positions(symbol: str = "") -> Dict[str, Dict[str, float]]:
    _require_copy_trader_mode("get_positions")
    query: Dict[str, Any] = {"productType": BITGET_PRODUCT_TYPE, "limit": "50"}
    symbol_id = symbol_contract_id(symbol)
    if symbol_id:
        query["symbol"] = symbol_id
    data = _bitget_private_request("GET", BITGET_ENDPOINTS["copy_current_track"], query=query)
    if isinstance(data, dict):
        rows = [dict(row or {}) for row in list(data.get("trackingList") or data.get("list") or [])]
    else:
        rows = [dict(row or {}) for row in list(data or [])]
    positions: Dict[str, Dict[str, float]] = {
        "long": {"size": 0.0, "entry_price": 0.0, "unrealized_pnl": 0.0, "margin": 0.0},
        "short": {"size": 0.0, "entry_price": 0.0, "unrealized_pnl": 0.0, "margin": 0.0},
    }
    entry_notional: Dict[str, float] = {"long": 0.0, "short": 0.0}
    for row in rows:
        hold = str(row.get("holdSide") or row.get("posSide") or "").strip().lower()
        if hold not in {"long", "short"}:
            continue
        size = abs(safe_float(row.get("openSize"), 0.0))
        if size <= 0:
            continue
        entry = safe_float(row.get("openPriceAvg"), 0.0)
        margin = abs(safe_float(row.get("openMarginSz"), 0.0))
        unrealized = safe_float(row.get("unrealizedPL"), safe_float(row.get("unrealizedPnl"), 0.0))
        positions[hold]["size"] += size
        positions[hold]["margin"] += margin
        positions[hold]["unrealized_pnl"] += unrealized
        entry_notional[hold] += (entry * size) if entry > 0 else 0.0
    for hold in ("long", "short"):
        size = positions[hold]["size"]
        positions[hold]["entry_price"] = (entry_notional[hold] / size) if size > 0 else 0.0
    return positions


def place_order(
    symbol: str,
    size: float,
    side: str,
    trade_side: str,
    *,
    hold_side: str = "",
    order_type: str = "market",
    price: float | None = None,
    client_oid: str = "",
) -> Dict[str, Any]:
    _require_copy_trader_mode("place_order")
    side_clean, trade_side_clean, hold_side_clean = _resolve_hedge_fields(side, trade_side, hold_side)
    qty = safe_float(size, 0.0)
    if qty <= 0:
        raise RuntimeError("invalid_size:{}".format(size))
    if trade_side_clean == "close":
        return close_position(symbol, hold_side_clean, qty, client_oid=client_oid)
    _enforce_copy_order_rate_limit()
    # Copy-trader opening path (official): use mix place-order with hedge fields.
    payload: Dict[str, Any] = {
        "symbol": symbol_contract_id(symbol),
        "productType": BITGET_PRODUCT_TYPE,
        "marginMode": BITGET_MARGIN_MODE,
        "marginCoin": BITGET_MARGIN_COIN,
        "size": str(qty),
        "side": side_clean,
        "tradeSide": trade_side_clean,
        "holdSide": hold_side_clean,
        "orderType": str(order_type or "market").strip().lower(),
        "clientOid": str(client_oid or "bot-{}-{}".format(int(time.time() * 1000), symbol_contract_id(symbol)[:12])),
    }
    if payload["orderType"] == "limit":
        payload["price"] = str(safe_float(price, 0.0))
        payload["force"] = "gtc"
    data = _bitget_private_request("POST", BITGET_ENDPOINTS["copy_trader_open_order"], body=payload)
    result = dict(data or {})
    order_id = str(result.get("orderId") or result.get("clientOid") or "")
    return {"id": order_id, "client_oid": str(result.get("clientOid") or ""), "info": result}


def close_position(symbol: str, hold_side: str, size: float, *, client_oid: str = "") -> Dict[str, Any]:
    _require_copy_trader_mode("close_position")
    hold = str(hold_side or "").strip().lower()
    if hold not in {"long", "short"}:
        raise RuntimeError("invalid_holdSide:{}".format(hold_side))
    qty = safe_float(size, 0.0)
    if qty <= 0:
        raise RuntimeError("invalid_size:{}".format(size))
    _enforce_copy_order_rate_limit()
    # Copy-trader closing path (official): use copy trader tracking close endpoint.
    track_rows = []
    for row in _bitget_copy_fetch_current_tracks(symbol):
        if symbol_key(str(row.get("symbol") or "")) != symbol_key(symbol):
            continue
        row_hold = str(row.get("holdSide") or row.get("posSide") or "").strip().lower()
        if row_hold != hold:
            continue
        row_size = abs(safe_float(row.get("openSize"), 0.0))
        if row_size <= 0:
            continue
        track_rows.append(dict(row or {}))
    if not track_rows:
        raise RuntimeError("no_tracking_position:{}:{}".format(symbol, hold))
    remaining = qty
    closed: List[Dict[str, Any]] = []
    for row in track_rows:
        if remaining <= 0:
            break
        payload: Dict[str, Any] = {
            "symbol": symbol_contract_id(symbol),
            "productType": BITGET_PRODUCT_TYPE,
            "trackingNo": str(row.get("trackingNo") or ""),
        }
        data = _bitget_private_request("POST", BITGET_ENDPOINTS["copy_close_positions"], body=payload)
        if isinstance(data, list):
            item = dict(data[0] or {}) if data else {}
        else:
            item = dict(data or {})
        closed.append(item)
        remaining -= abs(safe_float(row.get("openSize"), 0.0))
    if not closed:
        raise RuntimeError("copy_close_empty_response")
    return {
        "id": str((closed[-1] or {}).get("trackingNo") or ""),
        "client_oid": str(client_oid or ""),
        "info": {"closed": closed, "requested_size": qty, "remaining_size": max(remaining, 0.0)},
    }


def open_long(symbol: str, size: float, *, order_type: str = "market", price: float | None = None) -> Dict[str, Any]:
    return place_order(symbol, size, "buy", "open", hold_side="long", order_type=order_type, price=price)


def open_short(symbol: str, size: float, *, order_type: str = "market", price: float | None = None) -> Dict[str, Any]:
    return place_order(symbol, size, "sell", "open", hold_side="short", order_type=order_type, price=price)


def close_long(symbol: str, size: float) -> Dict[str, Any]:
    return place_order(symbol, size, "sell", "close", hold_side="long", order_type="market")


def close_short(symbol: str, size: float) -> Dict[str, Any]:
    return place_order(symbol, size, "buy", "close", hold_side="short", order_type="market")


def _bitget_copy_fetch_current_tracks(symbol: str = "") -> List[Dict[str, Any]]:
    _require_copy_trader_mode("_bitget_copy_fetch_current_tracks")
    query: Dict[str, Any] = {"productType": BITGET_PRODUCT_TYPE, "limit": "50"}
    symbol_id = symbol_contract_id(symbol)
    if symbol_id:
        query["symbol"] = symbol_id
    data = _bitget_private_request("GET", BITGET_ENDPOINTS["copy_current_track"], query=query)
    if isinstance(data, dict):
        return [dict(row or {}) for row in list(data.get("trackingList") or [])]
    if isinstance(data, list):
        return [dict(row or {}) for row in data]
    return []


def _bitget_copy_total_equity_usdt() -> float:
    _require_copy_trader_mode("_bitget_copy_total_equity_usdt")
    data = _bitget_private_request("GET", BITGET_ENDPOINTS["copy_total_detail"])
    payload = dict(data or {})
    return safe_float(payload.get("totalEquity"), 0.0)


def _bitget_copy_close_tracking(symbol: str, tracking_no: str = "") -> Dict[str, Any]:
    payload: Dict[str, Any] = {"productType": BITGET_PRODUCT_TYPE}
    symbol_id = symbol_contract_id(symbol)
    if symbol_id:
        payload["symbol"] = symbol_id
    if str(tracking_no or "").strip():
        payload["trackingNo"] = str(tracking_no).strip()
    data = _bitget_private_request("POST", BITGET_ENDPOINTS["copy_close_positions"], body=payload)
    if isinstance(data, list):
        for row in data:
            item = dict(row or {})
            if symbol_id and symbol_contract_id(str(item.get("symbol") or "")) == symbol_id:
                return item
        return dict(data[0] or {}) if data else {}
    return dict(data or {})


def _bitget_copy_modify_tpsl(symbol: str, tracking_no: str, stop_loss: float, take_profit: float) -> bool:
    if not str(tracking_no or "").strip():
        return False
    payload: Dict[str, Any] = {
        "symbol": symbol_contract_id(symbol),
        "productType": BITGET_PRODUCT_TYPE,
        "trackingNo": str(tracking_no).strip(),
    }
    if safe_float(take_profit, 0.0) > 0:
        payload["stopSurplusPrice"] = str(safe_float(take_profit, 0.0))
    if safe_float(stop_loss, 0.0) > 0:
        payload["stopLossPrice"] = str(safe_float(stop_loss, 0.0))
    if "stopSurplusPrice" not in payload and "stopLossPrice" not in payload:
        return False
    try:
        _bitget_private_request("POST", BITGET_ENDPOINTS["copy_modify_tpsl"], body=payload)
        return True
    except Exception:
        return False


def _bitget_copy_query_symbol_settings() -> List[Dict[str, Any]]:
    data = _bitget_private_request(
        "GET",
        BITGET_ENDPOINTS["copy_symbol_settings"],
        query={"productType": BITGET_PRODUCT_TYPE},
    )
    if isinstance(data, list):
        return [dict(row or {}) for row in data]
    if isinstance(data, dict):
        return [dict(row or {}) for row in list(data.get("list") or data.get("symbolList") or [])]
    return []


def _copy_allowed_symbol_keys() -> set[str]:
    global COPY_ALLOWED_SYMBOL_KEYS, COPY_ALLOWED_SYMBOL_KEYS_TS
    now = now_ts()
    if COPY_ALLOWED_SYMBOL_KEYS and (now - COPY_ALLOWED_SYMBOL_KEYS_TS) < COPY_SYMBOLS_CACHE_TTL_SEC:
        return set(COPY_ALLOWED_SYMBOL_KEYS)
    rows = _bitget_copy_query_symbol_settings()
    allowed: set[str] = set()
    for row in rows:
        item = dict(row or {})
        open_flag = str(item.get("openTrader") or item.get("open_trader") or "").strip().lower()
        if open_flag and open_flag not in {"yes", "y", "true", "1"}:
            continue
        sym = str(item.get("symbol") or "").strip()
        if not sym:
            continue
        allowed.add(symbol_key(sym))
    COPY_ALLOWED_SYMBOL_KEYS = set(allowed)
    COPY_ALLOWED_SYMBOL_KEYS_TS = now
    return set(allowed)


def _bitget_mix_order_detail(symbol: str, *, order_id: str = "", client_oid: str = "") -> Dict[str, Any]:
    symbol_id = symbol_contract_id(symbol)
    order_id = str(order_id or "").strip()
    client_oid = str(client_oid or "").strip()
    if not order_id and not client_oid:
        return {}
    last_exc: Exception | None = None
    for product_type in _bitget_order_product_types():
        base_query: Dict[str, Any] = {"symbol": symbol_id, "productType": product_type}
        if order_id:
            try:
                data = _bitget_private_request(
                    "GET",
                    BITGET_ENDPOINTS["mix_order_detail"],
                    query=dict(base_query, orderId=order_id),
                )
                return dict(data or {})
            except Exception as exc:
                last_exc = exc
        probe_client_oid = client_oid or order_id
        if probe_client_oid:
            try:
                data = _bitget_private_request(
                    "GET",
                    BITGET_ENDPOINTS["mix_order_detail"],
                    query=dict(base_query, clientOid=probe_client_oid),
                )
                return dict(data or {})
            except Exception as exc:
                last_exc = exc
    if last_exc:
        raise last_exc
    return {}


def _bitget_mix_cancel_order(symbol: str, *, order_id: str = "", client_oid: str = "") -> Dict[str, Any]:
    symbol_id = symbol_contract_id(symbol)
    order_id = str(order_id or "").strip()
    client_oid = str(client_oid or "").strip()
    if not order_id and not client_oid:
        raise RuntimeError("cancel_order_id_missing")
    last_exc: Exception | None = None
    for product_type in _bitget_order_product_types():
        base_payload: Dict[str, Any] = {
            "symbol": symbol_id,
            "productType": product_type,
            "marginCoin": BITGET_MARGIN_COIN,
        }
        if order_id:
            try:
                data = _bitget_private_request(
                    "POST",
                    BITGET_ENDPOINTS["mix_cancel_order"],
                    body=dict(base_payload, orderId=order_id),
                )
                return dict(data or {})
            except Exception as exc:
                last_exc = exc
        probe_client_oid = client_oid or order_id
        if probe_client_oid:
            try:
                data = _bitget_private_request(
                    "POST",
                    BITGET_ENDPOINTS["mix_cancel_order"],
                    body=dict(base_payload, clientOid=probe_client_oid),
                )
                return dict(data or {})
            except Exception as exc:
                last_exc = exc
    if last_exc:
        raise last_exc
    return {}


def _create_contract_order(
    symbol: str,
    order_type: str,
    side: str,
    qty: float,
    price: float | None,
    *,
    reduce_only: bool = False,
    initial_stop_loss: float | None = None,
    initial_take_profit: float | None = None,
) -> Dict[str, Any]:
    errors: List[str] = []
    if BITGET_USE_COPY_TRADER_API:
        trade_side = "close" if reduce_only else "open"
        hold_side = ""
        side_text = str(side or "").strip().lower()
        if trade_side == "open":
            hold_side = "long" if side_text == "buy" else "short"
        else:
            hold_side = "long" if side_text == "sell" else "short"
        return place_order(
            symbol,
            qty,
            side_text,
            trade_side,
            hold_side=hold_side,
            order_type=str(order_type or "market").lower(),
            price=price,
        )
    params = _build_contract_order_params(side, reduce_only=reduce_only)
    try:
        return dict(exchange.create_order(symbol, order_type, side, qty, price, params) or {})
    except Exception as exc:
        errors.append("ccxt_create_order:" + str(exc)[:220])
    payload: Dict[str, Any] = {
        "symbol": symbol_contract_id(symbol),
        "productType": BITGET_PRODUCT_TYPE,
        "marginMode": BITGET_MARGIN_MODE,
        "marginCoin": BITGET_MARGIN_COIN,
        "size": str(qty),
        "side": str(side or "").lower(),
        "tradeSide": "close" if reduce_only else "open",
        "orderType": str(order_type or "market").lower(),
        "holdSide": "long" if (str(side or "").lower() == "buy") == (not reduce_only) else "short",
        "clientOid": "bot-{}-{}".format(int(time.time() * 1000), symbol_contract_id(symbol)[:12]),
    }
    if payload["orderType"] == "limit":
        payload["price"] = str(safe_float(price, 0.0))
        payload["force"] = "gtc"
    if not reduce_only:
        init_sl = safe_float(initial_stop_loss, 0.0)
        init_tp = safe_float(initial_take_profit, 0.0)
        if init_tp > 0:
            payload["presetStopSurplusPrice"] = str(init_tp)
        if init_sl > 0:
            payload["presetStopLossPrice"] = str(init_sl)
    try:
        data = _bitget_private_request("POST", BITGET_ENDPOINTS["mix_place_order"], body=payload)
        order_id = str((dict(data or {})).get("orderId") or (dict(data or {})).get("clientOid") or "")
        return {"id": order_id, "client_oid": str((dict(data or {})).get("clientOid") or ""), "info": data}
    except Exception as exc:
        errors.append("mix_v2_place_order_fallback:" + str(exc)[:220])
    raise RuntimeError("order_failed:" + " | ".join(errors[:4]))

def base_asset(symbol: str) -> str:
    token = compact_symbol(symbol)
    if token.endswith("USDT"):
        token = token[:-4]
    return token.upper()


def fixed_order_notional_usdt(symbol: str, reference_price: float = 0.0) -> float:
    asset = base_asset(symbol)
    if asset in SPECIAL_NOTIONAL_SYMBOLS:
        return SPECIAL_NOTIONAL_USDT
    ref_price = safe_float(reference_price, 0.0)
    if ref_price > HIGH_PRICE_ORDER_THRESHOLD_USDT:
        return HIGH_PRICE_ORDER_NOTIONAL_USDT
    return BTC_ETH_MIN_ORDER_NOTIONAL_USDT if asset in HIGH_NOTIONAL_SYMBOLS else FIXED_ORDER_NOTIONAL_USDT


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


def _extract_structure_bars(context: Dict[str, Any], timeframe: str = "15m", limit: int = 40) -> List[Dict[str, float]]:
    bars = extract_recent_bars(context, timeframe, max(limit, 8))
    cleaned: List[Dict[str, float]] = []
    for row in bars:
        open_price = safe_float(row.get("open"), 0.0)
        high_price = safe_float(row.get("high"), 0.0)
        low_price = safe_float(row.get("low"), 0.0)
        close_price = safe_float(row.get("close"), 0.0)
        if open_price <= 0 or high_price <= 0 or low_price <= 0 or close_price <= 0:
            continue
        high_price = max(high_price, open_price, close_price, low_price)
        low_price = min(low_price, open_price, close_price, high_price)
        cleaned.append({"open": open_price, "high": high_price, "low": low_price, "close": close_price})
    return cleaned


def _find_recent_structure_anchor(side: str, bars: List[Dict[str, float]], entry_price: float, atr_price: float = 0.0) -> tuple[float, str]:
    if not bars or entry_price <= 0:
        return 0.0, "no_bars"
    exclude_recent = 3 if len(bars) >= 9 else 2
    analysis_bars = list(bars[:-exclude_recent] if len(bars) > (exclude_recent + 2) else bars)
    highs = [safe_float(row.get("high"), 0.0) for row in analysis_bars]
    lows = [safe_float(row.get("low"), 0.0) for row in analysis_bars]
    pivot_lows: List[float] = []
    pivot_highs: List[float] = []
    min_anchor_gap = max(safe_float(atr_price, 0.0) * 0.65, entry_price * 0.0035)
    end_idx = max(len(analysis_bars) - 2, 2)
    for idx in range(2, end_idx):
        low = lows[idx]
        high = highs[idx]
        if low > 0 and low <= lows[idx - 1] and low <= lows[idx + 1] and low <= lows[idx - 2] and low <= lows[idx + 2]:
            pivot_lows.append(low)
        if high > 0 and high >= highs[idx - 1] and high >= highs[idx + 1] and high >= highs[idx - 2] and high >= highs[idx + 2]:
            pivot_highs.append(high)
    if side == "long":
        valid_pivot_lows = [price for price in pivot_lows if 0 < price < entry_price]
        valid_far_lows = [price for price in valid_pivot_lows if (entry_price - price) >= min_anchor_gap]
        if valid_far_lows:
            return max(valid_far_lows), "pivot_low_15m_far"
        if valid_pivot_lows:
            return min(valid_pivot_lows), "pivot_low_15m_deep"
        fallback_lows = [low for low in lows[:-1] if 0 < low < entry_price]
        fallback_far_lows = [low for low in fallback_lows if (entry_price - low) >= min_anchor_gap]
        if fallback_far_lows:
            return max(fallback_far_lows[-min(len(fallback_far_lows), 16):]), "fallback_low_15m_far"
        if fallback_lows:
            return min(fallback_lows[-min(len(fallback_lows), 12):]), "fallback_low_15m"
    else:
        valid_pivot_highs = [price for price in pivot_highs if price > entry_price]
        valid_far_highs = [price for price in valid_pivot_highs if (price - entry_price) >= min_anchor_gap]
        if valid_far_highs:
            return min(valid_far_highs), "pivot_high_15m_far"
        if valid_pivot_highs:
            return max(valid_pivot_highs), "pivot_high_15m_deep"
        fallback_highs = [high for high in highs[:-1] if high > entry_price]
        fallback_far_highs = [high for high in fallback_highs if (high - entry_price) >= min_anchor_gap]
        if fallback_far_highs:
            return min(fallback_far_highs[-min(len(fallback_far_highs), 16):]), "fallback_high_15m_far"
        if fallback_highs:
            return max(fallback_highs[-min(len(fallback_highs), 12):]), "fallback_high_15m"
    return 0.0, "no_valid_anchor"


def _derive_structure_stop_15m(side: str, context: Dict[str, Any], entry_price: float, atr_price: float) -> tuple[float, float, str, bool]:
    tf15 = dict((context.get("multi_timeframe") or {}).get("15m") or {})
    bars = _extract_structure_bars(context, "15m", STRUCTURE_SWING_LOOKBACK_15M + 6)
    anchor, anchor_source = _find_recent_structure_anchor(side, bars, entry_price, atr_price)
    prior_bars = bars[:-1] if len(bars) > 1 else bars
    recent_slice = prior_bars[-min(len(prior_bars), 12):] if prior_bars else []
    range_high = max([safe_float(row.get("high"), 0.0) for row in recent_slice], default=0.0)
    range_low = min([safe_float(row.get("low"), 0.0) for row in recent_slice if safe_float(row.get("low"), 0.0) > 0], default=0.0)
    range_span = max(range_high - range_low, 0.0) if range_high > 0 and range_low > 0 else 0.0
    consolidating = range_span > 0 and range_span <= max(atr_price * 2.8, entry_price * 0.006)
    anti_sweep = max(atr_price * 0.28, entry_price * 0.0024)
    explosive_move = bool(tf15.get("explosive_move", False))
    if side == "long":
        if consolidating and range_low > 0 and (anchor <= 0 or anchor > (range_low + range_span * 0.45)):
            anchor = range_low
            anchor_source = "range_low_15m"
        base_stop = anchor - anti_sweep if 0 < anchor < entry_price else entry_price - max(atr_price, entry_price * 0.008)
        raw_gap = entry_price - base_stop
    else:
        if consolidating and range_high > 0 and (anchor <= 0 or anchor < (range_high - range_span * 0.45)):
            anchor = range_high
            anchor_source = "range_high_15m"
        base_stop = anchor + anti_sweep if anchor > entry_price else entry_price + max(atr_price, entry_price * 0.008)
        raw_gap = base_stop - entry_price
    min_gap = max(atr_price * 1.30, entry_price * 0.0075)
    max_gap = max(atr_price * (2.0 if explosive_move else 2.6), min_gap * 1.35)
    gap = clamp(raw_gap, min_gap, max_gap)
    stop_price = entry_price - gap if side == "long" else entry_price + gap
    return stop_price, anchor, anchor_source, consolidating


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
    "prebreakout_signals": [],
    "prebreakout_leaderboard": {},
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
            "prebreakout_signals": list(STATE.get("prebreakout_signals", []))[:PREBREAKOUT_TOP_PICK],
            "prebreakout_leaderboard": dict(STATE.get("prebreakout_leaderboard") or {}),
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


def persist_runtime_snapshot_throttled(*, force: bool = False) -> None:
    global LAST_STATE_SNAPSHOT_TS
    now = time.time()
    if force or (now - LAST_STATE_SNAPSHOT_TS) >= STATE_SNAPSHOT_MIN_INTERVAL_SEC:
        persist_runtime_snapshot()
        LAST_STATE_SNAPSHOT_TS = now


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
            if isinstance(base.get("prebreakout_signals"), list):
                STATE["prebreakout_signals"] = list(base["prebreakout_signals"])[:PREBREAKOUT_TOP_PICK]
            if isinstance(base.get("prebreakout_leaderboard"), dict):
                STATE["prebreakout_leaderboard"] = dict(base["prebreakout_leaderboard"])
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
    global LAST_STATE_SNAPSHOT_TS
    persist_flag = bool(kwargs.pop("_persist", True))
    force_persist = bool(kwargs.pop("_force_persist", False))
    with STATE_LOCK:
        for key, value in kwargs.items():
            STATE[key] = value
        STATE["last_update"] = tw_now_str()
        snapshot = dict(STATE)
    sync_runtime_views()
    if persist_flag:
        now = time.time()
        if force_persist or (now - LAST_STATE_SNAPSHOT_TS) >= STATE_SNAPSHOT_MIN_INTERVAL_SEC:
            persist_runtime_snapshot()
            LAST_STATE_SNAPSHOT_TS = now
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
    persist_runtime_snapshot_throttled()


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


def fetch_exchange_snapshot(*, force_live: bool = False) -> tuple[Dict[str, Any], Dict[str, Any], str]:
    global LAST_MARKETS_CACHE, LAST_TICKERS_CACHE, LAST_MARKETS_TS, LAST_TICKERS_TS
    markets: Dict[str, Any] = {}
    tickers: Dict[str, Any] = {}
    notes: List[str] = []
    now = time.time()
    use_markets_cache = (not force_live) and bool(LAST_MARKETS_CACHE) and (now - LAST_MARKETS_TS) <= EXCHANGE_SNAPSHOT_CACHE_TTL_SEC
    use_tickers_cache = (not force_live) and bool(LAST_TICKERS_CACHE) and (now - LAST_TICKERS_TS) <= EXCHANGE_SNAPSHOT_CACHE_TTL_SEC
    if use_markets_cache:
        markets = dict(LAST_MARKETS_CACHE)
        notes.append("markets_cache")
    if use_tickers_cache:
        tickers = dict(LAST_TICKERS_CACHE)
        notes.append("tickers_cache")
    if not markets:
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                markets = dict(exchange.load_markets() or {})
                LAST_MARKETS_CACHE = dict(markets)
                LAST_MARKETS_TS = time.time()
                notes.append("markets_live")
                break
            except Exception as exc:
                last_err = exc
                if is_rate_limit_error(exc) and attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                break
        if not markets and LAST_MARKETS_CACHE:
            markets = dict(LAST_MARKETS_CACHE)
            notes.append("markets_fallback_cache")
        if not markets and last_err:
            raise last_err
    if not tickers:
        last_err = None
        for attempt in range(3):
            try:
                tickers = dict(exchange.fetch_tickers() or {})
                LAST_TICKERS_CACHE = dict(tickers)
                LAST_TICKERS_TS = time.time()
                notes.append("tickers_live")
                break
            except Exception as exc:
                last_err = exc
                if is_rate_limit_error(exc) and attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                break
        if not tickers and LAST_TICKERS_CACHE:
            tickers = dict(LAST_TICKERS_CACHE)
            notes.append("tickers_fallback_cache")
        if not tickers and last_err:
            raise last_err
    return markets, tickers, ",".join(notes[:4])


def fetch_market_cap_snapshot(base_asset: str) -> Dict[str, Any]:
    base = str(base_asset or "").strip().lower()
    now = time.time()
    if base in MARKET_CAP_CACHE and (now - MARKET_CAP_CACHE_TS.get(base, 0)) < 6 * 3600:
        return dict(MARKET_CAP_CACHE[base])
    row = {
        "available": False,
        "market_cap_usd": None,
        "fdv_usd": None,
        "circulating_supply": None,
        "total_supply": None,
    }

    def _parse_market_entry(raw_entry: Dict[str, Any]) -> Dict[str, Any]:
        entry = dict(raw_entry or {})
        return {
            "available": True,
            "market_cap_usd": safe_float(entry.get("market_cap"), 0.0),
            "fdv_usd": safe_float(entry.get("fully_diluted_valuation"), 0.0),
            "circulating_supply": safe_float(entry.get("circulating_supply"), 0.0),
            "total_supply": safe_float(entry.get("total_supply"), 0.0),
        }

    gecko_base = "https://api.coingecko.com/api"
    gecko_v = "v3"
    data = safe_request_json(
        "{}/{}/coins/markets".format(gecko_base, gecko_v),
        params={"vs_currency": "usd", "symbols": base, "sparkline": "false"},
        timeout=7.0,
    )
    if isinstance(data, list) and data:
        row = _parse_market_entry(dict(data[0] or {}))
    else:
        search = safe_request_json(
            "{}/{}/search".format(gecko_base, gecko_v),
            params={"query": base},
            timeout=7.0,
        )
        candidates = list((dict(search or {}).get("coins") or []))
        candidate_id = ""
        for item in candidates[:12]:
            symbol_text = str((item or {}).get("symbol") or "").strip().lower()
            item_id = str((item or {}).get("id") or "").strip()
            if not item_id:
                continue
            if symbol_text == base:
                candidate_id = item_id
                break
        if not candidate_id and candidates:
            candidate_id = str((candidates[0] or {}).get("id") or "").strip()
        if candidate_id:
            by_id = safe_request_json(
                "{}/{}/coins/markets".format(gecko_base, gecko_v),
                params={"vs_currency": "usd", "ids": candidate_id, "sparkline": "false"},
                timeout=7.0,
            )
            if isinstance(by_id, list) and by_id:
                row = _parse_market_entry(dict(by_id[0] or {}))
    MARKET_CAP_CACHE[base] = row
    MARKET_CAP_CACHE_TS[base] = now
    return dict(row)


def safe_fetch_ohlcv_df(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    cache_key = "{}|{}|{}".format(symbol, timeframe, limit)
    now = time.time()
    ttl_map = {
        "1m": 18.0,
        "5m": 45.0,
        "15m": 90.0,
        "1h": 240.0,
        "4h": 600.0,
        "1d": 1200.0,
    }
    fresh_ttl = ttl_map.get(str(timeframe), 60.0)
    cached_now = OHLCV_CACHE.get(cache_key)
    if isinstance(cached_now, pd.DataFrame) and not cached_now.empty:
        age = now - OHLCV_CACHE_TS.get(cache_key, 0.0)
        if age <= fresh_ttl:
            return cached_now.copy()
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
        open_price = safe_float(row.get("o"), 0.0)
        high_price = safe_float(row.get("h"), 0.0)
        low_price = safe_float(row.get("l"), 0.0)
        close_price = safe_float(row.get("c"), 0.0)
        volume = safe_float(row.get("v"), 0.0)
        if open_price <= 0 or high_price <= 0 or low_price <= 0 or close_price <= 0:
            continue
        high_price = max(high_price, open_price, close_price, low_price)
        low_price = min(low_price, open_price, close_price, high_price)
        volume = max(volume, 0.0)
        rows.append(
            {
                "time": safe_int(row.get("t"), 0),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
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
    def _anomaly_from_ohlcv(tf: str, lookback_prev: int = 3) -> float | None:
        try:
            df = safe_fetch_ohlcv_df(symbol, tf, 16)
        except Exception:
            return None
        if df is None or df.empty or len(df) < (lookback_prev + 1):
            return None
        tail = df.tail(lookback_prev + 1)
        last_row = tail.iloc[-1]
        prev_rows = tail.iloc[:-1]
        last_notional = safe_float(last_row.get("c"), 0.0) * max(safe_float(last_row.get("v"), 0.0), 0.0)
        prev_notionals = [
            safe_float(r.get("c"), 0.0) * max(safe_float(r.get("v"), 0.0), 0.0)
            for _, r in prev_rows.iterrows()
        ]
        prev_avg = (sum(prev_notionals) / max(len(prev_notionals), 1)) if prev_notionals else 0.0
        if last_notional <= 0 or prev_avg <= 0:
            return None
        return round(last_notional / prev_avg, 4)

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
        "volume_anomaly_5m": 1.0,
        "volume_anomaly_15m": 1.0,
        "order_book_available": False,
        "recent_trades_available": False,
        "raw_orderbook_snapshot": {"bids": [], "asks": []},
        "recent_trades": {
            "window_sec": 60,
            "buy_notional": 0.0,
            "sell_notional": 0.0,
            "trade_count": 0,
            "large_trade_threshold_usdt": 0.0,
            "large_buy_count": 0,
            "large_sell_count": 0,
        },
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
        result["raw_orderbook_snapshot"] = {
            "bids": [[round(safe_float(row[0], 0.0), 8), round(safe_float(row[1], 0.0), 8)] for row in bids[:10]],
            "asks": [[round(safe_float(row[0], 0.0), 8), round(safe_float(row[1], 0.0), 8)] for row in asks[:10]],
        }
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
        recent_buy_notional = 0.0
        recent_sell_notional = 0.0
        recent_trade_count = 0
        recent_trade_rows: List[tuple[str, float]] = []
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
            if age_ms <= 60 * 1000:
                recent_trade_count += 1
                recent_trade_rows.append((side, notional))
                if side == "buy":
                    recent_buy_notional += notional
                elif side == "sell":
                    recent_sell_notional += notional
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
        large_trade_threshold = 0.0
        if recent_trade_rows:
            notionals = sorted(max(row[1], 0.0) for row in recent_trade_rows)
            p85_idx = int(max(min(len(notionals) - 1, math.floor((len(notionals) - 1) * 0.85)), 0))
            large_trade_threshold = max(notionals[p85_idx], 1.0)
        large_buy_count = 0
        large_sell_count = 0
        if large_trade_threshold > 0:
            for side, notional in recent_trade_rows:
                if notional < large_trade_threshold:
                    continue
                if side == "buy":
                    large_buy_count += 1
                elif side == "sell":
                    large_sell_count += 1
        result["recent_trades"] = {
            "window_sec": 60,
            "buy_notional": round(recent_buy_notional, 4),
            "sell_notional": round(recent_sell_notional, 4),
            "trade_count": int(recent_trade_count),
            "large_trade_threshold_usdt": round(large_trade_threshold, 4),
            "large_buy_count": int(large_buy_count),
            "large_sell_count": int(large_sell_count),
        }
        result["volume_anomaly_5m"] = round(vol_5m / vol_prev_5m, 4) if vol_prev_5m > 0 else _anomaly_from_ohlcv("5m", 3)
        result["volume_anomaly_15m"] = round(vol_15m / vol_prev_15m, 4) if vol_prev_15m > 0 else _anomaly_from_ohlcv("15m", 3)
        result["recent_trades_available"] = True
    except Exception as exc:
        result["errors"].append("trades:{}".format(str(exc)[:120]))
    if result["volume_anomaly_5m"] is None:
        fallback_5m = _anomaly_from_ohlcv("5m", 3)
        result["volume_anomaly_5m"] = fallback_5m if fallback_5m is not None else 1.0
    if result["volume_anomaly_15m"] is None:
        fallback_15m = _anomaly_from_ohlcv("15m", 3)
        result["volume_anomaly_15m"] = fallback_15m if fallback_15m is not None else 1.0
    result["volume_anomaly_5m"] = round(max(safe_float(result["volume_anomaly_5m"], 1.0), 0.0), 4)
    result["volume_anomaly_15m"] = round(max(safe_float(result["volume_anomaly_15m"], 1.0), 0.0), 4)
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

    def _first_numeric(keys: List[str], default: float = 0.0) -> float:
        for key in keys:
            if key in info:
                value = safe_float(info.get(key), default)
                if value != default:
                    return value
        return default

    buy_vol = safe_float(info.get("buyVolume"), 0.0)
    sell_vol = safe_float(info.get("sellVolume"), 0.0)
    if buy_vol > 0 or sell_vol > 0:
        result["long_short_ratio"] = round((buy_vol + 1e-9) / max(sell_vol, 1e-9), 4)
        result["top_trader_long_short_ratio"] = result["long_short_ratio"]
        result["whale_position_change_pct"] = round(((buy_vol - sell_vol) / max(buy_vol + sell_vol, 1e-9)) * 100, 4)
    if safe_float(result.get("long_short_ratio"), 0.0) <= 0:
        ls_ratio = _first_numeric(
            [
                "longShortRatio",
                "long_short_ratio",
                "buySellRatio",
                "buy_sell_ratio",
                "accountLongShortRatio",
            ],
            0.0,
        )
        if ls_ratio > 0:
            result["long_short_ratio"] = round(ls_ratio, 4)
    if safe_float(result.get("top_trader_long_short_ratio"), 0.0) <= 0:
        top_ls_ratio = _first_numeric(
            [
                "topTraderLongShortRatio",
                "eliteLongShortRatio",
                "topTraderAccountLongShortRatio",
            ],
            0.0,
        )
        if top_ls_ratio > 0:
            result["top_trader_long_short_ratio"] = round(top_ls_ratio, 4)
    if safe_float(result.get("whale_position_change_pct"), 0.0) == 0:
        whale_delta = _first_numeric(
            [
                "whalePositionChangePct",
                "whale_position_change_pct",
                "topTraderLongShortRatioChange",
            ],
            0.0,
        )
        if whale_delta != 0:
            result["whale_position_change_pct"] = round(whale_delta, 4)
    if safe_float(result.get("liquidation_volume_24h"), 0.0) <= 0:
        liq_volume = _first_numeric(
            [
                "liquidationVolume24h",
                "liquidation_volume_24h",
                "liqVolume24h",
                "liquidationValue24h",
            ],
            0.0,
        )
        if liq_volume > 0:
            result["liquidation_volume_24h"] = round(liq_volume, 4)
            result["liquidation_map_status"] = "from_ticker"
    try:
        fetch_long_short_ratio = getattr(exchange, "fetch_long_short_ratio", None)
        if callable(fetch_long_short_ratio) and safe_float(result.get("long_short_ratio"), 0.0) <= 0:
            ls_payload = fetch_long_short_ratio(symbol)
            if isinstance(ls_payload, dict):
                ls_value = safe_float(
                    ls_payload.get("longShortRatio", ls_payload.get("long_short_ratio")),
                    0.0,
                )
                if ls_value > 0:
                    result["long_short_ratio"] = round(ls_value, 4)
    except Exception:
        pass
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
    tf1h = dict((context.get("multi_timeframe") or {}).get("1h") or {})
    tf4h = dict((context.get("multi_timeframe") or {}).get("4h") or {})
    price = safe_float((context.get("basic_market_data") or {}).get("current_price"), 0.0)
    atr = max(safe_float(tf15.get("atr"), 0.0), price * 0.003 if price > 0 else 0.0)
    rr_target = 2.6 if total_score >= 72 else 2.3 if total_score >= 58 else LOCAL_TP_MIN_R
    rr_target = max(rr_target, LOCAL_TP_MIN_R)
    if side not in ("long", "short"):
        side = "long"
    entry = price
    stop, stop_anchor_used, stop_anchor_source, consolidation_guard = _derive_structure_stop_15m(side, context, entry, atr)
    if side == "long":
        if stop >= entry:
            stop = entry - max(atr * 0.75, entry * 0.004)
        risk = max(entry - stop, atr * 0.65, entry * 0.004)
        rr_take = entry + risk * rr_target
        resistances = [
            safe_float(tf15.get("recent_structure_high"), 0.0),
            safe_float(tf1h.get("recent_structure_high"), 0.0),
            safe_float(tf4h.get("recent_structure_high"), 0.0),
        ]
        structure_take = min([price_lv for price_lv in resistances if price_lv > entry], default=0.0)
        take_profit = max(rr_take, structure_take if structure_take > 0 else rr_take)
    else:
        if stop <= entry:
            stop = entry + max(atr * 0.75, entry * 0.004)
        risk = max(stop - entry, atr * 0.65, entry * 0.004)
        rr_take = entry - risk * rr_target
        supports = [
            safe_float(tf15.get("recent_structure_low"), 0.0),
            safe_float(tf1h.get("recent_structure_low"), 0.0),
            safe_float(tf4h.get("recent_structure_low"), 0.0),
        ]
        structure_take = max([price_lv for price_lv in supports if 0 < price_lv < entry], default=0.0)
        take_profit = min(rr_take, structure_take if structure_take > 0 else rr_take)
    rr_ratio = abs((take_profit - entry) / max(abs(entry - stop), 1e-9))
    if rr_ratio < LOCAL_TP_MIN_R:
        take_profit = entry + (abs(entry - stop) * LOCAL_TP_MIN_R) if side == "long" else entry - (abs(entry - stop) * LOCAL_TP_MIN_R)
        rr_ratio = abs((take_profit - entry) / max(abs(entry - stop), 1e-9))
    return {
        "price": round(entry, 6),
        "stop_loss": round(stop, 6),
        "take_profit": round(take_profit, 6),
        "rr_ratio": round(rr_ratio, 2),
        "atr": round(atr, 6),
        "stop_anchor_price": round(safe_float(stop_anchor_used, 0.0), 6),
        "stop_anchor_source": stop_anchor_source,
        "stop_same_bar_exception": False,
        "stop_consolidation_guard": bool(consolidation_guard),
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
    basic_market = dict((context.get("basic_market_data") or {}))
    tf15 = dict((context.get("multi_timeframe") or {}).get("15m") or {})
    current_price = safe_float(
        basic_market.get("current_price"),
        safe_float(tf15.get("last_close"), 0.0),
    )
    change_24h_pct = safe_float(
        basic_market.get("change_24h_pct"),
        0.0,
    )
    quote_volume_24h = safe_float(
        basic_market.get("quote_volume_24h"),
        0.0,
    )
    side = str(forced_side or "").strip().lower()
    if side not in ("long", "short"):
        side = "neutral"
    lane_note = {
        "short_gainers": "24h 漲幅候選，等待 OpenAI 從原始資料重建方向與進出場。",
        "prebreakout_scanner": "pre-breakout 候選，僅提供原始資料給 OpenAI。",
        "pending_advice": "觀察單重審候選，等待 OpenAI 重新判斷。",
    }.get(str(candidate_source or "").strip().lower(), "一般候選，僅提供原始資料給 OpenAI。")
    breakdown = {
        "Setup": "data_only_candidate",
        "Regime": "unknown",
    }
    return {
        "symbol": symbol,
        "direction": side,
        "side": side,
        "score": round(change_24h_pct, 2),
        "raw_score": round(change_24h_pct, 2),
        "priority_score": round(quote_volume_24h, 4),
        "direction_confidence": 0.0,
        "trend_confidence": 0.0,
        "entry_quality": 0.0,
        "signal_grade": "",
        "setup_label": breakdown["Setup"],
        "price": current_price,
        "stop_loss": 0.0,
        "take_profit": 0.0,
        "rr_ratio": 0.0,
        "margin_pct": round(clamp((fixed_order_notional_usdt(symbol, max(current_price, 1e-9)) / max(_get_symbol_max_leverage(symbol), 1)) / max(safe_float(STATE.get("equity"), 100.0), 100.0), 0.01, 0.25), 4),
        "est_pnl": 0.0,
        "breakdown": breakdown,
        "desc": lane_note,
        "trend_mode": "data_only",
        "hold_reason": "normal_manage",
        "trend_note": "Local bot is data-only. Direction and trade logic are OpenAI-only.",
        "candidate_source": candidate_source,
        "scanner_intent": "",
        "btc_drive_aligned": False,
        "btc_ret_12bars_pct": 0.0,
        "symbol_ret_12bars_pct": safe_float(tf15.get("ret_12bars_pct"), 0.0),
        "atr": safe_float(tf15.get("atr"), 0.0),
        "atr15": safe_float(tf15.get("atr"), 0.0),
        "reference_trade_plan": {},
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
        "quote_volume_24h": round(
            safe_float(
                ticker.get("quoteVolume"),
                safe_float(ticker.get("baseVolume"), 0.0) * max(safe_float(ticker.get("last"), 0.0), 0.0),
            ),
            4,
        ),
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
    tf5m = dict(multi_timeframe.get("5m") or {})
    tf1h = dict(multi_timeframe.get("1h") or {})
    tf4h = dict(multi_timeframe.get("4h") or {})
    price_now = safe_float(basic_market.get("current_price"), safe_float(ticker.get("last"), 0.0))
    atr_15m = safe_float(tf15.get("atr"), 0.0)
    recent_support = safe_float(tf15.get("recent_structure_low"), 0.0)
    recent_resistance = safe_float(tf15.get("recent_structure_high"), 0.0)
    calculated_metrics = {
        "atr_15m": round(atr_15m, 6),
        "atr_5m": round(safe_float(tf5m.get("atr"), 0.0), 6),
        "atr_1h": round(safe_float(tf1h.get("atr"), 0.0), 6),
        "distance_to_support_atr15": round((price_now - recent_support) / max(atr_15m, 1e-9), 4) if recent_support > 0 and atr_15m > 0 else 0.0,
        "distance_to_resistance_atr15": round((recent_resistance - price_now) / max(atr_15m, 1e-9), 4) if recent_resistance > 0 and atr_15m > 0 else 0.0,
        "entry_spread_cost_pct": round(safe_float(liquidity.get("spread_pct"), 0.0), 6),
        "recent_volume_vs_avg_15m": round(safe_float(tf15.get("vol_ratio"), 0.0), 4),
        "recent_range_vs_atr_15m": round(safe_float(tf15.get("current_bar_range_atr"), 0.0), 4),
    }
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
        "machine_rr_hint": round(LOCAL_TP_MIN_R, 2),
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
        "raw_orderbook_snapshot": dict(liquidity.get("raw_orderbook_snapshot") or {"bids": [], "asks": []}),
        "recent_trades": dict(liquidity.get("recent_trades") or {}),
        "calculated_metrics": calculated_metrics,
        "derivatives_context": derivatives,
        "news_context": {"available": False, "note": "News feed is currently unavailable in this deployment.", "items": []},
        "multi_timeframe": multi_timeframe,
        "timeframe_bars": timeframe_bars,
        "pre_breakout_radar": {"ready": len(aligned) >= 2, "phase": "watch", "direction": bias, "score": round(safe_float(tf15.get("adx"), 0.0), 2), "summary": "Multi-timeframe scan built from live candles, liquidity, and derivatives.", "note": "Live market structure monitor."},
        "execution_context": {"spread_pct": liquidity.get("spread_pct", 0.0), "top_depth_ratio": liquidity.get("depth_imbalance_10", 0.0), "api_error_streak": total_error_count, "status": "ok" if total_error_count == 0 else "degraded", "notes": errors + list(liquidity.get("errors", [])) + list(derivatives.get("errors", []))},
        "execution_policy": {
            "fixed_leverage": int(max(_get_symbol_max_leverage(symbol), 1)),
            "leverage_mode": "cross_max",
            "min_order_margin_usdt": round(fixed_order_notional_usdt(symbol, basic_market["current_price"]) / max(_get_symbol_max_leverage(symbol), 1), 4),
            "fixed_order_notional_usdt": fixed_order_notional_usdt(symbol, basic_market["current_price"]),
            "margin_pct_range": [0.01, 0.20],
        },
        "multi_timeframe_pressure_summary": pressure_summary,
        "multi_timeframe_pressure": pressure_map,
        "reference_trade_plan": reference_trade_plan,
        "reference_context": {"summary": "Live scan built from Bitget market, multi-timeframe candles, liquidity, and derivatives."},
    }


def _ai_required_min_bars(timeframe: str) -> int:
    if timeframe in ("15m", "1h"):
        return 16
    if timeframe == "4h":
        return 10
    if timeframe == "1d":
        return 6
    return 8


def market_context_ready_for_ai(context: Dict[str, Any]) -> tuple[bool, str]:
    core_required = ("15m", "1h")
    higher_tf_optional = ("4h", "1d")
    multi = dict((context or {}).get("multi_timeframe") or {})
    bars_map = dict((context or {}).get("timeframe_bars") or {})
    missing_core: List[str] = []
    weak_core: List[str] = []
    for tf in core_required:
        tf_row = dict(multi.get(tf) or {})
        close_price = safe_float(tf_row.get("last_close"), 0.0)
        if close_price <= 0:
            missing_core.append("missing_last_close_{}".format(tf))
            continue
        bars = list(bars_map.get(tf) or [])
        min_bars = _ai_required_min_bars(tf)
        if len(bars) < min_bars:
            weak_core.append("insufficient_bars_{}_{}".format(tf, len(bars)))
    if missing_core:
        return False, ",".join(missing_core[:2])
    if weak_core:
        return False, ",".join(weak_core[:2])
    optional_missing: List[str] = []
    for tf in higher_tf_optional:
        tf_row = dict(multi.get(tf) or {})
        close_price = safe_float(tf_row.get("last_close"), 0.0)
        bars = list(bars_map.get(tf) or [])
        min_bars = _ai_required_min_bars(tf)
        if close_price <= 0 or len(bars) < min_bars:
            optional_missing.append(
                "{}:close_ok={} bars={}/{}".format(
                    tf,
                    "yes" if close_price > 0 else "no",
                    len(bars),
                    min_bars,
                )
            )
    if optional_missing:
        return True, "degraded_htf_data:{}".format("|".join(optional_missing[:2]))
    return True, "ok"


def repair_market_context_for_ai(symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
    repaired = dict(context or {})
    if not symbol:
        return repaired
    multi = dict(repaired.get("multi_timeframe") or {})
    bars_map = dict(repaired.get("timeframe_bars") or {})
    for tf in ("15m", "1h", "4h", "1d"):
        tf_row = dict(multi.get(tf) or {})
        close_ok = safe_float(tf_row.get("last_close"), 0.0) > 0
        bars = list(bars_map.get(tf) or [])
        bars_ok = len(bars) >= _ai_required_min_bars(tf)
        if close_ok and bars_ok:
            continue
        try:
            df = safe_fetch_ohlcv_df(symbol, tf, TIMEFRAME_BAR_LIMIT)
            if df is not None and not df.empty:
                multi[tf] = build_timeframe_stats(df)
                bars_map[tf] = serialize_bars(df, TIMEFRAME_BAR_LIMIT)
        except Exception:
            continue
        tf_row = dict(multi.get(tf) or {})
        bars = list(bars_map.get(tf) or [])
        if safe_float(tf_row.get("last_close"), 0.0) <= 0 and bars:
            last_bar = list(bars[-1] or [])
            bar_close = safe_float(last_bar[3] if len(last_bar) > 3 else 0.0, 0.0)
            if bar_close > 0:
                tf_row["last_close"] = bar_close
                multi[tf] = tf_row
    repaired["multi_timeframe"] = multi
    repaired["timeframe_bars"] = bars_map
    return repaired


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
        "openai_market_regime": str(decision.get("market_regime") or ""),
        "openai_regime_note": str(decision.get("regime_note") or ""),
        "openai_trend_state": str(decision.get("trend_state") or ""),
        "openai_timing_state": str(decision.get("timing_state") or ""),
        "openai_breakout_assessment": str(decision.get("breakout_assessment") or ""),
        "openai_trade_side": str(decision.get("trade_side") or ""),
        "openai_rr_ratio": safe_float(decision.get("rr_ratio"), 0.0),
        "openai_take_profit": safe_float(decision.get("take_profit"), 0.0),
        "openai_scale_in_recommended": bool(decision.get("scale_in_recommended", False)),
        "openai_scale_in_price": safe_float(decision.get("scale_in_price"), 0.0),
        "openai_scale_in_qty_pct": safe_float(decision.get("scale_in_qty_pct"), 0.0),
        "openai_scale_in_condition": str(decision.get("scale_in_condition") or ""),
        "openai_scale_in_note": str(decision.get("scale_in_note") or ""),
        "openai_order_type": decision.get("order_type", ""),
        "openai_bot_instruction": str(decision.get("bot_instruction") or ""),
        "openai_decision_valid": bool(decision.get("valid", True)),
        "openai_validation_errors": list(decision.get("validation_errors") or []),
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
        "openai_next_recheck_ts": safe_float(decision.get("next_recheck_ts"), 0.0),
        "openai_next_recheck_at": str(decision.get("next_recheck_at") or ""),
        "openai_watch_structure_condition": str(decision.get("watch_structure_condition") or ""),
        "openai_watch_volume_condition": str(decision.get("watch_volume_condition") or ""),
        "openai_watch_trigger_candle": str(decision.get("watch_trigger_candle") or "none"),
        "openai_watch_retest_rule": str(decision.get("watch_retest_rule") or "none"),
        "openai_watch_volume_ratio_min": safe_float(decision.get("watch_volume_ratio_min"), 0.0),
        "openai_watch_micro_vwap_rule": str(decision.get("watch_micro_vwap_rule") or "none"),
        "openai_watch_micro_ema20_rule": str(decision.get("watch_micro_ema20_rule") or "none"),
        "openai_watch_confirmations": list(decision.get("watch_confirmations") or []),
        "openai_watch_invalidations": list(decision.get("watch_invalidations") or []),
        "will_order": action == "enter",
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
        pending = dict(STATE.get("fvg_orders") or {})
        equity = safe_float(STATE.get("equity"), 0.0)
    long_count = sum(1 for row in active if str(row.get("side") or "").lower() == "long")
    short_count = sum(1 for row in active if str(row.get("side") or "").lower() == "short")
    pending_long = sum(1 for row in pending.values() if _side_from_value(row.get("side")) == "long")
    pending_short = sum(1 for row in pending.values() if _side_from_value(row.get("side")) == "short")
    exposure = exposure_snapshot()
    return {
        "equity": equity,
        "active_position_count": len(active),
        "long_count": long_count,
        "short_count": short_count,
        "pending_order_count": len(pending),
        "long_total_count": long_count + pending_long,
        "short_total_count": short_count + pending_short,
        "same_direction_count": max(long_count + pending_long, short_count + pending_short),
        "open_symbols": [row.get("symbol") for row in active if row.get("symbol")],
        "position_symbols": [row.get("symbol") for row in active if row.get("symbol")],
        "pending_symbols": [str(sym) for sym in pending.keys()],
        "general_total_count": safe_int(exposure.get("total_general"), 0),
        "prebreakout_total_count": safe_int(exposure.get("total_prebreakout"), 0),
        "general_long_total_count": safe_int(exposure.get("long_total_general"), 0),
        "general_short_total_count": safe_int(exposure.get("short_total_general"), 0),
    }


def _fetch_standard_swap_equity() -> float:
    try:
        balance = exchange.fetch_balance({"type": "swap", "productType": BITGET_PRODUCT_TYPE, "marginCoin": BITGET_MARGIN_COIN})
        total = dict(balance.get("total") or {})
        return safe_float(total.get(BITGET_MARGIN_COIN), 0.0)
    except Exception:
        return 0.0


def _fetch_standard_swap_positions() -> List[Dict[str, Any]]:
    try:
        return [dict(row or {}) for row in list(exchange.fetch_positions(None, {"productType": BITGET_PRODUCT_TYPE, "marginCoin": BITGET_MARGIN_COIN}) or [])]
    except Exception:
        try:
            return [dict(row or {}) for row in list(exchange.fetch_positions() or [])]
        except Exception:
            return []


def _append_ccxt_positions(positions_payload: List[Dict[str, Any]], raw_positions: List[Dict[str, Any]]) -> None:
    for row in list(raw_positions or []):
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
                "intrabarHigh": mark,
                "intrabarLow": mark,
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


def sync_positions_once() -> None:
    set_backend_thread("positions", "running", "Syncing balance and active positions.")
    try:
        equity = 0.0
        positions_payload: List[Dict[str, Any]] = []
        if BITGET_USE_COPY_TRADER_API:
            tracks = _bitget_copy_fetch_current_tracks()
            for row in tracks:
                size = abs(safe_float(row.get("openSize"), 0.0))
                if size <= 0:
                    continue
                pos_side = str(row.get("holdSide") or row.get("posSide") or "").lower()
                side = "long" if pos_side == "long" else "short" if pos_side == "short" else ""
                if side not in ("long", "short"):
                    continue
                symbol_id = str(row.get("symbol") or "")
                symbol = contract_symbol_to_ccxt_symbol(symbol_id)
                entry = safe_float(row.get("openPriceAvg"), 0.0)
                mark = entry
                try:
                    ticker = exchange.fetch_ticker(symbol) if symbol else {}
                    mark = safe_float((ticker or {}).get("last"), entry)
                except Exception:
                    mark = entry
                leverage = safe_float(row.get("openLeverage"), 1.0)
                unrealized = 0.0
                if entry > 0 and mark > 0 and size > 0:
                    unrealized = (mark - entry) * size if side == "long" else (entry - mark) * size
                pct = 0.0
                if entry > 0:
                    pct = ((mark - entry) / entry) * 100 if side == "long" else ((entry - mark) / entry) * 100
                positions_payload.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "entryPrice": entry,
                        "markPrice": mark,
                        "intrabarHigh": mark,
                        "intrabarLow": mark,
                        "contracts": size,
                        "unrealizedPnl": round(unrealized, 4),
                        "leverage": leverage,
                        "percentage": round(pct, 4),
                        "drawdown_pct": round(pct, 4),
                        "leveraged_pnl_pct": round(pct, 4),
                        "trend_mode": "learning",
                        "hold_reason": "normal_manage",
                        "trend_confidence": 0.0,
                        "tracking_no": str(row.get("trackingNo") or ""),
                        "open_order_id": str(row.get("openOrderId") or ""),
                    }
                )
            bal = get_balance()
            equity = max(
                safe_float(bal.get("equity"), 0.0),
                safe_float(bal.get("available"), 0.0) + safe_float(bal.get("margin"), 0.0),
            )
        else:
            equity = _fetch_standard_swap_equity()
            if equity <= 0:
                equity = safe_float(STATE.get("equity"), 0.0)
            _append_ccxt_positions(positions_payload, _fetch_standard_swap_positions())
        if equity <= 0:
            equity = safe_float(STATE.get("equity"), 0.0)
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


def _fetch_live_position_snapshots(symbols: List[str]) -> Dict[str, Dict[str, float]]:
    snapshots: Dict[str, Dict[str, float]] = {}
    base_prices = _fetch_live_position_prices(symbols)
    for symbol in [str(sym or "") for sym in symbols if str(sym or "")]:
        mark_price = safe_float(base_prices.get(symbol), 0.0)
        intrabar_high = mark_price
        intrabar_low = mark_price
        try:
            df = safe_fetch_ohlcv_df(symbol, "1m", 3)
            if not df.empty:
                latest = df.iloc[-1]
                candle_high = safe_float(latest.get("h"), 0.0)
                candle_low = safe_float(latest.get("l"), 0.0)
                candle_close = safe_float(latest.get("c"), 0.0)
                if mark_price <= 0 and candle_close > 0:
                    mark_price = candle_close
                if candle_high > 0:
                    intrabar_high = max(intrabar_high, candle_high)
                if candle_low > 0:
                    intrabar_low = candle_low if intrabar_low <= 0 else min(intrabar_low, candle_low)
        except Exception:
            pass
        if mark_price <= 0:
            continue
        if intrabar_high <= 0:
            intrabar_high = mark_price
        if intrabar_low <= 0:
            intrabar_low = mark_price
        snapshots[symbol] = {
            "markPrice": round(mark_price, 8),
            "intrabarHigh": round(max(intrabar_high, mark_price), 8),
            "intrabarLow": round(min(intrabar_low, mark_price), 8),
        }
    return snapshots


def sync_active_position_prices_once() -> bool:
    with STATE_LOCK:
        active_positions = [dict(row or {}) for row in list(STATE.get("active_positions") or []) if row.get("symbol")]
        equity = safe_float(STATE.get("equity"), 0.0)
    if not active_positions:
        return False
    live_snapshots = _fetch_live_position_snapshots([str(row.get("symbol") or "") for row in active_positions])
    if not live_snapshots:
        # Fallback: keep risk rules running on the latest known marks if live fetch temporarily fails.
        process_position_rules(active_positions)
        set_backend_thread("positions", "running", "Live price fetch degraded; managing positions with last known prices.")
        return False
    changed = False
    updated_positions: List[Dict[str, Any]] = []
    for row in active_positions:
        symbol = str(row.get("symbol") or "")
        entry = safe_float(row.get("entryPrice"), 0.0)
        leverage = safe_float(row.get("leverage"), 1.0)
        snapshot = dict(live_snapshots.get(symbol) or {})
        mark = safe_float(snapshot.get("markPrice"), safe_float(row.get("markPrice"), 0.0))
        if mark <= 0:
            updated_positions.append(row)
            continue
        intrabar_high = max(safe_float(snapshot.get("intrabarHigh"), mark), mark)
        intrabar_low = min(safe_float(snapshot.get("intrabarLow"), mark), mark) if safe_float(snapshot.get("intrabarLow"), 0.0) > 0 else mark
        raw_move_pct = 0.0
        if entry > 0:
            if str(row.get("side") or "").lower() == "long":
                raw_move_pct = ((mark - entry) / entry) * 100
            else:
                raw_move_pct = ((entry - mark) / entry) * 100
        next_row = dict(row)
        changed = changed or abs(mark - safe_float(row.get("markPrice"), 0.0)) > 1e-12
        changed = changed or abs(intrabar_high - safe_float(row.get("intrabarHigh"), safe_float(row.get("markPrice"), 0.0))) > 1e-12
        changed = changed or abs(intrabar_low - safe_float(row.get("intrabarLow"), safe_float(row.get("markPrice"), 0.0))) > 1e-12
        next_row["markPrice"] = round(mark, 8)
        next_row["intrabarHigh"] = round(intrabar_high, 8)
        next_row["intrabarLow"] = round(intrabar_low, 8)
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
    used_behaviors = set()

    def _behavior_bucket(row: Dict[str, Any]) -> str:
        ctx = dict(row.get("openai_market_context") or {})
        multi = dict(ctx.get("multi_timeframe") or {})
        tf15 = dict(multi.get("15m") or {})
        tf1h = dict(multi.get("1h") or {})
        basic = dict(ctx.get("basic_market_data") or {})
        side = _side_from_value(row.get("side")) or "neutral"
        c24 = safe_float(basic.get("change_24h_pct"), 0.0)
        r15 = safe_float(tf15.get("ret_12bars_pct"), 0.0)
        trend15 = str(tf15.get("trend_label") or "na")
        trend1h = str(tf1h.get("trend_label") or "na")
        c24_bucket = "u" if c24 >= 8 else "n" if c24 > -8 else "d"
        r15_bucket = "u" if r15 >= 1.5 else "n" if r15 > -1.5 else "d"
        return "{}|{}|{}|{}|{}".format(side, trend15, trend1h, c24_bucket, r15_bucket)

    for row in ranked:
        if len(selected) >= limit:
            break
        asset = base_asset(row.get("symbol", ""))
        setup = "{}|{}|{}".format(row.get("candidate_source"), row.get("side"), row.get("setup_label"))
        behavior = _behavior_bucket(row)
        same_side_count = sum(1 for item in selected if item.get("side") == row.get("side"))
        if asset in used_assets:
            continue
        if setup in used_setups and len(ranked) > limit:
            continue
        if behavior in used_behaviors and len(ranked) > limit:
            continue
        if same_side_count >= MAX_SAME_DIRECTION_POSITIONS:
            continue
        selected.append(dict(row))
        used_assets.add(asset)
        used_setups.add(setup)
        used_behaviors.add(behavior)
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
        "side": "neutral",
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
        "scanner_intent": "",
        "openai_market_context": {},
        "market": {"symbol": symbol},
    }


def build_openai_constraints(
    symbol: str = "",
    reference_price: float = 0.0,
    *,
    candidate_source: str = "general",
) -> Dict[str, Any]:
    max_symbol_leverage = max(_get_symbol_max_leverage(symbol), 1) if symbol else max(int(OPENAI_TRADE_CONFIG.get("max_leverage", 25) or 25), 1)
    order_notional_usdt = fixed_order_notional_usdt(symbol, reference_price) if symbol else FIXED_ORDER_NOTIONAL_USDT
    source = str(candidate_source or "general").lower()
    strategy_lane = "neutral_bidirectional_execution"
    lane_max_positions = PREBREAKOUT_MAX_OPEN_POSITIONS if source == "prebreakout_scanner" else MAX_OPEN_POSITIONS
    lane_same_direction = PREBREAKOUT_MAX_OPEN_POSITIONS if source == "prebreakout_scanner" else MAX_SAME_DIRECTION_POSITIONS
    lane_min_rr = OPENAI_MIN_RR_FOR_ENTRY
    return {
        "fixed_leverage": int(max_symbol_leverage),
        "min_leverage": int(OPENAI_TRADE_CONFIG.get("min_leverage", 4) or 4),
        "max_leverage": int(max_symbol_leverage),
        "min_margin_pct": 0.01,
        "max_margin_pct": 0.20,
        "fixed_order_notional_usdt": order_notional_usdt,
        "min_order_margin_usdt": round(order_notional_usdt / max_symbol_leverage, 4),
        "trade_style": "short_term_intraday",
        "max_open_positions": lane_max_positions,
        "max_same_direction": lane_same_direction,
        "leverage_policy": "always_use_symbol_max",
        "min_rr_for_entry": round(lane_min_rr, 2),
        "execution_tp_mode": "ai_direct",
        "candidate_source": source,
        "strategy_lane": strategy_lane,
        "aggressive_style": "pnl_first_pre_breakout",
        "anti_chase_rule": "prefer_limit_on_stretched_price_with_fvg_or_pullback",
        "scanner_metadata_trust": "disabled_data_only_mode",
    }


def open_position_symbols() -> set[str]:
    with STATE_LOCK:
        return {str(row.get("symbol") or "") for row in list(STATE.get("active_positions") or []) if row.get("symbol")}


def open_position_symbol_keys() -> set[str]:
    keys: set[str] = set()
    for symbol in open_position_symbols():
        key = symbol_key(symbol)
        if key:
            keys.add(key)
    return keys


def pending_order_symbols() -> set[str]:
    with STATE_LOCK:
        return set((STATE.get("fvg_orders") or {}).keys())


def pending_order_symbol_keys() -> set[str]:
    keys: set[str] = set()
    for symbol in pending_order_symbols():
        key = symbol_key(symbol)
        if key:
            keys.add(key)
    return keys


def refresh_execution_position_snapshot() -> tuple[bool, str]:
    """Best-effort refresh to avoid stale local position state blocking execution."""
    try:
        sync_positions_once()
        return True, ""
    except Exception as exc:
        return False, str(exc)[:180]


def _side_from_value(raw_side: Any) -> str:
    side = str(raw_side or "").strip().lower()
    if side in ("buy", "long"):
        return "long"
    if side in ("sell", "short"):
        return "short"
    return ""


def is_prebreakout_source(source: Any) -> bool:
    return str(source or "").strip().lower() == "prebreakout_scanner"


def _signal_candidate_source(signal: Dict[str, Any]) -> str:
    return str(signal.get("candidate_source") or signal.get("source") or "general").strip().lower()


def _signal_lane(signal: Dict[str, Any]) -> str:
    return "prebreakout" if is_prebreakout_source(_signal_candidate_source(signal)) else "general"


def _strip_direction_hints(value: Any) -> Any:
    blocked_keys = {
        "side_hint",
        "prebreakout_side_hint",
        "scanner_side",
        "prebreakout_hint",
    }
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key or "").strip().lower()
            if key_text in blocked_keys or "side_hint" in key_text:
                continue
            cleaned[key] = _strip_direction_hints(item)
        return cleaned
    if isinstance(value, list):
        return [_strip_direction_hints(item) for item in value]
    return value


def _prioritize_openai_candidates_by_source(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    pending_rows = [dict(row) for row in rows if str(row.get("candidate_source") or "").strip().lower() == "pending_advice"]
    normal_rows = [dict(row) for row in rows if str(row.get("candidate_source") or "").strip().lower() != "pending_advice"]
    if not normal_rows:
        return pending_rows

    preferred_sources = [str(src or "").strip().lower() for src in OPENAI_SOURCE_ROTATION if str(src or "").strip()]
    buckets: Dict[str, List[Dict[str, Any]]] = {src: [] for src in preferred_sources}
    fallback: List[Dict[str, Any]] = []
    for row in normal_rows:
        src = str(row.get("candidate_source") or "general").strip().lower()
        if src in buckets:
            buckets[src].append(row)
        else:
            fallback.append(row)

    def _quote_volume_24h(signal: Dict[str, Any]) -> float:
        ctx = dict(signal.get("openai_market_context") or {})
        basic = dict(ctx.get("basic_market_data") or {})
        return safe_float(basic.get("quote_volume_24h"), 0.0)

    for source_key, source_rows in buckets.items():
        if source_key == "short_gainers":
            source_rows.sort(
                key=lambda row: (
                    -safe_float(
                        ((row.get("openai_market_context") or {}).get("basic_market_data") or {}).get("change_24h_pct"),
                        0.0,
                    ),
                    safe_int(row.get("rank"), 9999),
                    str(row.get("symbol") or ""),
                )
            )
        elif source_key in {"general", "prebreakout_scanner"}:
            source_rows.sort(
                key=lambda row: (
                    -_quote_volume_24h(row),
                    safe_int(row.get("rank"), 9999),
                    str(row.get("symbol") or ""),
                )
            )

    ordered: List[Dict[str, Any]] = []
    seen_symbols: set[str] = set()

    def _append_dedup(rows_to_add: List[Dict[str, Any]]) -> None:
        for item in rows_to_add:
            symbol = str(item.get("symbol") or "")
            if not symbol:
                continue
            if symbol in seen_symbols:
                continue
            seen_symbols.add(symbol)
            ordered.append(item)

    for src in preferred_sources:
        _append_dedup(buckets.get(src, []))
    _append_dedup(fallback)

    short_rows = [row for row in ordered if str(row.get("candidate_source") or "").strip().lower() == "short_gainers"]
    short_ready = any(bool(row.get("local_gate_ok")) for row in short_rows)
    if short_ready:
        # Keep short-gainer first, but still allow general/prebreakout candidates
        # so AI can capture early expansion setups instead of being locked to one source.
        non_short_rows = [
            row
            for row in ordered
            if str(row.get("candidate_source") or "").strip().lower() != "short_gainers"
        ]
        return short_rows + non_short_rows + pending_rows
    return ordered + pending_rows


def _prebreakout_snapshot_from_signal(signal: Dict[str, Any]) -> Dict[str, Any]:
    features = dict(signal.get("prebreakout_scanner_features") or {})
    liquidity = dict((signal.get("openai_market_context") or {}).get("liquidity_context") or {})
    flow = dict(signal.get("prebreakout_flow_context") or (signal.get("prebreakout_raw_candidate_payload") or {}).get("flow_context") or {})
    return {
        "ts": now_ts(),
        "symbol": str(signal.get("symbol") or ""),
        "price": safe_float(signal.get("price"), 0.0),
        "cvd_notional_1m": safe_float(flow.get("cvd_notional_1m"), 0.0),
        "depth_imbalance_10": safe_float(liquidity.get("depth_imbalance_10"), 0.0),
        "distance_to_breakout_atr15m": safe_float(features.get("distance_to_breakout_atr15m"), 0.0),
        "distance_to_breakdown_atr15m": safe_float(features.get("distance_to_breakdown_atr15m"), 0.0),
    }


def _prebreakout_material_change(prev_snapshot: Dict[str, Any], new_snapshot: Dict[str, Any]) -> tuple[bool, List[str]]:
    prev = dict(prev_snapshot or {})
    curr = dict(new_snapshot or {})
    if not prev:
        return True, ["snapshot_missing"]
    changes: List[str] = []
    prev_price = safe_float(prev.get("price"), 0.0)
    curr_price = safe_float(curr.get("price"), 0.0)
    if prev_price > 0 and curr_price > 0:
        move_pct = abs(((curr_price / prev_price) - 1.0) * 100.0)
        if move_pct >= PREBREAKOUT_PRICE_CHANGE_PCT:
            changes.append("price_move_pct={:.3f}".format(move_pct))
    cvd_delta = abs(safe_float(curr.get("cvd_notional_1m"), 0.0) - safe_float(prev.get("cvd_notional_1m"), 0.0))
    if cvd_delta >= PREBREAKOUT_CVD_CHANGE_USDT:
        changes.append("cvd_delta={:.1f}".format(cvd_delta))
    depth_delta = abs(safe_float(curr.get("depth_imbalance_10"), 0.0) - safe_float(prev.get("depth_imbalance_10"), 0.0))
    if depth_delta >= PREBREAKOUT_DEPTH_IMBALANCE_CHANGE:
        changes.append("depth_imbalance_delta={:.4f}".format(depth_delta))
    dist_breakout_delta = abs(
        safe_float(curr.get("distance_to_breakout_atr15m"), 0.0) - safe_float(prev.get("distance_to_breakout_atr15m"), 0.0)
    )
    dist_breakdown_delta = abs(
        safe_float(curr.get("distance_to_breakdown_atr15m"), 0.0) - safe_float(prev.get("distance_to_breakdown_atr15m"), 0.0)
    )
    if max(dist_breakout_delta, dist_breakdown_delta) >= PREBREAKOUT_TRIGGER_DIST_CHANGE_ATR:
        changes.append("trigger_distance_shift")
    return (len(changes) > 0), changes


def _is_btc_beta_symbol(symbol: str) -> bool:
    asset = base_asset(symbol)
    return bool(asset and asset != "BTC" and asset in BTC_BETA_SYMBOLS)


def _is_effective_pending_order(row: Dict[str, Any], *, now_ts_value: float | None = None) -> bool:
    now_value = now_ts() if now_ts_value is None else float(now_ts_value)
    order_id = str((row or {}).get("order_id") or "").strip()
    if not order_id:
        return False
    expires_ts = safe_float((row or {}).get("expires_ts"), 0.0)
    if expires_ts > 0 and now_value >= expires_ts:
        return False
    return True


def exposure_snapshot() -> Dict[str, Any]:
    with STATE_LOCK:
        active_positions = list(STATE.get("active_positions") or [])
        pending_orders_raw = dict(STATE.get("fvg_orders") or {})
        position_rules = dict(POSITION_RULES or {})
    now_value = now_ts()
    pending_orders = {
        str(symbol or ""): dict(row or {})
        for symbol, row in pending_orders_raw.items()
        if _is_effective_pending_order(dict(row or {}), now_ts_value=now_value)
    }
    long_positions = [row for row in active_positions if _side_from_value(row.get("side")) == "long"]
    short_positions = [row for row in active_positions if _side_from_value(row.get("side")) == "short"]
    long_pending = [row for row in pending_orders.values() if _side_from_value(row.get("side")) == "long"]
    short_pending = [row for row in pending_orders.values() if _side_from_value(row.get("side")) == "short"]
    symbol_source: Dict[str, str] = {}
    for symbol, row in position_rules.items():
        symbol_source[str(symbol or "")] = str((row or {}).get("candidate_source") or "").strip().lower()
    for symbol, row in pending_orders.items():
        src = str((row or {}).get("candidate_source") or "").strip().lower()
        if src:
            symbol_source[str(symbol or "")] = src
    active_prebreakout = [
        row for row in active_positions
        if is_prebreakout_source(symbol_source.get(str(row.get("symbol") or ""), ""))
    ]
    pending_prebreakout = [
        row for row in pending_orders.values()
        if is_prebreakout_source((row or {}).get("candidate_source"))
    ]
    long_general_total = 0
    short_general_total = 0
    for row in active_positions:
        symbol = str(row.get("symbol") or "")
        if is_prebreakout_source(symbol_source.get(symbol, "")):
            continue
        side = _side_from_value(row.get("side"))
        if side == "long":
            long_general_total += 1
        elif side == "short":
            short_general_total += 1
    prebreakout_total = len(active_prebreakout)
    total_all = len(active_positions)
    total_with_pending = len(active_positions) + len(pending_orders)
    total_general = max(total_all - prebreakout_total, 0)
    return {
        "active_positions": active_positions,
        "pending_orders": pending_orders,
        "pending_orders_raw": pending_orders_raw,
        "long_positions": len(long_positions),
        "short_positions": len(short_positions),
        "long_pending": len(long_pending),
        "short_pending": len(short_pending),
        "long_total": len(long_positions),
        "short_total": len(short_positions),
        "total": total_all,
        "total_all": total_all,
        "total_with_pending": total_with_pending,
        "total_general": total_general,
        "total_prebreakout": prebreakout_total,
        "long_total_general": long_general_total,
        "short_total_general": short_general_total,
        "long_symbols": {str(row.get("symbol") or "") for row in long_positions if row.get("symbol")},
        "short_symbols": {str(row.get("symbol") or "") for row in short_positions if row.get("symbol")},
        "pending_long_symbols": {str(row.get("symbol") or "") for row in long_pending if row.get("symbol")},
        "pending_short_symbols": {str(row.get("symbol") or "") for row in short_pending if row.get("symbol")},
        "prebreakout_symbols": {
            str(row.get("symbol") or "")
            for row in active_prebreakout
            if row.get("symbol")
        } | {
            str((row or {}).get("symbol") or "")
            for row in pending_prebreakout
            if (row or {}).get("symbol")
        },
        "symbol_source_map": symbol_source,
    }


def _mark_capacity_block(signal: Dict[str, Any], reason: str, ttl_sec: int = ORDER_BLOCK_EXPIRY_SEC) -> None:
    symbol = str(signal.get("symbol") or "")
    if not symbol:
        return
    now = now_ts()
    expires_ts = now + max(300, int(ttl_sec))
    with REVIEW_LOCK:
        existing = dict(REVIEW_TRACKER.get(symbol) or {})
        REVIEW_TRACKER[symbol] = {
            "status": "blocked",
            "side": _side_from_value(signal.get("side")) or str(existing.get("side") or ""),
            "candidate_source": str(signal.get("candidate_source") or existing.get("candidate_source") or ""),
            "updated_at": tw_now_str(),
            "tracking_status": "capacity_blocked",
            "tracking_reason": "{} until {}".format(str(reason or "capacity_blocked")[:80], tw_from_ts(expires_ts)),
            "trigger_ready": False,
            "current_price": safe_float(signal.get("price"), 0.0),
            "next_allowed_ts": expires_ts,
            "created_ts": safe_float(existing.get("created_ts"), now) or now,
            "decision": dict(signal.get("openai_trade_plan") or existing.get("decision") or {}),
        }


def observe_expire_ts(row: Dict[str, Any], default_now: float | None = None) -> float:
    now = default_now if default_now is not None else now_ts()
    created_ts = safe_float((row or {}).get("created_ts"), now)
    expires_ts = safe_float((row or {}).get("observe_expires_ts"), 0.0)
    if expires_ts <= 0:
        expires_ts = created_ts + OBSERVE_MANUAL_WINDOW_SEC
    return expires_ts


def prune_expired_observe_trackers() -> List[str]:
    now = now_ts()
    removed: List[str] = []
    with REVIEW_LOCK:
        for symbol, row in list(REVIEW_TRACKER.items()):
            status = str((row or {}).get("status") or "")
            if status == "observe":
                expires_ts = observe_expire_ts(dict(row or {}), now)
                if now >= expires_ts:
                    del REVIEW_TRACKER[symbol]
                    removed.append(symbol)
                continue
            if status == "blocked":
                next_allowed_ts = safe_float((row or {}).get("next_allowed_ts"), 0.0)
                if next_allowed_ts > 0 and now >= next_allowed_ts:
                    del REVIEW_TRACKER[symbol]
                    removed.append(symbol)
    return removed


def watched_observe_symbols() -> set[str]:
    removed = prune_expired_observe_trackers()
    if removed:
        persist_runtime_snapshot_throttled(force=True)
    with REVIEW_LOCK:
        return {
            str(symbol or "")
            for symbol, row in REVIEW_TRACKER.items()
            if str((row or {}).get("status") or "") == "observe" and str(symbol or "")
        }


def temporarily_hidden_symbols(now: float | None = None) -> set[str]:
    now_ts_value = safe_float(now, now_ts())
    hidden: set[str] = set()
    cooldown_sec = max(int(float(OPENAI_TRADE_CONFIG.get("cooldown_minutes", 60) or 60) * 60), 60)
    state_symbols = dict((OPENAI_TRADE_STATE or {}).get("symbols") or {})
    for symbol, row in state_symbols.items():
        sym = str(symbol or "")
        if not sym:
            continue
        last_sent_ts = safe_float((row or {}).get("last_sent_ts"), 0.0)
        if last_sent_ts <= 0:
            continue
        if (now_ts_value - last_sent_ts) < cooldown_sec:
            hidden.add(sym)
    with REVIEW_LOCK:
        for symbol, row in REVIEW_TRACKER.items():
            sym = str(symbol or "")
            if not sym:
                continue
            status = str((row or {}).get("status") or "").strip().lower()
            next_allowed_ts = safe_float((row or {}).get("next_allowed_ts"), 0.0)
            if status in ("skip", "blocked") and next_allowed_ts > now_ts_value:
                hidden.add(sym)
                continue
            if status == "observe":
                expires_ts = observe_expire_ts(dict(row or {}), now_ts_value)
                manual_requested_ts = safe_float((row or {}).get("manual_recheck_requested_ts"), 0.0)
                last_manual_sent_ts = safe_float((row or {}).get("last_manual_recheck_sent_ts"), 0.0)
                if expires_ts > now_ts_value and manual_requested_ts <= last_manual_sent_ts:
                    hidden.add(sym)
                    continue
    return hidden


def review_action(decision: Dict[str, Any]) -> str:
    decision = dict(decision or {})
    instruction = str(decision.get("bot_instruction") or "").strip().upper()
    if instruction in ("ENTER_MARKET", "ENTER_LIMIT"):
        return "enter"
    if instruction == "SKIP":
        return "skip"
    if bool(decision.get("should_trade", False)) and _side_from_value(decision.get("trade_side")) in ("long", "short"):
        return "enter"
    return "skip"


def _reason_to_zh(reason: Any) -> str:
    text = str(reason or "").strip()
    if not text:
        return ""
    mapping = {
        "no_ranked_candidates": "本輪沒有可送審候選",
        "invalid_decision": "AI 回傳格式不合法",
        "invalid_decision_schema": "AI 回傳欄位不符合本地 schema",
        "instruction_not_enter": "AI 指令不是 ENTER",
        "missing_trade_side": "AI 缺少明確方向",
        "missing_entry_or_stop": "AI ENTER 缺少進場或止損",
        "stale_decision": "AI 指令已過期",
        "already_active": "此幣已有倉位或掛單",
        "max_positions": "已達一般倉位上限",
        "max_positions_prebreakout": "已達爆發系統倉位上限",
        "max_same_direction": "同方向倉位已滿",
        "qty_zero": "下單數量為 0",
        "leverage_failed": "槓桿設定失敗",
        "observe_manual_only": "觀察單期間僅手動重送",
        "observe_expired": "觀察單已過期，等待下一輪",
        "recheck_cooldown": "重審冷卻中",
        "skip_cooldown": "同幣冷卻中",
        "capacity_blocked": "倉位容量暫時阻擋",
        "same_direction_full": "同方向容量暫時阻擋",
        "prebreakout_observe_waiting_change": "爆發候選等待明顯變化後再送審",
        "prebreakout_full": "爆發系統倉位已滿",
        "empty_decision": "AI 回傳為空",
        "bad_request": "OpenAI 請求格式錯誤",
        "budget_paused": "OpenAI 預算閘門暫停",
        "cooldown_active": "同幣 OpenAI 冷卻中",
        "global_interval_active": "全域送審間隔冷卻中",
        "cached_reuse": "沿用同 payload 既有決策",
        "below_min_score": "低於候選送審門檻",
        "not_ranked": "不在本輪送審順位",
        "empty_response_no_action": "AI 空回覆，本輪不執行",
        "empty_response_fallback_skip": "AI 空回覆，已回退為 SKIP",
        "empty_response_reuse_cached": "AI 空回覆，沿用舊決策",
        "non_executable_instruction": "AI 指令不可執行",
    }
    if text in mapping:
        return mapping[text]
    if text.startswith("non_executable_instruction:"):
        sub = text.split(":", 1)[1].strip()
        return "AI 指令不可執行（{}）".format(mapping.get(sub, sub))
    if text.startswith("skip_cooldown_until="):
        return "同幣冷卻中（{}）".format(text.split("=", 1)[1])
    if text.startswith("manual_only_until="):
        return "觀察單僅手動重送至 {}".format(text.split("=", 1)[1])
    if text.startswith("missing_timeframe:"):
        return "缺少必要 K 線資料（{}）".format(text.split(":", 1)[1])
    if text.startswith("missing_market_context:"):
        return "市場資料不完整（{}）".format(text.split(":", 1)[1])
    if text.startswith("context_not_ready:"):
        return "候選資料尚未準備完成（{}）".format(text.split(":", 1)[1])
    if text.startswith("bad_request:"):
        return "OpenAI 請求格式錯誤（{}）".format(text.split(":", 1)[1])
    if "_until=" in text:
        base, ts = text.split("_until=", 1)
        return "{}（到 {}）".format(mapping.get(base, base), ts)
    return text


def _humanize_blockers_zh(blockers: List[str]) -> List[str]:
    out: List[str] = []
    for raw in list(blockers or [])[:24]:
        text = str(raw or "").strip()
        if not text:
            continue
        if ":" in text:
            sym, reason = text.split(":", 1)
            out.append("{}:{}".format(sym, _reason_to_zh(reason)))
        else:
            out.append(_reason_to_zh(text))
    return out[:10]


def enter_decision_ready(decision: Dict[str, Any]) -> tuple[bool, str]:
    plan = dict(decision or {})
    if not bool(plan.get("valid", True)):
        return False, "invalid_decision_schema"
    instruction = str(plan.get("bot_instruction") or "").strip().upper()
    if instruction not in ("ENTER_MARKET", "ENTER_LIMIT"):
        return False, "instruction_not_enter"
    decision_side = _side_from_value(plan.get("trade_side"))
    if decision_side not in ("long", "short"):
        return False, "missing_trade_side"
    entry_price = safe_float(plan.get("entry_price"), 0.0)
    stop_loss = safe_float(plan.get("stop_loss"), 0.0)
    if entry_price <= 0 or stop_loss <= 0:
        return False, "missing_entry_or_stop"
    rr_ratio = safe_float(plan.get("rr_ratio"), 0.0)
    if rr_ratio < OPENAI_MIN_RR_FOR_ENTRY:
        return False, "rr_below_min:{:.2f}<{:.2f}".format(rr_ratio, OPENAI_MIN_RR_FOR_ENTRY)
    return True, "ok"


def update_watchlist_state() -> None:
    prune_expired_observe_trackers()
    watchlist = []
    now = now_ts()
    with REVIEW_LOCK:
        for symbol, row in REVIEW_TRACKER.items():
            if str(row.get("status") or "") != "observe":
                continue
            expires_ts = observe_expire_ts(row, now)
            watchlist.append(
                {
                    "symbol": symbol,
                    "status": "observe",
                    "side": row.get("side"),
                    "watch_trigger_type": row.get("watch_trigger_type"),
                    "watch_trigger_price": row.get("watch_trigger_price"),
                    "watch_invalidation_price": row.get("watch_invalidation_price"),
                    "recheck_reason": row.get("recheck_reason"),
                    "next_recheck_at": tw_from_ts(row.get("next_recheck_ts")),
                    "last_checked_at": tw_from_ts(row.get("last_checked_ts")),
                    "candidate_source": row.get("candidate_source"),
                    "updated_at": row.get("updated_at"),
                    "observe_expires_at": tw_from_ts(expires_ts),
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
    symbol_norm = symbol_key(symbol)
    if symbol_norm in open_position_symbol_keys() or symbol_norm in pending_order_symbol_keys():
        return {"ok": False, "reason": "already_active", "force_recheck": False}
    signal_source = str(signal.get("candidate_source") or "general").strip().lower()
    with REVIEW_LOCK:
        tracker = dict(REVIEW_TRACKER.get(symbol) or {})
    tracker_source = str(tracker.get("candidate_source") or "").strip().lower()
    if tracker and tracker_source and signal_source and tracker_source != signal_source:
        return {"ok": True, "reason": "fresh_other_source", "force_recheck": False}
    if not tracker:
        return {"ok": True, "reason": "fresh", "force_recheck": False}
    status = str(tracker.get("status") or "")
    now = now_ts()
    if status == "blocked":
        block_until = safe_float(tracker.get("next_allowed_ts"), 0.0)
        if block_until > now:
            return {
                "ok": False,
                "reason": "capacity_blocked",
                "force_recheck": False,
                "next_allowed_ts": block_until,
            }
        with REVIEW_LOCK:
            if symbol in REVIEW_TRACKER:
                del REVIEW_TRACKER[symbol]
        return {"ok": True, "reason": "capacity_block_expired", "force_recheck": False}
    if status == "skip" and now < safe_float(tracker.get("next_allowed_ts"), 0.0):
        return {
            "ok": False,
            "reason": "skip_cooldown",
            "force_recheck": False,
            "next_allowed_ts": safe_float(tracker.get("next_allowed_ts"), 0.0),
        }
    if status == "observe":
        expires_ts = observe_expire_ts(tracker, now)
        if now >= expires_ts:
            with REVIEW_LOCK:
                if symbol in REVIEW_TRACKER:
                    del REVIEW_TRACKER[symbol]
            return {
                "ok": False,
                "reason": "observe_expired",
                "force_recheck": False,
                "observe_expires_ts": expires_ts,
                "next_recheck_ts": expires_ts,
            }
        tracker_source = str(tracker.get("candidate_source") or signal.get("candidate_source") or "").strip().lower()
        manual_requested_ts = safe_float(tracker.get("manual_recheck_requested_ts"), 0.0)
        last_manual_sent_ts = safe_float(tracker.get("last_manual_recheck_sent_ts"), 0.0)
        allow_recheck = manual_requested_ts > last_manual_sent_ts
        stage = "manual_recheck_requested" if allow_recheck else "manual_waiting"
        reason_text = "manual_send_requested_at={}".format(tw_from_ts(manual_requested_ts)) if allow_recheck else ""
        gate_reason = "manual_recheck_requested" if allow_recheck else "observe_manual_only"
        if (not allow_recheck) and is_prebreakout_source(tracker_source):
            previous_snapshot = dict(tracker.get("prebreakout_last_snapshot") or {})
            current_snapshot = _prebreakout_snapshot_from_signal(signal)
            changed, change_reasons = _prebreakout_material_change(previous_snapshot, current_snapshot)
            if changed:
                allow_recheck = True
                gate_reason = "prebreakout_material_change"
                stage = "prebreakout_change_detected"
                reason_text = "material_change={}".format(",".join(change_reasons[:4]))
            else:
                stage = "prebreakout_waiting_change"
                reason_text = "observe_waiting_material_change_until={}".format(tw_from_ts(expires_ts))
                gate_reason = "prebreakout_observe_waiting_change"
        with REVIEW_LOCK:
            if symbol in REVIEW_TRACKER:
                REVIEW_TRACKER[symbol]["current_price"] = safe_float(signal.get("price"), 0.0)
                REVIEW_TRACKER[symbol]["last_checked_ts"] = now
                REVIEW_TRACKER[symbol]["check_count"] = safe_int(REVIEW_TRACKER[symbol].get("check_count"), 0) + 1
                REVIEW_TRACKER[symbol]["updated_at"] = tw_now_str()
                REVIEW_TRACKER[symbol]["tracking_status"] = stage
                REVIEW_TRACKER[symbol]["tracking_reason"] = reason_text
                REVIEW_TRACKER[symbol]["trigger_ready"] = bool(allow_recheck)
                REVIEW_TRACKER[symbol]["observe_expires_ts"] = expires_ts
                if is_prebreakout_source(tracker_source):
                    REVIEW_TRACKER[symbol]["prebreakout_last_snapshot"] = _prebreakout_snapshot_from_signal(signal)
        return {
            "ok": allow_recheck,
            "reason": gate_reason,
            "force_recheck": allow_recheck,
            "observe_expires_ts": expires_ts,
            "next_recheck_ts": expires_ts,
        }
    return {"ok": True, "reason": "reopen", "force_recheck": False}


def apply_review_tracker(signal: Dict[str, Any], result: Dict[str, Any]) -> str:
    decision = dict(result.get("decision") or {})
    if not decision:
        return "none"
    action = review_action(decision)
    symbol = str(signal.get("symbol") or "")
    now = now_ts()
    with REVIEW_LOCK:
        existing = dict(REVIEW_TRACKER.get(symbol) or {})
        row = {
            "status": action,
            "side": _side_from_value(decision.get("trade_side")) or signal.get("side"),
            "candidate_source": signal.get("candidate_source"),
            "updated_at": tw_now_str(),
            "tracking_status": "manual_only_waiting" if action == "observe" else "inactive",
            "tracking_reason": str(existing.get("tracking_reason") or ""),
            "trigger_ready": bool(existing.get("trigger_ready", False)),
            "current_price": safe_float(signal.get("price"), 0.0),
            "watch_trigger_type": str(decision.get("watch_trigger_type") or "none"),
            "watch_trigger_price": safe_float(decision.get("watch_trigger_price"), 0.0),
            "watch_invalidation_price": safe_float(decision.get("watch_invalidation_price"), 0.0),
            "recheck_reason": str(decision.get("recheck_reason") or decision.get("watch_structure_condition") or decision.get("watch_note") or ""),
            "decision": decision,
            "next_allowed_ts": 0.0,
            "created_ts": safe_float(existing.get("created_ts"), now),
            "last_checked_ts": safe_float(existing.get("last_checked_ts"), now),
            "check_count": safe_int(existing.get("check_count"), 0),
            "last_recheck_sent_ts": 0.0,
            "next_recheck_ts": 0.0,
            "recheck_cooldown_sec": AI_OBSERVE_RECHECK_COOLDOWN_SEC,
            "manual_recheck_requested_ts": safe_float(existing.get("manual_recheck_requested_ts"), 0.0),
            "last_manual_recheck_sent_ts": safe_float(existing.get("last_manual_recheck_sent_ts"), 0.0),
            "observe_expires_ts": safe_float(existing.get("observe_expires_ts"), 0.0),
            "last_result_status": str(result.get("status") or ""),
        }
        if action == "skip":
            row["next_allowed_ts"] = now + AI_SKIP_COOLDOWN_SEC
            row["tracking_status"] = "skip_cooldown"
            row["tracking_reason"] = str(decision.get("reason_to_skip") or "")
            row["trigger_ready"] = False
            decision["next_recheck_ts"] = row["next_allowed_ts"]
            decision["next_recheck_at"] = tw_from_ts(row["next_allowed_ts"])
            row["decision"] = decision
        if action == "observe":
            prior_created_ts = safe_float(existing.get("created_ts"), 0.0)
            row["created_ts"] = prior_created_ts if prior_created_ts > 0 else now
            row["observe_expires_ts"] = now + OBSERVE_MANUAL_WINDOW_SEC
            if is_prebreakout_source(row.get("candidate_source")):
                row["tracking_status"] = "prebreakout_waiting_change"
                row["tracking_reason"] = "material_change_auto_recheck_until={}".format(tw_from_ts(row["observe_expires_ts"]))
                row["prebreakout_last_snapshot"] = _prebreakout_snapshot_from_signal(signal)
                row["auto_recheck_on_change"] = True
            else:
                row["tracking_status"] = "manual_only_waiting"
                row["tracking_reason"] = "manual_send_only_until={}".format(tw_from_ts(row["observe_expires_ts"]))
            row["trigger_ready"] = False
            row["manual_recheck_requested_ts"] = 0.0
            if not str(decision.get("bot_instruction") or "").strip():
                decision["bot_instruction"] = "OBSERVE"
            if not str(decision.get("recheck_reason") or "").strip():
                expire_minutes = max(1, int(round(OBSERVE_MANUAL_WINDOW_SEC / 60.0)))
                if is_prebreakout_source(row.get("candidate_source")):
                    decision["recheck_reason"] = "目前為 pre-breakout 觀察單，僅在價格/CVD/掛單或 trigger distance 明顯變化時才會自動重審；{} 分鐘內有效。".format(expire_minutes)
                else:
                    decision["recheck_reason"] = "目前為觀察單，僅支援手動發送重審；{} 分鐘內有效，逾時自動移除。".format(expire_minutes)
            decision["next_recheck_ts"] = row["observe_expires_ts"]
            decision["next_recheck_at"] = tw_from_ts(row["observe_expires_ts"])
            row["recheck_reason"] = str(decision.get("recheck_reason") or row.get("recheck_reason") or "")
            row["decision"] = decision
        REVIEW_TRACKER[symbol] = row
    result["decision"] = dict(decision)
    update_watchlist_state()
    sync_openai_pending_advice()
    persist_runtime_snapshot_throttled(force=True)
    return action

def sync_openai_pending_advice() -> None:
    global OPENAI_TRADE_STATE
    pending: Dict[str, Any] = {}
    expired_symbols: List[str] = []
    with REVIEW_LOCK:
        now = now_ts()
        for symbol, row in list(REVIEW_TRACKER.items()):
            if str(row.get("status") or "") != "observe":
                continue
            expires_ts = observe_expire_ts(row, now)
            if now >= expires_ts:
                del REVIEW_TRACKER[symbol]
                expired_symbols.append(symbol)
                continue
            decision = dict(row.get("decision") or {})
            created_ts = safe_float(row.get("created_ts"), now)
            last_checked_ts = safe_float(row.get("last_checked_ts"), 0.0)
            manual_requested_ts = safe_float(row.get("manual_recheck_requested_ts"), 0.0)
            last_manual_sent_ts = safe_float(row.get("last_manual_recheck_sent_ts"), 0.0)
            pending[symbol] = {
                "symbol": symbol,
                "side": row.get("side"),
                "status": "manual_observe",
                "bot_tracking_enabled": False,
                "tracking_status": str(row.get("tracking_status") or "manual_only_waiting"),
                "tracking_reason": str(row.get("tracking_reason") or ""),
                "trigger_ready": bool(row.get("trigger_ready", False)),
                "candidate_source": row.get("candidate_source"),
                "created_ts": created_ts,
                "created_at": tw_from_ts(created_ts),
                "last_checked_ts": last_checked_ts,
                "last_checked_at": tw_from_ts(last_checked_ts),
                "last_recheck_sent_ts": last_manual_sent_ts,
                "last_recheck_sent_at": tw_from_ts(last_manual_sent_ts),
                "next_recheck_ts": expires_ts,
                "next_recheck_at": tw_from_ts(expires_ts),
                "recheck_cooldown_sec": safe_int(row.get("recheck_cooldown_sec"), AI_OBSERVE_RECHECK_COOLDOWN_SEC),
                "check_count": safe_int(row.get("check_count"), 0),
                "current_price": safe_float(row.get("current_price"), 0.0),
                "watch_note": str(decision.get("watch_note") or ""),
                "entry_plan": str(decision.get("entry_plan") or ""),
                "bot_instruction": str(decision.get("bot_instruction") or "OBSERVE"),
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
                "recheck_reason": str(
                    decision.get("recheck_reason")
                    or row.get("recheck_reason")
                    or decision.get("watch_structure_condition")
                    or decision.get("watch_note")
                    or ""
                ),
                "observe_expires_ts": expires_ts,
                "observe_expires_at": tw_from_ts(expires_ts),
                "manual_recheck_requested_ts": manual_requested_ts,
                "manual_recheck_requested_at": tw_from_ts(manual_requested_ts),
                "manual_only": True,
            }
            trigger = safe_float(decision.get("watch_trigger_price"), 0.0)
            current = safe_float(row.get("current_price"), 0.0)
            if not str(pending[symbol].get("recheck_reason") or "").strip():
                expire_minutes = max(1, int(round(OBSERVE_MANUAL_WINDOW_SEC / 60.0)))
                pending[symbol]["recheck_reason"] = "目前觀察單僅支援手動重送 AI，{} 分鐘到期自動移除。".format(expire_minutes)
            pending[symbol]["distance_pct"] = round((((current / trigger) - 1.0) * 100.0), 4) if trigger > 0 and current > 0 else None
    if expired_symbols:
        update_watchlist_state()
        persist_runtime_snapshot_throttled(force=True)
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


def _attach_pending_recheck_context(signal: Dict[str, Any], tracker: Dict[str, Any], eval_result: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(signal or {})
    decision = dict((tracker or {}).get("decision") or {})
    trigger_reason = str((eval_result or {}).get("reason") or (tracker or {}).get("tracking_reason") or "觀察條件已成立")
    manual_reask = "manual" in trigger_reason.lower()
    summary_text = "使用者手動要求重送 AI，請以最新市場資料重新評估是否可進場。" if manual_reask else "機器人已確認先前 AI 要求的觀察條件成立，現在必須重新送 AI 確認是否可執行。"
    note_prefix = "手動重審請求" if manual_reask else "剛達到你先前要求觀察的條件"
    ref = dict(enriched.get("external_reference") or enriched.get("reference_context") or {})
    ref.update(
        {
            "source": "stored_openai_pending_advice",
            "summary": summary_text,
            "setup": str(decision.get("entry_plan") or decision.get("watch_note") or "")[:220],
            "risk": str(decision.get("reason_to_skip") or decision.get("market_read") or "")[:220],
            "note": (
                "{}：{} | trigger={} price={} invalidation={} timeframe={} recheck={}".format(
                    note_prefix,
                    trigger_reason,
                    safe_float(decision.get("watch_trigger_price"), 0.0),
                    safe_float(enriched.get("price"), 0.0),
                    safe_float(decision.get("watch_invalidation_price"), 0.0),
                    str(decision.get("watch_timeframe") or ""),
                    str(decision.get("recheck_reason") or decision.get("watch_note") or "")[:120],
                )
            )[:220],
            "checklist": " / ".join(list(decision.get("watch_checklist") or [])[:4]),
            "confirmations": " / ".join(list(decision.get("watch_confirmations") or [])[:4]),
            "invalidations": " / ".join(list(decision.get("watch_invalidations") or [])[:4]),
        }
    )
    enriched["external_reference"] = ref
    enriched["pending_openai_advice"] = {
        "watch_trigger_type": str(decision.get("watch_trigger_type") or "none"),
        "watch_trigger_price": safe_float(decision.get("watch_trigger_price"), 0.0),
        "watch_invalidation_price": safe_float(decision.get("watch_invalidation_price"), 0.0),
        "watch_timeframe": str(decision.get("watch_timeframe") or ""),
        "watch_note": str(decision.get("watch_note") or ""),
        "recheck_reason": str(
            decision.get("recheck_reason")
            or decision.get("watch_structure_condition")
            or decision.get("watch_note")
            or ""
        ),
        "watch_structure_condition": str(decision.get("watch_structure_condition") or ""),
        "watch_volume_condition": str(decision.get("watch_volume_condition") or ""),
        "watch_checklist": list(decision.get("watch_checklist") or []),
        "watch_confirmations": list(decision.get("watch_confirmations") or []),
        "watch_invalidations": list(decision.get("watch_invalidations") or []),
        "trigger_reason": trigger_reason,
    }
    enriched["desc"] = str(decision.get("watch_note") or decision.get("entry_plan") or enriched.get("desc") or "")
    return enriched


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
        "next_partial_basis": str((next_partial or {}).get("basis") or "initial"),
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
        {"r_multiple": 1.0, "fraction": 0.25, "basis": "initial", "done": False},
        {"r_multiple": 1.5, "fraction": 0.25, "basis": "initial", "done": False},
        {"r_multiple": 2.0, "fraction": 0.30, "basis": "initial", "done": False},
        {"r_multiple": 3.0, "fraction": 0.50, "basis": "remaining", "done": False},
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
                "basis": str(item.get("basis") or "initial").lower(),
                "done": bool(item.get("done", False)),
            }
        )
    return normalized


def _position_side_to_order_side(side: str) -> str:
    return "buy" if str(side or "").lower() == "long" else "sell"


def _local_rr_target_from_decision(decision: Dict[str, Any]) -> float:
    rr_hint = safe_float((decision or {}).get("rr_ratio"), 0.0)
    if rr_hint >= LOCAL_TP_MIN_R:
        return clamp(rr_hint, LOCAL_TP_MIN_R, LOCAL_TP_MAX_R)
    return LOCAL_TP_DEFAULT_R


def _local_take_profit_from_r(side: str, entry_price: float, stop_loss: float, decision: Dict[str, Any]) -> float:
    side_norm = str(side or "").lower()
    entry_price = safe_float(entry_price, 0.0)
    stop_loss = safe_float(stop_loss, 0.0)
    if entry_price <= 0 or stop_loss <= 0:
        return 0.0
    if side_norm == "long" and stop_loss >= entry_price:
        return 0.0
    if side_norm == "short" and stop_loss <= entry_price:
        return 0.0
    risk = abs(entry_price - stop_loss)
    if risk <= 0:
        return 0.0
    rr_target = _local_rr_target_from_decision(decision)
    tp_price = entry_price + (risk * rr_target) if side_norm == "long" else entry_price - (risk * rr_target)
    return round(tp_price, 8)


def _extract_tf_rows(context: Dict[str, Any], timeframe: str = "15m", *, limit: int = 48) -> List[List[float]]:
    rows_raw = ((dict(context or {}).get("timeframe_bars") or {}).get(timeframe) or [])
    if isinstance(rows_raw, dict):
        rows_raw = rows_raw.get("rows") or rows_raw.get("candles") or rows_raw.get("bars") or []
    normalized: List[List[float]] = []
    for row in list(rows_raw or [])[-limit:]:
        if isinstance(row, dict):
            o = safe_float(row.get("open"), 0.0)
            h = safe_float(row.get("high"), 0.0)
            l = safe_float(row.get("low"), 0.0)
            c = safe_float(row.get("close"), 0.0)
            v = safe_float(row.get("volume"), 0.0)
            if o > 0 and h > 0 and l > 0 and c > 0:
                normalized.append([o, h, l, c, max(v, 0.0)])
            continue
        if isinstance(row, (list, tuple)):
            seq = list(row)
            if len(seq) >= 6:
                ts_candidate = safe_float(seq[0], 0.0)
                if ts_candidate >= 1_000_000_000:
                    seq = seq[1:6]
                else:
                    seq = seq[0:5]
            if len(seq) >= 5:
                o = safe_float(seq[0], 0.0)
                h = safe_float(seq[1], 0.0)
                l = safe_float(seq[2], 0.0)
                c = safe_float(seq[3], 0.0)
                v = safe_float(seq[4], 0.0)
                if o > 0 and h > 0 and l > 0 and c > 0:
                    normalized.append([o, h, l, c, max(v, 0.0)])
    return normalized


def _local_market_entry_chase_risk(side: str, symbol: str, entry_price: float, context: Dict[str, Any]) -> tuple[bool, str]:
    side_norm = _side_from_value(side)
    entry_price = safe_float(entry_price, 0.0)
    if side_norm not in ("long", "short") or entry_price <= 0:
        return False, ""
    tf15 = dict((dict(context or {}).get("multi_timeframe") or {}).get("15m") or {})
    atr = max(safe_float(tf15.get("atr"), 0.0), entry_price * 0.003)
    ema20 = safe_float(tf15.get("ema20"), safe_float(tf15.get("last_close"), entry_price))
    vwap = safe_float(tf15.get("vwap"), safe_float(tf15.get("last_close"), entry_price))
    structure_high = max(
        safe_float(tf15.get("recent_structure_high"), 0.0),
        safe_float(tf15.get("prior_structure_high_6"), 0.0),
    )
    structure_low = max(
        safe_float(tf15.get("recent_structure_low"), 0.0),
        safe_float(tf15.get("prior_structure_low_6"), 0.0),
    )
    base = base_asset(symbol)
    is_major = base in HIGH_NOTIONAL_SYMBOLS or base in {"BTC", "ETH", "BNB", "SOL", "XRP"}
    stretch_mult = 1.05 if is_major else 1.35
    extreme_mult = 0.55 if is_major else 0.40
    if side_norm == "long":
        anchor = max(ema20, vwap)
        stretched = (entry_price - anchor) >= (atr * stretch_mult)
        near_extreme = structure_high > 0 and (structure_high - entry_price) <= (atr * extreme_mult)
        if stretched and near_extreme:
            return True, "long_market_chasing_extension"
    else:
        anchor = min(x for x in (ema20, vwap) if x > 0) if (ema20 > 0 or vwap > 0) else entry_price
        stretched = (anchor - entry_price) >= (atr * stretch_mult)
        near_extreme = structure_low > 0 and (entry_price - structure_low) <= (atr * extreme_mult)
        if stretched and near_extreme:
            return True, "short_market_chasing_dump"
    return False, ""


def _derive_pullback_limit_entry(side: str, entry_price: float, context: Dict[str, Any]) -> float:
    side_norm = _side_from_value(side)
    entry_price = safe_float(entry_price, 0.0)
    if side_norm not in ("long", "short") or entry_price <= 0:
        return 0.0
    rows = _extract_tf_rows(context, "15m", limit=48)
    if len(rows) >= 3:
        candidates: List[float] = []
        for idx in range(2, len(rows)):
            a = rows[idx - 2]
            c = rows[idx]
            a_high = safe_float(a[1], 0.0)
            a_low = safe_float(a[2], 0.0)
            c_high = safe_float(c[1], 0.0)
            c_low = safe_float(c[2], 0.0)
            if side_norm == "long" and c_low > a_high > 0:
                zone_low = a_high
                zone_high = c_low
                mid = (zone_low + zone_high) / 2.0
                if 0 < mid < entry_price:
                    candidates.append(mid)
            if side_norm == "short" and c_high < a_low and c_high > 0:
                zone_low = c_high
                zone_high = a_low
                mid = (zone_low + zone_high) / 2.0
                if mid > entry_price:
                    candidates.append(mid)
        if candidates:
            if side_norm == "long":
                return round(max(candidates), 8)
            return round(min(candidates), 8)
    tf15 = dict((dict(context or {}).get("multi_timeframe") or {}).get("15m") or {})
    atr = max(safe_float(tf15.get("atr"), 0.0), entry_price * 0.003)
    prior_high = max(safe_float(tf15.get("prior_structure_high_6"), 0.0), safe_float(tf15.get("recent_structure_high"), 0.0))
    prior_low = max(safe_float(tf15.get("prior_structure_low_6"), 0.0), safe_float(tf15.get("recent_structure_low"), 0.0))
    if side_norm == "long":
        fallback = prior_low if 0 < prior_low < entry_price else (entry_price - atr * 0.75)
        if fallback < entry_price * 0.999:
            return round(max(fallback, entry_price - atr * 1.4), 8)
    else:
        fallback = prior_high if prior_high > entry_price else (entry_price + atr * 0.75)
        if fallback > entry_price * 1.001:
            return round(min(fallback, entry_price + atr * 1.4), 8)
    return 0.0


def _levels_consistent_for_side(side: str, entry: float, stop_loss: float, take_profit: float) -> bool:
    side_norm = str(side or "").lower()
    entry = safe_float(entry, 0.0)
    stop_loss = safe_float(stop_loss, 0.0)
    take_profit = safe_float(take_profit, 0.0)
    if entry <= 0 or stop_loss <= 0 or take_profit <= 0:
        return False
    if side_norm == "long":
        return stop_loss < entry < take_profit
    if side_norm == "short":
        return take_profit < entry < stop_loss
    return False


def _execution_level_error(side: str, entry: float, stop_loss: float, take_profit: float) -> str:
    side_norm = str(side or "").lower()
    entry = safe_float(entry, 0.0)
    stop_loss = safe_float(stop_loss, 0.0)
    take_profit = safe_float(take_profit, 0.0)
    if side_norm not in ("long", "short"):
        return "invalid_side"
    if entry <= 0:
        return "entry_non_positive"
    if stop_loss <= 0:
        return "stop_loss_non_positive"
    if take_profit <= 0:
        return "take_profit_non_positive"
    if side_norm == "long":
        if stop_loss >= entry:
            return "long_stop_not_below_entry"
        if take_profit <= entry:
            return "long_take_profit_not_above_entry"
    else:
        if stop_loss <= entry:
            return "short_stop_not_above_entry"
        if take_profit >= entry:
            return "short_take_profit_not_below_entry"
    return ""


def _enforce_structure_stop_guard(side: str, entry_price: float, stop_loss: float, context: Dict[str, Any]) -> tuple[float, str]:
    side_norm = str(side or "").lower()
    entry_price = safe_float(entry_price, 0.0)
    stop_loss = safe_float(stop_loss, 0.0)
    if side_norm not in ("long", "short") or entry_price <= 0:
        return stop_loss, "invalid_inputs"
    tf15 = dict((dict(context or {}).get("multi_timeframe") or {}).get("15m") or {})
    atr_price = max(safe_float(tf15.get("atr"), 0.0), entry_price * 0.003)
    structure_stop, _, anchor_source, _ = _derive_structure_stop_15m(side_norm, dict(context or {}), entry_price, atr_price)
    if structure_stop <= 0:
        return stop_loss, "no_structure_anchor"
    ai_valid = (side_norm == "long" and 0 < stop_loss < entry_price) or (side_norm == "short" and stop_loss > entry_price)
    ai_gap = abs(entry_price - stop_loss) if ai_valid else 0.0
    structure_gap = abs(entry_price - structure_stop)
    min_allowed_gap = structure_gap * 0.90
    if (not ai_valid) or ai_gap < min_allowed_gap:
        return structure_stop, "structure_guard_{}".format(anchor_source or "15m")
    return stop_loss, "keep_ai_stop"


def _repair_enter_levels(
    side: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    context: Dict[str, Any],
    decision: Dict[str, Any],
) -> tuple[float, float, str]:
    side_norm = _side_from_value(side)
    entry = safe_float(entry_price, 0.0)
    stop = safe_float(stop_loss, 0.0)
    take = safe_float(take_profit, 0.0)
    if side_norm not in ("long", "short") or entry <= 0:
        return stop, take, "invalid_inputs"
    notes: List[str] = []
    stop, guard_reason = _enforce_structure_stop_guard(side_norm, entry, stop, context)
    if guard_reason.startswith("structure_guard_"):
        notes.append(guard_reason)
    stop_valid = (side_norm == "long" and 0 < stop < entry) or (side_norm == "short" and stop > entry)
    if not stop_valid:
        tf15 = dict((dict(context or {}).get("multi_timeframe") or {}).get("15m") or {})
        atr_value = max(
            safe_float(tf15.get("atr"), 0.0),
            safe_float(tf15.get("a"), 0.0) * entry / 100.0,
            entry * 0.003,
        )
        fallback_gap = max(atr_value * 1.2, entry * 0.006)
        if side_norm == "long":
            stop = max(entry - fallback_gap, 1e-9)
        else:
            stop = entry + fallback_gap
        notes.append("fallback_atr_stop")
    if not _levels_consistent_for_side(side_norm, entry, stop, take):
        take = _local_take_profit_from_r(side_norm, entry, stop, decision)
        notes.append("rebuild_local_tp")
    return stop, take, ",".join(notes[:4])


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
    if side == "long" and not (0 < stop_loss < entry):
        stop_loss = 0.0
    if side == "short" and not (stop_loss > entry):
        stop_loss = 0.0
    if side == "long" and not (take_profit > entry):
        take_profit = 0.0
    if side == "short" and not (0 < take_profit < entry):
        take_profit = 0.0
    risk_reference_stop = 0.0
    recovered_source = ""
    for row in reversed(history):
        if str(row.get("symbol") or "") != symbol:
            continue
        src = str(row.get("candidate_source") or "").strip().lower()
        if src:
            recovered_source = src
            break
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
    if side == "long" and not (0 < risk_reference_stop < entry):
        risk_reference_stop = 0.0
    if side == "short" and not (risk_reference_stop > entry):
        risk_reference_stop = 0.0
    highest_price = max(entry, mark_price) if side == "long" else entry
    lowest_price = min(entry, mark_price) if side == "short" else entry
    return {
        "symbol": symbol,
        "side": side,
        "candidate_source": recovered_source or "general",
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
        "risk_plan_status": "ready" if stop_loss > 0 else "missing_initial_stop_from_ai",
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
    if qty <= 0 or stop_loss <= 0:
        return
    last_stage = str(rule.get("last_exchange_stop_sync_stage") or "")
    last_stop = safe_float(rule.get("last_exchange_stop_sync_price"), 0.0)
    last_qty = safe_float(rule.get("last_exchange_stop_sync_qty"), 0.0)
    last_ts = safe_float(rule.get("last_exchange_stop_sync_ts"), 0.0)
    last_ok = bool(rule.get("last_exchange_stop_sync_ok", False))
    qty_delta = max(last_qty * 0.15, qty * 0.15, 1e-8)
    stop_delta = max(risk * (0.20 if stage == "trailing" else 0.05), abs(stop_loss) * 0.0005, 1e-8)
    stage_changed = stage != last_stage
    qty_changed = abs(qty - last_qty) > qty_delta
    stop_improved = _stop_improved(side, stop_loss, last_stop, stop_delta)
    initial_retry_needed = stage == "initial" and not last_ok and (time.time() - last_ts) >= 10
    if not stage_changed and not qty_changed and not stop_improved and not initial_retry_needed:
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


def initialize_position_rule(signal: Dict[str, Any], qty: float, leverage: float, decision: Dict[str, Any], protection_result: Dict[str, Any] | None = None) -> None:
    symbol = str(signal.get("symbol") or "")
    if not symbol:
        return
    scale_in_price = safe_float(decision.get("scale_in_price"), 0.0)
    scale_in_qty_pct = clamp(decision.get("scale_in_qty_pct"), 0.0, 1.0)
    partials = default_position_partials()
    protection_result = dict(protection_result or {})
    initial_sl_ok = bool(protection_result.get("sl_ok", False))
    initial_stop = safe_float(signal.get("stop_loss"), 0.0)
    last_sync_stop = initial_stop if initial_sl_ok else 0.0
    last_sync_ts = time.time() if initial_sl_ok else 0.0
    POSITION_RULES[symbol] = {
        "symbol": symbol,
        "side": str(signal.get("side") or ""),
        "candidate_source": str(signal.get("candidate_source") or "general"),
        "initial_entry_price": safe_float(signal.get("price"), 0.0),
        "initial_stop_loss": initial_stop,
        "risk_reference_stop_loss": initial_stop,
        "initial_take_profit": safe_float(signal.get("take_profit"), 0.0),
        "active_stop_loss": initial_stop,
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
        "last_exchange_stop_sync_price": last_sync_stop,
        "last_exchange_stop_sync_stage": "initial",
        "last_exchange_stop_sync_qty": safe_float(qty, 0.0),
        "last_exchange_stop_sync_ts": last_sync_ts,
        "last_exchange_stop_sync_ok": initial_sl_ok,
        "created_at": tw_now_str(),
        "updated_at": tw_now_str(),
    }
    refresh_trailing_state()
    persist_runtime_snapshot_throttled(force=True)


def remove_position_rule(symbol: str) -> None:
    if symbol in POSITION_RULES:
        del POSITION_RULES[symbol]
        refresh_trailing_state()
        persist_runtime_snapshot_throttled(force=True)


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
    try:
        if BITGET_USE_COPY_TRADER_API:
            if str(side).lower() == "long":
                order = close_long(symbol, qty)
            else:
                order = close_short(symbol, qty)
        else:
            close_side = "sell" if str(side).lower() == "long" else "buy"
            order = _create_contract_order(symbol, "market", close_side, qty, None, reduce_only=True)
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
        return {"ok": True, "order_id": str((order or {}).get("id") or (order or {}).get("orderId") or (order or {}).get("trackingNo") or "")}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:220]}


def scale_in_position(symbol: str, side: str, rule: Dict[str, Any]) -> Dict[str, Any]:
    add_qty = safe_float(rule.get("initial_qty"), 0.0) * clamp(rule.get("scale_in_qty_pct"), 0.0, 1.0)
    if add_qty <= 0:
        return {"ok": False, "error": "scale_qty_zero"}
    order_side = "buy" if str(side).lower() == "long" else "sell"
    try:
        order = _create_contract_order(symbol, "market", order_side, add_qty, None, reduce_only=False)
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
        return {"ok": True, "order_id": str((order or {}).get("id") or (order or {}).get("orderId") or ""), "qty": add_qty}
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
        if side not in ("long", "short"):
            continue
        mark_price = safe_float(position.get("markPrice"), 0.0)
        intrabar_high = max(safe_float(position.get("intrabarHigh"), mark_price), mark_price)
        intrabar_low = min(safe_float(position.get("intrabarLow"), mark_price), mark_price) if safe_float(position.get("intrabarLow"), 0.0) > 0 else mark_price
        entry = safe_float(rule.get("initial_entry_price"), safe_float(position.get("entryPrice"), 0.0))
        stop = safe_float(rule.get("initial_stop_loss"), 0.0)
        take_profit = safe_float(rule.get("initial_take_profit"), 0.0)
        current_qty = safe_float(position.get("contracts"), 0.0)
        if mark_price <= 0 or entry <= 0 or current_qty <= 0:
            continue
        if side == "long" and (stop <= 0 or stop >= entry):
            rule["risk_plan_status"] = "missing_initial_stop_from_ai"
            rule["updated_at"] = tw_now_str()
            POSITION_RULES[symbol] = rule
            print("[POSITION_RULE_BLOCK] symbol={} side=long missing/invalid initial stop from AI, skip local reconstruction".format(symbol))
            continue
        elif side == "short" and (stop <= 0 or stop <= entry):
            rule["risk_plan_status"] = "missing_initial_stop_from_ai"
            rule["updated_at"] = tw_now_str()
            POSITION_RULES[symbol] = rule
            print("[POSITION_RULE_BLOCK] symbol={} side=short missing/invalid initial stop from AI, skip local reconstruction".format(symbol))
            continue
        if side == "long" and take_profit > 0 and take_profit <= entry:
            print("[POSITION_RULE_FIX] symbol={} side=long invalid take_profit {} <= entry {}, disable hard take-profit".format(symbol, round(take_profit, 8), round(entry, 8)))
            take_profit = 0.0
            rule["initial_take_profit"] = 0.0
        if side == "short" and take_profit > 0 and take_profit >= entry:
            print("[POSITION_RULE_FIX] symbol={} side=short invalid take_profit {} >= entry {}, disable hard take-profit".format(symbol, round(take_profit, 8), round(entry, 8)))
            take_profit = 0.0
            rule["initial_take_profit"] = 0.0
        if stop <= 0:
            continue
        risk = max(risk_unit_from_rule(rule), 1e-9)
        if side == "long":
            favorable_price = max(mark_price, intrabar_high)
            adverse_price = min(mark_price, intrabar_low)
            current_r = (favorable_price - entry) / risk
            rule["highest_price"] = max(safe_float(rule.get("highest_price"), entry), favorable_price)
            best_r = (safe_float(rule.get("highest_price"), favorable_price) - entry) / risk
        else:
            favorable_price = min(mark_price, intrabar_low)
            adverse_price = max(mark_price, intrabar_high)
            current_r = (entry - favorable_price) / risk
            low_ref = safe_float(rule.get("lowest_price"), entry) or entry
            rule["lowest_price"] = min(low_ref, favorable_price)
            best_r = (entry - safe_float(rule.get("lowest_price"), favorable_price)) / risk
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
            if best_r >= 3.0:
                managed_stop = max(managed_stop, entry + risk * 1.5)
                stop_stage = "lock_1_5r"
            if rule.get("trailing_active"):
                trail_stop = max(managed_stop, safe_float(rule.get("highest_price"), favorable_price) - risk)
                managed_stop = max(managed_stop, trail_stop)
                rule["trailing_stop"] = trail_stop
                if trail_stop > (entry + risk * 1.5 + 1e-12):
                    stop_stage = "trailing"
        else:
            if best_r >= 1.0:
                managed_stop = min(managed_stop, entry)
                stop_stage = "breakeven"
            if best_r >= 2.0:
                managed_stop = min(managed_stop, entry - risk)
                stop_stage = "lock_1r"
                rule["trailing_active"] = True
            if best_r >= 3.0:
                managed_stop = min(managed_stop, entry - risk * 1.5)
                stop_stage = "lock_1_5r"
            if rule.get("trailing_active"):
                trail_stop = min(managed_stop, safe_float(rule.get("lowest_price"), favorable_price) + risk)
                managed_stop = min(managed_stop, trail_stop)
                rule["trailing_stop"] = trail_stop
                if trail_stop < (entry - risk * 1.5 - 1e-12):
                    stop_stage = "trailing"
        rule["active_stop_loss"] = managed_stop
        rule["stop_stage"] = stop_stage
        hard_stop_hit = (side == "long" and adverse_price <= managed_stop) or (side == "short" and adverse_price >= managed_stop)
        if hard_stop_hit:
            result = reduce_position(symbol, side, current_qty, "hard_stop_loss_{}".format(stop_stage))
            if result.get("ok"):
                remove_position_rule(symbol)
                continue
        hard_take_profit_hit = take_profit > 0 and (
            (side == "long" and favorable_price >= take_profit)
            or (side == "short" and favorable_price <= take_profit)
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
            reached_r = max(current_r, best_r)
            if reached_r < trigger_r:
                continue
            partial_basis = str(partial.get("basis") or "initial").lower()
            qty_basis = current_qty if partial_basis == "remaining" else safe_float(rule.get("initial_qty"), current_qty)
            close_qty = min(current_qty, qty_basis * safe_float(partial.get("fraction"), 0.0))
            if close_qty <= 0:
                partial["done"] = True
                continue
            result = reduce_position(symbol, side, close_qty, "tp_{}R".format(trigger_r))
            if result.get("ok"):
                partial["done"] = True
                current_qty = max(current_qty - close_qty, 0.0)
                rule["remaining_qty"] = current_qty
                if trigger_r >= 2.0:
                    rule["trailing_active"] = True
                if trigger_r >= 3.0:
                    if side == "long":
                        rule["active_stop_loss"] = max(safe_float(rule.get("active_stop_loss"), stop), entry + risk * 1.5)
                    else:
                        rule["active_stop_loss"] = min(safe_float(rule.get("active_stop_loss"), stop), entry - risk * 1.5)
                    rule["stop_stage"] = "lock_1_5r"
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
    persist_runtime_snapshot_throttled()


def _apply_openai_trade_plan_to_signal(sig: Dict[str, Any], decision: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    plan = dict(decision or {})
    if not plan:
        return sig
    plan["decision_created_ts"] = now_ts()
    plan["decision_expire_ts"] = safe_float(plan.get("decision_expire_ts"), 0.0) or (now_ts() + ORDER_BLOCK_EXPIRY_SEC)
    sig["openai_trade_plan"] = plan
    sig["openai_trade_meta"] = {
        "model": str((result or {}).get("model") or ""),
        "status": str((result or {}).get("status") or ""),
        "payload_hash": str((result or {}).get("payload_hash") or ""),
    }
    sig["decision_source"] = "openai"
    decision_side = str(plan.get("trade_side") or "").lower()
    if decision_side in ("long", "short"):
        sig["side"] = decision_side
    sig["planned_entry_price"] = float(plan.get("entry_price", 0) or 0)
    sig["stop_loss"] = float(plan.get("stop_loss", 0) or 0)
    sig["stop_loss_guard_reason"] = "ai_direct"
    sig["ai_take_profit_hint"] = float(plan.get("take_profit", 0) or 0)
    sig["openai_take_profit"] = sig["ai_take_profit_hint"]
    entry_for_tp = safe_float(sig.get("planned_entry_price"), safe_float(sig.get("price"), 0.0))
    local_tp = _local_take_profit_from_r(str(sig.get("side") or decision_side), entry_for_tp, sig["stop_loss"], plan)
    sig["take_profit"] = local_tp
    sig["take_profit_source"] = "local_rr_engine"
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
        {"productType": BITGET_PRODUCT_TYPE, "marginCoin": BITGET_MARGIN_COIN, "holdSide": pos_side},
        {"productType": BITGET_PRODUCT_TYPE, "marginCoin": BITGET_MARGIN_COIN, "marginMode": BITGET_MARGIN_MODE, "holdSide": pos_side},
        {"productType": BITGET_PRODUCT_TYPE, "marginCoin": BITGET_MARGIN_COIN, "marginMode": BITGET_MARGIN_MODE, "posSide": pos_side},
        {"productType": BITGET_PRODUCT_TYPE, "marginCoin": BITGET_MARGIN_COIN, "marginMode": BITGET_MARGIN_MODE, "tradeSide": "open", "posSide": pos_side},
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
    raw_qty = fixed_order_notional_usdt(symbol, entry_price) / entry_price
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
    use_exchange_triggers = bool(ENABLE_EXCHANGE_TRIGGER_PROTECTION)
    close_side = "sell" if str(side).lower() == "buy" else "buy"
    pos_side = "long" if str(side).lower() == "buy" else "short"
    outcome = {"sl_ok": False, "tp_ok": False}
    sl_errors: List[str] = []
    tp_errors: List[str] = []
    if BITGET_USE_COPY_TRADER_API:
        tracking_no = ""
        requested_sl = safe_float(stop_loss, 0.0) > 0
        requested_tp = safe_float(take_profit, 0.0) > 0 and (not manage_take_profit_locally)
        for _ in range(3):
            try:
                for row in _bitget_copy_fetch_current_tracks(symbol):
                    if symbol_key(str(row.get("symbol") or "")) == symbol_key(symbol):
                        tracking_no = str(row.get("trackingNo") or "")
                        if tracking_no:
                            break
                if tracking_no:
                    break
            except Exception as exc:
                sl_errors.append("copy_track_lookup:" + str(exc)[:120])
            time.sleep(0.6)
        if requested_sl and not tracking_no:
            sl_errors.append("copy_tracking_no_missing")
        if tracking_no and (requested_sl or requested_tp):
            ok = _bitget_copy_modify_tpsl(
                symbol,
                tracking_no,
                stop_loss if requested_sl else 0.0,
                0.0 if manage_take_profit_locally else take_profit,
            )
            if requested_sl:
                outcome["sl_ok"] = bool(ok)
                if not bool(ok):
                    sl_errors.append("copy_modify_tpsl_failed")
            else:
                outcome["sl_ok"] = False
            if requested_tp:
                outcome["tp_ok"] = bool(ok)
                if not bool(ok):
                    tp_errors.append("copy_modify_tpsl_failed")
            else:
                outcome["tp_ok"] = True
        else:
            outcome["sl_ok"] = not requested_sl
            outcome["tp_ok"] = True if manage_take_profit_locally else (not requested_tp)
    elif use_exchange_triggers:
        sl_attempts = [
            {"reduceOnly": True, "stopPrice": str(stop_loss), "orderType": "stop", "posSide": pos_side, "tdMode": "cross", "marginMode": "cross"},
            {"reduceOnly": True, "stopLossPrice": str(stop_loss), "posSide": pos_side, "tdMode": "cross", "marginMode": "cross"},
            {"reduceOnly": True, "triggerPrice": str(stop_loss), "triggerType": "mark_price", "posSide": pos_side, "tdMode": "cross", "marginMode": "cross"},
            {"reduceOnly": True, "triggerPrice": str(stop_loss), "triggerType": "mark_price", "planType": "loss_plan", "posSide": pos_side, "tdMode": "cross", "marginMode": "cross"},
            {"reduceOnly": True, "stopLossTriggerPrice": str(stop_loss), "stopLossTriggerType": "mark_price", "posSide": pos_side, "tdMode": "cross", "marginMode": "cross"},
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
            except Exception as exc:
                sl_errors.append(str(exc)[:160])
                continue
        if manage_take_profit_locally:
            outcome["tp_ok"] = True
        else:
            for params in tp_attempts:
                try:
                    exchange.create_order(symbol, "market", close_side, qty, None, params)
                    outcome["tp_ok"] = True
                    break
                except Exception as exc:
                    tp_errors.append(str(exc)[:160])
                    continue
    else:
        outcome["sl_ok"] = safe_float(stop_loss, 0.0) > 0
        outcome["tp_ok"] = True if manage_take_profit_locally else (safe_float(take_profit, 0.0) > 0)
    with STATE_LOCK:
        protection_state = dict(STATE.get("protection_state") or {})
        protection_state[symbol] = {
            "sl_ok": outcome["sl_ok"],
            "tp_ok": outcome["tp_ok"],
            "sl": round(stop_loss, 8),
            "tp": round(take_profit, 8),
            "sl_mode": "copy_modify_tpsl" if BITGET_USE_COPY_TRADER_API else ("exchange_trigger" if use_exchange_triggers else "local_bot_managed"),
            "tp_mode": "local_bot_managed" if manage_take_profit_locally else ("copy_modify_tpsl" if BITGET_USE_COPY_TRADER_API else ("exchange_take_profit" if use_exchange_triggers else "local_bot_managed")),
            "updated_at": tw_now_str(),
            "sl_error": "" if outcome["sl_ok"] else " | ".join(sl_errors[:2]),
            "tp_error": "" if outcome["tp_ok"] else " | ".join(tp_errors[:2]),
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
            "candidate_source": str(signal.get("candidate_source") or "general"),
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
    created_ts = now_ts()
    with STATE_LOCK:
        pending = dict(STATE.get("fvg_orders") or {})
        pending[symbol] = {
            "symbol": symbol,
            "order_id": str(order.get("id") or ""),
            "client_oid": str(order.get("client_oid") or ""),
            "price": safe_float(signal.get("planned_entry_price", signal.get("price")), 0.0),
            "stop_loss": safe_float(signal.get("stop_loss"), 0.0),
            "take_profit": safe_float(signal.get("take_profit"), 0.0),
            "side": signal.get("side"),
            "candidate_source": str(signal.get("candidate_source") or "general"),
            "leverage": leverage,
            "qty": size_info.get("qty", 0.0),
            "order_usdt": size_info.get("order_usdt", 0.0),
            "decision": dict(signal.get("openai_trade_plan") or {}),
            "openai_market_context": dict(signal.get("openai_market_context") or {}),
            "created_ts": created_ts,
            "expires_ts": created_ts + ORDER_BLOCK_EXPIRY_SEC,
            "updated_at": tw_now_str(),
        }
        STATE["fvg_orders"] = pending


def place_order_from_signal(signal: Dict[str, Any]) -> Dict[str, Any]:
    symbol = str(signal.get("symbol") or "")
    decision = dict(signal.get("openai_trade_plan") or {})
    if not symbol or not decision:
        return {"ok": False, "error": "missing_plan"}
    decision_expire_ts = safe_float(decision.get("decision_expire_ts"), 0.0)
    if decision_expire_ts > 0 and now_ts() > decision_expire_ts:
        return {"ok": False, "error": "stale_decision"}
    result: Dict[str, Any] = {"ok": False, "error": "unknown"}
    should_refresh_after_order = False
    with ORDER_LOCK:
        decision_side = _side_from_value(decision.get("trade_side"))
        if decision_side not in ("long", "short"):
            return {"ok": False, "error": "missing_trade_side"}
        execution_score = max(
            safe_float(signal.get("score"), 0.0),
            safe_float(signal.get("priority_score"), 0.0),
            safe_float(signal.get("raw_score"), 0.0),
        )
        if execution_score < OPENAI_EXECUTION_MIN_SCORE:
            return {
                "ok": False,
                "error": "score_below_execution_gate:{:.2f}<{:.2f}".format(execution_score, OPENAI_EXECUTION_MIN_SCORE),
            }
        instruction = str(decision.get("bot_instruction") or "").strip().upper()
        planned_entry = safe_float(decision.get("entry_price"), safe_float(signal.get("price"), 0.0))
        if planned_entry <= 0:
            return {"ok": False, "error": "entry_non_positive"}
        raw_stop = safe_float(decision.get("stop_loss"), 0.0)
        raw_take = safe_float(decision.get("take_profit"), 0.0)
        repaired_stop, repaired_take, repair_note = _repair_enter_levels(
            decision_side,
            planned_entry,
            raw_stop,
            raw_take,
            dict(signal.get("openai_market_context") or {}),
            decision,
        )
        decision["entry_price"] = planned_entry
        decision["stop_loss"] = safe_float(repaired_stop, raw_stop)
        decision["take_profit"] = safe_float(repaired_take, raw_take)
        signal["risk_repair_note"] = str(repair_note or "")
        executable, execute_reason = enter_decision_ready(decision)
        if not executable:
            return {"ok": False, "error": "non_executable_instruction:{}".format(execute_reason)}
        symbol_norm = symbol_key(symbol)
        active_symbols = open_position_symbol_keys()
        pending_symbols = pending_order_symbol_keys()
        if symbol_norm in active_symbols or symbol_norm in pending_symbols:
            refreshed, refresh_error = refresh_execution_position_snapshot()
            if refreshed:
                active_symbols = open_position_symbol_keys()
                pending_symbols = pending_order_symbol_keys()
            if symbol_norm in active_symbols or symbol_norm in pending_symbols:
                detail = "already_active"
                if refresh_error:
                    detail = "{}|snapshot_refresh_failed={}".format(detail, refresh_error)
                return {"ok": False, "error": detail}
        exposure = exposure_snapshot()
        lane = _signal_lane(signal)
        if lane == "prebreakout":
            if safe_int(exposure.get("total_prebreakout"), 0) >= PREBREAKOUT_MAX_OPEN_POSITIONS:
                return {"ok": False, "error": "max_positions_prebreakout"}
        else:
            if safe_int(exposure.get("total_general"), 0) >= MAX_OPEN_POSITIONS:
                return {"ok": False, "error": "max_positions"}
            side_total = safe_int(exposure.get("long_total_general"), 0) if decision_side == "long" else safe_int(exposure.get("short_total_general"), 0)
            if side_total >= MAX_SAME_DIRECTION_POSITIONS:
                return {"ok": False, "error": "max_same_direction"}
        signal["side"] = decision_side
        order_side = "buy" if decision_side == "long" else "sell"
        leverage, _, lev_error, lev_ok = _force_set_symbol_max_leverage(symbol, order_side)
        if not lev_ok and not BITGET_USE_COPY_TRADER_API:
            return {"ok": False, "error": lev_error or "leverage_failed"}
        signal["planned_entry_price"] = planned_entry
        signal["price"] = planned_entry if planned_entry > 0 else safe_float(signal.get("price"), 0.0)
        order_type = "limit" if str(decision.get("order_type") or "").lower() == "limit" or instruction == "ENTER_LIMIT" else "market"
        signal["stop_loss"] = safe_float(decision.get("stop_loss"), 0.0)
        signal["stop_loss_guard_reason"] = "ai_direct"
        signal["ai_take_profit_hint"] = safe_float(decision.get("take_profit"), 0.0)
        signal["openai_take_profit"] = signal["ai_take_profit_hint"]
        signal["take_profit"] = _local_take_profit_from_r(decision_side, signal["price"], signal["stop_loss"], decision)
        signal["take_profit_source"] = "local_rr_engine"
        if safe_float(signal.get("take_profit"), 0.0) <= 0:
            return {"ok": False, "error": "local_take_profit_unavailable"}
        size_info = compute_order_size(symbol, signal["price"], leverage)
        qty = safe_float(size_info.get("qty"), 0.0)
        if qty <= 0:
            return {"ok": False, "error": "qty_zero"}
        try:
            if order_type == "limit":
                order = _create_contract_order(
                    symbol,
                    "limit",
                    order_side,
                    qty,
                    signal["price"],
                    reduce_only=False,
                    initial_stop_loss=signal["stop_loss"],
                    initial_take_profit=0.0,
                )
                register_pending_order(symbol, order, signal, leverage, size_info)
                result = {"ok": True, "pending": True, "order_id": str((order or {}).get("id") or (order or {}).get("orderId") or "")}
                should_refresh_after_order = False
            else:
                order = _create_contract_order(
                    symbol,
                    "market",
                    order_side,
                    qty,
                    None,
                    reduce_only=False,
                    initial_stop_loss=signal["stop_loss"],
                    initial_take_profit=0.0,
                )
                fill_price = safe_float(order.get("average"), safe_float(order.get("price"), safe_float(signal.get("price"), 0.0)))
                if fill_price > 0:
                    signal["price"] = fill_price
                protection_result = ensure_exchange_protection(symbol, order_side, qty, signal["stop_loss"], signal["take_profit"], manage_take_profit_locally=True)
                record_open_trade(signal, qty, leverage, size_info.get("order_usdt", 0.0), decision)
                initialize_position_rule(signal, qty, leverage, decision, protection_result=protection_result)
                result = {"ok": True, "pending": False, "order_id": str((order or {}).get("id") or (order or {}).get("orderId") or "")}
                should_refresh_after_order = True
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:220]}
    if bool(result.get("ok")) and should_refresh_after_order:
        refreshed, refresh_error = refresh_execution_position_snapshot()
        if not refreshed and refresh_error:
            result["post_sync_warning"] = "position_snapshot_refresh_failed:{}".format(refresh_error)
    return result


def manage_pending_limit_orders() -> None:
    with STATE_LOCK:
        pending = dict(STATE.get("fvg_orders") or {})
    filled_count = 0
    for symbol, row in pending.items():
        order_id = str(row.get("order_id") or "")
        client_oid = str(row.get("client_oid") or "")
        expires_ts = safe_float(row.get("expires_ts"), 0.0)
        if expires_ts > 0 and now_ts() >= expires_ts:
            if order_id or client_oid:
                try:
                    if BITGET_USE_COPY_TRADER_API:
                        _bitget_mix_cancel_order(symbol, order_id=order_id, client_oid=client_oid)
                    else:
                        exchange.cancel_order(order_id, symbol)
                except Exception:
                    pass
            clear_pending_order(symbol)
            continue
        if not order_id and not client_oid:
            clear_pending_order(symbol)
            continue
        if BITGET_USE_COPY_TRADER_API:
            try:
                order = _bitget_mix_order_detail(symbol, order_id=order_id, client_oid=client_oid)
            except Exception:
                order = {}
        else:
            try:
                order = exchange.fetch_order(order_id, symbol)
            except Exception:
                order = {}
        status = str(order.get("status") or "").lower()
        decision = dict(row.get("decision") or {})
        if status in ("canceled", "cancelled", "rejected", "expired", "failed"):
            clear_pending_order(symbol)
            continue
        if status in ("closed", "filled"):
            fill_price = safe_float(
                order.get("average"),
                safe_float(order.get("priceAvg"), safe_float(order.get("price"), safe_float(row.get("price"), 0.0))),
            )
            side = str(row.get("side") or "").lower()
            entry_price = fill_price if fill_price > 0 else safe_float(row.get("price"), 0.0)
            stop_loss = safe_float(row.get("stop_loss"), 0.0)
            local_take_profit = safe_float(row.get("take_profit"), 0.0)
            repaired_stop, repaired_take, repair_note = _repair_enter_levels(
                side,
                entry_price,
                stop_loss,
                local_take_profit,
                dict((row.get("openai_market_context") or {}) if isinstance(row, dict) else {}),
                decision,
            )
            stop_loss = safe_float(repaired_stop, stop_loss)
            local_take_profit = safe_float(repaired_take, local_take_profit)
            if local_take_profit <= 0:
                local_take_profit = safe_float(row.get("take_profit"), 0.0)
            protection_result = ensure_exchange_protection(
                symbol,
                "buy" if side == "long" else "sell",
                safe_float(row.get("qty"), 0.0),
                stop_loss,
                local_take_profit,
                manage_take_profit_locally=True,
            )
            initialize_position_rule(
                {
                    "symbol": symbol,
                    "side": row.get("side"),
                    "candidate_source": row.get("candidate_source"),
                    "price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": local_take_profit,
                    "risk_repair_note": str(repair_note or ""),
                },
                safe_float(row.get("qty"), 0.0),
                safe_float(row.get("leverage"), 0.0),
                decision,
                protection_result=protection_result,
            )
            append_trade_history(
                {
                    "time": tw_now_str(),
                    "symbol": symbol,
                    "side": row.get("side"),
                    "price": safe_float(row.get("price"), 0.0),
                    "score": 0.0,
                    "stop_loss": stop_loss,
                    "take_profit": local_take_profit,
                    "pnl_pct": None,
                    "order_usdt": safe_float(row.get("order_usdt"), 0.0),
                    "decision_source": "openai_limit_fill",
                    "candidate_source": str(row.get("candidate_source") or "general"),
                }
            )
            clear_pending_order(symbol)
            filled_count += 1
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
    if filled_count > 0:
        refresh_execution_position_snapshot()


def build_scan_universe(
    markets: Dict[str, Any],
    tickers: Dict[str, Any],
    excluded_symbols: set[str] | None = None,
) -> tuple[List[tuple[str, Dict[str, Any], Dict[str, Any]]], List[tuple[str, Dict[str, Any], Dict[str, Any]]], List[tuple[str, Dict[str, Any], Dict[str, Any]]], List[tuple[str, Dict[str, Any], Dict[str, Any]]]]:
    excluded = {str(sym or "") for sym in set(excluded_symbols or set()) if str(sym or "")}
    copy_allowed_keys: set[str] = set()
    if BITGET_USE_COPY_TRADER_API:
        try:
            copy_allowed_keys = _copy_allowed_symbol_keys()
        except Exception:
            copy_allowed_keys = set()

    def _quote_volume_24h(ticker_row: Dict[str, Any]) -> float:
        return safe_float(
            ticker_row.get("quoteVolume"),
            safe_float(ticker_row.get("baseVolume"), 0.0) * max(safe_float(ticker_row.get("last"), 0.0), 0.0),
        )

    tradeable = []
    for symbol, market in markets.items():
        if not isinstance(market, dict):
            continue
        if symbol in excluded:
            continue
        if not is_usdt_symbol(market) or not bool(market.get("active", True)) or market_type_label(market) == "spot":
            continue
        if copy_allowed_keys and symbol_key(symbol) not in copy_allowed_keys:
            continue
        ticker = dict(tickers.get(symbol) or {})
        if _quote_volume_24h(ticker) < MIN_SYMBOL_QUOTE_VOLUME:
            continue
        tradeable.append((symbol, market, ticker))
    by_volume = sorted(tradeable, key=lambda row: _quote_volume_24h(dict(row[2] or {})), reverse=True)
    by_change = sorted(tradeable, key=lambda row: safe_float(row[2].get("percentage"), 0.0), reverse=True)
    volume_universe = by_volume[:SCAN_SYMBOL_LIMIT]
    short_gainer_universe = [row for row in by_change if safe_float(row[2].get("percentage"), 0.0) >= SHORT_GAINER_MIN_24H_PCT][: max(SHORT_GAINER_TOP_PICK * 2, 6)]
    watch_universe = []
    watched = watched_observe_symbols()
    for symbol in watched:
        if symbol in excluded:
            continue
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


def build_prebreakout_signals(
    markets: Dict[str, Any],
    tickers: Dict[str, Any],
    context_map: Dict[str, Dict[str, Any]],
    btc_context: Dict[str, Any] | None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    try:
        leaderboard = PREBREAKOUT_SCANNER.run(
            top_pick=PREBREAKOUT_TOP_PICK,
            symbol_limit=PREBREAKOUT_SYMBOL_LIMIT,
            fixed_order_notional_usdt=FIXED_ORDER_NOTIONAL_USDT,
            universe_label="bitget_usdt_perp",
            markets=markets,
            tickers=tickers,
        )
    except Exception:
        return [], {"scanner_ts": safe_int(now_ts(), 0), "universe": "bitget_usdt_perp", "mode": "pre_breakout_discovery", "top_candidates": []}
    candidate_payloads = dict(leaderboard.get("candidate_payloads") or {})
    output_rows: List[Dict[str, Any]] = []
    for row in list(leaderboard.get("top_candidates") or [])[:PREBREAKOUT_TOP_PICK]:
        symbol = str((row or {}).get("symbol") or "")
        if not symbol:
            continue
        market = dict(markets.get(symbol) or {})
        ticker = dict(tickers.get(symbol) or {})
        if not market or not ticker:
            continue
        if symbol not in context_map:
            try:
                context_map[symbol] = build_market_context(symbol, ticker, market)
            except Exception:
                continue
        context = dict(context_map.get(symbol) or {})
        signal = build_signal_from_context(
            symbol,
            {"symbol": symbol, "pattern": "prebreakout_scanner"},
            context,
            btc_context,
            candidate_source="prebreakout_scanner",
        )
        payload = dict(candidate_payloads.get(symbol) or {})
        signal["rank"] = safe_int((row or {}).get("rank"), 0)
        signal["priority_score"] = safe_float(
            (row or {}).get("rank_metric_quote_volume_24h"),
            safe_float(signal.get("priority_score"), 0.0),
        )
        signal["prebreakout_scanner_features"] = dict(payload.get("scanner_features") or {})
        signal["prebreakout_raw_candidate_payload"] = dict(payload.get("raw_candidate_payload") or {})
        signal["prebreakout_flow_context"] = dict((signal.get("prebreakout_raw_candidate_payload") or {}).get("flow_context") or {})
        signal["scanner_intent"] = ""
        signal["desc"] = "pre-breakout candidate (data only)"
        output_rows.append(signal)
    leaderboard_view = {
        "scanner_ts": safe_int(leaderboard.get("scanner_ts"), safe_int(now_ts(), 0)),
        "universe": str(leaderboard.get("universe") or "bitget_usdt_perp"),
        "mode": str(leaderboard.get("mode") or "pre_breakout_discovery"),
        "top_candidates": list(leaderboard.get("top_candidates") or [])[:PREBREAKOUT_TOP_PICK],
    }
    return output_rows, leaderboard_view


def choose_review_candidates(
    general_top: List[Dict[str, Any]],
    short_gainers: List[Dict[str, Any]],
    prebreakout_top: List[Dict[str, Any]],
    pending_advice_signals: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    snapshot = exposure_snapshot()
    virtual_general_total = safe_int(snapshot.get("total_general"), 0)
    virtual_prebreakout_total = safe_int(snapshot.get("total_prebreakout"), 0)

    def _pool_sort(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            list(rows or []),
            key=lambda row: (
                safe_int(row.get("rank"), 9999),
                -safe_float(row.get("priority_score"), abs(safe_float(row.get("score"), 0.0))),
                -abs(safe_float(row.get("score"), 0.0)),
                str(row.get("symbol") or ""),
            ),
        )

    pools = [
        ("pending_advice", _pool_sort(pending_advice_signals)),
        ("short_gainers", _pool_sort(short_gainers)),
        ("general", _pool_sort(general_top)),
        ("prebreakout_scanner", _pool_sort(prebreakout_top)),
    ]

    for pool_name, pool_rows in pools:
        for row in pool_rows:
            symbol = str(row.get("symbol") or "")
            dedupe_key = "{}|{}".format(pool_name, symbol)
            if not symbol or dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            picked = dict(row)
            picked["side"] = _side_from_value(row.get("side")) or "neutral"
            picked["queue_pool"] = pool_name
            picked["queue_rank"] = safe_int(picked.get("rank"), 0)
            lane = _signal_lane(picked)
            if lane == "prebreakout":
                if virtual_prebreakout_total >= PREBREAKOUT_MAX_OPEN_POSITIONS:
                    picked["force_openai_recheck"] = False
                    picked["local_gate_ok"] = False
                    picked["local_gate_reason"] = "prebreakout_full"
                    picked["local_gate_next_allowed_ts"] = now_ts() + ORDER_BLOCK_EXPIRY_SEC
                    picked["local_gate_next_recheck_ts"] = 0.0
                    picked["observe_expires_ts"] = 0.0
                    candidates.append(picked)
                    continue
            else:
                if virtual_general_total >= MAX_OPEN_POSITIONS:
                    picked["force_openai_recheck"] = False
                    picked["local_gate_ok"] = False
                    picked["local_gate_reason"] = "max_positions_general"
                    picked["local_gate_next_allowed_ts"] = now_ts() + ORDER_BLOCK_EXPIRY_SEC
                    picked["local_gate_next_recheck_ts"] = 0.0
                    picked["observe_expires_ts"] = 0.0
                    candidates.append(picked)
                    continue

            gate = candidate_review_gate(picked)
            picked["force_openai_recheck"] = bool(gate.get("force_recheck", False))
            picked["local_gate_ok"] = bool(gate.get("ok", False))
            picked["local_gate_reason"] = str(gate.get("reason") or "")
            picked["local_gate_next_allowed_ts"] = safe_float(gate.get("next_allowed_ts"), 0.0)
            picked["local_gate_next_recheck_ts"] = safe_float(gate.get("next_recheck_ts"), 0.0)
            picked["observe_expires_ts"] = safe_float(gate.get("observe_expires_ts"), 0.0)
            candidates.append(picked)
            if bool(gate.get("ok", False)):
                if lane == "prebreakout":
                    virtual_prebreakout_total += 1
                else:
                    virtual_general_total += 1
    return candidates


def _refresh_signal_with_latest_snapshot(signal: Dict[str, Any]) -> Dict[str, Any]:
    symbol = str((signal or {}).get("symbol") or "")
    if not symbol:
        return dict(signal or {})
    row = dict(signal or {})
    ticker = exchange.fetch_ticker(symbol) or {}
    if not ticker:
        return row
    context = dict(row.get("openai_market_context") or {})
    basic = dict(context.get("basic_market_data") or {})
    last_price = safe_float(ticker.get("last"), safe_float(row.get("price"), safe_float(basic.get("current_price"), 0.0)))
    quote_volume_24h = safe_float(
        ticker.get("quoteVolume"),
        safe_float(ticker.get("baseVolume"), 0.0) * max(last_price, 0.0),
    )
    basic["current_price"] = round(last_price, 6) if last_price > 0 else safe_float(basic.get("current_price"), 0.0)
    basic["change_24h_pct"] = round(safe_float(ticker.get("percentage"), safe_float(basic.get("change_24h_pct"), 0.0)), 4)
    basic["quote_volume_24h"] = round(quote_volume_24h, 4)
    basic["base_volume_24h"] = round(safe_float(ticker.get("baseVolume"), safe_float(basic.get("base_volume_24h"), 0.0)), 4)
    context["basic_market_data"] = basic
    row["openai_market_context"] = context
    row["price"] = safe_float(basic.get("current_price"), safe_float(row.get("price"), 0.0))
    return row


def maybe_run_openai(
    general_top: List[Dict[str, Any]],
    short_gainers: List[Dict[str, Any]],
    prebreakout_top: List[Dict[str, Any]],
    pending_advice_signals: List[Dict[str, Any]],
    skip_symbols: set[str] | None = None,
) -> str:
    global OPENAI_TRADE_STATE, OPENAI_REVIEW_ROTATION_SIGNATURE, OPENAI_REVIEW_ROTATION_CURSOR

    def _rotation_signature(rows: List[Dict[str, Any]]) -> str:
        symbols = [str((row or {}).get("symbol") or "") for row in list(rows or []) if str((row or {}).get("symbol") or "")]
        return "|".join(symbols)

    def _rotate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        global OPENAI_REVIEW_ROTATION_SIGNATURE, OPENAI_REVIEW_ROTATION_CURSOR
        if not rows:
            return []
        signature = _rotation_signature(rows)
        if signature != OPENAI_REVIEW_ROTATION_SIGNATURE:
            OPENAI_REVIEW_ROTATION_SIGNATURE = signature
            OPENAI_REVIEW_ROTATION_CURSOR = 0
        OPENAI_REVIEW_ROTATION_CURSOR = OPENAI_REVIEW_ROTATION_CURSOR % len(rows)
        return list(rows[OPENAI_REVIEW_ROTATION_CURSOR:]) + list(rows[:OPENAI_REVIEW_ROTATION_CURSOR])

    def _advance_cursor(rows: List[Dict[str, Any]], sent_symbol: str = "") -> None:
        global OPENAI_REVIEW_ROTATION_CURSOR
        if not rows:
            return
        symbols = [str((row or {}).get("symbol") or "") for row in rows]
        if sent_symbol and sent_symbol in symbols:
            OPENAI_REVIEW_ROTATION_CURSOR = (symbols.index(sent_symbol) + 1) % len(rows)
            return
        OPENAI_REVIEW_ROTATION_CURSOR = (OPENAI_REVIEW_ROTATION_CURSOR + 1) % len(rows)

    try:
        manage_pending_limit_orders()
    except Exception:
        pass
    exposure = exposure_snapshot()
    total_slots = safe_int(exposure.get("total_all"), 0)
    if total_slots >= MAX_OPEN_POSITIONS:
        refreshed, _ = refresh_execution_position_snapshot()
        if refreshed:
            try:
                manage_pending_limit_orders()
            except Exception:
                pass
            exposure = exposure_snapshot()
            total_slots = safe_int(exposure.get("total_all"), 0)
    if total_slots >= MAX_OPEN_POSITIONS:
        AI_PANEL["openai_gate_debug"] = {
            "attempted": "",
            "status": "max_positions_global",
            "blockers": ["global:max_positions_full"],
            "blockers_zh": ["總持倉已滿，暫停送單"],
            "forced_first_send": False,
        }
        set_backend_thread("openai", "running", "Global max positions reached ({}). Skip OpenAI send.".format(MAX_OPEN_POSITIONS))
        refresh_openai_dashboard()
        return ""

    skip_symbols = set(str(sym or "") for sym in set(skip_symbols or set()) if str(sym or ""))
    reviewed_candidates_all = _prioritize_openai_candidates_by_source(
        choose_review_candidates(general_top, short_gainers, prebreakout_top, pending_advice_signals)
    )
    reviewed_candidates_all = _rotate_rows(reviewed_candidates_all)
    reviewed_candidates = list(reviewed_candidates_all)
    if skip_symbols:
        reviewed_candidates = [row for row in reviewed_candidates if str(row.get("symbol") or "") not in skip_symbols]
    if not reviewed_candidates:
        AI_PANEL["openai_gate_debug"] = {
            "attempted": "",
            "status": "no_ranked_candidates",
            "blockers": [],
            "blockers_zh": [],
            "forced_first_send": False,
        }
        refresh_openai_dashboard()
        return ""
    blockers: List[str] = []
    for reviewed in reviewed_candidates:
        if not reviewed.get("local_gate_ok"):
            blocker = "{}:{}".format(compact_symbol(reviewed.get("symbol")), reviewed.get("local_gate_reason"))
            if str(reviewed.get("local_gate_reason") or "") == "skip_cooldown":
                next_ts = safe_float(reviewed.get("local_gate_next_allowed_ts"), 0.0)
                if next_ts > 0:
                    blocker = "{}:skip_cooldown_until={}".format(compact_symbol(reviewed.get("symbol")), tw_from_ts(next_ts))
            if str(reviewed.get("local_gate_reason") or "") in ("recheck_cooldown", "observe_manual_only", "observe_expired"):
                next_ts = safe_float(reviewed.get("local_gate_next_recheck_ts"), 0.0)
                if next_ts <= 0:
                    next_ts = safe_float(reviewed.get("observe_expires_ts"), 0.0)
                if next_ts > 0:
                    blocker = "{}:manual_only_until={}".format(compact_symbol(reviewed.get("symbol")), tw_from_ts(next_ts))
            if str(reviewed.get("local_gate_reason") or "") in ("capacity_blocked", "same_direction_full"):
                next_ts = safe_float(reviewed.get("local_gate_next_allowed_ts"), 0.0)
                if next_ts > 0:
                    blocker = "{}:{}_until={}".format(
                        compact_symbol(reviewed.get("symbol")),
                        str(reviewed.get("local_gate_reason") or "capacity_blocked"),
                        tw_from_ts(next_ts),
                    )
            if str(reviewed.get("local_gate_reason") or "") in ("prebreakout_observe_waiting_change", "prebreakout_full"):
                next_ts = safe_float(reviewed.get("observe_expires_ts"), 0.0)
                if next_ts > 0:
                    blocker = "{}:{}_until={}".format(
                        compact_symbol(reviewed.get("symbol")),
                        str(reviewed.get("local_gate_reason") or ""),
                        tw_from_ts(next_ts),
                    )
            blockers.append(blocker)
            continue
        try:
            reviewed = _refresh_signal_with_latest_snapshot(reviewed)
        except Exception as exc:
            blockers.append("{}:latest_snapshot_failed({})".format(compact_symbol(reviewed.get("symbol")), str(exc)[:60]))
            continue
        reviewed["openai_market_context"] = repair_market_context_for_ai(str(reviewed.get("symbol") or ""), dict(reviewed.get("openai_market_context") or {}))
        context_ok, context_reason = market_context_ready_for_ai(dict(reviewed.get("openai_market_context") or {}))
        if not context_ok:
            blockers.append("{}:{}".format(compact_symbol(reviewed.get("symbol")), context_reason))
            continue
        if reviewed.get("candidate_source") == "pending_advice" and reviewed.get("force_openai_recheck"):
            with REVIEW_LOCK:
                tracker = dict(REVIEW_TRACKER.get(str(reviewed.get("symbol") or "")) or {})
            if tracker:
                reviewed = _attach_pending_recheck_context(
                    reviewed,
                    tracker,
                    {"ready": True, "reason": "manual_recheck_requested"},
                )
        source = str(reviewed.get("candidate_source") or "general").lower()
        candidate_top_pool = general_top
        if source == "short_gainers":
            candidate_top_pool = short_gainers
        elif source == "prebreakout_scanner":
            candidate_top_pool = prebreakout_top
        candidate = build_openai_trade_candidate(
            signal=reviewed,
            market=reviewed.get("market") or {},
            risk_status=STATE.get("risk_status") or {},
            portfolio=build_portfolio_snapshot(),
            top_candidates=list(candidate_top_pool or [])[:1],
            constraints=build_openai_constraints(
                str(reviewed.get("symbol") or ""),
                safe_float(reviewed.get("price"), safe_float(((reviewed.get("openai_market_context") or {}).get("basic_market_data") or {}).get("current_price"), 0.0)),
                candidate_source=str(reviewed.get("candidate_source") or "general"),
            ),
            rank_index=max(safe_int(reviewed.get("rank"), 1) - 1, 0),
        )
        if int((OPENAI_TRADE_STATE or {}).get("api_calls", 0) or 0) == 0:
            candidate["force_recheck"] = True
        if reviewed.get("candidate_source") == "short_gainers":
            candidate["short_gainer_context"] = {
                "change_24h_pct": safe_float(((reviewed.get("openai_market_context") or {}).get("basic_market_data") or {}).get("change_24h_pct"), 0.0),
                "scanner_reason": "top_24h_gainer_followup",
            }
        if source == "prebreakout_scanner":
            candidate["raw_candidate_payload"] = _strip_direction_hints(dict(reviewed.get("prebreakout_raw_candidate_payload") or {}))
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
            if reviewed.get("candidate_source") == "pending_advice" and reviewed.get("force_openai_recheck"):
                with REVIEW_LOCK:
                    tracker_row = dict(REVIEW_TRACKER.get(str(reviewed.get("symbol") or "")) or {})
                    if tracker_row and str(tracker_row.get("status") or "") == "observe":
                        sent_ts = now_ts()
                        tracker_row["last_manual_recheck_sent_ts"] = sent_ts
                        tracker_row["manual_recheck_requested_ts"] = 0.0
                        tracker_row["updated_at"] = tw_now_str()
                        tracker_row["tracking_status"] = "manual_recheck_sent"
                        tracker_row["tracking_reason"] = "manual_recheck_sent_at={}".format(tw_from_ts(sent_ts))
                        REVIEW_TRACKER[str(reviewed.get("symbol") or "")] = tracker_row
            if status in ("below_min_score", "not_ranked"):
                blockers.append("{}:{}".format(compact_symbol(reviewed.get("symbol")), status))
                continue
            if status in (
                "bad_request",
                "auth_error",
                "permission_error",
                "rate_limit",
                "error",
                "missing_api_key",
                "disabled",
                "budget_paused",
            ) and not decision:
                blockers.append("{}:{}".format(compact_symbol(reviewed.get("symbol")), status))
                reviewed["auto_order"] = flatten_openai_result(result)
                reviewed["auto_order"]["will_order"] = False
                reviewed["auto_order"]["order_error"] = str(result.get("error") or status)
                reviewed["auto_order"]["order_error_zh"] = _reason_to_zh(str(status))
                continue
            if status in ("cooldown_active", "global_interval_active", "cached_reuse") and not decision:
                blockers.append("{}:{}".format(compact_symbol(reviewed.get("symbol")), status))
                continue
            if decision and not bool(decision.get("valid", True)):
                blockers.append("{}:invalid_decision".format(compact_symbol(reviewed.get("symbol"))))
                reviewed["auto_order"] = flatten_openai_result(result)
                set_backend_thread("openai", "running", "OpenAI returned invalid decision schema for {}.".format(compact_symbol(reviewed.get("symbol"))))
                AI_PANEL["openai_gate_debug"] = {
                    "attempted": compact_symbol(reviewed.get("symbol")),
                    "status": "invalid_decision",
                    "blockers": blockers[:10],
                    "blockers_zh": _humanize_blockers_zh(blockers[:10]),
                    "forced_first_send": bool(candidate.get("force_recheck", False)),
                }
                refresh_openai_dashboard()
                _advance_cursor(reviewed_candidates_all, str(reviewed.get("symbol") or ""))
                return str(reviewed.get("symbol") or "")
            if decision:
                _apply_openai_trade_plan_to_signal(reviewed, decision, result)
                action = apply_review_tracker(reviewed, result)
                reviewed["auto_order"] = flatten_openai_result(result)
                if action == "enter":
                    order_result = place_order_from_signal(reviewed)
                    reviewed["openai_order_result"] = dict(order_result or {})
                    if bool(order_result.get("ok", False)):
                        set_backend_thread(
                            "openai",
                            "running",
                            "Order {} {} for {}".format(
                                "pending" if bool(order_result.get("pending", False)) else "placed",
                                reviewed.get("side"),
                                compact_symbol(reviewed.get("symbol")),
                            ),
                        )
                    if not bool(order_result.get("ok", False)):
                        reviewed["auto_order"]["will_order"] = False
                        reviewed["auto_order"]["order_error"] = str(order_result.get("error") or "order_failed")
                        reviewed["auto_order"]["order_error_zh"] = _reason_to_zh(str(order_result.get("error") or "order_failed"))
                        order_error = str(order_result.get("error") or "")
                        if any(
                            order_error.startswith(prefix)
                            for prefix in ("max_positions", "max_same_direction", "already_active")
                        ):
                            _mark_capacity_block(reviewed, order_error, ORDER_BLOCK_EXPIRY_SEC)
                        append_trade_history(
                            {
                                "time": tw_now_str(),
                                "symbol": reviewed.get("symbol"),
                                "side": "order_rejected_{}".format(reviewed.get("side") or ""),
                                "price": safe_float(reviewed.get("price"), 0.0),
                                "score": safe_float(reviewed.get("score"), 0.0),
                                "stop_loss": safe_float(reviewed.get("stop_loss"), 0.0),
                                "take_profit": safe_float(reviewed.get("take_profit"), 0.0),
                                "pnl_pct": None,
                                "decision_source": "openai_order_reject",
                                "candidate_source": str(reviewed.get("candidate_source") or "general"),
                                "error": str(order_result.get("error") or "order_failed")[:200],
                            }
                        )
                        set_backend_thread(
                            "openai",
                            "running",
                            "Order rejected {}: {}".format(compact_symbol(reviewed.get("symbol")), str(order_result.get("error") or "order_failed")[:140]),
                        )
            else:
                blockers.append("{}:empty_decision".format(compact_symbol(reviewed.get("symbol"))))
                continue
            source_lower = str(reviewed.get("candidate_source") or "").strip().lower()
            target_pool: List[Dict[str, Any]] = general_top
            if source_lower == "short_gainers":
                target_pool = short_gainers
            elif source_lower == "prebreakout_scanner":
                target_pool = prebreakout_top
            for idx, row in enumerate(target_pool):
                if row.get("symbol") == reviewed.get("symbol"):
                    target_pool[idx] = reviewed
                    break
            set_backend_thread("openai", "running", "OpenAI status: {}".format(status or "unknown"))
            AI_PANEL["openai_gate_debug"] = {
                "attempted": compact_symbol(reviewed.get("symbol")),
                "status": status,
                "blockers": blockers[:10],
                "blockers_zh": _humanize_blockers_zh(blockers[:10]),
                "forced_first_send": bool(candidate.get("force_recheck", False)),
            }
            refresh_openai_dashboard()
            _advance_cursor(reviewed_candidates_all, str(reviewed.get("symbol") or ""))
            return str(reviewed.get("symbol") or "")
        except Exception as exc:
            refresh_openai_dashboard()
            set_backend_thread("openai", "crashed", "OpenAI sync failed.", error=str(exc))
            _advance_cursor(reviewed_candidates_all, str(reviewed.get("symbol") or ""))
            return str(reviewed.get("symbol") or "")
    failed_send_markers = ("bad_request", "auth_error", "permission_error", "rate_limit", "error")
    had_send_failures = any(
        (":" + marker) in str(item or "")
        for item in list(blockers or [])
        for marker in failed_send_markers
    )
    final_status = "send_failed" if had_send_failures else "no_candidate_sent"
    AI_PANEL["openai_gate_debug"] = {
        "attempted": "",
        "status": final_status,
        "blockers": blockers[:10],
        "blockers_zh": _humanize_blockers_zh(blockers[:10]),
        "forced_first_send": False,
    }
    if had_send_failures:
        set_backend_thread("openai", "running", "OpenAI request failed this cycle. {}".format(" | ".join(blockers[:3]) if blockers else ""))
    else:
        set_backend_thread("openai", "running", "No eligible OpenAI candidate this cycle. {}".format(" | ".join(blockers[:3]) if blockers else ""))
    refresh_openai_dashboard()
    _advance_cursor(reviewed_candidates_all, "")
    return ""


def perform_scan_cycle() -> None:
    set_backend_thread("scan", "running", "Fetching markets and scan universe.")
    update_state(scan_progress="Loading Bitget tickers...", _persist=False)
    try:
        markets, tickers, snapshot_note = fetch_exchange_snapshot(force_live=True)
        if snapshot_note:
            set_backend_thread("scan", "running", "Exchange snapshot: {}".format(snapshot_note))
        hidden_symbols = temporarily_hidden_symbols()
        selected_union, volume_universe, short_gainer_universe, watch_universe = build_scan_universe(
            markets,
            tickers,
            excluded_symbols=hidden_symbols,
        )
        occupied_symbol_keys = open_position_symbol_keys() | pending_order_symbol_keys()
        cycle_scan_budget = max(int(SCAN_SYMBOLS_PER_CYCLE or 2), 1)
        cycle_short_universe: List[tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        cycle_volume_universe: List[tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        picked_symbols: set[str] = set()
        for symbol, market, ticker in list(short_gainer_universe or []):
            if len(cycle_short_universe) >= cycle_scan_budget:
                break
            if symbol in picked_symbols:
                continue
            if symbol_key(symbol) in occupied_symbol_keys:
                continue
            cycle_short_universe.append((symbol, market, ticker))
            picked_symbols.add(symbol)
        for symbol, market, ticker in list(volume_universe or []):
            if (len(cycle_short_universe) + len(cycle_volume_universe)) >= cycle_scan_budget:
                break
            if symbol in picked_symbols:
                continue
            if symbol_key(symbol) in occupied_symbol_keys:
                continue
            cycle_volume_universe.append((symbol, market, ticker))
            picked_symbols.add(symbol)
        cycle_scan_universe = list(cycle_short_universe) + list(cycle_volume_universe)
        context_map: Dict[str, Dict[str, Any]] = {}
        for idx, (symbol, market, ticker) in enumerate(cycle_scan_universe, start=1):
            update_state(scan_progress="Scanning {}/{} {}".format(idx, len(cycle_scan_universe), compact_symbol(symbol)), _persist=False)
            try:
                context_map[symbol] = build_market_context(symbol, ticker, market)
            except Exception as exc:
                context_map[symbol] = {"basic_market_data": {"current_price": safe_float(ticker.get("last"), 0.0)}, "multi_timeframe": {}, "timeframe_bars": {}}
                update_state(scan_progress="Context degraded for {}: {}".format(compact_symbol(symbol), str(exc)[:90]), _persist=False)
        btc_context = context_map.get("BTC/USDT:USDT")
        general_signals: List[Dict[str, Any]] = []
        for symbol, market, ticker in cycle_volume_universe:
            try:
                general_signals.append(build_signal_from_context(symbol, {"symbol": symbol, "pattern": "general_scan"}, context_map[symbol], btc_context, candidate_source="general"))
            except Exception as exc:
                general_signals.append(build_scan_error_signal(symbol, ticker, exc, "general"))
        short_gainer_signals: List[Dict[str, Any]] = []
        for symbol, market, ticker in cycle_short_universe:
            try:
                context = context_map[symbol]
                signal = build_signal_from_context(
                    symbol,
                    {"symbol": symbol, "pattern": "short_gainers"},
                    context,
                    btc_context,
                    candidate_source="short_gainers",
                )
                change_pct = safe_float(
                    ((context.get("basic_market_data") or {}).get("change_24h_pct")),
                    safe_float(ticker.get("percentage"), 0.0),
                )
                signal["desc"] = "24h 漲幅候選 | change_24h={:.2f}% | 等待 AI 重新推導方向".format(change_pct)
                signal["short_gainer_context"] = {"change_24h_pct": change_pct}
                short_gainer_signals.append(signal)
            except Exception:
                continue
        prebreakout_signals: List[Dict[str, Any]] = []
        prebreakout_leaderboard = {
            "scanner_ts": safe_int(now_ts(), 0),
            "universe": "bitget_usdt_perp",
            "mode": "pre_breakout_disabled_two_symbol_cycle",
            "top_candidates": [],
        }
        pending_advice_signals: List[Dict[str, Any]] = []
        for symbol, market, ticker in watch_universe:
            try:
                tracker = dict(REVIEW_TRACKER.get(symbol) or {})
                if str(tracker.get("status") or "") != "observe":
                    continue
                expires_ts = observe_expire_ts(tracker)
                if now_ts() >= expires_ts:
                    continue
                manual_requested_ts = safe_float(tracker.get("manual_recheck_requested_ts"), 0.0)
                last_manual_sent_ts = safe_float(tracker.get("last_manual_recheck_sent_ts"), 0.0)
                if manual_requested_ts <= last_manual_sent_ts:
                    continue
                if symbol not in context_map:
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
                signal["force_openai_recheck"] = True
                pending_advice_signals.append(signal)
            except Exception as exc:
                update_state(scan_progress="Pending advice tracking degraded for {}: {}".format(compact_symbol(symbol), str(exc)[:90]), _persist=False)
        hidden_symbols = temporarily_hidden_symbols()
        general_candidates = [row for row in list(general_signals or []) if str(row.get("symbol") or "") not in hidden_symbols]
        short_gainer_candidates = [row for row in list(short_gainer_signals or []) if str(row.get("symbol") or "") not in hidden_symbols]
        prebreakout_candidates = [row for row in list(prebreakout_signals or []) if str(row.get("symbol") or "") not in hidden_symbols]

        general_top = diversified_selection(general_candidates, GENERAL_TOP_PICK)
        short_gainer_top = sorted(
            short_gainer_candidates,
            key=lambda row: safe_float(((row.get("openai_market_context") or {}).get("basic_market_data") or {}).get("change_24h_pct"), 0.0),
            reverse=True,
        )[:SHORT_GAINER_TOP_PICK]
        for idx, row in enumerate(general_top, start=1):
            row["rank"] = idx
        for idx, row in enumerate(short_gainer_top, start=1):
            row["rank"] = idx
        short_symbols = {str(row.get("symbol") or "") for row in short_gainer_top if row.get("symbol")}
        if short_symbols:
            general_top = [row for row in list(general_top or []) if str(row.get("symbol") or "") not in short_symbols]
            for idx, row in enumerate(general_top, start=1):
                row["rank"] = idx
        prebreakout_top = sorted(
            list(prebreakout_candidates or []),
            key=lambda row: (
                safe_int(row.get("rank"), 9999),
                -safe_float(row.get("priority_score"), 0.0),
            ),
        )[:PREBREAKOUT_TOP_PICK]
        for idx, row in enumerate(prebreakout_top, start=1):
            row["rank"] = idx
        prebreakout_symbols = {str(row.get("symbol") or "") for row in prebreakout_top if row.get("symbol")}
        if prebreakout_symbols:
            general_top = [row for row in list(general_top or []) if str(row.get("symbol") or "") not in prebreakout_symbols]
            for idx, row in enumerate(general_top, start=1):
                row["rank"] = idx
        manage_pending_limit_orders()
        cycle_symbol_pool = {
            str((row or {}).get("symbol") or "")
            for row in (list(general_top or []) + list(short_gainer_top or []) + list(prebreakout_top or []) + list(pending_advice_signals or []))
            if str((row or {}).get("symbol") or "")
        }
        send_round_limit = max(int(OPENAI_TRADE_CONFIG.get("sends_per_scan", 2) or 2), 1)
        sent_symbols_this_cycle: set[str] = set()
        for _ in range(send_round_limit):
            sent_symbol = maybe_run_openai(
                general_top,
                short_gainer_top,
                prebreakout_top,
                pending_advice_signals,
                skip_symbols=sent_symbols_this_cycle,
            )
            if not sent_symbol:
                break
            sent_symbols_this_cycle.add(str(sent_symbol or ""))
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
            "prebreakout": [row.get("symbol") for row in prebreakout_top],
            "timeframes": list(TIMEFRAMES),
            "last_update": tw_now_str(),
        }
        AUTO_BACKTEST_STATE["scanned_markets"] = len(general_top) + len(short_gainer_top) + len(prebreakout_top)
        AUTO_BACKTEST_STATE["target_count"] = len(cycle_scan_universe)
        AUTO_BACKTEST_STATE["db_symbols"] = (
            [row.get("symbol") for row in general_top]
            + [row.get("symbol") for row in short_gainer_top]
            + [row.get("symbol") for row in prebreakout_top]
        )
        AUTO_BACKTEST_STATE["db_last_update"] = tw_now_str()
        update_state(
            top_signals=general_top,
            general_top_signals=general_top,
            short_gainer_signals=short_gainer_top,
            prebreakout_signals=prebreakout_top,
            prebreakout_leaderboard=prebreakout_leaderboard,
            pending_advice_signals=pending_advice_signals[:10],
            scan_progress="Scan complete. {} general / {} short-gainer / {} pre-breakout symbols ready.".format(
                len(general_top),
                len(short_gainer_top),
                len(prebreakout_top),
            ),
            _force_persist=True,
        )
        update_market_overview(general_top or prebreakout_top or short_gainer_top)
        refresh_learning_summary()
        update_watchlist_state()
        set_backend_thread(
            "scan",
            "running",
            "Updated {} ranked symbols, {} short-gainer candidates, and {} pre-breakout candidates.".format(
                len(general_top),
                len(short_gainer_top),
                len(prebreakout_top),
            ),
        )
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
        try:
            manage_pending_limit_orders()
        except Exception as exc:
            set_backend_thread("positions", "running", "Pending order sync degraded.", error=str(exc))
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
        force=bool(flask_request.args.get("force")),
    )
    compact = str(flask_request.args.get("compact") or "").strip().lower() in {"1", "true", "yes", "on"}
    if compact:
        payload = compact_state_lite_payload(payload)
    return jsonify(payload)


@app.route("/api/positions_state")
def api_positions_state():
    start_background_workers()
    payload = positions_cache.get_or_build(
        lambda: build_positions_payload(dict(STATE)),
        force=bool(flask_request.args.get("force")),
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
        force=bool(flask_request.args.get("force")),
    )
    compact = str(flask_request.args.get("compact") or "").strip().lower() in {"1", "true", "yes", "on"}
    if compact:
        payload = compact_ai_panel_payload(payload)
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


@app.route("/api/openai_manual_send", methods=["POST"])
def api_openai_manual_send():
    start_background_workers()
    payload = flask_request.get_json(silent=True) or {}
    symbol = str(payload.get("symbol") or "").strip()
    if not symbol:
        return jsonify({"ok": False, "message": "Missing symbol."}), 400
    now = now_ts()
    with REVIEW_LOCK:
        row = dict(REVIEW_TRACKER.get(symbol) or {})
        if not row or str(row.get("status") or "") != "observe":
            return jsonify({"ok": False, "message": "Symbol is not in observe list."}), 404
        expires_ts = observe_expire_ts(row, now)
        if now >= expires_ts:
            if symbol in REVIEW_TRACKER:
                del REVIEW_TRACKER[symbol]
            update_watchlist_state()
            sync_openai_pending_advice()
            persist_runtime_snapshot_throttled(force=True)
            return jsonify({"ok": False, "message": "Observe item expired and removed."}), 409
        row["manual_recheck_requested_ts"] = now
        row["tracking_status"] = "manual_recheck_requested"
        row["tracking_reason"] = "manual_send_requested_at={}".format(tw_from_ts(now))
        row["updated_at"] = tw_now_str()
        row["observe_expires_ts"] = expires_ts
        REVIEW_TRACKER[symbol] = row
    update_watchlist_state()
    sync_openai_pending_advice()
    persist_runtime_snapshot_throttled(force=True)
    return jsonify({"ok": True, "message": "Manual OpenAI resend queued for {}.".format(compact_symbol(symbol)), "expires_at": tw_from_ts(expires_ts)})


@app.route("/api/cancel_fvg_order", methods=["POST"])
def api_cancel_fvg_order():
    payload = flask_request.get_json(silent=True) or {}
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
    payload = flask_request.get_json(silent=True) or {}
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
    try:
        result = reduce_position(symbol, side, contracts, "manual_close_all")
        if not result.get("ok"):
            return {"symbol": symbol, "ok": False, "error": str(result.get("error") or "close_failed")[:220]}
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
        return {"symbol": symbol, "ok": True, "order_id": str(result.get("order_id") or "")}
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


@app.route("/api/debug/bitget_diag", methods=["GET"])
def api_bitget_diag():
    start_background_workers()
    api_key_live = str(env_or_blank("BITGET_API_KEY", "") or "")
    passphrase_live = str(
        env_or_blank("BITGET_PASSWORD", "")
        or env_or_blank("BITGET_PASSPHRASE", "")
        or env_or_blank("PASSWORD", "")
        or ""
    )
    report: Dict[str, Any] = {
        "mode": BITGET_API_MODE,
        "copy_mode_enabled": bool(BITGET_USE_COPY_TRADER_API),
        "product_type": BITGET_PRODUCT_TYPE,
        "copy_product_type": BITGET_COPY_PRODUCT_TYPE,
        "margin_coin": BITGET_MARGIN_COIN,
        "margin_mode": BITGET_MARGIN_MODE,
        "order_product_types": _bitget_order_product_types(),
        "copy_fallback_to_swap": bool(BITGET_COPY_PERMISSION_FALLBACK_TO_SWAP),
        "copy_strict_only": bool(BITGET_COPY_STRICT_ONLY),
        "account_name_hint": BITGET_ACCOUNT_NAME_HINT,
        "key_tail": _mask_secret_tail(api_key_live, keep=8),
        "passphrase_present": bool(passphrase_live),
        "passphrase_len": len(passphrase_live),
        "checks": [],
    }
    def _push(name: str, ok: bool, detail: str = "") -> None:
        report["checks"].append({"name": name, "ok": bool(ok), "detail": str(detail or "")[:240]})
    try:
        auth = _bitget_authorities_snapshot(force=True)
        authorities = sorted([str(x) for x in list(auth.get("authorities") or set()) if str(x)])
        trader_type = str(auth.get("trader_type") or "")
        report["authorities"] = authorities
        report["trader_type"] = trader_type
        _push("spot_account_info", True, "traderType={} authorities={}".format(trader_type or "-", ",".join(authorities[:14])))
    except Exception as exc:
        _push("spot_account_info", False, str(exc))
    try:
        total = _bitget_copy_total_equity_usdt()
        _push("copy_order_total_detail", True, "totalEquity={}".format(total))
    except Exception as exc:
        _push("copy_order_total_detail", False, str(exc))
    try:
        rows = _bitget_copy_fetch_current_tracks()
        _push("copy_current_track", True, "count={}".format(len(rows)))
    except Exception as exc:
        _push("copy_current_track", False, str(exc))
    try:
        balance = get_balance()
        _push("copy_balance", True, "equity={} available={}".format(balance.get("equity"), balance.get("available")))
    except Exception as exc:
        _push("copy_balance", False, str(exc))
    try:
        pos = get_positions()
        _push(
            "copy_positions_2way",
            True,
            "long={} short={}".format(
                safe_float((pos.get("long") or {}).get("size"), 0.0),
                safe_float((pos.get("short") or {}).get("size"), 0.0),
            ),
        )
    except Exception as exc:
        _push("copy_positions_2way", False, str(exc))
    try:
        live_positions = _fetch_standard_swap_positions()
        _push("ccxt_swap_positions", True, "count={}".format(len(live_positions)))
    except Exception as exc:
        _push("ccxt_swap_positions", False, str(exc))
    return jsonify(report)


if __name__ == "__main__":
    start_background_workers()
    app.run(host="0.0.0.0", port=int(env_or_blank("PORT", "5000") or 5000))

