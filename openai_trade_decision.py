from __future__ import annotations

import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Tuple

import requests

from bot_runtime_utils import atomic_json_load, atomic_json_save
from kline_payload_preprocessor import apply_kline_preprocessing_to_payload


def _env_bool(env_getter: Callable[[str, str], str], name: str, default: bool = False) -> bool:
    raw = str(env_getter(name, '1' if default else '0') or '').strip().lower()
    return raw in ('1', 'true', 'yes', 'on')


def _env_int(env_getter: Callable[[str, str], str], name: str, default: int) -> int:
    try:
        return int(float(env_getter(name, str(default)) or default))
    except Exception:
        return int(default)


def _env_float(env_getter: Callable[[str, str], str], name: str, default: float) -> float:
    try:
        return float(env_getter(name, str(default)) or default)
    except Exception:
        return float(default)


def default_trade_config(env_getter: Callable[[str, str], str]) -> Dict[str, Any]:
    monthly_budget_twd = max(_env_float(env_getter, 'OPENAI_TRADE_MONTHLY_BUDGET_TWD', 1000.0), 50.0)
    soft_ratio = min(max(_env_float(env_getter, 'OPENAI_TRADE_SOFT_BUDGET_RATIO', 0.85), 0.1), 1.0)
    hard_ratio = min(max(_env_float(env_getter, 'OPENAI_TRADE_HARD_BUDGET_RATIO', 0.95), soft_ratio), 1.0)
    return {
        'enabled': _env_bool(env_getter, 'OPENAI_TRADE_ENABLE', True),
        'model': str(env_getter('OPENAI_TRADE_MODEL', 'gpt-5.4-mini') or 'gpt-5.4-mini').strip(),
        'upgrade_model': str(env_getter('OPENAI_TRADE_UPGRADE_MODEL', 'gpt-5.4') or 'gpt-5.4').strip(),
        'fallback_model': str(env_getter('OPENAI_TRADE_FALLBACK_MODEL', 'gpt-5.4-mini') or 'gpt-5.4-mini').strip(),
        'allow_upgrade_model': _env_bool(env_getter, 'OPENAI_TRADE_ALLOW_UPGRADE', False),
        'monthly_budget_twd': monthly_budget_twd,
        'soft_budget_twd': round(monthly_budget_twd * soft_ratio, 2),
        'hard_budget_twd': round(monthly_budget_twd * hard_ratio, 2),
        'per_call_budget_twd': min(max(_env_float(env_getter, 'OPENAI_TRADE_PER_CALL_BUDGET_TWD', 0.20), 0.05), 2.0),
        'usd_to_twd': max(_env_float(env_getter, 'OPENAI_TRADE_USD_TO_TWD', 32.0), 1.0),
        'input_price_per_1m_usd': max(_env_float(env_getter, 'OPENAI_TRADE_PRICE_INPUT_PER_1M_USD', 0.15), 0.0),
        'output_price_per_1m_usd': max(_env_float(env_getter, 'OPENAI_TRADE_PRICE_OUTPUT_PER_1M_USD', 0.60), 0.0),
        'cached_input_price_per_1m_usd': max(_env_float(env_getter, 'OPENAI_TRADE_PRICE_CACHED_INPUT_PER_1M_USD', 0.015), 0.0),
        'top_k_per_scan': max(_env_int(env_getter, 'OPENAI_TRADE_TOP_K', 10), 1),
        'sends_per_scan': max(_env_int(env_getter, 'OPENAI_TRADE_SENDS_PER_SCAN', 1), 1),
        'cooldown_minutes': max(_env_int(env_getter, 'OPENAI_TRADE_SYMBOL_COOLDOWN_MINUTES', 20), 1),
        'same_payload_reuse_minutes': max(_env_int(env_getter, 'OPENAI_TRADE_SAME_PAYLOAD_REUSE_MINUTES', 180), 1),
        'global_min_interval_minutes': max(_env_int(env_getter, 'OPENAI_TRADE_GLOBAL_MIN_INTERVAL_MINUTES', 20), 0),
        'min_score_abs': max(_env_float(env_getter, 'OPENAI_TRADE_MIN_SCORE', 40.0), 0.0),
        'min_margin_pct': min(max(_env_float(env_getter, 'OPENAI_TRADE_MIN_MARGIN_PCT', 0.03), 0.005), 0.5),
        'max_margin_pct': min(max(_env_float(env_getter, 'OPENAI_TRADE_MAX_MARGIN_PCT', 0.08), 0.01), 0.8),
        'min_leverage': max(_env_int(env_getter, 'OPENAI_TRADE_MIN_LEVERAGE', 4), 1),
        'max_leverage': max(_env_int(env_getter, 'OPENAI_TRADE_MAX_LEVERAGE', 25), 1),
        'max_output_tokens': max(_env_int(env_getter, 'OPENAI_TRADE_MAX_OUTPUT_TOKENS', 560), 280),
        'request_timeout_sec': max(_env_float(env_getter, 'OPENAI_TRADE_TIMEOUT_SEC', 18.0), 5.0),
        'max_decision_latency_sec': max(_env_float(env_getter, 'OPENAI_TRADE_MAX_DECISION_LATENCY_SEC', 36.0), 12.0),
        'temperature': _clamp(_env_float(env_getter, 'OPENAI_TRADE_TEMPERATURE', 0.0), 0.0, 0.4),
        'base_url': str(env_getter('OPENAI_RESPONSES_URL', 'https://api.openai.com/v1/responses') or 'https://api.openai.com/v1/responses').strip(),
        'reasoning_effort': str(env_getter('OPENAI_TRADE_REASONING_EFFORT', 'none') or 'none').strip(),
        'retry_reasoning_effort': str(env_getter('OPENAI_TRADE_RETRY_REASONING_EFFORT', 'none') or 'none').strip(),
        'empty_retry_enabled': _env_bool(env_getter, 'OPENAI_TRADE_EMPTY_RETRY_ENABLE', True),
        'advice_ttl_minutes': max(_env_int(env_getter, 'OPENAI_TRADE_ADVICE_TTL_MINUTES', 240), 15),
    }


def _month_key(now_ts: float | None = None) -> str:
    now = datetime.fromtimestamp(float(now_ts or time.time()), tz=timezone.utc)
    return now.strftime('%Y-%m')


def _new_state(now_ts: float | None = None) -> Dict[str, Any]:
    return {
        'schema_version': 1,
        'month_key': _month_key(now_ts),
        'spent_estimated_usd': 0.0,
        'spent_estimated_twd': 0.0,
        'api_calls': 0,
        'input_tokens': 0,
        'output_tokens': 0,
        'cached_input_tokens': 0,
        'last_consulted_ts': 0.0,
        'last_top_candidates_signature': '',
        'symbols': {},
        'pending_advice': {},
        'recent_decisions': [],
        'last_error': '',
        'updated_at': '',
    }


def load_trade_state(path: str, now_ts: float | None = None) -> Dict[str, Any]:
    state = atomic_json_load(path, None)
    if not isinstance(state, dict):
        return _new_state(now_ts)
    base = _new_state(now_ts)
    for key, value in base.items():
        state.setdefault(key, value)
    if str(state.get('month_key') or '') != _month_key(now_ts):
        fresh = _new_state(now_ts)
        fresh['symbols'] = dict(state.get('symbols') or {})
        fresh['pending_advice'] = dict(state.get('pending_advice') or {})
        fresh['recent_decisions'] = list(state.get('recent_decisions') or [])[:20]
        fresh['last_error'] = str(state.get('last_error') or '')
        return fresh
    return state


def save_trade_state(path: str, state: Dict[str, Any]) -> None:
    atomic_json_save(path, state or {})


def _round(value: Any, digits: int = 4) -> float:
    try:
        return round(float(value or 0.0), digits)
    except Exception:
        return 0.0


def _short_text(value: Any, limit: int = 180) -> str:
    return str(value or '').replace('\n', ' ').strip()[:max(int(limit), 1)]


def base_asset(symbol: str) -> str:
    text = str(symbol or '').upper()
    if '/' in text:
        text = text.split('/')[0]
    if ':' in text:
        text = text.split(':')[0]
    for suffix in ('USDT', 'USD', 'PERP', '-SWAP', '_SWAP'):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text.strip('-_ ')


def _compact_number(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    try:
        number = float(value)
    except Exception:
        return value
    if not math.isfinite(number):
        return 0.0
    if abs(number) >= 100000:
        return round(number, 0)
    if abs(number) >= 1000:
        return round(number, 1)
    if abs(number) >= 1:
        return round(number, 3)
    return round(number, 5)


def _string_schema(max_length: int, min_length: int = 0) -> Dict[str, Any]:
    schema = {'type': 'string', 'maxLength': max(int(max_length or 1), 1)}
    if int(min_length or 0) > 0:
        schema['minLength'] = int(min_length)
    return schema


def _string_array_schema(max_items: int, item_max_length: int) -> Dict[str, Any]:
    return {
        'type': 'array',
        'maxItems': max(int(max_items or 1), 1),
        'items': _string_schema(item_max_length),
    }


def _compact_mapping(data: Dict[str, Any], keys: list[str], *, text_limit: int = 160) -> Dict[str, Any]:
    src = dict(data or {})
    out: Dict[str, Any] = {}
    for key in keys:
        if key not in src:
            continue
        value = src.get(key)
        if isinstance(value, (int, float, bool)) or value is None:
            out[key] = _compact_number(value)
        elif isinstance(value, list):
            compact_list = []
            for item in value[:5]:
                if isinstance(item, (int, float, bool)) or item is None:
                    compact_list.append(_compact_number(item))
                else:
                    compact_list.append(_short_text(item, text_limit))
            out[key] = compact_list
        elif isinstance(value, dict):
            out[key] = {
                str(k): (_compact_number(v) if isinstance(v, (int, float, bool)) or v is None else _short_text(v, text_limit))
                for k, v in list(value.items())[:10]
            }
        else:
            out[key] = _short_text(value, text_limit)
    return out


def _clamp(value: Any, low: float, high: float) -> float:
    try:
        v = float(value)
    except Exception:
        v = float(low)
    return max(float(low), min(float(high), v))


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        try:
            v = float(value)
            return v if math.isfinite(v) else float(default)
        except Exception:
            return float(default)
    text = str(value or '').strip()
    if not text:
        return float(default)
    try:
        v = float(text)
        return v if math.isfinite(v) else float(default)
    except Exception:
        pass
    m = re.search(r'[-+]?\d+(?:\.\d+)?', text)
    if m:
        try:
            v = float(m.group(0))
            return v if math.isfinite(v) else float(default)
        except Exception:
            pass
    return float(default)


def _single_watch_path(text: Any) -> str:
    raw = str(text or '').strip()
    if not raw:
        return ''
    parts = re.split(r'\s+or\s+|(?:锛寍,)\s*or\s+|鎴栬€厊 鎴?|锛屾垨', raw, maxsplit=1, flags=re.IGNORECASE)
    return str(parts[0] or '').strip()


def _cn_phrase(text: Any) -> str:
    raw = str(text or '').strip()
    if not raw:
        return ''
    lowered = raw.lower()
    mapping = {
        'market does not have trend': '市場目前沒有明確趨勢，偏向震盪或混沌。',
        'market currently has trend': '市場目前有趨勢。',
        'market is transitioning': '市場目前處於轉折或過渡階段。',
        'wait for breakout': '等待突破確認後再執行。',
        'wait for pullback': '等待回踩確認後再執行。',
        'enter now': '目前可直接執行進場。',
        'avoid near term': '近期不適合進場。',
    }
    return mapping.get(lowered, raw)


def _hash_payload(payload: Dict[str, Any]) -> str:
    try:
        packed = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    except Exception:
        packed = str(payload)
    return hashlib.sha256(packed.encode('utf-8')).hexdigest()


def _clean_numeric_mapping(src: Dict[str, Any], keys: list[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in keys:
        if key not in src:
            continue
        value = src.get(key)
        if isinstance(value, (int, float, bool)) or value is None:
            out[key] = _compact_number(value)
            continue
        if isinstance(value, list):
            compact_list = []
            for item in list(value)[:12]:
                if isinstance(item, (int, float, bool)) or item is None:
                    compact_list.append(_compact_number(item))
            if compact_list:
                out[key] = compact_list
            continue
        if isinstance(value, dict):
            compact_dict = {}
            for sub_key, sub_value in list(value.items())[:24]:
                if isinstance(sub_value, (int, float, bool)) or sub_value is None:
                    compact_dict[str(sub_key)] = _compact_number(sub_value)
            if compact_dict:
                out[key] = compact_dict
    return out


def _ensure_keys(target: Dict[str, Any], keys: list[str], default: Any = None) -> Dict[str, Any]:
    out = dict(target or {})
    for key in keys:
        if key not in out:
            out[key] = default
            continue
        value = out.get(key)
        if value is None:
            out[key] = default
            continue
        if isinstance(value, str) and not str(value).strip():
            out[key] = default
    return out


def _clean_levels(levels: Dict[str, Any]) -> Dict[str, Any]:
    src = dict(levels or {})
    return _clean_numeric_mapping(
        src,
        [
            'nearest_support',
            'nearest_resistance',
            'support_levels',
            'resistance_levels',
            'recent_high',
            'recent_low',
        ],
    )


def _clean_multi_timeframe(multi: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for tf, row in list(dict(multi or {}).items())[:6]:
        clean_row = _clean_numeric_mapping(
            dict(row or {}),
            [
                'n', 'c', 'a', 'r', 'x',
                'e20', 'e50', 'e200', 'm20', 'v',
                'mh', 'bbp', 'vr',
                'hi', 'lo', 'ph', 'pl', 'ch', 'cl',
                'xr',
            ],
        )
        if clean_row:
            out[str(tf)] = clean_row
    return out


def _clean_orderbook_levels(rows: Any, *, max_levels: int = 10) -> list[list[float]]:
    levels: list[list[float]] = []
    for row in list(rows or [])[:max(int(max_levels or 1), 1)]:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        price = _coerce_float(row[0], 0.0)
        size = _coerce_float(row[1], 0.0)
        if price <= 0 or size < 0:
            continue
        levels.append([round(price, 8), round(size, 8)])
    return levels


def _build_clean_payload(
    candidate: Dict[str, Any],
    *,
    logger: Callable[[str], None] | None = None,
) -> Dict[str, Any]:
    src = dict(candidate or {})
    symbol = str(src.get('symbol') or '').strip()
    market_context = dict(src.get('market_context') or {})
    market_state = dict(market_context.get('market_state') or {})
    ticker = _clean_numeric_mapping(
        dict(market_state.get('ticker') or {}),
        ['last', 'bid', 'ask', 'spread_pct', 'mark_price', 'index_price'],
    )
    basic_market_data = _clean_numeric_mapping(
        dict(market_context.get('basic_market_data') or {}),
        [
            'current_price',
            'change_24h_pct',
            'quote_volume_24h',
            'base_volume_24h',
            'market_cap_available',
            'market_cap_usd',
            'fdv_usd',
            'circulating_supply',
            'total_supply',
            'funding_rate',
            'open_interest',
            'open_interest_value_usdt',
            'long_short_ratio',
            'top_trader_long_short_ratio',
            'whale_position_change_pct',
            'mark_price',
            'index_price',
        ],
    )
    basic_symbol = str(((market_context.get('basic_market_data') or {}).get('symbol') or '')).strip()
    if basic_symbol:
        basic_market_data['symbol'] = basic_symbol[:32]
    basic_exchange = str(((market_context.get('basic_market_data') or {}).get('exchange') or '')).strip()
    if basic_exchange:
        basic_market_data['exchange'] = basic_exchange[:24]
    basic_market_type = str(((market_context.get('basic_market_data') or {}).get('market_type') or '')).strip()
    if basic_market_type:
        basic_market_data['market_type'] = basic_market_type[:24]
    basic_market_data = _ensure_keys(
        basic_market_data,
        [
            'symbol',
            'exchange',
            'market_type',
            'current_price',
            'change_24h_pct',
            'quote_volume_24h',
            'base_volume_24h',
            'market_cap_available',
            'market_cap_usd',
            'fdv_usd',
            'circulating_supply',
            'total_supply',
            'funding_rate',
            'open_interest',
            'open_interest_value_usdt',
            'long_short_ratio',
            'top_trader_long_short_ratio',
            'whale_position_change_pct',
            'mark_price',
            'index_price',
        ],
        None,
    )
    levels = _clean_levels(dict(src.get('levels') or market_context.get('levels') or {}))
    multi_timeframe = _clean_multi_timeframe(dict(src.get('multi_timeframe') or market_context.get('multi_timeframe') or {}))
    timeframe_bars = _compact_timeframe_bars(
        dict(src.get('timeframe_bars') or market_context.get('timeframe_bars') or {}),
        symbol=symbol,
        logger=logger,
    )
    liquidity_context = _clean_numeric_mapping(
        dict(src.get('liquidity_context') or market_context.get('liquidity_context') or {}),
        [
            'spread_pct',
            'bid_depth_5',
            'ask_depth_5',
            'bid_depth_10',
            'ask_depth_10',
            'depth_imbalance_10',
            'largest_bid_wall_price',
            'largest_bid_wall_size',
            'largest_ask_wall_price',
            'largest_ask_wall_size',
            'aggressive_buy_volume',
            'aggressive_sell_volume',
            'aggressive_buy_notional',
            'aggressive_sell_notional',
            'buy_sell_notional_ratio',
            'cvd_notional',
            'volume_anomaly_5m',
            'volume_anomaly_15m',
        ],
    )
    derivatives_context = _clean_numeric_mapping(
        dict(src.get('derivatives_context') or market_context.get('derivatives_context') or {}),
        [
            'funding_rate',
            'open_interest',
            'open_interest_value_usdt',
            'open_interest_change_pct_5m',
            'long_short_ratio',
            'top_trader_long_short_ratio',
            'whale_position_change_pct',
            'basis_pct',
            'liquidation_volume_24h',
            'leverage_heat_score',
            'mark_price',
            'index_price',
        ],
    )
    calculated_metrics = _clean_numeric_mapping(
        dict(src.get('calculated_metrics') or market_context.get('calculated_metrics') or {}),
        [
            'atr_15m',
            'atr_5m',
            'atr_1h',
            'distance_to_support_atr15',
            'distance_to_resistance_atr15',
            'entry_spread_cost_pct',
            'recent_volume_vs_avg_15m',
            'recent_range_vs_atr_15m',
        ],
    )
    raw_orderbook_src = dict(src.get('raw_orderbook_snapshot') or market_context.get('raw_orderbook_snapshot') or {})
    raw_orderbook_snapshot = {
        'bids': _clean_orderbook_levels(raw_orderbook_src.get('bids'), max_levels=10),
        'asks': _clean_orderbook_levels(raw_orderbook_src.get('asks'), max_levels=10),
    }
    recent_trades = _clean_numeric_mapping(
        dict(src.get('recent_trades') or market_context.get('recent_trades') or {}),
        [
            'window_sec',
            'buy_notional',
            'sell_notional',
            'trade_count',
            'large_trade_threshold_usdt',
            'large_buy_count',
            'large_sell_count',
        ],
    )
    derivatives_src = dict(src.get('derivatives_context') or market_context.get('derivatives_context') or {})
    leverage_heat = str(derivatives_src.get('leverage_heat') or '').strip()
    if leverage_heat:
        derivatives_context['leverage_heat'] = leverage_heat[:24]
    liquidation_map_status = str(derivatives_src.get('liquidation_map_status') or '').strip()
    if liquidation_map_status:
        derivatives_context['liquidation_map_status'] = liquidation_map_status[:24]
    risk: Dict[str, Any] = {}
    portfolio_src = dict(src.get('portfolio') or {})
    held_symbols: list[str] = []
    for key in ('held_symbols', 'position_symbols', 'open_symbols'):
        for item in list(portfolio_src.get(key) or []):
            token = str(item or '').strip()
            if token and token not in held_symbols:
                held_symbols.append(token)
    portfolio: Dict[str, Any] = {
        'held_symbols': held_symbols[:12],
        'held_symbol_count': len(held_symbols[:12]),
    }
    execution_policy = _clean_numeric_mapping(
        dict(src.get('execution_policy') or {}),
        ['fixed_leverage', 'min_order_margin_usdt', 'fixed_order_notional_usdt', 'margin_pct_range'],
    )
    leverage_mode = str((src.get('execution_policy') or {}).get('leverage_mode') or '').strip()
    if leverage_mode:
        execution_policy['leverage_mode'] = leverage_mode[:24]
    constraints = _clean_numeric_mapping(
        dict(src.get('constraints') or {}),
        [
            'min_margin_pct',
            'max_margin_pct',
            'min_leverage',
            'max_leverage',
            'fixed_leverage',
            'min_order_margin_usdt',
            'fixed_order_notional_usdt',
            'max_open_positions',
            'max_same_direction',
        ],
    )
    leverage_policy = str((src.get('constraints') or {}).get('leverage_policy') or '').strip()
    trade_style = str(src.get('trade_style') or (src.get('constraints') or {}).get('trade_style') or 'short_term_intraday').strip() or 'short_term_intraday'
    if leverage_policy:
        constraints['leverage_policy'] = leverage_policy[:40]
    constraints['trade_style'] = trade_style
    current_price = _coerce_float(src.get('current_price', basic_market_data.get('current_price', 0)), 0.0)
    raw_candidate_payload = dict(src.get('raw_candidate_payload') or {})
    raw_candidate_payload_clean: Dict[str, Any] = {}
    if raw_candidate_payload:
        raw_candidate_payload_clean = {
            'ticker_24h': _clean_numeric_mapping(
                dict(raw_candidate_payload.get('ticker_24h') or {}),
                ['lastPrice', 'priceChangePercent', 'quoteVolume', 'weightedAvgPrice'],
            ),
            'liquidity_context': _clean_numeric_mapping(
                dict(raw_candidate_payload.get('liquidity_context') or {}),
                [
                    'spread_pct', 'bid_depth_10', 'ask_depth_10', 'depth_imbalance_10',
                    'largest_bid_wall_price', 'largest_bid_wall_size',
                    'largest_ask_wall_price', 'largest_ask_wall_size',
                    'top_5_bid_liquidity', 'top_5_ask_liquidity',
                ],
            ),
            'orderbook_history': _clean_numeric_mapping(
                dict(raw_candidate_payload.get('orderbook_history') or {}),
                [
                    'depth_imbalance_change_1m', 'depth_imbalance_change_5m',
                    'largest_bid_wall_size_change_3m_pct', 'largest_ask_wall_size_change_3m_pct',
                    'bid_wall_following_price', 'ask_wall_getting_thinner', 'wall_pull_or_spoof_risk',
                ],
            ),
            'flow_context': _clean_numeric_mapping(
                dict(raw_candidate_payload.get('flow_context') or {}),
                [
                    'market_buy_notional_1m', 'market_sell_notional_1m',
                    'market_buy_sell_ratio_1m', 'market_buy_sell_ratio_5m',
                    'cvd_notional_1m', 'cvd_notional_5m',
                    'cvd_slope_1m', 'cvd_slope_5m', 'cvd_slope_15m',
                    'large_trade_count_1m',
                ],
            ),
            'derivatives_context': _clean_numeric_mapping(
                dict(raw_candidate_payload.get('derivatives_context') or {}),
                [
                    'funding_rate', 'basis_pct', 'open_interest_value_usdt',
                    'open_interest_change_pct_5m', 'open_interest_change_pct_15m', 'open_interest_change_pct_1h',
                ],
            ),
            'calculated_metrics': _clean_numeric_mapping(
                dict(raw_candidate_payload.get('calculated_metrics') or {}),
                [
                    'atr_15m', 'atr_5m', 'atr_1h',
                    'distance_to_support_atr15', 'distance_to_resistance_atr15',
                    'entry_spread_cost_pct', 'recent_volume_vs_avg_15m', 'recent_range_vs_atr_15m',
                ],
            ),
            'raw_orderbook_snapshot': {
                'bids': _clean_orderbook_levels(((raw_candidate_payload.get('raw_orderbook_snapshot') or {}).get('bids')), max_levels=10),
                'asks': _clean_orderbook_levels(((raw_candidate_payload.get('raw_orderbook_snapshot') or {}).get('asks')), max_levels=10),
            },
            'recent_trades': _clean_numeric_mapping(
                dict(raw_candidate_payload.get('recent_trades') or {}),
                [
                    'window_sec', 'buy_notional', 'sell_notional', 'trade_count',
                    'large_trade_threshold_usdt', 'large_buy_count', 'large_sell_count',
                ],
            ),
            'scanner_features': _clean_numeric_mapping(
                dict(raw_candidate_payload.get('scanner_features') or {}),
                [
                    'bb_width_pct_15m', 'bb_width_percentile_15m_120', 'atr_percentile_15m_120',
                    'range_tightness_atr_15m', 'range_duration_bars_15m', 'inside_bar_count_15m',
                    'higher_lows_count_15m', 'lower_highs_count_15m', 'volume_contraction_ratio_15m',
                    'range_high_15m', 'range_low_15m', 'range_mid_15m', 'close_position_in_range_15m',
                    'distance_to_range_high_atr15m', 'distance_to_range_low_atr15m',
                    'breakout_level', 'breakdown_level', 'distance_to_breakout_atr15m',
                    'distance_to_breakdown_atr15m', 'already_extended_atr15m',
                ],
            ),
            'timeframe_bars': _compact_timeframe_bars(
                dict(raw_candidate_payload.get('timeframe_bars') or {}),
                symbol=symbol,
                logger=logger,
            ),
        }
        raw_symbol = str(raw_candidate_payload.get('symbol') or '').strip()
        if raw_symbol:
            raw_candidate_payload_clean['symbol'] = raw_symbol[:40]
    orderbook_history = _clean_numeric_mapping(
        dict(src.get('orderbook_history') or market_context.get('orderbook_history') or raw_candidate_payload_clean.get('orderbook_history') or {}),
        [
            'depth_imbalance_change_1m', 'depth_imbalance_change_5m',
            'largest_bid_wall_size_change_3m_pct', 'largest_ask_wall_size_change_3m_pct',
            'bid_wall_following_price', 'ask_wall_getting_thinner', 'wall_pull_or_spoof_risk',
        ],
    )
    flow_context = _clean_numeric_mapping(
        dict(src.get('flow_context') or market_context.get('flow_context') or raw_candidate_payload_clean.get('flow_context') or {}),
        [
            'market_buy_notional_1m', 'market_sell_notional_1m',
            'market_buy_sell_ratio_1m', 'market_buy_sell_ratio_5m',
            'cvd_notional_1m', 'cvd_notional_5m',
            'cvd_slope_1m', 'cvd_slope_5m', 'cvd_slope_15m',
            'large_trade_count_1m',
        ],
    )
    clean_payload = {
        'symbol': symbol,
        'trade_style': trade_style,
        'current_price': _compact_number(current_price),
        'market_context': {
            'basic_market_data': basic_market_data,
            'market_state': {'ticker': ticker},
        },
        'levels': levels,
        'multi_timeframe': multi_timeframe,
        'timeframe_bars': timeframe_bars if timeframe_bars else None,
        'liquidity_context': liquidity_context,
        'orderbook_history': orderbook_history,
        'flow_context': flow_context,
        'derivatives_context': derivatives_context,
        'calculated_metrics': calculated_metrics,
        'raw_orderbook_snapshot': raw_orderbook_snapshot,
        'recent_trades': recent_trades,
        'risk': {},
        'portfolio': portfolio,
        'execution_policy': {},
        'constraints': constraints,
        'raw_candidate_payload': raw_candidate_payload_clean if str(src.get('candidate_source') or '').strip().lower() == 'prebreakout_scanner' else {},
        'force_recheck': False,
    }
    return clean_payload


def _stable_payload(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return _finalize_payload_for_send(_build_clean_payload(candidate))


def estimate_cost_usd(config: Dict[str, Any], *, input_tokens: int = 0, output_tokens: int = 0, cached_input_tokens: int = 0) -> float:
    return (
        max(int(input_tokens or 0), 0) / 1_000_000.0 * float(config.get('input_price_per_1m_usd', 0.0) or 0.0)
        + max(int(output_tokens or 0), 0) / 1_000_000.0 * float(config.get('output_price_per_1m_usd', 0.0) or 0.0)
        + max(int(cached_input_tokens or 0), 0) / 1_000_000.0 * float(config.get('cached_input_price_per_1m_usd', 0.0) or 0.0)
    )


def estimate_cost_twd(config: Dict[str, Any], *, input_tokens: int = 0, output_tokens: int = 0, cached_input_tokens: int = 0) -> float:
    usd = estimate_cost_usd(config, input_tokens=input_tokens, output_tokens=output_tokens, cached_input_tokens=cached_input_tokens)
    return usd * float(config.get('usd_to_twd', 32.0) or 32.0)


def _estimate_tokens_from_json(data: Any) -> int:
    try:
        packed = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    except Exception:
        packed = str(data)
    return max(int(len(packed) / 3.8), 1)


def _output_token_cost_twd(config: Dict[str, Any]) -> float:
    usd_per_token = float(config.get('output_price_per_1m_usd', 0.0) or 0.0) / 1_000_000.0
    return usd_per_token * float(config.get('usd_to_twd', 32.0) or 32.0)


def build_candidate_payload(
    *,
    signal: Dict[str, Any],
    market: Dict[str, Any],
    risk_status: Dict[str, Any],
    portfolio: Dict[str, Any],
    top_candidates: list[Dict[str, Any]],
    constraints: Dict[str, Any],
    rank_index: int,
) -> Dict[str, Any]:
    market_context = dict(signal.get('openai_market_context') or {})
    execution_policy = dict((market_context.get('execution_policy') or {}))
    basic_market_data = dict(market_context.get('basic_market_data') or {})
    levels = dict(market_context.get('levels') or {})
    multi_timeframe = dict(market_context.get('multi_timeframe') or {})
    timeframe_bars = dict(market_context.get('timeframe_bars') or {})
    liquidity_context = dict(market_context.get('liquidity_context') or {})
    orderbook_history = dict(
        market_context.get('orderbook_history')
        or (signal.get('prebreakout_raw_candidate_payload') or {}).get('orderbook_history')
        or {}
    )
    flow_context = dict(
        market_context.get('flow_context')
        or signal.get('prebreakout_flow_context')
        or (signal.get('prebreakout_raw_candidate_payload') or {}).get('flow_context')
        or {}
    )
    derivatives_context = dict(market_context.get('derivatives_context') or {})
    calculated_metrics = dict(market_context.get('calculated_metrics') or {})
    raw_orderbook_snapshot = dict(market_context.get('raw_orderbook_snapshot') or {})
    recent_trades = dict(market_context.get('recent_trades') or {})
    source_lower = str(signal.get('candidate_source') or signal.get('source') or 'general').strip().lower()
    held_symbols: list[str] = []
    for key in ('held_symbols', 'position_symbols', 'open_symbols'):
        for item in list((portfolio or {}).get(key) or []):
            token = str(item or '').strip()
            if token and token not in held_symbols:
                held_symbols.append(token)
    raw_candidate_payload = dict(
        signal.get('raw_candidate_payload')
        or signal.get('prebreakout_raw_candidate_payload')
        or {}
    )
    return {
        'symbol': str(signal.get('symbol') or ''),
        'trade_style': str(constraints.get('trade_style') or 'short_term_intraday'),
        'current_price': _compact_number(signal.get('price', basic_market_data.get('current_price'))),
        'market_context': {
            'basic_market_data': _compact_mapping(
                basic_market_data,
                [
                    'symbol',
                    'exchange',
                    'market_type',
                    'current_price',
                    'change_24h_pct',
                    'quote_volume_24h',
                    'base_volume_24h',
                    'market_cap_available',
                    'market_cap_usd',
                    'fdv_usd',
                    'circulating_supply',
                    'total_supply',
                    'funding_rate',
                    'open_interest',
                    'open_interest_value_usdt',
                    'long_short_ratio',
                    'top_trader_long_short_ratio',
                    'whale_position_change_pct',
                    'mark_price',
                    'index_price',
                ],
                text_limit=120,
            ),
            'market_state': {
                'ticker': _compact_mapping(
                    dict(((market_context.get('market_state') or {}).get('ticker') or {})),
                    ['last', 'bid', 'ask', 'spread_pct', 'mark_price', 'index_price'],
                    text_limit=120,
                ),
            },
        },
        'levels': _compact_mapping(levels, ['nearest_support', 'nearest_resistance', 'support_levels', 'resistance_levels', 'recent_high', 'recent_low'], text_limit=120),
        'multi_timeframe': {
            str(tf): _compact_tf_stats(dict(row or {}))
            for tf, row in list(multi_timeframe.items())[:6]
        },
        'timeframe_bars': _compact_timeframe_bars(timeframe_bars),
        'liquidity_context': _compact_mapping(
            liquidity_context,
            [
                'spread_pct',
                'bid_depth_5',
                'ask_depth_5',
                'bid_depth_10',
                'ask_depth_10',
                'depth_imbalance_10',
                'largest_bid_wall_price',
                'largest_bid_wall_size',
                'largest_ask_wall_price',
                'largest_ask_wall_size',
                'recent_trades_count',
                'aggressive_buy_volume',
                'aggressive_sell_volume',
                'aggressive_buy_notional',
                'aggressive_sell_notional',
                'buy_sell_notional_ratio',
                'cvd_notional',
                'volume_anomaly_5m',
                'volume_anomaly_15m',
            ],
            text_limit=140,
        ),
        'orderbook_history': _compact_mapping(
            orderbook_history,
            ['depth_imbalance_change_1m', 'depth_imbalance_change_5m', 'largest_bid_wall_size_change_3m_pct', 'largest_ask_wall_size_change_3m_pct', 'bid_wall_following_price', 'ask_wall_getting_thinner', 'wall_pull_or_spoof_risk'],
            text_limit=140,
        ),
        'flow_context': _compact_mapping(
            flow_context,
            ['market_buy_notional_1m', 'market_sell_notional_1m', 'market_buy_sell_ratio_1m', 'market_buy_sell_ratio_5m', 'cvd_notional_1m', 'cvd_notional_5m', 'cvd_slope_1m', 'cvd_slope_5m', 'cvd_slope_15m', 'large_trade_count_1m'],
            text_limit=140,
        ),
        'derivatives_context': _compact_mapping(
            derivatives_context,
            [
                'funding_rate',
                'open_interest',
                'open_interest_value_usdt',
                'open_interest_change_pct_5m',
                'open_interest_change_pct_15m',
                'open_interest_change_pct_1h',
                'long_short_ratio',
                'top_trader_long_short_ratio',
                'whale_position_change_pct',
                'basis_pct',
                'liquidation_volume_24h',
                'leverage_heat_score',
                'mark_price',
                'index_price',
            ],
            text_limit=140,
        ),
        'calculated_metrics': _compact_mapping(
            calculated_metrics,
            [
                'atr_15m',
                'atr_5m',
                'atr_1h',
                'distance_to_support_atr15',
                'distance_to_resistance_atr15',
                'entry_spread_cost_pct',
                'recent_volume_vs_avg_15m',
                'recent_range_vs_atr_15m',
            ],
            text_limit=120,
        ),
        'raw_orderbook_snapshot': {
            'bids': _clean_orderbook_levels(raw_orderbook_snapshot.get('bids'), max_levels=10),
            'asks': _clean_orderbook_levels(raw_orderbook_snapshot.get('asks'), max_levels=10),
        },
        'recent_trades': _compact_mapping(
            recent_trades,
            [
                'window_sec',
                'buy_notional',
                'sell_notional',
                'trade_count',
                'large_trade_threshold_usdt',
                'large_buy_count',
                'large_sell_count',
            ],
            text_limit=120,
        ),
        'risk': {},
        'portfolio': {
            'held_symbols': held_symbols[:12],
            'held_symbol_count': len(held_symbols[:12]),
        },
        'execution_policy': {},
        'constraints': _compact_mapping(
            dict(constraints or {}),
            [
                'min_margin_pct', 'max_margin_pct', 'min_leverage', 'max_leverage', 'fixed_leverage',
                'leverage_policy', 'min_order_margin_usdt', 'fixed_order_notional_usdt', 'trade_style',
                'max_open_positions', 'max_same_direction',
            ],
            text_limit=100,
        ),
        'raw_candidate_payload': raw_candidate_payload if source_lower == 'prebreakout_scanner' else {},
        'force_recheck': False,
    }


def _compact_news_context(news_context: Dict[str, Any]) -> Dict[str, Any]:
    src = dict(news_context or {})
    items = []
    for row in list(src.get('items') or [])[:3]:
        entry = dict(row or {})
        compact_row = {}
        for key in ('title', 'summary', 'published_at', 'url', 'source', 'coin'):
            if key in entry and entry.get(key) not in (None, ''):
                compact_row[key] = _short_text(entry.get(key), 120 if key != 'url' else 180)
        if compact_row:
            items.append(compact_row)
    return {
        'available': bool(src.get('available', False)),
        'note': _short_text(src.get('note') or '', 180),
        'items': items,
    }


def _compact_tf_stats(row: Dict[str, Any]) -> Dict[str, Any]:
    src = dict(row or {})
    out = {
        'n': int(src.get('bars', 0) or 0),
        'c': _compact_number(src.get('last_close')),
        'a': _compact_number(src.get('atr_pct')),
        'r': _compact_number(src.get('rsi')),
        'x': _compact_number(src.get('adx')),
        'e20': _compact_number(src.get('ema20')),
        'e50': _compact_number(src.get('ema50')),
        'e200': _compact_number(src.get('ema200')),
        'm20': _compact_number(src.get('ma20')),
        'v': _compact_number(src.get('vwap')),
        'mh': _compact_number(src.get('macd_hist')),
        'bbp': _compact_number(src.get('bb_position_pct')),
        'vr': _compact_number(src.get('vol_ratio')),
        'hi': _compact_number(src.get('recent_structure_high')),
        'lo': _compact_number(src.get('recent_structure_low')),
        'ph': _compact_number(src.get('prior_structure_high_6')),
        'pl': _compact_number(src.get('prior_structure_low_6')),
        'ch': _compact_number(src.get('current_bar_high')),
        'cl': _compact_number(src.get('current_bar_low')),
        'xr': bool(src.get('explosive_move', False)),
    }
    return {k: v for k, v in out.items() if v not in ('', None)}


def _compact_pressure_stats(row: Dict[str, Any]) -> Dict[str, Any]:
    src = dict(row or {})
    out = {
        'pp': _compact_number(src.get('pressure_price')),
        'sp': _compact_number(src.get('support_price')),
        'pa': _compact_number(src.get('pressure_distance_atr')),
        'sa': _compact_number(src.get('support_distance_atr')),
        'c20': _compact_number(src.get('close_vs_ema20_pct')),
        'c50': _compact_number(src.get('close_vs_ema50_pct')),
        'vr': _compact_number(src.get('volume_ratio')),
        'hh': int(src.get('hh_count', 0) or 0),
        'hl': int(src.get('hl_count', 0) or 0),
        'lh': int(src.get('lh_count', 0) or 0),
        'll': int(src.get('ll_count', 0) or 0),
    }
    return {k: v for k, v in out.items() if v not in ('', None)}


def _log_timeframe_drop(
    logger: Callable[[str], None] | None,
    *,
    symbol: str,
    timeframe: str,
    reason: str,
    raw_rows: Any,
) -> None:
    if not logger:
        return
    try:
        preview = json.dumps(list(raw_rows or [])[:2], ensure_ascii=False, separators=(',', ':'))
    except Exception:
        preview = str(list(raw_rows or [])[:2])
    logger(
        'clean_payload timeframe_bars removed: symbol={} timeframe={} reason={} rows_head2={}'.format(
            symbol or '',
            timeframe,
            reason,
            preview[:320],
        )
    )


def _parse_time_value(raw: Any) -> int:
    if isinstance(raw, bool):
        return 0
    try:
        value = float(raw)
    except Exception:
        return 0
    if not math.isfinite(value) or value <= 0:
        return 0
    return int(value)


def _normalize_ohlcv_row(raw_row: Any) -> tuple[bool, str, list[float], int]:
    if isinstance(raw_row, dict):
        sequence = [
            raw_row.get('time', raw_row.get('timestamp', raw_row.get('t', 0))),
            raw_row.get('open', raw_row.get('o', 0)),
            raw_row.get('high', raw_row.get('h', 0)),
            raw_row.get('low', raw_row.get('l', 0)),
            raw_row.get('close', raw_row.get('c', 0)),
            raw_row.get('volume', raw_row.get('v', 0)),
        ]
    elif isinstance(raw_row, (list, tuple)):
        sequence = list(raw_row)
    else:
        return False, 'row_not_array', [], 0
    timestamp = 0
    values: list[Any] = []
    if len(sequence) == 5:
        values = list(sequence)
    elif len(sequence) >= 6:
        ts_candidate = _parse_time_value(sequence[0])
        # Some sources put OHLCV first without timestamp. Only treat index-0 as time when it looks like a real epoch value.
        if ts_candidate >= 1_000_000_000:
            timestamp = ts_candidate
            values = list(sequence[1:6])
        else:
            values = list(sequence[0:5])
    else:
        return False, 'row_len_not_5_or_6plus', [], timestamp
    parsed: list[float] = []
    for idx, value in enumerate(values):
        if isinstance(value, bool):
            return False, f'row_value_not_number[{idx}]', [], timestamp
        try:
            number = float(value)
        except Exception:
            return False, f'row_value_not_number[{idx}]', [], timestamp
        if not math.isfinite(number):
            return False, f'row_value_not_finite[{idx}]', [], timestamp
        parsed.append(number)
    op, hi, lo, cl, vol = parsed
    if op <= 0 or hi <= 0 or lo <= 0 or cl <= 0:
        return False, 'ohlc_non_positive', [], timestamp
    if vol < 0:
        return False, 'volume_negative', [], timestamp
    if hi < max(op, cl, lo) or lo > min(op, cl, hi):
        # Repair swapped/dirty high-low while preserving true price envelope.
        fixed_hi = max(op, cl, hi, lo)
        fixed_lo = min(op, cl, hi, lo)
        hi, lo = fixed_hi, fixed_lo
    if hi <= 0 or lo <= 0:
        return False, 'high_low_non_positive_after_repair', [], timestamp
    if hi < max(op, cl, lo):
        return False, 'high_inconsistent_after_repair', [], timestamp
    if lo > min(op, cl, hi):
        return False, 'low_inconsistent_after_repair', [], timestamp
    return True, '', [_compact_number(op), _compact_number(hi), _compact_number(lo), _compact_number(cl), _compact_number(vol)], timestamp


def _extract_timeframe_row_sources(raw_value: Any) -> tuple[list[list[Any]], int, int]:
    sources: list[list[Any]] = []
    explicit_start_ts = 0
    explicit_interval_ms = 0
    if isinstance(raw_value, list):
        sources.append(list(raw_value))
        return sources, explicit_start_ts, explicit_interval_ms
    if not isinstance(raw_value, dict):
        return sources, explicit_start_ts, explicit_interval_ms
    explicit_start_ts = _parse_time_value(raw_value.get('start_ts', raw_value.get('start', 0)))
    explicit_interval_ms = _parse_time_value(raw_value.get('interval_ms', raw_value.get('interval', 0)))
    for key in ('rows', 'candles', 'bars', 'ohlcv', 'data'):
        candidate = raw_value.get(key)
        if isinstance(candidate, list) and candidate:
            sources.append(list(candidate))
    open_col = raw_value.get('open', raw_value.get('o'))
    high_col = raw_value.get('high', raw_value.get('h'))
    low_col = raw_value.get('low', raw_value.get('l'))
    close_col = raw_value.get('close', raw_value.get('c'))
    vol_col = raw_value.get('volume', raw_value.get('v'))
    time_col = raw_value.get('time', raw_value.get('timestamp', raw_value.get('t')))
    if all(isinstance(col, list) for col in (open_col, high_col, low_col, close_col, vol_col)):
        min_len = min(len(open_col), len(high_col), len(low_col), len(close_col), len(vol_col))
        col_rows: list[list[Any]] = []
        for idx in range(min_len):
            tval = time_col[idx] if isinstance(time_col, list) and idx < len(time_col) else 0
            col_rows.append([tval, open_col[idx], high_col[idx], low_col[idx], close_col[idx], vol_col[idx]])
        if col_rows:
            sources.append(col_rows)
    return sources, explicit_start_ts, explicit_interval_ms


def _compact_timeframe_bars(
    timeframe_bars: Dict[str, Any],
    *,
    symbol: str = '',
    logger: Callable[[str], None] | None = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for tf, raw_value in list(dict(timeframe_bars or {}).items())[:6]:
        tf_key = str(tf)
        row_limit = {
            '1m': 120,
            '5m': 120,
            '15m': 120,
            '1h': 120,
            '4h': 120,
            '1d': 90,
        }.get(tf_key, 6)
        row_sources, explicit_start_ts, explicit_interval_ms = _extract_timeframe_row_sources(raw_value)
        if not row_sources:
            _log_timeframe_drop(
                logger,
                symbol=symbol,
                timeframe=tf_key,
                reason='rows_not_found',
                raw_rows=[],
            )
            continue
        cleaned_rows: list[list[float]] = []
        timestamps: list[int] = []
        invalid_reasons: list[str] = []
        sample_rows: list[Any] = []
        for source_idx, source_rows in enumerate(row_sources):
            if not isinstance(source_rows, list) or not source_rows:
                continue
            source_tail = list(source_rows)[-row_limit:]
            if not sample_rows:
                sample_rows = list(source_tail)[:2]
            for row_idx, raw_row in enumerate(source_tail):
                ok, reason, normalized_row, row_ts = _normalize_ohlcv_row(raw_row)
                if not ok:
                    invalid_reasons.append(f'source{source_idx}[{row_idx}]={reason}')
                    continue
                cleaned_rows.append(normalized_row)
                if row_ts > 0:
                    timestamps.append(row_ts)
            if cleaned_rows:
                break
        if not cleaned_rows:
            reason = 'rows_empty_after_repair'
            if invalid_reasons:
                reason = '{} ({})'.format(reason, ';'.join(invalid_reasons[:3]))
            _log_timeframe_drop(logger, symbol=symbol, timeframe=tf_key, reason=reason, raw_rows=sample_rows)
            continue
        # Keep chronological order and ensure each row is a strict numeric OHLCV 5-tuple.
        cleaned_rows = [row for row in cleaned_rows if isinstance(row, list) and len(row) == 5]
        if not cleaned_rows:
            _log_timeframe_drop(logger, symbol=symbol, timeframe=tf_key, reason='rows_empty_after_shape_filter', raw_rows=sample_rows)
            continue
        start_ts = explicit_start_ts
        interval_ms = explicit_interval_ms
        if start_ts <= 0 and timestamps:
            start_ts = int(timestamps[0])
        if interval_ms <= 0 and len(timestamps) >= 2:
            diff = int(timestamps[1] - timestamps[0])
            if diff > 0:
                interval_ms = diff
        out[tf_key] = {
            'start_ts': int(start_ts),
            'interval_ms': int(interval_ms),
            'rows': cleaned_rows,
        }
    return out


def _finalize_payload_for_send(
    payload: Dict[str, Any],
    *,
    logger: Callable[[str], None] | None = None,
) -> Dict[str, Any]:
    clean = dict(payload or {})
    symbol = str(clean.get('symbol') or '')
    bars_clean = _compact_timeframe_bars(
        dict(clean.get('timeframe_bars') or {}),
        symbol=symbol,
        logger=logger,
    )
    clean['timeframe_bars'] = bars_clean if bars_clean else None

    # Replace heavy raw OHLCV rows with deterministic local summaries before OpenAI send.
    clean = apply_kline_preprocessing_to_payload(clean)

    raw_candidate_payload = dict(clean.get('raw_candidate_payload') or {})
    if raw_candidate_payload and isinstance(raw_candidate_payload.get('timeframe_bars'), dict):
        clean['raw_candidate_payload'] = apply_kline_preprocessing_to_payload(raw_candidate_payload)
    return clean


def _short_label(status: str) -> str:
    mapping = {
        'consulted': 'OpenAI consulted',
        'cached_reuse': 'Cached decision reused',
        'cooldown_active': 'Symbol cooldown active',
        'global_interval_active': 'Global send interval active',
        'same_payload_reuse': 'Same payload reused',
        'budget_paused': 'Budget paused',
        'below_min_score': 'Score below filter',
        'not_ranked': 'Outside OpenAI top K',
        'invalid_decision': 'Decision schema invalid',
        'review_deferred': 'Queued for later review',
        'disabled': 'OpenAI disabled',
        'missing_api_key': 'Missing API key',
        'auth_error': 'OpenAI auth error',
        'permission_error': 'OpenAI permission error',
        'bad_request': 'OpenAI bad request',
        'rate_limit': 'OpenAI rate limit',
        'empty_response': 'OpenAI empty response',
        'empty_response_reuse_cached': 'OpenAI empty response, reused cached decision',
        'empty_response_no_action': 'OpenAI empty response, no local action',
        'empty_response_fallback_skip': 'OpenAI empty response, fallback SKIP decision',
        'error': 'OpenAI error',
    }
    return mapping.get(str(status or ''), str(status or 'unknown'))


def _append_recent(state: Dict[str, Any], item: Dict[str, Any], limit: int = 14) -> None:
    rows = list(state.get('recent_decisions') or [])
    rows.insert(0, dict(item or {}))
    state['recent_decisions'] = rows[:max(int(limit), 1)]


def _top_candidates_signature(candidate: Dict[str, Any]) -> str:
    items = []
    for row in list(candidate.get('top_candidates') or [])[:5]:
        sym = str((row or {}).get('symbol') or '').strip()
        if sym:
            items.append(sym)
    return '|'.join(items)


def _build_recent_item(candidate: Dict[str, Any], *, status: str, action: str = '', detail: str = '', decision: Dict[str, Any] | None = None, model: str = '') -> Dict[str, Any]:
    decision = dict(decision or {})
    return {
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'symbol': str(candidate.get('symbol') or ''),
        'side': str(candidate.get('side') or ''),
        'candidate_source': str(candidate.get('candidate_source') or 'normal'),
        'status': str(status or ''),
        'status_label': _short_label(status),
        'action': str(action or ''),
        'detail': str(detail or '')[:280],
        'model': str(model or ''),
        'market_regime': str(decision.get('market_regime') or '')[:48],
        'regime_note': str(decision.get('regime_note') or '')[:180],
        'trend_state': str(decision.get('trend_state') or '')[:60],
        'timing_state': str(decision.get('timing_state') or '')[:60],
        'breakout_assessment': str(decision.get('breakout_assessment') or '')[:160],
        'trade_side': str(decision.get('trade_side') or '')[:40],
        'rr_ratio': _round(decision.get('rr_ratio'), 4),
        'scale_in_recommended': bool(decision.get('scale_in_recommended', False)),
        'scale_in_price': _round(decision.get('scale_in_price'), 8),
        'scale_in_qty_pct': _round(decision.get('scale_in_qty_pct'), 4),
        'scale_in_condition': str(decision.get('scale_in_condition') or '')[:180],
        'scale_in_note': str(decision.get('scale_in_note') or '')[:220],
        'order_type': str(decision.get('order_type') or ''),
        'bot_instruction': str(decision.get('bot_instruction') or '')[:32],
        'entry_price': _round(decision.get('entry_price'), 8),
        'candidate_entry_price': _round(decision.get('candidate_entry_price'), 8),
        'stop_loss': _round(decision.get('stop_loss'), 8),
        'take_profit': _round(decision.get('take_profit'), 8),
        'leverage': int(decision.get('leverage', 0) or 0),
        'margin_pct': _round(decision.get('margin_pct'), 4),
        'confidence': _round(decision.get('confidence'), 2),
        'thesis': str(decision.get('thesis') or '')[:280],
        'market_read': str(decision.get('market_read') or '')[:280],
        'entry_plan': str(decision.get('entry_plan') or '')[:280],
        'entry_reason': str(decision.get('entry_reason') or '')[:220],
        'stop_loss_reason': str(decision.get('stop_loss_reason') or '')[:220],
        'take_profit_plan': str(decision.get('take_profit_plan') or '')[:280],
        'if_missed_plan': str(decision.get('if_missed_plan') or '')[:220],
        'reference_summary': str(decision.get('reference_summary') or '')[:220],
        'watch_trigger_type': str(decision.get('watch_trigger_type') or 'none'),
        'watch_trigger_price': _round(decision.get('watch_trigger_price'), 8),
        'watch_invalidation_price': _round(decision.get('watch_invalidation_price'), 8),
        'watch_note': str(decision.get('watch_note') or '')[:220],
        'recheck_reason': str(decision.get('recheck_reason') or '')[:220],
        'watch_timeframe': str(decision.get('watch_timeframe') or '')[:80],
        'watch_price_zone_low': _round(decision.get('watch_price_zone_low'), 8),
        'watch_price_zone_high': _round(decision.get('watch_price_zone_high'), 8),
        'watch_structure_condition': str(decision.get('watch_structure_condition') or '')[:220],
        'watch_volume_condition': str(decision.get('watch_volume_condition') or '')[:220],
        'watch_checklist': [str(x).strip()[:140] for x in list(decision.get('watch_checklist') or []) if str(x).strip()][:6],
        'watch_confirmations': [str(x).strip()[:140] for x in list(decision.get('watch_confirmations') or []) if str(x).strip()][:6],
        'watch_invalidations': [str(x).strip()[:140] for x in list(decision.get('watch_invalidations') or []) if str(x).strip()][:6],
        'watch_trigger_candle': str(decision.get('watch_trigger_candle') or '')[:40],
        'watch_retest_rule': str(decision.get('watch_retest_rule') or '')[:40],
        'watch_volume_ratio_min': _round(decision.get('watch_volume_ratio_min'), 4),
        'watch_micro_vwap_rule': str(decision.get('watch_micro_vwap_rule') or '')[:40],
        'watch_micro_ema20_rule': str(decision.get('watch_micro_ema20_rule') or '')[:40],
        'watch_recheck_priority': _round(decision.get('watch_recheck_priority'), 2),
        'next_recheck_ts': _round(decision.get('next_recheck_ts'), 3),
        'next_recheck_at': str(decision.get('next_recheck_at') or '')[:40],
        'limit_cancel_price': _round(decision.get('limit_cancel_price'), 8),
        'limit_cancel_timeframe': str(decision.get('limit_cancel_timeframe') or '')[:80],
        'limit_cancel_condition': str(decision.get('limit_cancel_condition') or '')[:220],
        'limit_cancel_note': str(decision.get('limit_cancel_note') or '')[:220],
        'chase_if_triggered': bool(decision.get('chase_if_triggered', False)),
        'chase_trigger_price': _round(decision.get('chase_trigger_price'), 8),
        'chase_limit_price': _round(decision.get('chase_limit_price'), 8),
        'risk_notes': [str(x).strip()[:140] for x in list(decision.get('risk_notes') or []) if str(x).strip()][:4],
        'aggressive_note': str(decision.get('aggressive_note') or '')[:220],
        'reason_to_skip': str(decision.get('reason_to_skip') or '')[:220],
    }


def _extract_text(body: Dict[str, Any]) -> str:
    parsed = body.get('output_parsed')
    if parsed is not None:
        try:
            return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            return str(parsed)
    text = str(body.get('output_text') or '').strip()
    if text:
        return text
    output = body.get('output') or []
    parts = []
    for item in output:
        for content in list((item or {}).get('content') or []):
            if isinstance(content, dict):
                if content.get('json') is not None:
                    try:
                        return json.dumps(content.get('json'), ensure_ascii=False)
                    except Exception:
                        pass
                if content.get('parsed') is not None:
                    try:
                        return json.dumps(content.get('parsed'), ensure_ascii=False)
                    except Exception:
                        return str(content.get('parsed'))
                txt = content.get('text')
                if isinstance(txt, dict):
                    txt = txt.get('value')
                if txt:
                    parts.append(str(txt))
    return '\n'.join(parts).strip()


def _parse_json_text(text: str) -> Dict[str, Any]:
    raw = str(text or '').strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = raw.find('{')
    end = raw.rfind('}')
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except Exception:
            pass
    # Best-effort recovery for truncated JSON responses (e.g., incomplete by max_output_tokens).
    if start >= 0:
        clipped = raw[start:] if end <= start else raw[start:end + 1]
        # Remove trailing commas before object/array close.
        clipped = re.sub(r',\s*([}\]])', r'\1', clipped)
        # Balance braces when the tail is truncated.
        open_braces = clipped.count('{')
        close_braces = clipped.count('}')
        if open_braces > close_braces:
            clipped += '}' * (open_braces - close_braces)
        try:
            parsed = json.loads(clipped)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def _looks_like_trade_decision_object(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    keys = set(value.keys())
    must_any = {'action', 'bot_instruction', 'trade_side'}
    if not (keys & must_any):
        return False
    return bool(keys & {'entry_price', 'stop_loss', 'take_profit', 'rr_ratio', 'timing_state'})


def _collect_json_candidates(node: Any, out: list[Dict[str, Any]], *, depth: int = 0) -> None:
    if depth > 7 or node is None:
        return
    if isinstance(node, dict):
        if _looks_like_trade_decision_object(node):
            out.append(dict(node))
        for key, value in list(node.items())[:120]:
            if key in {'json', 'parsed'} and isinstance(value, dict):
                if _looks_like_trade_decision_object(value):
                    out.append(dict(value))
            if isinstance(value, (dict, list)):
                _collect_json_candidates(value, out, depth=depth + 1)
                continue
            if isinstance(value, str):
                parsed = _parse_json_text(value)
                if parsed and _looks_like_trade_decision_object(parsed):
                    out.append(parsed)
    elif isinstance(node, list):
        for item in list(node)[:120]:
            _collect_json_candidates(item, out, depth=depth + 1)


def _extract_decision_json(body: Dict[str, Any]) -> Dict[str, Any]:
    src = dict(body or {})
    parsed_direct = src.get('output_parsed')
    if _looks_like_trade_decision_object(parsed_direct):
        return dict(parsed_direct or {})
    direct_text = _extract_text(src)
    parsed_text = _parse_json_text(direct_text)
    if parsed_text and _looks_like_trade_decision_object(parsed_text):
        return parsed_text
    rows: list[Dict[str, Any]] = []
    _collect_json_candidates(src, rows, depth=0)
    for row in rows:
        if _looks_like_trade_decision_object(row):
            return row
    # Fall back to first parseable object even if keys are minimal; normalization can still repair.
    if parsed_text:
        return parsed_text
    return {}


def _response_usage(body: Dict[str, Any]) -> Dict[str, Any]:
    return dict((body or {}).get('usage') or {})


def _response_output_tokens(body: Dict[str, Any]) -> int:
    usage = _response_usage(body)
    try:
        return int(usage.get('output_tokens', 0) or 0)
    except Exception:
        return 0


def _json_schema() -> Dict[str, Any]:
    properties = {
        'should_trade': {'type': 'boolean'},
        'action': {'type': 'string', 'enum': ['enter', 'skip']},
        'market_regime': {'type': 'string', 'enum': ['trend_continuation', 'trend_pullback', 'range_reversion', 'consolidation_squeeze', 'transition_chop']},
        'trend_state': {'type': 'string', 'enum': ['trending_up', 'trending_down', 'range_mixed', 'transitioning', 'trend_unclear']},
        'timing_state': {'type': 'string', 'enum': ['enter_now', 'avoid_near_term']},
        'trade_side': {'type': 'string', 'enum': ['long', 'short', 'neutral']},
        'order_type': {'type': 'string', 'enum': ['market', 'limit']},
        'bot_instruction': {'type': 'string', 'enum': ['ENTER_MARKET', 'ENTER_LIMIT', 'SKIP']},
        'entry_price': {'type': 'number'},
        'stop_loss': {'type': 'number'},
        'take_profit': {'type': 'number'},
        'rr_ratio': {'type': 'number'},
        'market_read': _string_schema(220),
        'entry_plan': _string_schema(220),
        'stop_logic': _string_schema(220),
        'reason_to_skip': _string_schema(220),
        'stop_anchor_timeframe': _string_schema(40),
        'stop_anchor_source': _string_schema(120),
        'stop_anchor_price': {'type': 'number'},
        'stop_buffer_atr15': {'type': 'number'},
        'entry_to_stop_atr15': {'type': 'number'},
        'limit_cancel_price': {'type': 'number'},
        'limit_cancel_timeframe': _string_schema(40),
        'limit_cancel_condition': _string_schema(120),
        'limit_cancel_note': _string_schema(120),
    }
    return {
        'type': 'object',
        'additionalProperties': False,
        'properties': properties,
        'required': [
            'should_trade',
            'action',
            'market_regime',
            'trend_state',
            'timing_state',
            'trade_side',
            'entry_price',
            'stop_loss',
            'take_profit',
            'rr_ratio',
            'order_type',
            'bot_instruction',
        ],
    }


def _response_shape_hint() -> str:
    return (
        'Return exactly one complete compact JSON object. '
        'Use raw JSON numbers for numeric fields, concise strings, and short arrays. '
        'The only market data is the candidate payload JSON below.'
    )


def _candidate_stop_guard(candidate: Dict[str, Any], entry_price: float) -> Dict[str, float | bool]:
    market_context = dict(candidate.get('market_context') or {})
    tf15 = dict((market_context.get('multi_timeframe') or {}).get('15m') or {})
    bars15 = dict((market_context.get('timeframe_bars') or {}).get('15m') or {})
    rows = list(bars15.get('rows') or [])
    current_bar_high = _coerce_float(tf15.get('ch', 0), 0.0)
    current_bar_low = _coerce_float(tf15.get('cl', 0), 0.0)
    prior_high = _coerce_float(tf15.get('ph', tf15.get('hi', 0)), 0.0)
    prior_low = _coerce_float(tf15.get('pl', tf15.get('lo', 0)), 0.0)
    range_high = 0.0
    range_low = 0.0
    range_span = 0.0
    range_consolidating = False
    if rows:
        last_row = list(rows[-1] or [])
        if len(last_row) >= 4:
            current_bar_high = _coerce_float(last_row[1], current_bar_high)
            current_bar_low = _coerce_float(last_row[2], current_bar_low)
        previous_rows = [list(row or []) for row in rows[:-1] if isinstance(row, list) and len(row) >= 4]
        if previous_rows:
            lookback_rows = previous_rows[-18:]
            highs = [_coerce_float(row[1], 0.0) for row in lookback_rows if _coerce_float(row[1], 0.0) > 0]
            lows = [_coerce_float(row[2], 0.0) for row in lookback_rows if _coerce_float(row[2], 0.0) > 0]
            pivot_highs: list[float] = []
            pivot_lows: list[float] = []
            if len(lookback_rows) >= 5:
                for idx in range(2, len(lookback_rows) - 2):
                    hi = _coerce_float(lookback_rows[idx][1], 0.0)
                    lo = _coerce_float(lookback_rows[idx][2], 0.0)
                    if hi > 0:
                        left_hi_1 = _coerce_float(lookback_rows[idx - 1][1], 0.0)
                        left_hi_2 = _coerce_float(lookback_rows[idx - 2][1], 0.0)
                        right_hi_1 = _coerce_float(lookback_rows[idx + 1][1], 0.0)
                        right_hi_2 = _coerce_float(lookback_rows[idx + 2][1], 0.0)
                        if hi >= left_hi_1 and hi >= left_hi_2 and hi >= right_hi_1 and hi >= right_hi_2:
                            pivot_highs.append(hi)
                    if lo > 0:
                        left_lo_1 = _coerce_float(lookback_rows[idx - 1][2], 0.0)
                        left_lo_2 = _coerce_float(lookback_rows[idx - 2][2], 0.0)
                        right_lo_1 = _coerce_float(lookback_rows[idx + 1][2], 0.0)
                        right_lo_2 = _coerce_float(lookback_rows[idx + 2][2], 0.0)
                        if lo <= left_lo_1 and lo <= left_lo_2 and lo <= right_lo_1 and lo <= right_lo_2:
                            pivot_lows.append(lo)
            if pivot_highs:
                prior_high = pivot_highs[-1]
            elif highs:
                prior_high = max(highs)
            if pivot_lows:
                prior_low = pivot_lows[-1]
            elif lows:
                prior_low = min(lows)
            range_rows = lookback_rows[-12:]
            range_high = max([_coerce_float(row[1], 0.0) for row in range_rows], default=0.0)
            range_low = min([_coerce_float(row[2], 0.0) for row in range_rows if _coerce_float(row[2], 0.0) > 0], default=0.0)
            if range_high > 0 and range_low > 0 and range_high > range_low:
                range_span = range_high - range_low
    atr_pct = _coerce_float(tf15.get('a', 0), 0.0)
    atr_price = max(entry_price * atr_pct / 100.0, entry_price * 0.003 if entry_price > 0 else 0.0)
    if range_span > 0:
        range_consolidating = range_span <= max(atr_price * 2.8, entry_price * 0.006)
    return {
        'prior_high': prior_high,
        'prior_low': prior_low,
        'current_bar_high': current_bar_high,
        'current_bar_low': current_bar_low,
        'atr_price': atr_price,
        'atr_pct': atr_pct,
        'explosive': bool(tf15.get('xr', False)),
        'range_high': range_high,
        'range_low': range_low,
        'range_span': range_span,
        'range_consolidating': range_consolidating,
    }


def _extract_numeric_candidates(value: Any) -> list[float]:
    out: list[float] = []

    def _push(raw: Any) -> None:
        number = _coerce_float(raw, 0.0)
        if number > 0 and math.isfinite(number):
            out.append(number)

    if isinstance(value, (int, float, str)):
        _push(value)
    elif isinstance(value, list):
        for item in list(value)[:20]:
            if isinstance(item, dict):
                for key in ('price', 'level', 'value', 'p'):
                    if key in item:
                        _push(item.get(key))
                        break
            else:
                _push(item)
    elif isinstance(value, dict):
        for key in ('price', 'level', 'value', 'p'):
            if key in value:
                _push(value.get(key))
        for item in list(value.values())[:20]:
            if isinstance(item, (int, float, str)):
                _push(item)
    return out


def _extract_prices_from_text(text: Any) -> list[float]:
    raw = str(text or '').strip()
    if not raw:
        return []
    out: list[float] = []
    for token in re.findall(r'[-+]?\d+(?:\.\d+)?', raw):
        number = _coerce_float(token, 0.0)
        if number > 0 and math.isfinite(number):
            out.append(number)
    return out


def _pick_entry_hint_by_side(side: str, hints: list[float], reference_price: float) -> float:
    side_norm = str(side or '').lower()
    ref = _coerce_float(reference_price, 0.0)
    rows = [x for x in hints if x > 0 and math.isfinite(x)]
    if not rows:
        return 0.0
    if ref > 0:
        bounded = [x for x in rows if (ref * 0.10) <= x <= (ref * 10.0)]
        if bounded:
            rows = bounded
    if side_norm == 'long':
        if ref > 0:
            below = [x for x in rows if x < ref * 1.002]
            if below:
                return max(below)
        return min(rows, key=lambda x: abs(x - ref)) if ref > 0 else rows[0]
    if side_norm == 'short':
        if ref > 0:
            above = [x for x in rows if x > ref * 0.998]
            if above:
                return min(above)
        return min(rows, key=lambda x: abs(x - ref)) if ref > 0 else rows[0]
    return rows[0]


def _infer_entry_from_ai_hints(raw: Dict[str, Any], candidate: Dict[str, Any], side: str, fallback: float) -> float:
    src = dict(raw or {})
    ref = _coerce_float(candidate.get('current_price', candidate.get('entry_price', fallback)), fallback)
    hint_values: list[float] = []
    for key in (
        'entry_price', 'entry', 'entry_px', 'entryPrice',
        'limit_entry', 'limit_price', 'entry_limit', 'entry_limit_price',
        'planned_entry', 'plan_entry', 'fvg_entry_price', 'pullback_entry_price',
        'watch_trigger_price', 'watch_price_zone_low', 'watch_price_zone_high',
    ):
        if key in src:
            hint_values.extend(_extract_numeric_candidates(src.get(key)))
    watch_plan = dict(src.get('watch_plan') or {})
    for key in ('trigger_price', 'entry_price', 'entry', 'price_zone_low', 'price_zone_high'):
        if key in watch_plan:
            hint_values.extend(_extract_numeric_candidates(watch_plan.get(key)))
    for text_key in ('entry_plan', 'market_read', 'breakout_assessment', 'watch_note', 'recheck_reason'):
        hint_values.extend(_extract_prices_from_text(src.get(text_key)))
    picked = _pick_entry_hint_by_side(side, hint_values, ref)
    return picked if picked > 0 else _coerce_float(fallback, 0.0)


def _nearest_price(candidates: list[float], entry_price: float, *, above: bool) -> float:
    if entry_price <= 0:
        return 0.0
    if above:
        rows = [v for v in candidates if v > entry_price]
        return min(rows) if rows else 0.0
    rows = [v for v in candidates if 0 < v < entry_price]
    return max(rows) if rows else 0.0


def _raw_default_stop_take(candidate: Dict[str, Any], side: str, entry_price: float) -> tuple[float, float]:
    if entry_price <= 0:
        return 0.0, 0.0
    market_context = dict(candidate.get('market_context') or {})
    levels = dict(market_context.get('levels') or {})
    tf15 = dict((market_context.get('multi_timeframe') or {}).get('15m') or {})
    tf1h = dict((market_context.get('multi_timeframe') or {}).get('1h') or {})
    tf4h = dict((market_context.get('multi_timeframe') or {}).get('4h') or {})
    atr_pct = max(_coerce_float(tf15.get('a', 0), 0.0), 0.0)
    atr_price = max(entry_price * atr_pct / 100.0, entry_price * 0.003)
    buffer = max(atr_price * 0.28, entry_price * 0.002)
    constraints = dict(candidate.get('constraints') or {})
    min_rr_hint = max(_coerce_float(constraints.get('min_rr_for_entry', 0.0), 0.0), 0.0)
    rr_floor = _clamp(min_rr_hint if min_rr_hint > 0 else 1.35, 1.0, 4.0)
    rr_fallback = _clamp(rr_floor + 0.25, 1.25, 4.5)
    reward_floor = max(atr_price * max(rr_floor, 1.2), entry_price * 0.006)

    support_candidates: list[float] = []
    resistance_candidates: list[float] = []
    for key in ('nearest_support', 'recent_low', 'support_levels'):
        support_candidates.extend(_extract_numeric_candidates(levels.get(key)))
    for key in ('nearest_resistance', 'recent_high', 'resistance_levels'):
        resistance_candidates.extend(_extract_numeric_candidates(levels.get(key)))
    support_candidates.extend(_extract_numeric_candidates(tf15.get('lo')))
    support_candidates.extend(_extract_numeric_candidates(tf15.get('pl')))
    support_candidates.extend(_extract_numeric_candidates(tf1h.get('lo')))
    support_candidates.extend(_extract_numeric_candidates(tf1h.get('pl')))
    support_candidates.extend(_extract_numeric_candidates(tf4h.get('lo')))
    support_candidates.extend(_extract_numeric_candidates(tf4h.get('pl')))
    resistance_candidates.extend(_extract_numeric_candidates(tf15.get('hi')))
    resistance_candidates.extend(_extract_numeric_candidates(tf15.get('ph')))
    resistance_candidates.extend(_extract_numeric_candidates(tf1h.get('hi')))
    resistance_candidates.extend(_extract_numeric_candidates(tf1h.get('ph')))
    resistance_candidates.extend(_extract_numeric_candidates(tf4h.get('hi')))
    resistance_candidates.extend(_extract_numeric_candidates(tf4h.get('ph')))

    nearest_support = _nearest_price(support_candidates, entry_price, above=False)
    nearest_resistance = _nearest_price(resistance_candidates, entry_price, above=True)
    explosive = bool(tf15.get('xr', False))
    vol_regime_mult = 1.0
    if explosive:
        vol_regime_mult = max(vol_regime_mult, 1.25)
    if atr_pct >= 4.0:
        vol_regime_mult = max(vol_regime_mult, 1.5)
    if atr_pct >= 6.0:
        vol_regime_mult = max(vol_regime_mult, 1.8)
    min_gap = max(atr_price * (1.25 * vol_regime_mult), entry_price * (0.0065 * max(vol_regime_mult, 1.0)))
    max_gap = max(atr_price * (3.2 if explosive else 3.8) * vol_regime_mult, min_gap * 1.4)
    if str(side or '').lower() == 'short':
        stop_default = (
            nearest_resistance + buffer
            if nearest_resistance > 0
            else entry_price + max(buffer, entry_price * 0.015)
        )
        gap = _clamp(stop_default - entry_price, min_gap, max_gap)
        stop_default = entry_price + gap
        risk = max(gap, entry_price * 0.006)
        tp_default = _nearest_price(support_candidates, entry_price, above=False)
        if tp_default <= 0:
            tp_default = entry_price - max(risk * rr_fallback, reward_floor)
        if tp_default >= entry_price:
            tp_default = entry_price - max(risk * rr_fallback, reward_floor)
        min_rr_take = entry_price - (risk * rr_floor)
        if tp_default > min_rr_take:
            tp_default = min_rr_take
    else:
        stop_default = (
            nearest_support - buffer
            if nearest_support > 0
            else entry_price - max(buffer, entry_price * 0.015)
        )
        gap = _clamp(entry_price - stop_default, min_gap, max_gap)
        stop_default = entry_price - gap
        risk = max(gap, entry_price * 0.006)
        tp_default = _nearest_price(resistance_candidates, entry_price, above=True)
        if tp_default <= 0:
            tp_default = entry_price + max(risk * rr_fallback, reward_floor)
        if tp_default <= entry_price:
            tp_default = entry_price + max(risk * rr_fallback, reward_floor)
        min_rr_take = entry_price + (risk * rr_floor)
        if tp_default < min_rr_take:
            tp_default = min_rr_take
    return max(stop_default, 0.0), max(tp_default, 0.0)


def _sanitize_same_bar_stop(decision: Dict[str, Any], candidate: Dict[str, Any]) -> None:
    side = str(decision.get('trade_side') or candidate.get('side') or 'long').lower()
    entry_price = _coerce_float(decision.get('entry_price', 0), 0.0)
    stop_loss = _coerce_float(decision.get('stop_loss', 0), 0.0)
    if side not in {'long', 'short'} or entry_price <= 0 or stop_loss <= 0:
        return
    guard = _candidate_stop_guard(candidate, entry_price)
    atr_price = max(_coerce_float(guard.get('atr_price', 0), 0.0), entry_price * 0.003)
    atr_pct = max(_coerce_float(guard.get('atr_pct', 0), 0.0), 0.0)
    explosive = bool(guard.get('explosive', False))
    high_vol_mode = explosive or atr_pct >= 4.0
    same_bar_tol = max(atr_price * 0.18, entry_price * 0.0012)
    anti_sweep_buffer = max(atr_price * (0.40 if high_vol_mode else 0.28), entry_price * (0.003 if high_vol_mode else 0.002))
    min_gap = max(atr_price * (1.7 if high_vol_mode else 1.2), entry_price * (0.009 if high_vol_mode else 0.006))
    max_gap = max(
        atr_price * ((4.6 if explosive else 4.1) if high_vol_mode else (2.8 if explosive else 3.6)),
        min_gap * 1.3,
    )
    range_consolidating = bool(guard.get('range_consolidating', False))
    range_low = _coerce_float(guard.get('range_low', 0), 0.0)
    range_high = _coerce_float(guard.get('range_high', 0), 0.0)
    adjusted_stop = stop_loss
    adjusted = False
    if side == 'long':
        current_bar_low = _coerce_float(guard.get('current_bar_low', 0), 0.0)
        prior_low = _coerce_float(guard.get('prior_low', 0), 0.0)
        corrected_stop = prior_low - anti_sweep_buffer if 0 < prior_low < entry_price else 0.0
        if range_consolidating and 0 < range_low < entry_price:
            consolidation_stop = range_low - anti_sweep_buffer
            if corrected_stop <= 0 or consolidation_stop < corrected_stop:
                corrected_stop = consolidation_stop
        near_same_bar_low = current_bar_low > 0 and stop_loss >= (current_bar_low - same_bar_tol)
        needs_structure_repair = near_same_bar_low or stop_loss >= entry_price or stop_loss > corrected_stop
        if corrected_stop > 0 and corrected_stop < entry_price and needs_structure_repair:
            adjusted_stop = corrected_stop
            adjusted = True
        gap = entry_price - adjusted_stop
        clamped_gap = _clamp(gap, min_gap, max_gap)
        if not math.isclose(gap, clamped_gap, rel_tol=1e-6, abs_tol=max(entry_price * 1e-7, 1e-9)):
            adjusted_stop = entry_price - clamped_gap
            adjusted = True
    else:
        current_bar_high = _coerce_float(guard.get('current_bar_high', 0), 0.0)
        prior_high = _coerce_float(guard.get('prior_high', 0), 0.0)
        corrected_stop = prior_high + anti_sweep_buffer if prior_high > entry_price else 0.0
        if range_consolidating and range_high > entry_price:
            consolidation_stop = range_high + anti_sweep_buffer
            if corrected_stop <= 0 or consolidation_stop > corrected_stop:
                corrected_stop = consolidation_stop
        near_same_bar_high = current_bar_high > 0 and stop_loss <= (current_bar_high + same_bar_tol)
        needs_structure_repair = near_same_bar_high or stop_loss <= entry_price or stop_loss < corrected_stop
        if corrected_stop > entry_price and needs_structure_repair:
            adjusted_stop = corrected_stop
            adjusted = True
        gap = adjusted_stop - entry_price
        clamped_gap = _clamp(gap, min_gap, max_gap)
        if not math.isclose(gap, clamped_gap, rel_tol=1e-6, abs_tol=max(entry_price * 1e-7, 1e-9)):
            adjusted_stop = entry_price + clamped_gap
            adjusted = True
    if adjusted:
        decision['stop_loss'] = max(adjusted_stop, 0.0)
        note = '已依 15m 結構把止損放到結構外側，避開同根K掃損位；高波動時已自動加寬止損距離以降低被掃機率。'
        existing = str(decision.get('stop_loss_reason') or '').strip()
        decision['stop_loss_reason'] = note if not existing else f'{note} {existing}'


def _constraint_brief(candidate: Dict[str, Any], constraints: Dict[str, Any]) -> str:
    trade_style = str(candidate.get('trade_style') or constraints.get('trade_style') or 'short_term_intraday')
    min_rr = max(_coerce_float(constraints.get('min_rr_for_entry', 0.0), 0.0), 0.0)
    tp_mode = str(constraints.get('execution_tp_mode') or 'ai_direct')
    return (
        f'style={trade_style}; '
        f'leverage=symbol_max_{int(constraints.get("fixed_leverage", constraints.get("max_leverage", 25)) or 25)}x; '
        f'margin_pct={float(constraints.get("min_margin_pct", 0.03) or 0.03):.4f}-{float(constraints.get("max_margin_pct", 0.08) or 0.08):.4f}; '
        f'notional={float(constraints.get("fixed_order_notional_usdt", 20.0) or 20.0):.4f} USDT; '
        f'min_margin={float(constraints.get("min_order_margin_usdt", 0.1) or 0.1):.4f} USDT; '
        f'min_rr={min_rr:.2f}; '
        f'tp_execution_mode={tp_mode}'
    )


def _build_messages(
    candidate: Dict[str, Any],
    *,
    compact: bool = False,
    logger: Callable[[str], None] | None = None,
) -> list[Dict[str, Any]]:
    clean_payload = _build_clean_payload(candidate, logger=logger)
    clean_payload = _finalize_payload_for_send(clean_payload, logger=logger)
    constraints = dict(clean_payload.get('constraints') or {})
    trade_style = str(clean_payload.get('trade_style') or constraints.get('trade_style') or 'short_term_intraday')
    constraint_brief = _constraint_brief(clean_payload, constraints)
    system_text = (
        'You are the only trade-decision brain; local bot is execution-only and must follow your command. '
        'Local bot provides only raw market data and deterministic formula outputs; it does not provide direction bias or strategy advice. '
        'Do not assume any field name implies bullish or bearish conclusion. Treat all tags, summaries, anomaly marks, and metrics as evidence only, not direct signals. '
        'You must independently infer direction, edge, invalidation, and execution risk. '
        'If a field is null or missing, explicitly treat it as unavailable; do not infer unavailable flow, whale, liquidation, or sentiment data. '
        'Use full candidate payload data exactly as provided; do not ignore kline_summary, data_quality, compression_info, flow/liquidity, or derivatives fields. '
        'Output one JSON object only (no markdown). Numeric fields must be JSON numbers; text fields in Traditional Chinese. '
        'Allowed commands: ENTER_MARKET, ENTER_LIMIT, SKIP. OBSERVE is disabled. '
        'Required core: should_trade,action,market_regime,trend_state,timing_state,trade_side,order_type,bot_instruction,entry_price,stop_loss,take_profit,rr_ratio. '
        'Contract: enter => should_trade=true + trade_side long/short + entry/stop/tp > 0. '
        'skip => should_trade=false + entry=stop=tp=0 + concrete reason_to_skip. '
        'Direction must be neutral and based on market data only. '
        'Do not prefer long or short because of 24h change magnitude. '
        'For high-24h-gain symbols, be extra cautious on long entries: avoid blind chasing and avoid forcing long at stretched highs without structural confirmation. '
        'For high-24h-gain symbols, short setups are allowed and often valuable as advisory preference, but never by default: require real reversal evidence before short (structure break, failed reclaim, flow turn, and executable invalidation). '
        'Re-derive direction from 1D/4H/1H/15m/5m/1m structure + flow + liquidity + derivatives only. '
        'Long and short opportunities are symmetric and equal priority. '
        'Trend-following and counter-trend are both allowed on both sides under symmetric standards. '
        'Counter-trend (long or short) requires rejection/reclaim or breakdown/fakeout evidence, flow turn, and explicit invalidation. '
        'Allow breakdown continuation short when support breaks, retest fails to reclaim, flow is bearish, and downside room is sufficient. '
        'Allow upside breakout continuation long when resistance breaks, retest holds, flow is bullish, and upside room is sufficient. '
        'If long distance to nearest resistance <0.25 ATR15 => SKIP; if short distance to nearest support <0.25 ATR15 => SKIP. '
        'For fast-moving/high-volatility symbols, do not place tight stops: stop-loss must be set farther, outside structure invalidation and outside nearby liquidity sweep zones. '
        'When entry is momentum/chasing, use wider stop by design and reduce position size to keep risk controlled; repeated tiny-stop sweep has no edge. '
        'Entry planning advisory: structure resistance/support can be used as limit-entry anchors when feasible, but this is not priority over trend capture. '
        'Before ENTER, compare candidate with current held_symbols in payload as an advisory overlap check: high similarity is a caution signal, not a hard veto. '
        'Same direction alone (both bullish or both bearish) is NOT enough to reject. '
        'Keep this filter loose: only near-clone behavior (e.g., BTC/ETH/SOL style tight co-movement) should materially reduce conviction. '
        'Do not reject a candidate only because portfolio already has longs or shorts. '
        'Your decision basis should include: market analysis, trend analysis, smart-money/manipulation behavior analysis, and market sentiment analysis; these are advisory evidence, not hard rules. '
        'Final decision must depend only on current RR, structure, flow, liquidity, and execution risk.'
    )
    user_text = (
        f'Generate one {trade_style} tactical execution JSON decision.\n'
        f'Constraints (soft): {constraint_brief}.\n'
        'Analyze in this order: 1D/4H bias -> 1H trend -> 15m execution -> 5m/1m timing.\n'
        'HTF bias is guidance not veto. Use full payload and kline_summary (timeframe_bars.rows were compressed locally) for structure + flow judgment.\n'
        'Read compression_info and data_quality first, so you understand what was compressed and removed.\n'
        'Review kline_summary.<timeframe>.recent_2h_anomaly_candles and treat it as important context (e.g., up_needle/down_needle/sudden_pump/sudden_dump/volume_spike) for execution quality and stop placement.\n'
        'Needle events (up_needle/down_needle) are reference signals only and must not be used as the primary entry trigger by themselves.\n'
        'For each needle context, classify it as good needle vs bad needle using follow-through, reclaim/failure, volume-flow confirmation, and liquidity behavior; only good needle can support a setup, bad needle should reduce conviction or lead to SKIP.\n'
        'Integrate market/trend/smart-money/sentiment analysis as advisory judgment layers; not mandatory veto rules.\n'
        'Use portfolio.held_symbols as current holdings universe. Treat overlap as advisory: compare multi-timeframe structure + flow + volatility rhythm to estimate similarity risk.\n'
        'If only direction is similar but structure/flow details are different, you may still ENTER.\n'
        'If you choose SKIP due to overlap risk, state "highly similar走势重复" and name the overlapping held symbol; otherwise continue normal decision by RR/structure/flow/liquidity.\n'
        'Order execution suggestion: prefer LIMIT entries when executable to improve fill quality; you may use FVG gap / reclaim zone / pullback zone as limit anchors.\n'
        'This is advisory, not mandatory: if momentum/trend continuation is strong and delay harms expectancy, ENTER_MARKET is acceptable.\n'
        'Advisory alpha focus: proactively capture pre-expansion candidates on both sides; for explosive upside setup prefer long, for explosive downside setup prefer short.\n'
        'For pre-expansion setups, do not require perfect signals: if structure compression + flow turn + liquidity vacuum/trigger proximity are aligned and RR is executable, you may ENTER with clear invalidation instead of over-waiting.\n'
        'Treat this as a soft encouragement to catch upcoming expansion earlier, while still respecting stop structure and execution risk.\n'
        'When considering long on strong gainers: apply stricter anti-chase checks (distance from structure anchor, retest/reclaim quality, entry-to-stop efficiency). '
        'Default to ENTER_LIMIT on pullback/retest zones; avoid ENTER_MARKET at stretched highs unless break-retest confirmation is very clear and execution risk is still efficient. '
        'If price is overly stretched and no clean pullback/reclaim, prefer SKIP or wait. '
        'If trend structure is still healthy with executable risk, long is still allowed.\n'
        'For short_gainers candidates, countertrend short is an advisory-favored path only when reversal is clearly visible. '
        'Do not short merely because price has risen a lot; only short when reversal signals are explicit across structure + flow + execution context.\n'
        'Pick the best executable edge now from both sides equally: breakdown continuation short, upside breakout continuation long, continuation pullback, or reversal reclaim.\n'
        'If edge is clear now => ENTER_MARKET or ENTER_LIMIT. If not clear => SKIP.\n'
        'For ENTER provide concise market_read, entry_plan, stop_logic, stop_anchor_timeframe, stop_anchor_source, stop_anchor_price, stop_buffer_atr15, entry_to_stop_atr15.\n'
        'For SKIP set entry_price=0, stop_loss=0, take_profit=0, and reason_to_skip concrete.\n'
        'Output only one-line JSON object.'
    )
    payload_text = 'Candidate payload JSON:\n' + json.dumps(clean_payload, ensure_ascii=False, separators=(',', ':'))
    return [
        {'role': 'system', 'content': [{'type': 'input_text', 'text': system_text}]},
        {'role': 'user', 'content': [{'type': 'input_text', 'text': user_text}]},
        {'role': 'user', 'content': [{'type': 'input_text', 'text': payload_text}]},
    ]


def _fallback_trade_decision(candidate: Dict[str, Any], *, reason: str = '') -> Dict[str, Any]:
    candidate = dict(candidate or {})
    side = str(candidate.get('side') or 'long').lower()
    if side not in {'long', 'short'}:
        side = 'long'
    raw = {
        'should_trade': False,
        'action': 'skip',
        'trend_state': 'trend_unclear',
        'timing_state': 'avoid_near_term',
        'trade_side': side,
        'breakout_assessment': 'openai_empty_no_local_decision',
        'order_type': 'market',
        'bot_instruction': 'SKIP',
        'entry_price': 0.0,
        'stop_loss': 0.0,
        'take_profit': 0.0,
        'market_read': 'OpenAI 空回覆，本輪禁止本地推論。',
        'entry_plan': '本輪不下單，等待下一輪 OpenAI 有效 JSON。',
        'reason_to_skip': 'OpenAI 空回覆，未取得 AI 認可，不允許機器人自行下單。',
        'watch_trigger_type': 'none',
        'watch_trigger_price': 0.0,
        'watch_invalidation_price': 0.0,
        'watch_note': '',
        'recheck_reason': '',
        'watch_timeframe': '',
        'watch_price_zone_low': 0.0,
        'watch_price_zone_high': 0.0,
        'watch_structure_condition': '',
        'watch_volume_condition': '',
        'watch_trigger_candle': 'none',
        'watch_retest_rule': 'none',
        'watch_volume_ratio_min': 0.0,
        'watch_micro_vwap_rule': 'none',
        'watch_micro_ema20_rule': 'none',
        'watch_checklist': [],
        'watch_confirmations': [],
        'watch_invalidations': [],
        'watch_recheck_priority': 0.0,
        'limit_cancel_price': 0.0,
        'limit_cancel_timeframe': '',
        'limit_cancel_condition': '',
        'limit_cancel_note': '',
        'risk_notes': [str(reason or 'openai_empty_response')[:160]],
    }
    return _normalize_decision(raw, candidate)


def _build_request_body(
    candidate: Dict[str, Any],
    config: Dict[str, Any],
    *,
    structured: bool = True,
    model: str | None = None,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
    compact_prompt: bool = False,
    logger: Callable[[str], None] | None = None,
) -> Dict[str, Any]:
    effective_max_output_tokens = max(
        int(max_output_tokens or config.get('max_output_tokens', 900) or 900),
        220 if structured else 140,
    )
    body = {
        'model': str(model or config.get('model') or 'gpt-5.4-mini'),
        'input': _build_messages(candidate, compact=compact_prompt, logger=logger),
        'max_output_tokens': effective_max_output_tokens,
    }
    try:
        body['temperature'] = _clamp(config.get('temperature', 0.0), 0.0, 0.4)
    except Exception:
        body['temperature'] = 0.0
    text_format: Dict[str, Any]
    if structured:
        text_format = {
            'type': 'json_schema',
            'name': 'trade_decision',
            'schema': _json_schema(),
            'strict': True,
        }
    else:
        text_format = {
            'type': 'json_object',
        }
    body['text'] = {
        'verbosity': 'low',
        'format': text_format,
    }
    effort = str(reasoning_effort or config.get('reasoning_effort') or '').strip().lower()
    if effort in {'low', 'medium', 'high'}:
        body['reasoning'] = {'effort': effort}
    return body


def _response_diagnostics(body: Dict[str, Any]) -> str:
    src = dict(body or {})
    parts = []
    status = str(src.get('status') or '').strip()
    if status:
        parts.append(f'status={status}')
    incomplete = dict(src.get('incomplete_details') or {})
    if incomplete:
        reason = str(incomplete.get('reason') or '').strip()
        if reason:
            parts.append(f'incomplete_reason={reason}')
    try:
        usage = _response_usage(src)
        if usage:
            parts.append('usage_in={} usage_out={} usage_cached={}'.format(
                int(usage.get('input_tokens', 0) or 0),
                int(usage.get('output_tokens', 0) or 0),
                int(usage.get('input_cached_tokens', usage.get('cached_input_tokens', 0)) or 0),
            ))
    except Exception:
        pass
    item_types = []
    for item in list(src.get('output') or [])[:8]:
        kind = str((item or {}).get('type') or '').strip()
        if kind:
            item_types.append(kind)
        for content in list((item or {}).get('content') or [])[:8]:
            ctype = str((content or {}).get('type') or '').strip()
            if ctype:
                item_types.append(ctype)
    if item_types:
        parts.append('output_types={}'.format(','.join(item_types[:8])))
    return ' | '.join(parts[:6])


def _watch_trigger_type_from_side_timing(side: str, timing_state: str) -> str:
    if str(timing_state or '') == 'wait_pullback':
        return 'pullback_to_entry'
    return 'breakdown_confirm' if str(side or '').lower() == 'short' else 'breakout_reclaim'


def _infer_wait_timing_state(decision: Dict[str, Any]) -> str:
    text = ' '.join(
        [
            str(decision.get('entry_plan') or ''),
            str(decision.get('watch_note') or ''),
            str(decision.get('recheck_reason') or ''),
            str(decision.get('breakout_assessment') or ''),
            str(decision.get('watch_structure_condition') or ''),
            str(decision.get('watch_volume_condition') or ''),
        ]
    ).lower()
    if any(token in text for token in ('pullback', 'retest', '回踩', '回測', '回抽')):
        return 'wait_pullback'
    return 'wait_breakout'


def _should_promote_to_observe(decision: Dict[str, Any]) -> bool:
    action = str(decision.get('action') or '').strip().lower()
    timing_state = str(decision.get('timing_state') or '').strip().lower()
    if action == 'enter' and bool(decision.get('should_trade', False)) and timing_state == 'enter_now':
        # Keep enter intent aggressive: do not downgrade because of incidental watch text fields.
        return False
    if str(decision.get('watch_trigger_type') or 'none') != 'none':
        return True
    if str(decision.get('timing_state') or '') in {'wait_pullback', 'wait_breakout'}:
        return True
    if str(decision.get('watch_trigger_candle') or 'none') != 'none':
        return True
    if str(decision.get('watch_retest_rule') or 'none') != 'none':
        return True
    text = ' '.join(
        [
            str(decision.get('entry_plan') or ''),
            str(decision.get('watch_note') or ''),
            str(decision.get('recheck_reason') or ''),
            str(decision.get('reason_to_skip') or ''),
            str(decision.get('breakout_assessment') or ''),
        ]
    ).lower()
    has_waiting_language = any(
        token in text
        for token in (
            '等待', '先等', '等到', '確認後再', '再重新送', '再送 ai', '再執行',
            'wait ', 'wait_', 'wait for', 'after close', 'after retest', 'then re-ask',
            'retest then', 'reclaim then',
        )
    )
    has_executable_plan = (
        float(decision.get('entry_price', 0) or 0) > 0
        or float(decision.get('watch_trigger_price', 0) or 0) > 0
        or str(decision.get('order_type') or '') == 'limit'
    )
    has_wait_fields = (
        float(decision.get('watch_trigger_price', 0) or 0) > 0
        or str(decision.get('watch_structure_condition') or '').strip() != ''
        or str(decision.get('watch_volume_condition') or '').strip() != ''
        or str(decision.get('recheck_reason') or '').strip() != ''
    )
    return has_waiting_language and has_executable_plan and has_wait_fields


def _looks_like_fake_breakout(decision: Dict[str, Any]) -> bool:
    text = ' '.join(
        [
            str(decision.get('breakout_assessment') or ''),
            str(decision.get('market_read') or ''),
            str(decision.get('watch_volume_condition') or ''),
            str(decision.get('watch_structure_condition') or ''),
            str(decision.get('reason_to_skip') or ''),
            ' '.join([str(x) for x in list(decision.get('risk_notes') or [])[:6]]),
        ]
    ).lower()
    if not text:
        return False
    strong_tokens = (
        '假突破', 'fake breakout', 'false breakout', 'bull trap', 'bear trap',
        'trap move', 'liquidity grab', 'sweep and reject', 'failed breakout',
        'breakout failure',
    )
    weak_tokens = (
        'quick rejection', 'wick rejection', '衝高回落', '跌破收回',
        '量能不足', 'volume fade', 'no follow-through', 'breakout without follow-through',
    )
    breakout_context_tokens = (
        '突破', 'breakout', 'reclaim', '跌破', 'breakdown', '掃損', 'sweep', 'liquidity',
    )
    if any(token in text for token in strong_tokens):
        return True
    weak_hits = sum(1 for token in weak_tokens if token in text)
    has_breakout_context = any(token in text for token in breakout_context_tokens)
    return has_breakout_context and weak_hits >= 2


def _candidate_tf15(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return dict((((dict(candidate or {}).get('market_context') or {}).get('multi_timeframe') or {}).get('15m') or {}))


def _candidate_rows(candidate: Dict[str, Any], timeframe: str = '15m', *, limit: int = 48) -> list[list[float]]:
    bars = (((dict(candidate or {}).get('market_context') or {}).get('timeframe_bars') or {}).get(timeframe) or {})
    rows = bars.get('rows') if isinstance(bars, dict) else bars
    out: list[list[float]] = []
    for row in list(rows or [])[-limit:]:
        if isinstance(row, (list, tuple)) and len(row) >= 5:
            seq = list(row)[0:5]
            vals = [_coerce_float(x, 0.0) for x in seq]
            if all(v > 0 for v in vals[:4]):
                out.append(vals)
    return out


def _candidate_metric(candidate: Dict[str, Any], key: str, default: float = 0.0) -> float:
    root = dict(candidate or {})
    ctx = dict(root.get('market_context') or {})
    metrics = dict(ctx.get('calculated_metrics') or {})
    if key in root:
        return _coerce_float(root.get(key), default)
    return _coerce_float(metrics.get(key), default)


def _candidate_change_24h_pct(candidate: Dict[str, Any]) -> float:
    root = dict(candidate or {})
    basic = dict((dict(root.get('market_context') or {}).get('basic_market_data') or {}))
    return _coerce_float(
        root.get(
            'change_24h_pct',
            basic.get('change_24h_pct', basic.get('price_change_percent_24h', 0.0)),
        ),
        0.0,
    )


def _is_high_gainer_context(candidate: Dict[str, Any]) -> bool:
    source = str(candidate.get('candidate_source') or candidate.get('source') or '').strip().lower()
    if source in {'short_gainers', 'gainer_followup_discovery'}:
        return True
    return _candidate_change_24h_pct(candidate) >= 10.0


def _candidate_entry_stretched(candidate: Dict[str, Any], side: str, entry_price: float) -> bool:
    side_norm = str(side or '').lower()
    entry_price = _coerce_float(entry_price, 0.0)
    if side_norm not in {'long', 'short'} or entry_price <= 0:
        return False
    tf15 = _candidate_tf15(candidate)
    atr_pct = max(_coerce_float(tf15.get('a', 0), 0.0), 0.0)
    atr_price = max(entry_price * atr_pct / 100.0, entry_price * 0.003)
    ema20 = _coerce_float(tf15.get('e20', 0), 0.0)
    vwap = _coerce_float(tf15.get('v', 0), 0.0)
    structure_high = max(_coerce_float(tf15.get('hi', 0), 0.0), _coerce_float(tf15.get('ph', 0), 0.0))
    structure_low = max(_coerce_float(tf15.get('lo', 0), 0.0), _coerce_float(tf15.get('pl', 0), 0.0))
    symbol = str(candidate.get('symbol') or '')
    is_major = base_asset(symbol) in {'BTC', 'ETH', 'XAU', 'XAG', 'SOL', 'BNB', 'XRP'}
    stretch_mult = 1.05 if is_major else 1.35
    extreme_mult = 0.55 if is_major else 0.40
    if side_norm == 'long':
        anchor = max(ema20, vwap)
        if anchor <= 0:
            return False
        stretched = (entry_price - anchor) >= (atr_price * stretch_mult)
        near_high = structure_high > 0 and (structure_high - entry_price) <= (atr_price * extreme_mult)
        return bool(stretched and near_high)
    anchor_candidates = [x for x in (ema20, vwap) if x > 0]
    if not anchor_candidates:
        return False
    anchor = min(anchor_candidates)
    stretched = (anchor - entry_price) >= (atr_price * stretch_mult)
    near_low = structure_low > 0 and (entry_price - structure_low) <= (atr_price * extreme_mult)
    return bool(stretched and near_low)


def _infer_limit_entry_from_fvg(candidate: Dict[str, Any], side: str, reference_price: float) -> float:
    side_norm = str(side or '').lower()
    ref = _coerce_float(reference_price, 0.0)
    if side_norm not in {'long', 'short'} or ref <= 0:
        return 0.0
    rows = _candidate_rows(candidate, '15m', limit=48)
    if len(rows) >= 3:
        picks: list[float] = []
        for idx in range(2, len(rows)):
            a = rows[idx - 2]
            c = rows[idx]
            a_high = _coerce_float(a[1], 0.0)
            a_low = _coerce_float(a[2], 0.0)
            c_high = _coerce_float(c[1], 0.0)
            c_low = _coerce_float(c[2], 0.0)
            if side_norm == 'long' and c_low > a_high > 0:
                mid = (a_high + c_low) / 2.0
                if 0 < mid < ref:
                    picks.append(mid)
            if side_norm == 'short' and c_high < a_low and c_high > 0:
                mid = (c_high + a_low) / 2.0
                if mid > ref:
                    picks.append(mid)
        if picks:
            return _coerce_float(max(picks) if side_norm == 'long' else min(picks), 0.0)
    tf15 = _candidate_tf15(candidate)
    atr_pct = max(_coerce_float(tf15.get('a', 0), 0.0), 0.0)
    atr_price = max(ref * atr_pct / 100.0, ref * 0.003)
    prior_low = max(_coerce_float(tf15.get('pl', 0), 0.0), _coerce_float(tf15.get('lo', 0), 0.0))
    prior_high = max(_coerce_float(tf15.get('ph', 0), 0.0), _coerce_float(tf15.get('hi', 0), 0.0))
    if side_norm == 'long':
        fallback = prior_low if 0 < prior_low < ref else (ref - atr_price * 0.75)
        if fallback < ref * 0.999:
            return _coerce_float(max(fallback, ref - atr_price * 1.4), 0.0)
    else:
        fallback = prior_high if prior_high > ref else (ref + atr_price * 0.75)
        if fallback > ref * 1.001:
            return _coerce_float(min(fallback, ref + atr_price * 1.4), 0.0)
    return 0.0


def _enforce_non_chase_entry(decision: Dict[str, Any], candidate: Dict[str, Any], side: str) -> None:
    if str(decision.get('action') or '').lower() != 'enter':
        return
    if not bool(decision.get('should_trade', False)):
        return
    side_norm = str(side or decision.get('trade_side') or '').lower()
    if side_norm not in {'long', 'short'}:
        return
    current_price = _coerce_float(candidate.get('current_price', candidate.get('entry_price', 0)), 0.0)
    entry_price = _coerce_float(decision.get('entry_price', current_price), current_price)
    symbol = str(candidate.get('symbol') or '')
    is_major = base_asset(symbol) in {'BTC', 'ETH', 'XAU', 'XAG', 'SOL', 'BNB', 'XRP'}
    tf15 = _candidate_tf15(candidate)
    atr_pct = max(_coerce_float(tf15.get('a', 0), 0.0), 0.0)
    ref_price = max(current_price, entry_price, 0.0)
    atr_price = max(ref_price * atr_pct / 100.0, ref_price * 0.003) if ref_price > 0 else 0.0
    strict_gainer_long = bool(side_norm == 'long' and _is_high_gainer_context(candidate))
    strict_pullback_mult = 0.32 if is_major else 0.45
    strict_pullback_px = atr_price * strict_pullback_mult if strict_gainer_long else 0.0
    distance_to_res_atr15 = max(_candidate_metric(candidate, 'distance_to_resistance_atr15', 0.0), 0.0)
    near_resistance = bool(strict_gainer_long and distance_to_res_atr15 > 0 and distance_to_res_atr15 < 0.35)
    too_close_for_long_pullback = bool(
        strict_gainer_long
        and current_price > 0
        and strict_pullback_px > 0
        and (current_price - entry_price) < strict_pullback_px
    )
    text = ' '.join(
        [
            str(decision.get('entry_plan') or ''),
            str(decision.get('market_read') or ''),
            str(decision.get('breakout_assessment') or ''),
            str(decision.get('reason_to_skip') or ''),
        ]
    ).lower()
    says_no_chase = any(
        token in text
        for token in (
            '不要追價', '不追價', 'wait pullback', 'avoid chase', 'do not chase',
            'stretched', '過度延伸', '追高', '追空',
        )
    )
    if strict_gainer_long and (near_resistance or too_close_for_long_pullback):
        says_no_chase = True
    stretched = _candidate_entry_stretched(candidate, side_norm, entry_price if entry_price > 0 else current_price)
    strict_gainer_guard = bool(strict_gainer_long and (str(decision.get('order_type') or '').lower() != 'limit' or too_close_for_long_pullback or near_resistance))
    if not (says_no_chase or stretched or strict_gainer_guard):
        return
    existing_limit_ok = (
        str(decision.get('order_type') or '').lower() == 'limit'
        and current_price > 0
        and (
            (side_norm == 'long' and entry_price < current_price * 0.999)
            or (side_norm == 'short' and entry_price > current_price * 1.001)
        )
    )
    if existing_limit_ok and strict_gainer_long and current_price > 0 and strict_pullback_px > 0:
        existing_limit_ok = (current_price - entry_price) >= (strict_pullback_px * 0.9)
    if existing_limit_ok:
        return
    decision['order_type'] = 'limit'
    limit_entry = _infer_limit_entry_from_fvg(candidate, side_norm, entry_price if entry_price > 0 else current_price)
    if strict_gainer_long and current_price > 0 and strict_pullback_px > 0:
        min_pullback_entry = current_price - strict_pullback_px
        if min_pullback_entry > 0 and (limit_entry <= 0 or limit_entry > min_pullback_entry):
            limit_entry = min_pullback_entry
    if side_norm == 'long' and limit_entry >= max(entry_price, current_price) * 0.999:
        limit_entry = 0.0
    if side_norm == 'short' and limit_entry <= max(entry_price, current_price) * 1.001:
        limit_entry = 0.0
    if limit_entry > 0:
        decision['entry_price'] = limit_entry
        plan_note = '改為限價承接 FVG/回踩，避免追價。'
        if strict_gainer_long:
            plan_note = '漲幅榜做多啟用嚴格防追高：只用回踩限價進場，不追市價。'
        existing_plan = str(decision.get('entry_plan') or '').strip()
        decision['entry_plan'] = plan_note if not existing_plan else '{} {}'.format(plan_note, existing_plan)
        if not str(decision.get('limit_cancel_note') or '').strip():
            decision['limit_cancel_note'] = '若未成交且結構失效，取消此限價單。'
        return
    decision['action'] = 'skip'
    decision['should_trade'] = False
    decision['timing_state'] = 'avoid_near_term'
    decision['bot_instruction'] = 'SKIP'
    if strict_gainer_long:
        decision['reason_to_skip'] = '漲幅榜做多防追高：目前缺少可執行回踩/收回確認，避免高位追價本輪不做。'
    else:
        decision['reason_to_skip'] = '目前屬延伸段且非理想進場位，為避免追價本輪不做。'


def _normalize_decision(raw: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    raw = dict(raw or {})
    if isinstance(raw.get('decision'), dict):
        raw = dict(raw.get('decision') or {})
    elif isinstance(raw.get('trade_decision'), dict):
        raw = dict(raw.get('trade_decision') or {})

    allowed_actions = {'enter', 'skip'}
    allowed_regimes = {'trend_continuation', 'trend_pullback', 'range_reversion', 'consolidation_squeeze', 'transition_chop'}
    allowed_trend_states = {'trending_up', 'trending_down', 'range_mixed', 'transitioning', 'trend_unclear'}
    allowed_timing_states = {'enter_now', 'avoid_near_term'}
    allowed_sides = {'long', 'short', 'neutral'}
    allowed_order_types = {'market', 'limit'}
    allowed_instructions = {'ENTER_MARKET', 'ENTER_LIMIT', 'SKIP'}

    def _num(key: str, default: float = 0.0) -> float:
        return _coerce_float(raw.get(key, default), default)

    def _looks_like_schema_placeholder(value: Any) -> bool:
        if isinstance(value, dict):
            keyset = {str(k).strip().lower() for k in value.keys()}
            if not keyset:
                return False
            if 'type' in keyset and (
                keyset <= {'type', 'maxlength', 'minlength', 'enum', 'description', 'default', 'title'}
                or keyset <= {'type', 'max_length', 'min_length', 'enum', 'description', 'default', 'title'}
            ):
                return True
        text = str(value or '').strip().lower().replace(' ', '')
        if not text:
            return False
        return (
            text.startswith("{'type':'string'") or text.startswith('{"type":"string"')
            or ('maxlength' in text and 'type' in text and 'string' in text and text.startswith('{'))
        )

    def _safe_text_value(value: Any) -> str:
        if value is None:
            return ''
        if _looks_like_schema_placeholder(value):
            return ''
        if isinstance(value, (dict, list, tuple)):
            return ''
        return str(value).strip()

    def _txt(key: str, limit: int = 220) -> str:
        return _safe_text_value(raw.get(key))[:limit]

    def _txt_any(keys: list[str], limit: int = 220) -> str:
        for key in keys:
            text = _safe_text_value(raw.get(key))
            if text:
                return text[:limit]
        return ''

    def _normalize_side(value: Any) -> str:
        text = str(value or '').strip().lower()
        if text in {'buy', 'long', 'bull', 'up', '做多', '多', '看多'}:
            return 'long'
        if text in {'sell', 'short', 'bear', 'down', '做空', '空', '看空'}:
            return 'short'
        if text in {'none', 'na', 'n/a', 'flat', 'neutral', '觀望', '無', '没有', '沒有'}:
            return 'neutral'
        if 'long' in text or '做多' in text or '看多' in text:
            return 'long'
        if 'short' in text or '做空' in text or '看空' in text:
            return 'short'
        return ''

    def _normalize_action(value: Any) -> str:
        text = str(value or '').strip().lower()
        if text in allowed_actions:
            return text
        if text.startswith('enter') or '進場' in text or '入場' in text:
            return 'enter'
        if text.startswith('skip') or text.startswith('obs') or 'observe' in text or '觀察' in text or '等待' in text or '不做' in text or '略過' in text:
            return 'skip'
        return ''

    def _normalize_order_type(value: Any) -> str:
        text = str(value or '').strip().lower()
        if text in allowed_order_types:
            return text
        if 'limit' in text or '限價' in text or '掛單' in text or '挂单' in text:
            return 'limit'
        if 'market' in text or '市價' in text:
            return 'market'
        return ''

    def _normalize_bot_instruction(value: Any) -> str:
        text = str(value or '').strip().upper().replace('-', '_').replace(' ', '_')
        if text in allowed_instructions:
            return text
        if text.startswith('ENTER_LIMIT') or text in {'LIMIT', 'ENTRY_LIMIT', 'ENTERL'}:
            return 'ENTER_LIMIT'
        if text.startswith('ENTER_MARKET') or text in {'MARKET', 'ENTRY_MARKET', 'ENTERM'}:
            return 'ENTER_MARKET'
        if text.startswith('SKIP') or text.startswith('OBS') or text in {'PASS', 'WAIT', 'WATCH'}:
            return 'SKIP'
        return ''

    def _normalize_market_regime(value: Any) -> str:
        text = str(value or '').strip().lower()
        if text in allowed_regimes:
            return text
        if 'continuation' in text or '順勢' in text or '延續' in text:
            return 'trend_continuation'
        if 'pullback' in text or '回踩' in text or '回撤' in text:
            return 'trend_pullback'
        if 'range' in text or '震盪' in text or '區間' in text:
            return 'range_reversion'
        if 'squeeze' in text or '壓縮' in text or '爆發前' in text:
            return 'consolidation_squeeze'
        return 'transition_chop'

    def _normalize_trend_state(value: Any, side_hint: str = '') -> str:
        text = str(value or '').strip().lower()
        if text in allowed_trend_states:
            return text
        if 'up' in text or 'bull' in text or '多頭' in text:
            return 'trending_up'
        if 'down' in text or 'bear' in text or '空頭' in text:
            return 'trending_down'
        if 'transition' in text or '轉折' in text:
            return 'transitioning'
        if 'range' in text or 'mixed' in text or '震盪' in text:
            return 'range_mixed'
        if side_hint == 'long':
            return 'trending_up'
        if side_hint == 'short':
            return 'trending_down'
        return 'trend_unclear'

    instruction_action_map = {
        'ENTER_MARKET': 'enter',
        'ENTER_LIMIT': 'enter',
        'SKIP': 'skip',
    }

    side = _normalize_side(raw.get('trade_side') or raw.get('direction') or raw.get('side') or raw.get('position_side'))
    bot_instruction = _normalize_bot_instruction(raw.get('bot_instruction'))
    action = _normalize_action(raw.get('action'))
    order_type = _normalize_order_type(raw.get('order_type'))
    should_trade = bool(raw.get('should_trade', False))

    if not action and bot_instruction in instruction_action_map:
        action = instruction_action_map[bot_instruction]
    if action not in allowed_actions:
        action = 'enter' if should_trade else 'skip'

    if order_type not in allowed_order_types:
        order_type = 'limit' if bot_instruction == 'ENTER_LIMIT' else 'market'
    if bot_instruction not in allowed_instructions:
        bot_instruction = 'ENTER_LIMIT' if (action == 'enter' and order_type == 'limit') else ('ENTER_MARKET' if action == 'enter' else 'SKIP')

    if side not in allowed_sides:
        candidate_side = _normalize_side(candidate.get('side'))
        if candidate_side in {'long', 'short'}:
            side = candidate_side
        elif action != 'enter':
            side = 'neutral'
        else:
            side = ''

    decision = {
        'should_trade': should_trade,
        'action': action,
        'market_regime': _normalize_market_regime(raw.get('market_regime')),
        'trend_state': _normalize_trend_state(raw.get('trend_state'), side),
        'timing_state': 'enter_now' if action == 'enter' else 'avoid_near_term',
        'trade_side': side if side in allowed_sides else 'neutral',
        'order_type': order_type,
        'bot_instruction': bot_instruction,
        'entry_price': _num('entry_price', _num('entry', 0.0)),
        'stop_loss': _num('stop_loss', _num('sl', 0.0)),
        'take_profit': _num('take_profit', _num('tp', 0.0)),
        'rr_ratio': max(_num('rr_ratio', _num('rr', _num('risk_reward', 0.0))), 0.0),
        'market_read': _txt_any(['market_read', 'bot_note', 'analysis', 'note'], 220),
        'entry_plan': _txt_any(['entry_plan', 'execution_plan', 'plan', 'bot_note'], 220),
        'stop_logic': _txt_any(['stop_logic', 'stop_loss_reason', 'bot_note'], 220),
        'reason_to_skip': _txt_any(['reason_to_skip', 'skip_reason', 'bot_note'], 220),
        'stop_anchor_timeframe': _txt('stop_anchor_timeframe', 40),
        'stop_anchor_source': _txt('stop_anchor_source', 120),
        'stop_anchor_price': max(_num('stop_anchor_price', 0.0), 0.0),
        'stop_buffer_atr15': max(_num('stop_buffer_atr15', 0.0), 0.0),
        'entry_to_stop_atr15': max(_num('entry_to_stop_atr15', 0.0), 0.0),
        'limit_cancel_price': max(_num('limit_cancel_price', 0.0), 0.0),
        'limit_cancel_timeframe': _txt('limit_cancel_timeframe', 40),
        'limit_cancel_condition': _txt('limit_cancel_condition', 120),
        'limit_cancel_note': _txt('limit_cancel_note', 120),
    }

    warnings: list[str] = []
    for k in (
        'market_read',
        'entry_plan',
        'stop_logic',
        'reason_to_skip',
        'stop_anchor_source',
        'limit_cancel_condition',
        'limit_cancel_note',
    ):
        if _looks_like_schema_placeholder(raw.get(k)):
            warnings.append(f'{k}_schema_placeholder')
    if decision['action'] == 'enter':
        decision['should_trade'] = True
        decision['timing_state'] = 'enter_now'
        decision['bot_instruction'] = 'ENTER_LIMIT' if decision['order_type'] == 'limit' else 'ENTER_MARKET'
        if decision['trade_side'] not in {'long', 'short'}:
            warnings.append('enter_missing_trade_side_downgraded')
            decision['action'] = 'skip'
        elif decision['entry_price'] <= 0 or decision['stop_loss'] <= 0 or decision['take_profit'] <= 0:
            warnings.append('enter_missing_prices_downgraded')
            decision['action'] = 'skip'
    if decision['action'] == 'skip':
        decision['should_trade'] = False
        decision['timing_state'] = 'avoid_near_term'
        decision['bot_instruction'] = 'SKIP'
        decision['order_type'] = 'market'
        decision['entry_price'] = 0.0
        decision['stop_loss'] = 0.0
        decision['take_profit'] = 0.0
        if decision['trade_side'] not in {'long', 'short', 'neutral'}:
            decision['trade_side'] = 'neutral'
        if not str(decision.get('reason_to_skip') or '').strip():
            decision['reason_to_skip'] = '結構/流動性/RR 未達可執行標準，本輪不做。'
    _enforce_non_chase_entry(decision, candidate, str(decision.get('trade_side') or side or ''))

    if decision['market_regime'] not in allowed_regimes:
        decision['market_regime'] = 'transition_chop'
    if decision['trend_state'] not in allowed_trend_states:
        decision['trend_state'] = 'trend_unclear'
    if decision['timing_state'] not in allowed_timing_states:
        decision['timing_state'] = 'enter_now' if decision['action'] == 'enter' else 'avoid_near_term'

    decision['validation_errors'] = warnings[:16]
    decision['valid'] = True
    return decision


def consult_trade_decision(
    *,
    state: Dict[str, Any],
    state_path: str,
    api_key: str,
    config: Dict[str, Any],
    candidate: Dict[str, Any],
    logger: Callable[[str], None] | None = None,
    now_ts: float | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    now_ts = float(now_ts or time.time())
    state = load_trade_state(state_path, now_ts=now_ts) if not isinstance(state, dict) else load_trade_state(state_path, now_ts=now_ts) | dict(state or {})
    candidate = dict(candidate or {})
    symbol = str(candidate.get('symbol') or '')
    payload_hash = _hash_payload(_stable_payload(candidate))
    symbol_state = dict((state.get('symbols') or {}).get(symbol, {}) or {})
    force_recheck = bool(candidate.get('force_recheck', False))
    cooldown_sec = max(int(config.get('cooldown_minutes', 20) or 20), 1) * 60
    global_interval_sec = max(int(config.get('global_min_interval_minutes', 0) or 0), 0) * 60
    rank = int(candidate.get('rank', 99) or 99)

    if not bool(config.get('enabled', True)):
        result = {'status': 'disabled', 'decision': None, 'payload_hash': payload_hash, 'symbol_state': symbol_state}
        _append_recent(state, _build_recent_item(candidate, status='disabled', detail='OpenAI trade control is disabled.'))
        save_trade_state(state_path, state)
        return state, result

    if not str(api_key or '').strip():
        result = {'status': 'missing_api_key', 'decision': None, 'payload_hash': payload_hash, 'symbol_state': symbol_state}
        _append_recent(state, _build_recent_item(candidate, status='missing_api_key', detail='OPENAI_API_KEY is empty.'))
        save_trade_state(state_path, state)
        return state, result

    spent_twd = float(state.get('spent_estimated_twd', 0.0) or 0.0)
    if spent_twd >= float(config.get('hard_budget_twd', 950.0) or 950.0):
        result = {'status': 'budget_paused', 'decision': None, 'payload_hash': payload_hash, 'symbol_state': symbol_state}
        _append_recent(state, _build_recent_item(candidate, status='budget_paused', detail='Monthly OpenAI budget hard stop reached.'))
        save_trade_state(state_path, state)
        return state, result

    last_hash = str(symbol_state.get('last_payload_hash') or '')
    last_sent_ts = float(symbol_state.get('last_sent_ts', 0) or 0)
    cached_decision = dict(symbol_state.get('last_decision') or {})
    last_status = str(symbol_state.get('last_status') or '').strip().lower()
    empty_no_decision = last_status.startswith('empty_response') and last_status != 'empty_response_reuse_cached'
    if empty_no_decision:
        last_sent_ts = 0.0
        if not cached_decision:
            last_hash = ''
    same_payload_reuse_sec = max(int(config.get('same_payload_reuse_minutes', 180) or 180), 1) * 60
    # Cooldown is the only hard resend gate: once cooldown expires, same symbol may be sent again.
    effective_same_payload_reuse_sec = min(same_payload_reuse_sec, cooldown_sec)
    top_signature = _top_candidates_signature(candidate)

    if (
        not force_recheck
        and not empty_no_decision
        and cached_decision
        and last_hash == payload_hash
        and last_sent_ts > 0
        and (now_ts - last_sent_ts) < effective_same_payload_reuse_sec
    ):
        detail = 'Same payload unchanged within reuse window; cached decision reused to avoid duplicate OpenAI cost.'
        result = {
            'status': 'cached_reuse',
            'decision': cached_decision,
            'payload_hash': payload_hash,
            'symbol_state': symbol_state,
        }
        save_trade_state(state_path, state)
        return state, result

    reuse_cached_decision = cached_decision if cached_decision and last_hash == payload_hash else None
    if not force_recheck and not empty_no_decision and last_sent_ts > 0 and (now_ts - last_sent_ts) < cooldown_sec:
        next_allowed_ts = last_sent_ts + cooldown_sec
        detail = 'Symbol cooldown is active until {}.'.format(datetime.fromtimestamp(next_allowed_ts).strftime('%Y-%m-%d %H:%M:%S'))
        if reuse_cached_decision:
            detail += ' Same symbol payload was already sent in the current cooldown window; cached decision can be reused locally.'
        result = {
            'status': 'cooldown_active',
            'decision': reuse_cached_decision,
            'payload_hash': payload_hash,
            'symbol_state': symbol_state,
            'next_allowed_ts': next_allowed_ts,
        }
        _append_recent(state, _build_recent_item(candidate, status='cooldown_active', detail=detail))
        save_trade_state(state_path, state)
        return state, result

    last_consulted_ts = float(state.get('last_consulted_ts', 0) or 0)
    api_calls_total = int(state.get('api_calls', 0) or 0)
    top_changed = bool(top_signature) and top_signature != str(state.get('last_top_candidates_signature') or '')
    if global_interval_sec > 0 and api_calls_total > 0 and last_consulted_ts > 0 and (now_ts - last_consulted_ts) < global_interval_sec and not top_changed and not force_recheck:
        next_allowed_ts = last_consulted_ts + global_interval_sec
        detail = 'Global OpenAI interval is active until {}.'.format(datetime.fromtimestamp(next_allowed_ts).strftime('%Y-%m-%d %H:%M:%S'))
        if reuse_cached_decision:
            detail += ' Current symbol payload is unchanged, so the cached decision can be reused locally.'
        result = {
            'status': 'global_interval_active',
            'decision': reuse_cached_decision,
            'payload_hash': payload_hash,
            'symbol_state': symbol_state,
            'next_allowed_ts': next_allowed_ts,
        }
        save_trade_state(state_path, state)
        return state, result

    estimated_payload_tokens = _estimate_tokens_from_json(_stable_payload(candidate))
    estimated_instruction_tokens = 320
    estimated_output_tokens = max(int(config.get('max_output_tokens', 560) or 560), 760)
    est_call_cost_twd = estimate_cost_twd(
        config,
        input_tokens=estimated_payload_tokens + estimated_instruction_tokens,
        output_tokens=estimated_output_tokens,
    )
    soft_budget_twd = float(config.get('soft_budget_twd', 850.0) or 850.0)
    if spent_twd >= soft_budget_twd and not force_recheck:
        result = {'status': 'budget_paused', 'decision': None, 'payload_hash': payload_hash, 'symbol_state': symbol_state}
        _append_recent(state, _build_recent_item(candidate, status='budget_paused', detail='Soft budget reached. Only force_recheck can bypass this pause.'))
        save_trade_state(state_path, state)
        return state, result
    if spent_twd + est_call_cost_twd >= float(config.get('monthly_budget_twd', 1000.0) or 1000.0) * 1.02:
        result = {'status': 'budget_paused', 'decision': None, 'payload_hash': payload_hash, 'symbol_state': symbol_state}
        _append_recent(state, _build_recent_item(candidate, status='budget_paused', detail='Estimated call cost would exceed the monthly OpenAI budget cap.'))
        save_trade_state(state_path, state)
        return state, result

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    if logger:
        logger('OpenAI trade decision request: {} rank={}'.format(symbol, rank))

    try:
        base_url = str(config.get('base_url') or 'https://api.openai.com/v1/responses')
        timeout_sec = float(config.get('request_timeout_sec', 60.0) or 60.0)
        max_decision_latency_sec = max(float(config.get('max_decision_latency_sec', 45.0) or 45.0), 12.0)
        decision_started_at = time.monotonic()
        primary_model = str(config.get('model') or 'gpt-5.4').strip()
        upgrade_model = str(config.get('upgrade_model') or 'gpt-5.4').strip()
        fallback_model = str(config.get('fallback_model') or '').strip()
        allow_upgrade = bool(config.get('allow_upgrade_model', False))
        # Force non-reasoning mode for execution JSON reliability.
        # Empty responses were frequently caused by output budget being consumed before complete JSON.
        primary_effort = 'none'
        retry_effort = 'none'
        per_call_budget_twd = max(float(config.get('per_call_budget_twd', 0.20) or 0.20), 0.05)
        max_tokens = max(int(config.get('max_output_tokens', 560) or 560), 760)
        attempts = [
            {
                'model': primary_model,
                'structured': True,
                'effort': primary_effort,
                'max_tokens': max_tokens,
                'compact_prompt': True,
            },
        ]
        if bool(config.get('empty_retry_enabled', True)):
            if retry_effort != primary_effort:
                attempts.append(
                    {
                        'model': primary_model,
                        'structured': True,
                        'effort': retry_effort,
                        'max_tokens': int(max_tokens * 1.25),
                        'compact_prompt': True,
                    }
                )
            if allow_upgrade and upgrade_model and upgrade_model != primary_model:
                attempts.append(
                    {
                        'model': upgrade_model,
                        'structured': True,
                        'effort': retry_effort,
                        'max_tokens': int(max_tokens * 1.25),
                        'compact_prompt': True,
                    }
                )
            if fallback_model and fallback_model not in {primary_model, upgrade_model}:
                attempts.append(
                    {
                        'model': fallback_model,
                        'structured': True,
                        'effort': retry_effort,
                        'max_tokens': int(max_tokens * 1.35),
                        'compact_prompt': True,
                    }
                )
            # Last-resort parseable JSON fallback when schema attempts are truncated/incomplete.
            attempts.append(
                {
                    'model': primary_model,
                    'structured': False,
                    'effort': 'none',
                    'max_tokens': int(max_tokens * 1.15),
                    'compact_prompt': True,
                }
            )
        deduped_attempts = []
        seen_attempt_keys = set()
        for attempt in attempts:
            key = (
                str(attempt.get('model') or ''),
                bool(attempt.get('structured', False)),
                str(attempt.get('effort') or ''),
                int(attempt.get('max_tokens') or 0),
                bool(attempt.get('compact_prompt', False)),
            )
            if key in seen_attempt_keys:
                continue
            seen_attempt_keys.add(key)
            deduped_attempts.append(dict(attempt))
        if len(deduped_attempts) <= 3:
            attempts = list(deduped_attempts)
        else:
            # Keep a small retry budget, but always retain one non-structured fallback attempt.
            attempts = list(deduped_attempts[:2])
            tail = list(deduped_attempts[2:])
            non_structured_tail = next(
                (row for row in reversed(tail) if not bool(row.get('structured', True))),
                None,
            )
            attempts.append(dict(non_structured_tail or deduped_attempts[2]))

        body = {}
        raw_text = ''
        raw_json = {}
        selected_model = primary_model
        selected_attempt = {}
        empty_details = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_cached_input_tokens = 0
        actual_calls = 0
        last_attempt_exc: Exception | None = None
        had_successful_response = False
        schema_mode_rejected = False
        for idx, attempt in enumerate(attempts):
            elapsed = time.monotonic() - decision_started_at
            remaining_budget = max_decision_latency_sec - elapsed
            if remaining_budget <= 0:
                empty_details.append(
                    'decision_timeout_budget_exhausted elapsed={:.2f}s budget={:.2f}s'.format(
                        elapsed,
                        max_decision_latency_sec,
                    )
                )
                if logger:
                    logger('OpenAI decision budget exhausted: {} elapsed={:.2f}s'.format(symbol, elapsed))
                break
            selected_attempt = dict(attempt)
            selected_model = str(attempt.get('model') or primary_model)
            if schema_mode_rejected and bool(attempt.get('structured', False)):
                empty_details.append(
                    '{} structured={} skipped=schema_mode_rejected'.format(
                        selected_model,
                        bool(attempt.get('structured', False)),
                    )
                )
                continue
            try:
                request_body = _build_request_body(
                    candidate,
                    config,
                    structured=bool(attempt.get('structured', True)),
                    model=selected_model,
                    reasoning_effort=str(attempt.get('effort') or ''),
                    max_output_tokens=int(attempt.get('max_tokens') or max_tokens),
                    compact_prompt=bool(attempt.get('compact_prompt', False)),
                    logger=logger,
                )
                est_input_tokens_attempt = _estimate_tokens_from_json(request_body)
                est_output_tokens_attempt = max(int(request_body.get('max_output_tokens', attempt.get('max_tokens', max_tokens)) or 0), 0)
                projected_attempt_cost_twd = estimate_cost_twd(
                    config,
                    input_tokens=est_input_tokens_attempt,
                    output_tokens=est_output_tokens_attempt,
                )
                if projected_attempt_cost_twd > per_call_budget_twd:
                    base_input_cost_twd = estimate_cost_twd(config, input_tokens=est_input_tokens_attempt, output_tokens=0)
                    out_cost_per_token_twd = _output_token_cost_twd(config)
                    affordable_output_tokens = 0
                    if out_cost_per_token_twd > 0:
                        affordable_output_tokens = int(max((per_call_budget_twd - base_input_cost_twd) / out_cost_per_token_twd, 0))
                    min_needed_tokens = 260 if bool(attempt.get('structured', True)) else 160
                    if affordable_output_tokens >= min_needed_tokens and affordable_output_tokens < est_output_tokens_attempt:
                        request_body['max_output_tokens'] = affordable_output_tokens
                        est_output_tokens_attempt = affordable_output_tokens
                        projected_attempt_cost_twd = estimate_cost_twd(
                            config,
                            input_tokens=est_input_tokens_attempt,
                            output_tokens=est_output_tokens_attempt,
                        )
                    if projected_attempt_cost_twd > per_call_budget_twd:
                        detail = 'cost_cap_exceeded model={} projected_twd={:.4f} cap_twd={:.4f}'.format(
                            selected_model,
                            projected_attempt_cost_twd,
                            per_call_budget_twd,
                        )
                        empty_details.append(detail)
                        if logger:
                            logger('OpenAI attempt skipped by per-call cost cap: {} {}'.format(symbol, detail))
                        continue
                attempts_left = max(len(attempts) - idx - 1, 0)
                reserve_for_tail = 4.0 * attempts_left
                if remaining_budget <= max(1.5, reserve_for_tail):
                    empty_details.append(
                        'decision_timeout_budget_low remaining={:.2f}s reserve_tail={:.2f}s'.format(
                            remaining_budget,
                            reserve_for_tail,
                        )
                    )
                    if logger:
                        logger(
                            'OpenAI decision budget too low for another request: {} remaining={:.2f}s reserve_tail={:.2f}s'.format(
                                symbol,
                                remaining_budget,
                                reserve_for_tail,
                            )
                        )
                    break
                attempt_timeout = min(timeout_sec, max(4.0, remaining_budget - reserve_for_tail))
                resp = requests.post(base_url, headers=headers, json=request_body, timeout=attempt_timeout)
                actual_calls += 1
                resp.raise_for_status()
                body = resp.json()
                had_successful_response = True
                usage_i = _response_usage(body)
                total_input_tokens += int(usage_i.get('input_tokens', 0) or 0)
                total_output_tokens += int(usage_i.get('output_tokens', 0) or 0)
                total_cached_input_tokens += int(usage_i.get('input_cached_tokens', usage_i.get('cached_input_tokens', 0)) or 0)
                raw_text = _extract_text(body)
                raw_json = _parse_json_text(raw_text)
                if not raw_json:
                    raw_json = _extract_decision_json(body)
                    if raw_json and not raw_text:
                        try:
                            raw_text = json.dumps(raw_json, ensure_ascii=False)
                        except Exception:
                            raw_text = str(raw_json)
                if raw_json:
                    break
                diag = _response_diagnostics(body)
                empty_details.append('{} structured={} effort={} output_tokens={} {}'.format(
                    selected_model,
                    bool(attempt.get('structured', True)),
                    str(attempt.get('effort') or ''),
                    _response_output_tokens(body),
                    diag,
                ))
                if logger:
                    logger('OpenAI empty/invalid JSON response: {} | {}'.format(symbol, empty_details[-1]))
            except Exception as attempt_exc:
                last_attempt_exc = attempt_exc if isinstance(attempt_exc, Exception) else Exception(str(attempt_exc))
                detail = str(last_attempt_exc)
                try:
                    response = getattr(last_attempt_exc, 'response', None)
                    status_code = int(getattr(response, 'status_code', 0) or 0)
                    body_text = ''
                    if response is not None:
                        body_text = str(getattr(response, 'text', '') or '')
                    if status_code > 0:
                        detail = 'HTTP {} {}'.format(status_code, detail)
                    if body_text:
                        detail += ' | body=' + body_text[:180]
                    body_text_l = body_text.lower()
                    if (
                        status_code == 400
                        and 'invalid schema' in body_text_l
                        and 'trade_decision' in body_text_l
                    ):
                        schema_mode_rejected = True
                except Exception:
                    pass
                empty_details.append('{} structured={} effort={} request_error={}'.format(
                    selected_model,
                    bool(attempt.get('structured', True)),
                    str(attempt.get('effort') or ''),
                    detail[:220],
                ))
                if logger:
                    logger('OpenAI request attempt failed: {} | {}'.format(symbol, empty_details[-1]))
                continue
        if not raw_json:
            if (not had_successful_response) and last_attempt_exc is not None:
                raise last_attempt_exc
            est_cost_usd = estimate_cost_usd(config, input_tokens=total_input_tokens, output_tokens=total_output_tokens, cached_input_tokens=total_cached_input_tokens)
            est_cost_twd = est_cost_usd * float(config.get('usd_to_twd', 32.0) or 32.0)
            detail = 'OpenAI 本輪未回傳可解析的完整 JSON，已停止自動判斷：{}'.format(' ; '.join(empty_details))
            reusable_cached = dict(cached_decision or {}) if cached_decision and last_hash == payload_hash else {}
            if reusable_cached:
                reused = _normalize_decision(reusable_cached, candidate)
                symbol_state.update({
                    'last_payload_hash': payload_hash,
                    'last_sent_ts': now_ts,
                    'last_model': selected_model,
                    'last_decision': dict(reused or {}),
                    'last_status': 'empty_response_reuse_cached',
                    'last_response_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'last_cost_twd': round(est_cost_twd, 4),
                    'last_attempt': dict(selected_attempt),
                    'last_empty_response_ts': now_ts,
                    'last_empty_payload_hash': payload_hash,
                    'last_error': detail[:300],
                })
                state.setdefault('symbols', {})[symbol] = symbol_state
                state['api_calls'] = int(state.get('api_calls', 0) or 0) + max(actual_calls, 1)
                state['input_tokens'] = int(state.get('input_tokens', 0) or 0) + total_input_tokens
                state['output_tokens'] = int(state.get('output_tokens', 0) or 0) + total_output_tokens
                state['cached_input_tokens'] = int(state.get('cached_input_tokens', 0) or 0) + total_cached_input_tokens
                state['last_consulted_ts'] = now_ts
                state['last_top_candidates_signature'] = top_signature
                state['spent_estimated_usd'] = round(float(state.get('spent_estimated_usd', 0.0) or 0.0) + est_cost_usd, 6)
                state['spent_estimated_twd'] = round(float(state.get('spent_estimated_twd', 0.0) or 0.0) + est_cost_twd, 4)
                state['last_error'] = detail[:300]
                state['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                _append_recent(
                    state,
                    _build_recent_item(
                        candidate,
                        status='empty_response_reuse_cached',
                        action=str(reused.get('action') or ('enter' if reused.get('should_trade') else 'skip')),
                        detail='OpenAI 空回覆，沿用同 payload 最近一次可解析決策。',
                        decision=reused,
                        model=selected_model,
                    ),
                )
                save_trade_state(state_path, state)
                return state, {
                    'status': 'empty_response_reuse_cached',
                    'decision': reused,
                    'payload_hash': payload_hash,
                    'symbol_state': symbol_state,
                    'usage': {
                        'input_tokens': total_input_tokens,
                        'output_tokens': total_output_tokens,
                        'cached_input_tokens': total_cached_input_tokens,
                    },
                    'estimated_cost_twd': round(est_cost_twd, 4),
                    'estimated_cost_usd': round(est_cost_usd, 6),
                    'error': detail,
                    'fallback_used': True,
                    'model': selected_model,
                    'attempt': dict(selected_attempt),
                }
            fallback_decision = _fallback_trade_decision(candidate, reason='openai_empty_response')
            symbol_state.update({
                'last_payload_hash': payload_hash,
                'last_sent_ts': now_ts,
                'last_model': selected_model,
                'last_decision': dict(fallback_decision or {}),
                'last_status': 'empty_response_fallback_skip',
                'last_response_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'last_cost_twd': round(est_cost_twd, 4),
                'last_attempt': dict(selected_attempt),
                'last_empty_response_ts': now_ts,
                'last_empty_payload_hash': payload_hash,
                'last_error': detail[:300],
            })
            state.setdefault('symbols', {})[symbol] = symbol_state
            state['api_calls'] = int(state.get('api_calls', 0) or 0) + max(actual_calls, 1)
            state['input_tokens'] = int(state.get('input_tokens', 0) or 0) + total_input_tokens
            state['output_tokens'] = int(state.get('output_tokens', 0) or 0) + total_output_tokens
            state['cached_input_tokens'] = int(state.get('cached_input_tokens', 0) or 0) + total_cached_input_tokens
            state['spent_estimated_usd'] = round(float(state.get('spent_estimated_usd', 0.0) or 0.0) + est_cost_usd, 6)
            state['spent_estimated_twd'] = round(float(state.get('spent_estimated_twd', 0.0) or 0.0) + est_cost_twd, 4)
            state['last_error'] = detail[:300]
            state['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            _append_recent(
                state,
                _build_recent_item(
                    candidate,
                    status='empty_response_fallback_skip',
                    action='skip',
                    detail='OpenAI 空回覆或 JSON 不完整；已回退為可執行 SKIP 指令，本輪不下單。',
                    decision=fallback_decision,
                    model=selected_model,
                ),
            )
            save_trade_state(state_path, state)
            return state, {
                'status': 'empty_response_fallback_skip',
                'decision': fallback_decision,
                'payload_hash': payload_hash,
                'symbol_state': symbol_state,
                'usage': {
                    'input_tokens': total_input_tokens,
                    'output_tokens': total_output_tokens,
                    'cached_input_tokens': total_cached_input_tokens,
                },
                'estimated_cost_twd': round(est_cost_twd, 4),
                'estimated_cost_usd': round(est_cost_usd, 6),
                'error': detail,
                'fallback_used': False,
                'model': selected_model,
                'attempt': dict(selected_attempt),
            }
        decision = _normalize_decision(raw_json, candidate)
        decision_valid = bool(decision.get('valid', True))
        decision_status = 'consulted' if decision_valid else 'invalid_decision'
        decision_detail = (
            'OpenAI 已回傳可解析的結構化交易判斷。'
            if decision_valid
            else 'OpenAI 回傳 JSON 但未通過本地 schema/mapping 驗證；本輪不執行交易。'
        )
        decision_error = ';'.join(list(decision.get('validation_errors') or [])[:4]) if not decision_valid else ''
        usage = dict(body.get('usage') or {})
        input_tokens = total_input_tokens or int(usage.get('input_tokens', 0) or 0)
        output_tokens = total_output_tokens or int(usage.get('output_tokens', 0) or 0)
        cached_input_tokens = total_cached_input_tokens or int(usage.get('input_cached_tokens', usage.get('cached_input_tokens', 0)) or 0)
        est_cost_usd = estimate_cost_usd(config, input_tokens=input_tokens, output_tokens=output_tokens, cached_input_tokens=cached_input_tokens)
        est_cost_twd = est_cost_usd * float(config.get('usd_to_twd', 32.0) or 32.0)

        symbol_state.update({
            'last_payload_hash': payload_hash,
            'last_sent_ts': now_ts,
            'last_model': selected_model,
            'last_decision': dict(decision or {}),
            'last_status': decision_status,
            'last_response_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'last_cost_twd': round(est_cost_twd, 4),
            'last_attempt': dict(selected_attempt),
            'last_error': decision_error[:300],
        })
        state.setdefault('symbols', {})[symbol] = symbol_state
        state['api_calls'] = int(state.get('api_calls', 0) or 0) + max(actual_calls, 1)
        state['input_tokens'] = int(state.get('input_tokens', 0) or 0) + input_tokens
        state['output_tokens'] = int(state.get('output_tokens', 0) or 0) + output_tokens
        state['cached_input_tokens'] = int(state.get('cached_input_tokens', 0) or 0) + cached_input_tokens
        state['last_consulted_ts'] = now_ts
        state['last_top_candidates_signature'] = top_signature
        state['spent_estimated_usd'] = round(float(state.get('spent_estimated_usd', 0.0) or 0.0) + est_cost_usd, 6)
        state['spent_estimated_twd'] = round(float(state.get('spent_estimated_twd', 0.0) or 0.0) + est_cost_twd, 4)
        state['last_error'] = decision_error[:300]
        state['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        _append_recent(
            state,
            _build_recent_item(
                candidate,
                status=decision_status,
                action=str(decision.get('action') or ('enter' if decision.get('should_trade') else 'skip')),
                detail=decision_detail,
                decision=decision,
                model=selected_model,
            ),
        )
        save_trade_state(state_path, state)
        return state, {
            'status': decision_status,
            'decision': decision,
            'payload_hash': payload_hash,
            'symbol_state': symbol_state,
            'usage': usage,
            'estimated_cost_twd': round(est_cost_twd, 4),
            'estimated_cost_usd': round(est_cost_usd, 6),
            'raw_text': raw_text[:1200],
            'model': selected_model,
            'attempt': dict(selected_attempt),
            'error': decision_error[:300],
        }
    except Exception as exc:
        err = str(exc)
        status = 'error'
        detail = err
        try:
            response = getattr(exc, 'response', None)
            status_code = int(getattr(response, 'status_code', 0) or 0)
            body_text = ''
            if response is not None:
                try:
                    body_text = response.text or ''
                except Exception:
                    body_text = ''
            if status_code == 401:
                status = 'auth_error'
                detail = (
                    'OpenAI authentication failed (401). '
                    'Check OPENAI_API_KEY in deployment env. '
                    'Use an API key from platform.openai.com, not a ChatGPT login/subscription token. '
                    'Also remove quotes, spaces, and trailing newlines.'
                )
                if body_text:
                    detail += ' | body=' + body_text[:180]
            elif status_code == 403:
                status = 'permission_error'
                detail = 'OpenAI request forbidden (403). Check project permission, billing, and model access.'
                if body_text:
                    detail += ' | body=' + body_text[:180]
            elif status_code == 429:
                status = 'rate_limit'
                detail = 'OpenAI rate limit hit (429). Slow down requests or check usage limits.'
                if body_text:
                    detail += ' | body=' + body_text[:180]
            elif status_code == 400:
                status = 'bad_request'
                detail = 'OpenAI request body rejected (400). The deployment can reach the API, but one or more request fields are invalid for the current model/account.'
                if body_text:
                    detail += ' | body=' + body_text[:220]
        except Exception:
            pass
        usage_input_tokens = int(locals().get('total_input_tokens', 0) or 0)
        usage_output_tokens = int(locals().get('total_output_tokens', 0) or 0)
        usage_cached_input_tokens = int(locals().get('total_cached_input_tokens', 0) or 0)
        actual_calls = int(locals().get('actual_calls', 0) or 0)
        selected_model = str(locals().get('selected_model', config.get('model') or 'gpt-5.4-mini') or 'gpt-5.4-mini')
        selected_attempt = dict(locals().get('selected_attempt', {}) or {})
        est_cost_usd = estimate_cost_usd(
            config,
            input_tokens=usage_input_tokens,
            output_tokens=usage_output_tokens,
            cached_input_tokens=usage_cached_input_tokens,
        )
        est_cost_twd = est_cost_usd * float(config.get('usd_to_twd', 32.0) or 32.0)
        symbol_state.update({
            'last_payload_hash': payload_hash,
            'last_sent_ts': now_ts,
            'last_model': selected_model,
            'last_decision': dict(symbol_state.get('last_decision') or {}),
            'last_status': status,
            'last_response_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'last_cost_twd': round(est_cost_twd, 4),
            'last_attempt': dict(selected_attempt),
            'last_error': detail[:300],
        })
        state.setdefault('symbols', {})[symbol] = symbol_state
        if actual_calls > 0:
            state['api_calls'] = int(state.get('api_calls', 0) or 0) + actual_calls
            state['input_tokens'] = int(state.get('input_tokens', 0) or 0) + usage_input_tokens
            state['output_tokens'] = int(state.get('output_tokens', 0) or 0) + usage_output_tokens
            state['cached_input_tokens'] = int(state.get('cached_input_tokens', 0) or 0) + usage_cached_input_tokens
            state['spent_estimated_usd'] = round(float(state.get('spent_estimated_usd', 0.0) or 0.0) + est_cost_usd, 6)
            state['spent_estimated_twd'] = round(float(state.get('spent_estimated_twd', 0.0) or 0.0) + est_cost_twd, 4)
        state['last_consulted_ts'] = now_ts
        state['last_top_candidates_signature'] = top_signature
        state['last_error'] = detail[:300]
        state['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        _append_recent(state, _build_recent_item(candidate, status=status, detail=detail[:260], model=selected_model))
        save_trade_state(state_path, state)
        if logger:
            logger('OpenAI trade decision failed: {} | {}'.format(symbol, detail[:240]))
        return state, {
            'status': status,
            'decision': None,
            'payload_hash': payload_hash,
            'symbol_state': symbol_state,
            'error': detail,
            'estimated_cost_twd': round(est_cost_twd, 4),
            'estimated_cost_usd': round(est_cost_usd, 6),
            'usage': {
                'input_tokens': usage_input_tokens,
                'output_tokens': usage_output_tokens,
                'cached_input_tokens': usage_cached_input_tokens,
            },
            'model': selected_model,
            'attempt': dict(selected_attempt),
        }


def build_dashboard_payload(state: Dict[str, Any], config: Dict[str, Any], *, api_key_present: bool) -> Dict[str, Any]:
    state = dict(state or {})
    budget_total = float(config.get('monthly_budget_twd', 1000.0) or 1000.0)
    spent_twd = float(state.get('spent_estimated_twd', 0.0) or 0.0)
    remaining = max(budget_total - spent_twd, 0.0)
    pending = dict(state.get('pending_advice') or {})
    pending_rows = sorted(
        [dict(row or {}) for row in pending.values()],
        key=lambda row: (float(row.get('last_checked_ts', 0) or 0), float(row.get('created_ts', 0) or 0)),
        reverse=True,
    )
    recent_rows = [
        dict(row or {})
        for row in list(state.get('recent_decisions') or [])
    ][:12]
    return {
        'enabled': bool(config.get('enabled', True)),
        'api_key_present': bool(api_key_present),
        'model': str(config.get('model') or ''),
        'month_key': str(state.get('month_key') or ''),
        'budget_total_twd': round(budget_total, 2),
        'budget_soft_twd': round(float(config.get('soft_budget_twd', 0.0) or 0.0), 2),
        'budget_hard_twd': round(float(config.get('hard_budget_twd', 0.0) or 0.0), 2),
        'spent_estimated_twd': round(spent_twd, 4),
        'spent_estimated_usd': round(float(state.get('spent_estimated_usd', 0.0) or 0.0), 6),
        'remaining_estimated_twd': round(remaining, 4),
        'budget_paused': bool(spent_twd >= float(config.get('hard_budget_twd', budget_total) or budget_total)),
        'api_calls': int(state.get('api_calls', 0) or 0),
        'input_tokens': int(state.get('input_tokens', 0) or 0),
        'output_tokens': int(state.get('output_tokens', 0) or 0),
        'cached_input_tokens': int(state.get('cached_input_tokens', 0) or 0),
        'top_k_per_scan': int(config.get('top_k_per_scan', 5) or 5),
        'sends_per_scan': int(config.get('sends_per_scan', 1) or 1),
        'cooldown_minutes': int(config.get('cooldown_minutes', 20) or 20),
        'global_min_interval_minutes': int(config.get('global_min_interval_minutes', 0) or 0),
        'advice_ttl_minutes': int(config.get('advice_ttl_minutes', 240) or 240),
        'pending_advice_count': len(pending),
        'pending_advice': pending_rows[:20],
        'min_score_abs': float(config.get('min_score_abs', 0.0) or 0.0),
        'last_error': str(state.get('last_error') or ''),
        'updated_at': str(state.get('updated_at') or ''),
        'recent_decisions': recent_rows,
    }
