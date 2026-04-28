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
        'cooldown_minutes': max(_env_int(env_getter, 'OPENAI_TRADE_SYMBOL_COOLDOWN_MINUTES', 180), 1),
        'same_payload_reuse_minutes': max(_env_int(env_getter, 'OPENAI_TRADE_SAME_PAYLOAD_REUSE_MINUTES', 180), 1),
        'global_min_interval_minutes': max(_env_int(env_getter, 'OPENAI_TRADE_GLOBAL_MIN_INTERVAL_MINUTES', 20), 0),
        'min_score_abs': max(_env_float(env_getter, 'OPENAI_TRADE_MIN_SCORE', 40.0), 0.0),
        'min_margin_pct': min(max(_env_float(env_getter, 'OPENAI_TRADE_MIN_MARGIN_PCT', 0.03), 0.005), 0.5),
        'max_margin_pct': min(max(_env_float(env_getter, 'OPENAI_TRADE_MAX_MARGIN_PCT', 0.08), 0.01), 0.8),
        'min_leverage': max(_env_int(env_getter, 'OPENAI_TRADE_MIN_LEVERAGE', 4), 1),
        'max_leverage': max(_env_int(env_getter, 'OPENAI_TRADE_MAX_LEVERAGE', 25), 1),
        'max_output_tokens': max(_env_int(env_getter, 'OPENAI_TRADE_MAX_OUTPUT_TOKENS', 560), 320),
        'request_timeout_sec': max(_env_float(env_getter, 'OPENAI_TRADE_TIMEOUT_SEC', 18.0), 5.0),
        'max_decision_latency_sec': max(_env_float(env_getter, 'OPENAI_TRADE_MAX_DECISION_LATENCY_SEC', 36.0), 12.0),
        'temperature': 0.2,
        'base_url': str(env_getter('OPENAI_RESPONSES_URL', 'https://api.openai.com/v1/responses') or 'https://api.openai.com/v1/responses').strip(),
        'reasoning_effort': str(env_getter('OPENAI_TRADE_REASONING_EFFORT', 'medium') or 'medium').strip(),
        'retry_reasoning_effort': str(env_getter('OPENAI_TRADE_RETRY_REASONING_EFFORT', 'medium') or 'medium').strip(),
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
        '沒有',
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
    liquidity_context = _ensure_keys(
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
            'aggressive_buy_volume',
            'aggressive_sell_volume',
            'aggressive_buy_notional',
            'aggressive_sell_notional',
            'buy_sell_notional_ratio',
            'cvd_notional',
            'volume_anomaly_5m',
            'volume_anomaly_15m',
        ],
        '沒有',
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
    derivatives_src = dict(src.get('derivatives_context') or market_context.get('derivatives_context') or {})
    leverage_heat = str(derivatives_src.get('leverage_heat') or '').strip()
    if leverage_heat:
        derivatives_context['leverage_heat'] = leverage_heat[:24]
    liquidation_map_status = str(derivatives_src.get('liquidation_map_status') or '').strip()
    if liquidation_map_status:
        derivatives_context['liquidation_map_status'] = liquidation_map_status[:24]
    derivatives_context = _ensure_keys(
        derivatives_context,
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
            'leverage_heat',
            'leverage_heat_score',
            'liquidation_map_status',
            'mark_price',
            'index_price',
        ],
        '沒有',
    )
    risk = {
        'trading_ok': bool((src.get('risk') or {}).get('trading_ok', True)),
        'daily_loss_pct': _compact_number((src.get('risk') or {}).get('daily_loss_pct')),
        'consecutive_loss': int(((src.get('risk') or {}).get('consecutive_loss', 0) or 0)),
    }
    portfolio = {
        'equity': _compact_number((src.get('portfolio') or {}).get('equity')),
        'active_position_count': int(((src.get('portfolio') or {}).get('active_position_count', 0) or 0)),
        'same_direction_count': int(((src.get('portfolio') or {}).get('same_direction_count', 0) or 0)),
        'long_count': int(((src.get('portfolio') or {}).get('long_count', 0) or 0)),
        'short_count': int(((src.get('portfolio') or {}).get('short_count', 0) or 0)),
        'open_symbols': [str(x) for x in list((src.get('portfolio') or {}).get('open_symbols') or [])[:8] if str(x).strip()],
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
    orderbook_history = _ensure_keys(
        orderbook_history,
        [
            'depth_imbalance_change_1m',
            'depth_imbalance_change_5m',
            'largest_bid_wall_size_change_3m_pct',
            'largest_ask_wall_size_change_3m_pct',
            'bid_wall_following_price',
            'ask_wall_getting_thinner',
            'wall_pull_or_spoof_risk',
        ],
        '沒有',
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
    flow_context = _ensure_keys(
        flow_context,
        [
            'market_buy_notional_1m',
            'market_sell_notional_1m',
            'market_buy_sell_ratio_1m',
            'market_buy_sell_ratio_5m',
            'cvd_notional_1m',
            'cvd_notional_5m',
            'cvd_slope_1m',
            'cvd_slope_5m',
            'cvd_slope_15m',
            'large_trade_count_1m',
        ],
        '沒有',
    )
    clean_payload = {
        'symbol': symbol,
        'trade_style': trade_style,
        'candidate_source': _short_text(src.get('candidate_source') or 'general', 48),
        'strategy_lane': _short_text(src.get('strategy_lane') or (src.get('constraints') or {}).get('strategy_lane') or '', 64),
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
        'risk': risk,
        'portfolio': portfolio,
        'execution_policy': execution_policy,
        'constraints': constraints,
        'raw_candidate_payload': raw_candidate_payload_clean if str(src.get('candidate_source') or '').strip().lower() == 'prebreakout_scanner' else {},
        'force_recheck': bool(src.get('force_recheck', False)),
    }
    return clean_payload


def _stable_payload(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return _build_clean_payload(candidate)


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
    candidate_source = str(signal.get('candidate_source') or signal.get('source') or 'general')[:80]
    source_lower = candidate_source.strip().lower()
    if source_lower == 'short_gainers':
        strategy_lane = 'gainer_followup_discovery'
    elif source_lower == 'prebreakout_scanner':
        strategy_lane = 'pre_breakout_discovery'
    else:
        strategy_lane = 'trend_following_execution'
    raw_candidate_payload = dict(
        signal.get('raw_candidate_payload')
        or signal.get('prebreakout_raw_candidate_payload')
        or {}
    )
    return {
        'symbol': str(signal.get('symbol') or ''),
        'trade_style': str(constraints.get('trade_style') or 'short_term_intraday'),
        'candidate_source': candidate_source,
        'strategy_lane': strategy_lane,
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
        'risk': {
            'trading_ok': bool(risk_status.get('trading_ok', True)),
            'halt_reason': str(risk_status.get('halt_reason') or '')[:180],
            'daily_loss_pct': _compact_number(risk_status.get('daily_loss_pct')),
            'consecutive_loss': int(risk_status.get('consecutive_loss', 0) or 0),
        },
        'portfolio': {
            'equity': _compact_number(portfolio.get('equity')),
            'active_position_count': int(portfolio.get('active_position_count', 0) or 0),
            'same_direction_count': int(portfolio.get('same_direction_count', 0) or 0),
            'long_count': int(portfolio.get('long_count', 0) or 0),
            'short_count': int(portfolio.get('short_count', 0) or 0),
            'open_symbols': list(portfolio.get('open_symbols') or [])[:8],
        },
        'execution_policy': _compact_mapping(execution_policy, ['fixed_leverage', 'leverage_mode', 'min_order_margin_usdt', 'fixed_order_notional_usdt', 'margin_pct_range'], text_limit=100),
        'constraints': _compact_mapping(
            dict(constraints or {}),
            [
                'min_margin_pct', 'max_margin_pct', 'min_leverage', 'max_leverage', 'fixed_leverage',
                'leverage_policy', 'min_order_margin_usdt', 'fixed_order_notional_usdt', 'trade_style',
                'max_open_positions', 'max_same_direction', 'candidate_source', 'strategy_lane',
            ],
            text_limit=100,
        ),
        'raw_candidate_payload': raw_candidate_payload if source_lower == 'prebreakout_scanner' else {},
        'force_recheck': bool(signal.get('force_openai_recheck', False)),
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
        't': _short_text(src.get('trend_label') or '', 24),
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
        'sb': _short_text(src.get('structure_bias') or '', 20),
        'ts': _short_text(src.get('trend_stack') or '', 20),
        'rb': _short_text(src.get('recent_break') or '', 20),
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
            return {}
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
        'action': {'type': 'string', 'enum': ['enter', 'observe', 'skip']},
        'market_regime': {'type': 'string', 'enum': ['trend_continuation', 'trend_pullback', 'range_reversion', 'consolidation_squeeze', 'transition_chop']},
        'regime_note': _string_schema(180, 8),
        'trend_state': {'type': 'string', 'enum': ['trending_up', 'trending_down', 'range_mixed', 'transitioning', 'trend_unclear']},
        'timing_state': {'type': 'string', 'enum': ['enter_now', 'wait_pullback', 'wait_breakout', 'avoid_near_term']},
        'trade_side': {'type': 'string', 'enum': ['long', 'short']},
        'breakout_assessment': _string_schema(180, 10),
        'rr_ratio': {'type': 'number'},
        'scale_in_recommended': {'type': 'boolean'},
        'scale_in_price': {'type': 'number'},
        'scale_in_qty_pct': {'type': 'number'},
        'scale_in_condition': _string_schema(160),
        'scale_in_note': _string_schema(180),
        'order_type': {'type': 'string', 'enum': ['market', 'limit']},
        'bot_instruction': {'type': 'string', 'enum': ['ENTER_MARKET', 'ENTER_LIMIT', 'OBSERVE', 'SKIP']},
        'entry_price': {'type': 'number'},
        'candidate_entry_price': {'type': 'number'},
        'stop_loss': {'type': 'number'},
        'take_profit': {'type': 'number'},
        'market_read': _string_schema(320, 18),
        'entry_plan': _string_schema(320, 18),
        'leverage': {'type': 'integer'},
        'margin_pct': {'type': 'number'},
        'confidence': {'type': 'number'},
        'thesis': _string_schema(220),
        'reason_to_skip': _string_schema(220),
        'risk_notes': _string_array_schema(4, 120),
        'aggressive_note': _string_schema(180),
        'watch_trigger_type': {'type': 'string', 'enum': ['none', 'pullback_to_entry', 'breakout_reclaim', 'breakdown_confirm', 'volume_confirmation', 'manual_review']},
        'watch_trigger_price': {'type': 'number'},
        'watch_invalidation_price': {'type': 'number'},
        'watch_note': _string_schema(220),
        'recheck_reason': _string_schema(220),
        'watch_timeframe': _string_schema(40),
        'watch_price_zone_low': {'type': 'number'},
        'watch_price_zone_high': {'type': 'number'},
        'watch_structure_condition': _string_schema(240),
        'watch_volume_condition': _string_schema(240),
        'watch_checklist': _string_array_schema(5, 120),
        'watch_confirmations': _string_array_schema(5, 120),
        'watch_invalidations': _string_array_schema(5, 120),
        'watch_trigger_candle': {'type': 'string', 'enum': ['none', 'close_above', 'close_below']},
        'watch_retest_rule': {'type': 'string', 'enum': ['none', 'hold_above', 'hold_below', 'fail_above', 'fail_below']},
        'watch_volume_ratio_min': {'type': 'number'},
        'watch_micro_vwap_rule': {'type': 'string', 'enum': ['none', 'above', 'below']},
        'watch_micro_ema20_rule': {'type': 'string', 'enum': ['none', 'above', 'below']},
        'watch_recheck_priority': {'type': 'number'},
        'limit_cancel_price': {'type': 'number'},
        'limit_cancel_timeframe': _string_schema(40),
        'limit_cancel_condition': _string_schema(220),
        'limit_cancel_note': _string_schema(220),
        'stop_anchor_timeframe': _string_schema(40),
        'stop_anchor_source': _string_schema(120),
        'stop_anchor_price': {'type': 'number'},
        'stop_buffer_atr15': {'type': 'number'},
        'entry_to_stop_atr15': {'type': 'number'},
        'stop_logic': _string_schema(320),
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
            'breakout_assessment',
            'entry_price',
            'stop_loss',
            'take_profit',
            'rr_ratio',
            'order_type',
            'bot_instruction',
            'market_read',
            'entry_plan',
            'reason_to_skip',
            'watch_trigger_type',
            'watch_trigger_candle',
            'watch_retest_rule',
            'watch_micro_vwap_rule',
            'watch_micro_ema20_rule',
            'recheck_reason',
            'stop_logic',
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
    buffer = max(atr_price * 0.16, entry_price * 0.0012)
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
    min_gap = max(atr_price * 1.05, entry_price * 0.005)
    max_gap = max(atr_price * (2.15 if explosive else 2.8), min_gap * 1.4)
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
    same_bar_tol = max(atr_price * 0.18, entry_price * 0.0012)
    anti_sweep_buffer = max(atr_price * 0.16, entry_price * 0.0012)
    min_gap = max(atr_price * 1.0, entry_price * 0.005)
    max_gap = max(
        atr_price * (2.1 if bool(guard.get('explosive', False)) else 2.8),
        min_gap * 1.4,
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
        note = '已依 15m 結構把止損放到結構外側，優先使用前幾根 swing，並避開盤整中段與同根K掃損位。'
        existing = str(decision.get('stop_loss_reason') or '').strip()
        decision['stop_loss_reason'] = note if not existing else f'{note} {existing}'


def _constraint_brief(candidate: Dict[str, Any], constraints: Dict[str, Any]) -> str:
    trade_style = str(candidate.get('trade_style') or constraints.get('trade_style') or 'short_term_intraday')
    min_rr = max(_coerce_float(constraints.get('min_rr_for_entry', 0.0), 0.0), 0.0)
    tp_mode = str(constraints.get('execution_tp_mode') or 'ai_direct')
    strategy_lane = str(constraints.get('strategy_lane') or candidate.get('strategy_lane') or '').strip().lower()
    candidate_source = str(constraints.get('candidate_source') or candidate.get('candidate_source') or '').strip().lower()
    return (
        f'style={trade_style}; '
        f'leverage=symbol_max_{int(constraints.get("fixed_leverage", constraints.get("max_leverage", 25)) or 25)}x; '
        f'margin_pct={float(constraints.get("min_margin_pct", 0.03) or 0.03):.4f}-{float(constraints.get("max_margin_pct", 0.08) or 0.08):.4f}; '
        f'notional={float(constraints.get("fixed_order_notional_usdt", 20.0) or 20.0):.4f} USDT; '
        f'min_margin={float(constraints.get("min_order_margin_usdt", 0.1) or 0.1):.4f} USDT; '
        f'min_rr={min_rr:.2f}; '
        f'tp_execution_mode={tp_mode}; '
        f'strategy_lane={strategy_lane or "auto"}; '
        f'candidate_source={candidate_source or "auto"}'
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
    min_margin_pct = float(constraints.get('min_margin_pct', 0.03) or 0.03)
    max_margin_pct = float(constraints.get('max_margin_pct', 0.08) or 0.08)
    min_leverage = int(constraints.get('min_leverage', 4) or 4)
    max_leverage = int(constraints.get('max_leverage', 25) or 25)
    fixed_leverage = int(constraints.get('fixed_leverage', max_leverage) or max_leverage)
    min_order_margin_usdt = float(constraints.get('min_order_margin_usdt', 0.1) or 0.1)
    fixed_order_notional_usdt = float(constraints.get('fixed_order_notional_usdt', 20.0) or 20.0)
    min_rr_for_entry = max(_coerce_float(constraints.get('min_rr_for_entry', 0.0), 0.0), 0.0)
    trade_style = str(clean_payload.get('trade_style') or constraints.get('trade_style') or 'short_term_intraday')
    schema_hint = _response_shape_hint()
    constraint_brief = _constraint_brief(clean_payload, constraints)
    candidate_source = str(clean_payload.get('candidate_source') or constraints.get('candidate_source') or 'general').strip().lower()
    strategy_lane = str(clean_payload.get('strategy_lane') or constraints.get('strategy_lane') or '').strip().lower()
    if not strategy_lane:
        if candidate_source == 'short_gainers':
            strategy_lane = 'gainer_followup_discovery'
        elif candidate_source == 'prebreakout_scanner':
            strategy_lane = 'pre_breakout_discovery'
        else:
            strategy_lane = 'trend_following_execution'
    if strategy_lane in {'countertrend_reversal', 'countertrend_short_reversal', 'gainer_followup_discovery'}:
        lane_instruction = (
            'Source lane: short_gainers follow-up mode. This lane only means 24h movers are selected as candidates; '
            'do not assume reversal or continuation by default. Re-derive direction from raw structure/flow. '
            'Be aggressive when edge is clear, but avoid late chase entries; prefer executable pullback/FVG/VWAP limit entries when stretched.'
        )
    elif strategy_lane == 'pre_breakout_discovery':
        lane_instruction = (
            'Source lane: prebreakout candidate scanner mode. Scanner outputs are low-trust candidate-selection metadata only; '
            'they cannot decide direction, entry, stop, take-profit, RR, timing, bot instruction, or should_trade.'
        )
    else:
        lane_instruction = (
            'Source lane: general leaderboard trend-following mode. Follow direction, but never chase late extension; '
            'if price is stretched, prefer a precise limit entry at FVG/pullback zone instead of market chasing.'
        )
    prebreakout_rules = (
        'If candidate_source=prebreakout_scanner: treat it as candidate-selection only. '
        'Do not assume direction from source label. Re-derive setup from raw OHLCV, multi-timeframe indicators, liquidity_context, orderbook_history, flow_context, derivatives_context, and risk constraints. '
        'The scanner may find candidates, but it cannot decide side, entry_price, stop_loss, take_profit, RR, timing_state, bot_instruction, or should_trade. '
        'If raw data confirms compression + directional flow + executable invalidation, prefer ENTER_LIMIT near pullback/FVG/VWAP rather than waiting too late. '
        'If raw data shows wall pressure, too-close resistance/support, wide spread, unconfirmed flow, or unclear invalidation, return OBSERVE or SKIP. '
        if candidate_source == 'prebreakout_scanner'
        else ''
    )
    setup_priority_rules = (
        'Always evaluate three opportunity archetypes from raw data: '
        'trend-follow continuation/pullback, exhaustion reversal/reclaim, and pre-breakout compression release. '
        'Pick whichever has the clearest executable edge now; do not force one archetype because of source lane.'
    )
    rr_discipline_rules = (
        'RR discipline (soft guidance): prioritize asymmetric RR and clear invalidation. '
        'Prefer RR >= 1.20; RR around 1.00-1.20 can still ENTER when structure, flow, and liquidity confirmation are strong and execution is clean. '
        'If RR is below 1.00, default to OBSERVE or SKIP unless there is exceptional mispricing edge with very clear invalidation.'
    )
    soft_guidance_note = (
        'All rules/checklists below are soft guidance, not hard constraints. '
        'Prioritize raw data analysis and choose the most risk-adjusted executable plan.'
    )
    aggressive_precision_note = (
        'Lean more aggressive: when edge is reasonably clear and executable now, prefer ENTER over excessive waiting. '
        'Aggressive does not mean random; invalidation and execution quality must stay clear.'
    )
    if compact:
        system_text = (
            'You are the only trade-decision brain. Local bot is data-collection + constraint-check + execution only. '
            'Return exactly one valid compact JSON object, no markdown. Numeric fields must stay numeric. Human-readable text in Traditional Chinese. '
            + soft_guidance_note + ' '
            'Ignore machine scores/labels/prior plans; low-trust metadata cannot decide entry. '
            'Re-derive from raw OHLCV + multi-timeframe indicators + liquidity + derivatives + risk constraints. '
            'Always output: market_regime, trend_state, timing_state, action, trade_side, order_type, bot_instruction, entry_price, stop_loss, take_profit, rr_ratio. '
            'Market regime must be one of trend_continuation/trend_pullback/range_reversion/consolidation_squeeze/transition_chop, and execution must match regime. '
            'Aggressive but precise: if edge is clear and executable now, prefer ENTER; otherwise OBSERVE/SKIP with concrete reason. '
            'Avoid repetitive generic OBSERVE. If data quality is sufficient and no hard blocker exists, choose ENTER_LIMIT or ENTER_MARKET. '
            'Do not output bot_note; use market_read + entry_plan + stop_logic + recheck_reason instead. '
            '15m stop must be structural invalidation from prior closed swings (roughly 3-12 bars), not random wick. '
            'Use anti-sweep buffer (about 0.15-0.45 ATR15). Typical entry-to-stop risk ~0.8-2.2 ATR15, avoid >2.5 ATR15 unless better limit entry. '
            'OBSERVE must set entry_price=0, stop_loss=0, take_profit=0 and provide exactly one primary watch path. '
            'ENTER must be executable now with explicit entry_price and stop_loss. '
            'Mapping: enter_now=>enter/should_trade=true; wait_pullback|wait_breakout=>observe/should_trade=false; avoid_near_term=>skip/should_trade=false. '
            'bot_instruction must align with action: ENTER_MARKET or ENTER_LIMIT or OBSERVE or SKIP. '
            'action must be lower-case (enter/observe/skip), and order_type must be lower-case (market/limit). '
            + lane_instruction + ' '
            + prebreakout_rules + ' '
            + rr_discipline_rules + ' '
            'JSON shape reminder: ' + schema_hint
        )
        user_text = (
            f'Generate one {trade_style} tactical execution-analysis object.\n'
            f'Reference constraints (soft guidance): {constraint_brief}.\n'
            + soft_guidance_note + '\n'
            'Use 1D/4H bias + 1H trend quality + 15m execution + 5m confirmation + 1m micro timing.\n'
            'Ignore machine-generated scores/labels/plans; derive everything from raw payload.\n'
            'Do not long into nearby resistance, do not short into nearby support, and avoid late chase candles.\n'
            'Stop must be structural invalidation (prior closed 15m swing + anti-sweep buffer), not random or too-tight wick.\n'
            'If setup still needs waiting, return OBSERVE with one precise trigger path (no OR).\n'
            'If edge is clear now with executable invalidation and liquidity support, enter aggressively but not randomly.\n'
            'If no hard blocker exists (wide spread / wall pressure / unclear invalidation), do not stay in generic OBSERVE; prefer executable ENTER_LIMIT/ENTER_MARKET.\n'
            'For short_gainers or strong trend symbols, if structure + flow confirm and stop is executable, prefer ENTER now (market or executable limit), not repeated OBSERVE.\n'
            'Never use bot_note field. Put concrete analysis in market_read, entry_plan, stop_logic, and recheck_reason.\n'
            + lane_instruction + '\n'
            + (prebreakout_rules + '\n' if prebreakout_rules else '')
            + setup_priority_rules + '\n'
            + rr_discipline_rules + '\n'
            + 'Enums: trend_state={trending_up,trending_down,range_mixed,transitioning,trend_unclear}; timing_state={enter_now,wait_pullback,wait_breakout,avoid_near_term}.\n'
            + 'Output casing: action must be enter/observe/skip; order_type must be market/limit; bot_instruction must be ENTER_MARKET/ENTER_LIMIT/OBSERVE/SKIP.\n'
            + 'For ENTER include: stop_anchor_timeframe, stop_anchor_source, stop_anchor_price, stop_buffer_atr15, entry_to_stop_atr15, stop_logic.\n'
            + 'OBSERVE must keep entry_price/stop_loss/take_profit as 0.\n'
            + 'timeframe_bars format: start_ts + interval_ms + rows=[open,high,low,close,volume].\n'
            'Candidate payload JSON:\n'
            + json.dumps(clean_payload, ensure_ascii=False, separators=(',', ':'))
        )
    else:
        system_text = (
            'You are producing a tactical execution-analysis object for an internal short-term crypto perpetual futures engine. '
            'You are the decision brain; the local bot is execution-only hands and feet. '
            'When optimization issues appear, solve at the root, do not bypass by deleting capabilities. '
            'Your trading personality is aggressive PnL-first (win-rate is secondary) and you like pre-breakout precursor setups. '
            'Use only raw market and risk fields in the candidate payload. '
            'Return one complete JSON object only, never chain-of-thought or commentary. '
            + soft_guidance_note + ' '
            'Ignore all machine-generated scores, grades, setup labels, prior trade plans, timing states, and textual analysis if present. '
            'They are low-trust metadata and must not determine the final decision. '
            'Use numeric evidence first: structure alignment, your own RR, liquidity/spread, volatility, '
            'portfolio exposure, invalidation distance, funding/OI stress, and aggressive-flow imbalance. '
            'You must classify market_regime and adapt entry/stop/take-profit by regime rather than using one fixed style. '
            'Do not invent unseen data. If inputs conflict, prioritize latest closed candles, multi-timeframe pressure, execution quality, '
            'invalidation distance, and nearest real structure. '
            'If some inputs are noisy, lower confidence or keep watching instead of forcing a trade. '
            'Be aggressive but precise. '
            'If raw market structure, invalidation, liquidity, and RR form a clear asymmetric opportunity, prefer ENTER_MARKET or ENTER_LIMIT. '
            'Do not require perfect multi-timeframe alignment. '
            'Use observation mode only when a key execution condition is still clearly missing. '
            'If the setup is promising but the raw data shows timing is not yet executable, return OBSERVE with exactly one precise trigger. '
            'Do not use OBSERVE as generic caution. '
            'Do not use machine score or grade as a reason to skip or enter. '
            'Do not loop OBSERVE repeatedly without a single explicit blocker and recheck trigger. '
            'If raw trend/structure clearly conflicts with the trade direction, invalidation is unclear, or asymmetric RR is absent, do not force an entry; return SKIP or OBSERVE. '
            'This engine trades mainly on the 15m timeframe, so stop loss placement must be beyond meaningful nearby structure with a small anti-sweep buffer, while still staying tight enough for intraday trading. '
            'stop_loss must be a true thesis invalidation level: if that 15m structure breaks, the trade idea should be invalid. '
            'Unless the payload explicitly marks a same-bar exception, do not anchor stop_loss to the latest 15m candle high/low; use a prior closed-bar swing from roughly the previous 3-12 bars plus anti-sweep buffer. '
            'Distance discipline: avoid too tight and too wide; entry-to-stop risk should usually be around 0.8-2.2 ATR(15m). If required stop exceeds about 2.5 ATR(15m), prefer better limit entry or OBSERVE. '
            'Prioritize 1H/4H/1D range edges for trade desirability, but execute and invalidate on 15m structure. '
            'Local bot executes your take_profit directly and will not rewrite side/entry/stop/take-profit. '
            'For ENTER decisions, include stop_anchor_timeframe, stop_anchor_source, stop_anchor_price, stop_buffer_atr15, entry_to_stop_atr15, and stop_logic. '
            + aggressive_precision_note + ' '
            + setup_priority_rules + ' '
            + rr_discipline_rules + ' '
            + lane_instruction + ' '
            + prebreakout_rules + ' '
            'All human-readable explanation fields must be written in Traditional Chinese. Keep only enums, symbols, and timeframe tokens in English. '
            'JSON shape reminder: ' + schema_hint
        )
        user_text = (
            f'Generate one {trade_style} tactical execution-analysis object.\n'
            f'Reference constraints (soft guidance): {constraint_brief}; order_type should usually be market or limit.\n'
            + soft_guidance_note + '\n'
            'Compute entry_price, stop_loss, take_profit, and RR only from raw structure/current price/liquidity/invalidation in the payload.\n'
            'Output market_regime and regime_note explicitly, and make entry/stop/take-profit regime-specific: trend_continuation/trend_pullback can be aggressive, range_reversion must anchor to range edges, consolidation_squeeze must require clear trigger plus invalidation, transition_chop should reduce forced entries.\n'
            'For 15m short-term execution, stop_loss must sit at a reasonable break-of-structure invalidation level, not a fragile nearby wick.\n'
            'Select stop anchor from prior closed 15m candles (normally look back about 3-12 bars) and place stop beyond that defended swing high/low plus anti-sweep buffer.\n'
            'The anti-sweep buffer must reference 15m ATR_price, and is usually about 0.15-0.45 ATR(15m) beyond the swing anchor.\n'
            'Distance discipline: entry->stop risk should usually be around 0.8-2.2 ATR(15m); if stop would need to exceed ~2.5 ATR(15m), do not force entry now (prefer better ENTER_LIMIT or OBSERVE).\n'
            'Default rule: do not use the latest 15m candle high/low as the stop anchor. Only if payload clearly shows explosive same-bar exception may latest-bar anchor be allowed.\n'
            'Never place stop_loss in the middle of a tight 15m consolidation box; place it outside the structure edge.\n'
            'The execution bot will use your take_profit directly and will not rewrite side/entry/stop/take-profit.\n'
            'For ENTER decisions, include stop_anchor_timeframe, stop_anchor_source, stop_anchor_price, stop_buffer_atr15, entry_to_stop_atr15, and stop_logic.\n'
            + aggressive_precision_note + '\n'
            + setup_priority_rules + '\n'
            + rr_discipline_rules + '\n'
            'If the symbol is BTC, ETH, XAU, XAG, or SOL, keep the order sizing aggressive for a lower-volatility major: use the symbol-max leverage already provided and do not reduce the payload notional below the configured major-coin size.\n'
            'Classify trend_state and timing_state explicitly. Preferred action mapping: enter_now=>enter; wait_pullback/wait_breakout=>observe; avoid_near_term=>skip.\n'
            'Please set bot_instruction as one of ENTER_MARKET, ENTER_LIMIT, OBSERVE, SKIP. Suggested mapping: enter+market=>ENTER_MARKET, enter+limit=>ENTER_LIMIT, observe=>OBSERVE, skip=>SKIP.\n'
            'Execution bot only collects market data and executes your returned instruction; do not rely on bot-side analysis and do not output ambiguous wording.\n'
            'The local bot can only execute explicit machine-operable commands, so every ENTER must include exact entry_price and stop_loss.\n'
            'Be aggressive but precise: if edge is clear and executable now, prefer ENTER; if raw trend/structure conflicts, invalidation is unclear, or RR edge is absent, do not enter.\n'
            'When data quality is sufficient and there is no hard blocker, avoid generic OBSERVE; pick an executable ENTER path now.\n'
            'Do not output bot_note; use market_read, entry_plan, stop_logic, and recheck_reason for explanation.\n'
            + lane_instruction + '\n'
            + (prebreakout_rules + '\n' if prebreakout_rules else '')
            + 'If 15m execution edge is clear with explicit invalidation and asymmetric RR, do not downgrade to OBSERVE only due to minor higher-timeframe mismatch.\n'
            'If the plan still needs waiting for a future candle close, retest, reclaim, breakdown, pullback, or any X-minute observation, return observe now instead of hiding that wait inside entry_plan.\n'
            'Set trade_side and breakout_assessment explicitly. If scale-in is not recommended, return false and 0/empty fields.\n'
            'All numeric fields must be raw JSON numbers, and the reply must be one complete JSON object only.\n'
            'Analyze 1D/4H for larger bias, 1H for trend quality, 15m for the main entry frame, 5m for confirmation, and 1m only for micro-timing.\n'
            'timeframe_bars format: each timeframe uses start_ts + interval_ms, and rows are [open,high,low,close,volume] from oldest to newest.\n'
            'market_read and entry_plan must be explicit. reason_to_skip must never be generic and must cite concrete blockers from the payload.\n'
            'Choose one best executable path now: market, precise limit pullback/retest, or no trade yet with one precise recheck trigger.\n'
            'ENTER_LIMIT only when a resting order is executable now, entry_price is better than current price, not directly into nearby wall/resistance/support, stop_loss is a clear structural invalidation, and TP has room.\n'
            'Use OBSERVE when still waiting for close/breakout/retest/reclaim/breakdown, when price is too close to wall/support/resistance, or no clean pullback/FVG/VWAP/EMA entry zone.\n'
            'If OBSERVE: entry_price=0, stop_loss=0, take_profit=0; do not place executable prices in entry_price. If needed, use candidate_entry_price only.\n'
            'If OBSERVE, choose exactly one primary watch path only (no OR) and fill watch_trigger_candle, watch_retest_rule, watch_volume_ratio_min, watch_micro_vwap_rule, watch_micro_ema20_rule.\n'
            'Anti-chase suggestion: long near resistance (<0.25 ATR15m before valid break) or short near support (<0.25 ATR15m before valid break) should avoid ENTER_MARKET.\n'
            'If narrative says no chase / wait pullback / avoid extension, prefer ENTER_LIMIT or OBSERVE over ENTER_MARKET. pre_breakout_anticipation is usually ENTER_LIMIT or OBSERVE.\n'
            'Major-vs-alt nuance: ETH-like majors may use trend_pullback_continuation ENTER_LIMIT after sharp-drop reclaim near 15m/1h EMA200 or clear support, but describe it as support-rebound pullback long, not full timeframe trend flip.\n'
            'For LDO-like alts under ask wall/range high/near resistance with ask-heavy depth imbalance, prefer OBSERVE; do not ENTER_LIMIT under the wall unless pullback entry and invalidation are very clear.\n'
            'VWAP/EMA truth check: only say above VWAP/EMA when close > that level on the same timeframe. If only 1m is above VWAP while 5m/15m/1h are below, state it explicitly and do not generalize 1m strength upward.\n'
            'ATR unit rule: in multi_timeframe, a is atr_pct (not price points). For ATR price use ATR_price = close * atr_pct / 100.\n'
            'When describing stop buffer, separate entry->stop ATR risk and swing->stop anti-sweep ATR buffer; never treat atr_pct as direct price ATR.\n'
            'If order_type is limit, return limit_cancel_price, limit_cancel_timeframe, limit_cancel_condition, and limit_cancel_note; for market, set those fields to 0/empty.\n'
            'If should_trade is false, preserve the idea with one narrow watch plan and machine-readable tracking fields the bot can track exactly.\n'
            'Make watch fields explicit and machine-trackable with concrete price, timeframe, candle, retest, and volume conditions; never use vague watch text.\n'
            'Keep compact JSON only; no long explanation and no extra recheck loop.\n'
            'Final consistency check:\n'
            '- bot_instruction and timing_state must align; ENTER must be executable now.\n'
            '- OBSERVE must keep entry_price/stop_loss at 0 and use one machine trigger path.\n'
            '- stop_loss must be based on prior closed 15m swing invalidation (few bars back), not latest forming bar extreme or noise middle.\n'
            '- No data-opposite market_read, no false above VWAP/EMA claim, no atr_pct-as-price mistake.\n'
            '- Do not long directly under resistance or short directly above support.\n'
            '- Do not output no-chase text with ENTER_MARKET.\n'
            'Use Traditional Chinese for all human-readable analysis fields.\n'
            'Compact payload legend: multi_timeframe uses c=close a=atr_pct r=rsi x=adx e20/e50/e200 m20=ma20 v=vwap t=trend mh=macd_hist bbp=bb_position_pct vr=vol_ratio hi/lo=recent structure ph/pl=prior swing ch/cl=current bar high-low xr=explosive-move flag; multi_timeframe_pressure uses sb=structure_bias ts=trend_stack rb=recent_break pp/sp=pressure/support pa/sa=distance_atr c20/c50=close_vs_ema hh/hl/lh/ll=structure counts.\n'
            'Candidate payload JSON:\n'
            + json.dumps(clean_payload, ensure_ascii=False, separators=(',', ':'))
        )
    return [
        {'role': 'system', 'content': [{'type': 'input_text', 'text': system_text}]},
        {'role': 'user', 'content': [{'type': 'input_text', 'text': user_text}]},
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
        320 if structured else 160,
    )
    body = {
        'model': str(model or config.get('model') or 'gpt-5.4-mini'),
        'input': _build_messages(candidate, compact=compact_prompt, logger=logger),
        'max_output_tokens': effective_max_output_tokens,
    }
    text_format: Dict[str, Any]
    if structured:
        text_format = {
            'type': 'json_schema',
            'name': 'trade_decision',
            'schema': _json_schema(),
            'strict': False,
        }
    else:
        text_format = {
            'type': 'json_object',
        }
    body['text'] = {
        'verbosity': 'medium',
        'format': text_format,
    }
    effort = str(reasoning_effort or config.get('reasoning_effort') or '').strip().lower()
    if effort and effort not in {'none', 'off', 'disabled'}:
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
    stretched = _candidate_entry_stretched(candidate, side_norm, entry_price if entry_price > 0 else current_price)
    if not (says_no_chase or stretched):
        return
    existing_limit_ok = (
        str(decision.get('order_type') or '').lower() == 'limit'
        and current_price > 0
        and (
            (side_norm == 'long' and entry_price < current_price * 0.999)
            or (side_norm == 'short' and entry_price > current_price * 1.001)
        )
    )
    if existing_limit_ok:
        return
    decision['order_type'] = 'limit'
    limit_entry = _infer_limit_entry_from_fvg(candidate, side_norm, entry_price if entry_price > 0 else current_price)
    if side_norm == 'long' and limit_entry >= max(entry_price, current_price) * 0.999:
        limit_entry = 0.0
    if side_norm == 'short' and limit_entry <= max(entry_price, current_price) * 1.001:
        limit_entry = 0.0
    if limit_entry > 0:
        decision['entry_price'] = limit_entry
        plan_note = '改為限價承接 FVG/回踩，避免追價。'
        existing_plan = str(decision.get('entry_plan') or '').strip()
        decision['entry_plan'] = plan_note if not existing_plan else '{} {}'.format(plan_note, existing_plan)
        if not str(decision.get('limit_cancel_note') or '').strip():
            decision['limit_cancel_note'] = '若未成交且結構失效，取消此限價單。'
        return
    decision['action'] = 'observe'
    decision['should_trade'] = False
    decision['timing_state'] = 'wait_pullback'
    decision['watch_trigger_type'] = 'pullback_to_entry'
    decision['recheck_reason'] = '目前屬延伸段，先等可執行回踩/缺口承接再重審。'
    if not str(decision.get('watch_structure_condition') or '').strip():
        decision['watch_structure_condition'] = '等待 15m 回踩到可執行區，且收盤不破結構失效位。'


def _normalize_decision(raw: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    raw = dict(raw or {})
    if isinstance(raw.get('decision'), dict):
        raw = dict(raw.get('decision') or {})
    elif isinstance(raw.get('trade_decision'), dict):
        raw = dict(raw.get('trade_decision') or {})
    allowed_actions = {'enter', 'observe', 'skip'}
    allowed_regimes = {'trend_continuation', 'trend_pullback', 'range_reversion', 'consolidation_squeeze', 'transition_chop'}
    allowed_trend_states = {'trending_up', 'trending_down', 'range_mixed', 'transitioning', 'trend_unclear'}
    allowed_timing_states = {'enter_now', 'wait_pullback', 'wait_breakout', 'avoid_near_term'}
    allowed_sides = {'long', 'short', 'neutral'}
    allowed_order_types = {'market', 'limit'}
    allowed_instructions = {'ENTER_MARKET', 'ENTER_LIMIT', 'OBSERVE', 'SKIP'}
    allowed_watch = {'none', 'pullback_to_entry', 'breakout_reclaim', 'breakdown_confirm', 'volume_confirmation', 'manual_review'}
    allowed_trigger_candles = {'none', 'close_above', 'close_below'}
    allowed_retest_rules = {'none', 'hold_above', 'hold_below', 'fail_above', 'fail_below'}
    allowed_micro_rules = {'none', 'above', 'below'}

    def _num(key: str, default: float = 0.0) -> float:
        return _coerce_float(raw.get(key, default), default)

    def _txt(key: str, limit: int = 220) -> str:
        return str(raw.get(key) or '').strip()[:limit]

    def _txt_any(keys: list[str], limit: int = 220) -> str:
        for key in keys:
            text = str(raw.get(key) or '').strip()
            if text:
                return text[:limit]
        return ''

    instruction_action_map = {
        'ENTER_MARKET': 'enter',
        'ENTER_LIMIT': 'enter',
        'OBSERVE': 'observe',
        'SKIP': 'skip',
    }
    action_instruction_map = {
        ('enter', 'limit'): 'ENTER_LIMIT',
        ('enter', 'market'): 'ENTER_MARKET',
        ('observe', 'market'): 'OBSERVE',
        ('observe', 'limit'): 'OBSERVE',
        ('skip', 'market'): 'SKIP',
        ('skip', 'limit'): 'SKIP',
    }

    side = str(raw.get('trade_side') or raw.get('direction') or raw.get('side') or raw.get('position_side') or '').strip().lower()
    if side == 'buy':
        side = 'long'
    elif side == 'sell':
        side = 'short'
    elif side in {'none', 'na', 'n/a', 'flat'}:
        side = 'neutral'

    bot_instruction = str(raw.get('bot_instruction') or '').strip().upper()
    action = str(raw.get('action') or '').strip().lower()
    order_type = str(raw.get('order_type') or '').strip().lower()
    should_trade = bool(raw.get('should_trade', False))

    if not action and bot_instruction in instruction_action_map:
        action = instruction_action_map[bot_instruction]
    if action not in allowed_actions:
        if should_trade:
            action = 'enter'
        elif str(raw.get('watch_trigger_type') or 'none').strip().lower() != 'none':
            action = 'observe'
        else:
            action = 'skip'

    if not order_type:
        if bot_instruction == 'ENTER_LIMIT':
            order_type = 'limit'
        elif bot_instruction == 'ENTER_MARKET':
            order_type = 'market'
        elif action == 'enter':
            order_type = 'market'
        else:
            order_type = 'market'

    if bot_instruction not in allowed_instructions:
        bot_instruction = action_instruction_map.get((action, order_type), 'SKIP')

    timing_state = str(raw.get('timing_state') or '').strip().lower()
    if timing_state not in allowed_timing_states:
        if action == 'enter':
            timing_state = 'enter_now'
        elif action == 'observe':
            timing_state = 'wait_pullback'
        else:
            timing_state = 'avoid_near_term'

    if side not in allowed_sides:
        candidate_side = str(candidate.get('side') or '').strip().lower()
        if candidate_side in {'buy', 'long'}:
            side = 'long'
        elif candidate_side in {'sell', 'short'}:
            side = 'short'
        elif action != 'enter':
            side = 'neutral'
        else:
            side = ''

    decision = {
        'should_trade': should_trade,
        'action': action,
        'market_regime': str(raw.get('market_regime') or 'transition_chop').strip().lower(),
        'regime_note': _txt('regime_note', 180),
        'trend_state': str(raw.get('trend_state') or 'trend_unclear').strip().lower(),
        'timing_state': timing_state,
        'trade_side': side,
        'breakout_assessment': _txt_any(['breakout_assessment', 'setup_assessment', 'signal_assessment', 'bot_note'], 180),
        'rr_ratio': max(_num('rr_ratio', _num('rr', _num('risk_reward', 0.0))), 0.0),
        'scale_in_recommended': bool(raw.get('scale_in_recommended', False)),
        'scale_in_price': max(_num('scale_in_price', 0.0), 0.0),
        'scale_in_qty_pct': _clamp(_num('scale_in_qty_pct', 0.0), 0.0, 1.0),
        'scale_in_condition': _txt('scale_in_condition', 220),
        'scale_in_note': _txt('scale_in_note', 220),
        'order_type': order_type,
        'bot_instruction': bot_instruction,
        'entry_price': _num('entry_price', _num('entry', 0.0)),
        'candidate_entry_price': _num('candidate_entry_price', _num('candidate_entry', 0.0)),
        'stop_loss': _num('stop_loss', _num('sl', 0.0)),
        'take_profit': _num('take_profit', _num('tp', 0.0)),
        'market_read': _txt_any(['market_read', 'bot_note', 'analysis', 'note'], 320),
        'entry_plan': _txt_any(['entry_plan', 'execution_plan', 'plan', 'bot_note'], 320),
        'entry_reason': _txt('entry_reason', 220),
        'stop_loss_reason': _txt('stop_loss_reason', 220),
        'take_profit_plan': _txt('take_profit_plan', 280),
        'if_missed_plan': _txt('if_missed_plan', 220),
        'reference_summary': _txt('reference_summary', 220),
        'chase_if_triggered': bool(raw.get('chase_if_triggered', False)),
        'chase_trigger_price': max(_num('chase_trigger_price', 0.0), 0.0),
        'chase_limit_price': max(_num('chase_limit_price', 0.0), 0.0),
        'trail_trigger_atr_hint': _clamp(_num('trail_trigger_atr_hint', 0.0), 0.0, 6.0),
        'trail_pct_hint': _clamp(_num('trail_pct_hint', 0.0), 0.0, 0.3),
        'breakeven_atr_hint': _clamp(_num('breakeven_atr_hint', 0.0), 0.0, 4.0),
        'dynamic_take_profit_hint': max(_num('dynamic_take_profit_hint', 0.0), 0.0),
        'leverage': int(max(_num('leverage', 0.0), 0.0)),
        'margin_pct': max(_num('margin_pct', 0.0), 0.0),
        'confidence': _clamp(_num('confidence', 0.0), 0.0, 100.0),
        'thesis': _txt('thesis', 220),
        'reason_to_skip': _txt_any(['reason_to_skip', 'skip_reason', 'bot_note'], 220),
        'risk_notes': [str(x).strip()[:180] for x in list(raw.get('risk_notes') or []) if str(x).strip()][:8],
        'aggressive_note': _txt('aggressive_note', 220),
        'watch_trigger_type': str(raw.get('watch_trigger_type') or 'none').strip(),
        'watch_trigger_price': max(_num('watch_trigger_price', 0.0), 0.0),
        'watch_invalidation_price': max(_num('watch_invalidation_price', 0.0), 0.0),
        'watch_note': _txt_any(['watch_note', 'bot_note'], 220),
        'recheck_reason': _txt_any(['recheck_reason', 'watch_reason', 'bot_note'], 220),
        'watch_timeframe': _txt('watch_timeframe', 80),
        'watch_price_zone_low': max(_num('watch_price_zone_low', 0.0), 0.0),
        'watch_price_zone_high': max(_num('watch_price_zone_high', 0.0), 0.0),
        'watch_structure_condition': _txt('watch_structure_condition', 220),
        'watch_volume_condition': _txt('watch_volume_condition', 220),
        'watch_checklist': [str(x).strip()[:180] for x in list(raw.get('watch_checklist') or []) if str(x).strip()][:8],
        'watch_confirmations': [str(x).strip()[:180] for x in list(raw.get('watch_confirmations') or []) if str(x).strip()][:8],
        'watch_invalidations': [str(x).strip()[:180] for x in list(raw.get('watch_invalidations') or []) if str(x).strip()][:8],
        'watch_trigger_candle': str(raw.get('watch_trigger_candle') or 'none').strip(),
        'watch_retest_rule': str(raw.get('watch_retest_rule') or 'none').strip(),
        'watch_volume_ratio_min': max(_num('watch_volume_ratio_min', 0.0), 0.0),
        'watch_micro_vwap_rule': str(raw.get('watch_micro_vwap_rule') or 'none').strip(),
        'watch_micro_ema20_rule': str(raw.get('watch_micro_ema20_rule') or 'none').strip(),
        'watch_recheck_priority': _clamp(_num('watch_recheck_priority', 0.0), 0.0, 100.0),
        'limit_cancel_price': max(_num('limit_cancel_price', 0.0), 0.0),
        'limit_cancel_timeframe': _txt('limit_cancel_timeframe', 80),
        'limit_cancel_condition': _txt('limit_cancel_condition', 220),
        'limit_cancel_note': _txt('limit_cancel_note', 220),
        'stop_anchor_timeframe': _txt('stop_anchor_timeframe', 40),
        'stop_anchor_source': _txt('stop_anchor_source', 120),
        'stop_anchor_price': max(_num('stop_anchor_price', 0.0), 0.0),
        'stop_buffer_atr15': max(_num('stop_buffer_atr15', 0.0), 0.0),
        'entry_to_stop_atr15': max(_num('entry_to_stop_atr15', 0.0), 0.0),
        'stop_logic': _txt_any(['stop_logic', 'stop_loss_reason', 'bot_note'], 320),
    }

    errors: list[str] = []
    if decision['action'] not in allowed_actions:
        errors.append('enum_invalid:action')
    if decision['market_regime'] not in allowed_regimes:
        errors.append('enum_invalid:market_regime')
    if decision['trend_state'] not in allowed_trend_states:
        errors.append('enum_invalid:trend_state')
    if decision['timing_state'] not in allowed_timing_states:
        errors.append('enum_invalid:timing_state')
    if decision['trade_side'] not in allowed_sides:
        errors.append('enum_invalid:trade_side')
    if decision['order_type'] not in allowed_order_types:
        errors.append('enum_invalid:order_type')
    if decision['bot_instruction'] not in allowed_instructions:
        errors.append('enum_invalid:bot_instruction')
    if decision['watch_trigger_type'] not in allowed_watch:
        errors.append('enum_invalid:watch_trigger_type')
    if decision['watch_trigger_candle'] not in allowed_trigger_candles:
        errors.append('enum_invalid:watch_trigger_candle')
    if decision['watch_retest_rule'] not in allowed_retest_rules:
        errors.append('enum_invalid:watch_retest_rule')
    if decision['watch_micro_vwap_rule'] not in allowed_micro_rules:
        errors.append('enum_invalid:watch_micro_vwap_rule')
    if decision['watch_micro_ema20_rule'] not in allowed_micro_rules:
        errors.append('enum_invalid:watch_micro_ema20_rule')

    expected_action = instruction_action_map.get(decision['bot_instruction'])
    if expected_action and decision['action'] != expected_action:
        errors.append('mapping_invalid:action_vs_bot_instruction')

    if decision['timing_state'] == 'enter_now':
        if decision['action'] != 'enter' or not bool(decision['should_trade']):
            errors.append('mapping_invalid:timing_enter_now')
    if decision['timing_state'] in {'wait_pullback', 'wait_breakout'}:
        if decision['action'] != 'observe' or bool(decision['should_trade']):
            errors.append('mapping_invalid:timing_wait_state')
    if decision['timing_state'] == 'avoid_near_term':
        if decision['action'] != 'skip' or bool(decision['should_trade']):
            errors.append('mapping_invalid:timing_avoid_near_term')

    if decision['action'] == 'enter':
        if not bool(decision['should_trade']):
            errors.append('mapping_invalid:enter_should_trade_false')
        if decision['trade_side'] not in {'long', 'short'}:
            errors.append('enter_missing_trade_side')
        if decision['entry_price'] <= 0 or decision['stop_loss'] <= 0:
            errors.append('enter_missing_entry_or_stop')
        if not str(decision.get('stop_logic') or '').strip():
            errors.append('enter_missing_stop_logic')
    if decision['action'] in {'observe', 'skip'} and bool(decision['should_trade']):
        errors.append('mapping_invalid:non_enter_should_trade_true')
    if decision['bot_instruction'] == 'OBSERVE':
        if any([
            decision['entry_price'] > 0,
            decision['stop_loss'] > 0,
            decision['take_profit'] > 0,
        ]):
            errors.append('observe_prices_must_be_zero')
    if decision['action'] == 'observe':
        if str(decision.get('watch_trigger_type') or 'none') == 'none':
            errors.append('observe_missing_watch_trigger_type')
        if not str(decision.get('recheck_reason') or '').strip():
            errors.append('observe_missing_recheck_reason')
        if (
            float(decision.get('watch_trigger_price', 0) or 0) <= 0
            and not str(decision.get('watch_structure_condition') or '').strip()
            and not str(decision.get('watch_volume_condition') or '').strip()
        ):
            errors.append('observe_missing_machine_trigger_path')

    decision['validation_errors'] = errors[:16]
    decision['valid'] = len(decision['validation_errors']) == 0
    if not decision['valid']:
        decision['should_trade'] = False
        decision['action'] = 'skip'
        decision['bot_instruction'] = 'SKIP'
        decision['order_type'] = 'market'
        decision['entry_price'] = 0.0
        decision['stop_loss'] = 0.0
        decision['take_profit'] = 0.0
        if decision['trade_side'] not in {'long', 'short'}:
            decision['trade_side'] = 'neutral'
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
    cooldown_sec = max(int(config.get('cooldown_minutes', 180) or 180), 1) * 60
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
    top_signature = _top_candidates_signature(candidate)

    if (
        not force_recheck
        and not empty_no_decision
        and cached_decision
        and last_hash == payload_hash
        and last_sent_ts > 0
        and (now_ts - last_sent_ts) < same_payload_reuse_sec
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
    estimated_output_tokens = max(int(config.get('max_output_tokens', 560) or 560), 320)
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
        primary_effort = str(config.get('reasoning_effort') or 'medium').strip()
        retry_effort = str(config.get('retry_reasoning_effort') or primary_effort or 'medium').strip()
        per_call_budget_twd = max(float(config.get('per_call_budget_twd', 0.20) or 0.20), 0.05)
        max_tokens = max(int(config.get('max_output_tokens', 560) or 560), 320)
        attempts = [
            {
                'model': primary_model,
                'structured': False,
                'effort': primary_effort,
                'max_tokens': max_tokens,
                'compact_prompt': True,
            },
        ]
        if bool(config.get('empty_retry_enabled', True)):
            attempts.append(
                {
                    'model': primary_model,
                    'structured': False,
                    'effort': retry_effort,
                    'max_tokens': max_tokens,
                    'compact_prompt': True,
                }
            )
            if allow_upgrade and upgrade_model and upgrade_model != primary_model:
                attempts.append(
                    {
                        'model': upgrade_model,
                        'structured': True,
                        'effort': retry_effort,
                        'max_tokens': max_tokens,
                        'compact_prompt': True,
                    }
                )
            if fallback_model and fallback_model not in {primary_model, upgrade_model}:
                attempts.append(
                    {
                        'model': fallback_model,
                        'structured': False,
                        'effort': retry_effort,
                        'max_tokens': max_tokens,
                        'compact_prompt': True,
                    }
                )

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
        for _, attempt in enumerate(attempts):
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
                if remaining_budget <= 1.5:
                    empty_details.append(
                        'decision_timeout_budget_low remaining={:.2f}s'.format(remaining_budget)
                    )
                    if logger:
                        logger('OpenAI decision budget too low for another request: {} remaining={:.2f}s'.format(symbol, remaining_budget))
                    break
                attempt_timeout = min(timeout_sec, remaining_budget)
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
                        action=str(reused.get('action') or ('enter' if reused.get('should_trade') else 'observe')),
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
            symbol_state.update({
                'last_model': selected_model,
                'last_decision': {},
                'last_status': 'empty_response_no_action',
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
                    status='empty_response_no_action',
                    action='skip',
                    detail='OpenAI 空回覆或 JSON 不完整；本輪不採用本地推測，也不建立自動觀察單。',
                    model=selected_model,
                ),
            )
            save_trade_state(state_path, state)
            return state, {
                'status': 'empty_response_no_action',
                'decision': None,
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
        'cooldown_minutes': int(config.get('cooldown_minutes', 180) or 180),
        'global_min_interval_minutes': int(config.get('global_min_interval_minutes', 0) or 0),
        'advice_ttl_minutes': int(config.get('advice_ttl_minutes', 240) or 240),
        'pending_advice_count': len(pending),
        'pending_advice': pending_rows[:20],
        'min_score_abs': float(config.get('min_score_abs', 0.0) or 0.0),
        'last_error': str(state.get('last_error') or ''),
        'updated_at': str(state.get('updated_at') or ''),
        'recent_decisions': recent_rows,
    }
