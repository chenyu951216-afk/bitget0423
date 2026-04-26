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
        'usd_to_twd': max(_env_float(env_getter, 'OPENAI_TRADE_USD_TO_TWD', 32.0), 1.0),
        'input_price_per_1m_usd': max(_env_float(env_getter, 'OPENAI_TRADE_PRICE_INPUT_PER_1M_USD', 0.75), 0.0),
        'output_price_per_1m_usd': max(_env_float(env_getter, 'OPENAI_TRADE_PRICE_OUTPUT_PER_1M_USD', 4.50), 0.0),
        'cached_input_price_per_1m_usd': max(_env_float(env_getter, 'OPENAI_TRADE_PRICE_CACHED_INPUT_PER_1M_USD', 0.075), 0.0),
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
        'max_output_tokens': max(_env_int(env_getter, 'OPENAI_TRADE_MAX_OUTPUT_TOKENS', 560), 420),
        'request_timeout_sec': max(_env_float(env_getter, 'OPENAI_TRADE_TIMEOUT_SEC', 45.0), 5.0),
        'temperature': 0.2,
        'base_url': str(env_getter('OPENAI_RESPONSES_URL', 'https://api.openai.com/v1/responses') or 'https://api.openai.com/v1/responses').strip(),
        'reasoning_effort': str(env_getter('OPENAI_TRADE_REASONING_EFFORT', 'low') or 'low').strip(),
        'retry_reasoning_effort': str(env_getter('OPENAI_TRADE_RETRY_REASONING_EFFORT', 'low') or 'low').strip(),
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


def _string_schema(max_length: int) -> Dict[str, Any]:
    return {'type': 'string', 'maxLength': max(int(max_length or 1), 1)}


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
        ['current_price', 'change_24h_pct', 'quote_volume_24h', 'funding_rate', 'open_interest_value_usdt', 'mark_price', 'index_price'],
    )
    basic_symbol = str(((market_context.get('basic_market_data') or {}).get('symbol') or '')).strip()
    if basic_symbol:
        basic_market_data['symbol'] = basic_symbol[:32]
    levels = _clean_levels(dict(market_context.get('levels') or {}))
    multi_timeframe = _clean_multi_timeframe(dict(market_context.get('multi_timeframe') or {}))
    timeframe_bars = _compact_timeframe_bars(
        dict(market_context.get('timeframe_bars') or {}),
        symbol=symbol,
        logger=logger,
    )
    liquidity_context = _clean_numeric_mapping(
        dict(market_context.get('liquidity_context') or {}),
        [
            'spread_pct',
            'bid_depth_10',
            'ask_depth_10',
            'depth_imbalance_10',
            'largest_bid_wall_price',
            'largest_ask_wall_price',
            'buy_sell_notional_ratio',
            'cvd_notional',
            'volume_anomaly_5m',
            'volume_anomaly_15m',
        ],
    )
    derivatives_context = _clean_numeric_mapping(
        dict(market_context.get('derivatives_context') or {}),
        [
            'funding_rate',
            'open_interest_value_usdt',
            'open_interest_change_pct_5m',
            'basis_pct',
            'leverage_heat_score',
            'mark_price',
            'index_price',
        ],
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
        'derivatives_context': derivatives_context,
        'risk': risk,
        'portfolio': portfolio,
        'execution_policy': execution_policy,
        'constraints': constraints,
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
    breakdown = dict(signal.get('breakdown') or {})
    execution_quality = dict(signal.get('execution_quality') or {})
    market_context = dict(signal.get('openai_market_context') or {})
    style = dict((market_context.get('style') or {}))
    execution_policy = dict((market_context.get('execution_policy') or {}))
    signal_side = str(
        signal.get('side')
        or signal.get('direction')
        or dict(market_context.get('signal_context') or {}).get('side')
        or 'long'
    ).lower()
    if signal_side not in {'long', 'short'}:
        signal_side = 'long'
    compact_breakdown = _compact_mapping(
        breakdown,
        [
            'Setup', 'Regime', 'RegimeConf', 'RegimeConfidence', 'RegimeDir', 'RegimeBias',
            'MarketState', 'MarketStateConf', 'MarketTempo', 'TrendConfidence', '鏂瑰悜淇″績',
            'RR', 'EntryGate', '閫插牬鍝佽唱', 'ChaseRisk', '杩藉児棰ㄩ毆', 'VolRatio',
            'PreBreakoutScore', 'PreBreakoutDirection', 'PreBreakoutPhase',
            'AIScoreCoverage', 'AISampleCount', 'LearnEdge', 'SignalQuality',
            'VWAPDistanceATR', 'EMA20DistanceATR', 'SRDistanceATR',
        ],
        text_limit=140,
    )
    compact_market_context = {
        'style': _compact_mapping(style, ['holding_period', 'decision_priority'], text_limit=80),
        'signal_context': _compact_mapping(dict(market_context.get('signal_context') or {}), ['side', 'current_price', 'atr_15m', 'atr_4h'], text_limit=60),
        'latest_closed_candle': _compact_mapping(dict(market_context.get('latest_closed_candle') or {}), ['direction', 'shape'], text_limit=40),
        'momentum': _compact_mapping(dict(market_context.get('momentum') or {}), ['long_score', 'short_score', 'signals', 'volume_build', 'compression'], text_limit=80),
        'levels': _compact_mapping(dict(market_context.get('levels') or {}), ['nearest_support', 'nearest_resistance', 'support_levels', 'resistance_levels', 'recent_high', 'recent_low'], text_limit=80),
        'market_state': {
            'ticker': _compact_mapping(dict(((market_context.get('market_state') or {}).get('ticker') or {})), ['last', 'spread_pct', 'mark_price', 'index_price'], text_limit=120),
        },
        'basic_market_data': _compact_mapping(dict(market_context.get('basic_market_data') or {}), ['symbol', 'exchange', 'market_type', 'current_price', 'change_24h_pct', 'quote_volume_24h', 'funding_rate', 'open_interest_value_usdt'], text_limit=100),
        'liquidity_context': _compact_mapping(dict(market_context.get('liquidity_context') or {}), ['spread_pct', 'bid_depth_10', 'ask_depth_10', 'depth_imbalance_10', 'largest_bid_wall_price', 'largest_ask_wall_price', 'buy_sell_notional_ratio', 'cvd_notional', 'cvd_bias', 'volume_anomaly_5m', 'volume_anomaly_15m'], text_limit=120),
        'derivatives_context': _compact_mapping(dict(market_context.get('derivatives_context') or {}), ['funding_rate', 'open_interest_value_usdt', 'open_interest_change_pct_5m', 'basis_pct', 'leverage_heat', 'leverage_heat_score'], text_limit=120),
        'news_context': _compact_news_context(market_context.get('news_context') or {}),
        'multi_timeframe': {
            str(tf): _compact_tf_stats(dict(row or {}))
            for tf, row in list(dict(market_context.get('multi_timeframe') or {}).items())[:6]
        },
        'timeframe_bars': _compact_timeframe_bars(dict(market_context.get('timeframe_bars') or {})),
        'pre_breakout_radar': _compact_mapping(dict(market_context.get('pre_breakout_radar') or {}), ['ready', 'phase', 'direction', 'score'], text_limit=120),
        'execution_context': _compact_mapping(dict(market_context.get('execution_context') or {}), ['spread_pct', 'top_depth_ratio', 'api_error_streak', 'status'], text_limit=120),
        'multi_timeframe_pressure_summary': _compact_mapping(dict(market_context.get('multi_timeframe_pressure_summary') or {}), ['side', 'aligned_timeframes', 'opposing_timeframes', 'nearest_blocking_timeframe', 'nearest_blocking_price', 'nearest_blocking_distance_atr', 'nearest_backing_timeframe', 'nearest_backing_price', 'nearest_backing_distance_atr', 'stacked_blocking_within_1atr', 'stacked_blocking_within_2atr'], text_limit=80),
        'multi_timeframe_pressure': {
            str(tf): _compact_pressure_stats(dict(row or {}))
            for tf, row in list(dict(market_context.get('multi_timeframe_pressure') or {}).items())[:4]
        },
    }
    compact_market_context['signal_context']['market_bias'] = _short_text(((market_context.get('signal_context') or {}).get('side') or ''), 40)
    compact_market_context['signal_context']['scanner_side'] = signal_side
    compact_reference = _compact_mapping(
        dict(signal.get('external_reference') or signal.get('reference_context') or signal.get('scanner_reference') or {}),
        ['summary', 'bias', 'setup', 'risk', 'note', 'checklist', 'confirmations', 'invalidations', 'source'],
        text_limit=180,
    )
    compact_reference_trade_plan = _compact_mapping(
        dict(signal.get('reference_trade_plan') or market_context.get('reference_trade_plan') or {}),
        ['machine_entry_hint', 'machine_stop_loss_hint', 'machine_take_profit_hint', 'machine_rr_hint', 'machine_est_pnl_pct_hint', 'machine_stop_anchor_price', 'machine_stop_anchor_source', 'machine_stop_same_bar_exception', 'note'],
        text_limit=180,
    )
    compact_top_candidates = []
    for row in list(top_candidates or [])[:1]:
        compact_top_candidates.append(
            _compact_mapping(
                dict(row or {}),
                ['symbol', 'side', 'score', 'priority_score', 'entry_quality', 'rr_ratio', 'candidate_source'],
                text_limit=60,
            )
        )
    return {
        'symbol': str(signal.get('symbol') or ''),
        'side': signal_side,
        'trade_style': str(constraints.get('trade_style') or style.get('holding_period') or 'short_term_intraday'),
        'candidate_source': str(signal.get('candidate_source') or signal.get('source') or 'normal')[:80],
        'scanner_intent': str(signal.get('scanner_intent') or '')[:180],
        'rank': int(rank_index) + 1,
        'priority_score': _compact_number(signal.get('priority_score', abs(float(signal.get('score', 0) or 0)))),
        'current_price': _compact_number(signal.get('price')),
        'entry_quality': _compact_number(signal.get('entry_quality', breakdown.get('EntryGate'))),
        'setup_label': str(signal.get('setup_label') or breakdown.get('Setup') or ''),
        'signal_grade': str(signal.get('signal_grade') or breakdown.get('绛夌礆') or ''),
        'market_context': compact_market_context,
        'reference_trade_plan': compact_reference_trade_plan,
        'risk': {
            'trading_ok': bool(risk_status.get('trading_ok', True)),
            'halt_reason': str(risk_status.get('halt_reason') or '')[:180],
            'daily_loss_pct': _compact_number(risk_status.get('daily_loss_pct')),
            'consecutive_loss': int(risk_status.get('consecutive_loss', 0) or 0),
        },
        'portfolio': {
            'equity': _compact_number(portfolio.get('equity')),
            'active_position_count': int(portfolio.get('active_position_count', 0) or 0),
            'same_direction_count': int((portfolio.get('short_count') if signal_side == 'short' else portfolio.get('long_count')) or 0),
            'long_count': int(portfolio.get('long_count', 0) or 0),
            'short_count': int(portfolio.get('short_count', 0) or 0),
            'open_symbols': list(portfolio.get('open_symbols') or [])[:8],
        },
        'execution_policy': _compact_mapping(execution_policy, ['fixed_leverage', 'leverage_mode', 'min_order_margin_usdt', 'fixed_order_notional_usdt', 'margin_pct_range'], text_limit=80),
        'reference_context': compact_reference,
        'top_candidates': compact_top_candidates,
        'constraints': _compact_mapping(dict(constraints or {}), ['min_margin_pct', 'max_margin_pct', 'min_leverage', 'max_leverage', 'fixed_leverage', 'leverage_policy', 'min_order_margin_usdt', 'fixed_order_notional_usdt', 'trade_style', 'max_open_positions', 'max_same_direction'], text_limit=60),
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
            '1m': 6,
            '5m': 8,
            '15m': 8,
            '1h': 6,
            '4h': 4,
            '1d': 4,
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
        'regime_note': _string_schema(180),
        'trend_state': {'type': 'string', 'enum': ['trending_up', 'trending_down', 'range_mixed', 'transitioning', 'trend_unclear']},
        'timing_state': {'type': 'string', 'enum': ['enter_now', 'wait_pullback', 'wait_breakout', 'avoid_near_term']},
        'trade_side': {'type': 'string', 'enum': ['long', 'short']},
        'breakout_assessment': _string_schema(160),
        'rr_ratio': {'type': 'number'},
        'scale_in_recommended': {'type': 'boolean'},
        'scale_in_price': {'type': 'number'},
        'scale_in_qty_pct': {'type': 'number'},
        'scale_in_condition': _string_schema(160),
        'scale_in_note': _string_schema(180),
        'order_type': {'type': 'string', 'enum': ['market', 'limit']},
        'bot_instruction': {'type': 'string', 'enum': ['ENTER_MARKET', 'ENTER_LIMIT', 'OBSERVE', 'SKIP']},
        'entry_price': {'type': 'number'},
        'stop_loss': {'type': 'number'},
        'take_profit': {'type': 'number'},
        'market_read': _string_schema(240),
        'entry_plan': _string_schema(240),
        'leverage': {'type': 'integer'},
        'margin_pct': {'type': 'number'},
        'confidence': {'type': 'number'},
        'thesis': _string_schema(180),
        'reason_to_skip': _string_schema(180),
        'risk_notes': _string_array_schema(4, 120),
        'aggressive_note': _string_schema(180),
        'watch_trigger_type': {'type': 'string', 'enum': ['none', 'pullback_to_entry', 'breakout_reclaim', 'breakdown_confirm', 'volume_confirmation', 'manual_review']},
        'watch_trigger_price': {'type': 'number'},
        'watch_invalidation_price': {'type': 'number'},
        'watch_note': _string_schema(180),
        'recheck_reason': _string_schema(180),
        'watch_timeframe': _string_schema(40),
        'watch_price_zone_low': {'type': 'number'},
        'watch_price_zone_high': {'type': 'number'},
        'watch_structure_condition': _string_schema(180),
        'watch_volume_condition': _string_schema(180),
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
        'limit_cancel_condition': _string_schema(160),
        'limit_cancel_note': _string_schema(160),
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
            'market_read',
            'entry_plan',
            'reason_to_skip',
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
    if rows:
        last_row = list(rows[-1] or [])
        if len(last_row) >= 4:
            current_bar_high = _coerce_float(last_row[1], current_bar_high)
            current_bar_low = _coerce_float(last_row[2], current_bar_low)
        previous_rows = [list(row or []) for row in rows[:-1] if isinstance(row, list) and len(row) >= 4]
        if previous_rows:
            lookback_rows = previous_rows[-6:]
            highs = [_coerce_float(row[1], 0.0) for row in lookback_rows if _coerce_float(row[1], 0.0) > 0]
            lows = [_coerce_float(row[2], 0.0) for row in lookback_rows if _coerce_float(row[2], 0.0) > 0]
            if highs:
                prior_high = max(highs)
            if lows:
                prior_low = min(lows)
    atr_pct = _coerce_float(tf15.get('a', 0), 0.0)
    atr_price = max(entry_price * atr_pct / 100.0, entry_price * 0.003 if entry_price > 0 else 0.0)
    return {
        'prior_high': prior_high,
        'prior_low': prior_low,
        'current_bar_high': current_bar_high,
        'current_bar_low': current_bar_low,
        'atr_price': atr_price,
        'explosive': bool(tf15.get('xr', False)),
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
    atr_pct = max(_coerce_float(tf15.get('a', 0), 0.0), 0.0)
    atr_price = max(entry_price * atr_pct / 100.0, entry_price * 0.003)
    buffer = max(atr_price * 0.12, entry_price * 0.001)
    reward_floor = max(atr_price * 1.2, entry_price * 0.02)

    support_candidates: list[float] = []
    resistance_candidates: list[float] = []
    for key in ('nearest_support', 'recent_low', 'support_levels'):
        support_candidates.extend(_extract_numeric_candidates(levels.get(key)))
    for key in ('nearest_resistance', 'recent_high', 'resistance_levels'):
        resistance_candidates.extend(_extract_numeric_candidates(levels.get(key)))
    support_candidates.extend(_extract_numeric_candidates(tf15.get('lo')))
    support_candidates.extend(_extract_numeric_candidates(tf15.get('pl')))
    support_candidates.extend(_extract_numeric_candidates(tf15.get('cl')))
    resistance_candidates.extend(_extract_numeric_candidates(tf15.get('hi')))
    resistance_candidates.extend(_extract_numeric_candidates(tf15.get('ph')))
    resistance_candidates.extend(_extract_numeric_candidates(tf15.get('ch')))

    nearest_support = _nearest_price(support_candidates, entry_price, above=False)
    nearest_resistance = _nearest_price(resistance_candidates, entry_price, above=True)
    if str(side or '').lower() == 'short':
        stop_default = (
            nearest_resistance + buffer
            if nearest_resistance > 0
            else entry_price + max(buffer, entry_price * 0.015)
        )
        risk = max(stop_default - entry_price, entry_price * 0.006)
        tp_default = _nearest_price(support_candidates, entry_price, above=False)
        if tp_default <= 0:
            tp_default = entry_price - max(risk * 1.8, reward_floor)
        if tp_default >= entry_price:
            tp_default = entry_price - max(risk * 1.8, reward_floor)
    else:
        stop_default = (
            nearest_support - buffer
            if nearest_support > 0
            else entry_price - max(buffer, entry_price * 0.015)
        )
        risk = max(entry_price - stop_default, entry_price * 0.006)
        tp_default = _nearest_price(resistance_candidates, entry_price, above=True)
        if tp_default <= 0:
            tp_default = entry_price + max(risk * 1.8, reward_floor)
        if tp_default <= entry_price:
            tp_default = entry_price + max(risk * 1.8, reward_floor)
    return max(stop_default, 0.0), max(tp_default, 0.0)


def _sanitize_same_bar_stop(decision: Dict[str, Any], candidate: Dict[str, Any]) -> None:
    side = str(decision.get('trade_side') or candidate.get('side') or 'long').lower()
    entry_price = _coerce_float(decision.get('entry_price', 0), 0.0)
    stop_loss = _coerce_float(decision.get('stop_loss', 0), 0.0)
    if side not in {'long', 'short'} or entry_price <= 0 or stop_loss <= 0:
        return
    guard = _candidate_stop_guard(candidate, entry_price)
    if bool(guard.get('explosive', False)):
        return
    atr_price = max(_coerce_float(guard.get('atr_price', 0), 0.0), entry_price * 0.003)
    same_bar_tol = max(atr_price * 0.18, entry_price * 0.0012)
    anti_sweep_buffer = max(atr_price * 0.14, entry_price * 0.0010)
    adjusted = False
    if side == 'long':
        current_bar_low = _coerce_float(guard.get('current_bar_low', 0), 0.0)
        prior_low = _coerce_float(guard.get('prior_low', 0), 0.0)
        corrected_stop = prior_low - anti_sweep_buffer if 0 < prior_low < entry_price else 0.0
        near_same_bar_low = current_bar_low > 0 and stop_loss >= (current_bar_low - same_bar_tol)
        if corrected_stop > 0 and corrected_stop < entry_price and near_same_bar_low and stop_loss > corrected_stop:
            decision['stop_loss'] = corrected_stop
            adjusted = True
    else:
        current_bar_high = _coerce_float(guard.get('current_bar_high', 0), 0.0)
        prior_high = _coerce_float(guard.get('prior_high', 0), 0.0)
        corrected_stop = prior_high + anti_sweep_buffer if prior_high > entry_price else 0.0
        near_same_bar_high = current_bar_high > 0 and stop_loss <= (current_bar_high + same_bar_tol)
        if corrected_stop > entry_price and near_same_bar_high and stop_loss < corrected_stop:
            decision['stop_loss'] = corrected_stop
            adjusted = True
    if adjusted:
        note = '已自動把止損從當前15m K棒極值外移到前幾根 swing 外側，降低一般掃損機率。'
        existing = str(decision.get('stop_loss_reason') or '').strip()
        decision['stop_loss_reason'] = note if not existing else f'{note} {existing}'


def _constraint_brief(candidate: Dict[str, Any], constraints: Dict[str, Any]) -> str:
    trade_style = str(candidate.get('trade_style') or constraints.get('trade_style') or 'short_term_intraday')
    return (
        f'style={trade_style}; '
        f'leverage=symbol_max_{int(constraints.get("fixed_leverage", constraints.get("max_leverage", 25)) or 25)}x; '
        f'margin_pct={float(constraints.get("min_margin_pct", 0.03) or 0.03):.4f}-{float(constraints.get("max_margin_pct", 0.08) or 0.08):.4f}; '
        f'notional={float(constraints.get("fixed_order_notional_usdt", 20.0) or 20.0):.4f} USDT; '
        f'min_margin={float(constraints.get("min_order_margin_usdt", 0.1) or 0.1):.4f} USDT'
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
    trade_style = str(clean_payload.get('trade_style') or constraints.get('trade_style') or 'short_term_intraday')
    schema_hint = _response_shape_hint()
    constraint_brief = _constraint_brief(clean_payload, constraints)
    if compact:
        system_text = (
            'You are an internal crypto perpetual futures execution engine. '
            'Use only the supplied payload. Return exactly one valid JSON object, no markdown, no commentary. '
            'Numeric fields must stay numeric. Human-readable fields must be Traditional Chinese. '
            'Ignore all machine-generated scores, grades, setup labels, prior trade plans, timing states, and textual analysis if present. '
            'They are low-trust metadata and must not determine the final decision. '
            'Derive trend, structure, timing, entry, stop_loss, take_profit, RR, and should_trade only from raw OHLCV, multi-timeframe indicators, liquidity, derivatives, and risk constraints. '
            'You must classify the current market regime (trend_continuation / trend_pullback / range_reversion / consolidation_squeeze / transition_chop) and make entry/stop/take-profit logic regime-specific. '
            'Be aggressive but precise. '
            'If raw market structure, invalidation, liquidity, and RR form a clear asymmetric opportunity, prefer ENTER_MARKET or ENTER_LIMIT. '
            'Do not require perfect multi-timeframe alignment. '
            'If the setup is promising but the raw data shows timing is not yet executable, return OBSERVE with exactly one precise trigger. '
            'Do not use OBSERVE as generic caution. '
            'Do not use machine score or grade as a reason to skip or enter. '
            'If raw trend/structure clearly conflicts with the proposed direction, invalidation is unclear, or asymmetric RR is absent, do not force an entry; return SKIP or OBSERVE. '
            'This engine is optimized for 15m intraday execution, so stop loss placement must respect nearby real structure without using oversized swing distance. '
            'Unless the payload explicitly marks a same-bar exception, do not anchor stop_loss to the latest 15m candle high/low; use an earlier swing from prior bars plus a small anti-sweep buffer. '
            'JSON shape reminder: ' + schema_hint
        )
        user_text = (
            f'Generate one {trade_style} tactical execution-analysis object.\n'
            f'Hard bounds: {constraint_brief}.\n'
            'Use 1D/4H bias, 1H trend quality, 15m execution, 5m confirmation, 1m micro-timing.\n'
            'Ignore all machine-generated scores, grades, setup labels, prior trade plans, timing states, and textual analysis if present.\n'
            'They are low-trust metadata and must not determine the final decision.\n'
            'Derive trend, structure, timing, entry, stop_loss, take_profit, RR, and should_trade only from raw OHLCV, multi-timeframe indicators, liquidity, derivatives, and risk constraints.\n'
            'You must output market_regime and use regime-specific execution: trend_continuation/trend_pullback can be proactive; range_reversion should trade edges; consolidation_squeeze should wait/confirm squeeze break with explicit invalidation; transition_chop should avoid forced entries.\n'
            'For 15m short-term trading, place stop_loss just beyond the nearest meaningful swing high/low or invalidation structure with a small anti-sweep buffer; do not put it where normal wick noise can tag it, but do not place it excessively far either.\n'
            'Default rule: do not use the latest 15m candle high/low as the stop anchor. Look back several prior bars for the nearest meaningful swing first. Only if the payload clearly shows an explosive move / same-bar exception may you anchor to the latest bar extreme.\n'
            'If the symbol is BTC, ETH, XAU, XAG, or SOL, treat it as a lower-volatility major: keep the leverage at the symbol maximum in the payload and preserve at least the payload notional size instead of shrinking it to a small altcoin-sized trade.\n'
            'Enums: trend_state={trending_up,trending_down,range_mixed,transitioning,trend_unclear}; timing_state={enter_now,wait_pullback,wait_breakout,avoid_near_term}.\n'
            'Strict map: enter_now=>enter+should_trade=true; wait_pullback/wait_breakout=>observe+should_trade=false; avoid_near_term=>skip+should_trade=false.\n'
            'Set bot_instruction strictly as one of ENTER_MARKET, ENTER_LIMIT, OBSERVE, SKIP. Map enter+market=>ENTER_MARKET, enter+limit=>ENTER_LIMIT, observe=>OBSERVE, skip=>SKIP.\n'
            'Execution bot only collects market data and executes your returned instruction; do not rely on bot-side analysis and do not output ambiguous wording.\n'
            'Be aggressive but precise.\n'
            'If raw market structure, invalidation, liquidity, and RR form a clear asymmetric opportunity, prefer ENTER_MARKET or ENTER_LIMIT.\n'
            'Do not require perfect multi-timeframe alignment.\n'
            'If the setup is promising but the raw data shows timing is not yet executable, return OBSERVE with exactly one precise trigger.\n'
            'Do not use OBSERVE as generic caution.\n'
            'Do not use machine score or grade as a reason to skip or enter.\n'
            'If raw trend/structure is against the proposed direction, or invalidation/RR does not form a clear edge, do not enter and return SKIP or OBSERVE.\n'
            'If 15m execution structure is clear, invalidation is explicit, and RR is asymmetric, do not downgrade to OBSERVE only because of minor higher-timeframe mismatch.\n'
            'If the plan still requires waiting for a future candle close, retest, reclaim, breakdown, pullback, or any X-minute observation, you must return observe now; do not hide that waiting logic only inside entry_plan.\n'
            'If order_type=limit, fill limit_cancel_*; if market, limit_cancel_* must be 0/empty.\n'
            'If observe, choose exactly one primary watch path only, no OR, and fill watch_trigger_candle, watch_retest_rule, watch_volume_ratio_min, watch_micro_vwap_rule, watch_micro_ema20_rule.\n'
            'Observation fields must be explicit and machine-trackable: include price, timeframe, candle direction, retest rule, and any minimum volume threshold. Do not use vague wording such as wait for stability or watch a bit longer.\n'
            'Be only a little stricter than before: if price has already stretched, the retest has not confirmed, or 5m/1m follow-through is still one step short, switch to observe with one exact trigger; do not become broadly conservative.\n'
            'Do not be over-conservative. If current price is already near the trigger and invalidation is clear, prefer enter_now or a precise limit plan over vague observation.\n'
            'reason_to_skip must cite concrete blockers from the payload. market_read and entry_plan must be explicit.\n'
            'If force_recheck=true, either upgrade to an executable plan or state the one missing factor.\n'
            'Final consistency check:\n'
            '- If entry requires a future retest / candle close / reclaim / breakdown, output must be OBSERVE.\n'
            '- If output is ENTER, entry must be executable now or as a resting limit order with clear cancel rule.\n'
            '- If OBSERVE, output exactly one machine-trackable trigger.\n'
            '- If SKIP, reason_to_skip must come from raw data, not machine score or grade.\n'
            'Compact payload legend: multi_timeframe uses c=close a=atr_pct r=rsi x=adx e20/e50/e200 m20=ma20 v=vwap t=trend mh=macd_hist bbp=bb_position_pct vr=vol_ratio hi/lo=recent structure ph/pl=prior swing ch/cl=current bar high-low xr=explosive-move flag; multi_timeframe_pressure uses sb=structure_bias ts=trend_stack rb=recent_break pp/sp=pressure/support pa/sa=distance_atr c20/c50=close_vs_ema hh/hl/lh/ll=structure counts.\n'
            'timeframe_bars use start_ts + interval_ms with rows=[open,high,low,close,volume].\n'
            'Candidate payload JSON:\n'
            + json.dumps(clean_payload, ensure_ascii=False, separators=(',', ':'))
        )
    else:
        system_text = (
            'You are producing a tactical execution-analysis object for an internal short-term crypto perpetual futures engine. '
            'Use only raw market and risk fields in the candidate payload. '
            'Return one complete JSON object only, never chain-of-thought or commentary. '
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
            'If raw trend/structure clearly conflicts with the trade direction, invalidation is unclear, or asymmetric RR is absent, do not force an entry; return SKIP or OBSERVE. '
            'This engine trades mainly on the 15m timeframe, so stop loss placement must be beyond meaningful nearby structure with a small anti-sweep buffer, while still staying tight enough for intraday trading. '
            'Unless the payload explicitly marks a same-bar exception, do not anchor stop_loss to the latest 15m candle high/low; use an earlier swing from prior bars plus a small anti-sweep buffer. '
            'All human-readable explanation fields must be written in Traditional Chinese. Keep only enums, symbols, and timeframe tokens in English. '
            'JSON shape reminder: ' + schema_hint
        )
        user_text = (
            f'Generate one {trade_style} tactical execution-analysis object.\n'
            f'Hard bounds: {constraint_brief}; order_type must be market or limit.\n'
            'Compute entry_price, stop_loss, take_profit, and RR only from raw structure/current price/liquidity/invalidation in the payload.\n'
            'Output market_regime and regime_note explicitly, and make entry/stop/take-profit regime-specific: trend_continuation/trend_pullback can be aggressive, range_reversion must anchor to range edges, consolidation_squeeze must require clear trigger plus invalidation, transition_chop should reduce forced entries.\n'
            'For 15m short-term execution, stop_loss should sit slightly beyond the nearest valid swing high/low or invalidation structure so it is harder to get swept by ordinary noise, but it must not be left so far away that the trade stops fitting an intraday setup.\n'
            'Default rule: do not use the latest 15m candle high/low as the stop anchor. Look back several prior bars for the nearest meaningful swing first. Only if the payload clearly shows an explosive move / same-bar exception may you anchor to the latest bar extreme.\n'
            'If the symbol is BTC, ETH, XAU, XAG, or SOL, keep the order sizing aggressive for a lower-volatility major: use the symbol-max leverage already provided and do not reduce the payload notional below the configured major-coin size.\n'
            'Classify trend_state and timing_state explicitly. Action mapping is strict: enter_now=>enter; wait_pullback/wait_breakout=>observe; avoid_near_term=>skip.\n'
            'Set bot_instruction strictly as one of ENTER_MARKET, ENTER_LIMIT, OBSERVE, SKIP. Map enter+market=>ENTER_MARKET, enter+limit=>ENTER_LIMIT, observe=>OBSERVE, skip=>SKIP.\n'
            'Execution bot only collects market data and executes your returned instruction; do not rely on bot-side analysis and do not output ambiguous wording.\n'
            'Be aggressive but precise: if edge is clear and executable now, prefer ENTER; if raw trend/structure conflicts, invalidation is unclear, or RR edge is absent, do not enter.\n'
            'If 15m execution edge is clear with explicit invalidation and asymmetric RR, do not downgrade to OBSERVE only due to minor higher-timeframe mismatch.\n'
            'If the plan still needs waiting for a future candle close, retest, reclaim, breakdown, pullback, or any X-minute observation, return observe now instead of hiding that wait inside entry_plan.\n'
            'Set trade_side and breakout_assessment explicitly. If scale-in is not recommended, return false and 0/empty fields.\n'
            'All numeric fields must be raw JSON numbers, and the reply must be one complete JSON object only.\n'
            'Analyze 1D/4H for larger bias, 1H for trend quality, 15m for the main entry frame, 5m for confirmation, and 1m only for micro-timing.\n'
            'timeframe_bars format: each timeframe uses start_ts + interval_ms, and rows are [open,high,low,close,volume] from oldest to newest.\n'
            'market_read and entry_plan must be explicit. reason_to_skip must never be generic and must cite concrete blockers from the payload.\n'
            'Choose one best executable path now: market, precise limit pullback/retest, or no trade yet with one precise recheck trigger.\n'
            'If order_type is limit, return limit_cancel_price, limit_cancel_timeframe, limit_cancel_condition, and limit_cancel_note; for market, set those fields to 0/empty.\n'
            'If should_trade is false, preserve the idea with one narrow watch plan and machine-readable tracking fields the bot can track exactly.\n'
            'Make watch fields explicit and machine-trackable with concrete price, timeframe, candle, retest, and volume conditions; never use vague watch text.\n'
            'If force_recheck is true, either upgrade to an executable plan or explain the one missing factor still preventing execution.\n'
            'Final consistency check:\n'
            '- If entry requires a future retest / candle close / reclaim / breakdown, output must be OBSERVE.\n'
            '- If output is ENTER, entry must be executable now or as a resting limit order with clear cancel rule.\n'
            '- If OBSERVE, output exactly one machine-trackable trigger.\n'
            '- If SKIP, reason_to_skip must come from raw data, not machine score or grade.\n'
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
        int(max_output_tokens or config.get('max_output_tokens', 560) or 560),
        480 if structured else 260,
    )
    body = {
        'model': str(model or config.get('model') or 'gpt-5.4-mini'),
        'input': _build_messages(candidate, compact=compact_prompt, logger=logger),
        'max_output_tokens': effective_max_output_tokens,
    }
    body['text'] = {
        'verbosity': 'low',
        'format': {
            'type': 'json_object',
        }
    }
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
    if str(decision.get('watch_trigger_type') or 'none') != 'none':
        return True
    if str(decision.get('timing_state') or '') in {'wait_pullback', 'wait_breakout'}:
        return True
    text = ' '.join(
        [
            str(decision.get('entry_plan') or ''),
            str(decision.get('watch_note') or ''),
            str(decision.get('recheck_reason') or ''),
            str(decision.get('reason_to_skip') or ''),
            str(decision.get('breakout_assessment') or ''),
        ]
    )
    has_waiting_language = any(
        token in text
        for token in (
            '等待', '先等', '等到', '回踩', '回測', '回抽', '收盤', '站上', '站不上', '跌破', '突破', '確認後', '再執行',
            'retest', 'reclaim', 'close above', 'close below', 'wait ', 'wait_', 'pullback', 'breakout', 'breakdown',
        )
    )
    has_executable_plan = (
        float(decision.get('entry_price', 0) or 0) > 0
        or float(decision.get('watch_trigger_price', 0) or 0) > 0
        or str(decision.get('order_type') or '') == 'limit'
    )
    return has_waiting_language and has_executable_plan


def _normalize_decision(raw: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    raw = dict(raw or {})
    side = str(candidate.get('side') or 'long').lower()
    if side not in {'long', 'short'}:
        side = 'long'
    constraints = dict(candidate.get('constraints') or {})
    watch_plan = dict(raw.get('watch_plan') or {})
    entry_default = _coerce_float(candidate.get('current_price', candidate.get('entry_price', 0)), 0.0)
    if entry_default <= 0:
        ticker_ref = (((candidate.get('market_context') or {}).get('market_state') or {}).get('ticker') or {}).get('last', 0)
        entry_default = _coerce_float(ticker_ref, 0.0)
    if entry_default <= 0:
        entry_default = _coerce_float(((((candidate.get('market_context') or {}).get('basic_market_data') or {}).get('current_price') or 0)), 0.0)
    stop_default, tp_default = _raw_default_stop_take(candidate, side, entry_default)
    if side == 'long':
        if stop_default <= 0 and entry_default > 0:
            stop_default = entry_default * 0.985
        if tp_default <= 0 and entry_default > 0:
            tp_default = entry_default * 1.02
    else:
        if stop_default <= 0 and entry_default > 0:
            stop_default = entry_default * 1.015
        if tp_default <= 0 and entry_default > 0:
            tp_default = entry_default * 0.98
    decision = {
        'should_trade': bool(raw.get('should_trade', False)),
        'action': str(raw.get('action') or '').strip().lower(),
        'market_regime': str(raw.get('market_regime') or '').strip().lower(),
        'regime_note': str(raw.get('regime_note') or '').strip(),
        'trend_state': str(raw.get('trend_state') or '').strip().lower(),
        'timing_state': str(raw.get('timing_state') or '').strip().lower(),
        'trade_side': str(raw.get('trade_side') or side).strip().lower(),
        'breakout_assessment': str(raw.get('breakout_assessment') or '').strip(),
        'rr_ratio': max(_coerce_float(raw.get('rr_ratio', raw.get('rr', raw.get('risk_reward', 0))), 0.0), 0.0),
        'scale_in_recommended': bool(raw.get('scale_in_recommended', False)),
        'scale_in_price': max(_coerce_float(raw.get('scale_in_price', 0), 0.0), 0.0),
        'scale_in_qty_pct': _clamp(_coerce_float(raw.get('scale_in_qty_pct', 0), 0.0), 0.0, 1.0),
        'scale_in_condition': str(raw.get('scale_in_condition') or '').strip(),
        'scale_in_note': str(raw.get('scale_in_note') or '').strip(),
        'order_type': 'limit' if str(raw.get('order_type') or '').lower() == 'limit' else 'market',
        'bot_instruction': str(raw.get('bot_instruction') or '').strip().upper(),
        'entry_price': _coerce_float(raw.get('entry_price', entry_default), entry_default),
        'stop_loss': _coerce_float(raw.get('stop_loss', stop_default), stop_default),
        'take_profit': _coerce_float(raw.get('take_profit', tp_default), tp_default),
        'market_read': str(raw.get('market_read') or '').strip(),
        'entry_plan': str(raw.get('entry_plan') or '').strip(),
        'entry_reason': str(raw.get('entry_reason') or '').strip(),
        'stop_loss_reason': str(raw.get('stop_loss_reason') or '').strip(),
        'take_profit_plan': str(raw.get('take_profit_plan') or '').strip(),
        'if_missed_plan': str(raw.get('if_missed_plan') or '').strip(),
        'reference_summary': str(raw.get('reference_summary') or '').strip(),
        'chase_if_triggered': bool(raw.get('chase_if_triggered', False)),
        'chase_trigger_price': _coerce_float(raw.get('chase_trigger_price', 0), 0.0),
        'chase_limit_price': _coerce_float(raw.get('chase_limit_price', 0), 0.0),
        'trail_trigger_atr_hint': _coerce_float(raw.get('trail_trigger_atr_hint', 0), 0.0),
        'trail_pct_hint': _coerce_float(raw.get('trail_pct_hint', 0), 0.0),
        'breakeven_atr_hint': _coerce_float(raw.get('breakeven_atr_hint', 0), 0.0),
        'dynamic_take_profit_hint': _coerce_float(raw.get('dynamic_take_profit_hint', 0), 0.0),
        'leverage': int(_clamp(_coerce_float(raw.get('leverage', constraints.get('fixed_leverage', constraints.get('min_leverage', 4))), constraints.get('fixed_leverage', constraints.get('min_leverage', 4))), constraints.get('min_leverage', 4), constraints.get('max_leverage', 25))),
        'margin_pct': _clamp(_coerce_float(raw.get('margin_pct', constraints.get('min_margin_pct', 0.03)), constraints.get('min_margin_pct', 0.03)), constraints.get('min_margin_pct', 0.03), constraints.get('max_margin_pct', 0.08)),
        'confidence': _clamp(_coerce_float(raw.get('confidence', 0), 0.0), 0, 100),
        'thesis': str(raw.get('thesis') or '').strip(),
        'reason_to_skip': str(raw.get('reason_to_skip') or '').strip(),
        'risk_notes': [str(x).strip() for x in list(raw.get('risk_notes') or []) if str(x).strip()][:6],
        'aggressive_note': str(raw.get('aggressive_note') or '').strip(),
        'watch_trigger_type': str(raw.get('watch_trigger_type') or 'none').strip(),
        'watch_trigger_price': _coerce_float(raw.get('watch_trigger_price', 0), 0.0),
        'watch_invalidation_price': _coerce_float(raw.get('watch_invalidation_price', 0), 0.0),
        'watch_note': str(raw.get('watch_note') or '').strip(),
        'recheck_reason': str(raw.get('recheck_reason') or '').strip(),
        'watch_timeframe': str(raw.get('watch_timeframe', raw.get('timeframe')) or '').strip(),
        'watch_price_zone_low': _coerce_float(raw.get('watch_price_zone_low', 0), 0.0),
        'watch_price_zone_high': _coerce_float(raw.get('watch_price_zone_high', 0), 0.0),
        'watch_structure_condition': str(raw.get('watch_structure_condition') or '').strip(),
        'watch_volume_condition': str(raw.get('watch_volume_condition') or '').strip(),
        'watch_checklist': [str(x).strip() for x in list(raw.get('watch_checklist') or []) if str(x).strip()][:8],
        'watch_confirmations': [str(x).strip() for x in list(raw.get('watch_confirmations') or []) if str(x).strip()][:8],
        'watch_invalidations': [str(x).strip() for x in list(raw.get('watch_invalidations') or []) if str(x).strip()][:8],
        'watch_trigger_candle': str(raw.get('watch_trigger_candle') or 'none').strip(),
        'watch_retest_rule': str(raw.get('watch_retest_rule') or 'none').strip(),
        'watch_volume_ratio_min': _coerce_float(raw.get('watch_volume_ratio_min', 0), 0.0),
        'watch_micro_vwap_rule': str(raw.get('watch_micro_vwap_rule') or 'none').strip(),
        'watch_micro_ema20_rule': str(raw.get('watch_micro_ema20_rule') or 'none').strip(),
        'watch_recheck_priority': _clamp(_coerce_float(raw.get('watch_recheck_priority', 0), 0.0), 0, 100),
        'limit_cancel_price': _coerce_float(raw.get('limit_cancel_price', 0), 0.0),
        'limit_cancel_timeframe': str(raw.get('limit_cancel_timeframe') or '').strip(),
        'limit_cancel_condition': str(raw.get('limit_cancel_condition') or '').strip(),
        'limit_cancel_note': str(raw.get('limit_cancel_note') or '').strip(),
    }
    allowed_watch = {'none', 'pullback_to_entry', 'breakout_reclaim', 'breakdown_confirm', 'volume_confirmation', 'manual_review'}
    allowed_trigger_candles = {'none', 'close_above', 'close_below'}
    allowed_retest_rules = {'none', 'hold_above', 'hold_below', 'fail_above', 'fail_below'}
    allowed_micro_rules = {'none', 'above', 'below'}
    allowed_actions = {'enter', 'observe', 'skip'}
    allowed_regimes = {'trend_continuation', 'trend_pullback', 'range_reversion', 'consolidation_squeeze', 'transition_chop'}
    allowed_trend_states = {'trending_up', 'trending_down', 'range_mixed', 'transitioning', 'trend_unclear'}
    allowed_timing_states = {'enter_now', 'wait_pullback', 'wait_breakout', 'avoid_near_term'}
    if decision['watch_trigger_type'] not in allowed_watch:
        decision['watch_trigger_type'] = 'none'
    if not decision['watch_note'] and watch_plan.get('watch_trigger'):
        decision['watch_note'] = _single_watch_path(watch_plan.get('watch_trigger'))
    if not decision['recheck_reason'] and watch_plan.get('recheck_reason'):
        decision['recheck_reason'] = _single_watch_path(watch_plan.get('recheck_reason'))
    if not decision['watch_timeframe'] and watch_plan.get('timeframe'):
        decision['watch_timeframe'] = str(watch_plan.get('timeframe') or '').strip()
    if not decision['watch_checklist'] and watch_plan.get('checklist'):
        decision['watch_checklist'] = [str(x).strip() for x in list(watch_plan.get('checklist') or []) if str(x).strip()][:8]
    if not decision['watch_confirmations'] and watch_plan.get('confirmations'):
        decision['watch_confirmations'] = [str(x).strip() for x in list(watch_plan.get('confirmations') or []) if str(x).strip()][:8]
    if not decision['watch_invalidations'] and watch_plan.get('invalidations'):
        decision['watch_invalidations'] = [str(x).strip() for x in list(watch_plan.get('invalidations') or []) if str(x).strip()][:8]
    if not decision['watch_invalidations'] and watch_plan.get('invalidation'):
        decision['watch_invalidations'] = [str(watch_plan.get('invalidation') or '').strip()]
    if not decision['reason_to_skip'] and raw.get('should_trade_reason'):
        decision['reason_to_skip'] = str(raw.get('should_trade_reason') or '').strip()
    if decision['watch_trigger_type'] == 'none':
        watch_trigger_text = str(raw.get('watch_trigger') or watch_plan.get('watch_trigger') or '').lower()
        if 'pullback' in watch_trigger_text:
            decision['watch_trigger_type'] = 'pullback_to_entry'
        elif 'breakdown' in watch_trigger_text:
            decision['watch_trigger_type'] = 'breakdown_confirm'
        elif 'breakout' in watch_trigger_text or 'close above' in watch_trigger_text or 'hold above' in watch_trigger_text:
            decision['watch_trigger_type'] = 'breakout_reclaim'
    if decision['watch_trigger_price'] <= 0:
        if raw.get('watch_trigger_price') is None and watch_plan.get('watch_trigger'):
            match = re.search(r'([0-9]+(?:\.[0-9]+)?)', str(watch_plan.get('watch_trigger') or ''))
            if match:
                decision['watch_trigger_price'] = _coerce_float(match.group(1), 0.0)
        if decision['watch_trigger_price'] <= 0:
            decision['watch_trigger_price'] = _coerce_float(raw.get('watch_trigger_price', raw.get('entry_price', 0)), 0.0)
    if decision['watch_invalidation_price'] <= 0:
        if raw.get('watch_invalidation_price') is None and watch_plan.get('invalidation'):
            match = re.search(r'([0-9]+(?:\.[0-9]+)?)', str(watch_plan.get('invalidation') or ''))
            if match:
                decision['watch_invalidation_price'] = _coerce_float(match.group(1), 0.0)
        if decision['watch_invalidation_price'] <= 0:
            decision['watch_invalidation_price'] = _coerce_float(raw.get('watch_invalidation_price', raw.get('stop_loss', 0)), 0.0)
    if not decision['watch_note'] and raw.get('watch_trigger'):
        decision['watch_note'] = _single_watch_path(raw.get('watch_trigger'))
    if not decision['watch_invalidations'] and raw.get('invalidation'):
        decision['watch_invalidations'] = [str(raw.get('invalidation') or '').strip()]
    if decision['watch_trigger_candle'] not in allowed_trigger_candles:
        decision['watch_trigger_candle'] = 'none'
    if decision['watch_retest_rule'] not in allowed_retest_rules:
        decision['watch_retest_rule'] = 'none'
    if decision['watch_micro_vwap_rule'] not in allowed_micro_rules:
        decision['watch_micro_vwap_rule'] = 'none'
    if decision['watch_micro_ema20_rule'] not in allowed_micro_rules:
        decision['watch_micro_ema20_rule'] = 'none'
    trigger_text_full = ' '.join([
        str(raw.get('watch_trigger') or ''),
        str(watch_plan.get('watch_trigger') or ''),
        str(raw.get('recheck_reason') or ''),
        str(watch_plan.get('recheck_reason') or ''),
        str(raw.get('watch_note') or ''),
        str(raw.get('watch_structure_condition') or ''),
        str(raw.get('watch_volume_condition') or ''),
    ]).lower()
    if decision['watch_trigger_candle'] == 'none':
        if 'close above' in trigger_text_full:
            decision['watch_trigger_candle'] = 'close_above'
        elif 'close below' in trigger_text_full or 'breakdown under' in trigger_text_full:
            decision['watch_trigger_candle'] = 'close_below'
    if decision['watch_retest_rule'] == 'none':
        if 'hold above' in trigger_text_full:
            decision['watch_retest_rule'] = 'hold_above'
        elif 'hold below' in trigger_text_full:
            decision['watch_retest_rule'] = 'hold_below'
        elif 'fail below' in trigger_text_full or 'stays below' in trigger_text_full or 'remain below' in trigger_text_full:
            decision['watch_retest_rule'] = 'hold_below'
        elif 'fail above' in trigger_text_full or 'stays above' in trigger_text_full or 'remain above' in trigger_text_full:
            decision['watch_retest_rule'] = 'hold_above'
    if decision['watch_micro_vwap_rule'] == 'none':
        if 'below vwap' in trigger_text_full or 'loss of vwap' in trigger_text_full:
            decision['watch_micro_vwap_rule'] = 'below'
        elif 'above vwap' in trigger_text_full or 'reclaim vwap' in trigger_text_full:
            decision['watch_micro_vwap_rule'] = 'above'
    if decision['watch_micro_ema20_rule'] == 'none':
        if 'below ema20' in trigger_text_full or 'loss of ema20' in trigger_text_full:
            decision['watch_micro_ema20_rule'] = 'below'
        elif 'above ema20' in trigger_text_full or 'reclaim ema20' in trigger_text_full:
            decision['watch_micro_ema20_rule'] = 'above'
    if decision['watch_volume_ratio_min'] <= 0:
        match = re.search(r'above\s+([0-9]+(?:\.[0-9]+)?)', trigger_text_full)
        if match:
            decision['watch_volume_ratio_min'] = _coerce_float(match.group(1), 0.0)
    if decision['action'] not in allowed_actions:
        decision['action'] = 'enter' if decision['should_trade'] else ('observe' if decision['watch_trigger_type'] != 'none' else 'skip')
    if decision['market_regime'] not in allowed_regimes:
        if decision['trend_state'] in {'trending_up', 'trending_down'}:
            decision['market_regime'] = 'trend_pullback' if decision['timing_state'] == 'wait_pullback' else 'trend_continuation'
        elif decision['trend_state'] == 'range_mixed':
            decision['market_regime'] = 'range_reversion'
        elif decision['trend_state'] == 'transitioning':
            decision['market_regime'] = 'consolidation_squeeze'
        else:
            decision['market_regime'] = 'transition_chop'
    if decision['trend_state'] not in allowed_trend_states:
        decision['trend_state'] = 'trending_up' if side == 'long' and decision['should_trade'] else ('trending_down' if side == 'short' and decision['should_trade'] else 'trend_unclear')
    if decision['timing_state'] not in allowed_timing_states:
        if decision['action'] == 'enter':
            decision['timing_state'] = 'enter_now'
        elif decision['watch_trigger_type'] == 'pullback_to_entry':
            decision['timing_state'] = 'wait_pullback'
        elif decision['watch_trigger_type'] != 'none':
            decision['timing_state'] = 'wait_breakout'
        else:
            decision['timing_state'] = 'avoid_near_term'
    if decision['timing_state'] == 'enter_now':
        decision['action'] = 'enter'
        decision['should_trade'] = True
    elif decision['timing_state'] in {'wait_pullback', 'wait_breakout'}:
        decision['action'] = 'observe'
        decision['should_trade'] = False
        if decision['watch_trigger_type'] == 'none':
            decision['watch_trigger_type'] = _watch_trigger_type_from_side_timing(side, decision['timing_state'])
    elif decision['timing_state'] == 'avoid_near_term' and decision['action'] != 'observe':
        decision['action'] = 'skip'
        decision['should_trade'] = False
    if decision['action'] != 'observe' and _should_promote_to_observe(decision):
        decision['action'] = 'observe'
        decision['should_trade'] = False
        if decision['timing_state'] not in {'wait_pullback', 'wait_breakout'}:
            decision['timing_state'] = _infer_wait_timing_state(decision)
        if decision['watch_trigger_type'] == 'none':
            decision['watch_trigger_type'] = _watch_trigger_type_from_side_timing(side, decision['timing_state'])
    if decision['trade_side'] not in {'long', 'short'}:
        decision['trade_side'] = side
    if constraints.get('fixed_leverage'):
        decision['leverage'] = int(constraints.get('fixed_leverage') or decision['leverage'])
    _sanitize_same_bar_stop(decision, candidate)
    decision['market_read'] = _cn_phrase(decision['market_read'])
    decision['entry_plan'] = _cn_phrase(decision['entry_plan'])
    decision['regime_note'] = _cn_phrase(decision['regime_note'])
    decision['watch_note'] = _single_watch_path(decision['watch_note'])
    decision['recheck_reason'] = _single_watch_path(decision['recheck_reason'])
    decision['breakout_assessment'] = decision['breakout_assessment'][:180]
    decision['scale_in_condition'] = decision['scale_in_condition'][:220]
    decision['scale_in_note'] = decision['scale_in_note'][:220]
    decision['market_read'] = decision['market_read'][:280]
    decision['entry_plan'] = decision['entry_plan'][:280]
    decision['regime_note'] = decision['regime_note'][:180]
    decision['entry_reason'] = decision['entry_reason'][:220]
    decision['stop_loss_reason'] = decision['stop_loss_reason'][:220]
    decision['take_profit_plan'] = decision['take_profit_plan'][:280]
    decision['if_missed_plan'] = decision['if_missed_plan'][:220]
    decision['reference_summary'] = decision['reference_summary'][:220]
    decision['watch_note'] = decision['watch_note'][:220]
    decision['recheck_reason'] = decision['recheck_reason'][:220]
    decision['watch_timeframe'] = decision['watch_timeframe'][:80]
    decision['watch_structure_condition'] = decision['watch_structure_condition'][:220]
    decision['watch_volume_condition'] = decision['watch_volume_condition'][:220]
    decision['watch_checklist'] = [str(x)[:180] for x in decision.get('watch_checklist', [])][:8]
    decision['watch_confirmations'] = [str(x)[:180] for x in decision.get('watch_confirmations', [])][:8]
    decision['watch_invalidations'] = [str(x)[:180] for x in decision.get('watch_invalidations', [])][:8]
    decision['watch_trigger_candle'] = decision['watch_trigger_candle'][:40]
    decision['watch_retest_rule'] = decision['watch_retest_rule'][:40]
    decision['watch_micro_vwap_rule'] = decision['watch_micro_vwap_rule'][:40]
    decision['watch_micro_ema20_rule'] = decision['watch_micro_ema20_rule'][:40]
    decision['limit_cancel_timeframe'] = decision['limit_cancel_timeframe'][:80]
    decision['limit_cancel_condition'] = decision['limit_cancel_condition'][:220]
    decision['limit_cancel_note'] = decision['limit_cancel_note'][:220]
    decision['watch_trigger_price'] = max(_coerce_float(decision.get('watch_trigger_price', 0), 0.0), 0.0)
    decision['watch_invalidation_price'] = max(_coerce_float(decision.get('watch_invalidation_price', 0), 0.0), 0.0)
    decision['watch_price_zone_low'] = max(_coerce_float(decision.get('watch_price_zone_low', 0), 0.0), 0.0)
    decision['watch_price_zone_high'] = max(_coerce_float(decision.get('watch_price_zone_high', 0), 0.0), 0.0)
    if decision['watch_trigger_type'] == 'none':
        decision['watch_trigger_price'] = 0.0
        decision['watch_invalidation_price'] = 0.0
        decision['watch_price_zone_low'] = 0.0
        decision['watch_price_zone_high'] = 0.0
    if decision['action'] == 'observe' and decision['watch_trigger_type'] == 'none':
        decision['watch_trigger_type'] = _watch_trigger_type_from_side_timing(side, decision['timing_state'])
        decision['watch_trigger_price'] = decision['entry_price']
        decision['watch_invalidation_price'] = decision['stop_loss']
        decision['watch_price_zone_low'] = min(decision['entry_price'], decision['watch_trigger_price'])
        decision['watch_price_zone_high'] = max(decision['entry_price'], decision['watch_trigger_price'])
    if decision['action'] == 'observe' and decision['watch_trigger_price'] > 0:
        if decision['watch_price_zone_low'] <= 0 and decision['watch_price_zone_high'] <= 0:
            if decision['timing_state'] == 'wait_pullback' and decision['entry_price'] > 0:
                decision['watch_price_zone_low'] = min(decision['entry_price'], decision['watch_trigger_price'])
                decision['watch_price_zone_high'] = max(decision['entry_price'], decision['watch_trigger_price'])
            else:
                decision['watch_price_zone_low'] = decision['watch_trigger_price']
                decision['watch_price_zone_high'] = decision['watch_trigger_price']
    if decision['action'] == 'observe':
        if decision['watch_trigger_candle'] == 'none':
            decision['watch_trigger_candle'] = 'close_below' if side == 'short' else 'close_above'
        if decision['watch_retest_rule'] == 'none':
            decision['watch_retest_rule'] = 'hold_below' if side == 'short' else 'hold_above'
        if decision['watch_micro_vwap_rule'] == 'none':
            decision['watch_micro_vwap_rule'] = 'below' if side == 'short' else 'above'
        if decision['watch_micro_ema20_rule'] == 'none':
            decision['watch_micro_ema20_rule'] = 'below' if side == 'short' else 'above'
        if decision['watch_volume_ratio_min'] <= 0:
            decision['watch_volume_ratio_min'] = 1.05
        if not decision['watch_structure_condition']:
            decision['watch_structure_condition'] = '需要先完成主觸發與回測規則，再重新送 AI。'
        if not decision['watch_volume_condition']:
            decision['watch_volume_condition'] = '需要確認量能不是假突破後立刻衰退。'
        if not decision['recheck_reason']:
            trigger_price_text = str(_compact_number(decision.get('watch_trigger_price', 0)))
            timeframe_text = decision.get('watch_timeframe') or '15m'
            decision['recheck_reason'] = '等待{} 在 {} 完成 {}/{} 後重送 AI。'.format(
                timeframe_text,
                trigger_price_text if trigger_price_text not in {'0', '0.0'} else '主觸發價',
                decision['watch_trigger_candle'],
                decision['watch_retest_rule'],
            )
    decision['chase_trigger_price'] = max(_coerce_float(decision.get('chase_trigger_price', 0), 0.0), 0.0)
    decision['chase_limit_price'] = max(_coerce_float(decision.get('chase_limit_price', 0), 0.0), 0.0)
    if not bool(decision.get('chase_if_triggered')):
        decision['chase_trigger_price'] = 0.0
        decision['chase_limit_price'] = 0.0
    decision['trail_trigger_atr_hint'] = _clamp(decision.get('trail_trigger_atr_hint', 0), 0, 6)
    decision['trail_pct_hint'] = _clamp(decision.get('trail_pct_hint', 0), 0, 0.3)
    decision['breakeven_atr_hint'] = _clamp(decision.get('breakeven_atr_hint', 0), 0, 4)
    decision['dynamic_take_profit_hint'] = max(_coerce_float(decision.get('dynamic_take_profit_hint', 0), 0.0), 0.0)
    decision['limit_cancel_price'] = max(_coerce_float(decision.get('limit_cancel_price', 0), 0.0), 0.0)
    if not math.isfinite(decision['entry_price']) or decision['entry_price'] <= 0:
        decision['entry_price'] = entry_default
    if not math.isfinite(decision['stop_loss']) or decision['stop_loss'] <= 0:
        decision['stop_loss'] = stop_default
    if not math.isfinite(decision['take_profit']) or decision['take_profit'] <= 0:
        decision['take_profit'] = tp_default
    if side == 'long':
        if decision['stop_loss'] >= decision['entry_price']:
            decision['stop_loss'] = stop_default if stop_default < decision['entry_price'] else decision['entry_price'] * 0.985
        if decision['take_profit'] <= decision['entry_price']:
            decision['take_profit'] = tp_default if tp_default > decision['entry_price'] else decision['entry_price'] * 1.02
    else:
        if decision['stop_loss'] <= decision['entry_price']:
            decision['stop_loss'] = stop_default if stop_default > decision['entry_price'] else decision['entry_price'] * 1.015
        if decision['take_profit'] >= decision['entry_price']:
            decision['take_profit'] = tp_default if tp_default < decision['entry_price'] else decision['entry_price'] * 0.98
    _sanitize_same_bar_stop(decision, candidate)
    decision['rr_ratio'] = round(abs((decision['take_profit'] - decision['entry_price']) / max(abs(decision['entry_price'] - decision['stop_loss']), 1e-9)), 4)
    if decision['order_type'] != 'limit':
        decision['limit_cancel_price'] = 0.0
        decision['limit_cancel_timeframe'] = ''
        decision['limit_cancel_condition'] = ''
        decision['limit_cancel_note'] = ''
    else:
        if decision['limit_cancel_price'] <= 0:
            decision['limit_cancel_price'] = decision['watch_invalidation_price'] if decision['watch_invalidation_price'] > 0 else decision['stop_loss']
        if not decision['limit_cancel_timeframe']:
            decision['limit_cancel_timeframe'] = decision['watch_timeframe'] or '15m'
        if not decision['limit_cancel_condition']:
            decision['limit_cancel_condition'] = (
                'Cancel the resting long if the invalidation/support breaks before fill.'
                if side == 'long'
                else 'Cancel the resting short if invalidation/resistance is reclaimed before fill.'
            )
        if not decision['limit_cancel_note']:
            decision['limit_cancel_note'] = 'This cancellation rule applies only while the limit order is still resting and unfilled.'
    if decision['should_trade'] and not decision['thesis']:
        decision['thesis'] = 'Model approved the trade using the provided market, execution, and risk context.'
    generic_skip = str(decision.get('reason_to_skip') or '').strip().lower() in {
        '',
        'model rejected the setup after reviewing the full payload.',
        'model rejected the setup after reviewing the full payload',
    }
    if not decision['market_read']:
        decision['market_read'] = '{} | {}'.format(decision['trend_state'], decision['breakout_assessment'] or 'structure review incomplete').strip(' |')
    if not decision['regime_note']:
        regime_note_map = {
            'trend_continuation': '順勢延續盤，允許沿趨勢執行，但失效點必須明確。',
            'trend_pullback': '順勢回踩盤，優先回踩/回測進場，不追過度延伸。',
            'range_reversion': '區間盤，優先區間邊緣進場，中段避免硬做。',
            'consolidation_squeeze': '盤堅擠壓盤，等待明確觸發與失效後再執行。',
            'transition_chop': '轉折混沌盤，避免在方向不清時強行進場。',
        }
        decision['regime_note'] = regime_note_map.get(decision['market_regime'], '依盤型調整進場、停損與停利，不套用單一模板。')
    if not decision['entry_plan']:
        timing_map = {
            'enter_now': '目前結構可直接執行，依照既定進場與失效點操作。',
            'wait_pullback': '先不要追價，等回踩進場區後再執行。',
            'wait_breakout': '先不要進場，等突破或跌破確認後再執行。',
            'avoid_near_term': '這筆近期觀望，不適合直接執行。',
        }
        decision['entry_plan'] = timing_map.get(decision['timing_state'], timing_map['avoid_near_term'])
    if not decision['should_trade'] and generic_skip:
        if decision['action'] == 'observe':
            blocker = decision['recheck_reason'] or decision['watch_structure_condition'] or decision['watch_volume_condition'] or decision['market_read']
            decision['reason_to_skip'] = ('目前先不進場，等待：{}。'.format(blocker[:140]) if blocker else '目前先不進場，仍缺少關鍵確認。')
        else:
            blocker = decision['market_read'] or decision['breakout_assessment'] or decision['stop_loss_reason']
            decision['reason_to_skip'] = ('近期觀望，不適合執行：{}。'.format(blocker[:140]) if blocker else '近期觀望，因為結構與進場條件尚未對齊。')
    if decision['rr_ratio'] <= 0:
        decision['rr_ratio'] = round(abs((decision['take_profit'] - decision['entry_price']) / max(abs(decision['entry_price'] - decision['stop_loss']), 1e-9)), 4)
    if not decision['scale_in_recommended']:
        decision['scale_in_price'] = 0.0
        decision['scale_in_qty_pct'] = 0.0
        decision['scale_in_condition'] = ''
        decision['scale_in_note'] = ''
    if decision['action'] == 'enter':
        decision['timing_state'] = 'enter_now'
        decision['should_trade'] = True
    elif decision['action'] in ('observe', 'skip'):
        decision['should_trade'] = False
        if decision['action'] == 'skip':
            decision['timing_state'] = 'avoid_near_term'
    if decision['action'] == 'enter':
        decision['bot_instruction'] = 'ENTER_LIMIT' if decision['order_type'] == 'limit' else 'ENTER_MARKET'
    elif decision['action'] == 'observe':
        decision['bot_instruction'] = 'OBSERVE'
    else:
        decision['bot_instruction'] = 'SKIP'
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

    if not force_recheck and rank > int(config.get('top_k_per_scan', 5) or 5):
        result = {'status': 'not_ranked', 'decision': None, 'payload_hash': payload_hash, 'symbol_state': symbol_state}
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

    est_call_cost_twd = estimate_cost_twd(config, input_tokens=2200, output_tokens=500)
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
        primary_model = str(config.get('model') or 'gpt-5.4').strip()
        upgrade_model = str(config.get('upgrade_model') or 'gpt-5.4').strip()
        fallback_model = str(config.get('fallback_model') or '').strip()
        allow_upgrade = bool(config.get('allow_upgrade_model', False))
        primary_effort = str(config.get('reasoning_effort') or 'medium').strip()
        retry_effort = str(config.get('retry_reasoning_effort') or primary_effort or 'medium').strip()
        max_tokens = max(int(config.get('max_output_tokens', 560) or 560), 480)
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
                    'max_tokens': max_tokens + 120,
                    'compact_prompt': False,
                }
            )
            if allow_upgrade and upgrade_model and upgrade_model != primary_model:
                attempts.append(
                    {
                        'model': upgrade_model,
                        'structured': False,
                        'effort': retry_effort,
                        'max_tokens': max_tokens + 120,
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
        for _, attempt in enumerate(attempts):
            selected_attempt = dict(attempt)
            selected_model = str(attempt.get('model') or primary_model)
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
            resp = requests.post(base_url, headers=headers, json=request_body, timeout=timeout_sec)
            resp.raise_for_status()
            body = resp.json()
            usage_i = _response_usage(body)
            total_input_tokens += int(usage_i.get('input_tokens', 0) or 0)
            total_output_tokens += int(usage_i.get('output_tokens', 0) or 0)
            total_cached_input_tokens += int(usage_i.get('input_cached_tokens', usage_i.get('cached_input_tokens', 0)) or 0)
            actual_calls += 1
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
        if not raw_json:
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
            'last_status': 'consulted',
            'last_response_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'last_cost_twd': round(est_cost_twd, 4),
            'last_attempt': dict(selected_attempt),
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
        state['last_error'] = ''
        state['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        _append_recent(
            state,
            _build_recent_item(
                candidate,
                status='consulted',
                action=str(decision.get('action') or ('enter' if decision.get('should_trade') else 'skip')),
                detail='OpenAI 已回傳可解析的結構化交易判斷。',
                decision=decision,
                model=selected_model,
            ),
        )
        save_trade_state(state_path, state)
        return state, {
            'status': 'consulted',
            'decision': decision,
            'payload_hash': payload_hash,
            'symbol_state': symbol_state,
            'usage': usage,
            'estimated_cost_twd': round(est_cost_twd, 4),
            'estimated_cost_usd': round(est_cost_usd, 6),
            'raw_text': raw_text[:1200],
            'model': selected_model,
            'attempt': dict(selected_attempt),
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
