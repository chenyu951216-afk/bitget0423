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
        'max_output_tokens': max(_env_int(env_getter, 'OPENAI_TRADE_MAX_OUTPUT_TOKENS', 680), 520),
        'request_timeout_sec': max(_env_float(env_getter, 'OPENAI_TRADE_TIMEOUT_SEC', 45.0), 5.0),
        'temperature': 0.2,
        'base_url': str(env_getter('OPENAI_RESPONSES_URL', 'https://api.openai.com/v1/responses') or 'https://api.openai.com/v1/responses').strip(),
        'reasoning_effort': str(env_getter('OPENAI_TRADE_REASONING_EFFORT', 'medium') or 'medium').strip(),
        'retry_reasoning_effort': str(env_getter('OPENAI_TRADE_RETRY_REASONING_EFFORT', 'low') or 'low').strip(),
        'empty_retry_enabled': _env_bool(env_getter, 'OPENAI_TRADE_EMPTY_RETRY_ENABLE', False),
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


def _stable_payload(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'symbol': candidate.get('symbol'),
        'side': candidate.get('side'),
        'candidate_source': candidate.get('candidate_source'),
        'scanner_intent': candidate.get('scanner_intent'),
        'current_price': candidate.get('current_price'),
        'score': candidate.get('score'),
        'priority_score': candidate.get('priority_score'),
        'entry_quality': candidate.get('entry_quality'),
        'market_pattern': ((candidate.get('market') or {}).get('pattern')),
        'market_direction': ((candidate.get('market') or {}).get('direction')),
        'breakdown': candidate.get('breakdown') or {},
        'market_context': candidate.get('market_context') or {},
        'reference_trade_plan': candidate.get('reference_trade_plan') or {},
        'short_gainer_context': candidate.get('short_gainer_context') or {},
        'force_recheck': bool(candidate.get('force_recheck', False)),
        'execution_policy': candidate.get('execution_policy') or {},
        'constraints': candidate.get('constraints') or {},
    }


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
    signal_side = str(signal.get('side') or signal.get('direction') or ('long' if float(signal.get('score', 0) or 0) >= 0 else 'short')).lower()
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
    compact_market_context['signal_context']['scanner_side'] = 'long' if float(signal.get('score', 0) or 0) >= 0 else 'short'
    compact_reference = _compact_mapping(
        dict(signal.get('external_reference') or signal.get('reference_context') or signal.get('scanner_reference') or {}),
        ['summary', 'bias', 'setup', 'risk', 'note', 'checklist', 'confirmations', 'invalidations', 'source'],
        text_limit=180,
    )
    compact_reference_trade_plan = _compact_mapping(
        dict(signal.get('reference_trade_plan') or market_context.get('reference_trade_plan') or {}),
        ['machine_entry_hint', 'machine_stop_loss_hint', 'machine_take_profit_hint', 'machine_rr_hint', 'machine_est_pnl_pct_hint', 'note'],
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


def _compact_timeframe_bars(timeframe_bars: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for tf, rows in list(dict(timeframe_bars or {}).items())[:6]:
        tf_key = str(tf)
        row_limit = {
            '1m': 8,
            '5m': 10,
            '15m': 12,
            '1h': 10,
            '4h': 8,
            '1d': 6,
        }.get(tf_key, 8)
        raw_rows = list(rows or [])[-row_limit:]
        seq = []
        for row in raw_rows:
            if isinstance(row, dict):
                seq.append([
                    row.get('time', 0),
                    row.get('open', 0),
                    row.get('high', 0),
                    row.get('low', 0),
                    row.get('close', 0),
                    row.get('volume', 0),
                ])
            else:
                seq.append(list(row or []))
        if not seq:
            continue
        first_ts = 0
        interval_ms = 0
        compact_rows = []
        try:
            first_ts = int(seq[0][0] or 0)
        except Exception:
            first_ts = 0
        if len(seq) >= 2:
            try:
                interval_ms = int(seq[1][0] or 0) - int(seq[0][0] or 0)
            except Exception:
                interval_ms = 0
        for row in seq:
            if len(row) < 6:
                continue
            try:
                compact_rows.append([
                    _compact_number(row[1]),
                    _compact_number(row[2]),
                    _compact_number(row[3]),
                    _compact_number(row[4]),
                    _compact_number(row[5]),
                ])
            except Exception:
                continue
        if compact_rows:
            out[str(tf)] = {
                'start_ts': first_ts,
                'interval_ms': interval_ms,
                'rows': compact_rows,
            }
    return out


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
            'trend_state',
            'timing_state',
            'trade_side',
            'breakout_assessment',
            'rr_ratio',
            'scale_in_recommended',
            'scale_in_price',
            'scale_in_qty_pct',
            'scale_in_condition',
            'scale_in_note',
            'order_type',
            'entry_price',
            'stop_loss',
            'take_profit',
            'market_read',
            'entry_plan',
            'leverage',
            'margin_pct',
            'confidence',
            'thesis',
            'reason_to_skip',
            'risk_notes',
            'aggressive_note',
            'watch_trigger_type',
            'watch_trigger_price',
            'watch_invalidation_price',
            'watch_note',
            'recheck_reason',
        ],
    }


def _response_shape_hint() -> str:
    return (
        'Return exactly one complete JSON object that satisfies the schema. '
        'Use raw JSON numbers for numeric fields, concise strings, and short arrays. '
        'The only market data is the candidate payload JSON below.'
    )


def _constraint_brief(candidate: Dict[str, Any], constraints: Dict[str, Any]) -> str:
    trade_style = str(candidate.get('trade_style') or constraints.get('trade_style') or 'short_term_intraday')
    return (
        f'style={trade_style}; '
        f'leverage=symbol_max_{int(constraints.get("fixed_leverage", constraints.get("max_leverage", 25)) or 25)}x; '
        f'margin_pct={float(constraints.get("min_margin_pct", 0.03) or 0.03):.4f}-{float(constraints.get("max_margin_pct", 0.08) or 0.08):.4f}; '
        f'notional={float(constraints.get("fixed_order_notional_usdt", 20.0) or 20.0):.4f} USDT; '
        f'min_margin={float(constraints.get("min_order_margin_usdt", 0.1) or 0.1):.4f} USDT'
    )


def _build_messages(candidate: Dict[str, Any], *, compact: bool = False) -> list[Dict[str, Any]]:
    constraints = dict(candidate.get('constraints') or {})
    min_margin_pct = float(constraints.get('min_margin_pct', 0.03) or 0.03)
    max_margin_pct = float(constraints.get('max_margin_pct', 0.08) or 0.08)
    min_leverage = int(constraints.get('min_leverage', 4) or 4)
    max_leverage = int(constraints.get('max_leverage', 25) or 25)
    fixed_leverage = int(constraints.get('fixed_leverage', max_leverage) or max_leverage)
    min_order_margin_usdt = float(constraints.get('min_order_margin_usdt', 0.1) or 0.1)
    fixed_order_notional_usdt = float(constraints.get('fixed_order_notional_usdt', 20.0) or 20.0)
    trade_style = str(candidate.get('trade_style') or constraints.get('trade_style') or 'short_term_intraday')
    schema_hint = _response_shape_hint()
    constraint_brief = _constraint_brief(candidate, constraints)
    if compact:
        system_text = (
            'You are an internal crypto perpetual futures execution engine. '
            'Use only the supplied payload. Return exactly one valid JSON object, no markdown, no commentary. '
            'Numeric fields must stay numeric. Human-readable fields must be Traditional Chinese. '
            'Be aggressive but disciplined: when trend/structure, invalidation, liquidity, and RR are acceptable, prefer an executable enter or precise limit plan instead of defaulting to observe. '
            'Use observe only when one clearly testable execution condition is still missing. '
            'JSON shape reminder: ' + schema_hint
        )
        user_text = (
            f'Generate one {trade_style} tactical execution-analysis object.\n'
            f'Hard bounds: {constraint_brief}.\n'
            'Use 1D/4H bias, 1H trend quality, 15m execution, 5m confirmation, 1m micro-timing.\n'
            'Compute entry_price, stop_loss, take_profit, and RR yourself; machine hints are low-trust.\n'
            'Enums: trend_state={trending_up,trending_down,range_mixed,transitioning,trend_unclear}; timing_state={enter_now,wait_pullback,wait_breakout,avoid_near_term}.\n'
            'Strict map: enter_now=>enter+should_trade=true; wait_pullback/wait_breakout=>observe+should_trade=false; avoid_near_term=>skip+should_trade=false.\n'
            'If order_type=limit, fill limit_cancel_*; if market, limit_cancel_* must be 0/empty.\n'
            'If observe, choose exactly one primary watch path only, no OR, and fill watch_trigger_candle, watch_retest_rule, watch_volume_ratio_min, watch_micro_vwap_rule, watch_micro_ema20_rule.\n'
            'Do not be over-conservative. If current price is already near the trigger and invalidation is clear, prefer enter_now or a precise limit plan over vague observation.\n'
            'If score/entry_quality are strong and no hard blocker exists, do not reject only because price is slightly stretched or portfolio is somewhat crowded.\n'
            'reason_to_skip must cite concrete blockers from the payload. market_read and entry_plan must be explicit.\n'
            'If force_recheck=true, either upgrade to an executable plan or state the one missing factor.\n'
            'Compact payload legend: multi_timeframe uses c=close a=atr_pct r=rsi x=adx e20/e50/e200 m20=ma20 v=vwap t=trend mh=macd_hist bbp=bb_position_pct vr=vol_ratio hi/lo=recent structure; multi_timeframe_pressure uses sb=structure_bias ts=trend_stack rb=recent_break pp/sp=pressure/support pa/sa=distance_atr c20/c50=close_vs_ema hh/hl/lh/ll=structure counts.\n'
            'timeframe_bars use start_ts + interval_ms with rows=[open,high,low,close,volume].\n'
            'Candidate payload JSON:\n'
            + json.dumps(candidate, ensure_ascii=False, separators=(',', ':'))
        )
    else:
        system_text = (
            'You are producing a tactical execution-analysis object for an internal short-term crypto perpetual futures engine. '
            'Use every field in the candidate payload, especially multi-timeframe market context, latest closed candles, '
            'momentum, volatility, volume expansion, execution context, liquidity_context, derivatives_context, '
            'basic_market_data, timeframe_bars, risk state, and reference analysis. '
            'Return one complete JSON object only, never chain-of-thought or commentary. '
            'Use numeric evidence first: structure alignment, entry quality, your own RR, liquidity/spread, volatility, '
            'portfolio exposure, invalidation distance, funding/OI stress, crowding, basis, and aggressive-flow imbalance. '
            'Do not invent unseen data. If inputs conflict, prioritize latest closed candles, multi-timeframe pressure, execution quality, '
            'invalidation distance, and nearest real structure. '
            'If some inputs are noisy, lower confidence or keep watching instead of forcing a trade. '
            'Be aggressive but not reckless: act when structure, invalidation, liquidity, and confirmation align; stand down when the setup is messy. '
            'When the setup is reasonably clean and reward-to-risk is acceptable, prefer a concrete executable plan over a generic observe-only answer. '
            'You are allowed to approve an aggressive tactical entry when trend, invalidation, and liquidity are acceptable; do not default to over-cautious rejection. '
            'Use observation mode only when a key execution condition is still clearly missing. '
            'All human-readable explanation fields must be written in Traditional Chinese. Keep only enums, symbols, and timeframe tokens in English. '
            'JSON shape reminder: ' + schema_hint
        )
        user_text = (
            f'Generate one {trade_style} tactical execution-analysis object.\n'
            f'Hard bounds: {constraint_brief}; order_type must be market or limit.\n'
            'Compute entry_price, stop_loss, take_profit, and RR yourself from structure/current price/liquidity/invalidation; machine price/RR hints are low-trust only.\n'
            'Classify trend_state and timing_state explicitly. Action mapping is strict: enter_now=>enter; wait_pullback/wait_breakout=>observe; avoid_near_term=>skip.\n'
            'Set trade_side and breakout_assessment explicitly. If scale-in is not recommended, return false and 0/empty fields.\n'
            'All numeric fields must be raw JSON numbers, and the reply must be one complete JSON object only.\n'
            'Analyze 1D/4H for larger bias, 1H for trend quality, 15m for the main entry frame, 5m for confirmation, and 1m only for micro-timing.\n'
            'timeframe_bars format: each timeframe uses start_ts + interval_ms, and rows are [open,high,low,close,volume] from oldest to newest.\n'
            'market_read and entry_plan must be explicit. reason_to_skip must never be generic and must cite concrete blockers from the payload.\n'
            'Choose one best executable path now: market, precise limit pullback/retest, or no trade yet with one precise recheck trigger. If structure, invalidation, and confirmation already align, do not hide behind vague caution.\n'
            'If order_type is limit, return limit_cancel_price, limit_cancel_timeframe, limit_cancel_condition, and limit_cancel_note; for market, set those fields to 0/empty.\n'
            'If should_trade is false, preserve the idea with one narrow watch plan and machine-readable tracking fields the bot can track exactly.\n'
            'If force_recheck is true, either upgrade to an executable plan or explain the one missing factor still preventing execution.\n'
            'Use Traditional Chinese for all human-readable analysis fields.\n'
            'Compact payload legend: multi_timeframe uses c=close a=atr_pct r=rsi x=adx e20/e50/e200 m20=ma20 v=vwap t=trend mh=macd_hist bbp=bb_position_pct vr=vol_ratio hi/lo=recent structure; multi_timeframe_pressure uses sb=structure_bias ts=trend_stack rb=recent_break pp/sp=pressure/support pa/sa=distance_atr c20/c50=close_vs_ema hh/hl/lh/ll=structure counts.\n'
            'Candidate payload JSON:\n'
            + json.dumps(candidate, ensure_ascii=False, separators=(',', ':'))
        )
    return [
        {'role': 'system', 'content': [{'type': 'input_text', 'text': system_text}]},
        {'role': 'user', 'content': [{'type': 'input_text', 'text': user_text}]},
    ]


def _fallback_trade_decision(candidate: Dict[str, Any], *, reason: str = '') -> Dict[str, Any]:
    candidate = dict(candidate or {})
    side = str(candidate.get('side') or 'long').lower()
    constraints = dict(candidate.get('constraints') or {})
    reference_trade_plan = dict(candidate.get('reference_trade_plan') or {})
    market_context = dict(candidate.get('market_context') or {})
    pressure_summary = dict(market_context.get('multi_timeframe_pressure_summary') or {})
    ref_context = dict(candidate.get('reference_context') or {})
    current_price = _coerce_float(candidate.get('current_price', reference_trade_plan.get('machine_entry_hint', 0)), 0.0)
    if current_price <= 0:
        current_price = _coerce_float((((market_context.get('market_state') or {}).get('ticker') or {}).get('last', 0)), 0.0)
    entry_price = current_price if current_price > 0 else _coerce_float(reference_trade_plan.get('machine_entry_hint', 0), 0.0)
    stop_loss = _coerce_float(reference_trade_plan.get('machine_stop_loss_hint', 0), 0.0)
    take_profit = _coerce_float(reference_trade_plan.get('machine_take_profit_hint', 0), 0.0)
    nearest_blocking_price = _coerce_float(pressure_summary.get('nearest_blocking_price', 0), 0.0)
    nearest_backing_price = _coerce_float(pressure_summary.get('nearest_backing_price', 0), 0.0)
    watch_timeframe = str(pressure_summary.get('nearest_blocking_timeframe') or pressure_summary.get('nearest_backing_timeframe') or '15m').strip() or '15m'
    candidate_source = str(candidate.get('candidate_source') or 'normal')
    if side == 'short':
        if stop_loss <= entry_price:
            stop_loss = nearest_backing_price if nearest_backing_price > entry_price else (entry_price * 1.02 if entry_price > 0 else 0.0)
        if take_profit <= 0 or take_profit >= entry_price:
            take_profit = nearest_blocking_price if 0 < nearest_blocking_price < entry_price else (entry_price * 0.97 if entry_price > 0 else 0.0)
        watch_trigger_type = 'breakdown_confirm' if candidate_source == 'short_gainers' else 'pullback_to_entry'
        watch_trigger_price = nearest_blocking_price if nearest_blocking_price > 0 else (entry_price * 0.992 if entry_price > 0 else 0.0)
        watch_invalidation_price = nearest_backing_price if nearest_backing_price > 0 else stop_loss
        zone_low = min(watch_trigger_price, entry_price) if watch_trigger_price > 0 and entry_price > 0 else max(watch_trigger_price, 0.0)
        zone_high = max(watch_trigger_price, entry_price) if watch_trigger_price > 0 and entry_price > 0 else max(entry_price, 0.0)
        watch_note = '先等空方主觸發成立，再重新送 AI；目前不要預判追空。'
        structure_condition = '需要出現跌破或跌破後回抽站不上，且短線結構轉弱。'
        volume_condition = '偏好跌破時成交量放大、回抽時買量轉弱。'
        confirmations = [
            '觀察週期收在主觸發價下方且無法收回',
            '回抽主觸發價失敗或形成更低高點',
            '跌破或回抽失敗時賣量放大',
        ]
        invalidations = [
            '價格快速收回失效價上方且站穩',
            '觀察週期重新延伸出更高高點',
            '跌破失敗且賣方跟進不足',
        ]
        reason_to_skip = '目前先不追空，因為還沒有明確跌破或跌破後回抽失敗。'
        thesis = '只有在多方延續明確失效後，這筆逆勢放空才具備執行價值。'
        aggressive_note = '若跌破延續且回抽持續站不上，才考慮激進追空。'
    else:
        if stop_loss >= entry_price or stop_loss <= 0:
            stop_loss = nearest_backing_price if 0 < nearest_backing_price < entry_price else (entry_price * 0.985 if entry_price > 0 else 0.0)
        if take_profit <= entry_price:
            take_profit = nearest_blocking_price if nearest_blocking_price > entry_price else (entry_price * 1.03 if entry_price > 0 else 0.0)
        watch_trigger_type = 'pullback_to_entry'
        watch_trigger_price = nearest_backing_price if nearest_backing_price > 0 else entry_price
        watch_invalidation_price = stop_loss
        zone_low = min(watch_trigger_price, entry_price) if watch_trigger_price > 0 and entry_price > 0 else max(watch_trigger_price, 0.0)
        zone_high = max(watch_trigger_price, entry_price) if watch_trigger_price > 0 and entry_price > 0 else max(entry_price, 0.0)
        watch_note = '先等回踩承接或重新站回關鍵區，再重新送 AI；目前不要追價。'
        structure_condition = '需要回踩守住、重新站回，或延續結構再確認。'
        volume_condition = '偏好回踩量縮、重新上攻時買量回升。'
        confirmations = [
            '回踩守住觀察區且有買盤承接',
            '重新站回關鍵區且延續穩定',
            '反彈量能優於回踩量能',
        ]
        invalidations = [
            '價格跌破觀察區且收在失效價外',
            '反彈無法站回觀察區',
            '動能走弱且持續出現更低高點',
        ]
        reason_to_skip = '目前先不追多，因為回踩承接或重新站回關鍵區還未確認。'
        thesis = '趨勢可能仍然有效，但進場點要等結構更乾淨。'
        aggressive_note = '只有重新站回買盤區且量能改善時，才考慮激進追多。'
    return {
        'should_trade': False,
        'action': 'observe',
        'trend_state': 'trending_down' if side == 'short' else 'trending_up',
        'timing_state': 'wait_pullback' if watch_trigger_type == 'pullback_to_entry' else 'wait_breakout',
        'trade_side': side,
        'breakout_assessment': '等待確認，暫時保留觀察計畫。',
        'rr_ratio': round(abs((take_profit - entry_price) / max(abs(entry_price - stop_loss), 1e-9)), 4) if entry_price > 0 and stop_loss > 0 and take_profit > 0 else 0.0,
        'scale_in_recommended': False,
        'scale_in_price': 0.0,
        'scale_in_qty_pct': 0.0,
        'scale_in_condition': '',
        'scale_in_note': '',
        'order_type': 'limit',
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'market_read': str(ref_context.get('summary') or 'OpenAI 空回覆，因此先保留結構化觀察計畫。')[:280],
        'entry_plan': '現在先不要進場，先維持觀察，等觸發條件在指定週期明確出現後再重新送 AI。',
        'entry_reason': '這是根據同一份 payload 生成的保守觀察方案，先等結構更明確再執行。',
        'stop_loss_reason': '失效價放在最近有意義的結構外，避免一般噪音過早洗掉。',
        'take_profit_plan': '先以最近合理結構目標作為第一止盈參考，確認成功後再處理剩餘倉位。',
        'if_missed_plan': '如果價格沒有照觸發條件走，不追價，等待下一次回測或重新失效訊號。',
        'reference_summary': str(reason or 'OpenAI 沒有回可解析 JSON，因此機器人用同一份 payload 保留保守觀察計畫。')[:220],
        'chase_if_triggered': False,
        'chase_trigger_price': 0.0,
        'chase_limit_price': 0.0,
        'trail_trigger_atr_hint': 0.0,
        'trail_pct_hint': 0.0,
        'breakeven_atr_hint': 0.0,
        'dynamic_take_profit_hint': 0.0,
        'leverage': int(constraints.get('fixed_leverage', constraints.get('min_leverage', 20)) or 20),
        'margin_pct': _clamp(_coerce_float(constraints.get('min_margin_pct', 0.03), 0.03), _coerce_float(constraints.get('min_margin_pct', 0.03), 0.03), _coerce_float(constraints.get('max_margin_pct', 0.08), 0.08)),
        'confidence': 24.0,
        'thesis': thesis,
        'reason_to_skip': reason_to_skip,
        'risk_notes': [
            '因為 OpenAI 空回覆，所以先套用保守觀察方案。',
            '在主觸發沒有明確成立前，不要把它轉成實際交易。',
            '等觸發條件出現後，再用同一份 payload 重新送審。',
        ],
        'aggressive_note': aggressive_note,
        'watch_trigger_type': watch_trigger_type,
        'watch_trigger_price': watch_trigger_price,
        'watch_invalidation_price': watch_invalidation_price,
        'watch_note': watch_note,
        'recheck_reason': '只有當主觸發條件清楚成立後，才重新送 AI。',
        'watch_timeframe': watch_timeframe,
        'watch_price_zone_low': zone_low,
        'watch_price_zone_high': zone_high,
        'watch_structure_condition': structure_condition,
        'watch_volume_condition': volume_condition,
        'watch_trigger_candle': 'close_below' if side == 'short' else 'close_above',
        'watch_retest_rule': 'hold_below' if side == 'short' else 'hold_above',
        'watch_volume_ratio_min': 1.05,
        'watch_micro_vwap_rule': 'below' if side == 'short' else 'above',
        'watch_micro_ema20_rule': 'below' if side == 'short' else 'above',
        'watch_checklist': [
            '確認觀察週期收盤是否站上或跌破主觸發價',
            '確認回測是否符合指定的站穩或站不上規則',
            '確認量能不是瞬間衝一下就衰退',
        ],
        'watch_confirmations': confirmations,
        'watch_invalidations': invalidations,
        'watch_recheck_priority': 72.0,
        'limit_cancel_price': 0.0,
        'limit_cancel_timeframe': '',
        'limit_cancel_condition': '',
        'limit_cancel_note': '',
    }


def _build_request_body(
    candidate: Dict[str, Any],
    config: Dict[str, Any],
    *,
    structured: bool = True,
    model: str | None = None,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
    compact_prompt: bool = False,
) -> Dict[str, Any]:
    effective_max_output_tokens = max(
        int(max_output_tokens or config.get('max_output_tokens', 680) or 680),
        640 if structured else 320,
    )
    body = {
        'model': str(model or config.get('model') or 'gpt-5.4-mini'),
        'input': _build_messages(candidate, compact=compact_prompt),
        'max_output_tokens': effective_max_output_tokens,
    }
    effort = str(reasoning_effort if reasoning_effort is not None else config.get('reasoning_effort') or '').strip()
    if effort:
        body['reasoning'] = {'effort': effort}
    if structured:
        body['text'] = {
            'format': {
                'type': 'json_schema',
                'name': 'trade_decision',
                'schema': _json_schema(),
                'strict': True,
            }
        }
    else:
        body['text'] = {
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


def _normalize_decision(raw: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    raw = dict(raw or {})
    side = str(candidate.get('side') or 'long').lower()
    constraints = dict(candidate.get('constraints') or {})
    reference_trade_plan = dict(candidate.get('reference_trade_plan') or {})
    watch_plan = dict(raw.get('watch_plan') or {})
    entry_default = _coerce_float(candidate.get('current_price', candidate.get('entry_price', 0)), 0.0)
    stop_default = _coerce_float(reference_trade_plan.get('machine_stop_loss_hint', candidate.get('stop_loss', 0)), 0.0)
    tp_default = _coerce_float(reference_trade_plan.get('machine_take_profit_hint', candidate.get('take_profit', 0)), 0.0)
    if entry_default <= 0:
        ticker_ref = (((candidate.get('market_context') or {}).get('market_state') or {}).get('ticker') or {}).get('last', 0)
        entry_default = _coerce_float(ticker_ref, 0.0)
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
    if decision['trade_side'] not in {'long', 'short'}:
        decision['trade_side'] = side
    if constraints.get('fixed_leverage'):
        decision['leverage'] = int(constraints.get('fixed_leverage') or decision['leverage'])
    decision['market_read'] = _cn_phrase(decision['market_read'])
    decision['entry_plan'] = _cn_phrase(decision['entry_plan'])
    decision['watch_note'] = _single_watch_path(decision['watch_note'])
    decision['recheck_reason'] = _single_watch_path(decision['recheck_reason'])
    decision['breakout_assessment'] = decision['breakout_assessment'][:180]
    decision['scale_in_condition'] = decision['scale_in_condition'][:220]
    decision['scale_in_note'] = decision['scale_in_note'][:220]
    decision['market_read'] = decision['market_read'][:280]
    decision['entry_plan'] = decision['entry_plan'][:280]
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
        decision['watch_trigger_type'] = 'pullback_to_entry' if decision['timing_state'] == 'wait_pullback' else 'breakout_reclaim'
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
    score_abs = abs(float(candidate.get('score', 0) or 0))
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

    if not force_recheck and score_abs < float(config.get('min_score_abs', 43.0) or 43.0):
        result = {'status': 'below_min_score', 'decision': None, 'payload_hash': payload_hash, 'symbol_state': symbol_state}
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
    if last_status.startswith('empty_response'):
        last_sent_ts = 0.0
        if not cached_decision:
            last_hash = ''
    same_payload_reuse_sec = max(int(config.get('same_payload_reuse_minutes', 180) or 180), 1) * 60
    top_signature = _top_candidates_signature(candidate)

    if (
        not force_recheck
        and not last_status.startswith('empty_response')
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

    if not force_recheck and not last_status.startswith('empty_response') and last_sent_ts > 0 and (now_ts - last_sent_ts) < cooldown_sec:
        next_allowed_ts = last_sent_ts + cooldown_sec
        detail = 'Symbol cooldown is active until {}.'.format(datetime.fromtimestamp(next_allowed_ts).strftime('%Y-%m-%d %H:%M:%S'))
        if cached_decision and last_hash == payload_hash:
            detail += ' Same symbol payload was already sent in the current cooldown window.'
        result = {
            'status': 'cooldown_active',
            'decision': None,
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
        result = {
            'status': 'global_interval_active',
            'decision': None,
            'payload_hash': payload_hash,
            'symbol_state': symbol_state,
            'next_allowed_ts': next_allowed_ts,
        }
        save_trade_state(state_path, state)
        return state, result

    est_call_cost_twd = estimate_cost_twd(config, input_tokens=2200, output_tokens=500)
    soft_budget_twd = float(config.get('soft_budget_twd', 850.0) or 850.0)
    if spent_twd >= soft_budget_twd and score_abs < max(float(config.get('min_score_abs', 43.0) or 43.0) + 8.0, 52.0):
        result = {'status': 'budget_paused', 'decision': None, 'payload_hash': payload_hash, 'symbol_state': symbol_state}
        _append_recent(state, _build_recent_item(candidate, status='budget_paused', detail='Soft budget reached, only very strong signals are allowed to query OpenAI.'))
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
        logger('OpenAI trade decision request: {} rank={} score={:.2f}'.format(symbol, rank, score_abs))

    try:
        base_url = str(config.get('base_url') or 'https://api.openai.com/v1/responses')
        timeout_sec = float(config.get('request_timeout_sec', 60.0) or 60.0)
        primary_model = str(config.get('model') or 'gpt-5.4').strip()
        upgrade_model = str(config.get('upgrade_model') or 'gpt-5.4').strip()
        fallback_model = str(config.get('fallback_model') or '').strip()
        allow_upgrade = bool(config.get('allow_upgrade_model', False))
        primary_effort = str(config.get('reasoning_effort') or 'medium').strip()
        retry_effort = str(config.get('retry_reasoning_effort') or primary_effort or 'medium').strip()
        max_tokens = max(int(config.get('max_output_tokens', 680) or 680), 640)
        attempts = [
            {
                'model': primary_model,
                'structured': True,
                'effort': primary_effort,
                'max_tokens': max_tokens,
                'compact_prompt': True,
            },
        ]

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
        'min_score_abs': float(config.get('min_score_abs', 43.0) or 43.0),
        'last_error': str(state.get('last_error') or ''),
        'updated_at': str(state.get('updated_at') or ''),
        'recent_decisions': recent_rows,
    }
