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
        'same_payload_reuse_minutes': max(_env_int(env_getter, 'OPENAI_TRADE_SAME_PAYLOAD_REUSE_MINUTES', 45), 1),
        'global_min_interval_minutes': max(_env_int(env_getter, 'OPENAI_TRADE_GLOBAL_MIN_INTERVAL_MINUTES', 0), 0),
        'min_score_abs': max(_env_float(env_getter, 'OPENAI_TRADE_MIN_SCORE', 43.0), 0.0),
        'min_margin_pct': min(max(_env_float(env_getter, 'OPENAI_TRADE_MIN_MARGIN_PCT', 0.03), 0.005), 0.5),
        'max_margin_pct': min(max(_env_float(env_getter, 'OPENAI_TRADE_MAX_MARGIN_PCT', 0.08), 0.01), 0.8),
        'min_leverage': max(_env_int(env_getter, 'OPENAI_TRADE_MIN_LEVERAGE', 4), 1),
        'max_leverage': max(_env_int(env_getter, 'OPENAI_TRADE_MAX_LEVERAGE', 25), 1),
        'max_output_tokens': max(_env_int(env_getter, 'OPENAI_TRADE_MAX_OUTPUT_TOKENS', 1280), 800),
        'request_timeout_sec': max(_env_float(env_getter, 'OPENAI_TRADE_TIMEOUT_SEC', 45.0), 5.0),
        'temperature': 0.2,
        'base_url': str(env_getter('OPENAI_RESPONSES_URL', 'https://api.openai.com/v1/responses') or 'https://api.openai.com/v1/responses').strip(),
        'reasoning_effort': str(env_getter('OPENAI_TRADE_REASONING_EFFORT', 'medium') or 'medium').strip(),
        'retry_reasoning_effort': str(env_getter('OPENAI_TRADE_RETRY_REASONING_EFFORT', 'low') or 'low').strip(),
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


def _compact_mapping(data: Dict[str, Any], keys: list[str], *, text_limit: int = 160) -> Dict[str, Any]:
    src = dict(data or {})
    out: Dict[str, Any] = {}
    for key in keys:
        if key not in src:
            continue
        value = src.get(key)
        if isinstance(value, (int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, list):
            out[key] = [_short_text(x, text_limit) for x in value[:5]]
        elif isinstance(value, dict):
            out[key] = {str(k): _short_text(v, text_limit) for k, v in list(value.items())[:10]}
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
    compact_breakdown = _compact_mapping(
        breakdown,
        [
            'Setup', 'Regime', 'RegimeConf', 'RegimeConfidence', 'RegimeDir', 'RegimeBias',
            'MarketState', 'MarketStateConf', 'MarketTempo', 'TrendConfidence', '方向信心',
            'RR', 'EntryGate', '進場品質', 'ChaseRisk', '追價風險', 'VolRatio',
            'PreBreakoutScore', 'PreBreakoutDirection', 'PreBreakoutPhase',
            'AIScoreCoverage', 'AISampleCount', 'LearnEdge', 'SignalQuality',
            'VWAPDistanceATR', 'EMA20DistanceATR', 'SRDistanceATR',
        ],
        text_limit=140,
    )
    compact_market_context = {
        'style': _compact_mapping(style, ['holding_period', 'trade_goal', 'decision_priority'], text_limit=120),
        'signal_context': _compact_mapping(dict(market_context.get('signal_context') or {}), ['side', 'score', 'raw_score', 'priority_score', 'entry_quality', 'current_price', 'setup_label', 'signal_grade', 'regime', 'regime_confidence', 'trend_confidence', 'rotation_adj', 'score_jump', 'atr_15m', 'atr_4h'], text_limit=80),
        'latest_closed_candle': _compact_mapping(dict(market_context.get('latest_closed_candle') or {}), ['direction', 'shape', 'body_pct', 'upper_wick_pct', 'lower_wick_pct', 'range_pct_of_price', 'close_position_pct'], text_limit=80),
        'momentum': _compact_mapping(dict(market_context.get('momentum') or {}), ['long_score', 'short_score', 'signals', 'trend_4h_up', 'trend_1d_up', 'higher_lows', 'lower_highs', 'volume_build', 'compression'], text_limit=120),
        'levels': _compact_mapping(dict(market_context.get('levels') or {}), ['dist_high_atr', 'dist_low_atr', 'nearest_support', 'nearest_resistance', 'support_levels', 'resistance_levels', 'recent_high', 'recent_low'], text_limit=120),
        'market_state': {
            'ticker': _compact_mapping(dict(((market_context.get('market_state') or {}).get('ticker') or {})), ['last', 'bid', 'ask', 'spread_pct', 'quote_volume', 'percentage_24h', 'change_24h', 'high_24h', 'low_24h', 'mark_price', 'index_price'], text_limit=120),
            'support_resistance': _compact_mapping(dict(((market_context.get('market_state') or {}).get('support_resistance') or {})), ['support', 'resistance', 'distance_to_support_pct', 'distance_to_resistance_pct'], text_limit=80),
        },
        'basic_market_data': _compact_mapping(dict(market_context.get('basic_market_data') or {}), ['symbol', 'exchange', 'market_type', 'current_price', 'change_24h_pct', 'quote_volume_24h', 'base_volume_24h', 'market_cap_usd', 'fdv_usd', 'circulating_supply', 'total_supply', 'funding_rate', 'open_interest', 'open_interest_value_usdt', 'long_short_ratio', 'top_trader_long_short_ratio', 'whale_position_change_pct'], text_limit=140),
        'liquidity_context': _compact_mapping(dict(market_context.get('liquidity_context') or {}), ['spread_pct', 'bid_depth_5', 'ask_depth_5', 'bid_depth_10', 'ask_depth_10', 'depth_imbalance_10', 'largest_bid_wall_price', 'largest_bid_wall_size', 'largest_ask_wall_price', 'largest_ask_wall_size', 'recent_trades_count', 'aggressive_buy_volume', 'aggressive_sell_volume', 'aggressive_buy_notional', 'aggressive_sell_notional', 'buy_sell_notional_ratio', 'cvd_notional', 'cvd_bias', 'volume_anomaly_5m', 'volume_anomaly_15m', 'errors'], text_limit=160),
        'derivatives_context': _compact_mapping(dict(market_context.get('derivatives_context') or {}), ['funding_rate', 'next_funding_time', 'open_interest', 'open_interest_value_usdt', 'open_interest_change_pct_5m', 'long_short_ratio', 'top_trader_long_short_ratio', 'whale_position_change_pct', 'basis_pct', 'mark_price', 'index_price', 'liquidation_volume_24h', 'liquidation_map_status', 'leverage_heat', 'leverage_heat_score', 'errors'], text_limit=160),
        'news_context': dict(market_context.get('news_context') or {}),
        'multi_timeframe': {
            str(tf): _compact_mapping(
                dict(row or {}),
                ['bars', 'last_close', 'prev_close', 'atr', 'atr_pct', 'rsi', 'adx', 'plus_di', 'minus_di', 'ema9', 'ema20', 'ema50', 'ema100', 'ema200', 'ma20', 'ma60', 'vwap', 'trend_label', 'macd', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_mid', 'bb_lower', 'bb_width_pct', 'bb_position_pct', 'stoch_k', 'stoch_d', 'kdj_j', 'ret_3bars_pct', 'ret_12bars_pct', 'ret_24bars_pct', 'vol_ratio', 'avg_volume_20', 'avg_volume_60', 'last_volume', 'swing_high_20', 'swing_low_20', 'distance_to_swing_high_pct', 'distance_to_swing_low_pct', 'support_levels', 'resistance_levels', 'recent_structure_high', 'recent_structure_low'],
                text_limit=180,
            )
            for tf, row in list(dict(market_context.get('multi_timeframe') or {}).items())[:6]
        },
        'timeframe_bars': {
            str(tf): list(rows or [])[:100]
            for tf, rows in list(dict(market_context.get('timeframe_bars') or {}).items())[:6]
        },
        'pre_breakout_radar': _compact_mapping(dict(market_context.get('pre_breakout_radar') or {}), ['ready', 'phase', 'direction', 'score', 'summary', 'note', 'signals', 'tags'], text_limit=120),
        'execution_context': _compact_mapping(dict(market_context.get('execution_context') or {}), ['spread_pct', 'mark_last_deviation_pct', 'top_depth_ratio', 'api_error_streak', 'status', 'notes'], text_limit=120),
        'multi_timeframe_pressure_summary': _compact_mapping(dict(market_context.get('multi_timeframe_pressure_summary') or {}), ['side', 'aligned_timeframes', 'opposing_timeframes', 'nearest_blocking_timeframe', 'nearest_blocking_price', 'nearest_blocking_distance_atr', 'nearest_backing_timeframe', 'nearest_backing_price', 'nearest_backing_distance_atr', 'stacked_blocking_within_1atr', 'stacked_blocking_within_2atr'], text_limit=80),
        'multi_timeframe_pressure': {
            str(tf): _compact_mapping(
                dict(row or {}),
                ['structure_bias', 'trend_stack', 'swing_bias', 'recent_break', 'pressure_price', 'support_price', 'pressure_distance_pct', 'support_distance_pct', 'pressure_distance_atr', 'support_distance_atr', 'close_vs_ema20_pct', 'close_vs_ema50_pct', 'volume_ratio', 'hh_count', 'hl_count', 'lh_count', 'll_count'],
                text_limit=80,
            )
            for tf, row in list(dict(market_context.get('multi_timeframe_pressure') or {}).items())[:4]
        },
    }
    compact_reference = _compact_mapping(
        dict(signal.get('external_reference') or signal.get('reference_context') or signal.get('scanner_reference') or {}),
        ['summary', 'bias', 'setup', 'risk', 'note', 'checklist', 'confirmations', 'invalidations', 'source'],
        text_limit=180,
    )
    compact_reference_trade_plan = _compact_mapping(
        dict(market_context.get('reference_trade_plan') or {}),
        ['machine_entry_hint', 'machine_stop_loss_hint', 'machine_take_profit_hint', 'machine_rr_hint', 'machine_est_pnl_pct_hint', 'note'],
        text_limit=180,
    )
    return {
        'symbol': str(signal.get('symbol') or ''),
        'side': 'long' if float(signal.get('score', 0) or 0) >= 0 else 'short',
        'trade_style': str(constraints.get('trade_style') or style.get('holding_period') or 'short_term_intraday'),
        'signal_desc': _short_text(signal.get('desc'), 240),
        'candidate_source': str(signal.get('candidate_source') or signal.get('source') or 'normal')[:80],
        'scanner_intent': str(signal.get('scanner_intent') or '')[:180],
        'short_gainer_context': _compact_mapping(dict(signal.get('short_gainer_context') or {}), ['pct_24h', 'rank_score', 'quote_volume', 'spread_pct', 'ticker_last'], text_limit=80),
        'rank': int(rank_index) + 1,
        'score': _round(signal.get('score'), 4),
        'raw_score': _round(signal.get('raw_score', signal.get('score')), 4),
        'priority_score': _round(signal.get('priority_score', abs(float(signal.get('score', 0) or 0))), 4),
        'current_price': _round(signal.get('price'), 8),
        'entry_quality': _round(signal.get('entry_quality', breakdown.get('EntryGate')), 4),
        'est_pnl_pct': _round(signal.get('est_pnl'), 4),
        'regime': str(signal.get('regime') or breakdown.get('Regime') or 'neutral'),
        'regime_confidence': _round(signal.get('regime_confidence'), 4),
        'setup_label': str(signal.get('setup_label') or breakdown.get('Setup') or ''),
        'signal_grade': str(signal.get('signal_grade') or breakdown.get('等級') or ''),
        'trend_confidence': _round(signal.get('trend_confidence'), 4),
        'rotation_adj': _round(signal.get('rotation_adj'), 4),
        'breakdown': compact_breakdown,
        'execution_quality': _compact_mapping(execution_quality, ['execution_score', 'score', 'label', 'spread_pct', 'depth5', 'mark_last_dev_pct', 'penalty', 'reasons'], text_limit=120),
        'market': {
            'pattern': str(market.get('pattern') or ''),
            'direction': str(market.get('direction') or ''),
            'strength': _round(market.get('strength'), 4),
            'prediction': _short_text(market.get('prediction'), 180),
        },
        'market_context': compact_market_context,
        'reference_trade_plan': compact_reference_trade_plan,
        'risk': {
            'trading_ok': bool(risk_status.get('trading_ok', True)),
            'halt_reason': str(risk_status.get('halt_reason') or '')[:180],
            'daily_loss_pct': _round(risk_status.get('daily_loss_pct'), 4),
            'consecutive_loss': int(risk_status.get('consecutive_loss', 0) or 0),
        },
        'portfolio': {
            'equity': _round(portfolio.get('equity'), 4),
            'active_position_count': int(portfolio.get('active_position_count', 0) or 0),
            'same_direction_count': int(portfolio.get('same_direction_count', 0) or 0),
            'open_symbols': list(portfolio.get('open_symbols') or [])[:8],
        },
        'execution_policy': _compact_mapping(execution_policy, ['fixed_leverage', 'leverage_mode', 'min_order_margin_usdt', 'fixed_order_notional_usdt', 'margin_pct_range'], text_limit=80),
        'reference_context': compact_reference,
        'top_candidates': list(top_candidates or [])[:3],
        'constraints': dict(constraints or {}),
        'force_recheck': bool(signal.get('force_openai_recheck', False)),
    }


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
        'order_type': {'type': 'string', 'enum': ['market', 'limit']},
        'entry_price': {'type': 'number'},
        'stop_loss': {'type': 'number'},
        'take_profit': {'type': 'number'},
        'market_read': {'type': 'string'},
        'entry_plan': {'type': 'string'},
        'entry_reason': {'type': 'string'},
        'stop_loss_reason': {'type': 'string'},
        'take_profit_plan': {'type': 'string'},
        'if_missed_plan': {'type': 'string'},
        'reference_summary': {'type': 'string'},
        'chase_if_triggered': {'type': 'boolean'},
        'chase_trigger_price': {'type': 'number'},
        'chase_limit_price': {'type': 'number'},
        'trail_trigger_atr_hint': {'type': 'number'},
        'trail_pct_hint': {'type': 'number'},
        'breakeven_atr_hint': {'type': 'number'},
        'dynamic_take_profit_hint': {'type': 'number'},
        'leverage': {'type': 'integer'},
        'margin_pct': {'type': 'number'},
        'confidence': {'type': 'number'},
        'thesis': {'type': 'string'},
        'reason_to_skip': {'type': 'string'},
        'risk_notes': {'type': 'array', 'items': {'type': 'string'}},
        'aggressive_note': {'type': 'string'},
        'watch_trigger_type': {'type': 'string', 'enum': ['none', 'pullback_to_entry', 'breakout_reclaim', 'breakdown_confirm', 'volume_confirmation', 'manual_review']},
        'watch_trigger_price': {'type': 'number'},
        'watch_invalidation_price': {'type': 'number'},
        'watch_note': {'type': 'string'},
        'recheck_reason': {'type': 'string'},
        'watch_timeframe': {'type': 'string'},
        'watch_price_zone_low': {'type': 'number'},
        'watch_price_zone_high': {'type': 'number'},
        'watch_structure_condition': {'type': 'string'},
        'watch_volume_condition': {'type': 'string'},
        'watch_checklist': {'type': 'array', 'items': {'type': 'string'}},
        'watch_confirmations': {'type': 'array', 'items': {'type': 'string'}},
        'watch_invalidations': {'type': 'array', 'items': {'type': 'string'}},
        'watch_recheck_priority': {'type': 'number'},
        'limit_cancel_price': {'type': 'number'},
        'limit_cancel_timeframe': {'type': 'string'},
        'limit_cancel_condition': {'type': 'string'},
        'limit_cancel_note': {'type': 'string'},
    }
    return {
        'type': 'object',
        'additionalProperties': False,
        'properties': properties,
        'required': list(properties.keys()),
    }


def _response_shape_hint() -> str:
    return (
        'Return one JSON object only. '
        'Required numeric fields: entry_price, stop_loss, take_profit, chase_trigger_price, '
        'chase_limit_price, trail_trigger_atr_hint, trail_pct_hint, breakeven_atr_hint, '
        'dynamic_take_profit_hint, leverage, margin_pct, confidence, watch_trigger_price, '
        'watch_invalidation_price, watch_price_zone_low, watch_price_zone_high, watch_recheck_priority, limit_cancel_price. '
        'Required text/array fields: market_read, entry_plan, entry_reason, stop_loss_reason, take_profit_plan, '
        'if_missed_plan, reference_summary, thesis, reason_to_skip, risk_notes, aggressive_note, watch_note, '
        'recheck_reason, watch_timeframe, watch_structure_condition, watch_volume_condition, '
        'limit_cancel_timeframe, limit_cancel_condition, limit_cancel_note, '
        'watch_checklist, watch_confirmations, watch_invalidations. '
        'The only real market data is the candidate payload JSON below; do not treat any schema reminder as market data.'
    )


def _constraint_brief(candidate: Dict[str, Any], constraints: Dict[str, Any]) -> str:
    trade_style = str(candidate.get('trade_style') or constraints.get('trade_style') or 'short_term_intraday')
    return (
        f'style={trade_style}; '
        f'leverage={int(constraints.get("fixed_leverage", constraints.get("max_leverage", 25)) or 25)}x fixed; '
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
            'You are a crypto perpetual futures execution planner for short-term trading. '
            'Use only the supplied payload, especially timeframe_bars, multi_timeframe, multi_timeframe_pressure, '
            'liquidity_context, derivatives_context, execution_context, risk, portfolio, and reference_context. '
            'Return one complete JSON object only, with raw JSON numbers in numeric fields and no commentary outside the object. '
            'If the setup is unclear, return should_trade=false with a narrow watch plan instead of leaving anything blank. '
            'Be decisive when structure, invalidation, liquidity, and confirmation align; if the thesis is still good but stretched, prefer a better limit/recheck rather than a vague observe-only answer. '
            'JSON shape reminder: ' + schema_hint
        )
        user_text = (
            f'Decide whether to place a {trade_style} crypto perpetual futures order.\n'
            f'Hard bounds: {constraint_brief}.\n'
            'Horizon is 30 minutes to 6 hours: use 1D/4H for bias, 1H for trend quality, 15m for entry, 5m for confirmation, 1m only for micro-timing.\n'
            'Compute entry_price, stop_loss, take_profit, and RR yourself from structure/current price/liquidity/invalidation; machine hints are low-trust only.\n'
            'If limit, also return limit_cancel_price, limit_cancel_timeframe, limit_cancel_condition, limit_cancel_note; if market, set those limit-cancel fields to 0/empty.\n'
            'If not trading, still return the full object with a precise watch trigger, invalidation, timeframe, checklist, confirmations, invalidations, and recheck_reason so the bot can wait first and only ask again after the condition is clearly met.\n'
            'If force_recheck=true, either upgrade the previous watch idea into an executable plan or explain the single missing factor. For short_gainers, never short only because price rose; require failed continuation, exhaustion, reclaim failure, VWAP/EMA loss, or breakdown confirmation.\n'
            'Candidate payload JSON:\n'
            + json.dumps(candidate, ensure_ascii=False, separators=(',', ':'))
        )
    else:
        system_text = (
            'You are a crypto perpetual futures execution planner for short-term trading. '
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
            'Use observation mode only when a key execution condition is still clearly missing. '
            'JSON shape reminder: ' + schema_hint
        )
        user_text = (
            f'Decide whether to place a {trade_style} crypto perpetual futures order.\n'
            f'Hard bounds: {constraint_brief}; order_type must be market or limit; stop_loss and take_profit are mandatory protection orders.\n'
            'Compute entry_price, stop_loss, take_profit, and RR yourself from structure/current price/liquidity/invalidation; machine price/RR hints are low-trust only.\n'
            'All numeric fields must be raw JSON numbers, and the reply must be one complete JSON object only.\n'
            'If uncertain, still return the full schema with should_trade=false and a narrow watch plan.\n'
            'Analyze 1D/4H for larger bias, 1H for trend quality, 15m for the main entry frame, 5m for confirmation, and 1m only for micro-timing; optimize for holds around 30 minutes to 6 hours.\n'
            'Use timeframe_bars, multi_timeframe, multi_timeframe_pressure, liquidity_context, derivatives_context, execution_context, and news_context together rather than relying on a single block.\n'
            'market_read must explicitly mention trend state, key support/resistance, and fake-breakout risk.\n'
            'Choose one best executable path now: market, precise limit pullback/retest, or no trade yet with a precise recheck trigger. Be slightly selective, not overly selective: if the thesis is still good but price is stretched, prefer a better limit entry or confirmation trigger instead of blanket rejection. If structure, invalidation, and confirmation already align, do not hide behind vague caution.\n'
            'If order_type is limit, also return limit_cancel_price, limit_cancel_timeframe, limit_cancel_condition, and limit_cancel_note so the bot can cancel before fill if invalidated; for market, set those fields to 0/empty.\n'
            'If should_trade is false, preserve the idea with watch_trigger_type, watch_trigger_price, watch_invalidation_price, watch_note, recheck_reason, watch_timeframe, watch_price_zone_low/high, watch_structure_condition, watch_volume_condition, watch_checklist, watch_confirmations, watch_invalidations, and watch_recheck_priority. The watch trigger must be precise enough that the bot waits first and asks again only after the condition is clearly met.\n'
            'watch_trigger_type choices: none, pullback_to_entry, breakout_reclaim, breakdown_confirm, volume_confirmation, manual_review. If no practical watch condition exists, use none and 0 trigger/invalidation prices. Confidence should be 0-100.\n'
            'If force_recheck is true, the bot is returning because the previous watch condition appears met; either upgrade to an executable plan or explain the one missing factor still preventing execution. If candidate_source is short_gainers, do not approve only because price rose; require failed continuation, exhaustion wick, lower high, VWAP/EMA loss, breakdown confirmation, or another invalidation-based trigger.\n'
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
        watch_note = 'Wait for a clean bearish trigger before shorting; do not front-run a still-constructive bounce.'
        structure_condition = 'Need breakdown / failed reclaim / lower-high confirmation before the short becomes executable.'
        volume_condition = 'Prefer expanding sell volume on the loss of support and weaker bounce volume on the retest.'
        confirmations = [
            '15m closes below the trigger and cannot reclaim it',
            'Retest stalls under the broken level or prints a lower high',
            'Sell volume expands on breakdown or on failed reclaim',
        ]
        invalidations = [
            'Price quickly reclaims and holds above the invalidation level',
            'Higher highs continue across the watch timeframe',
            'Breakdown fails immediately with weak sell participation',
        ]
        reason_to_skip = 'Fallback observation plan: short setup is not opened without confirmed breakdown or failed reclaim.'
        thesis = 'Possible fade only after structure confirms that bullish continuation has failed.'
        aggressive_note = 'A more aggressive short is acceptable only after the trigger breaks cleanly and the retest stays weak.'
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
        watch_note = 'Wait for a cleaner pullback / hold before taking a long; do not chase a stretched print.'
        structure_condition = 'Need pullback support hold, reclaim, or continuation confirmation before long entry.'
        volume_condition = 'Prefer lighter sell volume on pullback and stronger rebound volume on the reclaim.'
        confirmations = [
            'Pullback holds the watch zone and buyers defend it',
            'Price reclaims the key level with stable follow-through',
            'Rebound volume improves versus the pullback leg',
        ]
        invalidations = [
            'Price loses the watch zone and closes through invalidation',
            'Rebound cannot reclaim the trigger area',
            'Momentum decays and lower highs keep printing into support',
        ]
        reason_to_skip = 'Fallback observation plan: long setup is not opened until pullback support or reclaim is confirmed.'
        thesis = 'Trend can remain valid, but execution should wait for a cleaner structure confirmation.'
        aggressive_note = 'A more aggressive long is acceptable only if price reclaims cleanly with improving participation.'
    return {
        'should_trade': False,
        'order_type': 'limit',
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'market_read': str(ref_context.get('summary') or 'OpenAI response was empty; preserve the setup as a structured watch plan.')[:280],
        'entry_plan': 'Do not enter now. Keep the setup in watch mode and recheck only after the trigger condition is visible in the supplied timeframe.',
        'entry_reason': 'Fallback plan uses the supplied payload only and waits for clearer structure confirmation before execution.',
        'stop_loss_reason': 'Invalidation stays beyond the closest meaningful structure so normal noise does not trigger an unnecessary stop.',
        'take_profit_plan': 'Use the nearest realistic short-term objective from structure as the first target; reassess runners only after confirmation.',
        'if_missed_plan': 'If price moves without confirming the trigger, do not chase; wait for the next clean retest or reclaim/failure sequence.',
        'reference_summary': str(reason or 'OpenAI returned no parseable JSON, so the bot preserved a conservative watch plan from the same payload.')[:220],
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
            'Fallback observation plan was used because the OpenAI response body was empty.',
            'Do not convert this into a live trade until the watch trigger actually confirms.',
            'Recheck with the same payload family once the trigger condition appears.',
        ],
        'aggressive_note': aggressive_note,
        'watch_trigger_type': watch_trigger_type,
        'watch_trigger_price': watch_trigger_price,
        'watch_invalidation_price': watch_invalidation_price,
        'watch_note': watch_note,
        'recheck_reason': 'Trigger condition became observable after the previous empty model response fallback.',
        'watch_timeframe': watch_timeframe,
        'watch_price_zone_low': zone_low,
        'watch_price_zone_high': zone_high,
        'watch_structure_condition': structure_condition,
        'watch_volume_condition': volume_condition,
        'watch_checklist': [
            'Check the watch timeframe close against the trigger level',
            'Check whether structure confirms or rejects the thesis',
            'Check whether volume confirms the move rather than fading immediately',
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
    body = {
        'model': str(model or config.get('model') or 'gpt-5.4-mini'),
        'input': _build_messages(candidate, compact=compact_prompt),
        'max_output_tokens': int(max_output_tokens or config.get('max_output_tokens', 1536) or 1536),
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


def _normalize_decision(raw: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    raw = dict(raw or {})
    side = str(candidate.get('side') or 'long').lower()
    constraints = dict(candidate.get('constraints') or {})
    reference_trade_plan = dict(candidate.get('reference_trade_plan') or {})
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
        'watch_timeframe': str(raw.get('watch_timeframe') or '').strip(),
        'watch_price_zone_low': _coerce_float(raw.get('watch_price_zone_low', 0), 0.0),
        'watch_price_zone_high': _coerce_float(raw.get('watch_price_zone_high', 0), 0.0),
        'watch_structure_condition': str(raw.get('watch_structure_condition') or '').strip(),
        'watch_volume_condition': str(raw.get('watch_volume_condition') or '').strip(),
        'watch_checklist': [str(x).strip() for x in list(raw.get('watch_checklist') or []) if str(x).strip()][:8],
        'watch_confirmations': [str(x).strip() for x in list(raw.get('watch_confirmations') or []) if str(x).strip()][:8],
        'watch_invalidations': [str(x).strip() for x in list(raw.get('watch_invalidations') or []) if str(x).strip()][:8],
        'watch_recheck_priority': _clamp(_coerce_float(raw.get('watch_recheck_priority', 0), 0.0), 0, 100),
        'limit_cancel_price': _coerce_float(raw.get('limit_cancel_price', 0), 0.0),
        'limit_cancel_timeframe': str(raw.get('limit_cancel_timeframe') or '').strip(),
        'limit_cancel_condition': str(raw.get('limit_cancel_condition') or '').strip(),
        'limit_cancel_note': str(raw.get('limit_cancel_note') or '').strip(),
    }
    allowed_watch = {'none', 'pullback_to_entry', 'breakout_reclaim', 'breakdown_confirm', 'volume_confirmation', 'manual_review'}
    if decision['watch_trigger_type'] not in allowed_watch:
        decision['watch_trigger_type'] = 'none'
    if constraints.get('fixed_leverage'):
        decision['leverage'] = int(constraints.get('fixed_leverage') or decision['leverage'])
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
    if not decision['should_trade'] and not decision['reason_to_skip']:
        decision['reason_to_skip'] = 'Model rejected the setup after reviewing the full payload.'
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
    same_payload_reuse_sec = max(int(config.get('same_payload_reuse_minutes', 45) or 45), 1) * 60
    top_signature = _top_candidates_signature(candidate)

    if (
        not force_recheck
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

    if not force_recheck and last_sent_ts > 0 and (now_ts - last_sent_ts) < cooldown_sec:
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
    top_changed = bool(top_signature) and top_signature != str(state.get('last_top_candidates_signature') or '')
    if global_interval_sec > 0 and last_consulted_ts > 0 and (now_ts - last_consulted_ts) < global_interval_sec and not top_changed and not force_recheck:
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
        retry_effort = primary_effort
        max_tokens = int(config.get('max_output_tokens', 1536) or 1536)
        attempts = [
            {'model': primary_model, 'structured': False, 'effort': primary_effort, 'max_tokens': max_tokens, 'compact_prompt': False},
            {'model': primary_model, 'structured': False, 'effort': retry_effort, 'max_tokens': min(max_tokens, 960), 'compact_prompt': True},
        ]
        if allow_upgrade and upgrade_model and upgrade_model != primary_model and rank <= 1 and score_abs >= max(float(config.get('min_score_abs', 43.0) or 43.0), 52.0):
            attempts.append({'model': upgrade_model, 'structured': False, 'effort': retry_effort, 'max_tokens': min(max_tokens, 960), 'compact_prompt': True})
        if fallback_model and fallback_model != primary_model:
            attempts.append({'model': fallback_model, 'structured': False, 'effort': retry_effort, 'max_tokens': min(max_tokens, 960), 'compact_prompt': True})

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
        for attempt_index, attempt in enumerate(attempts):
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
            if resp.status_code == 400 and bool(attempt.get('structured', True)):
                body_text = ''
                try:
                    body_text = resp.text or ''
                except Exception:
                    body_text = ''
                if logger:
                    logger('OpenAI structured request 400: {} | {}'.format(symbol, body_text[:260]))
                request_body = _build_request_body(
                    candidate,
                    config,
                    structured=False,
                    model=selected_model,
                    reasoning_effort=retry_effort,
                    max_output_tokens=min(max_tokens, 960),
                    compact_prompt=True,
                )
                resp = requests.post(base_url, headers=headers, json=request_body, timeout=timeout_sec)
            if resp.status_code in (403, 404, 429) and attempt_index < len(attempts) - 1:
                body_text = ''
                try:
                    body_text = resp.text or ''
                except Exception:
                    body_text = ''
                if logger:
                    logger('OpenAI attempt unavailable, retrying next model: {} status={} | {}'.format(
                        selected_model,
                        resp.status_code,
                        body_text[:180],
                    ))
                continue
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
            empty_details.append('{} structured={} effort={} output_tokens={}'.format(
                selected_model,
                bool(attempt.get('structured', True)),
                str(attempt.get('effort') or ''),
                _response_output_tokens(body),
            ))
            if logger:
                logger('OpenAI empty/invalid JSON response: {} | {}'.format(symbol, empty_details[-1]))
        if not raw_json:
            est_cost_usd = estimate_cost_usd(config, input_tokens=total_input_tokens, output_tokens=total_output_tokens, cached_input_tokens=total_cached_input_tokens)
            est_cost_twd = est_cost_usd * float(config.get('usd_to_twd', 32.0) or 32.0)
            detail = 'OpenAI returned no parseable trade JSON after retries: {}'.format(' ; '.join(empty_details))
            fallback_decision = _fallback_trade_decision(candidate, reason=detail)
            symbol_state.update({
                'last_payload_hash': payload_hash,
                'last_sent_ts': now_ts,
                'last_model': selected_model,
                'last_decision': dict(fallback_decision or {}),
                'last_status': 'consulted_fallback',
                'last_response_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'last_cost_twd': round(est_cost_twd, 4),
                'last_attempt': dict(selected_attempt),
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
                    status='consulted',
                    action='skip',
                    detail='OpenAI returned empty output; conservative fallback watch plan was generated from the same payload.',
                    decision=fallback_decision,
                    model='{} fallback'.format(selected_model),
                ),
            )
            save_trade_state(state_path, state)
            return state, {
                'status': 'consulted',
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
                'fallback_used': True,
                'model': '{} fallback'.format(selected_model),
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
        state['api_calls'] = int(state.get('api_calls', 0) or 0) + 1
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
                action='trade' if decision.get('should_trade') else 'skip',
                detail='OpenAI returned a structured trade plan.',
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
        if str((row or {}).get('status') or '') != 'global_interval_active'
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
