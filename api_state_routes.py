from __future__ import annotations

from typing import Any, Dict, Iterable


def _compact_openai_market_context(context: Dict[str, Any]) -> Dict[str, Any]:
    src = dict(context or {})
    if not src:
        return {}
    # Keep analysis summary but drop very heavy raw bar payloads for UI polling endpoints.
    src.pop('timeframe_bars', None)
    src.pop('multi_timeframe_pressure', None)
    src.pop('multi_timeframe_pressure_summary', None)
    return src


def _compact_signal_row(row: Dict[str, Any]) -> Dict[str, Any]:
    src = dict(row or {})
    if not src:
        return {}
    if isinstance(src.get('openai_market_context'), dict):
        src['openai_market_context'] = _compact_openai_market_context(dict(src.get('openai_market_context') or {}))
    src.pop('prebreakout_raw_candidate_payload', None)
    src.pop('prebreakout_flow_context', None)
    return src


def _compact_signal_list(rows: Any) -> Any:
    if not isinstance(rows, list):
        return rows
    return [_compact_signal_row(dict(item or {})) for item in rows]


def _compact_prebreakout_leaderboard(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    out.pop('candidate_payloads', None)
    return out


def pick_keys(payload: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    src = dict(payload or {})
    return {k: src.get(k) for k in keys if k in src}


def build_state_lite_payload(base_payload: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        'last_update', 'scan_progress', 'equity', 'total_pnl', 'threshold_info',
        'risk_status', 'market_info', 'latest_news_title', 'learn_summary',
        'lt_info', 'fvg_orders', 'top_signals', 'general_top_signals', 'short_gainer_signals', 'prebreakout_signals', 'prebreakout_leaderboard',
        'watchlist', 'backend_threads'
    ]
    return pick_keys(base_payload, keys)


def build_positions_payload(base_payload: Dict[str, Any]) -> Dict[str, Any]:
    keys = ['active_positions', 'trailing_info', 'protection_state', 'trade_history', 'watchlist']
    return pick_keys(base_payload, keys)


def build_ai_panel_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    keys = ['ai_panel', 'auto_backtest', 'trend_dashboard', 'top_signals', 'general_top_signals', 'short_gainer_signals', 'prebreakout_signals', 'prebreakout_leaderboard', 'watchlist', 'learn_summary', 'backend_threads']
    return pick_keys(payload, keys)


def compact_state_lite_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload or {})
    for key in ('top_signals', 'general_top_signals', 'short_gainer_signals', 'prebreakout_signals'):
        out[key] = _compact_signal_list(out.get(key))
    out['prebreakout_leaderboard'] = _compact_prebreakout_leaderboard(out.get('prebreakout_leaderboard'))
    return out


def compact_ai_panel_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload or {})
    for key in ('top_signals', 'general_top_signals', 'short_gainer_signals', 'prebreakout_signals'):
        out[key] = _compact_signal_list(out.get(key))
    out['prebreakout_leaderboard'] = _compact_prebreakout_leaderboard(out.get('prebreakout_leaderboard'))
    return out
