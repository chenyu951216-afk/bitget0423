from __future__ import annotations

import hashlib
import json
import math
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
        'model': str(env_getter('OPENAI_TRADE_MODEL', 'gpt-5-nano') or 'gpt-5-nano').strip(),
        'monthly_budget_twd': monthly_budget_twd,
        'soft_budget_twd': round(monthly_budget_twd * soft_ratio, 2),
        'hard_budget_twd': round(monthly_budget_twd * hard_ratio, 2),
        'usd_to_twd': max(_env_float(env_getter, 'OPENAI_TRADE_USD_TO_TWD', 32.0), 1.0),
        'input_price_per_1m_usd': max(_env_float(env_getter, 'OPENAI_TRADE_PRICE_INPUT_PER_1M_USD', 0.20), 0.0),
        'output_price_per_1m_usd': max(_env_float(env_getter, 'OPENAI_TRADE_PRICE_OUTPUT_PER_1M_USD', 1.25), 0.0),
        'cached_input_price_per_1m_usd': max(_env_float(env_getter, 'OPENAI_TRADE_PRICE_CACHED_INPUT_PER_1M_USD', 0.02), 0.0),
        'top_k_per_scan': max(_env_int(env_getter, 'OPENAI_TRADE_TOP_K', 3), 1),
        'cooldown_minutes': max(_env_int(env_getter, 'OPENAI_TRADE_SYMBOL_COOLDOWN_MINUTES', 180), 1),
        'min_score_abs': max(_env_float(env_getter, 'OPENAI_TRADE_MIN_SCORE', 38.0), 0.0),
        'min_margin_pct': min(max(_env_float(env_getter, 'OPENAI_TRADE_MIN_MARGIN_PCT', 0.03), 0.005), 0.5),
        'max_margin_pct': min(max(_env_float(env_getter, 'OPENAI_TRADE_MAX_MARGIN_PCT', 0.08), 0.01), 0.8),
        'min_leverage': max(_env_int(env_getter, 'OPENAI_TRADE_MIN_LEVERAGE', 4), 1),
        'max_leverage': max(_env_int(env_getter, 'OPENAI_TRADE_MAX_LEVERAGE', 25), 1),
        'max_output_tokens': max(_env_int(env_getter, 'OPENAI_TRADE_MAX_OUTPUT_TOKENS', 900), 200),
        'request_timeout_sec': max(_env_float(env_getter, 'OPENAI_TRADE_TIMEOUT_SEC', 35.0), 5.0),
        'temperature': 0.2,
        'base_url': str(env_getter('OPENAI_RESPONSES_URL', 'https://api.openai.com/v1/responses') or 'https://api.openai.com/v1/responses').strip(),
        'reasoning_effort': str(env_getter('OPENAI_TRADE_REASONING_EFFORT', 'low') or 'low').strip(),
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
        'symbols': {},
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


def _clamp(value: Any, low: float, high: float) -> float:
    try:
        v = float(value)
    except Exception:
        v = float(low)
    return max(float(low), min(float(high), v))


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
        'entry_price': candidate.get('entry_price'),
        'stop_loss': candidate.get('stop_loss'),
        'take_profit': candidate.get('take_profit'),
        'score': candidate.get('score'),
        'priority_score': candidate.get('priority_score'),
        'entry_quality': candidate.get('entry_quality'),
        'rr_ratio': candidate.get('rr_ratio'),
        'market_pattern': ((candidate.get('market') or {}).get('pattern')),
        'market_direction': ((candidate.get('market') or {}).get('direction')),
        'breakdown': candidate.get('breakdown_full') or {},
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
    return {
        'symbol': str(signal.get('symbol') or ''),
        'side': 'long' if float(signal.get('score', 0) or 0) >= 0 else 'short',
        'rank': int(rank_index) + 1,
        'score': _round(signal.get('score'), 4),
        'raw_score': _round(signal.get('raw_score', signal.get('score')), 4),
        'priority_score': _round(signal.get('priority_score', abs(float(signal.get('score', 0) or 0))), 4),
        'entry_price': _round(signal.get('price'), 8),
        'stop_loss': _round(signal.get('stop_loss'), 8),
        'take_profit': _round(signal.get('take_profit'), 8),
        'entry_quality': _round(signal.get('entry_quality', breakdown.get('EntryGate')), 4),
        'rr_ratio': _round(signal.get('rr_ratio', breakdown.get('RR')), 4),
        'est_pnl_pct': _round(signal.get('est_pnl'), 4),
        'regime': str(signal.get('regime') or breakdown.get('Regime') or 'neutral'),
        'regime_confidence': _round(signal.get('regime_confidence'), 4),
        'setup_label': str(signal.get('setup_label') or breakdown.get('Setup') or ''),
        'signal_grade': str(signal.get('signal_grade') or breakdown.get('等級') or ''),
        'trend_confidence': _round(signal.get('trend_confidence'), 4),
        'rotation_adj': _round(signal.get('rotation_adj'), 4),
        'breakdown_full': breakdown,
        'execution_quality': execution_quality,
        'market': {
            'pattern': str(market.get('pattern') or ''),
            'direction': str(market.get('direction') or ''),
            'strength': _round(market.get('strength'), 4),
            'prediction': str(market.get('prediction') or '')[:240],
        },
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
        'top_candidates': list(top_candidates or [])[:5],
        'constraints': dict(constraints or {}),
    }


def _short_label(status: str) -> str:
    mapping = {
        'consulted': 'OpenAI consulted',
        'cached_reuse': 'Cached decision reused',
        'cooldown_active': 'Symbol cooldown active',
        'same_payload_reuse': 'Same payload reused',
        'budget_paused': 'Budget paused',
        'below_min_score': 'Score below filter',
        'not_ranked': 'Outside OpenAI top K',
        'disabled': 'OpenAI disabled',
        'missing_api_key': 'Missing API key',
        'auth_error': 'OpenAI auth error',
        'permission_error': 'OpenAI permission error',
        'bad_request': 'OpenAI bad request',
        'rate_limit': 'OpenAI rate limit',
        'error': 'OpenAI error',
    }
    return mapping.get(str(status or ''), str(status or 'unknown'))


def _append_recent(state: Dict[str, Any], item: Dict[str, Any], limit: int = 14) -> None:
    rows = list(state.get('recent_decisions') or [])
    rows.insert(0, dict(item or {}))
    state['recent_decisions'] = rows[:max(int(limit), 1)]


def _build_recent_item(candidate: Dict[str, Any], *, status: str, action: str = '', detail: str = '', decision: Dict[str, Any] | None = None, model: str = '') -> Dict[str, Any]:
    decision = dict(decision or {})
    return {
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'symbol': str(candidate.get('symbol') or ''),
        'side': str(candidate.get('side') or ''),
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
    }


def _extract_text(body: Dict[str, Any]) -> str:
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


def _json_schema() -> Dict[str, Any]:
    return {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'should_trade': {'type': 'boolean'},
            'order_type': {'type': 'string', 'enum': ['market', 'limit']},
            'entry_price': {'type': 'number'},
            'stop_loss': {'type': 'number'},
            'take_profit': {'type': 'number'},
            'leverage': {'type': 'integer'},
            'margin_pct': {'type': 'number'},
            'confidence': {'type': 'number'},
            'thesis': {'type': 'string'},
            'reason_to_skip': {'type': 'string'},
            'risk_notes': {'type': 'array', 'items': {'type': 'string'}},
            'aggressive_note': {'type': 'string'},
        },
        'required': [
            'should_trade',
            'order_type',
            'entry_price',
            'stop_loss',
            'take_profit',
            'leverage',
            'margin_pct',
            'confidence',
            'thesis',
            'reason_to_skip',
            'risk_notes',
            'aggressive_note',
        ],
    }


def _build_messages(candidate: Dict[str, Any]) -> list[Dict[str, Any]]:
    constraints = dict(candidate.get('constraints') or {})
    min_margin_pct = float(constraints.get('min_margin_pct', 0.03) or 0.03)
    max_margin_pct = float(constraints.get('max_margin_pct', 0.08) or 0.08)
    min_leverage = int(constraints.get('min_leverage', 4) or 4)
    max_leverage = int(constraints.get('max_leverage', 25) or 25)
    system_text = (
        'You are a crypto perpetual futures execution planner. '
        'Use the full candidate payload. '
        'Make the final trading plan, not a generic analysis. '
        'Be decisive rather than overly conservative, but stay inside the leverage and margin bounds. '
        'If the setup is tradable, do not default to tiny size. '
        'Only reject the trade when the setup is clearly weak, conflicting, or poorly priced. '
        'Return valid JSON only.'
    )
    user_text = (
        'Decide whether to place a crypto perpetual futures order.\n'
        f'Hard bounds:\n- leverage must be between {min_leverage} and {max_leverage}\n'
        f'- margin_pct must be between {min_margin_pct:.4f} and {max_margin_pct:.4f}\n'
        '- order_type must be market or limit\n'
        '- stop_loss and take_profit must be valid for the side\n'
        '- confidence should be 0-100\n'
        'If you choose limit order, entry_price is the limit price.\n'
        'If you choose not to trade, still return a complete object and explain why.\n'
        'Candidate payload JSON:\n'
        + json.dumps(candidate, ensure_ascii=False, separators=(',', ':'))
    )
    return [
        {'role': 'system', 'content': [{'type': 'input_text', 'text': system_text}]},
        {'role': 'user', 'content': [{'type': 'input_text', 'text': user_text}]},
    ]


def _build_request_body(candidate: Dict[str, Any], config: Dict[str, Any], *, structured: bool = True) -> Dict[str, Any]:
    body = {
        'model': str(config.get('model') or 'gpt-5-nano'),
        'input': _build_messages(candidate),
        'max_output_tokens': int(config.get('max_output_tokens', 900) or 900),
    }
    effort = str(config.get('reasoning_effort') or '').strip()
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
    entry_default = float(candidate.get('entry_price', 0) or 0)
    stop_default = float(candidate.get('stop_loss', 0) or 0)
    tp_default = float(candidate.get('take_profit', 0) or 0)
    decision = {
        'should_trade': bool(raw.get('should_trade', False)),
        'order_type': 'limit' if str(raw.get('order_type') or '').lower() == 'limit' else 'market',
        'entry_price': float(raw.get('entry_price', entry_default) or entry_default),
        'stop_loss': float(raw.get('stop_loss', stop_default) or stop_default),
        'take_profit': float(raw.get('take_profit', tp_default) or tp_default),
        'leverage': int(_clamp(raw.get('leverage', constraints.get('min_leverage', 4)), constraints.get('min_leverage', 4), constraints.get('max_leverage', 25))),
        'margin_pct': _clamp(raw.get('margin_pct', constraints.get('min_margin_pct', 0.03)), constraints.get('min_margin_pct', 0.03), constraints.get('max_margin_pct', 0.08)),
        'confidence': _clamp(raw.get('confidence', 0), 0, 100),
        'thesis': str(raw.get('thesis') or '').strip(),
        'reason_to_skip': str(raw.get('reason_to_skip') or '').strip(),
        'risk_notes': [str(x).strip() for x in list(raw.get('risk_notes') or []) if str(x).strip()][:6],
        'aggressive_note': str(raw.get('aggressive_note') or '').strip(),
    }
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
    cooldown_sec = max(int(config.get('cooldown_minutes', 180) or 180), 1) * 60
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

    if score_abs < float(config.get('min_score_abs', 38.0) or 38.0):
        result = {'status': 'below_min_score', 'decision': None, 'payload_hash': payload_hash, 'symbol_state': symbol_state}
        return state, result

    if rank > int(config.get('top_k_per_scan', 3) or 3):
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
    if cached_decision and last_hash == payload_hash:
        result = {
            'status': 'cached_reuse',
            'decision': cached_decision,
            'payload_hash': payload_hash,
            'symbol_state': symbol_state,
            'cached': True,
        }
        _append_recent(
            state,
            _build_recent_item(
                candidate,
                status='cached_reuse',
                action='trade' if cached_decision.get('should_trade') else 'skip',
                detail='Same symbol payload already sent before, reused cached OpenAI decision.',
                decision=cached_decision,
                model=str(symbol_state.get('last_model') or config.get('model') or ''),
            ),
        )
        save_trade_state(state_path, state)
        return state, result

    if last_sent_ts > 0 and (now_ts - last_sent_ts) < cooldown_sec:
        next_allowed_ts = last_sent_ts + cooldown_sec
        detail = 'Symbol cooldown is active until {}.'.format(datetime.fromtimestamp(next_allowed_ts).strftime('%Y-%m-%d %H:%M:%S'))
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

    est_call_cost_twd = estimate_cost_twd(config, input_tokens=2200, output_tokens=500)
    soft_budget_twd = float(config.get('soft_budget_twd', 850.0) or 850.0)
    if spent_twd >= soft_budget_twd and score_abs < max(float(config.get('min_score_abs', 38.0) or 38.0) + 8.0, 52.0):
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
    request_body = _build_request_body(candidate, config, structured=True)
    if logger:
        logger('OpenAI trade decision request: {} rank={} score={:.2f}'.format(symbol, rank, score_abs))

    try:
        base_url = str(config.get('base_url') or 'https://api.openai.com/v1/responses')
        timeout_sec = float(config.get('request_timeout_sec', 35.0) or 35.0)
        resp = requests.post(base_url, headers=headers, json=request_body, timeout=timeout_sec)
        if resp.status_code == 400:
            body_text = ''
            try:
                body_text = resp.text or ''
            except Exception:
                body_text = ''
            if logger:
                logger('OpenAI structured request 400: {} | {}'.format(symbol, body_text[:260]))
            fallback_body = _build_request_body(candidate, config, structured=False)
            resp = requests.post(base_url, headers=headers, json=fallback_body, timeout=timeout_sec)
        resp.raise_for_status()
        body = resp.json()
        raw_text = _extract_text(body)
        raw_json = _parse_json_text(raw_text)
        decision = _normalize_decision(raw_json, candidate)
        usage = dict(body.get('usage') or {})
        input_tokens = int(usage.get('input_tokens', 0) or 0)
        output_tokens = int(usage.get('output_tokens', 0) or 0)
        cached_input_tokens = int(usage.get('input_cached_tokens', usage.get('cached_input_tokens', 0)) or 0)
        est_cost_usd = estimate_cost_usd(config, input_tokens=input_tokens, output_tokens=output_tokens, cached_input_tokens=cached_input_tokens)
        est_cost_twd = est_cost_usd * float(config.get('usd_to_twd', 32.0) or 32.0)

        symbol_state.update({
            'last_payload_hash': payload_hash,
            'last_sent_ts': now_ts,
            'last_model': str(config.get('model') or 'gpt-5-nano'),
            'last_decision': dict(decision or {}),
            'last_status': 'consulted',
            'last_response_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'last_cost_twd': round(est_cost_twd, 4),
        })
        state.setdefault('symbols', {})[symbol] = symbol_state
        state['api_calls'] = int(state.get('api_calls', 0) or 0) + 1
        state['input_tokens'] = int(state.get('input_tokens', 0) or 0) + input_tokens
        state['output_tokens'] = int(state.get('output_tokens', 0) or 0) + output_tokens
        state['cached_input_tokens'] = int(state.get('cached_input_tokens', 0) or 0) + cached_input_tokens
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
                model=str(config.get('model') or ''),
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
        state['last_error'] = err[:300]
        state['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        _append_recent(state, _build_recent_item(candidate, status=status, detail=detail[:260], model=str(config.get('model') or '')))
        save_trade_state(state_path, state)
        if logger:
            logger('OpenAI trade decision failed: {} | {}'.format(symbol, detail[:240]))
        return state, {
            'status': status,
            'decision': None,
            'payload_hash': payload_hash,
            'symbol_state': symbol_state,
            'error': detail,
        }


def build_dashboard_payload(state: Dict[str, Any], config: Dict[str, Any], *, api_key_present: bool) -> Dict[str, Any]:
    state = dict(state or {})
    budget_total = float(config.get('monthly_budget_twd', 1000.0) or 1000.0)
    spent_twd = float(state.get('spent_estimated_twd', 0.0) or 0.0)
    remaining = max(budget_total - spent_twd, 0.0)
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
        'top_k_per_scan': int(config.get('top_k_per_scan', 3) or 3),
        'cooldown_minutes': int(config.get('cooldown_minutes', 180) or 180),
        'min_score_abs': float(config.get('min_score_abs', 38.0) or 38.0),
        'last_error': str(state.get('last_error') or ''),
        'updated_at': str(state.get('updated_at') or ''),
        'recent_decisions': list(state.get('recent_decisions') or [])[:12],
    }
