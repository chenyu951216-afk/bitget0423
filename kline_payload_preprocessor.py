from __future__ import annotations

import json
import math
import sys
from typing import Any, Dict, List, Tuple

TIMEFRAMES: Tuple[str, ...] = ("1m", "5m", "15m", "1h", "4h", "1d")
KEPT_RECENT_CANDLES = 5
ZERO_VOLUME_RATIO_THRESHOLD = 0.5
MIN_ROWS_REQUIRED = 20
FLAT_PRICE_STREAK_THRESHOLD = 8


def _normalize_missing(value: Any) -> Any:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned == "" or cleaned == "沒有":
            return None
        return value
    if isinstance(value, list):
        return [_normalize_missing(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _normalize_missing(v) for k, v in value.items()}
    return value


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _extract_rows(timeframe_payload: Any) -> List[Any]:
    if isinstance(timeframe_payload, dict):
        rows = timeframe_payload.get("rows")
        return list(rows) if isinstance(rows, list) else []
    if isinstance(timeframe_payload, list):
        return list(timeframe_payload)
    return []


def _parse_candle(row: Any) -> Tuple[float | None, float | None, float | None, float | None, float | None]:
    if isinstance(row, dict):
        return (
            _to_float_or_none(row.get("open")),
            _to_float_or_none(row.get("high")),
            _to_float_or_none(row.get("low")),
            _to_float_or_none(row.get("close")),
            _to_float_or_none(row.get("volume")),
        )
    if isinstance(row, (list, tuple)) and len(row) >= 5:
        return (
            _to_float_or_none(row[0]),
            _to_float_or_none(row[1]),
            _to_float_or_none(row[2]),
            _to_float_or_none(row[3]),
            _to_float_or_none(row[4]),
        )
    return (None, None, None, None, None)


def _candle_shape(open_price: float | None, high: float | None, low: float | None, close: float | None) -> Dict[str, float | None]:
    if None in (open_price, high, low, close):
        return {"body_pct": None, "upper_wick_pct": None, "lower_wick_pct": None}
    candle_range = float(high) - float(low)
    if candle_range <= 0:
        return {"body_pct": None, "upper_wick_pct": None, "lower_wick_pct": None}
    body = abs(float(close) - float(open_price))
    upper_wick = max(float(high) - max(float(open_price), float(close)), 0.0)
    lower_wick = max(min(float(open_price), float(close)) - float(low), 0.0)
    return {
        "body_pct": _round_or_none((body / candle_range) * 100.0),
        "upper_wick_pct": _round_or_none((upper_wick / candle_range) * 100.0),
        "lower_wick_pct": _round_or_none((lower_wick / candle_range) * 100.0),
    }


def _is_invalid_ohlc(open_price: float | None, high: float | None, low: float | None, close: float | None, volume: float | None) -> bool:
    if None in (open_price, high, low, close, volume):
        return True
    if float(volume) < 0:
        return True
    if float(high) < float(open_price):
        return True
    if float(high) < float(close):
        return True
    if float(high) < float(low):
        return True
    if float(low) > float(open_price):
        return True
    if float(low) > float(close):
        return True
    return False


def _summarize_timeframe(rows: List[Any]) -> Dict[str, Any]:
    candles = [_parse_candle(row) for row in rows]
    rows_count = len(candles)

    highs = [c[1] for c in candles if c[1] is not None]
    lows = [c[2] for c in candles if c[2] is not None]
    volumes = [c[4] for c in candles if c[4] is not None]

    first_open = candles[0][0] if candles else None
    last_close = candles[-1][3] if candles else None
    max_high = max(highs) if highs else None
    min_low = min(lows) if lows else None

    total_volume = sum(volumes) if volumes else None
    latest_volume = candles[-1][4] if candles else None
    average_volume = (total_volume / len(volumes)) if (total_volume is not None and volumes) else None

    zero_volume_count = sum(1 for v in volumes if v == 0)
    zero_volume_ratio = (zero_volume_count / rows_count) if rows_count > 0 else None

    range_pct = None
    if max_high is not None and min_low is not None and last_close not in (None, 0):
        range_pct = ((max_high - min_low) / last_close) * 100.0

    return_pct = None
    if first_open not in (None, 0) and last_close is not None:
        return_pct = ((last_close - first_open) / first_open) * 100.0

    close_position_pct = None
    if max_high is not None and min_low is not None and last_close is not None:
        denom = max_high - min_low
        if denom > 0:
            close_position_pct = ((last_close - min_low) / denom) * 100.0

    invalid_ohlc = any(_is_invalid_ohlc(*candle) for candle in candles)
    insufficient_rows = rows_count < MIN_ROWS_REQUIRED
    too_many_zero_volume = (zero_volume_ratio is not None) and (zero_volume_ratio >= ZERO_VOLUME_RATIO_THRESHOLD)

    longest_flat_streak = 0
    current_flat_streak = 0
    for open_price, high, low, close, _volume in candles:
        is_flat = (
            open_price is not None
            and high is not None
            and low is not None
            and close is not None
            and abs(open_price - high) < 1e-12
            and abs(high - low) < 1e-12
            and abs(low - close) < 1e-12
        )
        if is_flat:
            current_flat_streak += 1
            longest_flat_streak = max(longest_flat_streak, current_flat_streak)
        else:
            current_flat_streak = 0
    flat_price_too_long = longest_flat_streak >= FLAT_PRICE_STREAK_THRESHOLD

    last_candle = {
        "open": None,
        "high": None,
        "low": None,
        "close": None,
        "volume": None,
        "body_pct": None,
        "upper_wick_pct": None,
        "lower_wick_pct": None,
    }
    if candles:
        last_open, last_high, last_low, last_close_value, last_volume = candles[-1]
        last_candle = {
            "open": _round_or_none(last_open),
            "high": _round_or_none(last_high),
            "low": _round_or_none(last_low),
            "close": _round_or_none(last_close_value),
            "volume": _round_or_none(last_volume),
            **_candle_shape(last_open, last_high, last_low, last_close_value),
        }

    recent_candles = []
    for open_price, high, low, close, volume in candles[-KEPT_RECENT_CANDLES:]:
        recent_candles.append(
            {
                "open": _round_or_none(open_price),
                "high": _round_or_none(high),
                "low": _round_or_none(low),
                "close": _round_or_none(close),
                "volume": _round_or_none(volume),
            }
        )

    return {
        "rows_count": rows_count,
        "first_open": _round_or_none(first_open),
        "last_close": _round_or_none(last_close),
        "max_high": _round_or_none(max_high),
        "min_low": _round_or_none(min_low),
        "total_volume": _round_or_none(total_volume),
        "latest_volume": _round_or_none(latest_volume),
        "average_volume": _round_or_none(average_volume),
        "zero_volume_count": zero_volume_count,
        "zero_volume_ratio": _round_or_none(zero_volume_ratio),
        "range_pct": _round_or_none(range_pct),
        "return_pct": _round_or_none(return_pct),
        "close_position_pct": _round_or_none(close_position_pct),
        "last_candle": last_candle,
        "recent_candles_compact": recent_candles,
        "abnormal_flags": {
            "invalid_ohlc": invalid_ohlc,
            "too_many_zero_volume": too_many_zero_volume,
            "insufficient_rows": insufficient_rows,
            "flat_price_too_long": flat_price_too_long,
        },
    }


def _distance_pct(current_price: float | None, target: Any, *, is_support: bool) -> float | None:
    target_value = _to_float_or_none(target)
    if current_price in (None, 0) or target_value is None:
        return None
    if is_support:
        return _round_or_none(((current_price - target_value) / current_price) * 100.0)
    return _round_or_none(((target_value - current_price) / current_price) * 100.0)


def _verify_liquidity(liquidity: Any) -> Dict[str, Any]:
    context = dict(liquidity or {}) if isinstance(liquidity, dict) else {}
    verification: Dict[str, Any] = {
        "status": "not_checked",
        "checks": [],
    }

    checks: List[Dict[str, Any]] = []

    bid_depth_10 = _to_float_or_none(context.get("bid_depth_10"))
    ask_depth_10 = _to_float_or_none(context.get("ask_depth_10"))
    provided_imbalance = _to_float_or_none(context.get("depth_imbalance_10"))
    if bid_depth_10 is not None and ask_depth_10 is not None and (bid_depth_10 + ask_depth_10) > 0:
        recomputed_imbalance = (bid_depth_10 - ask_depth_10) / (bid_depth_10 + ask_depth_10)
        if provided_imbalance is None:
            status = "verified"
            delta = None
        else:
            delta = abs(provided_imbalance - recomputed_imbalance)
            status = "verified" if delta <= 1e-4 else "mismatch_warning"
        checks.append(
            {
                "field": "depth_imbalance_10",
                "provided": _round_or_none(provided_imbalance),
                "recomputed": _round_or_none(recomputed_imbalance),
                "abs_diff": _round_or_none(delta),
                "status": status,
            }
        )

    aggressive_buy_notional = _to_float_or_none(context.get("aggressive_buy_notional"))
    aggressive_sell_notional = _to_float_or_none(context.get("aggressive_sell_notional"))
    provided_ratio = _to_float_or_none(context.get("buy_sell_notional_ratio"))
    if aggressive_buy_notional is not None and aggressive_sell_notional is not None:
        recomputed_ratio = None
        ratio_status = "verified"
        ratio_diff = None
        if aggressive_sell_notional > 0:
            recomputed_ratio = aggressive_buy_notional / aggressive_sell_notional
            if provided_ratio is not None:
                ratio_diff = abs(provided_ratio - recomputed_ratio)
                tolerance = max(1e-4, abs(recomputed_ratio) * 0.01)
                ratio_status = "verified" if ratio_diff <= tolerance else "mismatch_warning"
        else:
            ratio_status = "skipped_zero_denominator"
        checks.append(
            {
                "field": "buy_sell_notional_ratio",
                "provided": _round_or_none(provided_ratio),
                "recomputed": _round_or_none(recomputed_ratio),
                "abs_diff": _round_or_none(ratio_diff),
                "status": ratio_status,
            }
        )

    if checks:
        mismatch_exists = any(item.get("status") == "mismatch_warning" for item in checks)
        verification = {
            "status": "mismatch_warning" if mismatch_exists else "verified",
            "checks": checks,
        }

    context["local_verification"] = verification
    return context


def build_compact_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized_payload = _normalize_missing(payload)
    timeframe_bars = dict(normalized_payload.get("timeframe_bars") or {}) if isinstance(normalized_payload.get("timeframe_bars"), dict) else {}

    current_price = _to_float_or_none(normalized_payload.get("current_price"))

    levels_raw = dict(normalized_payload.get("levels") or {}) if isinstance(normalized_payload.get("levels"), dict) else {}
    levels = {
        "nearest_support": _to_float_or_none(levels_raw.get("nearest_support")),
        "nearest_resistance": _to_float_or_none(levels_raw.get("nearest_resistance")),
        "support_levels": list(levels_raw.get("support_levels") or []) if isinstance(levels_raw.get("support_levels"), list) else [],
        "resistance_levels": list(levels_raw.get("resistance_levels") or []) if isinstance(levels_raw.get("resistance_levels"), list) else [],
        "recent_high": _to_float_or_none(levels_raw.get("recent_high")),
        "recent_low": _to_float_or_none(levels_raw.get("recent_low")),
        "distance_to_support_pct": None,
        "distance_to_resistance_pct": None,
    }

    levels["distance_to_support_pct"] = _distance_pct(current_price, levels["nearest_support"], is_support=True)
    levels["distance_to_resistance_pct"] = _distance_pct(current_price, levels["nearest_resistance"], is_support=False)

    kline_summary: Dict[str, Any] = {}
    original_rows_count: Dict[str, int] = {}
    invalid_ohlc_timeframes: List[str] = []
    too_many_zero_volume_timeframes: List[str] = []
    insufficient_rows_timeframes: List[str] = []
    warnings: List[str] = []

    for timeframe in TIMEFRAMES:
        rows = _extract_rows(timeframe_bars.get(timeframe))
        original_rows_count[timeframe] = len(rows)
        summary = _summarize_timeframe(rows)
        kline_summary[timeframe] = summary

        abnormal = summary.get("abnormal_flags") or {}
        if bool(abnormal.get("invalid_ohlc")):
            invalid_ohlc_timeframes.append(timeframe)
        if bool(abnormal.get("too_many_zero_volume")):
            too_many_zero_volume_timeframes.append(timeframe)
        if bool(abnormal.get("insufficient_rows")):
            insufficient_rows_timeframes.append(timeframe)

    if invalid_ohlc_timeframes:
        warnings.append("invalid_ohlc_detected")
    if too_many_zero_volume_timeframes:
        warnings.append("too_many_zero_volume_detected")
    if insufficient_rows_timeframes:
        warnings.append("insufficient_rows_detected")

    compact_payload = {
        "symbol": normalized_payload.get("symbol"),
        "trade_style": normalized_payload.get("trade_style"),
        "current_price": _round_or_none(current_price),
        "market_context": dict(normalized_payload.get("market_context") or {}) if isinstance(normalized_payload.get("market_context"), dict) else {},
        "levels": levels,
        "multi_timeframe": dict(normalized_payload.get("multi_timeframe") or {}) if isinstance(normalized_payload.get("multi_timeframe"), dict) else {},
        "kline_summary": kline_summary,
        "liquidity_context": _verify_liquidity(normalized_payload.get("liquidity_context")),
        "derivatives_context": dict(normalized_payload.get("derivatives_context") or {}) if isinstance(normalized_payload.get("derivatives_context"), dict) else {},
        "risk": dict(normalized_payload.get("risk") or {}) if isinstance(normalized_payload.get("risk"), dict) else {},
        "portfolio": dict(normalized_payload.get("portfolio") or {}) if isinstance(normalized_payload.get("portfolio"), dict) else {},
        "execution_policy": dict(normalized_payload.get("execution_policy") or {}) if isinstance(normalized_payload.get("execution_policy"), dict) else {},
        "constraints": dict(normalized_payload.get("constraints") or {}) if isinstance(normalized_payload.get("constraints"), dict) else {},
        "data_quality": {
            "valid": len(invalid_ohlc_timeframes) == 0 and len(insufficient_rows_timeframes) == 0,
            "warnings": warnings,
            "invalid_ohlc_timeframes": invalid_ohlc_timeframes,
            "too_many_zero_volume_timeframes": too_many_zero_volume_timeframes,
            "insufficient_rows_timeframes": insufficient_rows_timeframes,
        },
        "compression_info": {
            "raw_timeframe_bars_removed": True,
            "kept_recent_candles_per_timeframe": KEPT_RECENT_CANDLES,
            "original_rows_count": original_rows_count,
            "compressed": True,
            "processing_notes": [
                "timeframe_bars.rows removed from outbound payload",
                "kline_summary generated with fixed formulas only",
                "multi_timeframe/levels/liquidity_context/derivatives_context/risk/portfolio/execution_policy/constraints preserved",
                "recent_candles_compact keeps only the latest 5 candles per timeframe",
            ],
        },
    }

    return compact_payload


def main(argv: List[str]) -> int:
    input_path = argv[1] if len(argv) > 1 else "payload.json"
    output_path = argv[2] if len(argv) > 2 else "compact_payload.json"

    with open(input_path, "r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)

    compact_payload = build_compact_payload(payload)

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(compact_payload, handle, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
