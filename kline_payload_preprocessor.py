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
RECENT_ANOMALY_WINDOW_MS = 2 * 60 * 60 * 1000
VOLUME_SPIKE_MULT_THRESHOLD = 2.2
NEEDLE_WICK_PCT_THRESHOLD = 55.0
MOVE_Z_SCORE_THRESHOLD = 2.0
MOVE_MIN_PCT_THRESHOLD = 2.0


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


def _timeframe_interval_ms(timeframe: str) -> int:
    tf = str(timeframe or "").strip().lower()
    if tf.endswith("m"):
        minutes = _to_float_or_none(tf[:-1])
        if minutes and minutes > 0:
            return int(minutes * 60 * 1000)
    if tf.endswith("h"):
        hours = _to_float_or_none(tf[:-1])
        if hours and hours > 0:
            return int(hours * 60 * 60 * 1000)
    if tf.endswith("d"):
        days = _to_float_or_none(tf[:-1])
        if days and days > 0:
            return int(days * 24 * 60 * 60 * 1000)
    return 0


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


def _detect_recent_2h_anomaly_candles(
    candles: List[Tuple[float | None, float | None, float | None, float | None, float | None]],
    *,
    timeframe: str,
    start_ts: int = 0,
    interval_ms: int = 0,
) -> List[Dict[str, Any]]:
    if not candles:
        return []
    if interval_ms <= 0:
        interval_ms = _timeframe_interval_ms(timeframe)
    if interval_ms <= 0:
        return []
    rows_count = len(candles)
    if rows_count <= 0:
        return []
    max_rows_recent = max(1, int(math.ceil(RECENT_ANOMALY_WINDOW_MS / max(interval_ms, 1))))
    recent_start_idx = max(0, rows_count - max_rows_recent)

    volumes = [float(c[4]) for c in candles if c[4] is not None and float(c[4]) >= 0]
    avg_volume = (sum(volumes) / len(volumes)) if volumes else 0.0
    std_volume = math.sqrt(sum((v - avg_volume) ** 2 for v in volumes) / len(volumes)) if volumes else 0.0

    move_pcts: List[float] = []
    for open_price, _high, _low, close, _volume in candles:
        if open_price in (None, 0) or close is None:
            continue
        move_pcts.append(abs((float(close) - float(open_price)) / float(open_price) * 100.0))
    avg_move = (sum(move_pcts) / len(move_pcts)) if move_pcts else 0.0
    std_move = math.sqrt(sum((m - avg_move) ** 2 for m in move_pcts) / len(move_pcts)) if move_pcts else 0.0

    anomalies: List[Dict[str, Any]] = []
    for idx in range(recent_start_idx, rows_count):
        open_price, high, low, close, volume = candles[idx]
        if _is_invalid_ohlc(open_price, high, low, close, volume):
            continue
        o = float(open_price)
        h = float(high)
        l = float(low)
        c = float(close)
        v = float(volume)
        shape = _candle_shape(o, h, l, c)
        move_pct = abs((c - o) / o * 100.0) if o > 0 else 0.0
        volume_ratio = (v / avg_volume) if avg_volume > 0 else 0.0
        volume_z = ((v - avg_volume) / std_volume) if std_volume > 0 else 0.0
        move_z = ((move_pct - avg_move) / std_move) if std_move > 0 else 0.0

        tags: List[str] = []
        if volume_ratio >= VOLUME_SPIKE_MULT_THRESHOLD or volume_z >= MOVE_Z_SCORE_THRESHOLD:
            tags.append("volume_spike")
        if float(shape.get("upper_wick_pct") or 0.0) >= NEEDLE_WICK_PCT_THRESHOLD:
            tags.append("up_needle")
        if float(shape.get("lower_wick_pct") or 0.0) >= NEEDLE_WICK_PCT_THRESHOLD:
            tags.append("down_needle")
        if (c > o) and (move_pct >= max(MOVE_MIN_PCT_THRESHOLD, avg_move + max(std_move * MOVE_Z_SCORE_THRESHOLD, 0.0))):
            tags.append("sudden_pump")
        if (c < o) and (move_pct >= max(MOVE_MIN_PCT_THRESHOLD, avg_move + max(std_move * MOVE_Z_SCORE_THRESHOLD, 0.0))):
            tags.append("sudden_dump")
        if not tags:
            continue

        ts_ms = 0
        if start_ts > 0:
            ts_ms = int(start_ts + (idx * interval_ms))

        anomalies.append(
            {
                "idx": idx,
                "time_ms": ts_ms if ts_ms > 0 else None,
                "open": _round_or_none(o),
                "high": _round_or_none(h),
                "low": _round_or_none(l),
                "close": _round_or_none(c),
                "volume": _round_or_none(v),
                "move_pct": _round_or_none(move_pct),
                "volume_ratio_vs_avg": _round_or_none(volume_ratio),
                "upper_wick_pct": shape.get("upper_wick_pct"),
                "lower_wick_pct": shape.get("lower_wick_pct"),
                "tags": tags,
            }
        )

    return anomalies[:12]


def _summarize_timeframe(rows: List[Any], *, timeframe: str = "", start_ts: int = 0, interval_ms: int = 0) -> Dict[str, Any]:
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

    recent_2h_anomaly_candles = _detect_recent_2h_anomaly_candles(
        candles,
        timeframe=timeframe,
        start_ts=int(start_ts or 0),
        interval_ms=int(interval_ms or 0),
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
        "recent_2h_anomaly_candles": recent_2h_anomaly_candles,
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


def _build_kline_artifacts(normalized_payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    timeframe_bars = dict(normalized_payload.get("timeframe_bars") or {}) if isinstance(normalized_payload.get("timeframe_bars"), dict) else {}

    kline_summary: Dict[str, Any] = {}
    original_rows_count: Dict[str, int] = {}
    removed_timeframe_rows_fields: List[str] = []
    invalid_ohlc_timeframes: List[str] = []
    too_many_zero_volume_timeframes: List[str] = []
    insufficient_rows_timeframes: List[str] = []
    warnings: List[str] = []

    for timeframe in TIMEFRAMES:
        timeframe_payload = timeframe_bars.get(timeframe)
        rows = _extract_rows(timeframe_payload)
        start_ts = 0
        interval_ms = 0
        if isinstance(timeframe_payload, dict):
            start_ts = int(_to_float_or_none(timeframe_payload.get("start_ts")) or 0)
            interval_ms = int(_to_float_or_none(timeframe_payload.get("interval_ms")) or 0)
        original_rows_count[timeframe] = len(rows)
        if rows:
            removed_timeframe_rows_fields.append(f"timeframe_bars.{timeframe}.rows")
        summary = _summarize_timeframe(rows, timeframe=timeframe, start_ts=start_ts, interval_ms=interval_ms)
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

    data_quality = {
        "valid": len(invalid_ohlc_timeframes) == 0 and len(insufficient_rows_timeframes) == 0,
        "warnings": warnings,
        "invalid_ohlc_timeframes": invalid_ohlc_timeframes,
        "too_many_zero_volume_timeframes": too_many_zero_volume_timeframes,
        "insufficient_rows_timeframes": insufficient_rows_timeframes,
    }
    compression_info = {
        "raw_timeframe_bars_removed": True,
        "kept_recent_candles_per_timeframe": KEPT_RECENT_CANDLES,
        "original_rows_count": original_rows_count,
        "compressed": True,
        "removed_fields": removed_timeframe_rows_fields,
        "processing_notes": [
            "timeframe_bars.rows removed from outbound payload",
            "kline_summary generated with fixed formulas only",
            "multi_timeframe/levels/liquidity_context/derivatives_context/risk/portfolio/execution_policy/constraints preserved",
            "recent_candles_compact keeps only the latest 5 candles per timeframe",
        ],
    }
    return kline_summary, data_quality, compression_info


def apply_kline_preprocessing_to_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    source_payload = dict(payload or {})
    normalized_payload = _normalize_missing(source_payload)
    compact_payload = dict(source_payload)
    current_price = _to_float_or_none(source_payload.get("current_price"))

    levels_raw = dict(source_payload.get("levels") or {}) if isinstance(source_payload.get("levels"), dict) else {}
    levels = dict(levels_raw)
    levels["distance_to_support_pct"] = _distance_pct(current_price, levels.get("nearest_support"), is_support=True)
    levels["distance_to_resistance_pct"] = _distance_pct(current_price, levels.get("nearest_resistance"), is_support=False)
    compact_payload["levels"] = levels

    kline_summary, data_quality, compression_info = _build_kline_artifacts(normalized_payload)
    compact_payload.pop("timeframe_bars", None)
    compact_payload["kline_summary"] = kline_summary
    compact_payload["liquidity_context"] = _verify_liquidity(source_payload.get("liquidity_context"))
    compact_payload["data_quality"] = data_quality
    compact_payload["compression_info"] = compression_info
    return compact_payload


def build_compact_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized_payload = _normalize_missing(payload)
    current_price = _to_float_or_none(normalized_payload.get("current_price"))
    levels_raw = dict(normalized_payload.get("levels") or {}) if isinstance(normalized_payload.get("levels"), dict) else {}
    levels = {
        "nearest_support": _to_float_or_none(levels_raw.get("nearest_support")),
        "nearest_resistance": _to_float_or_none(levels_raw.get("nearest_resistance")),
        "support_levels": list(levels_raw.get("support_levels") or []) if isinstance(levels_raw.get("support_levels"), list) else [],
        "resistance_levels": list(levels_raw.get("resistance_levels") or []) if isinstance(levels_raw.get("resistance_levels"), list) else [],
        "recent_high": _to_float_or_none(levels_raw.get("recent_high")),
        "recent_low": _to_float_or_none(levels_raw.get("recent_low")),
        "distance_to_support_pct": _distance_pct(current_price, levels_raw.get("nearest_support"), is_support=True),
        "distance_to_resistance_pct": _distance_pct(current_price, levels_raw.get("nearest_resistance"), is_support=False),
    }
    kline_summary, data_quality, compression_info = _build_kline_artifacts(normalized_payload)

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
        "data_quality": data_quality,
        "compression_info": compression_info,
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
