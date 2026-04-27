from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Tuple

import pandas as pd
import pandas_ta as ta


TIMEFRAME_LIMITS: Dict[str, int] = {
    "1m": 120,
    "5m": 120,
    "15m": 120,
    "1h": 120,
    "4h": 120,
    "1d": 90,
}


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


def clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def linear_score(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp(((safe_float(value, low) - low) / (high - low)) * 100.0, 0.0, 100.0)


def percentile_rank(series: List[float], value: float) -> float:
    clean = sorted([safe_float(x, 0.0) for x in series if math.isfinite(safe_float(x, 0.0))])
    if not clean:
        return 50.0
    count = sum(1 for x in clean if x <= value)
    return clamp((count / max(len(clean), 1)) * 100.0, 0.0, 100.0)


def compact_symbol(symbol: str) -> str:
    return str(symbol or "").replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")


def base_asset(symbol: str) -> str:
    token = compact_symbol(symbol)
    if token.endswith("USDT"):
        token = token[:-4]
    return token.upper()


def serialize_bars(df: pd.DataFrame, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if df is None or df.empty:
        return rows
    for _, row in df.tail(max(limit, 1)).iterrows():
        rows.append(
            {
                "time": safe_int(row.get("t"), 0),
                "open": safe_float(row.get("o"), 0.0),
                "high": safe_float(row.get("h"), 0.0),
                "low": safe_float(row.get("l"), 0.0),
                "close": safe_float(row.get("c"), 0.0),
                "volume": max(safe_float(row.get("v"), 0.0), 0.0),
            }
        )
    return rows


class PreBreakoutCandidateScanner:
    def __init__(self, exchange: Any):
        self.exchange = exchange
        self.order_book_history: Dict[str, List[Dict[str, Any]]] = {}
        self.open_interest_history: Dict[str, List[Dict[str, Any]]] = {}

    def _fetch_snapshot(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        markets = dict(self.exchange.load_markets() or {})
        tickers = dict(self.exchange.fetch_tickers() or {})
        return markets, tickers

    def _is_eligible_market(self, market: Dict[str, Any]) -> bool:
        if not isinstance(market, dict):
            return False
        if not bool(market.get("active", True)):
            return False
        if bool(market.get("spot", False)):
            return False
        if not (bool(market.get("swap", False)) or bool(market.get("future", False))):
            return False
        settle = str(market.get("settle") or "").upper()
        quote = str(market.get("quote") or "").upper()
        symbol = str(market.get("symbol") or "")
        return symbol.endswith("/USDT:USDT") or settle == "USDT" or quote == "USDT"

    def _tick_size(self, market: Dict[str, Any]) -> float:
        tick = safe_float(((market.get("precision") or {}).get("price")), 0.0)
        if tick > 0 and tick < 1:
            return tick
        if tick >= 1:
            return 10 ** (-int(tick))
        info = dict(market.get("info") or {})
        for row in list(info.get("filters") or []):
            if str(row.get("filterType") or "").upper() == "PRICE_FILTER":
                parsed = safe_float(row.get("tickSize"), 0.0)
                if parsed > 0:
                    return parsed
        return 0.0

    def _fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        rows = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not rows:
            return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v"])
        df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"])
        for col in ["o", "h", "l", "c", "v"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna().sort_values("t").reset_index(drop=True)
        return df

    def _fetch_all_timeframes(self, symbol: str) -> Dict[str, pd.DataFrame]:
        result: Dict[str, pd.DataFrame] = {}
        for tf, limit in TIMEFRAME_LIMITS.items():
            result[tf] = self._fetch_ohlcv(symbol, tf, limit)
        return result

    def _nearest_snapshot(self, history: List[Dict[str, Any]], now_ts: float, seconds: int) -> Dict[str, Any]:
        target = now_ts - max(seconds, 1)
        older = [row for row in history if safe_float(row.get("ts"), 0.0) <= target]
        return dict(older[-1]) if older else {}

    def _order_book_context(self, symbol: str, last_price: float) -> Dict[str, Any]:
        now = time.time()
        book = dict(self.exchange.fetch_order_book(symbol, limit=20) or {})
        bids = list(book.get("bids") or [])
        asks = list(book.get("asks") or [])
        bid_depth_10 = sum(safe_float(row[1], 0.0) for row in bids[:10])
        ask_depth_10 = sum(safe_float(row[1], 0.0) for row in asks[:10])
        top_5_bid = sum(safe_float(row[1], 0.0) for row in bids[:5])
        top_5_ask = sum(safe_float(row[1], 0.0) for row in asks[:5])
        denom = max(bid_depth_10 + ask_depth_10, 1e-9)
        depth_imbalance_10 = (bid_depth_10 - ask_depth_10) / denom
        largest_bid_wall = max(bids[:10], key=lambda row: safe_float(row[1], 0.0)) if bids else [0.0, 0.0]
        largest_ask_wall = max(asks[:10], key=lambda row: safe_float(row[1], 0.0)) if asks else [0.0, 0.0]
        best_bid = safe_float(bids[0][0], 0.0) if bids else 0.0
        best_ask = safe_float(asks[0][0], 0.0) if asks else 0.0
        spread_pct = ((best_ask - best_bid) / max(last_price, 1e-9)) * 100.0 if best_bid > 0 and best_ask > 0 and last_price > 0 else 99.0
        snapshot = {
            "ts": now,
            "last_price": last_price,
            "depth_imbalance_10": depth_imbalance_10,
            "largest_bid_wall_price": safe_float(largest_bid_wall[0], 0.0),
            "largest_bid_wall_size": safe_float(largest_bid_wall[1], 0.0),
            "largest_ask_wall_price": safe_float(largest_ask_wall[0], 0.0),
            "largest_ask_wall_size": safe_float(largest_ask_wall[1], 0.0),
        }
        history = list(self.order_book_history.get(symbol) or [])
        history.append(snapshot)
        history = [row for row in history if (now - safe_float(row.get("ts"), 0.0)) <= 8 * 60][-80:]
        self.order_book_history[symbol] = history
        prev_1m = self._nearest_snapshot(history, now, 60)
        prev_3m = self._nearest_snapshot(history, now, 180)
        prev_5m = self._nearest_snapshot(history, now, 300)
        depth_change_1m = depth_imbalance_10 - safe_float(prev_1m.get("depth_imbalance_10"), depth_imbalance_10)
        depth_change_5m = depth_imbalance_10 - safe_float(prev_5m.get("depth_imbalance_10"), depth_imbalance_10)
        prev_bid_size = safe_float(prev_3m.get("largest_bid_wall_size"), safe_float(largest_bid_wall[1], 0.0))
        prev_ask_size = safe_float(prev_3m.get("largest_ask_wall_size"), safe_float(largest_ask_wall[1], 0.0))
        bid_wall_change_3m_pct = ((safe_float(largest_bid_wall[1], 0.0) - prev_bid_size) / max(prev_bid_size, 1e-9)) * 100.0 if prev_bid_size > 0 else 0.0
        ask_wall_change_3m_pct = ((safe_float(largest_ask_wall[1], 0.0) - prev_ask_size) / max(prev_ask_size, 1e-9)) * 100.0 if prev_ask_size > 0 else 0.0
        bid_wall_following_price = (
            safe_float(snapshot.get("largest_bid_wall_price"), 0.0) > safe_float(prev_1m.get("largest_bid_wall_price"), 0.0)
            and safe_float(snapshot.get("last_price"), 0.0) >= safe_float(prev_1m.get("last_price"), 0.0)
        )
        ask_wall_getting_thinner = ask_wall_change_3m_pct <= -10.0
        wall_pull_or_spoof_risk = abs(ask_wall_change_3m_pct) >= 35.0 or abs(bid_wall_change_3m_pct) >= 35.0
        return {
            "bid_depth_10": round(bid_depth_10, 4),
            "ask_depth_10": round(ask_depth_10, 4),
            "depth_imbalance_10": round(depth_imbalance_10, 6),
            "largest_bid_wall_price": round(safe_float(largest_bid_wall[0], 0.0), 8),
            "largest_bid_wall_size": round(safe_float(largest_bid_wall[1], 0.0), 4),
            "largest_ask_wall_price": round(safe_float(largest_ask_wall[0], 0.0), 8),
            "largest_ask_wall_size": round(safe_float(largest_ask_wall[1], 0.0), 4),
            "top_5_bid_liquidity": round(top_5_bid, 4),
            "top_5_ask_liquidity": round(top_5_ask, 4),
            "spread_pct": round(spread_pct, 6),
            "depth_imbalance_change_1m": round(depth_change_1m, 6),
            "depth_imbalance_change_5m": round(depth_change_5m, 6),
            "largest_bid_wall_size_change_3m_pct": round(bid_wall_change_3m_pct, 4),
            "largest_ask_wall_size_change_3m_pct": round(ask_wall_change_3m_pct, 4),
            "bid_wall_following_price": bool(bid_wall_following_price),
            "ask_wall_getting_thinner": bool(ask_wall_getting_thinner),
            "wall_pull_or_spoof_risk": bool(wall_pull_or_spoof_risk),
        }

    def _flow_context(self, symbol: str) -> Dict[str, Any]:
        trades = list(self.exchange.fetch_trades(symbol, limit=500) or [])
        now_ms = int(time.time() * 1000)
        buy_1m = sell_1m = buy_5m = sell_5m = buy_15m = sell_15m = 0.0
        all_notional: List[float] = []
        large_buy_1m = large_sell_1m = 0.0
        for trade in trades:
            amount = safe_float(trade.get("amount"), 0.0)
            price = safe_float(trade.get("price"), 0.0)
            notional = amount * price
            if notional <= 0:
                continue
            all_notional.append(notional)
        threshold = sorted(all_notional)[int(len(all_notional) * 0.9)] if all_notional else 0.0
        threshold = max(threshold, 25_000.0)
        large_count_1m = 0
        for trade in trades:
            amount = safe_float(trade.get("amount"), 0.0)
            price = safe_float(trade.get("price"), 0.0)
            notional = amount * price
            side = str(trade.get("side") or "").lower()
            ts = safe_int(trade.get("timestamp"), 0)
            age = max(now_ms - ts, 0)
            if age <= 60_000:
                if side == "buy":
                    buy_1m += notional
                elif side == "sell":
                    sell_1m += notional
                if notional >= threshold:
                    large_count_1m += 1
                    if side == "buy":
                        large_buy_1m += notional
                    elif side == "sell":
                        large_sell_1m += notional
            if age <= 5 * 60_000:
                if side == "buy":
                    buy_5m += notional
                elif side == "sell":
                    sell_5m += notional
            if age <= 15 * 60_000:
                if side == "buy":
                    buy_15m += notional
                elif side == "sell":
                    sell_15m += notional
        cvd_1m = buy_1m - sell_1m
        cvd_5m = buy_5m - sell_5m
        cvd_15m = buy_15m - sell_15m
        large_trade_net = "buy" if large_buy_1m > large_sell_1m else "sell" if large_sell_1m > large_buy_1m else "neutral"
        return {
            "market_buy_notional_1m": round(buy_1m, 4),
            "market_sell_notional_1m": round(sell_1m, 4),
            "market_buy_sell_ratio_1m": round(buy_1m / max(sell_1m, 1e-9), 4) if (buy_1m > 0 or sell_1m > 0) else 1.0,
            "market_buy_sell_ratio_5m": round(buy_5m / max(sell_5m, 1e-9), 4) if (buy_5m > 0 or sell_5m > 0) else 1.0,
            "cvd_notional_1m": round(cvd_1m, 4),
            "cvd_notional_5m": round(cvd_5m, 4),
            "cvd_slope_1m": round(cvd_1m, 4),
            "cvd_slope_5m": round(cvd_5m / 5.0, 4),
            "cvd_slope_15m": round(cvd_15m / 15.0, 4),
            "large_trade_count_1m": int(large_count_1m),
            "large_trade_net_side": large_trade_net,
        }

    def _oi_change_pct(self, symbol: str, now_ts: float, seconds: int, current_oi_value: float) -> float:
        history = list(self.open_interest_history.get(symbol) or [])
        prev = [row for row in history if safe_float(row.get("ts"), 0.0) <= (now_ts - seconds)]
        if not prev:
            return 0.0
        prev_val = safe_float(prev[-1].get("oi_value"), 0.0)
        if prev_val <= 0 or current_oi_value <= 0:
            return 0.0
        return ((current_oi_value / prev_val) - 1.0) * 100.0

    def _derivatives_context(self, symbol: str, ticker: Dict[str, Any]) -> Dict[str, Any]:
        funding_rate = 0.0
        try:
            fetch_funding = getattr(self.exchange, "fetch_funding_rate", None)
            if callable(fetch_funding):
                fr = dict(fetch_funding(symbol) or {})
                funding_rate = safe_float(fr.get("fundingRate", fr.get("funding_rate")), 0.0)
        except Exception:
            funding_rate = 0.0
        mark_price = safe_float((ticker.get("info") or {}).get("markPrice"), safe_float(ticker.get("last"), 0.0))
        index_price = safe_float((ticker.get("info") or {}).get("indexPrice"), safe_float(ticker.get("last"), 0.0))
        basis_pct = ((mark_price - index_price) / max(index_price, 1e-9)) * 100.0 if mark_price > 0 and index_price > 0 else 0.0
        open_interest_value = 0.0
        try:
            fetch_oi = getattr(self.exchange, "fetch_open_interest", None)
            if callable(fetch_oi):
                oi = dict(fetch_oi(symbol) or {})
                open_interest_value = safe_float(oi.get("openInterestValue", oi.get("value")), 0.0)
                if open_interest_value <= 0:
                    open_interest = safe_float(oi.get("openInterestAmount", oi.get("openInterest", oi.get("amount"))), 0.0)
                    open_interest_value = open_interest * max(mark_price, 0.0)
        except Exception:
            open_interest_value = 0.0
        now = time.time()
        history = list(self.open_interest_history.get(symbol) or [])
        history.append({"ts": now, "oi_value": open_interest_value})
        history = [row for row in history if (now - safe_float(row.get("ts"), 0.0)) <= (2 * 3600)][-300:]
        self.open_interest_history[symbol] = history
        return {
            "funding_rate": round(funding_rate, 8),
            "basis_pct": round(basis_pct, 6),
            "open_interest_value_usdt": round(open_interest_value, 4),
            "open_interest_change_pct_5m": round(self._oi_change_pct(symbol, now, 5 * 60, open_interest_value), 4),
            "open_interest_change_pct_15m": round(self._oi_change_pct(symbol, now, 15 * 60, open_interest_value), 4),
            "open_interest_change_pct_1h": round(self._oi_change_pct(symbol, now, 60 * 60, open_interest_value), 4),
        }

    def _compression_structure_features(self, frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        df15 = frames.get("15m")
        df5 = frames.get("5m")
        if df15 is None or df15.empty or len(df15) < 40:
            raise ValueError("insufficient_15m")
        c15 = df15["c"]
        h15 = df15["h"]
        l15 = df15["l"]
        v15 = df15["v"]
        last_close = safe_float(c15.iloc[-1], 0.0)
        atr_series = ta.atr(h15, l15, c15, length=14)
        atr_price = max(safe_float(atr_series.iloc[-1], 0.0), last_close * 0.0015 if last_close > 0 else 0.0)
        bb = ta.bbands(c15, length=20, std=2.0)
        if isinstance(bb, pd.DataFrame) and not bb.empty:
            upper = bb.iloc[:, 2]
            lower = bb.iloc[:, 0]
            bb_width_series = ((upper - lower) / c15.replace(0, pd.NA)) * 100.0
        else:
            bb_width_series = pd.Series([0.0] * len(df15))
        bb_width_pct_15m = safe_float(bb_width_series.iloc[-1], 0.0)
        atr_pct_series = (atr_series / c15.replace(0, pd.NA)) * 100.0
        bb_width_percentile = percentile_rank(list(bb_width_series.tail(120)), bb_width_pct_15m)
        atr_percentile = percentile_rank(list(atr_pct_series.tail(120)), safe_float(atr_pct_series.iloc[-1], 0.0))
        lookback = 24
        recent_high = safe_float(h15.tail(lookback).max(), last_close)
        recent_low = safe_float(l15.tail(lookback).min(), last_close)
        range_span = max(recent_high - recent_low, 1e-9)
        range_mid = recent_low + range_span / 2.0
        range_tightness_atr = range_span / max(atr_price, 1e-9)
        inside_bar_count = 0
        higher_lows_count = 0
        lower_highs_count = 0
        lows = list(l15.tail(8))
        highs = list(h15.tail(8))
        for idx in range(1, len(df15.tail(12))):
            prev = df15.tail(12).iloc[idx - 1]
            curr = df15.tail(12).iloc[idx]
            if safe_float(curr.get("h"), 0.0) <= safe_float(prev.get("h"), 0.0) and safe_float(curr.get("l"), 0.0) >= safe_float(prev.get("l"), 0.0):
                inside_bar_count += 1
        for idx in range(1, len(lows)):
            if safe_float(lows[idx], 0.0) > safe_float(lows[idx - 1], 0.0):
                higher_lows_count += 1
        for idx in range(1, len(highs)):
            if safe_float(highs[idx], 0.0) < safe_float(highs[idx - 1], 0.0):
                lower_highs_count += 1
        recent_volume_avg = safe_float(v15.tail(6).mean(), 0.0)
        prior_volume_avg = safe_float(v15.tail(30).head(24).mean(), recent_volume_avg)
        volume_contraction_ratio = recent_volume_avg / max(prior_volume_avg, 1e-9) if prior_volume_avg > 0 else 1.0
        close_position_in_range = ((last_close - recent_low) / max(range_span, 1e-9))
        distance_to_high_atr = max((recent_high - last_close) / max(atr_price, 1e-9), 0.0)
        distance_to_low_atr = max((last_close - recent_low) / max(atr_price, 1e-9), 0.0)
        breakout_level = recent_high
        breakdown_level = recent_low
        distance_to_breakout_atr = max((breakout_level - last_close) / max(atr_price, 1e-9), 0.0)
        distance_to_breakdown_atr = max((last_close - breakdown_level) / max(atr_price, 1e-9), 0.0)
        ema20 = ta.ema(c15, length=20)
        already_extended = abs(last_close - safe_float(ema20.iloc[-1], last_close)) / max(atr_price, 1e-9) if isinstance(ema20, pd.Series) and not ema20.empty else 0.0
        range_duration = 0
        for row in reversed(list(c15.tail(30))):
            if recent_low <= safe_float(row, 0.0) <= recent_high:
                range_duration += 1
            else:
                break
        return {
            "atr_price_15m": round(atr_price, 8),
            "bb_width_pct_15m": round(bb_width_pct_15m, 6),
            "bb_width_percentile_15m_120": round(bb_width_percentile, 4),
            "atr_percentile_15m_120": round(atr_percentile, 4),
            "range_tightness_atr_15m": round(range_tightness_atr, 4),
            "range_duration_bars_15m": int(range_duration),
            "inside_bar_count_15m": int(inside_bar_count),
            "higher_lows_count_15m": int(higher_lows_count),
            "lower_highs_count_15m": int(lower_highs_count),
            "volume_contraction_ratio_15m": round(volume_contraction_ratio, 4),
            "range_high_15m": round(recent_high, 8),
            "range_low_15m": round(recent_low, 8),
            "range_mid_15m": round(range_mid, 8),
            "close_position_in_range_15m": round(clamp(close_position_in_range, 0.0, 1.0), 4),
            "distance_to_range_high_atr15m": round(distance_to_high_atr, 4),
            "distance_to_range_low_atr15m": round(distance_to_low_atr, 4),
            "breakout_level": round(breakout_level, 8),
            "breakdown_level": round(breakdown_level, 8),
            "distance_to_breakout_atr15m": round(distance_to_breakout_atr, 4),
            "distance_to_breakdown_atr15m": round(distance_to_breakdown_atr, 4),
            "already_extended_atr15m": round(already_extended, 4),
            "last_close_15m": round(last_close, 8),
        }

    def _side_hint(self, features: Dict[str, Any], flow: Dict[str, Any], order_book: Dict[str, Any], derivatives: Dict[str, Any]) -> str:
        long_votes = 0
        short_votes = 0
        if safe_int(features.get("higher_lows_count_15m"), 0) >= 4:
            long_votes += 1
        if safe_int(features.get("lower_highs_count_15m"), 0) >= 4:
            short_votes += 1
        if safe_float(features.get("close_position_in_range_15m"), 0.5) >= 0.55:
            long_votes += 1
        if safe_float(features.get("close_position_in_range_15m"), 0.5) <= 0.45:
            short_votes += 1
        dist_breakout = safe_float(features.get("distance_to_breakout_atr15m"), 99.0)
        dist_breakdown = safe_float(features.get("distance_to_breakdown_atr15m"), 99.0)
        if 0.20 <= dist_breakout <= 0.80:
            long_votes += 1
        if 0.20 <= dist_breakdown <= 0.80:
            short_votes += 1
        if safe_float(flow.get("cvd_slope_1m"), 0.0) > 0 and safe_float(flow.get("cvd_slope_5m"), 0.0) > 0:
            long_votes += 1
        if safe_float(flow.get("cvd_slope_1m"), 0.0) < 0 and safe_float(flow.get("cvd_slope_5m"), 0.0) < 0:
            short_votes += 1
        if safe_float(flow.get("market_buy_sell_ratio_5m"), 1.0) > 1.1:
            long_votes += 1
        if safe_float(flow.get("market_buy_sell_ratio_5m"), 1.0) < 0.9:
            short_votes += 1
        if bool(order_book.get("ask_wall_getting_thinner", False)):
            long_votes += 1
        if bool(order_book.get("bid_wall_following_price", False)):
            long_votes += 1
        if safe_float(order_book.get("largest_bid_wall_size_change_3m_pct"), 0.0) <= -10.0:
            short_votes += 1
        if safe_float(order_book.get("largest_ask_wall_size_change_3m_pct"), 0.0) >= 10.0:
            short_votes += 1
        funding_rate = abs(safe_float(derivatives.get("funding_rate"), 0.0))
        oi_change = safe_float(derivatives.get("open_interest_change_pct_15m"), 0.0)
        if funding_rate <= 0.001 and oi_change > 0:
            long_votes += 1
            short_votes += 1
        if long_votes >= 5 and long_votes > short_votes:
            return "long"
        if short_votes >= 5 and short_votes > long_votes:
            return "short"
        base_hint = str(features.get("prebreakout_side_hint") or "neutral").lower()
        return base_hint if base_hint in ("long", "short") else "neutral"

    def _hard_filters(
        self,
        market: Dict[str, Any],
        ticker: Dict[str, Any],
        features: Dict[str, Any],
        order_book: Dict[str, Any],
        *,
        fixed_order_notional_usdt: float,
    ) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        quote_volume = safe_float(ticker.get("quoteVolume"), 0.0)
        if quote_volume < 20_000_000.0:
            reasons.append("low_quote_volume_24h")
        spread_pct = safe_float(order_book.get("spread_pct"), 99.0)
        if spread_pct > 0.08:
            reasons.append("spread_too_wide")
        if not bool(market.get("active", True)):
            reasons.append("market_not_active")
        atr_price = safe_float(features.get("atr_price_15m"), 0.0)
        tick_size = self._tick_size(market)
        if atr_price <= max(tick_size * 8.0, 1e-12):
            reasons.append("atr_too_small_vs_tick")
        if safe_float(features.get("already_extended_atr15m"), 0.0) > 1.5:
            reasons.append("already_extended_over_1p5_atr")
        dist_high = safe_float(features.get("distance_to_range_high_atr15m"), 99.0)
        dist_low = safe_float(features.get("distance_to_range_low_atr15m"), 99.0)
        close_15m = safe_float(features.get("last_close_15m"), 0.0)
        if min(dist_high, dist_low) < 0.10 and close_15m > 0:
            reasons.append("too_close_to_range_edge")
        top_5_bid = safe_float(order_book.get("top_5_bid_liquidity"), 0.0)
        top_5_ask = safe_float(order_book.get("top_5_ask_liquidity"), 0.0)
        ref_price = max(close_15m, safe_float(ticker.get("last"), 0.0), 1e-9)
        bid_notional = top_5_bid * ref_price
        ask_notional = top_5_ask * ref_price
        if min(bid_notional, ask_notional) < max(fixed_order_notional_usdt, 1.0) * 3.0:
            reasons.append("order_book_too_thin_for_notional")
        return (len(reasons) == 0), reasons

    def _score(
        self,
        ticker: Dict[str, Any],
        features: Dict[str, Any],
        order_book: Dict[str, Any],
        flow: Dict[str, Any],
        derivatives: Dict[str, Any],
        side_hint: str,
        disqualify_reasons: List[str],
    ) -> Dict[str, float]:
        quote_volume = safe_float(ticker.get("quoteVolume"), 0.0)
        spread_pct = safe_float(order_book.get("spread_pct"), 0.0)
        top_depth = min(safe_float(order_book.get("top_5_bid_liquidity"), 0.0), safe_float(order_book.get("top_5_ask_liquidity"), 0.0))
        liquidity_score = (
            linear_score(quote_volume, 20_000_000.0, 300_000_000.0) * 0.50
            + (100.0 - linear_score(spread_pct, 0.01, 0.08)) * 0.30
            + linear_score(top_depth, 1000.0, 50000.0) * 0.20
        )
        compression_score = (
            (100.0 - safe_float(features.get("bb_width_percentile_15m_120"), 50.0)) * 0.45
            + (100.0 - safe_float(features.get("atr_percentile_15m_120"), 50.0)) * 0.30
            + linear_score(1.1 - safe_float(features.get("volume_contraction_ratio_15m"), 1.0), 0.0, 0.8) * 0.25
        )
        close_pos = safe_float(features.get("close_position_in_range_15m"), 0.5)
        structure_base = 50.0 + (close_pos - 0.5) * 100.0
        if side_hint == "short":
            structure_base = 50.0 + ((0.5 - close_pos) * 100.0)
        structure_score = clamp(
            structure_base
            + safe_int(features.get("higher_lows_count_15m"), 0) * (4.0 if side_hint == "long" else -2.0)
            + safe_int(features.get("lower_highs_count_15m"), 0) * (4.0 if side_hint == "short" else -2.0),
            0.0,
            100.0,
        )
        flow_bias = (
            safe_float(flow.get("market_buy_sell_ratio_5m"), 1.0) - 1.0
            + (safe_float(flow.get("cvd_slope_5m"), 0.0) / max(abs(safe_float(flow.get("cvd_notional_5m"), 1.0)), 1.0))
        )
        if side_hint == "short":
            flow_bias = -flow_bias
        flow_score = clamp(linear_score(flow_bias, -0.4, 0.8), 0.0, 100.0)
        funding_abs = abs(safe_float(derivatives.get("funding_rate"), 0.0))
        oi_change = safe_float(derivatives.get("open_interest_change_pct_15m"), 0.0)
        derivatives_score = clamp(
            (100.0 - linear_score(funding_abs, 0.0001, 0.0012)) * 0.45
            + linear_score(oi_change, -2.0, 5.0) * 0.55,
            0.0,
            100.0,
        )
        if side_hint == "long":
            trigger_value = safe_float(features.get("distance_to_breakout_atr15m"), 99.0)
        elif side_hint == "short":
            trigger_value = safe_float(features.get("distance_to_breakdown_atr15m"), 99.0)
        else:
            trigger_value = min(
                safe_float(features.get("distance_to_breakout_atr15m"), 99.0),
                safe_float(features.get("distance_to_breakdown_atr15m"), 99.0),
            )
        trigger_distance_score = clamp(100.0 - abs(trigger_value - 0.5) * 150.0, 0.0, 100.0)
        penalty_score = 0.0
        if bool(order_book.get("wall_pull_or_spoof_risk", False)):
            penalty_score += 8.0
        if safe_float(features.get("already_extended_atr15m"), 0.0) > 1.2:
            penalty_score += 6.0
        penalty_score += min(len(disqualify_reasons) * 4.0, 24.0)
        return {
            "liquidity_score": round(clamp(liquidity_score, 0.0, 100.0), 4),
            "compression_score": round(clamp(compression_score, 0.0, 100.0), 4),
            "structure_score": round(clamp(structure_score, 0.0, 100.0), 4),
            "flow_score": round(clamp(flow_score, 0.0, 100.0), 4),
            "derivatives_score": round(clamp(derivatives_score, 0.0, 100.0), 4),
            "trigger_distance_score": round(clamp(trigger_distance_score, 0.0, 100.0), 4),
            "penalty_score": round(clamp(penalty_score, 0.0, 100.0), 4),
        }

    def _prebreakout_score(self, scores: Dict[str, float]) -> float:
        return round(
            (
                safe_float(scores.get("liquidity_score"), 0.0) * 0.18
                + safe_float(scores.get("compression_score"), 0.0) * 0.22
                + safe_float(scores.get("structure_score"), 0.0) * 0.18
                + safe_float(scores.get("flow_score"), 0.0) * 0.22
                + safe_float(scores.get("derivatives_score"), 0.0) * 0.12
                + safe_float(scores.get("trigger_distance_score"), 0.0) * 0.08
                - safe_float(scores.get("penalty_score"), 0.0)
            ),
            4,
        )

    def run(
        self,
        *,
        top_pick: int = 3,
        symbol_limit: int = 24,
        fixed_order_notional_usdt: float = 40.0,
        universe_label: str = "bitget_usdt_perp",
        markets: Dict[str, Any] | None = None,
        tickers: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        scanner_ts = int(time.time())
        if not isinstance(markets, dict) or not isinstance(tickers, dict):
            markets, tickers = self._fetch_snapshot()
        universe: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        for symbol, market in markets.items():
            if not self._is_eligible_market(market):
                continue
            ticker = dict(tickers.get(symbol) or {})
            quote_volume = safe_float(ticker.get("quoteVolume"), 0.0)
            if quote_volume <= 0:
                continue
            universe.append((symbol, dict(market), ticker))
        universe = sorted(universe, key=lambda row: safe_float(row[2].get("quoteVolume"), 0.0), reverse=True)
        scoped_universe = universe[: max(int(symbol_limit), int(top_pick) * 2)]
        analyzed: List[Dict[str, Any]] = []
        payload_by_symbol: Dict[str, Dict[str, Any]] = {}
        for symbol, market, ticker in scoped_universe:
            try:
                frames = self._fetch_all_timeframes(symbol)
                features = self._compression_structure_features(frames)
                flow = self._flow_context(symbol)
                last_price = safe_float(ticker.get("last"), safe_float(features.get("last_close_15m"), 0.0))
                order_book = self._order_book_context(symbol, last_price)
                derivatives = self._derivatives_context(symbol, ticker)
                allow, disqualify_reasons = self._hard_filters(
                    market,
                    ticker,
                    features,
                    order_book,
                    fixed_order_notional_usdt=fixed_order_notional_usdt,
                )
                change_24h_pct = safe_float(ticker.get("percentage"), 0.0)
                quote_volume_24h = safe_float(ticker.get("quoteVolume"), 0.0)
                raw_payload = {
                    "symbol": symbol,
                    "scanner_ts": scanner_ts,
                    "ticker_24h": {
                        "symbol": symbol,
                        "lastPrice": round(last_price, 8),
                        "priceChangePercent": round(safe_float(ticker.get("percentage"), 0.0), 6),
                        "quoteVolume": round(safe_float(ticker.get("quoteVolume"), 0.0), 4),
                        "weightedAvgPrice": round(safe_float((ticker.get("info") or {}).get("weightedAvgPrice"), last_price), 8),
                    },
                    "timeframe_bars": {
                        tf: serialize_bars(df, TIMEFRAME_LIMITS.get(tf, 120))
                        for tf, df in frames.items()
                    },
                    "liquidity_context": dict(order_book),
                    "orderbook_history": {
                        "depth_imbalance_change_1m": order_book.get("depth_imbalance_change_1m"),
                        "depth_imbalance_change_5m": order_book.get("depth_imbalance_change_5m"),
                        "largest_bid_wall_size_change_3m_pct": order_book.get("largest_bid_wall_size_change_3m_pct"),
                        "largest_ask_wall_size_change_3m_pct": order_book.get("largest_ask_wall_size_change_3m_pct"),
                        "bid_wall_following_price": order_book.get("bid_wall_following_price"),
                        "ask_wall_getting_thinner": order_book.get("ask_wall_getting_thinner"),
                        "wall_pull_or_spoof_risk": order_book.get("wall_pull_or_spoof_risk"),
                    },
                    "flow_context": dict(flow),
                    "derivatives_context": dict(derivatives),
                    "scanner_features": dict(features),
                }
                row = {
                    "symbol": symbol,
                    "disqualify": not allow,
                    "rank_metric_quote_volume_24h": round(quote_volume_24h, 4),
                    "rank_metric_change_24h_pct": round(change_24h_pct, 4),
                }
                analyzed.append(row)
                payload_by_symbol[symbol] = {
                    "scanner_features": dict(features),
                    "raw_candidate_payload": raw_payload,
                }
            except Exception:
                continue
        ranked = sorted(
            [row for row in analyzed if not bool(row.get("disqualify", False))],
            key=lambda row: (
                safe_float(row.get("rank_metric_quote_volume_24h"), 0.0),
                abs(safe_float(row.get("rank_metric_change_24h_pct"), 0.0)),
            ),
            reverse=True,
        )
        top_candidates: List[Dict[str, Any]] = []
        for idx, row in enumerate(ranked[: max(int(top_pick), 1)], start=1):
            symbol = str(row.get("symbol") or "")
            top_candidates.append(
                {
                    "rank": idx,
                    "symbol": symbol,
                    "rank_metric_quote_volume_24h": round(safe_float(row.get("rank_metric_quote_volume_24h"), 0.0), 4),
                    "rank_metric_change_24h_pct": round(safe_float(row.get("rank_metric_change_24h_pct"), 0.0), 4),
                    "disqualify": False,
                }
            )
        return {
            "scanner_ts": scanner_ts,
            "universe": str(universe_label or "bitget_usdt_perp"),
            "mode": "pre_breakout_discovery",
            "top_candidates": top_candidates,
            "candidate_payloads": payload_by_symbol,
        }
