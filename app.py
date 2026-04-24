import os, sys, ccxt, threading, time, requests, gc, json, math, re
import numpy as np
sys.stdout.reconfigure(line_buffering=True)  # 鍗虫檪 flush logs
import pandas as pd
import pandas_ta as ta
from flask import Flask, render_template, jsonify, request
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from bot_runtime_utils import atomic_json_load, atomic_json_save, prune_mapping, safe_request_json, safe_request_text, snapshot_mapping
from bot_market_guard import MarketDirectionGuard
from bot_storage import BotStorage
import bot_news_disabled
from ai_dataset_guard import build_learning_weights as ext_build_learning_weights, learning_weight_summary as ext_learning_weight_summary
from ai_observer_tools import trigger_hit_leaderboard, neutral_failure_stats
from ai_execution_guard import exchange_quality_snapshot as exec_quality_snapshot, execution_gate, protection_failure_action
from ai_replay_store import save_decision_input_snapshot, load_decision_input_snapshots, ensure_replay_tables
from ai_market_context import build_market_consensus
from ai_session_tools import session_bucket_from_hour, build_session_bias as build_ext_session_bias
from ai_risk_alerts import derive_auto_mode
from ai_decision_intelligence import apply_decision_inertia, detect_market_tempo, classify_exit_type, weighted_trade_stats, recent_setup_loss_streak, confidence_position_multiplier, apply_exit_learning_to_params
from learning_engine import enrich_learning_trade, filter_learning_pool, phase_from_counts, build_decision_fingerprint, execution_quality_bucket
from routes_ai import build_ai_db_stats_payload, build_ai_learning_recent_payload, build_ai_debug_payload, build_ai_learning_health_payload, build_ai_strategy_matrix_payload, build_ai_decision_explain_payload, build_learning_sample_review_payload
from state_service import env_or_blank, build_learning_dataset_meta, DEFAULT_RUNTIME_STATE
from decision_calibrator import calibrate_trade_decision
from execution_engine import execution_score_from_snapshot
from position_engine import apply_position_formula
from signal_engine import build_signal_quality_snapshot
from decision_engine import merge_decision_explain, normalize_decision_summary, build_decision_funnel_payload
from decision_policy import DATASET_POLICY, DECISION_POLICY, RISK_POLICY, EXECUTION_POLICY, get_policy_snapshot
from scheduler import default_thread_specs
from trade_learning_service import LearningTaskQueue
from dashboard_state import state_lite_cache, ai_panel_cache, positions_cache
from api_state_routes import build_state_lite_payload, build_positions_payload, build_ai_panel_payload
from openai_trade_decision import (
    build_candidate_payload as build_openai_trade_candidate,
    build_dashboard_payload as build_openai_trade_dashboard,
    consult_trade_decision,
    default_trade_config,
    load_trade_state,
    save_trade_state,
)

app = Flask(__name__)

# =====================================================
# API 閰嶇疆
# =====================================================
bitget_config = {
    'apiKey':   env_or_blank('BITGET_API_KEY'),
    'secret':   env_or_blank('BITGET_SECRET'),
    'password': env_or_blank('BITGET_PASSWORD'),
    'enableRateLimit': True,
    'options': {'defaultType': 'swap', 'defaultMarginMode': 'cross'}
}
exchange = ccxt.bitget(bitget_config)
exchange.timeout = 10000   # 10绉?API 瓒呮檪锛岀禃涓嶇劇闄愮瓑寰?exchange.enableRateLimit = True
PANIC_API_KEY   = env_or_blank('PANIC_API_KEY')
OPENAI_API_KEY  = env_or_blank('OPENAI_API_KEY')
ANTHROPIC_KEY   = env_or_blank('ANTHROPIC_API_KEY')
OPENAI_TRADE_STATE_PATH = os.path.join(BASE_DIR, 'data', 'openai_trade_state.json')
OPENAI_TRADE_CONFIG = default_trade_config(lambda name, default='': env_or_blank(name, default))
OPENAI_TRADE_STATE = load_trade_state(OPENAI_TRADE_STATE_PATH)
OPENAI_TRADE_LOCK = threading.RLock()
OPENAI_SHORT_GAINERS_ENABLE = str(env_or_blank('OPENAI_SHORT_GAINERS_ENABLE', '1')).strip().lower() not in ('0', 'false', 'no', 'off')
OPENAI_SHORT_GAINERS_LIMIT = max(1, int(float(env_or_blank('OPENAI_SHORT_GAINERS_LIMIT', '3') or 3)))
OPENAI_SHORT_GAINERS_PREFILTER = max(10, int(float(env_or_blank('OPENAI_SHORT_GAINERS_PREFILTER', '80') or 80)))
OPENAI_SHORT_GAINERS_MIN_24H_PCT = max(0.5, float(env_or_blank('OPENAI_SHORT_GAINERS_MIN_24H_PCT', '35.0') or 35.0))
OPENAI_SHORT_GAINERS_MAX_PER_SCAN = max(0, int(float(env_or_blank('OPENAI_SHORT_GAINERS_MAX_PER_SCAN', '1') or 1)))
OPENAI_TRADE_PENDING_RECHECK_MAX_PER_SCAN = max(1, int(float(env_or_blank('OPENAI_TRADE_PENDING_RECHECK_MAX_PER_SCAN', '2') or 2)))
OPENAI_PENDING_RECHECK_MIN_GAP_SEC = max(30, int(float(env_or_blank('OPENAI_PENDING_RECHECK_MIN_GAP_SEC', '45') or 45)))
OPENAI_REVIEW_MAX_ACTIVE_POSITIONS = max(1, int(float(env_or_blank('OPENAI_REVIEW_MAX_ACTIVE_POSITIONS', '15') or 15)))
ORDER_THRESHOLD         = min(float(DECISION_POLICY.get('order_threshold', 60) or 60), 52.0)   # AI涓绘帶鐗堬細璧峰闁€妾绘斁瀵?ORDER_THRESHOLD_DEFAULT = min(float(DECISION_POLICY.get('order_threshold_default', ORDER_THRESHOLD) or ORDER_THRESHOLD), 52.0)   # AI涓绘帶鐗堥爯瑷€?ORDER_THRESHOLD_HIGH    = min(float(DECISION_POLICY.get('order_threshold_high', 80) or 80), 64.0)   # AI涓绘帶鐗堬細涓婇檺鏀惧浣嗕笉閹栨
ORDER_THRESHOLD_DROP    = max(float(DECISION_POLICY.get('order_threshold_drop', 2) or 2), 2.0)    # 姣忕┖涓€杓笅闄?2 鍒?ORDER_THRESHOLD_FLOOR   = min(float(DECISION_POLICY.get('order_threshold_floor', 55) or 55), 46.0)   # AI涓绘帶鐗堬細鏈€浣庡彲鏀惧

# =====================================================
# 鏍稿績浜ゆ槗鍙冩暩
# =====================================================
RISK_PCT              = RISK_POLICY['risk_pct']      # 姣忓柈鍚嶇洰璩囬噾浣跨敤绺借硣鐢?5%
ATR_RISK_PCT          = RISK_POLICY['atr_risk_pct']      # 姣忓柈瀵﹂殯棰ㄩ毆闋愮畻 1%锛堢敤鍋滄悕璺濋洟鎻涚畻鍊変綅锛?MIN_MARGIN_PCT        = RISK_POLICY['min_margin_pct']      # 鍕曟厠淇濊瓑閲戜笅闄?1%锛堣嚦灏戞姇鍏ョ附璩囬噾1%淇濊瓑閲戯級
MAX_MARGIN_PCT        = RISK_POLICY['max_margin_pct']      # 鍕曟厠淇濊瓑閲戜笂闄?8%
MAX_OPEN_POSITIONS    = RISK_POLICY['max_open_positions']         # 鐭窔绺芥寔鍊変笂闄?MAX_SAME_DIRECTION    = RISK_POLICY['max_same_direction']         # 鍚屾柟鍚戞渶澶?5 绛?TIME_STOP_BARS_15M    = RISK_POLICY['time_stop_bars_15m']        # 15 鏍?15m K 浠嶄笉璧板氨鏅傞枔姝㈡悕
FIXED_ORDER_NOTIONAL_USDT = 20.0  # 姣忓柈鍥哄畾鍚嶇洰鍊変綅 20U
FIXED_STOCK_ORDER_NOTIONAL_USDT = 40.0  # 鑲＄エ/鑲℃寚椤炲晢鍝佸浐瀹氬悕鐩€変綅 40U
NEWS_CACHE_TTL_SEC    = EXECUTION_POLICY['news_cache_ttl_sec']       # 鏂拌仦蹇彇 5 鍒嗛悩
ANTI_CHASE_ATR      = max(float(EXECUTION_POLICY.get('anti_chase_atr', 1.25) or 1.25), 1.8)      # AI涓绘帶鐗堬細杩藉児淇濊鏀瑰亸鎵ｅ垎锛屼笉纭搵
BREAKOUT_LOOKBACK   = EXECUTION_POLICY['breakout_lookback']        # 闋愬垽鏆存媺/鏆磋穼鐨勫崁闁撹瀵熸牴鏁?PULLBACK_BUFFER_ATR = EXECUTION_POLICY['pullback_buffer_atr']      # 閬垮厤杩藉児锛屽劒鍏堢瓑 0.35ATR 鍥炶俯/鍙嶅綀
SCALE_IN_MIN_RATIO = EXECUTION_POLICY['scale_in_min_ratio']      # 鍒嗘壒閫插牬绗簩鎵规渶浣庢瘮渚?SCALE_IN_MAX_RATIO = EXECUTION_POLICY['scale_in_max_ratio']      # 鍒嗘壒閫插牬绗簩鎵规渶楂樻瘮渚?FAKE_BREAKOUT_PENALTY = EXECUTION_POLICY['fake_breakout_penalty']      # 鍋囩獊鐮?鍋囪穼鐮存墸鍒?SQLITE_DB_PATH         = "/app/data/trading_bot.sqlite3"
LEGACY_LEARN_DB_PATH    = "/app/data/learn_db.json"
LEGACY_BACKTEST_DB_PATH = "/app/data/backtest_runs.json"
STATE_BACKUP_PATH       = "/app/data/state_backup.json"
RISK_STATE_PATH         = "/app/data/risk_state.json"

SCORE_SMOOTH_ALPHA  = EXECUTION_POLICY['score_smooth_alpha']      # 绌╁畾鍒嗘暩娆婇噸锛堣秺楂樿秺璺熷嵆鏅傦級
ENTRY_LOCK_SEC      = EXECUTION_POLICY['entry_lock_sec']       # 鍚屼竴骞ｇó 5 鍒嗛悩鍏т笉閲嶈闁嬫柊鍠?POST_CLOSE_COOLDOWN_SEC = 30 * 60                              # 鍚屼竴鍊嬪梗骞冲€夊緦 30 鍒嗛悩鍏т笉閲嶄笅
MIN_RR_HARD_FLOOR   = DECISION_POLICY['min_rr_hard_floor']      # 鑷嫊涓嬪柈鏈€浣?RR
TREND_AI_SEMI_TRADES = DATASET_POLICY['trend_ai_semi_trades']       # 瓒ㄥ嫝瀛哥繏 30 绛嗗緦鍗婁粙鍏?TREND_AI_FULL_TRADES = DATASET_POLICY['trend_ai_full_trades']       # 瓒ㄥ嫝瀛哥繏 50 绛嗗緦鍏ㄤ粙鍏?AI_MIN_SAMPLE_EFFECT = 10       # AI鎴愰暦淇濊鐗堬細鑷冲皯10绛嗗眬閮ㄦǎ鏈墠鍏佽ū褰㈡垚鏈夋晥褰遍熆
SYMBOL_BLOCK_MIN_TRADES = max(int(DATASET_POLICY.get('symbol_block_min_trades', 10) or 10), 18)    # AI涓绘帶鐗堬細寤跺緦骞ｇó灏侀帠鍟熺敤
SYMBOL_BLOCK_MIN_WINRATE = min(float(DATASET_POLICY.get('symbol_block_min_winrate', 40) or 40), 35.0) # AI涓绘帶鐗堬細鏀惧骞ｇó灏侀帠鍕濈巼
STRATEGY_CAPITAL_MIN_TRADES = DATASET_POLICY['strategy_capital_min_trades']  # 绛栫暐璩囬噾鏀惧ぇ鑷冲皯瑕?5 绛嗕互涓?STRATEGY_BLOCK_MIN_TRADES = max(int(DATASET_POLICY.get('strategy_block_min_trades', 11) or 11), 20)   # AI涓绘帶鐗堬細寤跺緦绛栫暐灏侀帠鍟熺敤
STRATEGY_BLOCK_MIN_WINRATE = min(float(DATASET_POLICY.get('strategy_block_min_winrate', 45) or 45), 40.0)# AI涓绘帶鐗堬細鏀惧绛栫暐灏侀帠鍕濈巼
NEUTRAL_REGIME_BLOCK = False      # AI涓绘帶鐗堬細neutral 鍏佽ū浣庡€変綅浜ゆ槗
DATASET_RESET_TW = "2026-04-05 13:45:00"  # 鍙扮仯鏅傞枔锛岄€欏€嬫檪闁撲箣寰屾墠绠楁柊鐗?AI 涓绘帶璩囨枡
LEARNING_DATASET_META = build_learning_dataset_meta(reset_from=env_or_blank('TREND_LEARNING_RESET_FROM', DATASET_RESET_TW))
TREND_LEARNING_RESET_FROM = LEARNING_DATASET_META.get('activated_from', '') or DATASET_RESET_TW
LEGACY_BOOTSTRAP_MIN_NEW_TRADES = max(int(TREND_AI_FULL_TRADES or 50), 50)
TREND_EARLY_EXIT_MIN_RUN = 1.20 # 骞冲€夊緦鑻ュ緦绾屽欢绾岃秴閬庢骞呭害锛岃鐐哄彲鑳藉お鏃╁嚭鍫?TREND_EARLY_EXIT_MIN_EDGE = 0.35# 骞冲€夊緦鍏堝洖韪╀笉瓒呴亷閫欏€嬫瘮渚嬶紝鎵嶇畻鍋ュ悍鍥炶俯寰屽欢绾?DECISION_PRIORITY_ORDER = list(DECISION_POLICY['decision_priority_order'])
SIGNAL_META_CACHE   = {}        # 鏈€杩戜竴娆″垎鏋愬揩鍙栵紙绲﹁拷韫?椹楄瓑鐢級
SCORE_CACHE         = {}        # 鍒嗘暩骞虫粦蹇彇
ENTRY_LOCKS         = {}        # 閫插牬閹栵紝閬垮厤 90鈫?0 鍙嶈瑙哥櫦
POST_CLOSE_LOCKS    = {}        # 骞冲€夊喎鍗婚帠锛屽悓骞ｅ钩鍊夊緦 30 鍒嗛悩鍏т笉閲嶄笅
PROTECTION_STATE    = {}        # 浜ゆ槗鎵€淇濊鍠璀夌媭鎱?AUTO_ORDER_AUDIT    = {}        # 瑷橀寗姣忚吉鐐轰綍娌掍笅鍠?API_ERROR_STREAK    = 0
PROTECTION_FAIL_STREAK = 0
AUTO_AI_MODE = 'normal'
AI_FULL_SCORE_CONTROL = True
AI_DISCOVERY_MIN_COUNT = 3
AI_DISCOVERY_BLEND_FLOOR = 0.08
AI_DISCOVERY_BLEND_CEIL = 0.72
AI_LEGACY_WEIGHT_READONLY = True
AI_MASTER_CONTROL_PATCH = True
LAST_MARKET_CONSENSUS = {}
CACHE_LOCK          = threading.RLock()
PROTECTION_LOCK     = threading.RLock()
AUDIT_LOCK          = threading.RLock()
MARKET_DIRECTION_GUARD = MarketDirectionGuard(required_confirmations=2, ttl_seconds=4 * 3600)
FVG_MONITOR_CACHE   = {}
FVG_MONITOR_LOCK    = threading.RLock()
PRE_BREAKOUT_RADAR_CACHE = {}
PRE_BREAKOUT_RADAR_LOCK = threading.RLock()
OPENAI_CONTEXT_CACHE = {}
OPENAI_CONTEXT_CACHE_TTL_SEC = 180
OPENAI_MARKETCAP_CACHE = {}
OPENAI_MARKETCAP_CACHE_TTL_SEC = 6 * 3600
OPENAI_NEWS_CACHE = {}
OPENAI_NEWS_CACHE_TTL_SEC = 15 * 60
OPENAI_CONTEXT_HTTP_TIMEOUT_SEC = 6.0

STORAGE = BotStorage(
    SQLITE_DB_PATH,
    legacy_learn_json=LEGACY_LEARN_DB_PATH,
    legacy_backtest_json=LEGACY_BACKTEST_DB_PATH,
)

RUNTIME_STATE = DEFAULT_RUNTIME_STATE
RUNTIME_STATE.update(meta=get_policy_snapshot())


def _dataset_meta():
    meta = dict(LEARNING_DATASET_META or {})
    meta.setdefault('activated_from', TREND_LEARNING_RESET_FROM)
    meta.setdefault('reset_from', TREND_LEARNING_RESET_FROM)
    meta.setdefault('reset_timezone', 'Asia/Taipei')
    meta.setdefault('dataset_mode', 'layered_live_only')
    meta.setdefault('legacy_policy', 'bootstrap_then_fade_out')
    return meta


MARKET_AUX_SCORE_CAP = 2.0
MARKET_AUX_THRESHOLD_CAP = 1.0
MARKET_NEUTRAL_THRESHOLD_ADD = 1.5


def _cap_market_aux(value, cap=MARKET_AUX_SCORE_CAP):
    try:
        cap = abs(float(cap or MARKET_AUX_SCORE_CAP))
        value = float(value or 0.0)
        return max(-cap, min(cap, value))
    except Exception:
        return 0.0


def _ai_effective_rows(closed_only=True):
    """绲变竴 AI 鎴愰暦闅庢鐨勬湁鏁堝鍠▓绠楋細鍎厛 trusted_live锛屽啀閫€鍥?soft_live / trend_live銆?""
    try:
        trusted = get_live_trades(closed_only=closed_only, pool='trusted_live')
        if trusted:
            return trusted
        soft_live = get_live_trades(closed_only=closed_only, pool='soft_live')
        if soft_live:
            return soft_live
        return get_trend_live_trades(closed_only=closed_only)
    except Exception:
        return []


def _ai_growth_control(effective_count=None):
    """AI 鎴愰暦淇濊妯″紡锛?    <30 绛嗭細鍙閷勪笉褰遍熆浜ゆ槗
    30~49 绛嗭細灏忓箙褰遍熆
    50+ 绛嗭細鍏ㄦ瑠鎺ョ
    """
    if effective_count is None:
        effective_count = len(_ai_effective_rows(closed_only=True))
    try:
        effective_count = int(effective_count or 0)
    except Exception:
        effective_count = 0

    if effective_count < TREND_AI_SEMI_TRADES:
        return {
            'phase': 'learning',
            'score_weight': 0.0,
            'threshold_weight': 0.0,
            'market_weight': 0.0,
            'confidence_weight': 0.0,
            'blend_cap': 0.0,
            'allow_ai_gate': False,
            'allow_profile_block': False,
            'note': f'AI鎴愰暦淇濊锛氬墠{TREND_AI_SEMI_TRADES}绛嗗彧瑷橀寗涓嶆帴绠?,
        }
    if effective_count < TREND_AI_FULL_TRADES:
        return {
            'phase': 'semi',
            'score_weight': 0.35,
            'threshold_weight': 0.35,
            'market_weight': 0.35,
            'confidence_weight': 0.35,
            'blend_cap': 0.35,
            'allow_ai_gate': True,
            'allow_profile_block': True,
            'note': f'AI鎴愰暦淇濊锛歿TREND_AI_SEMI_TRADES}-{TREND_AI_FULL_TRADES - 1}绛嗗皬骞呮帴绠?,
        }
    return {
        'phase': 'full',
        'score_weight': 1.0,
        'threshold_weight': 1.0,
        'market_weight': 1.0,
        'confidence_weight': 1.0,
        'blend_cap': 1.0,
        'allow_ai_gate': True,
        'allow_profile_block': True,
        'note': f'AI鎴愰暦淇濊锛歿TREND_AI_FULL_TRADES}+绛嗗叏娆婃帴绠?,
    }


def _execution_quality_state(sig):
    snap = dict((sig or {}).get('execution_quality') or {})
    score = execution_score_from_snapshot(snap)
    snap['execution_score'] = score
    return snap


def _ensure_sqlite_compat_schema():
    """瑁滈綂 API 鏈冪敤鍒扮殑娆勪綅锛岄伩鍏?schema 钀藉樊璁撴煡瑭㈢洿鎺ョ偢鎺夈€?""
    import sqlite3
    table_columns = {
        'learning_trades': {
            'updated_at': "TEXT DEFAULT ''",
            'created_at': "TEXT DEFAULT ''",
            'data_json': "TEXT DEFAULT '{}'",
        },
        'trade_history': {
            'updated_at': "TEXT DEFAULT ''",
            'created_at': "TEXT DEFAULT ''",
            'entry_time': "TEXT DEFAULT ''",
            'exit_time': "TEXT DEFAULT ''",
            'time': "TEXT DEFAULT ''",
            'data_json': "TEXT DEFAULT '{}'",
        },
        'risk_events': {
            'created_at': "TEXT DEFAULT ''",
            'event_time': "TEXT DEFAULT ''",
            'timestamp': "TEXT DEFAULT ''",
            'payload_json': "TEXT DEFAULT '{}'",
        },
        'audit_logs': {
            'created_at': "TEXT DEFAULT ''",
            'event_time': "TEXT DEFAULT ''",
            'timestamp': "TEXT DEFAULT ''",
            'payload_json': "TEXT DEFAULT '{}'",
        },
        'backtest_runs': {
            'created_at': "TEXT DEFAULT ''",
            'run_time': "TEXT DEFAULT ''",
            'timestamp': "TEXT DEFAULT ''",
            'payload_json': "TEXT DEFAULT '{}'",
            'summary_json': "TEXT DEFAULT '{}'",
            'result_json': "TEXT DEFAULT '{}'",
            'data_json': "TEXT DEFAULT '{}'",
        },
    }
    try:
        os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(SQLITE_DB_PATH)
        try:
            cur = conn.cursor()
            for table, columns in table_columns.items():
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
                exists = cur.fetchone()
                if not exists:
                    cols_sql = []
                    if table == 'learning_trades':
                        cols_sql.append('trade_id TEXT PRIMARY KEY')
                    elif table in ('trade_history', 'backtest_runs'):
                        cols_sql.append('id INTEGER PRIMARY KEY AUTOINCREMENT')
                    else:
                        cols_sql.append('id INTEGER PRIMARY KEY AUTOINCREMENT')
                    for name, spec in columns.items():
                        cols_sql.append(f"{name} {spec}")
                    cur.execute(f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(cols_sql)})")
                    continue
                cur.execute(f"PRAGMA table_info({table})")
                existing = {str(r[1]) for r in cur.fetchall()}
                for name, spec in columns.items():
                    if name not in existing:
                        cur.execute(f"ALTER TABLE {table} ADD COLUMN {name} {spec}")
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        print('SQLite schema 淇京澶辨晽:', e)


_ensure_sqlite_compat_schema()

# =====================================================
# 闃查€ｆ悕瑷畾
# =====================================================
MAX_DAILY_LOSS_PCT   = 0.15
MAX_CONSECUTIVE_LOSS = 3
COOLDOWN_MINUTES     = 120

RISK_STATE = {
    "daily_loss_usdt":    0.0,
    "daily_start_equity": 0.0,
    "consecutive_loss":   0,
    "cooldown_until":     None,
    "trading_halted":     False,
    "halt_reason":        "",
    "today_date":         "",
}
RISK_LOCK = threading.Lock()

# 鈹€鈹€ 鍕曟厠闁€妾荤媭鎱?鈹€鈹€
_DT = {
    "current":          52,   # AI涓绘帶鐗堣捣濮嬮杸妾?    "last_order_time":  None, # 鏈€杩戜笅鍠檪闁?    "full_rounds":      0,    # 閫ｇ簩婊垮€夎吉鏁?    "empty_rounds":     0,    # 闁€妾?5鏅傞€ｇ簩绌哄€夎吉鏁?    "no_order_rounds":  0,    # 閫ｇ簩鐒′笅鍠吉鏁革紙鏁存暩锛岄伩鍏峃one+1閷锛?}
_DT_LOCK = threading.Lock()

def _estimate_ai_threshold_target(top_sigs=None):
    """鐢?AI 瑭曞垎寰岀殑鍊欓伕瑷婅櫉鍝佽唱鑷嫊浼拌▓闁€妾伙紝涓﹀鐢?AI 鎴愰暦淇濊妯″紡銆?""
    control = _ai_growth_control()
    phase = str(control.get('phase') or 'learning')
    base_default = max(52.0, float(ORDER_THRESHOLD_DEFAULT or 52.0))
    sigs = list(top_sigs or [])[:8]
    if not sigs:
        if phase == 'learning':
            return base_default, control.get('note', 'AI鎴愰暦淇濊')
        return base_default if phase == 'semi' else 50.0, '鐒″€欓伕瑷婅櫉锛岀董鎸佽瀵?

    scored = []
    for sig in sigs:
        try:
            score = abs(float(sig.get('score', 0) or 0))
            rr = float(sig.get('rr_ratio', 0) or 0)
            eq = float(sig.get('entry_quality', 0) or 0)
            ai_cov = float((sig.get('breakdown') or {}).get('AIScoreCoverage', 0) or 0)
            ai_scnt = float((sig.get('breakdown') or {}).get('AISampleCount', 0) or 0)
            regime_conf = float(sig.get('regime_confidence', 0) or 0)
            anti_chase_ok = bool(sig.get('anti_chase_ok', True))
            profile = _ai_strategy_profile(
                str(sig.get('symbol') or ''),
                regime=str(sig.get('regime') or ((sig.get('breakdown') or {}).get('Regime')) or 'neutral'),
                setup=str(sig.get('setup_label') or ((sig.get('breakdown') or {}).get('Setup')) or ''),
            )
            quality = 0.0
            quality += (score - 50.0) / 18.0
            quality += max(min((rr - 1.0) * 1.1, 1.8), -1.2)
            quality += max(min((eq - 1.6) * 0.6, 1.2), -1.0)
            quality += ai_cov * 2.8
            quality += min(ai_scnt / 25.0, 1.8)
            quality += regime_conf * 1.2
            quality += float(profile.get('ev_per_trade', 0) or 0) * 14.0
            quality += (float(profile.get('win_rate', 50.0) or 50.0) - 50.0) * 0.03
            if not anti_chase_ok:
                quality -= 0.9
            if bool(profile.get('symbol_blocked')) or bool(profile.get('strategy_blocked')):
                quality -= 2.0
            scored.append((quality, sig, profile))
        except Exception:
            continue

    if not scored:
        return base_default, '鍊欓伕瑷婅櫉涓嶈冻锛岀董鎸佽瀵?

    scored.sort(key=lambda x: x[0], reverse=True)
    best_q = float(scored[0][0])
    avg_q = sum(float(x[0]) for x in scored[:3]) / max(min(3, len(scored)), 1)
    best_sig = scored[0][1]
    best_profile = scored[0][2]

    target = 52.0 - avg_q * 2.6 - max(best_q - 1.0, 0.0) * 1.55
    if bool(best_profile.get('ready')):
        target -= 1.8
    if bool(best_profile.get('symbol_blocked')) or bool(best_profile.get('strategy_blocked')):
        target += 1.2

    if phase == 'learning':
        target = base_default
    elif phase == 'semi':
        target = base_default * 0.65 + float(target) * 0.35

    target = max(50.0 if phase != 'full' else 44.0, min(64.0, target))
    note = 'AI鎴愰暦淇濊({}) | top {:.1f} | cov {:.2f} | 妯ｆ湰 {}'.format(
        phase,
        abs(float(best_sig.get('score', 0) or 0)),
        float(((best_sig.get('breakdown') or {}).get('AIScoreCoverage', 0) or 0)),
        int(((best_sig.get('breakdown') or {}).get('AISampleCount', 0) or 0)),
    )
    return round(target, 2), note

def update_dynamic_threshold(top_sigs=None):
    """AI 鑷富闁€妾伙細渚?AI 鍒嗘暩瑕嗚搵鐜囥€佸缈掓ǎ鏈垏鎸佸€夊鍔涘嫊鎱嬭鏁淬€?""
    global ORDER_THRESHOLD
    with _DT_LOCK:
        dt = _DT
        with STATE_LOCK:
            pos_count = len(STATE.get('active_positions', []))

        control = _ai_growth_control()
        phase = str(control.get('phase') or 'learning')
        target, note = _estimate_ai_threshold_target(top_sigs)
        if pos_count >= MAX_OPEN_POSITIONS:
            dt['full_rounds'] = dt.get('full_rounds', 0) + 1
            target += min(3.0, dt['full_rounds'] * (0.20 if phase != 'full' else 0.45))
        else:
            dt['full_rounds'] = 0

        strong_count = 0
        for sig in list(top_sigs or [])[:5]:
            try:
                sig_score = abs(float(sig.get('score', 0) or 0))
                ai_cov = float((sig.get('breakdown') or {}).get('AIScoreCoverage', 0) or 0)
                pwin = float((sig.get('decision_calibrator') or {}).get('p_win_est', 0.5) or 0.5)
                if sig_score >= max(48.0, target - 2.0) and (ai_cov >= 0.18 or pwin >= 0.51):
                    strong_count += 1
            except Exception:
                pass

        if strong_count == 0:
            dt['no_order_rounds'] = dt.get('no_order_rounds', 0) + 1
            if phase == 'full':
                target -= min(4.0, dt['no_order_rounds'] * 0.65)
            elif phase == 'semi':
                target -= min(1.5, dt['no_order_rounds'] * 0.20)
        else:
            dt['no_order_rounds'] = 0
            if strong_count >= 2:
                target -= min(2.5 if phase == 'full' else 0.8, strong_count * (0.7 if phase == 'full' else 0.18))

        prev = float(dt.get('current', ORDER_THRESHOLD_DEFAULT) or ORDER_THRESHOLD_DEFAULT)
        if phase == 'learning':
            new_val = round(max(52.0, float(ORDER_THRESHOLD_DEFAULT or 52.0)), 2)
        else:
            mix_ratio = 0.35 if phase == 'semi' else 0.65
            new_val = round(prev * (1.0 - mix_ratio) + float(target) * mix_ratio, 2)
            new_val = max(50.0 if phase == 'semi' else 44.0, min(64.0, new_val))
        dt['current'] = new_val
        ORDER_THRESHOLD = new_val
        dt['last_ai_note'] = note
        phase = 'AI绌嶆サ' if new_val <= 50 else 'AI鍧囪　' if new_val <= 61 else 'AI淇濆畧'
        print('馃 AI闁€妾绘洿鏂?{:.1f} 鈫?{:.1f} | {}'.format(prev, new_val, note))
        update_state(threshold_info={
            'current': new_val,
            'phase': phase,
            'full_rounds': dt.get('full_rounds', 0),
            'empty_rounds': dt.get('empty_rounds', 0),
            'no_order_rounds': dt.get('no_order_rounds', 0),
            'ai_note': note,
            'target': round(float(target), 2),
        })

def record_order_placed():
    """涓嬪柈寰屽儏鍋氶潪甯歌紩寰殑閬庣啽鎶戝埗锛屼笉鍐嶆妸闁€妾绘媺鍥炲浐瀹氬崁闁撱€?""
    global ORDER_THRESHOLD
    with _DT_LOCK:
        _DT['last_order_time'] = datetime.now()
        _DT['no_order_rounds'] = 0
        _DT['empty_rounds'] = 0
        prev = float(_DT.get('current', ORDER_THRESHOLD_DEFAULT) or ORDER_THRESHOLD_DEFAULT)
        nudged = round(min(prev + 0.3, 64.0), 2)
        _DT['current'] = max(44.0, min(64.0, nudged))
        ORDER_THRESHOLD = _DT['current']
        print('鈫╋笍 AI闁€妾诲井瑾胯嚦{}锛堥伩鍏嶇煭鏅傞枔閬庡害閫ｉ枊锛?.format(_DT['current']))
        update_state(threshold_info={
            'current': _DT['current'],
            'phase': 'AI绌嶆サ' if _DT['current'] <= 50 else 'AI鍧囪　' if _DT['current'] <= 61 else 'AI淇濆畧',
            'full_rounds': _DT.get('full_rounds', 0),
            'empty_rounds': _DT.get('empty_rounds', 0),
            'no_order_rounds': _DT.get('no_order_rounds', 0),
            'ai_note': _DT.get('last_ai_note', ''),
        })

# =====================================================
# 闁嬬洡鏅傛淇濊绯荤当锛堝彴鐏ｆ檪闁?UTC+8锛?# =====================================================
SESSION_STATE = {
    "eu_score":      0,   # 姝愭床鐩よ瀵熷垎鏁?(-2~+2)
    "us_score":      0,   # 缇庢床鐩よ瀵熷垎鏁?(-2~+2)
    "eu_score_date": "",  # 姝愮洡鍒嗘暩鐨勫彴鐏ｆ棩鏈?(YYYY-MM-DD)
    "us_score_date": "",  # 缇庣洡鍒嗘暩鐨勫彴鐏ｆ棩鏈?    "eu_score_time": "",  # 姝愮洡鍒嗘暩鐨勫彴鐏ｆ檪闁?(HH:MM)
    "us_score_time": "",  # 缇庣洡鍒嗘暩鐨勫彴鐏ｆ檪闁?    "europe_obs":    [],  # 瑙€瀵熸湡闁撶殑鍍规牸瑷橀寗
    "america_obs":   [],  # 瑙€瀵熸湡闁撶殑鍍规牸瑷橀寗
    "session_phase": "normal",
    "session_note":  "",
}
SESSION_LOCK = threading.Lock()

def get_tw_time():
    """鍙栧緱鍙扮仯鏅傞枔锛圲TC+8锛?""
    from datetime import timezone, timedelta
    tz_tw = timezone(timedelta(hours=8))
    return datetime.now(tz_tw)

def tw_now_str(fmt="%H:%M:%S"):
    """鍙扮仯鏅傞枔鏍煎紡鍖栧瓧涓?""
    return get_tw_time().strftime(fmt)

def tw_today():
    """鍙扮仯鏅傞枔浠婂ぉ鏃ユ湡"""
    return get_tw_time().strftime("%Y-%m-%d")

def get_session_status():
    """
    鍥炲偝鐣跺墠鏅傛鐙€鎱嬶細
    - normal: 姝ｅ父浜ゆ槗
    - eu_pause: 姝愮洡闁嬬洡鍓?0鍒嗛悩锛屽仠姝笅鏂板柈
    - eu_closed: 19:50-20:32 瀹屽叏鍋滄+骞冲€?    - eu_watch: 20:32鍓嶈瀵熸瓙鐩よ蛋鍕?    - us_pause: 缇庣洡闁嬬洡鍓?0鍒嗛悩锛屽仠姝笅鏂板柈
    - us_closed: 21:50-22:32 瀹屽叏鍋滄+骞冲€?    - us_watch: 22:32鍓嶈瀵熺編鐩よ蛋鍕?    """
    # 鏅傛淇濊宸插畬鍏ㄥ仠鐢紝鍥哄畾鍥炲偝姝ｅ父鐙€鎱嬨€?    return "normal", ""
    tw = get_tw_time()
    h = tw.hour
    m = tw.minute
    t = h * 60 + m  # 杞夋彌鎴愬垎閻?
    EU_PAUSE_START  = 19 * 60 + 30   # 19:30
    EU_CLOSE_START  = 19 * 60 + 50   # 19:50
    EU_WATCH_END    = 20 * 60 + 32   # 20:32
    EU_RESUME       = 20 * 60 + 35   # 20:35

    US_PAUSE_START  = 21 * 60 + 30   # 21:30
    US_CLOSE_START  = 21 * 60 + 50   # 21:50
    US_WATCH_END    = 22 * 60 + 32   # 22:32
    US_RESUME       = 22 * 60 + 35   # 22:35

    if EU_CLOSE_START <= t < EU_WATCH_END:
        return "eu_closed", "姝愮洡闁嬬洡瑙€瀵熸湡 (19:50-20:32)"
    elif EU_PAUSE_START <= t < EU_CLOSE_START:
        return "eu_pause", "姝愮洡闁嬬洡鍓嶆毇鍋滀笅鍠?(19:30-19:50)"
    elif EU_WATCH_END <= t < EU_RESUME:
        return "eu_watch_end", "姝愮洡瑙€瀵熺祼鏉燂紝瑷堢畻鍒嗘暩涓?
    elif US_CLOSE_START <= t < US_WATCH_END:
        return "us_closed", "缇庣洡闁嬬洡瑙€瀵熸湡 (21:50-22:32)"
    elif US_PAUSE_START <= t < US_CLOSE_START:
        return "us_pause", "缇庣洡闁嬬洡鍓嶆毇鍋滀笅鍠?(21:30-21:50)"
    elif US_WATCH_END <= t < US_RESUME:
        return "us_watch_end", "缇庣洡瑙€瀵熺祼鏉燂紝瑷堢畻鍒嗘暩涓?
    return "normal", ""

def observe_session_market(session="eu"):
    """
    瑙€瀵熼枊鐩よ蛋鍕紝瑷堢畻椤嶅瑭曞垎 (-2 ~ +2)
    閭忚集锛氱湅 BTC 鍦ㄨ瀵熸湡闁撶殑婕茶穼骞?    """
    # 鏅傛淇濊宸插仠鐢紝涓嶅啀鍋氫换浣曡瀵熴€佽鍒嗘垨 UI 鏇存柊銆?    return
    try:
        ticker = exchange.fetch_ticker("BTC/USDT:USDT")
        price  = float(ticker['last'])
        pct    = float(ticker.get('percentage', 0) or 0)

        with SESSION_LOCK:
            obs_key_map = {"eu": "europe_obs", "us": "america_obs", "europe": "europe_obs", "america": "america_obs"}
            score_key_map = {"eu": "eu_score", "us": "us_score", "europe": "eu_score", "america": "us_score"}
            date_key_map = {"eu": "eu_score_date", "us": "us_score_date", "europe": "eu_score_date", "america": "us_score_date"}
            time_key_map = {"eu": "eu_score_time", "us": "us_score_time", "europe": "eu_score_time", "america": "us_score_time"}
            key = obs_key_map.get(session, "{}_obs".format(session))
            SESSION_STATE.setdefault(key, [])
            SESSION_STATE[key].append(price)
            # 鍙繚鐣欐渶杩?0绛?            if len(SESSION_STATE[key]) > 20:
                SESSION_STATE[key] = SESSION_STATE[key][-20:]

            prices = SESSION_STATE[key]
            if len(prices) < 2:
                return

            # 瑷堢畻瑙€瀵熸湡闁撴疾璺?            first_price = prices[0]
            last_price  = prices[-1]
            change_pct  = (last_price - first_price) / first_price * 100

            # 瑭曞垎閭忚集
            if change_pct > 1.5:
                score = 2; note = "{}鐩ゅ挤鍕笂婕瞷:.1f}% +2鍒?.format(
                    "姝愭床" if session=="eu" else "缇庢床", change_pct)
            elif change_pct > 0.5:
                score = 1; note = "{}鐩ゅ皬骞呬笂婕瞷:.1f}% +1鍒?.format(
                    "姝愭床" if session=="eu" else "缇庢床", change_pct)
            elif change_pct < -1.5:
                score = -2; note = "{}鐩ゅ挤鍕笅璺寋:.1f}% -2鍒?.format(
                    "姝愭床" if session=="eu" else "缇庢床", abs(change_pct))
            elif change_pct < -0.5:
                score = -1; note = "{}鐩ゅ皬骞呬笅璺寋:.1f}% -1鍒?.format(
                    "姝愭床" if session=="eu" else "缇庢床", abs(change_pct))
            else:
                score = 0; note = "{}鐩ゆ┇鐩?0鍒?.format(
                    "姝愭床" if session=="eu" else "缇庢床")

            score_key = score_key_map.get(session, "{}_score".format(session))
            date_key  = date_key_map.get(session, "{}_score_date".format(session))
            time_key  = time_key_map.get(session, "{}_score_time".format(session))
            SESSION_STATE[score_key] = score
            SESSION_STATE[date_key]  = tw_today()        # 瑷橀寗鍙扮仯鏃ユ湡
            SESSION_STATE[time_key]  = tw_now_str("%H:%M")  # 瑷橀寗鍙扮仯鏅傞枔
            SESSION_STATE["session_note"] = note
            print("馃搳 {}鐩よ瀵? {} | BTC {:.2f}% | 鍒嗘暩鏈夋晥鑷虫槑鏃?榛?.format(
                "姝愭床" if session=="eu" else "缇庢床", note, change_pct))

            # 鍚屾鍒?STATE 绲?UI 椤ず
            update_state(session_info={
                "phase":    SESSION_STATE["session_phase"],
                "note":     note,
                "eu_score": SESSION_STATE["eu_score"],
                "us_score": SESSION_STATE["us_score"],
                "eu_time":  SESSION_STATE.get("eu_score_time",""),
                "us_time":  SESSION_STATE.get("us_score_time",""),
            })
    except Exception as e:
        print("瑙€瀵熷競鍫村け鏁? {}".format(e))

def get_session_score():
    """鏅傛淇濊宸插仠鐢紝涓嶅啀褰遍熆浜ゆ槗鍒嗘暩銆?""
    return 0

def session_monitor_thread():
    """鏅傛淇濊宸插仠鐢ㄣ€?""
    while True:
        time.sleep(600)

# =====================================================
# 澶х洡璧板嫝鍒嗘瀽绯荤当锛圔TC 鏃ョ窔 + 姝峰彶鍨嬫厠灏嶆瘮锛?# =====================================================
MARKET_STATE = {
    "pattern":      "鍒濆鍖栦腑",
    "direction":    "涓€?,
    "score":        0,
    "strength":     0.0,
    "detail":       "",
    "history_match": "",   # 姝峰彶鐩镐技鍨嬫厠
    "prediction":   "",    # 闋愭脯璧板嫝
    "last_update":  "",
    "btc_price":    0.0,
    "btc_change":   0.0,
    "long_term_pos": None, # 闀锋湡鍊変綅鐙€鎱?}
MARKET_LOCK = threading.Lock()

def find_similar_history(df, current_window=30, top_n=3):
    """
    鍦?BTC 姝峰彶鏃ョ窔涓壘鏈€鐩镐技鐨凨绶氬瀷鎱?    鐢ㄦ婧栧寲寰岀殑鏀剁洡鍍瑰簭鍒楀仛鐩镐技搴︽瘮灏嶏紙姝愬咕閲屽緱璺濋洟锛?    """
    try:
        closes = df['c'].values.astype(float)
        n = len(closes)
        if n < current_window + 30:
            return []

        # 鍙栨渶杩?current_window 鏍筀妫掍綔鐐虹暥鍓嶅瀷鎱?        current = closes[-(current_window):]
        # 妯欐簴鍖栵紙0-1 绡勫湇锛?        c_min, c_max = current.min(), current.max()
        if c_max == c_min:
            return []
        current_norm = (current - c_min) / (c_max - c_min)

        similarities = []
        # 寰炴鍙蹭腑婊戝嫊姣斿皪锛堣嚦灏戜繚鐣?00鏍瑰緦绾孠妫掔敤渚嗙湅绲愭灉锛?        search_end = n - current_window - 30
        for i in range(0, search_end - current_window, 5):  # 姣?鏍硅烦涓€鏍?            window = closes[i:i+current_window]
            w_min, w_max = window.min(), window.max()
            if w_max == w_min:
                continue
            window_norm = (window - w_min) / (w_max - w_min)

            # 瑷堢畻鐩镐技搴︼紙1 - 妯欐簴鍖栬窛闆級
            dist = np.sqrt(np.mean((current_norm - window_norm)**2))
            similarity = max(0, 1 - dist * 2)  # 0~1锛岃秺楂樿秺鐩镐技

            if similarity > 0.55:  # 鏀惧鍒?5%锛圔itget鏃ョ窔鏈夐檺锛?                # 鐪嬮€欏€嬫檪闁撻粸涔嬪緦30鏍圭殑婕茶穼
                future = closes[i+current_window:i+current_window+30]
                if len(future) >= 10:
                    future_ret = (future[-1] - future[0]) / future[0] * 100
                    # 鍙栧緱鏃ユ湡锛堢敤绱㈠紩鍙嶆帹锛?                    similarities.append({
                        'idx': i,
                        'similarity': round(similarity * 100, 1),
                        'future_ret': round(future_ret, 1),
                        'entry_price': round(closes[i+current_window-1], 0),
                    })

        # 鎸夌浉浼煎害鎺掑簭鍙栧墠N
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_n]
    except Exception as e:
        print("姝峰彶姣斿皪澶辨晽:", e)
        return []

def analyze_btc_market_trend():
    """
    鍒嗘瀽 BTC 鏃ョ窔璧板嫝锛岃瓨鍒ョ暥鍓嶅瀷鎱嬩甫灏嶆瘮姝峰彶
    鍥炲偝瑭崇窗鍒嗘瀽绲愭灉
    """
    try:

        # 鎶?BTC 鏃ョ窔 - 鎶撴渶澶?00鏍瑰仛姝峰彶姣斿皪锛堢磩1.5骞达級
        ohlcv = exchange.fetch_ohlcv("BTC/USDT:USDT", "1d", limit=1000)  # 鐩￠噺澶氭姄
        df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])

        # 鍧囩窔
        df['ma7']  = df['c'].rolling(7).mean()
        df['ma25'] = df['c'].rolling(25).mean()
        df['ma50'] = df['c'].rolling(50).mean()
        df['ma99'] = df['c'].rolling(min(99,len(df))).mean()

        curr  = float(df['c'].iloc[-1])
        ma7   = float(df['ma7'].iloc[-1])
        ma25  = float(df['ma25'].iloc[-1])
        ma50  = float(df['ma50'].iloc[-1])
        ma99  = float(df['ma99'].iloc[-1])
        prev  = float(df['c'].iloc[-2])
        change_pct = (curr - prev) / prev * 100

        # 杩戞湡楂樹綆榛?        high_30 = float(df['h'].tail(30).max())
        low_30  = float(df['l'].tail(30).min())
        high_7  = float(df['h'].tail(7).max())
        low_7   = float(df['l'].tail(7).min())
        range_30 = high_30 - low_30

        # ATR
        atr_s = ta.atr(df['h'], df['l'], df['c'], length=14)
        atr   = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else curr*0.02

        # 瓒ㄥ嫝鏂滅巼锛堢窔鎬у洖姝革級
        c7  = df['c'].tail(7).values
        c14 = df['c'].tail(14).values
        c30 = df['c'].tail(30).values
        x7  = np.arange(len(c7));  slope_7  = np.polyfit(x7,  c7,  1)[0]/curr*100
        x14 = np.arange(len(c14)); slope_14 = np.polyfit(x14, c14, 1)[0]/curr*100
        x30 = np.arange(len(c30)); slope_30 = np.polyfit(x30, c30, 1)[0]/curr*100

        # 鎴愪氦閲忚定鍕?        vol_7  = float(df['v'].tail(7).mean())
        vol_30 = float(df['v'].tail(30).mean())
        vol_ratio = vol_7 / max(vol_30, 1)

        # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
        # 鍨嬫厠璀樺垾
        # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
        pattern = ""; direction = "涓€?; score = 0; strength = 0.5
        detail = ""; history_match = ""; prediction = ""

        # 1. 鍧囩窔瀹屽叏澶氶牠鎺掑垪
        if curr > ma7 > ma25 > ma50 > ma99:
            pattern = "鍧囩窔瀹屽叏澶氶牠鎺掑垪"
            direction = "寮峰"
            score = 5; strength = 0.9
            detail = "BTC 鏃ョ窔4姊濆潎绶氬畬鍏ㄥ闋帓鍒楋紝澶х洡铏曟柤寮峰嫝鐗涘競绲愭"
            history_match = "姝峰彶灏嶆瘮锛?020骞?0鏈堛€?021骞?鏈堛€?024骞?鏈堝嚭鐝剧浉鍚岀祼妲?
            prediction = "鐭湡锛氱董鎸佸闋紝鍥炶璨峰叆 | 涓湡锛氳嫢閲忛厤鍚堝彲鍓垫柊楂?| 寤鸿锛氬仛澶氱偤涓?

        # 2. 鍧囩窔瀹屽叏绌洪牠鎺掑垪
        elif curr < ma7 < ma25 < ma50 < ma99:
            pattern = "鍧囩窔瀹屽叏绌洪牠鎺掑垪"
            direction = "寮风┖"
            score = -5; strength = 0.9
            detail = "BTC 鏃ョ窔4姊濆潎绶氬畬鍏ㄧ┖闋帓鍒楋紝澶х洡铏曟柤寮卞嫝鐔婂競绲愭"
            history_match = "姝峰彶灏嶆瘮锛?018骞翠笅鍗婂勾銆?022骞?-11鏈堝嚭鐝剧浉鍚岀祼妲?
            prediction = "鐭湡锛氬弽褰堝仛绌猴紝涓嶅疁杩藉 | 涓湡锛氬簳閮ㄤ笉鏄庤鎱?| 寤鸿锛氬仛绌哄弽褰堝毚鎺ф悕"

        # 3. 绐佺牬杩戞湡楂橀粸锛堢墰甯傜獊鐮达級
        elif curr > high_30 * 0.99 and slope_7 > 0.3:
            pattern = "绐佺牬杩戞湡30鏃ラ珮榛?
            direction = "澶?
            score = 4; strength = 0.8
            detail = "BTC 姝ｇ獊鐮磋繎30鏃ラ珮榛?{:.0f}锛岀獊鐮村瀷鎱嬬湅澶?.format(high_30)
            history_match = "姝峰彶灏嶆瘮锛氱獊鐮村瀷鎱嬪緦绾屼笂婕叉鐜囩磩65-70%"
            prediction = "鐭湡锛氬闋嫊鑳藉挤锛岃瀵熻兘鍚︾珯绌╅珮榛?| 寤鸿锛氬仛澶氾紝姝㈡悕楂橀粸涓嬫柟ATR脳1.5"

        # 4. 璺岀牬杩戞湡浣庨粸锛堢唺甯傝穼鐮达級
        elif curr < low_30 * 1.01 and slope_7 < -0.3:
            pattern = "璺岀牬杩戞湡30鏃ヤ綆榛?
            direction = "绌?
            score = -4; strength = 0.8
            detail = "BTC 璺岀牬杩?0鏃ヤ綆榛?{:.0f}锛岃穼鐮村瀷鎱嬬湅绌?.format(low_30)
            history_match = "姝峰彶灏嶆瘮锛氳穼鐮翠綆榛炲緦绾屼笅璺屾鐜囩磩60-65%"
            prediction = "鐭湡锛氱┖闋嫊鑳藉挤锛岄伩鍏嶆妱搴?| 寤鸿锛氳紩鍊夊仛绌烘垨瑙€鏈?

        # 5. 鍧囩窔绯剧簭锛堢洡鏁达級
        elif abs(curr - ma25) / curr < 0.02 and abs(slope_14) < 0.15:
            pattern = "鍧囩窔绯剧簭鐩ゆ暣"
            direction = "涓€?
            score = 0; strength = 0.3
            detail = "BTC 鍧囩窔绯剧簭锛屽競鍫存柟鍚戜笉鏄庯紝铏曟柤鐩ゆ暣闅庢"
            history_match = "姝峰彶灏嶆瘮锛氱洡鏁村緦绐佺牬鏂瑰悜姹哄畾涓嬩竴娉㈣定鍕?
            prediction = "绛夊緟鏂瑰悜閬告搰锛岀獊鐮村仛澶?璺岀牬鍋氱┖ | 寤鸿锛氶檷浣庡€変綅绛夊緟绐佺牬"

        # 6. 澶氶牠鍥炶锛堜富鍗囨氮涓洖瑾匡級
        elif curr > ma50 and curr < ma25 and slope_30 > 0.2:
            pattern = "澶氶牠涓诲崌娴腑鍥炶"
            direction = "澶?
            score = 3; strength = 0.7
            detail = "BTC 鍦ㄩ暦鏈熶笂鍗囪定鍕腑鍥炶鑷矼A25闄勮繎锛屽仴搴峰洖瑾?
            history_match = "姝峰彶灏嶆瘮锛氫富鍗囨氮鍥炶閫氬父鏄卜鍏ユ鏈冿紙鍥炶骞呭害10-20%锛?
            prediction = "鍥炶绲愭潫淇¤櫉锛氭棩绶氭敹寰㎝A25 | 寤鸿锛氬垎鎵瑰仛澶氾紝姝㈡悕MA50涓嬫柟"

        # 7. 姝昏矒褰堬紙鐔婂競涓弽褰堬級
        elif curr < ma50 and slope_7 > 0.5 and slope_30 < -0.1:
            pattern = "鐔婂競涓璨撳綀鍙嶅綀"
            direction = "绌?
            score = -3; strength = 0.7
            detail = "BTC 鍦ㄩ暦鏈熶笅闄嶈定鍕腑鍑虹従鍙嶅綀锛屽皬蹇冩璨撳綀"
            history_match = "姝峰彶灏嶆瘮锛氱唺甯傚弽褰堥€氬父鍦∕A50涓嬫柟澶姌"
            prediction = "鍙嶅綀鐩锛歁A25-MA50涔嬮枔 | 寤鸿锛氬弽褰堝仛绌猴紝涓嶈拷楂?

        # 8. 鎴愪氦閲忚悗绺┇鐩?        elif abs(slope_7) < 0.1 and vol_ratio < 0.7:
            pattern = "绺噺姗洡"
            direction = "涓€?
            score = 0; strength = 0.2
            detail = "BTC 鎴愪氦閲忚悗绺€佸児鏍兼┇鐩わ紝甯傚牬瑙€鏈涙儏绶掓績鍘?
            history_match = "姝峰彶灏嶆瘮锛氱府閲忔┇鐩ゅ緦閫氬父鏈変竴娉㈣純澶ц鎯?
            prediction = "钃勫嫝寰呯櫦鏂瑰悜鏈畾 | 寤鸿锛氱瓑寰呮斁閲忕獊鐮村緦璺熼€?

        # 闋愯ō锛氬急澶?寮辩┖
        else:
            if slope_14 > 0:
                pattern = "寮卞瓒ㄥ嫝"
                direction = "澶?
                score = 2; strength = 0.4
                detail = "BTC 鏃ョ窔鏂滅巼鍚戜笂锛屽急澶氭牸灞€"
                prediction = "瓒ㄥ嫝鍋忓浣嗗姏閬撲笉寮凤紝璎规厧鍋氬"
            else:
                pattern = "寮辩┖瓒ㄥ嫝"
                direction = "绌?
                score = -2; strength = 0.4
                detail = "BTC 鏃ョ窔鏂滅巼鍚戜笅锛屽急绌烘牸灞€"
                prediction = "瓒ㄥ嫝鍋忕┖浣嗗姏閬撲笉寮凤紝璎规厧鎿嶄綔"

        # 鐪熷姝峰彶鐩镐技搴︽瘮灏?        similar_cases = find_similar_history(df, current_window=20, top_n=3)
        if similar_cases:
            hist_lines = []
            bull_count = sum(1 for s in similar_cases if s['future_ret'] > 2)
            bear_count = sum(1 for s in similar_cases if s['future_ret'] < -2)
            for s in similar_cases:
                trend = "馃搱+{:.1f}%".format(s['future_ret']) if s['future_ret'] > 0 else "馃搲{:.1f}%".format(s['future_ret'])
                hist_lines.append("鐩镐技搴}% 鈫?寰岀簩30鏃}".format(s['similarity'], trend))
            hist_conclusion = "姝峰彶{}娆＄浉浼煎瀷鎱嬶細{}鐪嬪 / {}鐪嬬┖".format(
                len(similar_cases), bull_count, bear_count)
            history_match = hist_conclusion + " | " + " | ".join(hist_lines)
        else:
            # 鏁告摎涓嶈冻鏅傜敤鍨嬫厠鏂囧瓧鎻忚堪
            history_match = history_match or "鐩镐技搴︿笉瓒筹紙杩戞湡璧板嫝杓冪壒娈婏紝鐒￠珮搴︾浉浼兼鍙诧級"

        return {
            "pattern": pattern,
            "direction": direction,
            "score": score,
            "strength": strength,
            "detail": detail,
            "history_match": history_match,
            "prediction": prediction,
            "btc_price": round(curr, 2),
            "btc_change": round(change_pct, 2),
            "ma7": round(ma7, 2),
            "ma25": round(ma25, 2),
            "ma50": round(ma50, 2),
            "slope_7": round(slope_7, 3),
            "slope_30": round(slope_30, 3),
            "vol_ratio": round(vol_ratio, 2),
            "last_update": tw_now_str(),
        }
    except Exception as e:
        print("澶х洡鍒嗘瀽澶辨晽: {}".format(e))
        return None

def market_analysis_thread():
    """姣忓皬鏅傛洿鏂颁竴娆″ぇ鐩ゅ垎鏋?""
    print("澶х洡鍒嗘瀽鍩疯绶掑暉鍕?)
    time.sleep(20)  # 绛夋巸鎻忓煼琛岀窉鍏堝暉鍕?    while True:
        try:
            result = analyze_btc_market_trend()
            if result:
                with MARKET_LOCK:
                    MARKET_STATE.update(result)
                update_state(market_info=result)
                print("馃搳 澶х洡鍒嗘瀽: {} | {} | BTC {:.0f} ({:+.1f}%)".format(
                    result["pattern"], result["direction"],
                    result["btc_price"], result["btc_change"]))
                # 澶х洡鍒嗘瀽鑸囬暦鏈熷€変綅鍒ゆ柗鍒嗛洟锛岄渶缍撴柟鍚戠⒑瑾嶅緦鎵嶆渻鍒囨彌
                check_long_term_position()
        except Exception as e:
            print("澶х洡鍒嗘瀽鍩疯绶掗尟瑾? {}".format(e))
        time.sleep(3600)  # 姣忓皬鏅傛洿鏂?
# =====================================================
# 闀锋湡鍊変綅绯荤当锛堢崹绔嬫柤鐭窔7鍊嬪€変綅涔嬪锛?# =====================================================
LT_STATE = {
    "position": None,   # None / "long" / "short"
    "entry_price": 0.0,
    "entry_time": "",
    "symbol": "BTC/USDT:USDT",
    "contracts": 0.0,
    "unrealized_pnl": 0.0,
    "leverage": 5,      # 闀锋湡鍊変綅鐢ㄤ綆妲撴】
    "note": "",
}
LT_LOCK = threading.Lock()

# =====================================================
# FVG 闄愬児鎺涘柈杩借工绯荤当
# =====================================================
FVG_ORDERS = {}   # { symbol: { order_id, side, price, score, sl, tp, placed_time, support, resist } }
FVG_LOCK   = threading.Lock()

def register_fvg_order(symbol, order_id, side, price, score, sl, tp, support, resist, extra_meta=None):
    """鐧昏涓€绛?FVG 闄愬児鎺涘柈"""
    with FVG_LOCK:
        if symbol in FVG_ORDERS:
            print("鈿狅笍 FVG闃查噸瑜囷細{} 宸叉湁鎺涘柈锛岃烦閬?.format(symbol))
            return False
        FVG_ORDERS[symbol] = {
            "order_id":    order_id,
            "side":        side,
            "price":       price,
            "score":       score,
            "sl":          sl,
            "tp":          tp,
            "support":     support,
            "resist":      resist,
            "placed_time": tw_now_str("%H:%M:%S"),
            "created_ts":  time.time(),
            "curr_price":   price,
            "curr_score":   score,
            "status":      "鎺涘柈涓?,
        }
        if isinstance(extra_meta, dict):
            for key, value in extra_meta.items():
                if key in ('signal_payload', 'pending_fill_meta'):
                    continue
                FVG_ORDERS[symbol][key] = value
        print("馃搶 FVG鎺涘柈鐧昏: {} {} @{:.6f}".format(symbol, side, price))
        update_state(fvg_orders=dict(FVG_ORDERS))
        return True

def cancel_fvg_order(symbol, reason=""):
    """鍙栨秷涓︾櫥鍑轰竴绛?FVG 鎺涘柈"""
    with FVG_LOCK:
        if symbol not in FVG_ORDERS:
            return
        order = FVG_ORDERS.pop(symbol)
    with PENDING_LIMIT_LOCK:
        PENDING_LIMIT_META.pop(symbol, None)
    try:
        exchange.cancel_order(order["order_id"], symbol)
        print("馃棏 FVG鎺涘柈鍙栨秷: {} | 鍘熷洜: {}".format(symbol, reason))
    except Exception as e:
        print("FVG鍙栨秷澶辨晽(鍙兘宸叉垚浜?: {}".format(e))
    update_state(fvg_orders=dict(FVG_ORDERS))

def fvg_order_monitor_thread():
    """
    FVG 闄愬児鎺涘柈杩借工鍩疯绶掞紙姣?0绉掓鏌ヤ竴娆★級
    - 鎺涘柈鐙€鎱嬨€乼icker 鍥哄畾妾㈡煡
    - analyze(symbol) 鍙湪鍍规牸鎺ヨ繎澶辨晥鍗€鎴栧揩鍙栭亷鏈熸檪閲嶈窇
    - 鎺涘柈瓒呴亷4灏忔檪鏈垚浜?鈫?鍙栨秷
    """
    print("FVG鎺涘柈杩借工鍩疯绶掑暉鍕?)
    while True:
        try:
            with FVG_LOCK:
                syms = list(FVG_ORDERS.keys())
            for symbol in syms:
                try:
                    with FVG_LOCK:
                        if symbol not in FVG_ORDERS:
                            continue
                        order = dict(FVG_ORDERS[symbol])
                    with FVG_MONITOR_LOCK:
                        cache = FVG_MONITOR_CACHE.setdefault(symbol, {})
                    now_ts = time.time()
                    status = str(cache.get('order_status') or 'unknown')
                    if now_ts - float(cache.get('order_status_ts', 0) or 0) >= 20:
                        try:
                            od = exchange.fetch_order(order['order_id'], symbol)
                            status = od.get('status', '')
                            with FVG_MONITOR_LOCK:
                                cache['order_status'] = status
                                cache['order_status_ts'] = now_ts
                        except Exception:
                            pass
                    if status in ('closed', 'filled'):
                        with FVG_LOCK:
                            FVG_ORDERS.pop(symbol, None)
                        pending_meta = None
                        with PENDING_LIMIT_LOCK:
                            pending_meta = dict(PENDING_LIMIT_META.pop(symbol, None) or {})
                        print("鉁?FVG鎺涘柈鎴愪氦: {} @{}".format(symbol, order['price']))
                        if pending_meta:
                            try:
                                pending_sig = dict(pending_meta.get('signal') or {})
                                pending_sig['price'] = float(order.get('price', pending_sig.get('price', 0)) or pending_sig.get('price', 0))
                                pending_sig['decision_source'] = pending_sig.get('decision_source', 'openai')
                                finalize_open_position_entry(
                                    symbol,
                                    'buy' if str(order.get('side') or '').lower() == 'long' else 'sell',
                                    pending_sig,
                                    float(pending_meta.get('qty', 0) or 0),
                                    float(order.get('sl', pending_sig.get('stop_loss', 0)) or pending_sig.get('stop_loss', 0)),
                                    float(order.get('tp', pending_sig.get('take_profit', 0)) or pending_sig.get('take_profit', 0)),
                                    float(pending_meta.get('leverage', 1) or 1),
                                    float(pending_meta.get('order_usdt', 0) or 0),
                                    float(pending_meta.get('risk_usdt', 0) or 0),
                                    float(pending_meta.get('margin_pct', pending_sig.get('margin_pct', 0)) or pending_sig.get('margin_pct', 0)),
                                    dict(pending_meta.get('margin_ctx') or {}),
                                    protect=True,
                                )
                            except Exception as fill_err:
                                print("鎺涘柈鎴愪氦寰岃寤哄€夌媭鎱嬪け鏁?{}: {}".format(symbol, fill_err))
                        update_state(fvg_orders=dict(FVG_ORDERS))
                        continue
                    if status == 'canceled':
                        with FVG_LOCK:
                            FVG_ORDERS.pop(symbol, None)
                        with PENDING_LIMIT_LOCK:
                            PENDING_LIMIT_META.pop(symbol, None)
                        update_state(fvg_orders=dict(FVG_ORDERS))
                        continue
                    ticker = exchange.fetch_ticker(symbol)
                    curr = float(ticker['last'])
                    support = float(order.get('support', 0) or 0)
                    resist = float(order.get('resist', 0) or 0)
                    near_boundary = False
                    if order['side'] == 'long' and support > 0:
                        near_boundary = curr <= support * 1.003
                    elif order['side'] == 'short' and resist > 0:
                        near_boundary = curr >= resist * 0.997
                    sc = float(order.get('score', 0) or 0)
                    if near_boundary or (now_ts - float(cache.get('analysis_ts', 0) or 0) >= 180):
                        sc = extract_analysis_score(analyze(symbol))
                        with FVG_MONITOR_LOCK:
                            cache['analysis_score'] = sc
                            cache['analysis_ts'] = now_ts
                    else:
                        sc = float(cache.get('analysis_score', sc) or sc)
                    cancel_reason = None
                    ai_limit_cancel_price = float(order.get('limit_cancel_price', 0) or 0)
                    ai_limit_cancel_condition = str(order.get('limit_cancel_condition') or '')
                    ai_limit_cancel_note = str(order.get('limit_cancel_note') or '')
                    ai_limit_cancel_timeframe = str(order.get('limit_cancel_timeframe') or '')
                    if ai_limit_cancel_price > 0:
                        if order['side'] == 'long' and curr <= ai_limit_cancel_price:
                            cancel_reason = 'OpenAI鍙栨秷鎺涘柈 {:.6f} [{}] {}'.format(
                                ai_limit_cancel_price,
                                ai_limit_cancel_timeframe or 'price',
                                ai_limit_cancel_note or ai_limit_cancel_condition or '鍋氬鎺涘柈澶辨晥锛屽彇娑?,
                            )
                        elif order['side'] == 'short' and curr >= ai_limit_cancel_price:
                            cancel_reason = 'OpenAI鍙栨秷鎺涘柈 {:.6f} [{}] {}'.format(
                                ai_limit_cancel_price,
                                ai_limit_cancel_timeframe or 'price',
                                ai_limit_cancel_note or ai_limit_cancel_condition or '鍋氱┖鎺涘柈澶辨晥锛屽彇娑?,
                            )
                    if order['side'] == 'long' and sc < max(18, float(ORDER_THRESHOLD) * 0.55):
                        cancel_reason = '鍋氬鍒嗘暩涓嶈冻{}锛?30锛夛紝鍙栨秷鎺涘柈'.format(round(sc, 1))
                    elif order['side'] == 'short' and sc > -max(18, float(ORDER_THRESHOLD) * 0.55):
                        cancel_reason = '鍋氱┖鍒嗘暩涓嶈冻{}锛?-30锛夛紝鍙栨秷鎺涘柈'.format(round(sc, 1))
                    elif order['side'] == 'long' and support > 0 and curr < support * 0.998:
                        cancel_reason = '璺岀牬鏀拹{:.4f}锛屽彇娑堝仛澶氭帥鍠?.format(support)
                    elif order['side'] == 'short' and resist > 0 and curr > resist * 1.002:
                        cancel_reason = '绐佺牬澹撳姏{:.4f}锛屽彇娑堝仛绌烘帥鍠?.format(resist)
                    created_ts = float(order.get('created_ts', now_ts) or now_ts)
                    if not cancel_reason and (now_ts - created_ts) > 240 * 60:
                        cancel_reason = '鎺涘柈瓒呴亷4灏忔檪锛岃嚜鍕曞彇娑?
                    if cancel_reason:
                        cancel_fvg_order(symbol, cancel_reason)
                    else:
                        with FVG_LOCK:
                            if symbol in FVG_ORDERS:
                                FVG_ORDERS[symbol]['curr_price'] = round(curr, 6)
                                FVG_ORDERS[symbol]['curr_score'] = round(sc, 1)
                                FVG_ORDERS[symbol]['status'] = '鎺涘柈涓?| 鐝惧児{:.4f} | 鍒嗘暩{}'.format(curr, round(sc,1))
                        update_state(fvg_orders=dict(FVG_ORDERS))
                except Exception as e:
                    print('FVG杩借工{}閷: {}'.format(symbol, e))
        except Exception as e:
            print('FVG杩借工鍩疯绶掗尟瑾? {}'.format(e))
        time.sleep(30)

def open_long_term_position(direction, reason=""):
    """闁嬮暦鏈熷€変綅锛圔TC锛屼綆妲撴】5x锛?%璩囩敘锛?""
    try:
        with LT_LOCK:
            if LT_STATE["position"] is not None:
                print("闀锋湡鍊変綅宸插瓨鍦紝璺抽亷")
                return False

        ticker = exchange.fetch_ticker("BTC/USDT:USDT")
        price  = float(ticker['last'])
        with STATE_LOCK:
            equity = STATE.get("equity", 100)
        usdt   = equity * 0.05          # 鐢?%璩囩敘
        lev    = LT_STATE["leverage"]

        # 瑷畾妲撴】
        try:
            exchange.set_leverage(lev, "BTC/USDT:USDT")
        except:
            pass

        contracts = round(usdt * lev / price, 4)
        side = "buy" if direction == "long" else "sell"

        order = exchange.create_order(
            "BTC/USDT:USDT", "market", side, contracts,
            params={"tdMode": "cross"}
        )

        with LT_LOCK:
            LT_STATE["position"]    = direction
            LT_STATE["entry_price"] = price
            LT_STATE["entry_time"]  = tw_now_str("%Y-%m-%d %H:%M")
            LT_STATE["contracts"]   = contracts
            LT_STATE["note"]        = reason

        print("鉁?闀锋湡{}鍊夐枊鍊?BTC {:.2f} | {}寮?| 鍘熷洜:{}".format(
            "澶? if direction=="long" else "绌?,
            price, contracts, reason))
        return True
    except Exception as e:
        print("闀锋湡鍊変綅闁嬪€夊け鏁? {}".format(e))
        return False

def close_long_term_position(reason=""):
    """骞抽暦鏈熷€変綅"""
    try:
        with LT_LOCK:
            if LT_STATE["position"] is None:
                return False
            side      = LT_STATE["position"]
            contracts = LT_STATE["contracts"]

        close_side = "sell" if side == "long" else "buy"
        exchange.create_order(
            "BTC/USDT:USDT", "market", close_side, abs(contracts),
            params={"tdMode": "cross", "reduceOnly": True}
        )

        with LT_LOCK:
            entry = LT_STATE["entry_price"]
            LT_STATE["position"]  = None
            LT_STATE["contracts"] = 0.0

        ticker = exchange.fetch_ticker("BTC/USDT:USDT")
        curr   = float(ticker['last'])
        pnl    = (curr - entry) / entry * 100 if side=="long" else (entry - curr) / entry * 100
        print("馃摛 闀锋湡鍊変綅骞冲€?| 鎼嶇泭:{:+.2f}% | 鍘熷洜:{}".format(pnl, reason))
        return True
    except Exception as e:
        print("闀锋湡鍊変綅骞冲€夊け鏁? {}".format(e))
        return False

def check_long_term_position():
    """姣忓皬鏅傜敱澶х洡鍒嗘瀽鍩疯绶掑懠鍙紝鏍规摎澶х洡鏂瑰悜绠＄悊闀锋湡鍊変綅銆傞渶閫ｇ簩 2 娆″悓鏂瑰悜鎵嶅嫊浣滐紝闄嶄綆鑰﹀悎銆?""
    with MARKET_LOCK:
        direction = MARKET_STATE.get("direction", "涓€?)
        strength = MARKET_STATE.get("strength", 0)
        pattern = MARKET_STATE.get("pattern", "")
        prediction = MARKET_STATE.get("prediction", "")
    with LT_LOCK:
        curr_pos = LT_STATE["position"]
    if strength < 0.6:
        print("鈴?澶х洡寮峰害涓嶈冻({:.1f})锛岄暦鏈熷€変綅缍寔鐝剧媭".format(strength))
        return
    confirmed, confirm_count = MARKET_DIRECTION_GUARD.register(direction)
    if direction in ("寮峰", "澶?, "寮风┖", "绌?) and not confirmed:
        print("鈴?澶х洡鏂瑰悜 {} 绗?{} 娆＄⒑瑾嶏紝闀锋湡鍊変綅鏆笉鍒囨彌".format(direction, confirm_count))
        return
    if direction in ("寮峰", "澶?) and curr_pos != "long":
        if curr_pos == "short":
            close_long_term_position("鏂瑰悜杞夊锛屽钩绌哄€?)
        open_long_term_position("long", "{} | {}".format(pattern, prediction[:30]))
    elif direction in ("寮风┖", "绌?) and curr_pos != "short":
        if curr_pos == "long":
            close_long_term_position("鏂瑰悜杞夌┖锛屽钩澶氬€?)
        open_long_term_position("short", "{} | {}".format(pattern, prediction[:30]))
    elif direction == "涓€? and curr_pos is not None:
        close_long_term_position("澶х洡涓€э紝瑙€鏈?)

def check_risk_ok():
    """鍥炲偝 (鍙惁涓嬪柈, 鍘熷洜)锛堜笉鐢ㄩ帠锛岄伩鍏嶆閹栵級"""
    try:
        rs = RISK_STATE  # 鐩存帴璁€
        today = tw_today()

        # 鏂扮殑涓€澶╅噸缃棩铏ф悕
        if rs["today_date"] != today:
            rs["today_date"]         = today
            rs["daily_loss_usdt"]    = 0.0
            rs["daily_start_equity"] = STATE.get("equity", 0)
            rs["trading_halted"]     = False
            rs["halt_reason"]        = ""
            rs["cooldown_until"]     = None
            rs["consecutive_loss"]   = 0
            print("鏂扮殑涓€澶╋紝閲嶇疆棰ㄦ帶鐙€鎱?)

        if '绺借硣鐢㈣櫑鎼嶅凡閬? in str(rs.get("halt_reason", "") or ''):
            rs["trading_halted"] = False
            rs["halt_reason"] = ""

        # 鍍呬繚鐣欎汉宸?绯荤当绱氬仠鍠紝渚嬪淇濊鍠己澶憋紱绉婚櫎鏃ユ悕鑸囬€ｈ櫑鍋滃柈銆?        if rs["trading_halted"]:
            return False, rs["halt_reason"]

        return True, "姝ｅ父"
    except Exception as e:
        print("check_risk_ok 閷: {}".format(e))
        return True, "姝ｅ父"

def record_trade_result(pnl_usdt):
    """姣忕瓎骞冲€夊緦鍛煎彨锛屾洿鏂伴ⅷ鎺х媭鎱?""
    with RISK_LOCK:
        rs = RISK_STATE
        if pnl_usdt < 0:
            rs["daily_loss_usdt"]  += abs(pnl_usdt)
            rs["consecutive_loss"] += 1
            rs["cooldown_until"] = None
            append_risk_event('trade_loss', {
                'pnl_usdt': float(pnl_usdt or 0),
                'daily_loss_usdt': float(rs.get('daily_loss_usdt', 0) or 0),
                'consecutive_loss': int(rs.get('consecutive_loss', 0) or 0),
            })
        else:
            rs["consecutive_loss"] = 0  # 鍕濆埄閲嶇疆閫ｆ悕瑷堟暩
            rs["cooldown_until"] = None
            append_risk_event('trade_win_or_flat', {
                'pnl_usdt': float(pnl_usdt or 0),
                'daily_loss_usdt': float(rs.get('daily_loss_usdt', 0) or 0),
            })

def get_risk_status():
    """绲?UI 椤ず鐢紙涓嶇敤閹栵紝閬垮厤姝婚帠锛?""
    try:
        rs = RISK_STATE  # 鐩存帴璁€锛屼笉鍔犻帠
        if '绺借硣鐢㈣櫑鎼嶅凡閬? in str(rs.get("halt_reason", "") or ''):
            rs["trading_halted"] = False
            rs["halt_reason"] = ""
        ok = not rs.get("trading_halted", False)
        equity = STATE.get("equity", 1)
        start_eq = rs.get("daily_start_equity", equity) or equity
        return {
            "trading_ok":        ok,
            "halt_reason":       rs.get("halt_reason", ""),
            "consecutive_loss":  rs.get("consecutive_loss", 0),
            "daily_loss_usdt":   round(rs.get("daily_loss_usdt", 0), 2),
            "daily_loss_pct":    round((start_eq - equity) / max(start_eq, 1) * 100, 1) if equity > 0 else 0,
            "max_daily_loss_pct": int(MAX_DAILY_LOSS_PCT * 100),
            "cooldown_until":    None,
            "current_threshold": _DT.get("current", 50),
        }
    except Exception as e:
        return {"trading_ok": True, "halt_reason": "", "consecutive_loss": 0,
                "daily_loss_usdt": 0, "daily_loss_pct": 0, "max_daily_loss_pct": 15}



def _position_drawdown_pct(pos):
    try:
        entry = float(pos.get('entryPrice',0) or 0)
        mark = float(pos.get('markPrice',0) or 0)
        side = str(pos.get('side','') or '').lower()
        if entry <= 0 or mark <= 0:
            return 0.0
        if side == 'long':
            return round(max((entry - mark) / entry * 100.0, 0.0), 2)
        return round(max((mark - entry) / entry * 100.0, 0.0), 2)
    except Exception:
        return 0.0


def _position_leveraged_pnl_pct(pos):
    try:
        p = pos.get('percentage', None)
        if p is not None:
            return round(float(p or 0), 2)
    except Exception:
        pass
    try:
        dd = _position_drawdown_pct(pos)
        lev = float(pos.get('leverage',1) or 1)
        side = str(pos.get('side','') or '').lower()
        entry = float(pos.get('entryPrice',0) or 0)
        mark = float(pos.get('markPrice',0) or 0)
        favorable = (mark >= entry) if side == 'long' else (mark <= entry)
        signed = dd * lev
        return round(signed if favorable else -signed, 2)
    except Exception:
        return 0.0


def _manual_release_risk_state():
    with RISK_LOCK:
        RISK_STATE['trading_halted'] = False
        RISK_STATE['halt_reason'] = ''
        RISK_STATE['cooldown_until'] = None
        RISK_STATE['consecutive_loss'] = 0
    update_state(risk_status=get_risk_status())
    return {'ok': True, 'message': '宸叉墜鍕曡В闄らⅷ鎺ф毇鍋?}

# =====================================================
# 瑭曞垎娆婇噸锛堟豢鍒?00锛?# =====================================================
# =====================================================
# 鎸囨娆婇噸锛堟牴鎿?025閲忓寲鐮旂┒鏈€浣冲寲锛屾豢鍒?00锛?# 椤炲垾鍒嗛厤锛?#   瓒ㄥ嫝纰鸿獚(37): EMA+MACD+ADX+瓒ㄥ嫝绶?#   鍍规牸绲愭(29): 澹撳姏鏀拹+OB姗熸
#   娴侀噺/鍕曡兘(18): 娴佸嫊鎬?鎴愪氦閲?VWAP
#   鍕曡兘鎸洩(8):  RSI+KD
#   鎯呭婵剧恫(8):  K妫?鍦栧舰+鏂拌仦
# =====================================================


# =====================================================
# 瑭曞垎娆婇噸 - 鍚岄鎸囨鍏变韩绺藉垎闋愮畻锛屾湁骞惧€嬪氨闄ゅ咕鍊?# =====================================================
# 椤炲垾闋愮畻鍒嗛厤锛堢爺绌朵緷鎿氾細ICT/SMC + 閲忓寲鐮旂┒锛?# 瓒ㄥ嫝椤? 22鍒嗭細鏈€閲嶈锛岄伩鍏嶉€嗗嫝浜ゆ槗
# 绲愭椤? 22鍒嗭細OB/澹撳姏鏀拹鏄妲嬮€插嚭鏍稿績
# ICT椤?  20鍒嗭細BOS/CHoCH/鎺冨柈鏄従浠ｉ噺鍖栨牳蹇?# 鍕曢噺椤? 14鍒嗭細纰鸿獚鍕曡兘锛岄潪涓诲皫
# 閲忚兘椤? 12鍒嗭細璩囬噾娴佸悜椹楄瓑
# 鏂拌仦椤? 10鍒嗭細瀹忚鎯呯窉婵剧恫

# 鍚勯鍒?鈫?鎸囨娓呭柈锛堥噸绲勫緦锛?# 鏀瑰嫊瑾槑锛?#   KD 绉婚櫎锛堣垏RSI楂樺害閲嶈锛屾氮璨?鍒嗭級
#   鏂拌仦 10鈫?鍒嗭紙API涓嶇┅锛屼笉鎳変富灏庤鍒嗭級
#   chart_pat 闄嶇偤鐛ㄧ珛3鍒嗭紙浣庤Ц鐧肩巼锛?#   澶氭檪妗嗙⒑瑾?鏂板14鍒嗭紙15m+4H+鏃ョ窔涓€鑷达紝鏈€閲嶈鐨勫嫕鐜囦締婧愶級
_W_CAT = {
    "trend":     (22, ["ema_trend", "trendline", "adx"]),   # 22/3=7鍒唀ach
    "structure": (19, ["support_res", "order_block"]),       # 绉婚櫎chart_pat鍏变韩锛孫B+SR鍚?/10
    "ict":       (20, ["bos_choch", "liq_sweep", "candle", "fvg"]), # 20/4=5鍒唀ach
    "mtf":       (14, ["mtf_confirm"]),                      # 鈽呮柊澧烇細澶氭檪妗嗘柟鍚戜竴鑷?4鍒?    "momentum":  (10, ["macd", "rsi"]),                      # 绉婚櫎KD锛屽悇5鍒?    "volume":    (12, ["vwap", "whale"]),                    # 鍚?鍒?    "chart":     (3,  ["chart_pat"]),                        # 闄嶅埌3鍒嗭紙浣庤Ц鐧肩巼锛?    "news_cat":  (2,  ["news"]),                             # 闄嶅埌2鍒嗭紙涓嶇┅瀹氾級
}

W = {}
for cat, (budget, inds) in _W_CAT.items():
    per = round(budget / len(inds))
    for ind in inds:
        W[ind] = per

# 寰璁撶附鍒嗗墰濂?00
_total = sum(W.values())
if _total != 100:
    W["support_res"] += (100 - _total)

assert sum(W.values()) == 100, "娆婇噸绺藉拰{}涓嶇瓑鏂?00".format(sum(W.values()))

# 鏂拌仦鍠崹瑷堝垎
NEWS_WEIGHT = 0  # 鏂拌仦绯荤当宸插仠鐢紝涓嶅啀绱嶅叆鍒嗘暩


# =====================================================
# 瀛哥繏璩囨枡搴?/ SQLite 鍎插瓨灞?# =====================================================
def _default_learn_db_state():
    return {
            "trades": [],
            "pattern_stats": {},
            "symbol_stats": {},     # 姣忓€嬪梗鐨勫嫕鐜囩当瑷?            "atr_params": {"default_sl": 2.0, "default_tp": 3.5},
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
        }


def load_learn_db():
    try:
        return STORAGE.load_learning_state(default=_default_learn_db_state())
    except Exception as e:
        print("瀛哥繏DB璁€鍙栧け鏁楋紝鏀圭敤闋愯ō鍊? {}".format(e))
        return _default_learn_db_state()


def save_learn_db(db):
    try:
        STORAGE.save_learning_state(db)
    except Exception as e:
        print("瀛哥繏DB鍎插瓨澶辨晽: {}".format(e))


def load_backtest_db():
    try:
        return STORAGE.load_backtest_state(default={"runs": [], "summary": {}, "latest": {}})
    except Exception as e:
        print("鍥炴脯DB璁€鍙栧け鏁楋紝鏀圭敤闋愯ō鍊? {}".format(e))
        return {"runs": [], "summary": {}, "latest": {}}


def save_backtest_db(db):
    try:
        STORAGE.save_backtest_state(db)
    except Exception as e:
        print("鍥炴脯DB鍎插瓨澶辨晽: {}".format(e))


def persist_trade_history_record(rec):
    try:
        STORAGE.append_trade_history_record(rec)
    except Exception as e:
        print("trade_history 瀵叆 SQLite 澶辨晽: {}".format(e))


def hydrate_trade_history(limit=30):
    try:
        rows = STORAGE.load_recent_trade_history(limit=limit)
        if rows:
            with STATE_LOCK:
                STATE["trade_history"] = rows
    except Exception as e:
        print("trade_history 寰?SQLite 鎭㈠京澶辨晽: {}".format(e))


def append_risk_event(event_type, payload=None):
    try:
        STORAGE.append_risk_event(event_type, payload or {})
    except Exception as e:
        print("risk_event 瀵叆 SQLite 澶辨晽: {}".format(e))


def append_audit_log(category, message, payload=None):
    try:
        STORAGE.append_audit_log(category, message, payload or {})
    except Exception as e:
        print("audit_log 瀵叆 SQLite 澶辨晽: {}".format(e))


def _is_live_source(src):
    s = str(src or '').lower()
    return s.startswith('live')

def get_live_trades(closed_only=False, pool='all'):
    with LEARN_LOCK:
        trades = [enrich_learning_trade(dict(t or {}), reset_from=TREND_LEARNING_RESET_FROM) for t in list(LEARN_DB.get("trades", []) or [])]
    rows = [t for t in trades if _is_live_source(t.get("source"))]
    rows = filter_learning_pool(rows, pool=pool, closed_only=closed_only, reset_from=TREND_LEARNING_RESET_FROM)
    rows = _tag_dataset_layers(rows)
    return rows

def _parse_trade_time(trade):
    for key in ("exit_time", "entry_time"):
        raw = str((trade or {}).get(key) or "").strip()
        if not raw:
            continue
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(raw, fmt)
            except Exception:
                pass
    return None

def _legacy_new_split(rows):
    reset_raw = str(TREND_LEARNING_RESET_FROM or '').strip()
    if not reset_raw:
        return list(rows or []), []
    try:
        reset_dt = datetime.strptime(reset_raw, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return list(rows or []), []
    new_rows, legacy_rows = [], []
    for t in list(rows or []):
        trade_dt = _parse_trade_time(t)
        if trade_dt and trade_dt >= reset_dt:
            new_rows.append(t)
        else:
            legacy_rows.append(t)
    return new_rows, legacy_rows

def _tag_dataset_layers(rows):
    tagged = [dict(t or {}) for t in list(rows or [])]
    new_rows, legacy_rows = _legacy_new_split(tagged)
    new_ids = {str(t.get('id') or t.get('trade_id') or '') for t in new_rows}
    legacy_ids = {str(t.get('id') or t.get('trade_id') or '') for t in legacy_rows}
    out = []
    for t in tagged:
        tid = str(t.get('id') or t.get('trade_id') or '')
        layer = 'legacy'
        if tid and tid in new_ids:
            layer = 'new'
        elif tid and tid in legacy_ids:
            layer = 'legacy'
        else:
            trade_dt = _parse_trade_time(t)
            try:
                reset_dt = datetime.strptime(str(TREND_LEARNING_RESET_FROM or ''), "%Y-%m-%d %H:%M:%S")
                if trade_dt and trade_dt >= reset_dt:
                    layer = 'new'
            except Exception:
                pass
        t['dataset_layer'] = layer
        t['dataset_reset_from'] = TREND_LEARNING_RESET_FROM
        out.append(t)
    return out

def _legacy_trade_quality(trade):
    metric = float(_trade_learn_metric(trade) or 0.0)
    edge = float(_trade_edge_pct(trade) or 0.0)
    rr = float(trade.get('rr_ratio') or trade.get('rr') or ((trade.get('breakdown') or {}).get('RR')) or 0.0)
    result = str(trade.get('result') or '').lower()
    score = 0.0
    if result == 'win':
        score += 1.3
    elif result == 'loss':
        score -= 1.1
    score += max(min(metric / 1.8, 1.8), -1.8)
    score += max(min(edge / 0.9, 0.9), -0.9)
    score += max(min((rr - 1.15) * 0.9, 0.9), -0.9)
    setup_mode = _normalize_setup_mode(trade.get('setup_label') or ((trade.get('breakdown') or {}).get('Setup')) or '')
    if setup_mode == 'breakout' and result == 'loss':
        score -= 0.35
    return round(score, 4)

def _filter_legacy_bootstrap_rows(legacy_rows, new_rows=None):
    legacy_rows = list(legacy_rows or [])
    new_rows = list(new_rows or [])
    if not legacy_rows:
        return []
    if len(new_rows) >= LEGACY_BOOTSTRAP_MIN_NEW_TRADES:
        return []
    symbol_stats = {}
    for t in legacy_rows:
        sym = str(t.get('symbol') or '')
        rec = symbol_stats.setdefault(sym, {'count': 0, 'win': 0, 'metric': 0.0})
        rec['count'] += 1
        if str(t.get('result') or '').lower() == 'win':
            rec['win'] += 1
        rec['metric'] += float(_trade_learn_metric(t) or 0.0)
    filtered = []
    quarantine = []
    for t in legacy_rows:
        sym = str(t.get('symbol') or '')
        rec = symbol_stats.get(sym, {'count': 0, 'win': 0, 'metric': 0.0})
        cnt = int(rec.get('count', 0) or 0)
        wr = float(rec.get('win', 0) or 0) / max(cnt, 1)
        avg_metric = float(rec.get('metric', 0.0) or 0.0) / max(cnt, 1)
        q = _legacy_trade_quality(t)
        if cnt >= 10 and wr < 0.34 and avg_metric < -0.18:
            quarantine.append(t)
            continue
        if q <= -0.55:
            quarantine.append(t)
            continue
        filtered.append(t)
    try:
        LEARNING_DATASET_META['legacy_bootstrap_filtered'] = len(filtered)
        LEARNING_DATASET_META['legacy_bootstrap_quarantine'] = len(quarantine)
        LEARNING_DATASET_META['legacy_bootstrap_mode'] = 'mixed' if len(new_rows) < LEGACY_BOOTSTRAP_MIN_NEW_TRADES else 'new_only'
        LEARNING_DATASET_META['legacy_bootstrap_new_count'] = len(new_rows)
    except Exception:
        pass
    return filtered

def get_trend_live_trades(closed_only=False):
    rows = get_live_trades(closed_only=closed_only)
    new_rows, legacy_rows = _legacy_new_split(rows)
    try:
        LEARNING_DATASET_META['trend_new_count'] = len(new_rows)
        LEARNING_DATASET_META['trend_legacy_count'] = len(legacy_rows)
        LEARNING_DATASET_META['trend_reset_from'] = TREND_LEARNING_RESET_FROM
    except Exception:
        pass
    if not legacy_rows:
        return new_rows or rows
    if len(new_rows) >= LEGACY_BOOTSTRAP_MIN_NEW_TRADES:
        return new_rows
    legacy_filtered = _filter_legacy_bootstrap_rows(legacy_rows, new_rows=new_rows)
    merged = list(new_rows) + list(legacy_filtered)
    merged = _tag_dataset_layers(merged)
    return merged if merged else rows

def _trade_learn_metric(trade):
    """AI learning metric.
    Priority:
    - learn_pnl_pct: AI primary learning metric (prefer final realized close PnL converted from exchange)
    - leveraged_pnl_pct: realized ROI on used margin / leverage view
    - account_pnl_pct: realized account impact
    - pnl_pct: legacy fallback
    """
    if not isinstance(trade, dict):
        return 0.0
    for k in ("learn_pnl_pct", "leveraged_pnl_pct", "account_pnl_pct", "pnl_pct"):
        v = trade.get(k, None)
        if v is not None:
            try:
                return float(v or 0.0)
            except Exception:
                pass
    return 0.0

def _trade_edge_pct(trade):
    """Pure market edge without leverage. Useful for debug and strategy analysis."""
    if not isinstance(trade, dict):
        return 0.0
    for k in ("edge_pct", "raw_pnl_pct", "pnl_pct"):
        v = trade.get(k, None)
        if v is not None:
            try:
                return float(v or 0.0)
            except Exception:
                pass
    return 0.0

def _trend_learning_stage(closed_count=None, local_count=None, effective_count=None):
    cnt = len(get_trend_live_trades(closed_only=True)) if closed_count is None else int(closed_count or 0)
    local_cnt = cnt if local_count is None else int(local_count or 0)
    eff_cnt = float(cnt if effective_count is None else effective_count or 0)
    phase = phase_from_counts(cnt, local_cnt, eff_cnt)
    if phase == 'learning':
        return 'learning', 0.0
    if phase == 'semi':
        return 'semi', 0.5
    return 'full', 1.0

def _trade_post_move_profile(trade):
    if not isinstance(trade, dict):
        return {'run_pct': 0.0, 'pullback_pct': 0.0, 'continuation': False, 'reason': 'no_trade'}
    closes = [float(x) for x in (trade.get('post_candles') or []) if x is not None]
    exit_p = float(trade.get('exit_price', 0) or 0)
    leverage = max(float(trade.get('leverage', 1) or 1), 1.0)
    side = str(trade.get('side') or '').lower()
    if exit_p <= 0 or not closes:
        return {'run_pct': 0.0, 'pullback_pct': 0.0, 'continuation': False, 'reason': 'no_post_data'}
    if side == 'buy':
        run_pct = (max(closes) - exit_p) / max(exit_p, 1e-9) * 100.0 * leverage
        pullback_pct = max((exit_p - min(closes[:4] or closes)) / max(exit_p, 1e-9) * 100.0 * leverage, 0.0)
    else:
        run_pct = (exit_p - min(closes)) / max(exit_p, 1e-9) * 100.0 * leverage
        pullback_pct = max((max(closes[:4] or closes) - exit_p) / max(exit_p, 1e-9) * 100.0 * leverage, 0.0)
    learn_pnl = abs(float(trade.get('learn_pnl_pct', 0) or 0))
    min_run = max(TREND_EARLY_EXIT_MIN_RUN, learn_pnl * 0.75)
    max_pullback = max(TREND_EARLY_EXIT_MIN_EDGE, run_pct * 0.55)
    continuation = bool(run_pct >= min_run and pullback_pct <= max_pullback)
    reason = 'trend_continue' if continuation else 'normal_exit'
    return {
        'run_pct': round(run_pct, 4),
        'pullback_pct': round(pullback_pct, 4),
        'continuation': continuation,
        'reason': reason,
    }

def _trend_learning_profile(symbol='', regime='neutral', setup=''):
    rows = get_trend_live_trades(closed_only=True)
    stage, intervene_ratio = _trend_learning_stage(len(rows))
    regime = str(regime or 'neutral')
    setup_mode = _normalize_setup_mode(setup)

    def _match(items, by_symbol=False, by_regime=False):
        out = []
        for t in items:
            if by_symbol and str(t.get('symbol') or '') != str(symbol or ''):
                continue
            bd = dict(t.get('breakdown') or {})
            if by_regime and str(bd.get('Regime', 'neutral') or 'neutral') != regime:
                continue
            if setup_mode and _normalize_setup_mode(t.get('setup_label') or bd.get('Setup', '')) != setup_mode:
                continue
            out.append(t)
        return out

    local = _match(rows, by_symbol=True, by_regime=True)
    if len(local) >= 6:
        source = 'local'
        picked = local
    else:
        sym_rows = _match(rows, by_symbol=True, by_regime=False)
        if len(sym_rows) >= 8:
            source = 'symbol'
            picked = sym_rows
        else:
            reg_rows = _match(rows, by_symbol=False, by_regime=True)
            if len(reg_rows) >= 12:
                source = 'regime'
                picked = reg_rows
            else:
                source = 'global'
                picked = rows

    if not picked:
        return {
            'stage': stage, 'intervene_ratio': intervene_ratio, 'count': 0, 'continuation_rate': 0.0,
            'avg_run_pct': 0.0, 'avg_pullback_pct': 0.0, 'hold_bias': 0.0, 'source': 'none', 'note': '瓒ㄥ嫝妯ｆ湰涓嶈冻锛堥噸缃緦閲嶆柊绱锛?
        }

    profs = [_trade_post_move_profile(t) for t in picked]
    cont_hits = [p for p in profs if p.get('continuation')]
    count = len(profs)
    cont_rate = len(cont_hits) / max(count, 1)
    avg_run = sum(float(p.get('run_pct', 0) or 0) for p in profs) / max(count, 1)
    avg_pull = sum(float(p.get('pullback_pct', 0) or 0) for p in profs) / max(count, 1)
    if count >= 6 and cont_rate >= 0.38 and avg_run > max(avg_pull * 1.15, 0.9):
        hold_bias = min(1.0, (cont_rate - 0.30) * 1.9 + min(avg_run / max(avg_pull + 0.25, 1.0), 2.0) * 0.18)
    elif count >= 6 and cont_rate <= 0.18:
        hold_bias = -min(0.75, (0.24 - cont_rate) * 2.6)
    else:
        hold_bias = 0.0
    note = f'瓒ㄥ嫝瀛哥繏:{source}|妯ｆ湰{count}|寤剁簩鐜噞cont_rate*100:.0f}%|run{avg_run:.2f}|pull{avg_pull:.2f}'
    return {
        'stage': stage, 'intervene_ratio': intervene_ratio, 'count': count,
        'continuation_rate': round(cont_rate, 4), 'avg_run_pct': round(avg_run, 4),
        'avg_pullback_pct': round(avg_pull, 4), 'hold_bias': round(hold_bias, 4),
        'source': source, 'note': note,
    }

def _ui_trend_payload(symbol='', regime='neutral', setup=''):
    try:
        prof = _trend_learning_profile(symbol=symbol, regime=regime, setup=setup)
        stage = str(prof.get('stage') or 'learning')
        hold_bias = float(prof.get('hold_bias', 0.0) or 0.0)
        cont_rate = float(prof.get('continuation_rate', 0.0) or 0.0)
        count = int(prof.get('count', 0) or 0)
        intervene_ratio = float(prof.get('intervene_ratio', 0.0) or 0.0)
        source = str(prof.get('source') or 'none')
        source_bonus = {'local': 5.0, 'symbol': 4.0, 'regime': 2.5, 'global': 1.0}.get(source, 0.0)
        if stage == 'learning':
            confidence = min(58.0, 14.0 + count * 1.05 + cont_rate * 22.0 + source_bonus + max(hold_bias, 0.0) * 9.0)
        elif stage == 'semi':
            confidence = min(84.0, 34.0 + count * 0.38 + max(hold_bias, 0.0) * 26.0 + cont_rate * 24.0 + intervene_ratio * 10.0 + source_bonus)
        else:
            confidence = min(97.0, 46.0 + count * 0.22 + max(hold_bias, 0.0) * 29.0 + cont_rate * 28.0 + intervene_ratio * 12.0 + source_bonus)
        if hold_bias < -0.08:
            confidence = max(12.0, confidence - min(18.0, abs(hold_bias) * 22.0))
        hold_reason = 'trend_continuation' if hold_bias > 0.10 and stage in ('semi', 'full') else 'trend_caution' if hold_bias < -0.10 else 'normal_manage'
        mode_label = {'learning': 'learning', 'semi': 'partial', 'full': 'full'}.get(stage, 'learning')
        return {
            'trend_mode': mode_label,
            'hold_reason': hold_reason,
            'trend_confidence': round(max(0.0, min(confidence, 99.0)), 1),
            'trend_learning_count': count,
            'trend_continuation_rate': round(cont_rate * 100.0, 1),
            'trend_hold_bias': round(hold_bias, 4),
            'trend_note': str(prof.get('note') or ''),
            'trend_source': str(prof.get('source') or 'none'),
            'trend_avg_run_pct': float(prof.get('avg_run_pct', 0.0) or 0.0),
            'trend_avg_pullback_pct': float(prof.get('avg_pullback_pct', 0.0) or 0.0),
        }
    except Exception as e:
        return {
            'trend_mode': 'learning',
            'hold_reason': 'normal_manage',
            'trend_confidence': 0.0,
            'trend_learning_count': 0,
            'trend_continuation_rate': 0.0,
            'trend_hold_bias': 0.0,
            'trend_note': f'瓒ㄥ嫝璩囨枡閷: {e}',
            'trend_source': 'error',
            'trend_avg_run_pct': 0.0,
            'trend_avg_pullback_pct': 0.0,
        }

def _live_trade_stats(symbol=None, regime=None):
    rows = get_live_trades(closed_only=True)
    if symbol:
        rows = [t for t in rows if str(t.get("symbol")) == str(symbol)]
    if regime:
        rows = [t for t in rows if str((t.get("breakdown") or {}).get("Regime", "neutral")) == str(regime)]
    if not rows:
        return {"count": 0, "win_rate": 0.0, "avg_pnl": 0.0, "ev_per_trade": 0.0, "profit_factor": None, "max_drawdown_pct": None, "std_pnl": None}
    pnls = [_trade_learn_metric(t) for t in rows]
    wins = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p < 0]
    cnt = len(pnls)
    wr = (len(wins) / max(cnt, 1)) * 100.0
    avg = sum(pnls) / max(cnt, 1)
    pf = (sum(wins) / max(sum(losses), 1e-9)) if losses else (999.0 if wins else None)

    # v38锛氱敤 100 鍩烘簴娣ㄥ€兼洸绶氳▓绠楀洖鎾わ紝閬垮厤灏忔ǎ鏈檪鍑虹従 3000%+ 鍋囧洖鎾?    equity = 100.0
    peak = 100.0
    max_dd = 0.0
    for p in pnls:
        step = max(0.01, 1.0 + (float(p) / 100.0))
        equity *= step
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100.0)
    max_dd = min(max_dd, 100.0)

    mean = avg
    std = (sum((p - mean) ** 2 for p in pnls) / max(cnt, 1)) ** 0.5
    return {
        "count": cnt,
        "win_rate": round(wr, 2),
        "avg_pnl": round(avg, 4),
        "ev_per_trade": round(avg, 4),
        "profit_factor": None if pf is None else round(float(pf), 3),
        "max_drawdown_pct": round(float(max_dd), 3),
        "std_pnl": round(float(std), 4),
    }

def _ai_confidence_from_live(stats):
    cnt = int(stats.get("count", 0) or 0)
    std = float(stats.get("std_pnl", 0) or 0)
    base = min(cnt / 50.0, 1.0)
    stability = max(0.0, 1.0 - min(std / 3.0, 1.0))
    return round(base * stability, 3)

def _ai_status_from_live(stats):
    cnt = int(stats.get("count", 0) or 0)
    pf = stats.get("profit_factor", None)
    ev = float(stats.get("ev_per_trade", 0) or 0)
    dd = float(stats.get("max_drawdown_pct", 0) or 0)
    wr = float(stats.get("win_rate", 0) or 0)
    avg = float(stats.get("avg_pnl", 0) or 0)
    # v38锛氫笁闅庢锛屼笉璁?AI 澶棭鎺ョ
    if cnt < TREND_AI_SEMI_TRADES:
        return "warmup"
    if cnt < 50:
        return "observe"
    if (pf is not None and float(pf) < 0.95) or ev <= 0 or dd > 35 or wr < 42 or avg <= -0.25:
        return "reject"
    return "valid"

def _normalize_setup_mode(setup=''):
    s = str(setup or '')
    sl = s.lower()
    if ('绐佺牬' in s) or ('鐖嗙櫦' in s) or ('news' in sl) or ('breakout' in sl):
        return 'breakout'
    if (
        ('鍗€闁? in s) or ('闇囩洩' in s) or ('绠遍珨' in s) or ('鍧囧€煎洖姝? in s)
        or ('鎺冧綆鍥炴敹' in s) or ('鎺冮珮鍥炶惤' in s) or ('鍥炶' in s)
        or ('range' in sl) or ('mean reversion' in sl)
    ):
        return 'range'
    if ('鍥炶俯' in s) or ('绾屾敾' in s) or ('寤剁簩' in s) or ('鍙嶅綀绾岃穼' in s):
        return 'trend'
    return 'main'

def _regime_setup_fit(regime='neutral', setup=''):
    mode = _normalize_setup_mode(setup)
    regime = str(regime or 'neutral')
    if regime == 'range':
        if mode in ('trend', 'breakout'):
            return False, '闇囩洩甯傚牬涓嶈拷瓒ㄥ嫝/绐佺牬'
        return True, '闇囩洩甯傚牬閰嶅崁闁撴墦娉?
    if regime in ('news', 'breakout'):
        if mode == 'range':
            return False, '鐖嗙櫦鐩や笉鍋氬崁闁撳弽鍚?
        return True, '鐖嗙櫦鐩ゅ厑瑷辩獊鐮?瓒ㄥ嫝'
    # neutral / trend-like
    if mode == 'range':
        return False, '瓒ㄥ嫝/涓€х洡涓嶅劒鍏堝仛鍗€闁撻€嗗嫝'
    return True, '甯傚牬鑸囩瓥鐣ョ浉瀹?


def _normalize_market_state(state='neutral'):
    s = str(state or 'neutral').strip().lower()
    mapping = {
        'trend_pullback': 'trend_pullback',
        'pullback': 'trend_pullback',
        'trend_continuation': 'trend_continuation',
        'trend': 'trend_continuation',
        'range': 'range_rotation',
        'range_rotation': 'range_rotation',
        'breakout': 'breakout_ready',
        'breakout_ready': 'breakout_ready',
        'squeeze': 'squeeze_ready',
        'squeeze_ready': 'squeeze_ready',
        'fake_breakout': 'fake_breakout_reversal',
        'fake_breakout_reversal': 'fake_breakout_reversal',
        'reversal': 'fake_breakout_reversal',
        'news': 'news_expansion',
        'news_expansion': 'news_expansion',
        'neutral': 'neutral_transition',
        'neutral_transition': 'neutral_transition',
    }
    return mapping.get(s, s or 'neutral_transition')


def _classify_market_atlas(regime='neutral', setup='', breakdown=None, desc=''):
    bd = dict(breakdown or {})
    regime = str(regime or bd.get('Regime') or 'neutral')
    setup_text = str(setup or bd.get('Setup') or '')
    desc_text = str(desc or '')
    pre_s = float(bd.get('钃勫嫝绲愭', 0) or 0)
    bo_s = float(bd.get('绐佺牬闋愬垽', 0) or 0)
    fvg_rt = float(bd.get('FVG鍥炶俯鍝佽唱', 0) or 0)
    fake_s = float(bd.get('鍋囩獊鐮存烤缍?, 0) or 0)
    liq_s = float(bd.get('娴佸嫊鎬ф巸鍠?, 0) or 0)
    entry_q = float(bd.get('閫插牬鍝佽唱', 0) or 0)
    reg_conf = float(bd.get('RegimeConf', 0) or 0)
    state = 'neutral_transition'
    confidence = 0.38 + min(abs(entry_q) / 18.0, 0.12)
    note = '涓€ч亷娓?

    if fake_s <= -3 or '鍋囩獊鐮? in desc_text or '鎺冩祦鍕曟€у弽杞? in setup_text:
        state = 'fake_breakout_reversal'
        confidence = 0.68 + min(abs(fake_s) / 10.0, 0.22)
        note = '鍋囩獊鐮?鎺冩祦鍕曟€у緦鍙嶈綁'
    elif regime in ('news', 'breakout') and (bo_s >= 2 or pre_s >= 2):
        state = 'news_expansion' if regime == 'news' else 'breakout_ready'
        confidence = 0.66 + min((abs(bo_s) + abs(pre_s)) / 14.0, 0.24)
        note = '娑堟伅/鐖嗙櫦鎿村嫉绲愭'
    elif pre_s >= 3 and bo_s >= 2:
        state = 'squeeze_ready'
        confidence = 0.64 + min((pre_s + bo_s) / 16.0, 0.22)
        note = '鏀舵杺寰屾簴鍌欑獊鐮?
    elif fvg_rt >= 2 or ('鍥炶俯' in setup_text and regime in ('trend', 'neutral')):
        state = 'trend_pullback'
        confidence = 0.60 + min(abs(fvg_rt) / 12.0, 0.22)
        note = '瓒ㄥ嫝鍥炶俯/鍙嶅綀鎵挎帴'
    elif regime == 'range' or ('鍗€闁? in setup_text and abs(liq_s) <= 2):
        state = 'range_rotation'
        confidence = 0.58 + min(abs(entry_q) / 16.0, 0.18)
        note = '鍗€闁撲締鍥炶吉鍕?
    elif regime == 'trend' or ('寤剁簩' in setup_text) or abs(liq_s) >= 3:
        state = 'trend_continuation'
        confidence = 0.60 + min((abs(liq_s) + abs(entry_q)) / 18.0, 0.22)
        note = '瓒ㄥ嫝寤剁簩/闋嗗嫝鍔犻€?
    confidence = max(0.35, min(confidence + min(reg_conf * 0.12, 0.08), 0.95))
    return _normalize_market_state(state), round(confidence, 3), note


def _market_state_from_trade(trade):
    bd = dict((trade or {}).get('breakdown') or {})
    state = str(bd.get('MarketState') or '')
    if state:
        return _normalize_market_state(state)
    regime = str(bd.get('Regime', 'neutral') or 'neutral')
    setup = str((trade or {}).get('setup_label') or bd.get('Setup') or '')
    desc = str((trade or {}).get('desc') or '')
    state, _, _ = _classify_market_atlas(regime=regime, setup=setup, breakdown=bd, desc=desc)
    return state


def _market_state_profile(symbol='', regime='neutral', setup='', market_state=''):
    rows = get_trend_live_trades(closed_only=True)
    wanted = _normalize_market_state(market_state or 'neutral_transition')
    symbol = str(symbol or '')
    regime = str(regime or 'neutral')
    setup_mode = _normalize_setup_mode(setup)
    state_rows = []
    symbol_state_rows = []
    regime_state_rows = []
    for t in rows:
        bd = dict(t.get('breakdown') or {})
        t_state = _market_state_from_trade(t)
        if t_state != wanted:
            continue
        state_rows.append(t)
        if symbol and str(t.get('symbol') or '') == symbol:
            symbol_state_rows.append(t)
        if str(bd.get('Regime', 'neutral') or 'neutral') == regime:
            regime_state_rows.append(t)
    picked = []
    source = 'none'
    if len(symbol_state_rows) >= 5:
        picked = symbol_state_rows
        source = 'symbol_state'
    elif len(regime_state_rows) >= 7:
        picked = regime_state_rows
        source = 'regime_state'
    elif len(state_rows) >= 9:
        picked = state_rows
        source = 'market_state'
    if not picked:
        return {
            'market_state': wanted, 'count': 0, 'win_rate': 50.0, 'avg_pnl': 0.0,
            'confidence': 0.0, 'source': source, 'boost': 0.0,
            'note': f'甯傚牬鐙€鎱嬫ǎ鏈笉瓒?{wanted}'
        }
    count = len(picked)
    wins = sum(1 for t in picked if t.get('result') == 'win')
    win_rate = wins / max(count, 1) * 100.0
    avg_pnl = sum(_trade_learn_metric(t) for t in picked) / max(count, 1)
    confidence = min(0.95, 0.35 + count / 24.0)
    boost = (win_rate - 50.0) * 0.025 + avg_pnl * 10.0
    if setup_mode:
        matched_setup = [t for t in picked if _normalize_setup_mode((t.get('setup_label') or (t.get('breakdown') or {}).get('Setup') or '')) == setup_mode]
        if len(matched_setup) >= 4:
            count2 = len(matched_setup)
            wins2 = sum(1 for t in matched_setup if t.get('result') == 'win')
            win_rate2 = wins2 / max(count2, 1) * 100.0
            avg_pnl2 = sum(_trade_learn_metric(t) for t in matched_setup) / max(count2, 1)
            win_rate = (win_rate * 0.45) + (win_rate2 * 0.55)
            avg_pnl = (avg_pnl * 0.45) + (avg_pnl2 * 0.55)
            boost += (win_rate2 - 50.0) * 0.01 + avg_pnl2 * 4.0
            source += '+setup'
    return {
        'market_state': wanted, 'count': count, 'win_rate': round(win_rate, 2), 'avg_pnl': round(avg_pnl, 4),
        'confidence': round(confidence, 3), 'source': source,
        'boost': round(max(min(boost, 4.5), -4.5), 3),
        'note': f'甯傚牬瀛哥繏:{wanted}|{source}|妯ｆ湰{count}|鍕濈巼{win_rate:.1f}%|鍧囧埄{avg_pnl:+.3f}'
    }


def _symbol_hard_block(symbol=''):
    rows = [t for t in get_live_trades(closed_only=True) if str(t.get('symbol')) == str(symbol)]
    cnt = len(rows)
    if cnt < SYMBOL_BLOCK_MIN_TRADES:
        return False, ''
    wins = sum(1 for t in rows if t.get('result') == 'win')
    wr = wins / max(cnt, 1) * 100.0
    if wr < SYMBOL_BLOCK_MIN_WINRATE:
        return True, '瑭插梗瀵﹀柈瓒呴亷10绛嗕笖鍕濈巼浣庢柤40%锛屽皝閹栧梗绋?
    return False, ''


def _strategy_live_rows(symbol='', regime='neutral', setup=''):
    setup_mode = _normalize_setup_mode(setup)
    rows = []
    for t in get_live_trades(closed_only=True):
        if str(t.get('symbol') or '') != str(symbol):
            continue
        bd = dict(t.get('breakdown') or {})
        t_regime = str(bd.get('Regime', 'neutral') or 'neutral')
        t_setup = _normalize_setup_mode(t.get('setup_label') or bd.get('Setup') or t.get('setup') or '')
        if t_regime == str(regime or 'neutral') and t_setup == setup_mode:
            rows.append(t)
    return rows


def _strategy_hard_block(symbol='', regime='neutral', setup=''):
    rows = _strategy_live_rows(symbol=symbol, regime=regime, setup=setup)
    cnt = len(rows)
    if cnt < STRATEGY_BLOCK_MIN_TRADES:
        return False, ''
    wins = sum(1 for t in rows if t.get('result') == 'win')
    wr = wins / max(cnt, 1) * 100.0
    if wr < STRATEGY_BLOCK_MIN_WINRATE:
        return True, f'瑭茬瓥鐣ュ鍠秴閬?0绛嗕笖鍕濈巼浣庢柤{int(STRATEGY_BLOCK_MIN_WINRATE)}%锛屽皝閹栫瓥鐣?
    return False, ''


def _strategy_score_lookup(symbol='', regime='neutral', setup=''):
    setup_mode = _normalize_setup_mode(setup)
    wanted = f'{regime}|{setup}|{symbol}'
    best = None
    try:
        with AI_LOCK:
            board = list(AI_DB.get('strategy_scoreboard', []) or [])
            bt_rows = list((AUTO_BACKTEST_STATE.get('results') or []))
        for row in board:
            if str(row.get('strategy') or '') == wanted:
                best = dict(row)
                best['source'] = 'live_exact'
                return best
            if str(row.get('strategy') or '').endswith(f'|{symbol}') and str(row.get('strategy_mode') or 'main') == setup_mode:
                best = dict(row)
                best['source'] = 'live_mode'
        if best:
            return best
        for row in bt_rows:
            if str(row.get('symbol') or '') != str(symbol):
                continue
            row_mode = str(row.get('strategy_mode') or 'main')
            row_regime = str(row.get('market_regime') or 'neutral')
            if row_mode == setup_mode and row_regime == str(regime or 'neutral'):
                out = dict(row)
                out['count'] = int(row.get('trades', 0) or 0)
                out['source'] = 'backtest_exact'
                return out
        for row in bt_rows:
            if str(row.get('symbol') or '') == str(symbol) and str(row.get('strategy_mode') or 'main') == setup_mode:
                out = dict(row)
                out['count'] = int(row.get('trades', 0) or 0)
                out['source'] = 'backtest_mode'
                return out
    except Exception:
        pass
    return {}


def _strategy_margin_multiplier(symbol='', regime='neutral', setup=''):
    row = _strategy_score_lookup(symbol=symbol, regime=regime, setup=setup)
    count = int(row.get('count', row.get('trades', 0)) or 0)
    if count < STRATEGY_CAPITAL_MIN_TRADES:
        return 1.0, '绛栫暐妯ｆ湰涓嶈冻'
    ev = float(row.get('ev_per_trade', 0) or 0)
    wr = float(row.get('win_rate', 0) or 0)
    dd = float(row.get('max_drawdown_pct', 0) or 0)
    mult = 1.0
    note = '绛栫暐璩囬噾涓€?
    if ev >= 0.05 and wr >= 55 and dd <= 12:
        mult = 1.18 if count < 10 else 1.28
        note = '绛栫暐璩囬噾鏀惧ぇ'
    elif ev < 0 or wr < 45:
        mult = 0.72 if count >= 8 else 0.85
        note = '绛栫暐璩囬噾绺皬'
    return round(clamp(mult, 0.65, 1.35), 4), note


def _entry_quality_feedback(symbol='', regime='neutral', setup='', entry_quality=0):
    try:
        with AI_LOCK:
            eq_db = dict((AI_DB.get('entry_quality_feedback', {}) or {}))
        bin_key = 'hq' if float(entry_quality or 0) >= 7 else 'mq' if float(entry_quality or 0) >= 5 else 'lq'
        lookup_keys = [
            f'{symbol}|{regime}|{_normalize_setup_mode(setup)}|{bin_key}',
            f'{symbol}|{regime}|all|{bin_key}',
            f'all|{regime}|{_normalize_setup_mode(setup)}|{bin_key}',
        ]
        for key in lookup_keys:
            rec = dict(eq_db.get(key) or {})
            count = int(rec.get('count', 0) or 0)
            if count < AI_MIN_SAMPLE_EFFECT:
                continue
            loss_rate = float(rec.get('loss_rate', 0) or 0)
            avg = float(rec.get('avg_pnl', 0) or 0)
            if loss_rate >= 0.6 and avg < 0:
                return -2.5, '楂樺搧璩▕铏熻繎鏈熷け鐪?
            if loss_rate <= 0.35 and avg > 0:
                return 1.2, '閫插牬鍝佽唱鍥為浣?
    except Exception:
        pass
    return 0.0, ''


def _ai_risk_multiplier(symbol='', regime='neutral', setup='', score=0, breakdown=None):
    profile = _ai_strategy_profile(symbol, regime=regime, setup=setup)
    confidence = float(profile.get('confidence', 0) or 0)
    ev = float(profile.get('ev_per_trade', 0) or 0)
    wr = float(profile.get('win_rate', 0) or 0)
    mult = 1.0
    note = 'AI棰ㄦ帶涓€?
    if bool(profile.get('hard_block')):
        return 0.55, 'AI棰ㄦ帶灏侀帠绺€?
    if confidence < 0.5:
        mult *= 0.75
        note = 'AI淇″績涓嶈冻绺€?
    if ev > 0.05 and wr >= 55 and confidence >= 0.55:
        mult *= 1.08
        note = 'AI淇″績浣冲井鏀惧ぇ'
    elif ev < 0 or wr < 45:
        mult *= 0.78
        note = 'AI寮卞嫝绺€?
    if NEUTRAL_REGIME_BLOCK and str(regime or 'neutral') == 'neutral':
        mult *= 0.92
    return round(clamp(mult, 0.5, 1.2), 4), note


def _missed_move_feedback(trade):
    try:
        missed = float(trade.get('missed_move_pct', 0) or 0)
        pnl = float(_trade_learn_metric(trade) or 0)
        if missed > 2.0 and pnl >= 0:
            return 'stretch'
        if missed < 0.5 and pnl < 0:
            return 'tighten'
    except Exception:
        pass
    return ''


def _ai_warmup_mode():
    return len(get_live_trades(closed_only=True)) < 20

def save_full_state():
    """鍌欎唤绉诲嫊姝㈢泩鐙€鎱嬪埌纭"""
    try:
        dir2 = os.path.dirname(STATE_BACKUP_PATH)
        if dir2: os.makedirs(dir2, exist_ok=True)
        with TRAILING_LOCK:
            trail_copy = {k: {
                "side": v.get("side"),
                "entry_price": v.get("entry_price"),
                "highest_price": v.get("highest_price"),
                "lowest_price": v.get("lowest_price"),
                "trail_pct": v.get("trail_pct"),
                "initial_sl": v.get("initial_sl"),
                "atr": v.get("atr"),
            } for k, v in TRAILING_STATE.items()}
        backup = {
            "trailing_state": trail_copy,
            "threshold": {"current": _DT.get("current", ORDER_THRESHOLD_DEFAULT)},
            "timestamp": datetime.now().isoformat()
        }
        with open(STATE_BACKUP_PATH, 'w', encoding='utf-8') as f:
            json.dump(backup, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("鐙€鎱嬪倷浠藉け鏁? {}".format(e))

def load_full_state():
    """寰炵‖纰熸仮寰╃Щ鍕曟鐩堢媭鎱?""
    global ORDER_THRESHOLD
    try:
        if not os.path.exists(STATE_BACKUP_PATH):
            print("鈿狅笍 鐒＄媭鎱嬪倷浠斤紝寰為牠闁嬪")
            return
        with open(STATE_BACKUP_PATH, 'r', encoding='utf-8') as f:
            backup = json.load(f)
        with TRAILING_LOCK:
            for sym, ts in backup.get("trailing_state", {}).items():
                TRAILING_STATE[sym] = ts
        thresh = float(backup.get('threshold', {}).get('current', ORDER_THRESHOLD_DEFAULT) or ORDER_THRESHOLD_DEFAULT)
        thresh = max(46.0, min(72.0, thresh))
        with _DT_LOCK:
            _DT['current'] = thresh
        ORDER_THRESHOLD = thresh
        print('鉁?鐙€鎱嬪凡寰炲倷浠芥仮寰╋紝AI闁€妾?{}' .format(thresh))
    except FileNotFoundError:
        print("鈿狅笍 鐒＄媭鎱嬪倷浠斤紝寰為牠闁嬪")
    except Exception as e:
        print("鐙€鎱嬫仮寰╁け鏁? {}".format(e))

def save_risk_state():
    """鍌欎唤棰ㄦ帶鐙€鎱嬶紙JSON 鍙暀蹇収锛屼簨浠堕€?SQLite锛?""
    try:
        snapshot = {
            "today_date": RISK_STATE.get("today_date", ""),
            "daily_loss_usdt": RISK_STATE.get("daily_loss_usdt", 0),
            "consecutive_loss": RISK_STATE.get("consecutive_loss", 0),
            "trading_halted": RISK_STATE.get("trading_halted", False),
            "halt_reason": RISK_STATE.get("halt_reason", ""),
            "timestamp": datetime.now().isoformat()
        }
        atomic_json_save(RISK_STATE_PATH, snapshot, ensure_ascii=False, indent=2)
        append_risk_event('snapshot_saved', snapshot)
    except Exception as e:
        print("棰ㄦ帶鍌欎唤澶辨晽: {}".format(e))

def load_risk_state():
    """寰炵‖纰熸仮寰╅ⅷ鎺х媭鎱?""
    try:
        backup = atomic_json_load(RISK_STATE_PATH, None)
        if not backup:
            print("鈿狅笍 鐒￠ⅷ鎺у倷浠斤紝寰為牠闁嬪")
            return
        # 鍙仮寰╀粖澶╃殑璩囨枡
        today = tw_today()
        if backup.get("today_date") == today:
            with RISK_LOCK:
                RISK_STATE["today_date"]      = today
                RISK_STATE["daily_loss_usdt"] = backup.get("daily_loss_usdt", 0)
                RISK_STATE["consecutive_loss"]= backup.get("consecutive_loss", 0)
                RISK_STATE["trading_halted"]  = backup.get("trading_halted", False)
                RISK_STATE["halt_reason"]     = backup.get("halt_reason", "")
            print("鉁?棰ㄦ帶鐙€鎱嬪凡鎭㈠京锛堜粖鏃ヨ櫑鎼?{:.2f}U锛?.format(
                backup.get("daily_loss_usdt", 0)))
            append_risk_event('snapshot_restored', backup)
        else:
            print("鈿狅笍 棰ㄦ帶鍌欎唤鏄槰澶╃殑锛岄噸缃?)
    except FileNotFoundError:
        print("鈿狅笍 鐒￠ⅷ鎺у倷浠斤紝寰為牠闁嬪")
    except Exception as e:
        print("棰ㄦ帶鎭㈠京澶辨晽: {}".format(e))

LEARN_DB   = load_learn_db()
BACKTEST_DB = load_backtest_db()
LEARN_LOCK = threading.Lock()

def _rebuild_live_learning_db(db):
    db = dict(db or {})
    live_trades = [dict(t) for t in (db.get("trades", []) or []) if _is_live_source((t or {}).get("source"))]
    live_closed = [t for t in live_trades if t.get("result") in ("win", "loss")]

    rebuilt = {
        "trades": live_trades,
        "pattern_stats": {},
        "symbol_stats": {},
        "market_state_stats": {},
        "symbol_market_state_stats": {},
        "atr_params": dict((db.get("atr_params") or {"default_sl": 2.0, "default_tp": 3.5})),
        "total_trades": 0,
        "win_rate": 0.0,
        "avg_pnl": 0.0,
        "trend_learning_reset_from": TREND_LEARNING_RESET_FROM,
        "live_only_mode": True,
    }
    rebuilt["atr_params"]["default_sl"] = float(rebuilt["atr_params"].get("default_sl", 2.0) or 2.0)
    rebuilt["atr_params"]["default_tp"] = float(rebuilt["atr_params"].get("default_tp", 3.5) or 3.5)

    for trade in live_closed:
        bd = dict(trade.get("breakdown") or {})
        active_keys = [k for k, v in bd.items() if v != 0]
        pkey = "|".join(sorted(active_keys))
        metric = float(_trade_learn_metric(trade))
        atr_sl = float(trade.get("atr_mult_sl", 2.0) or 2.0)
        atr_tp = float(trade.get("atr_mult_tp", 3.0) or 3.0)

        if pkey not in rebuilt["pattern_stats"]:
            rebuilt["pattern_stats"][pkey] = {
                "win": 0, "loss": 0, "sample_count": 0, "total_pnl": 0.0,
                "avg_pnl": 0.0, "best_sl": atr_sl, "best_tp": atr_tp,
                "tp_candidates": [], "sl_candidates": []
            }
        ps = rebuilt["pattern_stats"][pkey]
        ps["sample_count"] += 1
        ps["total_pnl"] += metric
        ps["avg_pnl"] = round(ps["total_pnl"] / max(ps["sample_count"], 1), 4)
        if trade.get("result") == "win":
            ps["win"] += 1
            ps["tp_candidates"].append(atr_tp)
        else:
            ps["loss"] += 1
            ps["sl_candidates"].append(atr_sl)
        if ps["sample_count"] >= AI_MIN_SAMPLE_EFFECT:
            wr = ps["win"] / max(ps["sample_count"], 1)
            if wr >= 0.6 and ps["tp_candidates"]:
                ps["best_tp"] = round(min(max(ps["tp_candidates"]) * 1.1, 5.0), 2)
                ps["best_sl"] = round(max(ps.get("best_sl", 2.0) * 0.95, 1.8), 2)
            elif wr < 0.4:
                ps["best_sl"] = round(min(ps.get("best_sl", 2.0) * 0.85, 1.8), 2)
                ps["best_tp"] = round(max(ps.get("best_tp", 3.5) * 0.9, 2.8), 2)

        sym = str(trade.get("symbol") or "")
        if sym:
            ss = rebuilt["symbol_stats"].setdefault(sym, {"win": 0, "loss": 0, "count": 0, "total_pnl": 0.0, "total_margin_pct": 0.0})
            ss["count"] += 1
            ss["total_pnl"] += metric
            ss["total_margin_pct"] += float(trade.get("margin_pct", 0) or 0)
            if trade.get("result") == "win":
                ss["win"] += 1
            else:
                ss["loss"] += 1

        bd = dict(trade.get("breakdown") or {})
        market_state = str(bd.get("MarketState") or bd.get("Setup") or bd.get("Regime") or "neutral")
        ms = rebuilt["market_state_stats"].setdefault(market_state, {"count": 0, "win": 0, "loss": 0, "pnl_sum": 0.0})
        ms["count"] += 1
        ms["pnl_sum"] += metric
        if trade.get("result") == "win":
            ms["win"] += 1
        else:
            ms["loss"] += 1
        if sym:
            smk = f"{sym}|{market_state}"
            sms = rebuilt["symbol_market_state_stats"].setdefault(smk, {"count": 0, "win": 0, "loss": 0, "pnl_sum": 0.0})
            sms["count"] += 1
            sms["pnl_sum"] += metric
            if trade.get("result") == "win":
                sms["win"] += 1
            else:
                sms["loss"] += 1

    if live_closed:
        rebuilt["total_trades"] = len(live_closed)
        wins = sum(1 for t in live_closed if t.get("result") == "win")
        rebuilt["win_rate"] = round(wins / len(live_closed) * 100, 1)
        rebuilt["avg_pnl"] = round(sum(_trade_learn_metric(t) for t in live_closed) / len(live_closed), 4)

    return rebuilt

LEARN_DB = _rebuild_live_learning_db(LEARN_DB)
save_learn_db(LEARN_DB)


# =====================================================
# 鍏ㄥ煙鐙€鎱?# =====================================================
STATE = {
    "news_score":        0,
    "latest_news_title": "鏂拌仦绯荤当宸插仠鐢?,
    "news_sentiment":    "宸插仠鐢?,
    "top_signals":       [],
    "active_positions":  [],
    "scan_progress":     "鍟熷嫊涓紝绱勯渶 2 鍒嗛悩瀹屾垚棣栬吉鎺冩弿...",
    "trade_history":     [],
    "total_pnl":         0.0,
    "equity":            0.0,   # 甯虫埗绺借硣鐢紙鍗虫檪锛?    "last_update":       "--",
    "scan_count":        0,
    "halt_reason":       "",     # 棰ㄦ帶鍋滄鍘熷洜
    "risk_status":       {},     # 棰ㄦ帶鐙€鎱嬫憳瑕?    "trailing_info":     {},     # 绉诲嫊姝㈢泩杩借工鐙€鎱嬶紙绲I椤ず锛?    "session_info":      {},
    "market_info":       {"pattern":"鍒濆鍖栦腑","direction":"涓€?,"btc_price":0,"prediction":""},
    "lt_info":           {"position":None,"entry_price":0,"pnl":0,"pattern":"","prediction":""},
    "fvg_orders":        {},
    "threshold_info":    {"current": 60, "phase": "闋愯ō"},  # 鍕曟厠闁€妾昏硣瑷?    "auto_order_audit":  {},
    "protection_state":  {},
    "learn_summary": {
        "total_trades":    0,
        "win_rate":        0.0,
        "avg_pnl":         0.0,
        "current_sl_mult": 2.0,
        "current_tp_mult": 3.0,
        "top_patterns":    [],
        "worst_patterns":  [],
        "blocked_symbols": [],  # 鍕濈巼 < 40% 鐨勫梗锛岃瀵熶腑
    }
}
STATE_LOCK = threading.Lock()
BACKEND_THREAD_LABELS = {
    'position': '鎸佸€夌洠鎺?,
    'enhanced_position': '寮峰寲鎸佸€夌洠鎺?,
    'scan': '甯傚牬鎺冩弿',
    'trailing': '绉诲嫊姝㈢泩',
    'session': '浜ゆ槗鏅傛鐩ｆ帶',
    'market': '澶х洡鍒嗘瀽',
    'fvg_monitor': 'FVG 鎺涘柈鐩ｆ帶',
    'auto_backtest': '鑷嫊鍥炴脯',
    'memory_guard': '瑷樻喍楂斿畧璀?,
    'news': '鏂拌仦鏁寸悊',
}
BACKEND_THREAD_NOTES = {
    'position': '鍚屾鎸佸€夈€佸瑷楄垏淇濊鍠媭鎱?,
    'enhanced_position': '瑁滃挤鎸佸€夐璀夎垏鐣板父淇京',
    'scan': '鎺冩弿鎺掕姒溿€佹暣鐞嗗€欓伕銆佹焙瀹氭槸鍚﹂€佸',
    'trailing': '鎸佺簩鏇存柊淇濇湰鑸囩Щ鍕曟鐩?,
    'session': '鐩ｇ湅浜ゆ槗鏅傛鑸囩郴绲辩瘈濂?,
    'market': '鏇存柊澶х洡鏂瑰悜銆佸挤寮辫垏甯傚牬鍨嬫厠',
    'fvg_monitor': '杩借工 FVG 鎺涘柈鑸囧け鏁堝彇娑?,
    'auto_backtest': '缍 AI 鍊欓伕鍥炴脯鑸囩瓥鐣ユ帓琛?,
    'memory_guard': '鐩ｆ帶瑷樻喍楂旇垏鍩疯绌╁畾鎬?,
    'news': '鏁寸悊鏂拌仦鑸囦簨浠惰儗鏅?,
}

def update_state(**kwargs):
    with STATE_LOCK:
        STATE.update(kwargs)
        return dict(STATE)


def _backend_threads_snapshot():
    try:
        return dict(RUNTIME_STATE.get('backend_threads', {}) or {})
    except Exception:
        return dict(STATE.get('backend_threads', {}) or {})


def _set_backend_thread_state(name, status='running', detail='', error=''):
    thread_name = str(name or 'unknown')
    now_ts = time.time()
    row = dict(_backend_threads_snapshot().get(thread_name, {}) or {})
    if status == 'starting':
        row['starts'] = int(row.get('starts', 0) or 0) + 1
        row['last_started_at'] = tw_now_str()
        row['last_started_ts'] = now_ts
    if status in ('crashed', 'restarting'):
        row['restart_count'] = int(row.get('restart_count', 0) or 0) + (1 if status == 'crashed' else 0)
    row.update({
        'name': thread_name,
        'label': BACKEND_THREAD_LABELS.get(thread_name, thread_name),
        'note': BACKEND_THREAD_NOTES.get(thread_name, ''),
        'status': str(status or 'running'),
        'detail': str(detail or row.get('detail') or ''),
        'updated_at': tw_now_str(),
        'updated_ts': now_ts,
        'last_error': str(error or row.get('last_error') or '')[:240],
    })
    snapshot = _backend_threads_snapshot()
    snapshot[thread_name] = row
    try:
        RUNTIME_STATE.update(backend_threads=snapshot)
    except Exception:
        pass
    try:
        update_state(backend_threads=snapshot)
    except Exception:
        pass
    return snapshot


def _touch_backend_thread(name, detail=''):
    thread_name = str(name or 'unknown')
    snapshot = _backend_threads_snapshot()
    row = dict(snapshot.get(thread_name, {}) or {})
    if not row:
        return _set_backend_thread_state(thread_name, 'running', detail=detail)
    row.update({
        'status': 'running',
        'detail': str(detail or row.get('detail') or ''),
        'updated_at': tw_now_str(),
        'updated_ts': time.time(),
    })
    snapshot[thread_name] = row
    try:
        RUNTIME_STATE.update(backend_threads=snapshot)
    except Exception:
        pass
    try:
        update_state(backend_threads=snapshot)
    except Exception:
        pass
    return snapshot


def sync_openai_trade_state(push_runtime=True):
    with OPENAI_TRADE_LOCK:
        dashboard = build_openai_trade_dashboard(
            OPENAI_TRADE_STATE,
            OPENAI_TRADE_CONFIG,
            api_key_present=bool(OPENAI_API_KEY),
        )
    with AI_LOCK:
        if AI_PANEL.get('short_gainers'):
            dashboard['short_gainers'] = dict(AI_PANEL.get('short_gainers') or {})
        AI_PANEL['openai_trade'] = dashboard
        ai_panel_snapshot = dict(AI_PANEL)
    if push_runtime:
        try:
            update_state(ai_panel=ai_panel_snapshot, auto_backtest=dict(AUTO_BACKTEST_STATE))
            RUNTIME_STATE.update(ai_panel=ai_panel_snapshot)
        except Exception:
            pass
    return dashboard

def smooth_signal_score(symbol, raw_score):
    with CACHE_LOCK:
        prev = SCORE_CACHE.get(symbol, raw_score)
        stable = prev * (1 - SCORE_SMOOTH_ALPHA) + raw_score * SCORE_SMOOTH_ALPHA
        if abs(raw_score) >= 70 and abs(prev) < 45:
            stable = prev * 0.55 + raw_score * 0.45
        SCORE_CACHE[symbol] = stable
    return round(stable, 2)

def score_jump_alert(symbol, raw_score, stable_score):
    prev = SCORE_CACHE.get(symbol, stable_score)
    delta = raw_score - prev
    if abs(delta) >= 25:
        return '鍒嗘暩蹇畩 {:.1f}'.format(delta)
    return ''

def _cooldown_remaining_seconds(symbol):
    now_ts = time.time()
    with CACHE_LOCK:
        entry_ts = float(ENTRY_LOCKS.get(symbol, 0) or 0)
        close_ts = float(POST_CLOSE_LOCKS.get(symbol, 0) or 0)
    remain_entry = max(0.0, ENTRY_LOCK_SEC - (now_ts - entry_ts))
    remain_close = max(0.0, POST_CLOSE_COOLDOWN_SEC - (now_ts - close_ts))
    return max(remain_entry, remain_close)

def can_reenter_symbol(symbol):
    return _cooldown_remaining_seconds(symbol) <= 0

def get_symbol_cooldown_note(symbol):
    remain = _cooldown_remaining_seconds(symbol)
    if remain <= 0:
        return ''
    mins = int(math.ceil(remain / 60.0))
    with CACHE_LOCK:
        close_ts = float(POST_CLOSE_LOCKS.get(symbol, 0) or 0)
    if close_ts > 0 and (time.time() - close_ts) < POST_CLOSE_COOLDOWN_SEC:
        return '鍚屽梗骞冲€夊喎鍗讳腑 {} 鍒嗛悩'.format(mins)
    return '鍚屽梗閫插牬鍐峰嵒涓?{} 鍒嗛悩'.format(mins)

def touch_entry_lock(symbol):
    with CACHE_LOCK:
        ENTRY_LOCKS[symbol] = time.time()

def touch_post_close_lock(symbol):
    with CACHE_LOCK:
        POST_CLOSE_LOCKS[symbol] = time.time()

def fetch_real_atr(symbol, timeframe='15m', limit=60):
    try:
        d = pd.DataFrame(exchange.fetch_ohlcv(symbol, timeframe, limit=limit), columns=['t','o','h','l','c','v'])
        atr = ta.atr(d['h'], d['l'], d['c'], length=14)
        val = safe_last(atr, 0)
        if val > 0:
            return float(val)
    except Exception as e:
        print('fetch_real_atr澶辨晽 {}: {}'.format(symbol, e))
    return 0.0

def verify_protection_orders(symbol, side, sl_price, tp_price):
    side = (side or '').lower()
    try:
        orders = exchange.fetch_open_orders(symbol)
    except Exception as e:
        print('鏌ヨ淇濊鍠け鏁?{}: {}'.format(symbol, e))
        orders = []
    try:
        positions = exchange.fetch_positions([symbol])
    except Exception:
        positions = []
    sl_ok = False
    tp_ok = False
    has_position = any(abs(float((p or {}).get('contracts', 0) or 0)) > 0 for p in (positions or []))
    sl_keys = ['stop', 'stoploss', 'loss', 'sl']
    tp_keys = ['takeprofit', 'profit', 'tp']
    for o in orders:
        text_dump = json.dumps(o, ensure_ascii=False).lower()
        if not sl_ok and any(k in text_dump for k in sl_keys):
            sl_ok = True
        if not tp_ok and any(k in text_dump for k in tp_keys):
            tp_ok = True
    with PROTECTION_LOCK:
        PROTECTION_STATE[symbol] = {
            'sl_ok': sl_ok,
            'tp_ok': tp_ok,
            'has_position': has_position,
            'sl': round(float(sl_price or 0), 8),
            'tp': round(float(tp_price or 0), 8),
            'side': side,
            'updated_at': tw_now_str(),
        }
        snap = snapshot_mapping(PROTECTION_STATE)
    update_state(protection_state=snap)
    return sl_ok, tp_ok

def ensure_exchange_protection(sym, side, pos_side, qty, sl_price, tp_price, verify_wait_sec=1.0):
    """涓嬩富鍠緦绔嬪嵆瑁滄帥浜ゆ槗鎵€淇濊鍠紱鑻ユ鎼嶉璀夊け鏁楋紝鍥炲偝 sl_ok=False銆?""
    sl_side = 'sell' if side == 'buy' else 'buy'
    qty = float(qty or 0)
    sl_ok = False
    tp_ok = False

    if qty <= 0:
        with PROTECTION_LOCK:
            PROTECTION_STATE[sym] = {
                'sl_ok': False, 'tp_ok': False, 'sl': round(float(sl_price or 0), 8),
                'tp': round(float(tp_price or 0), 8), 'side': (side or '').lower(),
                'updated_at': tw_now_str(), 'note': 'qty<=0锛屾湭鎺涗繚璀峰柈'
            }
            snap = snapshot_mapping(PROTECTION_STATE)
        update_state(protection_state=snap)
        return False, False

    # 姝㈡悕鍠紙涓夌ó鏍煎紡渚濆簭鍢楄│锛?    try:
        exchange.create_order(sym, 'market', sl_side, qty, params={
            'reduceOnly':  True,
            'stopPrice':   str(sl_price),
            'orderType':   'stop',
            'posSide':     pos_side,
            'tdMode':      'cross',
        })
        print("姝㈡悕鍠垚鍔?鏍煎紡1): {} @{}".format(sym, sl_price))
        sl_ok = True
    except Exception:
        pass

    if not sl_ok:
        try:
            exchange.create_order(sym, 'market', sl_side, qty, params={
                'reduceOnly':    True,
                'stopLossPrice': str(sl_price),
                'posSide':       pos_side,
                'tdMode':        'cross',
            })
            print("姝㈡悕鍠垚鍔?鏍煎紡2): {} @{}".format(sym, sl_price))
            sl_ok = True
        except Exception:
            pass

    if not sl_ok:
        try:
            exchange.create_order(sym, 'market', sl_side, qty, params={
                'reduceOnly':   True,
                'triggerPrice': str(sl_price),
                'triggerType':  'mark_price',
                'posSide':      pos_side,
            })
            print("姝㈡悕鍠垚鍔?鏍煎紡3): {} @{}".format(sym, sl_price))
            sl_ok = True
        except Exception as e3:
            print("姝㈡悕涓夌ó鏍煎紡閮藉け鏁? {}".format(e3))

    # 姝㈢泩鍠紙鍏╃ó鏍煎紡渚濆簭鍢楄│锛?    try:
        exchange.create_order(sym, 'market', sl_side, qty, params={
            'reduceOnly':  True,
            'stopPrice':   str(tp_price),
            'orderType':   'takeProfit',
            'posSide':     pos_side,
            'tdMode':      'cross',
        })
        print("姝㈢泩鍠垚鍔?鏍煎紡1): {} @{}".format(sym, tp_price))
        tp_ok = True
    except Exception:
        pass

    if not tp_ok:
        try:
            exchange.create_order(sym, 'market', sl_side, qty, params={
                'reduceOnly':      True,
                'takeProfitPrice': str(tp_price),
                'posSide':         pos_side,
                'tdMode':          'cross',
            })
            print("姝㈢泩鍠垚鍔?鏍煎紡2): {} @{}".format(sym, tp_price))
            tp_ok = True
        except Exception as tp_err:
            print("姝㈢泩鎺涘柈澶辨晽锛屼緷璩寸Щ鍕曟鐩堢郴绲? {}".format(tp_err))

    # 浠ヤ氦鏄撴墍闁嬫斁鎺涘柈鍐嶆椹楄瓑锛岄伩鍏?create_order 鎴愬姛浣嗗叾瀵︽矑鎺涗笂
    try:
        time.sleep(max(float(verify_wait_sec), 0.2))
    except Exception:
        pass
    v_sl_ok, v_tp_ok = verify_protection_orders(sym, side, sl_price, tp_price)
    sl_ok = bool(sl_ok or v_sl_ok)
    tp_ok = bool(tp_ok or v_tp_ok)
    with PROTECTION_LOCK:
        PROTECTION_STATE[sym] = {
            'sl_ok': sl_ok,
            'tp_ok': tp_ok,
            'sl': round(float(sl_price or 0), 8),
            'tp': round(float(tp_price or 0), 8),
            'side': (side or '').lower(),
            'updated_at': tw_now_str(),
            'note': '浜ゆ槗鎵€姝㈡悕宸查璀? if sl_ok else '浜ゆ槗鎵€姝㈡悕椹楄瓑澶辨晽',
        }
        snap = snapshot_mapping(PROTECTION_STATE)
    update_state(protection_state=snap)
    return sl_ok, tp_ok


PENDING_LEARN_IDS = set()

def _parse_time_to_ms(s):
    try:
        return int(pd.Timestamp(str(s)).timestamp() * 1000)
    except Exception:
        return None

def resolve_exchange_exit_fill(symbol, entry_side=None, entry_time=None):
    """
    鍢楄│寰炰氦鏄撴墍鏈€杩戞垚浜や腑閭勫師鐪熸骞冲€夊児锛岄伩鍏?TP/SL 鐢变氦鏄撴墍瑙哥櫦鏅傚缈掓矑鏈夎閷勩€?    鍥炲偝: {exit_price, realized_pnl_usdt, fill_side, info}
    """
    result = {
        'exit_price': None,
        'realized_pnl_usdt': None,
        'fill_side': None,
        'info': '',
    }
    try:
        close_side = 'sell' if str(entry_side or '').lower() in ('buy', 'long') else 'buy'
        since_ms = _parse_time_to_ms(entry_time)
        candidates = []

        try:
            trades = exchange.fetch_my_trades(symbol, since=since_ms, limit=30)
        except Exception:
            trades = []

        for tr in trades or []:
            raw = json.dumps(tr, ensure_ascii=False).lower()
            side = str(tr.get('side') or '').lower()
            if side and side != close_side:
                continue
            if 'open' in raw and 'close' not in raw and 'reduce' not in raw:
                continue
            ts = tr.get('timestamp') or 0
            price = tr.get('price')
            if price is None:
                continue
            pnl = tr.get('realizedPnl')
            if pnl is None:
                info = tr.get('info') or {}
                for key in ('realizedPnl', 'achievedProfits', 'profit', 'closeProfit'):
                    if isinstance(info, dict) and info.get(key) is not None:
                        pnl = info.get(key)
                        break
            candidates.append((ts, float(price), float(pnl or 0), side, 'my_trades'))

        try:
            orders = exchange.fetch_closed_orders(symbol, since=since_ms, limit=20)
        except Exception:
            orders = []

        for od in orders or []:
            raw = json.dumps(od, ensure_ascii=False).lower()
            side = str(od.get('side') or '').lower()
            if side and side != close_side:
                continue
            if not any(k in raw for k in ('reduce', 'close', 'stop', 'tp', 'sl', 'profit', 'loss')):
                continue
            price = od.get('average') or od.get('price') or od.get('stopPrice')
            if price is None:
                continue
            ts = od.get('lastTradeTimestamp') or od.get('timestamp') or 0
            candidates.append((ts, float(price), None, side, 'closed_orders'))

        if candidates:
            candidates.sort(key=lambda x: x[0] or 0, reverse=True)
            ts, px, pnl, side, src = candidates[0]
            result.update({
                'exit_price': px,
                'realized_pnl_usdt': pnl,
                'fill_side': side,
                'info': src,
            })
    except Exception as e:
        print('resolve_exchange_exit_fill澶辨晽 {}: {}'.format(symbol, e))
    return result


def queue_learn_for_closed_symbol(sym, active_syms=None):
    """
    瑁滃挤锛氫笉绠℃槸姗熷櫒浜烘墜鍕曞钩鍊夛紝閭勬槸浜ゆ槗鎵€ TP/SL 瑙哥櫦锛屽彧瑕佸€変綅宸叉秷澶卞氨瑁滆瀛哥繏銆?    """
    try:
        if active_syms and sym in active_syms:
            return False

        with LEARN_LOCK:
            open_trade = None
            for t in reversed(LEARN_DB.get('trades', [])):
                if t.get('symbol') == sym and t.get('result') == 'open':
                    open_trade = t
                    break
            if not open_trade:
                return False
            trade_id = open_trade.get('id')
            if trade_id in PENDING_LEARN_IDS:
                return False

        fill = resolve_exchange_exit_fill(sym, open_trade.get('side'), open_trade.get('entry_time'))
        exit_price = fill.get('exit_price')
        realized_pnl_usdt = fill.get('realized_pnl_usdt')

        if exit_price is None:
            try:
                ticker = exchange.fetch_ticker(sym)
                exit_price = float(ticker.get('last') or 0)
            except Exception:
                exit_price = 0

        with LEARN_LOCK:
            for t in LEARN_DB.get('trades', []):
                if t.get('id') == trade_id and t.get('result') == 'open':
                    if exit_price:
                        t['exit_price'] = exit_price
                    if realized_pnl_usdt is not None:
                        t['realized_pnl_usdt'] = realized_pnl_usdt
                    break
            save_learn_db(LEARN_DB)
            PENDING_LEARN_IDS.add(trade_id)
        touch_post_close_lock(sym)

        print('鍋垫脯鍒板钩鍊? {}锛岄枊濮嬪缈掑垎鏋?.. exit_price={} source={} | 鍟熺敤30鍒嗛悩鍐峰嵒'.format(sym, exit_price, fill.get('info') or 'ticker'))
        _enqueue_closed_trade_learning(trade_id)
        return True
    except Exception as e:
        print('queue_learn_for_closed_symbol澶辨晽 {}: {}'.format(sym, e))
        return False



def _resolve_backtest_symbol(symbol=None):
    s = str(symbol or '').strip()
    if s and s.lower() not in ('auto', 'best', 'ai'):
        return s
    try:
        with AI_LOCK:
            rows = list((AUTO_BACKTEST_STATE.get('results') or []))
        if rows:
            return rows[0].get('symbol') or 'BTC/USDT:USDT'
    except Exception:
        pass
    try:
        with STATE_LOCK:
            sigs = list((STATE.get('top_signals') or []))
        if sigs:
            return sigs[0].get('symbol') or 'BTC/USDT:USDT'
    except Exception:
        pass
    return 'BTC/USDT:USDT'


def _normalize_wr_percent(value):
    try:
        x = float(value or 0)
    except Exception:
        return 0.0
    return x * 100.0 if 0 <= x <= 1 else x





def _ai_strategy_profile(symbol, regime='neutral', setup=''):
    """v35: 鐪?AI 鎺ョ + 涓夊堡鍥為€€銆傚洖娓彧鍋氬€欓伕锛屾帴绠″彧鍚冨鍠€?""
    strategy_key = f'{regime}|{setup}|{symbol}'
    setup_mode = _normalize_setup_mode(setup)
    profile = {
        'ready': False,
        'source': 'live_only',
        'sample_count': 0,
        'win_rate': 0.0,
        'avg_pnl': 0.0,
        'ev_per_trade': 0.0,
        'profit_factor': None,
        'max_drawdown_pct': None,
        'threshold_adjust': 0.0,
        'hard_block': False,
        'strategy': strategy_key,
        'strategy_mode': setup_mode,
        'note': 'AI妯ｆ湰涓嶈冻锛屾部鐢ㄤ富绛栫暐锛堝儏鍚冨鍠紝鍥炴脯鍙緵鍊欓伕锛?,
        'confidence': 0.0,
        'status': 'warmup',
        'symbol_blocked': False,
    }

    def _trade_setup_mode(trade):
        try:
            bd = dict(trade.get('breakdown') or {})
            raw = str(
                trade.get('setup_label')
                or bd.get('Setup')
                or trade.get('setup')
                or ''
            )
            return _normalize_setup_mode(raw)
        except Exception:
            return 'main'

    def _compute_stats_from_rows(rows):
        stats = weighted_trade_stats(rows, reset_from=None)
        if not stats:
            return {
                "count": 0, "effective_count": 0.0, "win_rate": 0.0, "avg_pnl": 0.0, "ev_per_trade": 0.0,
                "profit_factor": None, "max_drawdown_pct": None, "std_pnl": None, "weight_sum": 0.0
            }
        return stats

    try:
        bootstrap_rows = get_trend_live_trades(closed_only=True)
        trusted_rows = get_live_trades(closed_only=True, pool='trusted_live')
        soft_rows = get_live_trades(closed_only=True, pool='soft_live')
        quarantine_rows = get_live_trades(closed_only=True, pool='quarantine')
        all_rows = list(bootstrap_rows or trusted_rows or soft_rows or [])
        if not all_rows:
            all_rows = get_live_trades(closed_only=True, pool='all')
        symbol = str(symbol or '')
        regime = str(regime or 'neutral')
        fit_ok, fit_note = _regime_setup_fit(regime, setup)
        sym_block, sym_note = _symbol_hard_block(symbol)
        strat_block, strat_note = _strategy_hard_block(symbol, regime, setup)

        current_session_bucket = session_bucket_from_hour(get_tw_time().hour)
        local_rows = [
            t for t in all_rows
            if str(t.get('symbol') or '') == symbol
            and str((t.get('breakdown') or {}).get('Regime', 'neutral') or 'neutral') == regime
            and _trade_setup_mode(t) == setup_mode
        ]
        local_session_rows = [t for t in local_rows if str(t.get('session_bucket') or '') == current_session_bucket]
        mid_rows = [
            t for t in all_rows
            if str((t.get('breakdown') or {}).get('Regime', 'neutral') or 'neutral') == regime
            and _trade_setup_mode(t) == setup_mode
        ]
        mid_session_rows = [t for t in mid_rows if str(t.get('session_bucket') or '') == current_session_bucket]
        global_rows = [t for t in all_rows if _trade_setup_mode(t) == setup_mode]
        global_session_rows = [t for t in global_rows if str(t.get('session_bucket') or '') == current_session_bucket]
        if regime == 'range':
            global_rows = [t for t in global_rows if str((t.get('breakdown') or {}).get('Regime', 'neutral') or 'neutral') in ('range', 'neutral')]
        elif regime in ('news', 'breakout'):
            global_rows = [t for t in global_rows if str((t.get('breakdown') or {}).get('Regime', 'neutral') or 'neutral') in ('news', 'breakout', 'trend')]
        elif regime == 'trend':
            global_rows = [t for t in global_rows if str((t.get('breakdown') or {}).get('Regime', 'neutral') or 'neutral') in ('trend', 'neutral')]
        if len(global_rows) < 10:
            global_rows = list(all_rows)

        local_stats = _compute_stats_from_rows(local_rows)
        mid_stats = _compute_stats_from_rows(mid_rows)
        global_stats = _compute_stats_from_rows(global_rows)
        symbol_stats = _live_trade_stats(symbol=symbol, regime=None)

        local_cnt = int(local_stats.get('count', 0) or 0)
        mid_cnt = int(mid_stats.get('count', 0) or 0)
        global_cnt = int(global_stats.get('count', 0) or 0)
        trusted_local_cnt = len([t for t in trusted_rows if str(t.get('symbol') or '') == symbol and str((t.get('breakdown') or {}).get('Regime', 'neutral') or 'neutral') == regime and _trade_setup_mode(t) == setup_mode])

        local_session_stats = _compute_stats_from_rows(local_session_rows)
        mid_session_stats = _compute_stats_from_rows(mid_session_rows)
        global_session_stats = _compute_stats_from_rows(global_session_rows)
        if int(local_session_stats.get('count', 0) or 0) >= 5:
            stats = local_session_stats
            fallback_level = 'local_session'
            fallback_desc = '灞€閮ㄦ檪娈?
            fallback_detail = f'{symbol}|{regime}|{setup_mode}|{current_session_bucket}'
        elif local_cnt >= 8:
            stats = local_stats
            fallback_level = 'local'
            fallback_desc = '灞€閮?
            fallback_detail = f'{symbol}|{regime}|{setup_mode}'
        elif int(mid_session_stats.get('count', 0) or 0) >= 8:
            stats = mid_session_stats
            fallback_level = 'mid_session'
            fallback_desc = '涓堡鏅傛'
            fallback_detail = f'{regime}|{setup_mode}|{current_session_bucket}'
        elif mid_cnt >= 12:
            stats = mid_stats
            fallback_level = 'mid'
            fallback_desc = '涓堡'
            fallback_detail = f'{regime}|{setup_mode}'
        elif int(global_session_stats.get('count', 0) or 0) >= 10:
            stats = global_session_stats
            fallback_level = 'global_session'
            fallback_desc = '鍏ㄥ煙鏅傛'
            fallback_detail = f'live_all|{current_session_bucket}'
        else:
            stats = global_stats
            fallback_level = 'global'
            fallback_desc = '鍏ㄥ煙'
            fallback_detail = 'live_all'

        cnt = int(stats.get('count', 0) or 0)
        wr = float(stats.get('win_rate', 0) or 0)
        avg = float(stats.get('avg_pnl', 0) or 0)
        ev = float(stats.get('ev_per_trade', 0) or 0)
        pf = stats.get('profit_factor', None)
        dd = stats.get('max_drawdown_pct', None)
        conf = _ai_confidence_from_live(stats)
        status = _ai_status_from_live(stats)
        effective_count = float(stats.get('effective_count', cnt) or cnt)

        source_weight = {'local_session': 1.08, 'local': 1.0, 'mid_session': 0.82, 'mid': 0.7, 'global_session': 0.52, 'global': 0.4}.get(fallback_level, 0.4)
        conf = round(conf * source_weight, 3)
        suppress = recent_setup_loss_streak(all_rows, symbol=symbol, regime=regime, setup=setup)
        loss_streak = int(suppress.get('loss_streak', 0) or 0)
        if loss_streak >= 3:
            conf = round(conf * float(suppress.get('suppress_mult', 0.5) or 0.5), 3)

        phase = phase_from_counts(global_cnt, local_cnt, effective_count)

        profile.update({
            'sample_count': cnt,
            'win_rate': wr,
            'avg_pnl': avg,
            'ev_per_trade': ev,
            'profit_factor': pf,
            'max_drawdown_pct': dd,
            'confidence': conf,
            'status': status,
            'source': f'live_only:{fallback_level}',
            'effective_count': round(effective_count, 2),
            'loss_streak': loss_streak,
            'phase': phase,
            'trusted_local_count': trusted_local_cnt,
            'local_count': local_cnt,
            'mid_count': mid_cnt,
            'global_count': global_cnt,
            'soft_live_count': len(soft_rows),
            'trusted_live_count': len(trusted_rows),
            'bootstrap_live_count': len(all_rows),
            'strongest_local_count': max(local_cnt, trusted_local_cnt),
            'fallback_level': fallback_level,
            'quarantine_count': len(quarantine_rows),
            'ready': (
                not sym_block and status in ('valid', 'observe') and ((phase == 'full' and conf >= 0.22) or (fallback_level in ('mid', 'global', 'global_session') and effective_count >= 12 and conf >= 0.16))
            ),
            'symbol_blocked': sym_block,
            'strategy_blocked': strat_block,
            'source_weight': source_weight,
        })

        notes = [f'涓夊堡鍥為€€:{fallback_desc}', f'渚濇摎:{fallback_detail}', f'鐣跺墠鏅傛:{current_session_bucket}']
        notes.append(f'灞€閮▄local_cnt}锝滀腑灞mid_cnt}锝滃叏鍩焮global_cnt}')
        notes.append(f'鍙俊灞€閮▄trusted_local_cnt}锝渂ootstrap{len(all_rows)}锝渟oft{len(soft_rows)}锝滈殧闆len(quarantine_rows)}')
        notes.append(f'phase:{phase}')
        notes.append(f'鏈夋晥妯ｆ湰{effective_count:.1f}')

        if sym_block:
            profile['hard_block'] = True
            profile['threshold_adjust'] = 999.0
            notes.append(sym_note or '骞ｇó闀锋湡铏ф悕灏侀帠')
        elif strat_block:
            profile['hard_block'] = True
            profile['threshold_adjust'] = 999.0
            notes.append(strat_note or '绛栫暐闀锋湡鍋忓急灏侀帠')
        elif status == 'reject':
            profile['hard_block'] = False
            profile['threshold_adjust'] = 4.5
            notes.append('AI寮卞嫝绛栫暐锛屽崌楂橀杸妾昏瀵?)
            if pf is not None:
                notes.append(f'PF鍋忓急 {float(pf):.2f}')
            notes.append(f'EV鍋忓急 {ev:+.4f}')
        elif status == 'warmup':
            profile['threshold_adjust'] = -6.0 if fallback_level != 'global' else -2.5
            notes.append('鎺㈢储妯″紡')
            notes.append(f'妯ｆ湰 {cnt}/{TREND_AI_SEMI_TRADES}')
            if fallback_level == 'global':
                notes.append('灞€閮ㄤ笉瓒筹紝鏆€熷叏鍩熺稉椹?)
            elif fallback_level == 'mid':
                notes.append('灞€閮ㄤ笉瓒筹紝鏆€熶腑灞ょ稉椹?)
            else:
                notes.append(f'鍓峽TREND_AI_SEMI_TRADES}鍠劒鍏堢疮绌嶅鍠?)
        elif status == 'observe':
            profile['threshold_adjust'] = -1.5 if fit_ok else 1.0
            if fallback_level == 'global':
                profile['threshold_adjust'] += 1.0
            elif fallback_level == 'mid':
                profile['threshold_adjust'] += 0.5
            notes.append('鍗婃帴绠℃ā寮?)
            notes.append(f'妯ｆ湰 {cnt}/{TREND_AI_FULL_TRADES}')
            if not fit_ok:
                notes.append(fit_note)
        else:
            th_adj = 0.0
            if pf is not None and pf >= 1.35:
                th_adj -= 2.0
            elif pf is not None and pf < 1.08:
                th_adj += 3.0
            if ev >= 0.12:
                th_adj -= 1.5
            elif ev <= 0:
                th_adj += 2.5
            if wr >= 58:
                th_adj -= 1.0
            elif wr < 42:
                th_adj += 1.8
            if dd is not None and dd >= 10:
                th_adj += 2.0
            elif dd is not None and dd <= 3:
                th_adj -= 0.5
            if fallback_level == 'global':
                th_adj += 1.5
                notes.append('鍏ㄥ煙鍥為€€锛屼繚瀹堥亷婵?)
            elif fallback_level == 'mid':
                th_adj += 0.5
                notes.append('涓堡鍥為€€')
            else:
                notes.append('灞€閮ㄦ帴绠?)
            if not fit_ok:
                th_adj += 2.5
                notes.append(fit_note)
            else:
                notes.append(fit_note)
            if setup_mode == 'breakout':
                th_adj += 1.0
                notes.append('鐖嗙櫦绛栫暐淇濆畧閬庢烤')
            elif setup_mode == 'range':
                th_adj += 0.5 if regime != 'range' else -0.5
                notes.append('鍗€闁撶瓥鐣?)
            else:
                notes.append('瓒ㄥ嫝涓荤瓥鐣?)
            scnt = int(symbol_stats.get('count', 0) or 0)
            swr = float(symbol_stats.get('win_rate', 0) or 0)
            if scnt >= SYMBOL_BLOCK_MIN_TRADES and swr < SYMBOL_BLOCK_MIN_WINRATE:
                th_adj += 1.5
                notes.append('骞ｇó瀵﹀柈鍋忓急')
            profile['threshold_adjust'] = round(th_adj, 2)

        if loss_streak >= 3:
            profile['threshold_adjust'] = round(float(profile.get('threshold_adjust', 0) or 0) + 3.0, 2)
            notes.append(f'閫ｈ櫑鎶戝埗 x{float(suppress.get('suppress_mult', 0.5) or 0.5):.2f}')
            notes.append(f'鍚?setup 閫ｈ櫑 {loss_streak} 绛?)

        notes.append(f'EV/绛?{ev:+.4f}')
        if pf is not None:
            notes.append(f'PF {float(pf):.2f}')
        if dd is not None:
            notes.append(f'DD {float(dd):.2f}%')
        notes.append(f'淇″績 {conf*100:.0f}%')
        profile['note'] = '锝?.join(notes)
    except Exception as e:
        profile['note'] = f'AI绛栫暐璁€鍙栧け鏁?{str(e)[:40]}'
    return profile

def ai_decide_trade(sig, eff_threshold, mkt_ok, side_ok, same_dir_cnt, pos_syms, already_closing):
    symbol = str(sig.get('symbol') or '')
    score = abs(float(sig.get('score', 0) or 0))
    rr = float(sig.get('rr_ratio', 0) or 0)
    eq = float(sig.get('entry_quality', 0) or 0)
    bd = dict(sig.get('breakdown') or {})
    regime = str(bd.get('Regime', 'neutral') or 'neutral')
    setup = str(sig.get('setup_label') or bd.get('Setup', '') or '')
    profile = _ai_strategy_profile(symbol, regime=regime, setup=setup)
    market_state, market_state_conf, market_state_note = _classify_market_atlas(regime=regime, setup=setup, breakdown=bd, desc=str(sig.get('desc') or ''))
    market_profile = _market_state_profile(symbol=symbol, regime=regime, setup=setup, market_state=market_state)

    global_live_count = len(_ai_effective_rows(closed_only=True))
    growth_control = _ai_growth_control(global_live_count)
    phase = str(growth_control.get('phase') or 'learning')

    base_threshold = float(eff_threshold)
    fit_ok, fit_note = _regime_setup_fit(regime, setup)
    mode = str(profile.get('strategy_mode') or 'main')
    rotation_adj, rotation_notes = _symbol_rotation_adjustment(symbol)
    eq_adj, eq_note = _entry_quality_feedback(symbol, regime, setup, eq)
    execution_snapshot = _execution_quality_state(sig)

    if AI_FULL_SCORE_CONTROL:
        ai_cov = float(bd.get('AIScoreCoverage', 0) or 0)
        ai_scnt = int(bd.get('AISampleCount', 0) or 0)
        market_threshold_adj = _cap_market_aux(
            float(market_profile.get('boost', 0.0) or 0.0) * 0.18 * float(growth_control.get('market_weight', 0.0) or 0.0),
            MARKET_AUX_THRESHOLD_CAP,
        )
        ai_threshold = float(base_threshold) + float(profile.get('threshold_adjust', 0) or 0) * float(growth_control.get('threshold_weight', 0.0) or 0.0)
        ai_threshold -= market_threshold_adj
        if phase == 'learning':
            ai_threshold = max(ai_threshold, max(52.0, float(ORDER_THRESHOLD_DEFAULT or 52.0)))
        elif phase == 'semi':
            ai_threshold = max(ai_threshold, 50.0)
        elif phase == 'full':
            ai_threshold += max(0.0, 0.72 - ai_cov) * 1.2
        if bool(profile.get('symbol_blocked')) or bool(profile.get('strategy_blocked')):
            ai_threshold += 1.5
        ai_threshold = max(44.0, min(64.0, ai_threshold))

        ai_influence = float(growth_control.get('score_weight', 0.0) or 0.0)
        ai_score_adj = float(profile.get('ev_per_trade', 0) or 0) * 24.0
        ai_score_adj += (float(profile.get('win_rate', 50.0) or 50.0) - 50.0) * 0.06
        ai_score_adj += eq_adj
        ai_score_adj += ai_cov * 6.0
        market_score_adj = _cap_market_aux(float(market_profile.get('boost', 0.0) or 0.0) * (0.9 + market_state_conf * 0.35))
        ai_score_adj += market_score_adj
        ai_score_adj += min(ai_scnt / 18.0, 3.0)
        if not fit_ok:
            ai_score_adj -= 1.25
        if not bool(sig.get('anti_chase_ok', True)):
            ai_score_adj -= 0.85
        ai_score_adj *= ai_influence

        decision_calibrator = calibrate_trade_decision(
            score=score + rotation_adj + ai_score_adj,
            threshold=ai_threshold,
            rr_ratio=max(rr, 0.5),
            entry_quality=max(eq, 0.1),
            regime_confidence=float(sig.get('regime_confidence', 0.0) or 0.0),
            profile=profile,
            execution_quality=execution_snapshot,
            market_consensus=dict(LAST_MARKET_CONSENSUS or {}),
        )
        p_win_est = float(decision_calibrator.get('p_win_est', 0.5) or 0.5)
        ev_est = float(decision_calibrator.get('expected_value_est', 0.0) or 0.0)
        ai_conf_boost = max(0.0, p_win_est - 0.5) * 10.0 + ev_est * 12.0 + ai_cov * 2.0
        ai_conf_boost *= float(growth_control.get('confidence_weight', 0.0) or 0.0)
        effective_score = score + rotation_adj + ai_score_adj + ai_conf_boost
        gating = {
            'regime': bool(mkt_ok and (not (NEUTRAL_REGIME_BLOCK and regime == 'neutral' and phase == 'full'))),
            'setup': True,
            'risk': bool(side_ok and same_dir_cnt < MAX_SAME_DIRECTION and symbol not in already_closing),
            'symbol': bool(symbol not in pos_syms and symbol not in SHORT_TERM_EXCLUDED and can_reenter_symbol(symbol) and sig.get('allowed', True)),
            'trigger': bool(effective_score >= ai_threshold),
            'calibrated_winrate': True if not bool(growth_control.get('allow_ai_gate', False)) else bool(p_win_est >= (0.465 if phase == 'semi' else 0.485)),
            'positive_ev': True if not bool(growth_control.get('allow_ai_gate', False)) else bool(ev_est > (-0.015 if phase == 'semi' else -0.005)),
        }
        base_ok = all(gating.get(k, True) for k in DECISION_PRIORITY_ORDER) and gating.get('calibrated_winrate', True) and gating.get('positive_ev', True)
        ai_ok = True
        reasons = []
        reasons.append({'learning': '鎺㈢储妯″紡', 'semi': '鍗婃帴绠?, 'full': 'AI鐪熸帴绠?}[phase])
        reasons.append(str(growth_control.get('note') or 'AI鎴愰暦淇濊'))
        reasons.append('AI鍏ㄦ帶鍒嗗暉鐢?)
        reasons.append('姹虹瓥鍎厛搴?' + '>'.join(DECISION_PRIORITY_ORDER))
        reasons.append(f'鍏ㄥ煙妯ｆ湰 {global_live_count}')
        reasons.append(f'AI瑕嗚搵鐜?{ai_cov:.2f}')
        reasons.append(f'AI鐗瑰镜妯ｆ湰 {ai_scnt}')
        reasons.append(f'甯傚牬鐙€鎱?{market_state} conf {market_state_conf:.2f}')
        if market_profile.get('note'):
            reasons.append(str(market_profile.get('note')))
        if rotation_notes:
            reasons.extend(rotation_notes)
        if bool(growth_control.get('allow_profile_block', False)) and bool(profile.get('symbol_blocked')):
            ai_ok = False
            reasons.append('骞ｇó闀锋湡铏ф悕灏侀帠')
        if bool(growth_control.get('allow_profile_block', False)) and bool(profile.get('strategy_blocked')):
            ai_ok = False
            reasons.append('绛栫暐闀锋湡铏ф悕灏侀帠')
        if not fit_ok:
            reasons.append('绲愭涓嶅畬缇庝絾鍍呬綔AI杓斿姪鍙冭€?)
            reasons.append(fit_note)
        if not bool(sig.get('anti_chase_ok', True)):
            reasons.append('杩藉児棰ㄩ毆淇濈暀鐐篈I杓斿姪鐗瑰镜')
        if profile.get('ready'):
            reasons.append('AI绛栫暐宸插氨绶?)
        if profile.get('note'):
            reasons.append(str(profile.get('note')))
        if eq_note:
            reasons.append(eq_note + '锛堣紨鍔╋級')
        reasons.append('AI鍒嗘暩瑾挎暣 {:+.2f}'.format(ai_score_adj))
        reasons.append('鏍℃簴鍕濈巼 {:.1f}%'.format(p_win_est * 100.0))
        reasons.append('鏍℃簴EV {:+.3f}'.format(ev_est))
        if not base_ok:
            if effective_score < ai_threshold:
                reasons.append('AI缍滃悎鍒嗘暩鏈亷闁€妾?)
            if not gating.get('calibrated_winrate', True):
                reasons.append('鏍℃簴鍕濈巼涓嶈冻')
            if not gating.get('positive_ev', True):
                reasons.append('鏍℃簴EV涓嶈冻')
        normalized = normalize_decision_summary(
            allow_now=bool(base_ok and ai_ok),
            gating=gating,
            reasons=list(dict.fromkeys(reasons)),
            profile=dict(profile, phase=phase, allow_profile=ai_ok),
            effective_score=effective_score,
            effective_threshold=ai_threshold,
            decision_calibrator=decision_calibrator,
            signal_snapshot={'score': score, 'threshold_raw': base_threshold, 'threshold_calibrated': ai_threshold, 'execution_quality': execution_snapshot, 'market_state': market_state},
        )
        return {
            'allow_now': bool(base_ok and ai_ok),
            'effective_threshold': round(ai_threshold, 2),
            'effective_score': round(effective_score, 2),
            'rotation_adj': round(rotation_adj, 2),
            'reasons': list(dict.fromkeys(reasons)),
            'profile': dict(profile, phase=phase, market_state=market_state, market_profile=market_profile),
            'gating': gating,
            'decision_calibrator': decision_calibrator,
            'decision_explain': merge_decision_explain(gating=gating, calibrator=decision_calibrator, profile=dict(profile, phase=phase), reasons=reasons),
            **normalized,
        }

        base_threshold = float(eff_threshold)
    fit_ok, fit_note = _regime_setup_fit(regime, setup)
    mode = str(profile.get('strategy_mode') or 'main')

    if phase == 'learning':
        ai_threshold = max(52.0, min(64.0, float(base_threshold)))
        rr_floor = max(1.24, MIN_RR_HARD_FLOOR)
        min_entry_quality = 2.25
        if mode == 'breakout' or regime in ('news', 'breakout'):
            ai_threshold = max(ai_threshold, 51.0)
            rr_floor = max(rr_floor, 1.35)
            min_entry_quality = max(min_entry_quality, 2.85)
        elif mode == 'range' and regime == 'range':
            ai_threshold = max(48.0, ai_threshold - 1.0)
            rr_floor = max(1.20, rr_floor)
            min_entry_quality = max(1.8, min_entry_quality - 0.15)
    elif phase == 'semi':
        ai_threshold = max(50.0, min(66.0, base_threshold + float(profile.get('threshold_adjust', 0) or 0) * 0.35))
        rr_floor = max(1.28, MIN_RR_HARD_FLOOR)
        min_entry_quality = 2.25
        if mode == 'breakout' or regime in ('news', 'breakout'):
            ai_threshold = max(ai_threshold, 53.0)
            rr_floor = max(rr_floor, 1.42)
            min_entry_quality = max(min_entry_quality, 2.95)
        elif mode == 'range' and regime == 'range':
            min_entry_quality = min(min_entry_quality, 2.05)
            rr_floor = max(1.24, rr_floor)
    else:
        ai_threshold = max(50.0, min(68.0, base_threshold + float(profile.get('threshold_adjust', 0) or 0)))
        rr_floor = max(1.35, MIN_RR_HARD_FLOOR)
        min_entry_quality = 2.55
        if mode == 'breakout' or regime in ('news', 'breakout'):
            ai_threshold = max(ai_threshold, 54.0)
            rr_floor = max(rr_floor, 1.48)
            min_entry_quality = max(min_entry_quality, 3.0)
        if regime == 'range':
            if mode == 'range':
                min_entry_quality = min(min_entry_quality, 2.1)
                rr_floor = max(1.24, rr_floor)
            else:
                min_entry_quality = max(min_entry_quality, 2.8)

    rotation_adj, rotation_notes = _symbol_rotation_adjustment(symbol)
    eq_adj, eq_note = _entry_quality_feedback(symbol, regime, setup, eq)
    ai_influence = float(growth_control.get('score_weight', 0.0) or 0.0)
    ai_score_adj = float(profile.get('ev_per_trade', 0) or 0) * 20.0
    market_score_adj = _cap_market_aux(float(market_profile.get('boost', 0.0) or 0.0) * (0.8 + market_state_conf * 0.3))
    ai_score_adj += market_score_adj
    ai_score_adj += (float(profile.get('win_rate', 0) or 0) - 50.0) * 0.04
    ai_score_adj += eq_adj
    ai_score_adj *= ai_influence
    execution_snapshot = _execution_quality_state(sig)
    decision_calibrator = calibrate_trade_decision(
        score=score + rotation_adj + ai_score_adj,
        threshold=ai_threshold,
        rr_ratio=rr,
        entry_quality=eq,
        regime_confidence=float(sig.get('regime_confidence', 0.0) or 0.0),
        profile=profile,
        execution_quality=execution_snapshot,
        market_consensus=dict(LAST_MARKET_CONSENSUS or {}),
    )
    effective_score = score + rotation_adj + ai_score_adj + max(0.0, (decision_calibrator.get('p_win_est', 0.5) - 0.5) * 8.0) * float(growth_control.get('confidence_weight', 0.0) or 0.0)

    gating = {
        'regime': bool(mkt_ok and (not (NEUTRAL_REGIME_BLOCK and regime == 'neutral' and phase == 'full'))),
        'setup': bool(eq >= min_entry_quality and rr >= rr_floor and fit_ok),
        'risk': bool(side_ok and same_dir_cnt < MAX_SAME_DIRECTION and symbol not in already_closing),
        'symbol': bool(symbol not in pos_syms and symbol not in SHORT_TERM_EXCLUDED and can_reenter_symbol(symbol) and sig.get('allowed', True)),
        'trigger': bool(effective_score >= ai_threshold),
        'calibrated_winrate': True if not bool(growth_control.get('allow_ai_gate', False)) else bool(float(decision_calibrator.get('p_win_est', 0.0) or 0.0) >= (0.48 if phase != 'full' else 0.52)),
        'positive_ev': True if not bool(growth_control.get('allow_ai_gate', False)) else bool(float(decision_calibrator.get('expected_value_est', -1.0) or -1.0) > 0),
    }
    base_ok = all(gating.get(k, True) for k in DECISION_PRIORITY_ORDER) and gating.get('calibrated_winrate', True) and gating.get('positive_ev', True)

    ai_ok = True
    reasons = []
    reasons.append({'learning': '鎺㈢储妯″紡', 'semi': '鍗婃帴绠?, 'full': 'AI鐪熸帴绠?}[phase])
    reasons.append(str(growth_control.get('note') or 'AI鎴愰暦淇濊'))
    reasons.append('姹虹瓥鍎厛搴?' + '>'.join(DECISION_PRIORITY_ORDER))
    reasons.append('鍥炴脯鍙緵鍊欓伕鎺掑簭')
    reasons.append(f'鍏ㄥ煙妯ｆ湰 {global_live_count}')
    if rotation_notes:
        reasons.extend(rotation_notes)

    if phase == 'learning':
        reasons.append(f'鍓峽TREND_AI_SEMI_TRADES}鍠劒鍏堢疮绌嶅鍠?)
        if profile.get('sample_count') is not None:
            reasons.append('AI妯ｆ湰{}'.format(profile.get('sample_count', 0)))
        if profile.get('note'):
            reasons.append(str(profile.get('note')))
    elif phase == 'semi':
        if bool(profile.get('symbol_blocked')):
            ai_ok = False
            reasons.append('骞ｇó闀锋湡铏ф悕灏侀帠')
        elif bool(profile.get('strategy_blocked')):
            ai_ok = False
            reasons.append('绛栫暐闀锋湡铏ф悕灏侀帠')
        elif int(profile.get('sample_count', 0) or 0) >= TREND_AI_SEMI_TRADES and float(profile.get('avg_pnl', 0) or 0) <= 0 and float(profile.get('win_rate', 0) or 0) < 45:
            ai_ok = False
            reasons.append('鍗婃帴绠″皝閹栬櫑鎼嶇瓥鐣?)
        elif not fit_ok and mode in ('breakout', 'trend'):
            ai_ok = False
            reasons.append(fit_note)
        if profile.get('sample_count') is not None:
            reasons.append('AI妯ｆ湰{}'.format(profile.get('sample_count', 0)))
        if profile.get('note'):
            reasons.append(str(profile.get('note')))
    else:
        hard_block = bool(profile.get('hard_block'))
        if NEUTRAL_REGIME_BLOCK and regime == 'neutral':
            reasons.append('涓€х洡浣庡€変綅鏀捐')
        if not fit_ok:
            reasons.append(fit_note)
        if int(profile.get('sample_count', 0) or 0) < AI_MIN_SAMPLE_EFFECT and score < max(ai_threshold + 8.0, 62.0):
            reasons.append('灞€閮ㄦǎ鏈笉瓒筹紝鍍呴檷娆婁笉纭搵')
        ai_ok = not hard_block
        if profile.get('ready'):
            reasons.append('AI绛栫暐宸插氨绶?)
        else:
            reasons.append('AI鏈畬鍏ㄥ氨绶掞紝缍寔淇濆畧')
        if profile.get('sample_count') is not None:
            reasons.append('AI妯ｆ湰{}'.format(profile.get('sample_count', 0)))
        if profile.get('note'):
            reasons.append(str(profile.get('note')))

    reasons.append(f'甯傚牬鐙€鎱?{market_state} conf {market_state_conf:.2f}')
    if market_profile.get('note'):
        reasons.append(str(market_profile.get('note')))
    if eq_note:
        reasons.append(eq_note)
    reasons.append('AI鍒嗘暩瑾挎暣 {:+.2f}'.format(ai_score_adj))
    reasons.append('鏍℃簴鍕濈巼 {:.1f}%'.format(float(decision_calibrator.get('p_win_est', 0.0) or 0.0) * 100.0))
    reasons.append('鏍℃簴EV {:+.3f}'.format(float(decision_calibrator.get('expected_value_est', 0.0) or 0.0)))

    if not base_ok:
        if effective_score < ai_threshold:
            reasons.append('鍒嗘暩鏈亷AI闁€妾?)
        if eq < min_entry_quality:
            reasons.append('閫插牬鍝佽唱涓嶈冻')
        if rr < rr_floor:
            reasons.append('RR涓嶈冻')
        if not gating.get('calibrated_winrate', True):
            reasons.append('鏍℃簴鍕濈巼涓嶈冻')
        if not gating.get('positive_ev', True):
            reasons.append('鏍℃簴EV涓嶈冻')
    normalized = normalize_decision_summary(
        allow_now=bool(base_ok and ai_ok),
        gating=gating,
        reasons=list(dict.fromkeys(reasons)),
        profile=dict(profile, phase=phase, allow_profile=ai_ok),
        effective_score=effective_score,
        effective_threshold=ai_threshold,
        decision_calibrator=decision_calibrator,
        signal_snapshot={'score': score, 'threshold_raw': base_threshold, 'threshold_calibrated': ai_threshold, 'execution_quality': execution_snapshot, 'market_state': market_state},
    )
    return {
        'allow_now': bool(base_ok and ai_ok),
        'effective_threshold': round(ai_threshold, 2),
        'effective_score': round(effective_score, 2),
        'rotation_adj': round(rotation_adj, 2),
        'reasons': list(dict.fromkeys(reasons)),
        'profile': dict(profile, phase=phase, market_state=market_state, market_profile=market_profile),
        'gating': gating,
        'decision_calibrator': decision_calibrator,
        'decision_explain': merge_decision_explain(gating=gating, calibrator=decision_calibrator, profile=dict(profile, phase=phase), reasons=reasons),
        **normalized,
    }

def build_auto_order_reason(sig, eff_threshold, mkt_ok, side_ok, same_dir_cnt, pos_syms, already_closing, ai_decision=None):
    reasons = []
    if not side_ok:
        reasons.append('鏂瑰悜琛濈獊')
    if sig['symbol'] in pos_syms:
        reasons.append('宸叉湁鎸佸€?)
    if sig['symbol'] in already_closing:
        reasons.append('鍙嶅悜骞冲€変腑')
    if sig['symbol'] in SHORT_TERM_EXCLUDED:
        reasons.append('鐭窔鎺掗櫎鍚嶅柈')
    if not sig.get('allowed', True):
        reasons.append('姝峰彶鍕濈巼灏侀帠')
    if not mkt_ok:
        reasons.append('澶х洡鏂瑰悜涓嶇')
    if same_dir_cnt >= MAX_SAME_DIRECTION:
        reasons.append('鍚屽悜鎸佸€夊凡婊?)
    if not can_reenter_symbol(sig['symbol']):
        reasons.append(get_symbol_cooldown_note(sig['symbol']) or '閫插牬鍐峰嵒涓?)
    if AI_FULL_SCORE_CONTROL:
        reasons.append('鑸奟R/閫插牬鍝佽唱/鍨嬫厠鍏紡宸茶綁鐐篈I杓斿姪鐗瑰镜')
    if ai_decision:
        profile = dict(ai_decision.get('profile') or {})
        reasons.append('AI鏈夋晥鍒嗘暩 {}'.format(ai_decision.get('effective_score')))
        reasons.append('AI闁€妾?{}'.format(ai_decision.get('effective_threshold')))
        if profile.get('sample_count') is not None:
            reasons.append('AI妯ｆ湰 {}'.format(profile.get('sample_count', 0)))
        dc = dict(ai_decision.get('decision_calibrator') or {})
        if dc:
            reasons.append('AI鍕濈巼 {:.1f}%'.format(float(dc.get('p_win_est', 0.0) or 0.0) * 100.0))
            reasons.append('AIEV {:+.3f}'.format(float(dc.get('expected_value_est', 0.0) or 0.0)))
        if profile.get('hard_block'):
            reasons.append('AI灏侀帠姝ょ瓥鐣?)
        note = profile.get('note')
        if note:
            reasons.append(str(note))
    return list(dict.fromkeys(reasons))


def coin_selection_edge(sig):
    try:
        sig = dict(sig or {})
        bd = dict(sig.get('breakdown') or {})
        score = abs(float(sig.get('score', 0) or 0))
        side = 1 if float(sig.get('score', 0) or 0) >= 0 else -1
        rr = float(sig.get('rr_ratio', bd.get('RR', 0)) or 0)
        entry_quality = float(sig.get('entry_quality', bd.get('閫插牬鍝佽唱', bd.get('EntryQuality', 0))) or 0)
        regime = str(sig.get('regime') or bd.get('Regime') or 'neutral')
        setup = str(sig.get('setup_label') or bd.get('Setup') or '')
        regime_conf = float(sig.get('regime_confidence', bd.get('RegimeConf', bd.get('RegimeConfidence', 0))) or 0)
        trend_conf = float(sig.get('trend_confidence', bd.get('TrendConfidence', bd.get('鏂瑰悜淇″績', 0))) or 0)
        regime_bias = float(sig.get('regime_bias', bd.get('RegimeBias', 0)) or 0)
        chase = float(bd.get('杩藉児棰ㄩ毆', bd.get('ChaseRisk', 0)) or 0)
        vol_ratio = float(bd.get('VolRatio', bd.get('volume_ratio', 1.0)) or 1.0)
        marketability = dict(sig.get('marketability') or {})
        marketability_score = float(marketability.get('score', sig.get('marketability_score', 0.0)) or 0.0)
        ai_cov = float(bd.get('AIScoreCoverage', 0) or 0)
        ai_samples = int(bd.get('AISampleCount', 0) or 0)

        edge = 0.0
        notes = []

        edge += max(0.0, marketability_score - 2.2) * 0.45
        if marketability_score < 2.2:
            edge -= 2.0
            notes.append('marketability_weak')
        elif marketability_score >= 5.0:
            notes.append('liquid_active_market')

        if regime == 'trend':
            edge += 2.2 + min(regime_conf, 1.0) * 1.4
            notes.append('trend_regime')
        elif regime == 'news':
            edge -= 1.6
            notes.append('news_volatility_penalty')
        elif regime in ('range', 'neutral_range'):
            edge -= 0.9
            notes.append('range_market_penalty')
        elif regime == 'neutral':
            edge -= 0.45

        if (side > 0 and regime_bias > 0) or (side < 0 and regime_bias < 0):
            edge += min(abs(regime_bias) * 0.35, 1.5)
            notes.append('direction_aligned')
        elif abs(regime_bias) >= 2:
            edge -= 1.2
            notes.append('direction_conflict')

        if trend_conf >= 7:
            edge += 1.5
            notes.append('high_trend_confidence')
        elif trend_conf >= 5:
            edge += 0.6
        elif score >= 45:
            edge -= 0.8
            notes.append('low_trend_confidence')

        if rr >= 2.0:
            edge += 1.1
            notes.append('good_rr')
        elif rr >= 1.45:
            edge += 0.45
        elif rr > 0:
            edge -= 1.2
            notes.append('rr_too_thin')

        if entry_quality >= 7:
            edge += 1.0
            notes.append('clean_entry')
        elif entry_quality >= 5:
            edge += 0.35
        elif score >= 45:
            edge -= 0.8
            notes.append('entry_quality_weak')

        if 1.05 <= vol_ratio <= 3.2:
            edge += 0.7
            notes.append('healthy_volume_expansion')
        elif vol_ratio > 4.2:
            edge -= 1.0
            notes.append('volume_climax_risk')
        elif vol_ratio < 0.75:
            edge -= 0.55

        if chase >= 6:
            edge -= 2.2
            notes.append('anti_chase_penalty')
        elif chase >= 4:
            edge -= 0.9

        if bool(bd.get('PreBreakoutScore', 0)) and float(bd.get('PreBreakoutScore', 0) or 0) >= 55:
            edge += 0.9
            notes.append('pre_breakout_ready')

        if ai_cov >= 0.35:
            edge += min(ai_cov * 2.0, 1.3)
            notes.append('ai_coverage')
        if ai_samples >= 12:
            edge += min(ai_samples / 20.0, 1.0)
            notes.append('ai_sample_support')

        try:
            strat = _strategy_score_lookup(sig.get('symbol', ''), regime, setup)
            strat_count = int(strat.get('count', strat.get('trades', 0)) or 0)
            strat_ev = float(strat.get('ev_per_trade', 0) or 0)
            strat_wr = float(strat.get('win_rate', 0) or 0)
            if strat_count >= STRATEGY_CAPITAL_MIN_TRADES and strat_ev > 0 and strat_wr >= 52:
                edge += min(2.0, strat_ev * 18.0 + (strat_wr - 50.0) * 0.05)
                notes.append('learned_profitable_symbol')
            elif strat_count >= STRATEGY_BLOCK_MIN_TRADES and (strat_ev < 0 or strat_wr < 45):
                edge -= 1.6
                notes.append('learned_weak_symbol')
        except Exception:
            pass

        return round(max(-5.0, min(edge, 8.0)), 3), list(dict.fromkeys(notes))[:8]
    except Exception:
        return 0.0, []


def safe_last(series, default=0):
    try:
        v = series.iloc[-1]
        return float(v) if v == v else default
    except:
        return default

# =====================================================
# 闋愬垽鏆存媺 / 鏆磋穼鍓嶇殑钃勫嫝绲愭 + 杩藉児棰ㄩ毆鍋垫脯
# =====================================================
def _linreg_slope(values):
    try:
        arr = np.array(list(values), dtype=float)
        if len(arr) < 3:
            return 0.0
        x = np.arange(len(arr), dtype=float)
        slope = np.polyfit(x, arr, 1)[0]
        return float(slope)
    except:
        return 0.0

def analyze_pre_breakout_setup(d15, d4h):
    """
    鎵俱€岄倓娌掑櫞銆佷絾宸茬稉鍦ㄨ搫鍕€嶇殑绲愭锛?    - 娉㈠嫊鏀舵杺锛圔B瀵害/ATR绺皬锛?    - 闈犺繎鍗€闁撻珮/浣庨粸
    - 楂樹綆榛為€愭鎶珮/澹撲綆
    - 4H 涓昏定鍕㈠悓鍚?    """
    try:
        if len(d15) < 60 or len(d4h) < 30:
            return 0, "钃勫嫝鏁告摎涓嶈冻"

        c = d15['c'].astype(float)
        h = d15['h'].astype(float)
        l = d15['l'].astype(float)
        v = d15['v'].astype(float)
        curr = float(c.iloc[-1])
        atr = max(safe_last(ta.atr(h, l, c, length=14), curr * 0.004), curr * 0.003)

        bb = ta.bbands(c, length=20, std=2)
        if bb is None or bb.empty:
            return 0, "钃勫嫝鐒B"
        bb_up = safe_last(bb.iloc[:, 0], curr)
        bb_mid = safe_last(bb.iloc[:, 1], curr)
        bb_low = safe_last(bb.iloc[:, 2], curr)
        bb_width_now = max((bb_up - bb_low) / max(bb_mid, 1e-9), 0)
        width_hist = ((bb.iloc[:, 0] - bb.iloc[:, 2]) / bb.iloc[:, 1].replace(0, np.nan)).dropna().tail(40)
        width_med = float(width_hist.median()) if len(width_hist) else bb_width_now

        atr_series = ta.atr(h, l, c, length=14)
        atr_now = safe_last(atr_series, atr)
        atr_recent = float(pd.Series(atr_series).tail(8).mean()) if atr_series is not None else atr_now
        atr_prev = float(pd.Series(atr_series).tail(32).head(16).mean()) if atr_series is not None else atr_now
        atr_prev = atr_prev if atr_prev and atr_prev == atr_prev else atr_now

        range_high = float(h.tail(BREAKOUT_LOOKBACK).iloc[:-1].max())
        range_low = float(l.tail(BREAKOUT_LOOKBACK).iloc[:-1].min())
        near_high = (range_high - curr) / max(atr_now, 1e-9) <= 0.45 and curr <= range_high * 1.003
        near_low = (curr - range_low) / max(atr_now, 1e-9) <= 0.45 and curr >= range_low * 0.997

        lows_slope = _linreg_slope(l.tail(6).tolist())
        highs_slope = _linreg_slope(h.tail(6).tolist())

        vol_recent = float(v.tail(4).mean())
        vol_prev = float(v.tail(24).head(12).mean()) if len(v) >= 24 else vol_recent
        vol_expand = vol_recent > vol_prev * 1.08 if vol_prev > 0 else False

        ema21_4h = safe_last(ta.ema(d4h['c'], length=21), curr)
        ema55_4h = safe_last(ta.ema(d4h['c'], length=55), curr)
        trend_up = curr > ema21_4h > ema55_4h
        trend_dn = curr < ema21_4h < ema55_4h

        squeeze = bb_width_now < width_med * 0.88 and atr_recent < atr_prev * 0.9
        score = 0
        tags = []

        if squeeze and near_high and lows_slope > 0 and trend_up:
            score += 6
            tags.append("鏀舵杺閫艰繎鍓嶉珮")
            if vol_expand:
                score += 2
                tags.append("閲忚兘鎮勬倓鏀惧ぇ")
        elif squeeze and near_low and highs_slope < 0 and trend_dn:
            score -= 6
            tags.append("鏀舵杺閫艰繎鍓嶄綆")
            if vol_expand:
                score -= 2
                tags.append("閲忚兘鎮勬倓鏀惧ぇ")

        # 鍋囩獊鐮村墠鐨勫惛鏀讹細鍍规牸寰堟帴杩戝墠楂?鍓嶄綆锛屼絾灏氭湭澶у箙绌胯秺
        last_body = abs(float(c.iloc[-1]) - float(d15['o'].iloc[-1]))
        if near_high and trend_up and last_body < atr_now * 0.75 and curr <= range_high * 1.0015:
            score += 1
            tags.append("涓婃部鍚告敹涓?)
        elif near_low and trend_dn and last_body < atr_now * 0.75 and curr >= range_low * 0.9985:
            score -= 1
            tags.append("涓嬫部鍚告敹涓?)

        score = max(min(score, 8), -8)
        return score, "|".join(tags) if tags else "鐒℃槑椤搫鍕?
    except Exception:
        return 0, "钃勫嫝鍒嗘瀽澶辨晽"

def analyze_extension_risk(d15, direction_hint=0):
    """閬垮厤杩芥疾娈鸿穼锛氬凡闆㈠潎绶氬お閬?+ 閫ｇ簩鍠倞琛濆埡鏅傜洿鎺ラ檷娆娿€?""
    try:
        c = d15['c'].astype(float)
        o = d15['o'].astype(float)
        h = d15['h'].astype(float)
        l = d15['l'].astype(float)
        curr = float(c.iloc[-1])
        ema20 = safe_last(ta.ema(c, length=20), curr)
        atr = max(safe_last(ta.atr(h, l, c, length=14), curr * 0.004), curr * 0.003)
        bb = ta.bbands(c, length=20, std=2)
        bb_up = safe_last(bb.iloc[:, 0], curr) if bb is not None and not bb.empty else curr
        bb_low = safe_last(bb.iloc[:, 2], curr) if bb is not None and not bb.empty else curr
        ext = (curr - ema20) / max(atr, 1e-9)
        bull3 = all(c.iloc[-i] > o.iloc[-i] for i in [1,2,3])
        bear3 = all(c.iloc[-i] < o.iloc[-i] for i in [1,2,3])

        if direction_hint >= 0 and ext > ANTI_CHASE_ATR and curr >= bb_up * 0.995 and bull3:
            penalty = -10 if ext > 1.9 else -7
            return penalty, "澶氶牠閬庡害寤朵几锛岄伩鍏嶈拷楂?
        if direction_hint <= 0 and ext < -ANTI_CHASE_ATR and curr <= bb_low * 1.005 and bear3:
            penalty = 10 if ext < -1.9 else 7
            return penalty, "绌洪牠閬庡害寤朵几锛岄伩鍏嶈拷绌?
        return 0, "寤朵几姝ｅ父"
    except Exception:
        return 0, "寤朵几鍒嗘瀽澶辨晽"

def get_breakout_pullback_entry(symbol, side, current_price, atr):
    """
    杩藉児淇濊锛氬凡缍撻洟鍧囩窔/鍗€闁撳お閬犳檪锛屼笉鐩存帴甯傚児锛屾敼绛夊洖韪?鍙嶅綀銆?    """
    try:
        df = pd.DataFrame(exchange.fetch_ohlcv(symbol, '15m', limit=80), columns=['t','o','h','l','c','v'])
        if df.empty or len(df) < 30:
            return None, "pullback璩囨枡涓嶈冻"
        c = df['c'].astype(float)
        o = df['o'].astype(float)
        h = df['h'].astype(float)
        l = df['l'].astype(float)
        curr = float(current_price or c.iloc[-1])
        atr = max(float(atr or 0), curr * 0.003)
        ema20 = safe_last(ta.ema(c, length=20), curr)
        hh = float(h.tail(BREAKOUT_LOOKBACK).iloc[:-1].max())
        ll = float(l.tail(BREAKOUT_LOOKBACK).iloc[:-1].min())
        last_body = abs(float(c.iloc[-1]) - float(o.iloc[-1]))
        ext = (curr - ema20) / max(atr, 1e-9)

        if side == 'long':
            breakout_now = curr >= hh * 0.999 and last_body > atr * 0.7
            if ext > ANTI_CHASE_ATR or breakout_now:
                limit_price = max(ema20, hh - atr * PULLBACK_BUFFER_ATR)
                if limit_price < curr * 0.999:
                    return round(limit_price, 6), "杩介珮淇濊锛氭敼绛夊洖韪╁啀澶?
        else:
            breakout_now = curr <= ll * 1.001 and last_body > atr * 0.7
            if ext < -ANTI_CHASE_ATR or breakout_now:
                limit_price = min(ema20, ll + atr * PULLBACK_BUFFER_ATR)
                if limit_price > curr * 1.001:
                    return round(limit_price, 6), "杩界┖淇濊锛氭敼绛夊弽褰堝啀绌?
        return None, "寤朵几姝ｅ父"
    except Exception:
        return None, "pullback瑷堢畻澶辨晽"

# =====================================================
# 绲辫▓婵剧恫锛氳┎骞ｅ嫕鐜囨槸鍚﹂仈妯?# =====================================================
def is_symbol_allowed(symbol):
    """鑻ヨ┎骞ｅ鍠凡瓒呴亷10绛嗕笖鍕濈巼 <40%锛屽皝閹栦笅鍠紝鏀圭偤瑙€瀵?""
    with LEARN_LOCK:
        st = LEARN_DB.get("symbol_stats", {}).get(symbol, {})
    n = int(st.get("count", 0) or 0)
    if n < SYMBOL_BLOCK_MIN_TRADES:
        return True, n, 0.0   # 妯ｆ湰涓嶈冻锛屽厑瑷?    wr = float(st.get("win", 0) or 0) / max(n, 1) * 100.0
    return wr >= SYMBOL_BLOCK_MIN_WINRATE, n, round(wr, 1)

# =====================================================
# ADX锛氳定鍕㈠挤搴?# =====================================================
def analyze_adx(df):
    try:
        adx_df = ta.adx(df['h'], df['l'], df['c'], length=14)
        if adx_df is None or adx_df.empty:
            return 0, "ADX鐒℃暩鎿?
        adx_val = safe_last(adx_df['ADX_14'], 0)
        dmp     = safe_last(adx_df['DMP_14'], 0)
        dmn     = safe_last(adx_df['DMN_14'], 0)
        score = 0; tag = "ADX{:.0f}".format(adx_val)
        # 娉ㄦ剰锛氭鍑芥暩鐢?analyze() 鍛煎彨锛岄€忛亷 symbol 鍒ゆ柗涓绘祦/灞卞
        if adx_val > 30:
            score = W["adx"] if dmp > dmn else -W["adx"]
            tag  += "(寮峰)" if dmp > dmn else "(寮风┖)"
        elif adx_val > 20:
            score = W["adx"]//2 if dmp > dmn else -W["adx"]//2
            tag  += "(寮卞)" if dmp > dmn else "(寮辩┖)"
        else:
            tag += "(鐩ゆ暣)"
        return score, tag
    except:
        return 0, "ADX澶辨晽"

# =====================================================
# VWAP锛氱浉灏嶄綅缃?# =====================================================
def analyze_vwap(df):
    try:
        # 鎵嬪嫊瑷堢畻 VWAP锛屽畬鍏ㄤ笉闇€瑕?DatetimeIndex
        tp = (df['h'] + df['l'] + df['c']) / 3
        vwap_val = float((tp * df['v']).sum() / df['v'].sum())
        curr = float(df['c'].iloc[-1])
        if vwap_val <= 0:
            return 0, "VWAP鐒℃暩鎿?
        dist_pct = (curr - vwap_val) / vwap_val * 100
        if dist_pct > 1.0:
            return W["vwap"], "VWAP涓婃柟{:.1f}%".format(dist_pct)
        elif dist_pct < -1.0:
            return -W["vwap"], "VWAP涓嬫柟{:.1f}%".format(abs(dist_pct))
        elif dist_pct > 0.2:
            return W["vwap"]//2, "鎺ヨ繎VWAP涓婃柟"
        elif dist_pct < -0.2:
            return -W["vwap"]//2, "鎺ヨ繎VWAP涓嬫柟"
        else:
            return 0, "VWAP涓€?
    except:
        return 0, "VWAP澶辨晽"

# =====================================================
# Order Block锛氭妲嬪崁鍩熷伒娓?# =====================================================
# =====================================================
# ICT 姒傚康锛欱OS / CHoCH / 绺噺鍥炶
# =====================================================
# =====================================================
# FVG (Fair Value Gap / 鍚堢悊鍍规牸缂哄彛)
# =====================================================
def analyze_fvg(df):
    """
    FVG 鏄?SMC/ICT 鏍稿績姒傚康锛?    - 涓夋牴K妫掞紝涓枔閭ｆ牴鐨勪笂涓嬪奖绶氱暀涓嬬己鍙?    - 鍋氬FVG锛欿1鏈€楂?< K3鏈€浣庯紙鍚戜笂璺崇┖缂哄彛锛?    - 鍋氱┖FVG锛欿1鏈€浣?> K3鏈€楂橈紙鍚戜笅璺崇┖缂哄彛锛?    - 鍍规牸鍥炲埌 FVG 鍗€鍩?= 楂樻鐜囧弽杞夐粸
    - 閰嶅悎 OB 浣跨敤鏅備俊铏熸洿寮凤紙SMC鍏ュ牬鏍稿績锛?    """
    try:
        hi = df['h'].tolist()
        lo = df['l'].tolist()
        cl = df['c'].tolist()
        n  = len(cl)
        curr = cl[-1]
        score = 0; tags = []

        # 鎺冩弿鏈€杩?0鏍筀妫掓壘鏈～瑁滅殑FVG
        bullish_fvgs = []  # 鍋氬缂哄彛锛堟敮鎾愶級
        bearish_fvgs = []  # 鍋氱┖缂哄彛锛堝鍔涳級

        for i in range(2, min(30, n)):
            idx = n - 1 - i  # 寰炴渶杩戝線鍓嶆巸
            if idx < 2:
                break

            k1_h = hi[idx-2]; k1_l = lo[idx-2]
            k2_h = hi[idx-1]; k2_l = lo[idx-1]
            k3_h = hi[idx];   k3_l = lo[idx]

            # 鍋氬FVG锛欿1鏈€楂?< K3鏈€浣庯紙涓婂崌缂哄彛锛?            if k1_h < k3_l:
                gap_top    = k3_l
                gap_bottom = k1_h
                gap_size   = gap_top - gap_bottom

                # 妾㈡煡鏄惁宸茶濉
                filled = any(lo[j] <= gap_top and hi[j] >= gap_bottom
                             for j in range(idx+1, n))
                if not filled and gap_size > 0:
                    bullish_fvgs.append({
                        "top": gap_top,
                        "bottom": gap_bottom,
                        "size": gap_size,
                        "age": i  # 骞炬牴K妫掑墠
                    })

            # 鍋氱┖FVG锛欿1鏈€浣?> K3鏈€楂橈紙涓嬮檷缂哄彛锛?            elif k1_l > k3_h:
                gap_top    = k1_l
                gap_bottom = k3_h
                gap_size   = gap_top - gap_bottom

                filled = any(hi[j] >= gap_bottom and lo[j] <= gap_top
                             for j in range(idx+1, n))
                if not filled and gap_size > 0:
                    bearish_fvgs.append({
                        "top": gap_top,
                        "bottom": gap_bottom,
                        "size": gap_size,
                        "age": i
                    })

        # 鍒ゆ柗鐣跺墠鍍规牸鏄惁鍦?FVG 鍗€鍩熷収
        W_FVG = W.get("bos_choch", 7)  # 鍏辩敤 ICT 椤炲垎鏁?
        for fvg in bullish_fvgs[:3]:  # 鍙湅鏈€杩?鍊?            if fvg["bottom"] <= curr <= fvg["top"]:
                # 鍍规牸鍥炲埌鍋氬FVG鍗€鍩?鈫?鐪嬪
                freshness = max(1.0 - fvg["age"] / 30, 0.3)  # 瓒婃柊瓒婇噸瑕?                pts = round(W_FVG * freshness)
                score += pts
                tags.append("FVG鍋氬缂哄彛({:.4f}-{:.4f})".format(
                    fvg["bottom"], fvg["top"]))
                break

        for fvg in bearish_fvgs[:3]:
            if fvg["bottom"] <= curr <= fvg["top"]:
                # 鍍规牸鍥炲埌鍋氱┖FVG鍗€鍩?鈫?鐪嬬┖
                freshness = max(1.0 - fvg["age"] / 30, 0.3)
                pts = round(W_FVG * freshness)
                score -= pts
                tags.append("FVG鍋氱┖缂哄彛({:.4f}-{:.4f})".format(
                    fvg["bottom"], fvg["top"]))
                break

        # 鍍规牸鎺ヨ繎浣嗛倓娌掑埌FVG锛堥爯鏈熷洖鎾わ級
        if not tags:
            for fvg in bullish_fvgs[:2]:
                dist = (curr - fvg["top"]) / max(curr, 1e-9)
                if 0 < dist < 0.02:  # 璺濋洟缂哄彛闋傞儴2%浠ュ収
                    score += W_FVG // 2
                    tags.append("鎺ヨ繎FVG鏀拹缂哄彛")
                    break
            for fvg in bearish_fvgs[:2]:
                dist = (fvg["bottom"] - curr) / max(curr, 1e-9)
                if 0 < dist < 0.02:
                    score -= W_FVG // 2
                    tags.append("鎺ヨ繎FVG澹撳姏缂哄彛")
                    break

        return min(max(score, -W_FVG), W_FVG), "|".join(tags) or "鐒VG"
    except Exception as e:
        return 0, "FVG澶辨晽"

def analyze_ict(df4h, df15):
    """
    BOS (Break of Structure)锛氱獊鐮村墠楂?鍓嶄綆锛岀⒑瑾嶈定鍕㈡柟鍚?    CHoCH (Change of Character)锛氳定鍕㈣綁鎻涜▕铏燂紝鏈€閲嶈鐨勫弽杞変俊铏?    绺噺鍥炶锛氳定鍕腑鍥炶鏅傛垚浜ら噺绺皬锛屼唬琛ㄥ彧鏄洖瑾块潪鍙嶈綁
    """
    try:
        score = 0; tags = []

        c4 = df4h['c'].tolist()
        h4 = df4h['h'].tolist()
        l4 = df4h['l'].tolist()
        v4 = df4h['v'].tolist()
        n = len(c4)
        curr = c4[-1]

        if n < 20:
            return 0, "ICT鏁告摎涓嶈冻"

        # 鎵炬渶杩戠殑鎿哄嫊楂樹綆榛烇紙Swing High/Low锛?        def find_swings(highs, lows, lookback=5):
            swing_highs = []
            swing_lows = []
            for i in range(lookback, len(highs)-lookback):
                if highs[i] == max(highs[i-lookback:i+lookback+1]):
                    swing_highs.append((i, highs[i]))
                if lows[i] == min(lows[i-lookback:i+lookback+1]):
                    swing_lows.append((i, lows[i]))
            return swing_highs, swing_lows

        swing_highs, swing_lows = find_swings(h4, l4, lookback=3)

        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # 鏈€杩戝叐鍊嬫摵鍕曢珮榛?            sh1 = swing_highs[-2][1]  # 鍓嶅墠楂?            sh2 = swing_highs[-1][1]  # 鍓嶉珮
            # 鏈€杩戝叐鍊嬫摵鍕曚綆榛?            sl1 = swing_lows[-2][1]   # 鍓嶅墠浣?            sl2 = swing_lows[-1][1]   # 鍓嶄綆

            # BOS 鍋氬锛氱獊鐮村墠楂?鈫?涓婂崌瓒ㄥ嫝纰鸿獚
            if curr > sh2 * 1.001:
                score += 8
                tags.append("BOS绐佺牬鍓嶉珮{:.4f}".format(sh2))

            # BOS 鍋氱┖锛氳穼鐮村墠浣?鈫?涓嬮檷瓒ㄥ嫝纰鸿獚
            elif curr < sl2 * 0.999:
                score -= 8
                tags.append("BOS璺岀牬鍓嶄綆{:.4f}".format(sl2))

            # CHoCH 鍋氬锛氬師鏈笅闄嶈定鍕紙鍓嶉珮浣庢柤鏇村墠楂橈級锛屼絾鐝惧湪绐佺牬鍓嶉珮
            if sh2 < sh1 and curr > sh2 * 1.001:
                score += 6  # 椤嶅鍔犲垎锛岃定鍕㈣綁鎻?                tags.append("CHoCH瓒ㄥ嫝杞夊")

            # CHoCH 鍋氱┖锛氬師鏈笂鍗囪定鍕紙鍓嶄綆楂樻柤鏇村墠浣庯級锛屼絾鐝惧湪璺岀牬鍓嶄綆
            elif sl2 > sl1 and curr < sl2 * 0.999:
                score -= 6
                tags.append("CHoCH瓒ㄥ嫝杞夌┖")

        # 绺噺鍥炶鍋垫脯锛堝仛澶氱⒑瑾嶏級
        # 姊濅欢锛氭渶杩?鏍筀妫掑児鏍间笅璺屼絾鎴愪氦閲忕府灏?        if len(c4) >= 6 and len(v4) >= 6:
            recent_prices = c4[-4:-1]
            recent_vols = v4[-4:-1]
            avg_vol = sum(v4[-20:-4]) / max(len(v4[-20:-4]), 1)

            is_pullback = recent_prices[-1] < recent_prices[0]  # 杩戞湡鍥炶
            is_low_vol = sum(recent_vols) / 3 < avg_vol * 0.7   # 鎴愪氦閲忕府灏?0%

            if is_pullback and is_low_vol:
                # 澶ц定鍕㈡槸澶氶牠鎵嶅姞鍒?                if c4[-1] > sum(c4[-20:]) / 20:  # 鍍规牸鍦?0鏍瑰潎绶氫笂
                    score += 5
                    tags.append("绺噺鍋ュ悍鍥炶")
                else:
                    score -= 3
                    tags.append("绺噺寮卞嫝涓嬭穼")

        # 鐢?5鍒嗛悩纰鸿獚 BOS
        c15 = df15['c'].tolist()
        h15 = df15['h'].tolist()
        l15 = df15['l'].tolist()
        if len(h15) >= 20:
            sh15, sl15 = find_swings(h15, l15, lookback=3)
            if sh15 and sl15:
                last_sh15 = sh15[-1][1] if sh15 else 0
                last_sl15 = sl15[-1][1] if sl15 else float('inf')
                # 15鍒嗛悩涔熺獊鐮?鈫?澶氶€辨湡鍏辨尟
                if c15[-1] > last_sh15 * 1.001 and score > 0:
                    score += 3
                    tags.append("15m澶氶€辨湡BOS鍏辨尟")
                elif c15[-1] < last_sl15 * 0.999 and score < 0:
                    score -= 3
                    tags.append("15m澶氶€辨湡BOS鍏辨尟")

        return min(max(score, -W.get('bos_choch',7)), W.get('bos_choch',7)), "|".join(tags) or "鐒CT瑷婅櫉"
    except Exception as e:
        return 0, "ICT鍒嗘瀽澶辨晽"

def analyze_order_block(df4h, is_major=False):
    """
    鍋垫脯姗熸 Order Block锛?    - 寮峰姏鍠倞閬嬪嫊鍓嶇殑鏈€寰屼竴鏍瑰弽鍚慘妫掑嵆鐐?OB
    - 鍍规牸鍥炲埌 OB 鍗€鍩?鈫?楂樻鐜囧弽褰堥粸
    """
    try:
        score = 0; tags = []
        closes = df4h['c'].tolist()
        opens  = df4h['o'].tolist()
        highs  = df4h['h'].tolist()
        lows   = df4h['l'].tolist()
        curr   = closes[-1]

        # 鎵炬渶杩戠殑鐪嬪 OB锛堣穼寰屾€ユ疾鍓嶇殑鏈€寰屼竴鏍归櫚绶氾級
        for i in range(len(closes)-3, max(len(closes)-20, 2), -1):
            # 纰鸿獚寰岄潰鏈夊挤鍔涗笂婕?            move_up = (closes[i+1] - opens[i+1]) / max(abs(opens[i+1]), 1e-9)
            if closes[i] < opens[i] and move_up > (0.010 if is_major else 0.015):  # OB姊濅欢锛堜富娴?%/灞卞1.5%锛?                ob_high = opens[i]  # OB 鍗€鍩燂細闄扮窔鐨勯枊鐩ゅ埌鏈€楂?                ob_low  = lows[i]
                # 鐣跺墠鍍规牸鏄惁鍦?OB 鍗€鍩熷収锛堝洖娓級
                if ob_low <= curr <= ob_high * 1.01:
                    score += W["order_block"]
                    tags.append("鐪嬪OB鍗€鍩?{:.4f}-{:.4f})".format(ob_low, ob_high))
                    break

        # 鎵炬渶杩戠殑鐪嬬┖ OB锛堟疾寰屾€ヨ穼鍓嶇殑鏈€寰屼竴鏍归櫧绶氾級
        for i in range(len(closes)-3, max(len(closes)-20, 2), -1):
            move_dn = (opens[i+1] - closes[i+1]) / max(abs(opens[i+1]), 1e-9)
            if closes[i] > opens[i] and move_dn > (0.010 if is_major else 0.015):  # OB姊濅欢锛堜富娴?%/灞卞1.5%锛?                ob_low  = opens[i]
                ob_high = highs[i]
                if ob_low * 0.99 <= curr <= ob_high:
                    score -= W["order_block"]
                    tags.append("鐪嬬┖OB鍗€鍩?{:.4f}-{:.4f})".format(ob_low, ob_high))
                    break

        return min(max(score, -W["order_block"]), W["order_block"]), "|".join(tags) or "鐒B"
    except:
        return 0, "OB澶辨晽"

# =====================================================
# 娴佸嫊鎬ф巸鍠亷婵撅紙Liquidity Sweep锛?# =====================================================
def analyze_liquidity_sweep(df):
    """
    鍋垫脯鍋囩獊鐮?/ 娴佸嫊鎬ф巸鍠細
    - 鍍规牸鐭毇绐佺牬楂樹綆榛炲緦绔嬪埢鏀跺洖 鈫?鎺冨柈琛岀偤
    - 鎺冨柈寰屽弽鏂瑰悜涓嬪柈鍕濈巼鏇撮珮
    """
    try:
        score = 0; tags = []
        recent_high = df['h'].tail(20).iloc[:-1].max()  # 鎺掗櫎鏈€寰屼竴鏍?        recent_low  = df['l'].tail(20).iloc[:-1].min()
        last_high   = df['h'].iloc[-1]
        last_low    = df['l'].iloc[-1]
        last_close  = df['c'].iloc[-1]
        last_open   = df['o'].iloc[-1]

        # 鍚戜笂鎺冨柈锛圞妫掍笂褰辩窔绐佺牬楂橀粸寰屾敹鍥烇級鈫?鐪嬬┖
        upper_wick = last_high - max(df['c'].iloc[-1], df['o'].iloc[-1])
        lower_wick = min(df['c'].iloc[-1], df['o'].iloc[-1]) - last_low
        body = abs(df['c'].iloc[-1] - df['o'].iloc[-1])

        if last_high > recent_high * 1.0005 and last_close < recent_high * 0.999:
            score -= W["liq_sweep"]
            tags.append("鍚戜笂鎺冨柈({:.4f})".format(recent_high))
        elif last_low < recent_low * 0.9995 and last_close > recent_low * 1.001:
            score += W["liq_sweep"]
            tags.append("鍚戜笅鎺冨柈({:.4f})".format(recent_low))
        # 闀蜂笂褰辩窔锛堝彲鑳芥槸楂橀粸鎺冨柈锛夆啋 杓曞井鐪嬬┖
        elif upper_wick > body * 2 and last_close < recent_high:
            score -= W["liq_sweep"] // 2
            tags.append("闀蜂笂褰辩窔澹撳姏")
        # 闀蜂笅褰辩窔锛堝彲鑳芥槸浣庨粸鎺冨柈锛夆啋 杓曞井鐪嬪
        elif lower_wick > body * 2 and last_close > recent_low:
            score += W["liq_sweep"] // 2
            tags.append("闀蜂笅褰辩窔鏀拹")
        elif last_close > recent_high * 1.002:
            score += W["liq_sweep"] // 2
            tags.append("鏈夋晥绐佺牬楂橀粸")
        elif last_close < recent_low * 0.998:
            score -= W["liq_sweep"] // 2
            tags.append("鏈夋晥璺岀牬浣庨粸")

        return min(max(score, -W["liq_sweep"]), W["liq_sweep"]), "|".join(tags) or "鐒℃巸鍠?
    except:
        return 0, "鎺冨柈鍒嗘瀽澶辨晽"

# =====================================================
# 鑾婂 / 鎴愪氦閲?# =====================================================
def analyze_whale(df):
    try:
        score = 0
        avg_vol  = df['v'].tail(20).mean()
        last_vol = df['v'].iloc[-1]
        prev_vol = df['v'].iloc[-2]
        if last_vol > avg_vol * 2.0:   score += W["whale"]       # 鏀惧锛氬師鏈?鍊?        elif last_vol > avg_vol * 1.5: score += W["whale"]//2   # 鏀惧锛氬師鏈?鍊?        if last_vol > prev_vol * 1.2:  score += 2               # 鏀惧锛氬師鏈?.5鍊?        curr = df['c'].iloc[-1]
        if curr < df['l'].tail(50).min() * 1.03: score += 2
        vt = df['v'].tail(5).tolist(); pt = df['c'].tail(5).tolist()
        if pt[-1] < pt[0] and vt[-1] < vt[-3]: score += 1
        return min(score, W["whale"])
    except:
        return 0

# =====================================================
# K妫掑瀷鎱?# =====================================================
def analyze_candles(df):
    try:
        score = 0; tags = []
        o=df['o'].iloc[-1]; h=df['h'].iloc[-1]; l=df['l'].iloc[-1]; c=df['c'].iloc[-1]
        po=df['o'].iloc[-2]; pc=df['c'].iloc[-2]
        body=abs(c-o); rng=h-l if h!=l else 1e-9
        upper=h-max(c,o); lower=min(c,o)-l
        unit = W["candle"]
        if lower>body*2 and upper<body*0.3 and c>o:  score+=unit;   tags.append("閷樺瓙绶?)
        if upper>body*2 and lower<body*0.3 and c<o:  score-=unit;   tags.append("娴佹槦绶?)
        if c>o and pc<po and c>po and o<pc:           score+=unit;   tags.append("澶氶牠鍚炲櫖")
        if c<o and pc>po and c<po and o>pc:           score-=unit;   tags.append("绌洪牠鍚炲櫖")
        if body/rng<0.1:                              tags.append("鍗佸瓧鏄?)
        if c>o and body/rng>0.7:                      score+=unit//2; tags.append("寮峰嫝闄界窔")
        if c<o and body/rng>0.7:                      score-=unit//2; tags.append("寮峰嫝闄扮窔")
        if len(df)>=3:
            c2=df['c'].iloc[-3]; o2=df['o'].iloc[-3]
            c1=df['c'].iloc[-2]; o1=df['o'].iloc[-2]
            if c2<o2 and abs(c1-o1)<abs(c2-o2)*0.3 and c>o and c>(c2+o2)/2:
                score+=unit; tags.append("鏃╂櫒涔嬫槦")
            if c2>o2 and abs(c1-o1)<abs(c2-o2)*0.3 and c<o and c<(c2+o2)/2:
                score-=unit; tags.append("榛冩槒涔嬫槦")
        return min(max(score,-unit),unit), "|".join(tags) or "鐒＄壒娈奒妫?
    except:
        return 0, "K妫掑け鏁?

# =====================================================
# 鍦栧舰鍨嬫厠
# =====================================================
def analyze_chart_pattern(df):
    try:
        score=0; name=""
        hi=df['h'].tail(50).tolist(); lo=df['l'].tail(50).tolist()
        mid=len(lo)//2

        # W搴曪細鍏╀綆榛炵浉杩戯紝涓枔鏈夊弽褰?        lL,rL=min(lo[:mid]),min(lo[mid:])
        mH=max(hi[mid-8:mid+8]) if len(hi)>16 else 0
        if (lL<mH*0.96 and rL<mH*0.96 and
            abs(lL-rL)/max(abs(lL),1e-9)<0.06 and
            mH>max(lL,rL)*1.03):
            score+=W["chart_pat"]; name="W搴?

        # M闋細鍏╅珮榛炵浉杩戯紝涓枔鏈夊洖钀?鈫?娓涘垎
        lH,rH=max(hi[:mid]),max(hi[mid:])
        mL=min(lo[mid-8:mid+8]) if len(lo)>16 else 0
        if (lH>mL*1.04 and rH>mL*1.04 and
            abs(lH-rH)/max(abs(lH),1e-9)<0.06 and
            mL<min(lH,rH)*0.97):
            score-=W["chart_pat"]; name="M闋紙鐪嬬┖锛?

        # 涓夎褰?        rhi=hi[-15:]; rlo=lo[-15:]
        if max(rhi)-min(rhi)<max(rhi)*0.03 and rlo[-1]>rlo[0]:
            score+=W["chart_pat"]//2; name="涓婂崌涓夎"
        elif max(rlo)-min(rlo)<max(rlo)*0.03 and rhi[-1]<rhi[0]:
            score-=W["chart_pat"]//2; name="涓嬮檷涓夎锛堢湅绌猴級"

        # 闋偐闋?搴?        if len(hi)>=45:
            t=len(hi)//3
            h1,h2,h3=max(hi[:t]),max(hi[t:2*t]),max(hi[2*t:])
            if h2>h1*1.02 and h2>h3*1.02 and abs(h1-h3)/max(h1,1e-9)<0.08:
                score-=W["chart_pat"]; name="闋偐闋傦紙寮风湅绌猴級"
            l1,l2,l3=min(lo[:t]),min(lo[t:2*t]),min(lo[2*t:])
            if l2<l1*0.98 and l2<l3*0.98 and abs(l1-l3)/max(l1,1e-9)<0.08:
                score+=W["chart_pat"]; name="闋偐搴曪紙寮风湅澶氾級"

        return min(max(score,-W["chart_pat"]),W["chart_pat"]), name or "鐒℃槑椤舰鎱?
    except:
        return 0, "褰㈡厠澶辨晽"

# =====================================================
# Trend Magic锛圕CI + ATR 鑷仼鎳夎定鍕㈢窔锛?# =====================================================
def analyze_mtf_confirm(d15, d4h, d1d):
    """
    澶氭檪妗嗘柟鍚戠⒑瑾嶏紙Multi-TimeFrame锛?    15鍒嗛悩 + 4灏忔檪 + 鏃ョ窔 涓夊€嬫檪妗嗘柟鍚戜竴鑷存墠绲﹂珮鍒?    閫欐槸鎻愬崌鍕濈巼鏈€閲嶈鐨勯亷婵惧櫒
    """
    try:
        score = 0; tags = []
        W_MTF = W.get("mtf_confirm", 14)

        def get_direction(df):
            """鐢‥MA鍒ゆ柗瑭叉檪妗嗘柟鍚?""
            c = df['c']
            if len(c) < 20: return 0
            e9  = float(ta.ema(c, length=9).iloc[-1])
            e20 = float(ta.ema(c, length=20).iloc[-1])
            curr = float(c.iloc[-1])
            if pd.isna(e9) or pd.isna(e20): return 0
            if curr > e9 > e20: return 1   # 澶氶牠
            if curr < e9 < e20: return -1  # 绌洪牠
            return 0  # 涓€?
        dir_15 = get_direction(d15)
        dir_4h = get_direction(d4h)
        dir_1d = get_direction(d1d)

        dirs = [dir_15, dir_4h, dir_1d]
        bull = sum(1 for d in dirs if d == 1)
        bear = sum(1 for d in dirs if d == -1)

        if bull == 3:
            score = W_MTF         # 涓夋鍏ㄥ 鈫?婊垮垎
            tags.append("涓夋鍏辨尟鍋氬馃敟")
        elif bull == 2:
            score = round(W_MTF * 0.6)  # 鍏╂澶?            missing = ["15m","4H","鏃ョ窔"][dirs.index(next(d for d in dirs if d != 1))]
            tags.append("闆欐澶?{})".format(missing+"寰呯⒑瑾?))
        elif bear == 3:
            score = -W_MTF
            tags.append("涓夋鍏辨尟鍋氱┖馃敟")
        elif bear == 2:
            score = -round(W_MTF * 0.6)
            missing = ["15m","4H","鏃ョ窔"][dirs.index(next(d for d in dirs if d != -1))]
            tags.append("闆欐绌?{})".format(missing+"寰呯⒑瑾?))
        else:
            score = 0
            tags.append("澶氭涓€?鍒嗘")

        return min(max(score, -W_MTF), W_MTF), "|".join(tags)
    except Exception as e:
        return 0, "MTF澶辨晽"

def analyze_trend_magic(df, cci_period=20, atr_mult=1.5):
    """
    Trend Magic by GLAZ - CCI + ATR 绲勫悎
    - CCI > 0锛氳定鍕㈢窔鍙兘涓婄Щ锛堝闋ā寮忥紝钘嶇窔锛?    - CCI < 0锛氳定鍕㈢窔鍙兘涓嬬Щ锛堢┖闋ā寮忥紝绱呯窔锛?    - 鍍规牸绌胯秺瓒ㄥ嫝绶?鈫?瓒ㄥ嫝杞夋彌瑷婅櫉
    """
    try:
        c = df['c']
        h = df['h']
        l = df['l']
        n = len(c)
        if n < cci_period + 5:
            return 0, "TM鏁告摎涓嶈冻"

        # 瑷堢畻 CCI
        typical = (h + l + c) / 3
        tp_mean = typical.rolling(cci_period).mean()
        tp_mad  = typical.rolling(cci_period).apply(lambda x: abs(x - x.mean()).mean())
        cci = (typical - tp_mean) / (0.015 * tp_mad.replace(0, 1e-9))

        # 瑷堢畻 ATR
        atr_s = ta.atr(h, l, c, length=cci_period)

        # 鍒濆鍖栬定鍕㈢窔
        tm = [float(c.iloc[0])]
        for i in range(1, n):
            atr_val = float(atr_s.iloc[i]) if not pd.isna(atr_s.iloc[i]) else float(c.iloc[i]) * 0.01
            cci_val = float(cci.iloc[i]) if not pd.isna(cci.iloc[i]) else 0
            prev_tm = tm[-1]
            price   = float(c.iloc[i])

            if cci_val > 0:
                # 澶氶牠妯″紡锛氳定鍕㈢窔鍙兘涓婄Щ
                new_tm = max(prev_tm, price - atr_val * atr_mult)
            else:
                # 绌洪牠妯″紡锛氳定鍕㈢窔鍙兘涓嬬Щ
                new_tm = min(prev_tm, price + atr_val * atr_mult)
            tm.append(new_tm)

        curr      = float(c.iloc[-1])
        tm_curr   = tm[-1]
        tm_prev   = tm[-2] if len(tm) > 1 else tm_curr
        cci_curr  = float(cci.iloc[-1]) if not pd.isna(cci.iloc[-1]) else 0
        cci_prev  = float(cci.iloc[-2]) if not pd.isna(cci.iloc[-2]) else 0

        score = 0; tag = ""

        # 澶氶牠瑷婅櫉锛欳CI > 0 涓斿児鏍煎湪瓒ㄥ嫝绶氫笂鏂?        if cci_curr > 0 and curr > tm_curr:
            dist_pct = (curr - tm_curr) / max(tm_curr, 1e-9) * 100
            if dist_pct < 2.0:
                score = W.get("trendline", 7)  # 绶婅布瓒ㄥ嫝绶氭敮鎾愶紝寮风儓鍋氬
                tag = "TM澶氶牠绶婅布鏀拹"
            else:
                score = W.get("trendline", 7) // 2
                tag = "TM澶氶牠"

        # 绌洪牠瑷婅櫉锛欳CI < 0 涓斿児鏍煎湪瓒ㄥ嫝绶氫笅鏂?        elif cci_curr < 0 and curr < tm_curr:
            dist_pct = (tm_curr - curr) / max(tm_curr, 1e-9) * 100
            if dist_pct < 2.0:
                score = -W.get("trendline", 7)
                tag = "TM绌洪牠绶婅布澹撳姏"
            else:
                score = -W.get("trendline", 7) // 2
                tag = "TM绌洪牠"

        # 閲戝弶锛欳CI 寰炶矤杞夋锛堣定鍕㈣綁澶氾級
        elif cci_prev <= 0 and cci_curr > 0:
            score = W.get("trendline", 7)
            tag = "TM瓒ㄥ嫝杞夊馃數"

        # 姝诲弶锛欳CI 寰炴杞夎矤锛堣定鍕㈣綁绌猴級
        elif cci_prev >= 0 and cci_curr < 0:
            score = -W.get("trendline", 7)
            tag = "TM瓒ㄥ嫝杞夌┖馃敶"

        return min(max(score, -W.get('trendline', 7)), W.get('trendline', 7)), tag or "TM涓€?
    except Exception as e:
        return 0, "TM瑷堢畻澶辨晽"

def analyze_trend(df4h):
    """
    鐪熸鐨勮定鍕㈢窔鍒ゆ柗锛?    - 鐢ㄧ窔鎬у洖姝歌▓绠楁敮鎾愮窔鍜屽鍔涚窔鐨勬枩鐜?    - 鏂滅巼鍚戜笂 + 鍍规牸鍦ㄧ窔涓婃柟 = 涓婂崌瓒ㄥ嫝
    - 鏂滅巼鍚戜笅 + 鍍规牸鍦ㄧ窔涓嬫柟 = 涓嬮檷瓒ㄥ嫝
    """
    try:
        score=0; tags=[]
        lo=df4h['l'].tolist(); hi=df4h['h'].tolist()
        curr=df4h['c'].iloc[-1]; n=len(lo)
        unit=W["trendline"]
        if n < 10:
            return 0, "瓒ㄥ嫝绶氭暩鎿氫笉瓒?

        # 鐢ㄦ渶杩?0鏍筀妫掕▓绠?        recent_lo = lo[-20:]
        recent_hi = hi[-20:]
        x = list(range(len(recent_lo)))

        # 绶氭€у洖姝歌▓绠楁枩鐜?        def slope(vals):
            n_ = len(vals)
            sx = sum(x); sy = sum(vals)
            sxy = sum(x[i]*vals[i] for i in range(n_))
            sxx = sum(xi**2 for xi in x)
            denom = n_*sxx - sx*sx
            if denom == 0: return 0
            return (n_*sxy - sx*sy) / denom

        lo_slope = slope(recent_lo)
        hi_slope = slope(recent_hi)

        # 鏈€杩戞敮鎾愮窔鐨勫€硷紙鐢ㄦ渶寰屼竴榛烇級
        lo_intercept = sum(recent_lo)/len(recent_lo) - lo_slope * sum(x)/len(x)
        support_val = lo_slope * x[-1] + lo_intercept

        hi_intercept = sum(recent_hi)/len(recent_hi) - hi_slope * sum(x)/len(x)
        resist_val = hi_slope * x[-1] + hi_intercept

        # 鐢ˋTR鍒ゆ柗璺濋洟
        atr_approx = sum(hi[-14][i]-lo[-14][i] if isinstance(hi, list) else 0 for i in range(14)) if False else abs(curr * 0.01)
        try:
            atr_series = df4h['h'].tail(14) - df4h['l'].tail(14)
            atr_approx = float(atr_series.mean())
        except:
            atr_approx = curr * 0.01

        # 鏀拹瓒ㄥ嫝绶氬垽鏂?        if lo_slope > 0:  # 涓婂崌瓒ㄥ嫝鏀拹绶?            dist_from_support = (curr - support_val) / max(atr_approx, 1e-9)
            if dist_from_support < 1.0:   # 鍍规牸鎺ヨ繎涓婂崌鏀拹
                score += unit; tags.append("4H涓婂崌瓒ㄥ嫝鏀拹")
            elif dist_from_support > 5.0:  # 闆㈡敮鎾愬お閬?                score += unit//2; tags.append("涓婂崌瓒ㄥ嫝涓")
            else:
                score += unit//2; tags.append("4H涓婂崌瓒ㄥ嫝")
        elif lo_slope < -atr_approx * 0.05:  # 鏄庨’涓嬮檷
            if curr < support_val:
                score -= unit; tags.append("璺岀牬涓嬮檷瓒ㄥ嫝浣庨粸")
            else:
                score -= unit//2; tags.append("涓嬮檷瓒ㄥ嫝涓?)

        # 澹撳姏瓒ㄥ嫝绶氬垽鏂?        if hi_slope < 0:  # 涓嬮檷澹撳姏绶?            dist_from_resist = (resist_val - curr) / max(atr_approx, 1e-9)
            if dist_from_resist < 1.0:  # 鎺ヨ繎涓嬮檷澹撳姏
                score -= unit//2; tags.append("鍙楀涓嬮檷瓒ㄥ嫝绶?)
        elif hi_slope > 0:  # 涓婂崌澹撳姏绐佺牬
            if curr > resist_val:
                score += unit//2; tags.append("绐佺牬涓婂崌澹撳姏")

        return min(max(score, -unit), unit), "|".join(tags) or "瓒ㄥ嫝涓€?
    except Exception as e:
        return 0, "瓒ㄥ嫝绶氬け鏁?

def get_best_atr_params(breakdown_keys):
    with LEARN_LOCK:
        db   = LEARN_DB
        pkey = "|".join(sorted(breakdown_keys))
        if pkey in db["pattern_stats"]:
            st = db["pattern_stats"][pkey]
            if st.get("sample_count", 0) >= AI_MIN_SAMPLE_EFFECT:
                return st.get("best_sl", db["atr_params"]["default_sl"]),                        st.get("best_tp", db["atr_params"]["default_tp"])
        best_match=None; best_overlap=0; ks=set(breakdown_keys)
        for k,st in db["pattern_stats"].items():
            ov=len(ks & set(k.split("|")))
            if ov>best_overlap and st.get("sample_count",0)>=AI_MIN_SAMPLE_EFFECT:
                best_overlap=ov; best_match=st
        if best_match and best_overlap>=2:
            return best_match.get("best_sl",db["atr_params"]["default_sl"]),                    best_match.get("best_tp",db["atr_params"]["default_tp"])
        return db["atr_params"]["default_sl"], db["atr_params"]["default_tp"]


def get_learned_rr_target(symbol, regime, setup, breakdown_keys, sl_mult, tp_mult):
    """璁?TP 鐢?AI 瀛稿埌鐨?RR 姹哄畾锛岃€屼笉鏄浐瀹?ATR 鍊嶆暩銆?""
    base_rr = float(tp_mult or 3.0) / max(float(sl_mult or 2.0), 1e-9)
    rr_samples = []

    def _push(rr, weight=1.0):
        try:
            rr = float(rr or 0)
            weight = float(weight or 0)
        except Exception:
            return
        if rr > 0.5 and weight > 0:
            rr_samples.append((rr, weight))

    with LEARN_LOCK:
        db = LEARN_DB
        pkey = "|".join(sorted(breakdown_keys))
        pst = dict((db.get("pattern_stats", {}) or {}).get(pkey, {}) or {})
        if int(pst.get("sample_count", 0) or 0) >= AI_MIN_SAMPLE_EFFECT:
            _push(float(pst.get("best_tp", tp_mult) or tp_mult) / max(float(pst.get("best_sl", sl_mult) or sl_mult), 1e-9), 3.0)

        ks = set(breakdown_keys or [])
        best_match = None
        best_overlap = 0
        for k, st in (db.get("pattern_stats", {}) or {}).items():
            cnt = int(st.get("sample_count", 0) or 0)
            if cnt < AI_MIN_SAMPLE_EFFECT:
                continue
            ov = len(ks & set(str(k).split("|")))
            if ov > best_overlap:
                best_overlap = ov
                best_match = st
        if best_match and best_overlap >= 2:
            _push(float(best_match.get("best_tp", tp_mult) or tp_mult) / max(float(best_match.get("best_sl", sl_mult) or sl_mult), 1e-9), 2.0)

        rows = [t for t in db.get("trades", []) or [] if _is_live_source(t.get("source")) and t.get("result") in ("win", "loss")]
        if symbol:
            rows = [t for t in rows if str(t.get("symbol")) == str(symbol)]
        if regime:
            rows = [t for t in rows if str((t.get("breakdown") or {}).get("Regime", "neutral")) == str(regime)]
        if setup:
            rows = [t for t in rows if str((t.get("breakdown") or {}).get("Setup", "")) == str(setup)]
        rows = rows[-24:]
        for t in rows:
            rr = float(t.get("atr_mult_tp", 0) or 0) / max(float(t.get("atr_mult_sl", 0) or 0), 1e-9)
            w = 2.2 if t.get("result") == "win" else 0.9
            _push(rr, w)

    if rr_samples:
        total_w = sum(w for _, w in rr_samples)
        learned_rr = sum(rr * w for rr, w in rr_samples) / max(total_w, 1e-9)
    else:
        learned_rr = base_rr

    regime = str(regime or "neutral")
    if regime == 'range':
        learned_rr = min(max(learned_rr, 1.20), 2.20)
    elif regime in ('news', 'breakout'):
        learned_rr = min(max(learned_rr, 1.60), 3.60)
    elif regime == 'trend':
        learned_rr = min(max(learned_rr, 1.45), 3.20)
    else:
        learned_rr = min(max(learned_rr, 1.35), 2.80)

    return round(max(learned_rr, MIN_RR_HARD_FLOOR), 2)

# =====================================================
# 涓绘妧琛撳垎鏋愶紙鍏ㄩ€辨湡锛屾豢鍒?00锛?# =====================================================
# 鐭窔绂佹涓嬪柈锛堢暀绲﹂暦鏈熷€変綅锛?SHORT_TERM_EXCLUDED = {'BTC/USDT:USDT'}

# 涓绘祦骞ｆ竻鍠紙浣跨敤鏀惧鐗堣鍒嗭級
MAJOR_COINS = {
    'BTC/USDT:USDT','ETH/USDT:USDT','BNB/USDT:USDT','SOL/USDT:USDT',
    'XRP/USDT:USDT','ADA/USDT:USDT','DOGE/USDT:USDT','AVAX/USDT:USDT',
    'DOT/USDT:USDT','LINK/USDT:USDT','LTC/USDT:USDT','BCH/USDT:USDT',
    'UNI/USDT:USDT','ATOM/USDT:USDT','MATIC/USDT:USDT',
}


def analyze_market_regime_for_symbol(d15, d4h, d1d):
    """
    鏂瑰悜鍒ゆ柗鏍稿績锛?    涓嶅啀鍙湅绺藉垎锛岃€屾槸鍏堝垽鏂烽€欏€嬪梗鐝惧湪灞柤
    瓒ㄥ嫝寤剁簩 / 鍥炶俯绾屾敾 / 闇囩洩 / 鍙嶅綀鍙嶆娊銆?    """
    try:
        c15 = d15['c'].astype(float)
        h15 = d15['h'].astype(float)
        l15 = d15['l'].astype(float)
        c4 = d4h['c'].astype(float)
        c1 = d1d['c'].astype(float)

        curr = float(c15.iloc[-1])
        atr15 = max(safe_last(ta.atr(h15, l15, c15, length=14), curr * 0.004), curr * 0.003)

        e9_15  = safe_last(ta.ema(c15, length=9), curr)
        e21_15 = safe_last(ta.ema(c15, length=21), curr)
        e55_15 = safe_last(ta.ema(c15, length=55), curr)

        e21_4 = safe_last(ta.ema(c4, length=21), curr)
        e55_4 = safe_last(ta.ema(c4, length=55), curr)
        e20_1 = safe_last(ta.ema(c1, length=20), curr)
        e50_1 = safe_last(ta.ema(c1, length=50), curr)

        slope15 = _linreg_slope(c15.tail(8).tolist()) / max(curr, 1e-9) * 100
        slope4h = _linreg_slope(c4.tail(8).tolist()) / max(curr, 1e-9) * 100

        bull_stack = curr > e9_15 > e21_15 > e55_15 and curr > e21_4 > e55_4 and curr > e20_1 > e50_1
        bear_stack = curr < e9_15 < e21_15 < e55_15 and curr < e21_4 < e55_4 and curr < e20_1 < e50_1

        pullback_long = bull_stack and abs(curr - e21_15) / max(atr15, 1e-9) <= 0.9
        rebound_short = bear_stack and abs(curr - e21_15) / max(atr15, 1e-9) <= 0.9

        if bull_stack and slope15 > 0.08 and slope4h > 0.04:
            return 8 if pullback_long else 6, 1, ("澶氶牠鍥炶俯绾屾敾" if pullback_long else "澶氶牠寤剁簩"), True
        if bear_stack and slope15 < -0.08 and slope4h < -0.04:
            return -8 if rebound_short else -6, -1, ("绌洪牠鍙嶅綀绾岃穼" if rebound_short else "绌洪牠寤剁簩"), True

        if curr > e21_4 and curr > e20_1 and slope4h > 0:
            return 3, 1, "鍋忓浣嗘湭瀹屽叏鍏辨尟", True
        if curr < e21_4 and curr < e20_1 and slope4h < 0:
            return -3, -1, "鍋忕┖浣嗘湭瀹屽叏鍏辨尟", True

        return 0, 0, "鍗€闁撻渿鐩?, False
    except Exception:
        return 0, 0, "鏂瑰悜鍒ゆ柗澶辨晽", False


def analyze_entry_timing_quality(d15, d4h, direction_hint=0):
    """
    閫插牬鍝佽唱锛?    闋嗗嫝浣嗗お閬犱笉杩斤紱闋嗗嫝鍥炶俯銆佺獊鐮村緦绔欑┅銆侀噺鍍归厤鍚堟墠鍔犲垎銆?    """
    try:
        c = d15['c'].astype(float)
        o = d15['o'].astype(float)
        h = d15['h'].astype(float)
        l = d15['l'].astype(float)
        v = d15['v'].astype(float)
        curr = float(c.iloc[-1])
        atr = max(safe_last(ta.atr(h, l, c, length=14), curr * 0.004), curr * 0.003)
        ema9 = safe_last(ta.ema(c, length=9), curr)
        ema21 = safe_last(ta.ema(c, length=21), curr)
        vol_now = float(v.tail(3).mean()) if len(v) >= 3 else float(v.iloc[-1])
        vol_avg = float(v.tail(24).mean()) if len(v) >= 24 else vol_now
        hh = float(h.tail(20).iloc[:-1].max()) if len(h) > 21 else float(h.max())
        ll = float(l.tail(20).iloc[:-1].min()) if len(l) > 21 else float(l.min())

        body = abs(float(c.iloc[-1]) - float(o.iloc[-1]))
        close_pos = (float(c.iloc[-1]) - float(l.iloc[-1])) / max(float(h.iloc[-1]) - float(l.iloc[-1]), 1e-9)
        ext = (curr - ema21) / max(atr, 1e-9)

        score = 0
        tags = []

        if direction_hint > 0:
            if curr > ema9 > ema21:
                score += 2; tags.append("15m澶氶牠鎺掑垪")
            if abs(curr - ema21) / max(atr, 1e-9) <= 0.8:
                score += 3; tags.append("鍥炶俯鍧囩窔闄勮繎")
            if curr >= hh * 0.998 and close_pos > 0.65 and body > atr * 0.45 and vol_now > vol_avg * 1.1:
                score += 3; tags.append("绐佺牬甯堕噺绔欑┅")
            if ext > 1.5:
                score -= 4; tags.append("闆㈠潎绶氶亷閬?)
            if close_pos < 0.45 and curr >= hh * 0.998:
                score -= 2; tags.append("绐佺牬鏀朵笉绌?)
        elif direction_hint < 0:
            if curr < ema9 < ema21:
                score += 2; tags.append("15m绌洪牠鎺掑垪")
            if abs(curr - ema21) / max(atr, 1e-9) <= 0.8:
                score += 3; tags.append("鍙嶅綀鍧囩窔闄勮繎")
            low_close_pos = (float(h.iloc[-1]) - float(c.iloc[-1])) / max(float(h.iloc[-1]) - float(l.iloc[-1]), 1e-9)
            if curr <= ll * 1.002 and low_close_pos > 0.65 and body > atr * 0.45 and vol_now > vol_avg * 1.1:
                score += 3; tags.append("璺岀牬甯堕噺绔欑┅")
            if ext < -1.5:
                score -= 4; tags.append("闆㈠潎绶氶亷閬?)
            if low_close_pos < 0.45 and curr <= ll * 1.002:
                score -= 2; tags.append("璺岀牬鏀朵笉绌?)
        else:
            score -= 1
            tags.append("鏂瑰悜鏈槑")

        score = max(min(score, 8), -8)
        return score, "|".join(tags) if tags else "閫插牬鍝佽唱涓€鑸?
    except Exception:
        return 0, "閫插牬鍝佽唱澶辨晽"






def _calc_unified_targets(entry_price, atr_value, sl_mult, rr_target, side):
    entry_price = float(entry_price or 0)
    atr_value = max(float(atr_value or 0), entry_price * 0.001, 1e-9)
    sl_mult = max(float(sl_mult or 0), 0.8)
    rr_target = max(float(rr_target or 0), MIN_RR_HARD_FLOOR)
    side = str(side or '').lower()
    stop_dist = atr_value * sl_mult
    if side in ('long', 'buy', 'bull'):
        sl = round(entry_price - stop_dist, 6)
        tp = round(entry_price + stop_dist * rr_target, 6)
    else:
        sl = round(entry_price + stop_dist, 6)
        tp = round(entry_price - stop_dist * rr_target, 6)
    rr_ratio = abs(tp - entry_price) / max(abs(entry_price - sl), 1e-9)
    return sl, tp, round(rr_ratio, 4)


def analyze_breakout_forecast(d15, d4h, direction_hint=0):
    """鎻愬墠鍒ゆ柗绐佺牬钃勫嫝锛岄伩鍏嶇獊鐮村緦鎵嶈拷楂樸€?""
    try:
        c = d15['c'].astype(float); h = d15['h'].astype(float); l = d15['l'].astype(float); v = d15['v'].astype(float)
        curr = float(c.iloc[-1])
        atr = max(safe_last(ta.atr(h, l, c, length=14), curr * 0.004), curr * 0.003)
        lookback = min(20, max(len(d15) - 2, 8))
        recent_h = float(h.tail(lookback).iloc[:-1].max()) if len(h) > lookback else float(h.max())
        recent_l = float(l.tail(lookback).iloc[:-1].min()) if len(l) > lookback else float(l.min())
        range_now = max(recent_h - recent_l, atr)
        bb = ta.bbands(c, length=20, std=2.0)
        bb_up = safe_last(bb.iloc[:, 0], curr) if bb is not None and not bb.empty else curr
        bb_low = safe_last(bb.iloc[:, 2], curr) if bb is not None and not bb.empty else curr
        bb_width = abs(bb_up - bb_low) / max(curr, 1e-9)
        vol_now = float(v.tail(3).mean()) if len(v) >= 3 else float(v.iloc[-1])
        vol_base = max(float(v.tail(20).mean()) if len(v) >= 20 else vol_now, 1e-9)
        vol_ratio = vol_now / vol_base
        ema9 = safe_last(ta.ema(c, length=9), curr)
        ema21 = safe_last(ta.ema(c, length=21), curr)
        ext = abs(curr - ema21) / max(atr, 1e-9)
        score = 0
        tags = []
        meta = {'ready': False, 'near_break': False, 'distance_atr': 99.0, 'vol_ratio': round(vol_ratio, 3), 'ext_atr': round(ext, 3)}
        if direction_hint > 0:
            dist = (recent_h - curr) / max(atr, 1e-9)
            meta['distance_atr'] = round(dist, 3)
            if 0 <= dist <= 0.65 and curr >= ema9 >= ema21:
                score += 3; tags.append('绐佺牬鍓嶈布杩戦珮榛?)
                meta['near_break'] = True
            if bb_width <= 0.022 and range_now <= atr * 8.5:
                score += 2; tags.append('娉㈠嫊鏀舵杺钃勫嫝')
            if vol_ratio >= 1.08 and vol_ratio <= 1.9:
                score += 2; tags.append('閲忚兘婧拰鏀惧ぇ')
            if ext > 1.45:
                score -= 4; tags.append('閬庣啽鍏堢瓑鍥炶俯')
            if meta['near_break'] and score >= 5 and ext <= 1.2:
                score += 1; tags.append('鍙彁鏃╂簴鍌欑獊鐮?)
                meta['ready'] = True
        elif direction_hint < 0:
            dist = (curr - recent_l) / max(atr, 1e-9)
            meta['distance_atr'] = round(dist, 3)
            if 0 <= dist <= 0.65 and curr <= ema9 <= ema21:
                score -= 3; tags.append('绐佺牬鍓嶈布杩戜綆榛?)
                meta['near_break'] = True
            if bb_width <= 0.022 and range_now <= atr * 8.5:
                score -= 2; tags.append('娉㈠嫊鏀舵杺钃勫嫝')
            if vol_ratio >= 1.08 and vol_ratio <= 1.9:
                score -= 2; tags.append('閲忚兘婧拰鏀惧ぇ')
            if ext > 1.45:
                score += 4; tags.append('閬庣啽鍏堢瓑鍙嶅綀')
            if meta['near_break'] and abs(score) >= 5 and ext <= 1.2:
                score -= 1; tags.append('鍙彁鏃╂簴鍌欒穼鐮?)
                meta['ready'] = True
        return int(max(min(score, 7), -7)), '|'.join(tags) if tags else '鐒℃彁鍓嶇獊鐮寸祼妲?, meta
    except Exception as e:
        return 0, f'鎻愬墠绐佺牬澶辨晽:{str(e)[:20]}', {'ready': False, 'near_break': False, 'distance_atr': 99.0, 'vol_ratio': 1.0, 'ext_atr': 9.0}


def analyze_fvg_retest_quality(d15, d4h, direction_hint=0):
    """FVG 鍥炶俯/鍙嶅綀鍝佽唱锛岄伩鍏嶆甯稿洖韪╄瑾ゅ垽鎴愯拷鍍广€?""
    try:
        fvg_score, fvg_tag = analyze_fvg(d4h)
        c = d15['c'].astype(float); h = d15['h'].astype(float); l = d15['l'].astype(float)
        curr = float(c.iloc[-1])
        atr = max(safe_last(ta.atr(h, l, c, length=14), curr * 0.004), curr * 0.003)
        ema21 = safe_last(ta.ema(c, length=21), curr)
        ext = abs(curr - ema21) / max(atr, 1e-9)
        score = 0
        tags = []
        meta = {'active': False, 'is_pullback': False, 'is_chase_ok': False, 'ext_atr': round(ext, 3)}
        if direction_hint > 0 and (fvg_score > 0 or '鎺ヨ繎FVG鏀拹缂哄彛' in str(fvg_tag) or 'FVG鍋氬缂哄彛' in str(fvg_tag)):
            score += 2 if '鎺ヨ繎FVG鏀拹缂哄彛' in str(fvg_tag) else 4 if 'FVG鍋氬缂哄彛' in str(fvg_tag) else 1
            tags.append(str(fvg_tag))
            meta.update({'active': True, 'is_pullback': True})
            if ext <= 1.15:
                score += 1
                tags.append('FVG鍥炶俯鏈牬浣?)
                meta['is_chase_ok'] = True
        elif direction_hint < 0 and (fvg_score < 0 or '鎺ヨ繎FVG澹撳姏缂哄彛' in str(fvg_tag) or 'FVG鍋氱┖缂哄彛' in str(fvg_tag)):
            score -= 2 if '鎺ヨ繎FVG澹撳姏缂哄彛' in str(fvg_tag) else 4 if 'FVG鍋氱┖缂哄彛' in str(fvg_tag) else 1
            tags.append(str(fvg_tag))
            meta.update({'active': True, 'is_pullback': True})
            if ext <= 1.15:
                score -= 1
                tags.append('FVG鍙嶅綀鏈牬浣?)
                meta['is_chase_ok'] = True
        return int(max(min(score, 6), -6)), '|'.join(dict.fromkeys(tags)) if tags else '鐒VG鍥炶俯', meta
    except Exception as e:
        return 0, f'FVG鍥炶俯澶辨晽:{str(e)[:20]}', {'active': False, 'is_pullback': False, 'is_chase_ok': False, 'ext_atr': 9.0}

def analyze_fake_breakout(df, directional_bias=0):
    """
    鍋囩獊鐮?/ 鍋囪穼鐮撮亷婵?    鍥炲偝: (score_adjust, tag, meta)
    meta = {fakeout: bool, direction: 'up'/'down'/None, strength: float}
    """
    try:
        if df is None or len(df) < max(BREAKOUT_LOOKBACK + 3, 12):
            return 0, '璩囨枡涓嶈冻', {'fakeout': False, 'direction': None, 'strength': 0.0}

        sub = df.copy().reset_index(drop=True)
        last = sub.iloc[-1]
        prev = sub.iloc[-2]
        ref = sub.iloc[:-1].tail(max(BREAKOUT_LOOKBACK, 8))
        hh = float(ref['h'].max())
        ll = float(ref['l'].min())
        close_ = float(last['c'])
        high = float(last['h'])
        low = float(last['l'])
        open_ = float(last['o'])
        atr = safe_last(ta.atr(sub['h'], sub['l'], sub['c'], length=14), max(close_ * 0.004, 1e-9))
        body = abs(close_ - open_)
        upper = high - max(close_, open_)
        lower = min(close_, open_) - low
        score = 0
        tag = '鐒″亣绐佺牬'
        meta = {'fakeout': False, 'direction': None, 'strength': 0.0}

        broke_up = high > hh * 1.0008
        closed_back_in_up = close_ < hh and upper > body * 0.8
        broke_down = low < ll * 0.9992
        closed_back_in_down = close_ > ll and lower > body * 0.8

        if broke_up and closed_back_in_up:
            strength = min((high - close_) / max(atr, 1e-9), 3.0)
            meta = {'fakeout': True, 'direction': 'up', 'strength': round(strength, 2)}
            score = -min(8, max(3, int(round(2.5 + strength * 1.8))))
            if directional_bias < 0:
                score = abs(score)
            tag = '鍋囩獊鐮村洖钀?
        elif broke_down and closed_back_in_down:
            strength = min((close_ - low) / max(atr, 1e-9), 3.0)
            meta = {'fakeout': True, 'direction': 'down', 'strength': round(strength, 2)}
            score = min(8, max(3, int(round(2.5 + strength * 1.8))))
            if directional_bias > 0:
                score = -abs(score)
            tag = '鍋囪穼鐮村洖鏀?

        # 璺熺暥鍓嶅亸鍚戠浉鍙嶆檪锛岄澶栧姞閲嶆嚥缃?鐛庡嫷
        if meta['fakeout'] and directional_bias != 0:
            if directional_bias > 0 and meta['direction'] == 'up':
                score -= 2
            elif directional_bias < 0 and meta['direction'] == 'down':
                score += 2

        return score, tag, meta
    except Exception as e:
        return 0, f'鍋囩獊鐮村垎鏋愬け鏁?{str(e)[:24]}', {'fakeout': False, 'direction': None, 'strength': 0.0}

def analyze_legacy_shadow_1(symbol):
    is_major = symbol in MAJOR_COINS  # 鏄惁鐐轰富娴佸梗
    try:
        d15=pd.DataFrame(exchange.fetch_ohlcv(symbol,'15m',limit=ANALYZE_15M_LIMIT),columns=['t','o','h','l','c','v'])
        time.sleep(0.2)
        d4h=pd.DataFrame(exchange.fetch_ohlcv(symbol,'4h', limit=ANALYZE_4H_LIMIT), columns=['t','o','h','l','c','v'])
        time.sleep(0.2)
        d1d=pd.DataFrame(exchange.fetch_ohlcv(symbol,'1d', limit=ANALYZE_1D_LIMIT), columns=['t','o','h','l','c','v'])
        time.sleep(0.1)

        score=0.0; tags=[]; curr=d15['c'].iloc[-1]; breakdown={}

        regime_s, regime_bias, regime_tag, regime_ok = analyze_market_regime_for_symbol(d15, d4h, d1d)
        score += regime_s
        breakdown['鏂瑰悜鍝佽唱'] = regime_s
        tags.append(regime_tag)

        entry_s0, entry_tag0 = analyze_entry_timing_quality(d15, d4h, regime_bias)
        score += entry_s0
        breakdown['閫插牬鍝佽唱'] = entry_s0
        if entry_tag0:
            tags.append(entry_tag0)

        # RSI锛堝惈鑳岄洟鍋垫脯锛?        rsi_series = ta.rsi(d15['c'], length=14)
        rsi = safe_last(rsi_series, 50)
        rs = W["rsi"] if rsi<30 else W["rsi"]//2 if rsi<40 else -W["rsi"] if rsi>70 else -W["rsi"]//2 if rsi>60 else 0

        # RSI 鑳岄洟鍋垫脯锛堥珮鍕濈巼淇¤櫉锛?        try:
            if len(rsi_series) >= 10 and not rsi_series.isna().all():
                price_recent = d15['c'].tail(10).tolist()
                rsi_recent   = rsi_series.tail(10).tolist()
                # 鐪嬪鑳岄洟锛氬児鏍煎壍鏂颁綆浣哛SI娌掑壍鏂颁綆
                if price_recent[-1] < min(price_recent[:-1]) and rsi_recent[-1] > min(rsi_recent[:-1]):
                    rs = W["rsi"]
                    tags.append("RSI鐪嬪鑳岄洟馃敟")
                # 鐪嬬┖鑳岄洟锛氬児鏍煎壍鏂伴珮浣哛SI娌掑壍鏂伴珮
                elif price_recent[-1] > max(price_recent[:-1]) and rsi_recent[-1] < max(rsi_recent[:-1]):
                    rs = -W["rsi"]
                    tags.append("RSI鐪嬬┖鑳岄洟馃敟")
        except:
            pass

        score+=rs; breakdown['RSI({:.0f})'.format(rsi)]=rs
        if rs and 'RSI' not in str(tags):
            tags.append("RSI{:.0f}".format(rsi))

        # MACD锛堥噾鍙夋鍙?寮峰害锛?        macd=ta.macd(d15['c']); ms=0
        if macd is not None and 'MACDh_12_26_9' in macd.columns:
            mh=safe_last(macd['MACDh_12_26_9']); mp=float(macd['MACDh_12_26_9'].iloc[-2])
            ml=safe_last(macd['MACD_12_26_9']); ms_line=safe_last(macd['MACDs_12_26_9'])
            if mh>0 and mp<0:
                strength = min(abs(mh)/max(abs(ml),1e-9), 1.0)
                ms = int(W["macd"] * (0.7 + 0.3*strength))
                tags.append("MACD閲戝弶")
            elif mh<0 and mp>0:
                strength = min(abs(mh)/max(abs(ml),1e-9), 1.0)
                ms = -int(W["macd"] * (0.7 + 0.3*strength))
                tags.append("MACD姝诲弶")
            elif mh>0:
                ms=W["macd"]//2; tags.append("MACD澶?)
            else:
                ms=-W["macd"]//2; tags.append("MACD绌?)
        score+=ms; breakdown['MACD']=ms

        # 澶氭檪妗嗙⒑瑾?        mtf_s, mtf_tag = analyze_mtf_confirm(d15, d4h, d1d)
        score += mtf_s; breakdown['澶氭檪妗?] = mtf_s
        if mtf_tag and "涓€? not in mtf_tag:
            tags.append(mtf_tag)

        # 鏃ョ窔EMA
        e20=ta.ema(d1d['c'],length=20); e50=ta.ema(d1d['c'],length=50)
        e9=ta.ema(d1d['c'],length=9); es=0
        if e20 is not None and e50 is not None and not e20.empty and not e50.empty:
            v20=safe_last(e20); v50=safe_last(e50)
            v9=safe_last(e9,v20) if e9 is not None and not e9.empty else v20
            if curr>v20>v50:
                es=W["ema_trend"]; tags.append("鏃ョ窔澶氭帓")
            elif curr<v20<v50:
                es=-W["ema_trend"]; tags.append("鏃ョ窔绌烘帓")
            elif curr>v20:
                es=W["ema_trend"]//2; tags.append("EMA鏀拹")
            else:
                es=-W["ema_trend"]//2; tags.append("EMA鍙嶅")
            if e9 is not None and not e9.empty and len(e9) >= 2 and len(e20) >= 2:
                v9_prev = float(e9.iloc[-2]) if not pd.isna(e9.iloc[-2]) else v9
                v20_prev = float(e20.iloc[-2]) if not pd.isna(e20.iloc[-2]) else v20
                if v9_prev <= v20_prev and v9 > v20:
                    if is_major:
                        es = min(es + W["ema_trend"]//2, W["ema_trend"])
                        tags.append("EMA閲戝弶馃數")
                elif v9_prev >= v20_prev and v9 < v20:
                    if is_major:
                        es = max(es - W["ema_trend"]//2, -W["ema_trend"])
                        tags.append("EMA姝诲弶馃敶")
        score+=es; breakdown['鏃ョ窔EMA']=es

        # ADX
        adx_s,adx_tag=analyze_adx(d15)
        score+=adx_s; breakdown['ADX']=adx_s; tags.append(adx_tag)
        try:
            adx_df2 = ta.adx(d15['h'], d15['l'], d15['c'], length=14)
            adx_val2 = safe_last(adx_df2['ADX_14'], 25) if adx_df2 is not None else 25
            if adx_val2 < 20:
                score = score * 0.8
        except:
            pass

        # VWAP
        vwap_s,vwap_tag=analyze_vwap(d15)
        score+=vwap_s; breakdown['VWAP']=vwap_s
        if vwap_s!=0:
            tags.append(vwap_tag)

        # 4H 澹撳姏鏀拹
        r4h=d4h['h'].tail(20).max(); s4h=d4h['l'].tail(20).min(); mid4=(r4h+s4h)/2; ps=0
        atr_4h_s=ta.atr(d4h['h'],d4h['l'],d4h['c'],length=14)
        atr_4h=safe_last(atr_4h_s, curr*0.01)
        dist_res = (r4h - curr) / atr_4h if atr_4h>0 else 999
        dist_sup = (curr - s4h) / atr_4h if atr_4h>0 else 999
        sr_near = 0.5 if is_major else 0.3
        sr_mid  = 1.0 if is_major else 0.7
        if dist_res < 0.3:
            ps=W["support_res"];     tags.append("绐佺牬澹撳姏{:.4f}".format(r4h))
        elif dist_sup < 0.3:
            ps=-W["support_res"];    tags.append("璺岀牬鏀拹{:.4f}".format(s4h))
        elif dist_res < sr_near:
            ps=W["support_res"]//2;  tags.append("鎺ヨ繎澹撳姏{:.4f}".format(r4h))
        elif dist_sup < sr_near:
            ps=W["support_res"]//2;  tags.append("鎺ヨ繎鏀拹{:.4f}".format(s4h))
        elif dist_sup < sr_mid:
            ps=W["support_res"]//3;  tags.append("鏀拹鍗€闁撳収")
        elif curr>mid4:
            ps=W["support_res"]//4;  tags.append("鍗€闁撲笂鍗?)
        else:
            ps=-W["support_res"]//4; tags.append("鍗€闁撲笅鍗?)
        score+=ps; breakdown['澹撳姏鏀拹({:.4f}/{:.4f})'.format(s4h,r4h)]=ps

        # Trend Magic + 瓒ㄥ嫝绶?        tm_s, tm_tag = analyze_trend_magic(d4h)
        tl_s, tl_tag = analyze_trend(d4h)
        if (tm_s > 0 and tl_s > 0) or (tm_s < 0 and tl_s < 0):
            trend_final = tm_s
            if tl_tag != "瓒ㄥ嫝涓€?:
                tags.append(tl_tag)
        else:
            trend_final = (tm_s + tl_s) // 2
        trend_final = min(max(trend_final, -W["trendline"]), W["trendline"])
        score += trend_final; breakdown['TrendMagic'] = trend_final
        if tm_tag and tm_tag != "TM涓€?:
            tags.append(tm_tag)

        # K妫?        cs,cd=analyze_candles(d15)
        score+=cs; breakdown['K妫掑瀷鎱?]=cs
        if cd!="鐒＄壒娈奒妫?:
            tags.append(cd)

        # 鍦栧舰鍨嬫厠
        chs,chd=analyze_chart_pattern(d4h)
        score+=chs; breakdown['鍦栧舰鍨嬫厠']=chs
        if chd!="鐒″舰鎱?:
            tags.append(chd)

        # OB
        ob_s,ob_tag=analyze_order_block(d4h, is_major=is_major)
        score+=ob_s; breakdown['OB姗熸']=ob_s
        if ob_tag!="鐒B":
            tags.append(ob_tag)

        # ICT
        ict_s,ict_tag=analyze_ict(d4h, d15)
        score+=ict_s; breakdown['BOS/CHoCH']=ict_s
        if ict_tag!="鐒CT瑷婅櫉":
            tags.append(ict_tag)

        # FVG
        fvg_s,fvg_tag=analyze_fvg(d4h)
        fvg_bonus = min(max(fvg_s, -3), 3)
        score+=fvg_bonus; breakdown['FVG缂哄彛']=fvg_bonus
        if fvg_tag!="鐒VG":
            tags.append(fvg_tag)

        # 娴佸嫊鎬ф巸鍠?        liq_s,liq_tag=analyze_liquidity_sweep(d15)
        score+=liq_s; breakdown['娴佸嫊鎬ф巸鍠?]=liq_s
        if liq_tag!="鐒℃巸鍠?:
            tags.append(liq_tag)

        # 鑾婂閲忚兘
        ws=analyze_whale(d15)
        score+=ws; breakdown['鑾婂閲忚兘']=ws
        if ws>3:
            tags.append("鐣板父鏀鹃噺")

        # 鏆存媺 / 鏆磋穼鍓嶇疆钃勫嫝绲愭
        pre_s, pre_tag = analyze_pre_breakout_setup(d15, d4h)
        score += pre_s; breakdown['钃勫嫝绲愭'] = pre_s
        if pre_tag and '鐒℃槑椤? not in pre_tag and '涓嶈冻' not in pre_tag:
            tags.append(pre_tag)

        # 鎻愬墠绐佺牬闋愬垽锛堥伩鍏嶇獊鐮村緦鎵嶈拷锛?        bo_s, bo_tag, bo_meta = analyze_breakout_forecast(d15, d4h, regime_bias)
        score += bo_s; breakdown['绐佺牬闋愬垽'] = bo_s
        if bo_tag and '鐒℃彁鍓嶇獊鐮寸祼妲? not in bo_tag:
            tags.append(bo_tag)

        # FVG 鍥炶俯鍝佽唱锛堟甯稿洖韪╀笉鐣舵垚杩藉児锛?        fvg_rt_s, fvg_rt_tag, fvg_rt_meta = analyze_fvg_retest_quality(d15, d4h, regime_bias)
        score += fvg_rt_s; breakdown['FVG鍥炶俯鍝佽唱'] = fvg_rt_s
        if fvg_rt_tag and '鐒VG鍥炶俯' not in fvg_rt_tag:
            tags.append(fvg_rt_tag)

        # 鍋囩獊鐮?/ 鍋囪穼鐮撮亷婵?        fake_s, fake_tag, fake_meta = analyze_fake_breakout(d15, score)
        score += fake_s; breakdown['鍋囩獊鐮存烤缍?] = fake_s
        if fake_meta.get('fakeout'):
            tags.append(fake_tag)

        # 鏂拌仦
        raw_ns = STATE["news_score"]
        ns = round(max(min(raw_ns, 5), -5) / 5 * NEWS_WEIGHT)
        score += ns; breakdown['鏂拌仦鎯呯窉'] = ns

        # 鏅傛椤嶅鍒嗘暩
        sess_score = get_session_score()
        if sess_score != 0:
            score += sess_score
            breakdown['鏅傛鍒嗘暩'] = sess_score

        score=min(max(round(score,1),-100),100)

        # 杩芥疾娈鸿穼鎳茬桨锛氬凡闆㈠潎绶氬お閬犳檪鍏堥檷娆婏紝鍐嶇瓑鍥炶俯/鍙嶅綀閫?        ext_s, ext_tag = analyze_extension_risk(d15, score)
        if bool(fvg_rt_meta.get('is_chase_ok')) and ((score > 0 and ext_s < 0) or (score < 0 and ext_s > 0)):
            ext_s = int(round(ext_s * 0.35))
            ext_tag = str(ext_tag) + '|FVG姝ｅ父鍥炶俯鏀惧杩藉児鎳茬桨'
        if bool(bo_meta.get('ready')) and ((score > 0 and ext_s < 0) or (score < 0 and ext_s > 0)):
            ext_s = int(round(ext_s * 0.55))
            ext_tag = str(ext_tag) + '|鎻愬墠绐佺牬鏀惧杩藉児鎳茬桨'
        score += ext_s; breakdown['杩藉児棰ㄩ毆'] = ext_s
        if ext_s != 0:
            tags.append(ext_tag)

        # ===== ATR 鏀归€欒！锛歋L/TP 鏀圭敤 15m ATR =====
        atr15_s = ta.atr(d15['h'], d15['l'], d15['c'], length=14)
        atr15   = safe_last(atr15_s, curr * 0.01)

        atr4h_s = ta.atr(d4h['h'], d4h['l'], d4h['c'], length=14)
        atr4h   = safe_last(atr4h_s, curr * 0.02)

        # 姝ｅ紡鎷?15m ATR 鐣?SL / TP 鍩烘簴
        atr = atr15

        active_keys=[k for k,v in breakdown.items() if v!=0]
        sl_mult,tp_mult=get_best_atr_params(active_keys)

        # 灞卞骞ｆ尝鍕曠巼鍕曟厠瑾挎暣
        try:
            vol_now  = float(d15['v'].tail(96).sum())
            vol_prev = float(d15['v'].tail(192).head(96).sum())
            vol_ratio = vol_now / max(vol_prev, 1e-9)

            if vol_ratio > 2.5:
                tp_mult = round(min(tp_mult * 1.4, 6.0), 2)
                sl_mult = round(max(sl_mult * 0.85, 1.2), 2)
                tags.append("閲忚兘鏆村{:.1f}x鎿碩P".format(vol_ratio))
            elif vol_ratio > 1.5:
                tp_mult = round(min(tp_mult * 1.15, 5.0), 2)
                tags.append("閲忚兘鏀惧ぇ{:.1f}x".format(vol_ratio))
            elif vol_ratio < 0.5:
                tp_mult = round(max(tp_mult * 0.85, 1.5), 2)
                sl_mult = round(min(sl_mult * 1.1, 3.0), 2)
                tags.append("绺噺鏀剁穵TP")

            if curr < 0.01:
                sl_mult = round(max(sl_mult * 1.3, 1.5), 2)
                tp_mult = round(min(tp_mult * 1.5, 7.0), 2)
            elif curr < 1.0:
                sl_mult = round(max(sl_mult * 1.15, 1.3), 2)
                tp_mult = round(min(tp_mult * 1.2, 5.0), 2)
        except:
            pass

        sl_mult = max(sl_mult, 1.5)
        tp_mult = max(tp_mult, 2.5)
        if tp_mult / max(sl_mult, 0.1) < 1.5:
            tp_mult = round(sl_mult * 1.5, 2)

        # 鐐掗爞鐐掑簳鍋垫脯
        rsi_val = safe_last(ta.rsi(d15['c'], length=14), 50)
        overbought  = rsi_val > 75
        oversold    = rsi_val < 25
        if overbought and score > 0:
            score = score * 0.7
            tp_mult = round(tp_mult * 0.8, 2)
            tags.append("鈿狅笍RSI瓒呰卜鐐掗爞棰ㄩ毆")
            breakdown['鐐掗爞璀﹀憡'] = -5
        elif oversold and score < 0:
            score = score * 0.7
            tp_mult = round(tp_mult * 0.8, 2)
            tags.append("鈿狅笍RSI瓒呰常鐐掑簳棰ㄩ毆")
            breakdown['鐐掑簳璀﹀憡'] = -5

        # ===== 绲愭 / 娉㈠嫊婵剧恫 =====
        # 鏂瑰悜涓€鑷翠絾閫插牬浣嶇疆宸檪锛岄伩鍏嶅彧鏈夊垎鏁搁珮灏辩‖涓?        if score > 0 and regime_bias < 0:
            score *= 0.65
            breakdown['鏂瑰悜琛濈獊'] = -8
            tags.append('澶氬垎鏁镐絾鏂瑰悜琛濈獊')
        elif score < 0 and regime_bias > 0:
            score *= 0.65
            breakdown['鏂瑰悜琛濈獊'] = 8
            tags.append('绌哄垎鏁镐絾鏂瑰悜琛濈獊')

        # 绲变竴 TP/SL 鎺у埗锛氬厛瀛?RR锛屽啀鐢ㄥ悓涓€濂楃洰妯欏叕寮忚▓绠?        learned_rr = get_learned_rr_target(symbol, 'neutral', breakdown.get('Setup', ''), active_keys, sl_mult, tp_mult)
        if bool(fvg_rt_meta.get('is_chase_ok')):
            learned_rr = min(max(learned_rr + 0.10, 1.25), 3.6)
        if bool(bo_meta.get('ready')) and not bool(fake_meta.get('fakeout')):
            learned_rr = min(max(learned_rr + 0.15, 1.35), 3.8)
        side_label = 'long' if score > 0 else 'short'
        sl, tp, rr_ratio = _calc_unified_targets(curr, atr, sl_mult, learned_rr, side_label)

        # 鉁?闃插憜锛堜笉褰遍熆绛栫暐锛?        if 'tp' not in locals() or tp is None:
            return 0, '閷:no_tp', 0, 0, 0, 0, {'valid': False, 'reason': 'no_tp_sl'}, 0, 0, 0, 2.0, 3.0

        if 'sl' not in locals() or sl is None:
            return 0, '閷:no_sl', 0, 0, 0, 0, {'valid': False, 'reason': 'no_tp_sl'}, 0, 0, 0, 2.0, 3.0

        # RR / 閫插牬鍝佽唱鏀规垚 AI 杓斿姪鐗瑰镜锛屼笉鍐嶇洿鎺ョ暥纭€ч€插牬闁€妾?        if rr_ratio < 1.10:
            score *= 0.90
            breakdown['棰ㄥ牨姣斿亸浣?杓斿姪)'] = -2 if score > 0 else 2
            tags.append('棰ㄥ牨姣斿亸浣?杓斿姪)')
        elif rr_ratio >= 1.8:
            breakdown['棰ㄥ牨姣斿劒绉€(杓斿姪)'] = 2 if score > 0 else -2
            score += 2 if score > 0 else -2

        # 閫插牬鍝佽唱淇濈暀鐐?AI 鍙冭€冿紝涓嶅啀鐩存帴鍗℃瑷婅櫉
        if abs(entry_s0) <= 0:
            score *= 0.90
            breakdown['閫插牬鍝佽唱鍋忓急(杓斿姪)'] = -2 if score > 0 else 2
            tags.append('閫插牬鍝佽唱鍋忓急(杓斿姪)')

        atr_pct = atr / max(curr, 1e-9)
        if atr_pct > 0.045:
            score *= 0.75
            tags.append("楂樻尝鍕曢檷娆?)
            breakdown['楂樻尝鍕曢亷鐔?] = -4 if score > 0 else 4

        # 4H 涓昏定鍕㈠皪榻婏細閫?4H 瓒ㄥ嫝鏅傜洿鎺ラ檷娆婏紝閬垮厤楂樺垎閫嗗嫝纭笂
        ema21_4h = safe_last(ta.ema(d4h['c'], length=21), curr)
        ema55_4h = safe_last(ta.ema(d4h['c'], length=55), curr)
        if score > 0 and not (curr > ema21_4h > ema55_4h):
            score *= 0.7
            tags.append("閫?H瓒ㄥ嫝闄嶆瑠")
            breakdown['4H瓒ㄥ嫝涓嶉爢'] = -6
        elif score < 0 and not (curr < ema21_4h < ema55_4h):
            score *= 0.7
            tags.append("閫?H瓒ㄥ嫝闄嶆瑠")
            breakdown['4H瓒ㄥ嫝涓嶉爢'] = 6

        ep = round((atr * tp_mult) / curr * 100 * 20, 2)
        score = min(max(round(score, 1), -100), 100)
        breakdown['RR'] = round(rr_ratio, 2)
        breakdown['LearnedRR'] = round(learned_rr, 2)
        breakdown['RegimeBias'] = regime_bias * 2
        breakdown['EntryGate'] = entry_s0
        if bool(bo_meta.get('ready')):
            breakdown['Setup'] = '鎻愬墠绐佺牬闋愬垽' if score > 0 else '鎻愬墠璺岀牬闋愬垽'
        elif bool(fvg_rt_meta.get('is_pullback')):
            breakdown['Setup'] = 'FVG鍥炶俯鎵挎帴' if score > 0 else 'FVG鍙嶅綀鎵垮'

        del d15,d4h,d1d; gc.collect()
        return score,"|".join(tags),curr,sl,tp,ep,breakdown,atr,atr15,atr4h,sl_mult,tp_mult

    except Exception as e:
        import traceback
        print("analyze {} 澶辨晽: {} \n{}".format(symbol, e, traceback.format_exc()[-300:]))
        return 0,"閷:{}".format(str(e)[:40]),0,0,0,0,{},0,0,0,2.0,3.0

# =====================================================
# 鏂拌仦鍩疯绶?# =====================================================
NEWS_CACHE = bot_news_disabled.disabled_news_state()
NEWS_LOCK = threading.Lock()

def get_cached_news_score():
    with NEWS_LOCK:
        return dict(NEWS_CACHE)

def set_cached_news(score, sentiment, summary, latest_title):
    with NEWS_LOCK:
        NEWS_CACHE.update({
            "score": int(max(min(score, 5), -5)),
            "sentiment": sentiment or "宸插仠鐢?,
            "summary": summary or "",
            "latest_title": latest_title or "鏂拌仦绯荤当宸插仠鐢?,
            "updated_at": time.time(),
        })

def fetch_crypto_news():
    return bot_news_disabled.fetch_crypto_news()

def analyze_news_with_ai(news_list):
    return bot_news_disabled.analyze_news_with_ai(news_list)

def news_thread():
    bot_news_disabled.news_thread(update_state=update_state, set_cached_news=set_cached_news, sleep_sec=300)

# =====================================================
# 绉诲嫊姝㈢泩杩借工绯荤当
# =====================================================
# 瑷橀寗姣忓€嬪€変綅鐨勮拷韫ょ媭鎱?# { "BTC/USDT:USDT": {
#     "side": "long",
#     "entry_price": 70000,
#     "highest_price": 72000,   # 鍋氬鏅傜殑鏈€楂橀粸
#     "lowest_price":  68000,   # 鍋氱┖鏅傜殑鏈€浣庨粸
#     "trail_pct": 0.05,        # 鍥炴挙骞?瑙哥櫦骞冲€夛紙闋愯ō5%锛?#     "initial_sl": 69000,      # 鍒濆姝㈡悕鍍?#     "atr": 500,               # 闁嬪€夋檪鐨凙TR
# }}
TRAILING_STATE = {}
TRAILING_LOCK  = threading.Lock()
ORDER_LOCK     = threading.Lock()   # 闃叉鍚屾檪涓嬪绛嗗柈瓒呴亷7鍊嬫寔鍊?_ORDERED_THIS_SCAN = set()  # 鏈吉宸蹭笅鍠殑骞ｏ紙闃叉鍚岃吉閲嶈涓嬪柈锛?_ORDERED_LOCK = threading.Lock()

def detect_reversal(sym, side, current_price):
    """
    鍋垫脯瓒ㄥ嫝鍙嶈綁瑷婅櫉锛堢敤鏂煎垎鎵规鐩堢殑绶婃€ュ钩鍊夛級
    鍥炲偝 (鏄惁鍙嶈綁, 鍘熷洜)
    """
    try:
        ohlcv = exchange.fetch_ohlcv(sym, '15m', limit=20)
        df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
        c = df['c']; h = df['h']; l = df['l']; v = df['v']
        curr = float(c.iloc[-1])

        signals = []

        # 1. RSI 瓒呰卜/瓒呰常鑳岄洟
        rsi_s = ta.rsi(c, length=14)
        rsi = float(rsi_s.iloc[-1]) if not pd.isna(rsi_s.iloc[-1]) else 50
        if side == 'long' and rsi > 78:
            signals.append("RSI瓒呰卜{:.0f}".format(rsi))
        elif side == 'short' and rsi < 22:
            signals.append("RSI瓒呰常{:.0f}".format(rsi))

        # 2. 鎴愪氦閲忕暟甯告斁澶э紙鍙嶈綁瑷婅櫉锛?        vol_avg = float(v.tail(10).mean())
        vol_now = float(v.iloc[-1])
        if vol_now > vol_avg * 2.5:
            signals.append("閲忚兘鏆村{:.1f}x".format(vol_now/vol_avg))

        # 3. 寮峰姏鍙嶈綁K妫?        o_last = float(df['o'].iloc[-1])
        c_last = float(c.iloc[-1])
        h_last = float(h.iloc[-1])
        l_last = float(l.iloc[-1])
        body = abs(c_last - o_last)
        range_ = h_last - l_last
        if range_ > 0:
            if side == 'long':
                # 鍋氬鏅傚嚭鐝惧ぇ闄扮窔锛堝楂?60%锛?                if c_last < o_last and body / range_ > 0.6:
                    signals.append("寮峰姏闄扮窔鍙嶈綁")
                # 涓婂奖绶氶亷闀凤紙琚鍥烇級
                upper_shadow = h_last - max(c_last, o_last)
                if upper_shadow > body * 2:
                    signals.append("闀蜂笂褰辩窔澹撳洖")
            elif side == 'short':
                # 鍋氱┖鏅傚嚭鐝惧ぇ闄界窔锛堝楂?60%锛?                if c_last > o_last and body / range_ > 0.6:
                    signals.append("寮峰姏闄界窔鍙嶈綁")
                # 涓嬪奖绶氶亷闀凤紙琚拹璧凤級
                lower_shadow = min(c_last, o_last) - l_last
                if lower_shadow > body * 2:
                    signals.append("闀蜂笅褰辩窔鎾愯捣")

        # 4. 閫ｇ簩3鏍瑰弽鍚慘妫?        last3_c = c.iloc[-4:-1].values
        if side == 'long':
            if all(last3_c[i] < last3_c[i-1] for i in range(1,3)):
                signals.append("閫?鏍逛笅璺?)
        elif side == 'short':
            if all(last3_c[i] > last3_c[i-1] for i in range(1,3)):
                signals.append("閫?鏍逛笂婕?)

        # 闇€瑕?2 鍊嬩互涓婅▕铏熸墠纰鸿獚鍙嶈綁锛堥伩鍏嶅亣淇¤櫉锛?        if len(signals) >= 2:
            return True, "鍙嶈綁瑷婅櫉: " + "|".join(signals)
        return False, ""
    except Exception as e:
        return False, ""

def partial_close_position(sym, contracts, side, ratio, reason=""):
    """閮ㄥ垎骞冲€?""
    try:
        close_side = 'sell' if side == 'long' else 'buy'
        partial_qty = abs(contracts) * ratio
        partial_qty = exchange.amount_to_precision(sym, partial_qty)
        exchange.create_order(sym, 'market', close_side, partial_qty, params={
            'reduceOnly': True,
            'posSide':    side,
            'tdMode':     'cross',
        })
        print("馃摛 閮ㄥ垎骞冲€?{} {:.0f}% | {}".format(sym, ratio*100, reason))
        return True
    except Exception as e:
        print("閮ㄥ垎骞冲€夊け鏁?{}: {}".format(sym, e))
        return False

def update_trailing(sym, side, current_price, atr):
    """
    鍒嗘壒姝㈢泩 + 鍕曟厠姝㈡悕绯荤当
    鐩1锛?1.2ATR锛夆啋 骞冲€?5%锛屾鎼嶇Щ鑷充繚鏈?    鐩2锛?2.4ATR锛夆啋 鍐嶅钩鍊?5%锛屾鎼嶇Щ鑷?0.8ATR
    鐩3锛?4.2ATR锛夆啋 鍓╅閮ㄤ綅璺熻憲璧帮紝姝㈡悕鏄庨’鏀剁穵
    鍙嶈綁鍋垫脯        鈫?绔嬪嵆鍏ㄥ钩閹栧埄
    """
    with TRAILING_LOCK:
        if sym not in TRAILING_STATE:
            return False, "", 0

        ts      = TRAILING_STATE[sym]
        entry   = ts.get("entry_price", current_price)
        atr_val = ts.get("atr", current_price * 0.01)
        if atr_val <= 0: atr_val = current_price * 0.01

        partial_done = ts.get("partial_done", 0)  # 宸插畬鎴愬咕鎵规鐩?        bd = dict(ts.get("breakdown") or {})
        trend_prof = _trend_learning_profile(sym, regime=str(bd.get("Regime", "neutral") or "neutral"), setup=str(ts.get("setup_label") or bd.get("Setup", "") or ""))
        trend_stage = str(trend_prof.get("stage") or "learning")
        trend_ratio = float(trend_prof.get("intervene_ratio", 0.0) or 0.0)
        hold_bias = float(trend_prof.get("hold_bias", 0.0) or 0.0) * trend_ratio

        if side == "long":
            prev_high = ts.get("highest_price", entry)
            if current_price > prev_high:
                ts["highest_price"] = current_price
            highest    = ts.get("highest_price", current_price)
            profit_atr = (current_price - entry) / atr_val
            hint_tp = float(ts.get("dynamic_take_profit_hint", 0) or 0)

            # OpenAI 鍙彁渚涘缓璀板瀷鍕曟厠姝㈢泩浣嶏紱鍛戒腑寰屽彧鏀剁穵淇濊锛屼笉瑕嗚搵鍘熸湰绯荤当銆?            if hint_tp > entry and current_price >= hint_tp and not ts.get("dynamic_hint_armed"):
                ts["dynamic_hint_armed"] = True
                ts["trail_pct"] = min(float(ts.get("trail_pct", 0.05) or 0.05), 0.03)
                suggested_sl = max(entry, hint_tp - atr_val * 0.6)
                ts["initial_sl"] = max(float(ts.get("initial_sl", 0) or 0), min(current_price, suggested_sl))

            # 鈹€鈹€ 鍒嗘壒姝㈢泩 鈹€鈹€
            # 鐩1锛?1.5ATR 鈫?骞?0%锛屾鎼嶇Щ鍒颁繚鏈?            if profit_atr >= 1.2 and partial_done == 0:
                ts["partial_done"]  = 1
                ts["initial_sl"]    = max(ts.get("initial_sl", 0), entry)
                ts["trail_pct"]     = 0.05
                print("馃幆 鐩1閬旀垚 {} +{:.1f}ATR 鈫?骞?5%锛屾鎼嶇Щ淇濇湰".format(sym, profit_atr))
                return True, "鐩1骞冲€?5% +{:.1f}ATR".format(profit_atr), 0.25

            # 鐩2锛?2.4ATR 鈫?鍐嶅钩35%锛屾鎼嶇Щ鍒?0.8ATR
            elif profit_atr >= 2.4 and partial_done == 1:
                ts["partial_done"]  = 2
                ts["initial_sl"]    = max(ts.get("initial_sl", 0), entry + atr_val * 0.8)
                ts["trail_pct"]     = 0.04
                print("馃幆 鐩2閬旀垚 {} +{:.1f}ATR 鈫?鍐嶅钩35%锛屾鎼?0.8ATR".format(sym, profit_atr))
                return True, "鐩2骞冲€?5% +{:.1f}ATR".format(profit_atr), 0.35

            # 鐩3锛?4.2ATR 鈫?鍓╅璺熺穵
            elif profit_atr >= 4.2 and partial_done == 2:
                ts["partial_done"]  = 3
                ts["initial_sl"]    = max(ts.get("initial_sl", 0), current_price - atr_val * 1.2)
                ts["trail_pct"]     = 0.028
                print("馃幆 鐩3閬旀垚 {} +{:.1f}ATR 鈫?绶婄府绉诲嫊姝㈢泩".format(sym, profit_atr))

            # 鈹€鈹€ 姝㈡悕绉诲嫊锛堝彧鍗囦笉闄嶏級鈹€鈹€
            if profit_atr >= 4.2:
                new_sl = current_price - atr_val * 1.2
                ts["trail_pct"] = 0.028
            elif profit_atr >= 2.4:
                new_sl = entry + atr_val * 1.4
                ts["trail_pct"] = max(ts.get("trail_pct", 0.05) * 0.85, 0.03)
            elif profit_atr >= 1.2:
                new_sl = entry
            else:
                new_sl = ts.get("initial_sl", entry - atr_val * 2)

            if new_sl > ts.get("initial_sl", 0):
                ts["initial_sl"] = new_sl

            # 鈹€鈹€ 瑙哥櫦姊濅欢 鈹€鈹€
            trail_price = highest * (1 - ts.get("trail_pct", 0.05))
            current_sl  = ts.get("initial_sl", 0)

            # 寰炴渶楂橀粸鍥炴挙瑙哥櫦鍏ㄥ钩
            if current_price < trail_price and partial_done >= 1:
                pullback_atr = (highest - current_price) / max(atr_val, 1e-9)
                if hold_bias > 0 and trend_stage in ('semi', 'full') and profit_atr >= 1.2:
                    allow_pullback = 0.95 + hold_bias * 0.90
                    if partial_done >= 2:
                        allow_pullback += 0.20
                    if pullback_atr <= allow_pullback:
                        ts["hold_bias_active"] = round(hold_bias, 4)
                        ts["trail_pct"] = min(max(ts.get("trail_pct", 0.05) * (1.0 + hold_bias * 0.35), 0.032), 0.095)
                        return False, "", 0
                return True, "绉诲嫊姝㈢泩瑙哥櫦 宄?{:.6f} 鐝?{:.6f} 鍥炴挙{:.1f}%".format(
                    highest, current_price, (highest-current_price)/highest*100), 1.0

            # 璺岀牬姝㈡悕绶?            if current_sl > 0 and current_price < current_sl:
                sl_type = "淇濇湰姝㈡悕" if abs(current_sl-entry)<atr_val*0.1 else "绉诲嫊姝㈡悕"
                return True, "{} @{:.6f}".format(sl_type, current_price), 1.0

        elif side == "short":
            prev_low = ts.get("lowest_price", entry)
            if current_price < prev_low:
                ts["lowest_price"] = current_price
            lowest     = ts.get("lowest_price", current_price)
            profit_atr = (entry - current_price) / atr_val
            hint_tp = float(ts.get("dynamic_take_profit_hint", 0) or 0)

            # OpenAI 鍙彁渚涘缓璀板瀷鍕曟厠姝㈢泩浣嶏紱鍛戒腑寰屽彧鏀剁穵淇濊锛屼笉瑕嗚搵鍘熸湰绯荤当銆?            if 0 < hint_tp < entry and current_price <= hint_tp and not ts.get("dynamic_hint_armed"):
                ts["dynamic_hint_armed"] = True
                ts["trail_pct"] = min(float(ts.get("trail_pct", 0.05) or 0.05), 0.03)
                suggested_sl = min(entry, hint_tp + atr_val * 0.6)
                ts["initial_sl"] = min(float(ts.get("initial_sl", entry * 9) or entry * 9), max(current_price, suggested_sl))

            if profit_atr >= 1.2 and partial_done == 0:
                ts["partial_done"] = 1
                ts["initial_sl"]   = min(ts.get("initial_sl", float('inf')), entry)
                ts["trail_pct"]    = 0.05
                return True, "鐩1骞冲€?5% +{:.1f}ATR".format(profit_atr), 0.25

            elif profit_atr >= 2.4 and partial_done == 1:
                ts["partial_done"] = 2
                ts["initial_sl"]   = min(ts.get("initial_sl", float('inf')), entry - atr_val * 0.8)
                ts["trail_pct"]    = 0.04
                return True, "鐩2骞冲€?5% +{:.1f}ATR".format(profit_atr), 0.35

            elif profit_atr >= 4.2 and partial_done == 2:
                ts["partial_done"] = 3
                ts["initial_sl"]   = min(ts.get("initial_sl", float('inf')), current_price + atr_val * 1.2)
                ts["trail_pct"]    = 0.028

            if profit_atr >= 4.2:
                new_sl = current_price + atr_val * 1.2
                ts["trail_pct"] = 0.028
            elif profit_atr >= 2.4:
                new_sl = entry - atr_val * 1.4
            elif profit_atr >= 1.2:
                new_sl = entry
            else:
                new_sl = ts.get("initial_sl", entry + atr_val * 2)

            if new_sl < ts.get("initial_sl", float('inf')):
                ts["initial_sl"] = new_sl

            trail_price = lowest * (1 + ts.get("trail_pct", 0.05))
            current_sl  = ts.get("initial_sl", float('inf'))

            if current_price > trail_price and partial_done >= 1:
                rebound_atr = (current_price - lowest) / max(atr_val, 1e-9)
                if hold_bias > 0 and trend_stage in ('semi', 'full') and profit_atr >= 1.2:
                    allow_rebound = 0.95 + hold_bias * 0.90
                    if partial_done >= 2:
                        allow_rebound += 0.20
                    if rebound_atr <= allow_rebound:
                        ts["hold_bias_active"] = round(hold_bias, 4)
                        ts["trail_pct"] = min(max(ts.get("trail_pct", 0.05) * (1.0 + hold_bias * 0.35), 0.032), 0.095)
                        return False, "", 0
                return True, "绉诲嫊姝㈢泩瑙哥櫦 璋?{:.6f} 鐝?{:.6f} 鍙嶅綀{:.1f}%".format(
                    lowest, current_price, (current_price-lowest)/lowest*100), 1.0

            if current_sl < float('inf') and current_price > current_sl:
                sl_type = "淇濇湰姝㈡悕" if abs(current_sl-entry)<atr_val*0.1 else "绉诲嫊姝㈡悕"
                return True, "{} @{:.6f}".format(sl_type, current_price), 1.0

        # 鈹€鈹€ 鏅傞枔姝㈡悕锛?5 鏍?15m K 浠嶆湭鏈夋晥璧板嚭锛屽氨闆㈠牬 鈹€鈹€
        open_ts = ts.get("entry_time_ts", 0)
        time_stop_sec = ts.get("time_stop_sec", TIME_STOP_BARS_15M * 15 * 60)
        if open_ts and time.time() - open_ts >= time_stop_sec:
            move_pct = abs(current_price - entry) / max(entry, 1e-9)
            if move_pct < max(atr_val / max(entry, 1e-9) * 1.2, 0.006):
                if not (hold_bias > 0 and trend_stage in ('semi', 'full')):
                    return True, "鏅傞枔姝㈡悕 {} 鍒嗛悩鏈劔闆㈡垚鏈崁".format(int(time_stop_sec/60)), 1.0

        return False, "", 0


def close_position(sym, contracts, side):
    """骞冲€夊柈涓€鍊変綅"""
    try:
        close_side = 'sell' if side == 'long' else 'buy'
        exchange.create_order(sym, 'market', close_side, abs(contracts),
                              params={'reduceOnly': True})
        touch_post_close_lock(sym)
        with TRAILING_LOCK:
            if sym in TRAILING_STATE:
                del TRAILING_STATE[sym]
        print("绉诲嫊姝㈢泩骞冲€夋垚鍔? {} {}鍙?| 鍟熺敤30鍒嗛悩鍐峰嵒".format(sym, contracts))
        return True
    except Exception as e:
        print("绉诲嫊姝㈢泩骞冲€夊け鏁?{}: {}".format(sym, e))
        return False

def trailing_stop_thread():
    """鐛ㄧ珛鍩疯绶掞紝姣?绉掕拷韫ゆ墍鏈夋寔鍊?""
    print("绉诲嫊姝㈢泩鍩疯绶掑暉鍕?)
    while True:
        try:
            with STATE_LOCK:
                active = list(STATE["active_positions"])

            for pos in active:
                sym       = pos['symbol']
                side      = (pos.get('side') or '').lower()
                contracts = float(pos.get('contracts', 0) or 0)
                if abs(contracts) == 0:
                    continue

                # 鎶撳嵆鏅傚児鏍硷紙鍔?timeout 淇濊锛?                try:
                    ticker = exchange.fetch_ticker(sym)
                    curr   = float(ticker['last'])
                    time.sleep(0.2)  # 閬垮厤 API 闄愰€?                except:
                    continue

                # 鑻ラ€欏€嬪€変綅閭勬矑鍦ㄨ拷韫わ紝鍔犲叆锛堜笉鎶揔绶氾紝鐢ㄩ€插牬鍍圭洿鎺ヤ及绠楋級
                # 鏈夋柊鎸佸€夋檪锛屾竻闄よ┎骞ｇ殑 FVG 鎺涘柈瑷橀寗锛堝凡鎴愪氦鎴栧凡鎵嬪嫊涓嬪柈锛?                with FVG_LOCK:
                    if sym in FVG_ORDERS:
                        print("鉁?{} 宸叉湁鎸佸€夛紝娓呴櫎 FVG 鎺涘柈瑷橀寗".format(sym))
                        FVG_ORDERS.pop(sym, None)
                        update_state(fvg_orders=dict(FVG_ORDERS))

                with TRAILING_LOCK:
                    if sym not in TRAILING_STATE:
                        entry = float(pos.get('entryPrice', curr) or curr)
                        atr = float(SIGNAL_META_CACHE.get(sym, {}).get('atr15', 0) or SIGNAL_META_CACHE.get(sym, {}).get('atr', 0) or 0)
                        if atr <= 0:
                            atr = fetch_real_atr(sym, '15m', 60) or entry * 0.008
                        initial_sl = entry - atr * 2 if side == 'long' else entry + atr * 2
                        trail_pct  = 0.05  # 闋愯ō5%鍥炴挙

                        TRAILING_STATE[sym] = {
                            "side":          side,
                            "entry_price":   entry,
                            "highest_price": curr if side == 'long' else float('inf'),
                            "lowest_price":  curr if side == 'short' else float('inf'),
                            "trail_pct":     trail_pct,
                            "initial_sl":    initial_sl,
                            "atr":           atr,
                            "entry_time_ts": time.time(),
                            "time_stop_sec": TIME_STOP_BARS_15M * 15 * 60,
                            "partial_done": 0,
                        }
                        print("闁嬪杩借工 {} {} 鍥炴挙:{:.1f}% 姝㈡悕:{:.6f}".format(
                            sym, side, trail_pct*100, initial_sl))

                # 妾㈡煡鏄惁瑙哥櫦
                should_close, reason, close_ratio = update_trailing(sym, side, curr, 0)
                if should_close:
                    if 0 < close_ratio < 1.0:
                        # 鍒嗘壒姝㈢泩锛堥儴鍒嗗钩鍊夛級
                        print("馃幆 鍒嗘壒姝㈢泩 {} {:.0f}% | {}".format(sym, close_ratio*100, reason))
                        partial_close_position(sym, contracts, side, close_ratio, reason)
                        # 鏇存柊鎸佸€夋暩閲忥紙瀵﹂殯鏈冨湪涓嬫 position_thread 鏇存柊锛?                    else:
                        # 鍏ㄥ钩
                        print("馃摛 鍏ㄩ儴骞冲€?{} | {}".format(sym, reason))
                        close_position(sym, contracts, side)

                # 鍙嶈綁鍋垫脯锛堟湁鏈鐝惧埄娼ゆ墠鍋垫脯锛岄伩鍏嶆氮璨籄PI锛?                elif side in ('long', 'short'):
                    with TRAILING_LOCK:
                        ts_now = TRAILING_STATE.get(sym, {})
                        entry_p = ts_now.get("entry_price", curr)
                        profit_pct = (curr - entry_p)/entry_p if side=='long' else (entry_p - curr)/entry_p
                    if profit_pct > 0.01:  # 鏈夎秴閬?%鍒╂饯鎵嶅伒娓弽杞?                        is_reversal, rev_reason = detect_reversal(sym, side, curr)
                        if is_reversal:
                            print("鈿?鍙嶈綁瑷婅櫉锛亄} {} 鈫?绔嬪嵆骞冲€夐帠鍒?| {}".format(sym, side, rev_reason))
                            close_position(sym, contracts, side)
                    # 瑷橀寗鍒颁氦鏄撴鍙?                    close_rec = {
                        "symbol":      sym,
                        "side":        "绉诲嫊姝㈢泩骞冲€?,
                        "score":       0,
                        "price":       curr,
                        "stop_loss":   0,
                        "take_profit": 0,
                        "est_pnl":     0,
                        "order_usdt":  0,
                        "time":        tw_now_str(),
                        "reason":      reason,
                    }
                    with STATE_LOCK:
                        STATE["trade_history"].insert(0, close_rec)
                    persist_trade_history_record(close_rec)

            # 鏇存柊杩借工鐙€鎱嬪埌 UI
            with TRAILING_LOCK:
                ui_info = {}
                for s, ts in TRAILING_STATE.items():
                    side_t = ts.get('side','')
                    highest = ts.get('highest_price', 0)
                    lowest  = ts.get('lowest_price', float('inf'))
                    trail   = ts.get('trail_pct', 0.05)
                    if side_t == 'long' and highest != float('inf'):
                        trail_price = highest * (1 - trail)
                        ui_info[s] = {
                            "side": "鍋氬",
                            "peak": round(highest, 6),
                            "trail_price": round(trail_price, 6),
                            "trail_pct": round(trail * 100, 1),
                        }
                    elif side_t == 'short' and lowest != float('inf'):
                        trail_price = lowest * (1 + trail)
                        ui_info[s] = {
                            "side": "鍋氱┖",
                            "peak": round(lowest, 6),
                            "trail_price": round(trail_price, 6),
                            "trail_pct": round(trail * 100, 1),
                        }
            update_state(trailing_info=ui_info)
        except Exception as e:
            import traceback
            print("绉诲嫊姝㈢泩鐣板父: {}".format(e))
            print(traceback.format_exc())
        time.sleep(10)  # 姣?0绉掕拷韫や竴娆?
# =====================================================
# 鎸佸€?+ 甯虫埗璩囩敘鐩ｆ帶锛堟瘡 10 绉掞級
# =====================================================
PREV_POSITION_SYMS = set()

def position_thread():
    global PREV_POSITION_SYMS
    while True:
        try:
            # 鎶撳赋鎴剁附璩囩敘
            try:
                bal=exchange.fetch_balance()
                equity=float(bal.get('USDT',{}).get('total',0) or
                             bal.get('total',{}).get('USDT',0) or 0)
                update_state(equity=round(equity,4))
            except:
                pass

            raw=exchange.fetch_positions()
            active=[p for p in raw if abs(float(p.get('contracts',0) or 0))>0]
            pnl=sum(float(p.get('unrealizedPnl',0) or 0) for p in active)
            update_state(
                active_positions=active,
                total_pnl=round(pnl,4),
                last_update=tw_now_str()
            )

            curr_syms={p['symbol'] for p in active}
            closed_syms=PREV_POSITION_SYMS-curr_syms
            for sym in closed_syms:
                if not queue_learn_for_closed_symbol(sym, curr_syms):
                    print("璀﹀憡: {} 鐒″缈掔磤閷勶紙鍙兘鏄墜鍕曚笅鍠級".format(sym))

            # 瑁滃劅姗熷埗锛氶伩鍏嶄氦鏄撴墍 TP/SL 宸叉垚浜わ紝浣嗗洜閲嶅暉/婕忚吉瑭㈡矑琚閷?            with LEARN_LOCK:
                open_symbols = list({t.get('symbol') for t in LEARN_DB.get('trades', []) if t.get('result') == 'open' and t.get('symbol')})
            for sym in open_symbols:
                if sym not in curr_syms:
                    queue_learn_for_closed_symbol(sym, curr_syms)

            PREV_POSITION_SYMS=curr_syms
            # 姣忚吉鍌欎唤鐙€鎱?            save_full_state()
            save_risk_state()
        except Exception as e:
            print("鎸佸€夋洿鏂板け鏁? {}".format(e))
        time.sleep(10)

# =====================================================
# 瀛哥繏绯荤当锛氬钩鍊夊緦鍒嗘瀽
# =====================================================
def learn_from_closed_trade_legacy_shadow_1(trade_id):
    with LEARN_LOCK:
        trade = next((t for t in LEARN_DB["trades"] if t["id"] == trade_id), None)
    if not trade or trade["result"] != "open":
        PENDING_LEARN_IDS.discard(trade_id)
        return
    time.sleep(5)
    try:
        sym = trade["symbol"]
        side = trade["side"]
        exit_p = float(trade.get("exit_price", 0) or 0)
        entry_p = float(trade.get("entry_price", 0) or 0)
        leverage = float(trade.get("leverage", trade.get("planned_leverage", 1)) or 1)
        margin_pct = float(trade.get("margin_pct", RISK_PCT) or RISK_PCT)
        realized_pnl_usdt = float(trade.get("realized_pnl_usdt", 0) or 0)
        used_margin_usdt = float(trade.get("used_margin_usdt", trade.get("order_usdt", 0)) or 0)
        entry_equity = float(trade.get("entry_equity", STATE.get("equity", 0)) or 0)

        # 1) 绱斿児鏍奸倞闅涳紙涓嶅惈妲撴】锛?        raw_pct = ((exit_p - entry_p) / max(entry_p, 1e-9) * 100.0) if side == "buy" else ((entry_p - exit_p) / max(entry_p, 1e-9) * 100.0)

        # 2) 浜ゆ槗鎵€鏈€绲傚凡瀵︾従鎼嶇泭锛堝劒鍏堬級鈫?杞夋垚淇濊瓑閲?ROI锛岄伩鍏嶅缈掑€肩附鏄帴杩?0
        leveraged_pnl_pct = None
        if abs(realized_pnl_usdt) > 1e-12 and used_margin_usdt > 1e-9:
            leveraged_pnl_pct = (realized_pnl_usdt / used_margin_usdt) * 100.0
        if leveraged_pnl_pct is None:
            leveraged_pnl_pct = raw_pct * max(leverage, 1.0)

        # 3) 甯虫埗瑕栬鎼嶇泭锛堝闅涘凡瀵︾従鎼嶇泭 / 閫插牬鏅傝硣鐢級锛涙嬁涓嶅埌鏅傛墠閫€鍥炶垔浼扮畻
        account_pnl_pct = None
        if abs(realized_pnl_usdt) > 1e-12 and entry_equity > 1e-9:
            account_pnl_pct = (realized_pnl_usdt / entry_equity) * 100.0
        if account_pnl_pct is None:
            account_pnl_pct = leveraged_pnl_pct * max(margin_pct, 0.0001)

        # 绲?AI 瀛哥繏鐨勪富鍙ｅ緫锛氬劒鍏堜娇鐢ㄤ氦鏄撴墍鏈€绲傚凡瀵︾従鎼嶇泭鎻涚畻寰岀殑鐪熷绲愭灉
        learn_pnl_pct = leveraged_pnl_pct
        result = "win" if learn_pnl_pct > 0 else "loss"

        time.sleep(60)
        post_ohlcv = exchange.fetch_ohlcv(sym, '15m', limit=12)
        post_closes = [c[4] for c in post_ohlcv[-10:]]
        future_max = max(post_closes); future_min = min(post_closes)
        missed_pct = (future_max - exit_p) / max(exit_p, 1e-9) * 100 * max(leverage, 1.0) if side == "buy" else (exit_p - future_min) / max(exit_p, 1e-9) * 100 * max(leverage, 1.0)
        post_profile = _trade_post_move_profile({
            'side': side, 'exit_price': exit_p, 'post_candles': post_closes,
            'leverage': leverage, 'learn_pnl_pct': learn_pnl_pct,
        })
        exit_type = classify_exit_type({**trade, 'learn_pnl_pct': learn_pnl_pct}, post_profile)
        exec_bucket = execution_quality_bucket(trade.get('execution_snapshot') or trade.get('execution_quality'))

        bd = trade.get("breakdown", {})
        active_keys = [k for k, v in bd.items() if v != 0]
        pkey = "|".join(sorted(active_keys))

        with LEARN_LOCK:
            db = LEARN_DB
            for t in db["trades"]:
                if t["id"] == trade_id:
                    t["result"] = result
                    t["edge_pct"] = round(raw_pct, 4)
                    t["pnl_pct"] = round(raw_pct, 4)  # legacy鍏煎锛氫繚鐣欑磾鍍规牸閭婇殯
                    t["leveraged_pnl_pct"] = round(leveraged_pnl_pct, 4)
                    t["account_pnl_pct"] = round(account_pnl_pct, 4)
                    t["learn_pnl_pct"] = round(learn_pnl_pct, 4)
                    t["post_candles"] = post_closes
                    t["missed_move_pct"] = round(missed_pct, 2)
                    t["post_run_pct"] = round(float(post_profile.get('run_pct', 0) or 0), 4)
                    t["post_pullback_pct"] = round(float(post_profile.get('pullback_pct', 0) or 0), 4)
                    t["trend_continuation"] = bool(post_profile.get('continuation'))
                    t["trend_reason"] = str(post_profile.get('reason') or '')
                    t["exit_type"] = exit_type
                    t["execution_quality_bucket"] = exec_bucket
                    t["decision_fingerprint"] = build_decision_fingerprint(t)
                    t["exit_time"] = tw_now_str("%Y-%m-%d %H:%M:%S")
                    enriched = enrich_learning_trade(t, reset_from=TREND_LEARNING_RESET_FROM)
                    t.update(enriched)
                    break

            metric = float(learn_pnl_pct)

            # 鏇存柊鎸囨绲勫悎绲辫▓锛堢敤 learn_pnl_pct锛屼笉鍐嶇敤琚绺殑 raw pct锛?            if pkey not in db["pattern_stats"]:
                db["pattern_stats"][pkey] = {
                    "win": 0, "loss": 0, "sample_count": 0, "total_pnl": 0.0,
                    "avg_pnl": 0.0, "best_sl": trade.get("atr_mult_sl", 2.0),
                    "best_tp": trade.get("atr_mult_tp", 3.0), "tp_candidates": [], "sl_candidates": []
                }
            ps = db["pattern_stats"][pkey]
            ps["sample_count"] += 1
            ps["total_pnl"] += metric
            ps["avg_pnl"] = round(ps["total_pnl"] / max(ps["sample_count"], 1), 4)
            if result == "win":
                ps["win"] += 1; ps["tp_candidates"].append(trade.get("atr_mult_tp", 3.0))
            else:
                ps["loss"] += 1; ps["sl_candidates"].append(trade.get("atr_mult_sl", 2.0))
            if ps["sample_count"] >= AI_MIN_SAMPLE_EFFECT:
                wr = ps["win"] / max(ps["sample_count"], 1)
                if wr >= 0.6 and ps["tp_candidates"]:
                    ps["best_tp"] = round(min(max(ps["tp_candidates"]) * 1.1, 5.0), 2)
                    ps["best_sl"] = round(max(ps.get("best_sl", 2.0) * 0.95, 1.8), 2)
                elif wr < 0.4:
                    ps["best_sl"] = round(min(ps.get("best_sl", 2.0) * 0.85, 1.8), 2)
                    ps["best_tp"] = round(max(ps.get("best_tp", 3.5) * 0.9, 2.8), 2)

            # 鏇存柊骞ｇó绲辫▓
            ss = db.setdefault("symbol_stats", {})
            if sym not in ss:
                ss[sym] = {"win": 0, "loss": 0, "count": 0, "total_pnl": 0.0, "total_margin_pct": 0.0}
            ss[sym]["count"] += 1
            ss[sym]["total_pnl"] += metric
            ss[sym]["total_margin_pct"] += margin_pct
            if result == "win":
                ss[sym]["win"] += 1
            else:
                ss[sym]["loss"] += 1

            # 鍏ㄥ煙绲辫▓锛堝彧鐪?live锛?            all_closed = [t for t in db["trades"] if _is_live_source(t.get("source")) and t["result"] in ("win", "loss")]
            if all_closed:
                db["total_trades"] = len(all_closed)
                wins = sum(1 for t in all_closed if t["result"] == "win")
                db["win_rate"] = round(wins / len(all_closed) * 100, 1)
                db["avg_pnl"] = round(sum(_trade_learn_metric(t) for t in all_closed) / len(all_closed), 4)
                recent = all_closed[-20:]
                if len(recent) >= 10:
                    rwr = sum(1 for t in recent if t["result"] == "win") / len(recent)
                    if rwr >= 0.65:
                        db["atr_params"]["default_tp"] = round(min(db["atr_params"]["default_tp"] * 1.05, 5.0), 2)
                    elif rwr < 0.35:
                        db["atr_params"]["default_sl"] = round(max(db["atr_params"]["default_sl"] * 0.92, 1.2), 2)
                        db["atr_params"]["default_tp"] = round(max(db["atr_params"]["default_tp"] * 0.95, 1.5), 2)

            all_closed_count = len([t for t in db["trades"] if _is_live_source(t.get("source")) and t["result"] in ("win", "loss")])
            if all_closed_count >= 50 and all_closed_count % 10 == 0:
                _auto_adjust_weights(db)

            regime = str((trade.get('breakdown') or {}).get('Regime', 'neutral') or 'neutral')
            srs = db.setdefault('symbol_regime_stats', {})
            rk = f"{sym}|{regime}"
            if rk not in srs:
                srs[rk] = {'count': 0, 'win': 0, 'loss': 0, 'pnl_sum': 0.0, 'last_update': '--'}
            srs[rk]['count'] += 1
            srs[rk]['pnl_sum'] += metric
            srs[rk]['last_update'] = tw_now_str('%Y-%m-%d %H:%M:%S')
            if result == 'win':
                srs[rk]['win'] += 1
            else:
                srs[rk]['loss'] += 1

            market_state = _market_state_from_trade(trade)
            mss = db.setdefault('market_state_stats', {})
            ms = mss.setdefault(market_state, {'count': 0, 'win': 0, 'loss': 0, 'pnl_sum': 0.0, 'last_update': '--'})
            ms['count'] += 1
            ms['pnl_sum'] += metric
            ms['last_update'] = tw_now_str('%Y-%m-%d %H:%M:%S')
            if result == 'win':
                ms['win'] += 1
            else:
                ms['loss'] += 1
            smss = db.setdefault('symbol_market_state_stats', {})
            smk = f"{sym}|{market_state}"
            sms = smss.setdefault(smk, {'count': 0, 'win': 0, 'loss': 0, 'pnl_sum': 0.0, 'last_update': '--'})
            sms['count'] += 1
            sms['pnl_sum'] += metric
            sms['last_update'] = tw_now_str('%Y-%m-%d %H:%M:%S')
            if result == 'win':
                sms['win'] += 1
            else:
                sms['loss'] += 1

            save_learn_db(db)

        try:
            with AI_LOCK:
                AI_PANEL['last_learning'] = tw_now_str('%Y-%m-%d %H:%M:%S')
        except Exception:
            pass

        # 棰ㄦ帶鐢?USDT 鐩堣櫑锛氬劒鍏堜娇鐢ㄤ氦鏄撴墍鏈€绲傚凡瀵︾従鎼嶇泭锛屽惁鍓囨墠閫€鍥炰及绠?        pnl_usdt = float(trade.get("realized_pnl_usdt", 0) or 0)
        if abs(pnl_usdt) <= 1e-12:
            base_usdt = float(trade.get("used_margin_usdt", trade.get("order_usdt", 0)) or 0)
            if base_usdt <= 1e-9:
                base_usdt = float(STATE.get("equity", 10) or 10)
            pnl_usdt = (learn_pnl_pct / 100.0) * base_usdt
        record_trade_result(pnl_usdt)
        update_state(risk_status=get_risk_status())
        _refresh_learn_summary()
        print("鉁?瀛哥繏瀹屾垚 {} | edge:{:.4f}% | lev:{:.2f}% | acct:{:.4f}% | {}".format(sym, raw_pct, leveraged_pnl_pct, learn_pnl_pct, result))
        PENDING_LEARN_IDS.discard(trade_id)
    except Exception as e:
        PENDING_LEARN_IDS.discard(trade_id)
        print("瀛哥繏澶辨晽: {}".format(e))


LEARNING_QUEUE = LearningTaskQueue(learn_from_closed_trade_legacy_shadow_1, name='learning-queue')


def _enqueue_closed_trade_learning(trade_id):
    try:
        size = LEARNING_QUEUE.enqueue(trade_id)
        append_audit_log('ai', 'learning_enqueued', {'trade_id': trade_id, 'queue_size': size})
        return size
    except Exception as e:
        print(f'瀛哥繏鎺掗殜澶辨晽: {e}')
        append_audit_log('ai', 'learning_enqueue_failed', {'trade_id': trade_id, 'error': str(e)})
        return 0


def learn_from_closed_trade(trade_id):
    return _enqueue_closed_trade_learning(trade_id)

def _auto_adjust_weights(db):
    """鑸婂浐瀹氭瑠閲嶄繚鐣欑偤鍩虹鐗瑰镜锛屼笉鍐嶇洿鎺ヨ钃?W锛涙敼杓稿嚭 AI 鑷富閭忚集鎻愮ず銆?""
    try:
        trades = [t for t in db["trades"] if t["result"] in ("win","loss") and t.get("breakdown")]
        if len(trades) < 30:
            return

        indicator_stats = {}
        for t in trades[-360:]:
            bd = dict(t.get("breakdown") or {})
            metric = float(_trade_learn_metric(t) or 0.0)
            edge = max(min(metric / 2.5, 1.0), -1.0)
            for key, val in bd.items():
                if isinstance(val, bool):
                    num = 1.0 if val else -1.0
                elif isinstance(val, (int, float)):
                    num = float(val)
                else:
                    continue
                rec = indicator_stats.setdefault(key, {"count":0,"signed_edge_sum":0.0,"value_abs_sum":0.0})
                rec["count"] += 1
                rec["signed_edge_sum"] += edge * (1.0 if num > 0 else -1.0 if num < 0 else 0.0)
                rec["value_abs_sum"] += min(abs(num), 12.0)

        adaptive_hints = {}
        contrib = {}
        for key, st in indicator_stats.items():
            count = int(st.get("count", 0) or 0)
            if count < 12:
                continue
            signed_edge = float(st.get("signed_edge_sum", 0.0) or 0.0) / max(count, 1)
            avg_abs = float(st.get("value_abs_sum", 0.0) or 0.0) / max(count, 1)
            confidence = min(count / 45.0, 1.0) * min(avg_abs / 3.0, 1.0)
            edge = signed_edge * (0.55 + confidence * 0.45)
            if abs(edge) < 0.015:
                continue
            adaptive_hints[key] = {
                'edge': round(edge, 6),
                'confidence': round(confidence, 6),
                'count': count,
                'avg_abs_value': round(avg_abs, 6),
            }
            contrib[key] = round(abs(edge) * (0.5 + confidence), 6)

        db["indicator_contrib"] = contrib
        db["adaptive_indicator_hints"] = adaptive_hints
        print("馃 AI閭忚集鎻愮ず宸叉洿鏂?鍥哄畾娆婇噸鍍呬繚鐣欑偤鍩虹鐗瑰镜)锛屾彁绀烘暩:", len(adaptive_hints))

    except Exception as e:
        print("娆婇噸瑾挎暣澶辨晽:", e)


def _refresh_learn_summary():
    live_closed = get_live_trades(closed_only=True)
    with LEARN_LOCK:
        db = LEARN_DB
        stats = {}
        sym_stats = {}
        for t in live_closed:
            bd = t.get("breakdown", {}) or {}
            active_keys = [k for k,v in bd.items() if v not in (0, None, "", False)]
            pkey = "|".join(sorted(active_keys))
            if pkey:
                st = stats.setdefault(pkey, {"win":0,"sample_count":0,"total_pnl":0.0})
                st["sample_count"] += 1
                st["total_pnl"] += _trade_learn_metric(t)
                if t.get("result") == "win":
                    st["win"] += 1
            sym = str(t.get("symbol") or "")
            if sym:
                ss = sym_stats.setdefault(sym, {"win":0,"count":0,"total_pnl":0.0})
                ss["count"] += 1
                ss["total_pnl"] += _trade_learn_metric(t)
                if t.get("result") == "win":
                    ss["win"] += 1

        ranked = sorted(stats.items(), key=lambda x: (x[1]["total_pnl"]/max(x[1]["sample_count"],1)), reverse=True)
        top3=[{"pattern":k[:45],
               "avg_pnl":round(v["total_pnl"]/max(v["sample_count"],1),2),
               "win_rate":round(v["win"]/max(v["sample_count"],1)*100,0),
               "count":v["sample_count"]} for k,v in ranked[:3]] if ranked else []
        worst3=[{"pattern":k[:45],
                 "avg_pnl":round(v["total_pnl"]/max(v["sample_count"],1),2),
                 "win_rate":round(v["win"]/max(v["sample_count"],1)*100,0),
                 "count":v["sample_count"]} for k,v in ranked[-3:]] if len(ranked)>=3 else []
        blocked=[{"symbol":s,
                  "win_rate":round(v["win"]/v["count"]*100,1),
                  "count":v["count"],
                  "avg_pnl":round(v["total_pnl"]/max(v["count"],1),2)}
                 for s,v in sym_stats.items()
                 if v["count"]>=8 and (v["win"]/v["count"]<0.4 or (v["total_pnl"]/max(v["count"],1))<0)]

        open_pnl_usdt = 0.0
        try:
            open_pnl_usdt = round(sum(float(p.get('unrealizedPnl',0) or 0) for p in STATE.get('active_positions', [])), 4)
        except Exception:
            pass

        total_trades = len(live_closed)
        wins = sum(1 for t in live_closed if t.get("result") == "win")
        avg_pnl = round(sum(_trade_learn_metric(t) for t in live_closed) / max(total_trades,1), 2) if total_trades else 0.0
        summary = {
            "total_trades": total_trades,
            "win_rate": round(wins / max(total_trades,1) * 100, 1) if total_trades else 0.0,
            "avg_pnl": avg_pnl,
            "open_pnl_usdt": open_pnl_usdt,
            "current_sl_mult": db["atr_params"]["default_sl"],
            "current_tp_mult": db["atr_params"]["default_tp"],
            "top_patterns": top3,
            "worst_patterns": worst3,
            "blocked_symbols": blocked,
            "data_scope": "live_only",
        }
    update_state(learn_summary=summary)


# =====================================================
# 涓嬪柈锛堜娇鐢ㄧ附璩囩敘 5% + 鏈€楂樻妗匡級
# =====================================================
def get_fvg_entry_price(symbol, side, current_price, atr):
    """
    瑷堢畻鏈€鍎€插牬鍍规牸锛?    1) 鍏堝仛杩藉児淇濊锛岄伩鍏嶇獊鐮村緦鏈€寰屼竴妫掓墠鍘昏拷
    2) 鍐嶆壘 FVG 缂哄彛鎺涘洖韪?鍙嶅綀鍠?    """
    try:
        pb_price, pb_note = get_breakout_pullback_entry(symbol, side, current_price, atr)
        if pb_price is not None:
            return pb_price, pb_note

        ohlcv = exchange.fetch_ohlcv(symbol, '15m', limit=60)
        df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
        hi = df['h'].tolist()
        lo = df['l'].tolist()
        n  = len(hi)

        best_fvg_price = None
        best_dist      = float('inf')

        for i in range(2, min(30, n)):
            idx = n - 1 - i
            if idx < 2: break

            k1_h = hi[idx-2]; k3_l = lo[idx]
            k1_l = lo[idx-2]; k3_h = hi[idx]

            if side == 'long' and k1_h < k3_l:
                # 鍋氬 FVG锛堝悜涓婄己鍙ｏ級锛氱瓑鍍规牸鍥炲埌缂哄彛闋傞儴
                fvg_top    = k3_l
                fvg_bottom = k1_h
                fvg_mid    = (fvg_top + fvg_bottom) / 2
                # 鏈～瑁滅⒑瑾?                filled = any(lo[j] <= fvg_top and hi[j] >= fvg_bottom
                             for j in range(idx+1, n))
                if not filled:
                    dist = abs(current_price - fvg_mid) / max(atr, 1e-9)
                    if dist < best_dist:
                        best_dist = dist
                        best_fvg_price = fvg_top  # 鍦ㄧ己鍙ｉ爞閮ㄦ帥鍠?
            elif side == 'short' and k1_l > k3_h:
                # 鍋氱┖ FVG锛堝悜涓嬬己鍙ｏ級锛氱瓑鍍规牸鍙嶅綀鍒扮己鍙ｅ簳閮?                fvg_top    = k1_l
                fvg_bottom = k3_h
                fvg_mid    = (fvg_top + fvg_bottom) / 2
                filled = any(hi[j] >= fvg_bottom and lo[j] <= fvg_top
                             for j in range(idx+1, n))
                if not filled:
                    dist = abs(current_price - fvg_mid) / max(atr, 1e-9)
                    if dist < best_dist:
                        best_dist = dist
                        best_fvg_price = fvg_bottom  # 鍦ㄧ己鍙ｅ簳閮ㄦ帥鍠?
        # 璺濋洟鍒ゆ柗
        if best_fvg_price is None:
            return None, "鐒VG缂哄彛锛岀洿鎺ュ競鍍?

        if best_dist > 2.0:
            return None, "FVG缂哄彛澶仩({:.1f}ATR)锛屼笉鍕夊挤".format(best_dist)

        if best_dist < 0.3:
            # 宸茬稉鍦ㄧ己鍙ｅ収锛岀洿鎺ュ競鍍?            return None, "宸插湪FVG缂哄彛鍏э紝甯傚児閫插牬"

        return round(best_fvg_price, 6), "FVG闄愬児{:.6f}锛堣窛闆:.1f}ATR锛?.format(
            best_fvg_price, best_dist)

    except Exception as e:
        return None, "FVG瑷堢畻澶辨晽"

def clamp(v, lo, hi):
    try:
        return max(lo, min(hi, v))
    except:
        return lo

def calc_dynamic_margin_pct(score, atr_ratio, trend_aligned, squeeze_ready, extended_risk, same_side_count, market_dir="涓€?, market_strength=0.0):
    """
    鏍规摎瑷婅櫉鍝佽唱/绲愭/娉㈠嫊姹哄畾鐣朵笅淇濊瓑閲戞瘮渚嬶紝闄愬埗 1% ~ 8%銆?    - 寮辫▕铏燂細1%~2%
    - 閬庨杸妾伙細3.5%~5.5%
    - 寮峰叡鎸細鏈€楂?8%
    """
    s = abs(float(score or 0))
    atr_ratio = float(atr_ratio or 0)

    if s < 48:
        base = 0.01
    elif s < 50:
        base = 0.02
    elif s < 52:
        base = 0.04
    elif s < 54:
        base = 0.055
    else:
        base = 0.07

    adj = 0.0
    if trend_aligned:
        adj += 0.005
    if squeeze_ready:
        adj += 0.005

    if market_strength >= 0.6:
        if (market_dir in ("寮峰", "澶?) and score > 0) or (market_dir in ("寮风┖", "绌?) and score < 0):
            adj += 0.005
        elif market_dir != "涓€?:
            adj -= 0.01

    if extended_risk:
        adj -= 0.015
    if atr_ratio > 0.045:
        adj -= 0.01
    elif atr_ratio > 0.03:
        adj -= 0.005

    if same_side_count >= 4:
        adj -= 0.01
    elif same_side_count >= 2:
        adj -= 0.005

    return round(clamp(base + adj, MIN_MARGIN_PCT, MAX_MARGIN_PCT), 4)

def infer_margin_context(sig, same_side_count=0):
    try:
        desc = str(sig.get('desc', '') or '')
        breakdown = sig.get('breakdown', {}) or {}
        price = max(float(sig.get('price', 0) or 0), 1e-9)
        atr_val = float(sig.get('atr15') or sig.get('atr') or 0)
        atr_ratio = atr_val / price if price > 0 else 0.0

        trend_penalty = breakdown.get('4H瓒ㄥ嫝涓嶉爢', 0)
        trend_aligned = (trend_penalty == 0) and ('閫?H瓒ㄥ嫝闄嶆瑠' not in desc)
        squeeze_ready = any(k in desc for k in ['鏀舵杺', '鍚告敹涓?, '閲忚兘鎮勬倓鏀惧ぇ'])
        extended_risk = any(k in desc for k in ['閬庡害寤朵几', '閬垮厤杩介珮', '閬垮厤杩界┖', '楂樻尝鍕曢檷娆?, '鐐掗爞棰ㄩ毆', '鐐掑簳棰ㄩ毆'])

        with MARKET_LOCK:
            market_dir = MARKET_STATE.get('direction', '涓€?)
            market_strength = float(MARKET_STATE.get('strength', 0) or 0)

        margin_pct = calc_dynamic_margin_pct(
            score=sig.get('score', 0),
            atr_ratio=atr_ratio,
            trend_aligned=trend_aligned,
            squeeze_ready=squeeze_ready,
            extended_risk=extended_risk,
            same_side_count=same_side_count,
            market_dir=market_dir,
            market_strength=market_strength,
        )
        regime = str(breakdown.get('Regime', 'neutral') or 'neutral')
        setup = str(sig.get('setup_label') or breakdown.get('Setup', '') or '')
        learning_mult = get_margin_learning_multiplier(sig.get('symbol', ''), sig.get('score', 0), breakdown)
        strategy_mult, strategy_note = _strategy_margin_multiplier(sig.get('symbol', ''), regime, setup)
        ai_risk_mult, ai_risk_note = _ai_risk_multiplier(sig.get('symbol', ''), regime, setup, sig.get('score', 0), breakdown)
        profile = _ai_strategy_profile(sig.get('symbol', ''), regime=regime, setup=setup)
        conf_size_mult, conf_size_note = confidence_position_multiplier(float(profile.get('confidence', 0) or 0), str(breakdown.get('MarketTempo', 'normal') or 'normal'))
        exec_snapshot = _execution_quality_state(sig)
        calibrator = calibrate_trade_decision(
            score=abs(float(sig.get('score', 0) or 0)),
            threshold=float(ORDER_THRESHOLD_DEFAULT),
            rr_ratio=float(sig.get('rr_ratio', 0) or 0),
            entry_quality=float(sig.get('entry_quality', 0) or 0),
            regime_confidence=float(sig.get('regime_confidence', 0) or 0),
            profile=profile,
            execution_quality=exec_snapshot,
            market_consensus=dict(LAST_MARKET_CONSENSUS or {}),
        )
        signal_advantage_mult = 0.88 + max(0.0, float(calibrator.get('confidence_calibrated', 0.0) or 0.0) - 0.5) * 1.2
        execution_quality_mult = 0.78 + float(exec_snapshot.get('execution_score', 0.65) or 0.65) * 0.45
        market_state_discount = 1.0
        if regime in ('neutral', 'neutral_range'):
            market_state_discount *= 0.9
        if str(breakdown.get('MarketTempo', 'normal') or 'normal') == 'fast':
            market_state_discount *= 0.95
        layered = apply_position_formula(
            base_margin_pct=margin_pct * learning_mult * strategy_mult * ai_risk_mult * conf_size_mult,
            signal_advantage=signal_advantage_mult,
            execution_quality_mult=execution_quality_mult,
            market_state_discount=market_state_discount,
            min_margin_pct=MIN_MARGIN_PCT,
            max_margin_pct=MAX_MARGIN_PCT,
        )
        margin_pct = layered['margin_pct']
        return {
            'margin_pct': margin_pct,
            'learning_mult': learning_mult,
            'strategy_mult': strategy_mult,
            'ai_risk_mult': ai_risk_mult,
            'strategy_note': strategy_note,
            'ai_risk_note': ai_risk_note,
            'confidence_size_mult': conf_size_mult,
            'confidence_size_note': conf_size_note,
            'signal_advantage_mult': layered.get('signal_advantage_mult', 1.0),
            'execution_quality_mult': layered.get('execution_quality_mult', 1.0),
            'market_state_discount': layered.get('market_state_discount', 1.0),
            'decision_calibrator': calibrator,
            'atr_ratio': round(atr_ratio, 5),
            'trend_aligned': trend_aligned,
            'squeeze_ready': squeeze_ready,
            'extended_risk': extended_risk,
            'market_dir': market_dir,
            'market_strength': round(market_strength, 3),
        }
    except Exception as e:
        return {
            'margin_pct': RISK_PCT,
            'atr_ratio': 0.0,
            'trend_aligned': False,
            'squeeze_ready': False,
            'extended_risk': False,
            'market_dir': '涓€?,
            'market_strength': 0.0,
            'learning_mult': 1.0,
            'confidence_size_mult': 1.0,
            'confidence_size_note': 'fallback',
        }

def get_direction_position_count(side_name):
    try:
        with STATE_LOCK:
            active = list(STATE.get("active_positions", []))
        return sum(1 for p in active if (p.get("side") or "").lower() == side_name)
    except:
        return 0

def plan_scale_in_orders(sig, total_qty, entry_price, atr):
    """
    鍒嗘壒閫插牬瑕忓妰鍣細閬垮厤鍑藉紡缂哄け灏庤嚧宸查仈姊濅欢鍗荤劇娉曚笅鍠€?    鍥炲偝鏍煎紡锛?      {mode: single|scale_in, primary_qty, secondary_qty, secondary_price, note}
    """
    try:
        total_qty = float(total_qty or 0)
        entry_price = float(entry_price or 0)
        atr = float(atr or 0)
        if total_qty <= 0 or entry_price <= 0:
            return {
                "mode": "single",
                "primary_qty": total_qty,
                "secondary_qty": 0.0,
                "secondary_price": None,
                "note": "鍊変綅涓嶈冻锛屾敼鍠瓎閫插牬"
            }

        score = float(sig.get('score', 0) or 0)
        rr = float(sig.get('rr', sig.get('rrr', 0)) or 0)
        entry_quality = float(sig.get('entry_quality', 0) or 0)
        side = sig.get('side', 'long')
        setup = str(sig.get('setup') or sig.get('setup_name') or '')

        # 鍙湪鐩稿皪楂樺搧璩▕铏熸檪鍒嗘壒锛岄伩鍏嶅お寮辩殑鍠帥澶鍠?        should_scale = (
            score >= 60
            and rr >= 1.6
            and entry_quality >= 6
        ) or ('绐佺牬' in setup) or ('鍥炶俯' in setup) or ('pullback' in setup.lower())

        if not should_scale:
            return {
                "mode": "single",
                "primary_qty": total_qty,
                "secondary_qty": 0.0,
                "secondary_price": None,
                "note": "鍠瓎閫插牬"
            }

        ratio = (SCALE_IN_MIN_RATIO + SCALE_IN_MAX_RATIO) / 2.0
        secondary_qty = max(total_qty * ratio, 0.0)
        primary_qty = max(total_qty - secondary_qty, 0.0)
        if primary_qty <= 0:
            return {
                "mode": "single",
                "primary_qty": total_qty,
                "secondary_qty": 0.0,
                "secondary_price": None,
                "note": "涓诲柈涓嶈冻锛屾敼鍠瓎閫插牬"
            }

        atr = max(atr, entry_price * 0.003)
        if side == 'long':
            secondary_price = entry_price - atr * 0.35
        else:
            secondary_price = entry_price + atr * 0.35

        return {
            "mode": "scale_in",
            "primary_qty": primary_qty,
            "secondary_qty": secondary_qty,
            "secondary_price": round(secondary_price, 6),
            "note": "鍒嗘壒閫插牬锛氬厛涓诲柈锛屽洖韪?鍙嶅綀瑁滅浜屾壒"
        }
    except Exception as e:
        return {
            "mode": "single",
            "primary_qty": float(total_qty or 0),
            "secondary_qty": 0.0,
            "secondary_price": None,
            "note": f"鍒嗘壒瑕忓妰澶辨晽锛屾敼鍠瓎閫插牬: {e}"
        }

def compute_order_size(sym, entry_price, stop_price, equity, lev, margin_pct=None):
    """
    鍥哄畾姣忓柈鍚嶇洰鍊変綅 20U銆?    淇濊瓑閲?= 20U / 妲撴】锛涘彛鏁?= 20U / 閫插牬鍍广€?    """
    try:
        entry_price = float(entry_price)
        stop_price  = float(stop_price)
        equity      = max(float(equity), 1.0)
        lev         = max(float(lev), 1.0)
        stop_dist   = abs(entry_price - stop_price)
        if stop_dist <= 0:
            stop_dist = entry_price * 0.01

        fixed_notional_usdt = max(float(FIXED_ORDER_NOTIONAL_USDT if _is_crypto_usdt_swap_symbol(sym) else FIXED_STOCK_ORDER_NOTIONAL_USDT), 0.1)
        raw_qty = fixed_notional_usdt / max(entry_price, 1e-9)

        try:
            mkt = exchange.market(sym)
            min_amt = float(mkt.get('limits', {}).get('amount', {}).get('min') or 0)
            if min_amt > 0:
                raw_qty = max(raw_qty, min_amt)
        except:
            pass

        qty = float(exchange.amount_to_precision(sym, raw_qty))
        actual_notional_usdt = qty * entry_price
        used_margin_usdt = actual_notional_usdt / lev
        used_margin_pct = used_margin_usdt / equity if equity > 0 else 0.0
        est_risk_usdt = qty * stop_dist
        return qty, round(used_margin_usdt, 4), round(est_risk_usdt, 4), round(stop_dist, 6), round(float(used_margin_pct), 4)
    except Exception as e:
        print("鍊変綅瑷堢畻澶辨晽 {}: {}".format(sym, e))
        fixed_notional_usdt = max(float(FIXED_ORDER_NOTIONAL_USDT if _is_crypto_usdt_swap_symbol(sym) else FIXED_STOCK_ORDER_NOTIONAL_USDT), 0.1)
        qty = float(exchange.amount_to_precision(sym, fixed_notional_usdt / max(float(entry_price),1e-9)))
        used_margin_usdt = fixed_notional_usdt / max(float(lev), 1.0)
        used_margin_pct = used_margin_usdt / max(float(equity), 1.0)
        return qty, round(used_margin_usdt, 4), 0.0, abs(float(entry_price) - float(stop_price)), round(used_margin_pct, 4)

def tighten_position_for_session(sym, contracts, side, entry_price, mark_price):
    # 鏅傛淇濊宸插仠鐢紝涓嶅啀鍥犵壒瀹氭檪娈电府鍊夋垨骞冲€夈€?    return False
    try:
        pnl_pct = 0.0
        if entry_price and mark_price:
            if side == 'long':
                pnl_pct = (mark_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - mark_price) / entry_price

        # 鐩堝埄鍠府涓€鍗婂€変綅锛岃畵鍓╅閮ㄤ綅浜ょ郸绉诲嫊姝㈢泩锛涜櫑鎼嶅柈鐩存帴骞冲€夈€?        if pnl_pct > 0.004:
            partial_close_position(sym, contracts, side, 0.5, "闁嬬洡淇濊绺€?)
            with TRAILING_LOCK:
                if sym in TRAILING_STATE:
                    ts = TRAILING_STATE[sym]
                    ts['trail_pct'] = min(ts.get('trail_pct', 0.05), 0.03)
                    if side == 'long':
                        ts['initial_sl'] = max(ts.get('initial_sl', 0), entry_price)
                    else:
                        ts['initial_sl'] = min(ts.get('initial_sl', float('inf')), entry_price)
            print("馃洝 闁嬬洡淇濊绺€? {} 鐩堝埄鍠繚鐣欒定鍕㈠柈".format(sym))
        else:
            close_position(sym, contracts, side)
            print("馃洝 闁嬬洡淇濊骞冲€? {} 铏ф悕/鐒″埄娼ゅ柈鐩存帴闆㈠牬".format(sym))
    except Exception as e:
        print("闁嬬洡淇濊铏曠悊澶辨晽 {}: {}".format(sym, e))

def finalize_open_position_entry(sym, side, sig, qty, sl_price, tp_price, lev, order_usdt, est_risk_usdt, used_margin_pct, margin_ctx, protect=True):
    pos_side = 'long' if side == 'buy' else 'short'
    protected_qty = float(qty or 0)
    openai_plan = dict(sig.get('openai_trade_plan') or {})
    openai_meta = dict(sig.get('openai_trade_meta') or {})
    if protect:
        sl_ok, tp_ok = ensure_exchange_protection(sym, side, pos_side, protected_qty, sl_price, tp_price)
        if not (sl_ok and tp_ok):
            print("鉂?浜ゆ槗鎵€ SL/TP 淇濊鍠璀夊け鏁?sl_ok={} tp_ok={})锛岀珛鍒诲競鍍瑰钩鍊変繚璀? {}".format(sl_ok, tp_ok, sym))
            close_position(sym, protected_qty, 'long' if side == 'buy' else 'short')
            return False

    trade_id = "{}_{}".format(sym.replace('/', '').replace(':', ''), int(time.time()))
    rec = {
        "symbol": sym,
        "side": "鍋氬" if side == 'buy' else "鍋氱┖",
        "score": sig['score'],
        "price": sig['price'],
        "stop_loss": sl_price,
        "take_profit": tp_price,
        "est_pnl": sig['est_pnl'],
        "order_usdt": round(order_usdt, 2),
        "risk_usdt": round(est_risk_usdt, 2),
        "leverage": lev,
        "margin_pct": round(used_margin_pct * 100, 2),
        "scale_mode": sig.get('scale_plan', {}).get('mode', 'single'),
        "time": tw_now_str(),
        "learn_id": trade_id,
        "decision_source": sig.get('decision_source', 'rule_engine'),
        "openai_order_type": openai_plan.get('order_type', ''),
        "openai_confidence": round(float(openai_plan.get('confidence', 0) or 0), 2) if openai_plan else 0.0,
        "openai_model": openai_meta.get('model', ''),
    }
    with STATE_LOCK:
        STATE["trade_history"].insert(0, rec)
        if len(STATE["trade_history"]) > 30:
            STATE["trade_history"] = STATE["trade_history"][:30]
    persist_trade_history_record(rec)

    with STATE_LOCK:
        equity_now = STATE.get("equity", 0)
    learn_rec = {
        "id": trade_id,
        "symbol": sym,
        "side": side,
        "entry_price": sig['price'],
        "entry_score": sig['score'],
        "breakdown": sig.get('breakdown', {}),
        "atr_mult_sl": sig.get('sl_mult', 2.0),
        "atr_mult_tp": sig.get('tp_mult', 3.0),
        "margin_pct": used_margin_pct,
        "margin_learning_mult": margin_ctx.get('learning_mult', 1.0),
        "scale_mode": sig.get('scale_plan', {}).get('mode', 'single'),
        "order_usdt": round(order_usdt, 4),
        "used_margin_usdt": round(order_usdt, 4),
        "entry_equity": round(float(equity_now or 0), 4),
        "planned_leverage": round(float(lev or 0), 4),
        "entry_time": tw_now_str("%Y-%m-%d %H:%M:%S"),
        "exit_price": None,
        "exit_time": None,
        "pnl_pct": None,
        "setup_label": sig.get('setup_label') or sig.get('breakdown', {}).get('Setup', ''),
        "trend_learning_stage": _trend_learning_stage()[0],
        "expected_entry_price": sig.get('price'),
        "execution_snapshot": dict(_execution_quality_state(sig) or {}),
        "execution_gate": dict(sig.get('execution_gate') or {}),
        "execution_quality_bucket": execution_quality_bucket(sig.get('execution_quality') or {}),
        "decision_fingerprint": build_decision_fingerprint({
            'symbol': sym,
            'side': side,
            'setup_label': sig.get('setup_label') or sig.get('breakdown', {}).get('Setup', ''),
            'breakdown': sig.get('breakdown', {}),
            'execution_quality': dict(sig.get('execution_quality') or {}),
            'entry_time': tw_now_str("%Y-%m-%d %H:%M:%S"),
        }),
        "result": "open",
        "post_candles": [],
        "missed_move_pct": None,
        "post_run_pct": 0.0,
        "post_pullback_pct": 0.0,
        "trend_continuation": False,
        "trend_reason": "",
        "source": "live_bitget_openai" if openai_plan else "live_bitget_v32",
        "decision_source": sig.get('decision_source', 'rule_engine'),
        "openai_order_type": openai_plan.get('order_type', ''),
        "openai_confidence": round(float(openai_plan.get('confidence', 0) or 0), 2) if openai_plan else 0.0,
        "openai_model": openai_meta.get('model', ''),
    }
    if openai_plan:
        learn_rec["openai_plan"] = dict(openai_plan)
    learn_rec.update(_dataset_meta())
    learn_rec['dataset_layer'] = 'new'
    learn_rec['dataset_reset_from'] = TREND_LEARNING_RESET_FROM
    learn_rec = enrich_learning_trade(learn_rec, reset_from=TREND_LEARNING_RESET_FROM)
    with LEARN_LOCK:
        LEARN_DB["trades"].append(learn_rec)
        if len(LEARN_DB["trades"]) > 500:
            LEARN_DB["trades"] = LEARN_DB["trades"][-500:]
        save_learn_db(LEARN_DB)

    trail_pct = min(max(sig.get('atr', sig['price'] * 0.01) / sig['price'] * 3, 0.03), 0.10)
    if openai_plan:
        hint_trail_pct = float(openai_plan.get('trail_pct_hint', 0) or 0)
        if hint_trail_pct > 0:
            trail_pct = min(max(hint_trail_pct, 0.01), 0.12)
    with TRAILING_LOCK:
        TRAILING_STATE[sym] = {
            "side": side,
            "entry_price": sig['price'],
            "highest_price": sig['price'] if side == 'buy' else float('inf'),
            "lowest_price": sig['price'] if side == 'sell' else float('inf'),
            "trail_pct": trail_pct,
            "initial_sl": sl_price,
            "atr": sig.get('atr', sig['price'] * 0.01),
            "entry_time_ts": time.time(),
            "time_stop_sec": TIME_STOP_BARS_15M * 15 * 60,
            "partial_done": 0,
            "breakdown": dict(sig.get('breakdown') or {}),
            "setup_label": sig.get('setup_label') or sig.get('breakdown', {}).get('Setup', ''),
            "fixed_tp": tp_price,
            "breakeven_atr_hint": float(openai_plan.get('breakeven_atr_hint', 0) or 0),
            "trail_trigger_atr_hint": float(openai_plan.get('trail_trigger_atr_hint', 0) or 0),
            "dynamic_take_profit_hint": float(openai_plan.get('dynamic_take_profit_hint', 0) or 0),
        }
    record_order_placed()
    print("涓嬪柈鎴愬姛: {} {} @{} {}U 棰ㄩ毆{}U x{}鍊?SL:{} TP:{} 绉诲嫊鍥炴挙:{:.1f}% 渚嗘簮:{}".format(
        sym,
        side,
        sig['price'],
        round(order_usdt, 2),
        round(est_risk_usdt, 2),
        lev,
        sl_price,
        tp_price,
        trail_pct * 100,
        sig.get('decision_source', 'rule_engine'),
    ))
    return True


def place_order(sig):
    # 棰ㄦ帶妾㈡煡
    ok, reason = check_risk_ok()
    if not ok:
        print("棰ㄦ帶闃绘搵涓嬪柈: {}".format(reason))
        with STATE_LOCK:
            STATE["halt_reason"] = reason
        return

    # 闃查噸瑜囷細鏈吉宸蹭笅鍠殑骞ｄ笉鍐嶉噸瑜?    sym_check = sig['symbol']
    if not can_reenter_symbol(sym_check):
        print('鈿狅笍 {}锛岃烦閬?{}'.format(get_symbol_cooldown_note(sym_check) or '閫插牬鍐峰嵒涓?, sym_check))
        return
    with _ORDERED_LOCK:
        if sym_check in _ORDERED_THIS_SCAN:
            print("鈿狅笍 闃查噸瑜囷細{}鏈吉宸蹭笅鍠紝璺抽亷".format(sym_check))
            return
        _ORDERED_THIS_SCAN.add(sym_check)

    # 涓嬪柈閹栵細纰轰繚鍚屼竴鏅傞枔鍙湁涓€绛嗕笅鍠湪鍩疯
    with ORDER_LOCK:
        # 浜屾纰鸿獚鎸佸€夋暩閲忥紙鐣版涓嬪柈鍙兘閫犳垚瓒呴亷7鍊嬶級
        with STATE_LOCK:
            current_pos_count = len(STATE["active_positions"])
            # 鍚屾檪纰鸿獚閫欏€嬪梗娌掓湁鍦ㄦ寔鍊変腑
            pos_syms_now = {p['symbol'] for p in STATE["active_positions"]}
        if current_pos_count >= MAX_OPEN_POSITIONS:
            print("鎸佸€夊凡閬攞}鍊嬩笂闄愶紝鍙栨秷涓嬪柈: {}".format(MAX_OPEN_POSITIONS, sig['symbol']))
            with _ORDERED_LOCK:
                _ORDERED_THIS_SCAN.discard(sym_check)
            return
        if sym_check in pos_syms_now:
            print("鈿狅笍 闃查噸瑜囷細{}宸插湪鎸佸€変腑锛岃烦閬?.format(sym_check))
            with _ORDERED_LOCK:
                _ORDERED_THIS_SCAN.discard(sym_check)
            return

    try:
        sym=sig['symbol']
        side = 'buy' if sig['score'] > 0 else 'sell'
        sig['side'] = 'long' if side == 'buy' else 'short'  # 纰轰繚 sig['side'] 鏄?long/short
        openai_plan = dict(sig.get('openai_trade_plan') or {})
        preferred_order_type = 'limit' if str(openai_plan.get('order_type') or '').lower() == 'limit' else 'market'
        planned_entry_price = float(openai_plan.get('entry_price', sig.get('price', 0)) or sig.get('price', 0) or 0)
        if planned_entry_price > 0:
            sig['planned_entry_price'] = planned_entry_price

        same_dir_count = get_direction_position_count(sig['side'])
        if same_dir_count >= MAX_SAME_DIRECTION:
            print("鍚屾柟鍚戞寔鍊夊凡閬攞}绛嗕笂闄愶紝璺抽亷 {} {}".format(MAX_SAME_DIRECTION, sym, sig['side']))
            with _ORDERED_LOCK:
                _ORDERED_THIS_SCAN.discard(sym_check)
            return
        # 涓嬪柈鍓嶅挤鍒舵妸瑭插梗妲撴】瑷畾鍒?Bitget 鍙枊鐨勬渶澶у€?        lev = _get_symbol_max_leverage(sym)
        try:
            lev, lev_params, lev_err, lev_ok = _force_set_symbol_max_leverage(sym, side)
            if not lev_ok:
                raise RuntimeError(lev_err or 'failed to force max leverage on Bitget')
            sig['resolved_max_leverage'] = lev
            if openai_plan:
                openai_plan['leverage'] = lev
                sig['openai_trade_plan'] = openai_plan
            print("妲撴】瑷畾: {} {}x {}".format(sym, lev, ('params={}'.format(lev_params) if lev_params else '')))
        except Exception as lev_e:
            print("妲撴】瑷畾澶辨晽({}): {} | 鍙栨秷涓嬪柈".format(sym, lev_e))
            with _ORDERED_LOCK:
                _ORDERED_THIS_SCAN.discard(sym_check)
            return

        # 鍕曟厠淇濊瓑閲戯細鏍规摎鍒嗘暩 / 钃勫嫝 / 娉㈠嫊 / 鍚屽悜鎸佸€夛紝鑷嫊姹哄畾 3%~8%
        with STATE_LOCK: equity=STATE["equity"]
        if equity<=0: equity=10.0
        margin_ctx = infer_margin_context(sig, same_dir_count)
        margin_pct = margin_ctx['margin_pct']
        if openai_plan:
            margin_ctx = dict(margin_ctx or {})
            margin_pct = clamp(
                float(openai_plan.get('margin_pct', margin_pct) or margin_pct),
                max(MIN_MARGIN_PCT, float(OPENAI_TRADE_CONFIG.get('min_margin_pct', MIN_MARGIN_PCT) or MIN_MARGIN_PCT)),
                min(MAX_MARGIN_PCT, float(OPENAI_TRADE_CONFIG.get('max_margin_pct', MAX_MARGIN_PCT) or MAX_MARGIN_PCT)),
            )
            margin_ctx['openai_override'] = True
            margin_ctx['openai_margin_pct'] = margin_pct
        _gate = apply_execution_guard(sym, side, margin_pct)
        sig['execution_quality'] = dict(_gate.get('snapshot') or {})
        sig['execution_gate'] = dict(_gate.get('gate') or {})
        _gate_penalty = float((sig.get('execution_gate') or {}).get('score_penalty', 0.0) or 0.0)
        if _gate_penalty > 0:
            try:
                sig['score'] = round(float(sig.get('score', 0) or 0) - _gate_penalty, 2)
                bd_penalty = dict(sig.get('breakdown') or {})
                bd_penalty['鍩疯棰ㄩ毆鎵ｅ垎'] = -round(_gate_penalty, 2)
                sig['breakdown'] = bd_penalty
            except Exception:
                pass
        if not _gate.get('allow', True):
            print('閫佸柈鍓嶆渶寰屽畧闁€闃绘搵 {}: {}'.format(sym, (_gate.get('gate') or {}).get('reasons')))
            append_audit_log('execution_guard', '閫佸柈鍓嶆渶寰屽畧闁€闃绘搵', {'symbol': sym, 'side': side, 'gate': _gate})
            with _ORDERED_LOCK:
                _ORDERED_THIS_SCAN.discard(sym_check)
            return
        margin_pct = float(_gate.get('margin_pct', margin_pct) or margin_pct)
        sl_price = float(openai_plan.get('stop_loss', sig['stop_loss']) or sig['stop_loss'])
        tp_price = float(openai_plan.get('take_profit', sig['take_profit']) or sig['take_profit'])
        entry_for_size = planned_entry_price if planned_entry_price > 0 else sig['price']
        amt, order_usdt, est_risk_usdt, stop_distance, used_margin_pct = compute_order_size(sym, entry_for_size, sl_price, equity, lev, margin_pct)
        sig['margin_pct'] = used_margin_pct
        sig['margin_ctx'] = margin_ctx
        print("鍕曟厠淇濊瓑閲? {} score={} margin={}%(trend={} squeeze={} extended={} atr={})".format(
            sym, sig.get('score'), round(used_margin_pct*100,2),
            margin_ctx.get('trend_aligned'), margin_ctx.get('squeeze_ready'),
            margin_ctx.get('extended_risk'), margin_ctx.get('atr_ratio')
        ))
        if amt <= 0:
            print("鍊変綅澶皬鐒℃硶涓嬪柈: {}".format(sym))
            with _ORDERED_LOCK:
                _ORDERED_THIS_SCAN.discard(sym_check)
            return
        # (FVG 鍒ゆ柗寰屽彲鑳芥渻鏇存柊 sl_price/tp_price)

        # Step1: FVG 鏈€鍎児鍒ゆ柗
        atr_val  = sig.get('atr', sig['price'] * 0.01)
        if openai_plan and preferred_order_type == 'limit' and planned_entry_price > 0:
            fvg_price, fvg_note = planned_entry_price, 'OpenAI limit entry'
        elif openai_plan:
            fvg_price, fvg_note = None, 'OpenAI market entry'
        else:
            fvg_price, fvg_note = get_fvg_entry_price(sym, sig['side'], sig['price'], sig.get('atr15', atr_val))
        print("FVG鍒ゆ柗: {} 鈫?{}".format(sym, fvg_note))

        # Bitget 鍚堢磩蹇呰鍙冩暩
        pos_side = 'long' if side == 'buy' else 'short'
        order_params = {
            'tdMode':   'cross',      # 鍏ㄥ€?            'posSide':  pos_side,     # long/short锛圔itget闆欏悜鎸佸€夊繀闋堬級
        }

        scale_plan = plan_scale_in_orders(sig, amt, sig['price'], sig.get('atr15', atr_val))
        if openai_plan:
            scale_plan = {'mode': 'single', 'note': 'openai_trade_plan'}
        sig['scale_plan'] = scale_plan
        market_qty = amt

        if fvg_price is not None:
            # 闃查噸瑜囷細宸叉湁鎺涘柈灏辫烦閬?            with FVG_LOCK:
                already_pending = sym in FVG_ORDERS
            if already_pending:
                print("鈿狅笍 FVG闃查噸瑜囷細{} 宸叉湁鎺涘柈锛岃烦閬?.format(sym))
                return

            # 鎺涢檺鍍瑰柈绛夊洖鍒?FVG 缂哄彛
            try:
                order = exchange.create_order(sym, 'limit', side, amt, fvg_price, params=order_params)
                order_id = order.get('id', '')
                # 閲嶆柊瑷堢畻姝㈡悕姝㈢泩鍩烘柤FVG鍍癸紝涓﹀悓姝ラ噸绠楀€変綅
                if openai_plan:
                    sl_price = float(openai_plan.get('stop_loss', sl_price) or sl_price)
                    tp_price = float(openai_plan.get('take_profit', tp_price) or tp_price)
                else:
                    sl_atr = sig.get('sl_mult', 2.0) * atr_val
                    tp_atr = sig.get('tp_mult', 3.0) * atr_val
                    if sig['side'] == 'long':
                        sl_price = round(fvg_price - sl_atr, 6)
                        tp_price = round(fvg_price + tp_atr, 6)
                    else:
                        sl_price = round(fvg_price + sl_atr, 6)
                        tp_price = round(fvg_price - tp_atr, 6)
                sig['stop_loss']   = sl_price
                sig['take_profit'] = tp_price
                sig['price']       = fvg_price
                amt, order_usdt, est_risk_usdt, stop_distance, used_margin_pct = compute_order_size(sym, fvg_price, sl_price, equity, lev, margin_pct)
                register_fvg_order(
                    sym, order_id, sig['side'], fvg_price,
                    sig['score'], sl_price, tp_price,
                    sig.get('support', 0), sig.get('resist', 0),
                    extra_meta={
                        'strategy_source': 'openai' if openai_plan else 'fvg',
                        'order_kind': preferred_order_type,
                        'limit_cancel_price': float(sig.get('limit_cancel_price', 0) or 0),
                        'limit_cancel_timeframe': str(sig.get('limit_cancel_timeframe') or ''),
                        'limit_cancel_condition': str(sig.get('limit_cancel_condition') or ''),
                        'limit_cancel_note': str(sig.get('limit_cancel_note') or ''),
                    }
                )
                if openai_plan:
                    pending_sig = dict(sig)
                    pending_sig['price'] = fvg_price
                    pending_sig['stop_loss'] = sl_price
                    pending_sig['take_profit'] = tp_price
                    pending_sig['decision_source'] = sig.get('decision_source', 'openai')
                    with PENDING_LIMIT_LOCK:
                        PENDING_LIMIT_META[sym] = {
                            'qty': float(amt or 0),
                            'leverage': float(lev or 0),
                            'order_usdt': float(order_usdt or 0),
                            'risk_usdt': float(est_risk_usdt or 0),
                            'margin_pct': float(used_margin_pct or 0),
                            'margin_ctx': dict(margin_ctx or {}),
                            'signal': pending_sig,
                        }
                print("馃搶 FVG闄愬児鎺涘柈: {} {} @{:.6f} | {}".format(sym, side, fvg_price, fvg_note))
                return
            except Exception as fvg_err:
                print("FVG闄愬児涓嬪柈澶辨晽锛屾敼鐢ㄥ競鍍? {}".format(fvg_err))
                market_qty = amt
                order = exchange.create_order(sym, 'market', side, market_qty, params=order_params)
        else:
            with FVG_LOCK:
                if sym in FVG_ORDERS:
                    old_order = FVG_ORDERS.pop(sym, None)
                    if old_order:
                        try:
                            exchange.cancel_order(old_order['order_id'], sym)
                            print("馃棏 鍙栨秷鑸奆VG鎺涘柈鍐嶄笅甯傚児: {}".format(sym))
                        except:
                            pass
                        update_state(fvg_orders=dict(FVG_ORDERS))
            market_qty = scale_plan.get('primary_qty', amt) if scale_plan.get('mode') == 'scale_in' else amt
            market_qty = float(exchange.amount_to_precision(sym, max(market_qty, 0))) if market_qty > 0 else 0.0
            order = exchange.create_order(sym, 'market', side, market_qty or amt, params=order_params)

            if scale_plan.get('mode') == 'scale_in' and scale_plan.get('secondary_qty', 0) > 0 and scale_plan.get('secondary_price'):
                try:
                    secondary_qty = float(exchange.amount_to_precision(sym, scale_plan['secondary_qty']))
                    if secondary_qty > 0:
                        pullback_order = exchange.create_order(sym, 'limit', side, secondary_qty, scale_plan['secondary_price'], params=order_params)
                        print("馃獪 鍒嗘壒閫插牬鎺涘柈: {} 绗簩鎵?{}鍙?@{:.6f} | {}".format(sym, secondary_qty, scale_plan['secondary_price'], scale_plan.get('note', '')))
                        sig['scale_in_pending_order_id'] = pullback_order.get('id', '')
                except Exception as scale_err:
                    print("鍒嗘壒閫插牬鎺涘柈澶辨晽锛屼繚鐣欎富鍠? {}".format(scale_err))

        print("涓诲柈鎴愬姛: {} {} {}鍙?| {} | {}".format(sym, side, market_qty if market_qty else amt, fvg_note, scale_plan.get('note', '鍠瓎閫插牬')))
        touch_entry_lock(sym)
        finalize_open_position_entry(
            sym,
            side,
            sig,
            market_qty if market_qty else amt,
            sl_price,
            tp_price,
            lev,
            order_usdt,
            est_risk_usdt,
            used_margin_pct,
            margin_ctx,
            protect=True,
        )
    except Exception as e:
        print("涓嬪柈澶辨晽: {}".format(e))

# =====================================================
# 骞冲€夛紙姝ｇ⒑浣跨敤 reduceOnly锛?# =====================================================
def close_all():
    try:
        n=0
        positions=exchange.fetch_positions()
        for p in positions:
            c=float(p.get('contracts',0) or 0)
            if abs(c)>0:
                sym=p['symbol']
                side='sell' if p['side']=='long' else 'buy'
                try:
                    exchange.create_order(sym,'market',side,abs(c),params={
                        'reduceOnly':True,
                        'marginMode':'cross',
                    })
                    n+=1
                    print("骞冲€夋垚鍔? {} {}鍙?.format(sym,abs(c)))
                except Exception as pe:
                    print("骞冲€夊け鏁?{}: {}".format(sym,pe))
        return n
    except Exception as e:
        print("骞冲€夋暣楂斿け鏁? {}".format(e)); return 0

# =====================================================
# 涓绘巸鎻忓煼琛岀窉
# =====================================================
def _apply_openai_trade_plan_to_signal(sig, decision, result):
    plan = dict(decision or {})
    if not plan:
        return sig
    sig['openai_trade_plan'] = plan
    sig['openai_trade_meta'] = {
        'model': str((result or {}).get('symbol_state', {}).get('last_model') or OPENAI_TRADE_CONFIG.get('model') or ''),
        'status': str((result or {}).get('status') or ''),
        'estimated_cost_twd': float((result or {}).get('estimated_cost_twd', 0) or 0),
        'payload_hash': str((result or {}).get('payload_hash') or ''),
    }
    sig['decision_source'] = 'openai'
    sig['stop_loss'] = float(plan.get('stop_loss', sig.get('stop_loss', 0)) or sig.get('stop_loss', 0))
    sig['take_profit'] = float(plan.get('take_profit', sig.get('take_profit', 0)) or sig.get('take_profit', 0))
    planned_entry = float(plan.get('entry_price', sig.get('price', 0)) or sig.get('price', 0))
    sig['planned_entry_price'] = planned_entry
    sig['limit_cancel_price'] = float(plan.get('limit_cancel_price', sig.get('limit_cancel_price', 0)) or sig.get('limit_cancel_price', 0) or 0)
    sig['limit_cancel_timeframe'] = str(plan.get('limit_cancel_timeframe') or sig.get('limit_cancel_timeframe') or '')
    sig['limit_cancel_condition'] = str(plan.get('limit_cancel_condition') or sig.get('limit_cancel_condition') or '')
    sig['limit_cancel_note'] = str(plan.get('limit_cancel_note') or sig.get('limit_cancel_note') or '')
    if str(plan.get('order_type') or '').lower() != 'limit' and planned_entry > 0:
        sig['price'] = planned_entry
    return sig


def _get_symbol_max_leverage(symbol):
    lev = 0
    try:
        mkt = exchange.market(symbol)
        info = mkt.get('info', {})
        for field in ['maxLeverage', 'maxLev', 'leverageMax']:
            val = info.get(field)
            if not val:
                continue
            try:
                lev = max(lev, int(float(str(val))))
                if lev > 1:
                    break
            except Exception:
                continue
        if lev <= 1:
            try:
                limit_lev = int(float(mkt.get('limits', {}).get('leverage', {}).get('max', 0) or 0))
                lev = max(lev, limit_lev)
            except Exception:
                pass
        if lev <= 1:
            try:
                tiers = exchange.fetch_leverage_tiers([symbol])
                sym_tiers = tiers.get(symbol, [])
                if sym_tiers:
                    lev = max(lev, int(max(t.get('maxLeverage', lev) for t in sym_tiers)))
            except Exception:
                pass
    except Exception:
        pass
    if lev <= 1:
        lev = max(1, int(OPENAI_TRADE_CONFIG.get('max_leverage', 25) or 25))
    return max(int(lev or 1), 1)


def _force_set_symbol_max_leverage(symbol, side):
    pos_side = 'long' if str(side or '').lower() in ('buy', 'long') else 'short'
    lev = _get_symbol_max_leverage(symbol)
    market = {}
    info = {}
    try:
        market = exchange.market(symbol)
        info = dict(market.get('info', {}) or {})
    except Exception:
        market = {}
        info = {}
    market_id = str(market.get('id') or info.get('symbol') or symbol).strip()
    margin_coin = str(info.get('marginCoin') or market.get('settle') or market.get('quote') or 'USDT').strip()
    product_type = str(info.get('productType') or '').strip()

    margin_attempts = [
        {},
        {'tdMode': 'cross', 'holdSide': pos_side},
        {'marginMode': 'cross', 'holdSide': pos_side},
        {'tdMode': 'cross', 'posSide': pos_side},
        {'marginMode': 'cross', 'posSide': pos_side},
    ]
    for params in margin_attempts:
        try:
            if hasattr(exchange, 'set_margin_mode'):
                exchange.set_margin_mode('cross', symbol, params or None)
        except Exception:
            pass

    attempts = [
        {},
        {'tdMode': 'cross', 'holdSide': pos_side},
        {'marginMode': 'cross', 'holdSide': pos_side},
        {'tdMode': 'cross', 'posSide': pos_side},
        {'marginMode': 'cross', 'posSide': pos_side},
    ]
    errors = []
    for params in attempts:
        try:
            if params:
                exchange.set_leverage(lev, symbol, params)
            else:
                exchange.set_leverage(lev, symbol)
            return lev, params, '', True
        except Exception as e:
            errors.append(str(e))
            continue

    implicit_payloads = [
        {'symbol': market_id, 'marginCoin': margin_coin, 'leverage': str(lev), 'holdSide': pos_side},
        {'symbol': market_id, 'marginCoin': margin_coin, 'longLeverage': str(lev), 'shortLeverage': str(lev)},
        {'symbol': market_id, 'productType': product_type, 'marginCoin': margin_coin, 'leverage': str(lev), 'holdSide': pos_side},
        {'symbol': market_id, 'productType': product_type, 'marginCoin': margin_coin, 'longLeverage': str(lev), 'shortLeverage': str(lev)},
    ]
    implicit_methods = [
        'private_mix_post_v2_mix_account_set_leverage',
        'private_mix_post_mix_v1_account_setleverage',
        'privateMixPostV2MixAccountSetLeverage',
        'privateMixPostMixV1AccountSetLeverage',
    ]
    for method_name in implicit_methods:
        method = getattr(exchange, method_name, None)
        if not callable(method):
            continue
        for payload in implicit_payloads:
            try:
                clean_payload = {k: v for k, v in payload.items() if str(v or '').strip()}
                method(clean_payload)
                return lev, {'implicit': method_name, **clean_payload}, '', True
            except Exception as e:
                errors.append('{}: {}'.format(method_name, e))
                continue
    return lev, {}, ' | '.join(errors[:3]), False


def _fixed_order_notional_usdt_for_symbol(symbol):
    return float(FIXED_ORDER_NOTIONAL_USDT if _is_crypto_usdt_swap_symbol(symbol) else FIXED_STOCK_ORDER_NOTIONAL_USDT)


def _mtf_pressure_structure_snapshot(df, timeframe, current_price=0.0):
    if df is None or len(df) < 20:
        return {}
    closed_df = df.iloc[:-1].copy() if len(df) >= 30 else df.copy()
    if closed_df is None or len(closed_df) < 20:
        closed_df = df.copy()
    if closed_df is None or len(closed_df) < 20:
        return {}
    c = closed_df['c'].astype(float)
    h = closed_df['h'].astype(float)
    l = closed_df['l'].astype(float)
    v = closed_df['v'].astype(float)
    last = float(c.iloc[-1])
    price = _safe_num(current_price, last) or last
    atr = max(safe_last(ta.atr(h, l, c, length=14), last * 0.006), last * 0.003, 1e-9)
    ema20 = safe_last(ta.ema(c, length=20), last)
    ema50 = safe_last(ta.ema(c, length=50), last)
    upper_window = min(len(h), 50)
    lower_window = min(len(l), 50)
    range_high_20 = float(h.tail(min(len(h), 20)).max())
    range_low_20 = float(l.tail(min(len(l), 20)).min())
    range_high_50 = float(h.tail(upper_window).max())
    range_low_50 = float(l.tail(lower_window).min())
    recent_highs = h.tail(min(len(h), 6)).tolist()
    recent_lows = l.tail(min(len(l), 6)).tolist()
    hh_count = 0
    hl_count = 0
    lh_count = 0
    ll_count = 0
    for idx in range(1, len(recent_highs)):
        if recent_highs[idx] > recent_highs[idx - 1]:
            hh_count += 1
        elif recent_highs[idx] < recent_highs[idx - 1]:
            lh_count += 1
    for idx in range(1, len(recent_lows)):
        if recent_lows[idx] > recent_lows[idx - 1]:
            hl_count += 1
        elif recent_lows[idx] < recent_lows[idx - 1]:
            ll_count += 1
    trend_stack = 'bullish' if last >= ema20 >= ema50 else 'bearish' if last <= ema20 <= ema50 else 'mixed'
    swing_bias = 'bullish' if hh_count >= 3 and hl_count >= 3 else 'bearish' if lh_count >= 3 and ll_count >= 3 else 'mixed'
    structure_bias = trend_stack if trend_stack == swing_bias else ('bullish' if trend_stack == 'bullish' and hl_count >= ll_count else 'bearish' if trend_stack == 'bearish' and ll_count >= hl_count else 'mixed')
    prior_high = float(h.iloc[-7:-1].max()) if len(h) >= 7 else range_high_20
    prior_low = float(l.iloc[-7:-1].min()) if len(l) >= 7 else range_low_20
    recent_break = 'breakout' if last > prior_high else 'breakdown' if last < prior_low else 'inside'
    pressure_candidates = [x for x in [range_high_20, range_high_50, prior_high] if x > price]
    support_candidates = [x for x in [range_low_20, range_low_50, prior_low] if x < price]
    pressure_price = min(pressure_candidates) if pressure_candidates else max(range_high_20, range_high_50, prior_high)
    support_price = max(support_candidates) if support_candidates else min(range_low_20, range_low_50, prior_low)
    pressure_dist = max((pressure_price - price) / max(price, 1e-9) * 100.0, 0.0)
    support_dist = max((price - support_price) / max(price, 1e-9) * 100.0, 0.0)
    pressure_dist_atr = max((pressure_price - price) / atr, 0.0)
    support_dist_atr = max((price - support_price) / atr, 0.0)
    volume_ratio = float(v.tail(5).mean()) / max(float(v.tail(20).mean()), 1e-9)
    return {
        'timeframe': str(timeframe),
        'last_close': _safe_round_metric(last, 8),
        'atr': _safe_round_metric(atr, 8),
        'trend_stack': trend_stack,
        'swing_bias': swing_bias,
        'structure_bias': structure_bias,
        'recent_break': recent_break,
        'hh_count': int(hh_count),
        'hl_count': int(hl_count),
        'lh_count': int(lh_count),
        'll_count': int(ll_count),
        'ema20': _safe_round_metric(ema20, 8),
        'ema50': _safe_round_metric(ema50, 8),
        'close_vs_ema20_pct': _safe_round_metric((last - ema20) / max(last, 1e-9) * 100.0, 3),
        'close_vs_ema50_pct': _safe_round_metric((last - ema50) / max(last, 1e-9) * 100.0, 3),
        'pressure_price': _safe_round_metric(pressure_price, 8),
        'support_price': _safe_round_metric(support_price, 8),
        'pressure_distance_pct': _safe_round_metric(pressure_dist, 3),
        'support_distance_pct': _safe_round_metric(support_dist, 3),
        'pressure_distance_atr': _safe_round_metric(pressure_dist_atr, 3),
        'support_distance_atr': _safe_round_metric(support_dist_atr, 3),
        'range_high_20': _safe_round_metric(range_high_20, 8),
        'range_low_20': _safe_round_metric(range_low_20, 8),
        'range_high_50': _safe_round_metric(range_high_50, 8),
        'range_low_50': _safe_round_metric(range_low_50, 8),
        'volume_ratio': _safe_round_metric(volume_ratio, 3),
    }


def _summarize_mtf_structure_pressure(structure_map, side):
    rows = [dict(v or {}) for v in list((structure_map or {}).values()) if isinstance(v, dict) and v]
    if not rows:
        return {}
    side = str(side or '').lower()
    blocking_key = 'pressure_distance_atr' if side == 'long' else 'support_distance_atr'
    backing_key = 'support_distance_atr' if side == 'long' else 'pressure_distance_atr'
    blocking_price_key = 'pressure_price' if side == 'long' else 'support_price'
    backing_price_key = 'support_price' if side == 'long' else 'pressure_price'
    valid_blocking = [r for r in rows if _safe_num(r.get(blocking_key), 0.0) > 0]
    valid_backing = [r for r in rows if _safe_num(r.get(backing_key), 0.0) > 0]
    nearest_blocking = min(valid_blocking, key=lambda r: _safe_num(r.get(blocking_key), 9999.0)) if valid_blocking else {}
    nearest_backing = min(valid_backing, key=lambda r: _safe_num(r.get(backing_key), 9999.0)) if valid_backing else {}
    bullish_count = sum(1 for r in rows if str(r.get('structure_bias') or '') == 'bullish')
    bearish_count = sum(1 for r in rows if str(r.get('structure_bias') or '') == 'bearish')
    aligned_count = bullish_count if side == 'long' else bearish_count
    opposing_count = bearish_count if side == 'long' else bullish_count
    return {
        'side': side,
        'aligned_timeframes': int(aligned_count),
        'opposing_timeframes': int(opposing_count),
        'nearest_blocking_timeframe': str(nearest_blocking.get('timeframe') or ''),
        'nearest_blocking_price': _safe_round_metric(nearest_blocking.get(blocking_price_key), 8),
        'nearest_blocking_distance_atr': _safe_round_metric(nearest_blocking.get(blocking_key), 3),
        'nearest_backing_timeframe': str(nearest_backing.get('timeframe') or ''),
        'nearest_backing_price': _safe_round_metric(nearest_backing.get(backing_price_key), 8),
        'nearest_backing_distance_atr': _safe_round_metric(nearest_backing.get(backing_key), 3),
        'stacked_blocking_within_1atr': int(sum(1 for r in valid_blocking if _safe_num(r.get(blocking_key), 99.0) <= 1.0)),
        'stacked_blocking_within_2atr': int(sum(1 for r in valid_blocking if _safe_num(r.get(blocking_key), 99.0) <= 2.0)),
    }


def _build_openai_short_term_context(sig, market_info, constraints):
    symbol = str(sig.get('symbol') or '')
    side = 'long' if float(sig.get('score', 0) or 0) >= 0 else 'short'
    tf_plan = [('1m', 100), ('5m', 100), ('15m', 100), ('1h', 100), ('4h', 100), ('1d', 100)]
    tf_data = {}
    raw_frames = {}
    timeframe_bars = {}
    for tf, limit in tf_plan:
        df = _safe_fetch_ohlcv_df(symbol, tf, limit)
        if df is None or df.empty:
            continue
        raw_frames[tf] = df
        snap = _snapshot_from_df(df)
        if snap:
            tf_data[tf] = snap
        timeframe_bars[tf] = _serialize_ohlcv_rows(df, limit=limit)

    radar = {}
    try:
        if raw_frames.get('15m') is not None and raw_frames.get('4h') is not None:
            radar = analyze_pre_breakout_radar(raw_frames.get('15m'), raw_frames.get('4h'), raw_frames.get('1d'))
    except Exception as radar_e:
        radar = {'ready': False, 'note': 'pre_breakout_radar_error: {}'.format(radar_e)}

    ticker_context = {}
    try:
        ticker = exchange.fetch_ticker(symbol)
        last = float(ticker.get('last') or ticker.get('close') or sig.get('price') or 0)
        bid = float(ticker.get('bid') or 0)
        ask = float(ticker.get('ask') or 0)
        quote_volume = float(ticker.get('quoteVolume') or ticker.get('baseVolume') or 0)
        spread_pct = ((ask - bid) / max(last, 1e-9) * 100.0) if bid > 0 and ask > 0 else 0.0
        ticker_context = {
            'last': _safe_round_metric(last, 8),
            'bid': _safe_round_metric(bid, 8),
            'ask': _safe_round_metric(ask, 8),
            'spread_pct': _safe_round_metric(spread_pct, 4),
            'quote_volume': _safe_round_metric(quote_volume, 4),
            'percentage_24h': _safe_round_metric(ticker.get('percentage', 0), 3),
            'change_24h': _safe_round_metric(ticker.get('change', 0), 8),
            'high_24h': _safe_round_metric(ticker.get('high', 0), 8),
            'low_24h': _safe_round_metric(ticker.get('low', 0), 8),
            'mark_price': _safe_round_metric((ticker.get('info') or {}).get('markPrice', 0), 8),
            'index_price': _safe_round_metric((ticker.get('info') or {}).get('indexPrice', 0), 8),
            'raw_info': dict(ticker.get('info') or {}),
        }
    except Exception as ticker_e:
        ticker_context = {'error': str(ticker_e)[:180]}

    execution_context = {}
    try:
        execution_context = dict(exec_quality_snapshot(exchange, symbol, side) or {})
    except Exception as exec_e:
        execution_context = {'error': str(exec_e)[:180]}

    support = _safe_num(sig.get('support'), 0.0)
    resist = _safe_num(sig.get('resist'), 0.0)
    price = _safe_num(sig.get('price'), 0.0)
    breakout_ctx = {
        'support': _safe_round_metric(support, 8),
        'resistance': _safe_round_metric(resist, 8),
        'distance_to_support_pct': _safe_round_metric(((price - support) / max(price, 1e-9) * 100.0) if support > 0 and price > 0 else 0.0, 3),
        'distance_to_resistance_pct': _safe_round_metric(((resist - price) / max(price, 1e-9) * 100.0) if resist > 0 and price > 0 else 0.0, 3),
    }
    mtf_pressure_structure = {}
    for tf in ('15m', '1h', '4h', '1d'):
        snap = _mtf_pressure_structure_snapshot(raw_frames.get(tf), tf, price)
        if snap:
            mtf_pressure_structure[tf] = snap
    mtf_pressure_summary = _summarize_mtf_structure_pressure(mtf_pressure_structure, side)
    reference_context = dict(sig.get('external_reference') or sig.get('reference_context') or sig.get('scanner_reference') or {})
    tf_15m = dict(tf_data.get('15m') or {})
    tf_5m = dict(tf_data.get('5m') or {})
    tf_1h = dict(tf_data.get('1h') or {})
    tf_4h = dict(tf_data.get('4h') or {})
    tf_1d = dict(tf_data.get('1d') or {})
    momentum_context = {
        'long_score': _safe_round_metric(
            (1.0 if tf_15m.get('macd_hist', 0) > 0 else 0.0)
            + (1.0 if tf_15m.get('rsi', 50) >= 52 else 0.0)
            + (1.0 if tf_1h.get('trend_label') in ('uptrend', 'strong_uptrend', 'recovery_up') else 0.0)
            + (1.0 if tf_4h.get('trend_label') in ('uptrend', 'strong_uptrend', 'recovery_up') else 0.0),
            3,
        ),
        'short_score': _safe_round_metric(
            (1.0 if tf_15m.get('macd_hist', 0) < 0 else 0.0)
            + (1.0 if tf_15m.get('rsi', 50) <= 48 else 0.0)
            + (1.0 if tf_1h.get('trend_label') in ('downtrend', 'strong_downtrend', 'recovery_down') else 0.0)
            + (1.0 if tf_4h.get('trend_label') in ('downtrend', 'strong_downtrend', 'recovery_down') else 0.0),
            3,
        ),
        'signals': list(dict.fromkeys((radar.get('tags') or [])[:6] + [str(tf_15m.get('trend_label') or ''), str(tf_1h.get('trend_label') or '')])),
        'trend_4h_up': bool(tf_4h.get('trend_label') in ('uptrend', 'strong_uptrend', 'recovery_up')),
        'trend_1d_up': bool(tf_1d.get('trend_label') in ('uptrend', 'strong_uptrend', 'recovery_up')),
        'higher_lows': _safe_round_metric((tf_15m.get('recent_structure_low', 0) or 0) - (tf_5m.get('recent_structure_low', 0) or 0), 8),
        'lower_highs': _safe_round_metric((tf_15m.get('recent_structure_high', 0) or 0) - (tf_5m.get('recent_structure_high', 0) or 0), 8),
        'volume_build': bool((tf_5m.get('vol_ratio', 0) or 0) >= 1.15 or (tf_15m.get('vol_ratio', 0) or 0) >= 1.15),
        'compression': bool((tf_15m.get('bb_width_pct', 0) or 0) <= 6.0),
    }
    levels_context = {
        'nearest_support': _safe_round_metric((tf_15m.get('support_levels') or [support])[0] if (tf_15m.get('support_levels') or [support]) else support, 8),
        'nearest_resistance': _safe_round_metric((tf_15m.get('resistance_levels') or [resist])[0] if (tf_15m.get('resistance_levels') or [resist]) else resist, 8),
        'support_levels': list(tf_15m.get('support_levels') or [])[:3],
        'resistance_levels': list(tf_15m.get('resistance_levels') or [])[:3],
        'recent_high': _safe_round_metric(tf_15m.get('swing_high_20', resist), 8),
        'recent_low': _safe_round_metric(tf_15m.get('swing_low_20', support), 8),
        'dist_high_atr': _safe_round_metric(((tf_15m.get('swing_high_20', 0) or 0) - price) / max(tf_15m.get('atr', 0) or 1e-9, 1e-9) if price > 0 else 0.0, 3),
        'dist_low_atr': _safe_round_metric((price - (tf_15m.get('swing_low_20', 0) or 0)) / max(tf_15m.get('atr', 0) or 1e-9, 1e-9) if price > 0 else 0.0, 3),
    }
    microstructure_context = _build_market_microstructure_context(symbol, ticker_context, raw_frames)

    return {
        'style': {
            'holding_period': 'short_term_intraday',
            'trade_goal': 'short_term_perpetual_futures_entry',
            'decision_priority': 'entry_timing_momentum_structure_risk',
            'notes': [
                'This payload is for short-term crypto perpetual trading, not long-term investing.',
                'Use the multi-timeframe market structure and the latest closed candle shape heavily.',
                'Leverage is fixed by the bot to exchange max; do not become conservative by reducing leverage.',
                'Aggressive is acceptable only when price structure, liquidity, and trigger quality all align.',
                'If breakout chasing is justified, define exactly what price confirms the move and where chasing becomes too expensive.',
                'Use 1D and 4H for macro bias, 1H for trend continuation, 15m for entry location, 5m for confirmation, and 1m for micro-timing only.',
            ],
        },
        'signal_context': {
            'symbol': symbol,
            'side': side,
            'score': _safe_round_metric(sig.get('score'), 4),
            'raw_score': _safe_round_metric(sig.get('raw_score', sig.get('score')), 4),
            'priority_score': _safe_round_metric(sig.get('priority_score'), 4),
            'entry_quality': _safe_round_metric(sig.get('entry_quality'), 4),
            'current_price': _safe_round_metric(sig.get('price'), 8),
            'setup_label': str(sig.get('setup_label') or ''),
            'signal_grade': str(sig.get('signal_grade') or ''),
            'regime': str(sig.get('regime') or ''),
            'regime_confidence': _safe_round_metric(sig.get('regime_confidence'), 4),
            'trend_confidence': _safe_round_metric(sig.get('trend_confidence'), 4),
            'rotation_adj': _safe_round_metric(sig.get('rotation_adj'), 4),
            'score_jump': _safe_round_metric(sig.get('score_jump'), 4),
            'atr_15m': _safe_round_metric(sig.get('atr15'), 8),
            'atr_4h': _safe_round_metric(sig.get('atr4h'), 8),
            'signal_desc': str(sig.get('desc') or '')[:420],
        },
        'reference_trade_plan': {
            'machine_entry_hint': _safe_round_metric(sig.get('price'), 8),
            'machine_stop_loss_hint': _safe_round_metric(sig.get('stop_loss'), 8),
            'machine_take_profit_hint': _safe_round_metric(sig.get('take_profit'), 8),
            'machine_rr_hint': _safe_round_metric(sig.get('rr_ratio'), 4),
            'machine_est_pnl_pct_hint': _safe_round_metric(sig.get('est_pnl'), 4),
            'note': 'Machine-generated execution hints may be noisy. OpenAI should recalculate entry, stop, take profit, and RR from the full payload.',
        },
        'latest_closed_candle': dict(tf_15m.get('last_closed_candle') or {}),
        'momentum': momentum_context,
        'levels': levels_context,
        'market_state': {
            'broad_market': dict(market_info or {}),
            'ticker': ticker_context,
            'support_resistance': breakout_ctx,
        },
        'basic_market_data': dict(microstructure_context.get('basic_market_data') or {}),
        'liquidity_context': dict(microstructure_context.get('liquidity_context') or {}),
        'derivatives_context': dict(microstructure_context.get('derivatives_context') or {}),
        'news_context': dict(microstructure_context.get('news_context') or {}),
        'multi_timeframe': tf_data,
        'timeframe_bars': timeframe_bars,
        'multi_timeframe_pressure': mtf_pressure_structure,
        'multi_timeframe_pressure_summary': mtf_pressure_summary,
        'pre_breakout_radar': radar,
        'execution_context': execution_context,
        'reference_context': reference_context,
        'execution_policy': {
            'fixed_leverage': int(constraints.get('fixed_leverage', constraints.get('max_leverage', 1)) or 1),
            'leverage_mode': str(constraints.get('leverage_policy') or 'always_use_exchange_max'),
            'min_order_margin_usdt': _safe_round_metric(constraints.get('min_order_margin_usdt', 0.1), 4),
            'fixed_order_notional_usdt': _safe_round_metric(_fixed_order_notional_usdt_for_symbol(symbol), 4),
            'margin_pct_range': [
                _safe_round_metric(constraints.get('min_margin_pct', MIN_MARGIN_PCT), 4),
                _safe_round_metric(constraints.get('max_margin_pct', MAX_MARGIN_PCT), 4),
            ],
        },
    }


def _consult_openai_trade_for_signal(sig, rank_index, top_rows, market_info, risk_status, portfolio):
    top_candidates = []
    for row in list(top_rows or [])[:5]:
        top_candidates.append({
            'symbol': row.get('symbol'),
            'side': 'long' if float(row.get('score', 0) or 0) >= 0 else 'short',
            'score': round(float(row.get('score', 0) or 0), 2),
            'priority_score': round(float(row.get('priority_score', 0) or 0), 2),
            'entry_quality': round(float(row.get('entry_quality', 0) or 0), 2),
            'rr_ratio': round(float(row.get('rr_ratio', 0) or 0), 2),
            'candidate_source': str(row.get('candidate_source') or row.get('source') or 'normal')[:60],
        })
    fixed_leverage = _get_symbol_max_leverage(sig.get('symbol'))
    sig = dict(sig or {})
    fixed_order_notional_usdt = _fixed_order_notional_usdt_for_symbol(sig.get('symbol'))
    constraints = {
        'min_margin_pct': max(MIN_MARGIN_PCT, float(OPENAI_TRADE_CONFIG.get('min_margin_pct', MIN_MARGIN_PCT) or MIN_MARGIN_PCT)),
        'max_margin_pct': min(MAX_MARGIN_PCT, float(OPENAI_TRADE_CONFIG.get('max_margin_pct', MAX_MARGIN_PCT) or MAX_MARGIN_PCT)),
        'min_leverage': fixed_leverage,
        'max_leverage': fixed_leverage,
        'fixed_leverage': fixed_leverage,
        'leverage_policy': 'always_use_exchange_max',
        'min_order_margin_usdt': 0.1,
        'fixed_order_notional_usdt': fixed_order_notional_usdt,
        'trade_style': 'short_term_intraday',
        'max_open_positions': MAX_OPEN_POSITIONS,
        'max_same_direction': MAX_SAME_DIRECTION,
    }
    sig['openai_market_context'] = _build_openai_short_term_context(sig, market_info, constraints)
    candidate = build_openai_trade_candidate(
        signal=sig,
        market=market_info,
        risk_status=risk_status,
        portfolio=portfolio,
        top_candidates=top_candidates,
        constraints=constraints,
        rank_index=rank_index,
    )
    with OPENAI_TRADE_LOCK:
        global OPENAI_TRADE_STATE
        OPENAI_TRADE_STATE, result = consult_trade_decision(
            state=OPENAI_TRADE_STATE,
            state_path=OPENAI_TRADE_STATE_PATH,
            api_key=OPENAI_API_KEY,
            config=OPENAI_TRADE_CONFIG,
            candidate=candidate,
            logger=print,
            now_ts=time.time(),
        )
        save_trade_state(OPENAI_TRADE_STATE_PATH, OPENAI_TRADE_STATE)
    sync_openai_trade_state(push_runtime=False)
    return candidate, result


def _openai_pending_advice_map(state=None):
    src = state if isinstance(state, dict) else OPENAI_TRADE_STATE
    pending = src.setdefault('pending_advice', {})
    if not isinstance(pending, dict):
        src['pending_advice'] = {}
        pending = src['pending_advice']
    return pending


def _side_from_signal(sig):
    return 'long' if float((sig or {}).get('score', 0) or 0) >= 0 else 'short'


def _float_or_zero(value):
    try:
        value = float(value or 0)
        return value if math.isfinite(value) else 0.0
    except Exception:
        return 0.0


def _derive_openai_watch_plan(sig, decision):
    decision = dict(decision or {})
    side = _side_from_signal(sig)
    trigger_type = str(decision.get('watch_trigger_type') or 'none').strip()
    trigger_price = _float_or_zero(decision.get('watch_trigger_price'))
    invalidation_price = _float_or_zero(decision.get('watch_invalidation_price'))
    zone_low = _float_or_zero(decision.get('watch_price_zone_low'))
    zone_high = _float_or_zero(decision.get('watch_price_zone_high'))
    entry_price = _float_or_zero(decision.get('entry_price'))
    chase_price = _float_or_zero(decision.get('chase_trigger_price'))
    if trigger_type == 'none':
        if str(decision.get('order_type') or '').lower() == 'limit' and entry_price > 0:
            trigger_type = 'pullback_to_entry'
            trigger_price = entry_price
        elif bool(decision.get('chase_if_triggered')) and chase_price > 0:
            trigger_type = 'breakout_reclaim' if side == 'long' else 'breakdown_confirm'
            trigger_price = chase_price
        elif str(decision.get('if_missed_plan') or decision.get('entry_plan') or '').strip():
            trigger_type = 'manual_review'
    if trigger_type != 'manual_review' and trigger_price <= 0:
        if zone_low > 0 and zone_high > 0:
            trigger_price = (zone_low + zone_high) / 2.0
        else:
            trigger_type = 'none'
    if trigger_type == 'none':
        trigger_price = 0.0
        invalidation_price = 0.0
    return trigger_type, trigger_price, invalidation_price


def _store_openai_pending_advice(sig, decision, candidate=None, result=None, now_ts=None):
    if not decision or bool(decision.get('should_trade')):
        return None
    symbol = str((sig or {}).get('symbol') or '').strip()
    if not symbol:
        return None
    now_ts = float(now_ts or time.time())
    trigger_type, trigger_price, invalidation_price = _derive_openai_watch_plan(sig, decision)
    fallback_generated = bool((result or {}).get('fallback_used'))
    if fallback_generated:
        trigger_type = 'manual_review'
    if trigger_type == 'none':
        fallback_note = str(
            decision.get('watch_note')
            or decision.get('recheck_reason')
            or decision.get('reason_to_skip')
            or decision.get('entry_plan')
            or decision.get('if_missed_plan')
            or ''
        ).strip()
        if not fallback_note:
            return None
        trigger_type = 'manual_review'
        trigger_price = 0.0
        invalidation_price = round(_float_or_zero(decision.get('watch_invalidation_price') or decision.get('stop_loss')), 8)
    ttl_sec = max(int(OPENAI_TRADE_CONFIG.get('advice_ttl_minutes', 240) or 240), 15) * 60
    previous = dict((sig or {}).get('pending_openai_advice') or {})
    last_trigger_ts = now_ts if bool((sig or {}).get('force_openai_recheck')) else _float_or_zero(previous.get('last_trigger_ts'))
    trigger_count = int(previous.get('trigger_count', 0) or 0)
    advice = {
        'symbol': symbol,
        'side': _side_from_signal(sig),
        'source': str((sig or {}).get('candidate_source') or (sig or {}).get('source') or 'normal')[:60],
        'scanner_intent': str((sig or {}).get('scanner_intent') or '')[:120],
        'status': 'watching',
        'generated_by_fallback': fallback_generated,
        'created_ts': now_ts,
        'expires_ts': now_ts + ttl_sec,
        'last_checked_ts': now_ts,
        'last_trigger_ts': last_trigger_ts,
        'last_trigger_price': round(_float_or_zero(previous.get('last_trigger_price')), 8),
        'trigger_count': trigger_count,
        'trigger_armed': bool(previous.get('trigger_armed', True)),
        'trigger_type': trigger_type,
        'trigger_price': round(trigger_price, 8),
        'invalidation_price': round(invalidation_price, 8),
        'entry_price': round(_float_or_zero(decision.get('entry_price')), 8),
        'stop_loss': round(_float_or_zero(decision.get('stop_loss')), 8),
        'take_profit': round(_float_or_zero(decision.get('take_profit')), 8),
        'score': round(abs(float((sig or {}).get('score', 0) or 0)), 4),
        'priority_score': round(float((sig or {}).get('priority_score', 0) or 0), 4),
        'entry_quality': round(float((sig or {}).get('entry_quality', 0) or 0), 4),
        'reason_to_skip': str(decision.get('reason_to_skip') or '')[:260],
        'entry_plan': str(decision.get('entry_plan') or '')[:260],
        'if_missed_plan': str(decision.get('if_missed_plan') or '')[:220],
        'watch_note': str(decision.get('watch_note') or decision.get('entry_plan') or '')[:220],
        'recheck_reason': str(decision.get('recheck_reason') or decision.get('if_missed_plan') or '')[:220],
        'watch_timeframe': str(decision.get('watch_timeframe') or '')[:80],
        'watch_price_zone_low': round(_float_or_zero(decision.get('watch_price_zone_low')), 8),
        'watch_price_zone_high': round(_float_or_zero(decision.get('watch_price_zone_high')), 8),
        'watch_structure_condition': str(decision.get('watch_structure_condition') or '')[:220],
        'watch_volume_condition': str(decision.get('watch_volume_condition') or '')[:220],
        'watch_checklist': [str(x).strip()[:160] for x in list(decision.get('watch_checklist') or []) if str(x).strip()][:8],
        'watch_confirmations': [str(x).strip()[:160] for x in list(decision.get('watch_confirmations') or []) if str(x).strip()][:8],
        'watch_invalidations': [str(x).strip()[:160] for x in list(decision.get('watch_invalidations') or []) if str(x).strip()][:8],
        'watch_recheck_priority': round(float(decision.get('watch_recheck_priority', 0) or 0), 2),
        'market_read': str(decision.get('market_read') or '')[:220],
        'reference_summary': str(decision.get('reference_summary') or '')[:220],
        'model': str(((result or {}).get('symbol_state') or {}).get('last_model') or OPENAI_TRADE_CONFIG.get('model') or '')[:60],
        'current_price': round(_float_or_zero((sig or {}).get('price')), 8),
        'distance_pct': 0.0,
        'last_check_note': 'stored from OpenAI skip decision',
    }
    with OPENAI_TRADE_LOCK:
        pending = _openai_pending_advice_map(OPENAI_TRADE_STATE)
        pending[symbol] = advice
        save_trade_state(OPENAI_TRADE_STATE_PATH, OPENAI_TRADE_STATE)
    sync_openai_trade_state(push_runtime=False)
    return advice


def _refresh_openai_pending_advice_watch(sig, advice, now_ts=None, note='watching'):
    sig = dict(sig or {})
    advice = dict(advice or {})
    symbol = str(sig.get('symbol') or advice.get('symbol') or '').strip()
    if not symbol:
        return advice
    now_ts = float(now_ts or time.time())
    price = _float_or_zero(sig.get('price'))
    trigger_price = _float_or_zero(advice.get('trigger_price'))
    side = str(advice.get('side') or _side_from_signal(sig)).lower()
    trigger_type = str(advice.get('trigger_type') or 'none')
    zone_low = _float_or_zero(advice.get('watch_price_zone_low'))
    zone_high = _float_or_zero(advice.get('watch_price_zone_high'))
    distance_pct = 0.0
    if price > 0 and trigger_price > 0:
        distance_pct = abs(price - trigger_price) / max(trigger_price, 1e-9) * 100.0
    in_zone = bool(zone_low > 0 and zone_high > 0 and min(zone_low, zone_high) <= price <= max(zone_low, zone_high))
    condition_active = False
    if trigger_type == 'pullback_to_entry':
        condition_active = in_zone or (price <= trigger_price if side == 'long' else price >= trigger_price)
    elif trigger_type == 'breakout_reclaim':
        condition_active = in_zone or (price >= trigger_price if side == 'long' else price <= trigger_price)
    elif trigger_type == 'breakdown_confirm':
        condition_active = in_zone or (price <= trigger_price if side == 'short' else price >= trigger_price)
    elif trigger_type == 'volume_confirmation':
        condition_active = bool(trigger_price > 0) and (price >= trigger_price if side == 'long' else price <= trigger_price)
    elif trigger_type == 'manual_review':
        condition_active = in_zone or (bool(trigger_price > 0) and (price >= trigger_price if side == 'long' else price <= trigger_price))
    updates = {
        'status': str(note or 'watching'),
        'last_checked_ts': now_ts,
        'last_checked_at': tw_now_str(),
        'current_price': round(price, 8),
        'distance_pct': round(distance_pct, 4),
        'latest_score': round(abs(float(sig.get('score', 0) or 0)), 4),
        'latest_priority_score': round(float(sig.get('priority_score', 0) or 0), 4),
        'latest_entry_quality': round(float(sig.get('entry_quality', 0) or 0), 4),
    }
    if not bool(advice.get('trigger_armed', True)) and not condition_active:
        updates['trigger_armed'] = True
        updates['last_check_note'] = 'watch reset completed; waiting for a fresh trigger'
    _mark_openai_pending_advice(symbol, updates)
    advice.update(updates)
    return advice


def _clear_openai_pending_advice(symbol):
    symbol = str(symbol or '').strip()
    if not symbol:
        return
    with OPENAI_TRADE_LOCK:
        pending = _openai_pending_advice_map(OPENAI_TRADE_STATE)
        if symbol in pending:
            pending.pop(symbol, None)
            save_trade_state(OPENAI_TRADE_STATE_PATH, OPENAI_TRADE_STATE)
    sync_openai_trade_state(push_runtime=False)


def _get_openai_pending_advice(symbol, now_ts=None):
    symbol = str(symbol or '').strip()
    now_ts = float(now_ts or time.time())
    with OPENAI_TRADE_LOCK:
        pending = _openai_pending_advice_map(OPENAI_TRADE_STATE)
        advice = dict(pending.get(symbol) or {})
        if not advice:
            return None
        if bool(advice.get('generated_by_fallback')):
            pending.pop(symbol, None)
            save_trade_state(OPENAI_TRADE_STATE_PATH, OPENAI_TRADE_STATE)
            sync_openai_trade_state(push_runtime=False)
            return None
        expires_ts = _float_or_zero(advice.get('expires_ts'))
        if expires_ts > 0 and now_ts > expires_ts:
            advice['status'] = 'expired'
            pending.pop(symbol, None)
            save_trade_state(OPENAI_TRADE_STATE_PATH, OPENAI_TRADE_STATE)
            sync_openai_trade_state(push_runtime=False)
            return None
    return advice


def _pending_advice_should_block(sig, advice, now_ts=None):
    sig = dict(sig or {})
    advice = dict(advice or {})
    if not advice:
        return False
    now_ts = float(now_ts or time.time())
    symbol = str(sig.get('symbol') or '').strip()
    advice_symbol = str(advice.get('symbol') or '').strip()
    if symbol and advice_symbol and symbol != advice_symbol:
        return False
    expires_ts = _float_or_zero(advice.get('expires_ts'))
    if expires_ts > 0 and now_ts > expires_ts:
        return False
    status = str(advice.get('status') or '').strip().lower()
    if status in {'expired', 'cleared', 'filled', 'completed'}:
        return False
    current_side = _side_from_signal(sig)
    advice_side = str(advice.get('side') or current_side).strip().lower()
    if advice_side and advice_side != current_side:
        return False
    current_source = str(sig.get('candidate_source') or sig.get('source') or 'normal').strip().lower()
    advice_source = str(advice.get('source') or current_source).strip().lower()
    if advice_source and current_source and advice_source != current_source:
        return False
    trigger_type = str(advice.get('trigger_type') or 'none').strip().lower()
    trigger_price = _float_or_zero(advice.get('trigger_price'))
    zone_low = _float_or_zero(advice.get('watch_price_zone_low'))
    zone_high = _float_or_zero(advice.get('watch_price_zone_high'))
    has_zone = bool(zone_low > 0 and zone_high > 0)
    actionable_trigger = trigger_type != 'none' and (
        trigger_type == 'manual_review' or trigger_price > 0 or has_zone
    )
    if not actionable_trigger:
        return False
    fallback_generated = bool(advice.get('generated_by_fallback'))
    created_ts = _float_or_zero(advice.get('created_ts'))
    age_sec = max(now_ts - created_ts, 0.0) if created_ts > 0 else 0.0
    manual_block_sec = max(int(float(OPENAI_TRADE_CONFIG.get('manual_review_block_minutes', 20) or 20)), 5) * 60
    if trigger_type == 'manual_review' and trigger_price <= 0 and not has_zone:
        return False
    return True


def _mark_openai_pending_advice(symbol, updates):
    symbol = str(symbol or '').strip()
    if not symbol:
        return
    with OPENAI_TRADE_LOCK:
        pending = _openai_pending_advice_map(OPENAI_TRADE_STATE)
        row = dict(pending.get(symbol) or {})
        if not row:
            return
        row.update(dict(updates or {}))
        pending[symbol] = row
        save_trade_state(OPENAI_TRADE_STATE_PATH, OPENAI_TRADE_STATE)
    sync_openai_trade_state(push_runtime=False)


def _openai_pending_advice_trigger(sig, advice, now_ts=None):
    sig = dict(sig or {})
    advice = dict(advice or {})
    now_ts = float(now_ts or time.time())
    price = _float_or_zero(sig.get('price'))
    side = str(advice.get('side') or _side_from_signal(sig)).lower()
    trigger_type = str(advice.get('trigger_type') or 'none')
    trigger_price = _float_or_zero(advice.get('trigger_price'))
    invalidation_price = _float_or_zero(advice.get('invalidation_price'))
    zone_low = _float_or_zero(advice.get('watch_price_zone_low'))
    zone_high = _float_or_zero(advice.get('watch_price_zone_high'))
    fallback_generated = bool(advice.get('generated_by_fallback'))
    last_trigger_ts = _float_or_zero(advice.get('last_trigger_ts'))
    if not bool(advice.get('trigger_armed', True)):
        return False, 'pending advice awaiting reset'
    min_recheck_gap = OPENAI_PENDING_RECHECK_MIN_GAP_SEC
    if last_trigger_ts > 0 and (now_ts - last_trigger_ts) < min_recheck_gap:
        return False, 'pending advice recently rechecked'
    if price <= 0 or trigger_type == 'none':
        return False, ''
    if side == 'long' and invalidation_price > 0 and price <= invalidation_price:
        return False, 'pending advice invalidated'
    if side == 'short' and invalidation_price > 0 and price >= invalidation_price:
        return False, 'pending advice invalidated'
    in_zone = bool(zone_low > 0 and zone_high > 0 and min(zone_low, zone_high) <= price <= max(zone_low, zone_high))
    if trigger_type == 'pullback_to_entry':
        hit = in_zone or (price <= trigger_price if side == 'long' else price >= trigger_price)
    elif trigger_type == 'breakout_reclaim':
        hit = in_zone or (price >= trigger_price if side == 'long' else price <= trigger_price)
    elif trigger_type == 'breakdown_confirm':
        hit = in_zone or (price <= trigger_price if side == 'short' else price >= trigger_price)
    elif trigger_type == 'volume_confirmation':
        score_now = abs(float(sig.get('score', 0) or 0))
        quality_now = float(sig.get('entry_quality', 0) or 0)
        hit_price = (price >= trigger_price if side == 'long' else price <= trigger_price) if trigger_price > 0 else False
        hit = hit_price or score_now >= max(float(advice.get('score', 0) or 0) + 4.0, 52.0) or quality_now >= 4.0
    elif trigger_type == 'manual_review':
        score_now = abs(float(sig.get('score', 0) or 0))
        priority_now = float(sig.get('priority_score', 0) or 0)
        improve_hit = (
            score_now >= max(float(advice.get('score', 0) or 0) + 6.0, 54.0)
            or priority_now >= float(advice.get('priority_score', 0) or 0) + 6.0
        )
        price_hit = in_zone
        if trigger_price > 0:
            price_hit = price_hit or (price >= trigger_price if side == 'long' else price <= trigger_price)
        if fallback_generated:
            hit = price_hit and improve_hit
        else:
            if trigger_price > 0 or in_zone:
                hit = price_hit and improve_hit
            else:
                hit = improve_hit
    else:
        hit = False
    if hit:
        return True, '{} hit at {}'.format(trigger_type, round(price, 8))
    return False, ''


def _attach_openai_pending_reference(sig, advice, trigger_reason):
    sig = dict(sig or {})
    advice = dict(advice or {})
    ref = dict(sig.get('external_reference') or sig.get('reference_context') or {})
    ref.update({
        'source': 'stored_openai_pending_advice',
        'summary': 'The bot observed the previous OpenAI watch condition; re-check now before any order.',
        'setup': str(advice.get('entry_plan') or advice.get('watch_note') or '')[:220],
        'risk': str(advice.get('reason_to_skip') or '')[:220],
        'note': 'The condition you previously asked us to wait for is now met: {} | trigger={} price={} invalidation={} entry={} sl={} tp={} recheck={}'.format(
            trigger_reason,
            advice.get('trigger_type', ''),
            advice.get('trigger_price', 0),
            advice.get('invalidation_price', 0),
            advice.get('entry_price', 0),
            advice.get('stop_loss', 0),
            advice.get('take_profit', 0),
            str(advice.get('recheck_reason') or advice.get('if_missed_plan') or '')[:120],
        ),
        'checklist': ' / '.join(list(advice.get('watch_checklist') or [])[:4]),
        'confirmations': ' / '.join(list(advice.get('watch_confirmations') or [])[:4]),
        'invalidations': ' / '.join(list(advice.get('watch_invalidations') or [])[:4]),
    })
    sig['external_reference'] = ref
    sig['pending_openai_advice'] = advice
    sig['pending_openai_trigger_reason'] = trigger_reason
    sig['candidate_source'] = str(advice.get('source') or sig.get('candidate_source') or sig.get('source') or 'pending_advice')
    sig['source'] = sig['candidate_source']
    if advice.get('scanner_intent'):
        sig['scanner_intent'] = str(advice.get('scanner_intent') or '')
    return sig


def _symbol_family_key(symbol=''):
    base = str(symbol or '').split('/')[0].split(':')[0].upper()
    base = re.sub(r'^\d+', '', base)
    return base or str(symbol or '').upper()


def _signal_related_bucket(sig):
    sig = dict(sig or {})
    bd = dict(sig.get('breakdown') or {})
    side = 'long' if float(sig.get('score', 0) or 0) >= 0 else 'short'
    source = str(sig.get('candidate_source') or sig.get('source') or 'normal')
    setup = str(sig.get('setup_label') or bd.get('Setup') or '')
    regime = str(sig.get('regime') or bd.get('Regime') or '')
    market_state = str(bd.get('MarketState') or '')
    pct_24h = _float_or_zero((sig.get('short_gainer_context') or {}).get('pct_24h'))
    if pct_24h <= 0:
        pct_24h = _float_or_zero((sig.get('marketability') or {}).get('change_pct_24h'))
    pct_bucket = int(abs(pct_24h) // 8) if pct_24h else 0
    return (source, side, setup, regime, market_state, pct_bucket)


def _row_priority_for_related_dedupe(row):
    base = float((row or {}).get('priority_score', abs(float((row or {}).get('score', 0) or 0))) or 0)
    if (row or {}).get('_pending_openai_advice'):
        base += 50.0
    if (row or {}).get('_short_gainer_initial_review'):
        base += 20.0
    return round(base, 4)


def _dedupe_highly_related_signals(rows, max_per_family=1, max_per_bucket=2):
    kept = []
    family_seen = {}
    bucket_seen = {}
    for row in sorted(list(rows or []), key=_row_priority_for_related_dedupe, reverse=True):
        symbol = str((row or {}).get('symbol') or '')
        if not symbol:
            continue
        family = _symbol_family_key(symbol)
        bucket = _signal_related_bucket(row)
        if family_seen.get(family, 0) >= max_per_family:
            continue
        if bucket_seen.get(bucket, 0) >= max_per_bucket:
            continue
        clone = dict(row or {})
        clone['correlation_family'] = family
        clone['correlation_bucket'] = '|'.join(str(x) for x in bucket if str(x))
        kept.append(clone)
        family_seen[family] = family_seen.get(family, 0) + 1
        bucket_seen[bucket] = bucket_seen.get(bucket, 0) + 1
    return kept


def scan_thread():
    print("鎺冩弿鍩疯绶掑暉鍕曪紝绛夊緟10绉掕畵鍏朵粬鍩疯绶掑氨绶?..")
    update_state(scan_progress="鎺冩弿鍩疯绶掑暉鍕曚腑锛?0绉掑緦闁嬪绗?杓?..")
    _touch_backend_thread('scan', '鍟熷嫊涓紝绛夊緟鍏朵粬鍩疯绶掑氨绶?)
    time.sleep(10)
    _refresh_learn_summary()
    while True:
        try:
            _touch_backend_thread('scan', '婧栧倷闁嬪鏂颁竴杓競鍫存巸鎻?)
            # 姣忚吉闁嬪娓呯┖鏈吉涓嬪柈瑷橀寗
            with _ORDERED_LOCK:
                _ORDERED_THIS_SCAN.clear()
            AUTO_ORDER_AUDIT.clear()

            sc = STATE.get("scan_count", 0)
            print("=== 闁嬪绗瑊}杓巸鎻?=== 鏅傞枔:{}".format(sc+1, tw_now_str()))
            update_state(
                scan_progress="绗瑊}杓細鎶撳彇甯傚牬鏁告摎... {}".format(sc+1, tw_now_str()),
                last_update=tw_now_str()
            )
            _touch_backend_thread('scan', '绗瑊}杓細鎶撳彇甯傚牬璩囨枡'.format(sc + 1))
            try:
                tickers = exchange.fetch_tickers()
                print("fetch_tickers 鎴愬姛锛屽叡 {} 鍊嬪梗".format(len(tickers)))
            except Exception as ft_e:
                print("fetch_tickers 澶辨晽锛?0绉掑緦閲嶈│: {}".format(ft_e))
                time.sleep(10)
                continue

            scan_limit = max(20, int(COIN_SELECTOR_SCAN_LIMIT or 100))
            ranked = rank_tradable_markets(tickers, limit=COIN_SELECTOR_PREFILTER_LIMIT)

            # 鎺掗櫎鑲＄エ浠ｅ梗锛堝彧淇濈暀鍔犲瘑璨ㄥ梗锛?            STOCK_TOKENS = {
                'AAPL','GOOGL','GOOG','AMZN','TSLA','MSFT','META','NVDA','NFLX',
                'BABA','BIDU','JD','PDD','NIO','XPEV','LI','SNAP','TWTR','UBER',
                'LYFT','ABNB','COIN','HOOD','AMC','GME','SPY','QQQ','DJI',
                'MSTR','PLTR','SQ','PYPL','SHOP','INTC','AMD','QCOM','AVGO',
            }
            def is_crypto(sym):
                base = sym.split('/')[0].split(':')[0]
                if base in STOCK_TOKENS:
                    return False
                # 鎺掗櫎鍚皬鏁搁粸鎴栫湅璧蜂締鍍忚偂绁ㄧ殑锛堝 1000BONK 鏄梗锛?                return True

            marketability_by_symbol = {}
            symbols = []
            for sym, _, marketability in ranked:
                if not is_crypto(sym):
                    continue
                symbols.append(sym)
                marketability_by_symbol[sym] = marketability
                if len(symbols) >= scan_limit:
                    break
            print("鏈吉鎺冩弿 {} 鍊嬪梗".format(len(symbols)))
            _touch_backend_thread('scan', '绗瑊}杓細鍒嗘瀽 {} 鍊嬪梗'.format(sc + 1, len(symbols)))

            sigs=[]
            with LEARN_LOCK:
                sym_stats=LEARN_DB.get("symbol_stats",{})
            blocked_syms={s for s,v in sym_stats.items()
                          if v.get("count",0)>=7 and v.get("win",0)/v["count"]<0.4}

            for i,sym in enumerate(symbols):
                update_state(scan_progress="鎺冩弿 {}/{}锛歿}".format(i+1, len(symbols), sym))
                try:
                    time.sleep(0.5)  # 骞ｈ垏骞ｄ箣闁撻枔闅?.5绉掞紝閬垮厤rate limit
                    sc,desc,pr,sl,tp,ep,bd,atr,atr15,atr4h,sl_m,tp_m = analyze(sym)
                    allowed,sym_n,sym_wr=is_symbol_allowed(sym)
                    status="瑙€瀵熶腑(鍕濈巼{}%)".format(sym_wr) if not allowed else ""
                    if abs(sc)>=8:
                        stable_score = smooth_signal_score(sym, sc)
                        SIGNAL_META_CACHE[sym] = {
                            "atr": atr, "atr15": atr15, "atr4h": atr4h, "price": pr,
                            "marketability": marketability_by_symbol.get(sym, {}),
                            "raw_score": sc, "stable_score": stable_score, "updated_at": tw_now_str(), "ts": time.time(),
                            "setup_label": bd.get("Setup", ""),
                            "signal_grade": bd.get("绛夌礆", ""),
                            "direction_confidence": (lambda _dc, _tc: round(float(_dc if _dc not in (None, '', 0, 0.0) else float(_tc or 0) / 10.0), 1))(bd.get("鏂瑰悜淇″績"), bd.get("TrendConfidence", 0)),
                            "entry_quality": bd.get("閫插牬鍝佽唱", 0),
                            "rr_ratio": bd.get("RR", 0),
                            "regime": bd.get("Regime", "neutral"),
                            "regime_confidence": bd.get("RegimeConfidence", bd.get("TrendConfidence", bd.get("鏂瑰悜淇″績", 0))),
                        }
                        sigs.append({
                            "symbol":sym,"score":stable_score,"raw_score":sc,"desc":desc,"price":pr,
                            "stop_loss":sl,"take_profit":tp,"est_pnl":ep,
                            "direction":"鍋氬 鈻? if stable_score>0 else "鍋氱┖ 鈻?,
                            "breakdown": bd,
"atr": atr,
"atr15": atr15,
"atr4h": atr4h,
"sl_mult": sl_m,
"tp_mult": tp_m,
                            "allowed":allowed,"status":status,
                            "sym_trades":sym_n,"sym_wr":sym_wr,
                            "margin_pct": 0,
                            "entry_quality": bd.get("閫插牬鍝佽唱", 0),
                            "rr_ratio": bd.get("RR", 0),
                            "regime_bias": bd.get("RegimeBias", 0),
                            "setup_label": bd.get("Setup", ""),
                            "signal_grade": bd.get("绛夌礆", ""),
                            "direction_confidence": (lambda _dc, _tc: round(float(_dc if _dc not in (None, '', 0, 0.0) else float(_tc or 0) / 10.0), 1))(bd.get("鏂瑰悜淇″績"), bd.get("TrendConfidence", 0)),
                            "regime": bd.get("Regime", "neutral"),
                            "regime_confidence": bd.get("RegimeConfidence", bd.get("TrendConfidence", bd.get("鏂瑰悜淇″績", 0))),
                            "trend_confidence": bd.get("TrendConfidence", bd.get("鏂瑰悜淇″績", 0)),
                            "score_jump": score_jump_alert(sym, sc, stable_score),
                            "marketability": marketability_by_symbol.get(sym, {}),
                            "marketability_score": (marketability_by_symbol.get(sym, {}) or {}).get("score", 0.0),
                        })
                except Exception as sym_e:
                    print("鍒嗘瀽 {} 澶辨晽璺抽亷: {}".format(sym, sym_e))
                if i%5==0: gc.collect()

            for s in sigs:
                try:
                    ctx = infer_margin_context(s, same_side_count=0)
                    s['margin_pct'] = ctx.get('margin_pct', RISK_PCT)
                    s['margin_ctx'] = ctx
                except:
                    s['margin_pct'] = RISK_PCT
                try:
                    rot_adj, rot_notes = _symbol_rotation_adjustment(s.get('symbol', ''))
                except Exception:
                    rot_adj, rot_notes = 0.0, []
                s['rotation_adj'] = rot_adj
                s['rotation_notes'] = rot_notes
                selection_edge, selection_notes = coin_selection_edge(s)
                s['selection_edge'] = selection_edge
                s['selection_notes'] = selection_notes
                s['priority_score'] = round(abs(float(s.get('score', 0) or 0)) + rot_adj + selection_edge + float(s.get('entry_quality', 0) or 0) * 0.15 + min(float(s.get('rr_ratio', 0) or 0), 3.0) * 0.12, 2)

            sigs = _dedupe_highly_related_signals(sigs, max_per_family=1, max_per_bucket=2)

            # 鍒嗛枊鎺掑簭锛氬闋彇鍓?锛岀┖闋彇鍓?锛屾帓琛屾椤ず10鍊?            long_sigs  = sorted([s for s in sigs if s['score']>0], key=lambda x:(x.get('priority_score', abs(x['score'])), x['score']), reverse=True)[:6]
            short_gainer_signals = []
            short_gainer_rows = []
            if OPENAI_SHORT_GAINERS_ENABLE:
                try:
                    sig_by_symbol = {str(s.get('symbol') or ''): s for s in sigs}
                    short_gainer_rows = rank_short_gainer_markets(
                        tickers,
                        limit=OPENAI_SHORT_GAINERS_PREFILTER,
                        min_pct=OPENAI_SHORT_GAINERS_MIN_24H_PCT,
                    )
                    for _row in short_gainer_rows[:OPENAI_SHORT_GAINERS_LIMIT]:
                        _sym = str((_row or [''])[0] or '')
                        if not _sym or _sym in SHORT_TERM_EXCLUDED:
                            continue
                        _candidate = build_short_gainer_signal(_row, existing=sig_by_symbol.get(_sym))
                        if not _candidate:
                            continue
                        short_gainer_signals.append(_candidate)
                        marketability_by_symbol[_sym] = dict(_candidate.get('marketability') or marketability_by_symbol.get(_sym, {}))
                    short_gainer_signals = _dedupe_highly_related_signals(short_gainer_signals, max_per_family=1, max_per_bucket=1)
                    with AI_LOCK:
                        AI_PANEL['short_gainers'] = {
                            'enabled': True,
                            'status': 'watching',
                            'last_update': tw_now_str(),
                            'min_24h_pct': OPENAI_SHORT_GAINERS_MIN_24H_PCT,
                            'candidate_count': len(short_gainer_signals),
                            'send_limit_per_scan': OPENAI_SHORT_GAINERS_MAX_PER_SCAN,
                            'candidates': [
                                {
                                    'symbol': s.get('symbol'),
                                    'pct_24h': (s.get('short_gainer_context') or {}).get('pct_24h', 0),
                                    'score': abs(float(s.get('score', 0) or 0)),
                                    'price': s.get('price'),
                                    'trigger_price': s.get('local_watch_trigger_price', 0),
                                    'invalidation_price': s.get('local_watch_invalidation_price', 0),
                                    'status': s.get('watch_status') or 'local_watch',
                                }
                                for s in short_gainer_signals[:OPENAI_SHORT_GAINERS_LIMIT]
                            ],
                        }
                        _ai_panel_snapshot = dict(AI_PANEL)
                    update_state(ai_panel=_ai_panel_snapshot)
                    if short_gainer_signals:
                        print("OpenAI short gainer candidates prepared={}".format(len(short_gainer_signals)))
                except Exception as _sg_err:
                    print("short gainer leaderboard build failed: {}".format(_sg_err))
                    with AI_LOCK:
                        AI_PANEL['short_gainers'] = {
                            'enabled': True,
                            'status': 'error',
                            'last_update': tw_now_str(),
                            'error': str(_sg_err)[:180],
                        }
                        _ai_panel_snapshot = dict(AI_PANEL)
                    update_state(ai_panel=_ai_panel_snapshot)
            else:
                with AI_LOCK:
                    AI_PANEL['short_gainers'] = {'enabled': False, 'status': 'disabled', 'last_update': tw_now_str()}
                    _ai_panel_snapshot = dict(AI_PANEL)
                update_state(ai_panel=_ai_panel_snapshot)

            long_sigs  = sorted([s for s in sigs if s['score']>0], key=lambda x:(x.get('priority_score', abs(x['score'])), x['score']), reverse=True)[:6]
            short_sigs = sorted([s for s in sigs if s['score']<0], key=lambda x:(x.get('priority_score', abs(x['score'])), abs(x['score'])), reverse=True)[:4]
            top10 = sorted(long_sigs + short_sigs, key=lambda x:(x.get('priority_score', abs(x['score'])), abs(x['score'])), reverse=True)[:10]
            for _idx, _row in enumerate(top10):
                try:
                    _row['_normal_rank_index'] = _idx
                except Exception:
                    pass
            top7  = top10  # 鎺掕姒滈’绀?0鍊?            print("姝ラA: 鎺掕姒滄帓搴忓畬鎴愶紝鍏眥}鍊嬩俊铏?.format(len(top7)))
            _touch_backend_thread('scan', '绗瑊}杓細鎺掕姒滃畬鎴愶紝鍏?{} 鍊嬩綆鐩搁棞鍊欓伕'.format(sc + 1, len(top7)))
            with STATE_LOCK:
                STATE["top_signals"]=top10; STATE["scan_count"]+=1
                STATE["short_gainer_signals"]=short_gainer_signals[:OPENAI_SHORT_GAINERS_LIMIT]
                STATE["last_update"]=tw_now_str()
                STATE["scan_progress"]="绗瑊}杓畬鎴?| {} | 闁€妾?{}鍒?.format(STATE["scan_count"],STATE["last_update"],ORDER_THRESHOLD)
                STATE["auto_order_audit"]=dict(AUTO_ORDER_AUDIT)
            print("姝ラB: STATE鏇存柊瀹屾垚")

            with STATE_LOCK:
                active_pos = list(STATE["active_positions"])
                pos_syms   = {p['symbol'] for p in active_pos}
                pos_cnt    = len(active_pos)
            print("姝ラC: 鎸佸€夎畝鍙栧畬鎴愶紝鍏眥}鍊?.format(pos_cnt))

            # 鈹€鈹€ 鍙嶅悜鍋垫脯锛氭寔鍊夋柟鍚戣垏鏂拌▕铏熸柟鍚戠浉鍙?鈫?鍙钩鍊夛紝涓嶉枊鏂板€?鈹€鈹€
            sig_map = {s['symbol']: s['score'] for s in top7}
            already_closing = set()  # 闃叉閲嶈骞冲€夊悓涓€鍊嬪梗
            for pos in active_pos:
                sym_p = pos['symbol']
                if sym_p in already_closing:
                    continue
                pos_side = (pos.get('side') or '').lower()   # 'long' or 'short'
                new_score = sig_map.get(sym_p, None)
                if new_score is None:
                    continue  # 閫欒吉娌掓巸鍒伴€欏€嬪梗锛岃烦閬?                # 鍒ゆ柗鏂瑰悜琛濈獊
                is_reverse = (pos_side == 'long'  and new_score < -ORDER_THRESHOLD) or                              (pos_side == 'short' and new_score >  ORDER_THRESHOLD)
                if is_reverse:
                    contracts = float(pos.get('contracts', 0) or 0)
                    if abs(contracts) > 0:
                        close_side = 'sell' if pos_side == 'long' else 'buy'
                        already_closing.add(sym_p)
                        def _do_reverse_close(s, c, cs, score, mprice):
                            try:
                                exchange.create_order(s,'market',cs,abs(c),params={'reduceOnly':True})
                                touch_post_close_lock(s)
                                print("鍙嶅悜骞冲€夋垚鍔? {} 鏂板垎鏁?{:.1f} | 鍟熺敤30鍒嗛悩鍐峰嵒".format(s, score))
                                reverse_rec = {
                                        "symbol":s,"side":"鍙嶅悜骞冲€?,"score":score,
                                        "price":mprice,"stop_loss":0,"take_profit":0,
                                        "est_pnl":0,"order_usdt":0,
                                        "time":tw_now_str(),
                                    }
                                with STATE_LOCK:
                                    STATE["trade_history"].insert(0, reverse_rec)
                                persist_trade_history_record(reverse_rec)
                            except Exception as re:
                                print("鍙嶅悜骞冲€夊け鏁?{}: {}".format(s, re))
                        threading.Thread(
                            target=_do_reverse_close,
                            args=(sym_p, contracts, close_side, new_score, pos.get('markPrice',0)),
                            daemon=True
                        ).start()

            # 鈹€鈹€ 姝ｅ父闁嬪€夐倧杓紙涓嬪柈闁撻殧5绉掞紝閬垮厤rate limit锛夆攢鈹€
            # OpenAI 鏈€澶氳吉娴佹鏌ユ帓琛屾鍓?10锛涘闅涙寔鍊変粛鍙?MAX_OPEN_POSITIONS / 棰ㄦ帶闄愬埗銆?            order_scan_ts = time.time()
            pending_trigger_rows = []
            short_gainer_review_rows = []
            pending_watch_checked = 0
            top10_symbols = {str((s or {}).get('symbol') or '') for s in top10}
            pending_watch_candidates = list(sigs) + list(short_gainer_signals or [])
            if bool(OPENAI_TRADE_CONFIG.get('enabled', True) and OPENAI_API_KEY):
                for _sg in list(short_gainer_signals or []):
                    try:
                        _sym = str((_sg or {}).get('symbol') or '')
                        if not _sym:
                            continue
                        _pending = _get_openai_pending_advice(_sym, order_scan_ts)
                        if _pending and not _pending_advice_should_block(_sg, _pending, order_scan_ts):
                            _clear_openai_pending_advice(_sym)
                            _pending = None
                        if _pending:
                            continue
                        _row = dict(_sg)
                        _row['_short_gainer_initial_review'] = True
                        short_gainer_review_rows.append(_row)
                    except Exception as _sg_watch_err:
                        print('short gainer review queue failed: {}'.format(_sg_watch_err))
                try:
                    pending_map_snapshot = dict(_openai_pending_advice_map(OPENAI_TRADE_STATE) or {})
                    pending_symbols = set(pending_map_snapshot.keys())
                except Exception:
                    pending_map_snapshot = {}
                    pending_symbols = set()
                sig_symbols = {str((s or {}).get('symbol') or '') for s in pending_watch_candidates}
                for _sym in sorted(pending_symbols - sig_symbols):
                    try:
                        _tk = dict((tickers or {}).get(_sym) or {})
                        _px = _float_or_zero(_tk.get('last') or _tk.get('close') or _tk.get('markPrice') or _tk.get('bid') or _tk.get('ask'))
                        if _px > 0:
                            _ad = dict(pending_map_snapshot.get(_sym) or {})
                            _side_score = -0.01 if str(_ad.get('side') or '').lower() == 'short' else 0.01
                            pending_watch_candidates.append({
                                'symbol': _sym,
                                'score': _side_score,
                                'raw_score': _side_score,
                                'priority_score': 0.0,
                                'entry_quality': 0.0,
                                'rr_ratio': 0.0,
                                'price': _px,
                                'stop_loss': 0.0,
                                'take_profit': 0.0,
                                'desc': 'pending OpenAI advice background watch',
                                'breakdown': {},
                                'candidate_source': str(_ad.get('source') or 'pending_advice'),
                                'source': str(_ad.get('source') or 'pending_advice'),
                                'scanner_intent': str(_ad.get('scanner_intent') or 'background pending OpenAI advice watch'),
                            })
                    except Exception:
                        pass
                for _sig in pending_watch_candidates:
                    _sym = str((_sig or {}).get('symbol') or '')
                    if not _sym or _sym in top10_symbols:
                        continue
                    _advice = _get_openai_pending_advice(_sym, order_scan_ts)
                    if not _advice:
                        continue
                    pending_watch_checked += 1
                    _advice = _refresh_openai_pending_advice_watch(_sig, _advice, order_scan_ts, 'watching')
                    _hit, _why = _openai_pending_advice_trigger(_sig, _advice, order_scan_ts)
                    if _why == 'pending advice invalidated':
                        _clear_openai_pending_advice(_sym)
                        continue
                    if _hit:
                        _row = dict(_sig)
                        _row['_pending_openai_advice'] = _advice
                        _row['_pending_openai_trigger_reason'] = _why
                        _row['priority_score'] = round(float(_row.get('priority_score', abs(float(_row.get('score', 0) or 0))) or 0) + 25.0, 2)
                        pending_trigger_rows.append(_row)
                if pending_watch_checked or pending_trigger_rows or short_gainer_review_rows:
                    print("OpenAI pending advice watch checked={} triggered={} short_gainer_initial_queue={}".format(
                        pending_watch_checked,
                        len(pending_trigger_rows),
                        len(short_gainer_review_rows),
                    ))
            _order_rows = sorted(
                list(top10) + pending_trigger_rows + short_gainer_review_rows,
                key=lambda x:(
                    1 if x.get('_pending_openai_advice') else 0,
                    0 if x.get('_short_gainer_initial_review') else 1,
                    x.get('priority_score', abs(x['score'])),
                    abs(x['score'])
                ),
                reverse=True,
            )
            _order_rows = _dedupe_highly_related_signals(_order_rows, max_per_family=1, max_per_bucket=1)
            top10_for_order = []
            _seen_order_symbols = set()
            for _row in _order_rows:
                _sym = str((_row or {}).get('symbol') or '')
                if not _sym or _sym in _seen_order_symbols:
                    continue
                _seen_order_symbols.add(_sym)
                top10_for_order.append(_row)
                if len(top10_for_order) >= max(10, 10 + len(short_gainer_review_rows)):
                    break
            review_position_cap = min(MAX_OPEN_POSITIONS, OPENAI_REVIEW_MAX_ACTIVE_POSITIONS)
            if pos_cnt < review_position_cap:
                _touch_backend_thread('scan', '绗瑊}杓細閫佸鑸囦笅鍠浼颁腑'.format(sc + 1))
                order_delay = 0
                openai_sent_this_scan = 0
                openai_pending_rechecks_this_scan = 0
                openai_short_gainers_this_scan = 0
                openai_normal_attempts_this_scan = 0
                openai_pending_attempts_this_scan = 0
                openai_short_gainer_attempts_this_scan = 0
                openai_rank_limit = min(max(int(OPENAI_TRADE_CONFIG.get('top_k_per_scan', 10) or 10), 1), 10)
                openai_sends_per_scan = max(int(OPENAI_TRADE_CONFIG.get('sends_per_scan', 1) or 1), 1)
                openai_now_ts = order_scan_ts
                for rank_index, best in enumerate(top10_for_order):
                    # 澶х洡鏂瑰悜閬庢烤
                    with MARKET_LOCK:
                        mkt_dir = MARKET_STATE.get("direction", "涓€?)
                        mkt_str = MARKET_STATE.get("strength", 0)

                    # 寮峰害 >= 60% 鎵嶉亷婵炬柟鍚戯紝寮辩┖/寮卞涓嶉亷婵?                    signal_side = 'long' if best['score'] > 0 else 'short'
                    mkt_ok = True
                    if mkt_str >= 0.6:  # 鍙湁寮锋柟鍚戞墠閬庢烤
                        if mkt_dir in ("寮峰", "澶?) and signal_side == 'short':
                            mkt_ok = False  # 寮峰闋笉鍋氱┖
                        elif mkt_dir in ("寮风┖", "绌?) and signal_side == 'long':
                            mkt_ok = False  # 寮风┖闋笉鍋氬

                    # 澶х洡鍙仛杓斿姪娆婇噸锛岄伩鍏嶅柈涓€ BTC/ETH 澶х洡鍒ゆ柗涓诲皫鐭窔閬稿梗銆?                    eff_threshold = ORDER_THRESHOLD + (MARKET_NEUTRAL_THRESHOLD_ADD if mkt_dir == "涓€? and mkt_str >= 0.5 else 0)

                    same_dir_cnt_now = get_direction_position_count(signal_side)
                    entry_quality = float(best.get('entry_quality', 0) or 0)
                    rr_ratio = float(best.get('rr_ratio', 0) or 0)
                    regime_bias = float(best.get('regime_bias', 0) or 0)
                    side_ok = (signal_side == 'long' and regime_bias >= 0) or (signal_side == 'short' and regime_bias <= 0)
                    ai_decision = ai_decide_trade(best, eff_threshold, mkt_ok, side_ok, same_dir_cnt_now, pos_syms, already_closing)
                    allow_now = bool(ai_decision.get('allow_now'))
                    decision_source = 'rule_engine'
                    openai_result = {}
                    openai_decision = {}
                    openai_status = 'disabled'
                    openai_pending_advice = None
                    openai_pending_triggered = False
                    openai_pending_reason = ''
                    openai_panel = sync_openai_trade_state(push_runtime=False)
                    normal_rank_index = int(best.get('_normal_rank_index', rank_index) or rank_index)
                    hard_gate_reasons = []
                    openai_consult_block_reasons = []
                    soft_gate_reasons = []
                    symbol_cooldown_gate_reason = ''
                    if not mkt_ok:
                        soft_gate_reasons.append('澶х洡鏂瑰悜涓嶄竴鑷?)
                    if not side_ok:
                        soft_gate_reasons.append('鏂瑰悜鍋忓樊涓嶄竴鑷?)
                    if same_dir_cnt_now >= MAX_SAME_DIRECTION:
                        hard_gate_reasons.append('鍚屾柟鍚戞寔鍊夊凡婊?)
                    if best['symbol'] in pos_syms:
                        openai_consult_block_reasons.append('has_open_position')
                        hard_gate_reasons.append('鍚屽梗宸叉湁鎸佸€?)
                    if best['symbol'] in already_closing:
                        openai_consult_block_reasons.append('already_closing')
                        hard_gate_reasons.append('瑭插梗姝ｅ湪鍙嶅悜骞冲€?)
                    if best['symbol'] in SHORT_TERM_EXCLUDED:
                        openai_consult_block_reasons.append('short_term_excluded')
                        hard_gate_reasons.append('瑭插梗鍦ㄧ煭鏈熸帓闄ゅ悕鍠?)
                    if not can_reenter_symbol(best['symbol']):
                        hard_gate_reasons.append(get_symbol_cooldown_note(best['symbol']) or '鍚屽梗鍐峰嵒涓?)
                    if not best.get('allowed', True):
                        soft_gate_reasons.append('瑭插梗姝峰彶琛ㄧ従琚皝閹?)

                    openai_mode_active = bool(OPENAI_TRADE_CONFIG.get('enabled', True) and OPENAI_API_KEY)
                    if not openai_mode_active and rank_index >= MAX_OPEN_POSITIONS:
                        allow_now = False
                        hard_gate_reasons.append('瓒呭嚭瑕忓墖寮曟搸涓嬪柈鍓峽}鍚?.format(MAX_OPEN_POSITIONS))
                    if openai_mode_active:
                        allow_now = False
                        try:
                            sym_state = dict((OPENAI_TRADE_STATE.get('symbols') or {}).get(best['symbol'], {}) or {})
                            last_sent_ts = float(sym_state.get('last_sent_ts', 0) or 0)
                            last_openai_status = str(sym_state.get('last_status') or '').strip().lower()
                            cooldown_sec = max(int(OPENAI_TRADE_CONFIG.get('cooldown_minutes', 180) or 180), 1) * 60
                            openai_recently_sent = bool(last_openai_status not in ('empty_response', 'error', 'auth_error', 'permission_error', 'bad_request', 'rate_limit') and last_sent_ts > 0 and (openai_now_ts - last_sent_ts) < cooldown_sec)
                        except Exception:
                            openai_recently_sent = False
                        openai_pending_advice = dict(best.get('_pending_openai_advice') or {}) or _get_openai_pending_advice(best['symbol'], openai_now_ts)
                        if openai_pending_advice and not _pending_advice_should_block(best, openai_pending_advice, openai_now_ts):
                            _clear_openai_pending_advice(best['symbol'])
                            openai_pending_advice = None
                        if openai_pending_advice:
                            openai_pending_advice = _refresh_openai_pending_advice_watch(best, openai_pending_advice, openai_now_ts, 'watching')
                            openai_pending_reason = str(best.get('_pending_openai_trigger_reason') or '')
                            openai_pending_triggered = bool(openai_pending_reason)
                            if not openai_pending_triggered:
                                openai_pending_triggered, openai_pending_reason = _openai_pending_advice_trigger(best, openai_pending_advice, openai_now_ts)
                            if openai_pending_reason == 'pending advice invalidated':
                                _clear_openai_pending_advice(best['symbol'])
                                openai_pending_advice = None
                                openai_pending_triggered = False
                            elif openai_pending_triggered:
                                best = _attach_openai_pending_reference(best, openai_pending_advice, openai_pending_reason)
                                best['force_openai_recheck'] = True
                                openai_recently_sent = False
                        pending_waiting = bool(openai_pending_advice) and not openai_pending_triggered
                        candidate_source = str(best.get('candidate_source') or best.get('source') or 'normal')
                        is_short_gainer_review = candidate_source == 'short_gainers'
                        has_short_gainer_slot = openai_short_gainer_attempts_this_scan < OPENAI_SHORT_GAINERS_MAX_PER_SCAN
                        has_regular_slot = openai_normal_attempts_this_scan < openai_sends_per_scan
                        has_pending_slot = openai_pending_attempts_this_scan < OPENAI_TRADE_PENDING_RECHECK_MAX_PER_SCAN
                        rank_ok = is_short_gainer_review or openai_pending_triggered or normal_rank_index < openai_rank_limit
                        if openai_pending_triggered:
                            openai_lane = 'pending_recheck'
                            source_slot_ok = has_pending_slot
                        else:
                            openai_lane = 'short_gainers' if is_short_gainer_review else 'normal'
                            source_slot_ok = has_short_gainer_slot if is_short_gainer_review else has_regular_slot
                        effective_hard_gate_reasons = [] if openai_pending_triggered else openai_consult_block_reasons
                        can_send_openai = (
                            rank_ok and
                            not effective_hard_gate_reasons and
                            source_slot_ok and
                            not pending_waiting and
                            (openai_pending_triggered or not openai_recently_sent)
                        )
                        if can_send_openai:
                            if openai_lane == 'short_gainers':
                                openai_short_gainer_attempts_this_scan += 1
                            elif openai_lane == 'pending_recheck':
                                openai_pending_attempts_this_scan += 1
                            else:
                                openai_normal_attempts_this_scan += 1
                            if openai_pending_triggered:
                                openai_pending_advice = dict(openai_pending_advice or {})
                                openai_pending_advice.update({
                                    'status': 'triggered_recheck_sent',
                                    'last_trigger_ts': openai_now_ts,
                                    'trigger_count': int(openai_pending_advice.get('trigger_count', 0) or 0) + 1,
                                    'last_trigger_reason': openai_pending_reason,
                                })
                                best['pending_openai_advice'] = openai_pending_advice
                                _mark_openai_pending_advice(best['symbol'], {
                                    'status': 'triggered_recheck_sent',
                                    'last_trigger_ts': openai_now_ts,
                                    'last_trigger_price': round(float(best.get('price', 0) or 0), 8),
                                    'trigger_armed': False,
                                    'trigger_count': int(openai_pending_advice.get('trigger_count', 0) or 0),
                                    'last_trigger_reason': openai_pending_reason,
                                })
                            risk_snapshot = get_risk_status()
                            portfolio = {
                                'equity': STATE.get('equity', 0),
                                'active_position_count': pos_cnt,
                                'same_direction_count': same_dir_cnt_now,
                                'open_symbols': sorted(list(pos_syms))[:8],
                            }
                            _candidate, openai_result = _consult_openai_trade_for_signal(
                                best,
                                min(normal_rank_index, openai_rank_limit - 1) if is_short_gainer_review else normal_rank_index,
                                top10_for_order,
                                dict(MARKET_STATE),
                                risk_snapshot,
                                portfolio,
                            )
                            openai_status = str(openai_result.get('status') or 'unknown')
                            openai_decision = dict(openai_result.get('decision') or {})
                            openai_panel = sync_openai_trade_state(push_runtime=False)
                            if openai_status not in ('not_ranked', 'below_min_score', 'cooldown_active', 'global_interval_active', 'budget_paused', 'local_gate_block'):
                                if is_short_gainer_review:
                                    openai_short_gainers_this_scan += 1
                                elif openai_pending_triggered:
                                    openai_pending_rechecks_this_scan += 1
                                else:
                                    openai_sent_this_scan += 1
                            if openai_status in ('consulted', 'cached_reuse') and openai_decision:
                                decision_source = 'openai'
                                allow_now = bool(openai_decision.get('should_trade'))
                                if allow_now:
                                    _clear_openai_pending_advice(best['symbol'])
                                    _apply_openai_trade_plan_to_signal(best, openai_decision, openai_result)
                                    if hard_gate_reasons:
                                        allow_now = False
                                else:
                                    _store_openai_pending_advice(best, openai_decision, _candidate, openai_result, openai_now_ts)
                        else:
                            if rank_index >= openai_rank_limit:
                                openai_status = 'not_ranked'
                            elif openai_consult_block_reasons:
                                openai_status = 'local_gate_block'
                            elif pending_waiting:
                                openai_status = 'pending_advice_watching'
                            elif openai_recently_sent:
                                openai_status = 'cooldown_active'
                            elif openai_pending_triggered:
                                openai_status = 'pending_advice_triggered'
                            else:
                                openai_status = 'pending_advice_watching' if openai_pending_advice else 'review_deferred'
                    elif OPENAI_TRADE_CONFIG.get('enabled', True):
                        openai_status = 'missing_api_key'

                    reasons = build_auto_order_reason(best, ai_decision.get('effective_threshold', eff_threshold), mkt_ok, side_ok, same_dir_cnt_now, pos_syms, already_closing, ai_decision=ai_decision)
                    reasons = list(dict.fromkeys((ai_decision.get('reasons') or []) + (reasons or [])))
                    if hard_gate_reasons:
                        reasons.extend(hard_gate_reasons)
                    if soft_gate_reasons:
                        reasons.extend(soft_gate_reasons)
                    if openai_mode_active:
                        reasons.append('OpenAI鍊欓伕杓綁鍓峽}鍚嶏紝鏈吉鏈€澶氶€亄}鍊?.format(openai_rank_limit, openai_sends_per_scan))
                        if openai_status == 'not_ranked':
                            reasons.append('鏈€插叆OpenAI姹虹瓥鍚嶅柈')
                        elif openai_status == 'local_gate_block':
                            reasons.append('鏈湴纭ⅷ鎺у厛琛岄樆鎿嬶紝鏈€丱penAI')
                        elif openai_status in ('consulted', 'cached_reuse'):
                            if openai_decision.get('should_trade'):
                                reasons.append('OpenAI鏀捐 {} {}x {:.1f}%'.format(
                                    openai_decision.get('order_type', 'market'),
                                    int(openai_decision.get('leverage', 0) or 0),
                                    float(openai_decision.get('margin_pct', 0) or 0) * 100.0,
                                ))
                                if openai_decision.get('thesis'):
                                    reasons.append(str(openai_decision.get('thesis')))
                            else:
                                reasons.append('OpenAI鎷掑柈: {}'.format(openai_decision.get('reason_to_skip') or '妯″瀷瑾嶇偤涓嶉仼鍚?))
                        elif openai_status == 'cooldown_active':
                            reasons.append('OpenAI鍚屽梗鍐峰嵒涓?)
                        elif openai_status == 'budget_paused':
                            reasons.append('OpenAI鏈堥爯绠楀凡閬斾笂闄?)
                        elif openai_status == 'below_min_score':
                            reasons.append('鏈仈OpenAI閫佸鍒嗘暩')
                        elif openai_status == 'error':
                            reasons.append('OpenAI鍛煎彨澶辨晽锛屾湰杓笉涓嬪柈')
                        elif openai_status == 'empty_response':
                            reasons.append('OpenAI绌哄洖瑕嗭紝宸茶閷勬垚鏈甫閫插叆鍐峰嵒')
                        elif openai_status == 'review_deferred':
                            reasons.append('鏈吉OpenAI閫佸鍚嶉宸茬敤锛屼笅涓€杓辜绾屽線寰岄€?)
                        elif openai_status == 'pending_advice_watching':
                            reasons.append('OpenAI pending advice watching: {}'.format((openai_pending_advice or {}).get('watch_note') or (openai_pending_advice or {}).get('entry_plan') or 'waiting for trigger'))
                        elif openai_status == 'pending_advice_triggered':
                            reasons.append('OpenAI pending advice triggered but send slot is already used: {}'.format(openai_pending_reason or 'waiting next scan'))
                    elif openai_status == 'missing_api_key':
                        reasons.append('灏氭湭瑷畾 OPENAI_API_KEY锛屾毇鐢ㄨ鍓囧紩鎿?)
                    reasons = list(dict.fromkeys(reasons))

                    AUTO_ORDER_AUDIT[best['symbol']] = {
                        'will_order': bool(allow_now),
                        'reasons': reasons or ['绗﹀悎姊濅欢锛岃嚜鍕曚笅鍠?],
                        'threshold': round(ai_decision.get('effective_threshold', eff_threshold), 2),
                        'effective_score': round(float(ai_decision.get('effective_score', abs(best.get('score', 0))) or 0), 2),
                        'rotation_adj': round(float(ai_decision.get('rotation_adj', best.get('rotation_adj', 0)) or 0), 2),
                        'selection_edge': round(float(best.get('selection_edge', 0) or 0), 2),
                        'selection_notes': list(best.get('selection_notes') or [])[:8],
                        'marketability': dict(best.get('marketability') or {}),
                        'entry_quality': round(entry_quality, 2),
                        'rr_ratio': round(rr_ratio, 2),
                        'mkt_dir': mkt_dir,
                        'same_dir_cnt': same_dir_cnt_now,
                        'checked_at': tw_now_str(),
                        'ai_enabled': True,
                        'ai_ready': bool((ai_decision.get('profile') or {}).get('ready')),
                        'ai_source': (ai_decision.get('profile') or {}).get('source', 'none'),
                        'ai_strategy': (ai_decision.get('profile') or {}).get('strategy', ''),
                        'ai_sample_count': int((ai_decision.get('profile') or {}).get('sample_count', 0) or 0),
                        'ai_win_rate': round(float((ai_decision.get('profile') or {}).get('win_rate', 0) or 0), 2),
                        'ai_avg_pnl': round(float((ai_decision.get('profile') or {}).get('avg_pnl', 0) or 0), 2),
                        'ai_note': (ai_decision.get('profile') or {}).get('note', ''),
                        'ai_phase': (ai_decision.get('profile') or {}).get('phase', ''),
                        'p_win_est': round(float((ai_decision.get('decision_calibrator') or {}).get('p_win_est', 0.0) or 0.0), 4),
                        'expected_value_est': round(float((ai_decision.get('decision_calibrator') or {}).get('expected_value_est', 0.0) or 0.0), 4),
                        'confidence_calibrated': round(float((ai_decision.get('decision_calibrator') or {}).get('confidence_calibrated', 0.0) or 0.0), 4),
                        'dataset_version': _dataset_meta().get('dataset_version'),
                        'decision_source': decision_source,
                        'candidate_source': str(best.get('candidate_source') or best.get('source') or 'normal')[:60],
                        'scanner_intent': str(best.get('scanner_intent') or '')[:180],
                        'openai_enabled': bool(openai_mode_active),
                        'openai_status': openai_status,
                        'openai_status_label': {
                            'consulted': 'OpenAI宸叉焙绛?,
                            'cached_reuse': '娌跨敤鑸婃焙绛?,
                            'not_ranked': '鏈€插叆鍓嶅咕鍚?,
                            'cooldown_active': '鍚屽梗鍐峰嵒涓?,
                            'budget_paused': '闋愮畻鏆仠',
                            'missing_api_key': '缂哄皯API Key',
                            'auth_error': 'OpenAI 椹楄瓑澶辨晽',
                            'permission_error': 'OpenAI 娆婇檺涓嶈冻',
                            'bad_request': 'OpenAI 璜嬫眰鏍煎紡閷',
                            'rate_limit': 'OpenAI 閫熺巼闄愬埗',
                            'empty_response': 'OpenAI 绌哄洖瑕?,
                            'review_deferred': '寰呬笅杓€佸',
                            'below_min_score': '鍒嗘暩澶綆',
                            'local_gate_block': '鏈湴棰ㄦ帶闃绘搵',
                            'error': 'OpenAI閷',
                        }.get(openai_status, {
                            'pending_advice_watching': 'AI advice watching',
                            'pending_advice_triggered': 'AI advice triggered',
                        }.get(openai_status, openai_status)),
                        'openai_model': (openai_result.get('symbol_state') or {}).get('last_model', openai_panel.get('model', '')),
                        'openai_order_type': openai_decision.get('order_type', ''),
                        'openai_entry_price': round(float(openai_decision.get('entry_price', 0) or 0), 8) if openai_decision else 0.0,
                        'openai_stop_loss': round(float(openai_decision.get('stop_loss', 0) or 0), 8) if openai_decision else 0.0,
                        'openai_take_profit': round(float(openai_decision.get('take_profit', 0) or 0), 8) if openai_decision else 0.0,
                        'openai_leverage': int(openai_decision.get('leverage', 0) or 0) if openai_decision else 0,
                        'openai_margin_pct': round(float(openai_decision.get('margin_pct', 0) or 0), 4) if openai_decision else 0.0,
                        'openai_confidence': round(float(openai_decision.get('confidence', 0) or 0), 2) if openai_decision else 0.0,
                        'openai_thesis': str(openai_decision.get('thesis') or '')[:260] if openai_decision else '',
                        'openai_market_read': str(openai_decision.get('market_read') or '')[:260] if openai_decision else '',
                        'openai_entry_plan': str(openai_decision.get('entry_plan') or '')[:260] if openai_decision else '',
                        'openai_entry_reason': str(openai_decision.get('entry_reason') or '')[:220] if openai_decision else '',
                        'openai_stop_loss_reason': str(openai_decision.get('stop_loss_reason') or '')[:220] if openai_decision else '',
                        'openai_take_profit_plan': str(openai_decision.get('take_profit_plan') or '')[:260] if openai_decision else '',
                        'openai_if_missed_plan': str(openai_decision.get('if_missed_plan') or '')[:220] if openai_decision else '',
                        'openai_reference_summary': str(openai_decision.get('reference_summary') or '')[:220] if openai_decision else '',
                        'openai_chase_if_triggered': bool(openai_decision.get('chase_if_triggered', False)) if openai_decision else False,
                        'openai_chase_trigger_price': round(float(openai_decision.get('chase_trigger_price', 0) or 0), 8) if openai_decision else 0.0,
                        'openai_chase_limit_price': round(float(openai_decision.get('chase_limit_price', 0) or 0), 8) if openai_decision else 0.0,
                        'openai_risk_notes': list(openai_decision.get('risk_notes') or [])[:4] if openai_decision else [],
                        'openai_aggressive_note': str(openai_decision.get('aggressive_note') or '')[:220] if openai_decision else '',
                        'openai_reason_to_skip': str(openai_decision.get('reason_to_skip') or '')[:220] if openai_decision else '',
                        'openai_watch_trigger_type': str(openai_decision.get('watch_trigger_type') or '') if openai_decision else str((openai_pending_advice or {}).get('trigger_type') or ''),
                        'openai_watch_trigger_price': round(float(openai_decision.get('watch_trigger_price', 0) or 0), 8) if openai_decision else round(float((openai_pending_advice or {}).get('trigger_price', 0) or 0), 8),
                        'openai_watch_invalidation_price': round(float(openai_decision.get('watch_invalidation_price', 0) or 0), 8) if openai_decision else round(float((openai_pending_advice or {}).get('invalidation_price', 0) or 0), 8),
                        'openai_watch_note': str(openai_decision.get('watch_note') or '')[:220] if openai_decision else str((openai_pending_advice or {}).get('watch_note') or '')[:220],
                        'openai_recheck_reason': str(openai_decision.get('recheck_reason') or '')[:220] if openai_decision else str((openai_pending_advice or {}).get('recheck_reason') or openai_pending_reason or '')[:220],
                        'openai_watch_timeframe': str(openai_decision.get('watch_timeframe') or '')[:80] if openai_decision else str((openai_pending_advice or {}).get('watch_timeframe') or '')[:80],
                        'openai_watch_price_zone_low': round(float(openai_decision.get('watch_price_zone_low', 0) or 0), 8) if openai_decision else round(float((openai_pending_advice or {}).get('watch_price_zone_low', 0) or 0), 8),
                        'openai_watch_price_zone_high': round(float(openai_decision.get('watch_price_zone_high', 0) or 0), 8) if openai_decision else round(float((openai_pending_advice or {}).get('watch_price_zone_high', 0) or 0), 8),
                        'openai_watch_structure_condition': str(openai_decision.get('watch_structure_condition') or '')[:220] if openai_decision else str((openai_pending_advice or {}).get('watch_structure_condition') or '')[:220],
                        'openai_watch_volume_condition': str(openai_decision.get('watch_volume_condition') or '')[:220] if openai_decision else str((openai_pending_advice or {}).get('watch_volume_condition') or '')[:220],
                        'openai_watch_checklist': list(openai_decision.get('watch_checklist') or [])[:6] if openai_decision else list((openai_pending_advice or {}).get('watch_checklist') or [])[:6],
                        'openai_watch_confirmations': list(openai_decision.get('watch_confirmations') or [])[:6] if openai_decision else list((openai_pending_advice or {}).get('watch_confirmations') or [])[:6],
                        'openai_watch_invalidations': list(openai_decision.get('watch_invalidations') or [])[:6] if openai_decision else list((openai_pending_advice or {}).get('watch_invalidations') or [])[:6],
                        'openai_watch_recheck_priority': round(float(openai_decision.get('watch_recheck_priority', 0) or 0), 2) if openai_decision else round(float((openai_pending_advice or {}).get('watch_recheck_priority', 0) or 0), 2),
                        'openai_limit_cancel_price': round(float(openai_decision.get('limit_cancel_price', 0) or 0), 8) if openai_decision else 0.0,
                        'openai_limit_cancel_timeframe': str(openai_decision.get('limit_cancel_timeframe') or '')[:80] if openai_decision else '',
                        'openai_limit_cancel_condition': str(openai_decision.get('limit_cancel_condition') or '')[:220] if openai_decision else '',
                        'openai_limit_cancel_note': str(openai_decision.get('limit_cancel_note') or '')[:220] if openai_decision else '',
                        'openai_pending_advice': dict(openai_pending_advice or {}),
                        'openai_pending_triggered': bool(openai_pending_triggered),
                        'openai_pending_trigger_reason': str(openai_pending_reason or '')[:220],
                        'openai_cached': bool(openai_status == 'cached_reuse'),
                        'openai_budget_spent_twd': round(float(openai_panel.get('spent_estimated_twd', 0) or 0), 4),
                        'openai_budget_remaining_twd': round(float(openai_panel.get('remaining_estimated_twd', 0) or 0), 4),
                    }
                    best['auto_order'] = AUTO_ORDER_AUDIT[best['symbol']]
                    best['ai_decision'] = AUTO_ORDER_AUDIT[best['symbol']]
                    try:
                        save_decision_input_snapshot(SQLITE_DB_PATH, {
                            'symbol': best.get('symbol'),
                            'side': 'buy' if best.get('score', 0) >= 0 else 'sell',
                            'regime_snapshot': {
                                'regime': best.get('regime') or (best.get('breakdown') or {}).get('Regime') or 'neutral',
                                'bias': best.get('regime_bias', 0),
                                'confidence': best.get('regime_confidence', 0),
                            },
                            'setup_key': (best.get('setup_key') or (best.get('breakdown') or {}).get('Setup') or ''),
                            'signal_snapshot': build_signal_quality_snapshot(best),
                            'symbol_personality': (ai_decision.get('profile') or {}).get('symbol_personality', {}),
                            'sample_weight_summary': {
                                'source': (ai_decision.get('profile') or {}).get('source', 'none'),
                                'sample_count': int((ai_decision.get('profile') or {}).get('sample_count', 0) or 0),
                                'confidence': float((ai_decision.get('profile') or {}).get('confidence', 0) or 0),
                                'dataset_version': _dataset_meta().get('dataset_version'),
                                'learning_generation': _dataset_meta().get('learning_generation'),
                            },
                            'session_bucket': session_bucket_from_hour(get_tw_time().hour),
                            'market_consensus': dict(LAST_MARKET_CONSENSUS or {}),
                            'execution_quality': dict((best.get('execution_quality') or {})),
                            'decision': dict(AUTO_ORDER_AUDIT[best['symbol']]),
                            'gating': dict(ai_decision.get('gating') or {}),
                            'decision_calibrator': dict(ai_decision.get('decision_calibrator') or {}),
                            'position_formula': dict((best.get('margin_ctx') or {})),
                            'dataset_meta': _dataset_meta(),
                            'learn_version': 'v16_replay_inputs',
                        })
                    except Exception as _replay_err:
                        print('save_decision_input_snapshot澶辨晽: {}'.format(_replay_err))

                    if allow_now:  # 鍕曟厠闁€妾伙紙鍚?AI 鎺ョ锛?                        def _make_delayed(sig, delay):
                            def _run():
                                time.sleep(delay)
                                place_order(sig)
                            return _run
                        threading.Thread(
                            target=_make_delayed(best, order_delay),
                            daemon=True
                        ).start()
                        order_delay += 5  # 姣忕瓎鍠枔闅?绉?
            else:
                print("鎸佸€夊凡閬旈€佸鏆仠闁€妾?{}锛屾湰杓笉鍐嶉€?OpenAI / 涓嶉枊鏂板€?.format(review_position_cap))

            # 鏇存柊鍕曟厠闁€妾?            update_dynamic_threshold(top10)

            # 姣?0杓洿鏂颁竴娆″ぇ鐩ゅ垎鏋愶紙涓嶇瓑1灏忔檪锛?            if STATE.get("scan_count", 0) % 10 == 1:
                try:
                    result = analyze_btc_market_trend()
                    if result:
                        with MARKET_LOCK:
                            MARKET_STATE.update(result)
                        update_state(market_info=dict(MARKET_STATE))
                        print("馃搳 澶х洡(瀹氭湡鏇存柊): {} | {}".format(
                            result["pattern"], result["direction"]))
                except Exception as me:
                    print("澶х洡瀹氭湡鏇存柊澶辨晽: {}".format(me))

            # 鏇存柊棰ㄦ帶鎽樿
            print("姝ラD: 婧栧倷鏇存柊棰ㄦ帶... 鐣跺墠闁€妾?{}鍒?.format(ORDER_THRESHOLD))
            update_state(risk_status=get_risk_status())
            print("绗瑊}杓巸鎻忓畬鎴愶紝60绉掑緦闁嬪涓嬩竴杓?.format(STATE["scan_count"]))
            _touch_backend_thread('scan', '绗瑊}杓畬鎴愶紝60 绉掑緦涓嬩竴杓?.format(STATE["scan_count"]))
            time.sleep(60)  # 杓枔闅?0绉?            print("姝ラE: 60绉掍紤鎭祼鏉燂紝闁嬪涓嬩竴杓?)
        except Exception as e:
            import traceback
            print("鎺冩弿鐣板父: {}".format(e))
            print(traceback.format_exc())
            _set_backend_thread_state('scan', 'crashed', '鎺冩弿鍩疯绶掔暟甯革紝10 绉掑緦閲嶈│', str(e))
            time.sleep(10)

# =====================================================
# Flask 璺敱
# =====================================================
@app.route('/')
def index(): return render_template('index.html')


def _calc_max_drawdown(equity_curve):
    peak = equity_curve[0] if equity_curve else 0
    max_dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        if peak > 0:
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)
    return round(max_dd * 100, 2)

def run_simple_backtest_legacy_shadow_1(symbol="BTC/USDT:USDT", timeframe="15m", limit=800, fee_rate=0.0006):
    """
    杓曢噺鍥炴脯锛氱磵鍏?    - 钃勫嫝绲愭
    - 鍋囩獊鐮撮亷婵?    - 鍙嶈拷鍍?    - 鍒嗘壒閫插牬
    - 鍒嗘壒姝㈢泩
    璁撳洖娓洿璨艰繎瀵︾洡閭忚集銆?    """
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['ts','o','h','l','c','v'])
        if len(df) < 250:
            return {"ok": False, "error": "K绶氫笉瓒筹紝鑷冲皯闇€瑕?50鏍?}

        df['ema21'] = ta.ema(df['c'], length=21)
        df['ema55'] = ta.ema(df['c'], length=55)
        macd = ta.macd(df['c'], fast=12, slow=26, signal=9)
        df['macd'] = macd.iloc[:, 0]
        df['macds'] = macd.iloc[:, 1]
        df['rsi'] = ta.rsi(df['c'], length=14)
        adx_df = ta.adx(df['h'], df['l'], df['c'], length=14)
        df['adx'] = adx_df.iloc[:, 0]
        df['atr'] = ta.atr(df['h'], df['l'], df['c'], length=14)
        df = df.dropna().reset_index(drop=True)
        if len(df) < 120:
            return {"ok": False, "error": "鏈夋晥鎸囨璩囨枡涓嶈冻"}

        equity = 10000.0
        equity_curve = [equity]
        trades = []
        position = None

        for i in range(60, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            price = float(row['c'])
            atr = max(float(row['atr']), price * 0.003)
            ema_up = row['ema21'] > row['ema55']
            ema_dn = row['ema21'] < row['ema55']
            macd_up = row['macd'] > row['macds'] and prev['macd'] <= prev['macds']
            macd_dn = row['macd'] < row['macds'] and prev['macd'] >= prev['macds']
            adx_ok = row['adx'] >= 18
            win = df.iloc[max(0, i-BREAKOUT_LOOKBACK):i]
            hh = float(win['h'].max()) if len(win) else price
            ll = float(win['l'].min()) if len(win) else price
            recent_lows = df['l'].iloc[max(0, i-6):i].tolist()
            recent_highs = df['h'].iloc[max(0, i-6):i].tolist()
            squeeze_long = ema_up and len(recent_lows) >= 4 and _linreg_slope(recent_lows) > 0 and (hh - price) / max(atr, 1e-9) <= 0.5
            squeeze_short = ema_dn and len(recent_highs) >= 4 and _linreg_slope(recent_highs) < 0 and (price - ll) / max(atr, 1e-9) <= 0.5
            ext = (price - float(row['ema21'])) / max(atr, 1e-9)
            anti_chase_long = ext <= ANTI_CHASE_ATR
            anti_chase_short = ext >= -ANTI_CHASE_ATR
            sub = df.iloc[max(0, i-BREAKOUT_LOOKBACK-1):i+1].copy()
            fake_s, _, fake_meta = analyze_fake_breakout(sub, 1 if ema_up else -1 if ema_dn else 0)
            fakeout_long = fake_meta.get('fakeout') and fake_meta.get('direction') == 'up'
            fakeout_short = fake_meta.get('fakeout') and fake_meta.get('direction') == 'down'

            if position is None:
                if ema_up and (macd_up or squeeze_long) and row['rsi'] < 68 and adx_ok and anti_chase_long and not fakeout_long:
                    entry = min(price, max(float(row['ema21']), hh - atr * PULLBACK_BUFFER_ATR)) if price >= hh * 0.999 else price
                    sl = entry - atr * 1.55
                    rr_target = get_learned_rr_target(symbol, 'trend' if squeeze_long else 'neutral', '鏀舵杺绐佺牬鍟熷嫊' if squeeze_long else '瓒ㄥ嫝鍥炶俯绾屾敾', [symbol, 'backtest', 'long'], 1.55, (3.6 if squeeze_long else 3.0))
                    tp = entry + abs(entry - sl) * rr_target
                    pseudo_score = 55 if squeeze_long and adx_ok else 52 if macd_up else 50
                    margin_pct = calc_dynamic_margin_pct(pseudo_score, atr / max(price,1e-9), True, squeeze_long, not anti_chase_long, 0)
                    scale_in = 0.4 if squeeze_long and pseudo_score >= 55 else 0.0
                    blended_entry = entry * (1 - scale_in) + max(float(row['ema21']), entry - atr * PULLBACK_BUFFER_ATR) * scale_in
                    risk_budget = equity * ATR_RISK_PCT
                    cap_qty = (equity * margin_pct) / max(blended_entry, 1e-9)
                    risk_qty = risk_budget / max(blended_entry - sl, price * 0.002)
                    qty = max(min(risk_qty, cap_qty), 0)
                    position = {"side": "long", "entry": blended_entry, "sl": sl, "tp": tp, "qty": qty, "bar": i, "margin_pct": margin_pct, "partial_done": 0}
                elif ema_dn and (macd_dn or squeeze_short) and row['rsi'] > 32 and adx_ok and anti_chase_short and not fakeout_short:
                    entry = max(price, min(float(row['ema21']), ll + atr * PULLBACK_BUFFER_ATR)) if price <= ll * 1.001 else price
                    sl = entry + atr * 1.55
                    rr_target = get_learned_rr_target(symbol, 'trend' if squeeze_short else 'neutral', '鏀舵杺璺岀牬鍟熷嫊' if squeeze_short else '鍙嶅綀绾岃穼', [symbol, 'backtest', 'short'], 1.55, (3.6 if squeeze_short else 3.0))
                    tp = entry - abs(entry - sl) * rr_target
                    pseudo_score = 55 if squeeze_short and adx_ok else 52 if macd_dn else 50
                    margin_pct = calc_dynamic_margin_pct(pseudo_score, atr / max(price,1e-9), True, squeeze_short, not anti_chase_short, 0)
                    scale_in = 0.4 if squeeze_short and pseudo_score >= 55 else 0.0
                    blended_entry = entry * (1 - scale_in) + min(float(row['ema21']), entry + atr * PULLBACK_BUFFER_ATR) * scale_in
                    risk_budget = equity * ATR_RISK_PCT
                    cap_qty = (equity * margin_pct) / max(blended_entry, 1e-9)
                    risk_qty = risk_budget / max(sl - blended_entry, price * 0.002)
                    qty = max(min(risk_qty, cap_qty), 0)
                    position = {"side": "short", "entry": blended_entry, "sl": sl, "tp": tp, "qty": qty, "bar": i, "margin_pct": margin_pct, "partial_done": 0}
                continue

            exit_reason = None
            exit_price = price
            bars_held = i - position['bar']
            pnl = None

            if position['side'] == 'long':
                profit_atr = (price - position['entry']) / max(atr, 1e-9)
                if profit_atr >= 1.2 and position['partial_done'] == 0:
                    realized_qty = position['qty'] * 0.25
                    net = (price - position['entry']) * realized_qty - (position['entry'] + price) * realized_qty * fee_rate
                    equity += net
                    trades.append({"side": "long", "entry": round(position['entry'], 6), "exit": round(price, 6), "pnl": round(net, 4), "reason": 'TP1', "bars": bars_held, "margin_pct": round(position.get('margin_pct', RISK_PCT) * 100, 2)})
                    position['qty'] *= 0.75
                    position['sl'] = max(position['sl'], position['entry'])
                    position['partial_done'] = 1
                elif profit_atr >= 2.4 and position['partial_done'] == 1:
                    realized_qty = position['qty'] * 0.35
                    net = (price - position['entry']) * realized_qty - (position['entry'] + price) * realized_qty * fee_rate
                    equity += net
                    trades.append({"side": "long", "entry": round(position['entry'], 6), "exit": round(price, 6), "pnl": round(net, 4), "reason": 'TP2', "bars": bars_held, "margin_pct": round(position.get('margin_pct', RISK_PCT) * 100, 2)})
                    position['qty'] *= 0.65
                    position['sl'] = max(position['sl'], position['entry'] + atr * 0.8)
                    position['partial_done'] = 2
                if row['l'] <= position['sl']:
                    exit_price = position['sl']; exit_reason = 'SL'
                elif row['h'] >= position['tp']:
                    exit_price = position['tp']; exit_reason = 'TP'
                elif bars_held >= TIME_STOP_BARS_15M and abs(price - position['entry']) / position['entry'] < 0.006:
                    exit_reason = 'TIME'
                elif ema_dn and macd_dn:
                    exit_reason = 'REVERSE'
                pnl = (exit_price - position['entry']) * position['qty'] if exit_reason else None
            else:
                profit_atr = (position['entry'] - price) / max(atr, 1e-9)
                if profit_atr >= 1.2 and position['partial_done'] == 0:
                    realized_qty = position['qty'] * 0.25
                    net = (position['entry'] - price) * realized_qty - (position['entry'] + price) * realized_qty * fee_rate
                    equity += net
                    trades.append({"side": "short", "entry": round(position['entry'], 6), "exit": round(price, 6), "pnl": round(net, 4), "reason": 'TP1', "bars": bars_held, "margin_pct": round(position.get('margin_pct', RISK_PCT) * 100, 2)})
                    position['qty'] *= 0.75
                    position['sl'] = min(position['sl'], position['entry'])
                    position['partial_done'] = 1
                elif profit_atr >= 2.4 and position['partial_done'] == 1:
                    realized_qty = position['qty'] * 0.35
                    net = (position['entry'] - price) * realized_qty - (position['entry'] + price) * realized_qty * fee_rate
                    equity += net
                    trades.append({"side": "short", "entry": round(position['entry'], 6), "exit": round(price, 6), "pnl": round(net, 4), "reason": 'TP2', "bars": bars_held, "margin_pct": round(position.get('margin_pct', RISK_PCT) * 100, 2)})
                    position['qty'] *= 0.65
                    position['sl'] = min(position['sl'], position['entry'] - atr * 0.8)
                    position['partial_done'] = 2
                if row['h'] >= position['sl']:
                    exit_price = position['sl']; exit_reason = 'SL'
                elif row['l'] <= position['tp']:
                    exit_price = position['tp']; exit_reason = 'TP'
                elif bars_held >= TIME_STOP_BARS_15M and abs(price - position['entry']) / position['entry'] < 0.006:
                    exit_reason = 'TIME'
                elif ema_up and macd_up:
                    exit_reason = 'REVERSE'
                pnl = (position['entry'] - exit_price) * position['qty'] if exit_reason else None

            if exit_reason:
                fee = (position['entry'] + exit_price) * position['qty'] * fee_rate
                net = pnl - fee
                equity += net
                trades.append({"side": position['side'], "entry": round(position['entry'], 6), "exit": round(exit_price, 6), "pnl": round(net, 4), "reason": exit_reason, "bars": bars_held, "margin_pct": round(position.get('margin_pct', RISK_PCT) * 100, 2)})
                equity_curve.append(equity)
                position = None

        wins = sum(1 for t in trades if t['pnl'] > 0)
        losses = sum(1 for t in trades if t['pnl'] <= 0)
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        profit_factor = round(gross_profit / gross_loss, 3) if gross_loss > 0 else None
        win_rate = round((wins / len(trades) * 100), 2) if trades else 0.0
        avg_margin_pct = round((sum(t.get('margin_pct', RISK_PCT * 100) for t in trades) / len(trades)), 2) if trades else round(RISK_PCT * 100, 2)
        return {"ok": True, "symbol": symbol, "timeframe": timeframe, "trades": len(trades), "wins": wins, "losses": losses, "win_rate": win_rate, "profit_factor": profit_factor, "avg_margin_pct": avg_margin_pct, "margin_range_pct": [round(MIN_MARGIN_PCT * 100, 2), round(MAX_MARGIN_PCT * 100, 2)], "net_profit": round(equity - 10000.0, 2), "return_pct": round((equity / 10000.0 - 1) * 100, 2), "max_drawdown_pct": _calc_max_drawdown(equity_curve), "last_10_trades": trades[-10:]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.route('/api/backtest', methods=['GET'])
def api_backtest():
    symbol = _resolve_backtest_symbol(request.args.get('symbol', 'auto'))
    timeframe = request.args.get('timeframe', '15m')
    try:
        limit = int(request.args.get('limit', 800))
    except:
        limit = 800
    result = run_simple_backtest(symbol=symbol, timeframe=timeframe, limit=max(250, min(limit, 2000)))
    if isinstance(result, dict):
        result['selected_symbol'] = symbol
    return jsonify(result)

@app.route('/api/state')
def api_state_legacy_shadow_1():
    try:
        with STATE_LOCK:
            s = dict(STATE)

        # 娓呯悊 active_positions锛堢Щ闄や笉鑳?JSON 搴忓垪鍖栫殑娆勪綅锛?        clean_pos = []
        for p in s.get('active_positions', []):
            try:
                clean_pos.append({
                    'symbol':        str(p.get('symbol','') or ''),
                    'side':          str(p.get('side','') or ''),
                    'contracts':     float(p.get('contracts',0) or 0),
                    'entryPrice':    float(p.get('entryPrice',0) or 0),
                    'markPrice':     float(p.get('markPrice',0) or 0),
                    'unrealizedPnl': float(p.get('unrealizedPnl',0) or 0),
                    'percentage':    float(p.get('percentage',0) or 0),
                    'leverage':      float(p.get('leverage',1) or 1),
                    'drawdown_pct':  _position_drawdown_pct(p),
                    'leveraged_pnl_pct': _position_leveraged_pnl_pct(p),
                })
            except:
                pass
        s['active_positions'] = clean_pos

        # 瑁滀笂鍗虫檪棰ㄦ帶鐙€鎱?        s['risk_status'] = get_risk_status()

        # 瑁滀笂澶х洡鍜岄暦鏈熷€変綅
        with MARKET_LOCK:
            s['market_info'] = dict(MARKET_STATE)
        with LT_LOCK:
            s['lt_info'] = dict(LT_STATE)
        with FVG_LOCK:
            s['fvg_orders'] = dict(FVG_ORDERS)

        # 鍗虫檪绲勫悎 trailing_info锛堜笉绛?trailing_stop_thread 鏇存柊锛?        try:
            ui_trail = {}
            with TRAILING_LOCK:
                for sym, ts in TRAILING_STATE.items():
                    side_t = ts.get('side','')
                    trail  = ts.get('trail_pct', 0.05)
                    highest = ts.get('highest_price', 0)
                    lowest  = ts.get('lowest_price', float('inf'))
                    entry   = ts.get('entry_price', 0)
                    sl      = ts.get('initial_sl', 0)
                    if side_t in ('buy','long') and highest > 0:
                        trail_price = highest * (1 - trail)
                        stage = '淇濇湰' if abs(sl - entry) < entry * 0.001 else '閹栧埄'
                        ui_trail[sym] = {
                            'side': '鍋氬',
                            'peak': round(highest, 6),
                            'trail_price': round(trail_price, 6),
                            'trail_pct': round(trail * 100, 1),
                            'initial_sl': round(sl, 6),
                            'stage': stage,
                        }
                    elif side_t in ('sell','short') and lowest != float('inf'):
                        trail_price = lowest * (1 + trail)
                        ui_trail[sym] = {
                            'side': '鍋氱┖',
                            'peak': round(lowest, 6),
                            'trail_price': round(trail_price, 6),
                            'trail_pct': round(trail * 100, 1),
                            'initial_sl': round(sl, 6),
                        }
            s['trailing_info'] = ui_trail
        except:
            pass

        s['session_info'] = {}

        # 瑁滀笂鍕曟厠闁€妾昏硣瑷?        with _DT_LOCK:
            curr_thr = float(_DT.get('current', ORDER_THRESHOLD_DEFAULT) or ORDER_THRESHOLD_DEFAULT)
            s['threshold_info'] = {
                'current': curr_thr,
                'phase': 'AI绌嶆サ' if curr_thr <= 51 else 'AI鍧囪　' if curr_thr <= 60 else 'AI淇濆畧',
                'full_rounds': _DT.get('full_rounds', 0),
                'empty_rounds': _DT.get('empty_rounds', 0),
                'no_order_rounds': _DT.get('no_order_rounds', 0),
                'ai_note': _DT.get('last_ai_note', ''),
            }

        return jsonify(s)
    except Exception as e:
        print("api_state 閷: {}".format(e))
        return jsonify({"error": str(e), "scan_progress": "API閷: {}".format(str(e)[:50])})

@app.route('/api/learn_db')
def api_learn_db():
    with LEARN_LOCK: return jsonify(LEARN_DB)

@app.route('/api/close_all',methods=['POST'])
def api_close(): return jsonify({"status":"ok","closed":close_all()})

@app.route('/api/fvg_cancel', methods=['POST'])
def api_fvg_cancel():
    data   = request.get_json() or {}
    symbol = data.get('symbol','')
    if not symbol:
        return jsonify({"ok": False, "msg": "缂哄皯 symbol"})
    with FVG_LOCK:
        if symbol not in FVG_ORDERS:
            return jsonify({"ok": False, "msg": "鎵句笉鍒?{} 鐨勬帥鍠?.format(symbol)})
        order    = FVG_ORDERS.get(symbol, {})
        order_id = order.get("order_id","")
    if order_id:
        try:
            exchange.cancel_order(order_id, symbol)
            print("鎵嬪嫊鍙栨秷鎺涘柈: {} order_id={}".format(symbol, order_id))
        except Exception as e:
            print("鍙栨秷澶辨晽(鍙兘宸叉垚浜?: {}".format(e))
    with FVG_LOCK:
        FVG_ORDERS.pop(symbol, None)
    update_state(fvg_orders=dict(FVG_ORDERS))
    return jsonify({"ok": True, "msg": "{} 鎺涘柈宸插彇娑?.format(symbol)})

@app.route('/api/lt_open', methods=['POST'])
def api_lt_open():
    data      = request.get_json() or {}
    direction = data.get('direction', 'long')
    reason    = data.get('reason', '鎵嬪嫊鎿嶄綔')
    ok = open_long_term_position(direction, reason)
    return jsonify({"ok": ok, "msg": "闀锋湡鍊変綅宸查枊鍟? if ok else "闁嬪€夊け鏁?})

@app.route('/api/lt_close', methods=['POST'])
def api_lt_close():
    ok = close_long_term_position("鎵嬪嫊骞冲€?)
    return jsonify({"ok": ok, "msg": "闀锋湡鍊変綅宸插钩鍊? if ok else "骞冲€夊け鏁?})

@app.route('/api/lt_analyze', methods=['POST'])
def api_lt_analyze():
    result = analyze_btc_market_trend()
    if result:
        with MARKET_LOCK:
            MARKET_STATE.update(result)
        update_state(market_info=dict(MARKET_STATE))
        check_long_term_position()
        return jsonify({"ok": True, "result": result})
    return jsonify({"ok": False, "msg": "鍒嗘瀽澶辨晽"})

@app.route('/api/reset_cooldown',methods=['POST'])
def api_reset_cooldown():
    with RISK_LOCK:
        RISK_STATE["cooldown_until"]    = None
        RISK_STATE["consecutive_loss"]  = 0
        RISK_STATE["trading_halted"]    = False
        RISK_STATE["halt_reason"]       = ""
    update_state(risk_status=get_risk_status(), halt_reason="")
    append_risk_event('manual_release', {'action': 'reset_cooldown'})
    print("鍐烽潨鏈熷凡鎵嬪嫊瑙ｉ櫎")
    return jsonify({"status":"ok","msg":"鍐烽潨鏈熷凡瑙ｉ櫎锛屾仮寰╀氦鏄?})

# =====================================================
# Gunicorn hook锛堝柈 worker锛?# =====================================================
# =====================================================
# 鍩疯绶掑畧璀凤細浠讳綍鍩疯绶掓鎺夎嚜鍕曢噸鍟?# =====================================================
def watchdog(target_func, name):
    """鍖呰９鍩疯绶掑嚱鏁革紝姝绘帀鑷嫊閲嶅暉锛堟崟鎹夋墍鏈夐尟瑾わ級"""
    while True:
        _set_backend_thread_state(name, 'starting', '婧栧倷鍟熷嫊')
        try:
            print("=== 鍩疯绶掑暉鍕? {} ===".format(name))
            _set_backend_thread_state(name, 'running', '鍩疯涓?)
            target_func()
            print("=== 鍩疯绶掓甯哥祼鏉燂紙涓嶆噳鐧肩敓锛? {} ===".format(name))
            _set_backend_thread_state(name, 'stopped', '鍩疯绶掓剰澶栫祼鏉?)
        except BaseException as e:
            import traceback
            print("=== 鍩疯绶掑穿娼?{} : {} ===".format(name, e))
            print(traceback.format_exc())
            _set_backend_thread_state(name, 'crashed', '鍩疯绶掑穿娼帮紝绛夊緟鑷嫊閲嶅暉', str(e))
        print("=== 鍩疯绶?绉掑緦閲嶅暉: {} ===".format(name))
        _set_backend_thread_state(name, 'restarting', '5绉掑緦鑷嫊閲嶅暉')
        time.sleep(5)

def start_all_threads_legacy_shadow_1():
    # 鍟熷嫊鏅傛仮寰╁倷浠界媭鎱?    load_full_state()
    load_risk_state()
    threads = [
        (news_thread,            "news"),
        (position_thread,        "position"),
        (scan_thread,            "scan"),
        (trailing_stop_thread,    "trailing"),
        (market_analysis_thread,  "market"),
        (fvg_order_monitor_thread,"fvg_monitor"),
    ]
    for fn, name in threads:
        t = threading.Thread(
            target=watchdog,
            args=(fn, name),
            daemon=True,
            name=name
        )
        t.start()
    print("=== 鎵€鏈夊煼琛岀窉宸插暉鍕曪紙鍚畧璀烽噸鍟熸鍒讹級===")

def post_fork(server, worker):
    start_all_threads()
    print("=== [worker {}] 鍟熷嫊瀹屾垚 ===".format(worker.pid))



# =====================================================
# V6 寮峰寲鐗堬細鏂瑰悜鍏堣 + 绲愭瑙哥櫦 + 棰ㄥ牨姣旈亷婵?# =====================================================
DIRECTION_STRONG_GATE = 3.2
DIRECTION_WEAK_GATE   = 2.0
NO_TRADE_CHOP_ADX     = 17
MAX_SIGNAL_AGE_BARS   = 3


def _clip(v, lo, hi):
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return lo


def _ema_bias(close_s, fast=9, mid=21, slow=55):
    curr = float(close_s.iloc[-1])
    e1 = safe_last(ta.ema(close_s, length=fast), curr)
    e2 = safe_last(ta.ema(close_s, length=mid), curr)
    e3 = safe_last(ta.ema(close_s, length=slow), curr)
    slope = _linreg_slope(close_s.tail(8).tolist()) / max(curr, 1e-9) * 100
    if curr > e1 > e2 > e3 and slope > 0.03:
        return 1
    if curr < e1 < e2 < e3 and slope < -0.03:
        return -1
    return 0


def _detect_pullback_trigger(d15, side):
    c = d15['c'].astype(float); o = d15['o'].astype(float); h = d15['h'].astype(float); l = d15['l'].astype(float); v = d15['v'].astype(float)
    curr = float(c.iloc[-1])
    atr = max(safe_last(ta.atr(h, l, c, length=14), curr * 0.004), curr * 0.003)
    ema9 = safe_last(ta.ema(c, length=9), curr)
    ema21 = safe_last(ta.ema(c, length=21), curr)
    ema55 = safe_last(ta.ema(c, length=55), curr)
    vol_now = float(v.tail(2).mean()) if len(v) >= 2 else float(v.iloc[-1])
    vol_avg = float(v.tail(20).mean()) if len(v) >= 20 else vol_now
    low1 = float(l.iloc[-1]); high1 = float(h.iloc[-1]); close1 = float(c.iloc[-1]); open1 = float(o.iloc[-1])
    body = abs(close1-open1)
    candle_range = max(high1-low1, 1e-9)
    close_pos = (close1-low1)/candle_range
    bearish_close_pos = (high1-close1)/candle_range
    ext = abs(curr-ema21)/max(atr,1e-9)
    hh = float(h.tail(20).iloc[:-1].max()) if len(h) > 21 else float(h.max())
    ll = float(l.tail(20).iloc[:-1].min()) if len(l) > 21 else float(l.min())

    if side > 0:
        ok = curr > ema21 > ema55 and ema9 >= ema21 and ext <= 0.95 and close_pos > 0.58 and body > atr * 0.18
        if ok:
            sl = min(float(l.tail(4).min()), ema21 - atr * 0.55)
            entry = curr
            tp = entry + max(entry - sl, atr * 0.8) * 2.4
            quality = 7 + (1 if vol_now > vol_avg * 1.05 else 0) + (1 if curr >= hh * 0.995 else 0)
            return True, '瓒ㄥ嫝鍥炶俯绾屾敾', quality, entry, sl, tp
    else:
        ok = curr < ema21 < ema55 and ema9 <= ema21 and ext <= 0.95 and bearish_close_pos > 0.58 and body > atr * 0.18
        if ok:
            sl = max(float(h.tail(4).max()), ema21 + atr * 0.55)
            entry = curr
            tp = entry - max(sl - entry, atr * 0.8) * 2.4
            quality = 7 + (1 if vol_now > vol_avg * 1.05 else 0) + (1 if curr <= ll * 1.005 else 0)
            return True, '瓒ㄥ嫝鍙嶅綀绾岃穼', quality, entry, sl, tp
    return False, '', 0, curr, curr, curr


def _normalize_pre_breakout_score(v, lo=0.0, hi=100.0):
    try:
        return round(max(lo, min(hi, float(v or 0.0))), 2)
    except Exception:
        return round(lo, 2)


def analyze_pre_breakout_radar(d15, d4h, d1d=None):
    """
    闈為樆鏂峰瀷闋愮垎鐧奸浄閬旓細
    - 鍙彁渚涜瀵?鎺掑簭/椤ず璩囪▕
    - 涓嶇洿鎺ュ奖闊?AI 鍒嗘暩銆佸缈掓ǎ鏈垨涓嬪柈 gating
    """
    try:
        if d15 is None or d4h is None or len(d15) < 80 or len(d4h) < 40:
            return {
                'ready': False, 'score': 0.0, 'direction': '涓€?, 'phase': '璩囨枡涓嶈冻',
                'long_score': 0.0, 'short_score': 0.0, 'tags': [], 'signals': {}, 'note': '闋愮垎鐧奸浄閬旇硣鏂欎笉瓒?,
            }

        c = d15['c'].astype(float); o = d15['o'].astype(float); h = d15['h'].astype(float); l = d15['l'].astype(float); v = d15['v'].astype(float)
        curr = float(c.iloc[-1])
        atr = max(safe_last(ta.atr(h, l, c, length=14), curr * 0.004), curr * 0.003)
        ema9 = safe_last(ta.ema(c, length=9), curr)
        ema21 = safe_last(ta.ema(c, length=21), curr)
        ema55 = safe_last(ta.ema(c, length=55), curr)

        bb = ta.bbands(c, length=20, std=2.0)
        if bb is None or bb.empty:
            return {
                'ready': False, 'score': 0.0, 'direction': '涓€?, 'phase': '鐒B',
                'long_score': 0.0, 'short_score': 0.0, 'tags': [], 'signals': {}, 'note': '闋愮垎鐧奸浄閬旂己灏戝竷鏋楄硣鏂?,
            }
        bb_up = safe_last(bb.iloc[:, 0], curr)
        bb_mid = safe_last(bb.iloc[:, 1], curr)
        bb_low = safe_last(bb.iloc[:, 2], curr)
        width_now = max((bb_up - bb_low) / max(bb_mid, 1e-9), 0.0)
        width_hist = ((bb.iloc[:, 0] - bb.iloc[:, 2]) / bb.iloc[:, 1].replace(0, np.nan)).dropna().tail(36)
        width_med = float(width_hist.median()) if len(width_hist) else width_now
        squeeze_ratio = width_now / max(width_med, 1e-9)

        atr_series = ta.atr(h, l, c, length=14)
        atr_now = safe_last(atr_series, atr)
        atr_recent = float(pd.Series(atr_series).tail(10).mean()) if atr_series is not None else atr_now
        atr_prev = float(pd.Series(atr_series).tail(40).head(20).mean()) if atr_series is not None else atr_now
        atr_prev = atr_prev if atr_prev == atr_prev and atr_prev > 0 else atr_now
        atr_compress = atr_recent / max(atr_prev, 1e-9)

        lookback = max(int(BREAKOUT_LOOKBACK or 20), 20)
        hh = float(h.tail(lookback).iloc[:-1].max()) if len(h) > lookback else float(h.max())
        ll = float(l.tail(lookback).iloc[:-1].min()) if len(l) > lookback else float(l.min())
        dist_high_atr = (hh - curr) / max(atr_now, 1e-9)
        dist_low_atr = (curr - ll) / max(atr_now, 1e-9)
        near_high = dist_high_atr <= 0.85 and curr <= hh * 1.004
        near_low = dist_low_atr <= 0.85 and curr >= ll * 0.996

        last_n = 8
        lows_slope = _linreg_slope(l.tail(last_n).tolist())
        highs_slope = _linreg_slope(h.tail(last_n).tolist())
        clamp_highs = highs_slope <= max(curr * 0.00012, atr_now * 0.02)
        clamp_lows = lows_slope >= -max(curr * 0.00012, atr_now * 0.02)

        vol_recent = float(v.tail(4).mean()) if len(v) >= 4 else float(v.iloc[-1])
        vol_mid = float(v.tail(16).head(8).mean()) if len(v) >= 16 else vol_recent
        vol_long = float(v.tail(32).mean()) if len(v) >= 32 else vol_recent
        vol_build = vol_recent > vol_mid * 1.04 and vol_recent < vol_long * 1.75 if vol_mid > 0 and vol_long > 0 else False

        body = abs(float(c.iloc[-1]) - float(o.iloc[-1]))
        candle_range = max(float(h.iloc[-1] - l.iloc[-1]), 1e-9)
        close_pos = (float(c.iloc[-1]) - float(l.iloc[-1])) / candle_range
        upper_close_pos = (float(h.iloc[-1]) - float(c.iloc[-1])) / candle_range

        ema21_4h = safe_last(ta.ema(d4h['c'], length=21), curr)
        ema55_4h = safe_last(ta.ema(d4h['c'], length=55), curr)
        trend_up_4h = curr >= ema21_4h >= ema55_4h
        trend_dn_4h = curr <= ema21_4h <= ema55_4h

        trend_up_1d = trend_dn_1d = False
        if d1d is not None and len(d1d) >= 30:
            ema20_1d = safe_last(ta.ema(d1d['c'], length=20), float(d1d['c'].iloc[-1]))
            ema50_1d = safe_last(ta.ema(d1d['c'], length=50), float(d1d['c'].iloc[-1]))
            day_curr = float(d1d['c'].iloc[-1])
            trend_up_1d = day_curr >= ema20_1d >= ema50_1d
            trend_dn_1d = day_curr <= ema20_1d <= ema50_1d

        compress_ok = squeeze_ratio <= 0.92 and atr_compress <= 0.94
        launch_pad_long = near_high and clamp_lows and (trend_up_4h or trend_up_1d)
        launch_pad_short = near_low and clamp_highs and (trend_dn_4h or trend_dn_1d)
        micro_trigger_long = curr >= ema9 >= ema21 and close_pos >= 0.58 and body >= atr_now * 0.16
        micro_trigger_short = curr <= ema9 <= ema21 and upper_close_pos >= 0.58 and body >= atr_now * 0.16

        long_score = 0.0
        short_score = 0.0
        tags = []
        signals = {}

        if compress_ok:
            long_score += 20; short_score += 20
            tags.append('娉㈠嫊鏀舵杺')
            signals['compression'] = True
        if vol_build:
            long_score += 11; short_score += 11
            tags.append('閲忚兘鍫嗙')
            signals['volume_build'] = True
        if near_high:
            long_score += 19
            tags.append('閫艰繎涓婄罚')
            signals['near_break_high'] = round(dist_high_atr, 2)
        if near_low:
            short_score += 19
            tags.append('閫艰繎涓嬬罚')
            signals['near_break_low'] = round(dist_low_atr, 2)
        if clamp_lows and lows_slope > 0:
            long_score += 16
            tags.append('浣庨粸鎶珮')
            signals['higher_lows'] = round(lows_slope, 6)
        if clamp_highs and highs_slope < 0:
            short_score += 16
            tags.append('楂橀粸涓嬪')
            signals['lower_highs'] = round(highs_slope, 6)
        if trend_up_4h:
            long_score += 10
            tags.append('4H鍋忓')
            signals['trend_4h_up'] = True
        if trend_dn_4h:
            short_score += 10
            tags.append('4H鍋忕┖')
            signals['trend_4h_dn'] = True
        if trend_up_1d:
            long_score += 5
            tags.append('鏃ョ窔鍋忓')
            signals['trend_1d_up'] = True
        if trend_dn_1d:
            short_score += 5
            tags.append('鏃ョ窔鍋忕┖')
            signals['trend_1d_dn'] = True
        if micro_trigger_long:
            long_score += 8
            tags.append('鐭窔绾屾敾')
            signals['micro_long'] = True
        if micro_trigger_short:
            short_score += 8
            tags.append('鐭窔绾岃穼')
            signals['micro_short'] = True

        long_score = _normalize_pre_breakout_score(long_score)
        short_score = _normalize_pre_breakout_score(short_score)
        direction = '涓€?
        score = max(long_score, short_score)
        phase = '瑙€瀵?
        note = '灏氭湭褰㈡垚鏄庣⒑闋愮垎鐧煎劒鍕?
        if long_score >= short_score + 8 and long_score >= 52:
            direction = '鍋忓闋愮垎鐧?
            phase = '钃勫嫝寰呯櫦' if long_score < 68 else '鎺ヨ繎绐佺牬'
            note = '鍋忓闋愮垎鐧兼浠惰純瀹屾暣锛屽彲瑙€瀵熶笂绶ｇ獊鐮?
        elif short_score >= long_score + 8 and short_score >= 52:
            direction = '鍋忕┖闋愮垎鐧?
            phase = '钃勫嫝寰呯櫦' if short_score < 68 else '鎺ヨ繎璺岀牬'
            note = '鍋忕┖闋愮垎鐧兼浠惰純瀹屾暣锛屽彲瑙€瀵熶笅绶ｈ穼鐮?
        elif score >= 40:
            phase = '鏃╂湡钃勫嫝'
            note = '宸叉湁閮ㄥ垎闋愮垎鐧兼浠讹紝浣嗗皻鏈泦涓埌鍠伌'

        return {
            'ready': bool(score >= 40.0),
            'score': round(score, 2),
            'direction': direction,
            'phase': phase,
            'long_score': round(long_score, 2),
            'short_score': round(short_score, 2),
            'tags': list(dict.fromkeys(tags))[:8],
            'signals': signals,
            'note': note,
            'dist_high_atr': round(dist_high_atr, 2),
            'dist_low_atr': round(dist_low_atr, 2),
            'squeeze_ratio': round(squeeze_ratio, 3),
            'atr_compress_ratio': round(atr_compress, 3),
            'volume_build_ratio': round(vol_recent / max(vol_mid, 1e-9), 3) if vol_mid > 0 else 0.0,
        }
    except Exception as e:
        return {
            'ready': False, 'score': 0.0, 'direction': '涓€?, 'phase': '闆烽仈澶辨晽',
            'long_score': 0.0, 'short_score': 0.0, 'tags': [], 'signals': {}, 'note': f'闋愮垎鐧奸浄閬斿け鏁?{e}',
        }


def _cache_pre_breakout_radar(symbol, radar):
    try:
        with PRE_BREAKOUT_RADAR_LOCK:
            PRE_BREAKOUT_RADAR_CACHE[symbol] = {
                'ts': time.time(),
                'radar': dict(radar or {}),
            }
    except Exception:
        pass


def _get_pre_breakout_radar(symbol, ttl=180):
    try:
        with PRE_BREAKOUT_RADAR_LOCK:
            row = dict(PRE_BREAKOUT_RADAR_CACHE.get(symbol) or {})
        if not row:
            return {}
        ts = float(row.get('ts', 0) or 0)
        if ttl and ts and (time.time() - ts) > ttl:
            return {}
        return dict(row.get('radar') or {})
    except Exception:
        return {}


def _detect_squeeze_break_trigger(d15, side):
    c = d15['c'].astype(float); o = d15['o'].astype(float); h = d15['h'].astype(float); l = d15['l'].astype(float); v = d15['v'].astype(float)
    curr = float(c.iloc[-1])
    atr = max(safe_last(ta.atr(h, l, c, length=14), curr * 0.004), curr * 0.003)
    bb = ta.bbands(c, length=20, std=2.0)
    if bb is None or bb.empty:
        return False, '', 0, curr, curr, curr
    bb_up = safe_last(bb.iloc[:, 0], curr); bb_low = safe_last(bb.iloc[:, 2], curr)
    width = (bb_up - bb_low) / max(curr, 1e-9)
    width_med = float(((bb.iloc[:, 0] - bb.iloc[:, 2]) / c).tail(30).median()) if len(c) >= 30 else width
    vol_now = float(v.tail(2).mean()) if len(v) >= 2 else float(v.iloc[-1])
    vol_avg = float(v.tail(20).mean()) if len(v) >= 20 else vol_now
    body = abs(float(c.iloc[-1]) - float(o.iloc[-1]))
    hh = float(h.tail(20).iloc[:-1].max()) if len(h) > 21 else float(h.max())
    ll = float(l.tail(20).iloc[:-1].min()) if len(l) > 21 else float(l.min())
    squeeze = width < width_med * 0.82

    if side > 0 and squeeze and curr >= hh * 0.999 and body > atr * 0.55 and vol_now > vol_avg * 1.18:
        sl = min(float(l.tail(3).min()), curr - atr * 1.1)
        tp = curr + max(curr - sl, atr * 0.9) * 2.9
        return True, '鏀舵杺绐佺牬鍟熷嫊', 9, curr, sl, tp
    if side < 0 and squeeze and curr <= ll * 1.001 and body > atr * 0.55 and vol_now > vol_avg * 1.18:
        sl = max(float(h.tail(3).max()), curr + atr * 1.1)
        tp = curr - max(sl - curr, atr * 0.9) * 2.9
        return True, '鏀舵杺璺岀牬鍟熷嫊', 9, curr, sl, tp
    return False, '', 0, curr, curr, curr


def _detect_sweep_reclaim_trigger(d15, side):
    c = d15['c'].astype(float); o = d15['o'].astype(float); h = d15['h'].astype(float); l = d15['l'].astype(float); v = d15['v'].astype(float)
    curr = float(c.iloc[-1])
    atr = max(safe_last(ta.atr(h, l, c, length=14), curr * 0.004), curr * 0.003)
    ema9 = safe_last(ta.ema(c, length=9), curr)
    ema21 = safe_last(ta.ema(c, length=21), curr)
    vol_now = float(v.tail(2).mean()) if len(v) >= 2 else float(v.iloc[-1])
    vol_avg = float(v.tail(20).mean()) if len(v) >= 20 else vol_now
    prior_low = float(l.tail(24).iloc[:-1].min()) if len(l) > 25 else float(l.min())
    prior_high = float(h.tail(24).iloc[:-1].max()) if len(h) > 25 else float(h.max())
    candle_range = max(float(h.iloc[-1] - l.iloc[-1]), 1e-9)
    close_pos = (float(c.iloc[-1]) - float(l.iloc[-1])) / candle_range
    upper_close_pos = (float(h.iloc[-1]) - float(c.iloc[-1])) / candle_range

    if side > 0:
        swept = float(l.iloc[-1]) < prior_low * 0.999 and curr > prior_low and curr > ema9 and close_pos > 0.65
        if swept and vol_now > vol_avg * 0.95:
            sl = float(l.iloc[-1]) - atr * 0.2
            tp = curr + max(curr - sl, atr * 0.75) * 2.2
            q = 8 + (1 if curr > ema21 else 0)
            return True, '娴佸嫊鎬ф巸浣庡洖鏀?, q, curr, sl, tp
    else:
        swept = float(h.iloc[-1]) > prior_high * 1.001 and curr < prior_high and curr < ema9 and upper_close_pos > 0.65
        if swept and vol_now > vol_avg * 0.95:
            sl = float(h.iloc[-1]) + atr * 0.2
            tp = curr - max(sl - curr, atr * 0.75) * 2.2
            q = 8 + (1 if curr < ema21 else 0)
            return True, '娴佸嫊鎬ф巸楂樺洖钀?, q, curr, sl, tp
    return False, '', 0, curr, curr, curr


def _detect_range_reversal_trigger(d15, side):
    c = d15['c'].astype(float); o = d15['o'].astype(float); h = d15['h'].astype(float); l = d15['l'].astype(float); v = d15['v'].astype(float)
    curr = float(c.iloc[-1])
    atr = max(safe_last(ta.atr(h, l, c, length=14), curr * 0.004), curr * 0.003)
    adx_df = ta.adx(h, l, c, length=14)
    adx = safe_last(adx_df['ADX_14'], 18) if adx_df is not None and 'ADX_14' in adx_df else 18
    bb = ta.bbands(c, length=20, std=2.0)
    if bb is None or bb.empty:
        return False, '', 0, curr, curr, curr
    bb_up = safe_last(bb.iloc[:, 0], curr)
    bb_low = safe_last(bb.iloc[:, 2], curr)
    bb_mid = safe_last(bb.iloc[:, 1], curr)
    width = (bb_up - bb_low) / max(curr, 1e-9)
    rsi = safe_last(ta.rsi(c, length=14), 50)
    vol_now = float(v.tail(2).mean()) if len(v) >= 2 else float(v.iloc[-1])
    vol_avg = float(v.tail(20).mean()) if len(v) >= 20 else vol_now
    body = abs(float(c.iloc[-1]) - float(o.iloc[-1]))
    candle_range = max(float(h.iloc[-1] - l.iloc[-1]), 1e-9)
    close_pos = (float(c.iloc[-1]) - float(l.iloc[-1])) / candle_range
    upper_close_pos = (float(h.iloc[-1]) - float(c.iloc[-1])) / candle_range
    mean_rev_ok = adx <= 22 and width <= 0.028

    if side > 0:
        touched_low = curr <= bb_low * 1.01 or float(l.iloc[-1]) <= bb_low * 1.003
        reclaim = curr >= bb_mid * 0.994 or close_pos >= 0.62
        if mean_rev_ok and touched_low and reclaim and rsi <= 44 and body <= atr * 1.35:
            sl = min(float(l.tail(3).min()), curr - atr * 1.15)
            tp = max(bb_mid, curr + max(curr - sl, atr * 0.85) * 1.9)
            quality = 7.6 + (0.5 if vol_now <= vol_avg * 1.2 else 0.0)
            return True, '鍗€闁撲笅绶ｅ弽褰?, quality, curr, sl, tp
    else:
        touched_up = curr >= bb_up * 0.99 or float(h.iloc[-1]) >= bb_up * 0.997
        reclaim = curr <= bb_mid * 1.006 or upper_close_pos >= 0.62
        if mean_rev_ok and touched_up and reclaim and rsi >= 56 and body <= atr * 1.35:
            sl = max(float(h.tail(3).max()), curr + atr * 1.15)
            tp = min(bb_mid, curr - max(sl - curr, atr * 0.85) * 1.9)
            quality = 7.6 + (0.5 if vol_now <= vol_avg * 1.2 else 0.0)
            return True, '鍗€闁撲笂绶ｅ洖钀?, quality, curr, sl, tp
    return False, '', 0, curr, curr, curr


def _best_setup_v6(d15, preferred_side):
    candidates = []
    for fn in (_detect_pullback_trigger, _detect_squeeze_break_trigger, _detect_sweep_reclaim_trigger, _detect_range_reversal_trigger):
        ok, label, quality, entry, sl, tp = fn(d15, preferred_side)
        if ok:
            candidates.append((quality, label, entry, sl, tp))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    quality, label, entry, sl, tp = candidates[0]
    return {'setup_label': label, 'setup_quality': quality, 'entry': entry, 'sl': sl, 'tp': tp}


def _direction_profile_v6(d15, d4h, d1d):
    c15 = d15['c'].astype(float); h15 = d15['h'].astype(float); l15 = d15['l'].astype(float)
    c4 = d4h['c'].astype(float); h4 = d4h['h'].astype(float); l4 = d4h['l'].astype(float)
    c1 = d1d['c'].astype(float)
    curr = float(c15.iloc[-1])
    adx15_df = ta.adx(h15, l15, c15, length=14)
    adx4_df = ta.adx(h4, l4, c4, length=14)
    adx15 = safe_last(adx15_df['ADX_14'], 20) if adx15_df is not None and 'ADX_14' in adx15_df else 20
    adx4 = safe_last(adx4_df['ADX_14'], 20) if adx4_df is not None and 'ADX_14' in adx4_df else 20
    bias15 = _ema_bias(c15)
    bias4 = _ema_bias(c4)
    bias1 = _ema_bias(c1, fast=9, mid=20, slow=50)
    consensus = bias15 * 1.1 + bias4 * 1.5 + bias1 * 1.7
    bb = ta.bbands(c15, length=20, std=2.0)
    bb_up = safe_last(bb.iloc[:, 0], curr) if bb is not None and not bb.empty else curr
    bb_low = safe_last(bb.iloc[:, 2], curr) if bb is not None and not bb.empty else curr
    width = (bb_up - bb_low) / max(curr, 1e-9)
    atr15 = max(safe_last(ta.atr(h15, l15, c15, length=14), curr * 0.004), curr * 0.003)
    range20 = float(h15.tail(20).max() - l15.tail(20).min()) if len(h15) >= 20 else float(h15.max() - l15.min())
    chop = adx15 < NO_TRADE_CHOP_ADX and width < 0.022 and range20 < atr15 * 7.5 and abs(consensus) < DIRECTION_WEAK_GATE
    if chop:
        return 0, 0.0, '闇囩洩闆滆▕鍗€', False, adx15, adx4
    side = 1 if consensus > 0 else -1 if consensus < 0 else 0
    strong = abs(consensus) >= DIRECTION_STRONG_GATE
    label = ('寮峰鍏辨尟' if side > 0 else '寮风┖鍏辨尟') if strong else ('鍋忓绲愭' if side > 0 else '鍋忕┖绲愭') if side != 0 else '鏂瑰悜涓嶈冻'
    return side, abs(consensus), label, strong, adx15, adx4


def _ai_adaptive_scoring_profile(symbol='', regime='neutral', setup='', side=0, direction_conf_view=0.0, setup_q=0.0, rr_ratio=0.0):
    """AI 鑷仼鎳夎鍒嗭細鍙彁渚涙渶鍩虹鐨勭壒寰靛弮鑰冭垏鍙缈掓瑠閲嶏紝涓嶅啀鐢ㄥぇ閲忓姝诲垎鏁稿叕寮忎富灏庛€?""
    try:
        profile = _ai_strategy_profile(symbol, regime=regime, setup=setup)
    except Exception:
        profile = {}
    try:
        eq_adj, eq_note = _entry_quality_feedback(symbol, regime, setup, setup_q)
    except Exception:
        eq_adj, eq_note = 0.0, ''

    phase = str(profile.get('phase') or 'learning')
    status = str(profile.get('status') or 'warmup')
    conf = float(profile.get('confidence', 0.0) or 0.0)
    wr = float(profile.get('win_rate', 0.0) or 0.0)
    ev = float(profile.get('ev_per_trade', 0.0) or 0.0)
    pf = float(profile.get('profit_factor', 0.0) or 0.0) if profile.get('profit_factor') is not None else 0.0
    dd = float(profile.get('max_drawdown_pct', 0.0) or 0.0) if profile.get('max_drawdown_pct') is not None else 0.0
    effective_count = float(profile.get('effective_count', profile.get('sample_count', 0.0)) or 0.0)
    fallback_level = str(profile.get('fallback_level') or 'global')

    # 鍩烘簴娆婇噸鍙繚鐣欑偤銆屽彲閬嬩綔鐨勬渶灏忛鏋躲€嶏紝鐪熸鍋忛噸鐢卞缈掔祼鏋滄帹鍕曘€?    adapt = {
        'w_dir': 1.0,
        'w_setup': 1.0,
        'w_rr': 1.0,
        'w_momentum': 1.0,
        'w_anti': 1.0,
        'w_htf': 1.0,
        'bias': 0.0,
        'quality_adj': float(eq_adj or 0.0),
        'profile': profile,
        'notes': [],
    }

    learn_power = max(0.18, min(1.0, effective_count / max(float(TREND_AI_FULL_TRADES or 50), 1.0)))
    conf_edge = max(-0.35, min(0.35, (conf - 0.22) * 1.6))
    wr_edge = max(-0.40, min(0.40, (wr - 50.0) / 25.0))
    ev_edge = max(-0.30, min(0.30, ev * 4.0))
    pf_edge = max(-0.25, min(0.25, (pf - 1.0) * 0.35))
    dd_edge = max(-0.35, min(0.35, (8.0 - dd) / 18.0)) if dd > 0 else 0.08

    adapt['w_dir'] += (conf_edge * 0.75 + pf_edge * 0.20) * learn_power
    adapt['w_setup'] += (wr_edge * 0.80 + ev_edge * 0.25 + float(eq_adj or 0.0) * 0.05) * learn_power
    adapt['w_rr'] += (ev_edge * 0.85 + pf_edge * 0.30) * learn_power
    adapt['w_momentum'] += (pf_edge * 0.60 + conf_edge * 0.18) * learn_power
    adapt['w_anti'] += max(-0.22, min(0.40, ((dd - 6.0) / 18.0) + (0.08 if status == 'reject' else -0.04 if status == 'valid' else 0.0))) * learn_power
    adapt['w_htf'] += max(-0.20, min(0.35, ((dd - 5.0) / 20.0) + (0.06 if fallback_level.startswith('global') else -0.03 if fallback_level.startswith('local') else 0.0))) * learn_power

    if phase == 'full':
        adapt['bias'] += 0.22
        adapt['notes'].append('AI鍏ㄦ帴绠?)
    elif phase == 'semi':
        adapt['bias'] += 0.10
        adapt['notes'].append('AI鍗婃帴绠?)
    else:
        adapt['bias'] -= 0.04 if effective_count < 8 else 0.0
        adapt['notes'].append('AI瀛哥繏涓?)

    if status == 'valid':
        adapt['bias'] += 0.12
        adapt['notes'].append('绛栫暐鏈夋晥')
    elif status == 'observe':
        adapt['bias'] += 0.03
        adapt['notes'].append('瑙€瀵熸ā寮?)
    elif status == 'reject':
        adapt['bias'] -= 0.14
        adapt['notes'].append('绛栫暐寮卞嫝')

    if fallback_level.startswith('global'):
        adapt['bias'] -= 0.05
        adapt['notes'].append('鍏ㄥ煙鍥為€€')
    elif fallback_level.startswith('mid'):
        adapt['notes'].append('涓堡鍥為€€')
    else:
        adapt['bias'] += 0.03
        adapt['notes'].append('灞€閮ㄦ帴绠?)

    if rr_ratio >= 2.0:
        adapt['w_rr'] += 0.05 * learn_power
    elif rr_ratio < 1.2:
        adapt['w_rr'] -= 0.08 * learn_power
        adapt['bias'] -= 0.05

    if eq_note:
        adapt['notes'].append(eq_note)

    for k in ('w_dir', 'w_setup', 'w_rr', 'w_momentum', 'w_anti', 'w_htf'):
        adapt[k] = round(max(0.55, min(1.85, float(adapt[k]))), 4)
    adapt['bias'] = round(max(-0.35, min(0.35, float(adapt['bias']))), 4)
    return adapt


def _grade_signal_v6(direction_conf, setup_q, rr, anti_chase_penalty, htf_penalty):
    """绛夌礆鍙仛鏈€鍩虹椤ず锛屼富楂旇鍒嗙敱 AI 鏈€绲傚垎鏁告帶鍒躲€?""
    dc = max(0.0, min(float(direction_conf or 0.0), 10.0))
    sq = max(0.0, min(float(setup_q or 0.0), 10.0))
    rrv = max(0.0, min(float(rr or 0.0), 3.5))
    anti = max(0.0, min(float(anti_chase_penalty or 0.0), 12.0))
    htf = max(0.0, min(float(htf_penalty or 0.0), 12.0))
    base = dc * 0.34 + sq * 0.40 + min(rrv / 2.0, 1.25) * 0.26
    penalty = anti * 0.015 + htf * 0.012
    composite = max(0.0, min(1.15, base - penalty))
    if composite >= 0.88:
        return 'A+'
    if composite >= 0.76:
        return 'A'
    if composite >= 0.63:
        return 'B+'
    if composite >= 0.50:
        return 'B'
    if composite >= 0.34:
        return 'C'
    return 'D'


def analyze_legacy_shadow_2(symbol):
    is_major = symbol in MAJOR_COINS
    try:
        d15 = pd.DataFrame(exchange.fetch_ohlcv(symbol, '15m', limit=ANALYZE_15M_LIMIT), columns=['t','o','h','l','c','v'])
        time.sleep(0.18)
        d4h = pd.DataFrame(exchange.fetch_ohlcv(symbol, '4h', limit=ANALYZE_4H_LIMIT), columns=['t','o','h','l','c','v'])
        time.sleep(0.18)
        d1d = pd.DataFrame(exchange.fetch_ohlcv(symbol, '1d', limit=ANALYZE_1D_LIMIT), columns=['t','o','h','l','c','v'])
        time.sleep(0.08)
        if len(d15) < 80 or len(d4h) < 40 or len(d1d) < 40:
            return 0, '璩囨枡涓嶈冻', 0, 0, 0, 0, {}, 0, 0, 0, 2.0, 3.0

        pre_breakout_radar = analyze_pre_breakout_radar(d15, d4h, d1d)
        _cache_pre_breakout_radar(symbol, pre_breakout_radar)

        curr = float(d15['c'].iloc[-1])
        atr15 = max(safe_last(ta.atr(d15['h'], d15['l'], d15['c'], length=14), curr * 0.004), curr * 0.003)
        atr4h = max(safe_last(ta.atr(d4h['h'], d4h['l'], d4h['c'], length=14), curr * 0.008), curr * 0.006)
        atr = atr15
        breakdown = {}
        tags = []

        side, direction_conf, direction_label, direction_strong, adx15, adx4 = _direction_profile_v6(d15, d4h, d1d)
        direction_conf_view = max(0.0, min(10.0, direction_conf * 2.2 + max(adx15 - 15.0, 0.0) * 0.08 + max(adx4 - 15.0, 0.0) * 0.05 + (0.8 if direction_strong else 0.0)))
        breakdown['鏂瑰悜淇″績'] = round(direction_conf_view, 1)
        breakdown['ADX15'] = round(adx15, 1)
        breakdown['ADX4H'] = round(adx4, 1)
        tags.append(direction_label)

        if side == 0:
            return 0, '闇囩洩閬庢烤|鏂瑰悜涓嶈冻', curr, 0, 0, 0, {'鏂瑰悜淇″績':0, 'Setup':'NoTrade', '绛夌礆':'D'}, atr, atr15, atr4h, 2.0, 3.0

        setup = _best_setup_v6(d15, side)
        if not setup:
            # 娌掓湁鏄庣⒑瑙哥櫦锛岀董鎸佽瀵燂紱AI 浠嶅彲渚濇鍙茶〃鐝惧井瑾跨瓑寰呭垎鏁革紝閬垮厤鏁存壒瑷婅櫉闀锋湡鍍垫銆?            wait_profile = _ai_adaptive_scoring_profile(symbol, regime='neutral', setup='wait', side=side, direction_conf_view=direction_conf_view, setup_q=0.0, rr_ratio=1.15)
            base = 22 + direction_conf_view * (3.7 + max(wait_profile.get('w_dir', 6.9) - 6.9, -0.6)) + max(adx15 - 18.0, 0.0) * 0.32 + float(wait_profile.get('bias', 0.0) or 0.0)
            capped = min(base, 44)
            wait_quality = round(max(2.2, min(6.8, direction_conf_view * 0.44 + max(adx15 - 16.0, 0.0) * 0.08 + max(adx4 - 16.0, 0.0) * 0.05 + float(wait_profile.get('quality_adj', 0.0) or 0.0) * 0.18)), 2)
            wait_trend_conf = round(max(0.0, min(direction_conf_view * 9.6 + max(adx4 - 15.0, 0.0) * 1.28 + float(wait_profile.get('bias', 0.0) or 0.0) * 1.2, 99.0)), 1)
            wait_regime_conf = round(max(0.0, min(direction_conf_view * 8.5 + max(adx15 - 14.0, 0.0) * 1.08 + float(wait_profile.get('bias', 0.0) or 0.0) * 0.9, 99.0)), 1)
            wait_direction = round(max(direction_conf_view * 0.62 + wait_trend_conf / 21.0 + wait_regime_conf / 25.0, wait_trend_conf / 10.8, wait_regime_conf / 11.8), 1)
            wait_grade = _grade_signal_v6(wait_direction, wait_quality, 1.15, 0, 0)
            return side * capped, '鏂瑰悜鏈変絾鏈埌瑙哥櫦浣峾绛夊緟鍥炶俯/绐佺牬纰鸿獚', curr, 0, 0, 0, {
                '鏂瑰悜淇″績': wait_direction, 'Setup':'绛夊緟瑙哥櫦', '閫插牬鍝佽唱': wait_quality, 'RR':0, '绛夌礆':wait_grade,
                'TrendConfidence': wait_trend_conf,
                'RegimeConfidence': wait_regime_conf,
                'AI瑭曞垎妯″紡': '|'.join((wait_profile.get('notes') or [])[:3]),
            }, atr, atr15, atr4h, 2.0, 3.0

        setup_label = setup['setup_label']
        entry = float(setup['entry'])
        sl = float(setup['sl'])
        tp = float(setup['tp'])
        setup_q = float(setup['setup_quality'])
        tags.append(setup_label)
        breakdown['Setup'] = setup_label

        current_regime = 'neutral'
        try:
            current_regime = str((_fetch_regime_for_symbol(symbol) or {}).get('regime', 'neutral'))
        except Exception:
            current_regime = 'neutral'
        base_sl_mult = round(abs(entry - sl) / max(atr15, 1e-9), 2)
        base_tp_mult = round(abs(tp - entry) / max(atr15, 1e-9), 2)
        learned_rr = get_learned_rr_target(
            symbol,
            current_regime,
            setup_label,
            [k for k, v in breakdown.items() if v != 0] + [setup_label],
            base_sl_mult,
            base_tp_mult,
        )
        risk_dist = abs(entry - sl)
        if side > 0:
            tp = entry + risk_dist * learned_rr
        else:
            tp = entry - risk_dist * learned_rr

        ema21 = safe_last(ta.ema(d15['c'], length=21), curr)
        ext_atr = abs(curr - ema21) / max(atr15, 1e-9)
        anti_chase_penalty = 0
        if ext_atr > 1.35:
            anti_chase_penalty += 9
            tags.append('杩藉児棰ㄩ毆楂?)
        elif ext_atr > 1.05:
            anti_chase_penalty += 4
            tags.append('鍋忛洟鍧囩窔')

        # 闈犺繎4H鍙嶅悜妤靛€兼檪闄嶆瑠
        hh4 = float(d4h['h'].tail(30).max())
        ll4 = float(d4h['l'].tail(30).min())
        htf_penalty = 0
        if side > 0 and (hh4 - curr) / max(atr4h, 1e-9) < 0.55:
            htf_penalty += 5
            tags.append('鎺ヨ繎4H澹撳姏')
        if side < 0 and (curr - ll4) / max(atr4h, 1e-9) < 0.55:
            htf_penalty += 5
            tags.append('鎺ヨ繎4H鏀拹')

        rr_ratio = abs(tp - entry) / max(abs(entry - sl), 1e-9)
        breakdown['LearnedRR'] = round(learned_rr, 2)
        if rr_ratio < 1.55:
            htf_penalty += 8
            tags.append('棰ㄥ牨姣斾笉瓒?)
        elif rr_ratio >= 2.3:
            tags.append('棰ㄥ牨姣斿劒绉€')

        # 瑁滀笂灏戦噺杓斿姪鍥犲瓙锛屼絾涓嶅啀璁撳畠鍊戜富灏庢柟鍚?        rsi = safe_last(ta.rsi(d15['c'], length=14), 50)
        macd = ta.macd(d15['c'])
        hist = safe_last(macd['MACDh_12_26_9'], 0) if macd is not None and 'MACDh_12_26_9' in macd else 0
        helper = 0
        if side > 0:
            if 46 <= rsi <= 66:
                helper += 5; tags.append('RSI澶氶牠鐢滆湝鍗€')
            elif rsi > 74:
                helper -= 4; tags.append('RSI閬庣啽')
            if hist > 0:
                helper += 4; tags.append('MACD闋嗗')
        else:
            if 34 <= rsi <= 54:
                helper += 5; tags.append('RSI绌洪牠鐢滆湝鍗€')
            elif rsi < 26:
                helper -= 4; tags.append('RSI閬庡喎')
            if hist < 0:
                helper += 4; tags.append('MACD闋嗙┖')

        # 澶х洡鍚屽悜鍔犲垎锛岄€嗗悜鎵ｅ垎
        try:
            with MARKET_LOCK:
                mdir = MARKET_STATE.get('direction', '涓€?)
            if side > 0 and mdir in ('澶?, '寮峰'):
                helper += 4; tags.append('澶х洡闋嗛ⅷ')
            elif side < 0 and mdir in ('绌?, '寮风┖'):
                helper += 4; tags.append('澶х洡闋嗛ⅷ')
            elif mdir != '涓€?:
                helper -= 3; tags.append('澶х洡閫嗛ⅷ')
        except Exception:
            pass

        rr_feat = max(0.0, min(1.25, (rr_ratio - 1.0) / 1.35))
        dir_feat = max(0.0, min(1.0, direction_conf_view / 10.0))
        setup_feat = max(0.0, min(1.0, setup_q / 10.0))
        momentum_feat = max(0.0, min(1.0, (helper + 10.0) / 20.0))
        anti_feat = max(0.0, min(1.0, anti_chase_penalty / 12.0))
        htf_feat = max(0.0, min(1.0, htf_penalty / 12.0))
        ai_adapt = _ai_adaptive_scoring_profile(symbol, regime=current_regime, setup=setup_label, side=side, direction_conf_view=direction_conf_view, setup_q=setup_q, rr_ratio=rr_ratio)
        pos_score = (
            dir_feat * float(ai_adapt.get('w_dir', 1.0) or 1.0)
            + setup_feat * float(ai_adapt.get('w_setup', 1.0) or 1.0)
            + rr_feat * float(ai_adapt.get('w_rr', 1.0) or 1.0)
            + momentum_feat * float(ai_adapt.get('w_momentum', 1.0) or 1.0)
        )
        neg_score = (
            anti_feat * float(ai_adapt.get('w_anti', 1.0) or 1.0)
            + htf_feat * float(ai_adapt.get('w_htf', 1.0) or 1.0)
        )
        denom = max(
            float(ai_adapt.get('w_dir', 1.0) or 1.0)
            + float(ai_adapt.get('w_setup', 1.0) or 1.0)
            + float(ai_adapt.get('w_rr', 1.0) or 1.0)
            + float(ai_adapt.get('w_momentum', 1.0) or 1.0)
            + float(ai_adapt.get('w_anti', 1.0) or 1.0)
            + float(ai_adapt.get('w_htf', 1.0) or 1.0),
            1e-9,
        )
        net_strength = (pos_score - neg_score) / denom
        if direction_strong:
            net_strength += 0.035
        net_strength += float(ai_adapt.get('bias', 0.0) or 0.0)
        score_abs = round(max(0.0, min(100.0, 50.0 + net_strength * 58.0)), 1)
        score = round(score_abs if side > 0 else -score_abs, 1)

        sl_mult = round(abs(entry - sl) / max(atr15, 1e-9), 2)
        tp_mult = round(abs(tp - entry) / max(atr15, 1e-9), 2)
        est_pnl = round(abs(tp - entry) / max(entry, 1e-9) * 100 * 20, 2)
        entry_quality = round(max(1.0, min(10.0, (setup_feat * 6.2 + dir_feat * 2.1 + rr_feat * 1.4 - anti_feat * 0.7 - htf_feat * 0.6) * 1.55 + float(ai_adapt.get('quality_adj', 0.0) or 0.0) * 0.15)), 1)
        direction_for_grade = max(direction_conf_view, min(9.9, direction_conf_view + float(ai_adapt.get('bias', 0.0) or 0.0) * 2.0))
        grade = _grade_signal_v6(direction_for_grade, entry_quality, rr_ratio, anti_chase_penalty, htf_penalty)

        breakdown['閫插牬鍝佽唱'] = entry_quality
        breakdown['RR'] = round(rr_ratio, 2)
        breakdown['Setup'] = setup_label
        trend_conf_val = round(max(0.0, min(99.0, (dir_feat * float(ai_adapt.get('w_dir', 1.0) or 1.0) + setup_feat * float(ai_adapt.get('w_setup', 1.0) or 1.0) + rr_feat * float(ai_adapt.get('w_rr', 1.0) or 1.0) - anti_feat * float(ai_adapt.get('w_anti', 1.0) or 1.0) * 0.6 - htf_feat * float(ai_adapt.get('w_htf', 1.0) or 1.0) * 0.45) / max((float(ai_adapt.get('w_dir', 1.0) or 1.0) + float(ai_adapt.get('w_setup', 1.0) or 1.0) + float(ai_adapt.get('w_rr', 1.0) or 1.0) + float(ai_adapt.get('w_anti', 1.0) or 1.0) * 0.6 + float(ai_adapt.get('w_htf', 1.0) or 1.0) * 0.45), 1e-9) * 100.0)), 1)
        regime_conf_val = round(max(0.0, min(99.0, (dir_feat * 0.65 + momentum_feat * 0.22 + rr_feat * 0.18 - htf_feat * 0.14 - anti_feat * 0.12 + float(ai_adapt.get('bias', 0.0) or 0.0) * 0.2) * 100.0)), 1)
        direction_display = round(max(direction_conf_view, trend_conf_val / 10.0, regime_conf_val / 10.5), 1)
        breakdown['鏂瑰悜淇″績'] = round(max(direction_display, 0.0), 1)
        breakdown['TrendConfidence'] = trend_conf_val
        breakdown['RegimeConfidence'] = regime_conf_val
        breakdown['RegimeBias'] = side * round(direction_conf_view, 2)
        breakdown['杩藉児棰ㄩ毆'] = -anti_chase_penalty if side > 0 else anti_chase_penalty
        breakdown['楂橀殠浣嶉殠澹撳姏'] = -htf_penalty if side > 0 else htf_penalty
        breakdown['绛夌礆'] = grade
        breakdown['杓斿姪鍥犲瓙'] = helper if side > 0 else -helper
        breakdown['AI瑭曞垎妯″紡'] = '|'.join((ai_adapt.get('notes') or [])[:4])
        breakdown['AI娆婇噸'] = {
            'dir': round(float(ai_adapt.get('w_dir', 1.0) or 1.0), 2),
            'setup': round(float(ai_adapt.get('w_setup', 1.0) or 1.0), 2),
            'rr': round(float(ai_adapt.get('w_rr', 1.0) or 1.0), 2),
            'mom': round(float(ai_adapt.get('w_momentum', 1.0) or 1.0), 2),
            'anti': round(float(ai_adapt.get('w_anti', 1.0) or 1.0), 2),
            'htf': round(float(ai_adapt.get('w_htf', 1.0) or 1.0), 2),
            'bias': round(float(ai_adapt.get('bias', 0) or 0), 2),
        }

        desc = '|'.join(tags[:8])
        return score, desc, round(entry, 6), round(sl, 6), round(tp, 6), est_pnl, breakdown, atr, atr15, atr4h, sl_mult, tp_mult

    except Exception as e:
        import traceback
        print('analyze {} 澶辨晽(v6): {}\n{}'.format(symbol, e, traceback.format_exc()[-400:]))
        return 0, '閷:{}'.format(str(e)[:40]), 0, 0, 0, 0, {}, 0, 0, 0, 2.0, 3.0


# =====================================================
# V7 AI 寮峰寲灞わ細甯傚牬璀樺垾 / 鑷嫊鍥炴脯 / 30绛嗚嚜瀛哥繏 / 瑷樻喍楂旂董璀?# 閫欏堡鐩存帴鐤婂姞鍦ㄥ師鏈郴绲变笂锛屼笉鎷挎帀鏃㈡湁鍔熻兘銆?# =====================================================
AI_DB_PATH = "/app/data/ai_learning_db.json"
AUTO_BACKTEST_STATE = {
    "running": False,
    "last_run": "--",
    "summary": "灏氭湭鍟熷嫊",
    "results": [],
    "target_count": 70,
    "scanned_markets": 0,
    "data_timeframes": ["5m", "15m", "1h", "4h", "1d"],
    "db_last_update": "--",
    "db_symbols": 0,
    "last_duration_sec": 0,
    "errors": [],
}
AI_PANEL = {
    "regime": "鍒濆鍖栦腑",
    "symbol_regimes": {},
    "best_strategies": [],
    "openai_trade": {},
    "params": {
        "sl_mult": 2.0,
        "tp_mult": 3.5,
        "breakeven_atr": 0.9,
        "trail_trigger_atr": 1.4,
        "trail_pct": 0.035,
        "score_boost": {},
    },
    "memory": {
        "score_cache": 0,
        "signal_meta_cache": 0,
        "entry_locks": 0,
        "protection_state": 0,
        "fvg_orders": 0,
    },
    "last_learning": "--",
    "last_backtest": "--",
    "market_db_info": {
        "symbols": 0,
        "timeframes": ["5m", "15m", "1h", "4h", "1d"],
        "last_update": "--",
    },
}
AI_LOCK = threading.Lock()
PENDING_LIMIT_META = {}
PENDING_LIMIT_LOCK = threading.RLock()
AI_MARKET_TIMEFRAMES = ['5m', '15m', '1h', '4h', '1d']
AI_MARKET_LIMIT = int(os.getenv('AI_MARKET_LIMIT', '120'))
COIN_SELECTOR_SCAN_LIMIT = int(os.getenv('COIN_SELECTOR_SCAN_LIMIT', '100'))
COIN_SELECTOR_PREFILTER_LIMIT = int(os.getenv('COIN_SELECTOR_PREFILTER_LIMIT', '220'))
COIN_SELECTOR_MIN_QUOTE_VOLUME = float(os.getenv('COIN_SELECTOR_MIN_QUOTE_VOLUME', '250000'))
COIN_SELECTOR_MIN_DAILY_MOVE_PCT = float(os.getenv('COIN_SELECTOR_MIN_DAILY_MOVE_PCT', '0.25'))
COIN_SELECTOR_MAX_DAILY_MOVE_PCT = float(os.getenv('COIN_SELECTOR_MAX_DAILY_MOVE_PCT', '14.0'))
COIN_SELECTOR_MAX_SPREAD_PCT = float(os.getenv('COIN_SELECTOR_MAX_SPREAD_PCT', '0.16'))
SYMBOL_COOLDOWN_MINUTES = int(os.getenv('SYMBOL_COOLDOWN_MINUTES', '90'))
SYMBOL_REPEAT_LOOKBACK = int(os.getenv('SYMBOL_REPEAT_LOOKBACK', '18'))
SYMBOL_REPEAT_PENALTY = float(os.getenv('SYMBOL_REPEAT_PENALTY', '4.0'))
SYMBOL_EXPLORATION_BONUS = float(os.getenv('SYMBOL_EXPLORATION_BONUS', '2.0'))
SYMBOL_BALANCE_TARGET_SHARE = float(os.getenv('SYMBOL_BALANCE_TARGET_SHARE', '0.18'))
SYMBOL_BALANCE_SOFT_CAP = float(os.getenv('SYMBOL_BALANCE_SOFT_CAP', '0.30'))
AI_BACKTEST_LIMIT = int(os.getenv('AI_BACKTEST_KLINE_LIMIT', '700'))
AI_SNAPSHOT_LIMIT = int(os.getenv('AI_SNAPSHOT_KLINE_LIMIT', '360'))
AI_BACKTEST_SLEEP_SEC = int(os.getenv('AI_BACKTEST_SLEEP_SEC', '7200'))
ANALYZE_15M_LIMIT = int(os.getenv('ANALYZE_15M_KLINE_LIMIT', '180'))
ANALYZE_4H_LIMIT = int(os.getenv('ANALYZE_4H_KLINE_LIMIT', '120'))
ANALYZE_1D_LIMIT = int(os.getenv('ANALYZE_1D_KLINE_LIMIT', '90'))

def _default_ai_db():
    return {
        "regime_stats": {},
        "symbol_regime_stats": {},
        "market_state_stats": {},
        "symbol_market_state_stats": {},
        "indicator_weights": {},
        "ai_feature_model": {"features": {}, "meta": {"samples": 0, "wins": 0, "avg_pnl": 0.0, "updated_at": "--"}},
        "combo_stats": {},
        "param_sets": {
            "trend":  {"sl_mult": 1.9, "tp_mult": 3.8, "breakeven_atr": 1.0, "trail_trigger_atr": 1.8, "trail_pct": 0.032},
            "range":  {"sl_mult": 1.5, "tp_mult": 2.2, "breakeven_atr": 0.7, "trail_trigger_atr": 1.1, "trail_pct": 0.022},
            "news":   {"sl_mult": 2.5, "tp_mult": 4.8, "breakeven_atr": 1.2, "trail_trigger_atr": 2.0, "trail_pct": 0.045},
            "neutral":{"sl_mult": 2.0, "tp_mult": 3.3, "breakeven_atr": 0.9, "trail_trigger_atr": 1.5, "trail_pct": 0.035},
        },
        "backtests": [],
        "strategy_scoreboard": [],
        "market_snapshots": {},
        "market_history_meta": {
            "symbols": 0,
            "timeframes": AI_MARKET_TIMEFRAMES,
            "last_update": "--",
        },
        "last_learning": "--",
        "version": 2,
    }

def load_ai_db():
    try:
        db = atomic_json_load(AI_DB_PATH, None)
        if db is None:
            return _default_ai_db()
        base = _default_ai_db()
        for k, v in base.items():
            db.setdefault(k, v)
        return db
    except Exception:
        return _default_ai_db()

def save_ai_db(db):
    try:
        atomic_json_save(AI_DB_PATH, db, ensure_ascii=False, indent=2)
    except Exception as e:
        print('AI DB 鍎插瓨澶辨晽:', e)

AI_DB = load_ai_db()


def _is_crypto_usdt_swap_symbol(symbol):
    try:
        if not isinstance(symbol, str):
            return False
        if not symbol.endswith(':USDT'):
            return False
        base = symbol.split('/')[0].split(':')[0].upper()
        banned = {
            'AAPL','GOOGL','GOOG','AMZN','TSLA','MSFT','META','NVDA','NFLX','BABA','BIDU','JD','PDD','NIO','XPEV','LI',
            'SNAP','TWTR','UBER','LYFT','ABNB','COIN','HOOD','AMC','GME','SPY','QQQ','DJI','MSTR','PLTR','SQ','PYPL',
            'SHOP','INTC','AMD','QCOM','AVGO'
        }
        return base not in banned
    except Exception:
        return False


def _ticker_num(data, *keys, default=0.0):
    data = dict(data or {})
    for key in keys:
        try:
            value = data.get(key)
            if value is None and isinstance(data.get('info'), dict):
                value = data['info'].get(key)
            if value in (None, ''):
                continue
            value = float(value)
            if math.isnan(value) or math.isinf(value):
                continue
            return value
        except Exception:
            continue
    return float(default)


def marketability_from_ticker(symbol, ticker):
    ticker = dict(ticker or {})
    quote_vol = _ticker_num(ticker, 'quoteVolume', 'quoteVolume24h', 'usdtVolume', default=0.0)
    base_vol = _ticker_num(ticker, 'baseVolume', default=0.0)
    last = _ticker_num(ticker, 'last', 'close', default=0.0)
    bid = _ticker_num(ticker, 'bid', default=0.0)
    ask = _ticker_num(ticker, 'ask', default=0.0)
    pct = abs(_ticker_num(ticker, 'percentage', 'changePercentage', default=0.0))
    if quote_vol <= 0 and base_vol > 0 and last > 0:
        quote_vol = base_vol * last

    if bid > 0 and ask > 0:
        spread_pct = (ask - bid) / max((ask + bid) / 2.0, 1e-9) * 100.0
    else:
        spread_pct = 0.0

    reasons = []
    score = 0.0
    score += min(max(math.log10(max(quote_vol, 1.0)) - 5.2, 0.0) / 2.6, 1.0) * 3.0
    if pct >= COIN_SELECTOR_MIN_DAILY_MOVE_PCT:
        score += min(pct / 4.5, 1.0) * 2.0
    else:
        score -= 1.2
        reasons.append('quiet_market')
    if COIN_SELECTOR_MIN_DAILY_MOVE_PCT <= pct <= COIN_SELECTOR_MAX_DAILY_MOVE_PCT:
        score += 1.0
    else:
        score -= 1.6
        reasons.append('daily_move_outlier')
    if spread_pct > 0:
        if spread_pct <= COIN_SELECTOR_MAX_SPREAD_PCT:
            score += 1.2
        else:
            score -= min(2.0, (spread_pct - COIN_SELECTOR_MAX_SPREAD_PCT) * 8.0)
            reasons.append('wide_spread')
    if quote_vol < COIN_SELECTOR_MIN_QUOTE_VOLUME:
        score -= 2.0
        reasons.append('thin_volume')
    if symbol in SHORT_TERM_EXCLUDED:
        score -= 4.0
        reasons.append('short_term_excluded')

    score = round(max(0.0, min(score, 7.2)), 3)
    return {
        'score': score,
        'quote_volume': round(quote_vol, 2),
        'daily_move_pct': round(pct, 3),
        'spread_pct': round(spread_pct, 4),
        'tradable': bool(score >= 2.2 and quote_vol >= COIN_SELECTOR_MIN_QUOTE_VOLUME),
        'reasons': reasons,
    }


def rank_tradable_markets(tickers, limit=140):
    rows = []
    for sym, data in (tickers or {}).items():
        if not _is_crypto_usdt_swap_symbol(sym):
            continue
        marketability = marketability_from_ticker(sym, data)
        if not marketability.get('tradable') and len(rows) >= max(20, int(limit or 140)):
            continue
        rows.append((sym, data, marketability))
    rows.sort(
        key=lambda x: (
            float((x[2] or {}).get('score', 0.0) or 0.0),
            float((x[2] or {}).get('quote_volume', 0.0) or 0.0),
        ),
        reverse=True,
    )
    return rows[:max(1, int(limit or 140))]


def _ticker_pct_change(ticker):
    return _ticker_num(ticker, 'percentage', 'changePercentage', 'priceChangePercent', default=0.0)


def rank_short_gainer_markets(tickers, limit=10, min_pct=None):
    min_pct = OPENAI_SHORT_GAINERS_MIN_24H_PCT if min_pct is None else float(min_pct or 0.0)
    rows = []
    for sym, data in (tickers or {}).items():
        if not _is_crypto_usdt_swap_symbol(sym):
            continue
        ticker = dict(data or {})
        pct = _ticker_pct_change(ticker)
        if pct < min_pct:
            continue
        marketability = marketability_from_ticker(sym, ticker)
        if not marketability.get('tradable', False):
            continue
        spread = float(marketability.get('spread_pct', 0) or 0)
        quote_volume = float(marketability.get('quote_volume', 0) or 0)
        score = (
            pct * 1.15
            + float(marketability.get('score', 0) or 0) * 3.0
            + min(max(math.log10(max(quote_volume, 1.0)) - 5.0, 0.0), 4.0)
            - min(max(spread - COIN_SELECTOR_MAX_SPREAD_PCT, 0.0) * 16.0, 8.0)
        )
        rows.append((sym, score, marketability, pct, ticker))
    rows.sort(
        key=lambda x: (
            float(x[1] or 0.0),
            float((x[2] or {}).get('quote_volume', 0.0) or 0.0),
            float(x[3] or 0.0),
        ),
        reverse=True,
    )
    return rows[:max(1, int(limit or 10))]


def _enrich_signal_for_order_review(sig):
    sig = dict(sig or {})
    try:
        ctx = infer_margin_context(sig, same_side_count=0)
        sig['margin_pct'] = ctx.get('margin_pct', RISK_PCT)
        sig['margin_ctx'] = ctx
    except Exception:
        sig['margin_pct'] = RISK_PCT
    try:
        rot_adj, rot_notes = _symbol_rotation_adjustment(sig.get('symbol', ''))
    except Exception:
        rot_adj, rot_notes = 0.0, []
    sig['rotation_adj'] = rot_adj
    sig['rotation_notes'] = rot_notes
    try:
        selection_edge, selection_notes = coin_selection_edge(sig)
    except Exception:
        selection_edge, selection_notes = 0.0, []
    sig['selection_edge'] = selection_edge
    sig['selection_notes'] = selection_notes
    sig['priority_score'] = round(
        abs(float(sig.get('score', 0) or 0))
        + float(rot_adj or 0)
        + float(selection_edge or 0)
        + float(sig.get('entry_quality', 0) or 0) * 0.15
        + min(float(sig.get('rr_ratio', 0) or 0), 3.0) * 0.12,
        2,
    )
    return sig


def build_short_gainer_signal(row, existing=None):
    sym, rank_score, marketability, pct, ticker = row
    ticker = dict(ticker or {})
    marketability = dict(marketability or {})
    existing = dict(existing or {})
    try:
        if existing:
            pr = _float_or_zero(existing.get('price'))
            atr = _float_or_zero(existing.get('atr')) or pr * 0.006
            atr15 = _float_or_zero(existing.get('atr15')) or atr
            atr4h = _float_or_zero(existing.get('atr4h')) or atr
            bd = dict(existing.get('breakdown') or {})
            desc = str(existing.get('desc') or '')
            ep = _float_or_zero(existing.get('est_pnl'))
            sl_m = _float_or_zero(existing.get('sl_mult')) or 1.6
            tp_m = _float_or_zero(existing.get('tp_mult')) or 2.4
            allowed = bool(existing.get('allowed', True))
            sym_n = int(existing.get('sym_trades', 0) or 0)
            sym_wr = existing.get('sym_wr', 0)
            status = str(existing.get('status') or '')
        else:
            sc, desc, pr, _sl, _tp, ep, bd, atr, atr15, atr4h, sl_m, tp_m = analyze(sym)
            allowed, sym_n, sym_wr = is_symbol_allowed(sym)
            status = "watching({}%)".format(sym_wr) if not allowed else ""
    except Exception as e:
        print('short_gainer analyze skip {}: {}'.format(sym, e))
        return None
    if pr <= 0:
        pr = _ticker_num(ticker, 'last', 'close', 'markPrice', 'bid', 'ask', default=0.0)
    if pr <= 0:
        return None
    atr_base = max(_float_or_zero(atr15), _float_or_zero(atr), pr * 0.006)
    local_trigger_price = max(pr - atr_base * 0.65, pr * 0.05)
    local_zone_low = max(local_trigger_price - atr_base * 0.25, pr * 0.05)
    local_zone_high = local_trigger_price + atr_base * 0.45
    local_invalidation_price = pr + atr_base * 1.05
    sl = pr + atr_base * 1.65
    tp = max(pr - atr_base * 2.45, pr * 0.05)
    risk = max(sl - pr, pr * 0.002)
    reward = max(pr - tp, 0.0)
    rr = round(reward / max(risk, 1e-9), 2)
    forced_abs_score = max(42.0, min(76.0, 36.0 + float(pct or 0) * 1.25 + float(marketability.get('score', 0) or 0) * 2.2))
    bd = dict(bd or {})
    bd.update({
        'Setup': 'short_gainer_fade',
        'OpenAISource': 'short_gainers',
        'ScannerIntent': 'evaluate_short_after_24h_gainer',
        'ShortGainer24hPct': round(float(pct or 0), 3),
        'ShortGainerRankScore': round(float(rank_score or 0), 3),
        'RR': rr,
        'EntryGate': max(float(bd.get('EntryGate', bd.get('閫插牬鍝佽唱', 0)) or 0), 2.0),
        'RegimeBias': min(float(bd.get('RegimeBias', 0) or 0), -0.2),
    })
    ref = {
        'source': 'short_gainers',
        'summary': '24h top gainer is being evaluated for a possible exhaustion short.',
        'bias': 'short',
        'setup': 'Do not short only because it rose; require exhaustion, failed continuation, reclaim failure, or breakdown confirmation.',
        'risk': 'Strong gainers can keep squeezing upward; invalidation must be clear and close to structure.',
        'note': '24h_pct={} rank_score={} quote_volume={} spread_pct={}'.format(
            round(float(pct or 0), 3),
            round(float(rank_score or 0), 3),
            marketability.get('quote_volume', 0),
            marketability.get('spread_pct', 0),
        ),
        'checklist': 'upper wick / failed high / lower high / volume climax / VWAP loss / support breakdown',
    }
    sig = {
        'symbol': sym,
        'score': -round(forced_abs_score, 2),
        'raw_score': -round(forced_abs_score, 2),
        'desc': 'short gainer leaderboard fade candidate | 24h +{:.2f}% | {}'.format(float(pct or 0), desc),
        'price': pr,
        'stop_loss': sl,
        'take_profit': tp,
        'est_pnl': ep,
        'direction': 'short',
        'breakdown': bd,
        'atr': atr,
        'atr15': atr15,
        'atr4h': atr4h,
        'sl_mult': sl_m,
        'tp_mult': tp_m,
        'allowed': allowed,
        'status': status,
        'watch_status': 'local_watch',
        'local_watch_trigger_type': 'breakdown_confirm',
        'local_watch_trigger_price': round(local_trigger_price, 8),
        'local_watch_invalidation_price': round(local_invalidation_price, 8),
        'local_watch_zone_low': round(local_zone_low, 8),
        'local_watch_zone_high': round(local_zone_high, 8),
        'local_watch_note': 'Track first; ask OpenAI only after price breaks local trigger and begins confirming a short-gainer fade.',
        'sym_trades': sym_n,
        'sym_wr': sym_wr,
        'margin_pct': 0,
        'entry_quality': bd.get('EntryGate', 2.0),
        'rr_ratio': rr,
        'regime_bias': min(float(bd.get('RegimeBias', 0) or 0), -0.2),
        'setup_label': 'short_gainer_fade',
        'signal_grade': bd.get('绛夌礆', ''),
        'direction_confidence': max(2.0, min(8.0, float(pct or 0) / 2.5)),
        'regime': bd.get('Regime', 'neutral'),
        'regime_confidence': bd.get('RegimeConfidence', bd.get('TrendConfidence', 0)),
        'trend_confidence': bd.get('TrendConfidence', 0),
        'score_jump': '',
        'marketability': marketability,
        'marketability_score': marketability.get('score', 0.0),
        'candidate_source': 'short_gainers',
        'source': 'short_gainers',
        'scanner_intent': 'Ask OpenAI whether a 24h top gainer is now suitable to short.',
        'external_reference': ref,
        'short_gainer_context': {
            'pct_24h': round(float(pct or 0), 3),
            'rank_score': round(float(rank_score or 0), 3),
            'quote_volume': marketability.get('quote_volume', 0),
            'spread_pct': marketability.get('spread_pct', 0),
            'ticker_last': _ticker_num(ticker, 'last', 'close', 'markPrice', default=0.0),
            'local_trigger_price': round(local_trigger_price, 8),
            'local_invalidation_price': round(local_invalidation_price, 8),
            'local_zone_low': round(local_zone_low, 8),
            'local_zone_high': round(local_zone_high, 8),
        },
    }
    return _enrich_signal_for_order_review(sig)


def _short_gainer_local_trigger(sig, now_ts=None):
    sig = dict(sig or {})
    price = _float_or_zero(sig.get('price'))
    trigger = _float_or_zero(sig.get('local_watch_trigger_price') or (sig.get('short_gainer_context') or {}).get('local_trigger_price'))
    invalidation = _float_or_zero(sig.get('local_watch_invalidation_price') or (sig.get('short_gainer_context') or {}).get('local_invalidation_price'))
    if price <= 0 or trigger <= 0:
        return False, ''
    if invalidation > 0 and price >= invalidation:
        return False, 'short gainer local watch invalidated'
    if price <= trigger:
        return True, 'short_gainer local breakdown_confirm hit at {}'.format(round(price, 8))
    return False, ''


def fetch_top_volume_symbols(limit=70):
    try:
        tickers = exchange.fetch_tickers()
        ranked = rank_tradable_markets(tickers, limit=max(int(limit) * 2, int(limit)))
        symbols = [sym for sym, _, marketability in ranked if marketability.get('tradable', True)][:max(1, int(limit))]
        return symbols, len(ranked)
    except Exception as e:
        print('鎶撳墠{}鎴愪氦閲忓競鍫村け鏁? {}'.format(limit, e))
        return [], 0


def _safe_fetch_ohlcv_df(symbol, timeframe, limit):
    try:
        rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(rows, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        if df.empty:
            return None
        return df
    except Exception as e:
        print('鎶揔绶氬け鏁?{} {}: {}'.format(symbol, timeframe, e))
        return None


def _safe_round_metric(value, digits=6, default=0.0):
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            v = float(default)
        return round(v, digits)
    except Exception:
        return round(float(default), digits)


def _candle_shape_from_row(row):
    if row is None:
        return {}
    try:
        o = float(row['o'])
        h = float(row['h'])
        l = float(row['l'])
        c = float(row['c'])
        v = float(row['v'])
    except Exception:
        return {}
    candle_range = max(h - l, 1e-9)
    body = abs(c - o)
    upper_wick = max(h - max(o, c), 0.0)
    lower_wick = max(min(o, c) - l, 0.0)
    body_pct = body / candle_range
    upper_pct = upper_wick / candle_range
    lower_pct = lower_wick / candle_range
    close_pos = (c - l) / candle_range
    direction = 'bullish' if c > o else 'bearish' if c < o else 'flat'

    if body_pct <= 0.18:
        shape = 'doji'
    elif lower_pct >= 0.42 and upper_pct <= 0.20:
        shape = 'hammer' if direction == 'bullish' else 'hanging_man'
    elif upper_pct >= 0.42 and lower_pct <= 0.20:
        shape = 'shooting_star' if direction == 'bearish' else 'inverted_hammer'
    elif body_pct >= 0.72 and upper_pct <= 0.12 and lower_pct <= 0.12:
        shape = 'marubozu'
    elif body_pct >= 0.52:
        shape = 'strong_body'
    else:
        shape = 'mixed_body'

    return {
        'open': _safe_round_metric(o, 8),
        'high': _safe_round_metric(h, 8),
        'low': _safe_round_metric(l, 8),
        'close': _safe_round_metric(c, 8),
        'volume': _safe_round_metric(v, 4),
        'direction': direction,
        'shape': shape,
        'body_pct': _safe_round_metric(body_pct * 100.0, 2),
        'upper_wick_pct': _safe_round_metric(upper_pct * 100.0, 2),
        'lower_wick_pct': _safe_round_metric(lower_pct * 100.0, 2),
        'range_pct_of_price': _safe_round_metric(candle_range / max(c, 1e-9) * 100.0, 3),
        'close_position_pct': _safe_round_metric(close_pos * 100.0, 2),
    }


def _ema_stack_label(last, ema9, ema20, ema50, ema200=None):
    ema200 = float(ema200 or ema50 or last)
    if last >= ema9 >= ema20 >= ema50 >= ema200:
        return 'strong_uptrend'
    if last >= ema9 >= ema20 >= ema50:
        return 'uptrend'
    if last <= ema9 <= ema20 <= ema50 <= ema200:
        return 'strong_downtrend'
    if last <= ema9 <= ema20 <= ema50:
        return 'downtrend'
    if last >= ema20 >= ema50:
        return 'recovery_up'
    if last <= ema20 <= ema50:
        return 'recovery_down'
    return 'mixed'


def _snapshot_from_df(df):
    if df is None or len(df) < 20:
        return None
    closed_df = df.iloc[:-1].copy() if len(df) >= 30 else df.copy()
    if closed_df is None or len(closed_df) < 20:
        closed_df = df.copy()
    c = closed_df['c'].astype(float)
    o = closed_df['o'].astype(float)
    h = closed_df['h'].astype(float)
    l = closed_df['l'].astype(float)
    v = closed_df['v'].astype(float)
    last = float(c.iloc[-1])
    prev = float(c.iloc[-2]) if len(c) >= 2 else last
    atr = safe_last(ta.atr(h, l, c, length=14), 0)
    ema9 = safe_last(ta.ema(c, length=9), last)
    ema20 = safe_last(ta.ema(c, length=20), last)
    ema50 = safe_last(ta.ema(c, length=50), last)
    ema100 = safe_last(ta.ema(c, length=100), ema50 if len(c) >= 100 else last)
    ema200 = safe_last(ta.ema(c, length=200), ema50 if len(c) >= 60 else last)
    ma20 = safe_last(ta.sma(c, length=20), last)
    ma60 = safe_last(ta.sma(c, length=60), ma20 if len(c) >= 60 else last)
    rsi = safe_last(ta.rsi(c, length=14), 50)
    adx_df = ta.adx(h, l, c, length=14)
    adx = safe_last(adx_df.iloc[:, 0], 0) if adx_df is not None and not adx_df.empty else 0
    plus_di = safe_last(adx_df.iloc[:, 1], 0) if adx_df is not None and adx_df.shape[1] >= 2 else 0
    minus_di = safe_last(adx_df.iloc[:, 2], 0) if adx_df is not None and adx_df.shape[1] >= 3 else 0
    macd_df = ta.macd(c, fast=12, slow=26, signal=9)
    macd = safe_last(macd_df.iloc[:, 0], 0) if macd_df is not None and not macd_df.empty else 0
    macd_signal = safe_last(macd_df.iloc[:, 1], 0) if macd_df is not None and macd_df.shape[1] >= 2 else 0
    macd_hist = safe_last(macd_df.iloc[:, 2], 0) if macd_df is not None and macd_df.shape[1] >= 3 else 0
    bb = ta.bbands(c, length=20, std=2.0)
    bb_upper = safe_last(bb.iloc[:, 0], last) if bb is not None and not bb.empty else last
    bb_mid = safe_last(bb.iloc[:, 1], last) if bb is not None and bb.shape[1] >= 2 else last
    bb_lower = safe_last(bb.iloc[:, 2], last) if bb is not None and bb.shape[1] >= 3 else last
    bb_width_pct = (bb_upper - bb_lower) / max(last, 1e-9) * 100.0
    bb_pos = (last - bb_lower) / max(bb_upper - bb_lower, 1e-9)
    stoch_df = ta.stoch(h, l, c, k=14, d=3, smooth_k=3)
    stoch_k = safe_last(stoch_df.iloc[:, 0], 50) if stoch_df is not None and not stoch_df.empty else 50
    stoch_d = safe_last(stoch_df.iloc[:, 1], 50) if stoch_df is not None and stoch_df.shape[1] >= 2 else 50
    kdj_j = stoch_k * 3.0 - stoch_d * 2.0
    typical_price = (h + l + c) / 3.0
    vol_sum = float(v.sum()) if len(v) else 0.0
    vwap = float((typical_price * v).sum() / max(vol_sum, 1e-9)) if vol_sum > 0 else last
    avg_volume_20 = float(v.tail(20).mean()) if len(v) >= 20 else float(v.mean())
    avg_volume_60 = float(v.tail(60).mean()) if len(v) >= 60 else avg_volume_20
    last_volume = float(v.iloc[-1]) if len(v) else 0.0
    vol_ratio = float(v.tail(5).mean()) / max(float(v.tail(30).mean()), 1e-9)
    ret_3 = 0.0
    ret_12 = 0.0
    ret_24 = 0.0
    if len(c) >= 4:
        base = float(c.iloc[-4])
        ret_3 = (last - base) / max(base, 1e-9) * 100.0
    if len(c) >= 13:
        base = float(c.iloc[-13])
        ret_12 = (last - base) / max(base, 1e-9) * 100.0
    if len(c) >= 25:
        base = float(c.iloc[-25])
        ret_24 = (last - base) / max(base, 1e-9) * 100.0
    last_closed_candle = _candle_shape_from_row(closed_df.iloc[-1])
    recent_closed = []
    for _, row in closed_df.tail(3).iterrows():
        recent_closed.append(_candle_shape_from_row(row))
    swing_high_20 = float(h.tail(20).max()) if len(h) >= 20 else float(h.max())
    swing_low_20 = float(l.tail(20).min()) if len(l) >= 20 else float(l.min())
    pivot_highs = []
    pivot_lows = []
    wing = 2
    start_idx = max(len(closed_df) - 80, wing)
    end_idx = max(len(closed_df) - wing, start_idx)
    for idx in range(start_idx, end_idx):
        hi = float(h.iloc[idx])
        lo = float(l.iloc[idx])
        if hi >= float(h.iloc[idx - wing:idx + wing + 1].max()):
            pivot_highs.append(hi)
        if lo <= float(l.iloc[idx - wing:idx + wing + 1].min()):
            pivot_lows.append(lo)
    resistance_levels = sorted({round(float(x), 8) for x in pivot_highs if float(x) >= last})[:3]
    support_levels = sorted({round(float(x), 8) for x in pivot_lows if float(x) <= last}, reverse=True)[:3]
    recent_structure_high = max(resistance_levels[:1] or [swing_high_20])
    recent_structure_low = min(support_levels[:1] or [swing_low_20])
    return {
        'bars': int(len(closed_df)),
        'last_close': _safe_round_metric(last, 8),
        'prev_close': _safe_round_metric(prev, 8),
        'atr': _safe_round_metric(float(atr or 0), 8),
        'atr_pct': _safe_round_metric(float(atr or 0) / max(last, 1e-9) * 100.0, 3),
        'rsi': _safe_round_metric(float(rsi or 0), 2),
        'adx': _safe_round_metric(float(adx or 0), 2),
        'plus_di': _safe_round_metric(float(plus_di or 0), 2),
        'minus_di': _safe_round_metric(float(minus_di or 0), 2),
        'ema9': _safe_round_metric(float(ema9 or 0), 8),
        'ema20': _safe_round_metric(float(ema20 or 0), 8),
        'ema50': _safe_round_metric(float(ema50 or 0), 8),
        'ema100': _safe_round_metric(float(ema100 or 0), 8),
        'ema200': _safe_round_metric(float(ema200 or 0), 8),
        'ma20': _safe_round_metric(float(ma20 or 0), 8),
        'ma60': _safe_round_metric(float(ma60 or 0), 8),
        'vwap': _safe_round_metric(float(vwap or 0), 8),
        'trend_label': _ema_stack_label(last, ema9, ema20, ema50, ema200),
        'macd': _safe_round_metric(float(macd or 0), 8),
        'macd_signal': _safe_round_metric(float(macd_signal or 0), 8),
        'macd_hist': _safe_round_metric(float(macd_hist or 0), 8),
        'bb_upper': _safe_round_metric(float(bb_upper or 0), 8),
        'bb_mid': _safe_round_metric(float(bb_mid or 0), 8),
        'bb_lower': _safe_round_metric(float(bb_lower or 0), 8),
        'bb_width_pct': _safe_round_metric(bb_width_pct, 3),
        'bb_position_pct': _safe_round_metric(bb_pos * 100.0, 2),
        'stoch_k': _safe_round_metric(float(stoch_k or 0), 2),
        'stoch_d': _safe_round_metric(float(stoch_d or 0), 2),
        'kdj_j': _safe_round_metric(float(kdj_j or 0), 2),
        'ret_3bars_pct': _safe_round_metric(ret_3, 2),
        'ret_12bars_pct': _safe_round_metric(ret_12, 2),
        'ret_24bars_pct': _safe_round_metric(ret_24, 2),
        'vol_ratio': _safe_round_metric(vol_ratio, 2),
        'avg_volume_20': _safe_round_metric(avg_volume_20, 4),
        'avg_volume_60': _safe_round_metric(avg_volume_60, 4),
        'last_volume': _safe_round_metric(last_volume, 4),
        'swing_high_20': _safe_round_metric(swing_high_20, 8),
        'swing_low_20': _safe_round_metric(swing_low_20, 8),
        'distance_to_swing_high_pct': _safe_round_metric((swing_high_20 - last) / max(last, 1e-9) * 100.0, 3),
        'distance_to_swing_low_pct': _safe_round_metric((last - swing_low_20) / max(last, 1e-9) * 100.0, 3),
        'support_levels': support_levels[:3],
        'resistance_levels': resistance_levels[:3],
        'recent_structure_high': _safe_round_metric(recent_structure_high, 8),
        'recent_structure_low': _safe_round_metric(recent_structure_low, 8),
        'last_closed_candle': last_closed_candle,
        'recent_closed_candles': recent_closed,
    }


def _context_cache_get(bucket, key, ttl_sec):
    if ttl_sec <= 0:
        return None
    with CACHE_LOCK:
        row = dict(bucket.get(key) or {})
    if not row:
        return None
    ts = float(row.get('ts', 0) or 0)
    if ts <= 0 or (time.time() - ts) > ttl_sec:
        return None
    return row.get('data')


def _context_cache_set(bucket, key, data):
    with CACHE_LOCK:
        bucket[key] = {
            'ts': time.time(),
            'data': data,
        }


def _symbol_base_asset(symbol):
    return str(symbol or '').split('/')[0].split(':')[0].strip().upper()


def _symbol_quote_asset(symbol):
    tail = str(symbol or '').split('/')[1] if '/' in str(symbol or '') else 'USDT'
    return tail.split(':')[0].strip().upper()


def _bitget_rest_symbol(symbol):
    base = _symbol_base_asset(symbol)
    quote = _symbol_quote_asset(symbol)
    return '{}{}'.format(base, quote)


def _bitget_product_type(symbol):
    quote = _symbol_quote_asset(symbol)
    if quote == 'USDT':
        return 'USDT-FUTURES'
    if quote == 'USDC':
        return 'USDC-FUTURES'
    return 'COIN-FUTURES'


def _bitget_public_get(path, params=None, ttl_sec=60, cache_bucket=None, cache_key=''):
    bucket = cache_bucket if cache_bucket is not None else OPENAI_CONTEXT_CACHE
    final_key = cache_key or '{}?{}'.format(path, json.dumps(params or {}, sort_keys=True, ensure_ascii=False))
    cached = _context_cache_get(bucket, final_key, ttl_sec)
    if cached is not None:
        return cached
    url = 'https://api.bitget.com{}'.format(path)
    payload, err = safe_request_json(
        requests,
        'GET',
        url,
        timeout=OPENAI_CONTEXT_HTTP_TIMEOUT_SEC,
        retries=2,
        params=params or {},
    )
    if err or not isinstance(payload, dict):
        return None
    data = payload.get('data')
    _context_cache_set(bucket, final_key, data)
    return data


def _coingecko_market_snapshot(symbol):
    base = _symbol_base_asset(symbol).lower()
    cache_key = 'cg:{}'.format(base)
    cached = _context_cache_get(OPENAI_MARKETCAP_CACHE, cache_key, OPENAI_MARKETCAP_CACHE_TTL_SEC)
    if cached is not None:
        return cached
    url = 'https://api.coingecko.com/api/v3/coins/markets'
    rows, err = safe_request_json(
        requests,
        'GET',
        url,
        timeout=OPENAI_CONTEXT_HTTP_TIMEOUT_SEC,
        retries=2,
        params={
            'vs_currency': 'usd',
            'symbols': base,
            'include_tokens': 'all',
            'per_page': 3,
            'page': 1,
            'sparkline': 'false',
            'price_change_percentage': '24h',
        },
    )
    result = {}
    if isinstance(rows, list):
        exact = None
        for row in rows:
            if str((row or {}).get('symbol') or '').lower() == base:
                exact = dict(row or {})
                break
        exact = exact or (dict(rows[0]) if rows else {})
        result = {
            'market_cap_usd': _safe_round_metric(exact.get('market_cap', 0), 2),
            'fdv_usd': _safe_round_metric(exact.get('fully_diluted_valuation', 0), 2),
            'circulating_supply': _safe_round_metric(exact.get('circulating_supply', 0), 4),
            'total_supply': _safe_round_metric(exact.get('total_supply', 0), 4),
            'coingecko_id': str(exact.get('id') or '')[:80],
            'symbol_match': str(exact.get('symbol') or '')[:24],
        }
    if not result:
        result = {
            'market_cap_usd': 0.0,
            'fdv_usd': 0.0,
            'circulating_supply': 0.0,
            'total_supply': 0.0,
            'unavailable_reason': str(err or 'no_match')[:120],
        }
    _context_cache_set(OPENAI_MARKETCAP_CACHE, cache_key, result)
    return result


def _serialize_ohlcv_rows(df, limit=100):
    if df is None or df.empty:
        return []
    closed_df = df.iloc[:-1].copy() if len(df) >= 3 else df.copy()
    rows = []
    for row in closed_df.tail(max(int(limit or 0), 1)).itertuples(index=False):
        try:
            rows.append([
                int(getattr(row, 'ts')),
                round(float(getattr(row, 'o')), 8),
                round(float(getattr(row, 'h')), 8),
                round(float(getattr(row, 'l')), 8),
                round(float(getattr(row, 'c')), 8),
                round(float(getattr(row, 'v')), 4),
            ])
        except Exception:
            continue
    return rows


def _extract_trade_side(trade):
    side = str((trade or {}).get('side') or (trade or {}).get('takerSide') or '').lower().strip()
    info = dict((trade or {}).get('info') or {})
    if not side:
        side = str(info.get('side') or info.get('takerSide') or '').lower().strip()
    if side in ('buy', 'bid', 'b'):
        return 'buy'
    if side in ('sell', 'ask', 's'):
        return 'sell'
    return ''


def _fetch_symbol_liquidity_context(symbol, raw_frames=None):
    cache_key = 'liq:{}'.format(symbol)
    cached = _context_cache_get(OPENAI_CONTEXT_CACHE, cache_key, OPENAI_CONTEXT_CACHE_TTL_SEC)
    if cached is not None:
        return cached
    raw_frames = dict(raw_frames or {})
    result = {
        'spread_pct': 0.0,
        'bid_depth_5': 0.0,
        'ask_depth_5': 0.0,
        'bid_depth_10': 0.0,
        'ask_depth_10': 0.0,
        'depth_imbalance_10': 0.0,
        'largest_bid_wall_price': 0.0,
        'largest_bid_wall_size': 0.0,
        'largest_ask_wall_price': 0.0,
        'largest_ask_wall_size': 0.0,
        'recent_trades_count': 0,
        'aggressive_buy_volume': 0.0,
        'aggressive_sell_volume': 0.0,
        'aggressive_buy_notional': 0.0,
        'aggressive_sell_notional': 0.0,
        'buy_sell_notional_ratio': 0.0,
        'cvd_notional': 0.0,
        'cvd_bias': 'neutral',
        'volume_anomaly_5m': 0.0,
        'volume_anomaly_15m': 0.0,
        'errors': [],
    }
    try:
        ob = exchange.fetch_order_book(symbol, limit=20)
        bids = list((ob or {}).get('bids') or [])
        asks = list((ob or {}).get('asks') or [])
        best_bid = _safe_num(bids[0][0], 0.0) if bids else 0.0
        best_ask = _safe_num(asks[0][0], 0.0) if asks else 0.0
        mid = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else 0.0
        result['spread_pct'] = _safe_round_metric(((best_ask - best_bid) / max(mid, 1e-9) * 100.0) if mid > 0 else 0.0, 4)
        result['bid_depth_5'] = _safe_round_metric(sum(_safe_num(x[1], 0.0) for x in bids[:5]), 4)
        result['ask_depth_5'] = _safe_round_metric(sum(_safe_num(x[1], 0.0) for x in asks[:5]), 4)
        result['bid_depth_10'] = _safe_round_metric(sum(_safe_num(x[1], 0.0) for x in bids[:10]), 4)
        result['ask_depth_10'] = _safe_round_metric(sum(_safe_num(x[1], 0.0) for x in asks[:10]), 4)
        denom = max(result['bid_depth_10'] + result['ask_depth_10'], 1e-9)
        result['depth_imbalance_10'] = _safe_round_metric((result['bid_depth_10'] - result['ask_depth_10']) / denom, 4)
        if bids:
            largest_bid = max(bids[:10], key=lambda x: _safe_num(x[1], 0.0))
            result['largest_bid_wall_price'] = _safe_round_metric(largest_bid[0], 8)
            result['largest_bid_wall_size'] = _safe_round_metric(largest_bid[1], 4)
        if asks:
            largest_ask = max(asks[:10], key=lambda x: _safe_num(x[1], 0.0))
            result['largest_ask_wall_price'] = _safe_round_metric(largest_ask[0], 8)
            result['largest_ask_wall_size'] = _safe_round_metric(largest_ask[1], 4)
    except Exception as e:
        result['errors'].append('order_book:{}'.format(str(e)[:120]))
    try:
        trades = list(exchange.fetch_trades(symbol, limit=120) or [])
        buy_volume = 0.0
        sell_volume = 0.0
        buy_notional = 0.0
        sell_notional = 0.0
        for trade in trades:
            side = _extract_trade_side(trade)
            qty = _safe_num((trade or {}).get('amount', 0), 0.0)
            price = _safe_num((trade or {}).get('price', 0), 0.0)
            notional = qty * price
            if side == 'buy':
                buy_volume += qty
                buy_notional += notional
            elif side == 'sell':
                sell_volume += qty
                sell_notional += notional
        result['recent_trades_count'] = len(trades)
        result['aggressive_buy_volume'] = _safe_round_metric(buy_volume, 4)
        result['aggressive_sell_volume'] = _safe_round_metric(sell_volume, 4)
        result['aggressive_buy_notional'] = _safe_round_metric(buy_notional, 4)
        result['aggressive_sell_notional'] = _safe_round_metric(sell_notional, 4)
        result['buy_sell_notional_ratio'] = _safe_round_metric(buy_notional / max(sell_notional, 1e-9), 4)
        result['cvd_notional'] = _safe_round_metric(buy_notional - sell_notional, 4)
        result['cvd_bias'] = 'buy_dominant' if buy_notional > sell_notional * 1.08 else 'sell_dominant' if sell_notional > buy_notional * 1.08 else 'neutral'
    except Exception as e:
        result['errors'].append('trades:{}'.format(str(e)[:120]))
    try:
        d1m = raw_frames.get('1m')
        d5m = raw_frames.get('5m')
        d15m = raw_frames.get('15m')
        if d5m is not None and len(d5m) >= 30:
            recent_5m = float(d5m['v'].astype(float).tail(3).mean())
            base_5m = float(d5m['v'].astype(float).tail(24).head(18).mean())
            result['volume_anomaly_5m'] = _safe_round_metric(recent_5m / max(base_5m, 1e-9), 4)
        elif d1m is not None and len(d1m) >= 30:
            recent_1m = float(d1m['v'].astype(float).tail(5).mean())
            base_1m = float(d1m['v'].astype(float).tail(30).head(20).mean())
            result['volume_anomaly_5m'] = _safe_round_metric(recent_1m / max(base_1m, 1e-9), 4)
        if d15m is not None and len(d15m) >= 30:
            recent_15m = float(d15m['v'].astype(float).tail(2).mean())
            base_15m = float(d15m['v'].astype(float).tail(24).head(18).mean())
            result['volume_anomaly_15m'] = _safe_round_metric(recent_15m / max(base_15m, 1e-9), 4)
    except Exception as e:
        result['errors'].append('volume:{}'.format(str(e)[:120]))
    _context_cache_set(OPENAI_CONTEXT_CACHE, cache_key, result)
    return result


def _fetch_symbol_derivatives_context(symbol, ticker_context=None, liquidity_context=None):
    cache_key = 'drv:{}'.format(symbol)
    cached = _context_cache_get(OPENAI_CONTEXT_CACHE, cache_key, OPENAI_CONTEXT_CACHE_TTL_SEC)
    if cached is not None:
        return cached
    ticker_context = dict(ticker_context or {})
    liquidity_context = dict(liquidity_context or {})
    rest_symbol = _bitget_rest_symbol(symbol)
    product_type = _bitget_product_type(symbol)
    result = {
        'funding_rate': 0.0,
        'next_funding_time': '',
        'open_interest': 0.0,
        'open_interest_value_usdt': 0.0,
        'open_interest_change_pct_5m': 0.0,
        'long_short_ratio': 0.0,
        'top_trader_long_short_ratio': 0.0,
        'whale_position_change_pct': 0.0,
        'basis_pct': 0.0,
        'mark_price': _safe_round_metric(((ticker_context or {}).get('mark_price', 0)), 8),
        'index_price': _safe_round_metric(((ticker_context or {}).get('index_price', 0)), 8),
        'liquidation_volume_24h': 0.0,
        'liquidation_map_status': 'unavailable',
        'leverage_heat': 'unknown',
        'leverage_heat_score': 0.0,
        'errors': [],
    }
    ticker_info = dict((ticker_context or {}).get('raw_info') or {})
    mark_price = _safe_num(ticker_info.get('markPrice', ticker_context.get('mark_price', 0)), 0.0)
    index_price = _safe_num(ticker_info.get('indexPrice', ticker_context.get('index_price', 0)), 0.0)
    if mark_price > 0 and index_price > 0:
        result['basis_pct'] = _safe_round_metric((mark_price - index_price) / max(index_price, 1e-9) * 100.0, 4)
        result['mark_price'] = _safe_round_metric(mark_price, 8)
        result['index_price'] = _safe_round_metric(index_price, 8)
    try:
        fetch_funding_rate = getattr(exchange, 'fetch_funding_rate', None)
        if callable(fetch_funding_rate):
            funding = dict(fetch_funding_rate(symbol) or {})
            result['funding_rate'] = _safe_round_metric(funding.get('fundingRate', funding.get('funding_rate', 0)), 8)
            result['next_funding_time'] = str(funding.get('fundingDatetime') or funding.get('nextFundingTime') or '')[:40]
    except Exception as e:
        result['errors'].append('fetch_funding_rate:{}'.format(str(e)[:120]))
    if abs(result['funding_rate']) <= 0:
        try:
            funding_rows = _bitget_public_get(
                '/api/v2/mix/market/history-fund-rate',
                params={'symbol': rest_symbol, 'productType': product_type, 'pageSize': '1', 'pageNo': '1'},
                ttl_sec=60,
                cache_key='funding:{}:{}'.format(rest_symbol, product_type),
            ) or []
            first_row = dict(funding_rows[0] or {}) if isinstance(funding_rows, list) and funding_rows else {}
            result['funding_rate'] = _safe_round_metric(first_row.get('fundingRate', 0), 8)
            result['next_funding_time'] = str(first_row.get('fundingTime') or '')[:40]
        except Exception as e:
            result['errors'].append('history_funding:{}'.format(str(e)[:120]))
    try:
        fetch_open_interest = getattr(exchange, 'fetch_open_interest', None)
        if callable(fetch_open_interest):
            oi = dict(fetch_open_interest(symbol) or {})
            result['open_interest'] = _safe_round_metric(oi.get('openInterestAmount', oi.get('openInterest', oi.get('amount', 0))), 4)
            result['open_interest_value_usdt'] = _safe_round_metric(oi.get('openInterestValue', oi.get('value', 0)), 4)
    except Exception as e:
        result['errors'].append('fetch_open_interest:{}'.format(str(e)[:120]))
    if result['open_interest'] <= 0 and result['open_interest_value_usdt'] <= 0:
        try:
            oi = _bitget_public_get(
                '/api/v2/mix/market/open-interest',
                params={'symbol': rest_symbol, 'productType': product_type},
                ttl_sec=45,
                cache_key='oi:{}:{}'.format(rest_symbol, product_type),
            ) or {}
            if isinstance(oi, list):
                oi = dict(oi[0] or {}) if oi else {}
            result['open_interest'] = _safe_round_metric(oi.get('openInterest', oi.get('size', 0)), 4)
            result['open_interest_value_usdt'] = _safe_round_metric(oi.get('openInterestValue', oi.get('amount', 0)), 4)
        except Exception as e:
            result['errors'].append('open_interest_rest:{}'.format(str(e)[:120]))
    try:
        ratio_rows = _bitget_public_get(
            '/api/v2/mix/market/account-long-short',
            params={'symbol': rest_symbol, 'productType': product_type, 'period': '5m', 'limit': '2'},
            ttl_sec=60,
            cache_key='ls:{}:{}'.format(rest_symbol, product_type),
        ) or []
        if isinstance(ratio_rows, dict):
            ratio_rows = [ratio_rows]
        if ratio_rows:
            latest = dict(ratio_rows[0] or {})
            prev = dict(ratio_rows[1] or {}) if len(ratio_rows) > 1 else {}
            long_ratio = _safe_num(latest.get('longAccountRatio', latest.get('longRatio', 0)), 0.0)
            short_ratio = _safe_num(latest.get('shortAccountRatio', latest.get('shortRatio', 0)), 0.0)
            if short_ratio > 0:
                result['long_short_ratio'] = _safe_round_metric(long_ratio / max(short_ratio, 1e-9), 4)
            prev_long = _safe_num(prev.get('longAccountRatio', prev.get('longRatio', 0)), 0.0)
            prev_short = _safe_num(prev.get('shortAccountRatio', prev.get('shortRatio', 0)), 0.0)
            prev_ratio = prev_long / max(prev_short, 1e-9) if prev_short > 0 else 0.0
            if prev_ratio > 0 and result['long_short_ratio'] > 0:
                result['whale_position_change_pct'] = _safe_round_metric((result['long_short_ratio'] - prev_ratio) / max(prev_ratio, 1e-9) * 100.0, 4)
    except Exception as e:
        result['errors'].append('long_short_ratio:{}'.format(str(e)[:120]))
    if result['open_interest_value_usdt'] > 0:
        heat_score = abs(result['funding_rate']) * 10000.0
        heat_score += max(abs(result['whale_position_change_pct']), 0.0) * 0.2
        heat_score += min(result['open_interest_value_usdt'] / 5000000.0, 12.0)
        heat_score += min(abs(liquidity_context.get('cvd_notional', 0.0)) / 500000.0, 8.0)
        result['leverage_heat_score'] = _safe_round_metric(heat_score, 2)
        result['leverage_heat'] = 'hot' if heat_score >= 12 else 'warm' if heat_score >= 6 else 'cool'
    _context_cache_set(OPENAI_CONTEXT_CACHE, cache_key, result)
    return result


def _clean_html_text(text):
    text = re.sub(r'<[^>]+>', ' ', str(text or ''))
    text = text.replace('&amp;', '&').replace('&quot;', '"').replace('&#39;', "'")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _fetch_bitget_news_context(symbol):
    base = _symbol_base_asset(symbol)
    cache_key = 'news:{}'.format(base)
    cached = _context_cache_get(OPENAI_NEWS_CACHE, cache_key, OPENAI_NEWS_CACHE_TTL_SEC)
    if cached is not None:
        return cached
    sources = [
        'https://www.bitget.com/news',
        'https://www.bitget.com/support',
    ]
    items = []
    pattern = re.compile(r'<a[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>', re.I | re.S)
    keyword_candidates = {base.upper(), base.lower(), '{}usdt'.format(base.lower())}
    for url in sources:
        html, err = safe_request_text(requests, 'GET', url, timeout=OPENAI_CONTEXT_HTTP_TIMEOUT_SEC, retries=1)
        if err or not html:
            continue
        for match in pattern.finditer(html):
            href = str(match.group('href') or '').strip()
            title = _clean_html_text(match.group('title') or '')
            if not href or len(title) < 6:
                continue
            title_l = title.lower()
            if not any(k in title_l for k in keyword_candidates) and base not in title.upper():
                continue
            if href.startswith('/'):
                href = 'https://www.bitget.com{}'.format(href)
            items.append({
                'title': title[:180],
                'url': href[:260],
                'source': 'bitget_news',
            })
    deduped = []
    seen = set()
    for item in items:
        key = '{}|{}'.format(item.get('title'), item.get('url'))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= 3:
            break
    result = {
        'symbol': base,
        'items': deduped,
        'available': bool(deduped),
        'note': 'Bitget official news/support best-effort scrape',
    }
    if not deduped:
        result['note'] = 'No matching Bitget news/support items found from the public pages during this snapshot.'
    _context_cache_set(OPENAI_NEWS_CACHE, cache_key, result)
    return result


def _build_market_microstructure_context(symbol, ticker_context, raw_frames):
    liquidity = _fetch_symbol_liquidity_context(symbol, raw_frames=raw_frames)
    derivatives = _fetch_symbol_derivatives_context(symbol, ticker_context=ticker_context, liquidity_context=liquidity)
    supply = _coingecko_market_snapshot(symbol)
    last = _safe_num(ticker_context.get('last', 0), 0.0)
    return {
        'basic_market_data': {
            'symbol': symbol,
            'exchange': 'Bitget',
            'market_type': 'perpetual_swap',
            'current_price': _safe_round_metric(last, 8),
            'change_24h_pct': _safe_round_metric(ticker_context.get('percentage_24h', 0), 3),
            'quote_volume_24h': _safe_round_metric(ticker_context.get('quote_volume', 0), 4),
            'base_volume_24h': _safe_round_metric(((ticker_context.get('quote_volume', 0) or 0) / max(last, 1e-9)) if last > 0 else 0.0, 4),
            'market_cap_usd': _safe_round_metric(supply.get('market_cap_usd', 0), 2),
            'fdv_usd': _safe_round_metric(supply.get('fdv_usd', 0), 2),
            'circulating_supply': _safe_round_metric(supply.get('circulating_supply', 0), 4),
            'total_supply': _safe_round_metric(supply.get('total_supply', 0), 4),
            'funding_rate': _safe_round_metric(derivatives.get('funding_rate', 0), 8),
            'open_interest': _safe_round_metric(derivatives.get('open_interest', 0), 4),
            'open_interest_value_usdt': _safe_round_metric(derivatives.get('open_interest_value_usdt', 0), 4),
            'long_short_ratio': _safe_round_metric(derivatives.get('long_short_ratio', 0), 4),
            'top_trader_long_short_ratio': _safe_round_metric(derivatives.get('top_trader_long_short_ratio', 0), 4),
            'whale_position_change_pct': _safe_round_metric(derivatives.get('whale_position_change_pct', 0), 4),
        },
        'liquidity_context': liquidity,
        'derivatives_context': derivatives,
        'news_context': _fetch_bitget_news_context(symbol),
    }


def _persist_market_snapshot(db, symbol, regime_info, timeframe_data):
    snap_root = db.setdefault('market_snapshots', {})
    snap_root[symbol] = {
        'symbol': symbol,
        'regime': dict(regime_info or {}),
        'timeframes': dict(timeframe_data or {}),
        'updated_at': tw_now_str('%Y-%m-%d %H:%M:%S'),
    }
    meta = db.setdefault('market_history_meta', {})
    meta['symbols'] = len(snap_root)
    meta['timeframes'] = AI_MARKET_TIMEFRAMES
    meta['last_update'] = tw_now_str('%Y-%m-%d %H:%M:%S')


def _refresh_ai_panel_market_meta():
    with AI_LOCK:
        meta = dict((AI_DB.get('market_history_meta', {}) or {}))
        AI_PANEL['market_db_info'] = {
            'symbols': int(meta.get('symbols', 0) or 0),
            'timeframes': list(meta.get('timeframes', AI_MARKET_TIMEFRAMES) or AI_MARKET_TIMEFRAMES),
            'last_update': meta.get('last_update', '--') or '--',
        }
        AUTO_BACKTEST_STATE['db_symbols'] = AI_PANEL['market_db_info']['symbols']
        AUTO_BACKTEST_STATE['db_last_update'] = AI_PANEL['market_db_info']['last_update']
        AUTO_BACKTEST_STATE['data_timeframes'] = AI_PANEL['market_db_info']['timeframes']


_refresh_ai_panel_market_meta()

def get_margin_learning_multiplier(symbol, score, breakdown):
    try:
        with LEARN_LOCK:
            ss = LEARN_DB.get('symbol_stats', {}).get(symbol, {})
        count = int(ss.get('count', 0) or 0)
        if count < 5:
            return 1.0
        wr = float(ss.get('win', 0)) / max(count, 1)
        mult = 1.0
        if wr >= 0.62:
            mult += 0.08
        elif wr < 0.4:
            mult -= 0.10
        if abs(float(score or 0)) >= 70:
            mult += 0.04
        if isinstance(breakdown, dict):
            rr = float(breakdown.get('RR', 0) or 0)
            if rr >= 2.0:
                mult += 0.04
        return round(clamp(mult, 0.82, 1.15), 4)
    except Exception:
        return 1.0

def classify_market_regime(df15, df1h=None):
    try:
        c = df15['c'].astype(float)
        h = df15['h'].astype(float)
        l = df15['l'].astype(float)
        v = df15['v'].astype(float)
        curr = float(c.iloc[-1])
        adx = safe_last(ta.adx(h, l, c, length=14).iloc[:, 0], 18)
        atr = max(safe_last(ta.atr(h, l, c, length=14), curr * 0.004), curr * 0.003)
        atr_ratio = atr / max(curr, 1e-9)
        bb = ta.bbands(c, length=20, std=2)
        bb_up = safe_last(bb.iloc[:, 0], curr) if bb is not None and not bb.empty else curr
        bb_low = safe_last(bb.iloc[:, 2], curr) if bb is not None and not bb.empty else curr
        bb_width = (bb_up - bb_low) / max(curr, 1e-9)
        ret3 = abs(float(c.iloc[-1] - c.iloc[-4]) / max(c.iloc[-4], 1e-9) * 100) if len(c) >= 4 else 0
        vol_now = float(v.tail(3).mean()) if len(v) >= 3 else float(v.iloc[-1])
        vol_base = float(v.tail(30).head(20).mean()) if len(v) >= 30 else max(vol_now, 1.0)
        vol_ratio = vol_now / max(vol_base, 1e-9)
        slope = _linreg_slope(c.tail(14).tolist()) / max(curr, 1e-9) * 100
        dir_hint = '涓€?
        if slope > 0.06:
            dir_hint = '澶?
        elif slope < -0.06:
            dir_hint = '绌?
        if ret3 >= 2.2 and vol_ratio >= 1.8 and atr_ratio >= 0.01:
            regime = 'news'; confidence = 0.9; note = '鐭檪闁撶垎閲忔€ユ媺鎬ユ'
        elif adx >= 23 and abs(slope) >= 0.08 and bb_width >= 0.018:
            regime = 'trend'; confidence = min(0.95, 0.55 + adx / 50); note = 'ADX鑸囨枩鐜囧悓姝ワ紝灞柤瓒ㄥ嫝鐩?
        elif adx <= 18 and bb_width <= 0.02:
            regime = 'range'; confidence = 0.72; note = '浣嶢DX浣庢尝鍕曪紝鍋忓崁闁撶洡'
        else:
            regime = 'neutral'; confidence = 0.55; note = '娣峰悎绲愭锛岃蛋鍕㈡湭瀹屽叏瀹氬瀷'
        return {'regime': regime,'direction': dir_hint,'confidence': round(confidence, 3),'adx': round(adx, 2),'atr_ratio': round(atr_ratio, 5),'bb_width': round(bb_width, 5),'vol_ratio': round(vol_ratio, 2),'move_3bars_pct': round(ret3, 2),'note': note}
    except Exception as e:
        return {'regime': 'neutral', 'direction': '涓€?, 'confidence': 0.4, 'note': f'鍒ゅ畾澶辨晽:{e}'}

def get_regime_params(regime):
    with AI_LOCK:
        return dict(AI_DB.get('param_sets', {}).get(regime, AI_DB.get('param_sets', {}).get('neutral', {})))

# 鍩哄簳鍒ュ悕锛氫繚鐣?v1 鍋氱偤搴曞堡鐗瑰镜鐢㈢敓鍣紱鐪熸灏嶅 analyze / backtest 鏈冨湪寰屾缍佸埌澧炲挤鐗?_BASE_ANALYZE = analyze_legacy_shadow_1
_BASE_LEARN_FROM_CLOSED_TRADE = learn_from_closed_trade_legacy_shadow_1
_BASE_RUN_SIMPLE_BACKTEST = run_simple_backtest_legacy_shadow_1
_BASE_API_STATE = api_state_legacy_shadow_1

def _fetch_regime_for_symbol(symbol):
    try:
        d15 = _safe_fetch_ohlcv_df(symbol, '15m', max(ANALYZE_15M_LIMIT, 180))
        d1h = _safe_fetch_ohlcv_df(symbol, '1h', max(ANALYZE_4H_LIMIT, 180))
        info = classify_market_regime(d15, d1h)
        tempo = detect_market_tempo(d15)
        info.update(tempo)
        with AI_LOCK:
            prev = dict((AI_PANEL.get('symbol_regimes', {}) or {}).get(symbol, {}) or {})
        info = apply_decision_inertia(symbol, info, prev)
        return info
    except Exception as e:
        return {'regime': 'neutral', 'direction': '涓€?, 'confidence': 0.4, 'tempo': 'normal', 'note': f'鍒ゅ畾澶辨晽:{e}'}

def _safe_num(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def _ai_feature_scale(name, value):
    name = str(name or '').lower()
    v = _safe_num(value)
    av = abs(v)
    if av <= 1e-9:
        return 0.0
    if 'rr' in name:
        return math.tanh(v / 2.2)
    if 'conf' in name or 'quality' in name or 'gate' in name:
        return math.tanh(v / 4.0)
    if 'score' in name or 'bias' in name or 'pnl' in name or 'drawdown' in name:
        return math.tanh(v / 12.0)
    if 'atr' in name or 'width' in name or 'ratio' in name or 'vol' in name or 'move' in name:
        return math.tanh(v / 1.8)
    if av > 100:
        return math.tanh(v / 100.0)
    if av > 10:
        return math.tanh(v / 10.0)
    return math.tanh(v / 3.5)


def _sanitize_ai_feature_name(name):
    out = []
    for ch in str(name or '').strip().lower():
        if ch.isalnum() or ch in ('_', '-', '|', ':'):
            out.append(ch)
        elif ch in (' ', '/', '.'):
            out.append('_')
    return ''.join(out)[:80] or 'unknown'


def _infer_signal_side(score, entry=0.0, sl=0.0, tp=0.0):
    entry = _safe_num(entry)
    sl = _safe_num(sl)
    tp = _safe_num(tp)
    if entry and tp and sl:
        if tp > entry and sl < entry:
            return 1
        if tp < entry and sl > entry:
            return -1
    return 1 if _safe_num(score) >= 0 else -1



def _derive_signal_fingerprint(symbol, side, breakdown=None, regime_info=None, desc='', extra=None):
    bd = dict(breakdown or {})
    regime_info = dict(regime_info or {})
    fp = {}
    rr = _safe_num(bd.get('RR', 0.0))
    entry_gate = _safe_num(bd.get('EntryGate', bd.get('閫插牬鍝佽唱', 0.0)))
    vwap_bias = _safe_num(bd.get('VWAP', 0.0))
    regime_bias = _safe_num(bd.get('RegimeBias', bd.get('鏂瑰悜鍝佽唱', 0.0)))
    anti_chase = _safe_num(bd.get('杩藉児棰ㄩ毆', 0.0))
    tempo_score = _safe_num(regime_info.get('tempo_score', 0.0))
    vol_ratio = _safe_num(regime_info.get('vol_ratio', 0.0))
    conf = _safe_num(regime_info.get('confidence', 0.0))
    fp['rr_bucket'] = 'rr_hi' if rr >= 2.0 else 'rr_mid' if rr >= 1.35 else 'rr_low'
    fp['entry_bucket'] = 'entry_hi' if entry_gate >= 7 else 'entry_mid' if entry_gate >= 4 else 'entry_low'
    fp['tempo_bucket'] = 'tempo_fast' if tempo_score >= 0.55 else 'tempo_slow' if tempo_score <= -0.25 else 'tempo_normal'
    fp['chase_bucket'] = 'chase_risk' if anti_chase < 0 else 'chase_ok'
    fp['vol_bucket'] = 'vol_expand' if vol_ratio >= 1.35 else 'vol_dry' if vol_ratio <= 0.78 else 'vol_normal'
    fp['regime_align'] = 'align_yes' if regime_bias * side > 0 else 'align_no' if regime_bias * side < 0 else 'align_flat'
    fp['vwap_bucket'] = 'vwap_above' if vwap_bias * side > 0 else 'vwap_below' if vwap_bias * side < 0 else 'vwap_flat'
    fp['confidence_bucket'] = 'conf_hi' if conf >= 0.78 else 'conf_mid' if conf >= 0.55 else 'conf_low'
    session_bucket = str((regime_info.get('session_bucket') or bd.get('SessionBucket') or session_bucket_from_hour(get_tw_time().hour) or 'unknown')).strip() or 'unknown'
    fp['session_bucket'] = session_bucket
    if isinstance(desc, str) and desc:
        desc_l = desc.lower()
        if '绐佺牬' in desc or 'breakout' in desc_l:
            fp['trigger_family'] = 'breakout'
        elif '鍥炶俯' in desc or 'pullback' in desc_l:
            fp['trigger_family'] = 'pullback'
        elif '鎺? in desc or 'sweep' in desc_l:
            fp['trigger_family'] = 'liquidity_sweep'
        elif '鍗€闁? in desc or 'range' in desc_l:
            fp['trigger_family'] = 'range_revert'
    if not fp.get('trigger_family'):
        setup = str(bd.get('Setup') or '').lower()
        if 'break' in setup:
            fp['trigger_family'] = 'breakout'
        elif 'pull' in setup:
            fp['trigger_family'] = 'pullback'
        else:
            fp['trigger_family'] = 'generic'
    fp['symbol_family'] = str(symbol or 'NA').split('/')[0][:12] or 'NA'
    return fp


def _adaptive_indicator_hint_score(breakdown=None):
    bd = dict(breakdown or {})
    with AI_LOCK:
        hints = dict((AI_DB.get('adaptive_indicator_hints') or {}))
    if not hints:
        return 0.0, []
    raw = 0.0
    covered = []
    for key, value in bd.items():
        meta = hints.get(str(key))
        if not meta:
            continue
        val = _safe_num(value, None)
        if val is None:
            continue
        direction = 1.0 if val > 0 else -1.0 if val < 0 else 0.0
        strength = min(abs(float(val)), 10.0) / 10.0
        edge = float(meta.get('edge', 0.0) or 0.0)
        conf = float(meta.get('confidence', 0.0) or 0.0)
        contrib = direction * edge * (0.35 + strength * 0.65) * conf * 8.0
        raw += contrib
        covered.append((str(key), round(contrib, 4), int(meta.get('count', 0) or 0)))
    return raw, covered

def _extract_ai_signal_features(symbol, side, breakdown=None, regime_info=None, desc='', extra=None):
    bd = dict(breakdown or {})
    regime_info = dict(regime_info or {})
    feats = {}

    def add(name, value):
        key = _sanitize_ai_feature_name(name)
        val = _safe_num(value, None)
        if val is None:
            return
        if math.isnan(val) or math.isinf(val):
            return
        val = max(min(val, 4.0), -4.0)
        if abs(val) < 1e-9:
            return
        feats[key] = round(val, 6)

    add('side_bias', 1.0 if side > 0 else -1.0)
    add('regime_confidence', _ai_feature_scale('regime_conf', regime_info.get('confidence', 0)))
    add('tempo_score', _ai_feature_scale('tempo_score', regime_info.get('tempo_score', 0)))
    add('move_3bars_pct', _ai_feature_scale('move_3bars_pct', regime_info.get('move_3bars_pct', 0)))
    add('vol_ratio', _ai_feature_scale('vol_ratio', regime_info.get('vol_ratio', 0)))
    add('bb_width', _ai_feature_scale('bb_width', regime_info.get('bb_width', 0)))

    direction = str(regime_info.get('direction', '涓€?) or '涓€?)
    if direction == '澶?:
        add('market_direction_alignment', 1.0 if side > 0 else -1.0)
    elif direction == '绌?:
        add('market_direction_alignment', 1.0 if side < 0 else -1.0)

    regime = str(bd.get('Regime') or regime_info.get('regime') or 'neutral')
    add(f'regime::{regime}', 1.0)

    setup = str(bd.get('Setup') or '').strip()
    if setup:
        add(f'setup::{setup}', 1.0)

    add(f'symbol::{symbol}', 1.0)

    directional_keys = {'regimebias', '鏂瑰悜鍝佽唱', '4h瓒ㄥ嫝涓嶉爢', '杩藉児棰ㄩ毆', 'signalquality', 'learnedge', 'regimescoreadj'}
    skip = {'setup', 'regime', 'regimedir'}
    for k, v in bd.items():
        ks = _sanitize_ai_feature_name(k)
        if ks in skip:
            continue
        if isinstance(v, bool):
            add(f'flag::{ks}', 1.0 if v else -1.0)
            continue
        if isinstance(v, (int, float)):
            scaled = _ai_feature_scale(ks, v)
            if ks in directional_keys:
                scaled *= side
            add(f'bd::{ks}', scaled)
        elif isinstance(v, str) and v:
            add(f'cat::{ks}::{v}', 1.0)

    tags = []
    if isinstance(desc, str) and desc:
        tags.extend([p.strip() for p in desc.split('|') if p.strip()][:20])
    if isinstance(extra, (list, tuple)):
        tags.extend([str(x).strip() for x in extra if str(x).strip()][:20])
    for tag in tags[:30]:
        add(f'tag::{tag}', 1.0)

    fingerprint = _derive_signal_fingerprint(symbol, side, breakdown=bd, regime_info=regime_info, desc=desc, extra=extra)
    for fk, fv in fingerprint.items():
        add(f'fp::{fk}::{fv}', 1.0)

    with AI_LOCK:
        adaptive_hints = dict((AI_DB.get('adaptive_indicator_hints') or {}))
    for hk, meta in adaptive_hints.items():
        if hk not in bd:
            continue
        val = _safe_num(bd.get(hk), 0.0)
        if abs(val) <= 1e-9:
            continue
        hint_edge = float(meta.get('edge', 0.0) or 0.0)
        hint_conf = float(meta.get('confidence', 0.0) or 0.0)
        scaled = _ai_feature_scale(hk, val) * max(min(hint_edge * max(hint_conf, 0.15), 2.0), -2.0)
        add(f'hint::{hk}', scaled)

    return feats


def _trade_outcome_edge(trade):
    metric = float(_trade_learn_metric(trade) or 0.0)
    result = str(trade.get('result') or '')
    edge = math.tanh(metric / 1.4)
    if result == 'win':
        edge = max(edge, 0.35)
    elif result == 'loss':
        edge = min(edge, -0.35)
    return float(max(min(edge, 1.0), -1.0))


def _build_ai_feature_model_from_trades(trades):
    model = {'features': {}, 'meta': {'samples': 0, 'wins': 0, 'avg_pnl': 0.0, 'updated_at': tw_now_str('%Y-%m-%d %H:%M:%S')}}
    cleaned = [t for t in list(trades or []) if _is_live_source((t or {}).get('source')) and str((t or {}).get('result') or '') in ('win', 'loss')]
    if not cleaned:
        return model
    total_pnl = 0.0
    wins = 0
    for t in cleaned[-360:]:
        bd = dict(t.get('breakdown') or {})
        side = 1 if str(t.get('side') or '').lower() == 'long' else -1
        regime_info = {
            'regime': bd.get('Regime', 'neutral'),
            'direction': bd.get('RegimeDir', '涓€?),
            'confidence': bd.get('RegimeConf', 0.5),
            'tempo_score': bd.get('TempoScore', 0.0),
        }
        feats = _extract_ai_signal_features(
            str(t.get('symbol') or 'NA'),
            side,
            breakdown=bd,
            regime_info=regime_info,
            desc=t.get('desc') or '',
            extra=[t.get('setup_label') or '']
        )
        edge = _trade_outcome_edge(t)
        pnl_metric = float(_trade_learn_metric(t) or 0.0)
        total_pnl += pnl_metric
        if str(t.get('result') or '') == 'win':
            wins += 1
        for feat, val in feats.items():
            rec = model['features'].setdefault(feat, {'count': 0, 'edge_sum': 0.0, 'edge_abs': 0.0, 'wins': 0, 'pnl_sum': 0.0, 'value_abs_sum': 0.0})
            rec['count'] += 1
            rec['edge_sum'] += edge * float(val)
            rec['edge_abs'] += abs(edge * float(val))
            rec['pnl_sum'] += pnl_metric * float(val)
            rec['value_abs_sum'] += abs(float(val))
            if str(t.get('result') or '') == 'win':
                rec['wins'] += 1
    features_out = {}
    for feat, rec in model['features'].items():
        count = int(rec.get('count', 0) or 0)
        if count < 3:
            continue
        avg_edge = float(rec.get('edge_sum', 0.0) or 0.0) / max(count, 1)
        avg_pnl = float(rec.get('pnl_sum', 0.0) or 0.0) / max(count, 1)
        win_rate = float(rec.get('wins', 0) or 0) / max(count, 1)
        confidence = min(count / 18.0, 1.0)
        weight = avg_edge * 52.0 + avg_pnl * 8.0 + (win_rate - 0.5) * 10.0
        if abs(weight) < 0.18:
            continue
        features_out[feat] = {
            'weight': round(weight, 6),
            'count': count,
            'confidence': round(confidence, 6),
            'win_rate': round(win_rate, 6),
            'avg_pnl': round(avg_pnl, 6),
        }
    model['features'] = features_out
    model['meta'] = {
        'samples': len(cleaned[-360:]),
        'wins': wins,
        'avg_pnl': round(total_pnl / max(len(cleaned[-360:]), 1), 6),
        'updated_at': tw_now_str('%Y-%m-%d %H:%M:%S'),
    }
    return model


def _score_signal_with_ai_model(symbol, side, breakdown=None, regime_info=None, desc='', fallback_score=0.0, extra=None):
    with AI_LOCK:
        model = dict(AI_DB.get('ai_feature_model') or {})
    features = _extract_ai_signal_features(symbol, side, breakdown=breakdown, regime_info=regime_info, desc=desc, extra=extra)
    feature_model = dict(model.get('features') or {})
    covered = []
    raw = 0.0
    for feat, val in features.items():
        meta = feature_model.get(feat)
        if not meta:
            continue
        conf = max(0.12, float(meta.get('confidence', 0.0) or 0.0))
        weight = float(meta.get('weight', 0.0) or 0.0)
        contrib = float(val) * weight * conf
        raw += contrib
        covered.append((feat, round(contrib, 4), int(meta.get('count', 0) or 0)))
    coverage = min(len(covered) / 14.0, 1.0)
    meta = dict(model.get('meta') or {})
    sample_cnt = int(meta.get('samples', 0) or 0)
    sample_conf = min(sample_cnt / 60.0, 1.0)
    strategy = _strategy_score_lookup(symbol, str((breakdown or {}).get('Regime') or (regime_info or {}).get('regime') or 'neutral'), str((breakdown or {}).get('Setup') or ''))
    profile = _ai_strategy_profile(symbol, regime=str((breakdown or {}).get('Regime') or (regime_info or {}).get('regime') or 'neutral'), setup=str((breakdown or {}).get('Setup') or ''))
    strategy_boost = float(strategy.get('ev_per_trade', 0.0) or 0.0) * 22.0 + (float(strategy.get('win_rate', 50.0) or 50.0) - 50.0) * 0.10
    profile_boost = float(profile.get('ev_per_trade', 0.0) or 0.0) * 16.0 + (float(profile.get('win_rate', 50.0) or 50.0) - 50.0) * 0.06
    hint_score, hint_covered = _adaptive_indicator_hint_score(breakdown=breakdown)
    discovered_logic_count = len(covered) + len(hint_covered)
    discovery_strength = min(discovered_logic_count / 18.0, 1.0)
    base_from_model = raw * 7.2 + strategy_boost + profile_boost + hint_score
    growth_control = _ai_growth_control(int(profile.get('effective_count', profile.get('sample_count', 0)) or 0))
    base_blend = sample_conf * 0.52 + coverage * 0.28 + discovery_strength * 0.20
    blend_cap = AI_DISCOVERY_BLEND_CEIL * float(growth_control.get('blend_cap', 1.0) or 0.0)
    adaptive_blend = 0.0 if blend_cap <= 0 else max(AI_DISCOVERY_BLEND_FLOOR, min(blend_cap, base_blend))
    fallback_weight = 0.0 if AI_FULL_SCORE_CONTROL else max(0.15, 1.0 - sample_conf * 0.7)
    mixed = base_from_model * adaptive_blend + float(fallback_score or 0.0) * max(1.0 - adaptive_blend, fallback_weight)
    score = max(min(round(mixed, 2), 100.0), -100.0)
    top = sorted(covered + [('hint::' + f, c, n) for f, c, n in hint_covered], key=lambda x: abs(x[1]), reverse=True)[:12]
    return {
        'score': score,
        'coverage': round(coverage, 4),
        'sample_confidence': round(sample_conf, 4),
        'sample_count': sample_cnt,
        'raw': round(raw, 6),
        'strategy_boost': round(strategy_boost + profile_boost + hint_score, 4),
        'adaptive_blend': round(adaptive_blend, 4),
        'discovered_logic_count': int(discovered_logic_count),
        'top_contributors': [
            {'feature': f, 'contribution': c, 'count': n} for f, c, n in top
        ],
    }


def _signal_quality_from_breakdown(breakdown, side):
    bd = dict(breakdown or {})
    quality = 0.0
    notes = []
    rr = _safe_num(bd.get('RR', 0))
    entry_gate = _safe_num(bd.get('EntryGate', bd.get('閫插牬鍝佽唱', 0)))
    regime_bias = _safe_num(bd.get('RegimeBias', bd.get('鏂瑰悜鍝佽唱', 0)))
    if rr >= 2.0:
        quality += 2.2
        notes.append('RR浣?)
    elif rr >= 1.5:
        quality += 1.2
    elif 0 < rr < 1.2:
        quality -= 2.5
        notes.append('RR寮?)
    if entry_gate >= 4:
        quality += 2.0
        notes.append('閫插牬浣?)
    elif entry_gate <= 0:
        quality -= 2.2
        notes.append('閫插牬寮?)
    if regime_bias * side > 0:
        quality += min(abs(regime_bias) * 0.35, 2.0)
        notes.append('鏂瑰悜鍚屽悜')
    elif regime_bias * side < 0:
        quality -= min(abs(regime_bias) * 0.45, 3.0)
        notes.append('鏂瑰悜閫嗛ⅷ')
    if '楂樻尝鍕曢亷鐔? in bd:
        quality -= 1.6
        notes.append('娉㈠嫊閬庣啽')
    if '4H瓒ㄥ嫝涓嶉爢' in bd:
        quality -= 2.2
        notes.append('閫?H')
    if '棰ㄥ牨姣斾笉瓒? in bd:
        quality -= 2.0
    return round(quality, 2), notes


def _recent_symbol_trade_profile(symbol, lookback=SYMBOL_REPEAT_LOOKBACK):
    try:
        all_recent = list(get_live_trades(closed_only=False) or [])
    except Exception:
        all_recent = []
    lookback = max(int(lookback), 1)
    recent = list(all_recent[-lookback:])
    matched = []
    for t in reversed(recent):
        if str(t.get('symbol') or '') != str(symbol or ''):
            continue
        matched.append(t)
    count = len(matched)
    total_recent = len(recent)
    share = float(count) / max(total_recent, 1)
    last_minutes = None
    if matched:
        last_t = matched[0]
        ts = last_t.get('exit_time') or last_t.get('entry_time') or last_t.get('time')
        try:
            dt = parse_time_any(ts)
            if dt is not None:
                last_minutes = max((tw_now() - dt).total_seconds() / 60.0, 0.0)
        except Exception:
            last_minutes = None
    return {'count': count, 'last_minutes': last_minutes, 'total_recent': total_recent, 'share': round(share, 4)}



def _symbol_rotation_adjustment(symbol):
    adj = 0.0
    notes = []
    profile = _recent_symbol_trade_profile(symbol)
    recent_count = int(profile.get('count', 0) or 0)
    total_recent = int(profile.get('total_recent', 0) or 0)
    share = float(profile.get('share', 0) or 0)

    try:
        with LEARN_LOCK:
            ss = dict((LEARN_DB.get('symbol_stats', {}) or {}).get(symbol, {}) or {})
    except Exception:
        ss = {}

    n = int(ss.get('count', 0) or 0)
    wr = float(ss.get('win', 0) or 0) / max(n, 1) if n > 0 else 0.0
    avg_all = float(ss.get('total_pnl', 0) or 0) / max(n, 1) if n > 0 else 0.0
    strong_symbol = (n >= 6 and wr >= 0.56 and avg_all > 0)
    elite_symbol = (n >= 10 and wr >= 0.62 and avg_all > 0.03)

    if total_recent >= 8:
        base_target = min(max(SYMBOL_BALANCE_TARGET_SHARE, 0.10), 0.35)
        target_share = base_target
        soft_cap = max(base_target + 0.08, SYMBOL_BALANCE_SOFT_CAP)

        if strong_symbol:
            target_share = min(base_target + 0.05, 0.42)
            soft_cap = max(soft_cap, min(target_share + 0.10, 0.52))
        if elite_symbol:
            target_share = min(target_share + 0.04, 0.46)
            soft_cap = max(soft_cap, min(target_share + 0.12, 0.58))

        if share > soft_cap:
            overflow = (share - soft_cap) / max(0.15, 1.0 - soft_cap)
            penalty = min(max(overflow, 0.0) * 1.8, 1.8)
            if recent_count >= max(5, int(round(total_recent * soft_cap)) + 2):
                penalty = min(penalty + 0.35, 2.2)
            if strong_symbol:
                penalty *= 0.72
            if elite_symbol:
                penalty *= 0.58
            adj -= penalty
            notes.append('寮峰梗鍗犳瘮閬庨珮锛岀◢寰垎娴? if strong_symbol else '杩戞湡鍗犳瘮閬庨珮')
        elif share < target_share * 0.55 and recent_count <= 1:
            bonus = min((target_share - share) * 4.2, 0.95 if strong_symbol else 1.15)
            adj += bonus
            notes.append('杓嫊瑁滃钩琛?)

    if strong_symbol:
        strong_bonus = 0.55 if not elite_symbol else 0.9
        adj += strong_bonus
        notes.append('寮峰梗淇濈暀鍎厛')

    if n <= 1:
        adj += min(SYMBOL_EXPLORATION_BONUS * 0.42, 0.7)
        notes.append('鎺㈢储鏂板梗')
    elif n <= 4:
        adj += min(SYMBOL_EXPLORATION_BONUS * 0.22, 0.45)
        notes.append('瑁滄ǎ鏈?)
    elif n >= 10 and wr < 0.42 and avg_all < 0:
        adj -= 1.25
        notes.append('闀锋湡鍋忓急')

    # 鍘婚噸浣嗕繚鐣欓爢搴忥紝閬垮厤 audit 閲嶈澶瀛?    dedup_notes = []
    for x in notes:
        if x not in dedup_notes:
            dedup_notes.append(x)
    return round(adj, 2), dedup_notes



def _learning_edge(symbol, regime):
    edge = 0.0
    note = ''
    try:
        with LEARN_LOCK:
            ss = dict(LEARN_DB.get('symbol_stats', {}).get(symbol, {}) or {})
        n = int(ss.get('count', 0) or 0)
        if n >= 8:
            wr = float(ss.get('win', 0) or 0) / max(n, 1)
            avg_all = float(ss.get('total_pnl', 0) or 0) / max(n, 1)
            if wr >= 0.60 and avg_all > 0:
                edge += 1.6
                note = '瑭插梗姝峰彶杓冨挤'
            elif wr < 0.40 and avg_all < 0:
                edge -= 2.5
                note = '瑭插梗姝峰彶鍋忓急'
        with AI_LOCK:
            sr = dict((AI_DB.get('symbol_regime_stats', {}) or {}).get(f'{symbol}|{regime}', {}) or {})
        rn = int(sr.get('count', 0) or 0)
        if rn >= 6:
            rwr = float(sr.get('win', 0) or 0) / max(rn, 1)
            ravg = float(sr.get('pnl_sum', 0) or 0) / max(rn, 1)
            regime_cap = 2.0 if regime in ('news', 'breakout') else 3.0
            edge_raw = (rwr - 0.5) * 8.0 + ravg * 0.18
            if regime in ('news', 'breakout') and rn < 10:
                edge_raw = min(edge_raw, 0.8)
            edge += max(min(edge_raw, regime_cap), -regime_cap)
            if not note:
                note = '瑭插梗鍦ㄦ甯傚牬鍨嬫厠鏈夌当瑷堝劒鍕? if (rwr >= 0.55 and ravg > 0) else '瑭插梗鍦ㄦ甯傚牬鍨嬫厠闇€淇濆畧'
    except Exception:
        pass
    return round(edge, 2), note


def _apply_regime_to_signal(symbol, score, desc, entry, sl, tp, est_pnl, breakdown, atr, atr15, atr4h, sl_mult, tp_mult):
    regime_info = _fetch_regime_for_symbol(symbol)
    regime = regime_info.get('regime', 'neutral')
    params = get_regime_params(regime)
    breakdown = dict(breakdown or {})
    score = _safe_num(score)
    entry = _safe_num(entry)
    sl = _safe_num(sl)
    tp = _safe_num(tp)
    atr15 = max(_safe_num(atr15), 0.0)
    side = 1 if score >= 0 else -1
    rr = abs(tp - entry) / max(abs(entry - sl), 1e-9) if entry and sl and tp else _safe_num(breakdown.get('RR', 0))
    direction = regime_info.get('direction', '涓€?)
    conf = _safe_num(regime_info.get('confidence', 0.5))
    tempo = str(regime_info.get('tempo', 'normal') or 'normal')
    slope_dir = 1 if direction == '澶? else -1 if direction == '绌? else 0
    move = _safe_num(regime_info.get('move_3bars_pct', 0))
    volr = _safe_num(regime_info.get('vol_ratio', 1))
    bb_width = abs(_safe_num(regime_info.get('bb_width', 0)))
    chase_pen = abs(_safe_num(breakdown.get('杩藉児棰ㄩ毆', 0)))
    setup_name = str(breakdown.get('Setup', '') or '')
    setup_mode = _normalize_setup_mode(setup_name)

    score_boost = 0.0
    extra = []

    # 鍗€闁撶洡鍎厛杞夋垚鍗€闁撶瓥鐣ワ紝閬垮厤浠嶄互瓒ㄥ嫝/绐佺牬閭忚集铏曠悊
    if regime == 'range' and setup_mode != 'range':
        if side > 0:
            breakdown['Setup'] = '鍗€闁撲笅绶ｅ弽褰?
        else:
            breakdown['Setup'] = '鍗€闁撲笂绶ｅ洖钀?
        setup_name = str(breakdown['Setup'])
        setup_mode = 'range'
        extra.append('鍗€闁撶洡鏀圭敤鍧囧€煎洖姝?)

    # 1) 鍏堢湅 base 鍒嗘瀽鍝佽唱
    quality_boost, quality_notes = _signal_quality_from_breakdown(breakdown, side)
    score_boost += quality_boost
    extra.extend(quality_notes)

    # 2) 甯傚牬鍨嬫厠鏅鸿兘鍔犳瑠
    if regime == 'trend':
        if slope_dir == side:
            score_boost += _cap_market_aux(1.2 + conf * 0.8)
            extra.append('瓒ㄥ嫝鍚屽悜')
            if rr >= 1.6:
                score_boost += 0.6
                extra.append('瓒ㄥ嫝鐩R浣?)
        elif slope_dir != 0:
            score_boost -= _cap_market_aux(1.4 + conf * 0.6)
            extra.append('閫嗚定鍕?)
        else:
            score_boost -= 0.4
    elif regime == 'range':
        if setup_mode == 'range':
            score_boost += 1.2
            extra.append('鍗€闁撶洡浣跨敤鍗€闁撻倧杓?)
            if rr >= 1.25:
                score_boost += 0.4
            if bb_width <= 0.018:
                score_boost += 0.2
        else:
            score_boost -= 1.6
            extra.append('鍗€闁撶洡涓嶈拷瓒ㄥ嫝')
        if chase_pen >= 6:
            score_boost -= 1.0
            extra.append('鍗€闁撶洡閬垮厤杩藉児')
    elif regime == 'news':
        if move >= 3.2 or volr >= 2.4 or chase_pen >= 6:
            score_boost -= 2.0
            extra.append('鏆存媺鏆磋穼寰屽厛绛夊洖韪?)
        elif setup_mode == 'breakout' and abs(score) >= 66 and rr >= 1.9:
            score_boost += 1.0
            extra.append('娑堟伅鐩ょ獊鐮翠絾浠嶄繚瀹?)
        else:
            score_boost -= 1.0
            extra.append('娑堟伅鐩や繚瀹?)
    else:
        if rr >= 1.7:
            score_boost += 1.2
            extra.append('涓€х洡鐣欏挤鍕?)
        elif rr < 1.25:
            score_boost -= 1.4
            extra.append('涓€х洡娣樻卑寮盧R')

    # 3) 瀛哥繏璩囨枡鍔犳瑠
    learn_boost, learn_note = _learning_edge(symbol, regime)
    if learn_boost:
        score_boost += learn_boost * side
        if learn_note:
            extra.append(learn_note)

    eq_value = float(breakdown.get('閫插牬鍝佽唱', 0) or 0)
    eq_boost, eq_note = _entry_quality_feedback(symbol, regime, setup_name, eq_value)
    if eq_boost:
        score_boost += eq_boost * side
        if eq_note:
            extra.append(eq_note)

    strat_row = _strategy_score_lookup(symbol, regime, setup_name)

    # 4) 渚濆競鍫村瀷鎱嬭钃嬮ⅷ鎺у弮鏁革紝浣?TP 浠嶇敱 AI 瀛稿埌鐨?RR 渚嗘焙瀹?    new_sl_mult = float(params.get('sl_mult', sl_mult or 2.0))
    regime_rr_target = float(rr or max(float(tp_mult or 3.5) / max(float(sl_mult or 2.0), 1e-9), MIN_RR_HARD_FLOOR))
    strat_trades = int(strat_row.get('count', strat_row.get('trades', 0)) or 0)
    if strat_trades >= STRATEGY_CAPITAL_MIN_TRADES:
        strat_ev = float(strat_row.get('ev_per_trade', 0) or 0)
        strat_wr = float(strat_row.get('win_rate', 0) or 0)
        if strat_ev > 0.04 and strat_wr >= 55:
            regime_rr_target = min(max(regime_rr_target * 1.06, 1.25), 3.9)
            extra.append('绛栫暐鍎嫝鏀惧ぇ鍒╂饯鐩')
        elif strat_ev < 0 or strat_wr < 45:
            regime_rr_target = min(max(regime_rr_target * 0.94, 1.15), 3.4)
            new_sl_mult = min(max(new_sl_mult * 0.96, 1.2), 3.0)
            extra.append('绛栫暐鍋忓急绺煭鐩')
    if tempo == 'fast':
        new_sl_mult = min(max(new_sl_mult * 1.20, 1.2), 3.2)
        regime_rr_target = min(max(regime_rr_target * 1.30, 1.35), 4.4)
        extra.append('蹇瘈濂忔斁澶P/SL')
    elif tempo == 'slow':
        new_sl_mult = min(max(new_sl_mult * 0.96, 1.15), 3.0)
        regime_rr_target = min(max(regime_rr_target * 0.94, 1.2), 3.2)
        extra.append('鎱㈢瘈濂忔敹鏂傜洰妯?)

    if regime == 'news' and conf >= 0.8:
        new_sl_mult = max(new_sl_mult, 2.4)
        regime_rr_target = min(max(regime_rr_target, 1.8), 3.8)
    elif regime == 'range':
        new_sl_mult = min(new_sl_mult, 1.7)
        regime_rr_target = min(max(regime_rr_target, 1.2), 2.4)
    elif regime == 'trend':
        if slope_dir == side and rr >= 1.5:
            regime_rr_target = max(regime_rr_target, 1.6)
        regime_rr_target = min(max(regime_rr_target, 1.45), 3.4)
    else:
        regime_rr_target = min(max(regime_rr_target, 1.35), 2.8)

    if entry and atr15:
        if side > 0:
            sl = round(entry - atr15 * new_sl_mult, 6)
            tp = round(entry + abs(entry - sl) * regime_rr_target, 6)
        else:
            sl = round(entry + atr15 * new_sl_mult, 6)
            tp = round(entry - abs(entry - sl) * regime_rr_target, 6)
        sl_mult = round(new_sl_mult, 2)
        tp_mult = round(abs(tp - entry) / max(atr15, 1e-9), 2)
        rr = abs(tp - entry) / max(abs(entry - sl), 1e-9)
        est_pnl = round(abs(tp - entry) / max(entry, 1e-9) * 100 * 20, 2)

    ai_score_payload = _score_signal_with_ai_model(
        symbol,
        side,
        breakdown=breakdown,
        regime_info=regime_info,
        desc=desc,
        fallback_score=(score + score_boost),
        extra=extra,
    )
    final_score = round(float(ai_score_payload.get('score', score + score_boost) or 0.0), 1)
    market_state, market_state_conf, market_state_note = _classify_market_atlas(regime=regime, setup=str(breakdown.get('Setup') or ''), breakdown=breakdown, desc=desc)
    breakdown['Regime'] = regime
    breakdown['MarketState'] = market_state
    breakdown['MarketStateConf'] = round(market_state_conf, 3)
    breakdown['MarketStateNote'] = market_state_note
    breakdown['RegimeConf'] = round(conf, 3)
    breakdown['RegimeDir'] = direction
    breakdown['RegimeScoreAdj'] = round(score_boost, 2)
    breakdown['MarketTempo'] = tempo
    breakdown['TempoScore'] = round(float(regime_info.get('tempo_score', 0) or 0), 3)
    breakdown['DecisionInertia'] = round(float(regime_info.get('decision_inertia_delta', 0) or 0), 3)
    breakdown['SignalQuality'] = round(quality_boost, 2)
    breakdown['LearnEdge'] = round(learn_boost, 2)
    breakdown['RR'] = round(rr, 2)
    breakdown['AdaptiveSL'] = round(sl_mult, 2)
    breakdown['AdaptiveTP'] = round(tp_mult, 2)
    breakdown['AIScoreCoverage'] = round(float(ai_score_payload.get('coverage', 0.0) or 0.0), 3)
    breakdown['AISampleConfidence'] = round(float(ai_score_payload.get('sample_confidence', 0.0) or 0.0), 3)
    breakdown['AISampleCount'] = int(ai_score_payload.get('sample_count', 0) or 0)
    breakdown['AIStrategyBoost'] = round(float(ai_score_payload.get('strategy_boost', 0.0) or 0.0), 3)
    breakdown['AIAdaptiveBlend'] = round(float(ai_score_payload.get('adaptive_blend', 0.0) or 0.0), 3)
    breakdown['AIDiscoveredLogicCount'] = int(ai_score_payload.get('discovered_logic_count', 0) or 0)
    top_ai = ai_score_payload.get('top_contributors', []) or []
    if top_ai:
        breakdown['AITopFeature'] = str(top_ai[0].get('feature') or '')

    desc = (desc + '|' if desc else '') + '甯傚牬:{}({}/{:.0%}/{})'.format(regime, direction, conf, tempo)
    desc += '|鍨嬫厠:{}({:.0%})'.format(market_state, market_state_conf)
    if extra:
        desc += '|' + '|'.join(dict.fromkeys(extra))

    with AI_LOCK:
        AI_PANEL['symbol_regimes'][symbol] = dict(regime_info, score_adjust=round(score_boost, 2), quality=round(quality_boost, 2), learn_edge=round(learn_boost, 2), ai_score=round(final_score, 2), ai_score_coverage=round(float(ai_score_payload.get('coverage', 0.0) or 0.0), 3))
        AI_PANEL['regime'] = regime
        AI_PANEL['params'].update({'sl_mult': sl_mult,'tp_mult': tp_mult,'breakeven_atr': float(params.get('breakeven_atr', 0.9)),'trail_trigger_atr': float(params.get('trail_trigger_atr', 1.4)),'trail_pct': float(params.get('trail_pct', 0.035))})
    return final_score, desc, entry, sl, tp, est_pnl, breakdown, atr, atr15, atr4h, sl_mult, tp_mult

def analyze(symbol):
    base = _BASE_ANALYZE(symbol)
    try:
        result = _apply_regime_to_signal(symbol, *base)
    except Exception as e:
        print('AI regime overlay澶辨晽 {}: {}'.format(symbol, e))
        result = base

    try:
        score, desc, entry, sl, tp, est_pnl, breakdown, atr, atr15, atr4h, sl_mult, tp_mult = result
        breakdown = dict(breakdown or {})
        radar = _get_pre_breakout_radar(symbol)
        if radar:
            breakdown['PreBreakoutScore'] = round(float(radar.get('score', 0.0) or 0.0), 2)
            breakdown['PreBreakoutDirection'] = str(radar.get('direction') or '涓€?)
            breakdown['PreBreakoutPhase'] = str(radar.get('phase') or '瑙€瀵?)
            breakdown['PreBreakoutLong'] = round(float(radar.get('long_score', 0.0) or 0.0), 2)
            breakdown['PreBreakoutShort'] = round(float(radar.get('short_score', 0.0) or 0.0), 2)
            breakdown['PreBreakoutTag'] = '|'.join((radar.get('tags') or [])[:4])
            breakdown['PreBreakoutNote'] = str(radar.get('note') or '')
            if radar.get('ready'):
                suffix = '闋愮垎鐧?{}({}/{:.0f})'.format(
                    breakdown.get('PreBreakoutDirection', '涓€?),
                    breakdown.get('PreBreakoutPhase', '瑙€瀵?),
                    float(radar.get('score', 0.0) or 0.0),
                )
                desc = (desc + '|' if desc else '') + suffix
        return score, desc, entry, sl, tp, est_pnl, breakdown, atr, atr15, atr4h, sl_mult, tp_mult
    except Exception:
        return result

def _extract_strategy_key(trade):
    bd = trade.get('breakdown', {}) or {}
    regime = bd.get('Regime', 'neutral')
    setup = bd.get('Setup', 'unknown')
    symbol = trade.get('symbol', 'NA')
    return f'{regime}|{setup}|{symbol}'


def _enhanced_auto_learn():
    with LEARN_LOCK:
        closed = [t for t in LEARN_DB.get('trades', []) if _is_live_source(t.get('source')) and t.get('result') in ('win', 'loss')]
        total = len(closed)
    if total < 20:
        return
    db = AI_DB
    combo_stats = db.setdefault('combo_stats', {})
    regime_stats = db.setdefault('regime_stats', {})
    symbol_regime_stats = db.setdefault('symbol_regime_stats', {})
    market_state_stats = db.setdefault('market_state_stats', {})
    symbol_market_state_stats = db.setdefault('symbol_market_state_stats', {})
    entry_quality_feedback = db.setdefault('entry_quality_feedback', {})
    blocked_strategy_keys = set(db.setdefault('blocked_strategy_keys', []))
    blocked_symbols = set(db.setdefault('blocked_symbols', []))

    combo_stats.clear(); regime_stats.clear(); symbol_regime_stats.clear(); market_state_stats.clear(); symbol_market_state_stats.clear(); entry_quality_feedback.clear()

    recent_closed = closed[-240:]
    for t in recent_closed:
        key = _extract_strategy_key(t)
        metric = float(_trade_learn_metric(t) or 0.0)
        rec = combo_stats.setdefault(key, {
            'count': 0, 'win': 0, 'loss': 0,
            'pnl_sum': 0.0, 'pnl_list': [],
            'gross_win': 0.0, 'gross_loss': 0.0,
        })
        rec['count'] += 1
        if t.get('result') == 'win':
            rec['win'] += 1
            rec['gross_win'] += max(metric, 0.0)
        else:
            rec['loss'] += 1
            rec['gross_loss'] += abs(min(metric, 0.0))
        rec['pnl_sum'] += metric
        rec['pnl_list'].append(metric)

        regime = (t.get('breakdown', {}) or {}).get('Regime', 'neutral')
        rs = regime_stats.setdefault(regime, {'count': 0, 'win': 0, 'pnl_sum': 0.0})
        rs['count'] += 1
        if t.get('result') == 'win':
            rs['win'] += 1
        rs['pnl_sum'] += metric

        sym = t.get('symbol', 'NA')
        sk = f'{sym}|{regime}'
        sr = symbol_regime_stats.setdefault(sk, {'count': 0, 'win': 0, 'pnl_sum': 0.0})
        sr['count'] += 1
        if t.get('result') == 'win':
            sr['win'] += 1
        sr['pnl_sum'] += metric

        market_state = _market_state_from_trade(t)
        ms = market_state_stats.setdefault(market_state, {'count': 0, 'win': 0, 'pnl_sum': 0.0})
        ms['count'] += 1
        if t.get('result') == 'win':
            ms['win'] += 1
        ms['pnl_sum'] += metric
        smk = f'{sym}|{market_state}'
        sms = symbol_market_state_stats.setdefault(smk, {'count': 0, 'win': 0, 'pnl_sum': 0.0})
        sms['count'] += 1
        if t.get('result') == 'win':
            sms['win'] += 1
        sms['pnl_sum'] += metric

        setup_mode = _normalize_setup_mode((t.get('breakdown') or {}).get('Setup') or t.get('setup_label') or '')
        eq_val = float((t.get('breakdown') or {}).get('閫插牬鍝佽唱', 0) or 0)
        eq_bin = 'hq' if eq_val >= 7 else 'mq' if eq_val >= 5 else 'lq'
        for eq_key in (f'{sym}|{regime}|{setup_mode}|{eq_bin}', f'{sym}|{regime}|all|{eq_bin}', f'all|{regime}|{setup_mode}|{eq_bin}'):
            rec_eq = entry_quality_feedback.setdefault(eq_key, {'count': 0, 'loss': 0, 'pnl_sum': 0.0})
            rec_eq['count'] += 1
            rec_eq['pnl_sum'] += metric
            if t.get('result') == 'loss':
                rec_eq['loss'] += 1

    board = []
    new_blocked_strategy_keys = set()
    new_blocked_symbols = set()
    for key, rec in combo_stats.items():
        count = int(rec.get('count', 0) or 0)
        if count < 5:
            continue
        wins = int(rec.get('win', 0) or 0)
        wr = wins / max(count, 1)
        avg = float(rec.get('pnl_sum', 0.0) or 0.0) / max(count, 1)
        pnl_list = list(rec.get('pnl_list', []) or [])
        eq = 100.0
        peak = 100.0
        max_dd = 0.0
        for p in pnl_list:
            step = max(0.01, 1.0 + (float(p) / 100.0))
            eq *= step
            peak = max(peak, eq)
            if peak > 0:
                max_dd = max(max_dd, (peak - eq) / peak * 100.0)
        gross_win = float(rec.get('gross_win', 0.0) or 0.0)
        gross_loss = abs(float(rec.get('gross_loss', 0.0) or 0.0))
        pf = (gross_win / max(gross_loss, 1e-9)) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
        std = (sum((float(p) - avg) ** 2 for p in pnl_list) / max(len(pnl_list), 1)) ** 0.5
        conf = min(count / 50.0, 1.0) * max(0.0, 1.0 - min(std / 3.0, 1.0))

        if count >= 12 and (wr < 0.30 and avg < -0.20 and max_dd >= 25):
            new_blocked_strategy_keys.add(key)

        sym = str(key).split('|')[-1]
        if count >= 14 and wr < 0.28 and avg < -0.25:
            new_blocked_symbols.add(sym)

        score = (
            avg * 35.0 +
            ((wr * 100.0) - 50.0) * 0.6 +
            min(pf, 3.0) * 12.0 +
            min(count, 30) * 0.5 -
            max_dd * 0.35 +
            conf * 8.0
        )

        board.append({
            'strategy': key,
            'count': count,
            'win_rate': round(wr * 100, 1),
            'avg_pnl': round(avg, 4),
            'score': round(score, 3),
            'ev_per_trade': round(avg, 4),
            'profit_factor': round(pf, 3),
            'max_drawdown_pct': round(max_dd, 2),
            'confidence': round(conf, 3),
        })

    for key, rec in list(entry_quality_feedback.items()):
        count = int(rec.get('count', 0) or 0)
        avg = float(rec.get('pnl_sum', 0) or 0.0) / max(count, 1)
        rec['avg_pnl'] = round(avg, 4)
        rec['loss_rate'] = round(float(rec.get('loss', 0) or 0) / max(count, 1), 4)

    board.sort(key=lambda x: (x['score'], x['count'], x['profit_factor'], x['win_rate']), reverse=True)
    db['strategy_scoreboard'] = board[:20]
    db['blocked_strategy_keys'] = sorted(new_blocked_strategy_keys)
    db['blocked_symbols'] = sorted(new_blocked_symbols)

    recent = recent_closed[-60:]
    if recent:
        avg = sum(_trade_learn_metric(t) for t in recent) / len(recent)
        wr = sum(1 for t in recent if t.get('result') == 'win') / len(recent)
        miss_stretch = sum(1 for t in recent if _missed_move_feedback(t) == 'stretch')
        miss_tighten = sum(1 for t in recent if _missed_move_feedback(t) == 'tighten')
        for regime, p in db.get('param_sets', {}).items():
            if wr >= 0.60 and avg > 0:
                p['tp_mult'] = round(min(float(p.get('tp_mult', 3.0)) * 1.02, 5.2), 2)
                p['trail_pct'] = round(max(float(p.get('trail_pct', 0.03)) * 0.99, 0.018), 4)
            elif wr < 0.45:
                p['sl_mult'] = round(min(float(p.get('sl_mult', 2.0)) * 1.02, 3.0), 2)
                p['tp_mult'] = round(max(float(p.get('tp_mult', 3.0)) * 0.98, 2.0), 2)
            if miss_stretch >= 5:
                p['tp_mult'] = round(min(float(p.get('tp_mult', 3.0)) * 1.03, 5.4), 2)
                p['trail_trigger_atr'] = round(min(float(p.get('trail_trigger_atr', 1.4)) * 1.04, 2.8), 2)
            elif miss_tighten >= 5:
                p['tp_mult'] = round(max(float(p.get('tp_mult', 3.0)) * 0.985, 2.0), 2)
                p['sl_mult'] = round(max(float(p.get('sl_mult', 2.0)) * 0.99, 1.2), 2)

    db['param_sets'] = apply_exit_learning_to_params(db.get('param_sets', {}), recent_closed[-90:])
    try:
        db['ai_feature_model'] = _build_ai_feature_model_from_trades(recent_closed)
    except Exception as e:
        print('AI鐗瑰镜妯″瀷鏇存柊澶辨晽:', e)
    db['last_learning'] = tw_now_str('%Y-%m-%d %H:%M:%S')
    save_ai_db(db)
    with AI_LOCK:
        AI_PANEL['best_strategies'] = db.get('strategy_scoreboard', [])[:8]
        AI_PANEL['last_learning'] = db['last_learning']
        AI_PANEL['params'].update(db.get('param_sets', {}).get('neutral', {}))
        AI_PANEL['params']['score_boost']['ai_feature_samples'] = int(((db.get('ai_feature_model') or {}).get('meta') or {}).get('samples', 0) or 0)


def learn_from_closed_trade_legacy_shadow_2(trade_id):
    _BASE_LEARN_FROM_CLOSED_TRADE(trade_id)
    try:
        _enhanced_auto_learn()
    except Exception as e:
        print('澧炲挤瀛哥繏澶辨晽:', e)

def run_simple_backtest_legacy_shadow_2(symbol='BTC/USDT:USDT', timeframe='15m', limit=800, fee_rate=0.0006):
    base = _BASE_RUN_SIMPLE_BACKTEST(symbol=symbol, timeframe=timeframe, limit=limit, fee_rate=fee_rate)
    if not base.get('ok'):
        return base
    regime = _fetch_regime_for_symbol(symbol)
    params = get_regime_params(regime.get('regime', 'neutral'))
    base['market_regime'] = regime
    base['ai_params'] = params
    base['ai_comment'] = f"{symbol} 鐣跺墠灞柤 {regime.get('regime')}锛屽洖娓互 {regime.get('note')} 鍙冭€冭鍙?
    return base

# 姝ｅ紡灏嶅缍佸畾鍒板寮风増锛岄伩鍏嶄粛钀藉洖 legacy v1
run_simple_backtest = run_simple_backtest_legacy_shadow_2
_LEARNING_WORKER = learn_from_closed_trade_legacy_shadow_2

def run_multi_market_backtest(symbols=None):
    started_at = time.time()
    if symbols is None:
        symbols, eligible_count = fetch_top_volume_symbols(AI_MARKET_LIMIT)
    else:
        eligible_count = len(symbols)
    if not symbols:
        with AI_LOCK:
            AUTO_BACKTEST_STATE['running'] = False
            AUTO_BACKTEST_STATE['summary'] = '鎵句笉鍒板彲鍥炴脯甯傚牬'
            AUTO_BACKTEST_STATE['scanned_markets'] = 0
            AUTO_BACKTEST_STATE['errors'] = ['鐒″彲鐢ㄥ競鍫?]
        update_state(ai_panel=dict(AI_PANEL), auto_backtest=dict(AUTO_BACKTEST_STATE))
        return []

    results = []
    errors = []
    scoreboard = []
    scanned = 0
    for idx, sym in enumerate(symbols, start=1):
        try:
            timeframe_data = {}
            regime_seed_df = None
            for tf in AI_MARKET_TIMEFRAMES:
                df_tf = _safe_fetch_ohlcv_df(sym, tf, AI_SNAPSHOT_LIMIT)
                if df_tf is not None:
                    timeframe_data[tf] = _snapshot_from_df(df_tf)
                    if tf == '15m':
                        regime_seed_df = df_tf.rename(columns={'o': 'o', 'h': 'h', 'l': 'l', 'c': 'c', 'v': 'v'})
            if not timeframe_data:
                errors.append(f'{sym}: 鐒℃硶鎶撳彇澶氶€辨湡K绶?)
                continue

            bt = run_simple_backtest(symbol=sym, timeframe='15m', limit=AI_BACKTEST_LIMIT)
            if not bt.get('ok'):
                errors.append(f"{sym}: {bt.get('error', '鍥炴脯澶辨晽')}")
                continue

            regime_info = bt.get('market_regime') if isinstance(bt.get('market_regime'), dict) else None
            if not regime_info and regime_seed_df is not None:
                try:
                    regime_info = classify_market_regime(regime_seed_df)
                except Exception:
                    regime_info = {'regime': 'neutral', 'direction': '涓€?, 'confidence': 0.4}
            regime_info = regime_info or {'regime': 'neutral', 'direction': '涓€?, 'confidence': 0.4}

            scanned += 1
            trades_n = int(bt.get('trades', bt.get('total_trades', 0)) or 0)
            return_pct = round(float(bt.get('return_pct', bt.get('net_profit_pct', 0)) or 0), 2)
            pf_val = round(float(bt.get('profit_factor', 0) or 0), 3) if bt.get('profit_factor') is not None else None
            dd_val = round(float(bt.get('max_drawdown_pct', 0) or 0), 2)
            ev_per_trade = round(return_pct / max(trades_n, 1), 4)
            strategy_mode = 'breakout' if regime_info.get('regime') in ('news', 'breakout') else 'range' if regime_info.get('regime') == 'range' else 'main'
            result_row = {
                'symbol': sym,
                'win_rate': round(float(bt.get('win_rate', 0) or 0), 2),
                'return_pct': return_pct,
                'profit_factor': pf_val,
                'max_drawdown_pct': dd_val,
                'ev_per_trade': ev_per_trade,
                'trades': trades_n,
                'market_regime': regime_info.get('regime', 'neutral'),
                'strategy_mode': strategy_mode,
                'regime_confidence': round(float(regime_info.get('confidence', 0) or 0), 3),
                'timeframes': list(timeframe_data.keys()),
                'updated_at': tw_now_str('%Y-%m-%d %H:%M:%S'),
            }
            results.append(result_row)
            sample_conf = min(max(trades_n / 30.0, 0), 1)
            ev_component = ev_per_trade * 180.0
            pf_component = ((pf_val or 1.0) - 1.0) * 18.0
            dd_penalty = dd_val * 1.6
            win_component = ((result_row['win_rate'] - 50.0) * sample_conf * 0.35)
            score_val = ev_component + pf_component + win_component - dd_penalty + min(trades_n, 30) * 0.45
            scoreboard.append({
                'strategy': '{}|{}'.format(sym, regime_info.get('regime', 'neutral')),
                'strategy_mode': strategy_mode,
                'count': trades_n,
                'win_rate': result_row['win_rate'],
                'avg_pnl': return_pct,
                'ev_per_trade': ev_per_trade,
                'profit_factor': pf_val,
                'max_drawdown_pct': dd_val,
                'score': round(score_val, 2),
                'confidence': round(sample_conf, 2),
                'timeframes': '/'.join(result_row['timeframes']),
                'updated_at': result_row['updated_at'],
            })
            with AI_LOCK:
                _persist_market_snapshot(AI_DB, sym, regime_info, timeframe_data)
                AUTO_BACKTEST_STATE['scanned_markets'] = scanned
                AUTO_BACKTEST_STATE['summary'] = 'AI鍥炴脯閫茶涓?{}/{}锝滄垚鍔?{}锝滃け鏁?{}'.format(idx, len(symbols), scanned, len(errors))
        except Exception as e:
            errors.append(f'{sym}: {str(e)[:90]}')
            print('multi backtest澶辨晽 {}: {}'.format(sym, e))

    results.sort(key=lambda x: (x.get('ev_per_trade', 0), x.get('profit_factor') or 0, -(x.get('max_drawdown_pct', 0) or 0), x.get('trades', 0)), reverse=True)
    scoreboard.sort(key=lambda x: (x.get('score', 0), x.get('ev_per_trade', 0), x.get('count', 0)), reverse=True)
    with AI_LOCK:
        AUTO_BACKTEST_STATE['running'] = False
        AUTO_BACKTEST_STATE['last_run'] = tw_now_str('%Y-%m-%d %H:%M:%S')
        AUTO_BACKTEST_STATE['target_count'] = len(symbols)
        AUTO_BACKTEST_STATE['scanned_markets'] = scanned
        AUTO_BACKTEST_STATE['last_duration_sec'] = round(time.time() - started_at, 1)
        AUTO_BACKTEST_STATE['errors'] = errors[:12]
        AUTO_BACKTEST_STATE['summary'] = '瀹屾垚鍓峽}鎴愪氦閲忓競鍫村洖娓綔鎴愬姛 {}锝滃け鏁?{}锝滃€欓伕绺芥暩 {}'.format(len(symbols), scanned, len(errors), eligible_count)
        AUTO_BACKTEST_STATE['results'] = results[:12]
        AI_PANEL['last_backtest'] = AUTO_BACKTEST_STATE['last_run']
        AI_PANEL['best_strategies'] = scoreboard[:12]
        AI_DB['strategy_scoreboard'] = scoreboard[:20]
        AI_DB.setdefault('backtests', []).append({
            'time': AUTO_BACKTEST_STATE['last_run'],
            'target_count': len(symbols),
            'scanned_markets': scanned,
            'results': results[:20],
            'errors': errors[:20],
            'duration_sec': AUTO_BACKTEST_STATE['last_duration_sec'],
        })
        AI_DB['backtests'] = AI_DB['backtests'][-30:]
        meta = AI_DB.setdefault('market_history_meta', {})
        meta['symbols'] = len((AI_DB.get('market_snapshots', {}) or {}))
        meta['timeframes'] = AI_MARKET_TIMEFRAMES
        meta['last_update'] = AUTO_BACKTEST_STATE['last_run']
        AI_PANEL['market_db_info'] = {
            'symbols': meta['symbols'],
            'timeframes': meta['timeframes'],
            'last_update': meta['last_update'],
        }
        AUTO_BACKTEST_STATE['db_symbols'] = meta['symbols']
        AUTO_BACKTEST_STATE['db_last_update'] = meta['last_update']
        AUTO_BACKTEST_STATE['data_timeframes'] = meta['timeframes']
    save_ai_db(AI_DB)
    update_state(ai_panel=dict(AI_PANEL), auto_backtest=dict(AUTO_BACKTEST_STATE))
    return results

def auto_backtest_thread():
    while True:
        try:
            with AI_LOCK:
                AUTO_BACKTEST_STATE['running'] = True
                AUTO_BACKTEST_STATE['summary'] = '鑷嫊鍥炴脯涓?..'
            sync_ai_state_to_dashboard(force_regime=False)
            run_multi_market_backtest()
            sync_ai_state_to_dashboard(force_regime=False)
        except Exception as e:
            print('鑷嫊鍥炴脯鍩疯绶掑け鏁?', e)
            with AI_LOCK:
                AUTO_BACKTEST_STATE['running'] = False
                AUTO_BACKTEST_STATE['summary'] = '鍥炴脯澶辨晽: {}'.format(str(e)[:80])
        time.sleep(AI_BACKTEST_SLEEP_SEC)

def memory_guard_thread():
    while True:
        try:
            now_ts = time.time()
            with CACHE_LOCK:
                for sym, meta in list(SIGNAL_META_CACHE.items()):
                    ts = float((meta or {}).get('ts', 0) or 0) if isinstance(meta, dict) else 0.0
                    if ts and now_ts - ts > 900:
                        SIGNAL_META_CACHE.pop(sym, None)
                        SCORE_CACHE.pop(sym, None)
                for cache in [SIGNAL_META_CACHE, SCORE_CACHE, ENTRY_LOCKS, POST_CLOSE_LOCKS]:
                    prune_mapping(cache, max_size=500, prune_count=200)
            with PROTECTION_LOCK:
                prune_mapping(PROTECTION_STATE, max_size=500, prune_count=200)
            with AI_LOCK:
                AI_PANEL['memory'] = {'score_cache': len(SCORE_CACHE),'signal_meta_cache': len(SIGNAL_META_CACHE),'entry_locks': len(ENTRY_LOCKS),'post_close_locks': len(POST_CLOSE_LOCKS),'protection_state': len(PROTECTION_STATE),'fvg_orders': len(FVG_ORDERS)}
            gc.collect()
            update_state(ai_panel=dict(AI_PANEL), auto_backtest=dict(AUTO_BACKTEST_STATE))
        except Exception as e:
            print('瑷樻喍楂斿畧璀峰け鏁?', e)
        time.sleep(120)

def enhanced_position_thread():
    while True:
        try:
            with TRAILING_LOCK:
                for sym, ts in list(TRAILING_STATE.items()):
                    side = str(ts.get('side', '')).lower()
                    entry = float(ts.get('entry_price', 0) or 0)
                    atr = float(ts.get('atr', 0) or 0)
                    if entry <= 0 or atr <= 0: continue
                    ticker = exchange.fetch_ticker(sym)
                    mark = float(ticker.get('last', 0) or 0)
                    if mark <= 0: continue
                    params = get_regime_params((AI_PANEL.get('symbol_regimes', {}).get(sym) or {}).get('regime', 'neutral'))
                    breakeven_atr = float(ts.get('breakeven_atr_hint', 0) or params.get('breakeven_atr', 0.9))
                    trail_trigger_atr = float(ts.get('trail_trigger_atr_hint', 0) or params.get('trail_trigger_atr', 1.4))
                    trail_pct = float(ts.get('trail_pct', params.get('trail_pct', 0.035)) or params.get('trail_pct', 0.035))
                    if side in ('buy', 'long'):
                        profit_atr = (mark - entry) / max(atr, 1e-9)
                        if profit_atr >= breakeven_atr: ts['initial_sl'] = max(float(ts.get('initial_sl', 0) or 0), entry)
                        if profit_atr >= trail_trigger_atr: ts['trail_pct'] = min(float(ts.get('trail_pct', trail_pct) or trail_pct), trail_pct)
                    elif side in ('sell', 'short'):
                        profit_atr = (entry - mark) / max(atr, 1e-9)
                        if profit_atr >= breakeven_atr: ts['initial_sl'] = min(float(ts.get('initial_sl', entry * 9) or entry * 9), entry)
                        if profit_atr >= trail_trigger_atr: ts['trail_pct'] = min(float(ts.get('trail_pct', trail_pct) or trail_pct), trail_pct)
        except Exception as e:
            print('寮峰寲淇濇湰/鍕曟厠姝㈢泩澶辨晽:', e)
        time.sleep(8)



def extract_analysis_score(result):
    """鐩稿 analyze() 涓嶅悓鍥炲偝鏍煎紡锛岀┅瀹氬彇鍑哄垎鏁搞€?""
    try:
        if isinstance(result, (list, tuple)):
            if len(result) >= 1:
                return float(result[0] or 0)
        if isinstance(result, dict):
            for key in ('score', 'final_score', 'stable_score', 'raw_score'):
                if key in result:
                    return float(result.get(key) or 0)
        return 0.0
    except Exception:
        return 0.0


def sync_ai_state_to_dashboard(force_regime=False):
    """鎶?AI 闈㈡澘/鍥炴脯鐙€鎱嬪挤鍒跺悓姝ラ€?STATE锛岄伩鍏嶅墠绔叏鏄?--銆?""
    try:
        with AI_LOCK:
            ai_panel = dict(AI_PANEL)
            auto_bt = dict(AUTO_BACKTEST_STATE)
            params = dict((ai_panel.get('params') or {}))
            market_db = dict((ai_panel.get('market_db_info') or {}))
            if force_regime and (not ai_panel.get('regime') or ai_panel.get('regime') in ('鍒濆鍖栦腑', '--')):
                patt = ((STATE.get('market_info') or {}).get('pattern') or '').strip()
                if patt:
                    ai_panel['regime'] = patt
            ai_panel.setdefault('regime', 'neutral')
            ai_panel['params'] = {
                'sl_mult': params.get('sl_mult', 2.0),
                'tp_mult': params.get('tp_mult', 3.3),
                'breakeven_atr': params.get('breakeven_atr', 0.9),
                'trail_trigger_atr': params.get('trail_trigger_atr', 1.5),
                'trail_pct': params.get('trail_pct', 0.035),
            }
            ai_panel['market_db_info'] = {
                'symbols': market_db.get('symbols', auto_bt.get('db_symbols', 0)),
                'timeframes': market_db.get('timeframes', auto_bt.get('data_timeframes', ['5m','15m','1h','4h','1d'])),
                'last_update': market_db.get('last_update', auto_bt.get('db_last_update', '--')),
            }
            auto_bt.setdefault('target_count', AI_MARKET_LIMIT)
            auto_bt.setdefault('data_timeframes', ['5m','15m','1h','4h','1d'])
            update_state(ai_panel=ai_panel, auto_backtest=auto_bt)
    except Exception as e:
        print('鍚屾 AI 鐙€鎱嬪け鏁?', e)


@app.route('/api/ai_status')
def api_ai_status():
    sync_ai_state_to_dashboard(force_regime=True)
    with AI_LOCK:
        return jsonify({
            'ok': True,
            'ai_panel': dict(AI_PANEL),
            'auto_backtest': dict(AUTO_BACKTEST_STATE),
        })


@app.route('/api/ai_db_stats')
def api_ai_db_stats():
    live_open_all = get_live_trades(closed_only=False, pool='all')
    live_closed_all = get_live_trades(closed_only=True, pool='all')
    live_closed_soft = get_live_trades(closed_only=True, pool='soft_live')
    live_closed_trusted = get_live_trades(closed_only=True, pool='trusted_live')
    live_closed_trend = get_trend_live_trades(closed_only=True)
    quarantine_rows = get_live_trades(closed_only=True, pool='quarantine')
    effective_rows = _ai_effective_rows(closed_only=True)

    counts_by_symbol = {}
    for t in effective_rows:
        sym = str((t or {}).get('symbol') or '')
        if sym:
            counts_by_symbol[sym] = counts_by_symbol.get(sym, 0) + 1
    strongest_local_count = max(counts_by_symbol.values()) if counts_by_symbol else 0
    local_ready_symbols = sum(1 for c in counts_by_symbol.values() if c >= AI_MIN_SAMPLE_EFFECT)

    effective_count = len(effective_rows)
    growth_control = _ai_growth_control(effective_count)
    ai_phase = str(growth_control.get('phase') or 'learning')
    ai_ready = bool(effective_count >= TREND_AI_FULL_TRADES and local_ready_symbols > 0)

    def _avg_metric(rows, key, default=0.0):
        vals = []
        for r in rows:
            try:
                v = r.get(key, None)
                if v is not None:
                    vals.append(float(v or 0.0))
            except Exception:
                pass
        return round(sum(vals) / max(len(vals), 1), 4) if vals else float(default)

    latest_trade = dict((effective_rows[-1] if effective_rows else (live_closed_all[-1] if live_closed_all else {})) or {})
    latest_payload = {
        'exit_time': str(latest_trade.get('exit_time') or latest_trade.get('entry_time') or ''),
        'source': str(latest_trade.get('source') or 'live_only'),
        'setup': str(latest_trade.get('setup_label') or ((latest_trade.get('breakdown') or {}).get('Setup')) or ''),
        'symbol': str(latest_trade.get('symbol') or ''),
        'raw_pnl_pct': round(float(latest_trade.get('raw_pnl_pct', latest_trade.get('pnl_pct', 0)) or 0), 4),
    }

    recent_rows = effective_rows[-10:]
    recent_pnls = [float(_trade_learn_metric(t) or 0.0) for t in recent_rows]
    recent_ev_10 = round(sum(recent_pnls) / max(len(recent_pnls), 1), 4) if recent_pnls else 0.0
    recent_fake_breakout_loss_count = sum(1 for t in recent_rows if float(_trade_learn_metric(t) or 0.0) < 0 and str(t.get('setup_label') or ((t.get('breakdown') or {}).get('Setup')) or '').lower().find('breakout') >= 0)

    symbol_blocked_list = []
    for sym in sorted(counts_by_symbol.keys()):
        blocked, _note = _symbol_hard_block(sym)
        if blocked:
            symbol_blocked_list.append(sym)

    payload = {
        'ai_phase': ai_phase,
        'ai_ready': ai_ready,
        'avg_execution_integrity': _avg_metric(effective_rows, 'execution_integrity', 0.0),
        'avg_exit_integrity': _avg_metric(effective_rows, 'exit_integrity', 0.0),
        'avg_label_confidence': _avg_metric(effective_rows, 'label_confidence', 0.0),
        'backtest_run_count': len((BACKTEST_DB.get('runs') or [])) if isinstance(BACKTEST_DB, dict) else 0,
        'closed_live_count': len(live_closed_all),
        'data_scope': 'live_only',
        'last_learning': str((AI_PANEL.get('last_learning') or '-')),
        'latest': latest_payload,
        'local_ready_symbols': int(local_ready_symbols),
        'mode': ai_phase,
        'open_live_count': len(live_open_all),
        'quarantine_count': len(quarantine_rows),
        'recent_ev_10': recent_ev_10,
        'recent_fake_breakout_loss_count': int(recent_fake_breakout_loss_count),
        'recent_miss_good_trade_count': 0,
        'recent_pnl_pct': round(sum(recent_pnls), 4) if recent_pnls else 0.0,
        'soft_live_count': len(live_closed_soft),
        'strongest_local_count': int(strongest_local_count),
        'symbol_blocked_list': symbol_blocked_list,
        'symbols': sorted(counts_by_symbol.keys()),
        'trusted_live_count': len(live_closed_trusted),
        'effective_live_count': int(effective_count),
        'reset_from': TREND_LEARNING_RESET_FROM,
        'ai_influence_mode': ai_phase,
        'ai_influence_note': str(growth_control.get('note') or ''),
    }
    return jsonify(payload)


@app.route('/api/ai_learning_recent')
def api_ai_learning_recent():
    """鍥炲偝鏈€杩戝缈掑埌鐨勫鍠硣鏂欙紙寰?SQLite learning_trades 璁€鍙栵級"""
    limit_arg = request.args.get('limit', '20')
    try:
        limit = max(1, min(int(limit_arg), 200))
    except Exception:
        limit = 20
    return jsonify(build_ai_learning_recent_payload(
        sqlite_fetch_dicts=_sqlite_fetch_dicts,
        sqlite_order_clause=_sqlite_order_clause,
        limit=limit,
        sqlite_db_path=SQLITE_DB_PATH,
        json_module=json,
    ))


@app.route('/api/ai_symbol_stats')
def api_ai_symbol_stats():
    """鍥炲偝鍚勫梗瀛哥繏绛嗘暩鑸囧嫕鐜囷紝鏂逛究蹇€熸鏌?AI 瀛哥繏绲愭灉銆?""
    rows = []
    error = None
    try:
        import sqlite3
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                symbol,
                COUNT(*) AS total,
                SUM(CASE WHEN LOWER(COALESCE(result, '')) = 'win' THEN 1 ELSE 0 END) AS win_count,
                SUM(CASE WHEN LOWER(COALESCE(result, '')) = 'loss' THEN 1 ELSE 0 END) AS loss_count,
                ROUND(100.0 * SUM(CASE WHEN LOWER(COALESCE(result, '')) = 'win' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2) AS win_rate
            FROM learning_trades
            GROUP BY symbol
            ORDER BY total DESC, symbol ASC
            """
        )
        fetched = cur.fetchall()
        conn.close()
        rows = [
            {
                'symbol': r[0],
                'total': int(r[1] or 0),
                'win_count': int(r[2] or 0),
                'loss_count': int(r[3] or 0),
                'win_rate': float(r[4] or 0),
            }
            for r in fetched
        ]
    except Exception as e:
        error = str(e)

    return jsonify({
        'ok': error is None,
        'count': len(rows),
        'data': rows,
        'error': error,
    })


def _api_limit(default=50, max_value=500):
    try:
        limit = int(request.args.get('limit', default))
    except Exception:
        limit = default
    return max(1, min(limit, max_value))


def _api_offset(default=0, max_value=5000):
    try:
        offset = int(request.args.get('offset', default))
    except Exception:
        offset = default
    return max(0, min(offset, max_value))


def _sqlite_fetch_dicts(query, params=()):
    import sqlite3
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _sqlite_table_columns(table_name):
    import sqlite3
    conn = sqlite3.connect(SQLITE_DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table_name})")
        return [str(r[1]) for r in cur.fetchall()]
    except Exception:
        return []
    finally:
        conn.close()


def _sqlite_order_clause(table_name, preferred_cols, fallback='rowid DESC'):
    cols = set(_sqlite_table_columns(table_name))
    usable = [c for c in preferred_cols if c in cols]
    if not usable:
        return fallback
    expr = ', '.join([f"CASE WHEN {c} IS NULL OR {c}='' THEN 1 ELSE 0 END" for c in usable])
    expr += ', ' + ', '.join([f'{c} DESC' for c in usable])
    return expr


@app.route('/api/ai_full_learning')
def api_ai_full_learning():
    """瀹屾暣瀛哥繏璩囨枡锛岄爯瑷渶杩?0绛嗭紝閬垮厤涓€娆℃拡澶ぇ閫犳垚鍗￠爴銆?""
    limit = _api_limit(default=50, max_value=500)
    offset = _api_offset(default=0, max_value=5000)
    rows, error = [], None
    try:
        order_clause = _sqlite_order_clause('learning_trades', ['updated_at', 'created_at', 'exit_time', 'entry_time'])
        rows = _sqlite_fetch_dicts(
            f"""
            SELECT trade_id, symbol, result, source, entry_time, exit_time, created_at, updated_at, data_json
            FROM learning_trades
            ORDER BY {order_clause}
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        for row in rows:
            raw = row.get('data_json')
            try:
                row['data_json'] = json.loads(raw) if raw else {}
            except Exception:
                row['data_json'] = raw
    except Exception as e:
        error = str(e)
    return jsonify({'ok': error is None, 'limit': limit, 'offset': offset, 'count': len(rows), 'data': rows, 'error': error})


@app.route('/api/trade_history')
def api_trade_history_records():
    """鏈€杩戜氦鏄撶磤閷勶紝寰?SQLite trade_history 璁€鍙栥€?""
    limit = _api_limit(default=50, max_value=500)
    offset = _api_offset(default=0, max_value=5000)
    rows, error = [], None
    try:
        order_clause = _sqlite_order_clause('trade_history', ['updated_at', 'created_at', 'exit_time', 'entry_time', 'time'])
        rows = _sqlite_fetch_dicts(
            f"""
            SELECT *
            FROM trade_history
            ORDER BY {order_clause}
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        for row in rows:
            if 'data_json' in row:
                raw = row.get('data_json')
                try:
                    row['data_json'] = json.loads(raw) if raw else {}
                except Exception:
                    pass
    except Exception as e:
        error = str(e)
    return jsonify({'ok': error is None, 'limit': limit, 'offset': offset, 'count': len(rows), 'data': rows, 'error': error})


@app.route('/api/risk_logs')
def api_risk_logs():
    """鏈€杩戦ⅷ鎺т簨浠剁磤閷勩€?""
    limit = _api_limit(default=50, max_value=500)
    offset = _api_offset(default=0, max_value=5000)
    rows, error = [], None
    try:
        order_clause = _sqlite_order_clause('risk_events', ['created_at', 'event_time', 'timestamp'])
        rows = _sqlite_fetch_dicts(
            f"""
            SELECT *
            FROM risk_events
            ORDER BY {order_clause}
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        for row in rows:
            if 'payload_json' in row:
                raw = row.get('payload_json')
                try:
                    row['payload_json'] = json.loads(raw) if raw else {}
                except Exception:
                    pass
    except Exception as e:
        error = str(e)
    return jsonify({'ok': error is None, 'limit': limit, 'offset': offset, 'count': len(rows), 'data': rows, 'error': error})


@app.route('/api/audit_logs')
def api_audit_logs():
    """鏈€杩戠郴绲辩ń鏍?鍋甸尟绱€閷勩€?""
    limit = _api_limit(default=50, max_value=500)
    offset = _api_offset(default=0, max_value=5000)
    rows, error = [], None
    try:
        order_clause = _sqlite_order_clause('audit_logs', ['created_at', 'event_time', 'timestamp'])
        rows = _sqlite_fetch_dicts(
            f"""
            SELECT *
            FROM audit_logs
            ORDER BY {order_clause}
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        for row in rows:
            if 'payload_json' in row:
                raw = row.get('payload_json')
                try:
                    row['payload_json'] = json.loads(raw) if raw else {}
                except Exception:
                    pass
    except Exception as e:
        error = str(e)
    return jsonify({'ok': error is None, 'limit': limit, 'offset': offset, 'count': len(rows), 'data': rows, 'error': error})


@app.route('/api/backtest_runs')
def api_backtest_runs():
    """鏈€杩戝洖娓磤閷勩€?""
    limit = _api_limit(default=30, max_value=200)
    offset = _api_offset(default=0, max_value=2000)
    rows, error = [], None
    try:
        order_clause = _sqlite_order_clause('backtest_runs', ['created_at', 'run_time', 'timestamp'])
        rows = _sqlite_fetch_dicts(
            f"""
            SELECT *
            FROM backtest_runs
            ORDER BY {order_clause}
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        for row in rows:
            for key in ('payload_json', 'summary_json', 'result_json', 'data_json'):
                if key in row:
                    raw = row.get(key)
                    try:
                        row[key] = json.loads(raw) if raw else {}
                    except Exception:
                        pass
    except Exception as e:
        error = str(e)
    return jsonify({'ok': error is None, 'limit': limit, 'offset': offset, 'count': len(rows), 'data': rows, 'error': error})


@app.route('/api/ai_debug_last_decision')
def api_ai_debug_last_decision():
    """蹇€熺湅鏈€杩戣嚜鍕曚笅鍠?鏈笅鍠師鍥狅紝涓嶆敼鍕曚富娴佺▼锛屽彧璁€鍙栧揩鍙栫媭鎱嬨€?""
    try:
        with AUDIT_LOCK:
            audit_map = snapshot_mapping(AUTO_ORDER_AUDIT)
        with _DT_LOCK:
            threshold_state = dict(_DT)
        payload = build_ai_debug_payload(
            audit_map=audit_map,
            threshold_state=threshold_state,
            risk_status=get_risk_status(),
            market_state=MARKET_STATE,
            session_state={},
            now_text=tw_now_str('%Y-%m-%d %H:%M:%S'),
        )
        RUNTIME_STATE.update(
            threshold=threshold_state,
            risk_status=get_risk_status(),
            market_state=dict(MARKET_STATE or {}),
            session_state={},
            audit=payload.get('auto_order_audit', {}),
        )
        symbol = str(request.args.get('symbol') or '').strip()
        if symbol:
            payload['symbol'] = symbol
            payload['decision'] = payload.get('auto_order_audit', {}).get(symbol)
        return jsonify(payload)
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

def api_state_enhanced():
    resp = _BASE_API_STATE()
    try:
        payload = resp.get_json() or {}
    except Exception:
        payload = {}
    try:
        sync_ai_state_to_dashboard(force_regime=False)
        with AI_LOCK:
            payload['ai_panel'] = dict(AI_PANEL)
            payload['auto_backtest'] = dict(AUTO_BACKTEST_STATE)
        if 'learn_summary' not in payload:
            payload['learn_summary'] = dict(STATE.get('learn_summary', {}))
        payload['trend_dashboard'] = _ui_trend_payload()
        top_signals = []
        for s in list(payload.get('top_signals', []) or []):
            try:
                row = dict(s)
                bd = dict(row.get('breakdown') or {})
                regime = str(bd.get('Regime', payload.get('market_info', {}).get('pattern', 'neutral')) or 'neutral')
                row.update(_ui_trend_payload(symbol=row.get('symbol', ''), regime=regime, setup=row.get('setup_label') or bd.get('Setup', '')))
                top_signals.append(row)
            except Exception:
                top_signals.append(s)
        if top_signals:
            payload['top_signals'] = top_signals
        active_positions = []
        for p in list(payload.get('active_positions', []) or []):
            try:
                row = dict(p)
                sym = str(row.get('symbol') or '')
                sig = dict(SIGNAL_META_CACHE.get(sym) or {})
                regime = str(sig.get('regime') or ((AI_PANEL.get('symbol_regimes', {}) or {}).get(sym, {}) or {}).get('regime') or payload.get('market_info', {}).get('pattern', 'neutral') or 'neutral')
                row.update(_ui_trend_payload(symbol=sym, regime=regime, setup=sig.get('setup_label') or ''))
                active_positions.append(row)
            except Exception:
                active_positions.append(p)
        if active_positions:
            payload['active_positions'] = active_positions
        return jsonify(payload)
    except Exception as e:
        payload['api_state_fix_error'] = str(e)
        return jsonify(payload)


@app.route('/api/state_lite')
def api_state_lite():
    def _builder():
        base = _BASE_API_STATE()
        payload = (base.get_json() or {}) if hasattr(base, 'get_json') else {}
        slim = build_state_lite_payload(payload)
        slim['ai_panel'] = dict(AI_PANEL)
        slim['auto_backtest'] = dict(AUTO_BACKTEST_STATE)
        return slim
    return jsonify(state_lite_cache.get_or_build(_builder, force=bool(request.args.get('force'))))


@app.route('/api/positions_state')
def api_positions_state():
    def _builder():
        base = _BASE_API_STATE()
        payload = (base.get_json() or {}) if hasattr(base, 'get_json') else {}
        return build_positions_payload(payload)
    return jsonify(positions_cache.get_or_build(_builder, force=bool(request.args.get('force'))))


@app.route('/api/ai_panel_state')
def api_ai_panel_state():
    def _builder():
        base = api_state_enhanced()
        payload = (base.get_json() or {}) if hasattr(base, 'get_json') else {}
        return build_ai_panel_payload(payload)
    return jsonify(ai_panel_cache.get_or_build(_builder, force=bool(request.args.get('force'))))


@app.route('/api/cancel_fvg_order', methods=['POST'])
def api_cancel_fvg_order_alias():
    return api_fvg_cancel()

@app.route('/api/force_backtest', methods=['POST'])
def api_force_backtest():
    try:
        results = run_multi_market_backtest()
        sync_ai_state_to_dashboard(force_regime=False)
        return jsonify({'ok': True, 'results': results, 'summary': AUTO_BACKTEST_STATE.get('summary', ''), 'auto_backtest': dict(AUTO_BACKTEST_STATE), 'ai_panel': dict(AI_PANEL)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/risk_override', methods=['POST'])
def api_risk_override():
    try:
        payload = request.get_json(silent=True) or {}
        action = str(payload.get('action') or 'release').lower()
        if action == 'release':
            return jsonify(_manual_release_risk_state())
        return jsonify({'ok': False, 'message': '鏈煡 action'}), 400
    except Exception as e:
        return jsonify({'ok': False, 'message': str(e)}), 500

@app.route('/api/params', methods=['GET'])
def api_params():
    with AI_LOCK:
        return jsonify({'ok': True,'params': AI_PANEL.get('params', {}),'best_strategies': AI_PANEL.get('best_strategies', []),'auto_backtest': dict(AUTO_BACKTEST_STATE)})

def api_state():
    resp = _BASE_API_STATE()
    try:
        payload = resp.get_json()
        payload['ai_panel'] = dict(AI_PANEL)
        payload['auto_backtest'] = dict(AUTO_BACKTEST_STATE)
        return jsonify(payload)
    except Exception:
        return resp

def reconcile_exchange_state():
    """鍟熷嫊鏅傚悓姝ヤ氦鏄撴墍鐪熷鍊変綅鑸囨湰鍦颁繚璀风媭鎱嬶紝闄嶄綆鏈湴/浜ゆ槗鎵€涓嶅悓姝ラⅷ闅€?""
    try:
        positions = exchange.fetch_positions()
    except Exception as e:
        print('鍟熷嫊鍚屾鍊変綅澶辨晽: {}'.format(e))
        return
    live = []
    with PROTECTION_LOCK:
        for p in positions or []:
            contracts = float((p or {}).get('contracts', 0) or 0)
            if abs(contracts) <= 0:
                continue
            sym = p.get('symbol')
            live.append(sym)
            PROTECTION_STATE.setdefault(sym, {})
            PROTECTION_STATE[sym]['has_position'] = True
            PROTECTION_STATE[sym]['updated_at'] = tw_now_str()
        for sym in list(PROTECTION_STATE.keys()):
            if sym not in live:
                PROTECTION_STATE[sym]['has_position'] = False
                PROTECTION_STATE[sym]['updated_at'] = tw_now_str()
    update_state(protection_state=snapshot_mapping(PROTECTION_STATE))

def start_all_threads():
    load_full_state()
    load_risk_state()
    hydrate_trade_history(limit=30)
    reconcile_exchange_state()
    _refresh_ai_panel_market_meta()
    sync_openai_trade_state(push_runtime=False)
    LEARNING_QUEUE.start()
    sync_ai_state_to_dashboard(force_regime=True)
    append_audit_log('system', 'start_all_threads', {'sqlite_path': SQLITE_DB_PATH})
    threads = [(globals()[fn_name], name) for fn_name, name in default_thread_specs() if fn_name in globals()]
    for fn, name in threads:
        t = threading.Thread(target=watchdog, args=(fn, name), daemon=True, name=name)
        t.start()
    app.view_functions['api_state'] = api_state_enhanced
    backend_threads = _backend_threads_snapshot()
    update_state(ai_panel=dict(AI_PANEL), auto_backtest=dict(AUTO_BACKTEST_STATE), news_score=0, news_sentiment='宸插仠鐢?, latest_news_title='鏂拌仦绯荤当宸插仠鐢?, backend_threads=backend_threads)
    RUNTIME_STATE.update(ai_panel=dict(AI_PANEL), auto_backtest=dict(AUTO_BACKTEST_STATE), backend_threads=backend_threads)
    print('=== V11 AI / UI 淇鐗堝煼琛岀窉宸插暉鍕曪紙鏂拌仦绯荤当宸插仠鐢級 ===')



def load_learning_db():
    with LEARN_LOCK:
        return dict(LEARN_DB)

# =========================
# V15 澶栨帥澧炲挤锛堜繚璀峰柈鑷嫊铏曠疆 / replay / 甯傚牬鍏辫瓨 / 閫佸柈瀹堥杸锛?# =========================
ensure_replay_tables(SQLITE_DB_PATH)
_original_verify_protection_orders = verify_protection_orders
_original_ensure_exchange_protection = ensure_exchange_protection

def _set_auto_ai_mode(mode: str, reasons=None):
    global AUTO_AI_MODE
    mode = str(mode or 'normal')
    AUTO_AI_MODE = mode
    try:
        update_state(ai_mode=mode, ai_mode_reasons=list(reasons or []))
    except Exception:
        pass

def _refresh_market_consensus_light():
    global LAST_MARKET_CONSENSUS
    try:
        btc = exchange.fetch_ohlcv('BTC/USDT:USDT', '1h', limit=40)
        eth = exchange.fetch_ohlcv('ETH/USDT:USDT', '1h', limit=40)
        def _pack(rows):
            d = pd.DataFrame(rows, columns=['t','o','h','l','c','v'])
            return {'price': float(d['c'].iloc[-1]), 'ma_fast': float(d['c'].rolling(10).mean().iloc[-1]), 'ma_slow': float(d['c'].rolling(30).mean().iloc[-1])}
        breadth = 0.0
        try:
            breadth = float((AI_PANEL.get('market_db_info') or {}).get('symbols', 0) or 0) / 200.0 - 0.5
        except Exception:
            breadth = 0.0
        LAST_MARKET_CONSENSUS = build_market_consensus(_pack(btc), _pack(eth), {'breadth': breadth, 'volatility_state': AI_PANEL.get('regime', 'normal')})
    except Exception as e:
        LAST_MARKET_CONSENSUS = {'market_consensus_bias': 'mixed', 'market_consensus_strength': 0.0, 'error': str(e)}
    return LAST_MARKET_CONSENSUS

def verify_protection_orders(symbol, side, sl_price, tp_price):
    global PROTECTION_FAIL_STREAK
    sl_ok, tp_ok = _original_verify_protection_orders(symbol, side, sl_price, tp_price)
    if not sl_ok:
        PROTECTION_FAIL_STREAK += 1
    else:
        PROTECTION_FAIL_STREAK = 0
    mode_info = derive_auto_mode(api_error_streak=API_ERROR_STREAK, protection_fail_streak=PROTECTION_FAIL_STREAK, learning_stale_minutes=0.0, schema_ok=True)
    _set_auto_ai_mode(mode_info.get('mode', 'normal'), mode_info.get('reasons', []))
    return sl_ok, tp_ok

def ensure_exchange_protection(sym, side, pos_side, qty, sl_price, tp_price, verify_wait_sec=1.0):
    global PROTECTION_FAIL_STREAK
    sl_ok, tp_ok = _original_ensure_exchange_protection(sym, side, pos_side, qty, sl_price, tp_price, verify_wait_sec=verify_wait_sec)
    if not (sl_ok and tp_ok):
        # 绗簩娆¤鎺涢洐閲嶇⒑瑾?        time.sleep(1.0)
        sl_ok2, tp_ok2 = _original_ensure_exchange_protection(sym, side, pos_side, qty, sl_price, tp_price, verify_wait_sec=0.5)
        sl_ok = bool(sl_ok or sl_ok2)
        tp_ok = bool(tp_ok or tp_ok2)
    if not (sl_ok and tp_ok):
        PROTECTION_FAIL_STREAK += 1
        action = protection_failure_action(sym, {'sl_ok': sl_ok, 'tp_ok': tp_ok}, missing_seconds=3.5)
        append_risk_event('protection_missing_auto_action', action)
        append_audit_log('protection', '淇濊鍠璀夊け鏁楀凡鑷嫊铏曠疆', action)
        with RISK_LOCK:
            RISK_STATE['trading_halted'] = True
            RISK_STATE['halt_reason'] = '淇濊鍠己澶憋紝鑷嫊鏆仠鏂板柈'
        update_state(risk_status=get_risk_status(), halt_reason=RISK_STATE.get('halt_reason', ''))
        _set_auto_ai_mode('observe', ['淇濊鍠己澶憋紝鑷嫊鏆仠鏂板柈'])
    else:
        PROTECTION_FAIL_STREAK = 0
    return sl_ok, tp_ok

def _is_soft_execution_pause(gate):
    try:
        gate = dict(gate or {})
        reasons = [str(x) for x in (gate.get('reasons') or [])]
        joined = ' | '.join(reasons).lower()
        hard_words = ['api', 'timeout', 'offline', 'network', 'schema', 'error', '淇濊鍠?, 'maintenance', '鍋滄', 'stale']
        soft_words = ['娣卞害閬庤杽', 'depth', 'spread', '婊戝児', '钖?, 'liquidity', 'orderbook']
        if any(w in joined for w in hard_words):
            return False
        return any(w.lower() in joined for w in soft_words)
    except Exception:
        return False

def apply_execution_guard(symbol, side, margin_pct):
    global API_ERROR_STREAK
    try:
        snap = exec_quality_snapshot(exchange, symbol, side)
        if snap.get('notes'):
            API_ERROR_STREAK = min(API_ERROR_STREAK + 1, 10)
        else:
            API_ERROR_STREAK = max(API_ERROR_STREAK - 1, 0)
        gate = execution_gate(snap, api_error_streak=API_ERROR_STREAK)
        if gate.get('action') == 'pause':
            if _is_soft_execution_pause(gate):
                softened_gate = dict(gate or {})
                softened_gate['action'] = 'penalty'
                softened_gate['softened'] = True
                softened_gate['score_penalty'] = max(float(softened_gate.get('score_penalty', 0.0) or 0.0), 6.0)
                reasons = list(softened_gate.get('reasons') or [])
                if '娣卞害鍋忚杽锛屾敼鎵ｅ垎闄嶅€夎檿鐞? not in reasons:
                    reasons.append('娣卞害鍋忚杽锛屾敼鎵ｅ垎闄嶅€夎檿鐞?)
                softened_gate['reasons'] = reasons
                mp = float(margin_pct or 0) * min(float(softened_gate.get('margin_mult', 1.0) or 1.0), 0.42)
                return {'allow': True, 'margin_pct': mp, 'snapshot': snap, 'gate': softened_gate}
            return {'allow': False, 'margin_pct': margin_pct, 'snapshot': snap, 'gate': gate}
        mp = float(margin_pct or 0) * float(gate.get('margin_mult', 1.0) or 1.0)
        return {'allow': True, 'margin_pct': mp, 'snapshot': snap, 'gate': gate}
    except Exception as e:
        API_ERROR_STREAK = min(API_ERROR_STREAK + 1, 10)
        softened_gate = {'action': 'penalty', 'softened': True, 'score_penalty': 7.0, 'margin_mult': 0.38, 'reasons': ['execution guard error', 'execution guard 澶辨晽鏀圭偤闄嶅€夋墸鍒?]}
        mp = float(margin_pct or 0) * float(softened_gate.get('margin_mult', 0.38) or 0.38)
        return {'allow': True, 'margin_pct': mp, 'snapshot': {'error': str(e)}, 'gate': softened_gate}



@app.route('/api/decision_funnel')
def api_decision_funnel():
    limit = _api_limit(default=50, max_value=300)
    with AUDIT_LOCK:
        audit_map = snapshot_mapping(AUTO_ORDER_AUDIT)
    items = []
    for symbol, payload in dict(audit_map or {}).items():
        row = dict(payload or {})
        row['symbol'] = symbol
        items.append(row)
    items.sort(key=lambda x: (0 if x.get('can_trade') else 1, str(x.get('symbol') or '')))
    return jsonify(build_decision_funnel_payload(items, limit))


@app.route('/api/learning_sample_review')
def api_learning_sample_review():
    limit = _api_limit(default=50, max_value=300)
    live_closed = get_live_trades(closed_only=True, pool='all')
    return jsonify(build_learning_sample_review_payload(live_closed=live_closed, limit=limit, reset_from=TREND_LEARNING_RESET_FROM))

@app.route('/api/ai_learning_health')
def api_ai_learning_health():
    live_closed = get_live_trades(closed_only=True, pool='all')
    return jsonify(build_ai_learning_health_payload(live_closed=live_closed, reset_from=TREND_LEARNING_RESET_FROM))


@app.route('/api/ai_strategy_matrix')
def api_ai_strategy_matrix():
    live_closed = get_live_trades(closed_only=True, pool='all')
    return jsonify(build_ai_strategy_matrix_payload(live_closed=live_closed, reset_from=TREND_LEARNING_RESET_FROM))


@app.route('/api/ai_decision_explain')
def api_ai_decision_explain():
    symbol = str(request.args.get('symbol', '') or '').strip()
    if not symbol:
        return jsonify({'ok': False, 'error': 'missing symbol'}), 400
    with AUDIT_LOCK:
        audit_map = snapshot_mapping(AUTO_ORDER_AUDIT)
    replay_items = load_decision_input_snapshots(SQLITE_DB_PATH, limit=120)
    return jsonify(build_ai_decision_explain_payload(symbol=symbol, audit_map=audit_map, replay_items=replay_items))


@app.route('/api/ai_replay_inputs')
def api_ai_replay_inputs():
    limit = int(request.args.get('limit', 50) or 50)
    return jsonify({'ok': True, 'dataset_meta': _dataset_meta(), 'items': load_decision_input_snapshots(SQLITE_DB_PATH, limit=limit)})

@app.route('/api/ai_learning_weight_summary')
def api_ai_learning_weight_summary():
    db = load_learning_db()
    trades = list((db or {}).get('trades', []) or [])
    return jsonify({'ok': True, 'summary': ext_learning_weight_summary(trades, reset_from=TREND_LEARNING_RESET_FROM)})

@app.route('/api/neutral_failure_stats')
def api_neutral_failure_stats():
    db = load_learning_db()
    trades = list((db or {}).get('trades', []) or [])
    return jsonify({'ok': True, 'stats': neutral_failure_stats(trades)})

@app.route('/api/trigger_hit_leaderboard')
def api_trigger_hit_leaderboard():
    db = load_learning_db()
    trades = list((db or {}).get('trades', []) or [])
    return jsonify({'ok': True, 'items': trigger_hit_leaderboard(trades, limit=20)})

@app.route('/api/ai_market_consensus')
def api_ai_market_consensus():
    return jsonify({'ok': True, 'consensus': _refresh_market_consensus_light()})

@app.route('/api/ai_mode')
def api_ai_mode_v15():
    info = derive_auto_mode(api_error_streak=API_ERROR_STREAK, protection_fail_streak=PROTECTION_FAIL_STREAK, learning_stale_minutes=0.0, schema_ok=True)
    return jsonify({'ok': True, 'mode': AUTO_AI_MODE, 'derived': info})

@app.route('/api/ai_sandbox')
def api_ai_sandbox_v15():
    score = float(request.args.get('score', 60) or 60)
    threshold = float(request.args.get('threshold', ORDER_THRESHOLD) or ORDER_THRESHOLD)
    margin_pct = float(request.args.get('margin_pct', 0.04) or 0.04)
    rr = float(request.args.get('rr', 1.6) or 1.6)
    cons = _refresh_market_consensus_light()
    adjusted = {
        'score': round(score + (1.5 if cons.get('market_consensus_bias') == 'bull' else -1.5 if cons.get('market_consensus_bias') == 'bear' else 0), 4),
        'threshold': threshold,
        'margin_pct': margin_pct,
        'rr': rr,
        'market_consensus': cons,
        'mode': AUTO_AI_MODE,
    }
    return jsonify({'ok': True, 'sandbox': adjusted})

if __name__=='__main__':
    app.view_functions['api_state'] = api_state_enhanced
    start_all_threads()
    app.run(host='0.0.0.0',port=int(os.environ.get("PORT",8080)),threaded=True)

# =========================

