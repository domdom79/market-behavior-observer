# =============================================================================
# Market Behavior Observer - sanitized public agent
# - Neutral naming for structural references (ref_upper/ref_lower)
# - detect_swings returns anonymized structural references
# - compute_anonymized_metrics uses neutral names (no price levels persisted)
# - Learning stores aggregated/normalized features and coarse labels
# - Public-friendly: no actionable trading terminology or price levels emitted/persisted
# =============================================================================

import os
import glob
import time
import threading
import json
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any

import pandas as pd

# =========================
# CONFIG (sanitized defaults)
# =========================
MT4_TIME_FORMAT = "%Y.%m.%d %H:%M"

DEFAULT_CONFIG = {
    "TELEGRAM_TOKEN": "",
    "TELEGRAM_CHAT_ID": "",
    "SYMBOL": "",
    "LOT": 0.0,
    "PIP_OFFSET": 0.0,
    "NEWS_ENABLED": False,
    "LEARNING_ENABLED": True,
    "LEARN_HORIZON_MINUTES": 20,
    "LEARN_HIT_MOVE_USD": 2.0,
    "LEARN_MAX_OPEN_SIGNALS": 500,
    "MINIMAL_OFFSET_USD": 0.5,
    "M1_RESET_SECONDS": 3600
}

CONFIG_PATH = os.path.join(os.getcwd(), "config.json")
LEARN_DB_PATH = os.path.join(os.getcwd(), "learning_db.json")

def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                j = json.load(f)
            if isinstance(j, dict):
                cfg.update(j)
        except Exception as e:
            print(f"[CONFIG] ⚠️ Cannot read config.json: {e}")
    cfg["TELEGRAM_TOKEN"] = os.environ.get("TELEGRAM_TOKEN", cfg.get("TELEGRAM_TOKEN", ""))
    cfg["TELEGRAM_CHAT_ID"] = os.environ.get("TELEGRAM_CHAT_ID", cfg.get("TELEGRAM_CHAT_ID", ""))
    try:
        cfg["LOT"] = float(cfg["LOT"])
    except Exception:
        cfg["LOT"] = DEFAULT_CONFIG["LOT"]
    try:
        cfg["PIP_OFFSET"] = float(cfg["PIP_OFFSET"])
    except Exception:
        cfg["PIP_OFFSET"] = DEFAULT_CONFIG["PIP_OFFSET"]
    try:
        cfg["LEARN_HORIZON_MINUTES"] = int(cfg["LEARN_HORIZON_MINUTES"])
    except Exception:
        cfg["LEARN_HORIZON_MINUTES"] = DEFAULT_CONFIG["LEARN_HORIZON_MINUTES"]
    try:
        cfg["LEARN_HIT_MOVE_USD"] = float(cfg["LEARN_HIT_MOVE_USD"])
    except Exception:
        cfg["LEARN_HIT_MOVE_USD"] = DEFAULT_CONFIG["LEARN_HIT_MOVE_USD"]
    try:
        cfg["LEARN_MAX_OPEN_SIGNALS"] = int(cfg["LEARN_MAX_OPEN_SIGNALS"])
    except Exception:
        cfg["LEARN_MAX_OPEN_SIGNALS"] = DEFAULT_CONFIG["LEARN_MAX_OPEN_SIGNALS"]
    try:
        cfg["MINIMAL_OFFSET_USD"] = float(cfg.get("MINIMAL_OFFSET_USD", DEFAULT_CONFIG["MINIMAL_OFFSET_USD"]))
    except Exception:
        cfg["MINIMAL_OFFSET_USD"] = DEFAULT_CONFIG["MINIMAL_OFFSET_USD"]
    try:
        cfg["M1_RESET_SECONDS"] = int(cfg.get("M1_RESET_SECONDS", DEFAULT_CONFIG["M1_RESET_SECONDS"]))
    except Exception:
        cfg["M1_RESET_SECONDS"] = DEFAULT_CONFIG["M1_RESET_SECONDS"]
    cfg["NEWS_ENABLED"] = bool(cfg.get("NEWS_ENABLED", False))
    cfg["LEARNING_ENABLED"] = bool(cfg.get("LEARNING_ENABLED", True))
    return cfg

CFG = load_config()
MINIMAL_OFFSET_USD = CFG["MINIMAL_OFFSET_USD"]
M1_RESET_SECONDS = CFG["M1_RESET_SECONDS"]
SYMBOL = CFG["SYMBOL"]

# Paths and persistence (public-friendly)
SIGNALS_LOG_FILE = os.path.join(os.getcwd(), "signals_log.jsonl")
RECENT_SIGNALS_KEEP = 50
DAILY_REPORTS_DIR = os.path.join(os.getcwd(), "daily_reports")
PROPOSALS_DIR = os.path.join(os.getcwd(), "proposals")
os.makedirs(DAILY_REPORTS_DIR, exist_ok=True)
os.makedirs(PROPOSALS_DIR, exist_ok=True)

# Duplicate windows
DUPLICATE_WINDOW_SECONDS = {"M1": 30, "M5": 300, "M15": 900, "H1": 3600, "H4": 3600}

# File scanning (example MT4 path)
base_path = os.path.join(os.environ.get("APPDATA", ""), "MetaQuotes", "Terminal")
terminal_folders = glob.glob(os.path.join(base_path, "*"))
FILE_GLOBS = ["market_data*.csv", "market_data.csv"]
TF_MAP = {1: "M1", 5: "M5", 15: "M15", 60: "H1", 240: "H4"}
LTF_SET = {"M1", "M5", "M15"}
HTF_SET = {"H1", "H4"}

# State with neutral names
@dataclass
class TFState:
    tf: str
    price: Optional[float] = None
    ref_upper: Optional[float] = None    # neutral: recent structural upper reference
    ref_lower: Optional[float] = None    # neutral: recent structural lower reference
    directional_bias: Optional[str] = None
    observation_tag: Optional[str] = None
    magnitude: Optional[float] = None    # neutral: absolute distance between refs

states: Dict[str, TFState] = {tf: TFState(tf) for tf in ["M1", "M5", "M15", "H1", "H4"]}

signals_history: Dict[str, float] = {}
signals_history_lock = threading.Lock()

last_dfs: Dict[str, pd.DataFrame] = {}
learning_open: List[Dict[str, Any]] = []
learning_stats: Dict[str, Any] = {}
_last_learn_eval: float = 0.0

# Helpers
def fmt(x):
    return "NA" if x is None else f"{x:.6f}"

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Robust time parse
def parse_time_to_epoch(time_str: Any) -> int:
    if time_str is None:
        return 0
    s = str(time_str).strip()
    if s == "":
        return 0
    try:
        if s.isdigit():
            return int(s)
        val = float(s)
        if val > 1e9:
            return int(val)
    except Exception:
        pass
    try:
        dt = datetime.strptime(s, MT4_TIME_FORMAT)
        return int(dt.timestamp())
    except Exception:
        pass
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if not pd.isna(dt):
            return int(pd.Timestamp(dt).to_pydatetime().timestamp())
    except Exception:
        pass
    today = datetime.now().date()
    time_formats = ("%H:%M:%S", "%H:%M")
    for fmt_try in time_formats:
        try:
            t = datetime.strptime(s, fmt_try).time()
            dt = datetime.combine(today, t)
            return int(dt.timestamp())
        except Exception:
            continue
    if "." in s:
        s_core = s.split(".")[0]
        for fmt_try in time_formats:
            try:
                t = datetime.strptime(s_core, fmt_try).time()
                dt = datetime.combine(today, t)
                return int(dt.timestamp())
            except Exception:
                continue
    return 0

def _time_column_in_df(df: pd.DataFrame) -> Optional[str]:
    if df is None:
        return None
    candidates = [c for c in df.columns if any(k in c for k in ("time","date","datetime","timestamp"))]
    return candidates[0] if candidates else None

def _get_last_candle_epoch_for_tf(tf: str) -> int:
    try:
        df = last_dfs.get(tf)
        if df is None or df.empty:
            return 0
        tc = _time_column_in_df(df)
        if not tc:
            return 0
        last_val = df[tc].iloc[-1]
        return parse_time_to_epoch(last_val)
    except Exception:
        return 0

def _epoch_to_str(epoch: int) -> str:
    try:
        return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "NA"

def _make_duplicate_key(tf: str, tag: str, candle_epoch: int) -> str:
    return f"{tf}_{tag}_{int(candle_epoch)}"

def is_duplicate_observation(tf: str, tag: str) -> bool:
    candle_epoch = _get_last_candle_epoch_for_tf(tf)
    key = _make_duplicate_key(tf, tag, candle_epoch)
    window = DUPLICATE_WINDOW_SECONDS.get(tf, 300)
    now = time.time()
    with signals_history_lock:
        ts = signals_history.get(key)
        if ts is None:
            return False
        if (now - ts) < window:
            return True
        try:
            del signals_history[key]
        except KeyError:
            pass
        return False

def mark_observation(tf: str, tag: str):
    candle_epoch = _get_last_candle_epoch_for_tf(tf)
    key = _make_duplicate_key(tf, tag, candle_epoch)
    with signals_history_lock:
        signals_history[key] = time.time()

# CSV parsing
def scan_csvs() -> List[str]:
    paths = []
    for t in terminal_folders:
        d = os.path.join(t, "MQL4", "Files")
        if not os.path.isdir(d):
            continue
        for pat in FILE_GLOBS:
            paths += glob.glob(os.path.join(d, pat))
    return paths

def pick_active_files_dir(paths: List[str]) -> Optional[str]:
    best_dir = None
    best_m = -1.0
    for p in paths:
        try:
            m = os.path.getmtime(p)
        except Exception:
            continue
        if m > best_m:
            best_m = m
            best_dir = os.path.dirname(p)
    return best_dir

def parse_csv(path):
    try:
        with open(path, "r", encoding="cp1250") as f:
            f.readline()
            tfm_line = f.readline()
            tfm = int(tfm_line.split(";")[1]) if ";" in tfm_line else None
        tf = TF_MAP.get(tfm)
    except Exception:
        return None, None
    try:
        df = pd.read_csv(path, delimiter=";", encoding="cp1250", skiprows=3)
        df.columns = [c.lower() for c in df.columns]
        if not {"open", "high", "low", "close"}.issubset(df.columns):
            return None, None
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna()
        return tf, df
    except Exception:
        return None, None

# detect_swings returns anonymized structural refs (upper_refs, lower_refs)
def detect_swings(df: pd.DataFrame, lookback: int) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """
    Generic peak/trough detection:
    - returns lists of (index, price) for local highs (upper refs) and local lows (lower refs)
    - neutral naming to avoid trading terminology
    """
    upper_refs: List[Tuple[int, float]] = []
    lower_refs: List[Tuple[int, float]] = []
    try:
        highs = df['high'].astype(float).tolist()
        lows = df['low'].astype(float).tolist()
    except Exception:
        return upper_refs, lower_refs
    n = len(df)
    if n < (lookback * 2 + 1):
        return upper_refs, lower_refs
    for i in range(lookback, n - lookback):
        try:
            is_upper = highs[i] > max(highs[i - lookback:i]) and highs[i] > max(highs[i + 1:i + lookback + 1])
            is_lower = lows[i] < min(lows[i - lookback:i]) and lows[i] < min(lows[i + 1:i + lookback + 1])
            if is_upper:
                upper_refs.append((i, float(highs[i])))
            if is_lower:
                lower_refs.append((i, float(lows[i])))
        except Exception:
            continue
    return upper_refs, lower_refs

def atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    try:
        if df is None or len(df) < period + 1:
            return None
        high = df['high']; low = df['low']; close = df['close']
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr_series = tr.rolling(window=period, min_periods=1).mean()
        return float(atr_series.iloc[-1])
    except Exception:
        return None

# compute anonymized metrics using neutral names
def compute_anonymized_metrics(st: TFState, df: pd.DataFrame):
    """
    Produce normalized/aggregated features:
    - magnitude: absolute difference between ref_upper and ref_lower
    - magnitude_pct: magnitude normalized by current price
    - atr and atr_pct (optional)
    - offset_pct: minimal offset normalized by price
    Note: No absolute price levels are returned or persisted.
    """
    try:
        if st.ref_upper is None or st.ref_lower is None or st.price is None:
            return None
        magnitude = abs(st.ref_upper - st.ref_lower)
        atr_val = atr(df)
        price = st.price or 0.0
        magnitude_pct = (magnitude / price) * 100.0 if price and price > 0 else None
        atr_pct = (atr_val / price) * 100.0 if atr_val and price and price > 0 else None
        offset = max((atr_val * 0.5) if atr_val is not None else 0.0, MINIMAL_OFFSET_USD)
        offset_pct = (offset / price) * 100.0 if price and price > 0 else None
        return {
            "magnitude": round(magnitude, 6),
            "magnitude_pct": round(magnitude_pct, 6) if magnitude_pct is not None else None,
            "atr": round(atr_val, 6) if atr_val is not None else None,
            "atr_pct": round(atr_pct, 6) if atr_pct is not None else None,
            "offset_pct": round(offset_pct, 6) if offset_pct is not None else None
        }
    except Exception:
        return None

# Learning & persistence (anonymized)
def load_learning_db():
    global learning_open, learning_stats
    learning_open = []
    learning_stats = {}
    if not os.path.isfile(LEARN_DB_PATH):
        return
    try:
        with open(LEARN_DB_PATH, "r", encoding="utf-8") as f:
            db = json.load(f)
        if not isinstance(db, dict):
            return
        open_list = db.get("open", [])
        stats_map = db.get("stats", {})
        if isinstance(open_list, list):
            learning_open[:] = [it for it in open_list if isinstance(it, dict)]
        if isinstance(stats_map, dict):
            learning_stats.clear()
            for k, v in stats_map.items():
                if isinstance(v, dict):
                    learning_stats[k] = v
    except Exception as e:
        print(f"[LEARN] ⚠️ load_learning_db failed: {e}")
        learning_open = []
        learning_stats = {}

def save_learning_db():
    try:
        db = {"open": learning_open[-CFG["LEARN_MAX_OPEN_SIGNALS"]:] if isinstance(learning_open, list) else [], "stats": learning_stats}
        tmp = f"{LEARN_DB_PATH}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
        os.replace(tmp, LEARN_DB_PATH)
    except Exception as e:
        print(f"[LEARN] ⚠️ save_learning_db failed: {e}")

def learn_add_observation(tf: str, tag: str, direction: str, metrics: Dict[str, Any]):
    if not CFG.get("LEARNING_ENABLED", True):
        return
    now = time.time()
    obs_id = f"{int(now)}_{tf}_{tag}_{uuid.uuid4().hex[:8]}"
    rec = {
        "id": obs_id,
        "created_ts": now,
        "created_str": get_timestamp(),
        "tf": tf,
        "tag": tag,
        "direction": direction,
        "metrics": metrics or {},
        "evaluated": False
    }
    learning_open.append(rec)
    if len(learning_open) > CFG["LEARN_MAX_OPEN_SIGNALS"]:
        learning_open[:] = learning_open[-CFG["LEARN_MAX_OPEN_SIGNALS"]:]
    save_learning_db()

def learn_eval_observations():
    if not CFG.get("LEARNING_ENABLED", True):
        return
    changed = False
    horizon_seconds = CFG.get("LEARN_HORIZON_MINUTES", 20) * 60
    threshold_usd = CFG.get("LEARN_HIT_MOVE_USD", 2.0)
    for rec in learning_open:
        if rec.get("evaluated"):
            continue
        tf = rec.get("tf")
        df = last_dfs.get(tf)
        if df is None or df.empty:
            continue
        time_col = _time_column_in_df(df)
        if time_col is None:
            continue
        try:
            times = pd.to_datetime(df[time_col], format=MT4_TIME_FORMAT, errors="coerce")
        except Exception:
            continue
        rec_ts = datetime.strptime(rec.get("created_str"), "%Y-%m-%d %H:%M:%S")
        start_idx = None
        for idx, t in enumerate(times):
            if pd.isna(t):
                continue
            if t >= rec_ts:
                start_idx = idx
                break
        if start_idx is None:
            continue
        end_time = rec_ts + timedelta(seconds=horizon_seconds)
        end_idx = None
        for idx in range(start_idx, len(times)):
            t = times.iloc[idx]
            if pd.isna(t):
                continue
            if t <= end_time:
                end_idx = idx
            else:
                break
        if end_idx is None or end_idx <= start_idx:
            continue
        try:
            window = df.iloc[start_idx:end_idx+1]
            highs = window['high'].astype(float)
            lows = window['low'].astype(float)
            start_price = float(window['close'].iloc[0])
            max_up = float(highs.max() - start_price)
            max_down = float(start_price - lows.min())
        except Exception:
            continue
        label = "NO_SIGNIFICANT_MOVE"
        if max_up >= threshold_usd and max_up > max_down:
            label = "MOVE_UP"
        elif max_down >= threshold_usd and max_down > max_up:
            label = "MOVE_DOWN"
        rec["evaluated"] = True
        rec["result_label"] = label
        key = f"{rec.get('tf')}|{rec.get('direction')}"
        st = learning_stats.get(key, {"total": 0, "MOVE_UP": 0, "MOVE_DOWN": 0, "NO_SIGNIFICANT_MOVE": 0})
        st["total"] = st.get("total", 0) + 1
        st[label] = st.get(label, 0) + 1
        learning_stats[key] = st
        changed = True
    if changed:
        save_learning_db()

# Observation emission
def emit_observation(obs_type: str, tf: str, direction: str, tag: str, metrics: Dict[str, Any], candle_epoch: Optional[int] = None):
    ts = get_timestamp()
    candle_epoch_val = int(candle_epoch) if candle_epoch is not None else _get_last_candle_epoch_for_tf(tf) or int(time.time())
    candle_time_str = _epoch_to_str(candle_epoch_val)
    header = f"[OBS {ts}] {obs_type} ({tf} {direction})"
    body_lines = [f"Tag: {tag}", f"Metrics: {json.dumps(metrics, ensure_ascii=False)}"]
    body_lines.append(f"Recorded at: {ts}")
    body_lines.append(f"Candle time id: {candle_time_str}")
    body = "\n".join(body_lines)
    print(f"\n{header}\n{body}", flush=True)
    rec = {
        "id": uuid.uuid4().hex,
        "timestamp": ts,
        "type": obs_type,
        "tf": tf,
        "direction": direction,
        "tag": tag,
        "metrics": metrics,
        "candle_epoch": int(candle_epoch_val),
        "candle_time": candle_time_str
    }
    try:
        with open(SIGNALS_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[LOG] Cannot write observation: {e}", flush=True)

# Update state using neutral refs
def update(tf: str, df: pd.DataFrame):
    st = states[tf]
    if tf in LTF_SET:
        if len(df) < 8:
            return
    else:
        if len(df) < 30:
            return
    try:
        st.price = round(float(df.close.iloc[-1]), 6)
    except Exception:
        st.price = None
    upper_refs, lower_refs = detect_swings(df, 2 if tf in LTF_SET else 3)
    # use last available structural refs if present
    if upper_refs:
        st.ref_upper = round(upper_refs[-1][1], 6)
    if lower_refs:
        st.ref_lower = round(lower_refs[-1][1], 6)
    if st.ref_upper is not None and st.ref_lower is not None:
        st.magnitude = abs(st.ref_upper - st.ref_lower)
        st.directional_bias = "UP" if st.ref_upper > st.ref_lower else "DOWN"
    else:
        st.magnitude = None
        st.directional_bias = None

def process_observations():
    global _last_learn_eval
    paths = scan_csvs()
    if not paths:
        return
    for p in paths:
        tf, df = parse_csv(p)
        if tf in states and df is not None:
            update(tf, df)
            last_dfs[tf] = df.copy()
    if time.time() - _last_learn_eval > 10:
        learn_eval_observations()
        _last_learn_eval = time.time()
    # HTF anonymized observation
    for tf in HTF_SET:
        st = states[tf]
        if not (st.magnitude and st.directional_bias and st.price):
            continue
        candle_epoch = _get_last_candle_epoch_for_tf(tf)
        if not candle_epoch or candle_epoch <= 0:
            continue
        tag = "HTF_STRUCTURE"
        if is_duplicate_observation(tf, tag):
            continue
        df = last_dfs.get(tf)
        metrics = compute_anonymized_metrics(st, df) if df is not None else None
        emit_observation("BASE_HTF", tf, st.directional_bias, tag, metrics, candle_epoch=candle_epoch)
        mark_observation(tf, tag)
        learn_add_observation(tf, tag, st.directional_bias, metrics)
    # LTF coarse observation
    for tf in LTF_SET:
        st = states[tf]
        if not (st.directional_bias and st.magnitude and st.price and st.ref_upper is not None and st.ref_lower is not None):
            continue
        df = last_dfs.get(tf)
        tag = "LTF_EVENT"
        if is_duplicate_observation(tf, tag):
            continue
        metrics = compute_anonymized_metrics(st, df) if df is not None else None
        emit_observation("OBSERVATION", tf, st.directional_bias, tag, metrics, candle_epoch=_get_last_candle_epoch_for_tf(tf))
        mark_observation(tf, tag)
        learn_add_observation(tf, tag, st.directional_bias, metrics)

# CLI / loop
def market_loop():
    while True:
        try:
            process_observations()
            time.sleep(0.25)
        except Exception as e:
            print("[MARKET LOOP] Exception:", e)
            time.sleep(0.5)

def _show_recent_observations(n=50):
    if os.path.isfile(SIGNALS_LOG_FILE):
        with open(SIGNALS_LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()[-n:]
        for l in lines:
            try:
                j = json.loads(l)
                print(json.dumps(j, ensure_ascii=False))
            except Exception:
                print(l.strip())
    else:
        print("No observations log.")

def chat_loop():
    threading.Thread(target=market_loop, daemon=True).start()
    print("Agent started (observer-only). Commands: observations | learning show | learning wip | eval | show df <tf> | exit")
    while True:
        try:
            raw = input("\nYOU: ")
            if raw is None:
                continue
            cmd = raw.strip().lower()
            if cmd == "":
                continue
            if cmd in ("observations", "obs", "signals"):
                _show_recent_observations()
            elif cmd in ("learning show", "learning stats"):
                print("[LEARNING STATS]")
                if not learning_stats:
                    print("No learning stats.")
                else:
                    for k, v in learning_stats.items():
                        print(f"{k} -> {v}")
            elif cmd in ("learning wip", "learning wip list"):
                pending = [x for x in learning_open if not x.get("evaluated")]
                print(f"Pending anonymized observations: {len(pending)}")
                for rec in pending[-50:]:
                    print(json.dumps({
                        "id": rec.get("id"),
                        "created_str": rec.get("created_str"),
                        "tf": rec.get("tf"),
                        "tag": rec.get("tag"),
                        "direction": rec.get("direction"),
                        "metrics": rec.get("metrics"),
                        "evaluated": rec.get("evaluated"),
                        "result_label": rec.get("result_label", None)
                    }, ensure_ascii=False))
            elif cmd == "eval":
                print("[EVAL] Running lightweight evaluation pass...")
                learn_eval_observations()
                print("Done.")
            elif cmd.startswith("show df"):
                parts = cmd.split()
                if len(parts) < 3:
                    print("Usage: show df <M1|M5|M15|H1|H4>")
                    continue
                tf = parts[2].upper()
                df = last_dfs.get(tf)
                if df is None or df.empty:
                    print(f"No DF for {tf}")
                    continue
                try:
                    tc = _time_column_in_df(df)
                    last_epoch = _get_last_candle_epoch_for_tf(tf)
                    print(f"--- last {tf} (tail 5) ---")
                    print(df.tail(5).to_string(index=False))
                    print(f"Time col: {tc} last_epoch={last_epoch} ({_epoch_to_str(last_epoch)})")
                except Exception as e:
                    print("Error showing df:", e)
            elif cmd in ("exit", "quit"):
                print("Bye"); break
            else:
                print("Unknown command. Examples: observations | learning show | learning wip | eval | show df M1 | exit")
        except (EOFError, KeyboardInterrupt):
            print("\nShutting down.")
            break

if __name__ == "__main__":
    load_learning_db()
    chat_loop()