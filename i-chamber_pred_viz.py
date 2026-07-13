from scipy.optimize import curve_fit, brentq
from sklearn.metrics import mean_squared_error
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import streamlit as st
from streamlit_sortables import sort_items
from scipy.interpolate import interp1d
import re
import os
import concurrent.futures


# =========================================================
# GLOBAL CONFIG
# =========================================================
RUL_HORIZON = 180 # days; "Safe" (never reached) and anything beyond this saturate here.
MIN_MSE_FLOOR = 5e-5
# =========================================================
# SYNTHETIC FAULT GENERATORS
# =========================================================

def add_linear_ramp(arr, start_idx, max_offset):
    """Adds a gradual, linear degradation starting at start_idx."""
    res = arr.copy()
    tail_len = len(res) - start_idx
    if tail_len > 0:
        res[start_idx:] += np.linspace(0, max_offset, tail_len)
    return res

def add_exponential_curve(arr, start_idx, severity_factor):
    """Adds a compounding exponential curve starting at start_idx."""
    res = arr.copy()
    tail_len = len(res) - start_idx
    if tail_len > 0:
        # e^x curve scaled so the final point hits severity_factor
        curve = np.exp(np.linspace(0, 3, tail_len)) - 1 
        curve = (curve / np.max(curve)) * severity_factor
        res[start_idx:] += curve
    return res

def add_step_change(arr, start_idx, offset):
    """Instantly offsets the data after start_idx (e.g., sudden mechanical shift)."""
    res = arr.copy()
    if start_idx < len(res):
        res[start_idx:] += offset
    return res

def add_logarithmic_curve(arr, start_idx, target_offset):
    """Adds a logarithmic curve that rapidly rises and then heavily flattens."""
    res = arr.copy()
    tail_len = len(res) - start_idx
    if tail_len > 0:
        # np.log1p(linspace(0, 20)) ensures a strong initial spike that flattens out
        curve = np.log1p(np.linspace(0, 20, tail_len))
        # Scale the curve so the final point hits the exact target offset
        curve = (curve / np.max(curve)) * target_offset
        res[start_idx:] += curve
    return res

def stretch_or_squeeze_time(arr, start_idx, factor):
    """
    Time-warps the degradation tail. 
    factor < 1.0 = faster failure (squeezed). 
    factor > 1.0 = slower failure (stretched).
    """
    if start_idx >= len(arr) - 5:
        return arr.copy()
        
    head = arr[:start_idx]
    tail = arr[start_idx:]
    
    orig_x = np.linspace(0, 1, len(tail))
    new_len = int(len(tail) * factor)
    
    # Interpolate the tail to a new length
    f = interp1d(orig_x, tail, kind='linear', fill_value="extrapolate")
    new_x = np.linspace(0, 1, new_len)
    warped_tail = f(new_x)
    
    # Reassemble. If faster, pad end with NaNs so dataframe length matches.
    # If slower, clip the end.
    if factor < 1.0:
        pad_len = len(tail) - new_len
        warped_tail = np.concatenate([warped_tail, np.full(pad_len, np.nan)])
    else:
        warped_tail = warped_tail[:len(tail)]
        
    return np.concatenate([head, warped_tail])

def add_baseline_drift(arr, drift_max):
    """Adds a linear drift across the ENTIRE lifespan of the asset."""
    drift = np.linspace(0, drift_max, len(arr))
    return arr + drift

def inject_tail_noise(arr, start_idx, noise_multiplier=2.0):
    """Increases the variance of the signal near failure."""
    res = arr.copy()
    tail_len = len(res) - start_idx
    if tail_len > 0:
        # Calculate natural standard deviation of the healthy portion
        healthy_std = np.nanstd(arr[:start_idx]) if start_idx > 10 else np.nanstd(arr)
        noise = np.random.normal(0, healthy_std * noise_multiplier, tail_len)
        res[start_idx:] += noise
    return res

# =========================================================
# BREAK-BENCHMARK HELPERS (labels, grouping, propagation)
# =========================================================

def parse_base(col):
    """Extracts the base channel from a synthetic column name.
    'INJECT_Linear_1 (Base: 32)' -> '32'.  Falls back to the col itself."""
    m = re.search(r"\(Base:\s*([^)]+)\)", str(col))
    return m.group(1).strip() if m else str(col)


def init_break_label_store():
    """Ensures the dataset-scoped label store exists.
    Value semantics per channel:
        float  -> labeled break day (in elapsed_days space)
        None   -> explicitly 'no obvious break' (treated as a control, not a miss)
        absent -> not yet labeled
    """
    if 'break_labels' not in st.session_state:
        st.session_state['break_labels'] = {"original": {}, "synthetic": {}}
    return st.session_state['break_labels']


def get_unhealthy_channels(active_df, is_synthetic, threshold,
                           outlier_factor, outlier_window):
    """Original dataset: a channel is unhealthy if its smoothed signal crosses
    `threshold` anywhere. Synthetic dataset: every channel is unhealthy by
    construction, so we return all of them."""
    channels = get_available_channels(active_df)
    if is_synthetic:
        return channels

    unhealthy = []
    for ch in channels:
        smooth, _raw, _t = load_my_sensor_data(
            active_df, col=ch, outlier_factor=outlier_factor,
            outlier_window=outlier_window
        )
        arr = np.asarray(smooth, dtype=float)
        if arr.size and np.nanmax(arr) >= threshold:
            unhealthy.append(ch)
    return unhealthy


def order_candidates_inject_first(candidates):
    """For the synthetic set we must label the INJECT_/base channels BEFORE the
    MUTATE_ ones, so propagation has a source break to copy. Injected/base
    channels first, mutations last."""
    base_like = [c for c in candidates if not c.startswith("MUTATE_")]
    mutated   = [c for c in candidates if c.startswith("MUTATE_")]
    return base_like + mutated


def propagate_break_to_mutations(labeled_col, break_day, scope, candidates):
    """When a source (INJECT_/base) channel gets a break, copy that
    break day to every MUTATE_ channel derived from the SAME base."""
    labels = st.session_state['break_labels'][scope]
    base = parse_base(labeled_col)
    
    for c in candidates:
        if c.startswith("MUTATE_") and parse_base(c) == base and c not in labels:
            labels[c] = break_day

def autoseed_injected_breaks(active_df, scope, candidates):
    """Auto-fills INJECT_ channel breaks from recorded ground truth.
    Mutations are skipped so they can inherit the user's manual base labels."""
    truth = st.session_state.get('synthetic_truth', {})
    if not truth:
        return

    labels = st.session_state['break_labels'][scope]
    origin = active_df.index.min().normalize()

    for col in candidates:
        if col in labels:
            continue
        meta = truth.get(col)
        if not meta:
            continue
            
        # BUG FIX: Ensure we ONLY auto-seed injected faults. 
        # Mutations will be handled by the propagation function.
        if meta.get("kind") != "inject":
            continue
            
        ts = meta.get("break_timestamp")
        if pd.isna(ts):
            continue
            
        labels[col] = float((ts.normalize() - origin).days)

def rolling_iqr_filter(data, window=20, factor=1.5, center=True, keep_nans=True):
    """
    Filter out outliers using a Local Windowed Interquartile Range (IQR) method.
    Automatically handles both Pandas Series and DataFrames.
    """
    def _filter_series(series):
        s = pd.to_numeric(series, errors='coerce')
        original_nans = s.isna()

        Q1 = s.rolling(window=window, center=center, min_periods=1).quantile(0.25)
        Q3 = s.rolling(window=window, center=center, min_periods=1).quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - (factor * IQR)
        upper_bound = Q3 + (factor * IQR)

        mask = (s >= lower_bound) & (s <= upper_bound)
        s_filtered = s.where(mask, np.nan)
        s_filtered = s_filtered.interpolate(method='linear', limit_direction='both')

        if keep_nans:
            # .loc is safer than chained boolean indexing (avoids SettingWithCopyWarning)
            s_filtered.loc[original_nans] = np.nan

        return s_filtered

    if isinstance(data, pd.Series):
        return _filter_series(data)
    elif isinstance(data, pd.DataFrame):
        return data.apply(_filter_series)
    else:
        raise TypeError(f"Expected pd.Series or pd.DataFrame, got {type(data)}")


# ---------------------------------------------------------
# 1. The Standardized Mathematical Models
# ---------------------------------------------------------
def linear_model(t, m, c):
    return m * t + c

def logarithmic_model(t, a, b, d):
    # np.log1p(x) calculates log(1 + x), ensuring safe behavior at t=0
    return a * np.log1p(np.clip(b * t, 0, np.inf)) + d

def shifted_exponential_model(t, a, b, t0, d):
    return a * np.exp(np.clip(b * (t - t0), -50, 50)) + d

def softplus_model(t, a, b, t0, d):
    return a * np.logaddexp(0, b * (t - t0)) + d

def gompertz_model(t, a, b, c, d):
    return a * np.exp(-b * np.exp(-c * t)) + d

def arctan_model(t, L, k, t0, d):
    return L * (np.arctan(k * (t - t0)) / np.pi + 0.5) + d

def linear_sine_model(t, a, b, c, m, d):
    # a: amplitude, b: frequency, c: phase shift
    # m: linear slope, d: vertical offset (intercept)
    return a * np.sin(b * t + c) + (m * t + d)

# Ordered by requested default priority (Highest to Lowest)
AVAILABLE_MODELS = [
    'Linear', 
    'Logarithmic', 
    'Trending Sine',
    'Softplus', 
    'Shifted Exponential',  
    'Gompertz',
    
]


# ---------------------------------------------------------
# 1b. CENTRALIZED MODEL CONFIG (single source of truth)
# ---------------------------------------------------------
def build_models_config(y_min, y_max, y_range):
    d_lo = y_min - 0.2
    d_hi = y_max + 0.2  # use y_max consistently as the upper bound for offset d
    return {
        'Linear': {
            'func': linear_model,
            'p0': [y_range, y_min],
            'bounds': ([-np.inf, d_lo], [np.inf, d_hi]),
        },
        'Logarithmic': {
            'func': logarithmic_model,
            'p0': [y_range * 0.5, 10.0, y_min],
            'bounds': ([1e-5, 1e-5, d_lo], 
                       [y_range * 10.0, 500.0, d_hi]),
        },
        'Shifted Exponential': {
            'func': shifted_exponential_model,
            'p0': [y_range * 0.1, 5.0, 0.5, y_min],
            'bounds': ([1e-5, 0.01, 0.0, d_lo],
                       [y_range * 5.0, 50.0, 1.0, d_hi]),
        },
        'Softplus': {
            'func': softplus_model,
            'p0': [y_range * 0.5, 10.0, 0.5, y_min],
            'bounds': ([1e-3, 1e-3, 0.0, d_lo],
                       [y_range * 10.0, 500.0, 1.0, d_hi]),
        },
        'Gompertz': {
            'func': gompertz_model,
            'p0': [y_range * 1.1, 1.0, 0.1, y_min],
            'bounds': ([y_range * 0.8, 0.01, 1e-4, d_lo],
                       [max(2.0, y_range * 2.2), 100.0, 50.0, d_hi]),
        },
        'Trending Sine': {
            'func': linear_sine_model,
            # p0: [Amp, Freq (guess 2 periods), Phase, Slope, Offset]
            # We guess the amplitude is small (10% of range) and the slope handles the bulk of the rise.
            'p0': [y_range * 0.1, 2 * 2 * np.pi, 0.0, y_range, y_min],
            'bounds': (
                # Lower bounds: [Amp, Freq (min 1 period), Phase, Slope, Offset]
                [1e-5, 2 * np.pi, -np.pi, -np.inf, d_lo],           
                # Upper bounds: [Amp, Freq (max 5 periods), Phase, Slope, Offset]
                [y_range * 2.0, 10 * np.pi, np.pi, np.inf, d_hi]   
            ),
        },
    }


# ---------------------------------------------------------
# 2. Master Fitting Function (AIC & Priority Router)
# ---------------------------------------------------------
def evaluate_all_models(time_data, sensor_data, priority_ranking, eval_window=None,
                        plot=False, verbose=True, Title_addon="",
                        save_name=None, warm_start=None, maxfev=10000,min_mse_floor=5e-5):
    
    time_arr = np.asarray(time_data)
    sensor_arr = np.asarray(sensor_data)

    t_max = np.max(time_arr)
    time_norm = time_arr / t_max if t_max > 0 else time_arr

    valid_mask = ~np.isnan(sensor_arr)
    if valid_mask.sum() < 10:
        return {}, {}

    t_fit = time_norm[valid_mask]
    y_fit = sensor_arr[valid_mask]

    y_min = float(np.min(y_fit))
    y_max = float(np.max(y_fit))
    y_range = y_max - y_min

    models_config = build_models_config(y_min, y_max, y_range)

    results = {}

    for name, config in models_config.items():
        try:
            p0 = config['p0']
            k = len(p0)  # Number of parameters used for AIC penalty

            if warm_start is not None and name in warm_start and warm_start[name] is not None:
                try:
                    p0 = np.clip(warm_start[name], config['bounds'][0], config['bounds'][1])
                except Exception:
                    p0 = config['p0']

            params, _ = curve_fit(
                config['func'], t_fit, y_fit, p0=p0,
                bounds=config['bounds'], method='trf', maxfev=maxfev
            )
            preds = config['func'](t_fit, *params)

            # Evaluate on the tail window if specified
            if eval_window is not None and eval_window < len(y_fit):
                y_eval = y_fit[-eval_window:]
                preds_eval = preds[-eval_window:]
                n = len(y_eval)
            else:
                y_eval = y_fit
                preds_eval = preds
                n = len(y_fit)

            mse = mean_squared_error(y_eval, preds_eval)
            
            # Tail-Weighted Pseudo-AIC Calculation
            # Added 1e-10 to prevent math domain error if MSE is exactly 0
            aic = n * np.log(max(mse, min_mse_floor)) + 2 * k

            results[name] = {'params': params, 'mse': mse, 'aic': aic, 'func': config['func']}
            
        except Exception as e:
            results[name] = {'params': None, 'mse': float('inf'), 'aic': float('inf'),
                             'error': str(e), 'func': config['func']}

    # Filter out models that completely failed to converge
    valid_results = {n: d for n, d in results.items() if d['aic'] != float('inf')}
    if not valid_results:
        return {}, results

    # --- Step 1 & 2: Find the Best AIC ---
    best_aic = min(d['aic'] for d in valid_results.values())

    # --- Step 3 & 4: Filter Competitive Models (Delta AIC <= 2.0) ---
    competitive_models = {n: d for n, d in valid_results.items() if (d['aic'] - best_aic) <= 2.0}

    # --- Step 5: Apply User Priority Ranking to break ties ---
    # priority_ranking dict is structured as {'Linear': 1, 'Logarithmic': 2, ...}
    # Lower number = higher priority
    top_models_sorted = dict(sorted(competitive_models.items(),
                                    key=lambda item: priority_ranking.get(item[0], 99)))

    # Sort the full leaderboard for the UI display (purely by AIC score)
    full_leaderboard = dict(sorted(results.items(),
                                   key=lambda item: item[1].get('aic', float('inf'))))

    return top_models_sorted, full_leaderboard



# Put this at the top level of your script, outside any UI functions
def process_single_channel_worker(channel, active_df, outlier_factor, outlier_window, 
                                  target_thresh, break_algo, break_window, break_step, 
                                  break_sustained, z_factor, z_sustained, min_mse_floor, 
                                  req_break, step_days, window_size, eval_window, 
                                  override_model, manual_model, use_dynamic_variance, 
                                  variance_window, priority_dict, ema_span, origin_date,
                                  fleet_mean=0.0, fleet_std=1.0):
    """
    PURE FUNCTION: No Streamlit calls allowed here. 
    This runs entirely on a background CPU core.
    """
    import numpy as np
    import pandas as pd
    
    # 1. Load Data
    sensor_smooth, sensor_raw, time_arr = load_my_sensor_data(
        active_df, col=channel, outlier_factor=outlier_factor, outlier_window=outlier_window
    )
    
    time_arr_np = np.asarray(time_arr, dtype=float)
    smooth_np = np.asarray(sensor_smooth, dtype=float)
    raw_np = np.asarray(sensor_raw, dtype=float)  

    crossing_indices = np.where(smooth_np >= target_thresh)[0]
    actual_crossing_day = time_arr_np[crossing_indices[0]] if len(crossing_indices) > 0 else np.nan

    # 2. Structural Break Detection
    if break_algo == "Fleet Z-Score":
        break_idx, _break_time, eval_start_idx = detect_zscore_break(
            time_arr_np, smooth_np, fleet_mean, fleet_std, z_factor, z_sustained
        )
    else:
        break_idx, _break_time, eval_start_idx = detect_structural_break(
            time_arr_np, smooth_np, window=break_window, step=break_step, 
            sustained_wins=break_sustained, min_mse_floor=min_mse_floor
        )

    start_idx = 50
    if req_break and break_idx is not None:
        start_idx = max(50, eval_start_idx)
    elif req_break and break_idx is None:
        return [] # Return empty list if skipping

    max_idx = len(time_arr)
    timesteps = list(range(start_idx, max_idx, step_days))
    if len(timesteps) == 0:
        return []

    alpha = 2.0 / (ema_span + 1)  
    ema_n = ema_u = ema_l = None
    warm_start_cache = None

    def _ema_update(prev, new_raw):
        if np.isnan(new_raw):
            if prev is None:
                return prev, np.nan          
            bridged = alpha * float(RUL_HORIZON) + (1 - alpha) * prev
            return bridged, np.nan            
        if prev is None:
            return float(new_raw), float(new_raw)  
        smoothed = alpha * float(new_raw) + (1 - alpha) * prev
        return smoothed, smoothed

    channel_results = []
    
    # 3. The Timestep Loop (Running silently in background)
    for step_idx, current_cutoff in enumerate(timesteps):
        fit_start_idx = max(0, current_cutoff - window_size)
        
        hist_time = time_arr_np[fit_start_idx:current_cutoff]
        hist_smooth = smooth_np[fit_start_idx:current_cutoff]
        hist_raw = raw_np[fit_start_idx:current_cutoff] 
        current_day = hist_time[-1]

        eval_win = min(eval_window, len(hist_time))

        if override_model:
            best_model = manual_model
            mse_log_str = f"{manual_model}: (override - competition skipped)"
            raw_n, raw_u, raw_l = calculate_rul_headless(
                hist_time, hist_smooth, hist_raw, best_model, target_thresh,
                use_dynamic_variance=use_dynamic_variance, precomputed_params=None,
                variance_window=variance_window
            )
        else:
            top_models, all_models = evaluate_all_models(
                hist_time, hist_smooth, priority_ranking=priority_dict,
                eval_window=eval_win, plot=False, verbose=False,
                warm_start=warm_start_cache, min_mse_floor=min_mse_floor
            )
            
            aic_log_str = "All models failed"
            if all_models:
                aic_log_str = " | ".join(
                    [f"{k}: {v['aic']:.2f}" for k, v in all_models.items() 
                     if v.get('aic', float('inf')) != float('inf')]
                )

            if not top_models:
                best_model = "Linear"
                winner_params = None
            else:
                best_model = list(top_models.keys())[0]
                winner_params = top_models[best_model].get('params')

            warm_start_cache = {k: v.get('params') for k, v in all_models.items() 
                                if v.get('params') is not None}

            raw_n, raw_u, raw_l = calculate_rul_headless(
                hist_time, hist_smooth, hist_raw, best_model, target_thresh,
                use_dynamic_variance=use_dynamic_variance, precomputed_params=winner_params,
                variance_window=variance_window
            )

        ema_n, store_n = _ema_update(ema_n, raw_n)
        ema_u, store_u = _ema_update(ema_u, raw_u)
        ema_l, store_l = _ema_update(ema_l, raw_l)

        actual_rul = actual_crossing_day - current_day if not np.isnan(actual_crossing_day) else np.nan
        if not np.isnan(actual_rul) and actual_rul < 0:
            continue  
        
        current_date_str = (origin_date + pd.Timedelta(days=current_day)).strftime('%Y-%m-%d')
        
        channel_results.append({
            'Channel': channel,
            'Evaluation_Day': current_day,
            'Evaluation_Date': current_date_str,
            'Model_Used': f"{best_model} (Override)" if override_model else best_model,
            'Actual_RUL': actual_rul,
            'Nominal_RUL': store_n,     
            'Upper_Risk_RUL': store_u,
            'Lower_Risk_RUL': store_l,
            'All_Models_AIC': aic_log_str if not override_model else mse_log_str
        })
        
    return channel_results


# ---------------------------------------------------------
# 2b. SHARED RUL SOLVER (used by plotting + headless engine)
# ---------------------------------------------------------
# OPT-C: brentq instead of fsolve. brentq is derivative-free, bracketed, and won't
# wander. The ceiling check tells us whether a root can exist before bracketing.
SENTINEL_ALREADY_REACHED = -999.0

def build_dynamic_std_fn(std_slope, std_intercept):
    def get_dynamic_std(t_norm_val):
        return np.maximum(0.0, std_slope * t_norm_val + std_intercept)
    return get_dynamic_std

def solve_rul_root(func, params, target_val, t_max, get_dynamic_std,
                   sigma_factor, mode='nominal', t_hi=200.0):
    """
    Returns:
      'Safe'                      -> ceiling never reaches target
      SENTINEL_ALREADY_REACHED    -> already past target at t~0
      float (raw time)            -> solved crossing time
      None                        -> numerically unresolved
    """
    t_infinite = 10000.0
    ceiling_base = func(t_infinite, *params)
    if mode == 'upper':
        ceiling_val = ceiling_base + get_dynamic_std(t_infinite) * sigma_factor
    elif mode == 'lower':
        ceiling_val = ceiling_base - get_dynamic_std(t_infinite) * sigma_factor
    else:
        ceiling_val = ceiling_base

    if ceiling_val < target_val:
        return 'Safe'

    def target_equation(t_guess):
        base_val = func(t_guess, *params)
        if mode == 'upper':
            return (base_val + get_dynamic_std(t_guess) * sigma_factor) - target_val
        elif mode == 'lower':
            return (base_val - get_dynamic_std(t_guess) * sigma_factor) - target_val
        return base_val - target_val

    lo = 1e-9
    try:
        f_lo = target_equation(lo)
        if f_lo >= 0:
            return SENTINEL_ALREADY_REACHED  # already at/over target at the start

        # Expand the upper bracket until we get a sign change (monotone ascent).
        hi = t_hi
        f_hi = target_equation(hi)
        expand = 0
        while f_hi < 0 and expand < 8:
            hi *= 2.0
            f_hi = target_equation(hi)
            expand += 1

        if f_hi < 0:
            return None  # could not bracket a root in a sane range

        t_solution = brentq(target_equation, lo, hi, xtol=1e-9, maxiter=200)
        return t_solution * t_max
    except Exception:
        return None


def detect_structural_break(time_arr, sensor_arr, window=60, step=7, sustained_wins=2,
                            maxfev=2000, eval_lookback=None, min_mse_floor=5e-5):
    """
    Detects a structural break via sustained Exponential-vs-Linear model competition.

    Marker placement:
        The break is reported at the DETECTION point `i` (the window that confirmed
        the sustained streak), NOT the rewound start of the first window. The old
        `i - (sustained_wins-1)*step` reference labeled the break up to ~window-length
        too early, which reviewers saw as the marker sitting visibly in front of the
        actual bend. Reporting `i` is causally honest: we only claim a break once the
        detector had enough evidence (sustained_wins consecutive wins).

    Evaluation look-back:
        For the simulation we usually WANT to start evaluating slightly before the
        confirmed point, to capture the run-up into the break. `eval_lookback`
        (in array steps, i.e. days here) is subtracted from the detection index to
        produce `eval_start_idx`. If None, it defaults to (sustained_wins-1)*step,
        i.e. roughly the span of the first window in the winning streak.

    Returns
    -------
    trigger_idx : int or None
        Confirmed detection index (clamped to >= 0). Use this for the GRAPH MARKER.
    break_time : float or None
        time_arr at trigger_idx (the marker's x position).
    eval_start_idx : int or None
        Suggested index to BEGIN EVALUATION (trigger_idx - eval_lookback, clamped).
        Lets the simulation look back into the run-up while the marker stays at `i`.
    """
    time_arr = np.asarray(time_arr, dtype=float)
    sensor_arr = np.asarray(sensor_arr, dtype=float)

    if eval_lookback is None:
        eval_lookback = (sustained_wins - 1) * step

    if len(sensor_arr) < window:
        return None, None, None

    # Smooth with an EWMA so the model competition isn't dominated by point noise.
    smooth_sensor = pd.Series(sensor_arr).ewm(span=5, adjust=False, ignore_na=True).mean().values

    consecutive_exp_wins = 0
    trigger_idx = None

    for i in range(0, len(smooth_sensor) - window + 1, step):
        t_win_raw = time_arr[i: i + window]
        y_win_raw = smooth_sensor[i: i + window]

        valid = ~np.isnan(y_win_raw)
        if valid.sum() < 10:
            consecutive_exp_wins = 0
            continue

        t_win = t_win_raw[valid]
        y_win = y_win_raw[valid]

        # Local normalization so fits are well-conditioned per window.
        t_min, t_max = t_win[0], t_win[-1]
        t_norm = (t_win - t_min) / (t_max - t_min) if t_max > t_min else t_win - t_min

        y_min, y_max = np.min(y_win), np.max(y_win)
        y_range = y_max - y_min + 1e-5
        n = len(y_win)

        # --- Fit Linear Model ---
        
        try:
            popt_lin, _ = curve_fit(linear_model, t_norm, y_win,
                                    p0=[y_range, y_min], maxfev=maxfev)
            preds_lin = linear_model(t_norm, *popt_lin)
            mse_lin = mean_squared_error(y_win, preds_lin)
            aic_lin = n * np.log(max(mse_lin, min_mse_floor)) + 2 * 2
        except Exception:  # no bare except
            aic_lin = float('inf')

        # --- Fit Exponential Model ---
        try:
            bounds_exp = ([1e-5, 0.01, 0.0, y_min - 0.2],
                          [y_range * 5.0, 50.0, 1.0, y_max + 0.2])
            p0_exp = [y_range * 0.1, 5.0, 0.5, y_min]
            popt_exp, _ = curve_fit(shifted_exponential_model, t_norm, y_win,
                                    p0=p0_exp, bounds=bounds_exp, maxfev=maxfev)
            preds_exp = shifted_exponential_model(t_norm, *popt_exp)
            mse_exp = mean_squared_error(y_win, preds_exp)
            aic_exp = n * np.log(max(mse_exp, min_mse_floor)) + 2 * 4
        except Exception:  # no bare except
            aic_exp = float('inf')

        # --- Evaluate Winner ---
        if aic_exp < aic_lin - 2.0:
            consecutive_exp_wins += 1
            if consecutive_exp_wins >= sustained_wins:
                # Report the DETECTION point (this confirming window), clamped.
                trigger_idx = max(0, i)
                break
        else:
            consecutive_exp_wins = 0

    if trigger_idx is None:
        return None, None, None

    # Evaluation may begin a bit earlier than the marker to capture the run-up.
    eval_start_idx = max(0, trigger_idx - eval_lookback)
    return trigger_idx, time_arr[trigger_idx], eval_start_idx


@st.cache_data
def compute_fleet_baseline(_base_df, outlier_factor, outlier_window):
    """
    Computes the global fleet mean and standard deviation from the ORIGINAL data.
    The underscore prevents Streamlit from hashing the massive DataFrame.
    """
    channels = get_available_channels(_base_df)
    all_valid_data = []
    
    for ch in channels:
        sensor_smooth, _, _ = load_my_sensor_data(
            _base_df, col=ch, outlier_factor=outlier_factor, outlier_window=outlier_window
        )
        valid_vals = sensor_smooth.dropna().values
        if len(valid_vals) > 0:
            all_valid_data.append(valid_vals)
            
    if not all_valid_data:
        return 0.0, 1.0
        
    concat_vals = np.concatenate(all_valid_data)
    fleet_mean = float(np.mean(concat_vals))
    fleet_std = float(np.std(concat_vals))
    
    # Prevent divide-by-zero or zero-variance issues
    if fleet_std < 1e-5:
        fleet_std = 1.0
        
    return fleet_mean, fleet_std


def detect_zscore_break(time_arr, sensor_arr, fleet_mean, fleet_std, z_factor=3.0, sustained_wins=3, eval_lookback=7):
    """
    Detects a structural break based on fleet-wide standard deviation.
    """
    time_arr = np.asarray(time_arr, dtype=float)
    sensor_arr = np.asarray(sensor_arr, dtype=float)
    
    if len(sensor_arr) < sustained_wins:
        return None, None, None

    # Smooth the data slightly to avoid single-point noise spikes triggering the Z-score
    smooth_sensor = pd.Series(sensor_arr).ewm(span=5, adjust=False, ignore_na=True).mean().values
    
    # Define the anomaly threshold
    upper_threshold = fleet_mean + (z_factor * fleet_std)
    
    consecutive_anomalies = 0
    trigger_idx = None
    
    for i, val in enumerate(smooth_sensor):
        if pd.isna(val):
            consecutive_anomalies = 0
            continue
            
        # We assume the break trends upwards. If it can drop, use abs(val - fleet_mean)
        if val > upper_threshold:
            consecutive_anomalies += 1
            if consecutive_anomalies >= sustained_wins:
                trigger_idx = i
                break
        else:
            consecutive_anomalies = 0
            
    if trigger_idx is None:
        return None, None, None
        
    eval_start_idx = max(0, trigger_idx - eval_lookback)
    return trigger_idx, time_arr[trigger_idx], eval_start_idx


def add_baseline_drift(arr, drift_max):
    """Adds a linear drift across the ENTIRE lifespan of the asset."""
    drift = np.linspace(0, drift_max, len(arr))
    return arr + drift

def inject_tail_noise(arr, start_idx, noise_multiplier=2.0):
    """Increases the variance of the signal near failure."""
    res = arr.copy()
    tail_len = len(res) - start_idx
    if tail_len > 0:
        # Calculate natural standard deviation of the healthy portion
        healthy_std = np.nanstd(arr[:start_idx]) if start_idx > 10 else np.nanstd(arr)
        noise = np.random.normal(0, healthy_std * noise_multiplier, tail_len)
        res[start_idx:] += noise
    return res

# ---------------------------------------------------------
# 4. Dynamic/Static Variance & RUL Plotting
# ---------------------------------------------------------
def fit_and_plotly_model(time_raw, sensor_smooth, sensor_raw, model_choice, thresholds=None,
                         input_time_unit="Hours", target_time_unit="Days", warm_start_params=None,
                         title_addon="", sigma_factor=1.645, save_name=None, max_rul_display=RUL_HORIZON,
                         use_dynamic_variance=True, break_time_raw=None, precomputed_params=None,
                         variance_window=20):
    """
    OPT-A: Accepts `precomputed_params` so the deep-dive page can pass the winner's
    already-fitted params from evaluate_all_models and skip a redundant curve_fit.
    ROBUSTNESS: every return path now returns a 3-tuple (fig, fitted_series, rul_df).
    """
    if isinstance(sensor_smooth, pd.Series):
        orig_index = sensor_smooth.index
    elif isinstance(time_raw, pd.Series):
        orig_index = time_raw.index
    else:
        orig_index = range(len(sensor_smooth))

    time_arr = np.asarray(time_raw, dtype=float)
    sensor_arr = np.asarray(sensor_smooth, dtype=float)
    sensor_raw_arr = np.asarray(sensor_raw, dtype=float)  # OPT-F: hoist conversion

    raw_current_time = np.max(time_arr)
    t_max = raw_current_time if raw_current_time > 0 else 1.0
    time_norm = time_arr / t_max

    to_hours = {"Seconds": 1 / 3600, "Minutes": 1 / 60, "Hours": 1.0, "Days": 24.0, "2H": 2.0, "8H": 8.0}
    conversion_factor = to_hours[input_time_unit] / to_hours[target_time_unit]

    valid_mask = ~np.isnan(sensor_arr)
    t_fit = time_norm[valid_mask]
    y_fit = sensor_arr[valid_mask]

    if len(y_fit) == 0:
        return go.Figure(), pd.Series(dtype=float), pd.DataFrame()

    y_min = float(np.min(y_fit))
    y_max = float(np.max(y_fit))
    y_range = y_max - y_min

    config = build_models_config(y_min, y_max, y_range)[model_choice]

    # OPT-A: reuse precomputed params when available, else fit once.
    if precomputed_params is not None:
        params = precomputed_params
    else:
        p0 = config['p0']
        if warm_start_params is not None:
            try:
                p0 = np.clip(warm_start_params, config['bounds'][0], config['bounds'][1])
            except Exception:
                p0 = config['p0']
        try:
            params, _ = curve_fit(config['func'], t_fit, y_fit, p0=p0,
                                  bounds=config['bounds'], method='trf', maxfev=10000)
        except Exception as e:
            print(f"Error: {model_choice} model failed to converge. {e}")
            # ROBUSTNESS: consistent 3-tuple on failure.
            return go.Figure(), pd.Series(np.nan, index=orig_index), pd.DataFrame()

    fitted_series = pd.Series(config['func'](time_norm, *params), index=orig_index,
                              name=f"{model_choice}_Fit")
    residuals_series = pd.Series(sensor_raw_arr - fitted_series.values, index=orig_index,
                                 name=f"{model_choice}_Residuals")

    # ---------------------------------------------------------
    # VARIANCE CALCULATION
    # ---------------------------------------------------------
    rolling_std = residuals_series.rolling(window=variance_window, min_periods=1).std()
    rolling_std = rolling_std.bfill().fillna(0)

    valid_std_mask = ~np.isnan(rolling_std)
    if use_dynamic_variance and valid_std_mask.sum() > 1:
        std_slope, std_intercept = np.polyfit(time_norm[valid_std_mask],
                                              rolling_std[valid_std_mask], 1)
    else:
        std_slope, std_intercept = 0.0, rolling_std.iloc[-1]

    get_dynamic_std = build_dynamic_std_fn(std_slope, std_intercept)

    current_margin = get_dynamic_std(1.0) * sigma_factor
    cdf = 0.5 * (1.0 + math.erf(sigma_factor / math.sqrt(2)))
    risk_pct_upper = (1.0 - cdf) * 100
    risk_pct_lower = cdf * 100

    func = config['func']

    # ---------------------------------------------------------
    # RECORD GENERATION
    # ---------------------------------------------------------
    rul_records = []
    plot_end_time_raw = raw_current_time * 1.2

    if thresholds is not None:
        if not isinstance(thresholds, (list, tuple, np.ndarray)):
            thresholds = [thresholds]

        for thresh in sorted(thresholds):
            nom_time = solve_rul_root(func, params, thresh, t_max, get_dynamic_std, sigma_factor, 'nominal')
            upper_time = solve_rul_root(func, params, thresh, t_max, get_dynamic_std, sigma_factor, 'upper')
            lower_time = solve_rul_root(func, params, thresh, t_max, get_dynamic_std, sigma_factor, 'lower')

            def calc_rul(t_val):
                if t_val == 'Safe':
                    return 'Safe'
                if isinstance(t_val, (float, int)) and t_val == SENTINEL_ALREADY_REACHED:
                    return SENTINEL_ALREADY_REACHED
                return (t_val - raw_current_time) * conversion_factor if isinstance(t_val, (float, int)) else np.nan

            def calc_abs(t_val):
                if t_val == 'Safe' or (isinstance(t_val, (float, int)) and t_val == SENTINEL_ALREADY_REACHED):
                    return np.nan
                return t_val * conversion_factor if isinstance(t_val, (float, int)) else np.nan

            nom_rul = calc_rul(nom_time)
            upper_rul = calc_rul(upper_time)
            lower_rul = calc_rul(lower_time)
            nom_abs, upper_abs, lower_abs = calc_abs(nom_time), calc_abs(upper_time), calc_abs(lower_time)

            if nom_rul == 'Safe':
                status = 'Never Reached (Safe)'
            elif isinstance(nom_rul, float) and nom_rul < 0:
                status = 'Already Reached'
            elif isinstance(upper_rul, float) and upper_rul < 0:
                status = 'Envelope Breached'
            else:
                status = 'Predicted Future'

            valid_times = [t for t in [nom_abs, upper_abs, lower_abs] if not pd.isna(t)]
            if valid_times:
                plot_end_time_raw = max(plot_end_time_raw, max(valid_times) / conversion_factor * 1.1)

            rul_records.append({
                'Threshold': thresh, 'Status': status, 'Current_Margin': current_margin,
                'Nominal_RUL': nom_rul, 'Nominal_Abs_Time': nom_abs,
                'Upper_Band_RUL': upper_rul, 'Upper_Abs_Time': upper_abs,
                'Lower_Band_RUL': lower_rul, 'Lower_Abs_Time': lower_abs
            })

    rul_df = pd.DataFrame(rul_records)

    # ---------------------------------------------------------
    # PLOTTING
    # ---------------------------------------------------------
    time_arr_converted = time_arr * conversion_factor
    plot_end_time_converted = plot_end_time_raw * conversion_factor
    current_time_converted = raw_current_time * conversion_factor
    max_zoom_limit = current_time_converted * 2.5
    final_end_time = min(plot_end_time_converted, max_zoom_limit)

    time_smooth_converted = np.linspace(0, final_end_time, 500)
    time_smooth_norm = (time_smooth_converted / conversion_factor) / t_max

    smooth_preds = func(time_smooth_norm, *params)
    dynamic_std_smooth = get_dynamic_std(time_smooth_norm)
    upper_env_smooth = smooth_preds + (dynamic_std_smooth * sigma_factor)
    lower_env_smooth = smooth_preds - (dynamic_std_smooth * sigma_factor)

    # --- NEW: Extract and calculate calendar dates for hover text ---
    if isinstance(orig_index, pd.DatetimeIndex) and len(orig_index) > 0:
        origin_date = orig_index.min()
        raw_dates = orig_index.strftime('%Y-%m-%d').tolist()
        # Convert the smooth time array back to days to add to the origin date
        days_smooth = time_smooth_converted / conversion_factor
        smooth_dates = [(origin_date + pd.Timedelta(days=d)).strftime('%Y-%m-%d') for d in days_smooth]
    else:
        raw_dates = [''] * len(sensor_raw_arr)
        smooth_dates = [''] * len(time_smooth_converted)
    # ---------------------------------------------------------------

    fig = go.Figure()
    
    # --- UPDATED: Added customdata and hovertemplate to traces ---
    fig.add_trace(go.Scatter(x=time_arr_converted, y=sensor_raw_arr, mode='markers',
                             name='Max Envelope Data', marker=dict(color='gray', size=5, opacity=0.7),
                             hovertemplate='Actual Value: %{y:.3f}<extra></extra>'))
                             
    # 2. NOMINAL FIT: Kept the Date! Since this line is continuous, it serves as our single date label.
    fig.add_trace(go.Scatter(x=time_smooth_converted, y=smooth_preds, mode='lines',
                             name=f'{model_choice} Fit', line=dict(color='blue', width=2.5), opacity=0.8,
                             customdata=smooth_dates,
                             hovertemplate='<b>Date: %{customdata}</b><br>Pred: %{y:.3f}<extra></extra>'))
                             
    # 3. CONFIDENCE FILL: Skipping hover entirely so it doesn't duplicate the band lines
    fig.add_trace(go.Scatter(x=time_smooth_converted, y=lower_env_smooth, mode='lines',
                             line=dict(width=0), showlegend=False, hoverinfo='skip'))
                             
    band_name = f'±{sigma_factor}σ Confidence Band' + (' (Dynamic)' if use_dynamic_variance else ' (Static)')
    
    fig.add_trace(go.Scatter(x=time_smooth_converted, y=upper_env_smooth, mode='lines',
                             line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 0, 255, 0.15)', name=band_name,
                             hoverinfo='skip')) # <-- ensure hover is skipped on the transparent fill!
                             
    # 4. UPPER BAND: Removed date
    fig.add_trace(go.Scatter(x=time_smooth_converted, y=upper_env_smooth, mode='lines',
                             name=f'Upper Band ({risk_pct_upper:.1f}% Risk)',
                             line=dict(color='blue', width=1.5, dash='dashdot'), opacity=0.6,
                             hovertemplate='Upper: %{y:.3f}<extra></extra>'))
                             
    # 5. LOWER BAND: Removed date
    fig.add_trace(go.Scatter(x=time_smooth_converted, y=lower_env_smooth, mode='lines',
                             name=f'Lower Band ({risk_pct_lower:.1f}% Risk)',
                             line=dict(color='blue', width=1.5, dash='dot'), opacity=0.4,
                             hovertemplate='Lower: %{y:.3f}<extra></extra>'))
    # ---------------------------------------------------------
    # ANNOTATIONS & LEGEND / TITLES
    # ---------------------------------------------------------
    unit_short = {"Seconds": "s", "Minutes": "m", "Hours": "h", "Days": "d"}.get(target_time_unit, target_time_unit)
    colors = ['#FFA500', '#FF0000', '#8B0000', '#800080', '#000000']
    status_lines = []

    def format_rul(val):
        if val == 'Safe':
            return "Safe"
        if pd.isna(val):
            return "Unknown"
        if isinstance(val, (float, int)):
            if val < 0:
                return "Breached"
            if val > max_rul_display:
                return f"> {max_rul_display}{unit_short}"
            return f"{val:.1f}{unit_short}"
        return str(val)

    if not rul_df.empty:
        for idx, row in rul_df.iterrows():
            c = colors[idx % len(colors)]
            thresh = float(row['Threshold'])
            status = row['Status']

            n_str = format_rul(row['Nominal_RUL'])
            u_str = format_rul(row['Upper_Band_RUL'])
            l_str = format_rul(row['Lower_Band_RUL'])

            if status == 'Never Reached (Safe)' and u_str == 'Safe' and l_str == 'Safe':
                status_lines.append(f"<b>T={thresh:.2f}:</b> Safe")
                legend_lbl = f"T={thresh:.2f} (Safe)"
            else:
                status_lines.append(f"<b>T={thresh:.2f} RUL</b> ➔ <b>Nominal:</b> {n_str} | <b>{risk_pct_upper:.1f}% Risk:</b> {u_str} | <b>{risk_pct_lower:.1f}% Risk:</b> {l_str}")
                legend_lbl = f"T={thresh:.2f} (Nom: {n_str} | {risk_pct_upper:.1f}%: {u_str} | {risk_pct_lower:.1f}%: {l_str})"

            fig.add_trace(go.Scatter(x=[0, final_end_time], y=[thresh, thresh], mode='lines',
                                     name=legend_lbl, line=dict(color=c, width=2, dash='dash'), opacity=0.6))

            if not pd.isna(row['Nominal_Abs_Time']) and row['Nominal_Abs_Time'] <= final_end_time:
                fig.add_trace(go.Scatter(x=[row['Nominal_Abs_Time']], y=[thresh], mode='markers', showlegend=False,
                                         marker=dict(symbol='circle', color=c, size=10, line=dict(color='black', width=1))))
            if not pd.isna(row['Upper_Abs_Time']) and row['Upper_Abs_Time'] <= final_end_time:
                fig.add_trace(go.Scatter(x=[row['Upper_Abs_Time']], y=[thresh], mode='markers', showlegend=False,
                                         marker=dict(symbol='triangle-left', color=c, size=10, line=dict(color='black', width=1))))
            if not pd.isna(row['Lower_Abs_Time']) and row['Lower_Abs_Time'] <= final_end_time:
                fig.add_trace(go.Scatter(x=[row['Lower_Abs_Time']], y=[thresh], mode='markers', showlegend=False,
                                         marker=dict(symbol='triangle-right', color=c, size=10, line=dict(color='black', width=1))))

    # --- Structural Break Visuals ---
    if break_time_raw is not None:
        break_time_converted = break_time_raw * conversion_factor
        fig.add_vline(x=break_time_converted, line_width=2, line_dash="dash", line_color="orange")
        fig.add_annotation(x=break_time_converted, y=y_max * 1.05 if y_max > 0 else 0,
                           text="⚠️ Structural Break", showarrow=False, yshift=10,
                           font=dict(color="orange", size=12))
        break_title_str = f" | <span style='color:orange;'><b>⚠️ Break Detected at {break_time_converted:.1f}{unit_short}</b></span>"
    else:
        break_title_str = " | <span style='color:green;'><b>✅ Stable (No Break)</b></span>"

    max_target = max(thresholds) if thresholds is not None else 0
    absolute_y_max = max(y_max, max_target)
    absolute_y_min = min(0, y_min)
    y_padding = (absolute_y_max - absolute_y_min) * 0.15
    y_upper_limit = absolute_y_max + y_padding
    y_lower_limit = absolute_y_min - (y_padding if absolute_y_min < 0 else 0)

    main_title = f"Predictive Analytics | Engine: {model_choice} {title_addon}{break_title_str}"
    subtitle_html = "<br>".join([f"<span style='font-size:14px; color:gray;'>{line}</span>" for line in status_lines])
    full_title = f"<b>{main_title}</b><br>{subtitle_html}"

    n_thresh = len(thresholds) if thresholds is not None else 0
    
    fig.update_layout(
        title=full_title, 
        xaxis_title=f"Elapsed Time ({target_time_unit})", 
        yaxis_title="Sensor Value",
        xaxis=dict(range=[0, final_end_time]),
        yaxis=dict(range=[y_lower_limit, y_upper_limit], hoverformat=".3f"),
        hovermode="x unified", 
        template="plotly_white",
        
        # --- UPDATED: Legend pushed below the chart ---
        legend=dict(
            title="System Log", 
            orientation="v",         # Makes the legend items sit side-by-side
            yanchor="top",           # Anchors the top of the legend box...
            y=-0.15,                 # ...to just below the x-axis
            xanchor="center",        # Anchors the center of the legend box...
            x=0.5,                   # ...to the middle of the chart
            bordercolor="LightSteelBlue", 
            borderwidth=1
        ),
        
        # --- UPDATED: Added bottom margin (b=120) to prevent the legend from being cut off ---
        margin=dict(t=120 + (n_thresh * 15), b=120), 
        width=1200, 
        height=900
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.4)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.4)')

    if save_name is not None:
        fig.write_html(save_name)

    return fig, fitted_series, rul_df


# =========================================================
# DATA LOADING & EXTRACTION
# =========================================================

@st.cache_data
def parse_raw_csv(file_obj):
    """Parses the CSV once into a raw DataFrame to be cached and shared."""
    file_obj.seek(0)
    df = pd.read_csv(file_obj, parse_dates=['DateTime'])
    df.set_index('DateTime', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    return df

def get_available_channels(df):
    """Now accepts a dataframe directly."""
    cols = [col for col in df.columns if 'Error' not in col]
    if 'Thermo_Valve_Temperature_DeviationPct' in cols:
        cols.remove('Thermo_Valve_Temperature_DeviationPct')
    return [str(c) for c in cols]

@st.cache_data
def load_my_sensor_data(df, col='32', outlier_factor=3.0, outlier_window=42):
    """Now accepts a dataframe instead of a file object."""
    freq = '4h'
    df_resampled = df.resample(freq).mean(numeric_only=True)

    if col not in df_resampled.columns:
        return pd.Series(), pd.Series(), pd.Series()

    df_select = df_resampled[[col]].copy()
    df_select = df_select.interpolate(method='time', limit=1)
    df_select = rolling_iqr_filter(df_select, factor=outlier_factor, window=outlier_window)

    window = int(1 * 24 / 4)
    df_select[f'{col}_max'] = df_select[col].rolling(window=window * 5, min_periods=1).max()
    df_select[f'{col}_max_ema'] = df_select[f'{col}_max'].ewm(span=window * 5, adjust=False, ignore_na=True).mean()

    df_daily = df_select.resample('D').mean()
    df_daily['elapsed_days'] = (df_daily.index - df_daily.index.min()).days

    return df_daily[f'{col}_max_ema'], df_daily[f'{col}_max'], df_daily['elapsed_days']

# ---------------------------------------------------------
# 6. Headless Math Engine (No Plotly rendering for speed)
# ---------------------------------------------------------
def calculate_rul_headless(time_raw, sensor_smooth, sensor_raw, model_choice, threshold,
                           sigma_factor=1.645, use_dynamic_variance=True, precomputed_params=None,variance_window=20):
    """
    OPT-A: When `precomputed_params` is supplied (the winner's fit from
    evaluate_all_models), this skips curve_fit entirely - no second fit.
    OPT-C: uses the shared brentq-based solver.
    ROBUSTNESS: uses centralized bounds + `except Exception`.
    """
    time_arr = np.asarray(time_raw, dtype=float)
    sensor_arr = np.asarray(sensor_smooth, dtype=float)

    raw_current_time = np.max(time_arr)
    t_max = raw_current_time if raw_current_time > 0 else 1.0
    time_norm = time_arr / t_max

    valid_mask = ~np.isnan(sensor_arr)
    if valid_mask.sum() < 10:
        return np.nan, np.nan, np.nan

    t_fit = time_norm[valid_mask]
    y_fit = sensor_arr[valid_mask]
    y_min, y_max = float(np.min(y_fit)), float(np.max(y_fit))
    y_range = y_max - y_min

    config = build_models_config(y_min, y_max, y_range).get(
        model_choice, build_models_config(y_min, y_max, y_range)['Linear'])
    func = config['func']

    if precomputed_params is not None:
        params = precomputed_params
    else:
        try:
            params, _ = curve_fit(func, t_fit, y_fit, p0=config['p0'],
                                  bounds=config['bounds'], method='trf', maxfev=5000)
        except Exception:
            return np.nan, np.nan, np.nan

    # --- Variance ---
    fitted_vals = func(t_fit, *params)
    residuals = y_fit - fitted_vals
    rolling_std = pd.Series(residuals).rolling(window=variance_window, min_periods=1).std().bfill().fillna(0).values

    if use_dynamic_variance and len(rolling_std) > 1:
        std_slope, std_intercept = np.polyfit(t_fit, rolling_std, 1)
    else:
        std_slope, std_intercept = 0.0, rolling_std[-1] if len(rolling_std) > 0 else 0.0

    get_dynamic_std = build_dynamic_std_fn(std_slope, std_intercept)

    def _rul(mode):
        t_val = solve_rul_root(func, params, threshold, t_max, get_dynamic_std, sigma_factor, mode)
        if t_val == 'Safe' or t_val is None:
            return np.nan
        if isinstance(t_val, (float, int)) and t_val == SENTINEL_ALREADY_REACHED:
            return np.nan
        return t_val - raw_current_time

    return _rul('nominal'), _rul('upper'), _rul('lower')


# ---------------------------------------------------------
# HORIZON HELPER (single source of truth for capping)
# ---------------------------------------------------------
def to_horizon(rul, horizon=RUL_HORIZON):
    """NaN ('Safe'/never reached) and anything beyond the horizon saturate at `horizon`."""
    if pd.isna(rul):
        return float(horizon)
    return float(min(rul, horizon))

def generate_simulation_dashboards(raw_df):
    """
    Chart A: Status Heatmap (classification driven by tolerance window & safe horizon).
    Chart B: Bias Heatmap (driven by HORIZON-CAPPED columns -> defined everywhere).
    Chart C: Scatter (full width; actual UNCAPPED, predicted CAPPED @ horizon).
    """
    df = raw_df.copy()

    # FORCE CATEGORICAL Y-AXIS
    df['Channel'] = "CH-" + df['Channel'].astype(str)

    st.markdown("---")
    st.header("📊 Fleet Backtesting Results")

    st.markdown("### Operational Thresholds")
    # Added a third column for the RUL Evaluation Mode
    col_aw, col_hz, col_mode = st.columns(3)
    
    with col_aw:
        action_window = st.slider(
            "Action Window (Days)", min_value=5, max_value=90, value=30, step=1,
            help="Tolerance window (± days). A prediction is considered a True Positive if it falls within this margin of the actual RUL."
        )
    with col_hz:
        display_horizon = st.slider(
            "Safe Horizon (Days)", min_value=30, max_value=730, value=RUL_HORIZON, step=5,
            help="The Safe Horizon. If both the predicted and actual RUL exceed this timeframe, it is considered a True Negative."
        )
    with col_mode:
        rul_mode = st.selectbox(
            "RUL Evaluation Mode",
            options=[
                "Nominal (Center Line)", 
                "Conservative (Early Alarm / Upper Band)", 
                "Optimistic (Late Alarm / Lower Band)"
            ],
            help="Choose which statistical confidence band to evaluate against the actual failure."
        )

    # Map the selected mode to the correct dataframe column
    if "Conservative" in rul_mode:
        eval_col = 'Upper_Risk_RUL'
    elif "Optimistic" in rul_mode:
        eval_col = 'Lower_Risk_RUL'
    else:
        eval_col = 'Nominal_RUL'

    # Create a dynamic column that the rest of the dashboard will use
    df['Selected_RUL'] = df[eval_col]

    # -----------------------------------------------------
    # HORIZON-CAPPED COLUMNS (continuous charts B & C only).
    # NaN actual/predicted ("never reached") -> full horizon so a position exists.
    # -----------------------------------------------------
    df['Selected_RUL_c'] = df['Selected_RUL'].apply(lambda x: to_horizon(x, display_horizon))
    df['Actual_RUL_c']  = df['Actual_RUL'].apply(lambda x: to_horizon(x, display_horizon))

    # -----------------------------------------------------
    # CHART A STATUS: Updated Tolerance and Horizon Logic
    # -----------------------------------------------------
    df['Status'] = 'Unknown'

    # Treat NaN as infinite (never reaches threshold) for logical comparisons
    act_val = df['Actual_RUL'].fillna(np.inf)
    pred_val = df['Selected_RUL'].fillna(np.inf)

    # 1. True Negative: Both are safely beyond the display horizon
    is_tn = (act_val > display_horizon) & (pred_val > display_horizon)

    # 2. True Positive: The prediction is within the action window tolerance of the actual RUL
    is_tp = (abs(pred_val - act_val) <= action_window) & ~is_tn

    # 3. False Positive: Predicted failure is earlier than actual (and not a TP/TN)
    is_fp = (pred_val < act_val) & ~is_tp & ~is_tn

    # 4. False Negative: Predicted failure is later than actual (and not a TP/TN)
    is_fn = (pred_val > act_val) & ~is_tp & ~is_tn

    df.loc[is_tn, 'Status'] = 'True Negative'
    df.loc[is_tp, 'Status'] = 'True Positive'
    df.loc[is_fp, 'Status'] = 'False Positive'
    df.loc[is_fn, 'Status'] = 'False Negative'

    # -----------------------------------------------------
    # BIAS SCORE: computed from CAPPED columns -> no NaN holes in Chart B.
    # -----------------------------------------------------
    error = df['Selected_RUL_c'] - df['Actual_RUL_c']
    df['Bias_Score'] = np.clip((error + action_window) / (action_window * 2), 0.0, 1.0)

    df['Eval_Day_Rounded'] = df['Evaluation_Day'].round().astype(int)

    # Hover keeps RAW semantics: NaN prediction is explicitly "never reached", not a number.
    def _fmt_pred(x):
        if pd.isna(x):
            return "Never reaches threshold (Safe)"
        return f"{x:.1f} Days"

    df['Hover_Pred'] = df['Selected_RUL'].apply(_fmt_pred)
    df['Hover_Act'] = df['Actual_RUL'].apply(lambda x: f"{x:.1f} Days" if pd.notna(x) else "Never Breaches")

    pivot_pred = df.pivot_table(index='Channel', columns='Eval_Day_Rounded', values='Hover_Pred', aggfunc='first')
    pivot_act = df.pivot_table(index='Channel', columns='Eval_Day_Rounded', values='Hover_Act', aggfunc='first')
    pivot_date = df.pivot_table(index='Channel', columns='Eval_Day_Rounded', values='Evaluation_Date', aggfunc='first')
    
    color_map_status = {"True Positive": "#4CAF50", "False Positive": "#FFB74D",
                        "True Negative": "#E8F5E9", "False Negative": "#F44336"}

    # ==========================================
    # CHART A: Lifecycle Confusion Heatmap
    # ==========================================
    st.subheader("Chart A: Lifecycle Confusion Matrix (Categorical)")
    st.caption(
        f"Each cell shows the model's classification for that day (using **{rul_mode}**):  \n"
        f"🟩 **True Positive:** The prediction was accurate within the ±{action_window}-day tolerance window.  \n"
        f"⬜ **True Negative:** Both actual and predicted RUL are safely beyond the {display_horizon}-day horizon.  \n"
        f"🟧 **False Positive:** The model alarmed too early (predicted failure earlier than actual).  \n"
        f"🟥 **False Negative:** The model alarmed too late or missed the failure entirely (predicted failure later than actual).  \n"
    )

    status_map = {"True Negative": 0, "False Positive": 1, "True Positive": 2, "False Negative": 3}
    df['Status_Code'] = df['Status'].map(status_map)

    pivot_status_code = df.pivot_table(index='Channel', columns='Eval_Day_Rounded', values='Status_Code', aggfunc='first')
    pivot_status_text = df.pivot_table(index='Channel', columns='Eval_Day_Rounded', values='Status', aggfunc='first')
    customdata_a = np.dstack((pivot_status_text.values,
                              pivot_pred.reindex_like(pivot_status_text).values,
                              pivot_act.reindex_like(pivot_status_text).values,
                              pivot_date.reindex_like(pivot_status_text).values))

    colorscale_a = [[0.0, '#E8F5E9'], [0.33, '#FFB74D'], [0.66, '#4CAF50'], [1.0, '#F44336']]

    fig_a = go.Figure()
    fig_a.add_trace(go.Heatmap(
        z=pivot_status_code.values, x=pivot_status_code.columns, y=pivot_status_code.index,
        colorscale=colorscale_a, zmin=0, zmax=3, showscale=False, hoverongaps=False, customdata=customdata_a,
        hovertemplate="<b>Date:</b> %{customdata[3]}<br><b>Day:</b> %{x}<br><b>Channel:</b> %{y}<br><b>Result:</b> %{customdata[0]}<br><b>Predicted:</b> %{customdata[1]}<br><b>Actual:</b> %{customdata[2]}<extra></extra>"
        ))
    for name, color in [("True Negative (Safe)", '#E8F5E9'), ("False Positive (Early Alarm)", '#FFB74D'),
                        ("True Positive (Correct Detection)", '#4CAF50'), ("False Negative (Missed Crossing)", '#F44336')]:
        fig_a.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                                   marker=dict(size=15, color=color, symbol='square', line=dict(color='black', width=1)), name=name))
    fig_a.update_layout(
        xaxis_title="Simulation Day", yaxis_title="Channel", height=400, template="plotly_white", margin=dict(t=30, b=30),
        yaxis=dict(type='category', autorange="reversed"),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, title=None)
    )
    st.plotly_chart(fig_a, use_container_width=True)

    # ==========================================
    # CHART B: Directional Bias Heatmap
    # ==========================================
    st.subheader("Chart B: Directional Bias Matrix (Continuous)")
    st.caption(
        f"Shows the direction of each prediction's error against the **{rul_mode}**:  \n"
        f"Dark = conservative (predicted failure earlier than it occurred — the safe side).  \n"
        f"Light = optimistic (predicted failure later than it occurred — the risk side).  \n"
        f"Mid-tone = on target.  \n"
    )

    rocket_palette = sns.color_palette("mako", n_colors=256).as_hex()
    pivot_bias = df.pivot_table(index='Channel', columns='Eval_Day_Rounded', values='Bias_Score', aggfunc='first')

    df['Hover_Bias'] = df['Bias_Score'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    pivot_bias_text = df.pivot_table(index='Channel', columns='Eval_Day_Rounded', values='Hover_Bias', aggfunc='first')
    customdata_b = np.dstack((pivot_bias_text.reindex_like(pivot_bias).values,
                              pivot_pred.reindex_like(pivot_bias).values,
                              pivot_act.reindex_like(pivot_bias).values,
                              pivot_date.reindex_like(pivot_bias).values))

    fig_b = go.Figure(data=go.Heatmap(
        z=pivot_bias.values, x=pivot_bias.columns, y=pivot_bias.index,
        colorscale=rocket_palette, zmin=0.0, zmax=1.0, showscale=True, hoverongaps=False, customdata=customdata_b,
        colorbar=dict(title="Bias Score", tickvals=[0, 0.5, 1], ticktext=["0.0 (Early)", "0.5", "1.0 (Late)"]),
        hovertemplate="<b>Date:</b> %{customdata[3]}<br><b>Day:</b> %{x}<br><b>Channel:</b> %{y}<br><b>Bias Score:</b> %{customdata[0]}<br><b>Predicted:</b> %{customdata[1]}<br><b>Actual:</b> %{customdata[2]}<extra></extra>"
    ))
    fig_b.update_layout(
        xaxis_title="Simulation Day", yaxis_title="Channel", height=400, template="plotly_white", margin=dict(t=30, b=30),
        yaxis=dict(type='category', autorange="reversed")
    )
    st.plotly_chart(fig_b, use_container_width=True)

    # ==========================================
    # CHART C: Aggregated Scatter (full width)
    # ==========================================
    st.subheader("Chart C: Prediction Scatter")
    
    scatter_mode = st.radio(
        "Chart C View Mode:",
        ["Finite predictions only (Recommended)", "All evaluations (Includes Safe/Never reaches)"],
        horizontal=True
    )

    df_scatter = df.copy()

    if scatter_mode == "Finite predictions only (Recommended)":
        df_scatter = df_scatter[df_scatter['Selected_RUL'].notna()].copy()
        st.caption(
            "Showing only evaluations where the model predicted a finite crossing time. "
            "Points on the dashed diagonal are exact. Above the line = optimistic; below = conservative."
        )
    else:
        st.caption(
            f"Showing all evaluations. The x-axis places real crossings normally, but groups "
            f"'never crosses' cases into a right-most parking lane. The y-axis caps safe predictions "
            f"at {display_horizon} days (flattening along the top edge)."
        )

    df_scatter['Predicted RUL'] = df_scatter['Hover_Pred']
    df_scatter['Actual Outcome'] = df_scatter['Hover_Act']

    real_actual_max = df_scatter['Actual_RUL'].max(skipna=True)
    if pd.isna(real_actual_max):
        real_actual_max = display_horizon
    never_pos = real_actual_max * 1.15
    df_scatter['Actual_RUL_plot'] = df_scatter['Actual_RUL'].fillna(never_pos)

    if not df_scatter.empty:
        x_view_max = max(never_pos, action_window * 2) * 1.10
        y_view_max = display_horizon * 1.05

        fig_c = px.scatter(
            df_scatter, x="Actual_RUL_plot", y="Selected_RUL_c", color="Status",
            hover_data={
                "Actual_RUL_plot": False,  
                "Selected_RUL_c": False,    
                "Status": False,           
                "Channel": True,
                "Evaluation_Day": True,
                "Predicted RUL": True,
                "Actual Outcome": True
            },
            color_discrete_map=color_map_status
        )

        diag_max = min(x_view_max, y_view_max)
        fig_c.add_trace(go.Scatter(x=[0, diag_max], y=[0, diag_max], mode='lines',
                                   name='Perfect Prediction', line=dict(color='black', dash='dash')))
        
        fig_c.add_vline(x=action_window, line_width=1, line_dash="dot", line_color="gray")
        fig_c.add_hline(y=action_window, line_width=1, line_dash="dot", line_color="gray")
        
        if df_scatter['Actual_RUL'].isna().any():
            fig_c.add_vline(x=never_pos, line_width=1, line_dash="dashdot", line_color="lightgray")
            fig_c.add_annotation(x=never_pos, y=y_view_max, yanchor="top", showarrow=False,
                                 text="Never crosses →", font=dict(size=11, color="gray"),
                                 xshift=-5, xanchor="right")
            
        fig_c.add_hline(y=display_horizon, line_width=1, line_dash="dashdot", line_color="lightgray")

        if scatter_mode == "Finite predictions only (Recommended)" and not df_scatter['Actual_RUL'].isna().any():
            x_title = "Actual outcome (Days)"
        else:
            x_title = "Actual outcome (Days; safe cases grouped at right)"

        max_range = max(x_view_max, y_view_max)

        fig_c.update_layout(
            xaxis_title=x_title,
            yaxis_title=f"Predicted RUL ({rul_mode.split(' ')[0]}, capped @ {display_horizon})",
            xaxis=dict(range=[0, max_range]), 
            yaxis=dict(
                range=[0, max_range], 
                scaleanchor="x",  
                scaleratio=1      
            ),
            width=1200, height=800,
            margin=dict(l=80, r=80, t=100, b=80),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        col_left, col_center, col_right = st.columns([1, 3, 1])
        with col_center:
            st.plotly_chart(fig_c, use_container_width=False)



def page_live_simulation(active_df, base_df, priority_dict, outlier_factor, outlier_window,
                         use_dynamic_variance, break_algo, break_window, break_step, break_sustained,
                         z_factor, z_sustained, override_model, manual_model,
                         window_size, eval_window,variance_window, min_mse_floor):
    st.title("Fleet-Wide Live Simulation")
    st.markdown("Run the predictive engine across all channels and all historical timesteps to generate statistical confidence metrics.")

    st.markdown("### Simulation Setup")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        req_break = st.toggle("Require Structural Break", value=False,
                              help=" ON: only evaluate AFTER a detected bend. OFF: evaluate from Day 50 onward.")
    with col2:
        step_days = st.number_input("Timestep Interval (Days)", min_value=1, max_value=30, value=7,
                                    help=" Days the simulation jumps between calculations.")
    with col3:
        target_thresh = st.number_input("Target Threshold", min_value=0.1, max_value=5.0, value=0.2, step=0.1,
                                        help=" The critical limit line.")
    with col4:
        ema_span = st.number_input("EMA Smoothing Span", min_value=1, max_value=20, value=1, step=1,
                                   help=" 1 = no smoothing. 4 = moderate smoothing.")

    if 'sim_results' not in st.session_state:
        st.session_state['sim_results'] = None

    if st.button("🚀 Start Fleet Simulation", type="primary", use_container_width=True):
        
        # --- REMNANT FIX 1: Fetch channels from active_df ---
        channels = get_available_channels(active_df)
        results_list = []

        st.markdown("#### Simulation Progress")
        status_text = st.empty()
        progress_bar = st.progress(0.0)
        sub_status_text = st.empty()
        sub_progress_bar = st.progress(0.0)

        total_channels = len(channels)
        UI_THROTTLE = 5  

        if break_algo == "Fleet Z-Score":
            status_text.markdown("**Pre-computing Fleet Baseline...**")
            # --- REMNANT FIX 2: Anchor Z-Score strictly to base_df ---
            fleet_mean, fleet_std = compute_fleet_baseline(base_df, outlier_factor, outlier_window)
        else:
            fleet_mean, fleet_std = 0.0, 1.0
            
        origin_date = active_df.index.min().normalize()
        
        available_cores = os.cpu_count() or 1
        
        core_info_box = st.empty()
        
        # 2. Write the message INSIDE the placeholder
        core_info_box.info(f"⚡Multi-Core processing active. Utilizing **{available_cores} CPU cores**.")

        status_text.markdown(f"**Spinning up {available_cores} CPU cores...**")

        # --- THE MULTI-CORE DISPATCHER ---
        # max_workers=None defaults to the number of physical cores on your machine
        with concurrent.futures.ProcessPoolExecutor() as executor:
            
            # 1. Dispatch all jobs to the background cores
            futures = []
            for channel in channels:
                future = executor.submit(
                    process_single_channel_worker, 
                    channel, active_df, outlier_factor, outlier_window, 
                    target_thresh, break_algo, break_window, break_step, 
                    break_sustained, z_factor, z_sustained, min_mse_floor, 
                    req_break, step_days, window_size, eval_window, 
                    override_model, manual_model, use_dynamic_variance, 
                    variance_window, priority_dict, ema_span, origin_date,
                    fleet_mean, fleet_std
                )
                futures.append(future)

            # 2. Collect results as they finish and update the UI
            channels_completed = 0
            for future in concurrent.futures.as_completed(futures):
                channels_completed += 1
                
                try:
                    # Get the list of results for this specific channel
                    channel_results = future.result() 
                    if channel_results:
                        results_list.extend(channel_results)
                        
                    # Update UI in the safe Main Thread
                    progress_bar.progress(channels_completed / total_channels)
                    status_text.markdown(f"**Overall Fleet Progress:** {channels_completed} / {total_channels} channels complete.")
                    
                except Exception as e:
                    # If a worker crashes, catch it so the whole app doesn't die
                    st.error(f"Worker process failed on a channel: {e}")

        status_text.success("✅ Fleet Simulation Complete!")
        progress_bar.empty()
        core_info_box.empty()

        if results_list:
            st.session_state['sim_results'] = pd.DataFrame(results_list)
        else:
            st.warning("No evaluations were run.")

    if st.session_state['sim_results'] is not None:
        generate_simulation_dashboards(st.session_state['sim_results'])
        with st.expander("View Raw Simulation Logs"):
            st.dataframe(st.session_state['sim_results'], use_container_width=True)

# ---------------------------------------------------------
# 6. Synthetic Data Studio Page
# ---------------------------------------------------------
def page_synthetic_studio(base_df):
    st.title("🧪 Synthetic Data Studio")
    
    with st.expander("📖 Guide: The Physics of Synthetic Testing", expanded=True):
        st.markdown("""
        This studio safely isolates the original data and generates artificial failure scenarios to benchmark the prediction engine.
        
        **We use two distinct testing strategies:**
        * **1. Fault Injection (Testing Detection):** We take a perfectly healthy sensor and force it to fail at a random time. This proves the engine can catch *new* degradation it has never seen before.
        * **2. Fault Mutation (Testing Robustness):** We take an asset that already failed and distort its signal. This proves the engine won't break or false-alarm when exposed to noisy environments, sensor drift, or changing machine workloads.
        """)

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Inject Faults into Healthy Assets")
        num_healthy = st.number_input(
            "Synthetic Healthy Channels to Create", min_value=0, max_value=20, value=3
        )
        healthy_thresh = st.number_input(
            "Max value to be considered 'Healthy'", value=0.2, step=0.1,
            help="Filters the fleet based on 1-day median data to ignore noise."
        )
        
        st.markdown("<br>**Fault Profile Mix (Weights)**", unsafe_allow_html=True)
        st.caption("Adjust sliders to control how often each physical fault type is generated.")
        
        st.markdown("**Linear Ramp** | *Gradual wear (e.g., abrasive wear, filter clogging)*")
        w_lin = st.slider("Linear Ramp Weight", 0, 5, 1, label_visibility="collapsed")
        
        st.markdown("**Exponential Curve** | *Compounding damage (e.g., crack propagation, thermal runaway)*")
        w_exp = st.slider("Exponential Weight", 0, 5, 1, label_visibility="collapsed")
        
        st.markdown("**Logarithmic Curve** | *Rapid initial damage that flattens out (e.g., component break-in, plastic deformation)*")
        w_log = st.slider("Logarithmic Weight", 0, 5, 1, label_visibility="collapsed")
        
    with col2:
        st.subheader("2. Mutate Unhealthy Assets")
        num_unhealthy = st.number_input(
            "Mutated Unhealthy Channels to Create", min_value=0, max_value=20, value=3
        )
        unhealthy_thresh = st.number_input(
            "Min value to be considered 'Unhealthy'", value=0.2, step=0.1,
            help="Filters the fleet based on 1-day median data."
        )
        
        st.markdown("<br>**Mutation Mix (Weights)**", unsafe_allow_html=True)
        st.caption("Distort existing failure signatures to test algorithm robustness.")
        
        st.markdown("**Time-Warping** | *Tests varied workloads (faster/slower failures)*")
        w_warp = st.slider("Time-Warping Weight", 0, 5, 1, label_visibility="collapsed")
        
        st.markdown("**Baseline Drift** | *Tests sensor aging/calibration drift (prevents false early alarms)*")
        w_drift = st.slider("Baseline Drift Weight", 0, 5, 1, label_visibility="collapsed")
        
        st.markdown("**Tail Noise** | *Tests violent chatter near failure (challenges the outlier filter)*")
        w_noise = st.slider("Tail Noise Weight", 0, 5, 1, label_visibility="collapsed")

    st.markdown("---")

    if st.button("🧬 Generate Synthetic Fleet", type="primary", use_container_width=True):
        with st.spinner("Analyzing fleet and generating synthetic profiles..."):
            channels = get_available_channels(base_df)

            healthy_pool = []
            unhealthy_pool = []

            daily_df = base_df[channels].resample('1D').median(numeric_only=True)

            for ch in channels:
                max_val = daily_df[ch].max(skipna=True)
                if pd.isna(max_val):
                    continue
                if max_val < healthy_thresh:
                    healthy_pool.append(ch)
                elif max_val >= unhealthy_thresh:
                    unhealthy_pool.append(ch)

            if not healthy_pool and num_healthy > 0:
                st.error("No healthy channels found below your threshold. Cannot generate healthy synthetics.")
                return
            if not unhealthy_pool and num_unhealthy > 0:
                st.error("No unhealthy channels found above your threshold. Cannot mutate unhealthy channels.")
                return

            synth_df = pd.DataFrame(index=base_df.index)
            for ch in unhealthy_pool:
                synth_df[ch] = base_df[ch].copy()

            ground_truth = {}     # NEW: col_name -> metadata incl. break timestamp
            generated_count = 0

            # ---------- Pipeline 1: Healthy -> Fault (INJECTION) ----------
            if num_healthy > 0:
                fault_types = ["Linear", "Exponential", "Logarithmic"]
                fault_weights = [w_lin, w_exp, w_log]

                if sum(fault_weights) > 0:
                    for i in range(num_healthy):
                        ch = np.random.choice(healthy_pool)
                        arr = base_df[ch].values.copy()

                        valid_indices = np.where(~np.isnan(arr))[0]
                        if len(valid_indices) < 50:
                            continue

                        start_idx = valid_indices[int(len(valid_indices) * np.random.uniform(0.4, 0.8))]
                        f_type = np.random.choice(fault_types, p=np.array(fault_weights)/sum(fault_weights))

                        if f_type == "Linear":
                            arr = add_linear_ramp(arr, start_idx, max_offset=np.random.uniform(0.5, 1.5))
                        elif f_type == "Exponential":
                            arr = add_exponential_curve(arr, start_idx, severity_factor=np.random.uniform(0.5, 2.0))
                        elif f_type == "Logarithmic":
                            target = np.random.uniform(0.5, 10.0) * healthy_thresh
                            arr = add_logarithmic_curve(arr, start_idx, target_offset=target)

                        new_col_name = f"INJECT_{f_type}_{i+1} (Base: {ch})"
                        synth_df[new_col_name] = arr

                        # NEW: record exact ground-truth break as a timestamp.
                        ground_truth[new_col_name] = {
                            "break_timestamp": base_df.index[start_idx],
                            "fault_type": f_type,
                            "kind": "inject",
                            "base": ch,
                        }
                        generated_count += 1

            # ---------- Pipeline 2: Mutating Unhealthy (MUTATION) ----------
            if num_unhealthy > 0:
                mut_types = ["TimeWarp", "Drift", "Noise"]
                mut_weights = [w_warp, w_drift, w_noise]

                if sum(mut_weights) > 0:
                    for j in range(num_unhealthy):
                        ch = np.random.choice(unhealthy_pool)
                        arr = base_df[ch].values.copy()

                        smooth_arr = pd.Series(arr).rolling(window=20, min_periods=1).median().values
                        crossings = np.where(smooth_arr >= unhealthy_thresh)[0]
                        start_idx = crossings[0] if len(crossings) > 0 else len(arr) // 2

                        m_type = np.random.choice(mut_types, p=np.array(mut_weights)/sum(mut_weights))

                        if m_type == "TimeWarp":
                            arr = stretch_or_squeeze_time(arr, start_idx, factor=np.random.uniform(0.5, 1.5))
                        elif m_type == "Drift":
                            arr = add_baseline_drift(arr, drift_max=np.random.uniform(0.2, 0.6))
                        elif m_type == "Noise":
                            arr = inject_tail_noise(arr, start_idx, noise_multiplier=np.random.uniform(2.0, 4.0))

                        new_col_name = f"MUTATE_{m_type}_{j+1} (Base: {ch})"
                        synth_df[new_col_name] = arr

                        # NEW: Drift has no point break -> no usable ground truth.
                        # Others record where the existing crossing/mutation begins.
                        ground_truth[new_col_name] = {
                            "break_timestamp": base_df.index[start_idx],
                            "fault_type": m_type,
                            "kind": "mutate",
                            "base": ch,
                        }
                        generated_count += 1

            st.session_state['synthetic_df'] = synth_df
            st.session_state['synthetic_truth'] = ground_truth   # NEW
            st.success(f"✅ Generated {generated_count} synthetic channels! Toggle the Data Source in the sidebar to test them.")


def page_break_benchmark(active_df, base_df, is_synthetic, outlier_factor, outlier_window,
                         break_algo, break_window, break_step, break_sustained,
                         z_factor, z_sustained, min_mse_floor):
    st.title("🎯 Structural Break Detection Benchmark")
    init_break_label_store()

    scope = "synthetic" if is_synthetic else "original"
    labels = st.session_state['break_labels'][scope]

    st.markdown(
        "This page scores the structural-break detectors against ground-truth break "
        "points. Click **Start Simulation** to begin: injected synthetic channels are "
        "auto-seeded from their recorded break times, and only channels without known "
        "ground truth need manual labeling. The benchmark then runs automatically."
    )

    # -----------------------------------------------------------------
    # PHASE GATE
    # -----------------------------------------------------------------
    if st.button("🚀 Start Simulation", type="primary", use_container_width=True):
        st.session_state['bench_phase'] = 'init'
        st.session_state['bench_scope'] = scope
        st.session_state['label_idx'] = 0   # <--- NEW: Initialize the pointer

    phase = st.session_state.get('bench_phase')
    if phase is None or st.session_state.get('bench_scope') != scope:
        st.info("Press **Start Simulation** to begin labeling / benchmarking this dataset.")
        if labels:
            st.caption(f"Existing labels for **{scope}**: "
                       f"{sum(v is not None for v in labels.values())} breaks, "
                       f"{sum(v is None for v in labels.values())} skipped.")
        return

    # -----------------------------------------------------------------
    # PHASE 1: INIT -> build candidate list (the original/synthetic fork)
    # -----------------------------------------------------------------
    if phase == 'init':
        if is_synthetic:
            candidates = get_unhealthy_channels(active_df, True, None,
                                                outlier_factor, outlier_window)
            candidates = order_candidates_inject_first(candidates)
            autoseed_injected_breaks(active_df, scope, candidates)   # seed INJECT_ + propagate
            st.session_state['bench_candidates'] = candidates
            st.session_state['bench_phase'] = 'label'
            st.rerun()
        else:
            st.subheader("Step 1 — Unhealthy Threshold (Original Dataset)")
            thresh = st.number_input(
                "A channel is 'unhealthy' if its smoothed signal crosses this value:",
                min_value=0.01, max_value=10.0, value=0.2, step=0.05
            )
            if st.button("Find Unhealthy Channels ▶️"):
                candidates = get_unhealthy_channels(active_df, False, thresh,
                                                    outlier_factor, outlier_window)
                if not candidates:
                    st.warning("No channels cross that threshold. Lower it and retry.")
                    return
                st.session_state['bench_candidates'] = candidates
                st.session_state['bench_phase'] = 'label'
                st.rerun()
        return

    # -----------------------------------------------------------------
    # PHASE 2: LABEL -> sequential click-to-set-break
    # -----------------------------------------------------------------
    if phase == 'label':
        candidates = st.session_state['bench_candidates']
        
        # Isolate channels that actually require manual labeling
        manual_candidates = [c for c in candidates if not c.startswith("INJECT_")]
        
        # Failsafe: Ensure pointer exists
        if 'label_idx' not in st.session_state:
            st.session_state['label_idx'] = 0
            
        total_manual = len(manual_candidates)
        done_manual = st.session_state['label_idx']

        st.subheader("Step 2 — Label Structural Breaks")

        n_seeded = sum(1 for c in candidates if c.startswith("INJECT_") and c in labels)
        if is_synthetic and n_seeded:
            st.caption(f"✅ Seeded {n_seeded} injected break(s) from ground truth — "
                       f"no manual labeling needed for those.")

        st.progress(done_manual / max(1, total_manual) if total_manual else 1.0)

        # --- COMPLETION CHECK ---
        if done_manual >= total_manual:
            st.success("✅ All channels labeled (seeded + manual).")
            colA, colB = st.columns(2)
            if colA.button("▶️ Run Benchmark Now", type="primary"):
                st.session_state['bench_phase'] = 'run'
                st.rerun()
            if colB.button("🔄 Re-label from scratch"):
                st.session_state['break_labels'][scope] = {}
                st.session_state['bench_phase'] = 'init'
                st.session_state['label_idx'] = 0  # Reset pointer
                st.rerun()
            return

        # --- CURRENT CHANNEL UI ---
        col = manual_candidates[st.session_state['label_idx']]
        st.markdown(f"**Manual label {done_manual + 1} / {total_manual}:** `{col}`")

        smooth, raw, time_arr = load_my_sensor_data(
            active_df, col=col, outlier_factor=outlier_factor,
            outlier_window=outlier_window
        )
        time_np = np.asarray(time_arr, dtype=float)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_np, y=np.asarray(raw, dtype=float),
                                 mode="markers", name="raw",
                                 marker=dict(size=4, opacity=0.4, color="gray")))
        fig.add_trace(go.Scatter(x=time_np, y=np.asarray(smooth, dtype=float),
                                 mode="lines", name="smoothed",
                                 line=dict(color="blue", width=2)))
        
        # --- NEW: VISUAL INDICATOR ---
        current_label = labels.get(col)
        if current_label is not None:
            fig.add_vline(x=current_label, line_width=2, line_dash="dash", line_color="red")
            fig.add_annotation(x=current_label, y=np.nanmax(smooth),
                               text="📍 Break Placed", showarrow=False,
                               yshift=10, font=dict(color="red", size=14))

        fig.update_layout(template="plotly_white", height=500,
                          title=f"Click the break onset for {col}",
                          xaxis_title="Elapsed Days", yaxis_title="Sensor Value")

        st.caption("👉 Click the curve where the break begins. Then press **Next Channel**.")
        
        event = st.plotly_chart(fig, use_container_width=True,
                                on_select="rerun", selection_mode="points", key=f"lbl_{scope}_{col}")

        # --- HANDLE GRAPH CLICKS ---
        pts = (event or {}).get("selection", {}).get("points", [])
        if pts:
            break_day = float(pts[0]["x"])
            # Prevent infinite reruns: only update if the user clicked a NEW spot
            if labels.get(col) != break_day:
                labels[col] = break_day
                if is_synthetic and not col.startswith("MUTATE_"):
                    propagate_break_to_mutations(col, break_day, scope, candidates)
                st.rerun()

        # --- NEW: MANUAL PROGRESSION CONTROLS ---
        c1, c2, c3 = st.columns(3)
        
        has_label = col in labels
        
        if c1.button("⏭️ Next Channel", type="primary" if has_label else "secondary", disabled=not has_label):
            st.session_state['label_idx'] += 1
            st.rerun()
            
        if c2.button("🚫 No obvious break (Skip / Control)"):
            labels[col] = None
            st.session_state['label_idx'] += 1
            st.rerun()
            
        # Optional: Let the user clear a misclick
        if has_label and current_label is not None:
            if c3.button("🔄 Clear Label"):
                del labels[col]
                st.rerun()
        return
    # -----------------------------------------------------------------
    # PHASE 3: RUN -> score the detector against the labels
    # -----------------------------------------------------------------
    # -----------------------------------------------------------------
    # PHASE 3: RUN -> score the detector against the labels
    # -----------------------------------------------------------------
    if phase == 'run':
        st.subheader("Step 3 — Benchmark Results")
        tol = st.slider("Detection Tolerance (± Days)", 5, 90, 21, 1,
                        help="A detection within this many days of the true break counts as a hit.")

        if break_algo == "Fleet Z-Score":
            # Anchor to base_df
            fleet_mean, fleet_std = compute_fleet_baseline(
                base_df, outlier_factor, outlier_window)

        origin = active_df.index.min().normalize()
        origin = active_df.index.min().normalize()
        rows = []

        # --- NEW: Progress Bar ---
        status_text = st.empty()
        progress_bar = st.progress(0.0)
        total_labels = len(labels)

        for i, (col, true_day) in enumerate(labels.items()):
            status_text.markdown(f"**Running benchmark:** Analyzing `{col}` ({i+1}/{total_labels})")
            progress_bar.progress((i + 1) / total_labels)
            
            if col not in active_df.columns:
                continue
            smooth, raw, time_arr = load_my_sensor_data(
                active_df, col=col, outlier_factor=outlier_factor,
                outlier_window=outlier_window)
            time_np = np.asarray(time_arr, dtype=float)
            smooth_np = np.asarray(smooth, dtype=float)

            if break_algo == "Fleet Z-Score":
                b_idx, b_time, _ = detect_zscore_break(
                    time_np, smooth_np, fleet_mean, fleet_std,
                    z_factor, z_sustained)
            else:
                b_idx, b_time, _ = detect_structural_break(
                    time_np, smooth_np, window=break_window,
                    step=break_step, sustained_wins=break_sustained,min_mse_floor=min_mse_floor)

            detected = b_time is not None
            is_control = true_day is None

            # --- NEW: Clearer Categorization ---
            if is_control:
                status = "False Positive (Fired on Control)" if detected else "True Negative (Correctly Ignored)"
                latency = np.nan
            elif not detected:
                status = "False Negative (Missed Break)"
                latency = np.nan
            else:
                latency = b_time - true_day
                if abs(latency) <= tol:
                    status = "True Positive (Hit)"
                elif latency < -tol:
                    status = "False Positive (Fired Too Early)"
                else:
                    status = "False Negative (Fired Too Late)"

            rows.append({
                "Channel": col,
                "True Break (Day)": round(true_day, 1) if true_day is not None else "— (Control)",
                "Detected (Day)": round(b_time, 1) if detected else None,
                "Latency (Days)": round(latency, 1) if not np.isnan(latency) else None,
                "Status": status,
            })

        # Clear UI elements after loop
        status_text.empty()
        progress_bar.empty()

        res = pd.DataFrame(rows)
        if res.empty:
            st.info("No labeled channels to score.")
            return

        n_break = (res["True Break (Day)"] != "— (control)").sum()
        tp    = (res["Status"] == "Hit (TP)").sum()
        fn    = (res["Status"] == "Miss (FN)").sum()
        fp    = (res["Status"] == "False Positive").sum()
        early = (res["Status"] == "Early / False Trigger").sum()
        med_lat = res.loc[res["Status"] == "Hit (TP)", "Latency (Days)"].median()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Detection Recall", f"{tp / n_break:.0%}" if n_break else "—")
        m2.metric("Median Latency", f"{med_lat:.1f} d" if pd.notna(med_lat) else "—")
        m3.metric("False Alarms", int(fp + early))
        m4.metric("Misses", int(fn))

        # --- NEW: Better Summary Metrics ---
        n_break = (res["True Break (Day)"] != "— (Control)").sum()
        tp    = (res["Status"] == "True Positive (Hit)").sum()
        tn    = (res["Status"] == "True Negative (Correctly Ignored)").sum()
        fn    = res["Status"].str.contains("False Negative").sum()
        fp    = res["Status"].str.contains("False Positive").sum()
        med_lat = res.loc[res["Status"] == "True Positive (Hit)", "Latency (Days)"].median()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Detection Recall", f"{tp / n_break:.0%}" if n_break else "—")
        m2.metric("Median Hit Latency", f"{med_lat:.1f} d" if pd.notna(med_lat) else "—")
        m3.metric("False Alarms (FP)", int(fp))
        m4.metric("Misses (FN)", int(fn))

        # --- NEW: Horizontal Diverging Bar Chart ---
        hits = res[pd.to_numeric(res["Latency (Days)"], errors="coerce").notna()].copy()
        if not hits.empty:
            hits["Latency (Days)"] = pd.to_numeric(hits["Latency (Days)"])
            hits = hits.sort_values("Latency (Days)")  # Sort for clean waterfall look

            figh = px.bar(
                hits, 
                x="Latency (Days)", 
                y="Channel", 
                color="Status",
                orientation='h',
                title=f"Detection Latency per Channel (Green Zone = Perfect Hit ±{tol} Days)",
                color_discrete_map={
                    "True Positive (Hit)": "#4CAF50",
                    "False Positive (Fired Too Early)": "#FF9800",
                    "False Negative (Fired Too Late)": "#F44336"
                }
            )
            figh.add_vline(x=0, line_dash="solid", line_color="black", line_width=2)
            # Shade the tolerance window
            figh.add_vrect(x0=-tol, x1=tol, fillcolor="green", opacity=0.1, line_width=0, layer="below")
            
            figh.update_layout(height=max(400, len(hits) * 25), template="plotly_white")
            st.plotly_chart(figh, use_container_width=True)

        st.dataframe(res, use_container_width=True, hide_index=True)

        if st.button("🔁 Restart (clear phase)"):
            st.session_state['bench_phase'] = None
            st.rerun()


 # ---------------------------------------------------------
# 7. The Main UI Function (App Router)
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="RUL Predictor", layout="wide")

    st.sidebar.title("🧭 Navigation")
    app_mode = st.sidebar.radio(
        "Select View:",
        ["Deep-Dive Analysis", 
            "Synthetic Data Studio", 
            "Break Detection Benchmark", 
            "Fleet RUL Benchmark"   # Renamed and moved here
        ]
    )
    st.sidebar.markdown("---")

    st.sidebar.header("📁 Data Input")
    uploaded_file = st.sidebar.file_uploader("Upload Sensor Data (CSV)", type=['csv'])

    if uploaded_file is None:
        st.info("👋 Welcome! Please upload your sensor data CSV in the sidebar to begin.")
        st.stop()

    match = re.search(r'_(\d+)\.csv$', uploaded_file.name, re.IGNORECASE)
    if match:
        st.sidebar.success(f"📍 **Gauge Nr:** {match.group(1)}")
        
    # Parse the CSV once
    base_df = parse_raw_csv(uploaded_file)
    active_df = base_df

    # --- THE DATA SWITCHER ---
    is_synthetic = False   # NEW: tracks whether the synthetic fleet is active
    if 'synthetic_df' in st.session_state:
        st.sidebar.markdown("### 🧬 Data Source")
        data_toggle = st.sidebar.radio("Active Dataset:", ["Original Fleet", "Synthetic Fault Fleet"])
        if data_toggle == "Synthetic Fault Fleet":
            active_df = st.session_state['synthetic_df']
            is_synthetic = True   # NEW

    # =========================================================
    # CHANNEL SELECTOR (Top of Sidebar)
    # =========================================================
    selected_col = None
    if app_mode == "Deep-Dive Analysis":
        st.sidebar.markdown("---")
        st.sidebar.header("🎯 Channel Selector")

        raw_channels = get_available_channels(active_df)

        synth_options = []
        base_options = []
        display_to_col = {}  # Safe dictionary mapping

        for c in raw_channels:
            if c.startswith("INJECT_") or c.startswith("MUTATE_"):
                disp_name = f"{c}"
                display_to_col[disp_name] = c
                synth_options.append(disp_name)
            elif c in ['32', '73']:
                disp_name = f"{c} (Outlier/Deviating)"
                display_to_col[disp_name] = c
                base_options.append(disp_name)
            else:
                display_to_col[c] = c
                base_options.append(c)

        # Sort so Synthetic are ALWAYS at the top, then base options alphabetically
        synth_options.sort()
        base_options.sort()
        display_options = synth_options + base_options

        selected_display = st.sidebar.selectbox("Select Data Channel to Analyze", options=display_options)

        # Safely fetch the exact column name
        selected_col = display_to_col[selected_display]

    # =========================================================
    # SHARED PARAMETERS (Middle of Sidebar)
    # =========================================================
    st.sidebar.markdown("---")
    st.sidebar.header("🛠️ Shared Parameters")
    
    if app_mode in ["Fleet RUL Benchmark", "Break Detection Benchmark"]:
        st.sidebar.warning("⚠️ **Note:** Changing these parameters requires you to click 'Start Simulation' again to see fleet-wide effects.")
    elif app_mode == "Synthetic Data Studio":
        st.sidebar.info("🧬 **Note:** These parameters don't apply to the generation of synthetic data.")
    else:
        st.sidebar.info("⚡ **Live Mode:** Adjusting these parameters will update the charts instantly.")

    st.sidebar.markdown("### Predictive Model Context")
    window_size = st.sidebar.number_input(
        "Window Size (Lookback Days)", min_value=10, max_value=5000, value=300, step=10,
        help="Trailing days used to fit the predictive curves."
    )
    eval_window = st.sidebar.number_input(
        "AIC Evaluation Window (Last N Days)", min_value=1, max_value=window_size, value=min(50, window_size), step=1,
        help=" Recent days used to compute the Tail-Weighted AIC. Unlike simple error (MSE), AIC penalizes complex curves to prevent overfitting. A lower score is better."
    )

    st.sidebar.markdown("### Outlier Filtering")
    outlier_factor = st.sidebar.slider("IQR Outlier Factor", min_value=0.5, max_value=10.0, value=3.0, step=0.1)
    outlier_window = st.sidebar.number_input("Rolling Window (4h Periods)", min_value=5, max_value=200, value=42, step=1)

    st.sidebar.markdown("### Variance Configuration")
    use_dynamic_variance = st.sidebar.toggle(
        "Use Dynamic Variance (Linear Fit)", 
        value=True,
        help="ON: Calculates rate of Std change and projects it into the future. OFF: Uses a fixed Std from the trailing window."
    )
    variance_window = st.sidebar.number_input(
        "Variance Rolling Window (Days)", 
        min_value=2, 
        max_value=100, 
        value=20, 
        step=1,
        help="Trailing days used to calculate the series recend std. A smaller window reacts faster to sudden chatter, while a larger window creates smoother, more stable confidence bands."
    )

    st.sidebar.markdown("### Model Override")
    override_model = st.sidebar.toggle("Enable Manual Selection", value=False)
    manual_model = st.sidebar.selectbox("Force specific model:", options=AVAILABLE_MODELS, disabled=not override_model)
    
    st.sidebar.markdown("### Router Priority Ranking")
    with st.sidebar.expander("Configure Router Ranking", expanded=False):
        st.caption("Drag and drop to set tie-breaker priority (Top = Highest Priority).")
        raw_sorted = sort_items(AVAILABLE_MODELS, direction='vertical')

        # Fallback for Streamlit iframe render bug
        if not raw_sorted:
            sorted_models = AVAILABLE_MODELS
        else:
            sorted_models = raw_sorted

        user_priority_dict = {model: rank for rank, model in enumerate(sorted_models, start=1)}

        
    st.sidebar.markdown("### Structural Break Algorithm")
    break_algo = st.sidebar.radio(
        "Detection Method:",
        ["Exponential vs Linear", "Fleet Z-Score"],
        help=" Choose how the engine detects the onset of degradation."
    )

    if break_algo == "Exponential vs Linear":
        break_window = st.sidebar.number_input(
            "Evaluation Window (Days)", min_value=10, max_value=200, value=60, step=10,
            help=f"The size of the moving window used to compare models. A larger window is more stable against noise but might detect the bend slightly later."
        )
        col_s, col_t = st.sidebar.columns(2)
        with col_s:
            break_step = st.number_input(
                "Step Size (Days)", min_value=1, max_value=30, value=7, step=1,
                help=f"How many days the window slides forward each iteration. Smaller steps offer higher precision at the cost of compute time."
            )
        with col_t:
            break_sustained = st.number_input(
                "Sustained Wins", min_value=1, max_value=10, value=2, step=1,
                help=f"Consecutive times the Exponential curve must beat the Linear line to confirm a true break. Prevents false alarms from sudden noise spikes."
            )
        z_factor = None
        z_sustained = None
    else:
        z_factor = st.sidebar.number_input(
            "Z-Score Multiplier", min_value=1.0, max_value=10.0, value=3.0, step=0.5,
            help=f"The statistical anomaly threshold. A value of 3.0 means the signal must exceed 3 standard deviations above the historical fleet baseline."
        )
        z_sustained = st.sidebar.number_input(
            "Sustained Points (Days)", min_value=1, max_value=20, value=3, step=1,
            help=f"Consecutive days the signal must remain above the Z-Score threshold to be classified as a confirmed structural break."
        )
        break_window = None
        break_step = None
        break_sustained = None
        
    st.sidebar.markdown("### AIC Overfit Prevention")
    min_mse_floor = st.sidebar.number_input(
        "Minimum MSE Floor", 
        min_value=1e-8, 
        max_value=0.1, 
        value=5e-5, 
        step=1e-5,
        format="%.5f",
        help="Clamps the error to this minimum value to prevent the Exponential model from overfitting to microscopic noise on flat signal lines."
    )

    
    max_rul = RUL_HORIZON

    # =========================================================
    # PAGE ROUTER (Main Area)
    # =========================================================
    if app_mode == "Synthetic Data Studio":
        page_synthetic_studio(base_df)

    elif app_mode == "Break Detection Benchmark":
        page_break_benchmark(
            active_df, base_df, is_synthetic, outlier_factor, outlier_window,
            break_algo, break_window, break_step, break_sustained,
            z_factor, z_sustained
        )

    elif app_mode == "Deep-Dive Analysis":
        sensor_arr_smooth, sensor_array_raw, time_arr = load_my_sensor_data(
            active_df, col=selected_col, outlier_factor=outlier_factor, outlier_window=outlier_window
        )

        # --- Dynamic Structural Break Router ---
        if break_algo == "Fleet Z-Score":
            # Anchor to base_df
            fleet_mean, fleet_std = compute_fleet_baseline(base_df, outlier_factor, outlier_window)
            break_idx, break_time, _eval_start_idx = detect_zscore_break(
                time_arr, sensor_arr_smooth, fleet_mean, fleet_std, z_factor, z_sustained, min_mse_floor=min_mse_floor
            )
        else:
            break_idx, break_time, _eval_start_idx = detect_structural_break(
                time_arr, sensor_arr_smooth, window=break_window, step=break_step, sustained_wins=break_sustained, min_mse_floor=min_mse_floor
            )

        max_index = len(time_arr) - 1
        thresholds = [0.2, 0.3]

        st.markdown("### Time Navigation")

        # --- ADAPTIVE SLIDER BOUNDS ---
        slider_min = min(10, max_index)
        default_val = max(slider_min, max_index // 2)

        cutoff_idx = st.slider(
            "Select the Current Time Point (Data Cutoff)",
            min_value=slider_min,
            max_value=max_index,
            value=default_val
        )

        with st.spinner(f"Analyzing models and calculating RUL for index {cutoff_idx}..."):
            start_idx = max(0, cutoff_idx - window_size)
            sliced_time = time_arr[start_idx:cutoff_idx]
            sliced_sensor = sensor_arr_smooth[start_idx:cutoff_idx]
            sliced_sensor_raw = sensor_array_raw[start_idx:cutoff_idx]

            top_models, all_models = evaluate_all_models(
                sliced_time, sliced_sensor, priority_ranking=user_priority_dict,
                eval_window=eval_window, plot=False, verbose=False, min_mse_floor=min_mse_floor
            )

            if not top_models:
                st.error("Error: All models failed to converge. Try increasing the Window Size or selecting a different cutoff.")
                return

            if override_model:
                best_model_name = manual_model
                reuse_params = None
            else:
                best_model_name = list(top_models.keys())[0]
                reuse_params = top_models[best_model_name].get('params')

            fig, fitted_series, rul_df = fit_and_plotly_model(
                time_raw=sliced_time, sensor_smooth=sliced_sensor, sensor_raw=sliced_sensor_raw,
                model_choice=best_model_name, thresholds=thresholds, input_time_unit="Days",
                title_addon=f"| Channel: {selected_col} | Cutoff: {cutoff_idx}",
                max_rul_display=max_rul, use_dynamic_variance=use_dynamic_variance,
                break_time_raw=break_time, precomputed_params=reuse_params,
                variance_window=variance_window
            )

            plot_col, side_metrics_col = st.columns([3, 1])
            with plot_col:
                st.plotly_chart(fig, use_container_width=True)

            with side_metrics_col:
                st.markdown("### 📊 Model Router")

                best_aic_val = min([m.get('aic', float('inf')) for m in all_models.values()])

                leaderboard_data = []
                for rank, (name, metrics) in enumerate(all_models.items(), start=1):
                    is_winner = (name == best_model_name)

                    model_aic = metrics.get('aic', float('inf'))
                    delta_aic = model_aic - best_aic_val if model_aic != float('inf') else float('inf')
                    model_mse = metrics.get('mse', float('inf'))          # <-- NEW

                    leaderboard_data.append({
                        "Rank": ("🏆 Override" if (is_winner and override_model)
                                else "🏆 Algorithm" if is_winner
                                else "🏆 Tied (Ranked lower)" if name in top_models.keys()
                                else str(rank)),
                        "Model": name,
                        "AIC": f"{model_aic:.2f}" if model_aic != float('inf') else "Failed",
                        "Δ AIC": f"+{delta_aic:.2f}" if delta_aic != float('inf') else "N/A",
                        "MSE": (
                            "N/A" if model_mse == float('inf') else 
                            f"{model_mse:.3e} (clamped to {min_mse_floor})" if model_mse < min_mse_floor else 
                            f"{model_mse:.3e}"
                        ),
                    })

                st.dataframe(pd.DataFrame(leaderboard_data), use_container_width=True, hide_index=True)

                st.caption(
                    "**How the Algorithm Chooses:** \n"
                    "The engine uses the **Akaike Information Criterion (AIC)**. "
                    "Models with a **Δ AIC ≤ 2.0** are considered mathematically tied. "
                    "When a tie occurs, the engine defers to your Priority Ranking to select the most physically realistic model."
                )

                st.markdown("---")
                st.markdown("### ⏳ RUL Projections")
                if not rul_df.empty:
                    display_rul_df = rul_df[['Threshold', 'Status', 'Nominal_RUL', 'Upper_Band_RUL', 'Lower_Band_RUL']].copy()

                    def cap_df_rul(val):
                        if val == 'Safe':
                            return 'Safe'
                        if pd.isna(val):
                            return 'Unknown'
                        if isinstance(val, (int, float)):
                            if val == SENTINEL_ALREADY_REACHED or val < 0:
                                return 'Breached'
                            if val > max_rul:
                                return f"> {max_rul}"
                            return round(val, 2)
                        return val

                    for c in ['Nominal_RUL', 'Upper_Band_RUL', 'Lower_Band_RUL']:
                        display_rul_df[c] = display_rul_df[c].apply(cap_df_rul)

                    st.dataframe(display_rul_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No threshold data available for this model fit.")

    elif app_mode == "Fleet RUL Benchmark":
        page_live_simulation(
            active_df, base_df, user_priority_dict, outlier_factor, outlier_window,
            use_dynamic_variance, break_algo, break_window, break_step, break_sustained,
            z_factor, z_sustained, override_model, manual_model,
            window_size, eval_window,variance_window, min_mse_floor
        )


if __name__ == "__main__":
    main()
