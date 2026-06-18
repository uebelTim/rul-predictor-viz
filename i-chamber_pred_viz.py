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

# =========================================================
# GLOBAL CONFIG
# =========================================================
RUL_HORIZON = 365  # days; "Safe" (never reached) and anything beyond this saturate here.


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
def arctan_model(t, L, k, t0, d):
    return L * (np.arctan(k * (t - t0)) / np.pi + 0.5) + d

def gompertz_model(t, a, b, c, d):
    return a * np.exp(-b * np.exp(-c * t)) + d

def rational_model(t, a, b, d):
    return (a * t) / (b + t) + d

def linear_model(t, m, c):
    return m * t + c

def softplus_model(t, a, b, t0, d):
    return a * np.logaddexp(0, b * (t - t0)) + d

def shifted_exponential_model(t, a, b, t0, d):
    return a * np.exp(np.clip(b * (t - t0), -50, 50)) + d

AVAILABLE_MODELS = [
    'Gompertz', 'Arctangent', 'Softplus',
    'Shifted Exponential', 'Rational', 'Linear'
]


# ---------------------------------------------------------
# 1b. CENTRALIZED MODEL CONFIG (single source of truth)
# ---------------------------------------------------------
# ROBUSTNESS FIX: previously the models_config dict was duplicated 3x with
# inconsistent bounds (y_min+0.2 vs y_max+0.2 for the offset `d`). That could make
# the leaderboard winner differ from the actually-fitted curve. Now one builder is
# used everywhere, so bounds are guaranteed identical across all call sites.
def build_models_config(y_min, y_max, y_range):
    d_lo = y_min - 0.2
    d_hi = y_max + 0.2  # use y_max consistently as the upper bound for offset d
    return {
        'Rational': {
            'func': rational_model,
            'p0': [y_range, 0.5, y_min],
            'bounds': ([0.0, 1e-3, d_lo], [np.inf, np.inf, d_hi]),
        },
        'Arctangent': {
            'func': arctan_model,
            'p0': [y_range * 1.1, 20.0, 0.5, y_min],
            'bounds': ([y_range * 0.5, 0.1, 0.0, d_lo],
                       [max(2.0, y_range * 2.2), 500.0, 1.0, d_hi]),
        },
        'Gompertz': {
            'func': gompertz_model,
            'p0': [y_range * 1.1, 1.0, 0.1, y_min],
            'bounds': ([y_range * 0.8, 0.01, 1e-4, d_lo],
                       [max(2.0, y_range * 2.2), 100.0, 50.0, d_hi]),
        },
        'Linear': {
            'func': linear_model,
            'p0': [y_range, y_min],
            'bounds': ([-np.inf, d_lo], [np.inf, d_hi]),
        },
        'Softplus': {
            'func': softplus_model,
            'p0': [y_range * 0.5, 10.0, 0.5, y_min],
            'bounds': ([1e-3, 1e-3, 0.0, d_lo],
                       [y_range * 10.0, 500.0, 1.0, d_hi]),
        },
        'Shifted Exponential': {
            'func': shifted_exponential_model,
            'p0': [y_range * 0.1, 5.0, 0.5, y_min],
            'bounds': ([1e-5, 0.01, 0.0, d_lo],
                       [y_range * 5.0, 50.0, 1.0, d_hi]),
        },
    }


# ---------------------------------------------------------
# 2. Master Fitting Function
# ---------------------------------------------------------
def evaluate_all_models(time_data, sensor_data, priority_ranking, eval_window=None,
                        tolerance_pct=0.5, plot=False, verbose=True, Title_addon="",
                        save_name=None, warm_start=None, maxfev=10000):
    """
    OPT-A: Now stores the fitted `params` for every model so callers can reuse the
    winner's fit instead of re-fitting (avoids the double-fit per timestep).
    OPT-E: Accepts `warm_start` (dict model->params) to seed p0 from the previous
    timestep, which cuts curve_fit iterations dramatically.
    """
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
            # OPT-E: warm-start from previous fit if it's within bounds.
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

            if eval_window is not None and eval_window < len(y_fit):
                mse = mean_squared_error(y_fit[-eval_window:], preds[-eval_window:])
            else:
                mse = mean_squared_error(y_fit, preds)

            results[name] = {'params': params, 'mse': mse, 'func': config['func']}
        except Exception as e:
            results[name] = {'params': None, 'mse': float('inf'),
                             'error': str(e), 'func': config['func']}

    valid_results = {n: d for n, d in results.items() if d['mse'] != float('inf')}
    if not valid_results:
        return {}, results

    best_math_mse = min(d['mse'] for d in valid_results.values())
    # NOTE: tolerance_pct is a MULTIPLIER, not percentage points.
    # 0.5 => "within 50% of the best MSE".
    tolerance_threshold = best_math_mse * (1.0 + tolerance_pct)

    competitive_models = {n: d for n, d in valid_results.items()
                          if d['mse'] <= tolerance_threshold}
    top_models_sorted = dict(sorted(competitive_models.items(),
                                    key=lambda item: priority_ranking.get(item[0], 99)))
    full_leaderboard = dict(sorted(results.items(),
                                   key=lambda item: item[1].get('mse', float('inf'))))

    return top_models_sorted, full_leaderboard


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
                            maxfev=2000, eval_lookback=None):
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
            aic_lin = n * np.log(mse_lin + 1e-10) + 2 * 2
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
            aic_exp = n * np.log(mse_exp + 1e-10) + 2 * 4
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
def compute_fleet_baseline(_file_obj, outlier_factor, outlier_window):
    """
    Computes the global fleet mean and standard deviation.
    The underscore in _file_obj prevents Streamlit from hashing the massive file object.
    """
    channels = get_available_channels(_file_obj)
    all_valid_data = []
    
    for ch in channels:
        sensor_smooth, _, _ = load_my_sensor_data(
            _file_obj, col=ch, outlier_factor=outlier_factor, outlier_window=outlier_window
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

# ---------------------------------------------------------
# 4. Dynamic/Static Variance & RUL Plotting
# ---------------------------------------------------------
def fit_and_plotly_model(time_raw, sensor_smooth, sensor_raw, model_choice, thresholds=None,
                         input_time_unit="Hours", target_time_unit="Days", warm_start_params=None,
                         title_addon="", sigma_factor=1.645, save_name=None, max_rul_display=RUL_HORIZON,
                         use_dynamic_variance=True, break_time_raw=None, precomputed_params=None):
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
    rolling_std = residuals_series.rolling(window=20, min_periods=1).std()
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

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_arr_converted, y=sensor_raw_arr, mode='markers',
                             name='Max Envelope Data', marker=dict(color='gray', size=5, opacity=0.7)))
    fig.add_trace(go.Scatter(x=time_smooth_converted, y=smooth_preds, mode='lines',
                             name=f'{model_choice} Fit', line=dict(color='blue', width=2.5), opacity=0.8))
    fig.add_trace(go.Scatter(x=time_smooth_converted, y=lower_env_smooth, mode='lines',
                             line=dict(width=0), showlegend=False, hoverinfo='skip'))
    band_name = f'±{sigma_factor}σ Confidence Band' + (' (Dynamic)' if use_dynamic_variance else ' (Static)')
    fig.add_trace(go.Scatter(x=time_smooth_converted, y=upper_env_smooth, mode='lines',
                             line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 0, 255, 0.15)', name=band_name))
    fig.add_trace(go.Scatter(x=time_smooth_converted, y=upper_env_smooth, mode='lines',
                             name=f'Upper Band ({risk_pct_upper:.1f}% Risk)',
                             line=dict(color='blue', width=1.5, dash='dashdot'), opacity=0.6))
    fig.add_trace(go.Scatter(x=time_smooth_converted, y=lower_env_smooth, mode='lines',
                             name=f'Lower Band ({risk_pct_lower:.1f}% Risk)',
                             line=dict(color='blue', width=1.5, dash='dot'), opacity=0.4))

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
        title=full_title, xaxis_title=f"Elapsed Time ({target_time_unit})", yaxis_title="Sensor Value",
        xaxis=dict(range=[0, final_end_time]),
        yaxis=dict(range=[y_lower_limit, y_upper_limit], hoverformat=".3f"),
        hovermode="x unified", template="plotly_white",
        legend=dict(title="System Log", yanchor="top", y=1, xanchor="left", x=1.02,
                    bordercolor="LightSteelBlue", borderwidth=1),
        margin=dict(t=120 + (n_thresh * 15)), width=1400, height=800
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.4)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.4)')

    if save_name is not None:
        fig.write_html(save_name)

    return fig, fitted_series, rul_df


# ---------------------------------------------------------
# 5. Data Loading & Extraction Functions
# ---------------------------------------------------------
@st.cache_data
def get_available_channels(file_obj):
    file_obj.seek(0)
    df = pd.read_csv(file_obj, nrows=1)
    cols = [col for col in df.columns if 'Error' not in col and col != 'DateTime']
    if 'Thermo_Valve_Temperature_DeviationPct' in cols:
        cols.remove('Thermo_Valve_Temperature_DeviationPct')
    return [str(c) for c in cols]


@st.cache_data
def load_my_sensor_data(file_obj, col='32', outlier_factor=3.0, outlier_window=42):
    freq = '4h'
    file_obj.seek(0)
    df = pd.read_csv(file_obj, parse_dates=['DateTime'])

    df.set_index('DateTime', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df = df.resample(freq).mean(numeric_only=True)

    cols = [c for c in df.columns if 'Error' not in c]
    if 'Thermo_Valve_Temperature_DeviationPct' in cols:
        cols.remove('Thermo_Valve_Temperature_DeviationPct')

    df_select = df[cols]
    df_select = df_select.interpolate(method='time', limit=1)
    df_select = rolling_iqr_filter(df_select, factor=outlier_factor, window=outlier_window)

    # ROBUSTNESS: explicit .copy() to avoid SettingWithCopyWarning / cache aliasing.
    df_defect = df_select.copy()
    window = int(1 * 24 / 4)

    df_defect[f'{col}_max'] = df_defect[f'{col}'].rolling(window=window * 5, min_periods=1).max()
    df_defect[f'{col}_max_ema'] = df_defect[f'{col}_max'].ewm(span=window * 5, adjust=False, ignore_na=True).mean()

    df_defect_daily = df_defect.resample('D').mean()
    df_defect_daily['elapsed_days'] = (df_defect_daily.index - df_defect_daily.index.min()).days

    return df_defect_daily[f'{col}_max_ema'], df_defect_daily[f'{col}_max'], df_defect_daily['elapsed_days']


# ---------------------------------------------------------
# 6. Headless Math Engine (No Plotly rendering for speed)
# ---------------------------------------------------------
def calculate_rul_headless(time_raw, sensor_smooth, sensor_raw, model_choice, threshold,
                           sigma_factor=1.645, use_dynamic_variance=True, precomputed_params=None):
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
    rolling_std = pd.Series(residuals).rolling(window=20, min_periods=1).std().bfill().fillna(0).values

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
    col_aw, col_hz = st.columns(2)
    with col_aw:
        action_window = st.slider(
            "Action Window (Days)", min_value=5, max_value=90, value=30, step=1,
            help="[Updates Instantly] Tolerance window (± days). A prediction is considered a True Positive if it falls within this margin of the actual RUL."
        )
    with col_hz:
        display_horizon = st.slider(
            "Safe Horizon (Days)", min_value=30, max_value=730, value=RUL_HORIZON, step=5,
            help="[Updates Instantly] The Safe Horizon. If both the predicted and actual RUL exceed this timeframe, it is considered a True Negative."
        )

    # -----------------------------------------------------
    # HORIZON-CAPPED COLUMNS (continuous charts B & C only).
    # NaN actual/predicted ("never reached") -> full horizon so a position exists.
    # -----------------------------------------------------
    df['Nominal_RUL_c'] = df['Nominal_RUL'].apply(lambda x: to_horizon(x, display_horizon))
    df['Actual_RUL_c']  = df['Actual_RUL'].apply(lambda x: to_horizon(x, display_horizon))

    # -----------------------------------------------------
    # CHART A STATUS: Updated Tolerance and Horizon Logic
    # -----------------------------------------------------
    df['Status'] = 'Unknown'

    # Treat NaN as infinite (never reaches threshold) for logical comparisons
    act_val = df['Actual_RUL'].fillna(np.inf)
    pred_val = df['Nominal_RUL'].fillna(np.inf)

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
    error = df['Nominal_RUL_c'] - df['Actual_RUL_c']
    df['Bias_Score'] = np.clip((error + action_window) / (action_window * 2), 0.0, 1.0)

    df['Eval_Day_Rounded'] = df['Evaluation_Day'].round().astype(int)

    # Hover keeps RAW semantics: NaN prediction is explicitly "never reached", not a number.
    def _fmt_pred(x):
        if pd.isna(x):
            return "Never reaches threshold (Safe)"
        return f"{x:.1f} Days"

    df['Hover_Pred'] = df['Nominal_RUL'].apply(_fmt_pred)
    df['Hover_Act'] = df['Actual_RUL'].apply(lambda x: f"{x:.1f} Days" if pd.notna(x) else "Never Breaches")

    pivot_pred = df.pivot_table(index='Channel', columns='Eval_Day_Rounded', values='Hover_Pred', aggfunc='first')
    pivot_act = df.pivot_table(index='Channel', columns='Eval_Day_Rounded', values='Hover_Act', aggfunc='first')

    color_map_status = {"True Positive": "#4CAF50", "False Positive": "#FFB74D",
                        "True Negative": "#E8F5E9", "False Negative": "#F44336"}

    # ==========================================
    # CHART A: Lifecycle Confusion Heatmap
    # ==========================================
    st.subheader("Chart A: Lifecycle Confusion Matrix (Categorical)")
    st.caption(
        f"Each cell shows the model's classification for that day:  \n"
        f"🟩 **True Positive:** The prediction was accurate within the ±{action_window}-day tolerance window.  \n"
        f"⬜ **True Negative:** Both actual and predicted RUL are safely beyond the {display_horizon}-day horizon.  \n"
        f"🟧 **False Positive:** The model alarmed too early (predicted failure earlier than actual).  \n"
        f"🟥 **False Negative:** The model alarmed too late or missed the failure entirely (predicted failure later than actual).  \n"
        f"Use the time axis to identify whether a channel is over-alarming early in its life and stabilising later."
    )

    status_map = {"True Negative": 0, "False Positive": 1, "True Positive": 2, "False Negative": 3}
    df['Status_Code'] = df['Status'].map(status_map)

    pivot_status_code = df.pivot_table(index='Channel', columns='Eval_Day_Rounded', values='Status_Code', aggfunc='first')
    pivot_status_text = df.pivot_table(index='Channel', columns='Eval_Day_Rounded', values='Status', aggfunc='first')
    customdata_a = np.dstack((pivot_status_text.values,
                              pivot_pred.reindex_like(pivot_status_text).values,
                              pivot_act.reindex_like(pivot_status_text).values))

    colorscale_a = [[0.0, '#E8F5E9'], [0.33, '#FFB74D'], [0.66, '#4CAF50'], [1.0, '#F44336']]

    fig_a = go.Figure()
    fig_a.add_trace(go.Heatmap(
        z=pivot_status_code.values, x=pivot_status_code.columns, y=pivot_status_code.index,
        colorscale=colorscale_a, zmin=0, zmax=3, showscale=False, hoverongaps=False, customdata=customdata_a,
        hovertemplate="<b>Day:</b> %{x}<br><b>Channel:</b> %{y}<br><b>Result:</b> %{customdata[0]}<br><b>Predicted:</b> %{customdata[1]}<br><b>Actual:</b> %{customdata[2]}<extra></extra>"
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
        f"Shows the direction of each prediction's error, not only whether it was correct:  \n"
        f"Dark = conservative (predicted failure earlier than it occurred — the safe side).  \n"
        f"Light = optimistic (predicted failure later than it occurred — the risk side).  \n"
        f"Mid-tone = on target.  \n"
        f"The scale saturates at ±{action_window} days (your action window). "
        f"Channels that never cross the threshold are scored against the {display_horizon}-day "
        f"safe horizon, so the algorithm's behaviour remains visible rather than blank."
    )

    rocket_palette = sns.color_palette("mako", n_colors=256).as_hex()
    pivot_bias = df.pivot_table(index='Channel', columns='Eval_Day_Rounded', values='Bias_Score', aggfunc='first')

    df['Hover_Bias'] = df['Bias_Score'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    pivot_bias_text = df.pivot_table(index='Channel', columns='Eval_Day_Rounded', values='Hover_Bias', aggfunc='first')
    customdata_b = np.dstack((pivot_bias_text.reindex_like(pivot_bias).values,
                              pivot_pred.reindex_like(pivot_bias).values,
                              pivot_act.reindex_like(pivot_bias).values))

    fig_b = go.Figure(data=go.Heatmap(
        z=pivot_bias.values, x=pivot_bias.columns, y=pivot_bias.index,
        colorscale=rocket_palette, zmin=0.0, zmax=1.0, showscale=True, hoverongaps=False, customdata=customdata_b,
        colorbar=dict(title="Bias Score", tickvals=[0, 0.5, 1], ticktext=["0.0 (Early)", "0.5", "1.0 (Late)"]),
        hovertemplate="<b>Day:</b> %{x}<br><b>Channel:</b> %{y}<br><b>Bias Score:</b> %{customdata[0]}<br><b>Predicted:</b> %{customdata[1]}<br><b>Actual:</b> %{customdata[2]}<extra></extra>"
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
    
    # 1. Add a toggle to filter the view
    scatter_mode = st.radio(
        "Chart C View Mode:",
        ["Finite predictions only (Recommended)", "All evaluations (Includes Safe/Never reaches)"],
        horizontal=True
    )

    df_scatter = df.copy()

    if scatter_mode == "Finite predictions only (Recommended)":
        # Keep only rows where the algorithm predicted an actual numeric crossing
        df_scatter = df_scatter[df_scatter['Nominal_RUL'].notna()].copy()
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

    # 2. Prepare clean names for the hover tooltip
    df_scatter['Predicted RUL'] = df_scatter['Hover_Pred']
    df_scatter['Actual Outcome'] = df_scatter['Hover_Act']

    # 3. Create the parking lane for "never breaches"
    real_actual_max = df_scatter['Actual_RUL'].max(skipna=True)
    if pd.isna(real_actual_max):
        real_actual_max = display_horizon
    never_pos = real_actual_max * 1.15
    df_scatter['Actual_RUL_plot'] = df_scatter['Actual_RUL'].fillna(never_pos)

    if not df_scatter.empty:
        x_view_max = max(never_pos, action_window * 2) * 1.10
        y_view_max = display_horizon * 1.05

        # 4. Build the scatter, hiding internal variables in the hover_data config
        fig_c = px.scatter(
            df_scatter, x="Actual_RUL_plot", y="Nominal_RUL_c", color="Status",
            hover_data={
                "Actual_RUL_plot": False,  # Hide internal plot value
                "Nominal_RUL_c": False,    # Hide internal capped value
                "Status": False,           # Handled by the color label
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
        
        # Only draw the "Never crosses" parking lane if there are actually NaN values being plotted
        if df_scatter['Actual_RUL'].isna().any():
            fig_c.add_vline(x=never_pos, line_width=1, line_dash="dashdot", line_color="lightgray")
            fig_c.add_annotation(x=never_pos, y=y_view_max, yanchor="top", showarrow=False,
                                 text="Never crosses →", font=dict(size=11, color="gray"),
                                 xshift=-5, xanchor="right")
            
        fig_c.add_hline(y=display_horizon, line_width=1, line_dash="dashdot", line_color="lightgray")

        # Dynamic X-axis title based on the mode
        if scatter_mode == "Finite predictions only (Recommended)" and not df_scatter['Actual_RUL'].isna().any():
            x_title = "Actual outcome (Days)"
        else:
            x_title = "Actual outcome (Days; safe cases grouped at right)"

        # 1. Find the absolute maximum between X and Y so both axes share the same scale
        max_range = max(x_view_max, y_view_max)

        fig_c.update_layout(
            xaxis_title=x_title,
            yaxis_title=f"Predicted RUL (Days, capped @ {display_horizon})",
            
            # 2. Apply the identical range and lock the physical aspect ratio to 1:1
            xaxis=dict(range=[0, max_range]), 
            yaxis=dict(
                range=[0, max_range], 
                scaleanchor="x",  # Locks Y axis physical pixels to X axis
                scaleratio=1      # 1:1 aspect ratio
            ),
            
            width=1200,
            height=800,
            
            # 3. Add explicit symmetric margins to prevent uneven squeezing
            margin=dict(l=80, r=80, t=100, b=80),
            
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Center the square chart nicely on the page
        col_left, col_center, col_right = st.columns([1, 3, 1])
        with col_center:
            st.plotly_chart(fig_c, use_container_width=False)




def page_live_simulation(uploaded_file, priority_dict, outlier_factor, outlier_window,
                         use_dynamic_variance, break_algo, break_window, break_step, break_sustained,
                         z_factor, z_sustained, override_model, manual_model):
    st.title("Fleet-Wide Live Simulation")
    st.markdown("Run the predictive engine across all channels and all historical timesteps to generate statistical confidence metrics.")

    if uploaded_file is None:
        st.warning("Please upload a CSV file in the sidebar to begin.")
        return

    st.markdown("### Simulation Setup")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        req_break = st.toggle("Require Structural Break", value=True,
                              help="[Requires Simulation Re-run] ON: only evaluate AFTER a detected bend. OFF: evaluate from Day 50 onward.")
    with col2:
        step_days = st.number_input("Timestep Interval (Days)", min_value=1, max_value=30, value=7,
                                    help="[Requires Simulation Re-run] Days the simulation jumps between calculations.")
    with col3:
        target_thresh = st.number_input("Target Threshold", min_value=0.1, max_value=5.0, value=0.5, step=0.1,
                                        help="[Requires Simulation Re-run] The critical limit line.")
    with col4:
        ema_span = st.number_input("EMA Smoothing Span", min_value=1, max_value=20, value=1, step=1,
                                   help="[Requires Simulation Re-run] 1 = no smoothing. 4 = moderate smoothing.")

    if 'sim_results' not in st.session_state:
        st.session_state['sim_results'] = None

    if st.button("🚀 Start Fleet Simulation", type="primary", use_container_width=True):
        channels = get_available_channels(uploaded_file)
        results_list = []

        st.markdown("#### Simulation Progress")
        status_text = st.empty()
        progress_bar = st.progress(0.0)
        sub_status_text = st.empty()
        sub_progress_bar = st.progress(0.0)

        total_channels = len(channels)
        UI_THROTTLE = 5  # OPT-F: only refresh sub-status every Nth timestep

        if break_algo == "Fleet Z-Score":
            status_text.markdown("**Pre-computing Fleet Baseline...**")
            fleet_mean, fleet_std = compute_fleet_baseline(uploaded_file, outlier_factor, outlier_window)
        else:
            fleet_mean, fleet_std = 0.0, 1.0
            
        for idx, channel in enumerate(channels):
            status_text.markdown(f"**Overall Fleet Progress:** Processing Channel `{channel}` ({idx + 1} / {total_channels})")
            sub_status_text.markdown(f"Initializing data for Channel `{channel}`...")
            sub_progress_bar.progress(0.0)

            sensor_smooth, sensor_raw, time_arr = load_my_sensor_data(
                uploaded_file, col=channel, outlier_factor=outlier_factor, outlier_window=outlier_window
            )

            time_arr_np = np.asarray(time_arr, dtype=float)
            smooth_np = np.asarray(sensor_smooth, dtype=float)
            raw_np = np.asarray(sensor_raw, dtype=float)  # OPT-F: hoist out of inner loop

            crossing_indices = np.where(smooth_np >= target_thresh)[0]
            actual_crossing_day = time_arr_np[crossing_indices[0]] if len(crossing_indices) > 0 else np.nan

            # Detection point (marker) + look-back evaluation start.
            if break_algo == "Fleet Z-Score":
                break_idx, _break_time, eval_start_idx = detect_zscore_break(
                    time_arr_np, smooth_np, fleet_mean, fleet_std, z_factor, z_sustained
                )
            else:
                break_idx, _break_time, eval_start_idx = detect_structural_break(
                    time_arr_np, smooth_np, window=break_window, step=break_step, sustained_wins=break_sustained
                )

            start_idx = 50
            if req_break and break_idx is not None:
                # Begin evaluation a bit early (run-up), but never before the 50-point floor.
                start_idx = max(50, eval_start_idx)
            elif req_break and break_idx is None:
                sub_status_text.markdown(f"⏭️ *Skipping Channel `{channel}` (No Structural Break detected).*")
                sub_progress_bar.progress(1.0)
                progress_bar.progress((idx + 1) / total_channels)
                continue

            max_idx = len(time_arr)
            timesteps = list(range(start_idx, max_idx, step_days))
            total_steps = len(timesteps)

            if total_steps == 0:
                sub_status_text.markdown(f"⏭️ *Skipping Channel `{channel}` (Not enough historical data).*")
                sub_progress_bar.progress(1.0)
                progress_bar.progress((idx + 1) / total_channels)
                continue

            # OPT-D: manual O(1) running EMA state (no growing pd.Series rebuilds).
            alpha = 2.0 / (ema_span + 1)  # span=1 -> alpha=1 -> passthrough
            ema_n = ema_u = ema_l = None

            # OPT-E: warm-start cache (winner params reused as next-step seed).
            warm_start_cache = None

            # EMA rule:
            #  - real value  -> update EMA normally, store smoothed value
            #  - NaN AND EMA already running (prev not None) -> bridge EMA with RUL_HORIZON (365)
            #    to keep the running value moving, but STORE NaN ("never reached")
            #  - NaN AND EMA not started yet (prev is None) -> skip, store NaN
            def _ema_update(prev, new_raw):
                if np.isnan(new_raw):
                    if prev is None:
                        return prev, np.nan          # leading NaN: skip, store NaN
                    bridged = alpha * float(RUL_HORIZON) + (1 - alpha) * prev
                    return bridged, np.nan           # bridge EMA with 365, store NaN
                if prev is None:
                    return float(new_raw), float(new_raw)  # seed
                smoothed = alpha * float(new_raw) + (1 - alpha) * prev
                return smoothed, smoothed

            for step_idx, current_cutoff in enumerate(timesteps):
                hist_time = time_arr_np[:current_cutoff]
                hist_smooth = smooth_np[:current_cutoff]
                hist_raw = raw_np[:current_cutoff]  # OPT-F: slice of hoisted array

                current_day = hist_time[-1]

                # OPT-F: throttle UI updates
                if step_idx % UI_THROTTLE == 0 or step_idx == total_steps - 1:
                    sub_status_text.markdown(f"↳ Evaluating Timestep **{step_idx + 1} / {total_steps}** (Day {current_day:.1f})")
                    sub_progress_bar.progress((step_idx + 1) / total_steps)

                eval_win = min(50, len(hist_time))

                # OPT-A + OPT-E: single competition pass; reuse winner's params for RUL.
                if override_model:
                    # Override is the rare case: fit ONLY the forced model (no competition).
                    best_model = manual_model
                    mse_log_str = f"{manual_model}: (override - competition skipped)"
                    raw_n, raw_u, raw_l = calculate_rul_headless(
                        hist_time, hist_smooth, hist_raw, best_model, target_thresh,
                        use_dynamic_variance=use_dynamic_variance, precomputed_params=None
                    )
                else:
                    # Normal case: full competition, then REUSE the winner's fitted params.
                    top_models, all_models = evaluate_all_models(
                        hist_time, hist_smooth, priority_ranking=priority_dict,
                        eval_window=eval_win, plot=False, verbose=False,
                        warm_start=warm_start_cache  # OPT-E
                    )

                    mse_log_str = "All models failed"
                    if all_models:
                        mse_log_str = " | ".join(
                            [f"{k}: {v['mse']:.5f}" for k, v in all_models.items()
                             if v.get('mse', float('inf')) != float('inf')]
                        )

                    if not top_models:
                        best_model = "Linear"
                        winner_params = None
                    else:
                        best_model = list(top_models.keys())[0]
                        winner_params = top_models[best_model].get('params')

                    # OPT-E: refresh warm-start cache with all successfully-fitted params.
                    warm_start_cache = {k: v.get('params') for k, v in all_models.items()
                                        if v.get('params') is not None}

                    # OPT-A: pass winner params -> calculate_rul_headless does NOT re-fit.
                    raw_n, raw_u, raw_l = calculate_rul_headless(
                        hist_time, hist_smooth, hist_raw, best_model, target_thresh,
                        use_dynamic_variance=use_dynamic_variance, precomputed_params=winner_params
                    )

                # NaN-preserving EMA (bridges with 365 only after the EMA has started).
                ema_n, store_n = _ema_update(ema_n, raw_n)
                ema_u, store_u = _ema_update(ema_u, raw_u)
                ema_l, store_l = _ema_update(ema_l, raw_l)

                actual_rul = actual_crossing_day - current_day if not np.isnan(actual_crossing_day) else np.nan
                if not np.isnan(actual_rul) and actual_rul < 0:
                    continue  # asset already crossed -> don't log

                results_list.append({
                    'Channel': channel,
                    'Evaluation_Day': current_day,
                    'Model_Used': f"{best_model} (Override)" if override_model else best_model,
                    'Actual_RUL': actual_rul,
                    'Nominal_RUL': store_n,     # NaN when model said "never reached"
                    'Upper_Risk_RUL': store_u,
                    'Lower_Risk_RUL': store_l,
                    'All_Models_MSE': mse_log_str
                })

            progress_bar.progress((idx + 1) / total_channels)

        status_text.success("✅ Fleet Simulation Complete!")
        sub_status_text.empty()
        sub_progress_bar.empty()

        if results_list:
            st.session_state['sim_results'] = pd.DataFrame(results_list)
        else:
            st.warning("No evaluations were run.")

    if st.session_state['sim_results'] is not None:
        generate_simulation_dashboards(st.session_state['sim_results'])
        with st.expander("View Raw Simulation Logs"):
            st.dataframe(st.session_state['sim_results'], use_container_width=True)



# ---------------------------------------------------------
# 7. The Main UI Function (App Router)
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="RUL Predictor", layout="wide")

    st.sidebar.title("🧭 Navigation")
    app_mode = st.sidebar.radio("Select View:", ["Deep-Dive Analysis", "Live Fleet Simulation"])
    st.sidebar.markdown("---")

    st.sidebar.header("📁 Data Input")
    uploaded_file = st.sidebar.file_uploader("Upload Sensor Data (CSV)", type=['csv'])

    if uploaded_file is None:
        st.info("👋 Welcome! Please upload your sensor data CSV in the sidebar to begin.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.header("🛠️ Shared Parameters")

    st.sidebar.markdown("### Outlier Filtering")
    outlier_factor = st.sidebar.slider("IQR Outlier Factor", min_value=0.5, max_value=10.0, value=3.0, step=0.1,
                                       help="[Requires Simulation Re-run] How aggressively to flatten spikes.")
    outlier_window = st.sidebar.number_input("Rolling Window (4h Periods)", min_value=5, max_value=200, value=42, step=1,
                                             help="[Requires Simulation Re-run] 42 periods = 7 days of local context.")

    st.sidebar.markdown("### Variance Configuration")
    use_dynamic_variance = st.sidebar.toggle("Use Dynamic Variance (Linear Fit)", value=True,
                                             help="[Requires Simulation Re-run] ON: bands widen over time. OFF: parallel bands.")

    st.sidebar.markdown("### Model Override")
    override_model = st.sidebar.toggle("Enable Manual Selection", value=False,
                                       help="[Requires Simulation Re-run] Forces a specific model, ignoring the auto recommendation.")
    manual_model = st.sidebar.selectbox("Force specific model:", options=AVAILABLE_MODELS, disabled=not override_model)

    st.sidebar.markdown("### Structural Break Algorithm")
    break_algo = st.sidebar.radio(
        "Detection Method:", 
        ["Exponential vs Linear", "Fleet Z-Score"],
        help="[Requires Simulation Re-run] Choose how the engine detects the onset of degradation."
    )
    
    if break_algo == "Exponential vs Linear":
        break_window = st.sidebar.number_input("Evaluation Window (Days)", min_value=10, max_value=200, value=60, step=10)
        col_s, col_t = st.sidebar.columns(2)
        with col_s:
            break_step = st.number_input("Step Size (Days)", min_value=1, max_value=30, value=7, step=1)
        with col_t:
            break_sustained = st.number_input("Sustained Wins", min_value=1, max_value=10, value=2, step=1)
            
        z_factor = None # Not used
        z_sustained = None # Not used
    else:
        z_factor = st.sidebar.number_input("Z-Score Multiplier", min_value=1.0, max_value=10.0, value=3.0, step=0.5,
                                           help="How many standard deviations above the fleet mean defines a break.")
        z_sustained = st.sidebar.number_input("Sustained Points (Days)", min_value=1, max_value=20, value=3, step=1,
                                              help="Consecutive days the value must stay above the Z-score limit.")
        
        break_window = None
        break_step = None
        break_sustained = None

    st.sidebar.markdown("### Router Priority Ranking")
    with st.sidebar.expander("Configure Router Ranking", expanded=False):
        st.caption("Drag and drop to set tie-breaker priority (Top = Highest Priority).")
        sorted_models = sort_items(AVAILABLE_MODELS, direction='vertical')
        user_priority_dict = {model: rank for rank, model in enumerate(sorted_models, start=1)}

    # ROBUSTNESS: single horizon constant shared by both pages (no floating magic number).
    max_rul = RUL_HORIZON

    # =========================================================
    # PAGE 1: DEEP-DIVE ANALYSIS
    # =========================================================
    if app_mode == "Deep-Dive Analysis":
        st.sidebar.markdown("---")
        st.sidebar.header("🔍 Deep-Dive Settings")

        raw_channels = get_available_channels(uploaded_file)
        display_options = []
        for c in raw_channels:
            if c in ['32', '73']:
                display_options.append(f"{c} (Outlier/Deviating)")
            else:
                display_options.append(c)

        selected_display = st.sidebar.selectbox("1. Select Data Channel", options=display_options)
        selected_col = selected_display.split(" ")[0]

        window_size = st.sidebar.number_input("2. Window Size (Lookback Days)", min_value=10, max_value=5000, value=300, step=10,
                                              help="[Updates Instantly] Trailing days used for the predictive curves.")
        eval_window = st.sidebar.number_input("3. MSE Eval Window (Last N Days)", min_value=1, max_value=window_size,
                                              value=min(50, window_size), step=1,
                                              help="[Updates Instantly] Recent days used to score/rank models.")

        sensor_arr_smooth, sensor_array_raw, time_arr = load_my_sensor_data(
            uploaded_file, col=selected_col, outlier_factor=outlier_factor, outlier_window=outlier_window
        )

        # --- NEW ROUTER LOGIC ---
        if break_algo == "Fleet Z-Score":
            fleet_mean, fleet_std = compute_fleet_baseline(uploaded_file, outlier_factor, outlier_window)
            break_idx, break_time, _eval_start_idx = detect_zscore_break(
                time_arr, sensor_arr_smooth, fleet_mean, fleet_std, z_factor, z_sustained
            )
        else:
            break_idx, break_time, _eval_start_idx = detect_structural_break(
                time_arr, sensor_arr_smooth, window=break_window, step=break_step, sustained_wins=break_sustained
            )


        max_index = len(time_arr) - 1
        thresholds = [0.2, 0.5, 1.0]

        st.markdown("### Time Navigation")
        cutoff_idx = st.slider("Select the Current Time Point (Data Cutoff)", min_value=10, max_value=max_index, value=max_index // 2)

        with st.spinner(f"Analyzing models and calculating RUL for index {cutoff_idx}..."):
            start_idx = max(0, cutoff_idx - window_size)
            sliced_time = time_arr[start_idx:cutoff_idx]
            sliced_sensor = sensor_arr_smooth[start_idx:cutoff_idx]
            sliced_sensor_raw = sensor_array_raw[start_idx:cutoff_idx]

            top_models, all_models = evaluate_all_models(
                sliced_time, sliced_sensor, priority_ranking=user_priority_dict,
                eval_window=eval_window, plot=False, verbose=False
            )

            if not top_models:
                st.error("Error: All models failed to converge. Try increasing the Window Size or selecting a different cutoff.")
                return

            if override_model:
                best_model_name = manual_model
                reuse_params = None  # forced model wasn't necessarily fit in competition
            else:
                best_model_name = list(top_models.keys())[0]
                reuse_params = top_models[best_model_name].get('params')  # OPT-A reuse

            # OPT-A: pass winner params so fit_and_plotly_model doesn't re-fit.
            fig, fitted_series, rul_df = fit_and_plotly_model(
                time_raw=sliced_time, sensor_smooth=sliced_sensor, sensor_raw=sliced_sensor_raw,
                model_choice=best_model_name, thresholds=thresholds, input_time_unit="Days",
                title_addon=f"| Channel: {selected_col} | Cutoff: {cutoff_idx}",
                max_rul_display=max_rul, use_dynamic_variance=use_dynamic_variance,
                break_time_raw=break_time, precomputed_params=reuse_params
            )

            plot_col, side_metrics_col = st.columns([3, 1])
            with plot_col:
                st.plotly_chart(fig, use_container_width=True)

            with side_metrics_col:
                st.markdown("### 📊 Model Router")
                leaderboard_data = []
                for rank, (name, metrics) in enumerate(all_models.items(), start=1):
                    is_winner = (name == best_model_name)
                    leaderboard_data.append({
                        "Rank": ("🏆 Override" if (is_winner and override_model)
                                 else "🏆 Algorithm" if is_winner
                                 else "🏆" if name in top_models.keys()
                                 else str(rank)),
                        "Model": name,
                        "MSE": f"{metrics['mse']:.5f}"
                    })
                st.dataframe(pd.DataFrame(leaderboard_data), use_container_width=True, hide_index=True)

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

    # =========================================================
    # PAGE 2: LIVE FLEET SIMULATION
    # =========================================================
    elif app_mode == "Live Fleet Simulation":
        page_live_simulation(
            uploaded_file, user_priority_dict, outlier_factor, outlier_window,
            use_dynamic_variance, break_window, break_step, break_sustained, override_model, manual_model,
            break_algo, z_factor, z_sustained
        )


if __name__ == "__main__":
    main()
