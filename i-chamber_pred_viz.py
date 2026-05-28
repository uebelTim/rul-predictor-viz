from scipy.optimize import curve_fit, fsolve
from sklearn.metrics import mean_squared_error
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_sortables import sort_items 
#import diagnostic_utils as du 

def filter_outliers_quantile(df, factor=1.5, keep_nans=True):
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    original_nans = df_numeric.isna()
    
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (factor * IQR)
    upper_bound = Q3 + (factor * IQR)
    
    mask = (df_numeric >= lower_bound) & (df_numeric <= upper_bound)
    df_filtered = df_numeric.where(mask, np.nan)
    df_filtered = df_filtered.interpolate(method='linear', limit_direction='both')
    
    if df_filtered.isnull().values.any():
        df_filtered = df_filtered.interpolate(method='linear', axis=1, limit_direction='both')
        
    if keep_nans:
        df_filtered[original_nans] = np.nan
        
    return df_filtered

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
# 2. Master Fitting Function
# ---------------------------------------------------------
def evaluate_all_models(time_data, sensor_data, priority_ranking, eval_window=None, tolerance_pct=0.5, plot=False, verbose=True, Title_addon="", save_name=None):
    time_arr = np.asarray(time_data)
    sensor_arr = np.asarray(sensor_data)
    
    t_max = np.max(time_arr)
    time_norm = time_arr / t_max if t_max > 0 else time_arr
    
    y_min = np.min(sensor_arr)
    y_max = np.max(sensor_arr)
    y_range = y_max - y_min
    
    models_config = {
        'Rational': {
            'func': rational_model,
            'p0': [y_range, 0.5, y_min],
            'bounds': ([0.0, 1e-3, y_min - 0.2], [np.inf, np.inf, y_min + 0.2])
        },
        'Arctangent': {
            'func': arctan_model,
            'p0': [y_range * 1.1, 20.0, 0.5, y_min],
            'bounds': ([y_range * 0.5, 0.1, 0.0, y_min - 0.2], [max(2.0, y_range * 2.2), 500.0, 1.0, y_min + 0.2])
        },
        'Gompertz': {
            'func': gompertz_model,
            'p0': [y_range * 1.1, 1.0, 0.1, y_min],
            'bounds': (
                [y_range * 0.8, 0.01, 1e-4, y_min - 0.2], 
                [max(2.0, y_range * 2.2), 100.0, 50.0, y_min + 0.2])
            },
        'Linear': {
            'func': linear_model,
            'p0': [y_range, y_min],
            'bounds': ([-np.inf, y_min - 0.2], [np.inf, y_min + 0.2])
        },
        'Softplus': {
            'func': softplus_model,
            'p0': [y_range * 0.5, 10.0, 0.5, y_min],
            'bounds': (
                [1e-3, 1e-3, 0.0, y_min - 0.2], 
                [y_range * 10.0, 500.0, 1.0, y_min + 0.2]
            )
        },
        'Shifted Exponential': {
            'func': shifted_exponential_model,
            'p0': [y_range * 0.1, 5.0, 0.5, y_min],
            'bounds': (
                [1e-5, 0.01, 0.0, y_min - 0.2], 
                [y_range * 5.0, 50.0, 1.0, y_min + 0.2]
            )
        },
    }
    
    results = {}
    
    for name, config in models_config.items():
        try:
            params, _ = curve_fit(config['func'], time_norm, sensor_arr, p0=config['p0'], bounds=config['bounds'], maxfev=10000)
            preds = config['func'](time_norm, *params)
            
            if eval_window is not None and eval_window < len(sensor_arr):
                mse = mean_squared_error(sensor_arr[-eval_window:], preds[-eval_window:])
            else:
                mse = mean_squared_error(sensor_arr, preds)
                
            results[name] = {'params': params, 'mse': mse, 'func': config['func']}
        except Exception as e:
            results[name] = {'mse': float('inf'), 'error': str(e), 'func': config['func']}

    valid_results = {name: data for name, data in results.items() if data['mse'] != float('inf')}
    
    if not valid_results:
        return {}, results

    best_math_mse = min(data['mse'] for data in valid_results.values())
    tolerance_threshold = best_math_mse * (1.0 + tolerance_pct)
    
    competitive_models = {name: data for name, data in valid_results.items() if data['mse'] <= tolerance_threshold}
    top_models_sorted = dict(sorted(competitive_models.items(), key=lambda item: priority_ranking.get(item[0], 99)))
    full_leaderboard = dict(sorted(results.items(), key=lambda item: item[1].get('mse', float('inf'))))

    return top_models_sorted, full_leaderboard


# ---------------------------------------------------------
# 3. Structural Break Detector (Rolling Slope / Velocity)
# ---------------------------------------------------------
def detect_structural_break(time_arr, sensor_arr, window=21, threshold_pct=5.0, sustained_points=3):
    """
    Detects a structural break by tracking the rolling linear slope (velocity) of the data.
    Triggers when the upward slope is steep enough to cover 'threshold_pct' of the 
    total data range within a standardized 30-day period, sustained over several days.
    """
    time_arr = np.asarray(time_arr, dtype=float)
    sensor_arr = np.asarray(sensor_arr, dtype=float)
    
    if len(sensor_arr) < window + sustained_points:
        return None, None

    # 1. Lightly smooth the max envelope to prevent single-day noise from warping the slope
    smooth_sensor = pd.Series(sensor_arr).ewm(span=5, adjust=False).mean().values
    
    # 2. Determine the physical scale of the data
    data_range = np.max(sensor_arr) - np.min(sensor_arr)
    if data_range == 0:
        return None, None
        
    # Calculate the absolute slope required to trigger a break.
    # (e.g., slope needed to rise 5% of the total range over 30 days)
    required_slope = (data_range * (threshold_pct / 100.0)) / 30.0
    
    # 3. Calculate rolling slopes
    slopes = np.zeros(len(smooth_sensor))
    for i in range(window, len(smooth_sensor)):
        t_win = time_arr[i-window : i]
        y_win = smooth_sensor[i-window : i]
        
        # Calculate slope (m) for this window
        m, _ = np.polyfit(t_win, y_win, 1)
        slopes[i] = m

    # 4. Find where the slope exceeds the threshold for 'sustained_points' consecutive days
    consecutive_count = 0
    break_idx = None
    
    # Start looking only after the first window has fully formed
    for i in range(window, len(slopes)):
        if slopes[i] >= required_slope:
            consecutive_count += 1
            if consecutive_count >= sustained_points:
                # The actual break started 'sustained_points' ago when it first crossed
                break_idx = i - sustained_points + 1
                break
        else:
            consecutive_count = 0 # Reset if the slope dips back down
            
    if break_idx is not None:
        return break_idx, time_arr[break_idx]
        
    return None, None

# ---------------------------------------------------------
# 4. Dynamic/Static Variance & RUL Plotting
# ---------------------------------------------------------
def fit_and_plotly_model(time_raw, sensor_smooth, sensor_raw, model_choice, thresholds=None, 
                         input_time_unit="Hours", target_time_unit="Days", warm_start_params=None, 
                         title_addon="", sigma_factor=1.645, save_name=None, max_rul_display=365,
                         use_dynamic_variance=True, break_time_raw=None):
    if isinstance(sensor_smooth, pd.Series):
        orig_index = sensor_smooth.index
    elif isinstance(time_raw, pd.Series):
        orig_index = time_raw.index
    else:
        orig_index = range(len(sensor_smooth))
        
    time_arr = np.asarray(time_raw, dtype=float)
    sensor_arr = np.asarray(sensor_smooth, dtype=float)
    
    raw_current_time = np.max(time_arr)
    t_max = raw_current_time if raw_current_time > 0 else 1.0
    time_norm = time_arr / t_max
    
    y_min = np.min(sensor_arr)
    y_max = np.max(sensor_arr)
    y_range = y_max - y_min
    
    to_hours = {"Seconds": 1/3600, "Minutes": 1/60, "Hours": 1.0, "Days": 24.0, "2H": 2.0, "8H": 8.0}
    conversion_factor = to_hours[input_time_unit] / to_hours[target_time_unit]
    
    models_config = {
        'Rational': {'func': rational_model, 'p0': [y_range, 0.5, y_min], 'bounds': ([0.0, 1e-3, y_min - 0.2], [np.inf, np.inf, y_min + 0.2])},
        'Arctangent': {'func': arctan_model, 'p0': [y_range * 1.1, 20.0, 0.5, y_min], 'bounds': ([y_range * 0.5, 0.1, 0.0, y_min - 0.2], [max(2.0, y_range * 2.2), 500.0, 1.0, y_min + 0.2])},
        'Linear': {'func': linear_model, 'p0': [y_range, y_min], 'bounds': ([-np.inf, y_min - 0.2], [np.inf, y_min + 0.2])},
        'Gompertz': {'func': gompertz_model, 'p0': [y_range * 1.1, 1.0, 0.1, y_min], 'bounds': ([y_range * 0.8, 0.01, 1e-4, y_min - 0.2], [max(2.0, y_range * 2.2), 100.0, 50.0, y_min + 0.2])},
        'Softplus': {'func': softplus_model, 'p0': [y_range * 0.5, 10.0, 0.5, y_min], 'bounds': ([1e-3, 1e-3, 0.0, y_min - 0.2], [y_range * 10.0, 500.0, 1.0, y_min + 0.2])},
        'Shifted Exponential': {'func': shifted_exponential_model, 'p0': [y_range * 0.1, 5.0, 0.5, y_min], 'bounds': ([1e-5, 0.01, 0.0, y_min - 0.2], [y_range * 5.0, 50.0, 1.0, y_min + 0.2])},
    }
    config = models_config[model_choice]
    
    if warm_start_params is not None:
        try:
            config['p0'] = np.clip(warm_start_params, config['bounds'][0], config['bounds'][1])
        except Exception:
            pass

    try:
        params, _ = curve_fit(config['func'], time_norm, sensor_arr, p0=config['p0'], bounds=config['bounds'], maxfev=10000)
    except Exception as e:
        print(f"Error: {model_choice} model failed to converge. {e}")
        return pd.Series(np.nan, index=orig_index), pd.DataFrame()

    fitted_series = pd.Series(config['func'](time_norm, *params), index=orig_index, name=f"{model_choice}_Fit")
    residuals_series = pd.Series(sensor_raw - fitted_series, index=orig_index, name=f"{model_choice}_Residuals")
    
    # ---------------------------------------------------------
    # VARIANCE CALCULATION
    # ---------------------------------------------------------
    rolling_std = residuals_series.rolling(window=20, min_periods=1).std()
    rolling_std = rolling_std.bfill().fillna(0) 

    valid_mask = ~np.isnan(rolling_std)
    
    if use_dynamic_variance and valid_mask.sum() > 1:
        std_slope, std_intercept = np.polyfit(time_norm[valid_mask], rolling_std[valid_mask], 1)
    else:
        std_slope, std_intercept = 0.0, rolling_std.iloc[-1]

    def get_dynamic_std(t_norm_val):
        return np.maximum(0.0, std_slope * t_norm_val + std_intercept)

    current_margin = get_dynamic_std(1.0) * sigma_factor
    cdf = 0.5 * (1.0 + math.erf(sigma_factor / math.sqrt(2)))
    risk_pct_upper = (1.0 - cdf) * 100
    risk_pct_lower = cdf * 100

    # ---------------------------------------------------------
    # RUL ROOT FINDING
    # ---------------------------------------------------------
    def solve_for_dynamic_t(target_val, mode='nominal'):
        t_infinite = 10000.0 
        ceiling_base = config['func'](t_infinite, *params)
        
        if mode == 'upper':
            ceiling_val = ceiling_base + get_dynamic_std(t_infinite) * sigma_factor
        elif mode == 'lower':
            ceiling_val = ceiling_base - get_dynamic_std(t_infinite) * sigma_factor
        else:
            ceiling_val = ceiling_base

        if ceiling_val < target_val:
            return 'Safe' 

        def target_equation(t_guess):
            base_val = config['func'](t_guess, *params)
            if mode == 'upper':
                return (base_val + get_dynamic_std(t_guess) * sigma_factor) - target_val
            elif mode == 'lower':
                return (base_val - get_dynamic_std(t_guess) * sigma_factor) - target_val
            else:
                return base_val - target_val

        try:
            t_solution = fsolve(target_equation, 1.0)[0]
            
            if abs(target_equation(t_solution)) > 1e-3:
                if target_equation(1.0) >= 0:
                    return -999.0 
                return None 
                
            return t_solution * t_max
        except Exception:
            try:
                if target_equation(1.0) >= 0: return -999.0
            except: pass
            return None

    # ---------------------------------------------------------
    # RECORD GENERATION
    # ---------------------------------------------------------
    rul_records = []
    plot_end_time_raw = raw_current_time * 1.2 
    
    if thresholds is not None:
        if not isinstance(thresholds, (list, tuple, np.ndarray)):
            thresholds = [thresholds]
            
        for thresh in sorted(thresholds):
            nom_time = solve_for_dynamic_t(thresh, mode='nominal')
            upper_time = solve_for_dynamic_t(thresh, mode='upper') 
            lower_time = solve_for_dynamic_t(thresh, mode='lower') 
            
            def calc_rul(t_val):
                if t_val == 'Safe': return 'Safe'
                return (t_val - raw_current_time) * conversion_factor if isinstance(t_val, (float, int)) else np.nan
                
            def calc_abs(t_val):
                if t_val == 'Safe': return np.nan
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
            
            valid_times = [t for t in [nom_abs, upper_abs, lower_abs] if not np.isnan(t)]
            if valid_times:
                plot_end_time_raw = max(plot_end_time_raw, max(valid_times) / conversion_factor * 1.1)

            rul_records.append({
                'Threshold': thresh,
                'Status': status,
                'Current_Margin': current_margin,
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
    
    smooth_preds = config['func'](time_smooth_norm, *params)
    dynamic_std_smooth = get_dynamic_std(time_smooth_norm)
    
    upper_env_smooth = smooth_preds + (dynamic_std_smooth * sigma_factor)
    lower_env_smooth = smooth_preds - (dynamic_std_smooth * sigma_factor)
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_arr_converted, y=sensor_raw, mode='markers', name='Max Envelope Data',
        marker=dict(color='gray', size=5, opacity=0.7)
    ))

    fig.add_trace(go.Scatter(
        x=time_smooth_converted, y=smooth_preds, mode='lines', name=f'{model_choice} Fit',
        line=dict(color='blue', width=2.5), opacity=0.8
    ))

    fig.add_trace(go.Scatter(
        x=time_smooth_converted, y=lower_env_smooth, mode='lines', 
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))

    band_name = f'±{sigma_factor}σ Confidence Band' + (' (Dynamic)' if use_dynamic_variance else ' (Static)')
    fig.add_trace(go.Scatter(
        x=time_smooth_converted, y=upper_env_smooth, mode='lines', 
        line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 0, 255, 0.15)', 
        name=band_name
    ))

    fig.add_trace(go.Scatter(
        x=time_smooth_converted, y=upper_env_smooth, mode='lines', name=f'Upper Band ({risk_pct_upper:.1f}% Risk)',
        line=dict(color='blue', width=1.5, dash='dashdot'), opacity=0.6
    ))
    fig.add_trace(go.Scatter(
        x=time_smooth_converted, y=lower_env_smooth, mode='lines', name=f'Lower Band ({risk_pct_lower:.1f}% Risk)',
        line=dict(color='blue', width=1.5, dash='dot'), opacity=0.4
    ))
    
    # ---------------------------------------------------------
    # ANNOTATIONS & LEGEND / TITLES
    # ---------------------------------------------------------
    unit_short = {"Seconds": "s", "Minutes": "m", "Hours": "h", "Days": "d"}.get(target_time_unit, target_time_unit)
    colors = ['#FFA500', '#FF0000', '#8B0000', '#800080', '#000000']
    status_lines = []
    
    def format_rul(val):
        if val == 'Safe': return "Safe"
        if pd.isna(val): return "Unknown"
        if isinstance(val, (float, int)):
            if val < 0: return "Breached"
            if val > max_rul_display: return f"> {max_rul_display}{unit_short}"
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
            
            fig.add_trace(go.Scatter(
                x=[0, final_end_time], y=[thresh, thresh], mode='lines', name=legend_lbl,
                line=dict(color=c, width=2, dash='dash'), opacity=0.6
            ))
            
            if not pd.isna(row['Nominal_Abs_Time']) and row['Nominal_Abs_Time'] <= final_end_time:
                fig.add_trace(go.Scatter(
                    x=[row['Nominal_Abs_Time']], y=[thresh], mode='markers', showlegend=False,
                    marker=dict(symbol='circle', color=c, size=10, line=dict(color='black', width=1))
                ))
            if not pd.isna(row['Upper_Abs_Time']) and row['Upper_Abs_Time'] <= final_end_time:
                fig.add_trace(go.Scatter(
                    x=[row['Upper_Abs_Time']], y=[thresh], mode='markers', showlegend=False,
                    marker=dict(symbol='triangle-left', color=c, size=10, line=dict(color='black', width=1))
                ))
            if not pd.isna(row['Lower_Abs_Time']) and row['Lower_Abs_Time'] <= final_end_time:
                fig.add_trace(go.Scatter(
                    x=[row['Lower_Abs_Time']], y=[thresh], mode='markers', showlegend=False,
                    marker=dict(symbol='triangle-right', color=c, size=10, line=dict(color='black', width=1))
                ))

    # --- Structural Break Visuals ---
    break_title_str = ""
    if break_time_raw is not None:
        break_time_converted = break_time_raw * conversion_factor
        fig.add_vline(x=break_time_converted, line_width=2, line_dash="dash", line_color="orange")
        fig.add_annotation(
            x=break_time_converted, y=np.max(sensor_arr) * 1.05 if np.max(sensor_arr) > 0 else 0,
            text="⚠️ Structural Break", showarrow=False,
            yshift=10, font=dict(color="orange", size=12)
        )
        break_title_str = f" | <span style='color:orange;'><b>⚠️ Break Detected at {break_time_converted:.1f}{unit_short}</b></span>"
    else:
        break_title_str = f" | <span style='color:green;'><b>✅ Stable (No Break)</b></span>"

    # Axis Limits
    max_target = max(thresholds) if thresholds is not None else 0
    absolute_y_max = max(np.max(sensor_arr), max_target)
    absolute_y_min = min(0, np.min(sensor_arr))
    y_padding = (absolute_y_max - absolute_y_min) * 0.15
    y_upper_limit = absolute_y_max + y_padding
    y_lower_limit = absolute_y_min - (y_padding if absolute_y_min < 0 else 0)

    # Compile Title
    main_title = f"Predictive Analytics | Engine: {model_choice} {title_addon}{break_title_str}"
    subtitle_html = "<br>".join([f"<span style='font-size:14px; color:gray;'>{line}</span>" for line in status_lines])
    full_title = f"<b>{main_title}</b><br>{subtitle_html}"

    fig.update_layout(
        title=full_title,
        xaxis_title=f"Elapsed Time ({target_time_unit})",
        yaxis_title="Sensor Value",
        xaxis=dict(range=[0, final_end_time]),
        yaxis=dict(range=[y_lower_limit, y_upper_limit], hoverformat=".3f"),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(title="System Log", yanchor="top", y=1, xanchor="left", x=1.02, bordercolor="LightSteelBlue", borderwidth=1),
        margin=dict(t=120 + (len(thresholds)*15)), 
        width=1400, height=800
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
def calculate_global_baseline_slope(file_obj, baseline_pct=0.20):
    """
    Calculates the universal 'healthy' slope by averaging the first 20% 
    of all available columns (since they are identical component types).
    """
    file_obj.seek(0)
    df = pd.read_csv(file_obj, parse_dates=['DateTime'])
    df.set_index('DateTime', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    df_daily = df.resample('D').mean(numeric_only=True)
    
    cols = [c for c in df_daily.columns if 'Error' not in c and c != 'Thermo_Valve_Temperature_DeviationPct']
    global_mean = df_daily[cols].mean(axis=1).bfill().ffill()
    
    n_base = max(5, int(len(global_mean) * baseline_pct))
    y_base = global_mean.iloc[:n_base].values
    
    t_base = (global_mean.index[:n_base] - global_mean.index[0]).days.values
    
    if len(t_base) > 1:
        slope, _ = np.polyfit(t_base, y_base, 1)
    else:
        slope = 0.0
        
    return slope


@st.cache_data
def load_my_sensor_data(file_obj, col='32'):
    freq = '4h'
    column_names=[]
    
    file_obj.seek(0)
    df = pd.read_csv(file_obj, parse_dates=['DateTime'])
        
    df.set_index('DateTime', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df = df.resample(freq).mean(numeric_only=True)
    
    cols = [c for c in df.columns if 'Error' not in c]
    if 'Thermo_Valve_Temperature_DeviationPct' in cols:
        cols.remove('Thermo_Valve_Temperature_DeviationPct')
    column_names.extend(cols)
    
    df_select = df[cols]
    df_select = df_select.interpolate(method='time')
    df_select = df_select.ffill().bfill()
    
    try:
        df_select = filter_outliers_quantile(df_select, factor=3)
    except NameError:
        pass 
        
    df_defect = df_select
    window = int(1*24/4) 
    
    df_defect[f'{col}_max'] = df_defect[f'{col}'].rolling(window=window*5).max()
    df_defect[f'{col}_max_ema'] = df_defect[f'{col}_max'].ewm(span=window*5, adjust=False).mean()

    df_defect_daily = df_defect.resample('D').mean()
    df_defect_daily = df_defect_daily.bfill().ffill()
    df_defect_daily['elapsed_days'] = (df_defect_daily.index - df_defect_daily.index.min()).days
    
    return df_defect_daily[f'{col}_max_ema'], df_defect_daily[f'{col}_max'], df_defect_daily['elapsed_days']

# ---------------------------------------------------------
# 6. The Main UI Function
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="RUL Predictor", layout="wide")
    st.title("Predictive Maintenance: Dynamic RUL Engine")

    st.sidebar.header("📁 Data Input")
    uploaded_file = st.sidebar.file_uploader("Upload Sensor Data (CSV)", type=['csv'])

    if uploaded_file is None:
        st.info("👋 Welcome! Please upload your sensor data CSV in the sidebar to begin.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.header("🛠️ Algorithm Parameters")

    raw_channels = get_available_channels(uploaded_file)
    display_options = []
    for c in raw_channels:
        if c in ['32', '73']:
            display_options.append(f"{c} (Outlier/Deviating)")
        else:
            display_options.append(c)

    selected_display = st.sidebar.selectbox("1. Select Data Channel", options=display_options)
    selected_col = selected_display.split(" ")[0]

    window_size = st.sidebar.number_input("2. Window Size (Lookback)", min_value=10, max_value=5000, value=300, step=10)
    eval_window = st.sidebar.number_input("3. MSE Eval Window (Last N points)", min_value=1, max_value=window_size, value=min(50, window_size), step=1)

    st.sidebar.markdown("### 4. Model Override")
    override_model = st.sidebar.toggle("Enable Manual Selection", value=False)
    manual_model = st.sidebar.selectbox("Force specific model:", options=AVAILABLE_MODELS, disabled=not override_model)

    st.sidebar.markdown("### 5. Router Priority Ranking")
    with st.sidebar.expander("Configure Router Ranking", expanded=False):
        sorted_models = sort_items(AVAILABLE_MODELS, direction='vertical')
        user_priority_dict = {model: rank for rank, model in enumerate(sorted_models, start=1)}

    st.sidebar.markdown("### 6. Display Limits")
    max_rul = st.sidebar.number_input("Max RUL Cap (Days)", min_value=1, max_value=5000, value=365)
    
    st.sidebar.markdown("### 7. Variance Configuration")
    use_dynamic_variance = st.sidebar.toggle("Use Dynamic Variance (Linear Fit)", value=True, help="If off, uses a static variance (the last recorded standard deviation) across the entire future curve.")

    # NEW: Rolling Slope UI Parameters
    st.sidebar.markdown("### 8. Structural Break (Rolling Slope)")
    break_window = st.sidebar.number_input("Rolling Window Size", min_value=7, max_value=90, value=21, step=1, help="Days to look back to calculate the current trend slope.")
    
    col_s, col_t = st.sidebar.columns(2)
    with col_s:
        slope_thresh = st.number_input("Severity Threshold (%)", min_value=0.5, max_value=50.0, value=5.0, step=0.5, help="Slope required to rise this % of total range in 30 days.")
    with col_t:
        break_sustained = st.number_input("Sustained Days", min_value=1, max_value=14, value=3, step=1, help="Consecutive days the slope must stay high to trigger the break.")

    # 1. Load Data
    sensor_arr_smooth, sensor_array_raw, time_arr = load_my_sensor_data(uploaded_file, col=selected_col)
    
    # 2. GLOBAL STRUCTURAL BREAK DETECTION
    # Runs exactly once on the ENTIRE dataset to find the true historical break.
    break_idx, break_time = detect_structural_break(
        time_arr, 
        sensor_arr_smooth, 
        window=break_window,
        threshold_pct=slope_thresh,
        sustained_points=break_sustained
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
        
        # 3. Route the Curve Fitting
        top_models, all_models = evaluate_all_models(
            sliced_time, sliced_sensor, 
            priority_ranking=user_priority_dict, 
            eval_window=eval_window,         
            plot=False, verbose=False
        )
        
        if not top_models:
            st.error("Error: All models failed to converge. Try increasing the Window Size or selecting a different cutoff.")
            return
            
        if override_model:
            best_model_name = manual_model
        else:
            best_model_name = list(top_models.keys())[0]
        
        # 4. Fit, Calculate Variance, and Plot
        fig, fitted_series, rul_df = fit_and_plotly_model(
            time_raw=sliced_time,
            sensor_smooth=sliced_sensor,
            sensor_raw=sliced_sensor_raw, 
            model_choice=best_model_name,
            thresholds=thresholds,
            input_time_unit="Days", 
            title_addon=f"| Channel: {selected_col} | Cutoff: {cutoff_idx}",
            max_rul_display=max_rul,
            use_dynamic_variance=use_dynamic_variance,
            break_time_raw=break_time # Pass the globally found break_time here
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
                    "Rank": (
                        "🏆 Override" if (is_winner and override_model) 
                        else "🏆 Algorithm" if is_winner 
                        else "🏆" if name in top_models.keys() 
                        else str(rank)
                    ),
                    "Model": name,
                    "MSE": f"{metrics['mse']:.5f}"
                })
            
            df_leaderboard = pd.DataFrame(leaderboard_data)
            st.dataframe(df_leaderboard, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            st.markdown("### ⏳ RUL Projections")
            if not rul_df.empty:
                display_rul_df = rul_df[[
                    'Threshold', 'Status', 'Nominal_RUL', 'Upper_Band_RUL', 'Lower_Band_RUL'
                ]].copy()
                
                def cap_df_rul(val):
                    if val == 'Safe': return 'Safe'
                    if pd.isna(val): return 'Unknown'
                    if isinstance(val, (int, float)):
                        if val < 0: return 'Breached'
                        if val > max_rul: return f"> {max_rul}"
                        return round(val, 2)
                    return val

                display_rul_df['Nominal_RUL'] = display_rul_df['Nominal_RUL'].apply(cap_df_rul)
                display_rul_df['Upper_Band_RUL'] = display_rul_df['Upper_Band_RUL'].apply(cap_df_rul)
                display_rul_df['Lower_Band_RUL'] = display_rul_df['Lower_Band_RUL'].apply(cap_df_rul)
                
                st.dataframe(display_rul_df, use_container_width=True, hide_index=True)
            else:
                st.info("No threshold data available for this model fit.")

if __name__ == "__main__":
    main()