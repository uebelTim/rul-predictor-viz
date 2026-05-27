from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_sortables import sort_items 
import diagnostic_utils as du 

# ---------------------------------------------------------
# 1. The Standardized Mathematical Models (Pruned to 6)
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
# 2. Master Fitting and Comparison Function
# ---------------------------------------------------------
def evaluate_all_models(time_data, sensor_data, priority_ranking, eval_window=None, tolerance_pct=0.5, plot=False, verbose=True, Title_addon="", save_name=None):
    time_arr = np.asarray(time_data)
    sensor_arr = np.asarray(sensor_data)
    
    t_max = np.max(time_arr)
    time_norm = time_arr / t_max
    
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
            
            # --- NEW: Evaluation Window Slicing ---
            if eval_window is not None and eval_window < len(sensor_arr):
                # Calculate MSE using ONLY the last 'n' points
                mse = mean_squared_error(sensor_arr[-eval_window:], preds[-eval_window:])
            else:
                # Calculate MSE over the entire lookback window
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
# 3. Fitting and Envelope Generation Function
# ---------------------------------------------------------
def fit_and_plotly_model(time_raw, sensor_smooth, sensor_raw, model_choice, thresholds=None, 
                         input_time_unit="Hours", target_time_unit="Days", warm_start_params=None, 
                         title_addon="", sigma_factor=1.645, save_name=None, max_rul_display=365):
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
    
    pos_residuals_series = residuals_series.where(residuals_series > 0, 0)
    rolling_sigma_up = pos_residuals_series.rolling(window=20, min_periods=1).std()
    
    last_sigma = rolling_sigma_up.iloc[-1]
    if pd.isna(last_sigma):
        last_sigma = 0.0
        
    envelope_margin = last_sigma * sigma_factor
    cdf = 0.5 * (1.0 + math.erf(sigma_factor / math.sqrt(2)))
    risk_pct = (1.0 - cdf) * 100
    dynamic_risk_str = f"{risk_pct:.1f}%"

    flat_pct = 0.99
    def solve_for_t(target_val):
        f_time = None
        t_flat = np.nan
        c_max = np.nan
        status = 'Unknown'
        try:
            if model_choice == 'Rational':
                a, b, d = params
                c_max = a + d
                if target_val < c_max:
                    f_time = ((b * (target_val - d)) / (a - (target_val - d))) * t_max
                else:
                    t_flat = (((flat_pct * b) / (1 - flat_pct)) * t_max)
            elif model_choice == 'Arctangent':
                L, k, t0, d = params
                c_max = L + d
                if target_val < c_max:
                    inner_val = np.pi * (((target_val - d) / L) - 0.5)
                    f_time = (t0 + (1 / k) * np.tan(inner_val)) * t_max
                else:
                    t_flat = ((t0 + (1 / k) * np.tan(0.49 * np.pi)) * t_max)
            elif model_choice == 'Linear':
                m, c = params
                if m != 0:
                    f_time = (target_val - c) / m
                else:
                    status = 'Flat Line (No Failure)' if c < target_val else 'Flat Line (Already Failed)'
                    c_max = c
            elif model_choice == 'Gompertz':
                a, b, c, d = params
                c_max = a + d
                if target_val < c_max:
                    inner_log = -np.log((target_val - d) / a) / b
                    if inner_log > 0:
                        f_time = (-np.log(inner_log) / c) * t_max
                else:
                    t_flat = ((-np.log(-np.log(flat_pct) / b) / c) * t_max)
                    
            elif model_choice == 'Softplus':
                a, b, t0, d = params
                if target_val > d and a > 0 and b > 0:
                    inner_exp = np.exp((target_val - d) / a) - 1
                    if inner_exp > 0:
                        f_time = (t0 + np.log(inner_exp) / b) * t_max
                    else:
                        status = 'Math Error'
                else:
                    status = 'Flat Line (No Failure)'
                    
            elif model_choice == 'Shifted Exponential':
                a, b, t0, d = params
                if target_val > d and a > 0 and b > 0:
                    inner_val = (target_val - d) / a
                    if inner_val > 0:
                        f_time = (t0 + np.log(inner_val) / b) * t_max
                    else:
                        status = 'Math Error'
                else:
                    status = 'Flat Line (No Failure)'
        except Exception:
            status = 'Math Error'
            
        return f_time, t_flat, c_max, status

    rul_records = []
    plot_end_time_raw = raw_current_time * 1.2 
    
    if thresholds is not None:
        if not isinstance(thresholds, (list, tuple, np.ndarray)):
            thresholds = [thresholds]
            
        for thresh in sorted(thresholds):
            nom_time, t_flat, c_max, nom_err = solve_for_t(thresh)
            cons_time, _, _, cons_err = solve_for_t(thresh - envelope_margin)
            
            record = {
                'Threshold': thresh,
                'Status': 'Unknown',
                'Envelope_Margin': envelope_margin,
                'Nominal_RUL': np.nan,
                'Nominal_Abs_Time': np.nan,
                'Conservative_RUL': np.nan,
                'Conservative_Abs_Time': np.nan,
                'Ceiling_Max': c_max,
                'Time_to_Flatten': (t_flat * conversion_factor) if not np.isnan(t_flat) else np.nan
            }
            
            if nom_time is not None:
                record['Nominal_Abs_Time'] = nom_time * conversion_factor
                record['Nominal_RUL'] = (nom_time - raw_current_time) * conversion_factor
                plot_end_time_raw = max(plot_end_time_raw, nom_time * 1.1)
            
            if cons_time is not None:
                record['Conservative_Abs_Time'] = cons_time * conversion_factor
                record['Conservative_RUL'] = (cons_time - raw_current_time) * conversion_factor
                plot_end_time_raw = max(plot_end_time_raw, cons_time * 1.1)
            
            n_rul = record['Nominal_RUL']
            c_rul = record['Conservative_RUL']
            
            if not np.isnan(n_rul):
                if n_rul < 0:
                    record['Status'] = 'Already Reached'
                else:
                    if not np.isnan(c_rul) and c_rul < 0:
                        record['Status'] = 'Envelope Breached'
                    else:
                        record['Status'] = 'Predicted Future'
            else:
                if not np.isnan(c_rul):
                    if c_rul < 0:
                        record['Status'] = 'Envelope Breached'
                    else:
                        record['Status'] = 'Envelope Risk (Future)'
                elif nom_err != 'Unknown' and nom_err != 'Math Error':
                    record['Status'] = nom_err
                else:
                    record['Status'] = 'Never Reached (Safe)'
                    if not np.isnan(record['Time_to_Flatten']):
                        plot_end_time_raw = max(plot_end_time_raw, (record['Time_to_Flatten'] / conversion_factor) * 1.1)
            
            rul_records.append(record)

    rul_df = pd.DataFrame(rul_records)
            
    time_arr_converted = time_arr * conversion_factor
    plot_end_time_converted = plot_end_time_raw * conversion_factor
    current_time_converted = raw_current_time * conversion_factor
    max_zoom_limit = current_time_converted * 2.5
    final_end_time = min(plot_end_time_converted, max_zoom_limit)
    
    time_smooth_converted = np.linspace(0, final_end_time, 500)
    time_smooth_norm = (time_smooth_converted / conversion_factor) / t_max
    smooth_preds = config['func'](time_smooth_norm, *params)
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_arr_converted, y=sensor_arr, mode='markers', name='Actual Data',
        marker=dict(color='gray', size=5, opacity=0.7)
    ))

    hist_envelope = fitted_series + (rolling_sigma_up * sigma_factor)
    fig.add_trace(go.Scatter(
        x=time_arr_converted, y=fitted_series, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=time_arr_converted, y=hist_envelope, mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(0, 0, 255, 0.15)', name='Historical Variance'
    ))

    fig.add_trace(go.Scatter(
        x=time_smooth_converted, y=smooth_preds, mode='lines', name=f'{model_choice} Fit',
        line=dict(color='blue', width=2.5), opacity=0.8
    ))

    fig.add_trace(go.Scatter(
        x=time_smooth_converted, y=smooth_preds + envelope_margin, mode='lines', name='Projected Upper Env.',
        line=dict(color='blue', width=1.5, dash='dashdot'), opacity=0.6
    ))
    
    unit_short = {"Seconds": "s", "Minutes": "m", "Hours": "h", "Days": "d"}.get(target_time_unit, target_time_unit)
    colors = ['#FFA500', '#FF0000', '#8B0000', '#800080', '#000000']
    status_lines = []
    
    def format_rul(val, unit, max_val):
        if pd.isna(val) or math.isinf(val): 
            return "Safe"
        if val > max_val:
            return f"> {max_val}{unit}"
        return f"{val:.1f}{unit}"

    if not rul_df.empty:
        for idx, row in rul_df.iterrows():
            c = colors[idx % len(colors)]
            thresh = float(row['Threshold'])
            status = row['Status']
            status_lines.append(f"T={thresh:.2f}: {status}")
            
            if status in ['Predicted Future', 'Envelope Breached', 'Envelope Risk (Future)']:
                n_rul = row['Nominal_RUL']
                c_rul = row['Conservative_RUL']
                n_time = row['Nominal_Abs_Time']
                c_time = row['Conservative_Abs_Time']
                
                n_str = format_rul(n_rul, unit_short, max_rul_display)
                c_str = format_rul(c_rul, unit_short, max_rul_display)
                
                lbl = f"T={thresh:.2f} (Trend: {n_str} | {dynamic_risk_str}: {c_str})"
                    
                fig.add_trace(go.Scatter(
                    x=[0, final_end_time], y=[thresh, thresh], mode='lines', name=lbl,
                    line=dict(color=c, width=2, dash='dash'), opacity=0.6
                ))
                
                if not pd.isna(n_time) and n_time <= final_end_time:
                    fig.add_trace(go.Scatter(
                        x=[n_time], y=[thresh], mode='markers', showlegend=False,
                        marker=dict(symbol='circle', color=c, size=10, line=dict(color='black', width=1)),
                        hoverinfo='skip'
                    ))
                    fig.add_vline(x=n_time, line_dash="dot", line_color=c, opacity=0.3)
                
                if not pd.isna(c_time) and c_time <= final_end_time:
                    fig.add_trace(go.Scatter(
                        x=[c_time], y=[thresh], mode='markers', showlegend=False,
                        marker=dict(symbol='x', color=c, size=12, line=dict(color='black', width=1)),
                        hoverinfo='skip'
                    ))
                    fig.add_vline(x=c_time, line_dash="dashdot", line_color=c, opacity=0.3)
                    
            elif status == 'Already Reached':
                fig.add_trace(go.Scatter(
                    x=[0, final_end_time], y=[thresh, thresh], mode='lines', name=f"T={thresh:.2f} (Breached)",
                    line=dict(color=c, width=2, dash='dash'), opacity=0.6
                ))
                if not pd.isna(row['Nominal_Abs_Time']) and row['Nominal_Abs_Time'] <= final_end_time:
                    fig.add_trace(go.Scatter(
                        x=[row['Nominal_Abs_Time']], y=[thresh], mode='markers', showlegend=False,
                        marker=dict(symbol='circle', color=c, size=10, line=dict(color='black', width=1))
                    ))
                    
            elif status == 'Never Reached (Safe)':
                fig.add_trace(go.Scatter(
                    x=[0, final_end_time], y=[thresh, thresh], mode='lines', name=f"T={thresh:.2f} (Safe)",
                    line=dict(color='gray', width=2, dash='dot'), opacity=0.5
                ))
                if idx == 0 or rul_df.iloc[idx-1]['Status'] != 'Never Reached (Safe)':
                    ceil = row['Ceiling_Max']
                    if not pd.isna(ceil):
                        fig.add_trace(go.Scatter(
                            x=[0, final_end_time], y=[ceil, ceil], mode='lines', name=f'Ceiling ({ceil:.2f})',
                            line=dict(color='green', width=1.5, dash='dash')
                        ))

    max_target = max(thresholds) if thresholds is not None else 0
    absolute_y_max = max(np.max(sensor_arr), max_target)
    absolute_y_min = min(0, np.min(sensor_arr))
    y_padding = (absolute_y_max - absolute_y_min) * 0.15
    y_upper_limit = absolute_y_max + y_padding
    y_lower_limit = absolute_y_min - (y_padding if absolute_y_min < 0 else 0)

    main_title = f"Predictive Analytics | Engine: {model_choice} {title_addon}"
    subtitle_html = "<br>".join([f"<span style='font-size:12px; color:gray;'>{line}</span>" for line in status_lines])
    full_title = f"<b>{main_title}</b><br>{subtitle_html}"

    fig.update_layout(
        title=full_title,
        xaxis_title=f"Elapsed Time ({target_time_unit})",
        yaxis_title="Sensor Value",
        xaxis=dict(range=[0, final_end_time]),
        yaxis=dict(
            range=[y_lower_limit, y_upper_limit],
            hoverformat=".3f" 
        ),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(title="System Log", yanchor="top", y=1, xanchor="left", x=1.02, bordercolor="LightSteelBlue", borderwidth=1),
        margin=dict(t=120),
        width=1400, height=800
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.4)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.4)')

    if save_name is not None:
        fig.write_html(save_name)

    return fig, fitted_series, rul_df


# ---------------------------------------------------------
# 4. Data Loading & Extraction Functions
# ---------------------------------------------------------
@st.cache_data
def get_available_channels(path=r'data/alunorf_2_i-channels_ds.csv'):
    try:
        df = pd.read_csv(path, nrows=1)
        cols = [col for col in df.columns if 'Error' not in col and col != 'DateTime']
        if 'Thermo_Valve_Temperature_DeviationPct' in cols:
            cols.remove('Thermo_Valve_Temperature_DeviationPct')
        return [str(c) for c in cols]
    except FileNotFoundError:
        return ['32', '73', '15', '42']

@st.cache_data
def load_my_sensor_data(col='32'):
    freq = '4H'
    column_names=[]
    path = r'data/alunorf_2_i-channels_ds.csv'
    
    try:
        df = pd.read_csv(path, parse_dates=['DateTime'])
    except FileNotFoundError:
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='4H')
        df = pd.DataFrame({'DateTime': dates, col: np.linspace(0.1, 0.9, 1000) + np.random.normal(0, 0.05, 1000)})
        
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
        df_select = du.filter_outliers_quantile(df_select, factor=3)
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
# 5. The Main UI Function
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="RUL Predictor", layout="wide")
    st.title("Predictive Maintenance: Dynamic RUL Engine")

    # --- UI SIDEBAR CONTROLS ---
    st.sidebar.header("🛠️ Algorithm Parameters")

    raw_channels = get_available_channels()
    display_options = []
    for c in raw_channels:
        if c in ['32', '73']:
            display_options.append(f"{c} (Outlier/Deviating)")
        else:
            display_options.append(c)

    selected_display = st.sidebar.selectbox("1. Select Data Channel", options=display_options)
    selected_col = selected_display.split(" ")[0]

    window_size = st.sidebar.number_input("2. Window Size (Lookback)", min_value=10, max_value=5000, value=300, step=10)

    # --- NEW: Evaluation Window Field ---
    eval_window = st.sidebar.number_input("3. MSE Eval Window (Last N points)", min_value=1, max_value=window_size, value=min(100, window_size), step=1)

    st.sidebar.markdown("### 4. Model Override")
    override_model = st.sidebar.toggle("Enable Manual Selection", value=False)
    manual_model = st.sidebar.selectbox("Force specific model:", options=AVAILABLE_MODELS, disabled=not override_model)

    st.sidebar.markdown("### 5. Router Priority Ranking")
    st.sidebar.caption("Drag and drop models to set priority (Top = 1, Highest Priority).")
    with st.sidebar.expander("Configure Router Ranking", expanded=False):
        sorted_models = sort_items(AVAILABLE_MODELS, direction='vertical')
        user_priority_dict = {model: rank for rank, model in enumerate(sorted_models, start=1)}

    st.sidebar.markdown("### 6. Display Limits")
    max_rul = st.sidebar.number_input("Max RUL Cap (Days)", min_value=1, max_value=5000, value=365)

    # --- END SIDEBAR ---

    sensor_arr_smooth, sensor_array_raw, time_arr = load_my_sensor_data(col=selected_col)

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
            sliced_time, sliced_sensor, 
            priority_ranking=user_priority_dict, 
            eval_window=eval_window,         # <--- Passes the new parameter to the master logic
            plot=False, verbose=False
        )
        
        if not top_models:
            st.error("Error: All models failed to converge. Try increasing the Window Size or selecting a different cutoff.")
            return
            
        if override_model:
            best_model_name = manual_model
        else:
            best_model_name = list(top_models.keys())[0]
        
        fig, fitted_series, rul_df = fit_and_plotly_model(
            time_raw=sliced_time,
            sensor_smooth=sliced_sensor,
            sensor_raw=sliced_sensor_raw, 
            model_choice=best_model_name,
            thresholds=thresholds,
            input_time_unit="Days", 
            title_addon=f"| Channel: {selected_col} | Cutoff: {cutoff_idx}",
            max_rul_display=max_rul  
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
                    'Threshold', 'Status', 'Nominal_RUL', 'Conservative_RUL'
                ]].copy()
                
                def cap_df_rul(val):
                    if pd.isna(val) or (isinstance(val, (int, float)) and val > max_rul):
                        return f"> {max_rul}"
                    return round(val, 2)

                display_rul_df['Nominal_RUL'] = display_rul_df['Nominal_RUL'].apply(cap_df_rul)
                display_rul_df['Conservative_RUL'] = display_rul_df['Conservative_RUL'].apply(cap_df_rul)
                
                st.dataframe(display_rul_df, use_container_width=True, hide_index=True)
            else:
                st.info("No threshold data available for this model fit.")

if __name__ == "__main__":
    main()