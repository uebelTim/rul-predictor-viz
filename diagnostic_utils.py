import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
import shutil
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import savgol_filter, periodogram
from scipy.ndimage import gaussian_filter1d
import scipy.stats as spstats
import seaborn as sns
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import pprint
import plotly.graph_objects as go
import plotly.express as px
from plotly.io import show
from darts import TimeSeries
import darts
print('darts version:', darts.__version__)
from darts.models import NHiTSModel, TFTModel, TiDEModel, RNNModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import GaussianLikelihood, QuantileRegression, BetaLikelihood, GammaLikelihood
from darts.dataprocessing.transformers import Scaler,StaticCovariatesTransformer
from sklearn.preprocessing import RobustScaler
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.models import NaiveSeasonal
from darts.metrics import mape, mae # Import metrics
from typing import List, Union, Callable, Optional
from tqdm.auto import tqdm
import torch
if torch.cuda.is_available():
    print(f"✅ GPU is available: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
else:
    print("❌ GPU is NOT available. Running on CPU.")
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateFinder
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
torch.set_float32_matmul_precision('high')
import logging
# Set the logging level for PyTorch Lightning to WARNING
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
import optuna
import json
import wandb
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
from itertools import combinations
from functools import partial
import logging
import pickle
import re

logging.getLogger("darts").setLevel(logging.ERROR) 
import warnings

try:
    # Lightning ≥1.9
    from pytorch_lightning.utilities.warnings import PossibleUserWarning
except ImportError:
    try:
        # Lightning ≥2.0 (via lightning-fabric)
        from lightning_fabric.utilities.warnings import PossibleUserWarning
    except ImportError:
        # Fall back to generic UserWarning if the symbol is missing entirely
        PossibleUserWarning = UserWarning

warnings.filterwarnings(
    "ignore",
    message="The 'predict_dataloader' does not have many workers",
    category=PossibleUserWarning,
)
# Suppress the specific PyTorch pin_memory warning
warnings.filterwarnings("ignore", message=".*pin_memory.*")


# The new functionality for multivariate plotting requires the ipywidgets library.
# We perform a soft check here to provide a clear error message if it's not installed.
try:
    from ipywidgets import Tab, Output
    from IPython.display import display
    _IPYWIDGETS_AVAILABLE = True
except ImportError:
    _IPYWIDGETS_AVAILABLE = False



def get_basic_statistics(ts):
    """
    Compute basic descriptive statistics of the timeseries.
    """
    assert isinstance(ts, pd.Series), "Input must be a pandas Series"
    # Check if the series is empty
    if ts.empty:
        raise ValueError("The input time series is empty.")
    stats_dict = {
        'count': ts.count(),
        'mean': ts.mean(),
        'median': ts.median(),
        'variance': ts.var(),
        'std': ts.std(),
        'skew': ts.skew(),
        'kurtosis': ts.kurt(),  # we use excess kurtosis so kurtosis = 0 is normal normal distribution
    }
        
    return stats_dict

def get_basic_statistics_from_parallel_ts(ts_array):
    """
    Compute basic descriptive statistics for multiple parallel timeseries.
    
    Parameters:
    ts_array: can be either:
              - array of pandas series
              - pandas dataframe where columns are the parallel data
    
    Returns:
    Dictionary with two types of statistics:
    1. 'combined' - statistics for all data points combined
    2. 'individual' - statistics for each individual series
    """
    # Dictionary to store results
    result = {'combined': {}, 'individual': {}}
    
    # Check if input is a pandas DataFrame
    if isinstance(ts_array, pd.DataFrame):
        #check if datetime column is present
        if 'datetime' in ts_array.columns:
            ts_array = ts_array.set_index('datetime')
        elif pd.api.types.is_datetime64_any_dtype(ts_array.index):
            # If the index is already a datetime index, we can use it directly
            pass
        else:
            raise ValueError("DataFrame must contain a 'datetime' column to set as index.")    
        
        # For individual statistics
        for col in ts_array.columns:
            #make series with timestamp as index
            result['individual'][col] = get_basic_statistics(ts_array[col])
        
        # For combined statistics (stack all columns)
        combined_ts = ts_array.stack().reset_index(drop=True)
        result['combined'] = get_basic_statistics(combined_ts)
    else:
        # For individual statistics
        for i, series in enumerate(ts_array):
            result['individual'][f'series_{i}'] = get_basic_statistics(series)
        
        # For combined statistics (concatenate all series)
        combined_ts = pd.concat(ts_array).reset_index(drop=True)
        result['combined'] = get_basic_statistics(combined_ts)
    
    return result


def find_max_bins(data):
    max_bins = 1
    while True:
        counts, _ = np.histogram(data, bins=max_bins)
        if np.any(counts == 0):
            break
        max_bins += 1
    return max_bins - 1  # Subtract 1 because the last increment caused an empty bin



    
def create_hist_from_parallel_ts(ts_data):
    '''
    ts_array: time series data; can be either:
              - array of pandas series
              - pandas dataframe where columns are the parallel data
    '''
    # Check if input is a pandas DataFrame
    if isinstance(ts_data, pd.DataFrame):
        print('nr of rows in dataframe:', ts_data.shape[0])
        # Stack all columns into a single series
        ts = ts_data.stack().reset_index(drop=True)
        print('nr of  rows in stacked dataframe:', ts.shape[0])
    else:
        # Concatenate all time series into a single series
        ts = pd.concat(ts_data)
        
        # Reset the index to avoid any potential issues with duplicate indices
        ts = ts.reset_index(drop=True)
    
    # Create histogram using the combined data
    create_histogram(ts)
    
def compute_std_thresholds():
    pass

def cornish_fisher_quantile(z, skew, kurtosis):
    """
    Adjusts the standard normal quantile z using the Cornish-Fisher expansion.
    
    Parameters:
    z: standard normal quantile (e.g., 1.96 for approximately 97.5% quantile)
    skew: skewness of the distribution
    kurtosis: kurtosis of the distribution (Note: This is the raw kurtosis, not excess kurtosis)
    
    Returns:
    Adjusted quantile.
    """
    # Assuming kurtosis is the actual 4th moment; for excess kurtosis, kappa_excess + 3 = kurtosis.
    cf_z = (z +
            (z**2 - 1) * skew / 6 +
            (z**3 - 3 * z) * (kurtosis - 3) / 24 -
            (2 * z**3 - 5 * z) * (skew**2) / 36)
    return cf_z

def compute_std_boundaries_with_moments(trend, sigma, skew, kurtosis, alpha=0.025):
    """
    Compute envelope boundaries using Cornish-Fisher expansion based on known moments.
    
    Parameters:
    ts_array: numpy array of the time series (not used directly here, provided for consistency)
    trend: numpy array of the trend component (or simply use the time series mean)
    sigma: standard deviation of the residuals (or overall time series variability)
    skew: skewness of the residuals/time series
    kurtosis: kurtosis of the residuals/time series (raw kurtosis; if you have excess kurtosis add 3)
    alpha: significance level for the envelope (default gives 95% envelope).
    
    Returns:
    lower: lower envelope boundary
    upper: upper envelope boundary
    """
    # For a two-tailed interval, use the quantile z for 1 - alpha
    z_low = spstats.norm.ppf(alpha)
    z_high = spstats.norm.ppf(1 - alpha)
    print(f"Z-scores for alpha ({alpha}): {z_low}, {z_high}")
    adjusted_z_low = cornish_fisher_quantile(z_low, skew, kurtosis)
    adjusted_z_high = cornish_fisher_quantile(z_high, skew, kurtosis)
    print(f"Adjusted Z-scores: {adjusted_z_low}, {adjusted_z_high}")
    
    # Envelope boundaries: Use the given trend as the central tendency.
    upper = trend + sigma * adjusted_z_high
    lower = trend + sigma * adjusted_z_low
    return lower, upper

def compute_boundaries_from_parallel_stats(stats_dict, alpha=0.025):
    """
    Compute envelope boundaries using Cornish-Fisher expansion based on statistics 
    from parallel time series.
    
    Parameters:
    stats_dict: Dictionary from get_basic_statistics_from_parallel_ts function
    alpha: significance level for the envelope (default gives 95% envelope)
    
    Returns:
    Dictionary with lower and upper boundaries, both overall and per series if requested
    """
    # Get combined statistics
    combined_stats = stats_dict['combined']
    mean = combined_stats['mean']
    sigma = combined_stats['std']
    skew = combined_stats['skew']
    # Convert excess kurtosis to raw kurtosis
    kurtosis = combined_stats['kurtosis'] + 3  
    
    # Calculate overall boundaries
    lower, upper = compute_std_boundaries_with_moments(mean, sigma, skew, kurtosis, alpha)
    
    return {
        'lower_boundary': lower,
        'upper_boundary': upper
    }
    
def analyse_slope(data,linearize_time=True):
    """Analyzes the slope of time series data"""
    
    if linearize_time:
        #check if frequency of data is constant
        frequency = pd.infer_freq(data.index)
        #print(f"Frequency of data: {frequency}")
        if frequency is None:
            data = make_equidistant_index(data)
        
    rows = []
    for col in data.columns:
        # Simple linear regression on original data
        X = np.arange(len(data)).reshape(-1, 1)
        y = data[col].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        slope = model.coef_[0]
        intercept = model.intercept_
        y_end = model.predict(X[-1].reshape(1, -1))[0]
        
        # Calculate R-squared
        y_pred = model.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot+1e-10)  # Avoid division by zero
        
        
        row = {'col': col, 'slope': slope, 'intercept': intercept, 'y_end': y_end, 'r_squared': r_squared}
        rows.append(row)
    
    result = pd.DataFrame(rows)
    result = result.set_index('col')
    return result
    
def find_threshold_datetime(data, slope, intercept, threshold, max_years=None):
    """
    Calculate when a linear model reaches the applicable threshold value.
    
    Parameters:
    -----------
    data : pandas.DataFrame or pandas.Series
        The original time series data with datetime index
    slope : float
        Slope of the linear model (change in value per index step)
    intercept : float
        Intercept of the linear model
    threshold : tuple
        (lower_threshold, upper_threshold) values
    max_years : float, optional
        Maximum number of years to extrapolate in either direction.
        Default is None (limited only by pandas Timedelta max of ~292 years).
        
    Returns:
    --------
    datetime or None
        The datetime when the applicable threshold is reached, or None if 
        the threshold will never be reached or is already passed
    """
    lower_threshold, upper_threshold = threshold
    
    # Handle case where slope is nearly zero
    if abs(slope) < 1e-8:
        return None
    
    # Determine which threshold to use based on slope direction
    if slope > 0:
        # Line is increasing, use upper threshold
        target_threshold = upper_threshold
    else:
        # Line is decreasing, use lower threshold
        target_threshold = lower_threshold
    
    # Calculate index where threshold is reached
    target_idx = (target_threshold - intercept) / slope
    
    # Get the start date and time delta per index
    dates = data.index
    
    if len(dates) <= 1:
        return None  # Cannot calculate time delta with only one point
    
    # Check if we've already passed the threshold
    last_idx = len(data) - 1
    if (slope > 0 and target_idx < last_idx) or (slope < 0 and target_idx < last_idx):
        return None  # Threshold was already passed
    
    # Calculate average seconds per index step
    total_seconds = (dates[-1] - dates[0]).total_seconds()
    seconds_per_idx = total_seconds / (len(dates) - 1)
    
    # Apply year-based extrapolation limit if provided
    if max_years is not None:
        # Calculate maximum seconds from current range
        max_seconds = max_years * 365.25 * 24 * 60 * 60
        # Convert to maximum index positions
        max_idx_from_end = max_seconds / seconds_per_idx
        
        # Calculate max allowed target_idx
        max_allowed_idx = last_idx + max_idx_from_end
        min_allowed_idx = -max_idx_from_end
        
        # Check if target_idx is within allowed range
        if target_idx > max_allowed_idx or target_idx < min_allowed_idx:
            return None
    
    # Calculate total seconds to add
    seconds_to_add = seconds_per_idx * target_idx
    
    # Pandas Timedelta limits (about 292 years in seconds)
    MAX_SECONDS = 292 * 365.25 * 24 * 60 * 60
    
    # Check if within pandas limits
    if abs(seconds_to_add) > MAX_SECONDS:
        return None
    
    # Create the timestamp with bounds checking
    try:
        threshold_date = dates[0] + pd.Timedelta(seconds=seconds_to_add)
        return threshold_date
    except (OverflowError, pd.errors.OutOfBoundsTimedelta):
        return None  # Should not happen with the above check, but just in case
    
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def calculate_parameter_spread(df, ds_name="Dataset", linearize_time=False, save_dir=None, show_plots=True, group_by_family=True, family_keywords=None, thresholds=None, unit="Messwert (Rohdaten)", verbose=True):
    """
    Calculates raw, physical parameter spreads. 
    If family_keywords is provided (list of strings), groups columns containing those substrings.
    Otherwise, falls back to stripping trailing numbers from column names.
    """
    import os
    import re
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Enforce grouping to protect unit integrity
    if not group_by_family:
        if verbose: print("Notice: group_by_family forced to True to maintain physical unit integrity.")
        group_by_family = True

    # 1. Replicate deviation and threshold logic
    row_means = df.mean(axis=1)
    deviations = df.sub(row_means, axis=0)
    row_deviation_stds = deviations.std(axis=1)
    
    # 2. Use existing slope/anomaly analysis functions
    slope_df = analyse_slope(df, linearize_time=linearize_time)
    anomalies_dict = find_anomalies_quantile_per_column(df, factor=1.5)
    anomaly_counts = pd.Series({col: len(anomalies) for col, anomalies in anomalies_dict.items()})
    
    # 3. Consolidate per-column parameters into a single DataFrame (Swapped Variance for CV)
    params_df = pd.DataFrame({
        'Mean': df.mean(),
        'Std': df.std(),
        'CV': df.std() / df.mean().replace(0, 1e-10), 
        'IQR': df.quantile(0.75) - df.quantile(0.25),
        'Skewness': df.skew(),
        'Kurtosis': df.kurt(),
        'Slope': slope_df['slope'],
        'NrAnomalies': anomaly_counts
    })

    # --- GROUPING LOGIC ---
    def get_base_name(col):
        col_str = str(col)
        
        # New Behavior: Check against passed keyword list
        if family_keywords is not None and isinstance(family_keywords, list):
            for keyword in family_keywords:
                if keyword in col_str:
                    return keyword
            return "Uncategorized" # Catch-all for columns that miss the keywords
            
        # Old Behavior: Regex fallback
        return re.sub(r'\d+(?!.*\d)', '', col_str)
    
    # Apply grouping logic
    params_df['Family'] = [get_base_name(col) for col in params_df.index]
    groups = [(family, group.drop(columns=['Family'])) for family, group in params_df.groupby('Family')]

    all_spread_results = {}
    local_html_plots = [] 
    # --- EVALUATION LOOP ---
    for group_name, group_df in groups:
        # ==============================================================
        # BOXPLOTS DER ECHTEN ROHDATEN
        # ==============================================================
        fig_raw_box = go.Figure()
        
        sensor_names = group_df.index.tolist()
        valid_sensors = [s for s in sensor_names if s in df.columns]
        
        # 1. Die Boxplots zeichnen
        for sensor in valid_sensors:
            fig_raw_box.add_trace(go.Box(
                y=df[sensor],  
                name=str(sensor),
                boxpoints='outliers',
                marker_color='#3498db',
                showlegend=False # Boxplots selbst brauchen keine Legende
            ))

        # 2. --- SCHWELLENWERTE LOGIK (Interaktive Legende als Checkboxen) ---
        if thresholds and isinstance(thresholds, dict):
            thresh_styles = {
                'error_upper': {'color': '#e74c3c', 'dash': 'solid', 'name': 'Fehler (Oben)'},
                'error_lower': {'color': '#e74c3c', 'dash': 'solid', 'name': 'Fehler (Unten)'},
                'warn_upper': {'color': '#f39c12', 'dash': 'dash', 'name': 'Warnung (Oben)'},
                'warn_lower': {'color': '#f39c12', 'dash': 'dash', 'name': 'Warnung (Unten)'}
            }
            
            for key, style in thresh_styles.items():
                if key in thresholds:
                    val = thresholds[key]
                    fig_raw_box.add_trace(go.Scatter(
                        x=valid_sensors,
                        y=[val] * len(valid_sensors), # Linie über alle Sensoren spannen
                        mode='lines',
                        name=style['name'],
                        line=dict(color=style['color'], dash=style['dash'], width=2),
                        hoverinfo='name+y',
                        visible='legendonly' # <--- DER TRICK: Verhält sich wie eine abgewählte Checkbox!
                    ))

        # 3. Layout mit dynamischer Einheit updaten
        fig_raw_box.update_layout(
            title=f"[{ds_name}] Rohdaten-Verteilung pro Sensor - {group_name}",
            xaxis_title="Sensoren / Komponenten",
            yaxis_title=unit, # <--- Hier wird Ihre Einheit übergeben!
            template="plotly_white",
            showlegend=(thresholds is not None) # Legende (unsere "Tick-Boxen") aktivieren
        )
        
        local_html_plots.append({
            'dataset': ds_name,
            'family': group_name,
            'parameter': 'Raw Data',
            'plot_type': 'Boxplot',
            'html': fig_raw_box.to_html(full_html=False, include_plotlyjs='cdn')
        })
        # ==============================================================
        
        # ==============================================================
        # NEU: WORST OFFENDERS RANKING (Info-Karten)
        # ==============================================================
        # 1. Die Top 3 "Übeltäter" pro Kategorie berechnen
        top_slope = group_df['Slope'].abs().nlargest(3)
        top_cv = group_df['CV'].nlargest(3)
        top_anomalies = group_df['NrAnomalies'].nlargest(3)
        
        # Für den Mean: Größte absolute Abweichung vom Gruppen-Durchschnitt!
        mean_deviation = (group_df['Mean'] - group_df['Mean'].mean()).abs()
        top_mean = mean_deviation.nlargest(3)

        # 2. Eine kleine Hilfsfunktion für schönes HTML
        def build_ranking_list(series, unit="", is_int=False):
            li_html = ""
            for i, (sensor, val) in enumerate(series.items()):
                color = "#e74c3c" if i == 0 else "#333" # Der Schlimmste in Rot
                weight = "bold" if i == 0 else "normal"
                # Holen wir uns den echten Vorzeichen-Wert für Slope und Mean zurück
                display_val = group_df.loc[sensor, 'Slope'] if series.name == 'Slope' else val
                if series.name == 'Mean': display_val = group_df.loc[sensor, 'Mean'] - group_df['Mean'].mean()
                
                if not is_int:
                    li_html += f"<li style='margin-bottom: 5px; color: {color}; font-weight: {weight};'>{sensor}: {display_val:.3g} {unit}</li>"
                else:
                    li_html += f"<li style='margin-bottom: 5px; color: {color}; font-weight: {weight};'>{sensor}: {int(display_val)} {unit}</li>"
            return f"<ol style='margin: 0; padding-left: 20px; font-size: 14px;'>{li_html}</ol>"

        # 3. HTML-Cards mit Flexbox bauen
        cards_html = f"""
        <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px;">
            <div style="flex: 1; min-width: 250px; background: #fff; padding: 15px; border-left: 5px solid #e74c3c; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4 style="margin-top: 0; color: #2c3e50;">Größter Drift (Slope)</h4>
                <p style="font-size: 12px; color: #7f8c8d; margin-top: -10px;">Stärkster Anstieg/Abfall</p>
                {build_ranking_list(top_slope)}
            </div>
            <div style="flex: 1; min-width: 250px; background: #fff; padding: 15px; border-left: 5px solid #f39c12; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4 style="margin-top: 0; color: #2c3e50;">Stärkstes Rauschen (CV)</h4>
                <p style="font-size: 12px; color: #7f8c8d; margin-top: -10px;">Unruhigste Signale</p>
                {build_ranking_list(top_cv)}
            </div>
            <div style="flex: 1; min-width: 250px; background: #fff; padding: 15px; border-left: 5px solid #9b59b6; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4 style="margin-top: 0; color: #2c3e50;">Häufigste Anomalien</h4>
                <p style="font-size: 12px; color: #7f8c8d; margin-top: -10px;">Absolute Ausreißer-Anzahl</p>
                {build_ranking_list(top_anomalies, is_int=True)}
            </div>
            <div style="flex: 1; min-width: 250px; background: #fff; padding: 15px; border-left: 5px solid #3498db; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h4 style="margin-top: 0; color: #2c3e50;">Größter Offset (Mean)</h4>
                <p style="font-size: 12px; color: #7f8c8d; margin-top: -10px;">Abweichung vom Flotten-Schnitt</p>
                {build_ranking_list(top_mean)}
            </div>
        </div>
        """
        
        local_html_plots.append({
            'dataset': ds_name,
            'family': group_name,
            'parameter': 'Ranking', 
            'plot_type': 'Worst Offenders',
            'html': cards_html
        })
        # ==============================================================
        
        # Calculate the SPREAD (Min, Max, Mean, Median) for the current group
        spread_results = group_df.agg(['min', 'max', 'mean', 'median']).transpose()
        spread_results['IQR'] = group_df.quantile(0.75) - group_df.quantile(0.25)
        spread_results['Range'] = spread_results['max'] - spread_results['min']
        
        all_spread_results[group_name] = spread_results

        # --- PRINT STATEMENTS ---
        if verbose:
            print("\n" + "*" * 85)
            if group_name in ["Global", "Uncategorized"]:
                title = f"DATASET PARAMETER SPREAD - {ds_name}"
            else:
                title = f"DATASET PARAMETER SPREAD - {ds_name} | FAMILY: {group_name}"
            
            print(f"{title:^85}")
            
            for param in spread_results.index:
                row = spread_results.loc[param]
                fmt = ".3g" if param == "Slope" else ".3f"
                print(f"{param:<15} | {row['min']:<10{fmt}} | {row['max']:<10{fmt}} | {row['mean']:<10{fmt}} | {row['median']:<10{fmt}} | {row['IQR']:<10{fmt}}")
            
            print("*" * 85)
        
        
        # --- PLOTTING ---
        for param in group_df.columns:
            # ==============================================================
            # 1. Namen und Werte der einzelnen Sensoren extrahieren
            sensor_names = group_df.index.tolist()
            sensor_values = group_df[param].tolist()
            
            # 2. Den Durchschnitt (Mean) dieser Statistik über alle Sensoren berechnen
            group_mean_val = group_df[param].mean()
            
            # 3. Listen erweitern für die X- und Y-Achse
            x_bars = sensor_names + ['<b>AVERAGE</b>']
            y_bars = sensor_values + [group_mean_val]
            
            # 4. Farben definieren (Sensoren = Blau, Average = Rot/Orange)
            bar_colors = ["#338bc5"] * len(sensor_names) + ['#e74c3c']
            
            fig_bar = go.Figure(data=[go.Bar(
                x=x_bars,
                y=y_bars,
                text=[f"{v:.4f}" for v in y_bars], # Zeigt die Werte direkt auf den Balken
                textposition='auto',
                marker_color=bar_colors
            )])
            
            # Präfixe für die Titel (wie im restlichen Code)
            if group_name != "Global":
                title_prefix = f"[{ds_name} | {group_name}] " 
                file_prefix = f"{ds_name}_{group_name}_"
            else:
                title_prefix = f"[{ds_name}] "
                file_prefix = f"{ds_name}_"
                
            fig_bar.update_layout(
                title=f"{title_prefix}Sensor Breakdown for: {param}",
                xaxis_title="Components",
                yaxis_title=f"{param} Value",
                template="plotly_white",
                xaxis_tickangle=-45 # Leichtes anwinkeln bei vielen Sensoren
            )
            
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                bar_save_path = os.path.join(save_dir, f"{file_prefix}{param}_sensor_barchart.html")
                fig_bar.write_html(bar_save_path)
                
            local_html_plots.append({
                'dataset': ds_name,
                'family': group_name,
                'parameter': param,
                'plot_type': 'Sensor Breakdown',
                'html': fig_bar.to_html(full_html=False, include_plotlyjs='cdn')
            })
            
            if show_plots:
                fig_bar.show()
                
                
            # ==============================================================
            # --- ROBUST PERCENTILE LOGIC ---
            # Identify the 5th and 95th percentile values for the parameter
            target_min_val = group_df[param].min()#.quantile(0.05)
            target_max_val = group_df[param].max()#.quantile(0.95)
            
            # Find the specific sensor columns whose values are closest to those percentiles
            min_col = (group_df[param] - target_min_val).abs().idxmin()
            max_col = (group_df[param] - target_max_val).abs().idxmin()
            
            # Extract the actual values for the labels
            min_val = group_df[param].loc[min_col]
            max_val = group_df[param].loc[max_col]
            
            fmt = ".4f"
            
            # --- INJECT DS_NAME INTO THE TITLE PREFIX & FILE PREFIX ---
            if group_name not in ["Global", "Uncategorized"]:
                title_prefix = f"[{ds_name} | {group_name}] " 
                file_prefix = f"{ds_name}_{group_name}_"
            else:
                title_prefix = f"[{ds_name}] "
                file_prefix = f"{ds_name}_"
            
            # 1. INITIALIZE THE CORRECT FIGURE TYPE
            if param in ['Skewness', 'Kurtosis']:
                stats_dict_min = get_basic_statistics(df[min_col])
                title_min = f"Histogram of min col: {min_col}"
                title_min += f"<br> (Std: {stats_dict_min['std']:.3f}, Skew: {stats_dict_min['skew']:.3f}, Kurtosis: {stats_dict_min['kurtosis']:.3f})"
                
                stats_dict_max = get_basic_statistics(df[max_col])
                title_max = f"Histogram of max col: {max_col}"
                title_max += f"<br> (Std: {stats_dict_max['std']:.3f}, Skew: {stats_dict_max['skew']:.3f}, Kurtosis: {stats_dict_max['kurtosis']:.3f})"
                
                fig = make_subplots(
                    rows=2, cols=2,
                    specs=[[{"colspan": 2}, None], 
                           [{}, {}]],
                    subplot_titles=(
                        f"{title_prefix}Time Series", 
                        title_min, 
                        title_max,
                    ),
                    vertical_spacing=0.15
                )
                ts_row, ts_col = 1, 1 
            else:
                fig = go.Figure()
                ts_row, ts_col = None, None
            
            # Add trace for the Min column
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df[min_col], 
                mode='lines', 
                name=f'Min Col: {min_col} (Value: {min_val:{fmt}})',
                line=dict(width=2)
            ))
                    
            # Add trace for the Max column
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df[max_col], 
                mode='lines', 
                name=f'Max Col: {max_col} (Value: {max_val:{fmt}})',
                line=dict(width=2)
            ))
            
            if param == 'Slope':
                slope = slope_df.loc[str(min_col), 'slope']
                intercept = slope_df.loc[str(min_col), 'intercept']
                y_end = slope_df.loc[str(min_col), 'y_end']
                start_x, end_x = df.index[0], df.index[-1]
                
                fig.add_trace(go.Scatter(
                    x=[start_x, end_x],
                    y=[intercept, y_end],
                    mode='lines',
                    name=f'Slope Line Min: {min_col} (Slope: {slope:.3g})',
                    line=dict(width=2, dash='dash')
                ))
                
                slope = slope_df.loc[str(max_col), 'slope']
                intercept = slope_df.loc[str(max_col), 'intercept']
                y_end = slope_df.loc[str(max_col), 'y_end']
                
                fig.add_trace(go.Scatter(
                    x=[start_x, end_x],
                    y=[intercept, y_end],
                    mode='lines',
                    name=f'Slope Line Max: {max_col} (Slope: {slope:{fmt}})',
                    line=dict(width=2, dash='dash')
                ))
                
            if param == 'NrAnomalies':
                anomalies_min = anomalies_dict[min_col]
                anomalies_max = anomalies_dict[max_col]
                
                min_count_actual = anomaly_counts[min_col]
                max_count_actual = anomaly_counts[max_col]
                
                fig.add_trace(go.Scatter(
                    x=anomalies_min.index,
                    y=anomalies_min.values,
                    mode='markers',
                    name=f'Anomalies in Min Col: {min_col} (Count: {min_count_actual})',
                    marker=dict(color='red', size=8, symbol='x')
                ))
                fig.add_trace(go.Scatter(
                    x=anomalies_max.index,
                    y=anomalies_max.values,
                    mode='markers',
                    name=f'Anomalies in Max Col: {max_col} (Count: {max_count_actual})',
                    marker=dict(color='blue', size=8, symbol='x')
                ))
            
            # Update layout (Automatically includes ds_name via title_prefix)
            fig.update_layout(
                title=f"{title_prefix}Vergleich von Min und Max Sensor für: {param}",
                xaxis_title="Time",
                yaxis_title="Raw Sensor Value",
                legend_title="Columns",
                hovermode="x unified",
                template="plotly_white",
                height=900 if param in ['Skewness', 'Kurtosis'] else None
            )
            
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{file_prefix}{param}_spread.html")
                fig.write_html(save_path)
                
            
            
            if param in ['Skewness', 'Kurtosis']:
                hist1 = create_plotly_histogram(df[min_col], stats_dict_min, fig=fig, row=2, col=1)
                hist2 = create_plotly_histogram(df[max_col], stats_dict_max, fig=fig, row=2, col=2, add_legend=False)
                
            local_html_plots.append({
                'dataset': ds_name,
                'family': group_name,
                'parameter': param,
                'plot_type': 'Structural Comparison',
                'html': fig.to_html(full_html=False, include_plotlyjs='cdn')
            })
            
            if show_plots:
                fig.show()

    # --- RETURN LOGIC ---
    if not group_by_family or (len(all_spread_results) == 1 and "Uncategorized" in all_spread_results):
        flat_key = "Global" if not group_by_family else "Uncategorized"
        return all_spread_results[flat_key], local_html_plots
        
    return pd.concat(all_spread_results, names=['Family', 'Parameter']), local_html_plots


def compare_dataset_spreads(df1, df2, name1="Dataset_1", name2="Dataset_2", data_desc=None, save_path_base=None, show_plots=True):
    """
    Compares two datasets by looking at their individual spreads and 
    the combined global spread to determine how much they diverge.
    """
    if save_path_base is not None:
        #
        save_path = os.path.join(save_path_base, f"{data_desc}")
    else:
        save_path = os.path.join(r'C:\Users\uel\Documents\Code\Data',  f"spread_comparison_{name1}-{name2}_{data_desc if data_desc else ''}")
    
    #create folder if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Saving individual dataset spreads to: {save_path}")
        
    print(f"\n{'#'*50}")
    print(f"--- Evaluating {name1} ---")
    print(f"{'#'*50}")
    spread1 = calculate_parameter_spread(df1, save_dir=save_path, show_plots=show_plots)
    
    print(f"\n{'#'*50}")
    print(f"--- Evaluating {name2} ---")
    print(f"{'#'*50}")
    spread2 = calculate_parameter_spread(df2, save_dir=save_path, show_plots=show_plots)
    
    comparison_df = pd.DataFrame(index=spread1.index)
    
    # 1. Individual Spreads
    comparison_df[f'{name1}_Spread'] = spread1['spread_range']
    comparison_df[f'{name2}_Spread'] = spread2['spread_range']
    
    # 2. Global Min and Max (Your approach)
    # Compares the minimums of both datasets to find the absolute lowest value, same for max
    comparison_df['Global_Min'] = np.minimum(spread1['min'], spread2['min'])
    comparison_df['Global_Max'] = np.maximum(spread1['max'], spread2['max'])
    
    # 3. The Combined Spread (Global Max - Global Min)
    comparison_df['Combined_Spread'] = comparison_df['Global_Max'] - comparison_df['Global_Min']
    
    comparison_df = comparison_df.drop(columns=['Global_Min', 'Global_Max'])  # Drop if you only want spreads
    # 4. Spread Expansion Factor
    # Tells you how much the spread GROWS when you put the datasets together.
    # 1.0 means perfect overlap (no growth). >2.0 means severe separation.
    max_individual_spread = np.maximum(comparison_df[f'{name1}_Spread'], comparison_df[f'{name2}_Spread'])
    # Avoid division by zero
    comparison_df['Spread_Expansion_Factor'] = comparison_df['Combined_Spread'] / (max_individual_spread + 1e-10)

    comparison_df[f'{name1}_Mean'] = spread1['mean']
    comparison_df[f'{name2}_Mean'] = spread2['mean']
    # Optional: Keep the mean shift just as a secondary sanity check
    comparison_df['Mean_Shift'] = abs(spread1['mean'] - spread2['mean'])
    avg_internal_spread = (spread1['spread_range'] + spread2['spread_range']) / 2
    comparison_df['Shift_to_Noise_Factor'] = comparison_df['Mean_Shift'] / (avg_internal_spread + 1e-10)

    # --- FORMAT AND PRINT RESULTS ---
    print("\n" + "=" * 130)
    target_data_str = f" for Target Data: {data_desc}" if data_desc is not None else ""
    print(f"{f'DATASET SIMILARITY COMPARISON: {name1} - {name2} {target_data_str}':^130}")
    print("=" * 130)
    
    # Create a string-formatted copy of the DataFrame for printing
    print_df = pd.DataFrame(index=comparison_df.index, columns=comparison_df.columns)
    
    for idx in comparison_df.index:
        for col in comparison_df.columns:
            val = comparison_df.loc[idx, col]
            
            if pd.isna(val):
                print_df.loc[idx, col] = "NaN"
            elif idx == 'NrAnomalies' and col not in ['Expansion_Factor', 'Mean_Shift']:
                # Force integers ONLY for the actual anomaly counts/spreads
                # Using round() first prevents 0.99999 math errors before converting to int
                print_df.loc[idx, col] = f"{int(round(val))}"
            elif abs(val) < 0.001 and val != 0:
                # Scientific notation for extremely tiny values (like slope)
                print_df.loc[idx, col] = f"{val:.2e}"
            else:
                # Standard float formatting for everything else (including Expansion_Factor)
                print_df.loc[idx, col] = f"{val:.3f}"
            
    print(print_df.to_string())
    print("=" * 130)
    
    if save_path is not None:
        #save to json
        #change precision to 3 decimals for all values
        comparison_df = comparison_df.round(3)
        #compare_filepath = os.path.join(save_path, f"comparison_statistics.json")
        #comparison_df.to_json(compare_filepath, orient='index', indent=4)
        comparison_df.to_csv(os.path.join(save_path, f"comparison_statistics.csv"))
        #save as formatted string
        with open(os.path.join(save_path, f"comparison_statistics.txt"), 'w') as f:
            f.write(print_df.to_string())
            
    
    return comparison_df

def compare_cooling_cycle_subsystems(df1, df2, name1="Dataset_1", name2="Dataset_2", data_desc=None, show_plots=False):
    """
    Wrapper to split cooling cycle data by physical units (Temp, Pressure, Flow)
    and compare the spreads of each subsystem independently to prevent 
    large-magnitude units from overshadowing smaller ones.
    """
    # 1. Extract column groups based on your specific logic
    # We use df1 as the reference, assuming df1 and df2 share the same schema
    columns = df1.columns
    
    temp_cols = [col for col in columns if 'Temp' in col and 'Error' not in col]
    if 'Thermo_Valve_Temperature_DeviationPct' in temp_cols:
        temp_cols.remove('Thermo_Valve_Temperature_DeviationPct')
        
    pressure_cols = [col for col in columns if 'Pressure' in col and 'Error' not in col]
    
    flow_cols = [col for col in columns if 'Flow' in col and 'Error' not in col]
    
    # 2. Map subsystems to their columns
    subsystems = {
        'Temperatures': temp_cols,
        'Pressures': pressure_cols,
        'Flows': flow_cols
    }
    
    # Dictionary to store the comparison tables for each subsystem
    all_results = {}
    
    # 3. Iterate through each subsystem and run the comparison
    for system_name, cols in subsystems.items():
        if not cols:
            print(f"\nSkipping {system_name}: No matching columns found.")
            continue
            
        print(f"\n{'='*90}")
        print(f"{f'STARTING SUBSYSTEM ANALYSIS: {system_name.upper()}':^90}")
        print(f"{'='*90}")
        print(f"Columns included ({len(cols)}): {cols}")
        
        # Ensure we only use columns that exist in BOTH datasets to prevent KeyError
        valid_cols = [c for c in cols if c in df1.columns and c in df2.columns]
        
        sub_df1 = df1[valid_cols]
        sub_df2 = df2[valid_cols]
            
        save_path = os.path.join(r'C:\Users\uel\Documents\Code\Data',  f"spread_comparison_{name1}-{name2}_{data_desc}")
        # Run your existing global comparison function
        comparison_df = compare_dataset_spreads(
            sub_df1, 
            sub_df2, 
            name1=name1, 
            name2=name2, 
            save_path_base=save_path,
            show_plots=show_plots,
            data_desc=f"{system_name}"
        )
        
        # Store the result dataframe in our dictionary
        all_results[system_name] = comparison_df
        
    print(f"\n{'='*90}")
    print(f"{'ALL SUBSYSTEMS COMPLETED SUCCESSFULLY':^90}")
    print(f"{'='*90}")
    
    return all_results



def get_base_name(col):
        return re.sub(r'\d+(?!.*\d)', '', str(col))
    
def format_spread_val(x):
    if pd.isna(x): return "NaN"
    if x == 0: return "0.000"
    if abs(x) < 0.001: return f"{x:.3e}"
    if float(x).is_integer(): return f"{int(x)}"
    return f"{x:.3f}"
    
def get_divided_spread_string(df):
    # Convert dataframe to string using your clean number formatter
    raw_str = df.to_string(float_format=format_spread_val)
    lines = raw_str.split('\n')
    formatted_lines = []
    
    # Get the max width of the table so the dividers match perfectly
    line_len = max([len(l) for l in lines]) if lines else 100 
    
    header_passed = False
    first_family_passed = False
    
    for line in lines:
        # Detect the main header line
        if 'Family' in line and 'Parameter' in line:
            header_passed = True
            formatted_lines.append(line)
            # Add a strong divider directly under the header
            formatted_lines.append("=" * line_len)
            continue
            
        # Detect a new Family row (starts with a character, not a space)
        if header_passed and len(line) > 0 and not line[0].isspace():
            if first_family_passed:
                # Inject a dashed divider before the next family
                formatted_lines.append("-" * line_len)
            first_family_passed = True
            
        formatted_lines.append(line)
        
    return '\n'.join(formatted_lines)

from scipy.stats import ks_2samp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import re
import numpy as np
import pandas as pd

def compare_robust_dataset_spreads(df1, df2, name1="Dataset_1", name2="Dataset_2", data_desc=None, save_path_base=None, show_plots=True, plot_comparisons=True, family_keywords=None, standardize=True,  thresholds=None, unit="Messwert (Rohdaten)"):
    """
    ========================================================================================
    METRICS INTERPRETATION GUIDE
    ========================================================================================

    1. Delta_Mean (or Delta_Mean_Z): 
        *What it is:* The simple difference between the average parameter value in DS2 and DS1.
        *Interpretation:* -> Positive (+): The average sensor in DS2 has a HIGHER value for this parameter.
            (e.g., a positive Delta_Mean for 'CV' means DS2 is, on average, noisier).
            -> Negative (-): The average sensor in DS2 has a LOWER value for this parameter.

    2. Spread_Ratio (D2 / D1):
        *What it is:* A proportional comparison of the datasets. It is calculated differently 
        depending on the parameter type to avoid mathematical traps and to capture the most 
        meaningful physical insight.
        
        *For Magnitude Parameters (Std, CV, IQR, NrAnomalies): Ratio of Means*
            -> WHY: These parameters measure the physical "amount" of noise or spread. Because 
            they are strictly positive numbers, comparing the average (mean) value across all 
            sensors tells you what the "typical" sensor is doing.
            -> Example: A CV ratio of 1.50 means the typical DS2 sensor is 50% noisier than DS1.
            -> Example: A ratio of 0.80 means DS2 is 20% "tighter" or less noisy than DS1.
            
        *For Shape Parameters (Skewness, Kurtosis): Ratio of Interquartile Ranges (IQR)*
            -> WHY: Shape parameters can be positive, negative, or zero (e.g., left vs. right skew). 
            If a dataset has an equal mix of highly left-skewed and right-skewed sensors, the 
            "mean" skewness might be perfectly zero. Dividing by zero would crash the script, 
            and the mean wouldn't tell you anything useful anyway. Instead, we measure the IQR 
            (the spread of the skews) to see the variance in behaviors.
            -> Example: A Skewness ratio of 2.0 means the sensors in DS2 exhibit a much wider 
            mix of extreme shapes. DS1 behaves much more uniformly.

    3. KS_Statistic (Kolmogorov-Smirnov):
        *What it is:* Measures the maximum difference between the cumulative distribution 
        functions (CDFs) of DS1 and DS2 for a given parameter.
        *Interpretation:* Bounded between 0.0 and 1.0.
            -> 0.0 to 0.2: The fleets behave virtually identically for this parameter.
            -> 0.3 to 0.6: Noticeable systemic shift; the fleets are diverging.
            -> 0.7 to 1.0: Severe divergence; the fleets belong to completely different distributions.
    ========================================================================================
    """
    
    common_cols = df1.select_dtypes(include=[np.number]).columns.intersection(df2.select_dtypes(include=[np.number]).columns)
    
    # 1. Conditionally Standardize
    if standardize:
        b_mean, b_std = df1[common_cols].mean(), df1[common_cols].std().replace(0, 1e-10)
        df1_s = (df1[common_cols] - b_mean) / b_std
        df2_s = (df2[common_cols] - b_mean) / b_std
        z_suffix = "_Z"
        title_prefix = "ROBUST Z-SCORED COMPARISON"
        shift_label = "Z-Shift"
    else:
        df1_s = df1[common_cols].copy()
        df2_s = df2[common_cols].copy()
        z_suffix = ""
        title_prefix = "ROBUST ABSOLUTE COMPARISON"
        shift_label = "Raw-Shift"

    # 2. Directory Logic
    folder_name = f"robust_comparison_{name1}-{name2}_{data_desc if data_desc else ''}"
    save_path = os.path.join(save_path_base if save_path_base else os.getcwd(), folder_name)
    plots_dir = os.path.join(save_path, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 3. Stats Calculation 
    def get_p(df_s):
        slp = analyse_slope(df_s)
        anom = find_anomalies_quantile_per_column(df_s, factor=1.5)
        cnts = pd.Series({col: len(v) for col, v in anom.items()})
        return pd.DataFrame({
            'Mean': df_s.mean(),
            'Std': df_s.std(), 
            'IQR': df_s.quantile(0.75) - df_s.quantile(0.25),
            'CV': df_s.std() / df_s.mean().replace(0, 1e-10), # Added CV
            'Skewness': df_s.skew(), 
            'Kurtosis': df_s.kurt(), 
            'Slope': slp['slope'], 
            'NrAnomalies': cnts
        }), anom

    print(f"Calculating {'standardized ' if standardize else 'absolute '}statistics for {name1}...")
    p1, anom1 = get_p(df1_s)
    
    print(f"Calculating {'standardized ' if standardize else 'absolute '}statistics for {name2}...")
    p2, anom2 = get_p(df2_s)

    def get_strict_base(col):
        return re.sub(r'\d+(?!.*\d)', '', str(col))

    # 4. Comparison Logic
    rows = []
    inter_html_plots = []
    meta_hist_plots = []
    for param in p1.columns:
        s1, s2 = p1[param].dropna(), p2[param].dropna()
        common_idx = s1.index.intersection(s2.index)
        s1, s2 = s1[common_idx], s2[common_idx]
        
        # --- MATH LOGIC ---
        if param in ['Std', 'CV', 'IQR', 'NrAnomalies']:
            ratio_val = s2.mean() / (s1.mean() + 1e-10) 
        else:
            iqr1, iqr2 = (s1.quantile(0.75)-s1.quantile(0.25)), (s2.quantile(0.75)-s2.quantile(0.25))
            ratio_val = iqr2 / (iqr1 + 1e-10)

        rows.append({
            'Parameter': param,
            f'Delta_Mean{z_suffix}': s2.mean() - s1.mean(),
            'Spread_Ratio (D2/D1)': ratio_val,
            'KS_Statistic': ks_2samp(s1, s2)[0]
        })
        
        
        # ==========================================================
        # NEU: SIDE-BY-SIDE META-HISTOGRAMME (Flottenverteilung)
        # ==========================================================
        # 1. Statistik-Dictionaries für beide Serien bauen
        stats1 = {'mean': s1.mean(), 'std': s1.std(), 'skew': s1.skew(), 'kurtosis': s1.kurt()}
        stats2 = {'mean': s2.mean(), 'std': s2.std(), 'skew': s2.skew(), 'kurtosis': s2.kurt()}
        
        # 2. Figur mit Subplots erstellen (1 Zeile, 2 Spalten)
        fig_meta_hist = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"Dataset: {name1}", f"Dataset: {name2}"),
            horizontal_spacing=0.1 # Etwas Platz zwischen den Diagrammen
        )
        
        # 3. Ihre Histogramm-Funktion aufrufen und exakt in die Spalten zuweisen!
        # (Farbe und Trace-Name sind jetzt egal, da sie räumlich getrennt sind)
        fig_meta_hist = create_plotly_histogram(s1, stats1, fig=fig_meta_hist, row=1, col=1, add_legend=False)
        fig_meta_hist = create_plotly_histogram(s2, stats2, fig=fig_meta_hist, row=1, col=2, add_legend=False)
        
        # 4. Globalen Titel und Layout setzen
        fig_meta_hist.update_layout(
            title_text=f"Fleet Distribution Comparison: {param}",
            template="plotly_white",
            height=500 # Macht das Diagramm etwas kompakter für das Dashboard
        )
        
        # Wichtig: Da wir Subplots nutzen, müssen wir die X-Achsen-Titel separat setzen
        fig_meta_hist.update_xaxes(title_text=f"{param} Value", row=1, col=1)
        fig_meta_hist.update_xaxes(title_text=f"{param} Value", row=1, col=2)
        
        # 5. Für das Dashboard speichern
        meta_hist_plots.append({
            'dataset': f'{name1} vs {name2}',
            'family': 'Global', 
            'parameter': param,
            'plot_type': 'Fleet Distribution (Side-by-Side)',
            'html': fig_meta_hist.to_html(full_html=False, include_plotlyjs='cdn')
        })
        
        if show_plots: fig_meta_hist.show()
        # ==========================================================
        
        
        # --- PLOTTING ---
        if plot_comparisons and len(common_idx) > 0:
            families = list(set([get_strict_base(c) for c in common_idx]))
            
            global_max_gap = -1
            plot_data = None

            for fam in families:
                d1_cols = [c for c in common_idx if get_strict_base(c) == fam]
                d2_cols = [c for c in common_idx if get_strict_base(c) == fam]
                
                if d1_cols and d2_cols:
                    # 1. Target the 95th and 5th percentiles instead of absolute max/min
                    p_max_1, p_min_1 = s1[d1_cols].quantile(0.95), s1[d1_cols].quantile(0.05)
                    p_max_2, p_min_2 = s2[d2_cols].quantile(0.95), s2[d2_cols].quantile(0.05)

                    # 2. Calculate the structural gaps using the percentiles
                    gap_2_1 = abs(p_max_2 - p_min_1) # DS2 95th vs DS1 5th
                    gap_1_2 = abs(p_max_1 - p_min_2) # DS1 95th vs DS2 5th
                    
                    # 3. Find the actual column names (sensors) whose values are closest to those percentiles
                    col_max_1 = (s1[d1_cols] - p_max_1).abs().idxmin()
                    col_min_1 = (s1[d1_cols] - p_min_1).abs().idxmin()
                    
                    col_max_2 = (s2[d2_cols] - p_max_2).abs().idxmin()
                    col_min_2 = (s2[d2_cols] - p_min_2).abs().idxmin()

                    if gap_2_1 > global_max_gap:
                        global_max_gap = gap_2_1
                        plot_data = (
                            col_max_2, name2, df2, anom2, s2, 
                            col_min_1, name1, df1, anom1, s1  
                        )
                    
                    if gap_1_2 > global_max_gap:
                        global_max_gap = gap_1_2
                        plot_data = (
                            col_max_1, name1, df1, anom1, s1,
                            col_min_2, name2, df2, anom2, s2
                        )

            if plot_data:
                col_max, n_max, df_max, anom_max, s_max, col_min, n_min, df_min, anom_min, s_min = plot_data
                
                raw_y_max = df_max[col_max].dropna()
                raw_y_min = df_min[col_min].dropna()

                if param == 'Std': val_raw_max, val_raw_min = raw_y_max.std(), raw_y_min.std()
                elif param == 'CV': 
                    val_raw_max = raw_y_max.std() / raw_y_max.mean() if raw_y_max.mean() != 0 else np.nan
                    val_raw_min = raw_y_min.std() / raw_y_min.mean() if raw_y_min.mean() != 0 else np.nan
                elif param == 'IQR':
                    val_raw_max = raw_y_max.quantile(0.75) - raw_y_max.quantile(0.25)
                    val_raw_min = raw_y_min.quantile(0.75) - raw_y_min.quantile(0.25)
                elif param == 'Skewness': val_raw_max, val_raw_min = raw_y_max.skew(), raw_y_min.skew()
                elif param == 'Kurtosis': val_raw_max, val_raw_min = raw_y_max.kurt(), raw_y_min.kurt()
                elif param == 'Slope':
                    val_raw_max = np.polyfit(np.arange(len(raw_y_max)), raw_y_max, 1)[0] if len(raw_y_max) > 1 else 0
                    val_raw_min = np.polyfit(np.arange(len(raw_y_min)), raw_y_min, 1)[0] if len(raw_y_min) > 1 else 0
                elif param == 'NrAnomalies': val_raw_max, val_raw_min = len(anom_max[col_max]), len(anom_min[col_min])
                else: val_raw_max, val_raw_min = np.nan, np.nan

                fmt = ".0f" if param == 'NrAnomalies' else ".4g" if param == 'Slope' else ".4f"

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=raw_y_max.index, y=raw_y_max, mode='lines', 
                                         name=f"{n_max} Max: {col_max} (Raw: {val_raw_max:{fmt}})", line=dict(width=2)))
                fig.add_trace(go.Scatter(x=raw_y_min.index, y=raw_y_min, mode='lines', 
                                         name=f"{n_min} Min: {col_min} (Raw: {val_raw_min:{fmt}})", line=dict(width=2)))
                
                if param == 'Slope':
                    for y_data, l_name in [(raw_y_max, f'Slope Max'), (raw_y_min, f'Slope Min')]:
                        if len(y_data) > 1:
                            x_num = np.arange(len(y_data))
                            slope, intercept = np.polyfit(x_num, y_data, 1)
                            fig.add_trace(go.Scatter(x=[y_data.index[0], y_data.index[-1]], 
                                                     y=[intercept, intercept + (slope * x_num[-1])], 
                                                     mode='lines', name=l_name, line=dict(width=2, dash='dash')))

                if param == 'NrAnomalies':
                    fig.add_trace(go.Scatter(x=anom_max[col_max].index, y=raw_y_max.loc[anom_max[col_max].index], 
                                             mode='markers', name=f'Anomalies Max', marker=dict(color='blue', size=8, symbol='x')))
                    fig.add_trace(go.Scatter(x=anom_min[col_min].index, y=raw_y_min.loc[anom_min[col_min].index], 
                                             mode='markers', name=f'Anomalies Min', marker=dict(color='red', size=8, symbol='x')))

                shift_val = s_max[col_max] - s_min[col_min]
                fig.update_layout(title=f"{param}: Largest Difference ({shift_label}: {shift_val:+.3f})",
                                  xaxis_title="Time", yaxis_title="Raw Sensor Value",
                                  hovermode="x unified", template="plotly_white")
                
                fig.write_html(os.path.join(plots_dir, f"Comparison_{param}.html"))
                
                inter_html_plots.append({
                    'dataset': f'{name1} vs {name2}',
                    'family': fam if 'fam' in locals() else 'Global',
                    'parameter': param,
                    'plot_type': 'Inter-Dataset Shift',
                    'html': fig.to_html(full_html=False, include_plotlyjs='cdn')
                })
                
                if show_plots: fig.show()

    # ==========================================
    # --- FINAL FORMATTING ---
    # ==========================================
    print(f"\n-------------------- Baseline: {name1} --------------------")
    spread_ds1, plots_ds1 = calculate_parameter_spread(df1, ds_name=name1, save_dir=plots_dir, show_plots=False, group_by_family=False, family_keywords=family_keywords, thresholds=thresholds, unit=unit, verbose=True)

    print(f"\n-------------------- Baseline: {name2} --------------------")
    spread_ds2, plots_ds2 = calculate_parameter_spread(df2, ds_name=name2, save_dir=plots_dir, show_plots=False, group_by_family=False, family_keywords=family_keywords, thresholds=thresholds, unit=unit, verbose=True)
    
    comparison_df = pd.DataFrame(rows).set_index('Parameter')
    
    print("\n" + "=" * 110)
    print(f"{f'{title_prefix}: {name2} vs {name1}':^110}")
    print("=" * 110)
    
    print_df = comparison_df.copy()
    for col in print_df.columns:
        print_df[col] = print_df[col].apply(lambda x: f"{x:+.3f}" if "Delta" in col else f"{x:.3f}")
    
    print(print_df.to_string())
    print("=" * 110)
    
    if save_path:
        comparison_df.round(4).to_csv(os.path.join(save_path, "comparison_table.csv"))
        
        details_path = os.path.join(save_path, "comparison_details.txt")
        with open(details_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write(f"FLEET STATISTICAL ANALYSIS REPORT: {name2} vs {name1}\n")
            f.write(f"Target Data Description: {data_desc if data_desc else 'N/A'}\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"PART 1: {title_prefix} ({'Standardized to' if standardize else 'Absolute values vs'} {name1})\n")
            f.write("-" * 100 + "\n")
            f.write(print_df.to_string() + "\n\n")

            f.write(f"PART 2: INDIVIDUAL PARAMETER SPREADS (Raw Physical Units)\n")
            f.write("-" * 100 + "\n")
            f.write(f"BASELINE DATASET: {name1}\n")
            # Note: Assuming get_divided_spread_string is still in your script scope!
            f.write(get_divided_spread_string(spread_ds1) + "\n\n")
            
            f.write(f"COMPARISON DATASET: {name2}\n")
            f.write(get_divided_spread_string(spread_ds2) + "\n")
            
            f.write("\n" + "=" * 100 + "\n")
            f.write("REPORT END\n")

        print(f"Analysis saved to: {save_path}")
        
        print("\n==========================================")
        print("--- GENERATING HTML DASHBOARDS ---")
        print("==========================================")
        
        # Strings für die Textboxen generieren
        str_summary = print_df.to_string()
        str_spread1 = get_divided_spread_string(spread_ds1)
        str_spread2 = get_divided_spread_string(spread_ds2)

        # ==========================================================
        # 1. MAIN DASHBOARD (Alles Text + Paarweise Visualisierungen)
        # ==========================================================
        boxplots_main = [p for p in plots_ds1 + plots_ds2 if p['plot_type'] == 'Boxplot']
        bar_plots_main = [p for p in plots_ds1 + plots_ds2 if p['plot_type'] == 'Sensor Breakdown']
        main_dash_plots = boxplots_main + bar_plots_main + meta_hist_plots

        create_interactive_dashboard(
            page_title=f"MAIN DASHBOARD: {name2} vs {name1}",
            data_desc=data_desc,
            text_sections=[
                (f"PART 1: {title_prefix}", str_summary),
                (f"PART 2: Baseline Spread ({name1})", str_spread1),
                (f"PART 3: Comparison Spread ({name2})", str_spread2)
            ],
            html_plots=main_dash_plots, 
            save_path=os.path.join(save_path, f"1_Main_Comparison_{name1}_vs_{name2}_{data_desc}.html"),
            pair_by_parameter=True 
        )

        # ==========================================================
        # 2. INTRA-DATASET SPREAD DS1
        # ==========================================================
        p_html_ds1, mi_html_ds1 = analyze_relationships(df1, plot_matrix=True)

        # Plots für DS1 filtern
        ranking_ds1 = [p['html'] for p in plots_ds1 if p['plot_type'] == 'Worst Offenders']
        boxplots_ds1 = [p['html'] for p in plots_ds1 if p['plot_type'] == 'Boxplot']
        # NEU: Barplots für die Statistiken in DS1 filtern
        bar_plots_ds1 = [p['html'] for p in plots_ds1 if p['plot_type'] == 'Sensor Breakdown']
        struct_plots_ds1 = [p['html'] for p in plots_ds1 if p['plot_type'] == 'Structural Comparison']

        # HTML Content für DS1 bauen
        intra_content_ds1 = f'<div class="text-block"><h3>Parameter Spreads: {name1}</h3><pre>{str_spread1}</pre></div>\n'
        
        intra_content_ds1 += "<hr><h2>🚨 Worst Offenders (Top 3 Auffälligkeiten)</h2>\n"
        for rank_html in ranking_ds1: intra_content_ds1 += rank_html

        intra_content_ds1 += "<hr><h2>Rohdaten-Verteilung (Boxplots)</h2>\n"
        for box_html in boxplots_ds1: intra_content_ds1 += f'<div class="plot-block">{box_html}</div>\n'

        # NEU: Statistische Balkendiagramme unter die Boxplots einfügen
        intra_content_ds1 += "<hr><h2>Statistische Kennwerte (Balkendiagramme)</h2>\n"
        for bar_html in bar_plots_ds1: intra_content_ds1 += f'<div class="plot-block">{bar_html}</div>\n'

        intra_content_ds1 += f"""
        <hr><h2>Beziehungs-Analyse (Korrelation & Abhängigkeit)</h2>
        <div style="display: flex; flex-wrap: wrap; gap: 30px; justify-content: center; margin-bottom: 40px;">
            <div style="flex: 1; min-width: 450px; max-width: 800px; background: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">{p_html_ds1}</div>
            <div style="flex: 1; min-width: 450px; max-width: 800px; background: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">{mi_html_ds1}</div>
        </div>
        """

        intra_content_ds1 += "<hr><h2>Detail-Analyse: Sensoren mi größtem Unterschied</h2>\n"
        for struct_html in struct_plots_ds1: intra_content_ds1 += f'<div class="plot-block">{struct_html}</div>\n'

        create_interactive_dashboard(
            page_title=f"INTRA-DATASET SPREAD: {name1}",
            data_desc=data_desc,
            text_sections=[("Kompletter Bericht", intra_content_ds1)], 
            html_plots=[], 
            save_path=os.path.join(save_path, f"2_Intra_Spread_{name1}_{data_desc}.html"),
            pair_by_parameter=False
        )

        # ==========================================================
        # 3. INTRA-DATASET SPREAD DS2
        # ==========================================================
        p_html_ds2, mi_html_ds2 = analyze_relationships(df2, plot_matrix=True)

        # Plots für DS2 filtern
        ranking_ds2 = [p['html'] for p in plots_ds2 if p['plot_type'] == 'Worst Offenders']
        boxplots_ds2 = [p['html'] for p in plots_ds2 if p['plot_type'] == 'Boxplot']
        # NEU: Barplots für die Statistiken in DS2 filtern
        bar_plots_ds2 = [p['html'] for p in plots_ds2 if p['plot_type'] == 'Sensor Breakdown']
        struct_plots_ds2 = [p['html'] for p in plots_ds2 if p['plot_type'] == 'Structural Comparison']

        # HTML Content für DS2 bauen
        intra_content_ds2 = f'<div class="text-block"><h3>Parameter Spreads: {name2}</h3><pre>{str_spread2}</pre></div>\n'
        
        intra_content_ds2 += "<hr><h2>🚨 Worst Offenders (Top 3 Auffälligkeiten)</h2>\n"
        for rank_html in ranking_ds2: intra_content_ds2 += rank_html

        intra_content_ds2 += "<hr><h2>Rohdaten-Verteilung (Boxplots)</h2>\n"
        for box_html in boxplots_ds2: intra_content_ds2 += f'<div class="plot-block">{box_html}</div>\n'

        # NEU: Statistische Balkendiagramme unter die Boxplots einfügen
        intra_content_ds2 += "<hr><h2>Statistische Kennwerte (Balkendiagramme)</h2>\n"
        for bar_html in bar_plots_ds2: intra_content_ds2 += f'<div class="plot-block">{bar_html}</div>\n'

        intra_content_ds2 += f"""
        <hr><h2>Beziehungs-Analyse (Korrelation & Abhängigkeit)</h2>
        <div style="display: flex; flex-wrap: wrap; gap: 30px; justify-content: center; margin-bottom: 40px;">
            <div style="flex: 1; min-width: 450px; max-width: 800px; background: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">{p_html_ds2}</div>
            <div style="flex: 1; min-width: 450px; max-width: 800px; background: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">{mi_html_ds2}</div>
        </div>
        """

        intra_content_ds2 += "<hr><h2>Detail-Analyse: Sensoren mit größtem Unterschied</h2>\n"
        for struct_html in struct_plots_ds2: intra_content_ds2 += f'<div class="plot-block">{struct_html}</div>\n'

        create_interactive_dashboard(
            page_title=f"INTRA-DATASET SPREAD: {name2}",
            data_desc=data_desc,
            text_sections=[("Kompletter Bericht", intra_content_ds2)], 
            html_plots=[], 
            save_path=os.path.join(save_path, f"3_Intra_Spread_{name2}_{data_desc}.html"),
            pair_by_parameter=False
        )

        # ==========================================================
        # 4. INTER-DATASET SPREAD (Shifts zwischen den Datensätzen)
        # ==========================================================
        create_interactive_dashboard(
            page_title=f"INTER-DATASET SHIFTS: {name2} vs {name1}",
            data_desc=data_desc,
            text_sections=[(f"Shift Summary Table", str_summary)],
            html_plots=inter_html_plots,
            save_path=os.path.join(save_path, f"4_Inter_Dataset_Shifts.html"),
            pair_by_parameter=False
        )
        
    return comparison_df

def create_interactive_dashboard(page_title, data_desc, text_sections, html_plots, save_path, pair_by_parameter=False):
    import os
    
    # =========================================================
    # NEU: Das Erklärungs-Wörterbuch für die Statistiken
    # =========================================================
    param_descriptions = {
        'Mean': 'Der durchschnittliche Wert des Sensors. Zeigt an, ob ein Sensor systematisch höher oder niedriger misst als der Rest des Datensatzes.',
        'Std': 'Die Standardabweichung (absolutes Rauschen). Ein höheres Level bedeutet, dass der Sensorwert stärker schwankt und unruhiger ist.',
        'CV': 'Der Variationskoeffizient (relatives Rauschen). Setzt die Streuung ins Verhältnis zum Mittelwert. Ideal, um Rauschen unabhängig vom absoluten Messlevel zu vergleichen.',
        'IQR': 'Der Interquartilsabstand (robuste Streuung). Betrachtet nur die mittleren 50% der Daten und ignoriert extreme, einzelne Ausreißer.',
        'Skewness': 'Die Schiefe der Verteilung. Ein Wert nahe 0 bedeutet Symmetrie. Positive Werte bedeuten häufigere Ausreißer nach oben, negative nach unten.',
        'Kurtosis': 'Die Wölbung der Verteilung. Ein hoher Wert bedeutet: Das Signal verlässt gelegentlich drastisch seine sonst enge gestreute Verteilung (plötzliche Spikes). Ein niedriger Wert steht für ein gleichmäßiges Schwanken ohne überraschende Sprünge.',
        'Slope': 'Die Steigung (Drift). Zeigt an, ob der Sensor über den gemessenen Zeitraum systematisch ansteigt (positiv) oder abfällt (negativ).',
        'NrAnomalies': 'Die absolute Anzahl der statistischen Anomalien/Ausreißer basierend auf der IQR Methode (Faktor 1.5), die für diesen Sensor in seinem Zeitverlauf erkannt wurden.'
    }

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{page_title}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; color: #333; margin: 0; padding: 40px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .container {{ max-width: 1400px; margin: auto; }}
            .text-block {{ background: #fff; padding: 25px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 30px; overflow-x: auto; }}
            pre {{ font-family: Consolas, 'Courier New', monospace; font-size: 13px; line-height: 1.5; color: #1a1a1a; }}
            .plot-block {{ background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 40px; }}
            hr {{ border: 0; height: 1px; background: #e0e0e0; margin: 20px 0; }}
            .param-desc {{ font-size: 15px; color: #7f8c8d; font-style: italic; margin-top: -10px; margin-bottom: 25px; }} /* Styling für die Beschreibung */
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{page_title}</h1>
            <p><strong>Target Data Description:</strong> {data_desc if data_desc else 'N/A'}</p>
    """
    
    # 1. Flexible Text-Blöcke einfügen
    for section_title, content in text_sections:
        if content.strip().startswith('<div'):
            html_content += content
        else:
            html_content += f'<div class="text-block"><h3>{section_title}</h3><pre>{content}</pre></div>\n'

    # 2. Plot-Logik
    if pair_by_parameter:
        params = []
        for p in html_plots:
            if p['parameter'] not in params: params.append(p['parameter'])
            
        for p_name in params:
            html_content += f"<hr><h2 style='color:#2c3e50;'>Statistik: {p_name}</h2>\n"
            
            # NEU: Erklärung aus dem Dictionary einfügen (falls vorhanden)
            desc = param_descriptions.get(p_name, "")
            if desc:
                html_content += f"<p class='param-desc'>{desc}</p>\n"
                
            for plot_data in html_plots:
                if plot_data['parameter'] == p_name:
                    html_content += f"""
                    <div class="plot-block">
                        {plot_data['html']}
                    </div>
                    """
    else:
        current_dataset = None
        current_family = None
        for plot_data in html_plots:
            if plot_data['dataset'] != current_dataset:
                current_dataset = plot_data['dataset']
                html_content += f'<hr><h2 style="color:#2c3e50; margin-top:40px;">Dataset: {current_dataset}</h2>\n'
                current_family = None 
                
            if plot_data['family'] != current_family:
                current_family = plot_data['family']
                if current_family not in ["Global", "Uncategorized"]:
                    html_content += f'<h3 style="color:#7f8c8d; margin-top:30px;">Family: {current_family}</h3>\n'

            # NEU: Auch hier im Einzel-Dashboard die Beschreibung über den Plot setzen
            p_name = plot_data['parameter']
            desc = param_descriptions.get(p_name, "")
            
            html_content += f"""
            <h4 style="margin-top:0; color:#bdc3c7;">{p_name} - {plot_data['plot_type']}</h4>
            """
            
            if desc:
                 html_content += f"<p class='param-desc' style='font-size: 13px;'>{desc}</p>\n"
                 
            html_content += f"""
            <div class="plot-block">
                {plot_data['html']}
            </div>
            """

    html_content += "\n</div></body></html>"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"-> Saved Dashboard: {save_path}")
    
    
def analyze_parallel_deviation_from_mean(df, normalize=True,linearize_time=True):
    """
    Analyzes deviations from the row mean for each timestamp and identifies columns 
    with the most significant deviations.
    
    Parameters:
    df: pandas DataFrame where index is timestamps and columns are parallel series
    normalize: If True, normalizes deviations by dividing by the row standard deviation
               to account for different scales at different timestamps
    
    Returns:
    Dictionary containing:
    - total_absolute_deviation: Sum of absolute deviations for each column
    - frequency_of_significant_deviation: Count of times each column exceeded 1 std from mean
    - most_deviating_column: Name of the column with highest total deviation
    - frequent_outlier_column: Name of the column most frequently deviating significantly
    - mean_deviation: Mean deviation (positive or negative) for each column
    - mean_deviation_percentage: Mean deviation as a percentage of the row mean
    """
    
    df['mean'] = df.mean(axis=1)
    # Calculate row means (mean across all columns for each timestamp)
    row_means = df['mean']
    
    # Calculate deviations from the row mean
    deviations = df.sub(row_means, axis=0)
    
    # Calculate mean deviation for each column (this captures direction - positive or negative bias)
    mean_deviation = deviations.mean()
    
    # Calculate percentage deviations (deviation as percentage of row mean)
    # Add small epsilon to avoid division by zero
    overall_mean = df.values.mean()  # Global mean of all values
    mean_deviation_percentage = (mean_deviation / overall_mean) * 100
    
     # Calculate row standard deviations
    row_stds = df.std(axis=1)
    # Calculate row standard deviations of deviations
    row_deviation_stds = deviations.std(axis=1)
    # Normalize deviations by row standard deviation
    # Add small epsilon to avoid division by zero
    normalized_deviations = deviations.div(row_deviation_stds + 1e-10, axis=0)
    if normalize:
        deviation_matrix = normalized_deviations
    else:
        deviation_matrix = deviations
    
    
    #calculate basic stats
    stats_dict = get_basic_statistics_from_parallel_ts(df)
    #get column with highest std deviation from mean
    mean_std = stats_dict['combined']['std']
    #get columns with highest std deviation from mean
    std_deviation = df.std() - mean_std
    #get the column with the highest std deviation from mean
    max_std_deviation_col = std_deviation.idxmax()
    #get the std deviation of the highest std deviation column
    max_std_deviation = std_deviation[max_std_deviation_col]
    
    #get highest skewness
    skewness = df.skew()
    #get the column with the highest skewness
    max_skewness_col = skewness.abs().idxmax()
    #get the skewness of the highest skewness column
    max_skewness = skewness[max_skewness_col]
    
    #get highest kurtosis
    kurtosis = df.kurt()
    #get the column with the highest kurtosis
    max_kurtosis_col = kurtosis.abs().idxmax()
    #get the kurtosis of the highest kurtosis column
    max_kurtosis = kurtosis[max_kurtosis_col]

    #clac slope of original data
    slop_orig_df = analyse_slope(df,linearize_time=linearize_time)
    #calculate time when slope is at -1 or 1
    avg_slope_orig = slop_orig_df['slope'].mean()
    #get highest slope
    highest_slope_orig = slop_orig_df['slope'].idxmax()
    #get col of highest deviation slope from avg slope
    slope_diff_orig = (slop_orig_df['slope'] - avg_slope_orig).abs()
    max_slope_diff_col_orig = slope_diff_orig.idxmax()
    #get the slope of the highest deviation column
    max_slope_diff_orig = slop_orig_df['slope'][max_slope_diff_col_orig]
    #get date where slope is at -1 or 1
    slope_date = find_threshold_datetime(df, slop_orig_df['slope'][max_slope_diff_col_orig], slop_orig_df['intercept'][max_slope_diff_col_orig], (-1, 1), max_years=5)
    
    #calculate slope of data deviation
    slope_df = analyse_slope(deviation_matrix,linearize_time=linearize_time)
    #calculate avg slope
    avg_slope = slope_df['slope'].mean()
    #get highest slope
    highest_slope = slope_df['slope'].idxmax()
    #get col of highest deviation slope from avg slope
    slope_diff = (slope_df['slope'] - avg_slope).abs()
    max_slope_diff_col = slope_diff.idxmax()
    #get the slope of the highest deviation column
    max_slope_diff = slope_df['slope'][max_slope_diff_col]
    
    
    # Calculate absolute deviations for each column
    abs_deviations = deviation_matrix.abs()
    
    # Sum the absolute deviations for each column
    total_abs_deviation = abs_deviations.sum()
    
    # Count how often each column exceeds 1 standard deviation from the mean
    std = df.std()
    is_significant_deviation = abs_deviations > 3.0 if normalize else abs_deviations > 3* std
    frequency_of_significant_deviation = is_significant_deviation.sum()
    
    # Find the column with the highest total deviation
    most_deviating_column = total_abs_deviation.idxmax()
    
    # Find the column that most frequently has significant deviations
    frequent_outlier_column = frequency_of_significant_deviation.idxmax()
    
    #find the column that has the single highest deviation
    single_highest_deviation_column = abs_deviations.max().idxmax()
    
    print('*' * 50)
    print(f"\nMost deviating column: {most_deviating_column} with total deviation {total_abs_deviation[most_deviating_column]}")
    print(f"Frequent outlier column: {frequent_outlier_column} with frequency {frequency_of_significant_deviation[frequent_outlier_column]}")
    print(f'highest single deviation column: {single_highest_deviation_column} with deviation {abs_deviations.max()[single_highest_deviation_column]}')
    print(f'column with highest slope deviation from deviation data: {max_slope_diff_col} with slope {max_slope_diff} from avg slope {avg_slope}')
    print(f'column with highest slope deviation from original data: {max_slope_diff_col_orig} with slope {max_slope_diff_orig} from avg slope {avg_slope_orig}')
    str_add = f'at {slope_date} ' if slope_date else f'not in under 5 years'
    print(f'reaching threshold '+str_add)
    print(f'column with highest std deviation from mean std: {max_std_deviation_col} with std deviation {max_std_deviation} from mean {mean_std}')
    print(f'column with highest skewness: {max_skewness_col} with skewness {max_skewness}')
    print(f'column with highest kurtosis: {max_kurtosis_col} with kurtosis {max_kurtosis}')
    print('*' * 50)
    create_analysis_plots(df, normalized_deviations,'mean',stats_dict, title_add=' (mean of all columns)',linearize_time=linearize_time)
    create_analysis_plots(df, normalized_deviations, most_deviating_column,  stats_dict, title_add=' (most deviating column)',linearize_time=linearize_time)
    create_analysis_plots(df, normalized_deviations, frequent_outlier_column,  stats_dict, title_add=' (frequent outlier column)',linearize_time=linearize_time)
    create_analysis_plots(df, normalized_deviations, single_highest_deviation_column,  stats_dict, title_add=' (single highest deviation column)',linearize_time=linearize_time)
    create_analysis_plots(df, normalized_deviations, max_slope_diff_col,  stats_dict, title_add=' (highest slope deviation column)',linearize_time=linearize_time)
    create_analysis_plots(df, normalized_deviations, max_std_deviation_col,  stats_dict, title_add=' (highest std deviation column)',linearize_time=linearize_time)
    create_analysis_plots(df, normalized_deviations, max_skewness_col,  stats_dict, title_add=' (highest skewness column)',linearize_time=linearize_time)
    create_analysis_plots(df, normalized_deviations, max_kurtosis_col,  stats_dict, title_add=' (highest kurtosis column)',linearize_time=linearize_time)
    
    
    top_deviaters = {}
    top_deviaters[max_slope_diff_col] = {}
    top_deviaters[max_slope_diff_col]['avg deviation'] = mean_deviation[max_slope_diff_col]
    top_deviaters[max_slope_diff_col]['avg deviation percentage'] = mean_deviation_percentage[max_slope_diff_col]
    top_deviaters[max_slope_diff_col]['deviation'] = total_abs_deviation[max_slope_diff_col]
    top_deviaters[max_slope_diff_col]['frequency'] = frequency_of_significant_deviation[max_slope_diff_col]
    top_deviaters[max_slope_diff_col]['highest deviation'] = abs_deviations[max_slope_diff_col].max()
    
    
    print('\nTop 5 most deviating columns with total deviation:')
    #get the top 5 most deviating columns and ther total deviation and pack them in a dictionary
    top_5_deviating = total_abs_deviation.nlargest(5)
    for col in top_5_deviating.index:
        print(f"{col}: {total_abs_deviation[col]}")
        if col not in top_deviaters:
            top_deviaters[col] = {}
            top_deviaters[col]['avg deviation'] = mean_deviation[col]
            top_deviaters[col]['avg deviation percentage'] = mean_deviation_percentage[col]
            top_deviaters[col]['deviation'] = total_abs_deviation[col]
            top_deviaters[col]['frequency'] = frequency_of_significant_deviation[col]
            top_deviaters[col]['highest deviation'] = abs_deviations[col].max()
            
    print('\nTop 5 most frequent outlier columns with number of deviations:')
    top_5_frequent = frequency_of_significant_deviation.nlargest(5)
    for col in top_5_frequent.index:
        print(f"{col}: {frequency_of_significant_deviation[col]}")
        if col not in top_deviaters:
            top_deviaters[col] = {}
            top_deviaters[col]['avg deviation'] = mean_deviation[col]
            top_deviaters[col]['avg deviation percentage'] = mean_deviation_percentage[col]
            top_deviaters[col]['deviation'] = total_abs_deviation[col]
            top_deviaters[col]['frequency'] = frequency_of_significant_deviation[col]
            top_deviaters[col]['highest deviation'] = abs_deviations[col].max()
    
    print('\nTop 5 most mean deviation columns with mean deviation:')
    #get the top 5 most deviating columns and their mean deviation and pack them in a dictionary
    top_5_mean_deviation = mean_deviation.abs().nlargest(5)
    for col in top_5_mean_deviation.index:
        print(f"{col}: {mean_deviation[col]} - {mean_deviation_percentage[col]:.2f}%")
        if col not in top_deviaters:
            top_deviaters[col] = {}
            top_deviaters[col]['avg deviation'] = mean_deviation[col]
            top_deviaters[col]['avg deviation percentage'] = mean_deviation_percentage[col]
            top_deviaters[col]['deviation'] = total_abs_deviation[col]
            top_deviaters[col]['frequency'] = frequency_of_significant_deviation[col]
            top_deviaters[col]['highest deviation'] = abs_deviations[col].max()
            
    print('\nTop 5 highest single deviations:')
    #get the top 5 highest single deviations and their mean deviation and pack them in a dictionary
    top_5_single_deviation = abs_deviations.max().nlargest(5)
    for col in top_5_single_deviation.index:
        print(f"{col}: {top_5_single_deviation[col]}")
        if col not in top_deviaters:
            top_deviaters[col] = {}
            top_deviaters[col]['avg deviation'] = mean_deviation[col]
            top_deviaters[col]['avg deviation percentage'] = mean_deviation_percentage[col]
            top_deviaters[col]['deviation'] = total_abs_deviation[col]
            top_deviaters[col]['frequency'] = frequency_of_significant_deviation[col]
            top_deviaters[col]['highest deviation'] = abs_deviations[col].max()
            
    #sort the dictionary by avg deviation
    top_deviaters = dict(sorted(top_deviaters.items(), key=lambda item: item[1]['avg deviation'], reverse=True))
    top_deviaters_df = pd.DataFrame.from_dict(top_deviaters, orient='index')
    #sort by avg deviation
    top_deviaters_df = top_deviaters_df.sort_values(by='avg deviation', ascending=False)
    # print('\nTop most irregular columns:')
    # pprint.pprint(top_deviaters_df)

    # print('\nMost irregular columns:')
    # pprint.pprint(top_deviaters)
    
    results = pd.DataFrame()
    results['total_absolute_deviation'] = total_abs_deviation
    results['frequency_of_significant_deviation'] = frequency_of_significant_deviation
    results['mean_deviation'] = mean_deviation
    results['mean_deviation_percentage'] = mean_deviation_percentage
    #set index starting at 1
    results.index = range(1, len(results) + 1)

    
    return top_deviaters_df, results

def plot_acf_pacf(ts, lags=40):
    """
    Plot the AutoCorrelation Function (ACF) and Partial AutoCorrelation Function (PACF).
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plot_acf(ts, lags=lags, ax=plt.gca())
    plt.title("ACF")
    plt.subplot(1,2,2)
    plot_pacf(ts, lags=lags, ax=plt.gca(), method='ywm')
    plt.title("PACF")
    plt.tight_layout()
    plt.show()
    

def filter_ouliers(df, sigma_thr=10):
    """
    Filter out outliers from a DataFrame using Z-score method.
    """
    # FIX: Force all data to numeric types. 
    # 'errors="coerce"' turns strings/non-parseable data into NaN.
    df_numeric = df.apply(pd.to_numeric, errors='coerce')

    # Calculate Z-scores (Now safe because df_numeric is guaranteed to be numbers)
    z_scores = (df_numeric - df_numeric.mean()) / df_numeric.std()
    
    # Replace outliers with NaN
    df_filtered = df_numeric.where(np.abs(z_scores) <= sigma_thr, np.nan)
    
    # Replace nan with interpolation
    df_filtered = df_filtered.interpolate(method='linear', limit_direction='both')
    
    # Interpolate over axis 1 of still NaN values
    if df_filtered.isnull().values.any():
        df_filtered = df_filtered.interpolate(method='linear', axis=1, limit_direction='both')
    
    return df_filtered

def filter_outliers_quantile(df, factor=1.5, keep_nans=True):
    """
    Filter out outliers from a DataFrame using the Interquartile Range (IQR) method.
    'factor' is typically 1.5 for outliers and 3.0 for extreme outliers.
    'keep_nans' if True, preserves original NaN positions after interpolation.
    """
    # Force numeric types
    df_numeric = df.apply(pd.to_numeric, errors='coerce')

    # Snapshot original NaN positions before any processing
    original_nans = df_numeric.isna()

    # Calculate quartiles and IQR
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds
    lower_bound = Q1 - (factor * IQR)
    upper_bound = Q3 + (factor * IQR)

    # Replace outliers with NaN
    mask = (df_numeric >= lower_bound) & (df_numeric <= upper_bound)
    df_filtered = df_numeric.where(mask, np.nan)

    # Interpolate over outlier positions (not original NaNs)
    df_filtered = df_filtered.interpolate(method='linear', limit_direction='both')

    # Interpolate over axis=1 if still NaN values
    if df_filtered.isnull().values.any():
        df_filtered = df_filtered.interpolate(method='linear', axis=1, limit_direction='both')

    # Restore original NaN positions
    if keep_nans:
        df_filtered[original_nans] = np.nan

    return df_filtered


def find_anomalies_quantile(df, factor=1.5, global_quantiles=True):
    """
    Finds anomalies in a DataFrame using the Interquartile Range (IQR) method.
    
    Parameters:
    - factor: The IQR multiplier for bounds (default 1.5).
    - global_quantiles: If True, calculates one shared threshold across all data. 
                        If False, calculates independent thresholds per column.
    """
    # Force numeric types
    df_numeric = df.apply(pd.to_numeric, errors='coerce')

    if global_quantiles:
        # --- GLOBAL THRESHOLDS ---
        # Flatten and remove NaNs to get fleet-wide statistics
        all_values = df_numeric.to_numpy().flatten()
        all_values = all_values[~np.isnan(all_values)]
        
        Q1 = np.percentile(all_values, 25)
        Q3 = np.percentile(all_values, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - (factor * IQR)
        upper_bound = Q3 + (factor * IQR)
        
    else:
        # --- LOCAL THRESHOLDS ---
        # Calculate per-column statistics (pandas returns a Series)
        Q1 = df_numeric.quantile(0.25)
        Q3 = df_numeric.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - (factor * IQR)
        upper_bound = Q3 + (factor * IQR)

    # Create the anomaly mask
    # Pandas elegantly handles comparing the dataframe to either the 
    # global scalar values or the local Series values automatically!
    anomaly_mask = (df_numeric < lower_bound) | (df_numeric > upper_bound)

    # Extract the anomalies
    anomalies_per_column = {}
    for col in df_numeric.columns:
        anomalies_per_column[col] = df_numeric.loc[anomaly_mask[col], col]

    return anomalies_per_column


def visualize_data(df, title, ztitle):
    # Get chamber numbers and datetime values
    column_names = df.columns.tolist()
    time_vals = df['datetime'] if 'datetime' in df.columns else df.index

    #get global min and max
    min_val = df.iloc[:, 1:].min().min()  # Minimum voltage value
    max_val = df.iloc[:, 1:].max().max()  # Maximum voltage value
    value_range=[min_val, max_val]
    # Create 3D surface plot
    fig = go.Figure(data=[go.Surface(
        z=df.iloc[:, 1:].values,  # Voltage values (all columns except datetime)
        x=column_names[1:],       # Chamber numbers on x-axis
        y=time_vals,              # Time on y-axis
        colorscale='Viridis',
        colorbar=dict(title=title),
        cmin=value_range[0],         # Set minimum color value
        cmax=value_range[1]          # Set maximum color value
    )])

    # Create a more explicit 3D grid configuration
    fig.update_layout(
        title='3D Visualization of ' + title,
        scene=dict(
            # X-axis settings
            xaxis=dict(
                title=title,
                showgrid=True,
                gridcolor='rgb(128, 128, 128)',  # Solid gray, more visible
                gridwidth=1,                     # Thicker grid lines
                showline=True,
                showticklabels=True,
                zeroline=True,
                zerolinecolor='rgb(0, 0, 0)',    # Black zero line
                zerolinewidth=2,
                tickmode='array',           # Tell Plotly to use your specific list
                tickvals=column_names[1:],  # The locations of the ticks (your columns)
                ticktext=column_names[1:],
            ),
            # Y-axis settings
            yaxis=dict(
                title='Time',
                showgrid=True,
                gridcolor='rgb(128, 128, 128)',
                gridwidth=2,
                showline=True,
                showticklabels=True,
                zeroline=True,
                zerolinecolor='rgb(0, 0, 0)',
                zerolinewidth=2
            ),
            # Z-axis settings
            zaxis=dict(
                title=ztitle,
                range=value_range,
                showgrid=True,
                gridcolor='rgb(128, 128, 128)',
                gridwidth=2,
                showline=True,
                showticklabels=True,
                zeroline=True,
                zerolinecolor='rgb(0, 0, 0)',
                zerolinewidth=2
            ),
            # Background settings - use explicit RGB settings
            xaxis_backgroundcolor='rgb(230, 230, 230)',  # Lighter gray
            yaxis_backgroundcolor='rgb(230, 230, 230)',
            zaxis_backgroundcolor='rgb(230, 230, 230)',
            # Make sure backgrounds are visible
            xaxis_showbackground=True,
            yaxis_showbackground=True,
            zaxis_showbackground=True,
            # Camera positioning for better view of the grid
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
            aspectratio=dict(x=1.5, y=1.5, z=0.7)
        ),
        width=1200,
        height=1000,
        margin=dict(l=65, r=50, b=65, t=90)
    )

    # Show the figure
    fig.show()
    
    
    
    
import plotly.graph_objects as go
import pandas as pd  # Assuming you're using pandas for DataFrames
import numpy as np  # For handling inf/NaN

def visualize_data_list(dfs, title, ztitle, gap=1, long_format=True, 
                        datetime_col='time_stamp', chamber_col='series_id', value_col='value', 
                        aggfunc=None, abbreviate_labels=False, save_path=None):
    if not dfs:
        print("No DataFrames provided.")
        return

    # Compute global min and max across all DataFrames, and max_chambers
    global_min = float('inf')
    global_max = float('-inf')
    max_chambers = 0
    processed_dfs = []  # Store processed wide DataFrames

    for df in dfs:
        # Ensure datetime column is actual datetime
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        if long_format:
            # Pivot to wide format (ignores extra columns like 'client_id')
            if aggfunc is None:
                wide_df = df.pivot(index=datetime_col, columns=chamber_col, values=value_col)
            else:
                wide_df = pd.pivot_table(df, index=datetime_col, columns=chamber_col, values=value_col, aggfunc=aggfunc)
            # Sort columns for consistent ordering (numerical if possible, else alphabetical)
            wide_df = wide_df.sort_index(axis=1)
        else:
            wide_df = df.set_index(datetime_col) if datetime_col in df.columns else df
            wide_df = wide_df.sort_index(axis=1)

        # Ensure the DataFrame is numeric (coerce any non-numeric to NaN)
        wide_df = wide_df.apply(pd.to_numeric, errors='coerce')

        # Store for later use
        processed_dfs.append(wide_df)

        # Compute stats (skip if all NaN)
        if not wide_df.empty:
            min_val = wide_df.min().min()
            max_val = wide_df.max().max()
            if pd.notna(min_val):
                global_min = min(global_min, min_val)
            if pd.notna(max_val):
                global_max = max(global_max, max_val)
            num_chambers = len(wide_df.columns)
            max_chambers = max(max_chambers, num_chambers)

    if np.isinf(global_min) or np.isinf(global_max):
        print("No valid numeric data found in the DataFrames.")
        return

    value_range = [global_min, global_max]

    # Create the figure
    fig = go.Figure()

    # Lists for custom x-axis ticks
    all_tickvals = []
    all_ticktext = []

    # Add a surface trace for each processed wide DataFrame with x-offset
    for i, wide_df in enumerate(processed_dfs):
        if wide_df.empty:
            continue
        original_chambers = wide_df.columns.tolist()  # Chamber names (sorted)
        num_chambers = len(original_chambers)
        time_vals = wide_df.index  # Time on y-axis (assumed to be the index after pivot/set_index)

        # Compute offset numerical x positions
        start_pos = i * (max_chambers + gap)
        x_num = [start_pos + j for j in range(num_chambers)]

        # Add custom tick labels (e.g., "DF1_Water_Primary_Temperature")
        if abbreviate_labels:
            abbreviated = [f"DF{i+1}_" + "_".join(word[0] for word in ch.split("_")) for ch in original_chambers]
        else:
            abbreviated = [f"DF{i+1}_{ch}" for ch in original_chambers]
        all_tickvals.extend(x_num)
        all_ticktext.extend(abbreviated)

        # Add the surface trace
        fig.add_trace(go.Surface(
            z=wide_df.values,         # Values (numeric array)
            x=x_num,                  # Offset numerical x positions
            y=time_vals,              # Time on y-axis
            colorscale='Viridis',
            colorbar=dict(title=title) if i == 0 else None,  # Show colorbar only on first trace
            cmin=value_range[0],      # Global min for color
            cmax=value_range[1],      # Global max for color
            showscale=(i == 0)        # Ensure colorbar is only shown once
        ))

    # Calculate dynamic scaling for aspect ratio and width
    if all_tickvals:
        total_x_span = max(all_tickvals) - min(all_tickvals) + 1
        num_ticks = len(all_tickvals)
    else:
        total_x_span = 1
        num_ticks = 0
    dynamic_aspect_x = max(1.1, total_x_span * 0.1)  # Scale x based on total span (adjust 0.1 if needed)
    dynamic_width = 1000 + (num_ticks * 5)  # Add space per tick for labels

    # Update layout with dynamic scaling and label improvements
    fig.update_layout(
        title='3D Visualization of ' + title,
        scene=dict(
            # X-axis settings (now with custom ticks, rotation, and font)
            xaxis=dict(
                title='Series',
                showgrid=True,
                gridcolor='rgb(128, 128, 128)',  # Solid gray, more visible
                gridwidth=2,                     # Thicker grid lines
                showline=True,
                showticklabels=True,
                zeroline=True,
                zerolinecolor='rgb(0, 0, 0)',    # Black zero line
                zerolinewidth=2,
                tickmode='array',                # Custom ticks
                tickvals=all_tickvals,           # Numerical positions
                ticktext=all_ticktext,           # Labeled with DF prefix and original chamber
                #tickangle=-45,                   # Rotate labels for readability
                tickfont=dict(size=10)           # Smaller font to reduce crowding
            ),
            # Y-axis settings
            yaxis=dict(
                title='Time',
                showgrid=True,
                gridcolor='rgb(128, 128, 128)',
                gridwidth=2,
                showline=True,
                showticklabels=True,
                zeroline=True,
                zerolinecolor='rgb(0, 0, 0)',
                zerolinewidth=2
            ),
            # Z-axis settings
            zaxis=dict(
                title=ztitle,
                range=value_range,  # Global range for z-axis
                showgrid=True,
                gridcolor='rgb(128, 128, 128)',
                gridwidth=2,
                showline=True,
                showticklabels=True,
                zeroline=True,
                zerolinecolor='rgb(0, 0, 0)',
                zerolinewidth=2
            ),
            # Background settings - use explicit RGB settings
            xaxis_backgroundcolor='rgb(230, 230, 230)',  # Lighter gray
            yaxis_backgroundcolor='rgb(230, 230, 230)',
            zaxis_backgroundcolor='rgb(230, 230, 230)',
            # Make sure backgrounds are visible
            xaxis_showbackground=True,
            yaxis_showbackground=True,
            zaxis_showbackground=True,
            # Camera positioning for better view of the grid
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
            aspectratio=dict(x=1.5, y=1, z=0.8)  # Dynamic x scaling
        ),
        width=dynamic_width,  # Dynamic width based on number of ticks
        height=1200,
        margin=dict(l=65, r=50, b=65, t=90)
    )

    # Show the figure
    fig.show()
    
    if save_path:
        fig.write_image(save_path)
    
def plot_chamber(df, chamber_num, ax=None, linearize_time=True):
    # Use provided ax if available, otherwise create a new one
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        should_show = True
        title_add = f' for chamber {chamber_num}'
    else:
        should_show = False
        title_add = ''
        
    # Calculate slope of data
    slope_df = analyse_slope(df, linearize_time=linearize_time)
    slope = slope_df.loc[str(chamber_num), 'slope']
    intercept = slope_df.loc[str(chamber_num), 'intercept']
    y_end = slope_df.loc[str(chamber_num), 'y_end']
    start_x = df.index[0]
    end_x = df.index[-1]
    start_y = intercept
    end_y = y_end
    
    df_entries = len(df)
    print(f"Number of entries: {df_entries}")
    time_period_days = (df.index[-1] - df.index[0]).days
    print(f"Time period in days: {time_period_days}")
    #calculate average timepoints per 30 days
    avg_timepoints_per_month = 30 *df_entries // time_period_days
    print(f"Average timepoints per month: {avg_timepoints_per_month}")

    running_std = df[chamber_num].rolling(window=avg_timepoints_per_month, min_periods=1).std()
    
    # Plot the original data
    ax.plot(df.index, df[chamber_num], label='delta voltage', color='blue')
    ax.set_title('Delta Voltage' + title_add)
    ax.plot([start_x, end_x], [start_y, end_y], color='red', label=f'Slope: {slope:.6f}')
    ax.plot(df.index, running_std, label=f'Rolling 30-day StdDev', color='green')
    
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('mV')
    ax.legend()
    ax.grid(True)
    
    if should_show:
        plt.show()
    
def plot_chamber_deviation(normalized_deviations, chamber_num, ax=None,linearize_time=True):
    """
    Plot the deviation of a specific chamber over time.
    
    Parameters:
    normalized_deviations: DataFrame with normalized deviations
    chamber_num: Chamber number to plot
    ax: Optional matplotlib axes to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        should_show = True
        title_add = f' for Chamber {chamber_num}'
    else:
        should_show = False
        title_add = ''
        
    slope_df = analyse_slope(normalized_deviations,linearize_time=linearize_time)
    slope = slope_df.loc[str(chamber_num), 'slope']
    intercept = slope_df.loc[str(chamber_num), 'intercept']
    y_end = slope_df.loc[str(chamber_num), 'y_end']
    start_x = normalized_deviations.index[0]
    end_x = normalized_deviations.index[-1]
    start_y = intercept
    end_y = y_end
    
    df_entries = len(normalized_deviations)
    print(f"Number of entries: {df_entries}")
    time_period_days = (normalized_deviations.index[-1] - normalized_deviations.index[0]).days
    print(f"Time period in days: {time_period_days}")
    #calculate average timepoints per 30 days
    avg_timepoints_per_month = 30 *df_entries // time_period_days
    print(f"Average timepoints per month: {avg_timepoints_per_month}")

    running_std = normalized_deviations[chamber_num].rolling(window=avg_timepoints_per_month, min_periods=1).std()

    ax.plot(normalized_deviations.index, normalized_deviations[chamber_num], label='Deviation from Mean', color='blue')
    ax.plot([start_x, end_x], [start_y, end_y], color='red', label=f'Slope: {slope:.6f}')
    ax.plot(normalized_deviations.index, running_std, label=f'Rolling 30-day StdDev', color='green')
    ax.set_title(f'Deviation from Mean'+title_add)
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Deviation from timestamp mean (σ)')
    ax.legend()
    ax.grid(True)
    
    if should_show:
        plt.show()
        
        
def create_histogram(ts, stats_dict, nr_bins='auto', scale_type='density', x_scale=5, global_mean=None, ax=None, title_addon=None):
    '''
    Create a histogram with normal and skew-normal distribution curves.
    
    Parameters:
    -----------
    ts: pandas Series or DataFrame
        Time series data
    stats_dict: dict
        Dictionary containing statistical moments (mean, std, skew, kurtosis)
    nr_bins: int or 'auto', default 'auto'
        Number of bins for the histogram
    scale_type: str, default 'density'
        Type of scaling for y-axis ('probability', 'density', or 'count')
    x_scale: int, float, or 'all', default 5
        Number of standard deviations to show or 'all' for all data
    ax: matplotlib.axes.Axes, default None
        Axes to plot on, if None, a new figure is created
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import scipy.stats as spstats
    import pandas as pd
    
    assert scale_type in ['probability', 'density', 'count'], "scale_type must be 'probability', 'density', or 'count'"
    assert isinstance(x_scale, (int, float)) or x_scale == 'all', "x_scale must be an integer, float, (meaning the nr of std to show) or 'all' (for all data)"
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        should_show = True
    else:
        should_show = False
    
    mean_val = stats_dict['mean']
    std = stats_dict['std']
    skew = stats_dict['skew']
    kurt = stats_dict['kurtosis']
    
    if isinstance(ts, pd.DataFrame):
        ts = ts.stack().reset_index(drop=True)
    
    # Handle case where nr_bins='auto' by pre-computing bins first
    if nr_bins == 'auto':
        # Get bin edges first without weights
        _, bin_edges = np.histogram(ts, bins='auto')
        nr_bins = bin_edges
    
    # Plot histogram based on scale_type
    if scale_type == 'probability':
        weights = np.ones_like(ts) / len(ts)
        ax.hist(ts, bins=nr_bins, weights=weights, alpha=0.6, label="Data Distribution")
        ylabel = 'Probability'
    elif scale_type == 'density':
        ax.hist(ts, bins=nr_bins, density=True, alpha=0.6, label="Data Distribution")
        ylabel = 'Density'
    elif scale_type == 'count':
        ax.hist(ts, bins=nr_bins, density=False, alpha=0.6, label="Data Distribution")
        ylabel = 'Count'
    else:
        raise ValueError("scale_type must be 'probability', 'density', or 'count'")
    
    # Get histogram information for scaling curves
    hist, bin_edges = np.histogram(ts, bins=nr_bins)
    bin_width = bin_edges[1] - bin_edges[0]
    total_count = len(ts)
    
    x_min = ts.min()
    x_max = ts.max()
    # Set x-limits based on mean and std
    if x_scale == 'all':
        xmin = x_min - std
        xmax = x_max + std
    else:
        # Use mean and std to set x-limits
        xmin = mean_val - x_scale * std
        xmax = mean_val + x_scale * std
    ax.set_xlim(xmin, xmax)
    
    # Generate x values for the distribution curves
    x = np.linspace(xmin, xmax, 100)
    
    # Calculate PDFs (these are in density form)
    p = stats.norm.pdf(x, loc=mean_val, scale=std)
    ps = spstats.skewnorm.fit(ts)
    ys = spstats.skewnorm.pdf(x, *ps)
    
    # Scale the PDFs according to the selected scale type
    if scale_type == 'probability':
        # Scale density to probability
        p = p * bin_width
        ys = ys * bin_width
    elif scale_type == 'count':
        # Scale density to count
        p = p * total_count * bin_width
        ys = ys * total_count * bin_width
    # For density, no scaling needed
    
    # Plot the curves
    ax.plot(x, p, 'r', linewidth=2, label=f"Normal Fit")
    ax.plot(x, ys, 'g', linewidth=2, label="Skew-Normal Fit")
    
    # Plot the mean line
    ax.axvline(mean_val, color='blue', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.2f}')
    if global_mean is not None:
        ax.axvline(global_mean, color='orange', linestyle='--', linewidth=2, label=f'Global Mean = {global_mean:.2f}')
    ax.axvline(x_min, color='black', linestyle='--', linewidth=2, label=f'Min = {x_min:.2f}')
    ax.axvline(x_max, color='black', linestyle='--', linewidth=2, label=f'Max = {x_max:.2f}')
    
    # Add labels and title
    title=f"Data Distribution\nSTD = {std:.3f}, Skewness = {skew:.2f}, Kurtosis = {kurt:.2f}"
    if title_addon:
        title = title_addon+'\n' + title
    ax.set_title(title)
    ax.set_xlabel("Value (mV)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True)
    
    if should_show:
        plt.tight_layout()
        plt.show()
    
    return ax

def create_plotly_histogram(ts, stats_dict, nr_bins='auto', scale_type='density', x_scale=5, global_mean=None, fig=None, row=None, col=None, title_addon=None, add_legend=True):
    '''
    Create a histogram with normal and skew-normal distribution curves using Plotly.
    
    Parameters:
    -----------
    ts: pandas Series or DataFrame
        Time series data
    stats_dict: dict
        Dictionary containing statistical moments (mean, std, skew, kurtosis)
    nr_bins: int or 'auto', default 'auto'
        Number of bins for the histogram
    scale_type: str, default 'density'
        Type of scaling for y-axis ('probability', 'density', or 'count')
    x_scale: int, float, or 'all', default 5
        Number of standard deviations to show or 'all' for all data
    fig: plotly.graph_objects.Figure, default None
        Figure to plot on, if None, a new figure is created
    row: int, default None
        Row index if adding to subplots
    col: int, default None
        Col index if adding to subplots
    '''
    
    assert scale_type in ['probability', 'density', 'count'], "scale_type must be 'probability', 'density', or 'count'"
    assert isinstance(x_scale, (int, float)) or x_scale == 'all', "x_scale must be an integer, float, (meaning the nr of std to show) or 'all' (for all data)"
    
    # Check if we are creating a new figure or adding to an existing one
    standalone = False
    if fig is None:
        fig = go.Figure()
        standalone = True
        
    mean_val = stats_dict['mean']
    std = stats_dict['std']
    skew = stats_dict['skew']
    kurt = stats_dict['kurtosis']
    
    if isinstance(ts, pd.DataFrame):
        ts = ts.stack().reset_index(drop=True)
        
    # Drop NaNs to prevent Scipy fit errors
    ts = ts.dropna()

    # Pre-compute bins to match numpy behavior perfectly
    hist, bin_edges = np.histogram(ts, bins=nr_bins if nr_bins != 'auto' else 'auto')
    bin_width = bin_edges[1] - bin_edges[0]
    total_count = len(ts)
    
    # Map scale_type to Plotly histnorm
    if scale_type == 'probability':
        histnorm = 'probability'
        ylabel = 'Probability'
    elif scale_type == 'density':
        histnorm = 'probability density'
        ylabel = 'Density'
    elif scale_type == 'count':
        histnorm = '' # default
        ylabel = 'Count'

    # 1. Add Histogram Trace
    fig.add_trace(go.Histogram(
        x=ts,
        xbins=dict(start=bin_edges[0], end=bin_edges[-1], size=bin_width),
        histnorm=histnorm,
        name='Data Distribution',
        marker_color='royalblue',
        opacity=0.6,
        showlegend=add_legend
    ), row=row, col=col)
    
    x_min = ts.min()
    x_max = ts.max()
    
    # Set x-limits based on mean and std
    if x_scale == 'all':
        xmin = x_min - std
        xmax = x_max + std
    else:
        xmin = mean_val - x_scale * std
        xmax = mean_val + x_scale * std

    # Generate x values for the distribution curves
    x = np.linspace(xmin, xmax, 100)
    
   # Calculate PDFs ONLY if the data has actual variance
    if ts.std() > 1e-5 and len(ts) > 2:
        try:
            # Calculate PDFs (these are in density form)
            p = spstats.norm.pdf(x, loc=mean_val, scale=std)
            ps = spstats.skewnorm.fit(ts)
            ys = spstats.skewnorm.pdf(x, *ps)
            
            # Scale the PDFs according to the selected scale type
            if scale_type == 'probability':
                p = p * bin_width
                ys = ys * bin_width
            elif scale_type == 'count':
                p = p * total_count * bin_width
                ys = ys * total_count * bin_width

            # 2. Plot the curves
            fig.add_trace(go.Scatter(
                x=x, y=p, mode='lines', line=dict(color='red', width=2), 
                name='Normal Fit', showlegend=add_legend
            ), row=row, col=col)
            
            fig.add_trace(go.Scatter(
                x=x, y=ys, mode='lines', line=dict(color='green', width=2), 
                name='Skew-Normal Fit', showlegend=add_legend
            ), row=row, col=col)
            
        except Exception as e:
            print(f"  -> Skipping curve fit: Data distribution too tight or invalid ({e})")
    else:
        print(f"  -> Skipping curve fit: Sensor data is completely constant.")

    # 3. Add Vertical Lines
    fig.add_vline(x=mean_val, line=dict(color='blue', dash='dash', width=2), 
                  annotation_text=f'Mean: {mean_val:.2f}', row=row, col=col)
    if global_mean is not None:
        fig.add_vline(x=global_mean, line=dict(color='orange', dash='dash', width=2), 
                      annotation_text=f'Global Mean: {global_mean:.2f}', row=row, col=col)
    fig.add_vline(x=x_min, line=dict(color='black', dash='dash', width=2), 
                  annotation_text=f'Min: {x_min:.2f}', annotation_position="bottom right", row=row, col=col)
    fig.add_vline(x=x_max, line=dict(color='black', dash='dash', width=2), 
                  annotation_text=f'Max: {x_max:.2f}', annotation_position="bottom left", row=row, col=col)

    # 4. Handle Layout and Titles
    title=f"Data Distribution<br>STD = {std:.3f}, Skewness = {skew:.2f}, Kurtosis = {kurt:.2f}"
    if title_addon:
        title = title_addon + '<br>' + title

    # Apply limits and labels to the specific subplot axes
    fig.update_xaxes(range=[xmin, xmax], title_text="Value (mV)", row=row, col=col)
    fig.update_yaxes(title_text=ylabel, row=row, col=col)
    
    # Only update the global title if we are generating a standalone figure
    if standalone:
        fig.update_layout(title=title, template="plotly_white", barmode="overlay")
        fig.update_layout(width=1000, height=700, margin=dict(l=50, r=50, t=100, b=50))
        fig.show()

    return fig

def create_analysis_plots(df, normalized_deviations, chamber_num, stats_dict, x_scale='all', figsize=(15, 12), title_add='',linearize_time=True):
    """
    Create analysis plots for a specific chamber with all plots in one figure.
    
    Parameters:
    df: pandas DataFrame with time series data
    chamber_num: Chamber number to analyze
    normalized_deviations: DataFrame with normalized deviations
    stats_dict: Dictionary containing statistical moments (mean, std, skew, kurtosis)
    x_scale: Number of standard deviations to show or 'all' for all data
    figsize: Tuple specifying figure size (width, height)
    
    Returns:
    None
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    # Create figure with custom grid
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 2, figure=fig)
    
    # Create axes for each subplot
    ax1 = fig.add_subplot(gs[0, :])  # Top left - chamber data
    ax2 = fig.add_subplot(gs[1, :])  # Top right - deviation
    ax3 = fig.add_subplot(gs[2, :])  # Bottom row, spanning both columns - histogram
    
    # Plot the chamber data in the first subplot
    plot_chamber(df, chamber_num, ax=ax1, linearize_time=linearize_time)
    
    # Plot the deviation from mean in the second subplot
    plot_chamber_deviation(normalized_deviations, chamber_num, ax=ax2, linearize_time=linearize_time)
    
    # Create histogram with normal and skew-normal distribution curves in the third subplot
    create_histogram(df[chamber_num], stats_dict['individual'][f'{chamber_num}'], nr_bins='auto', 
                     scale_type='density', x_scale=x_scale, global_mean=stats_dict['combined']['mean'], ax=ax3)
    
    fig.suptitle(f'Analysis for Chamber {chamber_num}' +title_add, fontsize=16)
    # Adjust layout
    plt.tight_layout()
    plt.show()
    


def make_equidistant_index(df, interpol_method='linear', frequency='auto'):
    """
    Make the index of a DataFrame equidistant by resampling.
    
    Parameters:
    df: pandas DataFrame with datetime index
    
    Returns:
    df: pandas DataFrame with equidistant datetime index
    """
  
    assert pd.api.types.is_datetime64_any_dtype(df.index), "DataFrame index must be datetime type."
    assert interpol_method in ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'], "interpol_method must be one of ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']"
    #assert that frequency is valid pandas timedelta
    assert isinstance(frequency, (pd.Timedelta, str)), "frequency must be a pandas Timedelta or a string of a valid pandas frequency (e.g. '1T', '5T', '1H', etc.)"
    
    #get the mean of timesteps between rows
    time_diff = df.index.to_series().diff().mean()
    #get fastest time difference that happens at least 10 times
    fastest_time_diff = df.index.to_series().diff().value_counts().sort_values(ascending=False)
    #print('20 most common time differences:\n', fastest_time_diff.head(20))
    min_fastest_time_diff = fastest_time_diff[fastest_time_diff > 10].index[0]
    #print(f'using time difference of {min_fastest_time_diff} as frequency')
    #fastest_time_diff = df.index.to_series().diff().min()
    #get nearest minute
    if frequency == 'auto':
        frequency = pd.Timedelta(minutes=round(min_fastest_time_diff.total_seconds() // 60))
        print(f'Frequency: {frequency}')
    elif isinstance(frequency, str):
        frequency = pd.Timedelta(frequency)
        print(f'Frequency: {frequency}')
    print(f'Frequency: {frequency}')
    # Resample the DataFrame to the nearest minute and interpolate missing values
    
    # Resample to 1-minute frequency and interpolate missing values
    df = df.resample(frequency).mean().interpolate(method=interpol_method, limit_direction='both')
    
    return df

def isolation_forrest_outliers(df, chamber_num=None, contamination=0.05):
    """
    Identify outliers using Isolation Forest.
    
    Parameters:
    df: pandas DataFrame with time series data
    chamber_num: Chamber number to analyze
    contamination: Proportion of outliers in the data (default is 0.05)
    
    Returns:
    outliers: Series of outlier scores for the specified chamber
    """
    from sklearn.ensemble import IsolationForest
    
    # Extract the data for the specified chamber
    if chamber_num is None:
        data = df
    else:
        data = df[chamber_num].values.reshape(-1, 1)
    
    # Create an Isolation Forest model
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    
    # Fit the model and get the outlier scores
    outlier_scores = iso_forest.fit_predict(data)
    
    # Convert scores to a Series with the same index as the original DataFrame
    outliers = pd.Series(outlier_scores, index=df.index, name=f'Outliers_{chamber_num}')
    
    return outliers

    
def create_cooling_cycle_dataset(freq, save_file=None, add_ramp_list= None):
    systems_dict = {}
    column_names=[]
    #load and concat all alunorf 1+2 and elval i-chamber data
    path = r'alunorf_1_all_new.csv'
    df = pd.read_csv(path, parse_dates=['DateTime'])
    #set datetime as index
    df.set_index('DateTime', inplace=True)
    df = df.drop(columns=['NSID','Ident'])
    #df =  filter_ouliers(df)
    #remove duplicates time stamps
    df = df[~df.index.duplicated(keep='first')]
    #resample to hourly data 
    df = df.resample(freq).mean()
    #extract all column names with temperature in it
    temp_cols = [col for col in df.columns if 'Temp' in col if 'Error' not in col]
    if 'Thermo_Valve_Temperature_DeviationPct' in temp_cols:
        temp_cols.remove('Thermo_Valve_Temperature_DeviationPct')
    #print(f'temperature columns: {temp_cols}')
    pressure_cols = [col for col in df.columns if 'Pressure' in col if 'Error' not in col]
    #print(f'pressure columns: {pressure_cols}')
    flow_cols = [col for col in df.columns if 'Flow' in col if 'Error' not in col]
    #print(f'flow columns: {flow_cols}')
    column_names.extend(temp_cols)
    column_names.extend(pressure_cols)
    column_names.extend(flow_cols)
    select_cols = temp_cols + pressure_cols + flow_cols
    #print(f'selected columns: {select_cols}')
    df_select = df[select_cols]
    #interpolate missing values
    df_select = df_select.interpolate(method='time')
    #fill any remaining missing values with forward fill
    df_select = df_select.ffill().bfill()
    long_df =  prepare_wide_df_for_long_format(df_select, client_id='alunorf',installation_id='A')
    df_list = [long_df]
    systems_dict["Alunorf_A"] = df_select

    path = r'alunorf_2_all_new.csv'
    df = pd.read_csv(path, parse_dates=['DateTime'])
    #set datetime as index
    df.set_index('DateTime', inplace=True)
    df = df.drop(columns=['NSID','Ident'])
    #df =  filter_ouliers(df)
    #remove duplicates time stamps
    df = df[~df.index.duplicated(keep='first')]
    #resample to hourly data 
    df = df.resample(freq).mean()
    #extract all column names with temperature in it
    temp_cols = [col for col in df.columns if 'Temp' in col if 'Error' not in col]
    if 'Thermo_Valve_Temperature_DeviationPct' in temp_cols:
        temp_cols.remove('Thermo_Valve_Temperature_DeviationPct')
    #print(f'temperature columns: {temp_cols}')
    pressure_cols = [col for col in df.columns if 'Pressure' in col if 'Error' not in col]
    #print(f'pressure columns: {pressure_cols}')
    flow_cols = [col for col in df.columns if 'Flow' in col if 'Error' not in col]
    #print(f'flow columns: {flow_cols}')
    column_names.extend(temp_cols)
    column_names.extend(pressure_cols)
    column_names.extend(flow_cols)
    select_cols = temp_cols + pressure_cols + flow_cols
    #print(f'selected columns: {select_cols}')
    df_select = df[select_cols]
    #interpolate missing values
    df_select = df_select.interpolate(method='time')
    #fill any remaining missing values with forward fill
    df_select = df_select.ffill().bfill()
    if add_ramp:
        for col in add_ramp_list:
            assert col in df_select.columns, f"{col} not found in DataFrame columns"
            start_idx =0.9*len(df_select[col])
            max_value = df_select[col].max()
            start_ts = df_select.index[int(start_idx)]
            df_select = add_ramp_offset_to_timeseries(df_select, target_col=col, start_ts=start_ts, offset_value=1.2*max_value, offset_duration=pd.Timedelta(days=10))
            ramp_dict= {'client_id': 'alunorf','installation_id': 'B',
                        col: (start_ts, 1.2*max_value, pd.Timedelta(days=10),)
                        }
        print('ramp dict:', ramp_dict)
        with open('temps_ramp_dict.pkl', 'wb') as f:
            pickle.dump(ramp_dict, f)
    #print(f'number of missing values: {df_select.isna().sum().sum()}')
    #print(df_select.head())
    #print('entries in df:', len(df_select))
    long_df =  prepare_wide_df_for_long_format(df_select, client_id='alunorf',installation_id='B')
    df_list.append(long_df)
    systems_dict["Alunorf_B"] = df_select

    path = r'elval_all_new.csv'
    df = pd.read_csv(path, parse_dates=['DateTime'])
    #set datetime as index
    df.set_index('DateTime', inplace=True)
    df = df.drop(columns=['NSID','Ident'])
    #df =  filter_ouliers(df)
    #remove duplicates time stamps
    df = df[~df.index.duplicated(keep='first')]

    dft_splits =  split_dataframe_on_gaps(df)
    print(f'number of split dataframes due to gaps: {len(dft_splits)}')
    for i, part in enumerate(dft_splits):
        print(f"DataFrame {i+1}:")
        print(f"  Start: {part.index.min()}")
        print(f"  End:   {part.index.max()}")
        print(f"  Rows:  {len(part)}")
        print("-" * 10)

    char_list=['A','B','C','D','E']
    for i, df in enumerate(dft_splits):
        #resample to hourly data 
        df = df.resample(freq).mean()
        #extract all column names with temperature in it
        temp_cols = [col for col in df.columns if 'Temp' in col if 'Error' not in col]
        if 'Thermo_Valve_Temperature_DeviationPct' in temp_cols:
            temp_cols.remove('Thermo_Valve_Temperature_DeviationPct')
        #print(f'temperature columns: {temp_cols}')
        pressure_cols = [col for col in df.columns if 'Pressure' in col if 'Error' not in col]
        #print(f'pressure columns: {pressure_cols}')
        flow_cols = [col for col in df.columns if 'Flow' in col if 'Error' not in col]
        #print(f'flow columns: {flow_cols}')
        column_names.extend(temp_cols)
        column_names.extend(pressure_cols)
        column_names.extend(flow_cols)
        select_cols = temp_cols + pressure_cols + flow_cols
        #print(f'selected columns: {select_cols}')
        df_select = df[select_cols]

        #interpolate missing values
        df_select = df_select.interpolate(method='time')
        #fill any remaining missing values with forward fill
        df_select = df_select.ffill().bfill()
        #print(f'number of missing values: {df_select.isna().sum().sum()}')
        #print(df_select.head())
        long_df =  prepare_wide_df_for_long_format(df_select, client_id='elval',installation_id=char_list[i])
        print(f'creating df with installation id: {char_list[i]}')
        df_list.append(long_df)
        systems_dict[f"Elval_{char_list[i]}"] = df_select

        
    #pickle df_list for faster testing
    if save_file is not None:
        with open(save_file, 'wb') as f:
            pickle.dump(df_list, f)
            
    return df_list
            

def create_i_chamber_dataset(freq, save_file=None):
    
    df_list = []
    path = r'alunorf_1_ichannels.csv'
    df = pd.read_csv(path, parse_dates=['DateTime'])
    #change column DateTime to datetime and set as index    
    df.rename(columns={'DateTime': 'datetime'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime']).dt.floor(freq)
    #set datetime as index
    df.set_index('datetime', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    #resample to hourly data
    df = df.resample(freq).mean()
    df = df.interpolate(method='time').ffill().bfill()
    clean_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq)
    df = df.reindex(clean_index).interpolate(method='time').ffill().bfill()

    # Explicitly set the freq attribute
    df.index.freq = freq
    long_df =  prepare_wide_df_for_long_format(df, client_id='alunorf',installation_id='A')
    df_list.append(long_df)

    path = r'alunorf_2_ichannels.csv'
    df = pd.read_csv(path, parse_dates=['DateTime'])
    #change column DateTime to datetime and set as index    
    df.rename(columns={'DateTime': 'datetime'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime']).dt.floor(freq)
    #set datetime as index
    df.set_index('datetime', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    #resample to hourly data
    df = df.resample(freq).mean()
    df = df.interpolate(method='time').ffill().bfill()
    clean_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq)
    df = df.reindex(clean_index).interpolate(method='time').ffill().bfill()
    # Explicitly set the freq attribute
    df.index.freq = freq
    long_df =  prepare_wide_df_for_long_format(df, client_id='alunorf',installation_id='B')
    df_list.append(long_df)

    path = r'elval_1_ichannels.csv'
    df = pd.read_csv(path, parse_dates=['DateTime'])
    #change column DateTime to datetime and set as index    
    df.rename(columns={'DateTime': 'datetime'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime']).dt.floor(freq)
    #set datetime as index
    df.set_index('datetime', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    dft_splits = split_dataframe_on_gaps(df,expected_freq='2h')
    print(f'number of split dataframes due to gaps: {len(dft_splits)}')
    for i, part in enumerate(dft_splits):
        print(f"DataFrame {i+1}:")
        print(f"  Start: {part.index.min()}")
        print(f"  End:   {part.index.max()}")
        print(f"  Rows:  {len(part)}")
        print("-" * 10)

    char_list=['A','B','C','D','E']
    for i, df in enumerate(dft_splits):
        
        #resample to hourly data
        df = df.resample(freq).mean()
        df = df.interpolate(method='time').ffill().bfill()
        clean_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq)
        df = df.reindex(clean_index).interpolate(method='time').ffill().bfill()
        # Explicitly set the freq attribute
        df.index.freq = freq
        long_df =  prepare_wide_df_for_long_format(df, client_id='elval',installation_id=char_list[i])
        df_list.append(long_df)
        
    if save_file is not None:
        with open(save_file, 'wb') as f:
            pickle.dump(df_list, f)
            
    return df_list

def create_Shutter_dataset(freq, save_file=None):
    
    column_names=[]
    #load and concat all alunorf 1+2 and elval i-chamber data
    path = r'alunorf_1_all_new.csv'
    df = pd.read_csv(path, parse_dates=['DateTime'])
    #set datetime as index
    df.set_index('DateTime', inplace=True)
    df = df.drop(columns=['NSID','Ident'])
    #df = du.filter_ouliers(df)
    #remove duplicates time stamps
    df = df[~df.index.duplicated(keep='first')]
    #resample to hourly data 
    df = df.resample(freq).mean()
    #extract all column names with temperature in it
    #shutter_cols = [col for col in df.columns if 'Shutter' in col or 'XRay' in col if 'Error' not in col]#Shutter_Close
    shutter_cols = [col for col in df.columns if 'Shutter_Close' in col  if 'Error' not in col]
    print(f'shutter columns: {shutter_cols}')
    column_names.extend(shutter_cols)
    select_cols = shutter_cols
    #print(f'selected columns: {select_cols}')
    df_select = df[select_cols]
    #interpolate missing values
    df_select = df_select.interpolate(method='time')
    #fill any remaining missing values with forward fill
    df_select = df_select.ffill().bfill()
    #print(f'number of missing values: {df_select.isna().sum().sum()}')
    #print(df_select.head())
    #print('entries in df:', len(df_select))
    long_df = prepare_wide_df_for_long_format(df_select, client_id='alunorf',installation_id='A')
    df_list = [long_df]

    path = r'alunorf_2_all_new.csv'
    df = pd.read_csv(path, parse_dates=['DateTime'])
    #set datetime as index
    df.set_index('DateTime', inplace=True)
    df = df.drop(columns=['NSID','Ident'])
    #df = du.filter_ouliers(df)
    #remove duplicates time stamps
    df = df[~df.index.duplicated(keep='first')]
    #resample to hourly data 
    df = df.resample(freq).mean()
    #extract all column names with temperature in it
    #shutter_cols = [col for col in df.columns if 'Shutter' in col or 'XRay' in col if 'Error' not in col]
    shutter_cols = [col for col in df.columns if 'Shutter_Close' in col  if 'Error' not in col]
    #print(f'flow columns: {flow_cols}')
    column_names.extend(shutter_cols)
    select_cols = shutter_cols
    #print(f'selected columns: {select_cols}')
    df_select = df[select_cols]
    #interpolate missing values
    df_select = df_select.interpolate(method='time')
    #fill any remaining missing values with forward fill
    df_select = df_select.ffill().bfill()
    #print(f'number of missing values: {df_select.isna().sum().sum()}')
    #print(df_select.head())
    #print('entries in df:', len(df_select))
    start_idx =0.9*len(df_select['Head_1_Shutter_Close_Time_0'])
    start_ts = df_select.index[int(start_idx)]
    df_select = add_ramp_offset_to_timeseries(df_select, target_col='Head_1_Shutter_Close_Time_0', start_ts=start_ts, offset_value=3.0, offset_duration=pd.Timedelta(days=10))
    start_idx =0.9*len(df_select['Head_2_Shutter_Close_Time_0'])
    start_ts = df_select.index[int(start_idx)]
    df_select = add_ramp_offset_to_timeseries(df_select, target_col='Head_2_Shutter_Close_Time_0', start_ts=start_ts, offset_value=2.0, offset_duration=pd.Timedelta(days=3))
    ramp_dict= {'client_id': 'alunorf','installation_id': 'B',
                'Head_1_Shutter_Close_Time_0': (start_ts, 3.0, pd.Timedelta(days=10),),
                'Head_2_Shutter_Close_Time_0': (start_ts, 2.0, pd.Timedelta(days=3))}
    print('ramp dict:', ramp_dict)
    with open('shutter_close_ramp_dict.pkl', 'wb') as f:
        pickle.dump(ramp_dict, f)
    #train ramps
    start_idx =0.3*len(df_select['Head_1_Shutter_Close_Time_0'])
    start_ts = df_select.index[int(start_idx)]
    df_select = add_ramp_offset_to_timeseries(df_select, target_col='Head_1_Shutter_Close_Time_0', start_ts=start_ts, offset_value=3.0, offset_duration=pd.Timedelta(days=10))
    start_idx =0.3*len(df_select['Head_2_Shutter_Close_Time_0'])
    start_ts = df_select.index[int(start_idx)]
    df_select = add_ramp_offset_to_timeseries(df_select, target_col='Head_2_Shutter_Close_Time_0', start_ts=start_ts, offset_value=2.0, offset_duration=pd.Timedelta(days=3))
    long_df = prepare_wide_df_for_long_format(df_select, client_id='alunorf',installation_id='B')
    df_list.append(long_df)

    path = r'elval_all_new.csv'
    df = pd.read_csv(path, parse_dates=['DateTime'])
    #set datetime as index
    df.set_index('DateTime', inplace=True)
    df = df.drop(columns=['NSID','Ident'])
    #df = du.filter_ouliers(df)
    #remove duplicates time stamps
    df = df[~df.index.duplicated(keep='first')]

    dft_splits = split_dataframe_on_gaps(df)
    print(f'number of split dataframes due to gaps: {len(dft_splits)}')
    for i, part in enumerate(dft_splits):
        print(f"DataFrame {i+1}:")
        print(f"  Start: {part.index.min()}")
        print(f"  End:   {part.index.max()}")
        print(f"  Rows:  {len(part)}")
        print("-" * 10)

    char_list=['A','B','C','D','E']
    for i, df in enumerate(dft_splits):
        #resample to hourly data 
        df = df.resample(freq).mean()
        #extract all column names with temperature in it
        #shutter_cols = [col for col in df.columns if 'Shutter' in col or 'XRay' in col if 'Error' not in col]
        shutter_cols = [col for col in df.columns if 'Shutter_Close' in col  if 'Error' not in col]
        #print(f'flow columns: {flow_cols}')
        column_names.extend(shutter_cols)
        select_cols = shutter_cols
        #print(f'selected columns: {select_cols}')
        df_select = df[select_cols]

        #interpolate missing values
        df_select = df_select.interpolate(method='time')
        #fill any remaining missing values with forward fill
        df_select = df_select.ffill().bfill()
        #print(f'number of missing values: {df_select.isna().sum().sum()}')
        #print(df_select.head())
        long_df = prepare_wide_df_for_long_format(df_select, client_id='elval',installation_id=char_list[i])
        print(f'creating df with installation id: {char_list[i]}')
        df_list.append(long_df)
        
    if save_file is not None:
        with open(save_file, 'wb') as f:
            pickle.dump(df_list, f)
            
    return df_list
    
class LoadBestModelCallback(pl.Callback):
    def __init__(self):
        super().__init__()
    def on_train_end(self, trainer, pl_module):
        # Find the ModelCheckpoint callback
        checkpoint_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                checkpoint_callback = callback
                break
        
        if checkpoint_callback and checkpoint_callback.best_model_path:
            print(f"Loading best model from: {checkpoint_callback.best_model_path}")
            # Load the best checkpoint
            best_checkpoint = torch.load(checkpoint_callback.best_model_path)
            pl_module.load_state_dict(best_checkpoint['state_dict'])
        else:
            print("No best model found. Skipping loading.")
    def get_save_dir(self):
        return self.best_model_path
            
            

def build_nhits_probabilistic_model(
    # --- N-HiTS Hyperparameters ---
    input_chunk_length: int = 24,
    output_chunk_length: int = 12, # Should be <= n_forecast for direct prediction
    n_epochs: int = 50,
    save_dir: str = 'models',
    num_stacks: int = 3,
    num_blocks: int = 1,
    num_layers: int = 2,
    layer_widths: int = 512,
    # --- Probabilistic Configuration ---
    # Choose one: likelihood OR quantiles
    prediction_type='likelihood', # 'likelihood' or 'quantiles'
    distribution='normal', # Only used if prediction_type is 'likelihood'
    quantiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99], # Only used if prediction_type is 'quantiles'
    batch_size: int = 32,
    lr = 1e-3,
    weight_decay: float = 1e-5,
    optimizer = 'adam',
    pl_trainer_kwargs: dict = {"accelerator": "auto", 
                               "enable_progress_bar": True,
                               'enable_checkpointing': True,
                               'logger':False},
    random_state: int = 42,
    monitor_metric: str = "val_loss",
    callbacks: list = None,
    save_checkpoints: bool = True,
    verbose: bool = True,
    config: dict = None  # New optional parameter for config dict
) -> tuple:
    """
    Builds and initializes a probabilistic NHiTSModel.

    Args:
        (See extracted steps for details; adapted for NHiTSModel.)
        config: Optional dictionary to override hyperparameters.

    Returns:
        tuple: Contains:
            - model (NHiTSModel): The initialized model.
            - save_dir (str): The directory where the model is saved.
    """
    if config is not None:
        input_chunk_length = config.get('input_chunk_length', input_chunk_length)
        output_chunk_length = config.get('output_chunk_length', output_chunk_length)
        n_epochs = config.get('n_epochs', n_epochs)
        save_dir = config.get('save_dir', save_dir)
        num_stacks = config.get('num_stacks', num_stacks)
        num_blocks = config.get('num_blocks', num_blocks)
        num_layers = config.get('num_layers', num_layers)
        layer_widths = config.get('layer_widths', layer_widths)
        prediction_type = config.get('prediction_type', prediction_type)
        distribution = config.get('distribution', distribution)
        quantiles = config.get('quantiles', quantiles)
        batch_size = config.get('batch_size', batch_size)
        lr = config.get('lr', lr)
        weight_decay = config.get('weight_decay', weight_decay)
        optimizer = config.get('optimizer', optimizer)
        pl_trainer_kwargs = config.get('pl_trainer_kwargs', pl_trainer_kwargs)
        random_state = config.get('random_state', random_state)
        monitor_metric = config.get('monitor_metric', monitor_metric)
        callbacks = config.get('callbacks', callbacks)
        save_checkpoints = config.get('save_checkpoints', save_checkpoints)
        verbose = config.get('verbose', verbose)

    assert prediction_type in ['likelihood', 'quantiles',None], "prediction_type must be 'likelihood' or 'quantiles' or None (for no probabilistic prediction)"
    assert distribution in ['normal', 'beta','gamma'], "distribution must be 'normal', 'beta' or 'gamma'"
    
    # Extract LR scheduler params (defaults or from config)
    lr_scheduler_patience = config.get('patience', 5) if config is not None else 5
    lr_scheduler_factor = config.get('factor', 0.2) if config is not None else 0.2
    
    save_dir = save_dir + '/NHiTS'    
    os.makedirs(save_dir, exist_ok=True)
    monitor_metric = "val_loss" 
    
    pl_trainer_kwargs['callbacks'] = callbacks if callbacks is not None else []
    pl_trainer_kwargs['callbacks'].append(LoadBestModelCallback())
    early_stopping = EarlyStopping(monitor='val_loss', patience=6, min_delta=0.0, mode="min", verbose=True)
    pl_trainer_kwargs['callbacks'].append(early_stopping)

    #define the likelihood or quantiles based on the prediction_type
    if prediction_type == 'likelihood':
        if distribution == 'normal':
            likelihood = GaussianLikelihood()
        elif distribution == 'beta':
            likelihood = BetaLikelihood()
        elif distribution == 'gamma':
            likelihood = GammaLikelihood()
        else:
            raise ValueError("distribution must be 'normal', 'beta' or 'gamma'")
    elif prediction_type == 'quantiles':
        likelihood = QuantileRegression(quantiles) # 1st, 10th, 25th, 50th, 75th, 90th, 99th percentiles
        print(f"Quantiles: {quantiles}")
    elif prediction_type is None:
        likelihood = None
    else:
        raise ValueError("prediction_type must be 'likelihood' or 'quantiles'")

    optimizer_name = optimizer
    assert optimizer_name in ['adam', 'adamw'], "optimizer must be 'adam' or 'adamw'"
    if optimizer_name == 'adam':
        optimizer_cls = torch.optim.Adam  
    elif optimizer_name == 'adamw':
        optimizer_cls = torch.optim.AdamW  
    optimizer_kwargs = {'lr': lr,
                        'weight_decay': weight_decay}
    # --- Model Initialization ---
    model = NHiTSModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        num_stacks=num_stacks,
        num_blocks=num_blocks,
        num_layers=num_layers,
        layer_widths=layer_widths,
        likelihood=likelihood, # Pass the likelihood object
        n_epochs=n_epochs,    
        batch_size=batch_size,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        pl_trainer_kwargs=pl_trainer_kwargs,
        lr_scheduler_cls =ReduceLROnPlateau,
        lr_scheduler_kwargs={'monitor': monitor_metric,
                             'factor': lr_scheduler_factor,
                             'patience': lr_scheduler_patience,
                             'min_lr':1e-9},
        random_state=random_state,
        work_dir=save_dir,
        #model_name=model_name,  # Name of the model for saving checkpoints
        save_checkpoints=save_checkpoints,
        force_reset=True,
    )
    save_dir = os.path.join(model.work_dir, model.model_name)
    config = {
        'model_type': 'NHiTS',
        'input_chunk_length': input_chunk_length,
        'output_chunk_length': output_chunk_length,
        'num_stacks': num_stacks,
        'num_blocks': num_blocks,
        'num_layers': num_layers,
        'layer_widths': layer_widths,
        'prediction_type': prediction_type,
        'distribution': distribution,
        'quantiles': quantiles,
        'batch_size': batch_size,
        'lr': lr,
        'optimizer': optimizer_name,
        'weight_decay': weight_decay,
        'lr_scheduler': 'ReduceLROnPlateau',
        'monitor_metric': monitor_metric,
        'patience': lr_scheduler_patience,
        'factor': lr_scheduler_factor,
    }
    save_model_config(config, save_dir)
    return model, save_dir

    
def build_tft_probabilistic_model(
    # --- TFT Hyperparameters ---
    input_chunk_length: int = 24,
    output_chunk_length: int = 12,
    n_epochs: int = 50,
    save_dir: str = 'models',
    hidden_size: int = 64,
    lstm_layers: int = 2,
    num_attention_heads: int = 4,
    full_attention: bool = False,
    dropout: float = 0.1,
    # --- Probabilistic Configuration ---
    prediction_type='likelihood',  # 'likelihood' or 'quantiles'
    distribution='normal',  # Only used if prediction_type is 'likelihood'
    quantiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99],  # Only used if prediction_type is 'quantiles'
    batch_size: int = 32,
    lr = 1e-3,
    weight_decay: float = 1e-5,
    optimizer = 'adam',
    pl_trainer_kwargs: dict = {"accelerator": "gpu", 
                               "enable_progress_bar": True,
                               'enable_checkpointing': True},
    random_state: int = 42,
    monitor_metric: str = "val_loss",
    callbacks: list = None,
    save_checkpoints: bool = True,
    verbose: bool = True,
    model_name: str = None,  # Added for consistency with train function
    proj_name: str = 'i-chamber',  # Added for consistency
    config: dict = None  # New optional parameter for config dict
) -> TFTModel:
    """
    Builds and initializes a probabilistic TFTModel.

    Args:
        (See extracted steps for details; adapted for TFTModel.)
        config: Optional dictionary to override hyperparameters.

    Returns:
        TFTModel: The initialized model.
    """
    if config is not None:
        input_chunk_length = config.get('input_chunk_length', input_chunk_length)
        output_chunk_length = config.get('output_chunk_length', output_chunk_length)
        n_epochs = config.get('n_epochs', n_epochs)
        save_dir = config.get('save_dir', save_dir)
        hidden_size = config.get('hidden_size', hidden_size)
        lstm_layers = config.get('lstm_layers', lstm_layers)
        num_attention_heads = config.get('num_attention_heads', num_attention_heads)
        full_attention = config.get('full_attention', full_attention)
        dropout = config.get('dropout', dropout)
        prediction_type = config.get('prediction_type', prediction_type)
        distribution = config.get('distribution', distribution)
        quantiles = config.get('quantiles', quantiles)
        batch_size = config.get('batch_size', batch_size)
        lr = config.get('lr', lr)
        weight_decay = config.get('weight_decay', weight_decay)
        optimizer = config.get('optimizer', optimizer)
        pl_trainer_kwargs = config.get('pl_trainer_kwargs', pl_trainer_kwargs)
        random_state = config.get('random_state', random_state)
        monitor_metric = config.get('monitor_metric', monitor_metric)
        callbacks = config.get('callbacks', callbacks)
        save_checkpoints = config.get('save_checkpoints', save_checkpoints)
        verbose = config.get('verbose', verbose)
        model_name = config.get('model_name', model_name)
        proj_name = config.get('proj_name', proj_name)

    assert prediction_type in ['likelihood', 'quantiles', None], "prediction_type must be 'likelihood' or 'quantiles' or None"
    assert distribution in ['normal', 'beta', 'gamma'], "distribution must be 'normal', 'beta' or 'gamma'"
    
    # Extract LR scheduler params (defaults or from config)
    lr_scheduler_patience = config.get('patience', 3) if config is not None else 3
    lr_scheduler_factor = config.get('factor', 0.2) if config is not None else 0.2
    
    save_dir = save_dir + '/TFT'
    os.makedirs(save_dir, exist_ok=True)
    monitor_metric = "val_loss"
    
    pl_trainer_kwargs['callbacks'] = callbacks if callbacks is not None else []
    pl_trainer_kwargs['callbacks'].append(LoadBestModelCallback())
    early_stopping = EarlyStopping(monitor='val_loss', patience=6, min_delta=0.0, mode="min", verbose=True)
    pl_trainer_kwargs['callbacks'].append(early_stopping)
    
    # Define the likelihood or quantiles based on the prediction_type
    if prediction_type == 'likelihood':
        if distribution == 'normal':
            likelihood = GaussianLikelihood()
        elif distribution == 'beta':
            likelihood = BetaLikelihood()
        elif distribution == 'gamma':
            likelihood = GammaLikelihood()
        else:
            raise ValueError("distribution must be 'normal', 'beta' or 'gamma'")
    elif prediction_type == 'quantiles':
        likelihood = QuantileRegression(quantiles)
        print(f"Quantiles: {quantiles}")
    elif prediction_type is None:
        likelihood = None
    else:
        raise ValueError("prediction_type must be 'likelihood' or 'quantiles'")
    
    optimizer_name = optimizer
    assert optimizer_name in ['adam', 'adamw'], "optimizer must be 'adam' or 'adamw'"
    if optimizer_name == 'adam':
        optimizer_cls = torch.optim.Adam  
    elif optimizer_name == 'adamw':
        optimizer_cls = torch.optim.AdamW
    optimizer_kwargs = {'lr': lr,
                        'weight_decay': weight_decay}
    # --- Model Initialization ---
    model = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        num_attention_heads=num_attention_heads,
        full_attention=full_attention,
        dropout=dropout,
        likelihood=likelihood,
        n_epochs=n_epochs,    
        batch_size=batch_size,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        pl_trainer_kwargs=pl_trainer_kwargs,
        lr_scheduler_cls=ReduceLROnPlateau,
        lr_scheduler_kwargs={'monitor': monitor_metric,
                             'factor': lr_scheduler_factor,
                             'patience': lr_scheduler_patience,},
        random_state=random_state,
        work_dir=save_dir,
        model_name=model_name,  # For saving checkpoints
        save_checkpoints=save_checkpoints,
        force_reset=True,
        add_relative_index=True,
    )
    
    save_dir = os.path.join(model.work_dir, model.model_name)
    config = {
        'model_type': 'TFT',
        'input_chunk_length': input_chunk_length,
        'output_chunk_length': output_chunk_length,
        'hidden_size': hidden_size,
        'lstm_layers': lstm_layers,
        'num_attention_heads': num_attention_heads,
        'full_attention': full_attention,
        'dropout': dropout,
        'prediction_type': prediction_type,
        'distribution': distribution,
        'quantiles': quantiles,
        'batch_size': batch_size,
        'lr': lr,
        'optimizer': optimizer_name,
        'weight_decay': weight_decay,
        'lr_scheduler': 'ReduceLROnPlateau',
        'monitor_metric': monitor_metric,
        'patience': lr_scheduler_patience,
        'factor': lr_scheduler_factor,
    }
    save_model_config(config, save_dir)
    return model, save_dir


def build_tide_probabilistic_model(
    # --- TiDE Hyperparameters ---
    input_chunk_length: int = 24,
    output_chunk_length: int = 12,
    n_epochs: int = 50,
    save_dir: str = 'models',
    num_encoder_layers: int = 2,
    num_decoder_layers: int = 2,
    decoder_output_dim: int = 32,
    hidden_size: int = 128,
    temporal_decoder_hidden: int = 64,
    dropout: float = 0.1,
    use_reversible_instance_norm: bool = True,
    # --- Probabilistic Configuration ---
    prediction_type='likelihood',  # 'likelihood' or 'quantiles'
    distribution='normal',  # Only used if prediction_type is 'likelihood'
    quantiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99],  # Only used if prediction_type is 'quantiles'
    batch_size: int = 32,
    lr = 1e-3,
    weight_decay: float = 1e-5,
    optimizer = 'adam',
    pl_trainer_kwargs: dict = {"accelerator": "gpu",
                               "enable_progress_bar": True,
                               'enable_checkpointing': True},
    random_state: int = 42,
    monitor_metric: str = "val_loss",
    callbacks: list = None,
    save_checkpoints: bool = True,
    verbose: bool = True,
    model_name: str = None,  # Added for consistency with train function
    proj_name: str = 'i-chamber',  # Added for consistency
    config: dict = None  # New optional parameter for config dict
) -> TiDEModel:
    """
    Builds and initializes a probabilistic TiDEModel.
    Args:
        (See extracted steps for details; adapted for TiDEModel.)
        config: Optional dictionary to override hyperparameters.
    Returns:
        TiDEModel: The initialized model.
    """
    if config is not None:
        input_chunk_length = config.get('input_chunk_length', input_chunk_length)
        output_chunk_length = config.get('output_chunk_length', output_chunk_length)
        n_epochs = config.get('n_epochs', n_epochs)
        save_dir = config.get('save_dir', save_dir)
        num_encoder_layers = config.get('num_encoder_layers', num_encoder_layers)
        num_decoder_layers = config.get('num_decoder_layers', num_decoder_layers)
        decoder_output_dim = config.get('decoder_output_dim', decoder_output_dim)
        hidden_size = config.get('hidden_size', hidden_size)
        temporal_decoder_hidden = config.get('temporal_decoder_hidden', temporal_decoder_hidden)
        dropout = config.get('dropout', dropout)
        use_reversible_instance_norm = config.get('use_reversible_instance_norm', use_reversible_instance_norm)
        prediction_type = config.get('prediction_type', prediction_type)
        distribution = config.get('distribution', distribution)
        quantiles = config.get('quantiles', quantiles)
        batch_size = config.get('batch_size', batch_size)
        lr = config.get('lr', lr)
        weight_decay = config.get('weight_decay', weight_decay)
        optimizer = config.get('optimizer', optimizer)
        pl_trainer_kwargs = config.get('pl_trainer_kwargs', pl_trainer_kwargs)
        random_state = config.get('random_state', random_state)
        monitor_metric = config.get('monitor_metric', monitor_metric)
        callbacks = config.get('callbacks', callbacks)
        save_checkpoints = config.get('save_checkpoints', save_checkpoints)
        verbose = config.get('verbose', verbose)
        model_name = config.get('model_name', model_name)
        proj_name = config.get('proj_name', proj_name)

    assert prediction_type in ['likelihood', 'quantiles', None], "prediction_type must be 'likelihood' or 'quantiles' or None"
    assert distribution in ['normal', 'beta', 'gamma'], "distribution must be 'normal', 'beta' or 'gamma'"
    
    # Extract LR scheduler params (defaults or from config)
    lr_scheduler_patience = config.get('patience', 4) if config is not None else 4
    lr_scheduler_factor = config.get('factor', 0.2) if config is not None else 0.2
    print(f'lr_scheduler_patience: {lr_scheduler_patience}, lr_scheduler_factor: {lr_scheduler_factor}')
    
    save_dir = save_dir + '/TiDE'
    os.makedirs(save_dir, exist_ok=True)
    monitor_metric = "val_loss"
    
    pl_trainer_kwargs['callbacks'] = callbacks if callbacks is not None else []
    pl_trainer_kwargs['callbacks'].append(LoadBestModelCallback())
    early_stopping = EarlyStopping(monitor='val_loss', patience=6, min_delta=0.0, mode="min", verbose=True)
    pl_trainer_kwargs['callbacks'].append(early_stopping)
    
    # Define the likelihood or quantiles based on the prediction_type
    if prediction_type == 'likelihood':
        if distribution == 'normal':
            likelihood = GaussianLikelihood()
        elif distribution == 'beta':
            likelihood = BetaLikelihood()
        elif distribution == 'gamma':
            likelihood = GammaLikelihood()
        else:
            raise ValueError("distribution must be 'normal', 'beta' or 'gamma'")
    elif prediction_type == 'quantiles':
        likelihood = QuantileRegression(quantiles)
        print(f"Quantiles: {quantiles}")
    elif prediction_type is None:
        likelihood = None
    else:
        raise ValueError("prediction_type must be 'likelihood' or 'quantiles'")
    
    optimizer_name = optimizer
    assert optimizer_name in ['adam', 'adamw'], "optimizer must be 'adam' or 'adamw'"
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam  
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW
    optimizer_kwargs = {'lr': lr, 'weight_decay': weight_decay}
    
    # --- Model Initialization ---
    model = TiDEModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        decoder_output_dim=decoder_output_dim,
        hidden_size=hidden_size,
        temporal_decoder_hidden=temporal_decoder_hidden,
        dropout=dropout,
        use_reversible_instance_norm=use_reversible_instance_norm,
        likelihood=likelihood,
        n_epochs=n_epochs,    
        batch_size=batch_size,
        optimizer_cls=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        pl_trainer_kwargs=pl_trainer_kwargs,
        lr_scheduler_cls=ReduceLROnPlateau,
        lr_scheduler_kwargs={'monitor': monitor_metric,
                             'factor': lr_scheduler_factor,
                             'patience': lr_scheduler_patience,},
        random_state=random_state,
        work_dir=save_dir,
        model_name=model_name,  # For saving checkpoints
        save_checkpoints=save_checkpoints,
        force_reset=True,
    )
    
    save_dir = os.path.join(model.work_dir, model.model_name)
    config={
        'model_type': 'TiDE',
        'input_chunk_length': input_chunk_length,
        'output_chunk_length': output_chunk_length,
        'num_encoder_layers': num_encoder_layers,
        'num_decoder_layers': num_decoder_layers,
        'decoder_output_dim': decoder_output_dim,
        'hidden_size': hidden_size,
        'temporal_decoder_hidden': temporal_decoder_hidden,
        'dropout': dropout,
        'use_reversible_instance_norm': use_reversible_instance_norm,
        'prediction_type': prediction_type,
        'distribution': distribution,
        'quantiles': quantiles,
        'batch_size': batch_size,
        'lr': lr,
        'optimizer': optimizer_name,
        'weight_decay': weight_decay,
        'lr_scheduler': 'ReduceLROnPlateau',
        'monitor_metric': monitor_metric,
        'patience': lr_scheduler_patience,
        'factor': lr_scheduler_factor,
    }
    save_model_config(config, save_dir)
    return model, save_dir


def build_deepar_probabilistic_model(
    # --- DeepAR (RNN) Hyperparameters ---
    input_chunk_length: int = 24,
    output_chunk_length: int = 12, # Used to set training_length
    n_epochs: int = 50,
    save_dir: str = 'models',
    rnn_type: str = 'LSTM', # 'LSTM' or 'GRU'
    hidden_dim: int = 64,
    n_rnn_layers: int = 2,
    dropout: float = 0.1,
    # --- Probabilistic Configuration ---
    prediction_type='likelihood', # 'likelihood' or 'quantiles'
    distribution='normal', # Only used if prediction_type is 'likelihood'
    quantiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99], # Only used if prediction_type is 'quantiles'
    batch_size: int = 32,
    lr = 1e-3,
    weight_decay: float = 1e-5,
    optimizer = 'adam',
    pl_trainer_kwargs: dict = {"accelerator": "gpu", 
                               "enable_progress_bar": True,
                               'enable_checkpointing': True,
                               'logger': False},
    random_state: int = 42,
    monitor_metric: str = "val_loss",
    callbacks: list = None,
    save_checkpoints: bool = True,
    verbose: bool = True,
    model_name: str = None, 
    config: dict = None  # New optional parameter for config dict
) -> tuple:
    """
    Builds and initializes a probabilistic DeepAR (RNNModel).

    Args:
        (See extracted steps for details; adapted for RNNModel.)
        config: Optional dictionary to override hyperparameters.

    Returns:
        tuple: Contains:
            - model (RNNModel): The initialized model.
            - save_dir (str): The directory where the model is saved.
    """
    if config is not None:
        input_chunk_length = config.get('input_chunk_length', input_chunk_length)
        output_chunk_length = config.get('output_chunk_length', output_chunk_length)
        n_epochs = config.get('n_epochs', n_epochs)
        save_dir = config.get('save_dir', save_dir)
        rnn_type = config.get('rnn_type', rnn_type)
        hidden_dim = config.get('hidden_dim', hidden_dim)
        n_rnn_layers = config.get('n_rnn_layers', n_rnn_layers)
        dropout = config.get('dropout', dropout)
        prediction_type = config.get('prediction_type', prediction_type)
        distribution = config.get('distribution', distribution)
        quantiles = config.get('quantiles', quantiles)
        batch_size = config.get('batch_size', batch_size)
        lr = config.get('lr', lr)
        weight_decay = config.get('weight_decay', weight_decay)
        optimizer = config.get('optimizer', optimizer)
        pl_trainer_kwargs = config.get('pl_trainer_kwargs', pl_trainer_kwargs)
        random_state = config.get('random_state', random_state)
        monitor_metric = config.get('monitor_metric', monitor_metric)
        callbacks = config.get('callbacks', callbacks)
        save_checkpoints = config.get('save_checkpoints', save_checkpoints)
        verbose = config.get('verbose', verbose)
        model_name = config.get('model_name', model_name)

    assert prediction_type in ['likelihood', 'quantiles', None], "prediction_type must be 'likelihood' or 'quantiles' or None"
    assert distribution in ['normal', 'beta', 'gamma'], "distribution must be 'normal', 'beta' or 'gamma'"
    
    # Extract LR scheduler params (defaults or from config)
    lr_scheduler_patience = config.get('patience', 5) if config is not None else 5
    lr_scheduler_factor = config.get('factor', 0.2) if config is not None else 0.2
    
    save_dir = save_dir + '/DeepAR'    
    os.makedirs(save_dir, exist_ok=True)
    monitor_metric = "val_loss" 
    
    pl_trainer_kwargs['callbacks'] = callbacks if callbacks is not None else []
    pl_trainer_kwargs['callbacks'].append(LoadBestModelCallback())
    early_stopping = EarlyStopping(monitor='val_loss', patience=6, min_delta=0.0, mode="min", verbose=True)
    pl_trainer_kwargs['callbacks'].append(early_stopping)

    # Define the likelihood or quantiles based on the prediction_type
    if prediction_type == 'likelihood':
        if distribution == 'normal':
            likelihood = GaussianLikelihood()
        elif distribution == 'beta':
            likelihood = BetaLikelihood()
        elif distribution == 'gamma':
            likelihood = GammaLikelihood()
        else:
            raise ValueError("distribution must be 'normal', 'beta' or 'gamma'")
    elif prediction_type == 'quantiles':
        likelihood = QuantileRegression(quantiles) 
        print(f"Quantiles: {quantiles}")
    elif prediction_type is None:
        likelihood = None
    else:
        raise ValueError("prediction_type must be 'likelihood' or 'quantiles'")

    optimizer_name = optimizer
    assert optimizer_name in ['adam', 'adamw'], "optimizer must be 'adam' or 'adamw'"
    if optimizer_name == 'adam':
        optimizer_cls = torch.optim.Adam  
    elif optimizer_name == 'adamw':
        optimizer_cls = torch.optim.AdamW  
    optimizer_kwargs = {'lr': lr, 'weight_decay': weight_decay}

    # --- Model Initialization ---
    # Note: DeepAR in Darts is an RNNModel. 
    # It requires training_length >= input_chunk_length. 
    # We set training_length to encompass input + output horizons to allow sufficient history.
    
    model = RNNModel(
        model=rnn_type,
        input_chunk_length=input_chunk_length,
        training_length=input_chunk_length + output_chunk_length, # Crucial for RNNs to learn the full window
        hidden_dim=hidden_dim,
        n_rnn_layers=n_rnn_layers,
        dropout=dropout,
        likelihood=likelihood,
        n_epochs=n_epochs,    
        batch_size=batch_size,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        pl_trainer_kwargs=pl_trainer_kwargs,
        lr_scheduler_cls=ReduceLROnPlateau,
        lr_scheduler_kwargs={'monitor': monitor_metric,
                             'factor': lr_scheduler_factor,
                             'patience': lr_scheduler_patience,
                             'min_lr': 1e-9},
        random_state=random_state,
        work_dir=save_dir,
        model_name=model_name,
        save_checkpoints=save_checkpoints,
        force_reset=True,
    )
    
    save_dir = os.path.join(model.work_dir, model.model_name)
    config = {
        'model_type': 'DeepAR', # Logged as DeepAR (implementation is RNNModel)
        'input_chunk_length': input_chunk_length,
        'output_chunk_length': output_chunk_length,
        'rnn_type': rnn_type,
        'hidden_dim': hidden_dim,
        'n_rnn_layers': n_rnn_layers,
        'dropout': dropout,
        'prediction_type': prediction_type,
        'distribution': distribution,
        'quantiles': quantiles,
        'batch_size': batch_size,
        'lr': lr,
        'optimizer': optimizer_name,
        'weight_decay': weight_decay,
        'lr_scheduler': 'ReduceLROnPlateau',
        'monitor_metric': monitor_metric,
        'patience': lr_scheduler_patience,
        'factor': lr_scheduler_factor,
    }
    save_model_config(config, save_dir)
    return model, save_dir


def save_model_config(config, save_dir):   
    assert isinstance(config, dict), "config must be a dictionary"
    assert isinstance(save_dir, str), "save_dir must be a string"
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
        


def train_probabilistic_model(
    series: Union[TimeSeries, List[TimeSeries]],
    val_series: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
    model: Optional[Union [NHiTSModel,TFTModel,TiDEModel]] = None,
    n_forecast: int = 12, # Forecast horizon in number of time steps
    verbose: bool = True,
    save_dir: str = 'models',
    input_chunk_length: int = 24,
    output_chunk_length: int = 12, # Should be <= n_forecast for direct prediction
    n_epochs: int = 50,
    prediction_type='likelihood', # 'likelihood' or 'quantiles'
    distribution='normal', # Only used if prediction_type is 'likelihood'
    batch_size: int = 32,
    lr =1e-3,
    build_model: str = None, # If None, use the provided model, else build a new one from [njits, tft, or tide]
    future_covariates: Optional[List[TimeSeries]] = None,  # Optional list of future covariates
    past_covariates: Optional[List[TimeSeries]] = None,  # Optional list of past covariates
    return_history_df: bool = False,  # Whether to return the training history DataFrame
    scale_globally: bool = True,  # Whether to scale the series globally across all series
    use_scaling: bool = False,  # Whether to apply scaling
    ) -> tuple:
    """
    Trains an N-HiTS model for probabilistic forecasting and calculates
    the probability of crossing a threshold.

    Args:
        series (TimeSeries): The target time series to forecast.
        val_series (TimeSeries | None): Optional validation series for evaluation.
        n_forecast (int): The forecast horizon.
        input_chunk_length (int): The length of the input sequence fed to the model.
        output_chunk_length (int): The length of the output sequence predicted by the model.
                                   Must be <= n_forecast.
        n_epochs (int): Number of epochs for training.
        num_stacks, num_blocks, num_layers, layer_widths: N-HiTS architecture params.
        prediction_type (str): Type of probabilistic prediction ('likelihood' or 'quantiles').
        batch_size (int): Training batch size.
        optimizer_kwargs (dict): Arguments for the PyTorch optimizer.
        pl_trainer_kwargs (dict): Arguments for the PyTorch Lightning Trainer.
        random_state (int): Random seed for reproducibility.
        verbose (bool): Whether to print training progress.
        **kwargs: Additional arguments passed to NHiTSModel constructor.

    Returns:
        tuple: Contains:
            - model (NHiTSModel): The trained Darts model.
            - forecast (TimeSeries): The probabilistic forecast.
                - If likelihood was used: Contains samples drawn from the predicted distributions.
                - If quantiles were used: Contains the predicted quantiles as components.
            - crossing_probabilities (TimeSeries | None): A TimeSeries where each value
                represents the probability P(series > threshold) at that forecast step.
                Returns None if `quantiles` were used instead of `likelihood`, as direct
                probability calculation is less straightforward.
    """
     # Convert to lists for global training
    if isinstance(series, TimeSeries):
        series_list = [series]
    else:
        series_list = series
        
    if val_series is not None:
        if isinstance(val_series, TimeSeries):
            val_list = [val_series]
        else:
            val_list = val_series
        if len(val_list) != len(series_list):
            raise ValueError("Validation list must match the length of the series list.")
    else:
        val_list = None
        
    if future_covariates is not None and len(future_covariates) != len(series_list):
        raise ValueError("Future covariates list must match the length of the series list plus validation list.")
        
    
    #build model with standard parameters
    if build_model:
        assert build_model in ['nhits', 'tft', 'tide'], "build_model must be one of ['nhits', 'tft', 'tide']"
        assert isinstance(input_chunk_length, int) and input_chunk_length > 0, "input_chunk_length must be a positive integer"
        assert isinstance(output_chunk_length, int) and output_chunk_length > 0, "output_chunk_length must be a positive integer"
        assert isinstance(n_epochs, int) and n_epochs > 0, "n_epochs must be a positive integer"
        assert isinstance(batch_size, int) and batch_size > 0, "batch_size must be a positive integer"

        loss_history = LossHistoryCallback() 
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.0, mode="min", verbose=True)
        
        if build_model == 'nhits':
            model,save_dir= build_nhits_probabilistic_model(input_chunk_length=input_chunk_length,
                                                    output_chunk_length=output_chunk_length,
                                                    n_epochs=n_epochs,
                                                    batch_size=batch_size,
                                                    prediction_type=prediction_type,
                                                    distribution=distribution,
                                                    optimizer_kwargs={'lr': lr},
                                                    callbacks=[loss_history, early_stopping],
                                                    save_dir=save_dir,
                                                )
        elif build_model == 'tft':
            model,save_dir = build_tft_probabilistic_model(input_chunk_length=input_chunk_length,
                                                    output_chunk_length=output_chunk_length,
                                                    n_epochs=n_epochs,
                                                    batch_size=batch_size,
                                                    prediction_type=prediction_type,
                                                    distribution=distribution,
                                                    optimizer_kwargs={'lr': lr},
                                                    callbacks=[loss_history, early_stopping],
                                                    save_dir=save_dir,
                                                )
        elif build_model == 'tide':
            model,save_dir = build_tide_probabilistic_model(input_chunk_length=input_chunk_length,
                                                    output_chunk_length=output_chunk_length,
                                                    n_epochs=n_epochs,
                                                    batch_size=batch_size,
                                                    prediction_type=prediction_type,
                                                    distribution=distribution,
                                                    optimizer_kwargs={'lr': lr},
                                                    callbacks=[loss_history, early_stopping],
                                                    save_dir=save_dir,
                                                )
            
    else:
        #assert model either is a NHiTSModel or TFTModel or TideModel
        print('model is of type:', type(model))
        assert isinstance(model, (NHiTSModel, TFTModel, TiDEModel, RNNModel)), "model must be an instance of NHiTSModel, TFTModel or TiDEModel or RNNModel"
        if not hasattr(model, 'pl_trainer_kwargs'):
            model.pl_trainer_kwargs = {}
        trainer_kwargs = model.pl_trainer_kwargs.copy()
        callbacks = trainer_kwargs.get('callbacks', [])
        loss_history = None
        for callback in callbacks:
            if isinstance(callback, LossHistoryCallback):
                loss_history = callback
                print("Found existing LossHistoryCallback in model callbacks.")
                break
        # if loss_history is None:
        #     print("Adding LossHistoryCallback to model callbacks.")
        #     loss_history = LossHistoryCallback()
        #     trainer_kwargs['callbacks'] = callbacks + [loss_history]
        #     model.pl_trainer_kwargs = trainer_kwargs
        
        
        
    assert isinstance(n_forecast, int) and n_forecast > 0, "n_forecast must be a positive integer"
    
    if use_scaling:
        print(f'using scaling for training, scale_globally: {scale_globally}')
        if scale_globally:
            global_scaler = Scaler(RobustScaler())
            scaled_series_list = global_scaler.fit_transform(series_list)
            if val_list:
                scaled_val_list = global_scaler.transform(val_list)  # Use same scaler
            else:
                scaled_val_list = None
        else:
            scalers = [Scaler() for _ in series_list]
            scaled_series_list = [scalers[i].fit_transform(ser) for i, ser in enumerate(series_list)]
            if val_list:
                scaled_val_list = [scalers[i].transform(ser) for i, ser in enumerate(val_list)]  # Use same scalers
            else:
                scaled_val_list = None   
    else:
        scaled_series_list = series_list
        scaled_val_list = val_list if val_list else None  
           
    # --- Training ---
    model_type = type(model).__name__
    if verbose:

        print(f"Training {model_type} model with input_chunk_length={input_chunk_length}, output_chunk_length={output_chunk_length}, n_forecast={n_forecast}, n_epochs={n_epochs}, batch_size={batch_size}, lr={lr}, prediction_type={prediction_type}, distribution={distribution}")
    
    model.fit(scaled_series_list,
                val_series=scaled_val_list,
                future_covariates=future_covariates if model.supports_future_covariates else None,
                val_future_covariates=future_covariates if model.supports_future_covariates else None,
                past_covariates=past_covariates if model.supports_past_covariates else None,
                val_past_covariates=past_covariates if model.supports_past_covariates else None,
                epochs=n_epochs,
                verbose=verbose
            )
    print("Training complete.")
    
    
    if loss_history is not None:
        print("\nLoss history callback found. Plotting training and validation losses.")
        # Get training history and plot
        history_df = loss_history.get_history_df()
        print("\nTraining History:")
        print(history_df.tail())
        # Plot the losses
        loss_history.plot_losses()
        loss_history.plot_lr()
    
    # if return_history_df and loss_history is not None:
    #     print("Returning model and training history DataFrame.")
    #     history_df = loss_history.get_history_df()
    #     return model, history_df
    # else:
    print("Returning trained model.")
    return model

class LossHistoryCallback(Callback):
    """
    A Darts Callback to record training and validation loss history at each epoch.
    """
    def __init__(self):
        super().__init__()
        self.history = []
        self.has_validation = False
        # Add an ID to track instances
        import uuid
        self.instance_id = str(uuid.uuid4())[:8]

    def on_train_epoch_end(self, trainer, pl_module):
        """
        This hook is called after the training and validation steps of an epoch are complete.
        We can safely access all logged metrics here.
        """

        # Get the current epoch number
        epoch = trainer.current_epoch
        train_loss = trainer.logged_metrics.get('train_loss')
        train_loss = float(train_loss) if train_loss is not None else None
        
        current_lr = trainer.optimizers[0].param_groups[0]['lr']
        current_lr = float(current_lr) if current_lr is not None else None
        
        # Get the validation loss
        val_loss = trainer.logged_metrics.get('val_loss')
        if val_loss is not None:
            self.has_validation = True
        
        # Append all metrics for the current epoch
        entry = {
            'epoch': epoch,
            'train_loss': float(train_loss) if train_loss is not None else None,
            'val_loss': float(val_loss) if val_loss is not None else None,
            'lr': float(current_lr) if current_lr is not None else None
        }
        self.history.append(entry)


    def get_history_df(self):
        """
        Returns the collected loss history as a pandas DataFrame.
        """
        return pd.DataFrame(self.history)

    def plot_losses(self, save_path=None):
        """
        Plots the training and validation loss curves.
        """
        df = self.get_history_df()
        
        if df.empty:
            print("No loss history to plot.")
            return

        plt.figure(figsize=(8, 4))
        plt.plot(df['epoch'], df['train_loss'], label='Training Loss', color='blue', linewidth=2)
        # Plot validation loss only if it exists
        if self.has_validation and 'val_loss' in df.columns and not df['val_loss'].isnull().all():
            plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', color='red', linewidth=2, linestyle='--')
            plt.title('Training and Validation Loss Over Time')
        
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        #set y axis range to only be effected by the training loss
        plt.ylim(bottom=min(min(df['train_loss'])*1.5 ,min(df['train_loss'])*0.7 ), top=max(max(df['train_loss']) * 1.5, max(df['train_loss']) * 0.7))
        
        if save_path:
            save_path=os.path.join(save_path, f'loss_history.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        if self.has_validation:
            plt.figure(figsize=(8, 10))
            plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', color='red', linewidth=2, linestyle='--')
            plt.title('Validation Loss Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            if save_path:
                plt.savefig(save_path.replace('.png', '_val.png'), dpi=300, bbox_inches='tight')
            plt.show()
        
        return df
    
    def plot_lr(self, save_path=None):
        """
        Plots the learning rate over epochs if available.
        """
        if 'lr' not in self.history[0]:
            print("No learning rate history to plot.")
            return
        
        lrs = [entry.get('lr', None) for entry in self.history]
        epochs = [entry['epoch'] for entry in self.history]
        
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, lrs, label='Learning Rate', color='green', linewidth=2)
        plt.title('Learning Rate Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        if save_path:
            plt.savefig(save_path.replace('.png', '_lr.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
from pytorch_lightning.callbacks import Callback
    
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Assume 'model' is a trained Darts NHiTSModel
# from darts.models import NHiTSModel

def _calculate_likelihood_probabilities(parameter_forecast_df, threshold_upper, threshold_lower):
    """Calculates threshold crossing probabilities from likelihood parameters."""
    # This function expects a DataFrame with 'mu' and 'sigma' columns
    prob_values_upper = 1.0 - stats.norm.cdf(
        threshold_upper,
        loc=parameter_forecast_df['mu'],
        scale=parameter_forecast_df['sigma']
    )
    prob_values_lower = stats.norm.cdf(
        threshold_lower,
        loc=parameter_forecast_df['mu'],
        scale=parameter_forecast_df['sigma']
    )
    
    probs_df = pd.DataFrame({
        'Prob_Cross_Threshold_Upper': prob_values_upper,
        'Prob_Cross_Threshold_Lower': prob_values_lower
    }, index=parameter_forecast_df.index)

    dt_upper = probs_df['Prob_Cross_Threshold_Upper'].idxmax()
    dt_lower = probs_df['Prob_Cross_Threshold_Lower'].idxmax()
    highest_prob_upper = probs_df['Prob_Cross_Threshold_Upper'].max()
    highest_prob_lower = probs_df['Prob_Cross_Threshold_Lower'].max()

    highest_prob_dict = {
        'upper': {'datetime': dt_upper, 'probability': highest_prob_upper, 'threshold': threshold_upper},
        'lower': {'datetime': dt_lower, 'probability': highest_prob_lower, 'threshold': threshold_lower}
    }
    
    return probs_df, highest_prob_dict


def _calculate_quantile_probabilities(forecast_df, quantiles, threshold_upper, threshold_lower):
    """Estimates threshold crossing probabilities from quantile forecasts."""
    # This function expects a DataFrame from a SINGLE component's quantile forecast
    comp_name = '_'.join(forecast_df.columns[0].split('_')[:-1])
    
    upper_qs = sorted([q for q in quantiles if q >= 0.5])
    lower_qs = sorted([q for q in quantiles if q <= 0.5], reverse=True)

    upper_prob, lower_prob = 0.0, 0.0
    upper_dtime, lower_dtime = None, None

    for q in upper_qs:
        col_name = f"{comp_name}_{q}"
        if forecast_df[col_name].max() > threshold_upper:
            upper_prob = 1 - q  
            upper_dtime = forecast_df[col_name].idxmax()
            break
            
    for q in lower_qs:
        col_name = f"{comp_name}_{q}"
        if forecast_df[col_name].min() < threshold_lower:
            lower_prob = q
            lower_dtime = forecast_df[col_name].idxmin()
            break
    
    if upper_dtime is None:
        print('finding highest point of quantile', upper_qs[-1])
        #find the highest point of the highest quantile
        col_name = f"{comp_name}_{upper_qs[-1]}"
        upper_dtime = forecast_df[col_name].idxmax()
    if lower_dtime is None:
        print('finding lowest point of quantile', lower_qs[-1])
        #find the lowest point of the lowest quantile
        col_name = f"{comp_name}_{lower_qs[-1]}"
        lower_dtime = forecast_df[col_name].idxmin()
    highest_prob_dict = {
        'upper': {'datetime': upper_dtime, 'probability': upper_prob, 'threshold': threshold_upper},
        'lower': {'datetime': lower_dtime, 'probability': lower_prob, 'threshold': threshold_lower}
    }
    
    return highest_prob_dict




def analyze_forecast(
    forecast: TimeSeries,
    prediction_type: str,
    threshold_upper: float,
    threshold_lower: float,
    quantiles: list = None,
    mu_forecast: TimeSeries = None,
    sigma_forecast: TimeSeries = None
):
    """
    Analyzes probabilistic forecasts for uni/multivariate series.

    Args:
        forecast (TimeSeries): The probabilistic forecast to analyze.
        prediction_type (str): The type of probabilistic prediction ('likelihood' or 'quantiles').
        threshold_upper (float): The upper threshold for probability calculation.
        threshold_lower (float): The lower threshold for probability calculation.
        quantiles (list, optional): A list of quantiles for quantile-based prediction. Defaults to None.
        mu_forecast (TimeSeries, optional): The forecasted mean for likelihood-based prediction. Defaults to None.
        sigma_forecast (TimeSeries, optional): The forecasted standard deviation for likelihood-based prediction. Defaults to None.

    Returns:
        dict: A dictionary with analysis results for each component.
    """
    results = {}
    components = forecast.columns.tolist()

    for comp_name in components:
        comp_forecast = forecast[comp_name]
        comp_results = {}

        if prediction_type == 'likelihood':
            mu_comp = mu_forecast[comp_name]
            sigma_comp = sigma_forecast[comp_name]
            
            param_comp_df = mu_comp.to_dataframe()
            param_comp_df.columns = ['mu']
            param_comp_df['sigma'] = sigma_comp.to_series()
            
            # Note: _calculate_likelihood_probabilities is a placeholder for your actual implementation.
            probs_df, highest_prob = _calculate_likelihood_probabilities(param_comp_df, threshold_upper, threshold_lower)
            comp_results = {'type': 'likelihood', 'probabilities_df': probs_df, 'highest_probabilities': highest_prob}
            
        elif prediction_type == 'quantiles':
            forecast_df = comp_forecast.quantiles_df(quantiles=quantiles)
            
            # Note: _calculate_quantile_probabilities is a placeholder for your actual implementation.
            highest_prob = _calculate_quantile_probabilities(forecast_df, quantiles, threshold_upper, threshold_lower)
            comp_results = {'type': 'quantiles', 'highest_probabilities': highest_prob}
        
        results[comp_name] = comp_results
        
    return results
    
                                         
    
import pandas as pd
from darts import TimeSeries
from scipy import stats
import matplotlib.pyplot as plt



def predict_probabilistic_threshold(
    model,
    series,
    n_forecast: int,
    val_series: TimeSeries = None,
    prediction_type='likelihood',
    quantiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
    threshold=1.0,
    interactive_plot=False,
    **kwargs
) -> tuple:
    """
    Generates probabilistic forecasts using a trained N-HiTS model.

    Args:
        model (NHiTSModel): The trained N-HiTS model.
        series (TimeSeries): The target time series to forecast.
        n_forecast (int): The forecast horizon.
        prediction_type (str): Type of probabilistic prediction ('likelihood' or 'quantiles' or None).
        quantiles (list): List of quantiles to predict if prediction_type is 'quantiles'.
        single_chamber (str | None): Optional chamber name to predict. If None, all chambers are predicted.
        threshold (float | tuple): The threshold value(s) to calculate crossing probabilities.
                                     If a single float, both upper and lower thresholds are set to this value.
                                     If a tuple/list, the first value is the upper threshold and the second is the lower threshold.
        **kwargs: Additional arguments for the predict method.

    Returns:
        tuple: Contains:
            - forecast (TimeSeries): The probabilistic forecast.
            - crossing_probabilities (TimeSeries | None): A TimeSeries where each value
              represents the probability P(series > threshold) at that forecast step.
              Returns None if `quantiles` were used instead of `likelihood`, as direct
              probability calculation is less straightforward.
    """
    if isinstance(series, pd.DataFrame):
        # Convert DataFrame to Darts TimeSeries
        series = TimeSeries.from_dataframe(series, time_col='datetime', value_cols=series.columns.tolist(), freq='1h')
    elif isinstance(series, pd.Series):
        series = TimeSeries.from_series(series, freq='1h')

    series_orig = series.copy()
    scaler = Scaler()
    series = scaler.fit_transform(series)

    if isinstance(threshold, (int, float)):
        threshold_upper = threshold
        threshold_lower = -threshold
    elif isinstance(threshold, (list, tuple)) and len(threshold) == 2:
        threshold_upper = threshold[0]
        threshold_lower = threshold[1]
    else:
        raise ValueError("threshold must be a single value or a tuple/list of two values")

    print(f"Generating probabilistic forecast for {n_forecast} steps...")
    if prediction_type == 'likelihood':
        # Predict distribution parameters and sample from them
        # num_samples determines how many sample paths are generated
        forecast = model.predict(n=n_forecast, num_samples=1000)
        forecast = scaler.inverse_transform(forecast)  # Inverse transform to original scale
        if interactive_plot:
            plot_forecast_interactive(series_orig, forecast, threshold_upper, threshold_lower, val_series=val_series)
        else:
            _ =plot_probabilistic_forecast(series_orig, forecast, val_series=val_series)

        parameter_forecast_scaled = model.predict(n=n_forecast, predict_likelihood_parameters=True)
        mu_cols = [c for c in parameter_forecast_scaled.columns if c.endswith('_mu')]
        sigma_cols = [c for c in parameter_forecast_scaled.columns if c.endswith('_sigma')]
        mu_scaled_ts = parameter_forecast_scaled[mu_cols]
        sigma_scaled_ts = parameter_forecast_scaled[sigma_cols]
        # Un-scale mu using the scaler's full inverse_transform method
        mu_unscaled_ts = scaler.inverse_transform(mu_scaled_ts)
        # 1. Get the original data's range for each component
        series_orig_df = series_orig.to_dataframe()
        data_min_per_component = series_orig_df.min(axis=0)
        data_max_per_component = series_orig_df.max(axis=0)
        data_range_per_component = data_max_per_component - data_min_per_component
        # 2. Multiply the scaled sigma values by the per-component range
        sigma_unscaled_df = sigma_scaled_ts.to_dataframe() * data_range_per_component.values
        sigma_unscaled_ts = TimeSeries.from_dataframe(sigma_unscaled_df)
        # 3. Rename columns for consistency
        mu_unscaled_ts = mu_unscaled_ts.with_columns_renamed(mu_unscaled_ts.columns, [c.replace('_mu', '') for c in mu_unscaled_ts.columns])
        sigma_unscaled_ts = sigma_unscaled_ts.with_columns_renamed(sigma_unscaled_ts.columns, [c.replace('_sigma', '') for c in sigma_unscaled_ts.columns])

        # --- SANITY CHECK BLOCK ---
        print("\n--- SANITY CHECK: SCALING OF LIKELIHOOD PARAMETERS ---")
        # CORRECTED: Use .first_value() to extract the scalar from the TimeSeries object before formatting.
        series_min_val = series_orig.min(axis=None).first_value()
        series_max_val = series_orig.max(axis=None).first_value()
        mu_min_val = mu_unscaled_ts.min(axis=None).first_value()
        mu_max_val = mu_unscaled_ts.max(axis=None).first_value()
        sigma_min_val = sigma_unscaled_ts.min(axis=None).first_value()
        sigma_max_val = sigma_unscaled_ts.max(axis=None).first_value()


    elif prediction_type == 'quantiles':
        # Predict the specified quantiles directly
        forecast = model.predict(n=n_forecast, num_samples=1000)
        forecast = scaler.inverse_transform(forecast)  # Inverse transform to original scale
        if interactive_plot:
            plot_forecast_interactive(series_orig, forecast, threshold_upper, threshold_lower, val_series=val_series)
        else:
            _=plot_probabilistic_forecast(series_orig, forecast, val_series=val_series)

 
    if prediction_type in ['likelihood', 'quantiles']:
        analysis_results = analyze_forecast(forecast=forecast,
                                            prediction_type=prediction_type,
                                            threshold_upper=threshold_upper,
                                            threshold_lower=threshold_lower,
                                            quantiles=quantiles,
                                            mu_forecast=mu_unscaled_ts if prediction_type == 'likelihood' else None,
                                            sigma_forecast=sigma_unscaled_ts if prediction_type == 'likelihood' else None
                                            )
        for comp, res in analysis_results.items():
            print(f"\nComponent: {comp}")
            upper = res['highest_probabilities']['upper']
            lower = res['highest_probabilities']['lower']
            if upper['datetime']:
                print(f"  > Highest prob. of crossing upper threshold ({upper['threshold']:.2f}) is {upper['probability']:.2%} at {upper['datetime'].strftime('%Y-%m-%d %H:%M')}")
            if lower['datetime']:
                print(f"  > Highest prob. of crossing lower threshold ({lower['threshold']:.2f}) is {lower['probability']:.2%} at {lower['datetime'].strftime('%Y-%m-%d %H:%M')}")
   
        return forecast, analysis_results
    
    
    else:
        print("No probabilistic prediction requested. Returning non-probabilistic forecast.")
        # make non probabilistic prediction
        forecast = model.predict(n=n_forecast, num_samples=500)
        forecast = scaler.inverse_transform(forecast)  # Inverse transform to original scale
        if interactive_plot:
            plot_forecast_interactive(series_orig, forecast, threshold_upper, threshold_lower, val_series=val_series)
        else:
            plot_probabilistic_forecast(series_orig, forecast, val_series=val_series)
            
        return forecast, None
            
    
def predict_probabilistic_threshold_long_format(
    model: NHiTSModel,
    series: Union[TimeSeries, List[TimeSeries]],
    n_forecast: int,
    val_series: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
    prediction_type: str = 'likelihood',
    quantiles: List[float] = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
    threshold: Union[float, tuple[float, float]] = 1.0,
    interactive_plot: bool = False,
    future_covariates: Optional[List[TimeSeries]] = None,
    group_filter: Optional[dict[str, str]] = None,  # New: e.g., {'series_id': '1'} to predict only matching series
    **kwargs
) -> tuple[Union[TimeSeries, List[TimeSeries]], Optional[Union[dict, List[dict]]]]:
    """
    Generates probabilistic forecasts using a trained N-HiTS model on a list of TimeSeries (long format).

    Predicts all series by default, or filters to a subset/one based on group_filter.

    Args:
        model (NHiTSModel): The trained N-HiTS model.
        series (Union[TimeSeries, List[TimeSeries]]): The target time series list to forecast.
        n_forecast (int): The forecast horizon.
        val_series (Optional[Union[TimeSeries, List[TimeSeries]]]): Optional validation series list.
        prediction_type (str): Type of probabilistic prediction ('likelihood' or 'quantiles' or None).
        quantiles (list): List of quantiles to predict if prediction_type is 'quantiles'.
        threshold (float | tuple): The threshold value(s) to calculate crossing probabilities.
                                   If a single float, both upper and lower thresholds are set to this value.
                                   If a tuple/list, the first value is the upper threshold and the second is the lower threshold.
        interactive_plot (bool): Whether to use interactive plotting.
        future_covariates (Optional[List[TimeSeries]]): Optional list of future covariates (must match series list length).
        group_filter (Optional[Dict[str, str]]): Optional dict to filter series by static covariates (e.g., {'series_id': '1'}).
                                                 If provided, predicts only matching series (returns single TimeSeries if one match).
        **kwargs: Additional arguments for the predict method.

    Returns:
        tuple: Contains:
            - forecast (TimeSeries | List[TimeSeries]): Probabilistic forecast(s) (list if multiple, single if filtered to one).
            - analysis_results (dict | List[dict] | None): Analysis dict(s) with crossing probabilities (matches forecast structure).
              Returns None if non-probabilistic.
    """
    # Convert to lists for multi-series handling
    if isinstance(series, TimeSeries):
        series_list = [series]
    else:
        series_list = series

    if val_series is not None:
        if isinstance(val_series, TimeSeries):
            val_list = [val_series]
        else:
            val_list = val_series
        if len(val_list) != len(series_list):
            raise ValueError("val_series list must match length of series list.")
    else:
        val_list = [None] * len(series_list)

    if future_covariates and len(future_covariates) != len(series_list):
        raise ValueError("future_covariates must match length of series list.")

    # Optional: Filter to subset based on group_filter (using static covariates)
    if group_filter:
        filtered_series_list = []
        filtered_val_list = []
        filtered_cov_list = [] if future_covariates else None
        for i, ser in enumerate(series_list):
            if all(ser.static_covariates.get(key, None) == value for key, value in group_filter.items()):
                filtered_series_list.append(ser)
                filtered_val_list.append(val_list[i])
                if future_covariates:
                    filtered_cov_list.append(future_covariates[i])
        if not filtered_series_list:
            raise ValueError(f"No series match group_filter: {group_filter}")
        series_list = filtered_series_list
        val_list = filtered_val_list
        future_covariates = filtered_cov_list if future_covariates else None
        print(f"Filtered to {len(series_list)} matching series based on group_filter: {group_filter}")

    # Threshold parsing (unchanged)
    if isinstance(threshold, (int, float)):
        threshold_upper = threshold
        threshold_lower = -threshold
    elif isinstance(threshold, (list, tuple)) and len(threshold) == 2:
        threshold_upper = threshold[0]
        threshold_lower = threshold[1]
    else:
        raise ValueError("threshold must be a single value or a tuple/list of two values")

    # Scaling: One scaler per series
    scalers = [Scaler() for _ in series_list]
    series_orig_list = [ser.copy() for ser in series_list]
    scaled_series_list = [scalers[i].fit_transform(ser) for i, ser in enumerate(series_list)]

    print(f"Generating probabilistic forecast for {n_forecast} steps across {len(series_list)} series...")

    forecast_list = []
    analysis_results_list = None

    if prediction_type == 'likelihood':
        # Global predict on (filtered) list, with samples
        forecast_scaled_list = model.predict(
            n=n_forecast, series=scaled_series_list, future_covariates=future_covariates, num_samples=1000, **kwargs
        )
        # Inverse scale per series
        forecast_list = [scalers[i].inverse_transform(fore) for i, fore in enumerate(forecast_scaled_list)]

        # Plot per series (loop over lists)
        if interactive_plot:
            plot_forecast_interactive(series_orig_list, forecast_list, threshold_upper, threshold_lower, val_series=val_list)
        else:
            _ = plot_probabilistic_forecast(series_orig_list, forecast_list, val_series=val_list)

        # Extract and unscale parameters per series
        parameter_forecast_scaled_list = model.predict(
            n=n_forecast, series=scaled_series_list, future_covariates=future_covariates, predict_likelihood_parameters=True, **kwargs
        )
        mu_unscaled_list = []
        sigma_unscaled_list = []
        for i, param_fore_scaled in enumerate(parameter_forecast_scaled_list):
            mu_cols = [c for c in param_fore_scaled.columns if c.endswith('_mu')]
            sigma_cols = [c for c in param_fore_scaled.columns if c.endswith('_sigma')]
            mu_scaled_ts = param_fore_scaled[mu_cols]
            sigma_scaled_ts = param_fore_scaled[sigma_cols]

            # Unscale mu
            mu_unscaled_ts = scalers[i].inverse_transform(mu_scaled_ts)

            # Unscale sigma (manual range-based)
            series_orig_df = series_orig_list[i].to_dataframe()
            data_min_per_component = series_orig_df.min(axis=0)
            data_max_per_component = series_orig_df.max(axis=0)
            data_range_per_component = data_max_per_component - data_min_per_component
            sigma_unscaled_df = sigma_scaled_ts.to_dataframe() * data_range_per_component.values
            sigma_unscaled_ts = TimeSeries.from_dataframe(sigma_unscaled_df)

            # Rename columns
            mu_unscaled_ts = mu_unscaled_ts.with_columns_renamed(mu_unscaled_ts.columns, [c.replace('_mu', '') for c in mu_unscaled_ts.columns])
            sigma_unscaled_ts = sigma_unscaled_ts.with_columns_renamed(sigma_unscaled_ts.columns, [c.replace('_sigma', '') for c in sigma_unscaled_ts.columns])

            mu_unscaled_list.append(mu_unscaled_ts)
            sigma_unscaled_list.append(sigma_unscaled_ts)

        # # Sanity check per series
        # for i in range(len(series_list)):
        #     print(f"\n--- SANITY CHECK: SCALING OF LIKELIHOOD PARAMETERS (Series {i}) ---")
        #     series_min_val = series_orig_list[i].min(axis=None).first_values()[0]
        #     series_max_val = series_orig_list[i].max(axis=None).first_values()[0]
        #     mu_min_val = mu_unscaled_list[i].min(axis=None).first_values()[0]
        #     mu_max_val = mu_unscaled_list[i].max(axis=None).first_values()[0]
        #     sigma_min_val = sigma_unscaled_list[i].min(axis=None).first_values()[0]
        #     sigma_max_val = sigma_unscaled_list[i].max(axis=None).first_values()[0]
        #     # Add your print logic here if needed

    elif prediction_type == 'quantiles':
        # Global predict on (filtered) list
        forecast_scaled_list = model.predict(
            n=n_forecast, series=scaled_series_list, future_covariates=future_covariates, num_samples=1000, **kwargs
        )
        forecast_list = [scalers[i].inverse_transform(fore) for i, fore in enumerate(forecast_scaled_list)]

        # Plot per series
        for i in range(len(series_list)):
            if interactive_plot:
                plot_forecast_interactive(series_orig_list[i], forecast_list[i], threshold_upper, threshold_lower, val_series=val_list[i])
            else:
                _ = plot_probabilistic_forecast(series_orig_list[i], forecast_list[i], val_series=val_list[i])

    else:
        print("No probabilistic prediction requested. Returning non-probabilistic forecast.")
        # Global non-probabilistic predict
        forecast_scaled_list = model.predict(
            n=n_forecast, series=scaled_series_list, future_covariates=future_covariates, num_samples=500, **kwargs
        )
        forecast_list = [scalers[i].inverse_transform(fore) for i, fore in enumerate(forecast_scaled_list)]

        # Plot per series
        for i in range(len(series_list)):
            if interactive_plot:
                plot_forecast_interactive(series_orig_list[i], forecast_list[i], threshold_upper, threshold_lower, val_series=val_list[i])
            else:
                plot_probabilistic_forecast(series_orig_list[i], forecast_list[i], val_series=val_list[i])

        # If filtered to one, return single instead of list
        if group_filter and len(forecast_list) == 1:
            return forecast_list[0], None
        return forecast_list, None

    # Analysis per series
    if prediction_type in ['likelihood', 'quantiles']:
        analysis_results_list = []
        for i in range(len(forecast_list)):
            mu_fore = mu_unscaled_list[i] if prediction_type == 'likelihood' else None
            sigma_fore = sigma_unscaled_list[i] if prediction_type == 'likelihood' else None
            res = analyze_forecast(
                forecast=forecast_list[i],
                prediction_type=prediction_type,
                threshold_upper=threshold_upper,
                threshold_lower=threshold_lower,
                quantiles=quantiles,
                mu_forecast=mu_fore,
                sigma_forecast=sigma_fore
            )
            analysis_results_list.append(res)

            # Print per series/component
            print(f"\nSeries {i}:")
            for comp, res_comp in res.items():
                print(f"  Component: {comp}")
                upper = res_comp['highest_probabilities']['upper']
                lower = res_comp['highest_probabilities']['lower']
                if upper['datetime']:
                    print(f"    > Highest prob. of crossing upper threshold ({upper['threshold']:.2f}) is {upper['probability']:.2%} at {upper['datetime'].strftime('%Y-%m-%d %H:%M')}")
                if lower['datetime']:
                    print(f"    > Highest prob. of crossing lower threshold ({lower['threshold']:.2f}) is {lower['probability']:.2%} at {lower['datetime'].strftime('%Y-%m-%d %H:%M')}")

        # If filtered to one, return single instead of list
        if group_filter and len(forecast_list) == 1:
            return forecast_list[0], analysis_results_list[0]

        return forecast_list, analysis_results_list

    # If filtered to one, return single instead of list
    if group_filter and len(forecast_list) == 1:
        return forecast_list[0], None
    return forecast_list, None


    
import matplotlib.pyplot as plt
from darts import TimeSeries
from typing import Optional
import pandas as pd # Often needed for time operations




def plot_probabilistic_forecast(
    series: Union[TimeSeries, List[TimeSeries]],
    forecast: Union[TimeSeries, List[TimeSeries]],
    val_series: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
    context_multiplier: int = 10,
    quantiles: Optional[list] = [0.05, 0.25, 0.4, 0.5, 0.6, 0.75, 0.95,],
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (12, 8),
    series_label: str = "Series",
    forecast_label: str = "Forecast",
    title: str = "Probabilistic Forecast with Historical Context"
) -> Optional[plt.Axes]:
    """
    Plots a Darts forecast, handling univariate, multivariate, and lists of series (long format).

    For univariate series, it produces a single static plot.
    For multivariate series or lists of series, it creates an interactive tabbed interface with a
    separate plot for each component/series. This requires the 'ipywidgets' library.

    The historical portion shown ends just before the forecast starts
    and has a length defined by `context_multiplier` times the forecast horizon.

    Args:
        series (Union[TimeSeries, List[TimeSeries]]): The full historical TimeSeries or list (can be multivariate).
        forecast (Union[TimeSeries, List[TimeSeries]]): The forecast TimeSeries or list (can be multivariate).
        val_series (Optional[Union[TimeSeries, List[TimeSeries]]]): Optional validation TimeSeries or list.
        context_multiplier (int): The historical window length will be
                                `context_multiplier * len(forecast)`.
        quantiles (Optional[list]): Quantiles to display for probabilistic forecasts.
        ax (Optional[plt.Axes]): A Matplotlib Axes object. If provided, the series must
                                be univariate. This argument is ignored for
                                multivariate/lists.
        figsize (tuple): The figure size for each plot.
        series_label (str): Label for the historical series.
        forecast_label (str): Label for the forecast median.
        title (str): Base title for the plot(s).

    Returns:
        Optional[plt.Axes]:
        - For a univariate series, returns the Matplotlib Axes object used for plotting.
        - For multivariate/lists, displays an interactive widget and returns None.

    Raises:
        ValueError: If inputs are mismatched or empty.
        ImportError: If multivariate/lists are used and 'ipywidgets' is not installed.
        ValueError: If an 'ax' is provided for multivariate/lists.
    """
    # Convert to lists for consistency (handle single TimeSeries as list of 1)
    if isinstance(series, TimeSeries):
        series_list = [series]
    else:
        series_list = series

    if isinstance(forecast, TimeSeries):
        forecast_list = [forecast]
    else:
        forecast_list = forecast

    if val_series is not None:
        if isinstance(val_series, TimeSeries):
            val_list = [val_series]
        else:
            val_list = val_series
    else:
        val_list = [None] * len(series_list)

    # Validation: Ensure lists match in length
    if len(series_list) != len(forecast_list) or len(series_list) != len(val_list):
        raise ValueError("series, forecast, and val_series must have the same length if lists.")

    if any(len(fore) == 0 for fore in forecast_list):
        raise ValueError("Forecast TimeSeries cannot be empty.")

    series_list, forecast_list, val_list = _sort_lists_by_series_id(series_list, forecast_list, val_list)

    # Determine if we need tabs (multi-series or multivariate)
    is_multi_series = len(series_list) > 1
    is_multivariate = any(ser.width > 1 for ser in series_list)
    print(f"Detected {'multivariate' if is_multivariate else 'univariate'} series with {len(series_list)} components.")
    print('using tabbed interface:', is_multi_series or is_multivariate)

    if not is_multi_series and not is_multivariate:
        # --- 1. Single Univariate Case: Static plot ---
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        _plot_single_component(
            ax=ax,
            series_comp=series_list[0],
            forecast_comp=forecast_list[0],
            val_series_comp=val_list[0],
            context_multiplier=context_multiplier,
            quantiles=quantiles,
            series_label=series_label,
            forecast_label=forecast_label,
            title=title
        )
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()
        return ax

    else:
        # --- 2. Multi-Series or Multivariate Case: Interactive tabs ---
        if not _IPYWIDGETS_AVAILABLE:
            raise ImportError(
                "The 'ipywidgets' library is required for multi-series/multivariate plotting. "
                "Please install it (e.g., 'pip install ipywidgets') and enable it "
                "in your Jupyter environment."
            )
        if ax is not None:
            raise ValueError("The 'ax' argument cannot be used for multi-series/multivariate plotting.")

        if is_multivariate:
            # Handle multivariate as before (flattened tabs per component)
            tab_children = []
            tab_titles = []
            tab_widget = Tab()

            for idx, ser in enumerate(series_list):
                components = ser.columns
                for comp_idx, comp_name in enumerate(components):
                    tab_title = f"Series {idx} - {comp_name}" if is_multi_series else str(comp_idx)
                    tab_titles.append(tab_title)
                    output_widget = create_plot_output(
                        series_comp=ser[comp_name],
                        forecast_comp=forecast_list[idx][comp_name],
                        val_series_comp=val_list[idx][comp_name] if val_list[idx] else None,
                        context_multiplier=context_multiplier,
                        quantiles=quantiles,
                        series_label=series_label,
                        forecast_label=forecast_label,
                        title=f"{title}\n{tab_title}",
                        figsize=figsize
                    )
                    tab_children.append(output_widget)

            tab_widget.children = tab_children
            for i, tab_title in enumerate(tab_titles):
                tab_widget.set_title(i, tab_title)

            display(tab_widget)
            return None

        else:
            # Handle list of univariate series with grouping by static covariates
            # Group by client_id > installation_id > series_id
            groups = defaultdict(lambda: defaultdict(list))
            for idx, ser in enumerate(series_list):
                if ser.static_covariates is not None and not ser.static_covariates.empty:
                    # Assuming univariate, take first row as dict
                    cov = ser.static_covariates.iloc[0].to_dict()
                    client_id = cov.get('client_id', 'Unknown')
                    installation_id = cov.get('installation_id', 'Unknown')
                    series_id = cov.get('series_id', 'Unknown')
                else:
                    client_id = 'Unknown'
                    installation_id = 'Unknown'
                    series_id = f"Series {idx}"

                groups[client_id][installation_id].append((idx, ser, forecast_list[idx], val_list[idx], series_id))

            # Build nested tabs
            top_tab = Tab()
            top_children = []
            top_titles = []
            for client_id in sorted(groups.keys()):
                client_groups = groups[client_id]
                if len(client_groups) == 1:
                    # Only one installation
                    installation_id = next(iter(client_groups))
                    install_series = client_groups[installation_id]
                    if len(install_series) == 1:
                        # Single series: direct plot
                        idx, ser, fore, val, ser_id = install_series[0]
                        plot_title = f"{title}\nClient {client_id} - Install {installation_id} - Series {ser_id}"
                        output_widget = create_plot_output(
                            series_comp=ser,
                            forecast_comp=fore,
                            val_series_comp=val,
                            context_multiplier=context_multiplier,
                            quantiles=quantiles,
                            series_label=series_label,
                            forecast_label=forecast_label,
                            title=plot_title,
                            figsize=figsize
                        )
                        top_children.append(output_widget)
                        top_titles.append(f"Client {client_id} - Install {installation_id}")
                    else:
                        # Multiple series in one installation: sub-tab for series
                        sub_tab = Tab()
                        sub_children = []
                        sub_titles = []
                        for _, (idx, ser, fore, val, ser_id) in enumerate(sorted(install_series, key=lambda x: x[4])):
                            plot_title = f"{title}\nClient {client_id} - Install {installation_id} - Series {ser_id}"
                            output_widget = create_plot_output(
                                series_comp=ser,
                                forecast_comp=fore,
                                val_series_comp=val,
                                context_multiplier=context_multiplier,
                                quantiles=quantiles,
                                series_label=series_label,
                                forecast_label=forecast_label,
                                title=plot_title,
                                figsize=figsize
                            )
                            sub_children.append(output_widget)
                            sub_titles.append(f"Series {ser_id}")
                        sub_tab.children = sub_children
                        for i, t in enumerate(sub_titles):
                            sub_tab.set_title(i, t)
                        top_children.append(sub_tab)
                        top_titles.append(f"Client {client_id} - Install {installation_id}")
                else:
                    # Multiple installations: mid-tab for installations
                    mid_tab = Tab()
                    mid_children = []
                    mid_titles = []
                    for installation_id in sorted(client_groups.keys()):
                        install_series = client_groups[installation_id]
                        if len(install_series) == 1:
                            # Single series in installation: direct plot
                            idx, ser, fore, val, ser_id = install_series[0]
                            plot_title = f"{title}\nClient {client_id} - Install {installation_id} - Series {ser_id}"
                            output_widget = create_plot_output(
                                series_comp=ser,
                                forecast_comp=fore,
                                val_series_comp=val,
                                context_multiplier=context_multiplier,
                                quantiles=quantiles,
                                series_label=series_label,
                                forecast_label=forecast_label,
                                title=plot_title,
                                figsize=figsize
                            )
                            mid_children.append(output_widget)
                            mid_titles.append(f"Install {installation_id} - Series {ser_id}")
                        else:
                            # Multiple series: sub-tab for series
                            sub_tab = Tab()
                            sub_children = []
                            sub_titles = []
                            for _, (idx, ser, fore, val, ser_id) in enumerate(sorted(install_series, key=lambda x: x[4])):
                                plot_title = f"{title}\nClient {client_id} - Install {installation_id} - Series {ser_id}"
                                output_widget = create_plot_output(
                                    series_comp=ser,
                                    forecast_comp=fore,
                                    val_series_comp=val,
                                    context_multiplier=context_multiplier,
                                    quantiles=quantiles,
                                    series_label=series_label,
                                    forecast_label=forecast_label,
                                    title=plot_title,
                                    figsize=figsize
                                )
                                sub_children.append(output_widget)
                                sub_titles.append(f"Series {ser_id}")
                            sub_tab.children = sub_children
                            for i, t in enumerate(sub_titles):
                                sub_tab.set_title(i, t)
                            mid_children.append(sub_tab)
                            mid_titles.append(f"Install {installation_id}")
                    mid_tab.children = mid_children
                    for i, t in enumerate(mid_titles):
                        mid_tab.set_title(i, t)
                    top_children.append(mid_tab)
                    top_titles.append(f"Client {client_id}")

            top_tab.children = top_children
            for i, t in enumerate(top_titles):
                top_tab.set_title(i, t)

            # Display the widget
            display(top_tab)
            return None

from collections import defaultdict
from ipywidgets.embed import embed_minimal_html
from IPython.display import HTML, display
from collections import defaultdict
from ipywidgets.embed import embed_minimal_html
def plot_forecasts_with_dropdowns(
    series: Union[TimeSeries, List[TimeSeries]],
    forecast: Union[TimeSeries, List[TimeSeries]],
    historical_forecasts: Optional[List[List[TimeSeries]]] = None,
    figsize: tuple = (12, 6),
    mapes: Optional[Union[float, List[float]]] = None,
    **kwargs
    ):
    """
    Plots forecasts using cascading dropdowns (client -> installation -> series)
    plus an optional slider to inspect historical forecasts.

    Slider conventions:
        - value == -1 -> show the latest forecast (default)
        - value >= 0  -> show the corresponding historical forecast burst
    """
    from collections import defaultdict
    from typing import Dict, Any, Optional, Tuple
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import ipywidgets as widgets

    def _safe_compute_mape(true_ts: TimeSeries, pred_ts: TimeSeries) -> Optional[float]:
        try:
            yt = true_ts.slice_intersect(pred_ts)
            yp = pred_ts.slice_intersect(true_ts)
            if len(yt) == 0:
                return None
            try:
                return float(safe_mape_metric(yt, yp))  # type: ignore
            except Exception:
                y = yt.values().flatten()
                p = yp.values().flatten()
                denom = np.where(y == 0, np.nan, y)
                return float(np.nanmean(np.abs((y - p) / denom)) * 100.0)
        except Exception:
            return None

    
    series_list = [series] if isinstance(series, TimeSeries) else series
    forecast_list = [forecast] if isinstance(forecast, TimeSeries) else forecast

    data_map: Dict[str, Dict[str, List[Tuple[str, Tuple[str, str, str]]]]] = defaultdict(lambda: defaultdict(list))
    series_lookup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for i, s in enumerate(series_list):
        try:
            client = str(s.static_covariates['client_id'].values[0])
            install = str(s.static_covariates['installation_id'].values[0])
            series_id = str(s.static_covariates['series_id'].values[0])
            series_key = (client, install, series_id)

            # Latest forecast MAPE (priority: provided list -> compute -> None)
            mape_value = None
            if mapes is not None:
                if isinstance(mapes, (list, tuple)):
                    if i < len(mapes):
                        try:
                            mape_value = float(mapes[i])
                        except Exception:
                            mape_value = None
                else:
                    try:
                        mape_value = float(mapes)
                    except Exception:
                        mape_value = None

            latest_forecast = forecast_list[i]
            hist_list = []
            if historical_forecasts is not None and i < len(historical_forecasts):
                hist_candidate = historical_forecasts[i]
                if hist_candidate is not None:
                    hist_list = list(hist_candidate)

            series_lookup[series_key] = {
                "series": s,
                "series_id": series_id,
                "forecast": latest_forecast,
                "mape": mape_value,
                "historical": hist_list,
                "historical_mape_cache": [None] * len(hist_list),
                "latest_mape_cache": mape_value  # may be None; lazily computed if absent
            }

            # Only append to the navigation map once the lookup entry is valid
            data_map[client][install].append((series_id, series_key))

        except (KeyError, AttributeError, IndexError):
            print(f"Warning: Skipping series at index {i} due to missing data.")
            continue

    # --- Widgets ---
    client_dd = widgets.Dropdown(description='Client ID:')
    install_dd = widgets.Dropdown(description='Installation ID:')
    series_dd = widgets.Dropdown(description='Series ID:')
    hist_slider = widgets.IntSlider(
        description='Historical idx:',
        min=-1,
        max=-1,
        step=1,
        value=-1,
        disabled=True,
        readout=True,
        continuous_update=True
    )
    output_plot = widgets.Output()

    suppress_hist_update = False  # protects slider callback during programmatic updates

    # --- Helper to render the plot ---
    def render_plot(series_key: Tuple[str, str, str], hist_idx: int = -1) -> None:
        data = series_lookup[series_key]
        hist_list = data.get("historical", [])
        ground_truth = data['series']
        series_label = data.get("series_id", series_key[-1])

        use_hist = hist_idx >= 0 and hist_idx < len(hist_list)

        if use_hist:
            forecast_comp = hist_list[hist_idx]
            forecast_label = f"Historical Forecast #{hist_idx}"
            subtitle = forecast_label
            hist_mape_cache = data.get("historical_mape_cache")
            if hist_mape_cache is None or len(hist_mape_cache) != len(hist_list):
                hist_mape_cache = [None] * len(hist_list)
                data["historical_mape_cache"] = hist_mape_cache
            mape_value = hist_mape_cache[hist_idx]
            if mape_value is None:
                mape_value = _safe_compute_mape(ground_truth, forecast_comp)
                hist_mape_cache[hist_idx] = mape_value
        else:
            forecast_comp = data["forecast"]
            forecast_label = "Forecast"
            subtitle = "Latest Forecast"
            mape_value = data.get("latest_mape_cache")
            if mape_value is None:
                mape_value = _safe_compute_mape(ground_truth, forecast_comp)
                data["latest_mape_cache"] = mape_value

        title_base = f"Client: {client_dd.value} | Install: {install_dd.value}\nSeries: {series_label}"
        title = f"{title_base} — {subtitle}"
        if mape_value is not None:
            try:
                if not math.isnan(mape_value):
                    title += f" | MAPE: {mape_value:.2f}%"
            except Exception:
                pass

        plot_kwargs = dict(kwargs)
        plot_kwargs.setdefault("forecast_label", forecast_label)

        with output_plot:
            output_plot.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=figsize)
            _plot_single_component(
                ax=ax,
                series_comp=ground_truth,
                forecast_comp=forecast_comp,
                title=title,
                **plot_kwargs
            )
            plt.show()

    # --- Callbacks ---
    def on_client_change(change):
        if change['name'] != 'value' or not change['new']:
            return
        client = change['new']
        new_install_options = sorted(data_map[client].keys())
        install_dd.options = new_install_options
        install_dd.value = new_install_options[0] if new_install_options else None

    def on_install_change(change):
        if change['name'] != 'value' or not change['new']:
            return
        client = client_dd.value
        install = change['new']
        series_entries = data_map[client][install]
        series_entries_sorted = sorted(series_entries, key=lambda item: item[0])
        series_dd.options = [(sid, key) for sid, key in series_entries_sorted]
        series_dd.value = series_entries_sorted[0][1] if series_entries_sorted else None

    def on_series_change(change):
        if change['name'] != 'value' or not change['new']:
            return
        series_key = change['new']
        data = series_lookup[series_key]
        hist_list = data.get("historical", [])

        nonlocal suppress_hist_update
        suppress_hist_update = True

        if hist_list:
            hist_slider.min = -1
            hist_slider.max = len(hist_list) - 1
            hist_slider.disabled = False
            hist_slider.value = -1  # default to latest forecast
        else:
            hist_slider.min = -1
            hist_slider.max = -1
            hist_slider.value = -1
            hist_slider.disabled = True

        suppress_hist_update = False
        render_plot(series_key, hist_idx=-1)

    def on_hist_slider_change(change):
        if change['name'] != 'value' or change['new'] is None or suppress_hist_update:
            return
        current_series_key = series_dd.value
        if current_series_key is None:
            return
        render_plot(current_series_key, hist_idx=change['new'])

    # --- Bind callbacks ---
    client_dd.observe(on_client_change, names='value')
    install_dd.observe(on_install_change, names='value')
    series_dd.observe(on_series_change, names='value')
    hist_slider.observe(on_hist_slider_change, names='value')

    # --- Initialize UI ---
    client_dd.options = sorted(data_map.keys())
    if client_dd.options:
        client_dd.value = client_dd.options[0]

    ui = widgets.VBox([client_dd, install_dd, series_dd, hist_slider, output_plot])
    display(ui)
    
from IPython.display import display, HTML
def plot_forecasts_with_dropdowns_plotly(
    series: Union[TimeSeries, List[TimeSeries]],
    forecast: Union[TimeSeries, List[TimeSeries]],
    val_series: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
    figsize: tuple = (1600, 800),  # In pixels for Plotly
    save_html_path: str = None,
    show: bool = True,
    context_multiplier: int = 1,
    quantiles: Optional[list] = [0.05, 0.25, 0.4, 0.5, 0.6, 0.75, 0.95],
    series_label: str = 'Historical',
    forecast_label: str = 'Forecast',
    val_label: str = 'Validation',
    mapes: Optional[Union[float, List[float]]] = None,  # If provided, display in title
    historical_forecasts: Optional[List[List[TimeSeries]]] = None,  # NEW: List of lists of historical forecasts per series
    historical_mapes: Optional[List[List[float]]] = None,  # NEW: List of lists of MAPEs per historical forecast
    is_multivariate: Optional[bool] = None,  # New: None for auto-detect, True for multivariate (wide), False for univariate (long)
    **kwargs
):
    """
    Plots forecasts using cascading dropdowns for navigation in a shareable HTML file.
    The hierarchy is fixed to: client_id -> installation_id -> series_id.
    Uses Plotly.js for interactive plots and custom HTML/JS for cascading dropdowns.

    Now shows MAPE in the title and as an on-plot annotation (if provided or computable).
    Supports both univariate (long format) and multivariate (wide format) TimeSeries.
    """

    from collections import defaultdict
    import json, os
    from typing import List, Optional, Union
    import numpy as np
    try:
        from IPython.display import HTML, display
    except Exception:
        pass

    # Wrap singles into lists
    series_list = [series] if isinstance(series, TimeSeries) else series
    forecast_list = [forecast] if isinstance(forecast, TimeSeries) else forecast
    val_list = [val_series] if isinstance(val_series, TimeSeries) else val_series if val_series else [None] * len(series_list)

    # Fallbacks and validation for new params
    historical_forecasts = historical_forecasts or [[] for _ in series_list]
    historical_mapes = historical_mapes or [[] for _ in series_list]
    if len(historical_forecasts) != len(series_list):
        raise ValueError("historical_forecasts must match the number of series")
    if len(historical_mapes) != len(series_list):
        raise ValueError("historical_mapes must match the number of series")

    # Auto-detect format if not specified
    if is_multivariate is None:
        # Check if all series have 'series_id' in static_covariates (univariate mode)
        has_series_id = all(
            hasattr(s, 'static_covariates') and s.static_covariates is not None and 'series_id' in s.static_covariates.columns
            for s in series_list
        )
        is_multivariate = not has_series_id
        print(f"Auto-detected mode: {'multivariate' if is_multivariate else 'univariate'}")

    data_map = defaultdict(lambda: defaultdict(list))
    js_data = {}  # {client: {install: {series_id: {...}}}}

    def _safe_compute_mape(true_ts: TimeSeries, pred_ts: TimeSeries) -> Optional[float]:
        # Try safe_mape_metric if available; else fallback
        try:
            yt = true_ts.slice_intersect(pred_ts)
            yp = pred_ts.slice_intersect(true_ts)
            if len(yt) == 0:
                return None
            try:
                return float(safe_mape_metric(yt, yp))  # type: ignore
            except Exception:
                y = yt.values().flatten()
                p = yp.values().flatten()
                denom = np.where(y == 0, np.nan, y)
                return float(np.nanmean(np.abs((y - p) / denom)) * 100.0)
        except Exception:
            return None

    # Process based on mode
    if not is_multivariate:  # Univariate mode (original logic)
        for i, s in enumerate(series_list):
            try:
                client = str(s.static_covariates['client_id'].values[0])
                install = str(s.static_covariates['installation_id'].values[0])
                series_name = str(s.static_covariates['series_id'].values[0])

                data_map[client][install].append(series_name)

                # Process historical forecasts + main forecast into lists
                all_forecasts = historical_forecasts[i] + [forecast_list[i]]  # Historical first, main last
                all_mapes = []
                forecasts_data = []
                series_datas = []  # List of series_data per forecast
                val_datas = []  # List of val_data per forecast

                for j, fc in enumerate(all_forecasts):
                    # ENHANCED: Compute horizon and context_len PER FORECAST
                    horizon = len(fc)
                    context_len = context_multiplier * horizon
                    forecast_start = fc.time_index[0]
                    forecast_end = fc.time_index[-1]
                    s_sliced = s.slice_intersect(fc)

                    series_data = {
                        'time': [str(t) for t in s_sliced.time_index],
                        'values': s_sliced.values().flatten().tolist()
                    }
                    series_datas.append(series_data)

                    # ENHANCED: For validation, use slice_intersect with the forecast (to show overlapping ground truth only)
                    val_sliced = None
                    if val_list[i] is not None:
                        try:
                            val_sliced = val_list[i].slice_intersect(fc)
                        except Exception:
                            val_sliced = val_list[i][-context_len:] if context_len > 0 and len(val_list[i]) > context_len else val_list[i]
                    val_data = {
                        'time': [str(t) for t in val_sliced.time_index],
                        'values': val_sliced.values().flatten().tolist()
                    } if val_sliced is not None and len(val_sliced) > 0 else None
                    val_datas.append(val_data)

                    # Build forecast payload (quantiles or deterministic)
                    if quantiles:
                        quantiles_sorted = sorted(set(quantiles))
                        forecast_data = {
                            'time': [str(t) for t in fc.time_index],
                            'quantiles': {str(q): fc.quantile(q).values().flatten().tolist() for q in quantiles_sorted},
                            'has_median': 0.5 in quantiles_sorted
                        }
                        is_symmetric = all((1 - q in quantiles_sorted) for q in quantiles_sorted if q < 0.5)
                        forecast_data['is_symmetric'] = is_symmetric
                        if is_symmetric:
                            pairs = [(q, 1 - q) for q in quantiles_sorted if q < 0.5]
                            forecast_data['pairs'] = [[str(round(p[0],2)), str(round(p[1],2))] for p in sorted(pairs, key=lambda x: x[0])]
                        else:
                            forecast_data['pairs'] = []
                    else:
                        forecast_data = {
                            'time': [str(t) for t in fc.time_index],
                            'values': fc.values().flatten().tolist()
                        }
                    forecasts_data.append(forecast_data)

                    # Determine MAPE per forecast
                    mape_value = None
                    if j == len(all_forecasts) - 1 and mapes is not None:  # Main (last) forecast uses existing mapes logic
                        if isinstance(mapes, (list, tuple)):
                            if i < len(mapes):
                                try:
                                    mape_value = float(mapes[i])
                                except Exception:
                                    mape_value = None
                        else:
                            try:
                                mape_value = float(mapes)
                            except Exception:
                                mape_value = None
                    elif j < len(all_forecasts) - 1 and historical_mapes and j < len(historical_mapes[i]):
                        mape_value = historical_mapes[i][j]
                    elif val_list[i] is not None:
                        try:
                            point_fc = fc.quantile(0.5)
                        except Exception:
                            point_fc = fc
                        mape_value = _safe_compute_mape(val_list[i], point_fc)
                    all_mapes.append(mape_value)

                title = f"Client: {client} | Install: {install}<br>Series: {series_name}"

                if client not in js_data:
                    js_data[client] = {}
                if install not in js_data[client]:
                    js_data[client][install] = {}

                js_data[client][install][series_name] = {
                    'series_list': series_datas,
                    'val_list': val_datas,
                    'forecasts': forecasts_data,
                    'title': title,
                    'series_name': series_name,
                    'quantiles': bool(quantiles),
                    'mapes_list': all_mapes
                }
            except (KeyError, AttributeError, IndexError):
                print(f"Warning: Skipping series at index {i} due to missing static covariates or data issues.")
                continue

    else:  # Multivariate mode
        for i, s in enumerate(series_list):
            try:
                client = str(s.static_covariates['client_id'].values[0])
                install = str(s.static_covariates['installation_id'].values[0])

                components = s.components
                for component in components:
                    series_name = str(component)

                    data_map[client][install].append(series_name)

                    s_uni = s[component]
                    forecast_uni = forecast_list[i][component]
                    val_uni = val_list[i][component] if val_list[i] is not None else None
                    historical_forecasts_uni = [hf[component] for hf in historical_forecasts[i]] if historical_forecasts[i] else []

                    # Process historical forecasts + main forecast into lists
                    all_forecasts = historical_forecasts_uni + [forecast_uni]  # Historical first, main last
                    all_mapes = []
                    forecasts_data = []
                    series_datas = []  # List of series_data per forecast
                    val_datas = []  # List of val_data per forecast

                    for j, fc in enumerate(all_forecasts):
                        # Compute horizon and context_len PER FORECAST
                        horizon = len(fc)
                        context_len = context_multiplier * horizon
                        s_sliced = s_uni.slice_intersect(fc)

                        series_data = {
                            'time': [str(t) for t in s_sliced.time_index],
                            'values': s_sliced.values().flatten().tolist()
                        }
                        series_datas.append(series_data)

                        # For validation, use slice_intersect with the forecast
                        val_sliced = None
                        if val_uni is not None:
                            try:
                                val_sliced = val_uni.slice_intersect(fc)
                            except Exception:
                                val_sliced = val_uni[-context_len:] if context_len > 0 and len(val_uni) > context_len else val_uni
                        val_data = {
                            'time': [str(t) for t in val_sliced.time_index],
                            'values': val_sliced.values().flatten().tolist()
                        } if val_sliced is not None and len(val_sliced) > 0 else None
                        val_datas.append(val_data)

                        # Build forecast payload (quantiles or deterministic)
                        if quantiles:
                            quantiles_sorted = sorted(set(quantiles))
                            forecast_data = {
                                'time': [str(t) for t in fc.time_index],
                                'quantiles': {str(q): fc.quantile(q).values().flatten().tolist() for q in quantiles_sorted},
                                'has_median': 0.5 in quantiles_sorted
                            }
                            is_symmetric = all((1 - q in quantiles_sorted) for q in quantiles_sorted if q < 0.5)
                            forecast_data['is_symmetric'] = is_symmetric
                            if is_symmetric:
                                pairs = [(q, 1 - q) for q in quantiles_sorted if q < 0.5]
                                forecast_data['pairs'] = [[str(round(p[0],2)), str(round(p[1],2))] for p in sorted(pairs, key=lambda x: x[0])]
                            else:
                                forecast_data['pairs'] = []
                        else:
                            forecast_data = {
                                'time': [str(t) for t in fc.time_index],
                                'values': fc.values().flatten().tolist()
                            }
                        forecasts_data.append(forecast_data)

                        # Compute MAPE per forecast (prioritize computation in multivariate for per-component accuracy)
                        mape_value = None
                        if val_uni is not None:
                            try:
                                point_fc = fc.quantile(0.5)
                            except Exception:
                                point_fc = fc
                            mape_value = _safe_compute_mape(val_uni, point_fc)
                        # Optionally override main forecast with provided mapes (assuming per group)
                        if j == len(all_forecasts) - 1 and mapes is not None:
                            if isinstance(mapes, (list, tuple)) and i < len(mapes):
                                try:
                                    mape_value = float(mapes[i])  # Use group-level mape for all components
                                except Exception:
                                    pass
                            elif not isinstance(mapes, (list, tuple)):
                                try:
                                    mape_value = float(mapes)
                                except Exception:
                                    pass
                        all_mapes.append(mape_value)

                    title = f"Client: {client} | Install: {install}<br>Series: {series_name}"

                    if client not in js_data:
                        js_data[client] = {}
                    if install not in js_data[client]:
                        js_data[client][install] = {}

                    js_data[client][install][series_name] = {
                        'series_list': series_datas,
                        'val_list': val_datas,
                        'forecasts': forecasts_data,
                        'title': title,
                        'series_name': series_name,
                        'quantiles': bool(quantiles),
                        'mapes_list': all_mapes
                    }
            except (KeyError, AttributeError, IndexError):
                print(f"Warning: Skipping series at index {i} due to missing static covariates or data issues.")
                continue

    if not js_data:
        raise ValueError("No valid series data found. Check static covariates.")

    js_data_json = json.dumps(js_data)

    # Responsive sizing: max-width 100%, maintain requested aspect ratio
    plot_width = max(figsize[0], 300)
    plot_height = max(figsize[1], 200)
    aspect_ratio = plot_height / plot_width

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Forecast Plots</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            :root {{
                --plot-w: {plot_width}px;
                --plot-h: {plot_height}px;
            }}
            body {{ font-family: Arial, sans-serif; margin: 16px; }}
            .controls {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }}
            .dropdown {{ margin: 6px 0; padding: 6px; min-width: 200px; }}
            .slider {{ margin: 6px 0; width: 200px; }}
            #plot {{
                width: min(100vw - 32px, var(--plot-w));
                height: calc(min(100vw - 32px, var(--plot-w)) * {aspect_ratio:.6f});
                max-height: var(--plot-h);
            }}
            @media (min-width: {plot_width}px) {{
                #plot {{ width: var(--plot-w); height: var(--plot-h); }}
            }}
        </style>
    </head>
    <body>
        <h1>Forecast Visualization</h1>
        <div class="controls">
            <label for="client-dropdown">Client:</label>
            <select id="client-dropdown" class="dropdown"></select>

            <label for="install-dropdown">Installation:</label>
            <select id="install-dropdown" class="dropdown" disabled></select>

            <label for="series-dropdown">Series:</label>
            <select id="series-dropdown" class="dropdown" disabled></select>

            <label for="forecast-slider">Forecast Index:</label>
            <input type="range" id="forecast-slider" class="slider" min="0" max="0" value="0" step="1" disabled>
            <span id="slider-value">0</span>
        </div>
        <div id="plot"></div>

        <script>
            const data = {js_data_json};

            const clientSelect = document.getElementById('client-dropdown');
            const installSelect = document.getElementById('install-dropdown');
            const seriesSelect = document.getElementById('series-dropdown');

            const sortedClients = Object.keys(data).sort();
            sortedClients.forEach(client => {{
                const option = document.createElement('option');
                option.value = client;
                option.textContent = client;
                clientSelect.appendChild(option);
            }});

            clientSelect.addEventListener('change', function() {{
                const selectedClient = this.value;

                installSelect.innerHTML = '';
                installSelect.disabled = true;

                seriesSelect.innerHTML = '';
                seriesSelect.disabled = true;

                if (selectedClient) {{
                    const installs = Object.keys(data[selectedClient]).sort();
                    installs.forEach(install => {{
                        const option = document.createElement('option');
                        option.value = install;
                        option.textContent = install;
                        installSelect.appendChild(option);
                    }});
                    installSelect.disabled = installs.length === 0;

                    if (installs.length > 0) {{
                        installSelect.value = installs[0];
                        installSelect.dispatchEvent(new Event('change'));
                    }}
                }}
            }});

            installSelect.addEventListener('change', function() {{
                const selectedClient = clientSelect.value;
                const selectedInstall = this.value;

                seriesSelect.innerHTML = '';
                seriesSelect.disabled = true;

                if (selectedClient && selectedInstall) {{
                    const seriesEntries = Object.entries(data[selectedClient][selectedInstall]); // [[series_id, item], ...]
                    // Sort by displayed name, fallback to id
                    seriesEntries.sort((a, b) => {{
                        const nameA = (a[1].series_name || a[0]).toString();
                        const nameB = (b[1].series_name || b[0]).toString();
                        return nameA.localeCompare(nameB, undefined, {{numeric: true, sensitivity: 'base'}});
                    }});

                    seriesEntries.forEach(([seriesId, item]) => {{
                        const option = document.createElement('option');
                        option.value = seriesId; // keep id as value
                        option.textContent = item.series_name || seriesId; // show readable name
                        seriesSelect.appendChild(option);
                    }});

                    seriesSelect.disabled = seriesEntries.length === 0;

                    if (seriesEntries.length > 0) {{
                        seriesSelect.value = seriesEntries[0][0];
                        seriesSelect.dispatchEvent(new Event('change'));
                    }}
                }}
            }});

            seriesSelect.addEventListener('change', function() {{
                setupSlider();
                plotSelected();
            }});

            const slider = document.getElementById('forecast-slider');
            const sliderValue = document.getElementById('slider-value');

            function setupSlider() {{
                const selectedClient = clientSelect.value;
                const selectedInstall = installSelect.value;
                const selectedSeries = seriesSelect.value;
                if (!selectedClient || !selectedInstall || !selectedSeries) return;

                const item = data[selectedClient][selectedInstall][selectedSeries];
                const numForecasts = item.forecasts ? item.forecasts.length : 0;
                if (numForecasts > 0) {{
                    slider.min = 0;
                    slider.max = numForecasts - 1;
                    slider.value = numForecasts - 1;  // Default to last (current/main forecast)
                    slider.disabled = numForecasts <= 1;
                    sliderValue.textContent = slider.value;
                    sliderValue.style.display = numForecasts > 1 ? 'inline' : 'none';
                }} else {{
                    slider.disabled = true;
                    sliderValue.style.display = 'none';
                }}
            }}

            slider.addEventListener('input', function() {{
                sliderValue.textContent = this.value;
                plotSelected();
            }});

            function plotSelected() {{
                const selectedClient = clientSelect.value;
                const selectedInstall = installSelect.value;
                const selectedSeries = seriesSelect.value;

                if (!selectedClient || !selectedInstall || !selectedSeries) return;

                const item = data[selectedClient][selectedInstall][selectedSeries];
                const idx = parseInt(slider.value) || 0;
                const forecast = item.forecasts ? item.forecasts[idx] : null;
                const series = item.series_list ? item.series_list[idx] : null;
                const val = item.val_list ? item.val_list[idx] : null;
                const mape = item.mapes_list ? item.mapes_list[idx] : null;

                if (!forecast) return;  // Safety check

                // ENHANCED: Debug logging
                console.log('Selected Index:', idx);
                console.log('Series Start Time:', series ? series.time[0] : 'None');
                console.log('Val Length:', val ? val.time.length : 'None');
                console.log('Forecast Start Time:', forecast.time[0]);

                const traces = [];

                if (series && series.time.length > 0) {{
                    traces.push({{
                        x: series.time,
                        y: series.values,
                        mode: 'lines',
                        name: '{series_label}',
                        line: {{color: 'blue'}}
                    }});
                }}

                if (val && val.time.length > 0) {{
                    traces.push({{
                        x: val.time,
                        y: val.values,
                        mode: 'lines',
                        name: '{val_label}',
                        line: {{color: 'green'}}
                    }});
                }}

                if (item.quantiles) {{
                    if (forecast.has_median && forecast.quantiles['0.5']) {{
                        traces.push({{
                            x: forecast.time,
                            y: forecast.quantiles['0.5'],
                            mode: 'lines',
                            name: '{forecast_label} Median',
                            line: {{color: 'red'}}
                        }});
                    }}

                    if (forecast.is_symmetric && forecast.pairs.length > 0) {{
                        const colors = [
                            {{ fill: 'rgba(255,0,0,0.1)', line: 'rgba(255,0,0,0.2)' }},
                            {{ fill: 'rgba(255,0,128,0.1)', line: 'rgba(255,0,0,0.3)' }},
                            {{ fill: 'rgba(255,0,255,0.1)', line: 'rgba(255,0,0,0.4)' }}
                        ];
                        forecast.pairs.forEach((pair, index) => {{
                            const color = colors[index % colors.length];
                            const lower = pair[0];
                            const upper = pair[1];
                            const intervalPct = Math.round((parseFloat(upper) - parseFloat(lower)) * 100);
                            const legendLabel = `${{intervalPct}}% Interval`;
                            traces.push({{
                                x: forecast.time,
                                y: forecast.quantiles[upper],
                                mode: 'lines',
                                line: {{color: color.line}},
                                name: `Upper ${{upper}}`,
                                showlegend: false
                            }});
                            traces.push({{
                                x: forecast.time,
                                y: forecast.quantiles[lower],
                                mode: 'lines',
                                fill: 'tonexty',
                                fillcolor: color.fill,
                                line: {{color: color.line}},
                                name: legendLabel,
                                showlegend: true,
                                hovertemplate: `Quantile ${{lower}}<br>Time=%{{x}}<br>Value=%{{y:.4f}}<extra></extra>`
                            }});
                        }});
                    }} else {{
                        Object.keys(forecast.quantiles).forEach(q => {{
                            if (q !== '0.5') {{
                                traces.push({{
                                    x: forecast.time,
                                    y: forecast.quantiles[q],
                                    mode: 'lines',
                                    name: `Quantile ${{q}}`,
                                    line: {{color: 'red', dash: 'dot'}}
                                }});
                            }}
                        }});
                    }}
                }} else {{
                    traces.push({{
                        x: forecast.time,
                        y: forecast.values,
                        mode: 'lines',
                        name: '{forecast_label}',
                        line: {{color: 'red'}}
                    }});
                }}

                const titleSuffix = (mape !== null && mape !== undefined && !Number.isNaN(mape))
                    ? ` — MAPE: ${{mape.toFixed(2)}}%`
                    : '';
                const layout = {{
                    title: item.title + titleSuffix,
                    xaxis: {{title: 'Time'}},
                    yaxis: {{title: 'Value'}},
                    showlegend: true,
                    annotations: (mape !== null && mape !== undefined && !Number.isNaN(mape)) ? [
                        {{
                            text: `MAPE: ${{mape.toFixed(2)}}%`,
                            xref: 'paper', yref: 'paper',
                            x: 0.99, y: 0.01,
                            showarrow: false,
                            font: {{size: 12}},
                            align: 'right',
                            bgcolor: 'rgba(255,255,255,0.6)'
                        }}
                    ] : []
                }};

                const config = {{
                    responsive: true,
                    displaylogo: false,
                    modeBarButtonsToRemove: ['toggleSpikelines']
                }};

                Plotly.newPlot('plot', traces, layout, config);
            }}

            if (sortedClients.length > 0) {{
                clientSelect.value = sortedClients[0];
                clientSelect.dispatchEvent(new Event('change'));
            }} else {{
                document.getElementById('plot').innerHTML = '<p>No data available to plot.</p>';
            }}
        </script>
    </body>
    </html>
    """

    if save_html_path:
        os.makedirs(os.path.dirname(save_html_path), exist_ok=True)
        with open(save_html_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        print(f"HTML saved to {save_html_path}")

    if show:
        try:
            display(HTML(html_template))
        except Exception:
            # Fallback to printing the HTML if not in a notebook
            print(html_template)
    elif not save_html_path:
        print("Nothing to do: No save path provided and show=False.")


# from darts import TimeSeries 
from IPython.display import display, HTML
from collections import defaultdict
import json, os
from typing import List, Optional, Union

def plot_ground_truth_with_dropdowns_plotly(
    series: Union['TimeSeries', List['TimeSeries']],
    figsize: tuple = (1600, 800),
    save_html_path: str = None,
    show: bool = True,
    is_multivariate: Optional[bool] = None,  
    covars_transformer = None, 
    unit_name=None,
    **kwargs
):
    series_list = [series] if not isinstance(series, list) else series

    if covars_transformer is not None:
        series_list = covars_transformer.inverse_transform(series_list)
        series_list = [series_list] if not isinstance(series_list, list) else series_list

    if is_multivariate is None:
        has_series_id = all(
            hasattr(s, 'static_covariates') and s.static_covariates is not None and 'series_id' in s.static_covariates.columns
            for s in series_list
        )
        is_multivariate = not has_series_id

    data_map = defaultdict(lambda: defaultdict(list))
    js_data = {}

    if not is_multivariate: 
        for i, s in enumerate(series_list):
            try:
                client = str(s.static_covariates['client_id'].iloc[0])
                install = str(s.static_covariates['installation_id'].iloc[0])
                series_name = str(s.static_covariates['series_id'].iloc[0])
                
                unit = str(s.static_covariates['unit'].iloc[0]) if 'unit' in s.static_covariates.columns else 'Value' if unit_name is None else unit_name
                display_name = f"{client} | {install} | {series_name}"

                data_map[client][install].append(series_name)
                series_data = {
                    'time': [str(t) for t in s.time_index],
                    'values': s.values().flatten().tolist()
                }

                if client not in js_data: js_data[client] = {}
                if install not in js_data[client]: js_data[client][install] = {}

                js_data[client][install][series_name] = {
                    'series_data': series_data,
                    'series_name': series_name,
                    'display_name': display_name,
                    'unit': unit 
                }
            except (KeyError, AttributeError, IndexError):
                continue

    else: 
        for i, s in enumerate(series_list):
            try:
                client = str(s.static_covariates['client_id'].iloc[0])
                install = str(s.static_covariates['installation_id'].iloc[0])
                
                unit = str(s.static_covariates['unit'].iloc[0]) if 'unit' in s.static_covariates.columns else 'Value'

                components = s.components
                for component in components:
                    series_name = str(component)
                    display_name = f"{client} | {install} | {series_name}"
                    
                    data_map[client][install].append(series_name)
                    s_uni = s[component]
                    series_data = {
                        'time': [str(t) for t in s_uni.time_index],
                        'values': [round(v, 3) for v in s_uni.values().flatten().tolist()]
                    }

                    if client not in js_data: js_data[client] = {}
                    if install not in js_data[client]: js_data[client][install] = {}

                    js_data[client][install][series_name] = {
                        'series_data': series_data,
                        'series_name': series_name,
                        'display_name': display_name,
                        'unit': unit
                    }
            except (KeyError, AttributeError, IndexError):
                continue

    if not js_data:
        raise ValueError("No valid series data found. Check static covariates.")

    js_data_json = json.dumps(js_data)

    plot_width = max(figsize[0], 300)
    plot_height = max(figsize[1], 200)
    aspect_ratio = plot_height / plot_width

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Ground Truth Plots</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            :root {{
                --plot-w: {plot_width}px;
                --plot-h: {plot_height}px;
            }}
            body {{ font-family: Arial, sans-serif; margin: 16px; color: #333; }}
            .controls {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 8px; }}
            .dropdown {{ margin: 6px 0; padding: 6px; min-width: 180px; border: 1px solid #ccc; border-radius: 4px; }}
            
            .pin-btn {{ padding: 6px 12px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; transition: background 0.2s; }}
            .pin-btn:hover {{ background-color: #0056b3; }}
            .pin-btn:disabled {{ background-color: #cccccc; cursor: not-allowed; }}
            
            .filter-group {{ display: flex; align-items: center; gap: 8px; margin-left: auto; padding: 8px 12px; background: #f1f3f5; border-radius: 6px; border: 1px solid #ddd; }}
            .number-input {{ width: 60px; padding: 4px; border: 1px solid #ccc; border-radius: 4px; text-align: center; }}

            .pill-tray {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; min-height: 32px; }}
            .pill {{ display: flex; align-items: center; background: #f8f9fa; border: 1px solid #ddd; border-left: 6px solid #ccc; padding: 4px 8px; border-radius: 4px; font-size: 13px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
            .pill span {{ margin-right: 8px; font-weight: 500; }}
            .pill button {{ background: none; border: none; cursor: pointer; color: #999; font-weight: bold; font-size: 14px; padding: 0 4px; line-height: 1; }}
            .pill button:hover {{ color: #dc3545; }}

            #plot {{
                width: min(100vw - 32px, var(--plot-w));
                height: calc(min(100vw - 32px, var(--plot-w)) * {aspect_ratio:.6f});
                max-height: var(--plot-h);
            }}
            @media (min-width: {plot_width}px) {{
                #plot {{ width: var(--plot-w); height: var(--plot-h); }}
            }}
        </style>
    </head>
    <body>
        <h2>Ground Truth Comparison</h2>
        <div class="controls">
            <div>
                <label for="client-dropdown">Client:</label>
                <select id="client-dropdown" class="dropdown"></select>
            </div>
            <div>
                <label for="install-dropdown">Installation:</label>
                <select id="install-dropdown" class="dropdown" disabled></select>
            </div>
            <div>
                <label for="series-dropdown">Series:</label>
                <select id="series-dropdown" class="dropdown" disabled></select>
            </div>
            <button id="pin-button" class="pin-btn" disabled>Pin to Plot</button>
            
            <div class="filter-group">
                <input type="checkbox" id="iqr-toggle">
                <label for="iqr-toggle" style="font-weight: bold; cursor: pointer;">IQR Outlier Filter</label>
                <label for="iqr-factor">Factor:</label>
                <input type="number" id="iqr-factor" class="number-input" value="3" step="0.1" min="0.1" disabled>
            </div>
        </div>
        
        <div id="pill-tray" class="pill-tray"></div>
        <div id="plot"></div>

        <script>
            const data = {js_data_json};

            const clientSelect = document.getElementById('client-dropdown');
            const installSelect = document.getElementById('install-dropdown');
            const seriesSelect = document.getElementById('series-dropdown');
            const pinButton = document.getElementById('pin-button');
            const pillTray = document.getElementById('pill-tray');
            
            const iqrToggle = document.getElementById('iqr-toggle');
            const iqrFactor = document.getElementById('iqr-factor');

            let pinnedSeries = [];
            const colorPalette = ['#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF'];
            let colorIndex = 0;

            function getSeriesKey(c, i, s) {{ return `${{c}}::${{i}}::${{s}}`; }}

            // Trigger replot when filter settings change
            iqrToggle.addEventListener('change', () => {{
                iqrFactor.disabled = !iqrToggle.checked;
                plotSelected();
            }});
            
            iqrFactor.addEventListener('change', plotSelected);

            function updatePinButtonState() {{
                const c = clientSelect.value;
                const i = installSelect.value;
                const s = seriesSelect.value;
                if (!c || !i || !s) {{
                    pinButton.disabled = true;
                    return;
                }}
                const key = getSeriesKey(c, i, s);
                const isPinned = pinnedSeries.some(p => p.key === key);
                pinButton.disabled = isPinned;
                pinButton.textContent = isPinned ? 'Already Pinned' : 'Pin to Plot';
            }}

            const sortedClients = Object.keys(data).sort();
            sortedClients.forEach(client => {{
                const option = document.createElement('option');
                option.value = client;
                option.textContent = client;
                clientSelect.appendChild(option);
            }});

            clientSelect.addEventListener('change', function() {{
                const selectedClient = this.value;
                installSelect.innerHTML = '';
                installSelect.disabled = true;
                seriesSelect.innerHTML = '';
                seriesSelect.disabled = true;

                if (selectedClient) {{
                    const installs = Object.keys(data[selectedClient]).sort();
                    installs.forEach(install => {{
                        const option = document.createElement('option');
                        option.value = install;
                        option.textContent = install;
                        installSelect.appendChild(option);
                    }});
                    installSelect.disabled = installs.length === 0;

                    if (installs.length > 0) {{
                        installSelect.value = installs[0];
                        installSelect.dispatchEvent(new Event('change'));
                    }}
                }}
            }});

            installSelect.addEventListener('change', function() {{
                const selectedClient = clientSelect.value;
                const selectedInstall = this.value;
                seriesSelect.innerHTML = '';
                seriesSelect.disabled = true;

                if (selectedClient && selectedInstall) {{
                    const seriesEntries = Object.entries(data[selectedClient][selectedInstall]);
                    seriesEntries.sort((a, b) => {{
                        const nameA = (a[1].series_name || a[0]).toString();
                        const nameB = (b[1].series_name || b[0]).toString();
                        return nameA.localeCompare(nameB, undefined, {{numeric: true, sensitivity: 'base'}});
                    }});

                    seriesEntries.forEach(([seriesId, item]) => {{
                        const option = document.createElement('option');
                        option.value = seriesId; 
                        option.textContent = item.series_name || seriesId; 
                        seriesSelect.appendChild(option);
                    }});

                    seriesSelect.disabled = seriesEntries.length === 0;

                    if (seriesEntries.length > 0) {{
                        seriesSelect.value = seriesEntries[0][0];
                        seriesSelect.dispatchEvent(new Event('change'));
                    }}
                }}
            }});

            seriesSelect.addEventListener('change', function() {{
                updatePinButtonState();
                plotSelected();
            }});

            pinButton.addEventListener('click', () => {{
                const c = clientSelect.value;
                const i = installSelect.value;
                const s = seriesSelect.value;
                if (!c || !i || !s) return;
                
                const key = getSeriesKey(c, i, s);
                if (pinnedSeries.some(p => p.key === key)) return;

                const item = data[c][i][s];
                const displayName = item.display_name;
                const color = colorPalette[colorIndex % colorPalette.length];
                colorIndex++;

                pinnedSeries.push({{ key, client: c, install: i, seriesId: s, name: displayName, color }});
                
                renderPills();
                updatePinButtonState();
                plotSelected();
            }});

            function unpinSeries(key) {{
                pinnedSeries = pinnedSeries.filter(p => p.key !== key);
                renderPills();
                updatePinButtonState();
                plotSelected();
            }}

            function renderPills() {{
                pillTray.innerHTML = '';
                pinnedSeries.forEach(p => {{
                    const pill = document.createElement('div');
                    pill.className = 'pill';
                    pill.style.borderLeftColor = p.color;
                    
                    const label = document.createElement('span');
                    label.textContent = p.name;
                    
                    const btn = document.createElement('button');
                    btn.innerHTML = '&times;'; 
                    btn.onclick = () => unpinSeries(p.key);
                    
                    pill.appendChild(label);
                    pill.appendChild(btn);
                    pillTray.appendChild(pill);
                }});
            }}

            // Filtering Function
            function applyIQRFilter(times, values) {{
                const factor = parseFloat(iqrFactor.value) || 1.5;
                const validValues = values.filter(v => v !== null && !isNaN(v));
                if (validValues.length < 4) return {{ x: times, y: values }}; // Not enough data

                const sorted = [...validValues].sort((a, b) => a - b);
                const q1 = sorted[Math.floor(sorted.length * 0.25)];
                const q3 = sorted[Math.floor(sorted.length * 0.75)];
                const iqr = q3 - q1;
                const lowerBound = q1 - (factor * iqr);
                const upperBound = q3 + (factor * iqr);

                const filteredX = [];
                const filteredY = [];

                for (let j = 0; j < values.length; j++) {{
                    const val = values[j];
                    if (val !== null && val >= lowerBound && val <= upperBound) {{
                        filteredX.push(times[j]);
                        filteredY.push(val);
                    }}
                }}
                return {{ x: filteredX, y: filteredY }};
            }}

            function plotSelected() {{
                const selectedClient = clientSelect.value;
                const selectedInstall = installSelect.value;
                const selectedSeries = seriesSelect.value;
                const isFilterActive = iqrToggle.checked;

                const rawTraces = [];

                // Helper to process and push a trace
                function addTrace(itemData, traceName, traceColor) {{
                    let finalX = itemData.series_data.time;
                    let finalY = itemData.series_data.values;

                    if (isFilterActive) {{
                        const filtered = applyIQRFilter(finalX, finalY);
                        finalX = filtered.x;
                        finalY = filtered.y;
                    }}

                    rawTraces.push({{
                        x: finalX,
                        y: finalY,
                        name: traceName,
                        color: traceColor,
                        unit: itemData.unit || 'Value'
                    }});
                }}

                // 1. Add Pinned Series
                pinnedSeries.forEach(p => {{
                    const pItem = data[p.client]?.[p.install]?.[p.seriesId];
                    if (pItem && pItem.series_data) {{
                        addTrace(pItem, p.name, p.color);
                    }}
                }});

                // 2. Add Selected Series (if not already pinned)
                if (selectedClient && selectedInstall && selectedSeries) {{
                    const key = getSeriesKey(selectedClient, selectedInstall, selectedSeries);
                    const isCurrentlyPinned = pinnedSeries.some(p => p.key === key);
                    
                    if (!isCurrentlyPinned) {{
                        const item = data[selectedClient][selectedInstall][selectedSeries];
                        if (item && item.series_data && item.series_data.time.length > 0) {{
                            addTrace(item, item.display_name, 'blue');
                        }}
                    }}
                }}

                const uniqueUnits = [...new Set(rawTraces.map(t => t.unit))];
                const primaryUnit = uniqueUnits.length > 0 ? uniqueUnits[0] : 'Value';
                const secondaryUnit = uniqueUnits.length > 1 ? uniqueUnits[1] : null;

                const traces = rawTraces.map(t => {{
                    const yaxisID = (t.unit === primaryUnit) ? 'y' : 'y2';
                    return {{
                        x: t.x,
                        y: t.y,
                        mode: 'lines', // Leaving out the outlier creates a natural connected line over the gap
                        name: t.name,
                        line: {{color: t.color}},
                        yaxis: yaxisID
                    }};
                }});

                const layout = {{
                    title: 'Interactive Series Comparison',
                    xaxis: {{title: 'Time'}},
                    yaxis: {{title: primaryUnit}},
                    showlegend: true,
                    legend: {{ orientation: 'h', y: -0.2 }},
                    hoverlabel: {{ namelength: -1 }} 
                }};

                if (secondaryUnit) {{
                    layout.yaxis2 = {{
                        title: secondaryUnit,
                        overlaying: 'y',
                        side: 'right',
                        showgrid: false 
                    }};
                }}

                const config = {{
                    responsive: true,
                    displaylogo: false,
                    modeBarButtonsToRemove: ['toggleSpikelines']
                }};

                Plotly.newPlot('plot', traces, layout, config);
            }}

            if (sortedClients.length > 0) {{
                clientSelect.value = sortedClients[0];
                clientSelect.dispatchEvent(new Event('change'));
            }} else {{
                document.getElementById('plot').innerHTML = '<p>No data available to plot.</p>';
            }}
        </script>
    </body>
    </html>
    """

    if save_html_path:
        dir_name = os.path.dirname(save_html_path)
        if dir_name:  
            os.makedirs(dir_name, exist_ok=True)
            
        with open(save_html_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        print(f"HTML saved to {save_html_path}")

    if show:
        try:
            display(HTML(html_template))
        except Exception:
            print("HTML output generated, but could not be displayed inline.")
    elif not save_html_path:
        print("Nothing to do: No save path provided and show=False.")
        
        
def _sort_lists_by_series_id(
        series_list: List[TimeSeries],
        forecast_list: List[TimeSeries],
        val_list: List[Optional[TimeSeries]]
    ) -> tuple[List[TimeSeries], List[TimeSeries], List[Optional[TimeSeries]]]:
    """
    Sorts the input lists by 'series_id' from static_covariates (if available).
    Assumes 'series_id' is a single value per TimeSeries.
    Skips sorting if not found for all series.
    """
    # Check if all series have 'series_id' in static_covariates
    if all(ser.static_covariates is not None and 'series_id' in ser.static_covariates.columns for ser in series_list):
        # Extract series_id (convert to int for numeric sort if possible)
        ids = []
        for ser in series_list:
            sid = ser.static_covariates['series_id'].values[0]
            try:
                ids.append(int(sid))  # Numeric sort (e.g., 1 before 10)
            except ValueError:
                ids.append(sid)  # Fallback to string sort

        # Zip lists with ids, sort by id, unzip
        zipped = list(zip(ids, series_list, forecast_list, val_list))
        zipped_sorted = sorted(zipped, key=lambda x: x[0])
        sorted_ids, sorted_series, sorted_forecast, sorted_val = zip(*zipped_sorted)
        print("Sorted series by 'series_id' for tabbed plotting.")
        return list(sorted_series), list(sorted_forecast), list(sorted_val)
    else:
        print("Warning: 'series_id' not found in static_covariates for all series. Skipping sort (using original order).")
        return series_list, forecast_list, val_list

def _sort_lists_by_series_id_train(
        series_list: List[TimeSeries],
    ) -> tuple[List[TimeSeries], List[TimeSeries], List[Optional[TimeSeries]]]:
    """
    Sorts the input lists by 'series_id' from static_covariates (if available).
    Assumes 'series_id' is a single value per TimeSeries.
    Skips sorting if not found for all series.
    """
    # Check if all series have 'series_id' in static_covariates
    if all(ser.static_covariates is not None and 'series_id' in ser.static_covariates.columns for ser in series_list):
        # Extract series_id (convert to int for numeric sort if possible)
        ids = []
        for ser in series_list:
            sid = ser.static_covariates['series_id'].values[0]
            try:
                ids.append(int(sid))  # Numeric sort (e.g., 1 before 10)
            except ValueError:
                ids.append(sid)  # Fallback to string sort

    sorted_series = sorted(series_list, key=lambda x: ids[series_list.index(x)])
    print("Sorted series by 'series_id'.")
    return sorted_series
    
def _plot_single_component(
    ax: plt.Axes,
    series_comp: TimeSeries,
    forecast_comp: TimeSeries,
    val_series_comp: Optional[TimeSeries] = None,
    context_multiplier: int = 10,
    quantiles: Optional[list] = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
    series_label: str = "Series",
    forecast_label: str = "Forecast",
    title: str = "Probabilistic Forecast"
):
    """
    Internal helper to plot a single component of a time series forecast with full datetime x-ticks.

    Args:
        ax (plt.Axes): Matplotlib Axes object.
        series_comp (TimeSeries): Historical TimeSeries.
        forecast_comp (TimeSeries): Forecast TimeSeries.
        val_series_comp (Optional[TimeSeries]): Optional validation series.
        context_multiplier (int): How many times the forecast horizon to show in history.
        quantiles (list): Quantiles for confidence bands.
        series_label (str): Label for historical data.
        forecast_label (str): Label for forecast median.
        title (str): Plot title.
    """
    import pandas as pd
    import numpy as np

    forecast_horizon = len(forecast_comp)

    # --- 1. Slice historical context ---
    context_length = min(context_multiplier * forecast_horizon, len(series_comp))
    series_slice = series_comp.slice_intersect(forecast_comp)

    if len(series_slice) == 0:
        print("Warning: No overlapping historical data with forecast period.")
        # Still plot forecast even if no history
        all_times = forecast_comp.time_index
    else:
        all_times = np.concatenate([series_slice.time_index, forecast_comp.time_index])
    all_times = np.unique(all_times)  # Sorted unique timestamps

    # Convert to pandas Timestamps for .strftime()
    all_times_dt = pd.to_datetime(all_times)

    # --- 2. Plot historical data ---
    plot_kwargs = {"lw": 1.0}

    if len(series_slice) > 0:
        series_slice.plot(ax=ax, label=series_label, **plot_kwargs)

    # --- 3. Plot probabilistic forecast bands ---
    quantile_colors = ['blue', 'purple', 'orange', 'red']
    n_pairs = len(quantiles) // 2
    for i in range(n_pairs):
        q_low = quantiles[i]
        q_high = quantiles[-(i + 1)]
        label = f"{int(q_low*100)}–{int(q_high*100)}th percentile"
        forecast_comp.plot(
            low_quantile=q_low,
            high_quantile=q_high,
            ax=ax,
            label=label,
            c=quantile_colors[i % len(quantile_colors)]
        )

    # Plot median forecast
    forecast_comp.plot(
        low_quantile=0.5,
        high_quantile=0.5,
        ax=ax,
        label=forecast_label,
        new_plot=False,
        **plot_kwargs
    )

    # --- 4. Customize x-axis: Full datetime on every tick ---
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_xticks(all_times)
    ax.set_xticklabels(
        [t.strftime('%Y-%m-%d %H:%M:%S') for t in all_times_dt],
        rotation=45,
        ha='right',
        fontsize=8
    )

    # Improve layout to prevent label cutoff
    plt.tight_layout()

    # --- 5. Final touches ---
    ax.set_ylabel(series_comp.components[0])
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
def plot_forecast_interactive(
    series: Union[TimeSeries, List[TimeSeries]],
    forecast: Union[TimeSeries, List[TimeSeries]],
    threshold_upper: Optional[float] = None,
    threshold_lower: Optional[float] = None,
    val_series: Optional[Union[TimeSeries, List[TimeSeries]]] = None
) -> None:
    """
    Plots probabilistic forecasts with interactive buttons for multivariate or multi-series (long format) data.

    Args:
        series (Union[TimeSeries, List[TimeSeries]]): The historical time series data or list.
        forecast (Union[TimeSeries, List[TimeSeries]]): The probabilistic forecast or list to plot.
        threshold_upper (float): The upper threshold to display on the plot.
        threshold_lower (float): The lower threshold to display on the plot.
        val_series (Optional[Union[TimeSeries, List[TimeSeries]]]): Optional validation series or list.
    """
    # Convert to lists for consistency (handle single TimeSeries as list of 1)
    if isinstance(series, TimeSeries):
        series_list = [series]
    else:
        series_list = series

    if isinstance(forecast, TimeSeries):
        forecast_list = [forecast]
    else:
        forecast_list = forecast

    if val_series is not None:
        if isinstance(val_series, TimeSeries):
            val_list = [val_series]
        else:
            val_list = val_series
    else:
        val_list = [None] * len(series_list)

    # Validation: Ensure lists match in length
    if len(series_list) != len(forecast_list) or len(series_list) != len(val_list):
        raise ValueError("series, forecast, and val_series must have the same length if lists.")

    # Sort the lists by 'series_id' if available in static_covariates
    series_list, forecast_list, val_list = _sort_lists_by_series_id(series_list, forecast_list, val_list)

    # Determine if multi-series or multivariate
    is_multi_series = len(series_list) > 1
    is_multivariate = any(ser.width > 1 for ser in series_list)

    fig = go.Figure(layout=dict(width=1200, height=1000))

    # Prepare traces: Handle lists as groups
    trace_groups = []  # List of (start_idx, end_idx) for each series/group's traces
    buttons = []
    group_names = []

    for group_idx, ser in enumerate(series_list):
        # Get components (multivariate) or treat as single (univariate)
        components = ser.columns if ser.width > 1 else [f"Series {group_idx}"]  # Fallback name

        # Get group name for button (use series_id if available)
        group_name = f"Series {group_idx}"
        if ser.static_covariates is not None and 'series_id' in ser.static_covariates.columns:
            series_id = ser.static_covariates['series_id'].values[0]
            group_name = f"Series ID: {series_id}"
        group_names.append(group_name)

        group_start_trace = len(fig.data)  # Start of this group's traces

        for comp_idx, comp_name in enumerate(components):
            comp_series = ser[comp_name] if ser.width > 1 else ser
            comp_forecast = forecast_list[group_idx][comp_name] if ser.width > 1 else forecast_list[group_idx]
            comp_val = val_list[group_idx][comp_name] if val_list[group_idx] and ser.width > 1 else val_list[group_idx]

            # Historical and validation data
            fig.add_trace(go.Scatter(
                x=comp_series.time_index, 
                y=comp_series.values().flatten(),  # Flatten for univariate
                mode='lines', 
                name='Historical Data', 
                line=dict(color='black'), 
                visible=(group_idx == 0 and comp_idx == 0)  # Only first visible
            ))
            
            if comp_val:
                fig.add_trace(go.Scatter(
                    x=comp_val.time_index, 
                    y=comp_val.values().flatten(), 
                    mode='lines', 
                    name='Validation Data', 
                    line=dict(color='darkorange', dash='dash'), 
                    visible=(group_idx == 0 and comp_idx == 0)
                ))
            
            # Median forecast
            median_forecast = comp_forecast.quantile(0.5)
            fig.add_trace(go.Scatter(
                x=comp_forecast.time_index, 
                y=median_forecast.values().flatten(), 
                mode='lines', 
                name='Median Forecast', 
                line=dict(color='blue'), 
                visible=(group_idx == 0 and comp_idx == 0)
            ))
            
            # Confidence intervals
            for q_low, q_high, color_alpha in [(0.05, 0.95, 0.15), (0.25, 0.75, 0.3)]:
                low_quantile = comp_forecast.quantile(q_low)
                high_quantile = comp_forecast.quantile(q_high)
                
                fig.add_trace(go.Scatter(
                    x=comp_forecast.time_index, 
                    y=high_quantile.values().flatten(), 
                    mode='lines', 
                    line=dict(width=0), 
                    showlegend=False, 
                    visible=(group_idx == 0 and comp_idx == 0)
                ))
                fig.add_trace(go.Scatter(
                    x=comp_forecast.time_index, 
                    y=low_quantile.values().flatten(), 
                    mode='lines', 
                    line=dict(width=0), 
                    fill='tonexty', 
                    fillcolor=f'rgba(0, 0, 255, {color_alpha})', 
                    name=f'{int(round((q_high - q_low) * 100))}% Confidence Interval', 
                    visible=(group_idx == 0 and comp_idx == 0)
                ))

        group_end_trace = len(fig.data)  # End of this group's traces
        trace_groups.append((group_start_trace, group_end_trace))

    # Create buttons for switching groups (series or multivariate components)
    if is_multi_series or is_multivariate:
        buttons = []
        for group_idx, group_name in enumerate(group_names):
            visibility_mask = [False] * len(fig.data)
            start, end = trace_groups[group_idx]
            visibility_mask[start:end] = [True] * (end - start)
            button = dict(
                label=group_name, 
                method='update', 
                args=[
                    {'visible': visibility_mask}, 
                    {'title': f'Probabilistic Forecast for {group_name}'}
                ]
            )
            buttons.append(button)
        
        fig.update_layout(
            updatemenus=[dict(
                type="buttons", 
                direction="right", 
                active=0, 
                x=0.5, 
                xanchor="center", 
                y=1.15, 
                yanchor="top", 
                buttons=buttons
            )], 
            title_text=f"Probabilistic Forecast for {group_names[0]}"
        )
    else:
        fig.update_layout(title_text=f"Probabilistic Forecast for {group_names[0]}")
    
    if threshold_upper:
        # Threshold lines (global, always visible)
        fig.add_hline(
            y=threshold_upper, 
            line_dash="dot", 
            line_color="red", 
            annotation_text=f"Upper Threshold ({threshold_upper})"
        )
    if threshold_lower:
        fig.add_hline(
            y=threshold_lower, 
            line_dash="dot", 
            line_color="red", 
            annotation_text=f"Lower Threshold ({threshold_lower})"
        )

    fig.show()

    
def split_series(series, n_forecast, input_chunk_length, output_chunk_length,val_size=None):
    """
    Splits a TimeSeries into training and validation sets based on the forecast horizon.

    ebnf

    Kopieren
    Parameters:
    series (TimeSeries): The TimeSeries to split.
    n_forecast (int): The forecast horizon.
    input_chunk_length (int): The length of the input sequence fed to the model.
    output_chunk_length (int): The length of the output sequence predicted by the model

    Returns:
    tuple: (train_series, val_series) where each is a TimeSeries.
    """
    if isinstance(series, pd.DataFrame):
        # Convert DataFrame to Darts TimeSeries
        if 'datetime' in series.columns:
            series = TimeSeries.from_dataframe(series, time_col='datetime', value_cols=series.columns.tolist(), freq='1h')
        elif isinstance(series.index, pd.DatetimeIndex):
            series = TimeSeries.from_dataframe(series, value_cols=series.columns.tolist(), freq='1h')
        else:
            raise ValueError("DataFrame must have a 'datetime' column or a DateTimeIndex.")
        
    elif isinstance(series, pd.Series):
        series = TimeSeries.from_series(series, freq='1h')

    if val_size is not None:
        assert isinstance(val_size, float) and val_size < 0, "val_size must be a float < 0 to indicate the proportion of the series to use for validation."
        split_point = len(series) - val_size*len(series) # Calculate split point based on proportion
    else:
        split_point = len(series) - max(n_forecast, input_chunk_length + output_chunk_length)
    if split_point < 1:
            raise ValueError("Series is too short to create a validation set with the given parameters.")
    train_series, val_series = series.split_before(split_point)
    print(f"Training series length: {len(series)}, Validation series length: {len(val_series)}")

    return train_series, val_series



def split_series_long_format(
    series: Union[TimeSeries, List[TimeSeries], pd.DataFrame, pd.Series],
    n_forecast: int,
    input_chunk_length: int,
    output_chunk_length: int,
    val_size: Optional[float] = None
) -> tuple[List[TimeSeries], List[TimeSeries]]:
    """
    Splits TimeSeries (or list of them) into training and validation sets based on the forecast horizon.
    
    Parameters:
    series (Union[TimeSeries, List[TimeSeries], pd.DataFrame, pd.Series]): The TimeSeries or list to split.
    n_forecast (int): The forecast horizon.
    input_chunk_length (int): The length of the input sequence fed to the model.
    output_chunk_length (int): The length of the output sequence predicted by the model.
    val_size (float, optional): Proportion of the series to use for validation (between 0 and 1). If None, uses max(n_forecast, input_chunk_length + output_chunk_length).

    Returns:
    tuple: (train_series_list, val_series_list) where each is a list of TimeSeries.
    """
    # Convert to list of TimeSeries for consistency
    if isinstance(series, TimeSeries):
        series_list = [series]
    elif isinstance(series, list):
        series_list = series
    elif isinstance(series, pd.DataFrame):
        time_col = 'datetime' if 'datetime' in series.columns else None
        if time_col or isinstance(series.index, pd.DatetimeIndex):
            if not time_col and isinstance(series.index, pd.DatetimeIndex):
                series = series.reset_index().rename(columns={'index': 'datetime'})
                time_col = 'datetime'
            series_list = TimeSeries.from_group_dataframe(
                series, time_col=time_col, value_cols=series.columns.drop(time_col).tolist(), freq='1h'
            )
        else:
            raise ValueError("DataFrame must have a 'datetime' column or a DateTimeIndex.")
    elif isinstance(series, pd.Series):
        series_list = [TimeSeries.from_series(series, freq='1h')]
    else:
        raise ValueError("Input must be TimeSeries, list of TimeSeries, DataFrame, or Series.")

    
    train_list = []
    val_list = []
    split_points = []
    lengths=[]
    train_lengths=[]
    val_lenghts=[]
    for ser in series_list:
        if val_size is not None:
            assert isinstance(val_size, float) and val_size < 1, "val_size must be a float < 1 to indicate the proportion of the series to use for validation."
            split_point = int((1.0 - val_size) * len(ser)) # Calculate split point based on proportion
        else:
            split_point = len(ser) - max(n_forecast, input_chunk_length + output_chunk_length)
        split_points.append(split_point)
        #print('split point:', split_point, 'len ser:', len(ser))
        train_ser, val_ser = ser.split_before(split_point)
        train_list.append(train_ser)
        val_list.append(val_ser)
        lengths.append(len(ser))
        train_lengths.append(len(train_ser))
        val_lenghts.append(len(val_ser))
    
    print(f'differt  lengths: {np.unique(lengths)}, train lengths: {np.unique(train_lengths)}, val lengths: {np.unique(val_lenghts)}, split points {split_points}')
    print(f' Val proportions: {np.unique([round(val_lenghts[i]/lengths[i],2) for i in range(len(lengths))])}')
    train_series = train_list if len(train_list) > 1 else train_list[0]
    val_series = val_list if len(val_list) > 1 else val_list[0]
    return train_series, val_series

    


def generate_backtest_starts(
    series: TimeSeries,
    forecast_horizon: int,
    min_train_series_length: int,
    start: Union[int, float, pd.Timestamp, None] = None,  # NEW PARAMETER
    n_end_strides: int = 0,
    n_spaced_strides: int = 0,
    plot: bool = False,
) -> List[int]:
    """
    Generates a list of start points for Darts backtesting based on end and spaced strides.
    
    Parameters
    ----------
    start : int, float, pd.Timestamp or None, default None
        Controls where the spaced strides begin. Can be:
        - int: index position in the series
        - float: fraction of series length (e.g., 0.5 for halfway point)
        - pd.Timestamp: specific datetime
        - None: uses min_train_series_length as starting point
    """
    len_series = len(series)
    
    # Handle start parameter
    if start is None:
        # Default behavior: use min_train_series_length
        start_index = min_train_series_length
    elif isinstance(start, float) and 0 < start < 1:
        # Convert fraction to index
        start_index = int(len_series * start)
    elif isinstance(start, float) and start >= 1:
        # Treat as index
        start_index = int(start)
    elif isinstance(start, int):
        start_index = start
    elif isinstance(start, pd.Timestamp):
        # Convert timestamp to index
        try:
            start_index = series.time_index.get_loc(start)
        except KeyError:
            # If exact timestamp not found, find nearest
            start_index = series.time_index.get_indexer([start], method='nearest')[0]
    else:
        raise ValueError(f"Invalid start parameter type: {type(start)}. Must be int, float, pd.Timestamp, or None.")
    
    # Ensure start_index meets minimum requirements
    min_required_start = min_train_series_length
    if start_index < min_required_start:
        print(f"Warning: Start index {start_index} adjusted to {min_required_start} to ensure minimum training length.")
        start_index = min_required_start
    
    # Ensure start_index allows for at least one forecast
    max_allowed_start = len_series - forecast_horizon
    if start_index > max_allowed_start:
        raise ValueError(
            f"Start index {start_index} is too late. Latest possible start is {max_allowed_start} "
            f"to allow for forecast horizon of {forecast_horizon}."
        )
    
    # --- 1. Calculate End Strides ---
    end_starts = []
    if n_end_strides > 0:
        end_starts = [
            len_series - i * forecast_horizon for i in range(1, n_end_strides + 1)
        ]

    # --- 2. Calculate Spaced Strides ---
    spaced_starts = []
    if n_spaced_strides > 0:
        if end_starts:
            upper_bound = min(end_starts)
        else:
            upper_bound = len_series - forecast_horizon

        if upper_bound <= start_index:
            raise ValueError(
                f"Cannot generate spaced strides. The available window (from {start_index} to {upper_bound}) "
                f"is too small or invalid."
            )

        if n_spaced_strides == 1:
            spaced_starts = [int((start_index + upper_bound) / 2)]
        else:
            spaced_starts = np.linspace(
                start_index, upper_bound, n_spaced_strides + 1, dtype=int
            )[:-1]

    # --- 3. Combine, filter, and sort ---
    combined_starts = set(list(spaced_starts) + end_starts)
    valid_starts = []
    for start_point in sorted(list(combined_starts)):
        # Check minimum training length requirement
        if start_point < min_train_series_length:
            continue
        # Check that we don't exceed series bounds
        if start_point + forecast_horizon > len_series:
            continue
        valid_starts.append(start_point)

    if not valid_starts and (n_end_strides > 0 or n_spaced_strides > 0):
        raise ValueError(
            "Could not generate any valid backtest start points with the given parameters."
        )

    # --- 4. Optional Visualization ---
    if plot:
        plt.figure(figsize=(12, 6))
        # For multivariate, plot the first component for visualization
        series.univariate_component(0).plot(label=f"Time Series (Component 0)")
        
        # Add vertical line showing the custom start point
        if start is not None:
            plt.axvline(x=series.time_index[start_index], color='blue', linestyle='-', 
                       linewidth=2, label=f'Custom Start (index {start_index})', alpha=0.7)
        
        spaced_set = set(spaced_starts)
        end_set = set(end_starts)
        has_labeled_spaced = not spaced_set
        has_labeled_end = not end_set
        
        for start_pt in valid_starts:
            is_spaced = start_pt in spaced_set
            color = 'green' if is_spaced else 'red'
            label = None
            if is_spaced and not has_labeled_spaced:
                label = 'Spaced Start Points'
                has_labeled_spaced = True
            elif not is_spaced and not has_labeled_end:
                label = 'End Start Points'
                has_labeled_end = True
            plt.axvline(x=series.time_index[start_pt], color=color, linestyle='--', label=label)

        plt.title("Visualization of Backtest Start Points")
        plt.legend()
        plt.show()

    return valid_starts

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Only 1 TimeSeries (lists) were provided which is lower than the number of series (n=95) used to fit Scaler. This can result in a mismatch between the series and the underlying transformers"
)
from darts.metrics import rmse, mape, mae, mase, rmsse
from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import RobustScaler
import numpy as np
import json
from tqdm.auto import tqdm
from typing import List, Union, Dict, Optional, Any

def validate_model_simplified(
    model, 
    series: Union[TimeSeries, List[TimeSeries]],
    start: float,
    n_forecast: int,
    burst_stride: int = None,
    metrics: List[str] = ['mae', 'rmse', 'rmsse', 'mape', 'nll'], 
    num_samples: int = 100,
    coverage_quantiles: List[float] = [0.90, 0.50, 0.20],
    future_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
    past_covariates: Optional[Union[TimeSeries, List[TimeSeries]]] = None,
    plot=True,
    save_html_path: Optional[str] = None,
    covars_mapper=None,
    use_scaling=True,
    scale_globally=False,
) -> Dict[str, float]:

    # --- 1. Setup ---
    # Map string names to Darts metric functions
    supported_metrics = {
        'rmse': rmse, 'mae': mae, 'mase': mase, 'rmsse': rmsse,
        'mape': safe_mape_metric,  # Assuming these are available in your scope
        'nll': nll_from_samples_metric 
    }
    
    # Validation
    unknown = [m for m in metrics if m not in supported_metrics]
    if unknown: raise ValueError(f"Unsupported metrics: {unknown}")

    # Separate metrics handled by Darts vs. Custom (NLL)
    darts_metric_names = [m for m in metrics if m != 'nll']
    darts_funcs = [supported_metrics[m] for m in darts_metric_names]
    
    series_list = series if isinstance(series, list) else [series]
    burst_stride = burst_stride if burst_stride is not None else n_forecast
    
    # --- 2. Scaling ---
    print(f"Starting validation on {len(series_list)} time series...")
    
    # (Kept your logic, but cleaner implementation would use a Pipeline)
    scalers = []
    if use_scaling:
        if scale_globally:
            global_scaler = Scaler(RobustScaler())
            # Simplified split logic for fitting
            if isinstance(start, float) and 0 < start < 1.0:
                 # Assuming split logic exists or we fit on the whole provided history
                 # Ideally: global_scaler.fit(series_list) if we are careful about leakage
                 # For safety matching your logic:
                 train_series = [s.split_before(start)[0] for s in series_list]
                 global_scaler.fit(train_series)
            else:
                global_scaler.fit(series_list)
            series_list_scaled = global_scaler.transform(series_list)
            scalers = [global_scaler] * len(series_list)
        else:
            series_list_scaled = []
            for s in series_list:
                sc = Scaler()
                if isinstance(start, float) and 0 < start < 1.0:
                    sc.fit(s.split_before(start)[0])
                else:
                    sc.fit(s)
                series_list_scaled.append(sc.transform(s))
                scalers.append(sc)
    else:
        series_list_scaled = series_list
        scalers = [None] * len(series_list)

    # --- 3. Main Validation Loop ---
    all_results = {}
    
    # Storage for aggregation
    agg_metrics = {name: [] for name in metrics} # Store raw values per series
    agg_dropoff = {f'{name}_dropoff': [] for name in metrics}
    prediction_counts = []
    
    # For plotting later
    history_inv_forecasts = [] 
    history_mapes = []

    for i, s_scaled in enumerate(tqdm(series_list_scaled, desc="Validating")):
        s_original = series_list[i]
        scaler = scalers[i]
        
        # Metadata Setup
        if not hasattr(s_original, 'static_covariates') or s_original.static_covariates is None:
            continue
        ident = "_".join(s_original.static_covariates.values.flatten().astype(str))
        
        # Handle Covariates
        fc = future_covariates[i] if isinstance(future_covariates, list) else future_covariates
        pc = past_covariates[i] if isinstance(past_covariates, list) else past_covariates

        # Determine Start
        min_required_hist = model.input_chunk_length + model.output_chunk_length
        start_idx = int(len(s_scaled) * start)
        if start_idx < min_required_hist:
            start_idx = min_required_hist
            
        if start_idx + n_forecast > len(s_scaled): continue

        try:
            # A. Generate Forecasts (Scaled)
            # This is the heavy lifting - done once
            hist_fc = model.historical_forecasts(
                series=s_scaled,
                start=start_idx,
                forecast_horizon=n_forecast,
                stride=burst_stride,
                retrain=False,
                last_points_only=False,
                verbose=False,
                num_samples=num_samples,
                future_covariates=fc if model.supports_future_covariates else None,
                past_covariates=pc if model.supports_past_covariates else None,
                predict_likelihood_parameters=False
            )
            prediction_counts.append(len(hist_fc))

            # B. Inverse Transform (Forecasts & Target)
            if use_scaling and scaler:
                inv_hist_fc = scaler.inverse_transform(hist_fc)
                # Note: We need the full inverse target to compare against
                inv_target = s_original # Already unscaled
            else:
                inv_hist_fc = hist_fc
                inv_target = s_original

            history_inv_forecasts.append(inv_hist_fc)

            # C. Calculate Standard Metrics (Vectorized via Backtest)
            # OPTIMIZATION: Call backtest ONLY ONCE with reduction=None
            # When historical_forecasts is passed, backtest just acts as a metric calculator
            
            indv_scores = np.array([])
            if darts_funcs:
                indv_scores = model.backtest(
                    series=inv_target,
                    historical_forecasts=inv_hist_fc,
                    metric=darts_funcs,
                    reduction=None, # Return array of shape (num_forecasts, num_metrics)
                    verbose=False
                )
            
            # Extract standard metrics
            series_res = {}
            burst_mapes = [] # Store for plotting if needed
            
            for idx, m_name in enumerate(darts_metric_names):
                scores = indv_scores[:, idx]
                mean_score = np.mean(scores)
                
                series_res[m_name] = mean_score
                agg_metrics[m_name].append(scores) # Keep full array for burst averaging
                
                # Calculate Dropoff
                diffs = np.diff(scores)
                agg_dropoff[f'{m_name}_dropoff'].append(np.mean(diffs) if len(diffs) > 0 else 0.0)
                
                # Capture MAPE specifically for plotting logic
                if m_name == 'mape':
                    burst_mapes = scores.tolist()
            
            history_mapes.append(burst_mapes)

            # D. Calculate NLL (Custom Loop)
            if 'nll' in metrics:
                # Custom loop is unavoidable for custom metric on samples
                nll_scores = [nll_from_samples_metric(inv_target, fc) for fc in inv_hist_fc]
                nll_mean = np.mean(nll_scores)
                
                series_res['nll'] = nll_mean
                agg_metrics['nll'].append(nll_scores)
                
                diffs = np.diff(nll_scores)
                agg_dropoff['nll_dropoff'].append(np.mean(diffs) if len(diffs) > 0 else 0.0)

            # E. Coverage Metrics
            cov_res = _calculate_coverage_metrics(inv_hist_fc, inv_target, coverage_quantiles)
            series_res.update(cov_res)
            
            all_results[ident] = series_res

        except Exception as e:
            print(f"Error validating {ident}: {e}")
            continue

    if not all_results: raise ValueError("Validation failed for all series.")

    # --- 4. Final Aggregation ---
    final_metrics = {}
    
    # Averages
    for k in metrics:
        # Flatten all scores from all series to get a global average? 
        # Or average of averages? Your code did average of series averages.
        # Let's stick to your logic:
        series_means = [res[k] for res in all_results.values() if k in res]
        final_metrics[k] = np.mean(series_means)
        
    # Dropoffs
    for k, v in agg_dropoff.items():
        final_metrics[k] = np.mean(v) if v else 0.0
        
    final_metrics['avg_forecast_bursts'] = np.mean(prediction_counts)

    # Burst Metrics (Averaging across the 'time' axis of forecasts)
    burst_metrics = {}
    for m_name in metrics:
        # agg_metrics[m_name] is a list of arrays (one array per series)
        # We need to average the 1st element of all arrays, 2nd element, etc.
        arrays = agg_metrics[m_name]
        if not arrays: continue
        
        max_len = max(len(a) for a in arrays)
        means = []
        for t in range(max_len):
            vals = [a[t] for a in arrays if len(a) > t]
            means.append(float(np.mean(vals)))
        burst_metrics[m_name] = means
        
    final_metrics['burst_metrics'] = burst_metrics

    print("\n--- Final Metrics ---")
    print(json.dumps({k:v for k,v in final_metrics.items() if k!='burst_metrics'}, indent=2, default=str))

    # --- 5. Plotting (Optional) ---
    if plot or save_html_path:
        # One-shot prediction for visual check
        valid_indices = []
        pred_input = []
        val_targets = []
        
        # Calculate minimum required length: History needed + Future to predict
        min_len_for_plot = model.input_chunk_length + n_forecast
        
        # 1. Identify Valid Indices
        # We only keep indices where the series is long enough
        valid_indices = [
            i for i, s in enumerate(series_list_scaled) 
            if len(s) >= min_len_for_plot
        ]
        
        if not valid_indices:
            print("Warning: No series were long enough for the visualization split.")
        else:
            # 2. Filter Targets based on valid indices
            pred_input = []
            val_targets = []
            for i in valid_indices:
                s = series_list_scaled[i]
                # split_before returns (training_part, validation_part)
                train, val = s.split_before(len(s) - n_forecast)
                pred_input.append(train)
                val_targets.append(val)

            # 3. Filter Covariates using the SAME valid_indices
            # This ensures len(covariates) == len(pred_input)
            fc_filtered = None
            if model.supports_future_covariates and future_covariates:
                # Handle list vs single TimeSeries input
                if isinstance(future_covariates, list):
                    fc_filtered = [future_covariates[i] for i in valid_indices]
                else:
                    fc_filtered = future_covariates # Broadcast single covariate
            
            pc_filtered = None
            if model.supports_past_covariates and past_covariates:
                if isinstance(past_covariates, list):
                    pc_filtered = [past_covariates[i] for i in valid_indices]
                else:
                    pc_filtered = past_covariates

            # 4. Predict using filtered lists
            preds_scaled = model.predict(
                n=n_forecast, 
                series=pred_input, 
                num_samples=num_samples,
                future_covariates=fc_filtered,
                past_covariates=pc_filtered
            )
        
        # Inverse transform for plot
        preds = []
        vals = []
        originals = []
        
        for i, (p, v, s) in enumerate(zip(preds_scaled, val_targets, series_list_scaled)):
            scaler = scalers[i]
            if use_scaling and scaler:
                preds.append(scaler.inverse_transform(p))
                vals.append(scaler.inverse_transform(v))
                originals.append(scaler.inverse_transform(s))
            else:
                preds.append(p)
                vals.append(v)
                originals.append(s)

        # Assuming helper functions are available
        mapes_plot = [float(safe_mape_metric(v, p)) for v, p in zip(vals, preds)]
        
        if save_html_path:
            # Save logic here (same as your original)
            pass

    return final_metrics

from collections.abc import Iterable
from typing import Union, Sequence
def nll_params_metric(actual_series: TimeSeries, param_forecast: TimeSeries) -> float:
    """
    Robust NLL calculation for Gaussian parameters (Mu, Sigma).
    Handles overflows, NaNs, and infinite values.
    """
    # 1. Align series
    actual_common = actual_series.slice_intersect(param_forecast)
    param_common = param_forecast.slice_intersect(actual_series)
    
    y_true = actual_common.values().flatten()
    params = param_common.values()
    
    # 2. Extract parameters
    mu = params[:, 0]
    sigma = params[:, 1]

    # 3. SAFETY: Replace NaNs or Infs in model output with safe fallbacks
    # If model output is broken, we penalize it heavily but don't crash
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e10, neginf=-1e10)
    sigma = np.nan_to_num(sigma, nan=1.0, posinf=1e10, neginf=1e-6)

    # 4. SAFETY: Clip Sigma
    # Prevent div/0 (too small) and overflow (too big)
    sigma = np.clip(sigma, 1e-6, 1e10)
    
    # 5. SAFETY: Clip Residuals
    # If (y - mu) is > 1e15, squaring it will overflow float64 (approx 1.8e308)
    # We clip the residual to prevent the square term from exploding
    residual = y_true - mu
    residual = np.clip(residual, -1e15, 1e15) 

    # 6. Calculate NLL
    # Formula: 0.5 * log(2*pi) + log(sigma) + (resid^2) / (2*sigma^2)
    likelihood_term = 0.5 * np.log(2 * np.pi) + np.log(sigma)
    squared_error_term = (residual**2) / (2 * sigma**2)
    
    nll = likelihood_term + squared_error_term
    
    # Final check for any remaining trash values
    nll = np.nan_to_num(nll, nan=1e10, posinf=1e10, neginf=-1e10)
    
    return float(np.mean(nll))

def nll_from_samples_metric(actual_series: TimeSeries, sample_forecast: TimeSeries) -> float:
    """
    Robust NLL calculation deriving parameters from samples.
    """
    actual_common = actual_series.slice_intersect(sample_forecast)
    pred_common = sample_forecast.slice_intersect(actual_series)
    
    y_true = actual_common.values().flatten()
    
    # Extract stats
    if pred_common.n_samples < 2:
        mu = pred_common.values().flatten()
        sigma = np.full_like(mu, 1e-2) 
    else:
        mu = pred_common.mean().values().flatten()
        sigma = pred_common.std().values().flatten()
    
    # --- SAFETY BLOCK ---
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e10, neginf=-1e10)
    sigma = np.nan_to_num(sigma, nan=1.0, posinf=1e10, neginf=1e-6)
    
    sigma = np.clip(sigma, 1e-6, 1e10)
    
    residual = y_true - mu
    residual = np.clip(residual, -1e15, 1e15)
    # --------------------

    nll = 0.5 * np.log(2 * np.pi) + np.log(sigma) + ((residual)**2) / (2 * sigma**2)
    
    # Replace any resulting NaNs (e.g. from log(0) if clip failed)
    nll = np.nan_to_num(nll, nan=1e10, posinf=1e10, neginf=-1e10)

    return float(np.mean(nll))

import numpy as np
from typing import Union, Sequence, Iterable
from darts import TimeSeries

def safe_mape_metric(
    actual: Union[TimeSeries, Sequence[TimeSeries], Iterable],
    pred:   Union[TimeSeries, Sequence[TimeSeries], Iterable],
    use_median_forecast: bool = True,
) -> Union[float, np.ndarray]:
    """
    Robust MAPE metric compatible with Darts and strictly matching TimesFM evaluation logic.
    
    Logic:
    1. Handles iterables (vectorized return) for Darts backtesting.
    2. Aligns timestamps automatically using intersection.
    3. Reduces probabilistic forecasts to median (deterministic).
    4. MATH PARITY: Treats actuals=0 as NaN (ignores them), same as TimesFM evaluation.
       Formula: np.nanmean( |(y - yhat) / y| ) * 100
    """

    # --- PART 1: Darts Infrastructure (Crucial for backtest compatibility) ---
    
    # Case 1: Handle Iterables/Generators (Recursive call)
    if isinstance(actual, Iterable) and not isinstance(actual, TimeSeries):
        return np.array([
            safe_mape_metric(a, p, use_median_forecast=use_median_forecast)
            for a, p in zip(actual, pred)
        ], dtype=float)

    # Case 2: Validate Input Types
    if not isinstance(actual, TimeSeries) or not isinstance(pred, TimeSeries):
        raise TypeError("safe_mape_metric expects TimeSeries or iterables of TimeSeries.")

    # --- PART 2: Alignment & Probabilistic Handling ---
    
    # Align to common time index (Robustness)
    actual_i = actual.slice_intersect(pred)
    pred_i = pred.slice_intersect(actual)

    # Handle Probabilistic Forecasts (Collapse to Median)
    if getattr(pred_i, "n_samples", 1) > 1:
        # 0.5 quantile = Median (Robust central tendency)
        pred_i = pred_i.quantile(0.5) if use_median_forecast else pred_i.mean()

    # Extract NumPy arrays (Float64 for precision)
    y_true = actual_i.values(copy=False).flatten().astype(float)
    y_pred = pred_i.values(copy=False).flatten().astype(float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

    # --- PART 3: Calculation (TimesFM Parity) ---
    
    # LOGIC: Matches "merged_df['value'].replace(0, np.nan)" from TimesFM code.
    # We set 0s to NaN to exclude them from the mean calculation completely.
    
    # 1. Create a safe copy of actuals
    y_true_adj = y_true.copy()
    
    # 2. Set zeros to NaN (Using isclose for float safety)
    zeros_mask = np.isclose(y_true_adj, 0, atol=1e-9)
    y_true_adj[zeros_mask] = np.nan
    
    # 3. Calculate Absolute Percentage Error
    # This will result in NaNs where y_true was 0
    pct_error = np.abs((y_true_adj - y_pred) / y_true_adj)
    
    # 4. Take Mean ignoring NaNs (replicating pandas .mean behavior)
    if np.all(np.isnan(pct_error)):
        return np.nan
        
    m = np.nanmean(pct_error)
    
    # Return Percentage (0.05 -> 5.0%)
    return float(m * 100.0)


import sys
def _calculate_coverage_metrics(
    probabilistic_forecasts: List[TimeSeries],
    actual_series: TimeSeries,
    coverage_quantiles: List[float]
) -> Dict[str, float]:
    """
    Calculates the empirical coverage for different confidence intervals.
    """
    coverage_results = {}
    points_in_interval = {q: 0 for q in coverage_quantiles}
    total_points = {q: 0 for q in coverage_quantiles}

    forecast_coverage ={q: [] for q in coverage_quantiles}
    for forecast_burst in probabilistic_forecasts:
        actual_values = actual_series.slice_intersect(forecast_burst)
        
        if len(actual_values) == 0:
            continue

        for q in coverage_quantiles:
            alpha = (1 - q) / 2
            lower_bound = forecast_burst.quantile(alpha)
            upper_bound = forecast_burst.quantile(1 - alpha)
            
            # --- THE CORRECTED FIX IS HERE ---
            # Get the component name from the actual series (the target name)
            target_component_name = actual_values.components[0]
            
            # Rename the component of the bounds to match the actual_values series
            lower_bound_renamed = lower_bound.with_columns_renamed(
                lower_bound.components[0], target_component_name
            )
            upper_bound_renamed = upper_bound.with_columns_renamed(
                upper_bound.components[0], target_component_name
            )
            
            # Perform the check using the correctly renamed TimeSeries
            is_within_interval = (actual_values >= lower_bound_renamed) & (actual_values <= upper_bound_renamed)
            # --- END OF FIX ---
            points_in_interval[q] += is_within_interval.sum().item()
            total_points[q] += len(is_within_interval)
            forecast_coverage[q].append(is_within_interval.sum().item() / len(is_within_interval) if len(is_within_interval) > 0 else 0.0)
            
    # Calculate and return the final coverage percentages
    for q in coverage_quantiles:
        metric_name = f'coverage_{int(q*100)}'
        if total_points[q] > 0:
            coverage = points_in_interval[q] / total_points[q]
        else:
            coverage = 0.0
        forecast_coverge_dropoff = np.diff(np.array(forecast_coverage[q])) if len(forecast_coverage[q]) > 1 else np.array([0.0])
        coverage_results[metric_name] = coverage
        coverage_results[f'{metric_name}_dropoff'] = np.mean(forecast_coverge_dropoff) if len(forecast_coverge_dropoff) > 0 else 0.0
        
    return coverage_results

from pytorch_lightning import Trainer


def _plot_mean_metrics_over_bursts(mean_metric_over_bursts_dict, burst_dates=None, save_path_png=None, show=True):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"Skipping mean-metrics plot; required plotting libraries not available: {e}")
        return

    if not mean_metric_over_bursts_dict:
        print("Skipping mean-metrics plot; no data provided.")
        return

    def _find_metric_key(metric_dict, target_name):
        target = target_name.lower()
        for key in metric_dict:
            if key.lower() == target:
                return key
        return None

    mape_key = _find_metric_key(mean_metric_over_bursts_dict, "MAPE")
    rmsse_key = _find_metric_key(mean_metric_over_bursts_dict, "RMSSE")

    if mape_key is None and rmsse_key is None:
        print("Skipping mean-metrics plot; neither MAPE nor RMSSE found in provided metrics.")
        return

    plt.figure(figsize=(10, 5))
    ax_left = plt.gca()

    legend_handles = []
    legend_labels = []

    if mape_key is not None:
        mape_values = mean_metric_over_bursts_dict.get(mape_key, [])
        if mape_values:
            x = np.arange(1, len(mape_values) + 1)
            (mape_line,) = ax_left.plot(
                x,
                mape_values,
                marker="o",
                color="tab:blue",
                label=mape_key.upper(),
            )
            if burst_dates:
                ax_left.set_xticks(x)
                ax_left.set_xticklabels([date.strftime('%Y-%m-%d') for date in burst_dates[:len(x)]], rotation=90, ha='right')
            ax_left.set_ylabel(mape_key.upper(), color="tab:blue")
            ax_left.tick_params(axis="y", labelcolor="tab:blue")
            legend_handles.append(mape_line)
            legend_labels.append(mape_key.upper())
        else:
            print(f"Skipping {mape_key.upper()} plot; no values provided.")

    ax_right = None
    if rmsse_key is not None:
        rmsse_values = mean_metric_over_bursts_dict.get(rmsse_key, [])
        if rmsse_values:
            x_rmsse = np.arange(1, len(rmsse_values) + 1)
            ax_right = ax_left.twinx()
            (rmsse_line,) = ax_right.plot(
                x_rmsse,
                rmsse_values,
                marker="s",
                color="tab:orange",
                label=rmsse_key.upper(),
            )
            ax_right.set_ylabel(rmsse_key.upper(), color="tab:orange")
            ax_right.tick_params(axis="y", labelcolor="tab:orange")
            legend_handles.append(rmsse_line)
            legend_labels.append(rmsse_key.upper())
        else:
            print(f"Skipping {rmsse_key.upper()} plot; no values provided.")

    if not legend_handles:
        plt.close()
        print("Skipping mean-metrics plot; no valid data to display.")
        return

    ax_left.set_xlabel("Historical forecast burst index")
    ax_left.set_title("Mean metric values over individual historical forecasts")

    plt.legend(legend_handles, legend_labels, loc="upper right")
    plt.tight_layout()

    if save_path_png:
        plt.savefig(save_path_png, dpi=150)
        print(f"Saved mean-metrics plot to {save_path_png}")
    if show:
        plt.show()
    
def _plot_mean_mapes_by_groups(historical_forecasts, series_list, save_path=None, json_save_path=None, show=True, is_multivariate: Optional[bool] = None):
    # Step 1: Extract and group data
    groups = defaultdict(lambda: defaultdict(list))
    
    # Auto-detect format if not specified
    if is_multivariate is None:
        has_series_id = all(
            hasattr(s, 'static_covariates') and s.static_covariates is not None and 'series_id' in s.static_covariates.columns
            for s in series_list
        )
        is_multivariate = not has_series_id
        print(f"Auto-detected mode: {'multivariate' if is_multivariate else 'univariate'}")
    
    # --- Data Collection Loop (Univariate) ---
    if not is_multivariate:  
        for forecasts, series in zip(historical_forecasts, series_list):
            if not hasattr(series, 'static_covariates') or series.static_covariates is None:
                continue
            try:
                client_id = str(series.static_covariates['client_id'].values[0])
                installation_id = str(series.static_covariates['installation_id'].values[0])
                series_id = str(series.static_covariates['series_id'].values[0])
            except KeyError:
                continue
            
            burst_mapes = []
            for burst in forecasts:
                p50 = burst.quantile(0.5)
                y_true = series.slice_intersect(p50)
                y_pred = p50.slice_intersect(series)
                # Assuming safe_mape_metric is available in your scope
                burst_mapes.append(float(safe_mape_metric(y_true, y_pred)))
            
            if not burst_mapes:
                continue
            
            mean_mape = np.mean(burst_mapes)
            groups[client_id][installation_id].append((series_id, mean_mape))
    
    # --- Data Collection Loop (Multivariate) ---
    else:  
        for forecasts_list, series in zip(historical_forecasts, series_list):
            if not hasattr(series, 'static_covariates') or series.static_covariates is None:
                continue
            try:
                client_id = str(series.static_covariates['client_id'].values[0])
                installation_id = str(series.static_covariates['installation_id'].values[0])
            except KeyError:
                continue
            
            components = series.components
            for component in components:
                series_id = str(component)
                series_uni = series[component]
                burst_mapes = []
                for burst in forecasts_list:
                    burst_uni = burst[component]
                    p50 = burst_uni.quantile(0.5)
                    y_true = series_uni.slice_intersect(p50)
                    y_pred = p50.slice_intersect(series_uni)
                    burst_mapes.append(float(safe_mape_metric(y_true, y_pred)))
                
                if not burst_mapes:
                    continue
                
                mean_mape = np.mean(burst_mapes)
                groups[client_id][installation_id].append((series_id, mean_mape))
    
    # Step 5: Edge case - no data
    if not groups:
        print("No valid groups found for plotting.")
        return
    
    # Collect sorted list of (client_id, installation_id) pairs
    pairs = []
    for client_id in sorted(groups.keys()):
        for installation_id in sorted(groups[client_id].keys()):
            pairs.append((client_id, installation_id))
    
    num_subplots = len(pairs)
    if num_subplots == 0:
        print("No valid pairs found for plotting.")
        return


    # Collect all Mean MAPE values into a dict
    mape_dict = {}
    for c_id, installs in groups.items():
        mape_dict[c_id] = {}
        for inst_id, series_data in installs.items():
            mape_dict[c_id][inst_id] = {s_id: m for s_id, m in series_data}
            
    # --- NEW LOGIC: Outlier-Filtered Limit Calculation ---
    # 1. Collect all Mean MAPE values into a flat list
    all_mapes = [
        m for client in groups.values() 
        for install in client.values() 
        for _, m in install
    ]
    
    
    plot_limit = 0
    if all_mapes:
        # 2. Calculate Quartiles
        q1, q3 = np.percentile(all_mapes, [25, 75])
        iqr = q3 - q1
        
        # 3. Define Upper Fence (Standard Outlier Definition)
        upper_fence = q3 + (1.5 * iqr)
        
        # 4. Filter values to find the "Robust Max"
        # We only consider values within the fence to determine the axis limit
        non_outliers = [x for x in all_mapes if x <= upper_fence]
        
        # Fallback to max(all_mapes) if data is weird and everything is an outlier
        robust_max = max(non_outliers) if non_outliers else max(all_mapes)
        
        # 5. Add padding (e.g., 10%)
        plot_limit = robust_max * 1.1
    # ------------------------------------------------------

    # Compute max_series for dynamic scaling
    max_series = max(
        len(data) for client in groups.values() for data in client.values()
    ) if groups else 0
    
    height = 6 * num_subplots
    width = 0.5 * max_series + 2
    fig, axes = plt.subplots(nrows=num_subplots, ncols=1, figsize=(width, height), squeeze=False)
    axes = axes.flatten()
    
    # Step 3: Populate subplots
    for idx, (client_id, installation_id) in enumerate(pairs):
        ax = axes[idx]
        series_data = sorted(groups[client_id][installation_id])
        
        if not series_data:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            ax.axis('off')
            continue
        
        series_ids, mean_mapes = zip(*series_data)
        y = np.arange(len(series_ids))
        
        # Plotting
        bars = ax.bar(y, mean_mapes)
        
        # --- APPLY THE CALCULATED LIMIT ---
        # Note: Since this is a vertical bar chart, the values are on the Y-axis.
        ax.set_ylim(0, plot_limit)
        
        # Optional: Color bars red if they exceed the view limit (visual indicator of outlier)
        for bar, val in zip(bars, mean_mapes):
            if val > plot_limit:
                #bar.set_color('red')
                # Add text annotation for the cut-off value
                ax.text(bar.get_x() + bar.get_width()/2, plot_limit * 0.95, 
                        f'{val:.1f}', ha='center', va='top', color='white', fontweight='bold', rotation=90)

        ax.set_xticks(y)
        ax.set_xticklabels(series_ids, rotation=45, ha='right')
        ax.set_title(f"Client: {client_id} - Installation: {installation_id}")
        ax.set_ylabel("Mean MAPE (%)")
        ax.set_xlabel("Series ID")
        ax.grid(True, axis='y', alpha=0.5) # Grid only on Y usually looks cleaner
    
    fig.suptitle("Mean MAPE per Series by Client and Installation", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved grouped MAPE plot to {save_path}")
        
    if show:
        plt.show()
        plt.close(fig)
            
    
def evaluate_model_with_backtest_bursts(
    model: ForecastingModel,
    series: TimeSeries,
    n_forecast: int,
    metric: str = 'rmse',
    n_spaced_strides: int = 5,
    burst_nr: int = 7,
    burst_stride: int = 1,
    start: Union[int, float, pd.Timestamp, None] = None,
    retrain_epochs: int = None,
    val_mult_forecast: float =2,
    auto_load_best: bool = True,
    num_samples: int = 100,
    save_dir: str = None,
    verbose: bool = True,
    trial: Optional[optuna.Trial] = None,
) -> dict:
    """
    Validates a forecasting model using burst forecasting with backtesting and automatic best model loading.
    
    This function performs a comprehensive validation by:
    1. Generating multiple evenly-spaced starting points across the time series
    2. For each starting point, training the model on historical data up to that point
    3. Generating a "burst" of sequential forecasts (multiple consecutive forecasts with specified stride)
    4. Calculating performance metrics for each burst and aggregating results
    
    The "burst" approach allows evaluation of model performance degradation over multiple consecutive 
    forecasting steps, which is useful for understanding how forecast quality changes over time.
    
    Parameters:
    -----------
    model : ForecastingModel
        The Darts forecasting model to validate. Will be configured with automatic best model 
        loading via custom callbacks during training.
    series : TimeSeries
        The time series data to validate on. Can also accept pandas DataFrame which will be 
        converted to TimeSeries.
    n_forecast : int
        The forecast horizon - number of time steps to forecast in each individual forecast.
    metric : str, default='rmse'
        The evaluation metric to calculate. Supported options: 'rmse', 'mape', 'mae'.
    n_spaced_strides : int, default=5
        Number of evenly spaced starting points to generate across the time series for validation.
        More points provide better validation coverage but increase computation time.
    burst_nr : int, default=7
        Number of consecutive forecasts to generate in each burst. Higher values test model 
        performance over longer sequences of predictions.
    burst_stride : int, default=1
        Number of time steps between the start of each forecast in a burst. A stride of 1 means 
        forecasts start at consecutive time points.
    start : Union[int, float, pd.Timestamp, None], default=None
        Starting point for validation. Can be:
        - None: Uses middle of series (len(series) // 2)
        - float (0-1): Fraction of series length
        - int: Absolute index
        - pd.Timestamp: Specific datetime
    retrain_epochs : int, optional
        Number of epochs to use when retraining the model. If None, uses the model's 
        default n_epochs setting.
    val_mult_forecast : float, default=2
        Multiplier for validation dataset size relative to forecast horizon. Validation size 
        will be n_forecast * val_mult_forecast, with minimum constraints based on model requirements.
    auto_load_best : bool, default=True
        Whether to automatically load the best model weights after training using a custom 
        callback. Improves forecast quality by using optimal model state.
    num_samples : int, default=100
        Number of samples to generate for probabilistic forecasts. Only relevant for 
        probabilistic models.
    save_name : str, optional
        If provided, saves the model checkpoints to a subdirectory with this name under the
        model's work directory. Useful for organizing burst validation results.
        If None, uses the default checkpoint directory.
    verbose : bool, default=True
        Whether to print detailed progress information, training details, and intermediate results.
    plot_bursts : bool, default=False
        Whether to generate a visualization showing the time series, validation start points, 
        and forecast periods for each burst.
    
    Returns:
    --------
    dict
        A dictionary containing validation results with the following keys:
        
        'metrics' : dict
            Performance metrics containing:
            - 'overall_performance': float - Mean metric value across all bursts and forecasts
            - 'prediction_averages': np.ndarray - Average metric for each forecast position (shape: n_forecasts)
            - 'burst_averages': np.ndarray - Average metric for each burst (shape: n_bursts) 
            - 'performance_dropoff': dict with:
                - 'avg_dropoff_pct': float - Average percentage dropoff from first to last forecast in bursts
                - 'avg_sequential_dropoff_pct': float - Average percentage dropoff between consecutive forecasts
            - 'metric': str - Name of the metric used
            
        'burst_info' : list
            List of dictionaries, one per burst, containing:
            - 'start_point': int - Index where burst training data ends
            - 'first_forecast_start': pd.Timestamp - Start time of first forecast in burst
            - 'last_forecast_end': pd.Timestamp - End time of last forecast in burst  
            - 'forecasts': list - List of TimeSeries forecast objects for this burst
            
        'model_config' : dict
            Configuration information used during validation:
            - 'model_type': str - Class name of the model
            - 'retrain_epochs': int - Number of epochs used for training
            - 'auto_load_best': bool - Whether best model loading was enabled
            - 'val_size': int - Actual validation dataset size used
    """
    
    # Setup
    series = _convert_dataframe(series)
    #print dimensions of series
    print(f"Series dimensions: {series.width} components, length {len(series)}")
    start = _determine_start_point(start, len(series), getattr(model, 'min_train_series_length', n_forecast * 2))
    
    # Configure model with automatic best model loading
    model.n_epochs = retrain_epochs
    
    start_points = generate_backtest_starts(
        series=series, start=start, forecast_horizon=n_forecast,
        min_train_series_length=getattr(model, 'min_train_series_length', n_forecast * 2),
        n_spaced_strides=n_spaced_strides, n_end_strides=0, plot=False
    )
    #plot start points
    if verbose:
        print(f"Generated {len(start_points)} start points for burst validation.")
        plt.figure(figsize=(12, 6))
        series.univariate_component(0).plot(label=f"Time Series (Component 0)")
        
        for start_pt in start_points:
            print(f"Start point at index {start_pt}")
            print(f"Start point datetime: {series.get_timestamp_at_point(int(start_pt))}")
            datetime_start = series.get_timestamp_at_point(int(start_pt))
            plt.axvline(x=series.time_index[start_pt], color='blue', linestyle='--', label=f'Start Point (index {datetime_start})')
        plt.title("Visualization of Backtest Start Points")
        plt.legend()
        plt.show()
    
    if verbose:
        _print_config(model, series, start_points, retrain_epochs, auto_load_best, 
                     burst_nr, burst_stride, n_forecast)
    
    # Get metric function
    metric_func = _get_metric_function(metric)
    
    # Process each burst
    all_burst_metrics = {}
    val_size= n_forecast* val_mult_forecast
    input_chunk_length = getattr(model, 'input_chunk_length', 1)
    output_chunk_length = getattr(model, 'output_chunk_length', 1)
    val_size = max(val_size, input_chunk_length + output_chunk_length)
    training_histories = {}
    train_histories =[]
    val_histories = []
    lr_histories = []
    
    i=0
    burst_forecasts_dict = {}
    for start_point in tqdm(start_points, desc="Training positions", disable=not verbose):
        start_point = int(start_point)
        
        if verbose:
            print(f"Processing burst starting at index {start_point}...")
        
        # Step 1: Manual training with train/val split
        train_data,_  = series.split_after(start_point)
        
        loss_history_callback = LossHistoryCallback()  # Initialize our custom callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, min_delta=0.0, mode="min", verbose=verbose)
        #extract existing modelCheckpoint callback if it exists

        # Erstelle den ModelCheckpoint mit dem extrahierten Pfad
        model_checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',  # Überwache val_loss
            dirpath=save_dir+f'/burst_{i}', 
            filename='best_model_{epoch:02d}_{val_loss:.2f}',  # Dateiname-Format
            save_top_k=1,  # Speichere nur das beste Modell
            mode='min',  # Minimiere val_loss
            every_n_epochs=1,  # Speichere jede Epoche
            verbose=verbose
            )    

        callbacks = [loss_history_callback, early_stopping, model_checkpoint_callback, LoadBestModelCallback()]  # Add our custom callback
        trainer = Trainer(
            max_epochs=retrain_epochs,
            callbacks=callbacks,  # Pass our callback here.
            enable_progress_bar=True,          # Keep HPO output clean.
            # Add other trainer settings as needed, e.g., accelerator
            accelerator="auto",
            precision="64-true",
            enable_model_summary=False,
            logger=False
        )

            # 3. Train the model. The callback is guaranteed to be active.
        _train_model_with_validation(model, train_data, val_size, trainer, verbose)

        history_df = loss_history_callback.get_history_df()
        training_histories[f'burst_{i}'] = {
            'train_loss': history_df['train_loss'],
            'val_loss': history_df['val_loss'],
            'lr': history_df['lr'],
            }
        train_histories.append(history_df['train_loss'].tolist())
        val_histories.append(history_df['val_loss'].tolist())
        lr_histories.append(history_df['lr'].tolist())
        if verbose:
            print(f"    Training history for Burst {i}:")
            loss_history_callback.plot_losses() # Plot losses after each training
            loss_history_callback.plot_lr()     # Plot learning rate after each training
            
        print('generating burst forecasts...')
        
        #cut of the series after series + burst_nr *buurst_stride + n_forecast
        if start_point + burst_nr * burst_stride + n_forecast > len(series):
            print(f"    Skipping burst at {start_point} - not enough data for full burst.")
            continue
        
        #convert startpoint from integer to datetime
        start_point_dt = series.start_time()
        end_point_dt = series.time_index[start_point + burst_nr * burst_stride + n_forecast]
        eval_series = series.slice(start_ts=start_point_dt, end_ts=end_point_dt)
        
                    # Step 2: Generate burst forecasts using historical_forecasts (no retraining)
        burst_forecasts = model.historical_forecasts(
            series=eval_series,
            start=start_point,
            forecast_horizon=n_forecast,
            stride=burst_stride,
            retrain=False,  # Model already trained and best weights loaded
            last_points_only=False,
            verbose=verbose,
            num_samples=num_samples
        )
        
        
        # Step 3: Calculate metrics for this burst
        burst_metrics = _calculate_burst_metrics(burst_forecasts, series, metric_func, len(series.components),verbose=verbose)
        all_burst_metrics[f'burst_{i}'] = burst_metrics
        burst_forecasts_dict[i]= burst_forecasts
        if trial is not None:
            # Log the mean metric for this burst to Optuna
            trial.report(burst_metrics['burst_mean'], step=i)
            if trial.should_prune():
                raise optuna.TrialPruned(f"Trial pruned at burst {i} with mean metric {burst_metrics['burst_mean']:.4f}")
        i+= 1
        
        
        
        if verbose:
            print(f"    ✓ Generated {len(burst_forecasts)} forecasts")
    
    training_histories['all_train_histories'] = train_histories
    training_histories['all_val_histories'] = val_histories
    training_histories['all_lr_histories'] = lr_histories
    ##convert train_histories to array and pad if necessary
    max_length = max(len(hist) for hist in train_histories)
    train_hist_array = np.array([np.pad(hist, (0, max_length - len(hist)), mode='constant', constant_values=np.nan) for hist in train_histories])
    val_hist_array = np.array([np.pad(hist, (0, max_length - len(hist)), mode='constant', constant_values=np.nan) for hist in val_histories])
    lr_hist_array = np.array([np.pad(hist, (0, max_length - len(hist)), mode='constant', constant_values=np.nan) for hist in lr_histories])
    training_histories['mean_train_loss'] = np.mean(train_hist_array, axis=0).tolist()
    training_histories['mean_val_loss'] = np.mean(val_hist_array, axis=0).tolist()
    training_histories['mean_lr'] = np.mean(lr_hist_array, axis=0).tolist()
    training_histories['max_epochs'] = max_length
    # Store burst info
    burst_info = {
        'burst_length': burst_nr,
        'n_burst': n_spaced_strides,
        'prediction_length': n_forecast,
        'stride': burst_stride,
        'forecasts': burst_forecasts_dict
    }
    # Calculate final results
    results = {}
    results['metrics'] = _calculate_final_metrics(all_burst_metrics, metric,verbose=verbose)
    results['training_history'] = training_histories
    results['burst_info'] = burst_info
    results['model_config'] = {
        'model_type': type(model).__name__,
        'retrain_epochs': retrain_epochs,
        'auto_load_best': auto_load_best,
        'val_size': val_size,
    }
    
    
    if verbose:
        overall = results.get('overall_performance')
        print(f"Results: {metric.upper()}={overall:.4f}" if overall else "Validation complete")
        
    _print_results(results['metrics'])
    
    return results




def evaluate_model_with_backtest_bursts_long_format(
    model: ForecastingModel,
    series: Union[TimeSeries, List[TimeSeries]],
    n_forecast: int,
    metric: str = 'rmse',
    n_spaced_strides: int = 5,
    burst_nr: int = 7,
    burst_stride: int = 1,
    start: Union[int, float, pd.Timestamp, None] = None,
    retrain_epochs: int = None,
    val_mult_forecast: float = 2,
    auto_load_best: bool = True,
    num_samples: int = 100,
    future_covariates: TimeSeries = None,
    save_dir: str = None,
    verbose: bool = True,
    trial: Optional[optuna.Trial] = None,
) -> dict:
    """
    Validates a forecasting model using burst forecasting with backtesting on a list of series (long format, global processing).
    
    Trains/predicts globally on the list for efficiency.
    
    Parameters:
    -----------
    model : ForecastingModel
        The Darts forecasting model to validate.
    series : Union[TimeSeries, List[TimeSeries]]
        The time series data (or list) to validate on.
    ... (other params same as original)
    
    Returns:
    --------
    dict
        With 'metrics' (overall aggregates), 'per_burst' details, training histories, etc.
    """
    # Convert to list for global handling
    if isinstance(series, TimeSeries):
        series_list = [series]
    else:
        series_list = series

    # Sort by series_id if available
    series_list = _sort_lists_by_series_id_train(series_list)

    if verbose:
        print(f"Evaluating globally on {len(series_list)} series.")

    # Use first series as reference for start points (assume aligned timestamps)
    ref_series = _convert_dataframe(series_list[0])
    print(f"Reference series dimensions: {ref_series.width} components, length {len(ref_series)}")
    ref_start = _determine_start_point(start, len(ref_series), getattr(model, 'min_train_series_length', n_forecast * 2))

    # Generate start points (once, based on reference)
    start_points = generate_backtest_starts(
        series=ref_series, start=ref_start, forecast_horizon=n_forecast,
        min_train_series_length=getattr(model, 'min_train_series_length', n_forecast * 2),
        n_spaced_strides=n_spaced_strides, n_end_strides=0, plot=False
    )

    if verbose:
        print(f"Generated {len(start_points)} start points for burst validation.")
        # Plot start points on reference series
        plt.figure(figsize=(12, 6))
        ref_series.univariate_component(0).plot(label=f"Reference Time Series (Component 0)")
        for start_pt in start_points:
            datetime_start = ref_series.get_timestamp_at_point(int(start_pt))
            plt.axvline(x=ref_series.time_index[start_pt], color='blue', linestyle='--', label=f'Start Point ({datetime_start})')
        plt.title("Visualization of Backtest Start Points (Reference Series)")
        plt.legend()
        plt.show()

    # Get metric function (unchanged)
    metric_func = _get_metric_function(metric)

    # Global config (unchanged)
    model.n_epochs = retrain_epochs

    all_burst_metrics = {}
    training_histories = {}
    train_histories = []
    val_histories = []
    lr_histories = []
    burst_forecasts_dict = {}
    i = 0

    val_size = n_forecast * val_mult_forecast
    input_chunk_length = getattr(model, 'input_chunk_length', 1)
    output_chunk_length = getattr(model, 'output_chunk_length', 1)
    val_size = max(val_size, input_chunk_length + output_chunk_length)

    transformer = StaticCovariatesTransformer()
    series_list = transformer.fit_transform(series_list)
    
    for start_point in tqdm(start_points, desc="Global training positions", disable=not verbose):
        start_point = int(start_point)

        if verbose:
            print(f"Processing global burst starting at index {start_point}...")

        # Global split: Create train_list and val_list by splitting each series
        train_list = []
        val_list = []
        for ser in series_list:
            # Adjust start_point if series is shorter
            adj_start = min(start_point, len(ser) - val_size - 1)
            train_data, _ = ser.split_after(adj_start + val_size)
            split_timestamp = train_data.time_index[-val_size]
            train_ser, val_ser = train_data.split_before(split_timestamp)
            train_list.append(train_ser)
            val_list.append(val_ser)

        # Callbacks (unchanged)
        loss_history_callback = LossHistoryCallback()
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, min_delta=0.0, mode="min", verbose=verbose)
        model_checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=f"{save_dir}/burst_{i}",
            filename='best_model_{epoch:02d}_{val_loss:.2f}',
            save_top_k=1,
            mode='min',
            every_n_epochs=1,
            verbose=verbose
        )
        callbacks = [loss_history_callback, early_stopping, model_checkpoint_callback, LoadBestModelCallback()]

        trainer = Trainer(
            max_epochs=retrain_epochs,
            callbacks=callbacks,
            enable_progress_bar=True,
            accelerator="auto",
            precision="64-true",
            enable_model_summary=False,
            logger=False
        )

        
        
        model.fit(series=train_list,
                  val_series=val_list,
                  trainer=trainer,
                  future_covariates=future_covariates if model.supports_future_covariates else None,
                  verbose=True
                  )

        history_df = loss_history_callback.get_history_df()
        training_histories[f'burst_{i}'] = {
            'train_loss': history_df['train_loss'],
            'val_loss': history_df['val_loss'],
            'lr': history_df['lr'],
        }
        train_histories.append(history_df['train_loss'].tolist())
        val_histories.append(history_df['val_loss'].tolist())
        lr_histories.append(history_df['lr'].tolist())
        if verbose:
            print(f"    Global training history for Burst {i}:")
            loss_history_callback.plot_losses()
            loss_history_callback.plot_lr()

        print('Generating global burst forecasts...')

        # Global eval slices: Slice each series for burst
        eval_list = []
        for ser in series_list:
            if start_point + burst_nr * burst_stride + n_forecast > len(ser):
                print(f"    Skipping burst for one series - not enough data.")
                continue
            start_point_dt = ser.start_time()
            end_point_dt = ser.time_index[min(start_point + burst_nr * burst_stride + n_forecast, len(ser) - 1)]
            eval_ser = ser.slice(start_ts=start_point_dt, end_ts=end_point_dt)
            eval_list.append(eval_ser)

        if not eval_list:
            continue

        # Global historical forecasts on list
        burst_forecasts_list = model.historical_forecasts(
            series=eval_list,
            start=start_point,
            forecast_horizon=n_forecast,
            stride=burst_stride,
            retrain=False,
            last_points_only=False,
            verbose=verbose,
            num_samples=num_samples
        )

        # Calculate metrics (adapt for list of forecast lists)
        burst_metrics = _calculate_burst_metrics(burst_forecasts_list, series_list, metric_func, len(series_list[0].components), verbose=verbose)
        all_burst_metrics[f'burst_{i}'] = burst_metrics
        burst_forecasts_dict[i] = burst_forecasts_list

        if trial is not None:
            trial.report(burst_metrics['burst_mean'], step=i)
            if trial.should_prune():
                raise optuna.TrialPruned(f"Trial pruned at burst {i} with mean metric {burst_metrics['burst_mean']:.4f}")

        i += 1

    # Aggregation (unchanged, but now burst_metrics are global)
    training_histories['all_train_histories'] = train_histories
    training_histories['all_val_histories'] = val_histories
    training_histories['all_lr_histories'] = lr_histories
    max_length = max(len(hist) for hist in train_histories)
    train_hist_array = np.array([np.pad(hist, (0, max_length - len(hist)), mode='constant', constant_values=np.nan) for hist in train_histories])
    val_hist_array = np.array([np.pad(hist, (0, max_length - len(hist)), mode='constant', constant_values=np.nan) for hist in val_histories])
    lr_hist_array = np.array([np.pad(hist, (0, max_length - len(hist)), mode='constant', constant_values=np.nan) for hist in lr_histories])
    training_histories['mean_train_loss'] = np.mean(train_hist_array, axis=0).tolist()
    training_histories['mean_val_loss'] = np.mean(val_hist_array, axis=0).tolist()
    training_histories['mean_lr'] = np.mean(lr_hist_array, axis=0).tolist()
    training_histories['max_epochs'] = max_length

    burst_info = {
        'burst_length': burst_nr,
        'n_burst': n_spaced_strides,
        'prediction_length': n_forecast,
        'stride': burst_stride,
        'forecasts': burst_forecasts_dict
    }

    results = {}
    results['metrics'] = _calculate_final_metrics(all_burst_metrics, metric, verbose=verbose)
    results['training_history'] = training_histories
    results['burst_info'] = burst_info
    results['model_config'] = {
        'model_type': type(model).__name__,
        'retrain_epochs': retrain_epochs,
        'auto_load_best': auto_load_best,
        'val_size': val_size,
    }

    if verbose:
        overall = results['metrics'].get('overall_performance')
        print(f"Global Results: {metric.upper()}={overall:.4f}" if overall else "Validation complete")
        _print_results(results['metrics'])

    return results



# [Keep all helper functions from before - they remain the same]
def _convert_dataframe(series):
    if isinstance(series, pd.DataFrame):
        print("Converting DataFrame to Darts TimeSeries...")
        if 'datetime' in series.columns:
            return TimeSeries.from_dataframe(series, time_col='datetime', value_cols=series.columns.tolist(), freq='1h')
        elif isinstance(series.index, pd.DatetimeIndex):
            return TimeSeries.from_dataframe(series, value_cols=series.columns.tolist(), freq='1h')
        else:
            raise ValueError("DataFrame must have a 'datetime' column or a DateTimeIndex.")
    return series

def _determine_start_point(start, series_len, min_train_len):
    if start is None:
        return series_len // 2
    elif isinstance(start, float) and 0 < start < 1:
        return int(series_len * start)
    else:
        return max(int(start), min_train_len)

def _get_metric_function(metric):
    metric_map = {'rmse': 'rmse', 'mape': 'mape', 'mae': 'mae'}
    if metric not in metric_map:
        raise ValueError(f"Unsupported metric: {metric}")
    from darts import metrics
    return getattr(metrics, metric_map[metric])

def _train_model_with_validation(model, train_data, val_size, trainer, verbose):
    """
    Manually train model with proper train/val split for ModelCheckpoint.
    """
    # Calculate validation size
    val_size = int(val_size)
    input_chunk_length = getattr(model, 'input_chunk_length', 24)
    output_chunk_length = getattr(model, 'output_chunk_length', 24)
    if val_size < output_chunk_length:
        val_size = output_chunk_length
    
    # Calculate split point (from the end)
    split_index = len(train_data) - val_size
    split_timestamp = train_data.time_index[split_index]
    
    # Split using Darts methods
    train_series, val_series = train_data.split_before(split_timestamp)
    
    if verbose:
        print(f"    Training: {len(train_series)} points, Validation: {len(val_series)} points")
        print(f"    Split at: {split_timestamp}")
    

    # Fit with validation data - this enables ModelCheckpoint to work properly
    model.fit(
        series=train_series,
        val_series=val_series,  # This is crucial for ModelCheckpoint!
        trainer=trainer,
        verbose=True
    )
    
def _train_model_with_validation_long_format(model, train_data, val_size, trainer, verbose):
    """
    Manually train model with proper train/val split for ModelCheckpoint.
    """
    # Calculate validation size
    val_size = int(val_size)
    input_chunk_length = getattr(model, 'input_chunk_length', 24)
    output_chunk_length = getattr(model, 'output_chunk_length', 24)
    if val_size < output_chunk_length:
        val_size = output_chunk_length
    
    # Calculate split point (from the end)
    split_index = len(train_data) - val_size
    split_timestamp = train_data.time_index[split_index]
    
    # Split using Darts methods
    train_series, val_series = train_data.split_before(split_timestamp)
    
    if verbose:
        print(f"    Training: {len(train_series)} points, Validation: {len(val_series)} points")
        print(f"    Split at: {split_timestamp}")
    

    # Fit with validation data - this enables ModelCheckpoint to work properly
    model.fit(
        series=train_series,
        val_series=val_series,  # This is crucial for ModelCheckpoint!
        trainer=trainer,
        verbose=True
    )
        

def _calculate_burst_metrics(burst_forecasts, series, metric_func, n_components, verbose):
    # Store mean metric over all components for each forecast
    forecast_metrics = np.zeros(len(burst_forecasts))
    for i, fcast in enumerate(burst_forecasts):
        try:
            actual_values = series[fcast.time_index]
            metric_values = metric_func(actual_values, fcast, series_reduction=None)  # Per-component
            forecast_metrics[i] = np.mean(metric_values)  # Mean over all 90 components
            
        except Exception as e:
            if verbose:
                print(f"    Error calculating metric for forecast {i}: {e}")
            forecast_metrics[i] = np.nan
    
    # Return both per-forecast metrics and overall mean
    burst_mean = np.nanmean(forecast_metrics)  # Mean over all predictions in burst
    print(f"Burst mean {metric_func.__name__}: {burst_mean:.4f}" if verbose else "")
    burst_dict = {}
    burst_dict['burst_mean'] = burst_mean  # Overall mean for this burst
    for i, forecast_metric in enumerate(forecast_metrics):
        burst_dict[f'forecast_{i}'] = forecast_metric
    if verbose:
        print(f"Per-forecast {metric_func.__name__} metrics: {burst_dict}")
    
    return burst_dict     # Shape: (n_forecasts,) - mean over 90 components for each forecast


def _calculate_final_metrics(all_burst_metrics, metric_name='rmse',verbose=True):
    import pandas as pd
    import numpy as np
    
    # Convert all_burst_metrics to a proper DataFrame
    # Handle case where burst metrics might be dictionaries
    processed_metrics = []
    
    print('all_burst_metrics:', all_burst_metrics)
    
    per_burst_means = {}
    per_prediction_sums = {}
    for burst_nr, forecast_dict in all_burst_metrics.items():
        burst_metrics = []
        forecast_metrics = []
        for forecast_nr, metric_value in forecast_dict.items():
            burst_metrics.append(metric_value)
            if forecast_nr not in per_prediction_sums:
                per_prediction_sums[forecast_nr] = 0.0
            per_prediction_sums[forecast_nr]+= metric_value 
            forecast_metrics.append(metric_value)
            
        per_burst_means[burst_nr] = np.nanmean(burst_metrics)  # Mean over all forecasts in this burst
            
    per_prediction_means = {}
    for i, metric_sum in per_prediction_sums.items():
        per_prediction_means[i] = metric_sum / len(all_burst_metrics)
    
    x_values = np.arange(len(per_prediction_means))
    z = np.polyfit(x_values, list(per_prediction_means.values()), 1)
    p = np.poly1d(z)
    #calculate difference between trend line start and end
    trend_start = p(x_values[0])
    trend_end = p(x_values[-1])
    dropoff_pct = (trend_start - trend_end) / trend_start * 100
    sequential_dropoff_pct = dropoff_pct / len(per_prediction_means) if len(per_prediction_means) > 0 else None
    
    if verbose:
        #plot burst prediction 
        fig = plt.figure(figsize=(12, 6))
        for burst_nr, burst_mean in all_burst_metrics.items():
            forecast_metrics = [v for k, v in burst_mean.items()]
            x_values = np.arange(len(forecast_metrics))
            plt.plot(x_values, forecast_metrics, marker='o', label=f'Burst {burst_nr}')
        plt.title(f"Per-Burst {metric_name.upper()} Means")
        plt.xlabel('Forecast Position')
        plt.ylabel(f'{metric_name.upper()}')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        #plot the performances and plot the performance dropoff
        fig = plt.figure(figsize=(12, 6))
        plt.plot(x_values, list(per_prediction_means.values()), marker='o', label='Per-Prediction Means')
        #fit line
        plt.plot(x_values, p(x_values), linestyle='--', color='red', label='Trend Line')
        plt.plot(x_values, np.mean(list(per_burst_means.values())) * np.ones_like(x_values), linestyle=':',color='black', label=f'Overall Mean = {np.nanmean(list(per_burst_means.values())):.4f}')
        #add dropoff percentage
        plt.title(f"Per-Prediction {metric_name.upper()} Means")
        plt.xlabel('Forecast Position')
        plt.ylabel(f'{metric_name.upper()}')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    overall_performance = np.nanmean(list(per_burst_means.values()))  # Mean over all bursts
    prediction_averages = np.array(list(per_prediction_means.values()))  # Shape: (n_forecasts,)
    burst_averages = np.array(list(per_burst_means.values()))  # Shape: (n_bursts,)
    
    return {
        'overall_performance': overall_performance,
        'prediction_averages': prediction_averages,
        'burst_averages': burst_averages,
        'performance_dropoff': {
            'avg_dropoff_pct': dropoff_pct,
            'avg_sequential_dropoff_pct': sequential_dropoff_pct
        },
        'metric':metric_name,
    }
    

def _print_config(model, series, start_points, retrain_epochs, auto_load_best, burst_length, burst_stride, n_forecast):
    print("-" * 60)
    print(f"MODEL: {type(model).__name__}")
    print(f"Series: {len(series)} points, {len(series.components)} components")
    print(f"Epochs: {retrain_epochs or getattr(model, 'n_epochs', 'N/A')}")
    print(f"Auto load best: {auto_load_best}")
    print(f"Validation points: {start_points}")
    print(f"Burst config: {burst_length} forecasts, stride={burst_stride}, horizon={n_forecast}")
    print("-" * 60)

def _print_results(results):
    print("-" * 60)
    print("VALIDATION RESULTS")
    if results['overall_performance']:
        print(f"Overall {results['metric'].upper()}: {results['overall_performance']:.4f}")
    if results['performance_dropoff']:
        print(f"Avg performance dropoff: {results['performance_dropoff']['avg_dropoff_pct']:.2f}%")
        print(f"Avg sequential perf dropoff: {results['performance_dropoff']['avg_sequential_dropoff_pct']:.2f}%")  
    if results['prediction_averages'] is not None:
        print(f"Avg per-prediction {results['metric'].upper()}: {np.round(results['prediction_averages'],4)}")
    print("-" * 60)

def _plot_burst_visualization(series, start, burst_info, model, burst_length, burst_stride, n_forecast):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 8))
    series.univariate_component(0).plot(label="Time Series", alpha=0.7)
    plt.axvline(x=series.time_index[start], color='blue', linewidth=2, 
               label=f'Validation Start', alpha=0.8)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(burst_info)))
    for i, (burst, color) in enumerate(zip(burst_info, colors)):
        plt.axvline(x=series.time_index[burst['start_point']], color='green', 
                   linestyle='--', alpha=0.8, label='Burst Starts' if i == 0 else "")
        for j, forecast in enumerate(burst['forecasts']):
            plt.axvspan(forecast.start_time(), forecast.end_time(), 
                       color=color, alpha=0.2)
            if i == 0 and j == 0:
                plt.axvline(x=forecast.start_time(), color='orange', linestyle=':', 
                           alpha=0.6, label='Forecast Periods')
    
    plt.title(f"Burst Validation with Best Model Loading - {type(model).__name__}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
import os
from pathlib import Path
from darts.models import NHiTSModel

def load_model(model_path: str) -> NHiTSModel:
    """
    Loads a Darts forecasting model by automatically finding the correct
    'work_dir' and 'model_name' from a single path.

    This function robustly finds the model's run directory by searching
    upwards from the given path for a directory containing a 'checkpoints' folder.

    Parameters:
    model_path (str): A path to any file or folder within a Darts model's
                      run directory structure.

    Returns:
    ForecastingModel: The loaded Darts forecasting model.
    """
    try:
        # Resolve the input path to an absolute path
        p = Path(model_path).resolve()

        # Determine the starting directory for our search.
        # If the provided path is a file, start from its parent directory.
        # Otherwise, start from the directory itself.
        search_dir = p.parent if p.is_file() else p
        
        # --- Find the model's run directory ---
        # We traverse up the directory tree from our starting point.
        # The correct 'model_run_dir' is the first directory we encounter
        # that contains a 'checkpoints' subdirectory.
        model_run_dir = None
        while search_dir != search_dir.parent: # Loop until we hit the filesystem root
            if (search_dir / 'checkpoints').is_dir():
                model_run_dir = search_dir
                break
            search_dir = search_dir.parent

        # If we looped all the way to the root and found nothing, raise an error.
        if model_run_dir is None:
            raise FileNotFoundError(
                f"Could not find a 'checkpoints' subdirectory in the path or any of its parents: {model_path}"
            )

        # The 'model_name' for Darts is the name of the run directory
        model_name = model_run_dir.name
        
        # The 'work_dir' is the parent directory of the run directory
        work_dir = str(model_run_dir.parent)

        print(f"Found model run directory: '{model_run_dir}'")
        print(f"Attempting to load model '{model_name}' from work directory '{work_dir}'...")

        # Load the model using the determined work_dir and model_name
        # The 'checkpoint' parameter is automatically found by Darts within the folder.
        model = NHiTSModel.load_from_checkpoint(model_name=model_name, work_dir=work_dir)
        
        print(f"Model '{model_name}' loaded successfully.")
        return model
    
    except Exception as e:
        print(f"Error loading model from path '{model_path}': {e}")
        raise


class HPOptimizer:
    def __init__(self, df_list, n_forecast, freq, epochs=50 ,project_name='i-chamber',group_name=None,models=None,long_format=False,thourough_evaluation=False, ramp_dict=None, rolling_window=None):
        #self.series = series
        self.df_list = df_list
        self.epochs = epochs
        self.n_forecast = n_forecast
        self.now = datetime.now()
        self.project_name = project_name
        self.save_name = os.path.join('models', project_name, f"hp_search_/{self.now.strftime('%Y-%m-%d_%H_%M_%S')}")
        if not os.path.exists(self.save_name):
            os.makedirs(self.save_name)
        self.best_error = np.inf
        self.group_name = f"{self.now.strftime('%Y-%m-%d_%H_%M_%S')}_"+group_name if group_name else f"{self.now.strftime('%Y-%m-%d_%H_%M_%S')}"
        self.models = models if models else ['nhits', 'deepar', 'tide']
        self.long_format = long_format
        #self.future_covariates = future_covariates
        self.thourough_evaluation = thourough_evaluation
        #self.covars_mapper = covars_mapper
        self.freq = freq
        self.ramp_dict = ramp_dict
        self.rolling_window = rolling_window
   
        
        
    def objective(self,trial):
        # Shared hyperparameters
        model_type = trial.suggest_categorical('model_type', self.models)
        steps_per_day = int(pd.Timedelta('1D') / pd.Timedelta(self.freq))
        input_chunk_length = trial.suggest_int('input_chunk_length', steps_per_day*7, steps_per_day*50, step=steps_per_day)
        output_chunk_length = trial.suggest_int('output_chunk_length', 6, min(steps_per_day*2,self.n_forecast,input_chunk_length), step=6)
        prediction_type = trial.suggest_categorical('prediction_type', ['likelihood', 'quantiles'])
        distribution = trial.suggest_categorical('distribution', ['normal'])
        batch_size = trial.suggest_int('batch_size', 16, 128, step=16)
        lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'adamw'])
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
        scaling= trial.suggest_categorical('global_scaaling', [True, False])
        if scaling:
            global_scaling = trial.suggest_categorical('global_scaling', [True, False])
            use_scaling=True
        else:
            global_scaling=False
            use_scaling=False
        use_wide_dfs = False #trial.suggest_categorical('use_wide_dfs', [True, False])
        
        run = wandb.init(project=f"hp_search_{self.project_name}", group=self.group_name)
        config = {
            'model_type': model_type,
            'input_chunk_length': input_chunk_length,
            'output_chunk_length': output_chunk_length,
            'prediction_type': prediction_type,
            'distribution': distribution,
            'batch_size': batch_size,
            'lr': lr,
            'optimizer': optimizer,
            'weight_decay': weight_decay,
            'scaling': scaling,
            'global_scaling': global_scaling,
            'use_wide_dfs': use_wide_dfs,
            'frequency': self.freq,
            'n_forecast': self.n_forecast,
            'trial_number': trial.number
        }
        print(f'config: {config}')
        
        self.series, self.future_covariates, self.past_covariates, self.covars_mapper = convert_long_dfs_to_darts_timeseries(self.df_list,
                                                                                            freq=self.freq, 
                                                                                            time_col='datetime', 
                                                                                            value_col='value', 
                                                                                            group_col_names=['series_id', 'client_id', 'installation_id'], 
                                                                                            future_horizon=self.n_forecast, 
                                                                                            future_horizon_unit='periods',
                                                                                            time_feature_names=['hour', 'is_daytime', 'weekday', 'dayofyear', 'weekofyear', 'month', 'quarter'], 
                                                                                            return_covars_mapper=True,
                                                                                            use_wide_df=use_wide_dfs,
                                                                                            rolling_window=self.rolling_window,)
        #save the covars mapper with pickle
        if not os.path.exists('covars_mapper.pkl'):
            with open(f'covars_mapper.pkl', 'wb') as f:
                pickle.dump(self.covars_mapper, f)
                
        #save the future covariates with pickle
        if not os.path.exists('future_covariates_{self.freq}.pkl'):
            with open(f'future_covariates_{self.freq}.pkl', 'wb') as f:
                pickle.dump(self.future_covariates, f)
        
        if not os.path.exists('series_list_{self.freq}.pkl'):
            with open(f'series_list_{self.freq}.pkl', 'wb') as f:
                pickle.dump(self.series, f)
                

    # Model-specific hyperparameters (conditional)
        if model_type == 'nhits':
            num_stacks = trial.suggest_int('num_stacks', 3, 5)
            num_blocks = trial.suggest_int('num_blocks', 1, 3)
            num_layers = trial.suggest_int('num_layers', 2, 4)
            layer_widths = trial.suggest_int('layer_widths', 256, 1028, step=256)
            config.update({
                'num_stacks': num_stacks,
                'num_blocks': num_blocks,
                'num_layers': num_layers,
                'layer_widths': layer_widths
            })
            
        elif model_type == 'tft':
            hidden_size = trial.suggest_int('hidden_size', 16, 128, step=16)  # Optimized range for TFT
            lstm_layers = trial.suggest_int('lstm_layers', 1, 4)
            num_attention_heads = trial.suggest_int('num_attention_heads', 2, 8, step=2)
            dropout = trial.suggest_float('dropout', 0.0, 0.3, step=0.05)
            full_attention = trial.suggest_categorical('full_attention', [True, False])
            config.update({
                'hidden_size': hidden_size,
                'lstm_layers': lstm_layers,
                'num_attention_heads': num_attention_heads,
                'dropout': dropout,
                'full_attention': full_attention
            })

        elif model_type == 'tide':
            num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 3)
            num_decoder_layers = trial.suggest_int('num_decoder_layers', 1, 3)
            decoder_output_dim = trial.suggest_int('decoder_output_dim', 16, 64, step=16)
            hidden_size = trial.suggest_int('hidden_size', 64, 256, step=64)
            temporal_decoder_hidden = trial.suggest_int('temporal_decoder_hidden', 32, 256, step=32)
            dropout = trial.suggest_float('dropout', 0.0, 0.3, step=0.05)
            use_reversible_instance_norm = trial.suggest_categorical('use_reversible_instance_norm', [True, False])
            config.update({
                'num_encoder_layers': num_encoder_layers,
                'num_decoder_layers': num_decoder_layers,
                'decoder_output_dim': decoder_output_dim,
                'hidden_size': hidden_size,
                'temporal_decoder_hidden': temporal_decoder_hidden,
                'dropout': dropout,
                'use_reversible_instance_norm': use_reversible_instance_norm
            })

        # --- ADDED DEEPAR CONFIGURATION HERE ---
        elif model_type == 'deepar':
            hidden_dim = trial.suggest_int('hidden_dim', 16, 128, step=16)
            n_rnn_layers = trial.suggest_int('n_rnn_layers', 1, 4)
            dropout = trial.suggest_float('dropout', 0.0, 0.3, step=0.05)
            rnn_type = trial.suggest_categorical('rnn_type', ['LSTM', 'GRU'])
            config.update({
                'hidden_dim': hidden_dim,
                'n_rnn_layers': n_rnn_layers,
                'dropout': dropout,
                'rnn_type': rnn_type
            })

        print(f"Trial {trial.number} - Config: {config}")

        wandb.config.update(config)
        trial_dir = os.path.join(self.save_name, f"trial_{trial.number}")
        if not os.path.exists(trial_dir):
            os.makedirs(trial_dir)
            
        loss_history = LossHistoryCallback()
        
        # Build the model based on type
        if model_type == 'nhits':
            model, save_dir = build_nhits_probabilistic_model(
                input_chunk_length=input_chunk_length,
                output_chunk_length=output_chunk_length,
                n_epochs=self.epochs,
                num_stacks=num_stacks,
                num_blocks=num_blocks,
                num_layers=num_layers,
                layer_widths=layer_widths,
                prediction_type=prediction_type,
                distribution=distribution,
                batch_size=batch_size,
                lr=lr,
                optimizer=optimizer,
                weight_decay=weight_decay,
                save_dir=trial_dir,
                callbacks=[loss_history],
                save_checkpoints=True,
            )
        elif model_type == 'tft':
            model, save_dir = build_tft_probabilistic_model(
                input_chunk_length=input_chunk_length,
                output_chunk_length=output_chunk_length,
                n_epochs=self.epochs,
                hidden_size=hidden_size,
                lstm_layers=lstm_layers,
                num_attention_heads=num_attention_heads,
                full_attention=full_attention,
                dropout=dropout,
                prediction_type=prediction_type,
                distribution=distribution,
                batch_size=batch_size,
                lr=lr,
                optimizer=optimizer,
                weight_decay=weight_decay,
                save_dir=trial_dir,
                callbacks=[loss_history],
                save_checkpoints=True,
            )
        elif model_type == 'tide':
            model, save_dir = build_tide_probabilistic_model(
                input_chunk_length=input_chunk_length,
                output_chunk_length=output_chunk_length,
                n_epochs=self.epochs,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                decoder_output_dim=decoder_output_dim,
                hidden_size=hidden_size,
                temporal_decoder_hidden=temporal_decoder_hidden,
                dropout=dropout,
                use_reversible_instance_norm=use_reversible_instance_norm,
                prediction_type=prediction_type,
                distribution=distribution,
                batch_size=batch_size,
                lr=lr,
                optimizer=optimizer,
                weight_decay=weight_decay,
                save_dir=trial_dir,
                callbacks=[loss_history],
                save_checkpoints=True,
            )
        # --- ADDED DEEPAR BUILD CALL HERE ---
        elif model_type == 'deepar':
            model, save_dir = build_deepar_probabilistic_model(
                input_chunk_length=input_chunk_length,
                output_chunk_length=output_chunk_length,
                n_epochs=self.epochs,
                rnn_type=rnn_type,
                hidden_dim=hidden_dim,
                n_rnn_layers=n_rnn_layers,
                dropout=dropout,
                prediction_type=prediction_type,
                distribution=distribution,
                batch_size=batch_size,
                lr=lr,
                optimizer=optimizer,
                weight_decay=weight_decay,
                save_dir=trial_dir,
                callbacks=[loss_history],
                save_checkpoints=True,
            )
            
        
        wandb.config.update({'save_dir': save_dir})
            
        if self.long_format:
            if self.thourough_evaluation:
                results = evaluate_model_with_backtest_bursts_long_format(model=model,
                                                        series=self.series,
                                                        n_forecast=self.n_forecast,
                                                        metric='rmse',
                                                        n_spaced_strides=2,
                                                        burst_nr=90,
                                                        burst_stride=24*1,
                                                        retrain_epochs=self.epochs,
                                                        save_dir= trial_dir,
                                                        verbose=self.verbose,
                                                        trial=trial,
                                                        )
                error_score = results['metrics']['overall_performance']
                print(f"Trial {trial.number} - Score: {error_score}, Params: {trial.params}")
                
                epochs = list(range(results['training_history']['max_epochs']))
                for epoch in range(max(len(epochs), results['burst_info']['burst_length'])):

                    log_dict = {
                        'mean_train_loss': results['training_history']['mean_train_loss'][epoch] if epoch < len(results['training_history']['mean_train_loss']) else np.nan,
                        'mean_val_loss': results['training_history']['mean_val_loss'][epoch] if epoch < len(results['training_history']['mean_val_loss']) else np.nan,
                        'mean_lr': results['training_history']['mean_lr'][epoch] if epoch < len(results['training_history']['mean_lr']) else np.nan,
                    }
                    if results['metrics']['prediction_averages'].size > epoch:
                        log_dict['prediction_average'] = results['metrics']['prediction_averages'][epoch]
                    if results['metrics']['burst_averages'].size > epoch:
                        log_dict['burst_average'] = results['metrics']['burst_averages'][epoch]
                    run.log(log_dict, step=epoch)

                if error_score > 100:
                    error_score = np.inf
                    
                run.log({
                    'error_score': error_score,
                    'trial_number': trial.number,
                    'performance_dropoff_pct': results['metrics']['performance_dropoff']['avg_dropoff_pct'],
                    'sequential_dropoff_pct': results['metrics']['performance_dropoff']['avg_sequential_dropoff_pct'],
                    'burst_length': results['burst_info']['burst_length'],
                    'n_burst': results['burst_info']['n_burst'],
                    'prediction_length': results['burst_info']['prediction_length'],
                    'stride': results['burst_info']['stride'],
                    'sampling_freq':self.series.index.freqstr,
                    })
                
                if self.ramp_dict:
                    figs = visualize_ramp_predictions(self.ramp_dict, model=model, future_horizon=self.n_forecast, freq=self.freq, static_transformer=self.covars_mapper,)
                    for idx, fig in figs.items():
                        run.log({f'ramp_forecast_{idx}': wandb.Image(fig)})
               
                        
            else:
                train_list, val_list = split_series_long_format(self.series, n_forecast=self.n_forecast, input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length,val_size=0.3)
                print('start training...')
                model = train_probabilistic_model(model=model, 
                                                series=train_list,
                                                val_series=val_list, 
                                                n_epochs=self.epochs, 
                                                future_covariates=self.future_covariates, 
                                                past_covariates=self.past_covariates,
                                                verbose=True, 
                                                return_history_df=True, 
                                                input_chunk_length=input_chunk_length, 
                                                output_chunk_length=output_chunk_length,
                                                prediction_type=prediction_type,
                                                distribution=distribution,
                                                batch_size=batch_size,
                                                lr=lr,
                                                n_forecast=self.n_forecast,
                                                use_scaling=use_scaling,
                                                scale_globally=global_scaling,
                                            )
                history_df= loss_history.get_history_df()
                for entry in history_df.itertuples():
                    run.log({
                        'train_loss': entry.train_loss,
                        'val_loss': entry.val_loss,
                        'lr': entry.lr,
                        'epoch': entry.epoch
                    }, step=entry.Index)
                
                results = validate_model_simplified(model=model,
                                         series=val_list,
                                         n_forecast=self.n_forecast, 
                                         future_covariates=self.future_covariates, 
                                         past_covariates=self.past_covariates,
                                         start=0.0 ,
                                         burst_stride=self.n_forecast,
                                         plot=False, 
                                         save_html_path=save_dir+'/validation_results.html',
                                         covars_mapper=self.covars_mapper,
                                         use_scaling=use_scaling,
                                         scale_globally=global_scaling,)
                
                error_score = results['mape']
                nll_score = results['nll']
                burst_mapes = results['burst_metrics']['mape']
                burst_rmsses = results['burst_metrics']['rmsse']
                run.log({k: v for k, v in results.items()})
                run.log({'burst_mapes': wandb.plot.line_series(
                    xs=list(range(len(burst_mapes))),   
                    ys=[burst_mapes],
                    keys=['mape'],
                    title='Burst MAPEs',
                    xname='Burst Number'
                )})
                run.log({'burst_rmsses': wandb.plot.line_series(
                    xs=list(range(len(burst_rmsses))),   
                    ys=[burst_rmsses],
                    keys=['rmsse'],
                    title='Burst RMSEs',
                    xname='Burst Number'
                )})
                run.log({#'validation_plots':wandb.Html(os.path.join(save_dir+'/validation_results.html')),
                         'burst_metrics':wandb.Image(os.path.join(save_dir+'/validation_results_burst_metrics.png')),
                         'series_mapes':wandb.Image(os.path.join(save_dir+'/validation_results_series_mapes.png')),})
        
                if self.ramp_dict:
                    figs = visualize_ramp_predictions(long_df_list=self.df_list ,ramp_dict=self.ramp_dict, model=model, future_horizon=self.n_forecast, freq=self.freq, static_transformer=self.covars_mapper,)
                    for idx, fig in enumerate(figs):
                        run.log({f'ramp_forecast_{idx}': wandb.Image(fig)})
               
        else:
            results = evaluate_model_with_backtest_bursts(model=model,
                                                        series=self.series,
                                                        n_forecast=self.n_forecast,
                                                        metric='rmse',
                                                        n_spaced_strides=2,
                                                        burst_nr=90,
                                                        burst_stride=24*1,
                                                        retrain_epochs=self.epochs,
                                                        save_dir= trial_dir,
                                                        verbose=False,
                                                        trial=trial,
                                                        )
        

        run.finish()
        #create a folder named best_model in the save_name directory and copy the model files there if the error_score is the best so far
        if error_score < self.best_error:
            self.best_error = error_score
            best_model_dir = os.path.join(self.save_name, 'best_model')
            if not os.path.exists(best_model_dir):
                os.makedirs(best_model_dir)
            #copy all files from trial_dir to best_model_dir
            for file_name in os.listdir(trial_dir):
                full_file_name = os.path.join(trial_dir, file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, best_model_dir)
            print(f"New best model found! Trial {trial.number} with error score {error_score}. Model files copied to {best_model_dir}")
        
        return error_score
        
    def start_hpo(self, n_trials, verbose = True):
        self.verbose = verbose
        self.best_error = np.inf
        
        study = optuna.create_study(direction="minimize",pruner=optuna.pruners.MedianPruner(n_startup_trials=20),sampler=optuna.samplers.TPESampler(seed=42,multivariate=True),)
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True,)
        
        print("Best trial:")
        trial = study.best_trial
        best_params = trial.params
        best_score = trial.value
        print("  Error: {}".format(best_score))
        print("  Params: ", best_params)
        
        
        save_dir = os.path.join(self.save_name,'results')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Save best parameters into a file
        params_file = os.path.join(save_dir, 'best_params.json')
        with open(params_file, 'w') as f:
            json.dump(best_params, f, indent=4)
            
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(os.path.join(save_dir, 'optimization_history.html'))
        show(fig)
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(os.path.join(save_dir, 'parallel_coordinate.html'))
        show(fig)
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(os.path.join(save_dir, 'param_importances.html'))
        show(fig)
        importance_dict = optuna.importance.get_param_importances(study)
        sorted_keys=list(importance_dict.keys())
        print(sorted_keys)
        fig = optuna.visualization.plot_contour(study, params=[sorted_keys[0], sorted_keys[1]])
        fig.write_html(os.path.join(save_dir, 'param_contour.html'))
        show(fig)
        
 
        return best_score, best_params
    
    
    

    
import pandas as pd
from typing import Optional, Dict, Any

def prepare_wide_df_for_long_format(
    wide_df: pd.DataFrame,
    client_id: str,
    installation_id: Optional[str] = None,
    static_features: Optional[Dict[str, Any]] = None,
    series_units: Optional[Dict[str, str]] = None, 
    keep_wide: bool = False,
) -> pd.DataFrame:
    """
    Converts a single wide-format DataFrame into a long-format DataFrame.
    """
    df = wide_df.copy()
    df.reset_index(inplace=True)
    df.rename(columns={df.columns[0]: 'time_stamp'}, inplace=True)

    if keep_wide == False:
        long_df = pd.melt(
            df,
            id_vars=['time_stamp'],
            var_name='series_id',
            value_name='value'
        )
        long_df['series_id'] = long_df['series_id'].astype(str)
        
        if series_units:
            long_df['unit'] = long_df['series_id'].map(series_units)
        else:
            # FIX 3: Assign the result to a new column in long_df
            long_df['unit'] = infer_units_from_series_ids(long_df['series_id'])
            
    else:
        long_df = df.copy()

    # Add the client identifier to every row
    long_df['client_id'] = client_id
    if installation_id:
        long_df['installation_id'] = installation_id

    # Add any other static features for this client
    if static_features:
        for key, val in static_features.items():
            long_df[key] = val

    print(f"Converted wide DataFrame with shape {wide_df.shape} to long format with shape {long_df.shape}")
    
    return long_df

def infer_units_from_series_ids(series_id_series: pd.Series) -> pd.Series:
    """
    Infers units from the 'series_id' column using the unit_names.csv metadata file.
    """ 
    path = "unit_names.csv"
    
    # FIX 1: Specify the semicolon separator
    units_df = pd.read_csv(path, sep=';') 
    
    # FIX 2: Use the actual column names from the CSV
    unit_mapping = dict(zip(units_df['ColumnName'], units_df['Dimension']))
    
    inferred_units = series_id_series.map(unit_mapping)
    
    # Fill empty values with 'Values' (handles NaN from pandas reading empty CSV dimensions)
    inferred_units = inferred_units.fillna('Values')
    
    return inferred_units


def convert_long_dfs_to_darts_timeseries(
    long_df_list: Union[pd.DataFrame, List[pd.DataFrame]],
    static_col_names: List[str] = 'unit',
    group_col_names: List[str] = ['client_id', 'series_id', 'installation_id'],
    time_col: str = 'time_stamp',
    value_col: str = 'value',
    freq: str = 'h',
    time_feature_names: List[str] = [],  # Optional: e.g., ['minute', 'hour', 'weekday', 'dayofyear', 'month']
    future_horizon: Optional[Union[int, pd.Timestamp]] = None,  # Optional: Required if time_feature_names provided
    future_horizon_unit: str = 'periods',  # 'periods' (int) or 'end_timestamp' (pd.Timestamp)
    return_covars_mapper = False,
    use_wide_df=False,
    rolling_window: int = None,
) -> tuple[List[TimeSeries], Optional[List[TimeSeries]]]:
    """
    Converts a pre-processed long-format DataFrame (or list of them) into a list of Darts TimeSeries.

    This function assumes you have already converted each of your original wide
    DataFrames into the required long format.

    Optionally generates future covariates from time-based features (e.g., hour, weekday)
    by calling generate_future_covariates with the concatenated DataFrame.

    Args:
        long_df_list (Union[pd.DataFrame, List[pd.DataFrame]]): A single long-format pandas DataFrame or a list of them.
            Each DataFrame MUST contain the columns specified in
            `time_col`, `value_col`, `group_col_names`, and `static_col_names`.
        static_col_names (List[str]): A list of column names to be treated as
            static covariates (e.g., ['client_id', 'location']).
        group_col_names (List[str], optional): A list of column names that uniquely
            identify each time series. Defaults to ['client_id', 'series_id', 'installation_id'].
        time_col (str, optional): The name of the timestamp column. Defaults to 'time_stamp'.
            If not found, attempts to detect and rename from common alternatives ('datetime', 'timestamp', etc.).
        value_col (str, optional): The name of the target value column. Defaults to 'value'.
        freq (str, optional): The frequency to resample the time series to. Defaults to 'H'.
        time_feature_names (List[str], optional): Time-based features to calculate for future covariates.
            Supported: 'minute' (0-59), 'hour' (0-23), 'weekday' (0-6), 'dayofyear' (1-366), 'month' (1-12).
        future_horizon (Optional[Union[int, pd.Timestamp]]): If provided along with time_feature_names,
            generate future covariates extended by this many periods or to this end timestamp.
        future_horizon_unit (str, optional): 'periods' (int horizon) or 'end_timestamp' (pd.Timestamp).
            Defaults to 'periods'.

    Returns:
        Tuple[List[TimeSeries], Optional[List[TimeSeries]]]: A tuple of (target TimeSeries list,
        future covariates TimeSeries list). Future covariates are None if not requested.
    """
    # Step 1: Handle input validation and concatenation
    if isinstance(long_df_list, pd.DataFrame):
        if long_df_list.empty:
            raise ValueError("The input DataFrame cannot be empty.")
        final_long_df = long_df_list.copy()
    elif isinstance(long_df_list, list):
        if not long_df_list:
            raise ValueError("The input list 'long_df_list' cannot be empty.")
        final_long_df = pd.concat(long_df_list, ignore_index=True)
        if final_long_df.empty:
            raise ValueError("The concatenated DataFrame is empty.")
    else:
        raise ValueError("long_df_list must be a pd.DataFrame or a List[pd.DataFrame].")

    # Step 1.5: Detect and standardize time_col if it doesn't exist (flexibility for variable names)
    if time_col not in final_long_df.columns:
        possible_time_cols = ['time_stamp', 'datetime', 'timestamp']
        for poss_col in possible_time_cols:
            if poss_col in final_long_df.columns:
                final_long_df = final_long_df.rename(columns={poss_col: time_col})
                print(f"Renamed '{poss_col}' to specified '{time_col}' for consistency.")
                break
        else:
            # If no column found, check if index is datetime and reset it
            if isinstance(final_long_df.index, pd.DatetimeIndex):
                final_long_df = final_long_df.reset_index().rename(columns={'index': time_col})
                print(f"Reset datetime index and renamed to '{time_col}'.")
            else:
                raise ValueError(f"Specified time_col '{time_col}' not found in DataFrame columns: {list(final_long_df.columns)}. "
                                 f"Tried alternatives: {possible_time_cols}. Ensure the DataFrame has a valid time column or datetime index.")

    print("--- Final Combined Long DataFrame ---")
    print(final_long_df.head())
    print(f"\nTotal rows: {len(final_long_df)}")
    print(f"Unique series found: {final_long_df.groupby(group_col_names).ngroups}")
    
    
    if use_wide_df:
        final_df = final_long_df.pivot_table(index=[time_col] + ['client_id', 'installation_id'], 
                                        columns='series_id', 
                                        values='value').reset_index()
        final_df.columns.name = None  # Remove the columns name to avoid confusion
        print("--- Converted Wide DataFrame ---")
        print(final_df.head())
        list_of_series = TimeSeries.from_group_dataframe(
                                                        final_df,
                                                        time_col=time_col,
                                                        group_cols=['client_id', 'installation_id'], #only group by client_id and installation_id
                                                        value_cols=None, #use all columns except time_col and group_cols
                                                        static_cols=static_col_names,
                                                        freq=freq,
                                                        fill_missing_dates=True,
                                                        fillna_value=0,
                                                        verbose=True
                                                    )
    else:

        list_of_series = TimeSeries.from_group_dataframe(
                                                        final_long_df,
                                                        time_col=time_col,
                                                        group_cols=group_col_names,
                                                        value_cols=value_col,
                                                        static_cols=static_col_names,
                                                        freq=freq,
                                                        fill_missing_dates=True,
                                                        fillna_value=0,
                                                        verbose=True
                                                    )
    print(f"Converted to {len(list_of_series)} TimeSeries objects with frequency '{freq}'.")
    
    # --- Step 3: Generate PAST COVARIATES ---
    past_covariates_list = None
    
    # Only generate if a window is provided
    if rolling_window is not None:
        if use_wide_df:
             print("Warning: Past covariate generation skipped (not supported in use_wide_df mode)")
        else:
            past_covariates_list = generate_past_covariates(
                long_df=final_long_df,
                group_col_names=group_col_names,
                time_col=time_col,
                value_col=value_col,
                rolling_window=rolling_window, # Pass the window size
                freq=freq,
                verbose=False
            )
    # Step 4: Optionally generate future covariates using the separate function
    future_covariates_list = None
    if time_feature_names and future_horizon is not None:
        future_covariates_list = generate_future_covariates(
            long_df=final_long_df if not use_wide_df else final_df,
            time_feature_names=time_feature_names,
            group_col_names=group_col_names if not use_wide_df else ['client_id', 'installation_id'],
            time_col=time_col,
            freq=freq,
            future_horizon=future_horizon,
            future_horizon_unit=future_horizon_unit,
            verbose=True
        )
    elif time_feature_names and future_horizon is None:
        raise ValueError("future_horizon is required when time_feature_names are provided.")
    print(f'unique series in future covariates: {len(future_covariates_list) if future_covariates_list else 0}')
    #transform static covariates to numeric values
    if group_col_names:
        covars_transformer = StaticCovariatesTransformer()
        list_of_series = covars_transformer.fit_transform(list_of_series)
        if future_covariates_list:
            transformer = StaticCovariatesTransformer()
            future_covariates_list = transformer.fit_transform(future_covariates_list)


    #convert all series in list to float32
    for i in range(len(list_of_series)):
        list_of_series[i] = list_of_series[i].astype(np.float32)
    for i in range(len(future_covariates_list or [])):
        future_covariates_list[i] = future_covariates_list[i].astype(np.float32)
        
    if return_covars_mapper:
        return list_of_series, future_covariates_list, past_covariates_list, covars_transformer
    else:
        return list_of_series, future_covariates_list, past_covariates_list

    

def generate_past_covariates(
    long_df: pd.DataFrame,
    group_col_names: List[str] = ['client_id', 'series_id', 'installation_id'],
    time_col: str = 'time_stamp',
    value_col: str = 'value',
    rolling_window: int = 24,
    freq: str = 'h',
    verbose: bool = True
) -> List[TimeSeries]:
    """
    Calculates rolling statistics (Trend, Variance) and converts them to Darts Past Covariates.

    Args:
        long_df (pd.DataFrame): The source dataframe (Long format).
        group_col_names (List[str]): Columns identifying unique series.
        time_col (str): Timestamp column name.
        value_col (str): The target column to calculate stats from.
        rolling_window (int): The window size for rolling calculations.
        freq (str): Frequency of the series.

    Returns:
        List[TimeSeries]: A list of multivariate TimeSeries containing 'trend' and 'variance'.
    """
    if verbose:
        print(f"--- Generating Past Covariates (Rolling Window={rolling_window}) ---")

    # 1. Work on a copy to avoid SettingWithCopy warnings on the original DF
    df = long_df.copy()

    # 2. GENERATE FEATURES (The actual calculation)
    # We use transform so the result aligns perfectly with the original rows
    # Note: We group by the identifiers to ensure rolling stats don't bleed across different series
    grouped = df.groupby(group_col_names)[value_col]
    
    # Calculate Trend (Rolling Mean)
    df['trend'] = grouped.transform(lambda x: x.rolling(window=rolling_window, min_periods=1).mean())
    
    # Calculate Variance (Rolling Variance)
    df['variance'] = grouped.transform(lambda x: x.rolling(window=rolling_window, min_periods=1).var())

    # Fill NaNs generated by the rolling window (usually the first few indices)
    df['trend'] = df['trend'].fillna(method='bfill').fillna(0)
    df['variance'] = df['variance'].fillna(method='bfill').fillna(0)

    if verbose:
        print("Calculated columns: ['trend', 'variance']")

    # 3. CONVERT TO DARTS TIMESERIES
    # We select only the new feature columns for this list
    past_covariates_list = TimeSeries.from_group_dataframe(
        df,
        time_col=time_col,
        group_cols=group_col_names,
        value_cols=['trend', 'variance'],  # We explicitly grab the generated features
        freq=freq,
        fill_missing_dates=True,
        fillna_value=0,
        verbose=verbose
    )

    # 4. OPTIMIZATION: Cast to float32
    for i in range(len(past_covariates_list)):
        past_covariates_list[i] = past_covariates_list[i].astype(np.float32)

    return past_covariates_list


def generate_future_covariates(
    long_df: pd.DataFrame,
    time_feature_names: List[str],
    group_col_names: List[str],
    time_col: str,
    freq: str,
    future_horizon: Union[int, pd.Timestamp],
    future_horizon_unit: str = 'periods',
    verbose: bool = True
) -> List[TimeSeries]:
    """
    Generates future covariate TimeSeries from time-based features, using a concatenated long-format DataFrame.

    Computes features (e.g., hour, weekday) for historical + extended future timestamps per group.
    Ensures alignment with target series derived from the same DataFrame.

    Args:
        long_df: Concatenated long-format DataFrame (e.g., from convert_long_dfs_to_darts_timeseries).
        time_feature_names: Time-based features to calculate (e.g., ['hour', 'weekday']).
            Supported: 'minute' (0-59), 'hour' (0-23), 'weekday' (0-6), 'dayofyear' (1-366), 'month' (1-12).
        group_col_names: Columns defining unique series (e.g., ['client_id', 'series_id', 'installation_id']).
        time_col: Timestamp column (e.g., 'time_stamp' or 'datetime').
        freq: Time series frequency (e.g., 'H').
        future_horizon: Extend covariates by this many periods or to this end timestamp.
        future_horizon_unit: 'periods' (int horizon) or 'end_timestamp' (pd.Timestamp). Defaults to 'periods'.
        verbose: Print logging info (default: True).

    Returns:
        List of future covariate TimeSeries (one per group, aligned with targets from the same DF).
    """
    if not time_feature_names:
        raise ValueError("At least one time_feature_names must be provided.")

    # Detect and standardize time_col if it doesn't exist (flexibility for variable names)
    if time_col not in long_df.columns:
        possible_time_cols = ['time_stamp', 'datetime', 'timestamp']
        for poss_col in possible_time_cols:
            if poss_col in long_df.columns:
                long_df = long_df.rename(columns={poss_col: time_col})
                if verbose:
                    print(f"Renamed '{poss_col}' to specified '{time_col}' for consistency.")
                break
        else:
            # If no column found, check if index is datetime and reset it
            if isinstance(long_df.index, pd.DatetimeIndex):
                long_df = long_df.reset_index().rename(columns={'index': time_col})
                if verbose:
                    print(f"Reset datetime index and renamed to '{time_col}'.")
            else:
                raise ValueError(f"Specified time_col '{time_col}' not found. Tried alternatives: {possible_time_cols}. Ensure the DataFrame has a valid time column.")

    # Ensure time_col is datetime
    if not pd.api.types.is_datetime64_any_dtype(long_df[time_col]):
        long_df[time_col] = pd.to_datetime(long_df[time_col])

    if verbose:
        print(f"Generating future covariates from features: {time_feature_names} (horizon: {future_horizon}, unit: {future_horizon_unit})")
    
    extended_dfs = []
    grouped = long_df.groupby(group_col_names)
    
    for group_key, group_df in grouped:
        # Get historical timestamps, sort them
        hist_ts = sorted(group_df[time_col].unique())
        last_ts = hist_ts[-1]
        
        # Generate future timestamps
        if future_horizon_unit == 'periods':
            if not isinstance(future_horizon, int) or future_horizon <= 0:
                raise ValueError("future_horizon must be a positive integer for 'periods'.")
            # The recommended and most robust way
            future_ts = pd.date_range(start=last_ts + pd.to_timedelta(freq), periods=future_horizon, freq=freq)
        elif future_horizon_unit == 'end_timestamp':
            if not isinstance(future_horizon, pd.Timestamp):
                raise ValueError("future_horizon must be a pd.Timestamp for 'end_timestamp'.")
            if future_horizon <= last_ts:
                raise ValueError("future_horizon end_timestamp must be after the last historical timestamp.")
            future_ts = pd.date_range(start=last_ts + pd.Timedelta(1, freq), end=future_horizon, freq=freq)
        else:
            raise ValueError("future_horizon_unit must be 'periods' or 'end_timestamp'.")
        
        # Full timestamps: historical + future
        full_ts = pd.DatetimeIndex(hist_ts).union(future_ts)
        
        # Create extended DF for this group
        extended_group_df = pd.DataFrame({time_col: full_ts})
        
        # Add group_col_names (repeat for all rows)
        group_key_list = group_key if isinstance(group_key, tuple) else (group_key,)
        for g_col, g_val in zip(group_col_names, group_key_list):
            extended_group_df[g_col] = g_val
        
        # Calculate time-based features for full timestamps
        for feature in time_feature_names:
            if feature == 'minute':
                extended_group_df[feature] = extended_group_df[time_col].dt.minute
            elif feature == 'hour':
                extended_group_df[feature] = extended_group_df[time_col].dt.hour
            elif feature == 'weekday':
                extended_group_df[feature] = extended_group_df[time_col].dt.weekday
            elif feature == 'dayofyear':
                extended_group_df[feature] = extended_group_df[time_col].dt.dayofyear
            elif feature == 'weekofyear':
                extended_group_df[feature] = extended_group_df[time_col].dt.isocalendar().week
            elif feature == 'month':
                extended_group_df[feature] = extended_group_df[time_col].dt.month
            elif feature == 'quarter':
                extended_group_df[feature] = extended_group_df[time_col].dt.quarter
            elif feature == 'is_daytime':
                extended_group_df[feature] = ((extended_group_df[time_col].dt.hour >= 6) & (extended_group_df[time_col].dt.hour < 18)).astype(int)
            else:
                raise ValueError(f"Unsupported time feature: '{feature}'. Supported: 'minute', 'hour', 'weekday', 'dayofyear', 'weekofyear', 'month', 'quarter'.")
            #convert feature into sine and cosine components to capture cyclic nature
            if feature in ['minute', 'hour', 'weekday', 'dayofyear', 'weekofyear', 'month', 'quarter']:
                period = {'minute': 60,'hour': 24,'weekday': 7,'dayofyear': 365,'weekofyear': 52,'month': 12,'quarter': 4}[feature]
                extended_group_df[f'{feature}_sin'] = np.sin(2 * np.pi * extended_group_df[feature] / period)
                extended_group_df[f'{feature}_cos'] = np.cos(2 * np.pi * extended_group_df[feature] / period)
                
        extended_dfs.append(extended_group_df)
        
    for feature in time_feature_names:
        if feature in ['minute', 'hour', 'weekday', 'dayofyear', 'weekofyear', 'month', 'quarter']:
            time_feature_names.extend([f'{feature}_sin', f'{feature}_cos'])
    
    # Concatenate all extended groups
    extended_df = pd.concat(extended_dfs, ignore_index=True)
    print('columns in extended df:', extended_df.columns.tolist())
    
    #plot distribution of time features
    if verbose:
        # Plot distributions of all time features in a single row of subplots
        n_features = len(time_feature_names)
        fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 4), squeeze=False)
        axes = axes.flatten()
        for i, feature in enumerate(time_feature_names):
            sns.histplot(extended_df[feature], bins=30, kde=False, ax=axes[i])
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Count')
        plt.tight_layout()
        plt.show()
    
    # Create future covariate TimeSeries
    future_covariates_list = TimeSeries.from_group_dataframe(
        extended_df,
        time_col=time_col,
        group_cols=group_col_names,
        value_cols=time_feature_names,
        freq=freq,
        fill_missing_dates=True,
        fillna_value=None,  # Time features are computed, so no fill value needed
        verbose=False
    )
    
    if verbose:
        print(f"Generated {len(future_covariates_list)} future covariate TimeSeries.")
    
    return future_covariates_list


# def generate_past_covariates(
#     lags:List[int],
#     windows:List[int],
#     window_stats:List[str],
#     wide_df: pd.DataFrame,
#     group_col_names: List[str],
#     time_col: str,
#     freq: str,
#     verbose: bool = True
# ) -> List[TimeSeries]:
#     """
#     generates a range of additional Features from the timestamp column to be used as past covariates.
#     Possible features are:
#     - Lag features with various lags
#     - Rolling statistics (mean, std, min, max) over various windows
    

#     Args:
#         lags (List[int]): List of integer lags to create lag features for.
#         windows (List[int]): List of integer window sizes for rolling statistics.
#         window_stats (List[str]): List of statistics to compute over rolling windows. Options: 'mean', 'std', 'min', 'max'.
#         wide_df (pd.DataFrame): Input wide-format DataFrame with time series data.
#         group_col_names (List[str]): Columns that uniquely identify each time series.
#         time_col (str): Name of the timestamp column.
#         freq (str): Frequency string (e.g., 'H' for hourly) for resampling.
#         verbose (bool, optional): If True, prints progress information. Defaults to True.

#     Returns:
#         List[TimeSeries]: _description_
#     """

#     assert all(isinstance(lag, int) and lag > 0 for lag in lags), "All lags must be positive integers."
#     assert all(isinstance(window, int) and window > 0 for window in windows), "All windows must be positive integers."
#     assert all(stat in ['mean', 'std', 'min', 'max'] for stat in window_stats), "window_stats must be from ['mean', 'std', 'min', 'max']"
    
#     if verbose:
#         print(f"Generating past covariates with lags: {lags}, windows: {windows}, stats: {window_stats}")
#     extended_dfs = []
#     columns = [col for col in wide_df.columns if col not in group_col_names + [time_col]]
#     covars_df = pd.DataFrame(index=wide_df.index)
#     for col in columns:
#         if lags:
#             for lag in lags:
#                 covars_df[f'{col}_lag_{lag}'] = wide_df.groupby(group_col_names)[col].shift(lag)
#         if windows:
#             for window in windows:
#                 for stat in window_stats:
#                     if stat == 'mean':
#                         covars_df[f'{col}_roll_mean_{window}'] = wide_df.groupby(group_col_names)[col].shift(1).rolling(window=window, min_periods=1).mean().reset_index(level=group_col_names, drop=True)
#                     elif stat == 'std':
#                         covars_df[f'{col}_roll_std_{window}'] = wide_df.groupby(group_col_names)[col].shift(1).rolling(window=window, min_periods=1).std().reset_index(level=group_col_names, drop=True)
#                     elif stat == 'min':
#                         covars_df[f'{col}_roll_min_{window}'] = wide_df.groupby(group_col_names)[col].shift(1).rolling(window=window, min_periods=1).min().reset_index(level=group_col_names, drop=True)
#                     elif stat == 'max':
#                         covars_df[f'{col}_roll_max_{window}'] = wide_df.groupby(group_col_names)[col].shift(1).rolling(window=window, min_periods=1).max().reset_index(level=group_col_names, drop=True)
                    
        
        
        
def analyze_relationships(df, mi_bins=10, plot_matrix=True):
    """
    Erzeugt Pearson- und MI-Matrizen für das Dashboard.
    """
    import plotly.graph_objects as go
    import numpy as np
    from sklearn.metrics import mutual_info_score
    
    # 1. Datenbereinigung (nur Numerik, keine Konstanten)
    numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
    nunique = numeric_df.nunique()
    numeric_df = numeric_df.drop(columns=nunique[nunique <= 1].index)
    
    # 2. Pearson Korrelation
    corr_matrix = numeric_df.corr(method='pearson')
    
    # 3. Mutual Information Matrix
    cols = numeric_df.columns
    mi_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    
    for i in range(len(cols)):
        for j in range(i, len(cols)):
            c1, c2 = cols[i], cols[j]
            # Daten binned für MI Berechnung
            d1 = pd.cut(numeric_df[c1], bins=mi_bins, labels=False, duplicates='drop')
            d2 = pd.cut(numeric_df[c2], bins=mi_bins, labels=False, duplicates='drop')
            score = mutual_info_score(d1, d2)
            mi_matrix.loc[c1, c2] = score
            mi_matrix.loc[c2, c1] = score

    # 4. Heatmaps für das Dashboard erstellen
    def create_heatmap(matrix, title, colorscale):
        fig = go.Figure(data=go.Heatmap(
            z=matrix.values, x=matrix.columns, y=matrix.index,
            colorscale=colorscale, zmin=0 if "Information" in title else -1, zmax=1 if "Information" in title else 1
        ))
        fig.update_layout(title=title, template="plotly_white", height=500, width=600)
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    pearson_html = create_heatmap(corr_matrix, "Pearson Korrelation (Linear)", "RdBu")
    mi_html = create_heatmap(mi_matrix, "Mutual Information (Abhängigkeit)", "Viridis")
    
    return pearson_html, mi_html



def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Helper function to ensure a DataFrame has a DatetimeIndex."""
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    # Check for common datetime column names
    for col_name in ['datetime', 'DateTime']:
        if col_name in df.columns:
            # Convert column to datetime and set as index
            df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
            df = df.set_index(col_name)
            return df

    raise ValueError("DataFrame does not have a DatetimeIndex or a 'datetime'/'DateTime' column.")


def merge_and_interpolate(df1: pd.DataFrame,
                          df2: pd.DataFrame,
                          interpolate_method: str = 'linear') -> pd.DataFrame:
    """
    Merges two DataFrames with different sampling frequencies by interpolating the
    lower frequency DataFrame onto the higher frequency DataFrame's index.

    Args:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.
        interpolate_method (str): The interpolation method to use.
                                  Examples: 'linear', 'time', 'polynomial', 'spline'.
                                  Defaults to 'linear'.

    Returns:
        pd.DataFrame: The merged DataFrame with interpolated values.
    """
    # Ensure both dataframes have a datetime index
    print("Preparing DataFrames...")
    df1_processed = _prepare_df(df1)
    df2_processed = _prepare_df(df2)

    # Identify which DataFrame has the higher frequency (more rows)
    if len(df1_processed) >= len(df2_processed):
        df_high = df1_processed
        df_low = df2_processed
    else:
        df_high = df2_processed
        df_low = df1_processed

    # Reindex the low-frequency DataFrame to match the high-frequency index
    df_low_resampled = df_low.reindex(df_high.index)

    # Interpolate the missing values
    df_low_interpolated = df_low_resampled.interpolate(method=interpolate_method)

    # Fill any remaining NaNs at the beginning or end using the nearest value
    df_low_filled = df_low_interpolated.bfill().ffill()

    # Join the two DataFrames
    # Use suffixes if there are overlapping column names
    if not df_high.columns.intersection(df_low_filled.columns).empty:
        merged_df = df_high.join(df_low_filled, lsuffix='_high', rsuffix='_low')
    else:
        merged_df = df_high.join(df_low_filled)

    return merged_df


def split_dataframe_on_gaps(df, expected_freq=None, tolerance_factor=50):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a pandas DatetimeIndex.")
    
    df = df.sort_index()
    if len(df) < 2:
        return [df]

    # Calculate time differences
    diffs = df.index.to_series().diff()

    # Infer frequency if not provided
    if expected_freq is None:
        # Use median/mode of the non-null differences to find the 'normal' step
        clean_diffs = diffs.dropna()
        if clean_diffs.empty:
            return [df]
        expected_freq = clean_diffs.mode()[0]

    # Define the threshold for a gap
    if isinstance(expected_freq, str):
        expected_freq = pd.Timedelta(expected_freq)
        
    threshold = expected_freq * tolerance_factor
    
    # Identify where the gap is larger than the threshold
    is_gap = diffs > threshold
    
    # Create group IDs (increment every time a gap is found)
    group_ids = is_gap.cumsum()
    
    # Split data
    splits = [group for _, group in df.groupby(group_ids)]
    
    return splits


def add_ramp_offset_to_timeseries(df, target_col, start_ts, offset_value, offset_duration):
    total_steps = len(df)
    # index position that is 20 % from the END  (0-based)
    ramp_start_idx = df.index.get_loc(start_ts)
    #print('original column:\n',df[target_col].iloc[ramp_start_idx:ramp_start_idx+50])
    #print(f'start_ts for ramp offset: {start_ts}')
    # 5 days later (frequency-aware)
    end_ts = start_ts + offset_duration
    # ------------------------------------------------------------------
    # 2. Build the offset series (same index as df_select)
    # ------------------------------------------------------------------
    # initialise with zeros
    offset = pd.Series(0.0, index=df.index, dtype=float)
    # mask for the ramp interval
    ramp_mask = (df.index >= start_ts) & (df.index <= end_ts)
    # time distance from start_ts (in hours – works for any freq)
    time_diff = (df.index[ramp_mask] - start_ts).total_seconds() / 3600.0
    total_hours = offset_duration.total_seconds() / 3600.0
    # linear ramp: 0 → +20 over the 5-day window
    ramp_values = offset_value * (time_diff / total_hours)
    offset.loc[ramp_mask] = ramp_values
    # ------------------------------------------------------------------
    # 3. Apply the offset to the target column
    # ------------------------------------------------------------------
    # safety check
    if target_col not in df.columns:
        raise KeyError(f"Column '{target_col}' not found in df_select")
    df[target_col] = df[target_col] + offset
    #print 50 values of the modified target column around the ramp
    ramp_start_idx = df.index.get_loc(start_ts)

    #print('modified column:\n',df[target_col].iloc[ramp_start_idx:ramp_start_idx+50])
    return df

def visualize_ramp_predictions(
    long_df_list: Union[pd.DataFrame, List[pd.DataFrame]],
    ramp_dict: dict,
    model,
    target_scaler=None,
    static_transformer=None,
    freq: str = '4h',
    time_feature_names: List[str] = ['hour', 'is_daytime', 'weekday', 'dayofyear', 'weekofyear', 'month', 'quarter'], 
    future_horizon: int = None,
    plot_margin_factor: float = 0.5,
    n_samples: int = 100,
    # UPDATED: Default to None so we don't accidentally drop columns
    static_cols_order: List[str] = ['series_id','client_id', 'installation_id'],
):
    # ---------------------------------------------------------
    # 1. Efficient Selection & Cleaning
    # ---------------------------------------------------------
    client_id = ramp_dict.get('client_id')
    installation_id = ramp_dict.get('installation_id')
    target_series_names = [k for k in ramp_dict.keys() if k not in ['client_id', 'installation_id']]
    
    print(f"--- Filtering data for Client: {client_id}, Series: {target_series_names} ---")

    if isinstance(long_df_list, pd.DataFrame):
        dfs_to_process = [long_df_list]
    else:
        dfs_to_process = long_df_list

    filtered_chunks = []
    
    for df in dfs_to_process:
        df = df.loc[:, ~df.columns.duplicated()]

        # Clean Identifiers
        if 'client_id' in df.columns:
            df['client_id'] = df['client_id'].astype(str).str.strip()
        if 'series_id' in df.columns:
            df['series_id'] = df['series_id'].astype(str).str.strip()
        if 'installation_id' in df.columns:
             df['installation_id'] = df['installation_id'].astype(str).str.strip()

        if 'client_id' in df.columns:
            if df.empty or df['client_id'].iloc[0] != str(client_id).strip():
                continue
        
        mask = df['series_id'].isin(target_series_names)
        if installation_id and 'installation_id' in df.columns:
            # Cast installation_id to string for comparison safety
            mask = mask & (df['installation_id'].astype(str) == str(installation_id))
            
        chunk = df[mask].copy()
        if not chunk.empty:
            filtered_chunks.append(chunk)

    if not filtered_chunks:
        raise ValueError("No data found matching the client/series in ramp_dict.")

    filtered_long_df = pd.concat(filtered_chunks, ignore_index=True)

    # ---------------------------------------------------------
    # 2. Convert to Darts TimeSeries
    # ---------------------------------------------------------
    group_cols = ['client_id', 'series_id']
    if installation_id and 'installation_id' in filtered_long_df.columns:
        group_cols.append('installation_id')
    
    # Determine the final order of static columns
    # If user provided specific order, use it. Otherwise use the natural grouping order.
    if static_cols_order:
        final_static_cols = static_cols_order
    else:
        final_static_cols = group_cols

    filtered_long_df = filtered_long_df.sort_values(by=group_cols + ['time_stamp'])
    
    ts_list_raw = [] 
    future_covs_list = [] if time_feature_names else None

    for ids, group in filtered_long_df.groupby(group_cols):
        print(f"Processing TimeSeries for IDs: {ids}")
        if not isinstance(ids, tuple):
            ids = (ids,)
        
        # Create map of all available static data
        static_data_map = {col: val for col, val in zip(group_cols, ids)}
        
        # Select only the columns needed, in the correct order
        # We fill missing keys with 'Unknown' to prevent crashes if a column is missing
        ordered_static_data = {}
        for k in final_static_cols:
            if k in static_data_map:
                ordered_static_data[k] = static_data_map[k]
            else:
                # Fallback if installation_id is expected by transformer but missing in group
                ordered_static_data[k] = "Unknown" 

        static_df = pd.DataFrame([ordered_static_data])

        ts = TimeSeries.from_dataframe(
            group, 
            time_col='time_stamp', 
            value_cols='value', 
            freq=freq,
            fill_missing_dates=True,
            fillna_value=None 
        )
        
        ts = ts.with_static_covariates(static_df)
        ts_list_raw.append(ts)

    # ---------------------------------------------------------
    # 3. Generate Future Covariates
    # ---------------------------------------------------------
    future_covs_list = None
    if time_feature_names:
        # We pass the same dataframe and the same grouping columns.
        # This guarantees the output list aligns 1:1 with ts_list_raw 
        # and has the same static covariates structure.
        future_covs_list = generate_future_covariates(
            long_df=filtered_long_df,
            time_feature_names=time_feature_names,
            group_col_names=static_cols_order, # <--- IMPORTANT: Matches Target Grouping
            time_col='time_stamp',
            freq=freq,
            future_horizon=future_horizon,
            verbose=False
        )
        
        if len(future_covs_list) != len(ts_list_raw):
            raise ValueError("Covariates generation mismatch! Different number of series generated.")
        
    # ---------------------------------------------------------
    # 3. Apply Transformations
    # ---------------------------------------------------------
    ts_list_input = ts_list_raw.copy() 
    ts_list_input = [ts.astype(np.float32) for ts in ts_list_input]

    if static_transformer:
        print(f"--- Transforming Static Covariates (Cols: {final_static_cols}) ---")
        try:
            ts_list_input = static_transformer.transform(ts_list_input)
        except Exception as e:
            print("\n!!! TRANSFORM ERROR !!!")
            print(f"Your transformer expects specific columns.")
            print(f"You provided: {final_static_cols}")
            print(f"Try changing 'static_cols_order' to match training (e.g. swap client/series or add installation_id).")
            raise e
        
    future_covs_input = None
    if future_covs_list:    
        transformer = StaticCovariatesTransformer()
        future_covs_input = transformer.fit_transform(future_covs_list)
        future_covs_input = [cov.astype(np.float32) for cov in future_covs_input]
        
    if target_scaler:
        ts_list_input = target_scaler.transform(ts_list_input)

    # ---------------------------------------------------------
    # 4. Predict and Visualize
    # ---------------------------------------------------------
    figs = []
    for i, (series_raw, series_model_input) in enumerate(zip(ts_list_raw, ts_list_input)):
        
        try:
            curr_series_name = series_raw.static_covariates['series_id'].iloc[0]
        except KeyError:
            curr_series_name = list(ramp_dict.keys())[i]

        if curr_series_name not in ramp_dict:
            print(f"Skipping {curr_series_name} (Not found in ramp_dict).")
            continue

        print(f"--- Processing Ramp: {curr_series_name} ---")
        
        start_ts, offset_val, duration = ramp_dict[curr_series_name]
        end_ts = start_ts + duration
        
        series_covs = future_covs_input[i] if future_covs_input else None

        try:
            train_input, _ = series_model_input.split_before(start_ts)
        except ValueError:
            print(f"Skipping {curr_series_name}: Start timestamp {start_ts} out of bounds.")
            continue

        prediction_horizon_ts = duration + (duration * 0.2)
        dummy_slice = series_model_input.slice(start_ts, start_ts + prediction_horizon_ts)
        n_steps = len(dummy_slice)

        predict_kwargs = {
            'n': n_steps,
            'series': train_input,
            'num_samples': n_samples,
        }
        if model.supports_future_covariates and series_covs:
            predict_kwargs['future_covariates'] = series_covs
            
        pred_scaled = model.predict(**predict_kwargs)
        print(f'prediction start: {pred_scaled.start_time()}, end: {pred_scaled.end_time()}')
        print(f'ramp start: {start_ts}, end: {end_ts}')

        if target_scaler:
            pred = target_scaler.inverse_transform(pred_scaled)
            actual_series = series_raw
            hist_unscaled, _ = actual_series.split_before(start_ts)
        else:
            pred = pred_scaled
            actual_series = series_model_input
            hist_unscaled = train_input

        # ---------------------------
        # C. Plotting with Multiple CIs
        # ---------------------------
        margin = duration * plot_margin_factor
        plot_start = start_ts - margin
        plot_end = end_ts + margin
        
        fig = plt.figure(figsize=(14, 6))
        
        # 1. Plot Actuals
        try:
            actual_series.slice(plot_start, plot_end).plot(label='Actual', color='black')
        except:
            actual_series.plot(label='Actual (Full)', color='black')

        # 2. Plot Prediction Intervals (Layered from widest to narrowest)
        # Note: We set label=None for the outer ones to keep legend clean, or label explicitly.
        
        # 80% CI (10th - 90th percentile) - Lightest Blue
        pred.plot(
            low_quantile=0.10, high_quantile=0.90, 
            color='tab:red', alpha=0.15, label='80% CI'
        )
        
        # 50% CI (25th - 75th percentile) - Medium Blue
        pred.plot(
            low_quantile=0.25, high_quantile=0.75, 
            color='tab:purple', alpha=0.25, label='50% CI'
        )
        
        # 20% CI (40th - 60th percentile) - Darkest Blue
        pred.plot(
            low_quantile=0.40, high_quantile=0.60, 
            color='tab:blue', alpha=0.40, label='20% CI'
        )

        # 3. Plot Median Line (Solid) on top
        pred.plot(low_quantile=0.5, high_quantile=0.5, color='tab:blue', lw=2, label='Prediction (Median)')

        # 4. Connector Line (Gap Filling)
        last_hist_val = hist_unscaled.last_value()
        last_hist_ts = hist_unscaled.end_time()
        first_pred_val = pred.values()[0][0] # Get median of first prediction step
        # Ideally get median: pred.quantile(0.5).values()[0][0]
        # But for simple visualization, simple indexing often works if n_samples=1, 
        # otherwise specifically calculate median:
        first_pred_median = np.median(pred.values()[0])

        plt.plot(
            [last_hist_ts, pred.start_time()], 
            [last_hist_val, first_pred_median], 
            color='tab:blue', linestyle=':', alpha=0.8
        )

        plt.axvline(start_ts, color='red', linestyle='--', alpha=0.7, label='Ramp Start')
        plt.axvspan(start_ts, end_ts, color='orange', alpha=0.1, label='Ramp Duration')
        plt.title(f"Prediction vs Ramp: {curr_series_name}\nTarget Offset: {offset_val}")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        figs.append(fig)
        
    return figs