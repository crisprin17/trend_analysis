import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import norm, linregress, t
from scipy.signal import savgol_filter
from statsmodels.tsa.seasonal import STL

class TrendResult:
    def __init__(self, direction, strength, confidence, details=None):
        """
        Encapsulate trend analysis result with direction, strength, and confidence.
        
        Parameters:
        - direction: "Upward", "Downward", or "No Trend"
        - strength: "Strong", "Moderate", "Weak"
        - confidence: float between 0 and 1
        - details: dictionary with additional metrics
        """
        self.direction = direction
        self.strength = strength
        self.confidence = confidence
        self.details = details or {}
    
    def __str__(self):
        return f"{self.strength} {self.direction} (conf: {self.confidence:.2f})"
    
    def to_dict(self):
        return {
            'direction': self.direction,
            'strength': self.strength,
            'confidence': self.confidence,
            'details': self.details
        }

def classify_trend_strength(slope, std_err, mean_value, relative_change):
    """
    Classify trend strength based on multiple criteria.
    """
    # Normalized slope (change per day relative to mean)
    norm_slope = abs(slope) / mean_value if mean_value > 0 else 0
    
    # T-statistic for slope
    t_stat = abs(slope) / std_err if std_err > 0 else 0
    
    if t_stat > 2.576 and norm_slope > 0.05 and relative_change > 0.15:  # 99% confidence
        return "Strong"
    elif t_stat > 1.96 and norm_slope > 0.02 and relative_change > 0.10:  # 95% confidence
        return "Moderate"
    elif t_stat > 1.645 and norm_slope > 0.01 and relative_change > 0.05:  # 90% confidence
        return "Weak"
    else:
        return "None"

def calculate_trend_confidence(p_value, variance_ratio, runs_z_score, relative_change):
    """
    Calculate overall confidence in trend detection.
    """
    # Convert p-value to confidence score (1 - p_value)
    p_confidence = max(0, 1 - p_value)
    
    # Convert variance ratio to confidence score
    var_confidence = min(1, variance_ratio * 2)  # Scale up to 1
    
    # Convert runs test to confidence score
    runs_confidence = max(0, 1 - abs(runs_z_score) / 3)  # Scale z-score to 0-1
    
    # Convert relative change to confidence score
    change_confidence = min(1, relative_change * 5)  # Scale up to 1
    
    # Weighted average of confidence scores
    weights = [0.4, 0.3, 0.1, 0.2]  # More weight to p-value and variance ratio
    confidence_scores = [p_confidence, var_confidence, runs_confidence, change_confidence]
    
    return np.average(confidence_scores, weights=weights)

def decompose_time_series(data, value_column="event_value", period=7):
    """
    Decompose time series into trend, seasonal, and residual components.
    Handles cases with no seasonality.
    """
    # Handle missing values if any
    values = pd.Series(data[value_column]).fillna(method='ffill').fillna(method='bfill')
    
    # Check if we have enough data points for decomposition
    if len(values) < 2 * period:
        # Return simple trend if not enough data
        trend = values.rolling(window=min(3, len(values)), center=True).mean()
        return {
            'trend': trend,
            'seasonal': pd.Series(np.zeros_like(values)),
            'residual': values - trend,
            'strength': {
                'seasonal': 0.0,
                'trend': max(0, 1 - np.var(values - trend) / np.var(values)) if np.var(values) > 0 else 0
            }
        }
    
    # Apply STL decomposition
    try:
        stl = STL(values, period=period)
        result = stl.fit()
        
        # Calculate variances
        total_var = np.var(values)
        seasonal_var = np.var(result.seasonal)
        residual_var = np.var(result.resid)
        trend_var = np.var(result.trend)
        
        # Calculate strength metrics safely
        if seasonal_var > 0:
            seasonal_strength = max(0, 1 - residual_var / seasonal_var)
        else:
            seasonal_strength = 0.0
        
        if trend_var > 0:
            trend_strength = max(0, 1 - residual_var / trend_var)
        else:
            trend_strength = 0.0
        
        return {
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.resid,
            'strength': {
                'seasonal': seasonal_strength,
                'trend': trend_strength
            }
        }
    
    except Exception as e:
        # Fallback to simple trend if decomposition fails
        trend = values.rolling(window=min(3, len(values)), center=True).mean()
        return {
            'trend': trend,
            'seasonal': pd.Series(np.zeros_like(values)),
            'residual': values - trend,
            'strength': {
                'seasonal': 0.0,
                'trend': max(0, 1 - np.var(values - trend) / np.var(values)) if np.var(values) > 0 else 0
            }
        }

def detect_trend(data, value_column="event_value", confidence_level=0.95):
    """
    Detect trend using three core statistical tests with seasonal decomposition:
    1. Slope Significance Test (on deseasonalized data)
    2. Variance Analysis
    3. Runs Test for Randomness
    """
    # First, decompose the time series
    decomposition = decompose_time_series(data, value_column)
    
    # Use deseasonalized data (trend + residual) for slope test
    deseasonalized = decomposition['trend'] + decomposition['residual']
    y = np.array(deseasonalized)
    x = np.arange(len(y))
    
    # Calculate first and last points for trend direction
    first_third = y[:len(y)//3].mean()
    last_third = y[-len(y)//3:].mean()
    overall_trend = last_third - first_third
    
    # Perform regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    mean_value = np.mean(y)
    
    # Calculate relative slope using absolute values
    slope_relative = abs(slope) / abs(mean_value) if mean_value != 0 else 0
    slope_significant = p_value <= 0.05
    slope_negligible = slope_relative < 0.01
    
    # Variance Analysis
    trend_var = np.var(decomposition['trend'])
    total_var = np.var(data[value_column])
    variance_ratio = trend_var / total_var if total_var > 0 else 0
    
    variance_significant = variance_ratio >= 0.3
    high_noise = variance_ratio < 0.2
    
    # Runs Test
    median = np.median(y)
    above_median = y > median
    
    if len(y) == 0 or np.all(y == y[0]):
        runs_significant = False
        random_pattern = True
        runs_z_score = 0
    else:
        runs = np.diff(above_median).astype(bool).sum() + 1
        n1 = np.sum(above_median)
        n2 = len(y) - n1
        
        if n1 == 0 or n2 == 0:
            runs_significant = True
            random_pattern = False
            runs_z_score = 3.0
        else:
            runs_mean = (2 * n1 * n2) / (n1 + n2) + 1
            runs_var = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1))
            runs_z_score = (runs - runs_mean) / np.sqrt(runs_var) if runs_var > 0 else 0
            runs_significant = abs(runs_z_score) >= 1.96
            random_pattern = abs(runs_z_score) < 1.645
    
    # Calculate evidence for no trend
    no_trend_evidence = [
        slope_negligible,
        not slope_significant,
        high_noise,
        random_pattern
    ]
    no_trend_score = sum(no_trend_evidence)
    
    # Calculate significance scores
    significance_scores = {
        'slope': min(1.0, slope_relative / 0.05) if slope_significant else 0,
        'variance': min(1.0, variance_ratio / 0.5) if variance_significant else 0,
        'runs': min(1.0, abs(runs_z_score) / 2.5) if runs_significant else 0
    }
    
    # Test weights
    test_weights = {
        'slope': 0.4,
        'variance': 0.4,
        'runs': 0.2
    }
    
    # Calculate confidence
    confidence = sum(
        score * test_weights[test]
        for test, score in significance_scores.items()
    )
    
    # Determine trend direction using both slope and overall trend
    if no_trend_score >= 3:
        direction = "No Trend"
        strength = "None"
        confidence = max(0.8, 1 - confidence)
    else:
        if confidence > 0.2:
            # Use overall trend for direction instead of just slope
            direction = "Upward" if overall_trend > 0 else "Downward"
            if confidence > 0.7:
                strength = "Strong"
            elif confidence > 0.4:
                strength = "Moderate"
            else:
                strength = "Weak"
        else:
            direction = "No Trend"
            strength = "None"
            confidence = 0.5
    
    # Compile details
    details = {
        'decomposition': decomposition['strength'],
        'tests': {
            'slope': {
                'value': slope,
                'relative': slope_relative,
                'p_value': p_value,
                'significant': slope_significant,
                'negligible': slope_negligible,
                'score': significance_scores['slope']
            },
            'variance': {
                'ratio': variance_ratio,
                'significant': variance_significant,
                'high_noise': high_noise,
                'score': significance_scores['variance']
            },
            'runs': {
                'z_score': runs_z_score,
                'significant': runs_significant,
                'random': random_pattern,
                'score': significance_scores['runs']
            }
        },
        'no_trend_evidence': {
            'score': no_trend_score,
            'criteria': {
                'slope_negligible': slope_negligible,
                'not_significant': not slope_significant,
                'high_noise': high_noise,
                'random_pattern': random_pattern
            }
        },
        'test_scores': significance_scores
    }
    
    return TrendResult(direction, strength, confidence, details)

def calculate_loess_trend(data, value_column="event_value", frac=0.1):
    """
    Calculate trend using LOESS smoothing with seasonal adjustment.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data containing timestamps and values
    value_column : str, default="event_value"
        Name of the column containing the values to analyze
    frac : float, default=0.1
        Smoothing parameter for LOESS. Lower values (0.1) provide better detection accuracy
        based on empirical analysis.
    """
    data = data.copy()
    
    # First decompose the series
    decomposition = decompose_time_series(data, value_column)
    
    # Use deseasonalized data
    deseasonalized = decomposition['trend'] + decomposition['residual']
    
    # Create temporary dataframe with deseasonalized data
    temp_data = data.copy()
    temp_data[value_column] = deseasonalized
    
    # Get trend detection on deseasonalized data
    trend_result = detect_trend(temp_data, value_column)
    
    # Enhanced trend detection logic based on empirical analysis
    if trend_result.direction == "No Trend":
        trend_values = decomposition['trend']
        derivative = np.gradient(trend_values)
        mean_derivative = np.mean(derivative)
        derivative_std = np.std(derivative)
        trend_strength = decomposition['strength']['trend']
        
        # More sensitive threshold for weak trends when trend strength is high
        if trend_strength > 0.95:  # High trend strength threshold
            if abs(mean_derivative) > derivative_std:
                trend_result.direction = "Upward" if mean_derivative > 0 else "Downward"
                trend_result.strength = "Weak"
                trend_result.confidence = max(0.3, min(0.5, trend_strength - 0.5))
        else:
            # Original more conservative threshold for lower trend strengths
            if abs(mean_derivative) > 1.5 * derivative_std:
                trend_result.direction = "Upward" if mean_derivative > 0 else "Downward"
                trend_result.strength = "Weak"
                trend_result.confidence = 0.3
    
    # Add decomposition details
    trend_result.details['decomposition'] = {
        'seasonal_strength': decomposition['strength']['seasonal'],
        'trend_strength': decomposition['strength']['trend']
    }
    
    return trend_result

def log_likelihood_ratio_test(data, value_column="event_value"):
    """Calculate trend using LLR test with seasonal adjustment."""
    data = data.copy()
    
    # First decompose the series
    decomposition = decompose_time_series(data, value_column)
    
    # Use deseasonalized data
    deseasonalized = decomposition['trend'] + decomposition['residual']
    
    # Create temporary dataframe with deseasonalized data
    temp_data = data.copy()
    temp_data[value_column] = deseasonalized
    
    # Get trend detection on deseasonalized data
    trend_result = detect_trend(temp_data, value_column)
    
    # If no trend detected, check residuals
    if trend_result.direction == "No Trend":
        x = np.arange(len(deseasonalized))
        y = deseasonalized
        
        slope_up, intercept_up = np.polyfit(x, y, 1)
        slope_down, intercept_down = np.polyfit(x, -y, 1)
        slope_down = -slope_down
        
        residuals_up = y - (slope_up * x + intercept_up)
        residuals_down = y - (slope_down * x + intercept_down)
        
        ssr_up = np.sum(residuals_up ** 2)
        ssr_down = np.sum(residuals_down ** 2)
        
        if abs(ssr_up - ssr_down) > 0.2 * np.mean([ssr_up, ssr_down]):
            trend_result.direction = "Upward" if ssr_up < ssr_down else "Downward"
            trend_result.strength = "Weak"
            trend_result.confidence = 0.3
    
    # Add decomposition details
    trend_result.details['decomposition'] = {
        'seasonal_strength': decomposition['strength']['seasonal'],
        'trend_strength': decomposition['strength']['trend']
    }
    
    return trend_result

def combine_trend_results(loess_result, llr_result, historical_results=None):
    """Combine results with emphasis on no-trend detection."""
    # Use test scores from both methods
    loess_scores = loess_result.details['test_scores']
    llr_scores = llr_result.details['test_scores']
    
    # Calculate combined scores
    combined_scores = {
        test: max(loess_scores[test], llr_scores[test])
        for test in ['slope', 'variance', 'runs']
    }
    
    # Use test weights from detect_trend
    test_weights = {
        'slope': 0.5,
        'variance': 0.3,
        'runs': 0.2
    }
    
    # Calculate combined confidence
    combined_confidence = sum(
        score * test_weights[test]
        for test, score in combined_scores.items()
    )
    
    # Check no-trend evidence from both methods
    loess_no_trend = loess_result.details['no_trend_evidence']['score']
    llr_no_trend = llr_result.details['no_trend_evidence']['score']
    
    # Strong evidence for no trend if either method suggests it
    if max(loess_no_trend, llr_no_trend) >= 3:
        direction = "No Trend"
        strength = "None"
        combined_confidence = 0.8  # High confidence in no trend
    else:
        # Determine direction and strength
        if combined_confidence < 0.3:  # Increased threshold for trend detection
            direction = "No Trend"
            strength = "None"
            combined_confidence = 0.6  # Moderate confidence in no trend
        else:
            # Use direction from method with higher confidence
            if loess_result.confidence >= llr_result.confidence:
                direction = loess_result.direction
            else:
                direction = llr_result.direction
            
            # Determine strength based on combined confidence
            if combined_confidence > 0.7:
                strength = "Strong"
            elif combined_confidence > 0.5:
                strength = "Moderate"
            else:
                strength = "Weak"
    
    # Compile details
    details = {
        'test_scores': combined_scores,
        'method_scores': {
            'loess': loess_result.confidence,
            'llr': llr_result.confidence
        },
        'no_trend_evidence': {
            'loess_score': loess_no_trend,
            'llr_score': llr_no_trend
        },
        'method_agreement': loess_result.direction == llr_result.direction
    }
    
    return TrendResult(direction, strength, combined_confidence, details)

def analyze_trend(user_data, value_column="event_value", window_size=7):
    """Analyze trends with improved ensemble method combination."""
    user_data = user_data.copy()
    user_data["local_date"] = pd.to_datetime(user_data["local_date"])
    
    if len(user_data) < window_size * 2:
        raise ValueError("At least two weeks of data required")
    
    trend_results = []
    historical_results = []  # Store previous results for context
    current_date = user_data["local_date"].min() + pd.Timedelta(days=window_size * 2)
    
    while current_date <= user_data["local_date"].max():
        interval_data = user_data[user_data["local_date"] < current_date].copy()
        first_week = interval_data.iloc[:window_size]
        second_week = interval_data.iloc[window_size:window_size * 2]
        
        # Get individual method results
        loess_result = calculate_loess_trend(second_week, value_column)
        llr_result = log_likelihood_ratio_test(second_week, value_column)
        
        # Combine results using ensemble method
        combined_result = combine_trend_results(
            loess_result, 
            llr_result,
            historical_results
        )
        
        # Store results
        result_entry = {
            'week_start': second_week['local_date'].min(),
            'week_end': second_week['local_date'].max(),
            'loess_result': loess_result.to_dict(),
            'llr_result': llr_result.to_dict(),
            'combined_result': combined_result.to_dict(),
            'final_direction': combined_result.direction,
            'final_strength': combined_result.strength,
            'confidence': combined_result.confidence,
            'methods_agree': loess_result.direction == llr_result.direction
        }
        
        trend_results.append(result_entry)
        historical_results.append(combined_result)
        
        current_date += pd.Timedelta(days=window_size)
    
    return pd.DataFrame(trend_results)

def get_trend_summary(trend_results):
    """
    Generate a human-readable summary of trend analysis results.
    
    Parameters:
    - trend_results: DataFrame from analyze_trend
    
    Returns:
    - str: Summary of trend patterns
    """
    summaries = []
    
    for _, result in trend_results.iterrows():
        combined = result['combined_result']
        confidence_level = "high" if combined['confidence'] > 0.8 else \
                         "moderate" if combined['confidence'] > 0.6 else "low"
        
        summary = f"Week of {result['week_start'].strftime('%Y-%m-%d')}: "
        
        if combined['direction'] == "No Trend":
            summary += f"No significant trend detected (confidence: {confidence_level})"
        else:
            summary += f"{combined['strength']} {combined['direction'].lower()} trend "
            summary += f"detected (confidence: {confidence_level})"
        
        if not result['methods_agree']:
            summary += " - Methods disagree, further review recommended"
        
        summaries.append(summary)
    
    return "\n".join(summaries)

# Example usage:
# analyzed_trends = analyze_trend(user1_data)
# print(analyzed_trends)
