import pandas as pd
import numpy as np
from trend_analysis import calculate_loess_trend
import time
import matplotlib.pyplot as plt
from datetime import datetime

def generate_synthetic_data():
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    
    # Generate different trend patterns
    upward = [50 + i*0.1 + np.random.normal(0, 2) for i in range(365)]
    downward = [100 - i*0.15 + np.random.normal(0, 3) for i in range(365)]
    cyclic = [75 + 20*np.sin(i/30) + i*0.05 + np.random.normal(0, 2) for i in range(365)]
    stable = [80 + np.random.normal(0, 4) for _ in range(365)]
    
    datasets = {
        'upward': pd.DataFrame({'timestamp': dates, 'event_value': upward}),
        'downward': pd.DataFrame({'timestamp': dates, 'event_value': downward}),
        'cyclic': pd.DataFrame({'timestamp': dates, 'event_value': cyclic}),
        'stable': pd.DataFrame({'timestamp': dates, 'event_value': stable})
    }
    return datasets

def evaluate_trend(data, pattern_name, frac):
    start_time = time.time()
    result = calculate_loess_trend(data, value_column='event_value', frac=frac)
    execution_time = time.time() - start_time
    
    metrics = {
        'pattern': pattern_name,
        'frac': frac,
        'direction': result.direction,
        'strength': result.strength,
        'confidence': result.confidence,
        'execution_time': execution_time,
        'trend_strength': result.details['decomposition']['trend_strength'],
        'seasonal_strength': result.details['decomposition']['seasonal_strength']
    }
    
    return metrics

def generate_report():
    # Test different frac values
    frac_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    datasets = generate_synthetic_data()
    all_results = []
    
    for frac in frac_values:
        for pattern_name, data in datasets.items():
            metrics = evaluate_trend(data, pattern_name, frac)
            all_results.append(metrics)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Generate report
    report = f"""
LOESS Parameter Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

1. Performance Analysis by Smoothing Parameter (frac)
{'-'*80}
{results_df.groupby('frac')['execution_time'].agg(['mean', 'std', 'min', 'max']).round(4).to_string()}

2. Trend Detection Accuracy
{'-'*80}
Pattern-wise results for different frac values:
{results_df.pivot_table(
    index='pattern', 
    columns='frac',
    values=['direction', 'strength', 'confidence'],
    aggfunc='first'
).round(3).to_string()}

3. Trend Strength Analysis
{'-'*80}
Average trend strength by pattern and frac value:
{results_df.pivot_table(
    index='pattern',
    columns='frac',
    values='trend_strength',
    aggfunc='first'
).round(3).to_string()}

4. Key Findings
{'-'*80}
a) Optimal frac value for each pattern:
{results_df.loc[results_df.groupby('pattern')['confidence'].idxmax()][['pattern', 'frac', 'confidence', 'strength']].to_string(index=False)}

b) Overall best performing frac value:
{results_df.groupby('frac')['confidence'].mean().idxmax()} (Average confidence: {results_df.groupby('frac')['confidence'].mean().max():.3f})

5. Recommendations
{'-'*80}
Based on the analysis above, here are the recommended settings:
- For stable patterns: frac = {results_df[results_df['pattern'] == 'stable'].loc[results_df[results_df['pattern'] == 'stable']['confidence'].idxmax(), 'frac']}
- For trending patterns: frac = {results_df[results_df['pattern'] != 'stable'].groupby('frac')['confidence'].mean().idxmax()}
- For general use: frac = {results_df.groupby('frac')['confidence'].mean().idxmax()}
"""
    
    # Save report
    with open('loess_analysis_report.txt', 'w') as f:
        f.write(report)
    
    print("Report has been generated and saved to 'loess_analysis_report.txt'")
    print("\nKey findings summary:")
    print("-" * 40)
    print(f"Best overall frac value: {results_df.groupby('frac')['confidence'].mean().idxmax()}")
    print(f"Average execution time: {results_df['execution_time'].mean():.4f} seconds")
    print(f"Best average confidence: {results_df.groupby('frac')['confidence'].mean().max():.3f}")

if __name__ == "__main__":
    generate_report()
