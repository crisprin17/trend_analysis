import pandas as pd
import numpy as np
from trend_analysis import calculate_loess_trend
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create synthetic patient data with different trend patterns
def generate_synthetic_data():
    np.random.seed(42)  # For reproducibility
    
    # Time range
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    
    # Generate different trend patterns
    upward = [50 + i*0.1 + np.random.normal(0, 2) for i in range(365)]  # Linear upward
    downward = [100 - i*0.15 + np.random.normal(0, 3) for i in range(365)]  # Linear downward
    cyclic = [75 + 20*np.sin(i/30) + i*0.05 + np.random.normal(0, 2) for i in range(365)]  # Cyclic with trend
    stable = [80 + np.random.normal(0, 4) for _ in range(365)]  # Stable with noise
    
    datasets = {
        'upward': pd.DataFrame({'timestamp': dates, 'event_value': upward}),
        'downward': pd.DataFrame({'timestamp': dates, 'event_value': downward}),
        'cyclic': pd.DataFrame({'timestamp': dates, 'event_value': cyclic}),
        'stable': pd.DataFrame({'timestamp': dates, 'event_value': stable})
    }
    return datasets

def evaluate_trend(data, pattern_name):
    # Time the analysis
    start_time = time.time()
    result = calculate_loess_trend(data, value_column='event_value', frac=0.3)
    execution_time = time.time() - start_time
    
    # Calculate metrics
    metrics = {
        'pattern': pattern_name,
        'direction': result.direction,
        'strength': result.strength,
        'confidence': result.confidence,
        'execution_time': execution_time,
    }
    
    # Print detailed results for each pattern
    print(f"\nPattern: {pattern_name.upper()}")
    print(f"Direction: {metrics['direction']}")
    print(f"Strength: {metrics['strength']}")
    print(f"Confidence: {metrics['confidence']:.2f}")
    print(f"Execution Time: {metrics['execution_time']:.3f} seconds")
    
    if result.details:
        print("\nDetailed Statistics:")
        for key, value in result.details.items():
            print(f"{key}: {value}")
    
    return metrics

# Generate and analyze synthetic data
datasets = generate_synthetic_data()
results = []

print("\nAnalyzing different trend patterns with LOESS method:")
print("-" * 80)

for pattern_name, data in datasets.items():
    metrics = evaluate_trend(data, pattern_name)
    results.append(metrics)

# Convert results to DataFrame for easy comparison
results_df = pd.DataFrame(results)
print("\nSummary of all patterns:")
print("-" * 80)
print(results_df[['pattern', 'direction', 'strength', 'confidence', 'execution_time']].to_string(index=False))
