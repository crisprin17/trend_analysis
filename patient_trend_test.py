import pandas as pd
import numpy as np
from trend_analysis import calculate_loess_trend
import time
from datetime import datetime, timedelta

def generate_patient_data():
    np.random.seed(42)
    
    # Generate one year of daily data
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    
    # Different patient scenarios
    scenarios = {
        'gradual_improvement': [
            # Starting at 150, gradually improving to 100 with daily fluctuations
            150 - (i * 0.137) + np.random.normal(0, 3) for i in range(365)
        ],
        'rapid_deterioration': [
            # Starting at 90, rapid increase to 140 over 2 months, then stabilizing
            90 + (min(i, 60) * 0.833) + np.random.normal(0, 4) for i in range(365)
        ],
        'seasonal_variation': [
            # Base value 100 with seasonal pattern and slight improvement
            100 + 15 * np.sin(2 * np.pi * i / 90) - (i * 0.02) + np.random.normal(0, 3) 
            for i in range(365)
        ],
        'stable_controlled': [
            # Stable around 95 with minor fluctuations
            95 + np.random.normal(0, 2) for _ in range(365)
        ],
        'fluctuating': [
            # Fluctuating between 100-130 with occasional spikes
            115 + 15 * np.sin(2 * np.pi * i / 30) + np.random.normal(0, 5) + 
            (10 if np.random.random() < 0.1 else 0)  # Occasional spikes
            for i in range(365)
        ]
    }
    
    datasets = {}
    for scenario, values in scenarios.items():
        datasets[scenario] = pd.DataFrame({
            'timestamp': dates,
            'event_value': values
        })
    
    return datasets

def analyze_patient_trends():
    print("\nPatient Trend Analysis Report")
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    datasets = generate_patient_data()
    results = []
    
    for scenario, data in datasets.items():
        print(f"\nAnalyzing scenario: {scenario.replace('_', ' ').title()}")
        print("-" * 40)
        
        # Calculate trend
        start_time = time.time()
        result = calculate_loess_trend(data, value_column='event_value')
        execution_time = time.time() - start_time
        
        # Print detailed results
        print(f"Direction: {result.direction}")
        print(f"Strength: {result.strength}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Execution Time: {execution_time:.3f} seconds")
        
        if result.details:
            print("\nDetailed Statistics:")
            print(f"Trend Strength: {result.details['decomposition']['trend_strength']:.3f}")
            print(f"Seasonal Strength: {result.details['decomposition']['seasonal_strength']:.3f}")
            
            if 'tests' in result.details:
                tests = result.details['tests']
                print("\nStatistical Tests:")
                print(f"Slope: {tests['slope']['value']:.4f} (p-value: {tests['slope']['p_value']:.4e})")
                print(f"Variance Ratio: {tests['variance']['ratio']:.4f}")
                print(f"Runs Test Z-score: {tests['runs']['z_score']:.4f}")
        
        # Store results
        results.append({
            'scenario': scenario,
            'direction': result.direction,
            'strength': result.strength,
            'confidence': result.confidence,
            'execution_time': execution_time,
            'trend_strength': result.details['decomposition']['trend_strength']
        })
    
    # Print summary table
    print("\nSummary of All Scenarios")
    print("=" * 80)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

if __name__ == "__main__":
    analyze_patient_trends()
