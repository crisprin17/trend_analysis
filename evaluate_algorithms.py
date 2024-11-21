import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error, r2_score
from trend_analysis import analyze_trend, calculate_loess_trend, log_likelihood_ratio_test
from generate_synthetic_data import generate_test_dataset
from scipy.stats import pearsonr
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess

def calculate_trend_metrics(true_trends, predicted_results):
    """
    Calculate accuracy, precision, recall, and F1 score for trend predictions.
    Now handles TrendResult objects and includes strength in evaluation.
    """
    # Convert trend labels to numeric
    direction_map = {"Upward": 1, "Downward": -1, "No Trend": 0}
    strength_map = {"Strong": 3, "Moderate": 2, "Weak": 1, "None": 0}
    
    # Extract directions and strengths
    y_true = [direction_map[t] for t in true_trends]
    y_pred = [direction_map[r.direction] for r in predicted_results]
    
    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # Calculate strength-weighted metrics
    strengths = [strength_map[r.strength] for r in predicted_results]
    confidence_scores = [r.confidence for r in predicted_results]
    
    weighted_accuracy = accuracy * np.mean(confidence_scores)
    
    return {
        'accuracy': accuracy,
        'weighted_accuracy': weighted_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_confidence': np.mean(confidence_scores),
        'avg_strength': np.mean(strengths)
    }

def evaluate_method_agreement(df):
    """
    Analyze how often the two methods agree/disagree.
    Now includes confidence and strength in the analysis.
    """
    total_comparisons = 0
    agreements = 0
    strong_agreements = 0  # Both methods agree with high confidence
    disagreements = []
    
    for patient_id in df['patient_id'].unique():
        patient_data = df[df['patient_id'] == patient_id].copy()
        
        # Analyze each week starting from week 2
        for week in range(1, 6):  # Weeks 2-6
            week_start = patient_data['local_date'].min() + pd.Timedelta(days=week*7)
            week_end = week_start + pd.Timedelta(days=7)
            week_data = patient_data[
                (patient_data['local_date'] >= week_start) & 
                (patient_data['local_date'] < week_end)
            ].copy()
            
            if len(week_data) > 0:
                loess_result = calculate_loess_trend(week_data)
                llr_result = log_likelihood_ratio_test(week_data)
                true_trend = week_data['true_trend'].iloc[0]
                
                total_comparisons += 1
                
                if loess_result.direction == llr_result.direction:
                    agreements += 1
                    if loess_result.confidence > 0.8 and llr_result.confidence > 0.8:
                        strong_agreements += 1
                else:
                    disagreements.append({
                        'patient_id': patient_id,
                        'week': week + 1,
                        'loess_result': loess_result.to_dict(),
                        'llr_result': llr_result.to_dict(),
                        'true_trend': true_trend
                    })
    
    agreement_rate = agreements / total_comparisons if total_comparisons > 0 else 0
    strong_agreement_rate = strong_agreements / total_comparisons if total_comparisons > 0 else 0
    
    return {
        'agreement_rate': agreement_rate,
        'strong_agreement_rate': strong_agreement_rate,
        'total_comparisons': total_comparisons,
        'agreements': agreements,
        'strong_agreements': strong_agreements,
        'disagreements': disagreements
    }

def calculate_goodness_of_fit(data, value_column='event_value'):
    """Calculate various goodness of fit metrics for both methods."""
    x = np.arange(len(data))
    y = data[value_column].values
    
    # LOESS fit
    loess_smoothed = lowess(y, x, frac=0.3, return_sorted=False)
    loess_mse = mean_squared_error(y, loess_smoothed)
    loess_r2 = r2_score(y, loess_smoothed)
    loess_correlation, _ = pearsonr(y, loess_smoothed)
    
    # Linear fit for LLR
    slope, intercept = np.polyfit(x, y, 1)
    linear_fit = slope * x + intercept
    llr_mse = mean_squared_error(y, linear_fit)
    llr_r2 = r2_score(y, linear_fit)
    llr_correlation, _ = pearsonr(y, linear_fit)
    
    return {
        'loess': {
            'mse': loess_mse,
            'r2': loess_r2,
            'correlation': loess_correlation
        },
        'llr': {
            'mse': llr_mse,
            'r2': llr_r2,
            'correlation': llr_correlation
        }
    }

def calculate_overall_trend_quality(data, value_column='event_value'):
    """Calculate metrics for overall trend quality."""
    # Calculate signal-to-noise ratio
    smoothed = savgol_filter(data[value_column].values, window_length=7, polyorder=3)
    noise = data[value_column].values - smoothed
    snr = np.var(smoothed) / np.var(noise) if np.var(noise) > 0 else float('inf')
    
    # Calculate trend consistency (autocorrelation)
    autocorr = data[value_column].autocorr(lag=1)
    
    # Calculate variance explained by trend
    total_variance = np.var(data[value_column])
    residual_variance = np.var(noise)
    variance_explained = 1 - (residual_variance / total_variance)
    
    # Calculate normalized trend strength
    x = np.arange(len(data))
    slope, _ = np.polyfit(x, data[value_column].values, 1)
    normalized_slope = slope / np.mean(data[value_column])
    
    return {
        'signal_to_noise_ratio': snr,
        'trend_consistency': autocorr,
        'variance_explained': variance_explained,
        'trend_strength': normalized_slope
    }

def evaluate_trend_detection_accuracy(df):
    """
    Evaluate accuracy of both methods against ground truth trends.
    Now includes strength and confidence analysis.
    """
    results = {
        'loess': {'true_trends': [], 'predicted_results': []},
        'llr': {'true_trends': [], 'predicted_results': []}
    }
    
    for patient_id in df['patient_id'].unique():
        patient_data = df[df['patient_id'] == patient_id].copy()
        
        # Analyze each week starting from week 2
        for week in range(1, 6):  # Weeks 2-6
            week_start = patient_data['local_date'].min() + pd.Timedelta(days=week*7)
            week_end = week_start + pd.Timedelta(days=7)
            week_data = patient_data[
                (patient_data['local_date'] >= week_start) & 
                (patient_data['local_date'] < week_end)
            ].copy()
            
            if len(week_data) > 0:
                true_trend = week_data['true_trend'].iloc[0]
                loess_result = calculate_loess_trend(week_data)
                llr_result = log_likelihood_ratio_test(week_data)
                
                results['loess']['true_trends'].append(true_trend)
                results['loess']['predicted_results'].append(loess_result)
                results['llr']['true_trends'].append(true_trend)
                results['llr']['predicted_results'].append(llr_result)
    
    # Calculate metrics for both methods
    loess_metrics = calculate_trend_metrics(
        results['loess']['true_trends'],
        results['loess']['predicted_results']
    )
    llr_metrics = calculate_trend_metrics(
        results['llr']['true_trends'],
        results['llr']['predicted_results']
    )
    
    return {
        'loess': loess_metrics,
        'llr': llr_metrics
    }

def main():
    # Generate test data
    print("Generating test data...")
    df = generate_test_dataset()
    
    print("\n1. Trend Detection Accuracy:")
    accuracy_results = evaluate_trend_detection_accuracy(df)
    
    print("\nLOESS Method Performance:")
    print(f"Basic Accuracy: {accuracy_results['loess']['accuracy']:.3f}")
    print(f"Weighted Accuracy: {accuracy_results['loess']['weighted_accuracy']:.3f}")
    print(f"Precision: {accuracy_results['loess']['precision']:.3f}")
    print(f"Recall: {accuracy_results['loess']['recall']:.3f}")
    print(f"F1 Score: {accuracy_results['loess']['f1']:.3f}")
    print(f"Average Confidence: {accuracy_results['loess']['avg_confidence']:.3f}")
    print(f"Average Strength: {accuracy_results['loess']['avg_strength']:.3f}")
    
    print("\nLLR Method Performance:")
    print(f"Basic Accuracy: {accuracy_results['llr']['accuracy']:.3f}")
    print(f"Weighted Accuracy: {accuracy_results['llr']['weighted_accuracy']:.3f}")
    print(f"Precision: {accuracy_results['llr']['precision']:.3f}")
    print(f"Recall: {accuracy_results['llr']['recall']:.3f}")
    print(f"F1 Score: {accuracy_results['llr']['f1']:.3f}")
    print(f"Average Confidence: {accuracy_results['llr']['avg_confidence']:.3f}")
    print(f"Average Strength: {accuracy_results['llr']['avg_strength']:.3f}")
    
    print("\n2. Method Agreement Analysis:")
    agreement_results = evaluate_method_agreement(df)
    print(f"Overall Agreement Rate: {agreement_results['agreement_rate']:.2%}")
    print(f"Strong Agreement Rate: {agreement_results['strong_agreement_rate']:.2%}")
    print(f"Total Comparisons: {agreement_results['total_comparisons']}")
    print(f"Number of Agreements: {agreement_results['agreements']}")
    print(f"Strong Agreements: {agreement_results['strong_agreements']}")
    
    print("\nDisagreement Details:")
    for d in agreement_results['disagreements']:
        print(f"\nPatient {d['patient_id']}, Week {d['week']}:")
        print(f"LOESS: {d['loess_result']['direction']} ({d['loess_result']['strength']}, conf: {d['loess_result']['confidence']:.2f})")
        print(f"LLR: {d['llr_result']['direction']} ({d['llr_result']['strength']}, conf: {d['llr_result']['confidence']:.2f})")
        print(f"True Trend: {d['true_trend']}")
    
    print("\n3. Goodness of Fit Analysis:")
    all_fits = []
    for patient_id in df['patient_id'].unique():
        patient_data = df[df['patient_id'] == patient_id].copy()
        fit_metrics = calculate_goodness_of_fit(patient_data)
        all_fits.append({
            'patient_id': patient_id,
            'loess_r2': fit_metrics['loess']['r2'],
            'llr_r2': fit_metrics['llr']['r2'],
            'loess_mse': fit_metrics['loess']['mse'],
            'llr_mse': fit_metrics['llr']['mse']
        })
    
    fits_df = pd.DataFrame(all_fits)
    print("\nAverage R² Scores:")
    print(f"LOESS: {fits_df['loess_r2'].mean():.3f}")
    print(f"LLR: {fits_df['llr_r2'].mean():.3f}")
    print("\nAverage MSE:")
    print(f"LOESS: {fits_df['loess_mse'].mean():.3f}")
    print(f"LLR: {fits_df['llr_mse'].mean():.3f}")
    
    print("\n4. Overall Trend Quality Analysis:")
    all_qualities = []
    for patient_id in df['patient_id'].unique():
        patient_data = df[df['patient_id'] == patient_id].copy()
        quality_metrics = calculate_overall_trend_quality(patient_data)
        quality_metrics['patient_id'] = patient_id
        all_qualities.append(quality_metrics)
    
    quality_df = pd.DataFrame(all_qualities)
    print("\nAverage Trend Quality Metrics:")
    print(f"Signal-to-Noise Ratio: {quality_df['signal_to_noise_ratio'].mean():.3f}")
    print(f"Trend Consistency: {quality_df['trend_consistency'].mean():.3f}")
    print(f"Variance Explained: {quality_df['variance_explained'].mean():.3f}")
    print(f"Trend Strength: {quality_df['trend_strength'].mean():.3f}")
    
    # Save detailed results
    fits_df.to_csv('analysis_output/goodness_of_fit_metrics.csv', index=False)
    quality_df.to_csv('analysis_output/trend_quality_metrics.csv', index=False)
    
    # Create comprehensive evaluation summary
    evaluation_summary = pd.DataFrame({
        'metric': [
            'LOESS Basic Accuracy', 'LOESS Weighted Accuracy', 'LOESS Average Confidence',
            'LLR Basic Accuracy', 'LLR Weighted Accuracy', 'LLR Average Confidence',
            'Method Agreement Rate', 'Strong Agreement Rate',
            'LOESS R²', 'LLR R²',
            'Signal-to-Noise Ratio', 'Trend Consistency', 'Variance Explained'
        ],
        'value': [
            accuracy_results['loess']['accuracy'],
            accuracy_results['loess']['weighted_accuracy'],
            accuracy_results['loess']['avg_confidence'],
            accuracy_results['llr']['accuracy'],
            accuracy_results['llr']['weighted_accuracy'],
            accuracy_results['llr']['avg_confidence'],
            agreement_results['agreement_rate'],
            agreement_results['strong_agreement_rate'],
            fits_df['loess_r2'].mean(),
            fits_df['llr_r2'].mean(),
            quality_df['signal_to_noise_ratio'].mean(),
            quality_df['trend_consistency'].mean(),
            quality_df['variance_explained'].mean()
        ]
    })
    
    evaluation_summary.to_csv('analysis_output/evaluation_summary.csv', index=False)

if __name__ == "__main__":
    main()
