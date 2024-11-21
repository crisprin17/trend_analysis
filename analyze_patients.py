import pandas as pd
import numpy as np
import plotly.graph_objects as go
from trend_analysis import calculate_loess_trend
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def load_and_prepare_data(file_path):
    """Load and prepare the patient data for analysis."""
    df = pd.read_csv(file_path)
    df['local_date'] = pd.to_datetime(df['local_date'])
    return df

def analyze_patient_trend(patient_data):
    """Analyze trend for a patient's data using LOESS."""
    result = calculate_loess_trend(patient_data)
    return result.direction

def evaluate_trends(df):
    """Evaluate trends for all patients and calculate performance metrics."""
    results = []
    metrics = {}
    
    # Group by patient
    for patient_id, patient_data in df.groupby('patient_id'):
        # Get ground truth (from the 'trend' column)
        true_trend = patient_data['trend'].iloc[0]
        
        # Predict trend using LOESS
        predicted_trend = analyze_patient_trend(patient_data)
        
        results.append({
            'patient_id': patient_id,
            'true_trend': true_trend,
            'predicted_trend': predicted_trend
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate metrics per patient
    for patient_id in results_df['patient_id'].unique():
        patient_mask = results_df['patient_id'] == patient_id
        patient_true = results_df.loc[patient_mask, 'true_trend']
        patient_pred = results_df.loc[patient_mask, 'predicted_trend']
        
        accuracy = accuracy_score([patient_true.iloc[0]], [patient_pred.iloc[0]])
        precision, recall, f1, _ = precision_recall_fscore_support(
            [patient_true.iloc[0]], [patient_pred.iloc[0]], 
            average='weighted', zero_division=0
        )
        
        metrics[patient_id] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # Calculate overall metrics
    overall_accuracy = accuracy_score(results_df['true_trend'], results_df['predicted_trend'])
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        results_df['true_trend'], results_df['predicted_trend'],
        average='weighted', zero_division=0
    )
    
    metrics['overall'] = {
        'accuracy': overall_accuracy,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1
    }
    
    return results_df, metrics

def plot_confusion_matrix(results_df):
    """Create and save confusion matrix visualization."""
    confusion_data = pd.crosstab(
        results_df['true_trend'], 
        results_df['predicted_trend'],
        normalize='index'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=confusion_data.values,
        x=confusion_data.columns,
        y=confusion_data.index,
        colorscale='RdBu',
        text=np.around(confusion_data.values, 2),
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        showscale=True
    ))
    
    fig.update_layout(
        title='Confusion Matrix (Normalized)',
        xaxis_title='Predicted Trend',
        yaxis_title='True Trend'
    )
    
    fig.write_html("confusion_matrix.html")
    return fig

def main():
    # Load data
    df = load_and_prepare_data('synthetic_patient_data.csv')
    
    # Evaluate trends
    results_df, metrics = evaluate_trends(df)
    
    # Print metrics
    print("\nPerformance Metrics:")
    print("=" * 50)
    
    # Print per-patient metrics
    for patient_id, patient_metrics in metrics.items():
        if patient_id == 'overall':
            continue
        print(f"\nPatient {patient_id}:")
        print("-" * 20)
        for metric, value in patient_metrics.items():
            print(f"{metric.capitalize()}: {value:.3f}")
    
    # Print overall metrics
    print("\nOverall Performance:")
    print("-" * 20)
    for metric, value in metrics['overall'].items():
        print(f"{metric.capitalize()}: {value:.3f}")
    
    # Create confusion matrix plot
    plot_confusion_matrix(results_df)
    print("\nConfusion matrix has been saved as 'confusion_matrix.html'")

if __name__ == "__main__":
    main()
