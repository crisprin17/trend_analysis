import pandas as pd
import numpy as np
from trend_analysis import calculate_loess_trend
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_data(file_path):
    """Load and prepare the patient data."""
    df = pd.read_csv(file_path)
    df['local_date'] = pd.to_datetime(df['local_date'])
    return df

def analyze_patient(data):
    """Analyze trend for a patient's data."""
    result = calculate_loess_trend(data)
    return result

def create_combined_visualization(all_results):
    """Create a combined visualization with all patients."""
    n_patients = len(all_results)
    
    # Create subplots for patient data
    fig = make_subplots(
        rows=n_patients, 
        cols=1,
        subplot_titles=[f"Patient {r['patient_id']} ({r['name']})" for r in all_results],
        vertical_spacing=0.05
    )
    
    # Add patient plots
    for idx, result in enumerate(all_results, 1):
        # Plot actual values
        fig.add_trace(
            go.Scatter(
                x=result['data']['local_date'],
                y=result['data']['event_value'],
                mode='lines+markers',
                name=f'Patient {result["patient_id"]} Values',
                line=dict(color='blue'),
                showlegend=False
            ),
            row=idx, 
            col=1
        )
        
        # Add trend line
        x_trend = np.linspace(0, len(result['data'])-1, len(result['data']))
        y_trend = np.polyfit(x_trend, result['data']['event_value'], 1)
        y_trend_line = np.polyval(y_trend, x_trend)
        
        fig.add_trace(
            go.Scatter(
                x=result['data']['local_date'],
                y=y_trend_line,
                mode='lines',
                name=f'Patient {result["patient_id"]} Trend',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=idx, 
            col=1
        )
        
        # Add annotation with trend info
        fig.add_annotation(
            text=(f"True: {result['true_trend']}<br>"
                  f"Detected: {result['result'].direction}<br>"
                  f"Age: {result['age']}<br>"
                  f"Condition: {result['condition']}"),
            xref=f"x{idx}", yref=f"y{idx}",
            x=0.02, y=0.95,
            xanchor='left',
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
    
    # Update layout
    fig.update_layout(
        height=300*n_patients,
        title_text="Patient Trend Analysis",
        showlegend=False
    )
    
    return fig

def create_results_table(all_results):
    """Create a separate table visualization."""
    table_data = []
    for r in all_results:
        slope = r['result'].details['tests']['slope']['value'] if 'tests' in r['result'].details else np.nan
        table_data.append([
            r['patient_id'],
            r['name'],
            1,  # Week
            r['true_trend'],
            r['result'].direction,
            f"{slope:.4f}" if not np.isnan(slope) else "N/A",
            r['result'].confidence,
            r['result'].strength
        ])
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Patient ID', 'Name', 'Week', 'True Trend', 'Detected Trend', 'Slope', 'Confidence', 'Strength'],
            font=dict(size=12, color='white'),
            fill_color='darkblue',
            align='left'
        ),
        cells=dict(
            values=list(zip(*table_data)),
            font=dict(size=11),
            fill_color='lavender',
            align='left'
        )
    )])
    
    fig.update_layout(
        title_text="Trend Analysis Results",
        height=400
    )
    
    return fig

def main():
    # Load data
    df = load_data('synthetic_patient_data.csv')
    
    print("\nTrend Analysis Results")
    print("=" * 50)
    
    all_results = []
    
    # Analyze each patient
    for patient_id in df['patient_id'].unique():
        patient_data = df[df['patient_id'] == patient_id].copy()
        true_trend = patient_data['true_trend'].iloc[0]
        
        # Analyze trend
        result = analyze_patient(patient_data)
        
        # Store results
        all_results.append({
            'patient_id': patient_id,
            'name': patient_data['name'].iloc[0],
            'age': patient_data['age'].iloc[0],
            'condition': patient_data['condition'].iloc[0],
            'data': patient_data,
            'true_trend': true_trend,
            'result': result
        })
        
        # Print results
        print(f"\nPatient {patient_id} ({patient_data['name'].iloc[0]})")
        print("-" * 30)
        print(f"True Trend: {true_trend}")
        print(f"Detected Trend: {result.direction}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Strength: {result.strength}")
        
        # Print detailed statistics
        print("\nDetailed Statistics:")
        if 'tests' in result.details:
            tests = result.details['tests']
            print(f"Slope: {tests['slope']['value']:.4f}")
            print(f"Variance Ratio: {tests['variance']['ratio']:.4f}")
            print(f"Trend Strength: {result.details['decomposition']['trend_strength']:.4f}")
    
    # Create and save visualizations
    plots_fig = create_combined_visualization(all_results)
    plots_fig.write_html("trend_analysis_plots.html")
    
    table_fig = create_results_table(all_results)
    table_fig.write_html("trend_analysis_table.html")
    
    print("\nVisualizations have been saved as:")
    print("- trend_analysis_plots.html")
    print("- trend_analysis_table.html")
    
    # Save results as CSV
    table_data = []
    for r in all_results:
        slope = r['result'].details['tests']['slope']['value'] if 'tests' in r['result'].details else np.nan
        table_data.append({
            'Patient ID': r['patient_id'],
            'Name': r['name'],
            'Age': r['age'],
            'Condition': r['condition'],
            'Week': 1,
            'True Trend': r['true_trend'],
            'Detected Trend': r['result'].direction,
            'Confidence': r['result'].confidence,
            'Strength': r['result'].strength,
            'Slope': slope
        })
    
    results_df = pd.DataFrame(table_data)
    results_df.to_csv('trend_analysis_results.csv', index=False)
    print("- trend_analysis_results.csv")

if __name__ == "__main__":
    main()
