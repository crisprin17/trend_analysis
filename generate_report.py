import pandas as pd
import numpy as np
from trend_analysis import analyze_trend, calculate_loess_trend, log_likelihood_ratio_test, TrendResult, decompose_time_series
from generate_synthetic_data import generate_test_dataset
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def analyze_single_week(week_data):
    """Analyze trend for a single week of data."""
    if len(week_data) < 3:  # Need at least 3 points for trend
        return None
    
    # First decompose to remove weekly pattern
    decomp = decompose_time_series(week_data, period=7)
    
    # Get individual method results on deseasonalized data
    deseason_data = week_data.copy()
    deseason_data['event_value'] = decomp['trend'] + decomp['residual']
    
    loess_result = calculate_loess_trend(deseason_data)
    llr_result = log_likelihood_ratio_test(deseason_data)
    
    # Combine results
    if loess_result.direction == llr_result.direction:
        final_direction = loess_result.direction
        final_strength = max(loess_result.strength, llr_result.strength)
        confidence = max(loess_result.confidence, llr_result.confidence)
    else:
        if loess_result.confidence > llr_result.confidence:
            final_direction = loess_result.direction
            final_strength = loess_result.strength
            confidence = loess_result.confidence
        else:
            final_direction = llr_result.direction
            final_strength = llr_result.strength
            confidence = llr_result.confidence
    
    return {
        'loess_result': loess_result.to_dict(),
        'llr_result': llr_result.to_dict(),
        'final_direction': final_direction,
        'final_strength': final_strength,
        'confidence': confidence,
        'seasonal_strength': decomp['strength']['seasonal']
    }

def create_patient_visualization(patient_data):
    """Create interactive plotly visualization for a patient."""
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add daily steps trace
    fig.add_trace(
        go.Scatter(
            x=patient_data['local_date'],
            y=patient_data['event_value'],
            name="Daily Steps",
            mode='lines+markers',
            marker=dict(size=8),
            line=dict(color='blue', width=1)
        ),
        secondary_y=False
    )
    
    # Add decomposition
    decomp = decompose_time_series(patient_data, period=7)
    
    # Add trend line
    fig.add_trace(
        go.Scatter(
            x=patient_data['local_date'],
            y=decomp['trend'],
            name="Trend (Deseasonalized)",
            line=dict(color='red', width=2)
        ),
        secondary_y=False
    )
    
    # Add seasonal pattern if significant
    if decomp['strength']['seasonal'] > 0.3:
        fig.add_trace(
            go.Scatter(
                x=patient_data['local_date'],
                y=patient_data['event_value'].mean() + decomp['seasonal'],
                name="Seasonal Pattern",
                line=dict(color='green', width=1, dash='dot')
            ),
            secondary_y=False
        )
    
    # Add treatment start line
    treatment_date = patient_data['local_date'].min() + pd.Timedelta(days=14)
    y_range = [patient_data['event_value'].min(), patient_data['event_value'].max()]
    
    fig.add_shape(
        type="line",
        x0=treatment_date,
        x1=treatment_date,
        y0=y_range[0],
        y1=y_range[1],
        line=dict(color="purple", width=2, dash="dash")
    )
    
    # Add treatment start annotation
    fig.add_annotation(
        x=treatment_date,
        y=y_range[1],
        text="Treatment Start",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )
    
    # Update layout
    fig.update_layout(
        title=f"Activity Pattern: {patient_data['name'].iloc[0]}<br>Seasonal Strength: {decomp['strength']['seasonal']:.2f}",
        xaxis_title="Date",
        yaxis_title="Daily Steps",
        height=400,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def create_comparison_visualization(df):
    """Create comparison visualization for all patients."""
    # Calculate weekly averages for each patient
    weekly_avg = df.groupby(['patient_id', pd.Grouper(key='local_date', freq='W')])['event_value'].mean().reset_index()
    
    # Create figure
    fig = go.Figure()
    
    for patient_id in df['patient_id'].unique():
        patient_data = weekly_avg[weekly_avg['patient_id'] == patient_id]
        patient_name = df[df['patient_id'] == patient_id]['name'].iloc[0]
        
        # Get trend for this patient
        decomp = decompose_time_series(
            df[df['patient_id'] == patient_id],
            period=7
        )
        
        fig.add_trace(
            go.Scatter(
                x=patient_data['local_date'],
                y=patient_data['event_value'],
                name=f"{patient_name} ({patient_id})",
                mode='lines+markers'
            )
        )
    
    # Add treatment start line
    treatment_date = df['local_date'].min() + pd.Timedelta(days=14)
    y_range = [weekly_avg['event_value'].min(), weekly_avg['event_value'].max()]
    
    fig.add_shape(
        type="line",
        x0=treatment_date,
        x1=treatment_date,
        y0=y_range[0],
        y1=y_range[1],
        line=dict(color="purple", width=2, dash="dash")
    )
    
    fig.update_layout(
        title="Weekly Average Steps Comparison",
        xaxis_title="Date",
        yaxis_title="Average Daily Steps",
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def create_trend_distribution_chart(df):
    """Create trend distribution visualization."""
    # Analyze trends by week
    trend_results = []
    
    for patient_id in df['patient_id'].unique():
        patient_data = df[df['patient_id'] == patient_id].copy()
        current_date = patient_data['local_date'].min() + pd.Timedelta(days=7)
        week = 2
        
        while current_date <= patient_data['local_date'].max():
            week_data = patient_data[
                (patient_data['local_date'] >= current_date) &
                (patient_data['local_date'] < current_date + pd.Timedelta(days=7))
            ].copy()
            
            if len(week_data) > 0:
                result = analyze_single_week(week_data)
                if result:
                    trend_results.append({
                        'patient_id': patient_id,
                        'week': week,
                        'trend': result['final_direction']
                    })
            
            current_date += pd.Timedelta(days=7)
            week += 1
    
    # Convert to DataFrame for plotting
    trend_df = pd.DataFrame(trend_results)
    trend_counts = pd.crosstab(trend_df['patient_id'], trend_df['trend'])
    
    # Create stacked bar chart
    fig = go.Figure()
    
    for trend in trend_counts.columns:
        fig.add_trace(
            go.Bar(
                name=trend,
                x=trend_counts.index,
                y=trend_counts[trend],
                text=trend_counts[trend],
                textposition='auto',
            )
        )
    
    fig.update_layout(
        title="Trend Distribution by Patient",
        xaxis_title="Patient ID",
        yaxis_title="Number of Weeks",
        barmode='stack',
        height=400,
        showlegend=True
    )
    
    return fig

def generate_patient_table_html(df):
    """Generate HTML table rows for patient results."""
    rows = []
    
    for patient_id in df['patient_id'].unique():
        patient_data = df[df['patient_id'] == patient_id].copy()
        current_date = patient_data['local_date'].min() + pd.Timedelta(days=7)
        week = 2
        
        while current_date <= patient_data['local_date'].max():
            week_data = patient_data[
                (patient_data['local_date'] >= current_date) &
                (patient_data['local_date'] < current_date + pd.Timedelta(days=7))
            ].copy()
            
            if len(week_data) > 0:
                result = analyze_single_week(week_data)
                if result:
                    def get_trend_class(trend):
                        if isinstance(trend, dict):
                            trend = trend.get('direction', 'No Trend')
                        if str(trend).startswith('Up'):
                            return 'trend-up'
                        elif str(trend).startswith('Down'):
                            return 'trend-down'
                        return 'trend-none'
                    
                    row = f"""
                    <tr>
                        <td>{patient_id}</td>
                        <td>Week {week}</td>
                        <td class="{get_trend_class(result['loess_result']['direction'])}">
                            {result['loess_result']['direction']} ({result['loess_result']['strength']})
                        </td>
                        <td class="{get_trend_class(result['llr_result']['direction'])}">
                            {result['llr_result']['direction']} ({result['llr_result']['strength']})
                        </td>
                        <td class="{get_trend_class(result['final_direction'])}">
                            {result['final_direction']} ({result['final_strength']})
                        </td>
                        <td>{result['confidence']:.2f}</td>
                        <td>{result['seasonal_strength']:.2f}</td>
                    </tr>
                    """
                    rows.append(row)
            
            current_date += pd.Timedelta(days=7)
            week += 1
    
    return "\n".join(rows)

def update_report_with_results():
    """Update the HTML report with actual results and visualizations."""
    print("Generating test data...")
    df = generate_test_dataset()
    
    print("Creating visualizations...")
    # Create individual patient visualizations
    patient_plots = {}
    for patient_id in df['patient_id'].unique():
        patient_data = df[df['patient_id'] == patient_id].copy()
        fig = create_patient_visualization(patient_data)
        patient_plots[patient_id] = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    # Create comparison visualization
    comparison_plot = create_comparison_visualization(df)
    comparison_html = comparison_plot.to_html(full_html=False, include_plotlyjs='cdn')
    
    # Create trend distribution chart
    trend_dist_plot = create_trend_distribution_chart(df)
    trend_dist_html = trend_dist_plot.to_html(full_html=False, include_plotlyjs='cdn')
    
    print("Generating patient results table...")
    table_rows = generate_patient_table_html(df)
    
    print("Reading template...")
    with open('analysis_report.html', 'r') as f:
        html_content = f.read()
    
    print("Updating report...")
    # Add visualizations
    visualization_html = f"""
    <div class="section">
        <h2>3. Patient Comparisons</h2>
        {comparison_html}
        
        <h3>Trend Distribution</h3>
        {trend_dist_html}
    </div>
    
    <div class="section">
        <h2>4. Individual Patient Visualizations</h2>
        {''.join(f'<div class="patient-viz">{plot}</div>' for plot in patient_plots.values())}
    </div>
    """
    
    # Update content
    html_content = html_content.replace(
        '<!-- Add more rows dynamically from your data -->',
        table_rows
    )
    html_content = html_content.replace(
        '</body>',
        f'{visualization_html}</body>'
    )
    
    # Add additional CSS for visualizations
    html_content = html_content.replace(
        '</style>',
        """
        .patient-viz {
            margin: 20px 0;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
        """
    )
    
    with open('analysis_report_with_results.html', 'w') as f:
        f.write(html_content)
    
    print("Report generated: analysis_report_with_results.html")

if __name__ == "__main__":
    update_report_with_results()
