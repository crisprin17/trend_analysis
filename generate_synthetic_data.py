import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class PatientProfile:
    def __init__(self, id, name, age, condition, story):
        self.id = id
        self.name = name
        self.age = age
        self.condition = condition
        self.story = story

def generate_patient_step_data(profile, start_date, num_days, base_steps, pattern_type, noise_level=0.2):
    """
    Generate synthetic step data for a patient based on their profile and response pattern.
    
    Parameters:
    - profile: PatientProfile object
    - start_date: Start date of data collection
    - num_days: Number of days to generate data for
    - base_steps: Base number of daily steps
    - pattern_type: Response pattern to treatment
    - noise_level: Amount of random noise to add
    """
    dates = [datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=x) 
            for x in range(num_days)]
    
    # Generate base steps with weekly pattern (less steps on weekends)
    base_values = np.array([base_steps * (0.7 if d.weekday() >= 5 else 1.0) 
                           for d in dates])
    
    # Treatment starts at day 14 (week 2)
    treatment_start = 14
    
    # Initialize ground truth trends for each week
    weekly_trends = []
    
    if pattern_type == "clear_improvement":
        # Linear improvement with minimal noise
        values = base_values.copy()
        values[treatment_start:] += np.linspace(0, 3000, num_days-treatment_start)
        weekly_trends = ["No Trend", "Upward", "Upward", "Upward", "Upward", "Upward"]
    
    elif pattern_type == "clear_deterioration":
        # Linear deterioration with minimal noise
        values = base_values.copy()
        values[treatment_start:] -= np.linspace(0, 2000, num_days-treatment_start)
        weekly_trends = ["No Trend", "Downward", "Downward", "Downward", "Downward", "Downward"]
    
    elif pattern_type == "improvement":
        values = base_values.copy()
        values[treatment_start:] += np.linspace(0, 2000, num_days-treatment_start)
        weekly_trends = ["No Trend", "Upward", "Upward", "Upward", "Upward", "Upward"]
    
    elif pattern_type == "deterioration":
        values = base_values.copy()
        values[treatment_start:] -= np.linspace(0, 1500, num_days-treatment_start)
        weekly_trends = ["No Trend", "Downward", "Downward", "Downward", "Downward", "Downward"]
    
    elif pattern_type == "oscillating":
        values = base_values.copy()
        post_treatment_days = num_days - treatment_start
        oscillation = 1000 * np.sin(np.linspace(0, 4*np.pi, post_treatment_days))
        values[treatment_start:] += oscillation
        weekly_trends = ["No Trend", "Upward", "Downward", "Upward", "Downward", "Upward"]
    
    elif pattern_type == "no_change":
        values = base_values.copy()
        weekly_trends = ["No Trend", "No Trend", "No Trend", "No Trend", "No Trend", "No Trend"]
    
    elif pattern_type == "initial_improvement_then_plateau":
        values = base_values.copy()
        improvement_period = 10
        plateau_start = treatment_start + improvement_period
        values[treatment_start:plateau_start] += np.linspace(0, 1500, improvement_period)
        values[plateau_start:] += 1500
        weekly_trends = ["No Trend", "Upward", "Upward", "No Trend", "No Trend", "No Trend"]
    
    elif pattern_type == "delayed_response":
        values = base_values.copy()
        delay_days = 10
        delayed_start = treatment_start + delay_days
        values[delayed_start:] += np.linspace(0, 1800, num_days-delayed_start)
        weekly_trends = ["No Trend", "No Trend", "No Trend", "Upward", "Upward", "Upward"]
    
    # Add random noise
    noise = np.random.normal(0, noise_level * base_steps, num_days)
    values = values + noise
    
    # Ensure no negative values
    values = np.maximum(values, 0)
    
    # Create weekly trend data
    week_labels = []
    for i in range(len(dates)):
        week_num = i // 7
        if week_num < len(weekly_trends):
            week_labels.append(weekly_trends[week_num])
        else:
            week_labels.append("No Trend")
    
    return pd.DataFrame({
        'patient_id': profile.id,
        'local_date': dates,
        'event_value': values.astype(int),
        'name': profile.name,
        'age': profile.age,
        'condition': profile.condition,
        'true_trend': week_labels
    })

def generate_test_dataset():
    """
    Generate a comprehensive test dataset with different patient personas.
    """
    # Define patient personas
    personas = [
        PatientProfile(
            "P001", "Alice Thompson", 35,
            "Treatment Resistant Depression",
            "Clear linear improvement case - Consistent daily increase"
        ),
        PatientProfile(
            "P002", "Bob Martinez", 42,
            "Treatment Resistant Depression",
            "Clear linear deterioration case - Consistent daily decrease"
        ),
        PatientProfile(
            "P003", "Carol Chen", 28,
            "Treatment Resistant Depression",
            "Shows initial improvement followed by relapse"
        ),
        PatientProfile(
            "P004", "David Wilson", 45,
            "Treatment Resistant Depression",
            "Shows minimal response to treatment"
        ),
        PatientProfile(
            "P005", "Emma Rodriguez", 31,
            "Treatment Resistant Depression",
            "Shows delayed but steady improvement"
        ),
        PatientProfile(
            "P006", "Frank Johnson", 39,
            "Treatment Resistant Depression",
            "Shows decline despite treatment"
        ),
        PatientProfile(
            "P007", "Grace Kim", 33,
            "Treatment Resistant Depression",
            "Shows cyclical patterns of improvement and decline"
        ),
        PatientProfile(
            "P008", "Henry Patel", 37,
            "Treatment Resistant Depression",
            "Shows initial decline followed by gradual improvement"
        )
    ]
    
    # Generate 6 weeks (42 days) of data for each persona
    start_date = '2023-01-01'
    datasets = []
    
    # Pattern assignments based on personas' stories
    patterns = {
        # Clear linear improvement with minimal noise
        "P001": ("clear_improvement", 5000),
        # Clear linear deterioration with minimal noise
        "P002": ("clear_deterioration", 4000),
        # Regular patterns for other patients
        "P003": ("oscillating", 3000),
        "P004": ("no_change", 4500),
        "P005": ("initial_improvement_then_plateau", 5500),
        "P006": ("deterioration", 4000),
        "P007": ("oscillating", 3500),
        "P008": ("delayed_response", 4800)
    }
    
    for persona in personas:
        pattern, base_steps = patterns[persona.id]
        # Use lower noise for clear cases
        noise_level = 0.02 if pattern in ["clear_improvement", "clear_deterioration"] else 0.15
        data = generate_patient_step_data(
            persona, start_date, 42, base_steps, pattern,
            noise_level=noise_level
        )
        datasets.append(data)
    
    # Combine all datasets
    return pd.concat(datasets).reset_index(drop=True)

if __name__ == "__main__":
    # Generate test data
    test_data = generate_test_dataset()
    test_data.to_csv('synthetic_patient_data.csv', index=False)
    
    # Print summary statistics for each patient
    print("\nGenerated synthetic walking data for the following personas:")
    for patient_id in test_data['patient_id'].unique():
        patient_data = test_data[test_data['patient_id'] == patient_id]
        name = patient_data['name'].iloc[0]
        condition = patient_data['condition'].iloc[0]
        avg_steps = patient_data['event_value'].mean()
        print(f"\nPatient {patient_id}: {name}")
        print(f"Condition: {condition}")
        print(f"Average daily steps: {int(avg_steps)}")
