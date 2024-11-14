import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

# Assuming 'filtered_data' and 'optimized_models_df' are already loaded as in the previous steps.

# Function to plot auto-correlation for each patient’s craving and cues
def plot_autocorrelation(patient_data, patient_id):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sm.graphics.tsa.plot_acf(patient_data['y_craving'], lags=20, ax=axes[0])
    axes[0].set_title(f'Autocorrelation of Craving (Patient {patient_id})')
    sm.graphics.tsa.plot_acf(patient_data['z_cues'], lags=20, ax=axes[1])
    axes[1].set_title(f'Autocorrelation of Cues (Patient {patient_id})')
    plt.tight_layout()
    plt.show()

# Loop through patients and generate auto-correlation plots
for patient in optimized_models_df['Patient'].unique():
    patient_data = filtered_data[filtered_data['SubjectID'] == patient].dropna(subset=['y_craving', 'z_cues'])
    if len(patient_data) >= 2:
        plot_autocorrelation(patient_data, patient)

# Export optimized models to an Excel file
optimized_models_df.to_excel("patient_scores.xlsx", index=False)

# Calculate auto-correlation for craving and cues for each patient
autocorrelation_results = []
for index, row in optimized_models_df.iterrows():
    patient_id = row['Patient']
    patient_data = filtered_data[filtered_data['SubjectID'] == patient_id].dropna(subset=['y_craving', 'z_cues'])
    if len(patient_data) >= 2:
        craving_acf = sm.tsa.stattools.acf(patient_data['y_craving'], nlags=20)
        cues_acf = sm.tsa.stattools.acf(patient_data['z_cues'], nlags=20)
        autocorrelation_results.append({'Patient': patient_id, 'Craving_ACF': craving_acf, 'Cues_ACF': cues_acf})

# Store auto-correlation results in a DataFrame
autocorrelation_df = pd.DataFrame(autocorrelation_results)

# Summary statistics for craving and cues for each patient
summary_stats = []
for patient in optimized_models_df['Patient'].unique():
    patient_data = filtered_data[filtered_data['SubjectID'] == patient].dropna(subset=['y_craving', 'z_cues'])
    if len(patient_data) >= 2:
        craving_stats = patient_data['y_craving'].describe()
        cues_stats = patient_data['z_cues'].describe()
        summary_stats.append({
            'Patient': patient,
            'Craving_Mean': craving_stats['mean'],
            'Craving_Std': craving_stats['std'],
            'Craving_Min': craving_stats['min'],
            'Craving_Max': craving_stats['max'],
            'Cues_Mean': cues_stats['mean'],
            'Cues_Std': cues_stats['std'],
            'Cues_Min': cues_stats['min'],
            'Cues_Max': cues_stats['max'],
        })

# Create a summary DataFrame
summary_df = pd.DataFrame(summary_stats)

# Cross-correlation analysis to find the optimal lag between craving and cues
cross_correlation_results = []
for index, row in optimized_models_df.iterrows():
    patient_id = row['Patient']
    patient_data = filtered_data[filtered_data['SubjectID'] == patient_id].dropna(subset=['y_craving', 'z_cues'])
    if len(patient_data) >= 2:
        ccf = sm.tsa.stattools.ccf(patient_data['y_craving'], patient_data['z_cues'], adjusted=False)
        max_lag = np.argmax(np.abs(ccf))
        max_ccf = ccf[max_lag]
        cross_correlation_results.append({
            'Patient': patient_id,
            'Max_Lag': max_lag,
            'Max_CCF': max_ccf,
            'Cross_Correlation': ccf
        })

# Create a DataFrame for cross-correlation results
if cross_correlation_results:
    cross_correlation_df = pd.DataFrame(cross_correlation_results)
    print(cross_correlation_df.describe())  # Overview of cross-correlation results
    print(cross_correlation_df.head())  # Display the first few rows
else:
    print("No cross-correlation results available.")

# Check for missing values
print(cross_correlation_df.isnull().sum())

# Plot cross-correlation for each patient
for index, row in cross_correlation_df.iterrows():
    patient_id = row['Patient']
    ccf = row['Cross_Correlation']
    lags = range(len(ccf))
    plt.figure(figsize=(10, 6))
    plt.plot(lags, ccf)
    plt.xlabel("Lag")
    plt.ylabel("Cross-Correlation")
    plt.title(f"Cross-Correlation Function for Patient {patient_id}")
    plt.grid(True)
    plt.show()
