################################################################################
#### Step 1 = After import of CuesCrav.npy --> reduction of time points  #######
################################################################################

import numpy as np
import pandas as pd

# Load the data from CuesCrav.npy
file_path = 'CuesCrav.npy'
data = np.load(file_path, allow_pickle=True)
Luse, Lcrav, Lcues = data  # Assuming the order remains [Luse, Lcrav, Lcues]

# Limit each subject's data to 4 time points for the analysis of cues on craving
data_points_per_subject = 4
Luse_reduced = [sublist[:data_points_per_subject] for sublist in Luse]
Lcrav_reduced = [sublist[:data_points_per_subject] for sublist in Lcrav]
Lcues_reduced = [sublist[:data_points_per_subject] for sublist in Lcues]

# Prepare the reduced dataset with a unique SubjectID for each participant
num_subjects = len(Luse_reduced)
df_reduced = pd.DataFrame({
    'Luse': [val for sublist in Luse_reduced for val in sublist],
    'Lcrav': [val for sublist in Lcrav_reduced for val in sublist],
    'Lcues': [val for sublist in Lcues_reduced for val in sublist],
    'SubjectID': [i for i in range(num_subjects) for _ in range(data_points_per_subject)]
})

# Add a time column to represent the sequence of time points
df_reduced['Time'] = df_reduced.groupby('SubjectID').cumcount()

# Print summary information for validation
total_rows = len(df_reduced)
print(f"Total number of rows: {total_rows}")
num_subjects = df_reduced['SubjectID'].nunique()
print(f"Total number of unique subjects: {num_subjects}")
time_points_per_subject = df_reduced.groupby('SubjectID')['Time'].count()
print("Number of time points per SubjectID:")
print(time_points_per_subject)

# Display the first few rows of the reduced DataFrame
df_reduced.head()




################################################################################
###### Step 2 = Make the SARIMAX with the simulated data  ######################
################################################################################

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from itertools import product
import matplotlib.pyplot as plt
import numpy as np

# Assuming df_reduced is already loaded as the reduced dataset
all_subjects = df_reduced['SubjectID'].unique()

# Define the range for AR (p) and MA (q)
p_range = range(0, 3)
q_range = range(0, 3)

# Function to test stationarity
def test_stationarity(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    return dftest[1]

optimized_models = []

for subject in all_subjects:
    subject_data = df_reduced[df_reduced['SubjectID'] == subject].dropna(subset=['Lcrav', 'Lcues'])
    
    if len(subject_data) < 2:
        continue

    # Test for stationarity in both time series
    p_value_craving = test_stationarity(subject_data['Lcrav'])
    p_value_cues = test_stationarity(subject_data['Lcues'])

    if p_value_craving < 0.05 and p_value_cues < 0.05:
        best_aic_model1 = float('inf')
        best_aic_model2 = float('inf')
        best_order_model1 = None
        best_order_model2 = None

        for p, q in product(p_range, q_range):
            try:
                # Model 1: Craving as dependent, Cues as exogenous
                model1 = ARIMA(subject_data['Lcrav'], order=(p, 0, q), exog=subject_data[['Lcues']])
                result1 = model1.fit()
                if result1.aic < best_aic_model1:
                    best_aic_model1 = result1.aic
                    best_order_model1 = (p, q)

                # Model 2: Cues as dependent, Craving as exogenous
                model2 = ARIMA(subject_data['Lcues'], order=(p, 0, q), exog=subject_data[['Lcrav']])
                result2 = model2.fit()
                if result2.aic < best_aic_model2:
                    best_aic_model2 = result2.aic
                    best_order_model2 = (p, q)

            except Exception as e:
                continue
        
        # Calculate cross-correlation for optimal lag
        cross_correlation = sm.tsa.stattools.ccf(subject_data['Lcrav'], subject_data['Lcues'])
        max_corr_lag = np.argmax(np.abs(cross_correlation[:10]))

        # Store results
        optimized_models.append((subject, best_order_model1, best_aic_model1, best_order_model2, best_aic_model2))

# Organize results into a DataFrame
optimized_models_df = pd.DataFrame(optimized_models, columns=[
    'SubjectID', 'Best Order Model 1', 'Best AIC Model 1', 'Best Order Model 2', 'Best AIC Model 2'
])

# Determine the preferred model for each subject
optimized_models_df['Preferred Model'] = optimized_models_df.apply(
    lambda row: 'Model 1 (Craving)' if row['Best AIC Model 1'] < row['Best AIC Model 2'] else 'Model 2 (Cues)',
    axis=1
)

print(optimized_models_df)

# Try to have a statistic in %
model_preference_counts = optimized_models_df['Preferred Model'].value_counts(normalize=True) * 100
model_preference_counts_total = optimized_models_df['Preferred Model'].value_counts()
print("\nProportion of subjects for each favorite model (%):")
print(model_preference_counts)
print("\nTotal number of subjects preferring each model :")
print(model_preference_counts_total)


# Try to have plots for some subjects
preferred_model_df = optimized_models_df[optimized_models_df['Preferred Model'] == 'Model 1 (Craving)']
subject_ids = preferred_model_df['SubjectID'].values

# Define a seed for reproducibility of the random selection
np.random.seed(42) 

# Select up to 4 subjects if available
num_samples = min(4, len(subject_ids))
sample_subject_ids = np.random.choice(subject_ids, num_samples, replace=False)

# Plotting the selected subjects
plt.figure(figsize=(14, 10))
for i, subject_id in enumerate(sample_subject_ids, 1):
    subject_data = df_reduced[df_reduced['SubjectID'] == subject_id]
    
    plt.subplot(2, 2, i)
    plt.plot(subject_data['Time'], subject_data['Lcrav'], marker='o', linestyle='-')
    plt.xlabel("Time Points")
    plt.ylabel("Craving (Lcrav)")
    plt.title(f"Craving (Lcrav) over Time for Subject {subject_id} (Preferred Model)")
    plt.xticks(range(5))  # Shows time points from 0 to 4 without decimals
    plt.grid(True)

plt.tight_layout()
plt.show()
