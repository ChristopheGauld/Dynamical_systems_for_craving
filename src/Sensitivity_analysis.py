"""
Créé : Oct 20 2024
@Author: Christophe Gauld
"""

########################################################################
## This analysis is the same as the global model, but only for alcohol # (cocaine below)
########################################################################

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from itertools import product

file_path = '/content/Data_alcohol.xlsx' # Data_model_craving
data = pd.read_excel(file_path)
filtered_data = data.groupby('SubjectID').apply(lambda group: group.iloc[4:-4]).reset_index(drop=True)
all_patients = filtered_data['SubjectID'].unique()

# Range AR and MA
p_range = range(0, 3)  # Pour AR
q_range = range(0, 3)  # Pour MA
# Set AR (p) and MA (q) parameter ranges to 0-2 to balance model complexity and computation
# Justification: Limits complexity while capturing main temporal dynamics + prevents overfitting and reduces computational load for each patient

# Stationnarity
def test_stationarity(timeseries):
    # Check if the time series is constant before applying adfuller
    if np.var(timeseries) == 0:
        return 1.0  # Return a p-value of 1 to indicate non-stationarity if constant
    else:
        dftest = adfuller(timeseries, autolag='AIC')
        return dftest[1]

optimized_models = []

for patient in all_patients:
    patient_data = filtered_data[filtered_data['SubjectID'] == patient].dropna(subset=['y_craving', 'z_cues'])

    if len(patient_data) < 2:
        continue

    # Check stationnarity
    p_value_craving = test_stationarity(patient_data['y_craving'])
    p_value_cues = test_stationarity(patient_data['z_cues'])

    # Continue only if series ok stationar
    if p_value_craving < 0.05 and p_value_cues < 0.05:
        best_aic_model1 = float('inf')
        best_aic_model2 = float('inf')
        best_order_model1 = None
        best_order_model2 = None

        for p, q in product(p_range, q_range):
            try:
                # Model 1: Craving dependant, Cues exogene
                # Check if y_craving is constant before fitting the model
                if np.var(patient_data['y_craving']) != 0:
                    model1 = ARIMA(patient_data['y_craving'], order=(p, 0, q), exog=patient_data['z_cues'])
                    result1 = model1.fit()
                    if result1.aic < best_aic_model1:
                        best_aic_model1 = result1.aic
                        best_order_model1 = (p, q)

            except Exception as e:
                continue

        ## Calcul de l'auto-corrélation croisée pour déterminer le lag optimal
        #cross_correlation = sm.tsa.stattools.ccf(patient_data['y_craving'], patient_data['z_cues'])
        #max_corr_lag = np.argmax(np.abs(cross_correlation[:10]))  # Limite à un lag max de 10

        optimized_models.append((patient, best_order_model1, best_aic_model1, best_order_model2, best_aic_model2))

optimized_models_df = pd.DataFrame(optimized_models, columns=['Patient', 'Best order model 1', 'Best AIC model 1', 'Best order model 2', 'Best AIC model 2'])

for i in range(len(optimized_models_df)):
    if optimized_models_df.loc[i, 'Best AIC model 1'] < optimized_models_df.loc[i, 'Best AIC model 2']:
        optimized_models_df.loc[i, 'Preferred model'] = 'Model 1 (Craving)'
    else:
        optimized_models_df.loc[i, 'Preferred model'] = 'Model 2 (Cues)'

print(optimized_models_df)




########################################################################
################## # Sensitiviy analysis alcohol/global ################
########################################################################



def analyze_data(file_path):
    """
    Analyzes the craving data and returns a DataFrame with optimized model parameters.
    """
    data = pd.read_excel(file_path)
    filtered_data = data.groupby('SubjectID').apply(lambda group: group.iloc[4:-4]).reset_index(drop=True)
    all_patients = filtered_data['SubjectID'].unique()
    p_range = range(0, 3)
    q_range = range(0, 3)

    def test_stationarity(timeseries):
        if np.var(timeseries) == 0:
            return 1.0
        else:
            dftest = adfuller(timeseries, autolag='AIC')
            return dftest[1]

    optimized_models = []
    for patient in all_patients:
        patient_data = filtered_data[filtered_data['SubjectID'] == patient].dropna(subset=['y_craving', 'z_cues'])
        if len(patient_data) < 2:
            continue

        p_value_craving = test_stationarity(patient_data['y_craving'])
        p_value_cues = test_stationarity(patient_data['z_cues'])

        if p_value_craving < 0.05 and p_value_cues < 0.05:
            best_aic_model1 = float('inf')
            best_order_model1 = None
            for p, q in product(p_range, q_range):
                try:
                    if np.var(patient_data['y_craving']) != 0:
                        model1 = ARIMA(patient_data['y_craving'], order=(p, 0, q), exog=patient_data['z_cues'])
                        result1 = model1.fit()
                        if result1.aic < best_aic_model1:
                            best_aic_model1 = result1.aic
                            best_order_model1 = (p, q)
                except Exception as e:
                    continue

            optimized_models.append((patient, best_order_model1, best_aic_model1))

    optimized_models_df = pd.DataFrame(optimized_models, columns=['Patient', 'Best Order Model 1', 'Best AIC Model 1'])
    return optimized_models_df


# Analyze Data_alcohol.xlsx
df_alcohol = analyze_data('/content/Data_alcohol.xlsx')

# Analyze Data_model_craving.xlsx (replace with actual file path if different)
df_craving = analyze_data('/content/Data_model_craving.xlsx')


# Comparison
comparison_df = pd.merge(df_alcohol, df_craving, on='Patient', suffixes=('_alcohol', '_craving'))

# Example of significance testing (AIC difference) - You'll need a more robust statistical test if required
comparison_df['AIC_Difference'] = comparison_df['Best AIC Model 1_alcohol'] - comparison_df['Best AIC Model 1_craving']
print("Comparison of AIC values between datasets:")
print(comparison_df)

# Calculate a p-value for the difference (Replace with an appropriate statistical test).
# Example of using a t-test (assuming AIC differences are normally distributed)

from scipy import stats
t_statistic, p_value = stats.ttest_ind(comparison_df['Best AIC Model 1_alcohol'], comparison_df['Best AIC Model 1_craving'])
print(f"T-statistic: {t_statistic}, P-value: {p_value}")
if p_value > 0.05 :
  print("No difference in AIC")
else:
  print("Significant difference in AIC")








########################################################################
## This analysis is the same as the global model, but only for cocaine ##
########################################################################

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from itertools import product

file_path = '/content/Data_cocaine.xlsx' 
data = pd.read_excel(file_path)
filtered_data = data.groupby('SubjectID').apply(lambda group: group.iloc[4:-4]).reset_index(drop=True)
all_patients = filtered_data['SubjectID'].unique()

# Range AR and MA
p_range = range(0, 3)  # Pour AR
q_range = range(0, 3)  # Pour MA
# Set AR (p) and MA (q) parameter ranges to 0-2 to balance model complexity and computation
# Justification: Limits complexity while capturing main temporal dynamics + prevents overfitting and reduces computational load for each patient

# Stationnarity
def test_stationarity(timeseries):
    # Check if the time series is constant before applying adfuller
    if np.var(timeseries) == 0:
        return 1.0  # Return a p-value of 1 to indicate non-stationarity if constant
    else:
        dftest = adfuller(timeseries, autolag='AIC')
        return dftest[1]

optimized_models = []

for patient in all_patients:
    patient_data = filtered_data[filtered_data['SubjectID'] == patient].dropna(subset=['y_craving', 'z_cues'])

    if len(patient_data) < 2:
        continue

    # Check stationnarity
    p_value_craving = test_stationarity(patient_data['y_craving'])
    p_value_cues = test_stationarity(patient_data['z_cues'])

    # Continue only if series ok stationar
    if p_value_craving < 0.05 and p_value_cues < 0.05:
        best_aic_model1 = float('inf')
        best_aic_model2 = float('inf')
        best_order_model1 = None
        best_order_model2 = None

        for p, q in product(p_range, q_range):
            try:
                # Model 1: Craving dependant, Cues exogene
                # Check if y_craving is constant before fitting the model
                if np.var(patient_data['y_craving']) != 0:
                    model1 = ARIMA(patient_data['y_craving'], order=(p, 0, q), exog=patient_data['z_cues'])
                    result1 = model1.fit()
                    if result1.aic < best_aic_model1:
                        best_aic_model1 = result1.aic
                        best_order_model1 = (p, q)

            except Exception as e:
                continue

        ## Calcul de l'auto-corrélation croisée pour déterminer le lag optimal
        #cross_correlation = sm.tsa.stattools.ccf(patient_data['y_craving'], patient_data['z_cues'])
        #max_corr_lag = np.argmax(np.abs(cross_correlation[:10]))  # Limite à un lag max de 10

        optimized_models.append((patient, best_order_model1, best_aic_model1, best_order_model2, best_aic_model2))

optimized_models_df = pd.DataFrame(optimized_models, columns=['Patient', 'Best order model 1', 'Best AIC model 1', 'Best order model 2', 'Best AIC model 2'])

for i in range(len(optimized_models_df)):
    if optimized_models_df.loc[i, 'Best AIC model 1'] < optimized_models_df.loc[i, 'Best AIC model 2']:
        optimized_models_df.loc[i, 'Preferred model'] = 'Model 1 (Craving)'
    else:
        optimized_models_df.loc[i, 'Preferred model'] = 'Model 2 (Cues)'

print(optimized_models_df)


########################################################################
################## # Sensitivity analysis cocaine/global ################
########################################################################

def analyze_data(file_path):
    """
    Analyzes the craving data and returns a DataFrame with optimized model parameters.
    """
    data = pd.read_excel(file_path)
    filtered_data = data.groupby('SubjectID').apply(lambda group: group.iloc[4:-4]).reset_index(drop=True)
    all_patients = filtered_data['SubjectID'].unique()
    p_range = range(0, 3)
    q_range = range(0, 3)

    def test_stationarity(timeseries):
        if np.var(timeseries) == 0:
            return 1.0
        else:
            dftest = adfuller(timeseries, autolag='AIC')
            return dftest[1]

    optimized_models = []
    for patient in all_patients:
        patient_data = filtered_data[filtered_data['SubjectID'] == patient].dropna(subset=['y_craving', 'z_cues'])
        if len(patient_data) < 2:
            continue

        p_value_craving = test_stationarity(patient_data['y_craving'])
        p_value_cues = test_stationarity(patient_data['z_cues'])

        if p_value_craving < 0.05 and p_value_cues < 0.05:
            best_aic_model1 = float('inf')
            best_order_model1 = None
            for p, q in product(p_range, q_range):
                try:
                    if np.var(patient_data['y_craving']) != 0:
                        model1 = ARIMA(patient_data['y_craving'], order=(p, 0, q), exog=patient_data['z_cues'])
                        result1 = model1.fit()
                        if result1.aic < best_aic_model1:
                            best_aic_model1 = result1.aic
                            best_order_model1 = (p, q)
                except Exception as e:
                    continue

            optimized_models.append((patient, best_order_model1, best_aic_model1))

    optimized_models_df = pd.DataFrame(optimized_models, columns=['Patient', 'Best Order Model 1', 'Best AIC Model 1'])
    return optimized_models_df

# Analyze Data_cocaine.xlsx
df_cocaine = analyze_data('/content/Data_cocaine.xlsx')

# Analyze Data_model_craving.xlsx (replace with actual file path if different)
df_craving = analyze_data('/content/Data_model_craving.xlsx')

# Comparison
comparison_df = pd.merge(df_cocaine, df_craving, on='Patient', suffixes=('_cocaine', '_craving'))

# Example of significance testing (AIC difference)
comparison_df['AIC_Difference'] = comparison_df['Best AIC Model 1_cocaine'] - comparison_df['Best AIC Model 1_craving']
print("Comparison of AIC values between datasets:")
print(comparison_df)

# Calculate a p-value for the difference
from scipy import stats
t_statistic, p_value = stats.ttest_ind(comparison_df['Best AIC Model 1_cocaine'], comparison_df['Best AIC Model 1_craving'])
print(f"T-statistic: {t_statistic}, P-value: {p_value}")
if p_value > 0.05:
    print("No difference in AIC")
else:
    print("Significant difference in AIC")
