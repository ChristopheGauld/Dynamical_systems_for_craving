"""
Créé : Oct 28 2023
@Author: Christophe Gauld 
"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from itertools import product

file_path = '/Users/christophe/Desktop/DynaPsy/2 Dyna EMA Craving/Data/Data_model_craving.xlsx' 
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
                model1 = ARIMA(patient_data['y_craving'], order=(p, 0, q), exog=patient_data['z_cues'])
                result1 = model1.fit()
                if result1.aic < best_aic_model1:
                    best_aic_model1 = result1.aic
                    best_order_model1 = (p, q)

                # Model 2: Cues dependant, Craving exogene
                model2 = ARIMA(patient_data['z_cues'], order=(p, 0, q), exog=patient_data['y_craving'])
                result2 = model2.fit()
                if result2.aic < best_aic_model2:
                    best_aic_model2 = result2.aic
                    best_order_model2 = (p, q)

            except Exception as e:
                continue
        
        # Calcul de l'auto-corrélation croisée pour déterminer le lag optimal
        cross_correlation = sm.tsa.stattools.ccf(patient_data['y_craving'], patient_data['z_cues'])
        max_corr_lag = np.argmax(np.abs(cross_correlation[:10]))  # Limite à un lag max de 10

        optimized_models.append((patient, best_order_model1, best_aic_model1, best_order_model2, best_aic_model2))

optimized_models_df = pd.DataFrame(optimized_models, columns=['Patient', 'Best Order Model 1', 'Best AIC Model 1', 'Best Order Model 2', 'Best AIC Model 2'])

for i in range(len(optimized_models_df)):
    if optimized_models_df.loc[i, 'Best AIC Model 1'] < optimized_models_df.loc[i, 'Best AIC Model 2']:
        optimized_models_df.loc[i, 'Preferred Model'] = 'Model 1 (Craving)'
    else:
        optimized_models_df.loc[i, 'Preferred Model'] = 'Model 2 (Cues)'

print(optimized_models_df)

