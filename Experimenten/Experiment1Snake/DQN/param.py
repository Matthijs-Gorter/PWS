import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, ParameterSampler, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import os

# Laad dataset
data = load_iris()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter ruimte
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Aantal testen
n_iter = 50

# Genereer parametercombinaties
param_list = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=42))

# Maak log map
os.makedirs('hyperparameter_logs', exist_ok=True)

# Uitvoeren van alle testen
for i, params in enumerate(param_list):
    model = RandomForestClassifier(**params)
    
    # Cross-validatie
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    # Maak resultaten dataframe
    results_df = pd.DataFrame({
        **params,
        'mean_accuracy': mean_score,
        'std_accuracy': std_score
    }, index=[0])
    
    # Sla op als CSV
    filename = os.path.join('hyperparameter_logs', f'trial_{i+1:03d}.csv')
    results_df.to_csv(filename, index=False)
    print(f'Test {i+1}/{n_iter} opgeslagen in {filename}')

# Vind beste parameters
log_files = [os.path.join('hyperparameter_logs', f) for f in os.listdir('hyperparameter_logs') if f.endswith('.csv')]
all_results = pd.concat([pd.read_csv(f) for f in log_files], ignore_index=True)
best_params = all_results.loc[all_results['mean_accuracy'].idxmax()]

print("\nBeste hyperparameters:")
print(best_params.to_string())