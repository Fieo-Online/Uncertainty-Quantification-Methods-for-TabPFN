from tabpfn import TabPFNRegressor
from tabicl import TabICLRegressor
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import openml, time, os
from src import dataloader, training, utils, uq_Methods
from datetime import datetime
from sklearn.neural_network import MLPRegressor
os.environ["TABPFN_TOKEN"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiZGM3YzVkZmQtYTJmNC00MDZmLWIxMjYtNjgzN2JlYWQ5NTMzIiwiZXhwIjoxODEwNDY1ODg0fQ.kT5eCkRjpz4UMq0ipkEwJq7HK1rwqKVgLgs9UVjttn8"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
dataset = openml.datasets.get_dataset(46772)
#X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
X, y, uq, y_true = dataloader.generate_synthetic_data(type="linear", noise=True)
#X, y = dataloader.sklearn_data("fetch_california_housing")
print(f"This is X: {X}")
print(f"This is y: {y}")
#print(f"This is Uncertainty per point: {uq}")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)
#X_train = dataloader.create_feature_noise(X_train, noise_level=0.2)
#y_train = dataloader.create_target_noise(y_train, noise_level=0.2)

model = TabPFNRegressor(ignore_pretraining_limits=True)
modelB = TabICLRegressor()
start = time.time()
model.fit(X_train, y_train)
modelB.fit(X_train,y_train)

quantile_grid = np.linspace(0.01, 0.99, 99)
preds = model.predict(X_test[:1], output_type="quantiles", quantiles=[0.1,0.9])
predsB = modelB.predict(X_test[:1],output_type="quantiles", quantiles=[0.1,0.9])
print(preds)
print(predsB)
#martingale = uq_Methods.martingale(preds)
#martingale_uq = uq_Methods.martingale_uq(martingale)
#print(*martingale, sep="\n")
#print(f"These are Vars:{martingale_uq}")

#preds = model.predict(X_test)
#predsB = modelB.predict(X_test)

#pred_q = np.array(preds)
#uq_test = pred_q[2]-pred_q[0]
#print("Runtime:"+str(time.time()-start))
#print(len(uq_test))
#print(uq_test)
#print(len(predsB))
##utils.simple_plot(preds,predsB,timestamp)
#utils.mse_plot(y_test,preds[50],predsB,martingale_uq)
y_min = float(np.min(y_train))
y_max = float(np.max(y_train))
y_grid = np.linspace(y_min,y_max,100)
quantiles_levels = np.linspace(0.01, 0.99, 99) 

# tabpfn_preds hat die Shape (99, 200) -> 99 Quantile für 200 Punkte
tabpfn_preds = np.array(preds) 

n_eval = tabpfn_preds.shape[1] # 200 Punkte
K = len(y_grid)                # Deine Stützstellen (z.B. 100)

# Leere Matrix für den korrekten Input vorbereiten (Shape: K x n_eval)
cdf_conditionals = np.zeros((K, n_eval))

# Für jeden Testpunkt die Quantile in eine CDF umwandeln
for i in range(n_eval):
    # Wichtig: y_grid sind die x-Werte für interp, 
    # tabpfn_preds[:, i] sind die bekannten Y-Werte der Quantile,
    # quantiles_levels sind die bekannten Wahrscheinlichkeiten.
    cdf_conditionals[:, i] = np.interp(
        y_grid, 
        xp=tabpfn_preds[:, i],    # Die von TabPFN vorhergesagten Y-Werte
        fp=quantiles_levels,      # Die dazugehörigen Wahrscheinlichkeiten (0.01 bis 0.99)
        left=0.0, right=1.0       # Standardwerte außerhalb des Bereichs
    )
ppd = uq_Methods.get_posterior(y_grid,len(y_train),100,50,cdf_conditionals,alpha_dimension=1)
test = uq_Methods.martingale_uq(ppd)
utils.mse_plot_(y_true,preds[49], test)
#utils.plot(X_test,preds)
#utils.plot(X_test,predsB)

#print("MSE:"+str(mean_squared_error(y_test, preds)))

#ppd_x, ppd_y = utils.get_ppd(preds, quantile_grid)
#utils.simple_plot(y_test, preds, timestamp)
#utils.plot(ppd_x,ppd_y)


#plt.plot(X, y, marker='o', color='royalblue', linewidth=2)
#plt.title("Funktion")
#plt.show()



# Plot anzeigen
#plt.show()
"""
# Wir plotten nur die ersten 50 Datenpunkte, damit man etwas erkennt
pred_q = np.array(pred_q).T #Zeilen und Spalten tauschen damit es funktioniert
n_samples = 50
x_axis = np.arange(n_samples)

# Get quantiles from pre_q
q_low = pred_q[:n_samples, 0]   # 0.05 Quantil
q_med = pred_q[:n_samples, 1]   # 0.50 Quantil (Median)
q_high = pred_q[:n_samples, 2]  # 0.95 Quantil

# Create plot
plt.figure(figsize=(12, 6))

# 0.9 quantile is in backround
plt.fill_between(x_axis, q_low, q_high, color='orange', alpha=0.3, label='90% Vorhersageintervall')

# Median as line
plt.plot(x_axis, q_med, color='red', lw=2, label='Median-Vorhersage (q=0.5)')

# 3. Real values as points
plt.scatter(x_axis, y_test[:n_samples], color='blue', s=30, label='Echte Werte', zorder=3)

plt.title('TabPFN Uncertainty Quantification: 90% Prediction Intervals')
plt.xlabel('Index des Test-Datensatzes')
plt.ylabel('Zielvariable (Target)')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()



plt.figure(figsize=(6,6))

#Shows correlation between Prediction and testdata. When most of the data have the about the same value, a linear function will form
plt.scatter(y_test, preds, alpha=0.5)

min_val = min(y_test.min(), preds.min())
max_val = max(y_test.max(), preds.max())

plt.plot([min_val, max_val], [min_val, max_val])

plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Prediction vs Ground Truth")

#The closer the plot-values are to the x-axis(to x=0) the better the prediction
residuals = y_test - preds

plt.figure(figsize=(7,5))

plt.scatter(preds, residuals, alpha=0.5)

plt.axhline(0)

plt.xlabel("Predictions")
plt.ylabel("Residuals")
plt.title("Residual Plot")

#Histogramm gives a better overview of outliers in the plot. The less outliers the more centered the plot is
plt.figure(figsize=(7,5))

plt.hist(residuals, bins=30)

plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Residual Distribution")

plt.show()
"""