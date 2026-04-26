import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten

import pyswarms as ps
from deap import base, creator, tools, algorithms
import random
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r'D:\gitRepos\EV-Charging-Load-Forecasting-ML\Drixter\cleandata.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

print("--- STATIONARITY CHECK (ADF TEST) ---")

result_adf = adfuller(df['demand_kWh'].iloc[:5000]) 
print(f'ADF Statistic: {result_adf[0]:.4f}')
print(f'p-value: {result_adf[1]:.4f}')

plt.figure(figsize=(12, 8))
decomp = seasonal_decompose(df['demand_kWh'].iloc[:500], model='additive', period=24)
decomp.plot()
plt.suptitle('Seasonal Decomposition: Visualizing Patterns', fontsize=14)
plt.show()

plt.figure(figsize=(10, 4))
plot_acf(df['demand_kWh'].iloc[:500], lags=48)
plt.title("Autocorrelation: Seasonal Spikes at 24 and 48 hours")
plt.show()

data = df[['demand_kWh']].values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

train_size = int(len(data_scaled) * 0.8)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

print("\nFitting SARIMA Model...")
sarima_model = SARIMAX(train_data, order=(1,1,1), seasonal_order=(1,1,1,24))
sarima_results = sarima_model.fit(disp=False)

sarima_pred = sarima_results.get_forecast(steps=len(test_data)).predicted_mean
sarima_pred = sarima_pred.reshape(-1, 1)

residuals = test_data - sarima_pred

def create_sequences(data, window=24):
    X, y = [], []
    for i in range(len(data)-window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

window = 24
X_res, y_res = create_sequences(residuals, window)
X_res = X_res.reshape((X_res.shape[0], X_res.shape[1], 1))


def build_tcn(filters, kernel, lr):
    model = Sequential()
    model.add(Conv1D(filters=int(filters), kernel_size=int(kernel), activation='relu', input_shape=(window, 1)))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return model

def fitness_eval(filters, kernel, lr):
    kernel = max(2, min(int(kernel), window))
    model = build_tcn(filters, kernel, lr)
    model.fit(X_res, y_res, epochs=3, batch_size=32, verbose=0)
    pred = model.predict(X_res, verbose=0)
    return mean_squared_error(y_res, pred)

print("\nRunning PSO Optimization...")
bounds = ([16, 2, 0.0001], [128, 5, 0.01])
optimizer = ps.single.GlobalBestPSO(n_particles=5, dimensions=3, options={'c1':0.5, 'c2':0.3, 'w':0.9}, bounds=bounds)

best_cost_pso, best_pos_pso = optimizer.optimize(
    lambda x: np.array([fitness_eval(p[0], p[1], p[2]) for p in x]), iters=5)

print("PSO Best Hyperparameters:", best_pos_pso)
pso_model = build_tcn(best_pos_pso[0], best_pos_pso[1], best_pos_pso[2])
pso_model.fit(X_res, y_res, epochs=10, verbose=0)

print("\nRunning GA Optimization...")
if "FitnessMin" in creator.__dict__: del creator.FitnessMin
if "Individual" in creator.__dict__: del creator.Individual

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("filters", random.randint, 16, 128)
toolbox.register("kernel", random.randint, 2, 5)
toolbox.register("lr", random.uniform, 0.0001, 0.01)
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.filters, toolbox.kernel, toolbox.lr), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_individual(ind):
    return (fitness_eval(ind[0], ind[1], ind[2]),)

toolbox.register("evaluate", eval_individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

pop = toolbox.population(n=5)
algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=5, verbose=False)

best_ind = tools.selBest(pop, k=1)[0]
print("GA Best Hyperparameters:", best_ind)

ga_model = build_tcn(best_ind[0], best_ind[1], best_ind[2])
ga_model.fit(X_res, y_res, epochs=10, verbose=0)

res_pso = pso_model.predict(X_res).flatten()
res_ga = ga_model.predict(X_res).flatten()

final_pso = sarima_pred[window:].flatten() + res_pso
final_ga = sarima_pred[window:].flatten() + res_ga
actual = test_data.flatten()[window:]

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2*np.abs(y_pred-y_true)/(np.abs(y_true)+np.abs(y_pred)))

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    smape_val = smape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name} RESULTS")
    print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | SMAPE: {smape_val:.2f}% | R2: {r2:.4f}")
    return mae, rmse, smape_val, r2

metrics_pso = evaluate("SARIMA-PSO-TCN", actual, final_pso)
metrics_ga = evaluate("SARIMA-GA-TCN", actual, final_ga)

plt.figure(figsize=(16,6))
plt.plot(actual, label='Actual', color='black', alpha=0.7)
plt.plot(final_pso, label='PSO Hybrid', color='blue', linestyle='--')
plt.plot(final_ga, label='GA Hybrid', color='red', linestyle='-.')
plt.title("Final Hybrid Model Comparison: Actual vs Forecast")
plt.legend()
plt.show()