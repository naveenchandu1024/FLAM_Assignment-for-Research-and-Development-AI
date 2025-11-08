import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from model import model

params = np.load("../data/fit_params.npy")
theta_fit, M_fit, X_fit = params


df = pd.read_csv("../data/xy_data_with_t.csv")
t_est = df["t_est"].values
x_data = df["x"].values
y_data = df["y"].values


t_uniform = np.linspace(6, 60, 500)  
f_x = interp1d(t_est, x_data, kind='cubic', fill_value='extrapolate')
f_y = interp1d(t_est, y_data, kind='cubic', fill_value='extrapolate')

x_expected = f_x(t_uniform)
y_expected = f_y(t_uniform)


pred = model(params, t_uniform)
x_pred, y_pred = pred[:, 0], pred[:, 1]


L1_distance = np.sum(np.abs(x_pred - x_expected) + np.abs(y_pred - y_expected))

print(f"✅ L1 distance (uniform 500 samples) = {L1_distance:.4f}")
