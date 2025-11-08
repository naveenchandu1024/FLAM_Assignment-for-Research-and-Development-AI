import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
from model import model

print("Loading dataset...")


df = pd.read_csv("../data/xy_data_with_t.csv")
x_data = df["x"].values
y_data = df["y"].values
t_data = df["t_est"].values

print(f"Data loaded successfully! Total points: {len(t_data)}")


data_points = np.column_stack((x_data, y_data))

# Residuals function for Phase A (L2)
def residuals(params, t, data_points):
    pred = model(params, t)
    return (pred - data_points).ravel()


initial_guess = [0.8, 0.0, 11.5]

# Bounds for Phase A 
lb = [0, -0.05, 0]
ub = [np.deg2rad(50), 0.05, 100]

print("\n Phase A: Running L2 least squares optimization...")
res_ls = least_squares(residuals, initial_guess, args=(t_data, data_points), bounds=(lb, ub))
theta_ls, M_ls, X_ls = res_ls.x
print(" Phase A complete! Initial fit parameters:")
print(f"  θ = {theta_ls:.4f}, M = {M_ls:.4f}, X = {X_ls:.4f}")

# Phase B: L1 minimization
def L1_objective(params):
    pred = model(params, t_data)
    return np.sum(np.abs(pred[:,0] - x_data) + np.abs(pred[:,1] - y_data))

bounds_phaseB = list(zip(lb, ub))
print("\n Phase B: Refining parameters by L1 minimization with maxiter=1000...")
res_l1 = minimize(L1_objective, x0=res_ls.x, bounds=bounds_phaseB, method='SLSQP', options={'maxiter':10000})
theta_fit, M_fit, X_fit = res_l1.x

print("\n Optimization complete!")
print("Estimated parameters after Phase B:")
print(f"  θ (theta) = {theta_fit:.4f} rad ({np.rad2deg(theta_fit):.2f}°)")
print(f"  M         = {M_fit:.4f}")
print(f"  X         = {X_fit:.4f}")

np.save("../data/fit_params.npy", res_l1.x)
print("\nParameters saved to '../data/fit_params.npy'")

latex_eq = f"\\left(t*\\cos({theta_fit}) - e^{{{M_fit}|t|}}*\\sin(0.3*t)*\\sin({theta_fit}) + {X_fit}, 42 + t*\\sin({theta_fit}) + e^{{{M_fit}|t|}}*\\sin(0.3*t)*\\cos({theta_fit})\\right)"
print("\nLaTeX Equation for Submission:\n", latex_eq)
