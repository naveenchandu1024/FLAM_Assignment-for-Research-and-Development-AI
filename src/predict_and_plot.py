import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import model
from scipy.interpolate import interp1d


print(" Loading optimized parameters...")
params = np.load("../data/fit_params.npy")
theta, M, X = params
print(f"Parameters loaded: θ={theta:.4f}, M={M:.4f}, X={X:.4f}")


print("\n Loading dataset...")
df = pd.read_csv("../data/xy_data_with_t.csv")
t_data = df["t_est"].values
x_actual = df["x"].values
y_actual = df["y"].values
print(f"Data loaded successfully! Total points: {len(df)}")

t_uniform = np.linspace(6, 60, 500)


pred_uniform = model(params, t_uniform)
x_pred, y_pred = pred_uniform[:, 0], pred_uniform[:, 1]

f_x = interp1d(t_data, x_actual, kind='cubic', fill_value='extrapolate')
f_y = interp1d(t_data, y_actual, kind='cubic', fill_value='extrapolate')
x_expected = f_x(t_uniform)
y_expected = f_y(t_uniform)

L1_uniform = np.sum(np.abs(x_pred - x_expected) + np.abs(y_pred - y_expected))
print(f"\nL1 error (uniform sampling) = {L1_uniform:.4f}")

#  Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(x_actual, y_actual, s=8, label="Actual Data", alpha=0.6)
plt.plot(x_pred, y_pred, color="red", linewidth=2, label="Fitted Curve (Predicted)")
plt.title("Actual vs Predicted Data (Parametric Curve Fit)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../data/fit.png", dpi=200)
plt.show()

print("\n Plot generated and saved as 'fit.png'.")
