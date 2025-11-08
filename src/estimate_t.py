
import pandas as pd
import numpy as np
df = pd.read_csv("C:/Users/charan/Desktop/curve_fit_assignment/data/xy_data.csv")
x = df['x'].values
y = df['y'].values
dx = np.diff(x)
dy = np.diff(y)
dist = np.sqrt(dx**2 + dy**2)
cum_dist = np.insert(np.cumsum(dist), 0, 0)  
t_est = 6 + (cum_dist - cum_dist.min()) * (60 - 6) / (cum_dist.max() - cum_dist.min())
df['t_est'] = t_est
df.to_csv("../data/xy_data_with_t.csv", index=False)

print("Arc-length t_est values saved to '../data/xy_data_with_t.csv'")
print(df.head())
