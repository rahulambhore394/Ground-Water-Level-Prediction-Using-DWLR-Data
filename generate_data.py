# generate_data.py
import pandas as pd
import numpy as np

# Set a seed for reproducibility
np.random.seed(42)

# Create 500 days of data
N_DAYS = 500
dates = pd.date_range(start='2023-01-01', periods=N_DAYS, freq='D')

# Simulate Rainfall (higher in summer/monsoon, lower otherwise)
rainfall = np.abs(np.sin(np.linspace(0, 2 * np.pi, N_DAYS) * 2) * 10) + np.random.rand(N_DAYS) * 5

# Simulate Temperature
temp = 25 + np.sin(np.linspace(0, 2 * np.pi, N_DAYS)) * 10 + np.random.randn(N_DAYS) * 2

# Simulate Depth_to_Water_Level_m (Target variable: lower is better/closer to surface)
# Trend: Water level is influenced by a slight annual trend + rainfall (negative correlation: more rain -> lower depth) + its own historical level.
base_level = 15 + np.cumsum(np.random.randn(N_DAYS) * 0.05)
water_level_m = base_level - rainfall * 0.5 + np.random.randn(N_DAYS) * 0.5

# Ensure water level stays positive and within a reasonable range
water_level_m = np.clip(water_level_m, 10, 30)

data = pd.DataFrame({
    'Date': dates,
    'Water_Level_m': water_level_m.round(2),
    'Rainfall_mm': rainfall.round(2),
    'Temperature_C': temp.round(2)
})

data.to_csv('DWLR_simulated_data.csv', index=False)
print("✅ DWLR_simulated_data.csv created successfully.")