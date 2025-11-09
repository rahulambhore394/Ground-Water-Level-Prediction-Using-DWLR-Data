# model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import json
import numpy as np

# --- IMPORTANT: PLACE YOUR ACTUAL DWLR DATASET HERE ---
DATA_PATH = 'DWLR_processed.csv' 
COEFF_FILE = 'model_coefficients.json'
TARGET_COL = 'Water_Level_m'
FEATURE_COLS = ['Rainfall_mm', 'Temperature_C'] 

def preprocess_and_train():
    try:
        # Load the actual DWLR data
        df = pd.read_csv(DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {DATA_PATH}. Please provide your actual DWLR data CSV.")
        return
    except KeyError as e:
        print(f"ERROR: Column {e} is missing in your DWLR_Actual_Data.csv.")
        return

    print("\n--- Data Head ---")
    print(df.head())
    
    # 1. Handle Missing Values
    df = df.fillna(method='ffill').dropna()
    print(f"\nData size after cleaning: {len(df)}")
    
    # --- Feature Engineering ---
    df['Water_Level_Lag1'] = df[TARGET_COL].shift(1)
    df['Water_Level_Lag7'] = df[TARGET_COL].shift(7)
    df['Month'] = df.index.month
    df['Day_of_Year'] = df.index.dayofyear
    df = df.dropna() 

    # Define features (X) and target (Y)
    features = ['Water_Level_Lag1', 'Water_Level_Lag7'] + FEATURE_COLS + ['Month', 'Day_of_Year']
    X = df[features]
    Y = df[TARGET_COL]
    
    # Split data for training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    
    # --- Model Training: Linear Regression ---
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # --- Model Evaluation ---
    Y_pred = model.predict(X_test)
    r2 = r2_score(Y_test, Y_pred)
    
    print("\n--- Model Performance (on Test Set) ---")
    print(f"R-squared (R2): {r2:.4f}")

    # --- Save Model Parameters and Data for HTML/JS ---
    model_params = {
        'intercept': model.intercept_,
        'coefficients': dict(zip(features, model.coef_)),
        'r2_score': r2
    }
    
    # Get the latest 7 days of data for the dashboard's prediction calculation
    latest_data = df.tail(7)[TARGET_COL].tolist()
    
    output_data = {
        'model_params': model_params,
        'latest_water_levels': {
            'latest_level': latest_data[-1],
            'lag1': latest_data[-1],
            'lag7': latest_data[0] # The first of the last 7 days is the 7-day lag
        },
        'historical_data': {
            'dates': df.index.strftime('%Y-%m-%d').tolist(),
            'levels': df[TARGET_COL].tolist()
        },
        'avg_rainfall': df['Rainfall_mm'].mean()
    }

    with open(COEFF_FILE, 'w') as f:
        json.dump(output_data, f)

    print(f"✅ Model parameters and data saved to {COEFF_FILE} for the dashboard.")

if __name__ == "__main__":
    # --- SIMULATE DATA CREATION IF YOUR REAL FILE IS MISSING ---
    # This block ensures the script runs for demonstration purposes if you haven't provided the CSV.
    if not pd.io.common.file_exists(DATA_PATH):
        print("\nWARNING: DWLR_Actual_Data.csv not found. Creating a simulated file for demonstration.")
        N_DAYS = 500
        dates = pd.date_range(start='2023-01-01', periods=N_DAYS, freq='D')
        np.random.seed(42)
        rainfall = np.abs(np.sin(np.linspace(0, 2 * np.pi, N_DAYS) * 2) * 10) + np.random.rand(N_DAYS) * 5
        base_level = 15 + np.cumsum(np.random.randn(N_DAYS) * 0.05)
        water_level_m = np.clip(base_level - rainfall * 0.5 + np.random.randn(N_DAYS) * 0.5, 10, 30)
        temp = 25 + np.sin(np.linspace(0, 2 * np.pi, N_DAYS)) * 10 + np.random.randn(N_DAYS) * 2
        data = pd.DataFrame({'Date': dates, TARGET_COL: water_level_m.round(2), 'Rainfall_mm': rainfall.round(2), 'Temperature_C': temp.round(2)})
        data.to_csv(DATA_PATH, index=False)
        print("Simulated data created. Run the script again now.")
    else:
        preprocess_and_train()