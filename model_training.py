# model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib 

# --- IMPORTANT: PLACE YOUR ACTUAL DWLR DATASET HERE ---
DATA_PATH = 'DWLR_processed.csv' 

# Define the expected columns in your dataset
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
    
    # Check for target/feature presence (minimal check)
    if TARGET_COL not in df.columns or not all(col in df.columns for col in FEATURE_COLS):
        print(f"ERROR: Ensure your CSV has columns: Date, {TARGET_COL}, {FEATURE_COLS}")
        return

    # 1. Handle Missing Values (Impute with forward fill)
    df = df.fillna(method='ffill').dropna()
    print(f"\nData size after cleaning: {len(df)}")
    
    # --- Feature Engineering for Time Series Linear Regression ---
    
    # 2. Lagged Features (The core of time-series LR)
    df['Water_Level_Lag1'] = df[TARGET_COL].shift(1) # Yesterday's level
    df['Water_Level_Lag7'] = df[TARGET_COL].shift(7) # Level from 7 days ago
    
    # 3. Time-based Features (Seasonality)
    df['Month'] = df.index.month
    df['Day_of_Year'] = df.index.dayofyear
    
    # Drop rows with NaN values created by the shift (first 7 rows)
    df = df.dropna() 

    # Define features (X) and target (Y)
    features = ['Water_Level_Lag1', 'Water_Level_Lag7'] + FEATURE_COLS + ['Month', 'Day_of_Year']
    X = df[features]
    Y = df[TARGET_COL]
    
    # Split data for training and testing (shuffle=False for time-series)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

    print(f"\nTotal Samples (After Lagging): {len(df)}")
    print(f"Training Samples: {len(X_train)}")
    print(f"Testing Samples: {len(X_test)}")
    
    # --- Model Training: Linear Regression ---
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # --- Model Evaluation ---
    Y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    
    print("\n--- Model Performance (on Test Set) ---")
    print(f"R-squared (R2): {r2:.4f}")
    
    # --- Save Model and Data for Streamlit App ---
    joblib.dump(model, 'groundwater_model.pkl')
    joblib.dump(features, 'model_features.pkl')
    joblib.dump(r2, 'model_r2_score.pkl') # Save R2 for the dashboard
    
    # Save the full processed dataset for the app's visualization
    df.to_csv('DWLR_processed.csv')

    print("✅ Model trained and saved successfully.")

if __name__ == "__main__":
    preprocess_and_train()