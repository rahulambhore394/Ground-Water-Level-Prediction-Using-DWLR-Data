# model_training.py (CORRECTED)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib 

DATA_PATH = 'DWLR_processed.csv'

def preprocess_and_train():
    try:
        # Load the simulated data
        df = pd.read_csv(DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {DATA_PATH}. Please run generate_data.py first.")
        return

    # **CORRECTION:** Removed Streamlit st.write/st.dataframe
    print("\n--- Data Head ---")
    print(df.head())

    # --- Feature Engineering for Time Series Linear Regression ---
    
    # 1. Lagged Features (Previous day and previous week water level)
    # This is the most crucial part for time series prediction with LR.
    df['Water_Level_Lag1'] = df['Water_Level_m'].shift(1) # Yesterday's level
    df['Water_Level_Lag7'] = df['Water_Level_m'].shift(7) # Level from 7 days ago
    
    # 2. Time-based Features (Seasonality)
    df['Month'] = df.index.month
    df['Day_of_Year'] = df.index.dayofyear
    
    # Drop rows with NaN values created by the shift (first 7 rows)
    df = df.dropna() 

    # Define features (X) and target (Y)
    features = ['Water_Level_Lag1', 'Water_Level_Lag7', 'Rainfall_mm', 'Temperature_C', 'Month', 'Day_of_Year']
    X = df[features]
    Y = df['Water_Level_m']
    
    # Split data for training and testing
    # Use shuffle=False for time-series data
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
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

    # --- Save Model and Data for Streamlit App ---
    joblib.dump(model, 'groundwater_model.pkl')
    joblib.dump(features, 'model_features.pkl')
    
    # Save the full processed dataset for the app's visualization
    df.to_csv('DWLR_processed.csv')

    print("✅ Model trained and saved successfully (groundwater_model.pkl, model_features.pkl).")
    print("✅ Processed data saved (DWLR_processed.csv).")

if __name__ == "__main__":
    preprocess_and_train()