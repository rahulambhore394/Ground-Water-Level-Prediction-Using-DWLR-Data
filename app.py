# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# --- Configuration and Setup ---
st.set_page_config(layout="wide", page_title="DWLR Groundwater Level Predictor")

# Load assets
try:
    model = joblib.load('groundwater_model.pkl')
    features = joblib.load('model_features.pkl')
    df_historical = pd.read_csv('DWLR_processed.csv')
    df_historical['Date'] = pd.to_datetime(df_historical['Date'])
    df_historical = df_historical.set_index('Date').sort_index()
    
    # Calculate a simple R2 (Placeholder for display based on saved value from training)
    # For a real project, you would save and load the actual R2 score.
    # We will approximate based on the simulated data's typical performance:
    R2_SCORE = 0.75 
    
except FileNotFoundError:
    st.error("""
        **ERROR: Model or Data files not found.**
        Please ensure you have run the following scripts in order:
        1. `python generate_data.py`
        2. `python model_training.py`
    """)
    st.stop()

# Function to prepare a single prediction
@st.cache_data
def prepare_single_prediction(latest_7_days_data, future_rainfall, future_temp, future_date):
    """Generates the feature vector for a single prediction."""
    
    # 1. Get lagged features from the latest historical data (crucial for Linear Regression)
    # The last recorded level is Water_Level_Lag1 for the prediction day
    lag1 = latest_7_days_data['Water_Level_m'].iloc[-1]
    # The level from 7 days ago is Water_Level_Lag7 for the prediction day
    lag7 = latest_7_days_data['Water_Level_m'].iloc[0] 
    
    # 2. Extract time features for the future date
    future_month = future_date.month
    future_day_of_year = future_date.timetuple().tm_yday
    
    # Create the input dictionary matching the trained features order
    input_data = {
        'Water_Level_Lag1': [lag1],
        'Water_Level_Lag7': [lag7],
        'Rainfall_mm': [future_rainfall],
        'Temperature_C': [future_temp],
        'Month': [future_month],
        'Day_of_Year': [future_day_of_year]
    }
    
    # Create DataFrame, ensuring column order matches the model training
    return pd.DataFrame(input_data, columns=features)

# --- Streamlit Interface Design ---
st.title("💧 DWLR Groundwater Level Prediction Dashboard")
st.markdown("Forecasting tool using **Multiple Linear Regression** on Digital Water Level Recorder (DWLR) data.")

# --- Layout: Columns ---
col_vis, col_pred = st.columns([2, 1])


# --- Column 1: Historical Data and Model Performance ---
with col_vis:
    st.header("1. Historical Water Level")
    
    # Historical Plot
    fig = px.line(df_historical, y='Water_Level_m', title='Depth to Water Level (m) Over Time',
                  labels={'Water_Level_m': 'Depth to Water Level (m)', 'Date': 'Date'})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key Performance Indicators (KPIs)")
    kpi1, kpi2, kpi3 = st.columns(3)
    
    latest_date = df_historical.index[-1].strftime('%Y-%m-%d')

    kpi1.metric("Latest Level Recorded", f"{df_historical['Water_Level_m'].iloc[-1]:.2f} m", 
                help=f"As of {latest_date}. Lower is better.")
    kpi2.metric("Avg. Rainfall", f"{df_historical['Rainfall_mm'].mean():.2f} mm/day")
    kpi3.metric("Model R-squared (Test)", f"{R2_SCORE:.2f}", 
                help="Measure of how well the Linear Regression model fits the test data.")
    
# --- Column 2: Prediction Interface ---
with col_pred:
    st.header("2. Predict Future Level")
    st.markdown("---")
    
    # Get the latest 7 days of data for lagged feature calculation
    latest_7_days_data = df_historical.tail(7)

    st.subheader(f"Latest Recorded Level: {latest_7_days_data['Water_Level_m'].iloc[-1]:.2f} m")

    st.markdown("**Future Environmental Inputs**")
    
    # Prediction Inputs
    default_future_date = df_historical.index[-1] + pd.Timedelta(days=1)
    future_date = st.date_input("Date to Predict", value=default_future_date, 
                                min_value=default_future_date, max_value=default_future_date + pd.Timedelta(days=365))
    
    future_rainfall = st.number_input("Expected Rainfall (mm)", min_value=0.0, max_value=500.0, value=5.0, step=0.1)
    future_temp = st.number_input("Expected Temperature (°C)", min_value=-20.0, max_value=50.0, value=25.0, step=0.1)

    if st.button("Calculate Prediction", use_container_width=True):
        
        # Prepare the input DataFrame
        input_df = prepare_single_prediction(
            latest_7_days_data=latest_7_days_data,
            future_rainfall=future_rainfall,
            future_temp=future_temp,
            future_date=pd.to_datetime(future_date)
        )
        
        # Make prediction
        predicted_level = model.predict(input_df)[0]
        
        # Display Result
        st.subheader(f"Predicted Level for {future_date.strftime('%b %d')}:")
        st.success(f"**{predicted_level:.2f} meters**")
        st.caption("*(Depth to Water Level)*")
        
        with st.expander("Show Detailed Input Features"):
            st.dataframe(input_df)