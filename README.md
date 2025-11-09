

# 💧 DWLR Groundwater Level Prediction (HTML/JS Deployment)

## Overview

This project provides a complete Time-Series Forecasting solution to predict the **Depth to Water Level** (in meters) using Actual Digital Water Level Recorder (DWLR) sensor data. The core model is a **Multiple Linear Regression (MLR)**, which utilizes lagged water levels, rainfall, and temperature as features.

The solution is split into a Python-based ML core for training and an HTML/JavaScript interface for the dashboard deployment, ensuring no external web frameworks (like Streamlit or Plotly) are needed for the frontend.

### Key Components

  * **Data Source:** `DWLR_Actual_Data.csv` (User-provided)
  * **Model:** Multiple Linear Regression (MLR)
  * **Model Persistence:** JSON (saving coefficients)
  * **Web Dashboard:** HTML, CSS, and pure JavaScript
  * **Visualization:** Chart.js (JavaScript Library)

-----

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

  * Python 3.8+
  * Web Browser (to open `index.html`)

### 1\. Installation

Open your terminal or command prompt and install the required Python libraries:

```bash
pip install pandas numpy scikit-learn
```

### 2\. Data Preparation

> **Crucial Step:** You must place your actual DWLR time-series data file in the project directory and name it:
> `DWLR_Actual_Data.csv`

Ensure this file contains at least the following columns: `Date`, `Water_Level_m`, `Rainfall_mm`, and `Temperature_C`.

### 3\. File Structure

Your project directory should contain these files:

```plaintext
dwlr_prediction/
├── index.html                  # The complete HTML/CSS/JavaScript Dashboard
├── model_training.py           # Python script for data processing and ML training
├── DWLR_Actual_Data.csv        # (Your actual DWLR data file)
└── model_coefficients.json     # (Generated in Step 4.1)
```

-----

## 💻 Usage

Run the following two commands in order from your project directory.

### 4.1 Step 1: Train the Model and Save Parameters

Run the Python training script. This will train the Linear Regression model, evaluate it, and save the necessary parameters (coefficients, intercept, and data) to a JSON file.

```bash
python model_training.py
```

Output: Model performance metrics and the creation of `model_coefficients.json`.

### 4.2 Step 2: Open the Dashboard

Open the `index.html` file directly in your web browser.

  * **Method:** Right-click on `index.html` and select "Open with" -\> "Your Browser" (Chrome, Firefox, Edge, etc.).

The JavaScript in the file will automatically load the `model_coefficients.json` data and make the dashboard functional.

-----

## 🛠️ Project Methodology: The Workflow

The project is executed in two distinct stages:

### Stage A: Backend (Python) - ML Core

The `model_training.py` script handles the entire machine learning workflow:

  * **Feature Engineering:** Creates Lagged Features (`Water_Level_Lag1`, `Lag7`) and Seasonal Features (`Month`, `Day_of_Year`) from the time-series data.
  * **Training:** Trains the Scikit-learn `LinearRegression` model.
  * **Deployment Prep:** Extracts the model's core mathematical components (intercept and coefficients) and essential historical data. This information is saved to `model_coefficients.json` for the frontend.

### Stage B: Frontend (HTML/JS) - Dashboard

The `index.html` file handles all user interaction without a server:

  * **Model Loading:** JavaScript uses the browser's file access to read and parse `model_coefficients.json`.
  * **Visualization:** `Chart.js` renders the historical water level plot using the data loaded from the JSON file.
  * **Live Prediction:** When the user enters inputs, the JavaScript function `makePrediction()` performs the complete Linear Regression equation using the loaded coefficients and the new feature inputs (Lagged, Rainfall, Temperature, Month, Day\_of\_Year), calculating the forecast entirely within the browser.

-----

## 🧑‍🤝‍🧑 Team and Contribution

| Team Member | Contribution Area | Files / Tasks Handled |
| :--- | :--- | :--- |
| **Rahul Ambhore** | Data Preparation & Foundation | Data loading/integration (setting up `DWLR_Actual_Data.csv`), initial data cleaning, and foundational project setup. |
| **Aditya Ghuge** | ML Modeling & Persistence | Core feature engineering (Lagged Features), implementation of the Linear Regression model, model training (`model_training.py`), evaluation, and saving coefficients to JSON. |
| **Namdev Yevtikar** | Dashboard Development & Deployment | Creation of the HTML/CSS structure, all JavaScript prediction logic and data loading, and integration of `Chart.js` for visualizations (`index.html`). |