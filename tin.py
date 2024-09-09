import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta

# Set up your API and base URL for fetching data
api_key = "l333ljg4122qws9kxkb4hly7a8dje27vk46c7zkceih11wmnrj7lqreku176"
base_url = "https://metals-api.com/api"

# Function to fetch data for a given timeframe, splitting into chunks of 365 days
def fetch_data(start_date, end_date):
    date_format = "%Y-%m-%d"
    start_date = datetime.strptime(start_date, date_format)
    end_date = datetime.strptime(end_date, date_format)

    all_data = {}

    while start_date <= end_date:
        current_end_date = min(start_date + timedelta(days=365), end_date)  # Fetch data in chunks of 365 days
        params = {
            "access_key": api_key,
            "base": "USD",
            "symbols": "TIN",
            "start_date": start_date.strftime(date_format),
            "end_date": current_end_date.strftime(date_format)
        }
        response = requests.get(f"{base_url}/timeseries", params=params)

        if response.status_code == 200:
            data = response.json()
            if data.get('success', False):
                all_data.update(data.get("rates", {}))
            else:
                st.error(f"API request failed: {data.get('error', {}).get('info')}")
                break
        else:
            st.error(f"Error fetching data: {response.status_code}")
            break

        start_date = current_end_date + timedelta(days=1)  # Move to the next chunk

    return all_data if all_data else None

# Streamlit App Configuration
st.set_page_config(page_title="Tin Price Prediction", layout="wide")

# Sidebar for user inputs
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Tin_-_periodic_table.jpg/330px-Tin_-_periodic_table.jpg",
        width=200)
    st.title("Tin Price Predictor")
    st.info("Select a start date to fetch data and predict future tin prices.")

    # User input for start date
    start_date = st.date_input("Start Date", datetime(2023, 7, 1))  # Default to a realistic date

    # User input for prediction period
    prediction_period = st.selectbox("Select Prediction Period", ["6 Months", "3 Months", "3 Weeks", "1 Week"])

    # Calculate the end date based on selected prediction period
    if prediction_period == "6 Months":
        end_date = start_date + timedelta(days=6 * 30)  # Approximate 6 months
    elif prediction_period == "3 Months":
        end_date = start_date + timedelta(days=3 * 30)  # Approximate 3 months
    elif prediction_period == "3 Weeks":
        end_date = start_date + timedelta(weeks=3)
    elif prediction_period == "1 Week":
        end_date = start_date + timedelta(weeks=1)

    st.write(f"Prediction period will end on: {end_date.strftime('%Y-%m-%d')}")

# Convert dates to strings for API
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Calculate the number of days for prediction
prediction_days = (end_date - start_date).days

# Main section for displaying data and results
st.title("Tin Price Prediction Dashboard")

# Fetch and combine data
data = fetch_data(start_date_str, end_date_str)

if data:
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.reset_index().rename(columns={"index": "ds", "TIN": "y"})
    df = df[["ds", "y"]]

    # Display data
    st.subheader("📊 Fetched Data")
    st.write(df.head(30))  # Display the data in the UI

    # Plot the data
    st.subheader("📈 Tin Price Over Time")
    st.line_chart(df.set_index('ds')['y'])


    # Prophet model training and forecasting
    st.subheader("🔮 Prophet Forecast")
    model = Prophet(
        changepoint_prior_scale=0.1,
        yearly_seasonality=True,
        weekly_seasonality=True
    )
    model.fit(df)

    future = model.make_future_dataframe(periods=prediction_days)
    forecast = model.predict(future)

    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # Evaluate the model
    st.subheader("📉 Model Performance Metrics")
    df_cv = cross_validation(model, initial='14 days', period='7 days', horizon='7 days')
    df_performance = performance_metrics(df_cv)
    st.write(df_performance)

    # Get user input for a specific prediction date
    st.subheader("📅 Predict Tin Price for a Specific Date")
    user_input = st.text_input("Enter the date for which you want to predict the price (YYYY-MM-DD):")

    if user_input:
        try:
            predicted_price = model.predict(pd.DataFrame({'ds': [user_input]}))['yhat'].values[0]
            st.success(f"The predicted price of tin on {user_input} is: ${predicted_price:.2f}")
            st.balloons()
        except Exception as e:
            st.error(f"Error predicting price: {e}")

    # ARIMA Model
    st.subheader("🔄 ARIMA Forecast")

    # Check for stationarity
    result = adfuller(df['y'])
    st.write('ADF Statistic:', result[0])
    st.write('p-value:', result[1])

    if result[1] < 0.05:  # The series is stationary
        arima_model = ARIMA(df['y'], order=(5, 1, 0))
        arima_result = arima_model.fit()

        arima_forecast = arima_result.get_forecast(steps=prediction_days)
        arima_conf_int = arima_forecast.conf_int()
        arima_pred = arima_forecast.predicted_mean

        # Plot ARIMA forecast
        plt.figure(figsize=(10, 6))
        plt.plot(df['ds'], df['y'], label='Historical')
        plt.plot(pd.date_range(start=df['ds'].iloc[-1], periods=prediction_days + 1, freq='D')[1:], arima_pred,
                 label='ARIMA Forecast')
        plt.fill_between(pd.date_range(start=df['ds'].iloc[-1], periods=prediction_days + 1, freq='D')[1:],
                         arima_conf_int.iloc[:, 0], arima_conf_int.iloc[:, 1], color='pink', alpha=0.3)
        plt.legend()
        plt.title('ARIMA Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price')
        st.pyplot(plt)
    else:
        st.write("The time series is not stationary. ARIMA might not provide reliable predictions.")
else:
    st.write("⚠️ No data fetched. Please check the date range or API details.")

# Custom CSS for styling
st.markdown("""
    <style>
        .css-18e3th9 {
            padding: 1.5rem 1rem;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-size: 1rem;
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }
        .css-1v0mbdj {
            display: flex;
            justify-content: center;
        }
        .css-1adrfps {
            color: #FF6347;
        }
    </style>
    """, unsafe_allow_html=True)
