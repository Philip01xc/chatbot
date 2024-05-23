from flask import Flask, request, jsonify, render_template
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import io
import base64
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)

# Load and preprocess data
file_path = "C:/Users/Dell/Desktop/pay_loads_202405170848/pay_loads_202405170848.csv"
data = pd.read_csv(file_path)
print(data)

# Ensure 'created_at' is in datetime format
data['created_at'] = pd.to_datetime(data['created_at'])

# Define holidays
holidays = pd.to_datetime(['2023-12-25', '2024-01-01', '2024-04-07', '2024-05-01', '2024-12-25', '2024-12-31'])

# Create daily transactions data
time_series_data = data.set_index('created_at')
daily_transactions = time_series_data.resample('D').size()
df = pd.DataFrame({'y': daily_transactions})
print(df)
df['ds'] = df.index
df['holiday'] = df['ds'].isin(holidays).astype(int)

# Check for stationarity and difference the data if necessary
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    return result[1]  # p-value

p_value = check_stationarity(df['y'])
if p_value > 0.05:
    df['y_diff'] = df['y'].diff().dropna()
else:
    df['y_diff'] = df['y']

print("Opening index function")
@app.route('/')
def index():
    return render_template('index.html')

print("Using chat function")
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '').lower()
    
    if 'forecast' in user_input:
        forecast_image = create_forecast()
        response = {
            'message': 'Here is the forecast for the next 30 days:',
            'image': forecast_image
        }
    else:
        response = {
            'message': "I didn't understand that. Type 'forecast' to see the predictions."
        }

    return jsonify(response)

print("Creating a forecast")
def create_forecast():
    try:
        # Use the differenced data if necessary
        if 'y_diff' in df.columns:
            y = df['y_diff'].dropna()
        else:
            y = df['y']
        
        print("Starting the ARIMA model")
        print(f"y: {y}")
        print(f"Exogenous Variables (holidays): {df['holiday']}")

        # Adjust ARIMA parameters if necessary
        model = ARIMA(y, exog=df['holiday'], order=(5, 1, 0))
        model_fit = model.fit()

        print("ARIMA model fitted")

        future_holidays = pd.DataFrame({'ds': pd.date_range(start=daily_transactions.index[-1] + pd.Timedelta(days=1), periods=30)})
        future_holidays['holiday'] = future_holidays['ds'].isin(holidays).astype(int)

        forecast_steps = 30
        forecast = model_fit.forecast(steps=forecast_steps, exog=future_holidays['holiday'])

        forecast_dates = pd.date_range(start=daily_transactions.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)
        forecast_series = pd.Series(forecast, index=forecast_dates)

        plt.figure(figsize=(12, 6))
        plt.plot(daily_transactions, label='Observed')
        plt.plot(forecast_series, label='Forecast with Holiday Effects', color='orange')
        plt.title('Transaction Forecast for the Next 30 Days')
        plt.xlabel('Date')
        plt.ylabel('Number of Transactions')
        plt.legend()
        buf = io.BytesIO()
        print("Saving image chart")
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f'data:image/png;base64,{image_base64}'
    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
