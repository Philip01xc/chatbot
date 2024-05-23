from flask import Flask, request, jsonify, render_template
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load and preprocess data
file_path = "C:/Users/Dell/Desktop/pay_loads_202405170848/pay_loads_202405170848.csv"
data = pd.read_csv(file_path)

# Ensure 'created_at' is in datetime format
data['created_at'] = pd.to_datetime(data['created_at'])

# Define holidays
holidays = pd.to_datetime(['2023-12-25', '2024-01-01', '2024-04-07', '2024-05-01', '2024-12-25', '2024-12-31'])

# Create daily transactions data
time_series_data = data.set_index('created_at')
daily_transactions = time_series_data.resample('D').size()
df = pd.DataFrame({'y': daily_transactions})
df['ds'] = df.index
df['holiday'] = df['ds'].isin(holidays).astype(int)

@app.route('/')
def index():
    return render_template('index.html')

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

def create_forecast():
    model = ARIMA(df['y'], exog=df['holiday'], order=(5, 1, 0))
    model_fit = model.fit()

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
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{image_base64}'

if __name__ == "__main__":
    app.run(debug=True)
ECHO is on.
