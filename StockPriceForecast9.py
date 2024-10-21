import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Load the data
file_path = '/vs code/MyProject/auto_regression/2330-training.csv'
data = pd.read_csv(file_path)

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')

# Clean data: Remove commas from numeric columns and convert them to float
for col in ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']:
    data[col] = data[col].replace({',': ''}, regex=True).astype(float)

# Prepare the data for Prophet model (Prophet requires 'ds' and 'y' columns)
data_prophet = pd.DataFrame()
data_prophet['ds'] = data['Date']  # Prophet requires the date column to be named 'ds'
data_prophet['y'] = data['x1']  # 'y' is the target column we want to forecast (e.g., closing price 'x1')

# Initialize Prophet model with higher uncertainty sampling and changepoint sensitivity
model = Prophet(interval_width=0.95, changepoint_prior_scale=0.5, uncertainty_samples=500)
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)  # Adding monthly seasonality

# Fit the model to the historical data
model.fit(data_prophet)

# Make a DataFrame to hold future dates (2 months = ~60 days)
future_dates = model.make_future_dataframe(periods=60)

# Predict future values
forecast = model.predict(future_dates)

# Plot the forecasted data with uncertainty intervals
fig, ax = plt.subplots(figsize=(10, 6))  # Single plot
model.plot(forecast, ax=ax)
ax.set_title('Stock Price Forecast with Enhanced Uncertainty Widening')

# Overlay the historical data (black dots) as a line for continuity
ax.plot(data_prophet['ds'], data_prophet['y'], color='black', lw=2, label='Actual Data')

# Add annotations for clarity (e.g., Forecast Initialization and Upward Trend)
ax.axvline(x=data_prophet['ds'].iloc[-1], color='red', linestyle='--')  # Forecast Initialization
ax.text(data_prophet['ds'].iloc[-1], forecast['yhat'].iloc[-1] + 100, 'Forecast Initialization', color='red')
ax.annotate('Upward Trend', xy=(future_dates.iloc[20], forecast['yhat'].iloc[20]), 
            xytext=(future_dates.iloc[20], forecast['yhat'].iloc[20] + 100),
            arrowprops=dict(facecolor='green', shrink=0.05))

# Add legend for clarity
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()
