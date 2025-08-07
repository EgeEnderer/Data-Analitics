import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from statsmodels.graphics.tukeyplot import results
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from pmdarima.datasets import load_lynx
from pmdarima.arima.utils import nsdiffs


lynx=load_lynx()

# Load data

df = pd.read_csv('Data.csv')

df.head()

df.info()

# Convert 'Days' to datetime format
df['Days'] = pd.to_datetime(df['Days'], format='%d/%m/%Y')
# Resample the data monthly and sum 'Sales' to get monthly total sales
Total_revenue = df.groupby(pd.Grouper(key='Days', freq='D'))['Total_revenue'].sum()

Total_revenue.head()

# Visualize Total Revenue

plt.figure(figsize=(12, 6))
plt.plot(Total_revenue, label='Total Revenue')
plt.title('Total Revenue Over Time')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()
plt.show()

# Time Series Decomposition
decomposition = seasonal_decompose(Total_revenue, model='additive')
fig = decomposition.plot()
fig.set_size_inches(14, 10)
plt.show()


# Check for stationarity using Augmented Dickey-Fuller test
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('Augmented Dickey-Fuller Test Results:')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    return result[1] < 0.05


is_stationary = check_stationarity(Total_revenue)
print(f"\nTime series is {'stationary' if is_stationary else 'non-stationary'}")

# If non-stationary, take first difference
if not is_stationary:
    sales_diff = Total_revenue.diff().dropna()
    print("\nChecking stationarity of differentiated series:")
    is_stationary_diff = check_stationarity(sales_diff)
else:
    sales_diff = Total_revenue

# Find and fit ARIMA model
auto_model = auto_arima(Total_revenue,
                        start_p=0, start_q=0,
                        test = 'adf',
                        max_p=3, max_q=3,
                        m=4,
                        seasonal=True,
                        d = None,
                        max_d=1,
                        D = None,
                        max_D=1,
                        trace=True,
                        error_action='ignore',
                        suppress_warnings=True,

                        stepwise=True)
auto_model.summary()

model = ARIMA(Total_revenue,
              order=auto_model.order,
              seasonal_order=auto_model.seasonal_order)
results = model.fit()


# Generate forecasts
forecast_periods = 90
forecast = results.get_forecast(steps=forecast_periods)
mean_forecast = forecast.predicted_mean

# Get confidence intervals
conf_int_95 = forecast.conf_int(alpha=0.05)
conf_int_80 = forecast.conf_int(alpha=0.20)
conf_int_70 = forecast.conf_int(alpha=0.30)

# Create visualization
plt.figure(figsize=(15, 7))

# Plot historical data and forecast
plt.plot(Total_revenue, label='Historical Data', color='blue')
plt.plot(mean_forecast, label='Forecast', color='red', linewidth=2)

# Plot confidence intervals
plt.fill_between(mean_forecast.index,
                 conf_int_95.iloc[:, 0],
                 conf_int_95.iloc[:, 1],
                 color='red', alpha=0.1,
                 label='95% CI')

plt.fill_between(mean_forecast.index,
                 conf_int_80.iloc[:, 0],
                 conf_int_80.iloc[:, 1],
                 color='red', alpha=0.2,
                 label='80% CI')

plt.fill_between(mean_forecast.index,
                 conf_int_70.iloc[:, 0],
                 conf_int_70.iloc[:, 1],
                 color='red', alpha=0.3,
                 label='70% CI')

plt.title('Sales Forecast with ARIMA and Multiple Confidence Intervals')
plt.xlabel('Date')
plt.ylabel('Sales ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# Get final values
last_forecast = mean_forecast.iloc[-1]
ranges_95_lower = conf_int_95.iloc[-1, 0]
ranges_95_upper = conf_int_95.iloc[-1, 1]
ranges_80_lower = conf_int_80.iloc[-1, 0]
ranges_80_upper = conf_int_80.iloc[-1, 1]
ranges_70_lower = conf_int_70.iloc[-1, 0]
ranges_70_upper = conf_int_70.iloc[-1, 1]

# Create info text
info_text = f'Final Forecast: ${last_forecast:,.0f}\n\n' \
            f'95% CI: ${ranges_95_lower:,.0f} to ${ranges_95_upper:,.0f}\n' \
            f'80% CI: ${ranges_80_lower:,.0f} to ${ranges_80_upper:,.0f}\n' \
            f'70% CI: ${ranges_70_lower:,.0f} to ${ranges_70_upper:,.0f}'

plt.text(0.02, 0.98, info_text,
         transform=plt.gca().transAxes,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.show()

# Calculate and display metrics
print("\nModel Performance Metrics:")
mse = mean_squared_error(Total_revenue, results.fittedvalues)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Total_revenue, results.fittedvalues)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Print forecast information
print("\nForecast Ranges for Final Period:")
print(f"Point Forecast: ${last_forecast:,.2f}")
print("\nConfidence Intervals:")
print(f"95% CI: ${ranges_95_lower:,.2f} to ${ranges_95_upper:,.2f}")
print(f"80% CI: ${ranges_80_lower:,.2f} to ${ranges_80_upper:,.2f}")
print(f"70% CI: ${ranges_70_lower:,.2f} to ${ranges_70_upper:,.2f}")

# Calculate and print interval widths
print("\nInterval Widths as Percentage of Forecast:")
print(f"95% CI: ±{((ranges_95_upper - ranges_95_lower) / 2 / last_forecast * 100):,.1f}%")
print(f"80% CI: ±{((ranges_80_upper - ranges_80_lower) / 2 / last_forecast * 100):,.1f}%")
print(f"70% CI: ±{((ranges_70_upper - ranges_70_lower) / 2 / last_forecast * 100):,.1f}%")

# Create forecast dataframe
forecast_df = pd.DataFrame({
    'Forecast': mean_forecast,
    '95% Lower': conf_int_95.iloc[:, 0],
    '95% Upper': conf_int_95.iloc[:, 1],
    '80% Lower': conf_int_80.iloc[:, 0],
    '80% Upper': conf_int_80.iloc[:, 1],
    '70% Lower': conf_int_70.iloc[:, 0],
    '70% Upper': conf_int_70.iloc[:, 1]
})
fc = forecast_df['Forecast']
fivelower = forecast_df['95% Lower']
fiveupper = forecast_df['95% Upper']
twentylower = forecast_df['80% Lower']
twentyupper = forecast_df['80% Upper']
thirtylower = forecast_df['70% Lower']
thirtyupper = forecast_df['70% Upper']

print("\nDetailed Forecast with Confidence Intervals:")
print(forecast_df.to_string())
np.savetxt('forecast_arima.csv', [p for p in zip(fc,
                                                 fivelower,
                                                 fiveupper,
                                                 twentylower,
                                                 twentyupper,
                                                 thirtylower,
                                                 thirtyupper)], delimiter=',', fmt='%s')
# -----------------

# Hot-Winters Model

# Try multiple Holt-Winters specifications
models = []
specifications = [
    {
        'name': 'Fixed Parameters',
        'model': ExponentialSmoothing(
            Total_revenue,
            seasonal_periods=7,
            trend='add',
            seasonal='add',
            damped_trend=True
        ).fit(
            smoothing_level=0.2,
            smoothing_trend=0.1,
            smoothing_seasonal=0.1,
            damping_trend=0.98,
            optimized=False
        )
    },
    {
        'name': 'Multiplicative Seasonal',
        'model': ExponentialSmoothing(
            Total_revenue,
            seasonal_periods=7,
            trend='add',
            seasonal='mul',
            damped_trend=True
        ).fit(
            smoothing_level=0.2,
            smoothing_trend=0.1,
            smoothing_seasonal=0.1,
            damping_trend=0.98,
            optimized=False
        )
    },
    {
        'name': 'Multiplicative Trend',
        'model': ExponentialSmoothing(
            Total_revenue,
            seasonal_periods=7,
            trend='mul',
            seasonal='add',
            damped_trend=True
        ).fit(
            smoothing_level=0.2,
            smoothing_trend=0.1,
            smoothing_seasonal=0.1,
            damping_trend=0.98,
            optimized=False
        )
    }
]

# Evaluate each model
results = []
for spec in specifications:
    model = spec['model']
    name = spec['name']

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(Total_revenue, model.fittedvalues))
    mae = mean_absolute_error(Total_revenue, model.fittedvalues)

    results.append({
        'name': name,
        'rmse': rmse,
        'mae': mae,
        'model': model
    })

# Find best model
best_model = min(results, key=lambda x: x['rmse'])
hw_model = best_model['model']
hw_forecast = hw_model.forecast(forecast_periods)

# Visualization
plt.figure(figsize=(15, 12))

# First subplot: All models
plt.subplot(3, 1, 1)
plt.plot(Total_revenue.index, Total_revenue, label='Historical Data', color='blue')
for result in results:
    plt.plot(result['model'].fittedvalues.index,
             result['model'].fittedvalues,
             label=f"{result['name']} Fitted",
             alpha=0.5)
plt.title('Model Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# Second subplot: Best model forecast
plt.subplot(3, 1, 2)
plt.plot(Total_revenue.index, Total_revenue, label='Historical Data', color='blue')
plt.plot(hw_forecast.index, hw_forecast,
         label=f'Forecast ({best_model["name"]})',
         color='green', linestyle='--')
plt.title(f'Best Model Forecast: {best_model["name"]}')
plt.legend()
plt.grid(True, alpha=0.3)

# Third subplot: Residuals of best model
plt.subplot(3, 1, 3)
residuals = Total_revenue - hw_model.fittedvalues
plt.plot(Total_revenue.index, residuals, label='Residuals', color='gray')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('Best Model Residuals')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print Results
print("\nModel Comparison:")
for result in results:
    print(f"\n{result['name']}:")
    print(f"RMSE: ${result['rmse']:.2f}")
    print(f"MAE: ${result['mae']:.2f}")

print(f"\nBest Model: {best_model['name']}")
print(f"Best RMSE: ${best_model['rmse']:.2f}")

# Print Best Model Parameters
print("\nBest Model Parameters:")
print(f"- Smoothing level (α): {hw_model.params['smoothing_level']:.3f}")
print(f"- Trend smoothing (β): {hw_model.params['smoothing_trend']:.3f}")
print(f"- Seasonal smoothing (γ): {hw_model.params['smoothing_seasonal']:.3f}")
print(f"- Damping parameter (φ): {hw_model.params['damping_trend']:.3f}")

# Print Forecast
print("\nBest Model Results:")
print(hw_forecast.to_string())

# Save results
final_results = pd.DataFrame({
    'Actual': Total_revenue,
    'Fitted': hw_model.fittedvalues,
    'Residuals': residuals
})



np.savetxt('forecast_hw.csv', [p for p in zip(hw_forecast)], delimiter=',', fmt='%s')