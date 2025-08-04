import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pmdarima as pm
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA


df = pd.read_csv("forecastdata.csv")
df.info()
df.plot()

df_train = df.values[:85]
df_test = df.values[85:]

acf_original = plot_acf(df_train)
pacf_original = plot_pacf(df_train)

adf_test = adfuller(df_train)
print(f'p-value: {adf_test[1]}')

df_train_diff = df_train.diff().dropna()
df_train_diff.plot()

adf_test = adfuller(df_train_diff)
print(f'p-value: {adf_test[1]}')

acf_original = plot_acf(df_train_diff)
pacf_original = plot_pacf(df_train_diff)

plt.show()

model=ARIMA(df_train, order=(2,1,0))
result=model.fit()
print(result.summary())

residuals = result.resid[1:]
fig, ax = plt.subplots(1,2)
residuals.plot(title='Residuals', ax=ax[0])
residuals.plot(title = 'Density', kind = 'kde', ax=ax[1])
plt.show()

acf_res = plot_acf(residuals)
pacf_res = plot_pacf(residuals)

forecast_test = result.forecast(len(df_test))
df['forecast_manual'] = [None]*len(df_train) + list(forecast_test)
df.plot()
plt.show()

auto_arima = pm.auto_arima(df_train, stepwise=False)
auto_arima
print(auto_arima.summary())

forecast_test_auto = auto_arima.predict(n_periods=len(df_test))
df['forecast_auto'] = [None]*len(df_train) + list(forecast_test_auto)

df.plot()
plt.show()