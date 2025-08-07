
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




acf_original = plot_acf(df)
pacf_original = plot_pacf(df)

adf_test = adfuller(df)
print(f'p-value: {adf_test[1]}')

df_diff = df.diff().dropna()
df_diff.plot()

adf_test = adfuller(df_diff)
print(f'p-value: {adf_test[1]}')

acf_original_diff = plot_acf(df_diff)
pacf_original_diff = plot_pacf(df_diff)

plt.show()
