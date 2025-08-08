import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv("data.csv")

df.set_index(pd.to_datetime(df['Days'] , format='%d/%m/%Y'), inplace=True)

dfi = df.interpolate(method='spline', order=5)

plt.plot(dfi['ARPU_M1'])
plt.plot(dfi['ARPU_M2'])
plt.plot(dfi['ARPU_M3'])
plt.plot(dfi['ARPU_M4'])
plt.plot(dfi['ARPU_M5'])
plt.plot(dfi['ARPU_M6'])
plt.plot(dfi['ARPU_M7'])
plt.show()

print(dfi.to_string())


np.savetxt('interpolation.csv',(dfi['ARPU_M1'],
                                        dfi['ARPU_M2'],
                                        dfi['ARPU_M3'],
                                        dfi['ARPU_M4'],
                                        dfi['ARPU_M5'],
                                        dfi['ARPU_M6'],
                                        dfi['ARPU_M7']), delimiter = ',')
