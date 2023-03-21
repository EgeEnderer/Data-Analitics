import pandas as pd
import matplotlib.style
import matplotlib as mp
import numpy as nmp
mp.style.use('_mpl-gallery')
db = pd.read_csv('datasd.csv')
#print(db)
#print(db.isnull().sum())
db = db.dropna()
