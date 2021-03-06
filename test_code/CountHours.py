import pandas as pd
import datetime
import numpy as np
import sys

args = sys.argv

if len(sys.argv) < 2:
    fname = 'cage1'
else:
    fname = args[1]

df = pd.read_csv(fname+'.txt', header=None, dtype={0:str}, parse_dates=[0], index_col=[0])
df['count'] = 1

df_tmp = df.resample('H').sum()
print(df_tmp)
df_tmp.to_csv(fname+'_H.csv')

df_day = df.resample('D').sum()
print(df_day)
df_day.to_csv(fname+'_D.csv')

df_min = df.resample('1Min').sum()
print(df_min)
df_min.to_csv(fname+'_Min.csv')
                �+��c)0C}��e����i�(�<��W�8c"�d�hqx1��馬f_�g��{Qx����dL2�Cmgp�