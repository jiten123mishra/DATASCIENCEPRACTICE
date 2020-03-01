import pandas as pd
import numpy as np
df = pd.read_csv(r'Advertising_outlier.csv')
#DETECT AND REMOVE OUTLIER FOR ALL COLUMN USING Z SCORE 
print(df.shape)
from scipy import stats
df_with_out_outlier_usingz=df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
print(df_with_out_outlier_usingz.shape)
#Z 1 = 68
#z 2 = 95
#z 3 =99.7
#DETECT AND REMOVE OUTLIER FOR ALL COLUMN USING IQR SCORE 
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df_with_out_outlier_using_IQR = df[((df >= (Q1 - 1.5 * IQR))& (df <= (Q3 + 1.5 * IQR))).all(axis=1)]
print(df.shape)
print(df_with_out_outlier_using_IQR.shape)
#print(df)
#print (((df >= (Q1 - 1.5 * IQR)) & (df <= (Q3 + 1.5 * IQR))))