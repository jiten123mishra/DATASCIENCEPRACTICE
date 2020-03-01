#How to deal with outlier should we drop the rows with outliers or transform it 
#https://www.theanalysisfactor.com/outliers-to-drop-or-not-to-drop/

#WAYS TO DETECT AND REMOVE OUTLIER IN MULTIPLE REGRESSION 
import pandas as pd 
import numpy  as np
#Single column 
l=[186, 176, 158, 180, 186, 168, 168, 164, 178, 170, 189, 195, 172,
     187, 180, 186, 185, 168, 179, 178, 183, 179, 170, 175, 186, 159,
     161, 178, 175, 185, 175, 162, 173, 172, 177, 175, 172, 177, 180,22,12,8]

#PLOT NORMAL DISTRIBUTION  TO OBSERVE SKEW NESS OF DATA 
import scipy.stats as stats
import matplotlib.pyplot as plt
l.sort()
hmean = np.mean(l)
hstd = np.std(l)
pdf = stats.norm.pdf(l, hmean, hstd)
#print(plt.plot(l, pdf))

#DETECT OULIER USING Z STATS
outliers=[]
data_after_removing_outliers=[]
def detect_outlier(data_1):
    
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
        if np.abs(z_score) < threshold:
            data_after_removing_outliers.append(y)        
    return outliers,data_after_removing_outliers
outliers,nooutliers=detect_outlier(l)
#print(outliers)
#print(nooutliers)

#FOR A DATA FRAME :
# Importing the dataset
dataset = pd.read_csv(r'C:\JITENDRA\SOFT\TEST\selflearn-master (2)\selflearn-master\02.MULTILINEAR REGRESION\MULTILINEAR_FOR_ADV_DATA\05.DETECT AND REMOVE OUTLIER\Advertising_outlier.csv')
#Display head of data set 
#print std deviation and mean of a column 
df_mean=dataset['TV'].mean()
df_std=dataset['TV'].std()
print(df_mean)
print(df_std)
#plot normal distribution of a column 
dataset['TV'].plot.kde()
#DETECT AND REMOVE OUTLIER FOR ALL COLUMN
print(dataset.shape)
from scipy import stats
df_with_out_outlier=dataset[(np.abs(stats.zscore(dataset)) < 3).all(axis=1)]
print(df_with_out_outlier.shape)
print(df_with_out_outlier)
#Here df_without_utlier will contain 18 record but if we calculate df with outlier we will get 0 record 
#as outlier is present for TV column not for all .To see outlier row we can use below method 

#FOR DATAFRAME FOR A SINGLE COLUMN 
df=dataset
df1=df[((df.TV - df.TV.mean()) / df.TV.std()).abs() < 3]
print(df1)
df2=df[((df.TV - df.TV.mean()) / df.TV.std()).abs() > 3]
print(df2)
#HERE we are getting df1 as without outlier dataframe and df2 as with outlier data frame 


#DETECTING OUTLIERS USING BOX PLOT 
#outliers can also be detected visually using box plot 
#print(plt.boxplot(df.Sales))#no outliers
#print(plt.boxplot(df.TV))#outliers 

#DETECTING OUTLIERS USING IQR 

def subset_by_iqr(df, column):
    # Calculate Q1, Q2 and IQR
    q1 = df[column].quantile(0.25)                 
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    filter = (df[column] >= q1 - 1.5*iqr) & (df[column] <= q3 + 1.5*iqr)
    return df.loc[filter]  
df3 = subset_by_iqr(df, 'TV')
#After removing outliers 
print(df3)