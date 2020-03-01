#All methods to deal with imbalanced class
#https://elitedatascience.com/imbalanced-classes
#problem with imbalanced dataset :suppose we have 100 data point 95 data are class 1  and 5 are of class 2 
#After training if we test our data on 40 data points where 2 are of class 2 and 38 are of class1 . even our model not able to detect any class 2 result it will still show
#90% plus result 
# Load libraries
import numpy as np
import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.utils import resample
# Load iris data
df = pd.read_csv("iris.csv")
df =df.iloc[45:65,:]

# print("Before downsampling")
# print(df.species.value_counts())
# # Separate majority and minority classes
# index=np.where(df.species=='Iris-versicolor')
# df_majority = df.iloc[index]
# index=np.where(df.species=='Iris-setosa')
# df_minority = df.iloc[index]

# # Downsample majority class
# df_majority_downsampled = resample(df_majority, 
                                 # replace=False,    # sample without replacement
                                 # n_samples=5,     # to match minority class
                                 # random_state=123) # reproducible results
 
# # Combine minority class with downsampled majority class
# df_downsampled = pd.concat([df_majority_downsampled, df_minority])

# print("After downsampling")
# print(df_downsampled.species.value_counts())
# # Downsample majority class
# df_minority_upsampled = resample(df_minority, 
                                 # replace=True,    # sample without replacement
                                 # n_samples=15,     # to match minority class
                                 # random_state=123) # reproducible results
 
# # Combine minority class with downsampled majority class
# df_upsampled = pd.concat([df_minority_upsampled, df_majority])

# print("After upsampling")
# print(df_upsampled.species.value_counts())

# print(df['species'].value_counts())
#
##Divide X (input) and Y (output)
#X = df.loc[:, df.columns != 'spicies']
#y = df.loc[:, df.columns == 'species']
##Dividing both x and y to training and test data 
##from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#


##Divide X (input) and Y (output)
X = df.loc[:, df.columns != 'spicies']
y = df.loc[:, df.columns == 'species']
print("Before ")
print(y['species'].value_counts())
#over sampling 

from imblearn.over_sampling import SMOTE
smote = SMOTE(ratio='minority')
X_sm, y_sm = smote.fit_sample(X, y)


#under sampling 

from imblearn.under_sampling import TomekLinks
tl = TomekLinks(return_indices=True, ratio='majority')
X_tl, y_tl, id_tl = tl.fit_sample(X, y)



#OVERSAMPLING USING SMOTE 
# As we have seen ratio of no-subscription to subscription instances is 89:11 we will do oversampling using SMOTE 
# from imblearn.over_sampling import SMOTE
# os = SMOTE(random_state=0)
# os_data_X,os_data_y=os.fit_sample(X_train, y_train)
# os_data_X = pd.DataFrame(data=os_data_X,columns=X_train.columns )
# os_data_y= pd.DataFrame(data=os_data_y,columns=y_train.columns)
# print(os_data_y['y'].value_counts())