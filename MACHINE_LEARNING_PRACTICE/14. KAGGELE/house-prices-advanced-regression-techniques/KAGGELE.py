#OBJECTIVE 
#TO predict the sale price of the house 
#IMPORT LIBRARY
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#***************************
#
#PART:1:HANDEL NULL
#
#**************************

#HERE BOTH TRAIN NAD TEST HAVE NULL VALUES 
#HOW TO HANDEL NULL VALUES 
#null values greater than 50 percent drop column 
#null value less than 1 percent i.e 10  replace with mode 
#null value  1 to 40 % replace with mean 
#drop the row which has null value  


#REPLACE WITH MEAN, MODE ,DROP 
#df['col']=df['col'].fillna(df['col'].mode()[0])
#df['col']=df['col'].fillna(df['col'].mean())
#df.drop(['col'],axis=1,inplace=True)
#df.dropna(inplace=True)


#WE NEED TO HNADLE IN BOTH CASES 
#TEST

#SEE HOW MANY NULLS ARE PRESENT  IN BOTH TEST AND TRAIN 
#print(df_train.isnull().sum())
test_df=pd.read_csv('test.csv')
#print(test_df.isnull().sum())
#MSZONING IN TEST HAVE 4 NULL VLUES 
#WE WILL HANDEL IT BY REPLACING WITH MODE 
test_df['MSZoning']=test_df['MSZoning'].fillna(test_df['MSZoning'].mode())
#LotFrontage have 227 null so we will replace with mean 
test_df['LotFrontage']=test_df['LotFrontage'].fillna(test_df['LotFrontage'].mean())
test_df.drop(['Alley'],axis=1,inplace=True)

#TRAIN 
train_df=pd.read_csv('train.csv')
print(train_df.shape)
#print(train_df.isnull().sum())

def replace_mod(df,l):
    for x in l:
        df[x]=df[x].fillna(df[x].mode()[0])

def replace_mean(df,l):
    for x in l:
        df[x]=df[x].fillna(df[x].mean())
def drop_column(df,l):
    for x in l:
        df.drop([x],axis=1,inplace=True)


mod_col_train=['BsmtCond','BsmtQual','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','MasVnrType','MasVnrArea','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical']
mean_col_train=['LotFrontage']
drop_col_train=['Alley','GarageYrBlt','PoolQC','Fence','MiscFeature','Id']
replace_mod(train_df,mod_col_train)
replace_mean(train_df,mean_col_train)
drop_column(train_df,drop_col_train)

test_df=pd.read_csv('test.csv')
mod_col_train=['MSZoning','Utilities','BsmtCond','BsmtQual','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','MasVnrType','MasVnrArea','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical']
mean_col_train=['LotFrontage']
drop_col_train=['Alley','GarageYrBlt','PoolQC','Fence','MiscFeature','Id']
replace_mod(test_df,mod_col_train)
replace_mean(test_df,mean_col_train)
drop_column(test_df,drop_col_train)
test_df.dropna(inplace=True)

#print(train_df.isnull().sum())
#print(test_df.isnull().sum())
print(train_df.shape)
print(test_df.shape)



#*********************************
#PART :2: HANDEL CATEGORICAL FEATURES 
#
#*********************************
#ISSUE FOR INTERVIEW :
#In test data set we have some categories which is not present in train  data so 
#train data after one hot we will have some column which will  not be present in test data so
#column number will not be same in training and test data 
#print(train_df.info())

#solution :
#STEP:1:combine train df test df and create a final df 
#handel all categorical value 
#split train and test based on number of rows they had earlier 
final_df=pd.concat([train_df,test_df],axis=0,sort='False')
print(final_df.shape)
#print(final_df.isnull().sum())
x=final_df.select_dtypes(['object'])
#Get all the column which has datatype=object 
categorical_column=x.columns.tolist()
print(categorical_column)
print(final_df['Heating'].value_counts())

#final_df=pd.get_dummies(final_df, drop_first=True)
#final_df =final_df.T.drop_duplicates().T
#print(final_df.shape)
#print(final_df.info())



def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final      

final_df=category_onehot_multcols(categorical_column)
final_df =final_df.loc[:,~final_df.columns.duplicated()]

print(final_df.shape)


#SPLIT INTO TRAIN AND TEST 

df_Train=final_df.iloc[:1460,:]
df_Test=final_df.iloc[1461:,:]

#Drop sales price from df_test
df_Test.drop(['SalePrice'],axis=1,inplace=True)

#SPLIT TRAIN PART INTO TRAIN AND TEST 
X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']
x=df_Train.drop(['SalePrice'],axis=1)
y=df_Train['SalePrice']
#Prediciton and selecting the Algorithm
import xgboost
regressor=xgboost.XGBRegressor()

n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]
base_score=[0.25,0.5,0.75,1]

# USE RANDOMSERACH CV 
# DEFINE F=GRID TO SEARCH 
# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }
# Set up the random search with 4-fold cross validation
from sklearn.model_selection import RandomizedSearchCV
random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=2, n_iter=10,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 3, 
            return_train_score = True,
            random_state=42)

#random_cv.fit(X_train,y_train)
#print(random_cv.best_estimator_)

regressor=xgboost.XGBRegressor(base_score=0.75, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.05, max_delta_step=0,
             max_depth=3, min_child_weight=4, missing=None, n_estimators=1100,
             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)


from sklearn.model_selection import train_test_split
# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42,shuffle=True)
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))
