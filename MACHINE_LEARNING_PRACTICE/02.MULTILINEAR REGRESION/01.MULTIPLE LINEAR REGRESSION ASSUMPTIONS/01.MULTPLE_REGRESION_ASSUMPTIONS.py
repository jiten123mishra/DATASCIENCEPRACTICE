import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'Advertising.csv')
#Display head of data set 
#print(dataset.head())

#DATA CLEANING 
# AS Unnamed column is just a  column is not contributing 
#column 1 TV 2. Radio 3. NewsPaper
x = dataset.iloc[:, [1,2,3]].values
y = dataset.iloc[:, 4].values

#ASSUMPTION OF LINEAR REGRESSION 
#1.LINEAR RELATION BETWEEN FEATURE AND TARGET
#2.LITTLE OR NO MULTI COLLINEARITY BETWEEN FEATURES OR INDEPEDENT VARIABLE 
#3.HOMOSCADACITY ASSUMPTION 
#4.NORMAL DISTRIBUTION OF ERROR TERM 
#5.NO CORRELATION BETWEEN RESIDUALS 

#1.LINEAR RELATION BETWEEN FEATURE AND TARGET
#VISUALISE DATA DEPENDENT VARIABLE WITH EVERY  OTHER INDEPENDENT VARIABLE 
import seaborn as sns
sns.pairplot(dataset, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=7, aspect=0.7)

#2.LITTLE OR NO MULTI COLLINEARITY BETWEEN FEATURES OR INDEPEDENT VARIABLE 
# CHECK FOR MULTICOLLINEARITTY 
#WAY-1 
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
x = dataset.iloc[:, [1,2,3]]
vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif["features"] = x.columns
print(vif)
#INTERPRETATION :
#two variables will be highly correlated, and each will have the same high VIF.
#1 means values are not corelated 
#2-5 means variables are moderately corelated 
#>5 Highly corelated we can not use all  variables together for prediction  which have same high VIF 
#WAY-2
#CORRELATION MATRIX 
import seaborn as sns
x = dataset.iloc[:, [1,2,3]]

plt.figure(figsize = (16,5))
sns.heatmap(x.corr(),annot=True,linewidths=.5)
#INTERPRETATION :
#Each box showing correlation coefficients : correlation coeefiecinets varies from -1 to 1 represents realation between 
#two variable in both strength and magnitude -1 and 1  are extreme 
#SOLUTION FOR MULTICOLLINEARITY 
#DROP ONE FEATURE WHICH HAS LESS R2 score or combined both variable  and create a new one  ex :math mark and science mark 

#3.HOMOSCADACITY ASSUMPTION 
#Homoscadacity is a situation where residual/error is the same across all values of the independent variables
#plot a graph x axis y hat y axis yhat-y
#Ex of heteroscadacity :Predicting spending on luxury item based on income 
#poor people low income low spending residual less 
#rich people high income some spending high some spending low amount/magnitude/size  of residual is more 
#This situation represents heteroscedasticity because the size of the error varies across values of the independent variable
#PROBLEM :
#Heteroscadacity leads to biased standard error and biased standard errors lead to incorrect conclusions about the significance of the regression coefficients.
#SOLUTION 
#box-cox transformation 
#HOW TO CHECK 
#A scatter plot of residual values vs predicted values is a goodway to check for homoscedasticity.
#INTERPRETATION 
#There should be no clear pattern in the distribution and if there is a specific pattern,the data is heteroscedastic.


from sklearn.linear_model import LinearRegression 
regressor=LinearRegression()


x = dataset.iloc[:, [1,2,3]].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
regressor.fit(x_train,y_train)

#making prediction 
y_pred=regressor.predict(x_test)

#visualizing homoscedacity 
#visualising training set with optimum line 
plt.scatter(y_pred,y_pred-y_test,color='red')        
plt.title('y pred vs diff')
plt.xlabel('y pred')
plt.ylabel('diff')
plt.show()

#4.NORMAL DISTRIBUTION OF ERROR TERM 
#The error(residuals) follow a normal distribution.
#If we are taking  large size sample for multiple times then there is no need of this assumption ,central limit theorom 
#can be verified in 2 ways 
#way-1 
#Residual distribution diagram from assumption 3 , we can see mean close to 0
#way-2 QQ Plot 
import statsmodels.api as sm
mod_fit =sm.OLS(y_train,x_train).fit()
res=mod_fit.resid
fig=sm.qqplot(res,fit=True,line='45')
plt.show()

#5.NO CORRELATION BETWEEN RESIDUALS 
#Autocorrelation occurs when the residual errors are dependent on each other.
#The presence of correlation in error terms drastically reduces model’s accuracy.This usually occurs in time series models where the next instant is dependent on previous instant.
# Autocorrelation can be tested with the help of Durbin-Watson test can be obtained from regressor OLS summary
#INTERPRETATION :
#test statistic varies from 0 to 4 
# value 2 no serial correlation  value closer to 0 strong positive correlation value closer to 4 strong negative serial correlation 
#MODEL
x = dataset.iloc[:, [1,2,3]].values
y = dataset.iloc[:, 4].values
import statsmodels.api as sm1
regressor_OLS = sm1.OLS(y,x).fit()
print(regressor_OLS.summary())

#In this summary we are getting Durbin-Watson:                   2.044 so no auto correlation present 

#INTERPRETING OLS SUMMMARY 
#REFER NOTE BOOK

#FINDING OPTIMUM INDEPENDENT VARIABLE

#SAVE MODEL

#PREDICT MODEL FROM SAVED FILE 



