import pandas as pd
from pandas import datetime
import matplotlib.pyplot as plt
#sales = pd.read_csv('sales-cars.csv')
#print(sales.info())
#as we can see month as datatype  object we need it as datetime64 format 
def parser(x):
    return datetime.strptime(x,'%Y-%m')
sales = pd.read_csv('sales-cars.csv',index_col=0,parse_dates=[0],date_parser=parser) 
#print(sales.head())  

 
#Now we need to check stationary 
#stationary means mean ,variance and covariance is constant over period of time 
#WAY:1.1
#print(sales.plot())  

#In graph also we can see graph is increasing so for any two given time frames mean will be different  
#WAY:1
X = sales.values
split = int(len(X) / 2)
print(split)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))
#here both means are not similar so data is not stationary 
#WAY:2
#ADFT (Augmented Dickey-Fuller test)
#Null Hypothesis (H0): If failed to be rejected, it suggests the time series has a unit root, meaning it is non-stationary. It has some time dependent structure.
#Alternate Hypothesis (H1): The null hypothesis is rejected; it suggests the time series does not have a unit root, meaning it is stationary. It does not have time-dependent structure.

#interpreting ADFT Result
#p-value > 0.05: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.
#p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.
#ADFT stats < 5% critical value 
from statsmodels.tsa.stattools import adfuller
X = sales.iloc[:,0].values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
#conclusion it is not stationary so ARIMA MODEL can not be applied 
#WAY:3
#ACF(auto correlation function)
from statsmodels.graphics.tsaplots import plot_acf
#print(plot_acf(sales))

#As graph declining slowly then it is not stationary 
#If graph decayssuddenly then it is stationary 

#MAKE NON STATIONARY TO STATIONARY 
#To make non stationary data to stationary we need to take a diff
sales_diff=sales.diff(periods=1)  
#Take everything except null values  
sales_diff=sales_diff[1:] 

#AGAIN CHECK for stationarity using acf graph 
#print(plot_acf(sales_diff))
#now we can see it is suddenly decreasing then it is stationary 

#MODELLING :
###############################
#AR:(Auto regression )
###############################
#An autoregression model is a linear regression model that uses lagged variables as input variables.
#We could calculate the linear regression model manually using the LinearRegession class in 
#scikit-learn and manually specify the lag input variables to use.
#Alternately, the statsmodels library provides an autoregression model that automatically selects an 
#appropriate lag value using statistical tests and trains a linear regression model
X=sales.values
#print(X.size())
train=X[1:len(X)-9] #27 data points for train 
test=X[len(X)-9:]  #9 data points for test
predictions=[]
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error

model_ar=AR(train)
model_ar_fit=model_ar.fit()
predictions = model_ar_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
print("predicted data vs actual data ")
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))

#ERROR and plotting 
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot results
#plt.plot(test)
#plt.plot(predictions, color='red')
#plt.show()

##############################################
#ARIMA MODEL
#############################################
from statsmodels.tsa.arima_model import ARIMA 
#In case of AR it automatically finds the periods 
#incase of ARIMA we need to find proper value of 
#p:periods (Number of previous unit considered for forecast)
#d:Integration order :how many time differentiation has been done 
#q :number of periods in moving avarage 
model_arima=ARIMA(train,order=(3,1,1))
model_arima_fit = model_arima.fit()

predictions=model_arima_fit.forecast(steps=9)[0]
print("predicted data vs actual data ")
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
print("AIC ERROR",model_arima_fit.aic)
#lower the AIC better is the model

#plt.plot(test)
#plt.plot(predictions, color='red')
#plt.show()

# How to find best combination of p d and q 
#WAY:1
import itertools
p=d=q=range(0,5)
pdq=list(itertools.product(p,d,q))
#print(pdq)
import warnings 
warnings.filterwarnings('ignore')
for param in pdq:
    try:
        model_arima=ARIMA(train,order=param)
        model_arima_fit = model_arima.fit()
        print(param,model_arima_fit.aic)
    except:
        continue


#WAY:2
#auto_arima() uses a stepwise approach to search multiple combinations of p,d,q parameters and chooses the best model that has the least AIC.
#This is not running because pmdarima not able to install 

#from pyramid.arima import auto_arima
#stepwise_model = auto_arima(train, start_p=1, start_q=1,
#                           max_p=3, max_q=3, m=12,
#                           start_P=0, seasonal=True,
#                           d=1, D=1, trace=True,
#                           error_action='ignore',  
#                           suppress_warnings=True, 
#                           stepwise=True)
#print(stepwise_model.aic())
#        
#suppose after analysis we found  that PDQ 4,1,0 has the best combination with low AIC value
print("Final prediction")
model_arima=ARIMA(train,order=(4,1,0))
model_arima_fit = model_arima.fit()
print(X[0])
from pandas import datetime
start_index = datetime(1990, 12, 25)
end_index = datetime(1990, 12, 26)
#forecast = model_arima_fit.predict(start=start_index, end=end_index,dynamic=False)
#print(sales.head())
        