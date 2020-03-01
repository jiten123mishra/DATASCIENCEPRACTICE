import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'Advertising.csv')
#Display head of data set 
print(dataset.head())

#DATA CLEANING 
# AS Unnamed column is just a  column is not contributing 
#column 1 TV 2. Radio 3. NewsPaper
x = dataset.iloc[:, [1,2,3]].values
y = dataset.iloc[:, 4].values

#UNDERSTAND DATA USING MATPLOTLIB 
dataset.plot.scatter(x = 'TV', y = 'Sales')
dataset.plot.scatter(x = 'Radio', y = 'Sales')
dataset.plot.scatter(x = 'Newspaper', y = 'Sales')
#SAMETHING CAN BE DONE BY SEABORN IN 1 LINE 
# visualize the relationship between the features and the response using scatterplots
import seaborn as sns
sns.pairplot(dataset, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=7, aspect=0.7)


#DETECTING MULTICOLINEARITY :
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(x, i) for i in range(x.shape[1])]
print(vif)
#https://www.ritchieng.com/machine-learning-evaluate-linear-regression-model/


#SPLITING TRAINING AND TEST DATA 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
#LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
## Predicting the Test set results
y_pred = regressor.predict(x_test)
##check accuracy of model
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))

print("The intercept is:", regressor.intercept_)
print("The slope is: ", regressor.coef_)

#OPTIMIZATION BY BACKWARD ELIMINATION 

len1=y.size
import statsmodels.formula.api as sm
#Adding one column with ones which will be intercept
x = np.append(arr = np.ones((len1, 1)).astype(int), values = x, axis = 1)
x = x.astype('float64') 
x_opt = x[:, [0,1,2,3]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
#print(regressor_OLS.summary())
#As newspaper p>t value is more we can eliminate it
x_opt = x[:, [0,1,2]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
#print(regressor_OLS.summary())

#LESSONS LEARNT :
#Newspaper is not contributing for sales  so we can remove it 

x_opt_train = x_train[:, [0,1]]

from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(x_opt_train, y_train)
y_pred = regressor.predict(x_test)
print(r2_score(y_test,y_pred))
#
import pickle
filename = 'finalized_model_adv.sav'
pickle.dump(regressor1, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
#esult = loaded_model.score(x_test, y_test)
#rint(result)

print("sales for TV , Radio  40,50",loaded_model.predict([[40,50]]))
