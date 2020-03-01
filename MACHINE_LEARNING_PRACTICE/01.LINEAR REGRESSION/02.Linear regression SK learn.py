import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv(r"SALARY_DATA.csv")
import pickle 

#Getting X and Y 
# here x is selecting all rows , all columns except last one 
# here y is selecting all rows and 1st index column 
x = df.iloc[:, :-1].values
y = df.iloc[:, 1].values


#SPLITTING DATA INTO TRAINING AND TEST
#1/3rd data is test data and 2/3 data is training data 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#IMPORT LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#pass data to algorithm to adjust value of m and c 
#training phase 
regressor.fit(x_train,y_train)

#making prediction 
y_pred=regressor.predict(x_test)

#visualising training set with optimum line 
plt.scatter(x_train,y_train,color='red')
#plottig x train and yprediction from training data
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

#visualising test set  with optimum line 
plt.scatter(x_test,y_test,color='red')
#plottig x train and yprediction from training data
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()

#visualizing homoscedacity 
#visualising training set with optimum line 
plt.scatter(y_pred,y_pred-y_test,color='red')        
plt.title('y pred vs diff')
plt.xlabel('y pred')
plt.ylabel('diff')
plt.show()


#Plotting a bar graph between y prediction and y test 
df1= pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)
df1.plot(kind='bar',figsize=(16,10))

#check accuracy of model
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))

#getting root mean square error and coefficients 
print("The intercept is:", regressor.intercept_)
print("The slope is: ", regressor.coef_)

#Saving linear regression training model  and use it 
filename = 'finalized_model.sav'
pickle.dump(regressor, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)
print("result from model sav file is ",result)

#prdicting output from pickle file 
print("Salary for experience of 10 years is  ",loaded_model.predict([[10]]))

from sklearn.externals import joblib
filename = 'finalized_model1.sav'
joblib.dump(regressor, filename)
loaded_model = joblib.load(filename)
result = loaded_model.score(x_test, y_test)
print("result from model joblibfile is ",result)