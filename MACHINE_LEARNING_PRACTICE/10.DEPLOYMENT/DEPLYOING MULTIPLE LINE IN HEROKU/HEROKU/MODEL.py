import pandas as pd
import pickle

dataset = pd.read_csv(r'Advertising.csv')
x = dataset.iloc[:, [1,2,3]].values
y = dataset.iloc[:, 4].values


#SPLITING TRAINING AND TEST DATA 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
#LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[100, 200, 300]]))
