import pandas as pd
import numpy as np
df=pd.read_csv("iris.csv")

y = df['species']
x = df[['petal_length','petal_width','sepal_length','sepal_width']]
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
logreg = LogisticRegression(multi_class = 'multinomial',solver='lbfgs')
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
print("Accuracy of logistic regression classifier on test set:Accuracy score")
y_pred = logreg.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))
#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix")
print(confusion_matrix)

#MAKE INDIVIDUAL PREDICTION 
res=logreg.predict([[2,5.7,2.2,2.3]])
#PER CLASS PROBABILITY
perclass=logreg.predict_proba([[2,5.7,2.2,2.3]])
probabilities=perclass.flatten()
types=y.unique()
for i in range( 0,types.size):
    print(str(types[i])+" has probability "+str(round(probabilities[i],2)))
print("So selecting result ",res)    

print(metrics.accuracy_score(y_test, y_pred))     

