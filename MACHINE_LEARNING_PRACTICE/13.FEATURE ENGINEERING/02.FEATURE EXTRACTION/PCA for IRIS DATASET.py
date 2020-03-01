#https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
import pandas as pd 
from  sklearn import  datasets

#LOADING IRIS DATA FROM SKLEARN 
iris=datasets.load_iris()
x= iris.data
y= iris.target

#LOADING IRIS DATA FROM CSV FILE 
df=pd.read_csv('iris.csv')

df['species_label'], _=pd.factorize(df['species'])
#factorize method is used to get numerical values for categorical values 
y = df['species_label']
x = df[['petal_length','petal_width','sepal_length','sepal_width']]
print("Shape of data frame",x.shape)
#Standardization 
#Standardization is all about scaling your data in such a way that all the variables and their values lie within a similar range
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
# Two ways to use PCA a. tell how many principal component we want  b .How much varience we want to preserve in percentage
pca = PCA(n_components=2,whiten=True)
#pca=PCA(0.95)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
print("Shape of principal DF",principalDf.shape)

print(pca.explained_variance_ratio_)
#The explained variance tells you how much information (variance) can be attributed to each of the principal components
#This is important as while you can convert 4 dimensional space to 2 dimensional space, you lose some of the variance
#(information) when you do this. By using the attribute explained_variance_ratio_, you can see that the first principal
# component contains 72.77% of the variance and the second principal component contains 23.03% of the variance. Together,
# the two components contain 95.80% of the information.

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
logreg = LogisticRegression(multi_class = 'multinomial',solver='lbfgs')
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
print("Accuracy of logistic regression classifier on test set:Accuracy score",metrics.accuracy_score(y_test, y_pred))

x_train, x_test, y_train, y_test = train_test_split(principalDf, y, test_size=0.3, random_state=0)
logreg = LogisticRegression(multi_class = 'multinomial',solver='lbfgs')
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
print("Accuracy of logistic regression classifier on test set:Accuracy score",metrics.accuracy_score(y_test, y_pred))

