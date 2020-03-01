import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np
#DATA About a bank with customer and calls detail with the prediction , the person  has opened term deposit or not as prediction 
df = pd.read_csv("BANKDATA.csv")
#data description::
#DATA FRAME EXPLORATION 
print(df.shape)
#STEP:1 CHECK FOR NULL VALUES 
#print(df.isnull().sum())
#STEP:2 HANDEL CATEGORICAL DATA 
df=pd.get_dummies(df,drop_first=True)
print(df.shape)
#STEP:3
#CHECK FOR OUTLIERS 
#Q1 = df.quantile(0.25)
#Q3 = df.quantile(0.75)
#IQR = Q3 - Q1
#df = df[((df >= (Q1 - 1.5 * IQR))& (df <= (Q3 + 1.5 * IQR))).all(axis=1)]
#print(df.shape)

#STEP:4:
#EXTRACT RELEVANTFEATURES  USING CORRELATION 
# #TASK:1:
#Create a new data frame by considering the columns which are highly correlated with target or label 
cor = df.corr()
cor_target = abs(cor["y"])
relevant_features = cor_target[cor_target>0.2]
df_relevant=df[relevant_features.index]
print(df_relevant.shape)
#TASK2:
#from the df which which is highly correlated with target drop those column which are highly correlated with each other 
#drop target column
# Create correlation matrix
corr_matrix =df_relevant.corr().abs()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
df_relevant.drop(df[to_drop], axis=1)
print(df_relevant.shape)
df=df_relevant

#STEP:5
#CHECK FOR OVERSAMPLING /UNDERSAMPLING 
print(df['y'].value_counts())

#DO OVERSAMPLING USING SMOTE 

#Divide X (input) and Y (output)
X = df.loc[:, df.columns != 'y']
y = df.loc[:, df.columns == 'y']
#Dividing both x and y to training and test data 
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#OVERSAMPLING USING SMOTE 
#As we have seen ratio of no-subscription to subscription instances is 89:11 we will do oversampling using SMOTE 
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=X_train.columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=y_train.columns)
print(os_data_y['y'].value_counts())
x_train=os_data_X 
y_train=os_data_y

#perform logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='liblinear')
logreg.fit(x_train, y_train) 
y_pred = logreg.predict(x_test)

#ACCURACY
print("Accuracy of logistic regression classifier on test set")
y_pred = logreg.predict(x_test)
from sklearn import metrics
print("from metrics.accuracy_score  ",metrics.accuracy_score(y_test, y_pred))
#2ndway
print("from model.score ",logreg.score(x_test, y_test))

#ROC_ACCURACY
print(" ROC Accuracy score")
probs = logreg.predict_proba(x_test)
print(metrics.roc_auc_score(y_test, probs[:, 1]))

#CONFUSION METRICS
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix")
print(confusion_matrix)


#Evaluation of model using cross validation 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#cv = KFold(n_splits=10, random_state=42, shuffle=False)
scores = cross_val_score(logreg, X, y,cv=10 , scoring='accuracy')
print("10 fold cross validation result :")
print(scores)
print("mean of scores ",scores.mean())