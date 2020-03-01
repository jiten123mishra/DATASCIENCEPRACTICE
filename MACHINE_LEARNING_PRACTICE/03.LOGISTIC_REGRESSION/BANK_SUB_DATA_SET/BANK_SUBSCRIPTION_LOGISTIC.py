import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np
#DATA About a bank with customer and calls detail with the prediction , the person  has opened term deposit or not as prediction 
df = pd.read_csv("BANKDATA.csv")
#data description::
#https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8


#DATA FRAME EXPLORATION 
print(df.shape)
print(list(df.columns))

#GET COLUMN WITH  NULL VALUES INFO 
print(df.info())
column_names_null_list=df.columns[df.isnull().any()].tolist()
NAS=df[column_names_null_list]
print("column names with null value with null count ",NAS.isna().sum())  
#If null value found treat with putting most mean , mode or remove 

#CHECK FOR UNDERSAMPLING 
print(df['y'].value_counts())
#calculate %of subscription 
count_no_sub=len(df[df['y']==0])
count_sub =len(df[df['y']==1])
pct_of_sub=(count_sub/(count_sub+count_no_sub))*100
pct_of_no_sub=100-pct_of_sub
print("total number of observation",len(df['y']))
print("percentage of subscription",pct_of_sub)
print("percentage of no  subscription",pct_of_no_sub)
#CONCLUSION:1
#Our classes are imbalanced, and the ratio of no-subscription to subscription instances is 89:11


#EFFECT OF INDIVIDUAL COLUMN ON SUBSCRIPTION 
print(df.groupby('y').mean())
#This will give us avarage of every column with respect to  person who have taken subscription and who has not taken subscription 

##VISUALIZATIONS
##This will give visualization of number of people subscribed or not with respect to each job category
#import matplotlib.pyplot as plt
#pd.crosstab(df.job,df.y).plot(kind='bar')
#plt.title('Purchase Frequency for Job Title')
#plt.xlabel('Job')
#plt.ylabel('Frequency of Purchase')
##This will give visualization of number of people subscribed or not with respect to each month category
#pd.crosstab(df.month,df.y).plot(kind='bar')
#plt.title('Purchase Frequency for Month')
#plt.xlabel('Month')
#plt.ylabel('Frequency of Purchase')


#DATA REPAIR 
data=df
#Let us group basic.4y, basic.9y and basic.6y together and call them basic for education column 
data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.6y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.4y', 'Basic', data['education'])
print(data.head())
#CREATE DUMMY VARIABLE 
#columns name before dummy variable 
print(list(df.columns))
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
data_final=data[to_keep]
df=data_final
#After dummy variable columns are
print("After dummy variable") 
print(list(df.columns))
#length
print("Total column after dummy variable",len(list(df.columns)))

#Divide X (input) and Y (output)
X = df.loc[:, df.columns != 'y']
y = df.loc[:, df.columns == 'y']
#Dividing both x and y to training and test data 
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#OVERSAMPLING USING SMOTE 
#As we have seen ratio of no-subscription to subscription instances is 89:11 we will do oversampling using SMOTE 
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=X_train.columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=y_train.columns)

# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of subscription in oversampled data",len(os_data_y[os_data_y['y']==1]))
#So we got a perfectly oversampled data with no of subscription = no of no subscription in over sampled data 
#NOTE:we over sample only training data 

#RECURSIVE FEATURE ELIMINATION
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='liblinear')
rfe = RFE(estimator=logreg, step=1,verbose=1,n_features_to_select=None)
#make verbose=0 if you dont want to see interactive logs
#Details 
#https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)
#here every feature will be ranked 
rank=list(rfe.ranking_)
columnss=list( X.columns)
selecetd_features=[]
for i in range(len(rank)-1):
    if rank[i]==1:
        selecetd_features.append(columnss[i])
print("After recursive feature elimination we got final features",selecetd_features)
print("Number of features after RFE is ", len(selecetd_features))
#So we  got required 21 columns from total 62 columns 
#Now we can build a model and check the significance  
cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 'default_no', 'default_unknown', 
      'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 
      'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"] 
X=os_data_X[cols]
y=os_data_y['y']
    
#IMPLEMENTING MODEL AND ANALYZING THE RESULT SUMMARY 
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())
#The p-values for most of the variables are smaller than 0.05, except four variables, therefore, we will remove them.

cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 
      'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 
      'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"] 
X=os_data_X[cols]
y=os_data_y['y']
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())
#Now all columns are significants

#LOGISTIC REGRESSION 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg.fit(X_train, y_train) 
y_pred = logreg.predict(X_test)

#ACCURACY
print("Accuracy of logistic regression classifier on test set")
y_pred = logreg.predict(X_test)
from sklearn import metrics
print("from metrics.accuracy_score  ",metrics.accuracy_score(y_test, y_pred))
#2ndway
print("from model.score ",logreg.score(X_test, y_test))
#ROC_ACCURACY
print(" ROC Accuracy score")
probs = logreg.predict_proba(X_test)
print(metrics.roc_auc_score(y_test, probs[:, 1]))

#REFERENCE :https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
#CONFUSION MATRIX :
#it is a performance measurement for machine learning classification problem where output can be two or more classes. 
#It is a table with 4 different combinations of predicted and actual values.
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix")
print(confusion_matrix)
#[[TRUE POSITIVE FALSE POSITIVE(TYPE-I)]
#  [FALSE NEGATIVE      TRUE NEGATIVE]]

#TP:You predicted positive and it’s true.
#TN:You predicted negative and it’s true.
#FP:You predicted positive and it’s false.
#FN: You predicted negative and it’s false.
#CONCLUSION FROM CONFUSION MATRIX :Total correct prediction =TP+TN Total incorrect prediction =FP+FN

#PRECISION :
#Out of all the positive classes we have predicted correctly, how many are actually positive.
# the ability of the classifier to not label a sample as positive if it is negative.
#PRECISION=TP/(TP+FP)

#RECALL :
#Out of all the positive classes, how much we predicted correctly. It should be high as possible.
#the ability of the classifier to find all the positive samples.
#RECALL=TP/(TP+FN)

#F SCORE :
#The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, 
#where an F-beta score reaches its best value at 1 and worst score at 0.
#f=(2* recall*precision)/(recall+precision)

#OBTAIN PRECISION RECALL F SCORE 
print("classification report")
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
#interpretation:
#Of the entire test set, 74% of the customer’s preferred term deposits that were promoted.

#ROC CURVE ( receiver operating characteristic (ROC))
#FPR=(False Positive Rate)=(FP/(FP+TN))
#TPR=(True Positive Rate)=(TP/TP+FN))
#The ROC curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right") 
plt.savefig('Log_ROC')
plt.show()

#AUC ROC SCORE , F SCORE ,ACCURACY 

#Evaluation of model using cross validation 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#cv = KFold(n_splits=10, random_state=42, shuffle=False)
scores = cross_val_score(logreg, X, y,cv=10 , scoring='accuracy')
print("10 fold cross validation result :")
print(scores)
print("mean of scores ",scores.mean())

# MAKE INDIVIDUAL PREDICTION 
print("List of final columns")
print(str(X.columns))
res=logreg.predict([[4.153,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0]])
#PER CLASS PROBABILITY
print("A perso having euribor 3 month rate 4.153,he is not a blue collar job , not a house maid ,with marital status is unknown  and he is illitrate ,booked on november and has p outcome failure ")
perclass=logreg.predict_proba([[4.153,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0]])
probabilities=perclass.flatten()
types=y.unique()
for i in range( 0,types.size):
    print(str(types[i])+" has probability "+str(round(probabilities[i],2)))
print("So selecting result ",res)    
#FOR OTHER COMBINATION ASK DOUBT 



