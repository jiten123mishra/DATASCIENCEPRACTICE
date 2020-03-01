#DATASET DESCRIPTION
#The dataset contains 6366 observations of 9 variables:
#rate_marriage: woman's rating of her marriage (1 = very poor, 5 = very good)
#age: woman's age
#yrs_married: number of years married
#children: number of children
#religious: woman's rating of how religious she is (1 = not religious, 4 = strongly religious)
#educ: level of education (9 = grade school, 12 = high school, 14 = some college, 16 = college graduate, 17 = some graduate school, 20 = advanced degree)
#occupation: woman's occupation (1 = student, 2 = farming/semi-skilled/unskilled, 3 = "white collar", 4 = teacher/nurse/writer/technician/skilled, 5 = managerial/business, 6 = professional with advanced degree)
#occupation_husb: husband's occupation (same coding as above)
#affairs: time spent in extra-marital affairs

#IMPORT
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score

# load dataset
df = sm.datasets.fair.load_pandas().data
# add "affair" column: 1 represents having affairs, 0 represents not
df['affair'] = (df.affairs > 0).astype(int)
#print(df.head())

#DATA EXPLORATION

#MEAN
#print(df.groupby('affair').mean())
#This means person having good rating have less chance of affair 
#as age and yrs_married increasing chance of affair increaing 
#print(df.groupby('rate_marriage').mean())
#An increase in age, yrs_married, and children appears to correlate with a declining marriage rating.

#HISTOGRAM
# histogram of education i.e Education level with entire population frequency
#print(df.educ.hist())
#plt.title('Histogram of Education')
#plt.xlabel('Education Level')
#plt.ylabel('Frequency')
##This means maximum people have education level 14

# histogram of marriage rating
#df.rate_marriage.hist()
#plt.title('Histogram of Marriage Rating')
#plt.xlabel('Marriage Rating')
#plt.ylabel('Frequency')
##This means maximum people have given rating 5 for their marraige 

# barplot of marriage rating grouped by affair (True or False)
#pd.crosstab(df.rate_marriage, df.affair.astype(bool)).plot(kind='bar')
#plt.title('Marriage Rating Distribution by Affair Status')
#plt.xlabel('Marriage Rating')
#plt.ylabel('Frequency')
##People having rating 4 have highest number of affairs 

#HANDLING CATEGORICAL DATA by DMATRICES
# As we have many categorical data let us assume that ccupation and occupation_husb are the only two categorical data 
#Using dmatrices from patsy we can achieve effect of 
#4 things a.onehotencoding b.dummy variable trap c. addig  intercept d.segregate input output  e.selecting required column combinedly 
# create dataframes with an intercept column and dummy variables for
# occupation and occupation_husb

y, x = dmatrices('affair ~ rate_marriage + age + yrs_married + children + \
                  religious + educ + C(occupation) + C(occupation_husb)',
                  df, return_type="dataframe")
print(x.columns)
#We can change names of newly added columns
x = x.rename(columns = {'C(occupation)[T.2.0]':'occ_2',
                        'C(occupation)[T.3.0]':'occ_3',
                        'C(occupation)[T.4.0]':'occ_4',
                        'C(occupation)[T.5.0]':'occ_5',
                        'C(occupation)[T.6.0]':'occ_6',
                        'C(occupation_husb)[T.2.0]':'occ_husb_2',
                        'C(occupation_husb)[T.3.0]':'occ_husb_3',
                        'C(occupation_husb)[T.4.0]':'occ_husb_4',
                        'C(occupation_husb)[T.5.0]':'occ_husb_5',
                        'C(occupation_husb)[T.6.0]':'occ_husb_6'})


#here we can observe there is 6 categories for womens occupation but we are getting 5 as one column got removed because dummy variable trap
#also we have a intercept column 

##converting y to 1 D nd array so that it can fit into machine learning now both x and y are data frame 
#print(type(x))
#print(type(y))
## flatten y into a 1-D array
y = np.ravel(y)
#print(type(y))
model = LogisticRegression()
model = model.fit(x, y)

# check the accuracy on the training set
print("model_score =",model.score(x, y))

#MEAN AND NULL ERROR RATE 
print("mean of affair column is ",y.mean())
#NULL ERROR RATE This is how often you would be wrong if you always predicted the majority class
#Here 32% of women have affair so majority class is No affair .which means that you could obtain 68% accuracy by always predicting "no". So we're doing better than the null error rate, but not by much
#2nd way to calculate null error rate is that total instance of majority class/total observation .

#DERIVING MEANING FROM MODEL COEFFICIENTS 
#print(x.columns, model.coef_)
#Increases in marriage rating and religiousness correspond to a decrease in the likelihood of having an affair.
#For both, wife's occupation and the husband's occupation, the lowest likelihood of having an affair corresponds
#to the baseline occupation (student), since all of the dummy coefficients are positive.
#RULE :Model cofficient larger+ve  means it will directly impact  mdel cefiecient larger -ve means will impact inversly 

#DERIVING MODEL BY SPLITTING INTO TRAIN AND TEST 
#So far, we have trained and tested on the same set. Let's instead split the data into a training set and a testing set.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
model2 = LogisticRegression(solver='liblinear')
model2.fit(x_train, y_train)
y_pred = model2.predict(x_test)
df1= pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

#GENERATAING EVALUATION MATRICES
probs = model2.predict_proba(x_test)
#ACCURACY
print("Accuracy score")
print(metrics.accuracy_score(y_test, y_pred))
#ROC_ACCURACY
print(" ROC Accuracy score")
print(metrics.roc_auc_score(y_test, probs[:, 1]))
#CONFUSION MATRIX
print(" Cofusion matrix score")
print(metrics.confusion_matrix(y_test, y_pred))
#CLASSIFICATION REPORT 
print(" Classification report score")
print(metrics.classification_report(y_test, y_pred))

#K FOLD  CROSS VALIDATION 
#That k-fold cross validation is a procedure used to estimate the skill of the model on new data.
#STEP:1 Shuffle the dataset randomly.
#STEP:2 Split the dataset into k groups
#STEP:3 For each unique group:
#        a.Take one group as a test data set
#        b.Take the remaining groups as a training data set
#        c.Fit a model on the training set and evaluate it on the test set
#Evaluation of model using cross validation 
from sklearn.model_selection import KFold
#cv = KFold(n_splits=10, random_state=42, shuffle=False)
scores = cross_val_score(model2, x, y,cv=10 , scoring='accuracy')
print("10 fold cross validation result :")
print(scores)
print("mean of scores ",scores.mean())
#K FOLD CROSS VALIDATION SIGNIFICANCE 
#Model accuracy can vary significantly from one fold to the next, especially with small data sets, 
#but the average accuracy across the folds gives you an idea of how the model might perform on unseen data.

#PREDICTING PROBBAILITY 
print(model.predict_proba(np.array([[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 3, 29, 1, 1, 3,16]])))
print("probability of having affair is 23%")