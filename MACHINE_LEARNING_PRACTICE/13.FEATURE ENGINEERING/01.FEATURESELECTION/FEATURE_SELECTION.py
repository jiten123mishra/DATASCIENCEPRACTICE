##DATA SET 

#importing libraries
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
#Loading the dataset
x = load_boston()
df = pd.DataFrame(x.data, columns = x.feature_names)
df["MEDV"] = x.target
X = df.drop("MEDV",1)   #Feature Matrix
y = df["MEDV"]          #Target Variable
#print(df.head())
#print(df.info())


#WE have  13 FEATURE COLUMN 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT'   X is a Dataframe 
#We have one output or target column "MEDV"       y is a series 

#######################################################
#FEATURE SELECTION USING FILTER METHOD : CORRELATION 
########################################################
#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = df.corr()
#sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
#plt.show()

#This graph will show how all features are correlated with each other as
 
#Here we are intreseted in which features are highly correlated with y i.e output variable 
#Correlation with output variable
cor_target = abs(cor["MEDV"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
#print(relevant_features)

#RM         0.695360
#PTRATIO    0.507787
#LSTAT      0.737663
#MEDV       1.000000
#We cn see RM , PSTAT,LSTAT are highly correlated But in assumptions of linear regression we know that the fatures should be highly correlated with output but 
#should not be highly correlated among themselves 
#So RM, PTRATIO, LSTAT highly correlated or not?
#can be found from heatmap or below code 
#print(df[["LSTAT","PTRATIO","RM"]].corr())

#we can see RM and LSTAT are highly correlated with each other so we can  take any one 
#so Final selected features will be PTRATIO and RM 

# ALL ABOVE TASK CAN BE DONE PROGRAMATICALLY
# #TASK:1:
# #Create a new data frame by considering the columns which are highly correlated with target or label 
# cor_target = abs(cor["MEDV"])
# #print(cor_target)
# relevant_features = cor_target[cor_target>0.5]
# df_relevant=df[relevant_features.index] 


# #TASK2:
# #from the df which which is highly correlated with target drop those column which are highly correlated with each other 
# #drop target column
# cor = df_relevant.corr().abs()
# to_drop = [column for column in cor.columns if any(cor[column] > 0.95)]

# f_relevant=df_relevant.drop(df_relevant[to_drop], axis=1)
# print(df_relevant.info())


#######################################################
#FEATURE SELECTION USING WRAPPER METHOD : 1.BACKWARD ELIMINATION 
########################################################
#Adding constant column of ones, mandatory for sm.OLS model
X_1 = sm.add_constant(X)
#X_1 is the new feature after add a constant column of ones in X 
#Fitting sm.OLS model
model = sm.OLS(y,X_1).fit()
#Pvalue for each column 
#print(round(model.pvalues,6))

#Backward Elimination
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols] # create data frame by adding columns 
    X_1 = sm.add_constant(X_1) #Add column with ones
    #print(list(X_1.columns))
    model = sm.OLS(y,X_1).fit()  # OLS model fit 
    p = pd.Series(model.pvalues.values[1:],index = cols) #create a data frame p with  index as column name and p value as 1st column 
    pmax = max(p)                            #Get highest value of p as pmax 
    feature_with_p_max = p.idxmax() # get index of the df where value of column is  maximum 
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
#print("Best features selected through backward elimination is ",selected_features_BE)
#['CRIM', 'ZN', 'CHAS', 'NOX', 'RM', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']


#######################################################
#FEATURE SELECTION USING WRAPPER METHOD : 2.RECURSIVE FEATURE ELIMINATION 
########################################################
#3 PARTS 
#PART -A EXAMPLE SHOWING SUPPORT AND RANKING 
#PART -B FINDING OPTIMUM NUMBER OF FEATURES 
#PART -C FIND BEST FEATURE USING OPTIMUM NUMBER OF FEATURES 

#PART -A
###################################################
#For recurive feature seection we need 2 things 1. A model and 2. Number of features
#In this example PART A number of features 7 taken for deonstartion purpose only 
model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, 7)
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  
#Fitting the data to model
model.fit(X_rfe,y)
print(rfe.support_)#support = False means redundant column 
print(rfe.ranking_)#ranking =1 means most suitable 

#PART -B
###################################################
#WAY:1
#Feature ranking with recursive feature elimination and cross-validated selection of the best number of features.
#step:number of features to remove at each iteration
#cv:integer, to specify the number of folds for cross validation 
from sklearn.feature_selection import RFECV
model = LinearRegression()
selector = RFECV(model, step=1, cv=10)
selector = selector.fit(X, y)
#Feature ranking with recursive feature elimination and cross-validated selection of the best number of features.
print("optimum number of features selected through cross validation ",selector.n_features_ )

#WAY-2
#Optimum number of features can be found out by itearting a loop with value 1 to 13 i.e total length of raw features and keep track of accuracy 
#for the number of features we are getting maximum accuracy that will be the optimum feature number 
#no of features
nof_list=np.arange(1,13)            
high_score=0
#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))

#PART-C:
###############################################################
cols = list(X.columns)
model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, 6)             
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,y)  
#Fitting the data to model
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)

#EMBEDDED METHOD :
#Embedded methods are iterative in a sense that takes care of each iteration
# of the model training process and carefully extract those features which 
#contribute the most to the training for a particular iteration. Regularization 
# methods are the most commonly used embedded methods which penalize a feature given a coefficient threshold.

#Here we will use lasso CV 
reg = LassoCV()
reg.fit(X, y)
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))

#To see which coeffiecient made zero by lasso
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
# Tosee visually 
imp_coef = coef.sort_values()
#import matplotlib
#matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
#imp_coef.plot(kind = "barh")
#plt.title("Feature importance using Lasso Model")


#FROM INTERNET 
#https://towardsdatascience.com/the-5-feature-selection-algorithms-every-data-scientist-need-to-know-3a6b566efd2
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l1"), max_features=5)
embeded_lr_selector.fit(X, y)

embeded_lr_support = embeded_lr_selector.get_support()
embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
print(str(len(embeded_lr_feature)), 'selected features')
