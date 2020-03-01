# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 16:14:47 2019

@author: INE12363221
"""

from sklearn import datasets
import xgboost as xgb

iris = datasets.load_iris()
X = iris.data
y = iris.target
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
#       base_score=0.5, booster='gbtree', colsample_bylevel=1,
#       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
#       max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
#       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
#       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#       silent=True, subsample=1

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)
mod = xgb.XGBClassifier(
    gamma=1,                 
    learning_rate=0.01,
    max_depth=3,
    n_estimators=10000,                                                                    
    subsample=0.8,
    random_state=34
) 
from sklearn.metrics import precision_score, recall_score, accuracy_score
mod.fit(X_train, Y_train)
Y_pred = mod.predict(X_test)
print(Y_pred)
print(Y_test)
print("Precision = {}".format(precision_score(Y_test, Y_pred, average='macro')))
print("Recall = {}".format(recall_score(Y_test, Y_pred, average='macro')))
print("Accuracy = {}".format(accuracy_score(Y_test, Y_pred)))


#SEARCHING OPTIMUM PARAMETER 
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
clf = xgb.XGBClassifier()
parameters = {
     "eta"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
     "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
     "min_child_weight" : [ 1, 3, 5, 7 ],
     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
     }
# Define cross validation
from sklearn.model_selection import KFold
kfold = KFold(n_splits=2, random_state=42)
# AUC and accuracy as score
from sklearn.metrics import accuracy_score, make_scorer
scoring = {'AUC':'roc_auc', 'Accuracy':make_scorer(accuracy_score)}
grid = GridSearchCV(
        clf,
  param_grid=parameters,
  cv=kfold,
  scoring=None,
  refit=True,
  verbose=1,
  n_jobs=2
)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42,shuffle=True,stratify=y)

from sklearn.metrics import confusion_matrix
model=grid.fit(X_train, y_train)
predict = model.predict(X_test)
print('Best AUC Score: {}'.format(model.best_score_))
print('Accuracy: {}'.format(accuracy_score(y_test, predict)))
print(confusion_matrix(y_test,predict))
print(model.best_params_)

#ALWAYS REMEMBER FOR MULTI CALSS CLASSIFICATION 
#SCORING=AUC is not supported  thats why scoring = None mase  



#OUTPUT:
#Best AUC Score: 0.975
#Accuracy: 0.9
#[[10  0  0]
# [ 0  9  1]
# [ 0  2  8]]
#{'colsample_bytree': 0.3, 'eta': 0.05, 'gamma': 0.1, 'max_depth': 3, 'min_child_weight': 3}