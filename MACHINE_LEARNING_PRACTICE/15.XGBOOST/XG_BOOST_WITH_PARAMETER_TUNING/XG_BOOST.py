#theory
#https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63d
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
df = pd.read_csv('income.csv')
print(df.info())
# Convert target to categorical
#here target is 
col = pd.Categorical(df.high_income)
df["high_income"] = col.codes
print(df.head(2))

#DATA CLEANING 

# Treat ? workclass as unknown
df.loc[df['workclass'] == '?', 'workclass'] = 'Unknown'
#print(df.native_country.unique())
# Too many categories, just convert to US and Non-US
df.loc[df['native_country']!='United-States','native_country']='non_usa'
# Get columns list for categorical and numerical
categorical_features = df.select_dtypes('object').columns.tolist()
numerical_features = df.select_dtypes('int64').columns.tolist()
print(numerical_features)

# Convert columns to categorical
for name in categorical_features:
  col = pd.Categorical(df[name])
  df[name] = col.codes
# Normalize numerical features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])
print(df.head(5))

#NOTE:
#MIn max scaler:
#This estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one.
x=df.drop('high_income', axis=1)
y=df['high_income']
print(x.shape)
#select  k best features 
from sklearn.feature_selection import SelectKBest, chi2
x = SelectKBest(chi2, k=10).fit_transform(x, y)


from sklearn.model_selection import train_test_split
# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42,shuffle=True,stratify=df['high_income'])

#benefit of stratify 
# if the data set has a large amount of each class, stratified sampling is pretty much the same as random sampling. 
# But if one class isn't much represented in the data set, which may be the case in your dataset since you plan 
# to oversample the minority class, then stratified sampling may yield a different target class distribution in the
# train and test sets than what random sampling may yield.

#select k best 
#oversampling or under sampling 
#xgb  classifier 
#k fold cross validation 
#xgb classifier
#https://medium.com/@juniormiranda_23768/ensemble-methods-tuning-a-xgboost-model-with-scikit-learn-54ff669f988a
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
#NOTE 
#MEANING OF ABOVE PARAMETERS :
   #eta [default=0.3, alias: learning_rate] Step size shrinkage used in update to prevents overfitting 
   #max_depth [default=6] Maximum depth of a tree
   #gamma gamma [default=0, alias: min_split_loss]Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.

# Define cross validation
from sklearn.model_selection import KFold
kfold = KFold(n_splits=2, random_state=42)
# AUC and accuracy as score
from sklearn.metrics import accuracy_score, make_scorer
scoring = {'AUC':'roc_auc', 'Accuracy':make_scorer(accuracy_score)}


# Define grid search
grid = GridSearchCV(
        clf,
  param_grid=parameters,
  cv=kfold,
  scoring=scoring,
  refit='AUC',
  verbose=1,
  n_jobs=2
)
from sklearn.metrics import confusion_matrix
model=grid.fit(X_train, y_train)
predict = model.predict(X_test)
print('Best AUC Score: {}'.format(model.best_score_))
print('Accuracy: {}'.format(accuracy_score(y_test, predict)))
print(confusion_matrix(y_test,predict))
print(model.best_params_)
model.dump_model('dump.raw.txt')
