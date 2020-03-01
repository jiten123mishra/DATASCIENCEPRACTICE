#BASED ON FEATURES DECIDE WHETHER LABEL IS 0 or 1 

#LIBRARIES
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

#PART:01 :DATA PROCESSING 
dataset=pd.read_csv('Churn_Modelling.csv')

print(dataset.info())
#now we came to know surname geography and gender are of Object type 
print(dataset.isnull().sum(axis=0))
#to see if we have any null value 

#As we know Rownumber ,customer ID and surname not going to make any impact on prediction 
x=dataset.iloc[:,3:13]
y=dataset.iloc[:,13]
#print(x.head())
#print(y.head())
#Handle categorical features
#WAY:1 
x=pd.get_dummies(x,drop_first=True)
print(x.info())
#WAY:2
#geography=pd.get_dummies(dataset["Geography"],drop_first=True)
#gender=pd.get_dummies(dataset["Gender"],drop_first=True)
#x=pd.concat([x,geography,gender],axis=1)
#x=x.drop(['Geography','Gender'],axis=1)
#print(x.info())

#splitting the dataset into the training set and test set 
x=x.values
y=y.values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state=42 )

#Feature scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
#why  Feature scaling is required 
#need to bring all datas  to same scale as age and training score difference is too high 




#PART:2
#lets make the ANN 

#ANN
import tensorflow.keras 
#Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. 
from tensorflow.keras.models import Sequential
#The Sequential model is a linear stack of layers.
from tensorflow.keras.layers import Dense
#Dense implements the operation: output = activation(dot(input, kernel) + bias) 
#where activation is the element-wise activation function passed as the activation argument, 
#kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True).
from tensorflow.keras.layers import LeakyReLU,PReLU,ELU
from tensorflow.keras.layers import Dropout
#Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.

#INITIALIZING ANN
classifier=Sequential()
#Adding input layer and first hidden layer
classifier.add(Dense(units=10, input_shape=(11,),kernel_initializer='he_uniform'))
classifier.add(Dropout(0.3))
#NUMBER OF NODES IN INPUT LAYER :11 as number of column is 11 in input feature OUTPUT will be 6 which will be input to hidden layer 
# Adding the second hidden layer
classifier.add(Dense(units=20, kernel_initializer = 'he_uniform',activation='relu'))
classifier.add(Dropout(0.4))
#Adding third hidden layer 
classifier.add(Dense(units=6, kernel_initializer = 'he_normal',activation='relu'))
classifier.add(Dropout(0.2))
# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])

#for binary classification :binary_crossentropy
#for multi class classification :categorical_crossentropy 

# Fitting the ANN to the Training set
model_history=classifier.fit(x_train, y_train,validation_split=0.33, batch_size = 10, nb_epoch = 50)

# list all data in history
print(model_history.history.keys())

# summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#NOTE :
#accuracy:training accuracy 
#val_accuracy :validation or test accuracy 
#loss :training loss 
#val_loss:validation or test loss 

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)
from sklearn import metrics
#GENERATAING EVALUATION MATRICES
#CONFUSION MATRIX
print(" Cofusion matrix score")
print(metrics.confusion_matrix(y_test, y_pred))
probs = classifier.predict_proba(x_test)
#ACCURACY
print("Accuracy score")
print(metrics.accuracy_score(y_test, y_pred))




