import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv(r'Advertising.csv')
#Display head of data set 

#DATA CLEANING 
# AS Unnamed column is just a  column is not contributing 
#column 1 TV 2. Radio 3. NewsPaper
X = dataset.iloc[:, [1,2,3]].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

def cost_function(X, Y, B):
 m = len(Y)
 J = np.sum((X.dot(B)-Y) ** 2)/(2 * m)
 return J

#In Batch Gradient Descent Function we take parameters
#X: Feature Matrix
#Y: an array of target values
#B: initial value of theta
#alpha: learning rate
#iterations: max no. of iterations for algorithm
 
def batch_gradient_descent(X, Y, B, alpha, iterations):
    cost_history = np.zeros(iterations)              
    m = len(Y)
    for iteration in range(iterations):
        #print(iteration)
        # Hypothesis Values
        yhat = X.dot(B)
        # Changing Values of B using Gradient
        B = B - alpha *(X.T.dot(yhat - Y)) / m
        # New Cost Value
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost
    
    return B, cost_history


def stocashtic_gradient_descent(X,Y,B1,alpha,iterations):
    '''
    X    = Matrix of X with added bias units
    y    = Vector of Y
    theta=Vector of thetas np.random.randn(j,1)
    learning_rate 
    iterations = no of iterations
    
    Returns the final theta vector and array of cost history over no of iterations
    '''
    m = len(Y)
    cost_history = np.zeros(iterations)
    for it in range(iterations):
        cost =0.0
        for i in range(m):
            rand_ind = np.random.randint(0,m)
            X_i = X[rand_ind,:].reshape(1,X.shape[1])
            y_i = Y[rand_ind].reshape(1,1)
            print(X_i.shape)
            print(B1.shape)
            print(y_i.shape)
            yhat = X_i.dot(B1)
            print(yhat.shape)
            loss=yhat - y_i
            print(X_i.shape)
            print(loss.shape)
            gradient=X_i.T.dot(loss)
            B1 = B1 - alpha *gradient/ m
            cost += cost_function(X_i,y_i,B1)
        cost_history[it]  = cost  
    return B1, cost_history
def pred(X,B):
    return(X.dot(B))


def r2(y_,y):
 sst = np.sum((y-y.mean())**2)
 ssr = np.sum((y_-y)**2)
 r2 = 1-(ssr/sst)
 return(r2)
 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)
#below steps are done to add one column with value 1 to x vector because we are considering x0=1 
print(X_train[0])
X_train = np.c_[np.ones(len(X_train),dtype='int64'),X_train]
print(X_train[0])
X_test = np.c_[np.ones(len(X_test),dtype='int64'),X_test]

B = np.zeros(X_train.shape[1])
print("BBB",B)
alpha = 0.005
iter_ = 1
newB, cost_history = batch_gradient_descent(X_train, y_train, B, alpha, iter_)
y_ = pred(X_test,newB)
print(r2(y_,y_test))
B = np.zeros(X_train.shape[1])
newB, cost_history = stocashtic_gradient_descent(X_train, y_train, B, alpha, iter_)
y_ = pred(X_test,newB)
print(r2(y_,y_test))