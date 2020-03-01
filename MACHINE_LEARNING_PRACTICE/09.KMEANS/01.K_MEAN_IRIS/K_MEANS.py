import pandas as pd
from pandas import Series,DataFrame

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import classification_report

df=pd.read_csv('iris.csv')

#df['species_label'], _=pd.factorize(df['species'])
#factorize method is used to get numerical values for categorical values 
y = df['species']
x = df[['petal_length','petal_width','sepal_length','sepal_width']]

from sklearn.cluster import KMeans
km = KMeans(n_clusters = 3, n_jobs = 4, random_state=21)
km.fit(x)

#EVALUATION OF K  MEANS 
def converter(prvt):
    if prvt == 'Iris-setosa':
        return 1
    elif prvt == 'Iris-virginica':
        return 2
    else:
        return 0

y_pred = km.labels_
y_test =df['species'].apply(converter)

print("Accuracy of K means clustering on test set:Accuracy score")
print(metrics.accuracy_score(y_test, y_pred))
#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix")
print(confusion_matrix)
print("classification report for the same ")
print(classification_report(y_test, y_pred))
#Form new dataframe with cluster
df['Cluster'] =  km.labels_
#print(df)

#How to find optimum value of k using elbow method 
#https://pythonprogramminglanguage.com/kmeans-elbow-method/
# k means determine k
import numpy as np
from scipy.spatial.distance import cdist
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(x)
    kmeans = KMeans(n_clusters = k, init = 'k-means++', random_state = 42)
    kmeans.fit(x)
    distortions.append(kmeans.inertia_)

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

#When K increases, the centroids are closer to the clusters centroids.
#The improvements will decline, at some point rapidly, creating the elbow shape.
#That point is the optimal value for K. In the image above, K=3.

#kmeans.inertia_
#Sum of squared distances of samples to their closest cluster center.

#init{‘k-means++’, ‘random’ or an ndarray}
#Method for initialization, defaults to ‘k-means++’:
#
#‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. See section Notes in k_init for more details.
