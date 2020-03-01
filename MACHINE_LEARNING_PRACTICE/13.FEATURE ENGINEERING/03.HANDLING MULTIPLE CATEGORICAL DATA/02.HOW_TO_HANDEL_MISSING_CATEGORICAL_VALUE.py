#WAYS TO HANDEL CATEGORICAL MISSING VALUES 
#1.DELETE ROWS 
#2.REPLACE WITH MOST FREQUENT VALUE
#3.APPLY CLASSIFIER ALGOTITHM TO PREDICT MISSING VALUES 
#4.APPLY UNSUPERVISED MODEL TO PREDICT MISSING  VALUES 
import pandas as pd 
import numpy as np
df = pd.DataFrame({"column1": ['a',np.nan,'c','a'], "column2": ['d','e','f','f'], "column3": ['g','h','i','h'],"output": ['yes','no','yes','yes']})

#How to find NAn information 

print(df.info())

print(df.isna().sum())
#1.Deleterow with NaN values
df1 = df.dropna(how='any',axis=0)
print(df1)
#2.replace with most repeated 
#list of columns can be obtained from print(df.isna().sum())
cols = ["column1"]
df[cols]=df[cols].fillna(df.mode().iloc[0])
print(df)
#Use classification algorithm 
#making feature as column2, column3 and output with label as column1 

#4. Kmeans clustering  with features as column2 and column 3 with k=2 
