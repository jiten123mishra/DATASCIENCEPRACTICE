#SIMPLE ONE HOT ENCODING 
import pandas as pd
import numpy as np
df = pd.DataFrame({"column1": ['a','b','c'], "column2": ['d','e','f'], "column3": ['g','h','i']})
#print(df)
#print(pd.get_dummies(df))
#print(pd.get_dummies(df, drop_first=True))

#********************
#COUNT FREQUENCY ENCODING 
#***********************
df = pd.read_csv('mercedesbenz.csv', usecols=['X1', 'X2'])
print(df.head())
#lets see how many unique values x1 has
print(len(df['X1'].unique()))
#27 means if we apply one hot encoding then 26 new columns will be added 
print(len(df['X2'].unique()))
#44 means if we apply one hot encoding then 43 new columns will be added 
print()
#One simple solution will be replace all those variable by their count 
#pro :most repeated value will get higher value 
#con :If some of the labels have the same count, then they will be replaced with the same count and they will loose some valuable information.
#lets do it

# first we make a dictionary that maps each label to the counts
df_frequency_map_X2 = df.X2.value_counts().to_dict()
df_frequency_map_X1 = df.X1.value_counts().to_dict()
print(df_frequency_map_X2)
df["X2_label"] = df.X2.map(df_frequency_map_X2)
df["X1_label"] = df.X1.map(df_frequency_map_X1)
#print(df.head())
#as at is repeated 6 times at will be replaced by 6 and so at will be replaced by 6 


#***********************************
#ONE HOT ENCODING FOR MANY VARIABLES 
#***********************************
df = pd.read_csv('mercedesbenz.csv', usecols=['X1', 'X2'])
#using 10 most frequent labels convert them into dummy variables using onehotencoding
#Suggested by  winning solution of the KDD 2009 cup

# let's find the top 10 most frequent categories for the variable X2
print(df.X2.value_counts().sort_values(ascending=False).head(20))
# let's make a list with the most frequent categories of the variable

top_10_labels = [y for y in df.X2.value_counts().sort_values(ascending=False).head(10).index]
print(top_10_labels)
# get whole set of dummy variables, for all the categorical variables

def one_hot_encoding_top_x(df, variable, top_x_labels):
    # function to create the dummy variables for the most frequent labels
    # we can vary the number of most frequent labels that we encode
    
    for label in top_x_labels:
        df[variable+'_'+label] = np.where(df[variable]==label, 1, 0)

# read the data again
df = pd.read_csv('mercedesbenz.csv', usecols=['X1', 'X2'])

# encode X2 into the 10 most frequent categories
one_hot_encoding_top_x(df, 'X2', top_10_labels)
print(df.head())
