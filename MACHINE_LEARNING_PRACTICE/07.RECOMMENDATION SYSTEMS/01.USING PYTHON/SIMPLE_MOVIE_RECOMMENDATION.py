# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:03:48 2019

@author: INE12363221
"""
#Here our objective is to get similar moviees using correlation if one movie name is given 
#movie1 got rating 3,4,5,2,1
#movie2 got rating 1,4,4,NaN,3
#movie3 got rating 2,3,4,5,NaN
#How movie1 is similar to movie 2 and movie3 

import numpy as np
import pandas as pd
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)
movie_titles = pd.read_csv("Movie_Id_Titles")
#Now let's get the movie titles:
df = pd.merge(df,movie_titles,on='item_id')
print(df.head())
#Now let's create a matrix that has the user ids on one access and the movie title on another axis.
 #Each cell will then consist of the rating the user gave to that movie. Note there will be a lot of NaN values,
 #because most people have not seen most of the movies.
 
 #result of pivot table 
# =============================================================================
# user rating movie
# user1 4.0    A
# user2 3.0    B 
# user3 5.0    A
# user3 4.0    A
# user1 3.0    A
#        A    B 
# user1  3.5  Nan
# user2  Nan  3.0
# user3  5.0  4.0
# NOTE :if same user found more than one time then it will take avarage of rating as index can not be duplicate 
# =============================================================================
 #
 
 
moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
print(moviemat.head())
#Now let's grab the user ratings for those two movies:
starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']

#lets find Similarity of star wars with other movies 
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
#similar_to_starwars contains all movies name and correlation coefficeient with star wars 
#create a new  dataframe corr_starwars  with column name as correlation 
corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
print(corr_starwars.head())
#Now if we sort the dataframe by correlation, we should get the most similar movies,
 #however note that we get some results that don't really make sense. This is because 
 #there are a lot of movies only watched once by users who also watched star wars (it was the most popular movie).
 
#print(corr_starwars.sort_values('Correlation',ascending=False).head(10))
#Let's fix this by filtering out movies that have less than 100 reviews 
 
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
corr_starwars = corr_starwars.join(ratings['num of ratings'])
print(corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head())
