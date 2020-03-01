# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 08:23:50 2019

@author: INE12363221
"""
import pandas as pd 
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

credits=pd.read_csv("tmdb_5000_credits.csv")
movies_df=pd.read_csv("tmdb_5000_movies.csv")
#print(credits.head())
print(movies_df['genres'].head(2))

#From movies_df we will take overview column as it contains summary of the movie . This overview represents content of the movie 
#based on the content we will do content based recommendation i.e which movies have similar content 

#In credits df we can see it has movie_id and title  and movies_df we have id and overview  we need to merge this two dataframe on id 
# so rename movie_id to id then merge
credits_rename=credits.rename(index=str,columns={"movie_id":"id"})
movies_df_merge=movies_df.merge(credits_rename,on='id')
print(movies_df_merge.info())
#print(credits_rename.info())

#As we are considering only id that is unique id of movie ,overview,original_title rest columns we can drop rest
movies_cleaned_df=movies_df_merge[["id","overview","original_title","tagline","keywords","genres","cast"]]
#replace null overviews with empty string 
movies_cleaned_df['overview']=movies_cleaned_df['overview'].fillna('')
movies_cleaned_df['tagline']=movies_cleaned_df['tagline'].fillna('')
movies_cleaned_df['original_title']=movies_cleaned_df['original_title'].fillna('')
#make original_title to smaller case so insensitive search can be done 
def make_small(x):
    return x.lower()
movies_cleaned_df['original_title']=movies_cleaned_df['original_title'].apply(make_small)
print(movies_cleaned_df['original_title'])
from nltk.tokenize import word_tokenize
def remove_unnecessary(text):
    tokens = word_tokenize(text)
    # convert to lower case
    #tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    import string
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    #make smaller case for case insensitive search 
    #words = [w.lower() for w in words]
    str1 = " ".join(words) 
    return str1 
#movies
#movies_cleaned_df['overview'] = movies_cleaned_df.apply(overview,axis=1)
movies_cleaned_df['overview']  = movies_cleaned_df['overview'] .apply(remove_unnecessary)
movies_cleaned_df['tagline']  = movies_cleaned_df['tagline'] .apply(remove_unnecessary)

features = ['overview','tagline']
##Step 3: Create a column in DF which combines all selected features
for feature in features:
	movies_cleaned_df[feature] = movies_cleaned_df[feature].fillna('')

def combine_features(row):
	try:
		return row["tagline"]+" "+row["overview"]
	except:
		print ("Error:", row)	

movies_cleaned_df["combined_features"] = movies_cleaned_df.apply(combine_features,axis=1)
#print(movies_cleaned_df.head())

##Step 4: Create count matrix from this new combined column
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
count_matrix = cv.fit_transform(movies_cleaned_df["combined_features"])

##Step 5: Compute the Cosine Similarity based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(count_matrix) 
movie_user_likes = "avatar"
s=movies_cleaned_df['original_title']
print(s.loc[s.str.startswith(movie_user_likes, na=False)])
movie_user_likes=movie_user_likes.lower()

def get_title_from_index(index):
	return movies_cleaned_df[movies_cleaned_df.index == index].original_title.values[0]

def get_index_from_title(title):
    return movies_cleaned_df[movies_cleaned_df.original_title == title].index.values[0]


## Step 6: Get index of this movie from its title
movie_index = get_index_from_title(movie_user_likes)
#print(cosine_sim[movie_index])
similar_movies =  list(enumerate(cosine_sim[movie_index]))
#print(similar_movies)
#print(movies_cleaned_df[movies_cleaned_df.original_title == 'Avatar'].index.values[0])
    
## Step 7: Get a list of similar movies in descending order of similarity score
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)

## Step 8: Print titles of first 50 movies
i=0
for element in sorted_similar_movies:
		print (get_title_from_index(element[0]))
		i=i+1
		if i>5:
			break