# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 22:51:25 2019

@author: INE12363221
"""
text=["london paris london","paris paris london "]
#We need to come up with a count matrix 
#[[2,1],[1,2]]
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
count_matrix = vectorizer.fit_transform(text)
print(count_matrix.toarray())
from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity(count_matrix))