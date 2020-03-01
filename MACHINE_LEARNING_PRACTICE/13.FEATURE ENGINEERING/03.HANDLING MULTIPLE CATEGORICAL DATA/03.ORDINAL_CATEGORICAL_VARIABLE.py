# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:32:26 2019

@author: INE12363221
"""
import pandas as pd
import datetime
df_base = datetime.datetime.today()
df_date_list = [df_base - datetime.timedelta(days=x) for x in range(0, 20)]
df = pd.DataFrame(df_date_list)
df.columns = ['day']
df['day_of_week'] = df['day'].dt.weekday_name
print(df.head())
#WAY:1 
#PROVIDING CORRESPONDING NUMERICAL VALUES AS PER  OUR CHOICE LIKE GOOD 1 BETTTR 2 BEST 3 
weekday_map = {'Monday':1,
               'Tuesday':2,
               'Wednesday':3,
               'Thursday':4,
               'Friday':5,
               'Saturday':6,
               'Sunday':7
}

df['day_ordinal'] = df.day_of_week.map(weekday_map)
#print(df)
#USING SK LEARN LABEL ENCODING 
from sklearn import preprocessing  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
df['day_ordinal_sk']= label_encoder.fit_transform(df['day_of_week']) 
print(df)
#here we can not  control what value need to be assigned for what 
#more or less result is same  


