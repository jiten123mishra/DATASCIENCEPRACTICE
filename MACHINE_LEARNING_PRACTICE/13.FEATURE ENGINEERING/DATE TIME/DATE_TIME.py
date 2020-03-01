# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 21:49:59 2019

@author: INE12363221
"""
import pandas as pd

use_cols = ['loan_amnt', 'grade', 'purpose', 'issue_d', 'last_pymnt_d']
data = pd.read_csv('loan1.csv', usecols=use_cols)
#see first five rows 
#print(data.head())

#See the types of data number of rows and column 
#print(data.info())     

#create a new column and covert issue_d and last_pymt_dt to date 
data['issue_dt'] = pd.to_datetime(data.issue_d, errors = 'coerce')
data['last_pymt_dt'] = pd.to_datetime(data.last_pymnt_d, errors = 'coerce')
print(data.head())


# let's see how much money Lending Club has disbursed
# (i.e., lent) over the years to the different risk
# markets (grade variable)

fig = data.groupby(['issue_dt', 'grade'])['loan_amnt'].sum().unstack().plot(
    figsize=(14, 8), linewidth=2)

fig.set_title('Disbursed amount in time')
fig.set_ylabel('Disbursed Amount (US Dollars)')