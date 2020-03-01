# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 07:30:51 2019

@author: INE12363221
"""

import pandas as pd
df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
                         "bar", "bar", "bar", "bar"],
                   "B": ["one", "one", "one", "two", "two",
                          "one", "one", "two", "two"],
                    "C": ["small", "large", "large", "small",
                          "small", "large", "small", "small",
                          "large"],
                "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                  "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})
print(df)
table = pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'], aggfunc=np.sum)
moviemat = df.pivot_table(index='D',columns='A',values='E')
print(moviemat)
