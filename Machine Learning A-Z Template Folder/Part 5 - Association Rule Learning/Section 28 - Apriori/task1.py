"""
Created on Tue Oct  2 16:37:26 2018

@author: shadab
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
#dataset is in dataFrame but for apriori we need list of list as dataset so make it
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

#training apriori on dataset
#apyori is a file of functions of apriori so import it
#minimum support is selected by supposing that dataset is of a week and we want the product to be bought 3 times a day
#minimum length should be atleast 2 of rules
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
#results = [rules]
results = list(rules)
print(results)