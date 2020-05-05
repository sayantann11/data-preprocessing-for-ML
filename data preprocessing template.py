import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, 3].values
np.set_printoptions(threshold=np.nan)

from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state = 0)
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)"""
