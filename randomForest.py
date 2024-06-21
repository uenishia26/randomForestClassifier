import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree #Plot tree sklearn module allows us to plot trees
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import StratifiedKFold


df = pd.read_csv("formatedData.csv")

X,y = df.iloc[:,:8], df.iloc[:,-1]

"""
    By looking at the classification report and in particular the f1 score, 
    and take the avg div by numSplits you can see which model performed better 
    There are two classifiers use both to train and predict 
    
"""

skf = StratifiedKFold(shuffle=True, random_state=42, n_splits=5)
rfc = RandomForestClassifier(random_state=42, n_estimators=100)
bbClassifier = BalancedBaggingClassifier(rfc,sampling_strategy='auto',replacement=False,random_state=42)
sumMacroF1Score = 0

for train_index, test_index in skf.split(X,y): 
    xtrain = X.iloc[train_index,:8]
    xtest = X.iloc[test_index, :8]
    ytrain = y.iloc[train_index]
    ytest = y.iloc[test_index]
    bbClassifier.fit(xtrain, ytrain)
    ypredict = bbClassifier.predict(xtest)
    print(classification_report(ytest,ypredict))


""""
    *** Plotting Trees ***
        singularTree = clf.estimators_[0]
        plt.figure(figsize=(12,8))
        plot_tree(singularTree, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
        plt.show()
"""


