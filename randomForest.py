import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df ['species'] = pd.Categorical.from_codes(iris.target, iris.target_names) #Just adding label name 

X, y = df.iloc[:,:4], df.iloc[:,4]
#print(df['species'].unique())

#Before labelEncoding indexing order is [setosa, versicolor, virginica]

lEncoder = LabelEncoder()
y = lEncoder.fit_transform(y)

xtrain, xtest, ytrain, ytest = train_test_split(X,y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_jobs=2, random_state=10, n_estimators=10)
clf.fit(xtrain, ytrain)
ypredict = clf.predict(xtest)


singularTree = clf.estimators_[0]
plt.figure(figsize=(12,8))
plot_tree(singularTree, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()



