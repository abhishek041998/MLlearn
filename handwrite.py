import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_mldata
dataset= fetch_mldata("MNIST original")



X=dataset.data
y=dataset.target

some_digit=X[156]
some_digit_image=some_digit.reshape(28,28)

plt.imshow(some_digit_image)
plt.show()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)


from sklearn.tree import DecisionTreeClassifier
df=DecisionTreeClassifier(max_depth=13)

df.fit(X_train,y_train)
df.score(X_train,y_train)

df.fit(X_test,y_test)
df.score(X_test,y_test)