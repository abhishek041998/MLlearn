import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset =pd.read_excel('blood.xlsx')
X=dataset.iloc[:,1].values
y=dataset.iloc[:,-1].values
X=X.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)

from sklearn.linear_model import LinearRegression
li=LinearRegression()

li.fit(X_train,y_train)
li.score(X_train,y_train)

plt.scatter(X_train, y_train)
plt.plot(X_train, li.predict(X_train), 'r-')

li.fit(X_test,y_test)
li.score(X_test,y_test)

plt.scatter(X_test, y_test)
plt.plot(X_test, li.predict(X_test), 'r-')

li.fit(X,y)
li.score(X,y)

plt.scatter(X, y)
plt.plot(X, li.predict(X), 'r-')

li.predict([[20]])

li.coef_
li.intercept_