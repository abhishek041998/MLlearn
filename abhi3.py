import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('bank2.csv')
columns=dataset.columns.values[0].split(';')
columns = [column.replace('"', '') for column in columns]
dataset=dataset.values
dataset=[items[0].split(';') for items in dataset]
dataset=pd.DataFrame(dataset,columns = columns)

dataset['job']=dataset['job'].str.replace('"','')
dataset['marital']=dataset['marital'].str.replace('"','')
dataset['education']=dataset['education'].str.replace('"','')
dataset['default']=dataset['default'].str.replace('"','')
dataset['housing']=dataset['housing'].str.replace('"','')
dataset['loan']=dataset['loan'].str.replace('"','')
dataset['contact']=dataset['contact'].str.replace('"','')
dataset['month']=dataset['month'].str.replace('"','')
dataset['day_of_week']=dataset['day_of_week'].str.replace('"','')
dataset['poutcome']=dataset['poutcome'].str.replace('"','')
dataset['y']=dataset['y'].str.replace('"','')


X = dataset.iloc[:, 0:20].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
X[:,1]=lab.fit_transform(X[:,1])
X[:,2]=lab.fit_transform(X[:,2])
X[:,3]=lab.fit_transform(X[:,3])
X[:,4]=lab.fit_transform(X[:,4])
X[:,5]=lab.fit_transform(X[:,5])
X[:,6]=lab.fit_transform(X[:,6])
X[:,7]=lab.fit_transform(X[:,7])
X[:,8]=lab.fit_transform(X[:,8])
X[:,9]=lab.fit_transform(X[:,9])
X[:,14]=lab.fit_transform(X[:,14])
y=lab.fit_transform(y)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [1,2,3,4,5,6,7,8,9,14])
X=one.fit_transform(X)
X=X.toarray()
