import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('bank2.csv',na_values='unknown')



X=dataset.iloc[:,0:20].values
y=dataset.iloc[:,-1].values


from  sklearn.impute import SimpleImputer

sim=SimpleImputer()

X[:,[0,10,11,12,13,15,16,17,18,19]]=sim.fit_transform(X[:,[0,10,11,12,13,15,16,17,18,19]])

temp=pd.DataFrame(X[:,[1,2,3,4,5,6,7,8,9,14]])

temp[0].value_counts()
temp[1].value_counts()
temp[2].value_counts()
temp[3].value_counts()
temp[4].value_counts()
temp[5].value_counts()
temp[6].value_counts()
temp[7].value_counts()
temp[8].value_counts()
temp[9].value_counts()

temp[0]=temp[0].fillna('admin')
temp[1]=temp[1].fillna('married')
temp[2]=temp[2].fillna('university.degree')
temp[3]=temp[3].fillna('no')
temp[4]=temp[4].fillna('yes')
temp[5]=temp[5].fillna('no' )
temp[6]=temp[6].fillna('cellular')
temp[7]=temp[7].fillna('may')
temp[8]=temp[6].fillna('thu')
temp[9]=temp[9].fillna('nonexistent')


X[:,[1,2,3,4,5,6,7,8,9,14]]=temp
#del(temp)



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




from sklearn.preprocessing import OneHotEncoder
one=OneHotEncoder(categorical_features=[1,2,3,4,5,6,7,8,9,14])
X=one.fit_transform(X)
X=X.toarray()
y=lab.fit_transform(y)
lab.classes_

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)

from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier(n_neighbors=3)

kn.fit(X_train,y_train)
kn.score(X_train,y_train)

kn.fit(X_test,y_test)
kn.score(X_test,y_test)


kn.fit(X,y)
kn.score(X,y)