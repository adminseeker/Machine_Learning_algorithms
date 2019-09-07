import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Churn_Modelling.csv')

x=dataset.iloc[:, 3:13].values
y=dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1=LabelEncoder()
labelencoder_X_2=LabelEncoder()
x[:,1]=labelencoder_X_1.fit_transform(x[:,1])
x[:,2]=labelencoder_X_1.fit_transform(x[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
x=onehotencoder.fit_transform(x).toarray()
x=x[:,1:]


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#import keras
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()

classifier.add(Dense(output_dim=6,init="uniform",activation="relu",input_dim=11))
classifier.add(Dense(output_dim=6,init="uniform",activation="relu"))
classifier.add(Dense(output_dim=1,init="uniform",activation="sigmoid"))

classifier.compile(optimizer="adam", loss="binary_crossentropy",metrics=["accuracy"])
classifier.fit(X_train,Y_train,batch_size=10,nb_epoch=100)

Y_pred=classifier.predict(X_test)
Y_pred=(Y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test, Y_pred)