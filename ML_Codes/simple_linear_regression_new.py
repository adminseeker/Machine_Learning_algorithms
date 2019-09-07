import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Salary_Data.csv')

x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=1/3,random_state=0)

'''from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)'''

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

Y_pred=regressor.predict(X_test)

plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train, regressor.predict(X_train))
plt.title("Salary vs Experience(Training set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(X_test,Y_test,color="red")
plt.plot(X_train, regressor.predict(X_train))
plt.title("Salary vs Experience(Testing set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()