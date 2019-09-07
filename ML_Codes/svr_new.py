import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')

x=dataset.iloc[:, 1:2].values
y=dataset.iloc[:, 2:3].values

#from sklearn.model_selection import train_test_split
#X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_Y=StandardScaler()
x=sc_X.fit_transform(x)
y=sc_Y.fit_transform(y)

from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(x,y)

#var1=np.array([[6.5]])
#
#var2=sc_X.fit_transform(var1)

Y_pred=sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#var=np.array([[6.5]])
#lin_reg.predict(var)
#
#Y_pred=regressor.predict(var)


X_grid=np.arange(min(x),max(x),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(x,y,color="red")
plt.plot(X_grid, regressor.predict(X_grid),color="blue")
plt.title("Truth or Bluf (SVR)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
