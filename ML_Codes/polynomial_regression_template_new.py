import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')

x=dataset.iloc[:, 1:2].values
y=dataset.iloc[:, 2].values

#from sklearn.model_selection import train_test_split
#X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)

'''from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)'''



#frfkf



var=np.array([[6.5]])
lin_reg.predict(var)

Y_pred=regressor.predict(var)


X_grid=np.arange(min(x),max(x),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(x,y,color="red")
plt.plot(X_grid, regressor.predict(X_grid),color="blue")
plt.title("Truth or Bluf (Regression Model)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

