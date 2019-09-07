import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')

x=dataset.iloc[:, 1:2].values
y=dataset.iloc[:, 2:3].values

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(x,y)

var=np.array([[6.5]])
regressor.predict(var)
Y_pred=regressor.predict(var)


X_grid=np.arange(min(x),max(x),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(x,y,color="red")
plt.plot(X_grid, regressor.predict(X_grid),color="blue")
plt.title("Truth or Bluf (random forest Regression )")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
