import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('50_Startups.csv')

x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 4].values


#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X=LabelEncoder()
#x[:,3]=labelencoder_X.fit_transform(x[:,3])
#onehotencoder=OneHotEncoder(categorical_features=[3])
#x=onehotencoder.fit_transform(x).toarray()

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

ct_X = ColumnTransformer([('encoder',OneHotEncoder(),[3])], remainder='passthrough')
x=np.array(ct_X.fit_transform(x), dtype=np.float)

x=x[:, 1:]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

Y_pred=regressor.predict(X_test)

import statsmodels.formula.api as sm
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
X_opt=x[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=x[:,[0,3]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()


