import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.preprocessing 
import sklearn.model_selection

import itertools
%matplotlib inline

data = pd.read_csv('Cars93.csv')
#data[:2]

Y = np.array(data['Price'])
columns = ['MPG.city', 'MPG.highway', 'EngineSize', 'Horsepower', 'RPM', 'Rev.per.mile', 
          'Fuel.tank.capacity', 'Length', 'Width', 'Turn.circle', 'Weight']
X = np.array(data[columns])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_scaled, Y, test_size=0.3)
print(np.shape(Y_test), np.shape(X_test))
#print(X_test[:2])
#print(Y_test[1])
print(X_test[:,1])

regresion = sklearn.linear_model.LinearRegression()
R2 = []
a = np.arange(0, 11)
for i in range(0, len(a)):
    b = list(itertools.combinations(a, i+1))
    x = X_train[:,b]
    regresion.fit(X_train[:,b], Y_train)
    R2.append(regresion.score(X_train[:,b], Y_train))
f = []
for i in range(10):
    f.append(i)
plt.scatter(f, R2)