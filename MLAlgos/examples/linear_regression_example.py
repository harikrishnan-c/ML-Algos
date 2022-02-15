import numpy as np
import sys
from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from supervised_learning.linear_regression import LinearRegression

m = 3
n = 100
X = 2 * np.random.rand(n, m)
y = (np.multiply(X, [3.0, 4.0, 8.0])).sum(axis=1).reshape(n, 1) + 4.0 + np.random.rand(n, 1)
X_train = X[:80]
y_train = y[:80]
X_test = X[80:]
y_test = y[80:]
reg = LinearRegression()
reg.fit(X_train, y_train, fit_intercept=False)
print(reg.coef_)
print(reg.intercept_)
print(reg.predict(X_test))
