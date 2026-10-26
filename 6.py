import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = np.arange(1, 51).reshape(-1, 1)
rng = np.random.default_rng(42)
y = 2 * X.ravel()**2 + 3 * X.ravel() + rng.normal(0, 100, 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear = LinearRegression().fit(X_train, y_train)

poly = PolynomialFeatures(degree=2, include_bias=False)
Xp_train = poly.fit_transform(X_train)
Xp_test = poly.transform(X_test)
polynomial = LinearRegression().fit(Xp_train, y_train)

for name, pred in [
    ("Linear", linear.predict(X_test)),
    ("Polynomial", polynomial.predict(Xp_test))
]:
    print(name)
    print("MAE:", mean_absolute_error(y_test, pred))
    print("MSE:", mean_squared_error(y_test, pred))
    print("R2:", r2_score(y_test, pred))

x_line = np.linspace(1, 50, 200).reshape(-1, 1)

plt.scatter(X, y, label="Actual Data")
plt.plot(x_line, linear.predict(x_line), label="Linear")
plt.plot(x_line, polynomial.predict(poly.transform(x_line)),
         label="Polynomial")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()