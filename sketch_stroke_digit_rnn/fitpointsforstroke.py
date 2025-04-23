import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

x_data = np.array([0, 9, 12, 9, 12, 4, 12], dtype=float)
y_data = np.array([0, 16, 25, 17, 48, 15, 32], dtype=float)

# Define the constrained exponential model
def exp_through_origin(x, a, b):
    return a * (np.exp(b * x) - 1)

# Fit the curve
params, _ = curve_fit(exp_through_origin, x_data, y_data, p0=(1, 0.1))
a, b = params

# Predict y values
y_pred = exp_through_origin(x_data, a, b)

# Calculate R²
r2 = r2_score(y_data, y_pred)

# Print results
print(f"Fitted equation: y = {a:.4f} * (e^({b:.4f}x) - 1)")
print(f"R² = {r2:.4f}")

# Plot data and fit
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = exp_through_origin(x_fit, a, b)

plt.scatter(x_data, y_data, label='Data')
plt.plot(x_fit, y_fit, color='red', label='Fitted Exp Curve')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Exponential Fit Through Origin')
plt.show()