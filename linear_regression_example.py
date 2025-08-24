# linear_regression_example.py
# Simple Linear Regression example
# Predict house prices based on size

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Data: house size (sq ft) vs price ($)
X = np.array([1000, 1500, 2000, 2500]).reshape(-1,1)  # Features
Y = np.array([150000, 200000, 250000, 300000])       # Labels

# Create model
model = LinearRegression()
model.fit(X, Y)  # Train the model

# Predict price for 1800 sq ft house
predicted_price = model.predict([[1800]])
print("Predicted price for 1800 sq ft house:", predicted_price[0])

# Plot the data and regression line
plt.scatter(X, Y, color='blue')            # Original data
plt.plot(X, model.predict(X), color='red') # Regression line
plt.xlabel("Size (sq ft)")
plt.ylabel("Price ($)")
plt.title("House Price Prediction")
plt.show()
