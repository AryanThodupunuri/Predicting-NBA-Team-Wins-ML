import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics

# Load the training and test data
try:
    data_train = pd.read_csv('NBA_train.csv')
    data_test = pd.read_csv('NBA_test.csv')
except FileNotFoundError:
    print("Error: Data files not found.")
    exit(1)

# Display basic information about the dataset
print("First 5 rows of the training dataset:")
print(data_train.head())

print("\nInformation about the training dataset:")
print(data_train.info())

print("\nSummary statistics of the training dataset:")
print(data_train.describe())

# Create a new column 'PointDiff' by subtracting 'oppPTS' from 'PTS'
data_train['PointDiff'] = data_train['PTS'] - data_train['oppPTS']

# Visualize the relationship between 'PointDiff' and 'W' on the training set
plt.figure(figsize=(15, 10))
sns.scatterplot(x='PointDiff', y='W', data=data_train)
plt.title('Point difference vs. Win totals (Training Set)')
plt.xlabel('PointDiff')
plt.ylabel('W')
plt.savefig('PointDiff_vs_W_Training.png')
plt.show()

# Prepare the data for regression
X = data_train[['PointDiff']]
y = data_train['W']

# Split the data into a training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Display the model's coefficients, intercept, and equation
print("\nModel Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("Regression Equation: W = {:.5f} * PointDiff + {:.5f}".format(model.coef_[0], model.intercept_))

# Evaluate the model on the training set
training_r2 = model.score(X_train, y_train)
print("\nR-squared (training set):", training_r2)

# Predict 'W' values on the test set
y_pred = model.predict(X_test)

# Visualize the results on the training set
plt.figure(figsize=(15, 10))
sns.scatterplot(x='PointDiff', y='W', data=data_train)
plt.plot(X_train, model.predict(X_train), color='red')
plt.title('Point difference vs. Win totals (Training Set)')
plt.xlabel('PointDiff')
plt.ylabel('W')
plt.savefig('PointDiff_vs_W_Training.png')
plt.show()

# Visualize the results on the test set
plt.figure(figsize=(15, 10))
sns.scatterplot(x='PointDiff', y='W', data=data_train)
plt.plot(X_test, y_pred, color='red')
plt.title('Point difference vs. Win totals (Test Set)')
plt.xlabel('PointDiff')
plt.ylabel('W')
plt.savefig('PointDiff_vs_W_Test.png')
plt.show()

# Evaluate the model on the test set (1st evaluation)
test_r2 = model.score(X_test, y_test)
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
cv_scores = cross_val_score(model, X, y, cv=5)
print("\nR-squared (test set):", test_r2)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Cross-validation scores:", cv_scores)
print("Average cross-validation score:", cv_scores.mean())

# Add 'PointDiff' column to the test data, predict 'W' values, and round them
X = data_test[['PointDiff']]
y_pred = model.predict(X)
data_test['W_pred'] = y_pred.round(0).astype(int)

# Display the predicted win totals for the 2022-23 season
plt.figure(figsize=(15, 10))
sns.scatterplot(x='PointDiff', y='W', data=data_test)
plt.plot(X, y_pred, color='red', linewidth=3)
plt.title('Point difference vs. Win totals (2022-23 Season)')
plt.xlabel('PointDiff')
plt.ylabel('W')
plt.savefig('PointDiff_vs_W_2022-23.png')
plt.show()

# Display the actual and predicted win totals for the 2022-23 season
df = data_test[['Team', 'W', 'W_pred']]
plt.figure(figsize=(15, 10))
plt.bar(df['Team'], df['W'], label='Actual W', alpha=0.7)
plt.bar(df['Team'], df['W_pred'], label='Predicted W', alpha=0.7)
plt.xlabel('Team')
plt.ylabel('Win Totals')
plt.title('Actual vs. Predicted Win Totals for the 2022-23 Season')
plt.xticks(rotation=90)  # Rotate x-axis labels for readability
plt.legend()
plt.savefig('Actual_vs_Predicted_W_2022-23.png')
plt.show()

# Evaluate the model on the 2022-23 season (2nd evaluation)
test_r2 = model.score(X, data_test['W'])
mae = metrics.mean_absolute_error(data_test['W'], y_pred)
mse = metrics.mean_squared_error(data_test['W'], y_pred)
rmse = np.sqrt(mse)
print("\nR-squared (test set):", test_r2)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
cv_scores = cross_val_score(model, X, data_test['W'], cv=5)
print("Cross-validation scores:", cv_scores)
print("Average cross-validation score:", cv_scores.mean())