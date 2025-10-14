# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('PROJ A DF.csv')  # Update with your actual dataset filename
print("Dataset loaded successfully.")

from sklearn.impute import SimpleImputer

# Separate numeric and categorical columns BEFORE one-hot encoding
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# 1. Impute numeric columns
num_imputer = SimpleImputer(strategy='median')
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

# 2. Impute categorical columns (if any)
if len(cat_cols) > 0:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# 3. Now do one-hot encoding for categorical columns
if len(cat_cols) > 0:
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# After this, no NaN values should remain
print("Any NaNs remaining?", df.isnull().sum().sum())


print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Visualize distributions of numeric variables
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols].hist(figsize=(15,12), bins=30)
plt.suptitle("Histograms of Numeric Features")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Scatter plot for median_income vs median_house_value
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='median_income', y='median_house_value')
plt.title("Median Income vs Median House Value")
plt.show()


# Handle missing values: fill numeric missing values with median
df.fillna(df.median(), inplace=True)

# Identify categorical columns (object dtype)
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical columns detected:", cat_cols)

# One-hot encode categorical variables if any
if len(cat_cols) > 0:
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print("Categorical columns converted to numeric using one-hot encoding.")

# Confirm no missing values remain
print("\nMissing values after imputation:")
print(df.isnull().sum())

X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

print("\nData types in features (should all be numeric):")
print(X.dtypes.value_counts())

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

X_train_simple = X_train[['median_income']]
X_test_simple = X_test[['median_income']]

model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train)
y_pred_simple = model_simple.predict(X_test_simple)

print("\nSimple Linear Regression Results:")
print(f"Intercept: {model_simple.intercept_:.2f}")
print(f"Coefficient for median_income: {model_simple.coef_[0]:.2f}")

mse_simple = mean_squared_error(y_test, y_pred_simple)
rmse_simple = np.sqrt(mse_simple)
r2_simple = r2_score(y_test, y_pred_simple)

print(f"MSE: {mse_simple:.2f}")
print(f"RMSE: {rmse_simple:.2f}")
print(f"R-squared: {r2_simple:.4f}")

# Visualize actual vs predicted for simple regression
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_simple, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Median House Value')
plt.ylabel('Predicted Median House Value')
plt.title('Simple Linear Regression: Actual vs Predicted')
plt.show()

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

y_pred_multi = model_multi.predict(X_test)

print("\nMultiple Linear Regression Results:")
print(f"Intercept: {model_multi.intercept_:.2f}")

# Display coefficients with feature names
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model_multi.coef_})
print(coefficients.sort_values(by='Coefficient', ascending=False))

mse_multi = mean_squared_error(y_test, y_pred_multi)
rmse_multi = np.sqrt(mse_multi)
r2_multi = r2_score(y_test, y_pred_multi)

print(f"MSE: {mse_multi:.2f}")
print(f"RMSE: {rmse_multi:.2f}")
print(f"R-squared: {r2_multi:.4f}")

# Visualize actual vs predicted for multiple regression
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_multi, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Median House Value')
plt.ylabel('Predicted Median House Value')
plt.title('Multiple Linear Regression: Actual vs Predicted')
plt.show()

print("\nSummary:")
print(f"- Simple Linear Regression R2: {r2_simple:.4f}")
print(f"- Multiple Linear Regression R2: {r2_multi:.4f}")