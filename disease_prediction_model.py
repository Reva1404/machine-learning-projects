# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# 1. Load dataset
df = pd.read_csv('PROJ C DF.csv')  # Replace with your dataset filename
print(df.columns)

# 2. Basic Exploration
print(df.head())
print(df.info())
print(df['disease'].value_counts())  # Target variable distribution

# 3. Exploratory Data Analysis (EDA)

# Plot distribution of target variable
sns.countplot(x='disease', data=df)
plt.title('Distribution of Disease')
plt.show()

# Summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Visualize relationships with target
for col in df.columns:
    if col != 'disease':
        plt.figure(figsize=(6,4))
        if df[col].dtype == 'object':
            sns.countplot(x=col, hue='disease', data=df)
        else:
            sns.boxplot(x='disease', y=col, data=df)
        plt.title(f'{col} vs Disease')
        plt.show()

# 4. Data Preprocessing

# Handle missing values (example: drop or impute)
df = df.dropna()  

# Encode categorical variables if any (example)
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Separate features and target
X = df.drop('disease', axis=1)
y = df['disease']

# Scale numeric features 
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 5. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Model Development and Hyperparameter Tuning

models = {
    'LogisticRegression': LogisticRegression(max_iter=500),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

params = {
    'LogisticRegression': {'C': [0.1, 1, 10]},
    'DecisionTree': {'max_depth': [3, 5, 10, None]},
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
}

best_models = {}
for model_name in models:
    print(f"Training and tuning {model_name}...")
    clf = GridSearchCV(models[model_name], param_grid=params[model_name], cv=5, scoring='f1', n_jobs=-1)
    clf.fit(X_train, y_train)
    best_models[model_name] = clf.best_estimator_
    print(f"Best params for {model_name}: {clf.best_params_}")

# 7. Model Evaluation
for model_name, model in best_models.items():
    print(f"\nEvaluating {model_name}...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    if y_proba is not None:
        print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# 8. Insights 
dt_model = best_models['DecisionTree']
if hasattr(dt_model, 'feature_importances_'):
    feat_importances = pd.Series(dt_model.feature_importances_, index=X.columns)
    feat_importances = feat_importances.sort_values(ascending=False)
    plt.figure(figsize=(10,6))
    sns.barplot(x=feat_importances.values, y=feat_importances.index)
    plt.title('Feature Importance - Decision Tree')
    plt.show()
