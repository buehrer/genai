import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Step 1: Load the data
data = pd.read_csv('C:\Users\nbabar\test\git-stuff\genai\diabetes_prediction_dataset.csv')

# Step 2: Preprocess the data
# Assuming the target column is named 'target' and all other columns are features
X = data.drop(columns=['target'])
y = data['target']

# Handle missing values if any
X.fillna(X.mean(), inplace=True)

# Encode categorical variables if any
X = pd.get_dummies(X)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'regression',  # Change to 'binary' or 'multiclass' for classification
    'metric': 'rmse',  # Change to 'binary_logloss' or 'multi_logloss' for classification
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100, early_stopping_rounds=10)

# Step 5: Evaluate the model
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# For classification, you can use accuracy_score
# accuracy = accuracy_score(y_test, (y_pred > 0.5).astype(int))
# print(f'Accuracy: {accuracy}')