import lightgbm as lgb
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score


print("Loading data...")

# Step 1: Load the data
print("Loading data...")
data = pd.read_csv(r'C:\Users\nbabar\test\git-stuff\genai\diabetes_prediction_dataset.csv', low_memory=False)
print("Data loaded successfully.")
print("Columns in the dataset:")
print(data.columns)



# Step 2: Preprocess the data

# Handle missing values if any
data = data.fillna(data.mean())


# Encode categorical variables
le = LabelEncoder()
categorical_columns = ['gender', 'smoking_history']
for col in categorical_columns:
    data[col] = le.fit_transform(data[col].astype(str))


# Prepare features and target
X = data.drop(columns=['diabetes'])
y = data['diabetes']

# Convert all columns to float
X = X.astype(float)

print("Data preprocessing completed.")
print("Feature columns:", X.columns)
print("Target column: diabetes")

# Step 3: Split the data into training and testing sets
train, test = train_test_split(data, test_size=0.2)

# y is the column to predict
X_train = train.drop(columns=['diabetes'])
X_test = test.drop(columns=['diabetes'])
y_train = train['diabetes']
y_test = test['diabetes']


# Step 4: Train the model
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'binary',  # for binary classification
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

print("starting training...")
model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[test_data], callbacks=[lgb.early_stopping(stopping_rounds=5)]) 


print("Saving model...")

# save model to file
model.save_model("model.txt")


print("Starting predicting...")

# predict 
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# evaluate
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]  # Convert probabilities to binary predictions
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Accuracy: {accuracy}')