import lightgbm as lgb
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder


print("Loading data...")

# Step 1: Load the data
data = pd.read_csv(r'C:\Users\nbabar\test\git-stuff\genai\diabetes_prediction_dataset.csv', low_memory=False)
print("Data loaded successfully.")
print("Columns in the dataset:")
print(data.columns)


# Step 2: Preprocess the data
print("Preprocessing data...")


# Clean the 'diabetes' column
data['diabetes'] = data['diabetes'].astype(str).str.replace(r'[^\d]', '', regex=True)
data['diabetes'] = pd.to_numeric(data['diabetes'], errors='coerce')

# Fill NaN values with a default value, e.g., 0
data['diabetes'] = data['diabetes'].fillna(0)

# Now convert to int
y = data['diabetes'].astype(int)


# Handle the 'gender' column
data['gender'] = data['gender'].astype(str).str.split().str[0]

# Handle the 'smoking_history' column
data['smoking_history'] = data['smoking_history'].astype(str).str.split().str[0]

# Encode categorical variables
le = LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])
data['smoking_history'] = le.fit_transform(data['smoking_history'])



# Separate features and target
X = data.drop('diabetes', axis=1)

# Convert all feature columns to float
X = X.astype(float)

print("Data preprocessing completed.")
print("Feature columns:", X.columns)
print("Target column: diabetes")

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


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
model = lgb.train(params, train_data, num_boost_round=20, valid_sets=[test_data], callbacks=[lgb.early_stopping(stopping_rounds=5)]) 


print("Saving model...")

# save model to file
model.save_model("model.txt")


print("Starting predicting...")

# predict 
print("Starting predicting...")
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Accuracy: {accuracy}')