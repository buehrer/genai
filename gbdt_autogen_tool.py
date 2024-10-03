from typing import Annotated, Literal
import os
from dotenv import load_dotenv
from autogen import ConversableAgent
import openai
from openai import OpenAIError
import promptflow
from promptflow.tracing import trace as trace_nabila, start_trace

# Instrument traces
start_trace()



# Load environment variables from the .env file
load_dotenv()

# Get the API key and other configurations from the environment variables
api_key = os.getenv("GPT4O_API_KEY")
model = os.getenv("GPT4O_MODEL")
base_url = os.getenv("GPT4O_BASE")
api_version = os.getenv("GPT4O_VERSION")
api_type = os.getenv("OPENAI_API_TYPE")


if not api_key:
    raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set")

if not api_version:
    raise ValueError("GPT4O_API_VERSION environment variable is not set")


# define train model function 
@trace_nabila

def train_model(dataset: str, target_column: str, delimiter: str) -> str:
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from sklearn.metrics import mean_squared_error, accuracy_score
    import lightgbm as lgb
    from sklearn.preprocessing import StandardScaler

    print("Loading data...")

    # Step 1: Load the data
    data = pd.read_csv(dataset, delimiter=delimiter, low_memory=False)
    print("Data loaded successfully.")
    print("Columns in the dataset:")
    print(data.columns)

    # Check if the target column exists
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset. Available columns: {data.columns}")


    # Step 2: Preprocess the data
    print("Preprocessing data...")

    # Clean the target column
    data[target_column] = pd.to_numeric(data[target_column], errors='coerce')

    # Fill NaN values with a default value, e.g., 0
    data[target_column] = data[target_column].fillna(0)

    # Now convert to int
    y = data[target_column].astype(int)

    # Adjust labels to be in the range [0, num_classes - 1]
    y = y - y.min()

    # Handle categorical columns
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].astype(str).str.split().str[0]
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # Separate features and target
    X = data.drop(target_column, axis=1)

    # Convert all feature columns to float
    X = X.astype(float)

    print("Data preprocessing completed.")
    print("Feature columns:", X.columns)
    print(f"Target column: {target_column}")

    
    # Step 3: Split the data into training and testing sets
    # random_state=42 is used for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    # Step 4: Train the model
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # Determine the number of classes
    num_classes = len(y.unique())

    params = {
        'objective': 'multiclass',  # for multi-class classification
        'num_class': num_classes,   # number of classes
        'metric': 'multi_logloss',  # for multi-class classification
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 1,

    }

    model = lgb.train(params, train_data, num_boost_round=20, valid_sets=[test_data], callbacks=[lgb.early_stopping(stopping_rounds=50)])

    # Step 5: Evaluate the model
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_max = [list(x).index(max(x)) for x in y_pred]  # Get the index of the max probability

    mse = mean_squared_error(y_test, y_pred_max)
    print(f'Mean Squared Error: {mse}')

    # For classification, you can use accuracy_score
    accuracy = accuracy_score(y_test, y_pred_max)
    print(f'Accuracy: {accuracy}')



# Let's first define the assistant agent that suggests tool calls.
assistant = ConversableAgent(
    name="Assistant",
    system_message="You are a helpful AI assistant. "
    "You can help with training machine learning models. "
    "Return 'TERMINATE' when the task is done.",
    llm_config={
        "config_list": [
            {
                "model": model,  # Use the verified model name
                "api_type": api_type,  # Use the verified API type
                "api_key": api_key,  # Use the verified API key
                "base_url": base_url,  # Use the verified base URL
                "api_version": api_version  # Use the verified API version
            }
        ]
    },
)



# The user proxy agent is used for interacting with the assistant agent
# and executes tool calls.
user_proxy = ConversableAgent(
    name="User",
    llm_config=False,
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
)


# Register the tool signature with the assistant agent.
assistant.register_for_llm(name="train_model", description="Train a machine learning model")(train_model)

# Register the tool function with the user proxy agent.
user_proxy.register_for_execution(name="train_model")(train_model)



try:
    chat_result = user_proxy.initiate_chat(assistant, message='Train a model using the dataset at C:/Users/nbabar/test/git-stuff/genai/wine_data.txt to predict quality and delimiter ";"')
    print(chat_result)
except OpenAIError as e:
    print(f"An error occurred: {e}")    



try:
    chat_result = user_proxy.initiate_chat(assistant, message='Train a model using the dataset at C:/Users/nbabar/test/git-stuff/genai/wine_data.txt to predict quality and delimiter ";"')
    print(chat_result)
except OpenAIError as e:
    print(f"An error occurred: {e}")    


#train_model("C:/Users/nbabar/test/git-stuff/genai/diabetes_prediction_dataset.csv", "diabetes", ",")