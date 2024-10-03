import argparse
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import promptflow
from promptflow.tracing import trace as trace_nabila, start_trace
import matplotlib.pyplot as plt  


# Instrument traces
start_trace()


def infer_delimiter(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline()
        if ',' in first_line:
            return ','
        elif ';' in first_line:
            return ';'
        else:
            raise ValueError("Unknown delimiter. Please specify the delimiter using the --delimiter argument.")

@trace_nabila
def train_model(dataset, target_column, delimiter):
    print("Loading data...")

    # Step 1: Load the data
    data = pd.read_csv(dataset, delimiter=delimiter, low_memory=False)
    print("Data loaded successfully.")
    print("Columns in the dataset:")
    # this next line 
    print(data.columns)

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


    # prepare configuration for cross validation test harness
    seed = 7
    # prepare models
    from sklearn import svm
    from sklearn.svm import SVC
    from sklearn.linear_model import SGDClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    models = []
    models.append(('SupportVectorClassifier', SVC()))
    models.append(('StochasticGradientDecentC', SGDClassifier()))
    models.append(('RandomForestClassifier', RandomForestClassifier()))
    models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
    models.append(('GaussianNB', GaussianNB()))
    models.append(('KNeighborsClassifier', KNeighborsClassifier()))
    models.append(('AdaBoostClassifier', AdaBoostClassifier()))
    models.append(('LogisticRegression', LogisticRegression()))
    # evaluate each model in turn
    results = []
    names = []
    import sklearn as sk
    from sklearn import model_selection
    scoring = 'accuracy'
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    # boxplot algorithm comparison
    # import the library
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

# train_model("C:/Users/buehrer/OneDrive - Microsoft/projects/genai/wine_data.txt", "quality", ";")
train_model("C:/Users/nbabar/test/git-stuff/genai/wine_data.txt", "quality", ";")
# train_model("C:/Users/nbabar/test/git-stuff/genai/diabetes_prediction_dataset.csv", "diabetes", ",")
