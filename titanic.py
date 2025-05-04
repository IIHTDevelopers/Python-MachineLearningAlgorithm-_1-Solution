import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib


# Function 1: Load, clean, and encode Titanic dataset
def load_and_prepare_data(path="titanic.csv"):
    df = pd.read_csv(path)
    df['sex'] = df['sex'].fillna('unknown')
    df['embarked'] = df['embarked'].fillna('S')
    df['age'] = df['age'].fillna(df['age'].median())
    df['fare'] = df['fare'].fillna(df['fare'].median())

    le_sex = LabelEncoder()
    df['sex'] = le_sex.fit_transform(df['sex'])

    le_embarked = LabelEncoder()
    df['embarked'] = le_embarked.fit_transform(df['embarked'])

    print("✅ Data loaded, cleaned, and encoded.")
    print(print(df.columns.tolist()))
    return df


# Function 2: Perform EDA (Only Max and Std for Fare)
def explore_data(df):
    max_fare = df['fare'].max()
    std_fare = df['fare'].std()
    print(f"\n💰 Fare - Max: {max_fare}, Std Dev: {std_fare:.2f}")
    return round(max_fare, 4), round(std_fare, 2)

# Function 3: Sigmoid activation for a single value
def sigmoid_demo():
    z = 0
    sigmoid = 1 / (1 + np.exp(-z))
    print(f"\n🧠 Sigmoid(0) = {sigmoid:.4f}")
    return sigmoid

# Function 4: Custom cost function (Log Loss)
def cost_function(y_true, y_pred_prob):
    epsilon = 1e-15
    y_pred_prob = np.clip(y_pred_prob, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred_prob) + (1 - y_true) * np.log(1 - y_pred_prob))


# Function 5: Train and evaluate model
def train_and_evaluate(X_train, y_train, X_test, y_test, path="titanic_log_model.pkl"):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    joblib.dump(model, path)
    print(f"\n✅ Model trained and saved to '{path}'")

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    cost = cost_function(y_test.values, y_pred_prob)

    print(f"\n🎯 Log Loss (Custom Cost): {cost:.4f}")
    print("🔍 Sample Predictions:", y_pred[:10])


# --------- Main Logic ---------
if __name__ == "__main__":
    df = load_and_prepare_data("titanic.csv")

    explore_data(df)
    sigmoid_demo()

    features = ['pclass', 'sex', 'age', 'fare', 'embarked']
    X = df[features]
    y = df['survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_and_evaluate(X_train, y_train, X_test, y_test)
