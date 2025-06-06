import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

# Function 1: Load and preprocess the dataset
def load_and_preprocess(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    df = df.dropna()
    print("✅ Data loaded and cleaned.")
    print(print(df.columns.tolist()))
    return df

# Function 2: Show mean of displacement and min of horsepower
def show_key_stats(df):
    displacement_mean = df['displacement'].mean()
    horsepower_min = df['horsepower'].min()
    print(f"\n📏 Mean Displacement: {displacement_mean:.2f}")
    print(f"⚙️  Minimum Horsepower: {horsepower_min}")

# Function 3: Prepare data for training
def prepare_data(df, features, target):
    X = df[features]
    y = df[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print("\n🧪 Data prepared and split.")
    return X_train, X_test, y_train, y_test, scaler

# Function 4: Train the model and save it
def train_and_save_model(X_train, y_train, model_path="linear_model.pkl"):
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print(f"\n✅ Model trained and saved to '{model_path}'")
    return model

# Function 5: Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"\n🎯 Mean Squared Error: {mse:.4f}")
    print("🔍 Sample Predictions:", y_pred[:10])

# ---- MAIN SCRIPT ----
if __name__ == "__main__":
    features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model-year']
    target = 'mpg'

    df = load_and_preprocess("auto-mpg.csv")
    show_key_stats(df)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, features, target)
    model = train_and_save_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
