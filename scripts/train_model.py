import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_model(input_path, model_output_path):
    df = pd.read_csv(input_path)
    X = df.drop("Disease", axis=1)
    y = df["Disease"]

    # Verify target variable is not continuous
    print("Target variable type:", y.dtype)
    print("Unique values in target variable:", y.unique())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(multi_class='ovr', max_iter=1000)
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Model trained and saved to {model_output_path}")

if __name__ == "__main__":
    train_model('data/processed/preprocessed_data.csv', 'models/disease_prediction_model.pkl')
