import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import joblib

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    # Separate the target variable
    y = df['Disease']
    X = df.drop('Disease', axis=1)

    # Encode categorical variables in features
    label_encoders = {}
    for column in X.select_dtypes(include=["object"]).columns:
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column])

    # Encode the target variable
    label_encoders['Disease'] = LabelEncoder()
    y = label_encoders['Disease'].fit_transform(y)

    # Normalize numerical variables in features
    scaler = StandardScaler()
    X[X.columns] = scaler.fit_transform(X[X.columns])

    # Combine the features and target back into one DataFrame
    df_processed = pd.concat([X, pd.Series(y, name='Disease')], axis=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print(f"Data preprocessed and saved to {output_path}")

if __name__ == "__main__":
    preprocess_data('data/raw/synthetic_data.csv', 'data/processed/preprocessed_data.csv')
