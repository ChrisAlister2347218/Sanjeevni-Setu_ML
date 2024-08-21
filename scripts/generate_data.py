import pandas as pd
import numpy as np
import os

def generate_synthetic_data(n_samples=1000, seed=42):
    np.random.seed(seed)
    data = {
        "Age": np.random.randint(20, 80, n_samples),
        "Gender": np.random.choice(["Male", "Female"], n_samples),
        "Blood_Pressure": np.random.randint(90, 180, n_samples),
        "Cholesterol_Level": np.random.randint(150, 300, n_samples),
        "Blood_Sugar_Level": np.random.randint(70, 200, n_samples),
        "BMI": np.random.uniform(18, 40, n_samples),
        "Smoking_Status": np.random.choice(["Never", "Former", "Current"], n_samples),
        "Physical_Activity_Level": np.random.choice(["Low", "Moderate", "High"], n_samples),
        "Family_History": np.random.choice(["Yes", "No"], n_samples),
        "Medical_History": np.random.choice(["None", "Heart Disease", "Diabetes", "Cancer", "COPD"], n_samples),
        "Disease": np.random.choice(["Heart Disease", "Diabetes", "Cancer", "COPD"], n_samples)
    }

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    os.makedirs('data/raw', exist_ok=True)
    df = generate_synthetic_data()
    df.to_csv('data/raw/synthetic_data.csv', index=False)
    print("Synthetic data generated and saved to data/raw/synthetic_data.csv")
