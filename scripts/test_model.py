import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_new_data(new_data, label_encoders, scaler):
    df = pd.DataFrame(new_data, index=[0])

    # Encode categorical variables
    for column, le in label_encoders.items():
        if column in df.columns:
            try:
                df[column] = le.transform(df[column])
            except ValueError:
                # Handle unseen labels
                df[column] = df[column].apply(lambda x: le.transform(['Unknown'])[0] if x not in le.classes_ else le.transform([x])[0])
    
    # Normalize numerical variables
    df[df.columns] = scaler.transform(df[df.columns])

    return df

def predict_disease(new_data):
    # Load encoders and scaler
    label_encoders = joblib.load('models/label_encoders.pkl')
    scaler = joblib.load('models/scaler.pkl')
    model = joblib.load('models/disease_prediction_model.pkl')

    # Add 'Unknown' class to all label encoders to handle unseen values
    for column, le in label_encoders.items():
        if 'Unknown' not in le.classes_:
            le.classes_ = np.append(le.classes_, 'Unknown')

    # Preprocess new data
    df_processed = preprocess_new_data(new_data, label_encoders, scaler)

    # Make prediction
    probabilities = model.predict_proba(df_processed)[0]

    # Map probabilities to disease names
    diseases = label_encoders['Disease'].classes_
    prediction = {disease: prob for disease, prob in zip(diseases, probabilities)}

    return prediction

if __name__ == "__main__":
    # Example new patient data
    new_patient_data = {
        "Age": 18,
        "Gender": "Male",
        "Blood_Pressure": 200,
        "Cholesterol_Level": 200,
        "Blood_Sugar_Level": 500,
        "BMI": 40.9,
        "Smoking_Status": "Current",
        "Physical_Activity_Level": "Low",
        "Family_History": "Yes",
        "Medical_History": "None"
    }

    prediction = predict_disease(new_patient_data)
    print("Disease prediction probabilities:", prediction)
