import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(data_path, model_path, report_path):
    df = pd.read_csv(data_path)
    X = df.drop("Disease", axis=1)
    y = df["Disease"]

    model = joblib.load(model_path)

    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, target_names=['Heart Disease', 'Diabetes', 'Cancer', 'COPD'])
    cm = confusion_matrix(y, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{cm}")

    # Save the evaluation report
    os.makedirs(report_path, exist_ok=True)
    with open(os.path.join(report_path, 'classification_report.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")

    # Plot and save the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Heart Disease', 'Diabetes', 'Cancer', 'COPD'], yticklabels=['Heart Disease', 'Diabetes', 'Cancer', 'COPD'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(report_path, 'confusion_matrix.png'))
    plt.show()

if __name__ == "__main__":
    evaluate_model('data/processed/preprocessed_data.csv', 'models/disease_prediction_model.pkl', 'reports')
