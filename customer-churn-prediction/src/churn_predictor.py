import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def generate_dummy_data(num_samples=1000):
    """
    Generates dummy customer churn data for demonstration purposes.
    """
    np.random.seed(42)
    data = {
        'customer_id': range(num_samples),
        'age': np.random.randint(18, 70, num_samples),
        'gender': np.random.choice(['Male', 'Female'], num_samples),
        'monthly_charges': np.random.uniform(20, 100, num_samples),
        'total_charges': np.random.uniform(50, 5000, num_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], num_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], num_samples),
        'tech_support': np.random.choice(['Yes', 'No'], num_samples),
        'churn': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]) # 30% churn rate
    }
    df = pd.DataFrame(data)
    # Introduce some correlation
    df.loc[df['contract_type'] == 'Month-to-month', 'churn'] = np.random.choice([0, 1], df[df['contract_type'] == 'Month-to-month'].shape[0], p=[0.5, 0.5])
    df.loc[df['monthly_charges'] > 80, 'churn'] = np.random.choice([0, 1], df[df['monthly_charges'] > 80].shape[0], p=[0.4, 0.6])
    return df

def preprocess_data(df):
    """
    Preprocesses the dummy data for model training.
    - Handles categorical features using one-hot encoding.
    - Fills missing values (if any, though not expected in dummy data).
    """
    print("Preprocessing data...")
    # Convert categorical features to numerical using one-hot encoding
    df = pd.get_dummies(df, columns=['gender', 'contract_type', 'internet_service', 'tech_support'], drop_first=True)
    return df

def train_and_evaluate_model(df):
    """
    Trains a RandomForestClassifier model and evaluates its performance.
    """
    print("Splitting data into training and testing sets...")
    # Define features (X) and target (y)
    X = df.drop(['customer_id', 'churn'], axis=1)
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training RandomForestClassifier model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    print("Evaluating model performance...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)
    return model, accuracy, report

if __name__ == "__main__":
    print("Starting customer churn prediction pipeline...")

    # 1. Generate dummy data
    raw_data = generate_dummy_data(num_samples=1500)
    print(f"Generated {raw_data.shape[0]} samples of dummy data.")

    # 2. Preprocess data
    processed_data = preprocess_data(raw_data)
    print(f"Data after preprocessing: {processed_data.shape[0]} rows, {processed_data.shape[1]} columns.")

    # 3. Train and evaluate model
    model, accuracy, report = train_and_evaluate_model(processed_data)

    print("\nCustomer churn prediction pipeline finished successfully!")
