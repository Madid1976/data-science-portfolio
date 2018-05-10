from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import joblib

def train_model(X_train, y_train, model_name=\"random_forest\", **kwargs):
    """
    Trains a machine learning model for churn prediction.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        model_name (str): Name of the model to train. Options: \"random_forest\", \"logistic_regression\", \"gradient_boosting\", \"decision_tree\".
        **kwargs: Additional arguments for the model constructor.
        
    Returns:
        object: The trained model.
    """
    print(f"Training {model_name} model...")
    
    if model_name == \"random_forest\":
        model = RandomForestClassifier(random_state=42, **kwargs)
    elif model_name == \"logistic_regression\":
        model = LogisticRegression(random_state=42, solver=\"liblinear\", **kwargs)
    elif model_name == \"gradient_boosting\":
        model = GradientBoostingClassifier(random_state=42, **kwargs)
    elif model_name == \"decision_tree\":
        model = DecisionTreeClassifier(random_state=42, **kwargs)
    elif model_name == \"svm\":
        model = SVC(random_state=42, probability=True, **kwargs) # probability=True for ROC AUC
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
        
    model.fit(X_train, y_train)
    print(f"{model_name} model training complete.")
    return model

def save_model(model, filepath):
    """
    Saves the trained model to a specified file using joblib.
    """
    try:
        joblib.dump(model, filepath)
        print(f"Model saved successfully to {filepath}")
    except Exception as e:
        print(f"Error saving model to {filepath}: {e}")

def load_model(filepath):
    """
    Loads a trained model from a specified file using joblib.
    """
    try:
        model = joblib.load(filepath)
        print(f"Model loaded successfully from {filepath}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading model from {filepath}: {e}")
        return None

if __name__ == \"__main__\":
    import os
    import pandas as pd
    from data_loader import generate_dummy_churn_data
    from preprocessing import preprocess_churn_data

    # Generate dummy data for testing
    dummy_df = generate_dummy_churn_data(num_samples=500)
    X_train, _, y_train, _, _ = preprocess_churn_data(dummy_df)

    if X_train is not None:
        # Train a Random Forest model
        rf_model = train_model(X_train, y_train, model_name=\"random_forest\")
        model_path = os.path.join(os.path.dirname(__file__), \"..\", \"models\", \"random_forest_churn_model.joblib\")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        save_model(rf_model, model_path)

        # Load the model back
        loaded_model = load_model(model_path)
        if loaded_model:
            print("Model training, saving, and loading test successful!")
