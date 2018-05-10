from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the performance of a trained machine learning model.
    
    Args:
        model (object): The trained machine learning model.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): True target variable for testing.
        
    Returns:
        dict: A dictionary containing various evaluation metrics.
    """
    print("Evaluating model performance...")
    
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }
    
    # For binary classification, also calculate ROC AUC if model supports predict_proba
    if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        
    print("Model evaluation complete.")
    return metrics

def print_metrics(metrics):
    """
    Prints the evaluation metrics in a readable format.
    """
    print("\n--- Model Evaluation Metrics ---")
    for metric, value in metrics.items():
        if metric == "confusion_matrix":
            print(f"  {metric.replace("_", " ").title()}:\n{np.array(value)}")
        elif metric == "classification_report":
            print(f"  {metric.replace("_", " ").title()}:\n")
            for class_label, class_metrics in value.items():
                if isinstance(class_metrics, dict):
                    print(f"    Class {class_label}:")
                    for sub_metric, sub_value in class_metrics.items():
                        print(f"      {sub_metric.replace("_", " ").title()}: {sub_value:.4f}")
                else:
                    print(f"    {class_label.replace("_", " ").title()}: {class_metrics:.4f}")
        else:
            print(f"  {metric.replace("_", " ").title()}: {value:.4f}")
    print("------------------------------")

if __name__ == "__main__":
    import os
    import pandas as pd
    import joblib
    from data_loader import generate_dummy_churn_data
    from preprocessing import preprocess_churn_data
    from model import train_model, save_model

    # Generate dummy data for testing
    dummy_df = generate_dummy_churn_data(num_samples=500)
    X_train, X_test, y_train, y_test, _ = preprocess_churn_data(dummy_df)

    if X_train is not None:
        # Train a Random Forest model
        rf_model = train_model(X_train, y_train, model_name="random_forest")
        
        # Evaluate the model
        metrics = evaluate_model(rf_model, X_test, y_test)
        print_metrics(metrics)

        print("\nModel evaluation example finished!")
