import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_churn_data(df, target_column=\'churn\'):
    """
    Preprocesses the customer churn DataFrame.
    - Handles categorical features using one-hot encoding.
    - Scales numerical features.
    - Splits data into training and testing sets.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        
    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Training features (X_train).
            - pd.DataFrame: Testing features (X_test).
            - pd.Series: Training target variable (y_train).
            - pd.Series: Testing target variable (y_test).
            - ColumnTransformer: The fitted preprocessor object.
    """
    print("Starting data preprocessing...")

    # Separate target variable
    X = df.drop(columns=[target_column, \'customer_id\'])
    y = df[target_column]

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=\'object\').columns
    numerical_features = X.select_dtypes(include=[
        \'int64\', \'float64\'
    ]).columns

    # Create a column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (\'num\', StandardScaler(), numerical_features),
            (\'cat\', OneHotEncoder(handle_unknown=\'ignore\'), categorical_features),
        ],
        remainder=\'passthrough\',  # Keep other columns not specified
    )

    # Split data before fitting preprocessor to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit and transform the training data, transform test data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names after one-hot encoding
    cat_feature_names = preprocessor.named_transformers_[
        \'cat\'
    ].get_feature_names_out(categorical_features)
    transformed_feature_names = list(numerical_features) + list(cat_feature_names)

    X_train_final = pd.DataFrame(
        X_train_processed, columns=transformed_feature_names, index=X_train.index
    )
    X_test_final = pd.DataFrame(
        X_test_processed, columns=transformed_feature_names, index=X_test.index
    )

    print("Data preprocessing complete. Data split into training and testing sets.")
    return X_train_final, X_test_final, y_train, y_test, preprocessor


if __name__ == \'__main__\':
    import os
    from data_loader import load_churn_data

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, \'..\', \'data\', \'dummy_telecom_churn.csv\')
    df = load_churn_data(data_path)

    if df is not None:
        X_train, X_test, y_train, y_test, preprocessor = preprocess_churn_data(df)

        print("\nPreprocessing successful!")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
        print("\nSample of preprocessed training data:")
        print(X_train.head())
