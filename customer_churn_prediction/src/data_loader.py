import pandas as pd
import numpy as np

def generate_dummy_churn_data(num_samples=1000):
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

def load_churn_data(filepath='data/telecom_churn.csv'):
    """
    Loads customer churn data from a specified CSV file.
    If the file does not exist, it generates dummy data.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File not found at {filepath}. Generating dummy data...")
        dummy_df = generate_dummy_churn_data()
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        dummy_df.to_csv(filepath, index=False)
        print(f"Dummy data saved to {filepath}")
        return dummy_df

if __name__ == '__main__':
    import os
    # Example usage
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'data', 'dummy_telecom_churn.csv')
    df = load_churn_data(data_path)
    print(df.head())
    print(f"Number of churned customers: {df['churn'].sum()}")
