import pandas as pd
import os

def load_and_describe_data():
    """
    Load and describe the NHANES dataset files in the data directory
    """
    # Use the current directory's parent directory to find data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    
    print(f"Looking for data in: {data_dir}")
    
    # List all CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    datasets = {}
    
    # Load each CSV file and print basic information
    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        print(f"\nLoading {file}...")
        try:
            # Read only first few rows to get structure
            df = pd.read_csv(file_path, nrows=5)
            print(f"Sample of {file}:")
            print(df.head())
            
            # Read full file to get stats
            full_df = pd.read_csv(file_path)
            datasets[file] = full_df
            print(f"Shape: {full_df.shape}")
            print(f"Columns: {list(full_df.columns)}")
            print(f"Data types:\n{full_df.dtypes}")
            print(f"Missing values: {full_df.isnull().sum().sum()}")
            
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")
    
    return datasets

if __name__ == "__main__":
    print("Exploring NHANES dataset...")
    datasets = load_and_describe_data()
    print("\nExploration complete!")