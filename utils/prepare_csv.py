import pandas as pd
import os

def remove_last_rows_from_csv_recursive(directory: str) -> None:
    """
    Recursively removes the last 5 rows from all CSV files in the specified directory.

    Args:
        directory (str): The directory path where the CSV files are located.

    Returns:
        None
    """

    # Walk through all directories and files in the directory
    for root, dirs, files in os.walk(directory):
        for filename in files:

            if filename.endswith('.csv'):
                file_path = os.path.join(root, filename)
                df = pd.read_csv(file_path)

                if len(df) > 5:
                    df = df[:-5]
                    df.to_csv(file_path, index=False)
                    print(f"Processed {file_path}")

                else:
                    print(f"File {file_path} has less than 5 rows, not modified.")

if __name__ == "__main__":
    directory = 'logs'
    remove_last_rows_from_csv_recursive(directory)
