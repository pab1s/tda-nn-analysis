import os
import pandas as pd
from typing import Tuple

def generate_latex_tables(directory) -> Tuple[str, str]:
    """
    Generate LaTeX tables from CSV files in the specified directory.

    Args:
        directory (str): The directory path where the CSV files are located.

    Returns:
        tuple: A tuple containing the LaTeX code for the tables generated for models optimized with Adam and SGD.
    """

    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    data = {
        'Model': [],
        'Optimizer': [],
        'Batch Size': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1_score': []
    }

    # Process each CSV file
    for csv_file in csv_files:

        # Extract details from the filename
        parts = csv_file.split('_')
        model_name = '_'.join(parts[2:parts.index('CarDataset')])
        optimizer = parts[parts.index('CarDataset') + 1]
        batch_size = parts[parts.index('CarDataset') + 2]

        df = pd.read_csv(os.path.join(directory, csv_file))

        last_row = df.iloc[-1]

        data['Model'].append(model_name)
        data['Optimizer'].append(optimizer)
        data['Batch Size'].append(batch_size)
        data['val_accuracy'].append(last_row['val_accuracy'])
        data['val_precision'].append(last_row['val_precision'])
        data['val_recall'].append(last_row['val_recall'])
        data['val_f1_score'].append(last_row['val_f1_score'])

    results_df = pd.DataFrame(data)

    adam_df = results_df[results_df['Optimizer'] == 'Adam']
    sgd_df = results_df[results_df['Optimizer'] == 'SGD']

    adam_latex = adam_df.to_latex(index=False, float_format="%.4f",
                                  caption="Validation Metrics for Models Optimized with Adam",
                                  label="tab:adam_metrics",
                                  column_format='|l|l|l|l|l|l|l|')
    sgd_latex = sgd_df.to_latex(index=False, float_format="%.4f",
                                caption="Validation Metrics for Models Optimized with SGD",
                                label="tab:sgd_metrics",
                                column_format='|l|l|l|l|l|l|l|')

    return adam_latex, sgd_latex

def generate_latex_transforms_tables(directory) -> str:
    """
    Generate LaTeX tables from transforms files in the specified directory.

    Args:
        directory (str): The directory path where the CSV files are located.

    Returns:
        str: The LaTeX table as a string.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
    """
    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    data = {
        'Transformation': [],
        'Batch Size': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1_score': []
    }
    
    # Process each CSV file
    for csv_file in csv_files:

        # Extract details from the filename
        parts = csv_file.split('_')

        # Assumes transformation is always the fifth element
        transformation = parts[4]

        # Assumes batch size is always the seventh element
        batch_size = parts[6]
        
        df = pd.read_csv(os.path.join(directory, csv_file))
        
        # Extract the last row's validation metrics
        last_row = df.iloc[-1]
        
        data['Transformation'].append(transformation)
        data['Batch Size'].append(batch_size)
        data['val_accuracy'].append(last_row['val_accuracy'])
        data['val_precision'].append(last_row['val_precision'])
        data['val_recall'].append(last_row['val_recall'])
        data['val_f1_score'].append(last_row['val_f1_score'])
    
    results_df = pd.DataFrame(data)
    
    # Generate LaTeX tables
    latex_table = results_df.to_latex(index=False, float_format="%.4f", 
                                      caption="Validation Metrics for Different Transformations", 
                                      label="tab:transformation_metrics", 
                                      column_format='|c|c|c|c|c|c|')
    
    return latex_table


if __name__ == "__main__":
    directory = 'logs/Modelo'
    adam_table, sgd_table = generate_latex_tables(directory)
    print("Adam Optimizer Table:\n", adam_table)
    print("\nSGD Optimizer Table:\n", sgd_table)
