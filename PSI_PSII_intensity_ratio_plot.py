# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:52:36 2024

@author: Schre082

Code that takes CSV files with 2 columns, z-stack number 
and the ratio of PSI average intensity/ PSII average intensity
and produces a figure that includes all CSV files in the folder.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def process_and_plot_ratio(csv_directory_path, output_directory_path, output_filename):
    """
    Processes the CSV files in the provided directory, plots the PSI/PSII ratio vs Z values,
    and saves the plot as an image.

    Parameters:
        csv_directory_path (str): Path to the folder containing CSV files.
        output_directory_path (str): Path to the folder where the output plot will be saved.
        output_filename (str): Name of the output plot file.

    Returns:
        None
    """
    # Create the folder to save the output plot if it doesn't exist
    os.makedirs(output_directory_path, exist_ok=True)

    # Full path to the output file
    output_file_path = os.path.join(output_directory_path, output_filename)

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Loop through each CSV file in the folder
    for filename in os.listdir(csv_directory_path):
        if filename.endswith('.csv'):
            # Full path to the CSV file
            file_path = os.path.join(csv_directory_path, filename)
            
            # Read the CSV file
            data = pd.read_csv(file_path)
            
            # Ensure the CSV has the expected structure
            if data.shape[1] != 2:
                print(f"Skipping {filename}: unexpected number of columns.")
                continue
            
            # Extract Z values and PSI/PSII ratios
            z_values = data.iloc[:, 0]  # First column (Z values)
            ratio_values = data.iloc[:, 1]  # Second column (PSI/PSII ratios)
            
            # Plot the data
            plt.plot(z_values, ratio_values, label=filename.split('.')[0])

    # Add plot labels, title, and legend
    plt.xlabel('Z Value (one z-slice = 0.64 μm)', fontsize=12)
    plt.ylabel('PSI/PSII Ratio', fontsize=12)
    plt.title('PSI/PSII Ratio vs Z-value', fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True)

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_file_path)
    print(f"Plot saved as: {output_file_path}")
    plt.close()

if __name__ == "__main__":
    # Define the folder containing the CSV files
    csv_directory_path = r'path to intensity ratio csv file folder'

    # Define the folder to save the output plot
    output_directory_path = r'directory to output folder'

    # Name of the output plot file
    output_filename = "PSI_PSII_ratio_plot.png"

    # Call the function to process the CSV files and generate the plot
    process_and_plot_ratio(csv_directory_path, output_directory_path, output_filename)

