# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:44:32 2024

@author: Schre082

Code used to produce a figure in which the average of fPSI vs depth graphs
of different measurements on the same leaf is shown with standard deviation
input should be 2 folders each containing multiple measurements of one leaf
with data about the z-slice of the z-stack (multiphoton measurements), and the 
amplitude of PSI (photosystem I) as determined by fitting the data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


def process_and_plot_folders(csv_directories, plots_directory_path):
    """
    Processes CSV files from two input folders, calculates fPSI, truncates to shared depth,
    and generates a combined plot with the averages and shaded error bars for each folder.

    Parameters:
    csv_directories (list of str): List of paths to the directories containing CSV files.
    plots_directory_path (str): Path to save the generated plot.
    """
    # Create the plots directory if it doesn't exist
    os.makedirs(plots_directory_path, exist_ok=True)

    # Define the constant C (this depends on microscope detectors)
    C = 6.58

    # Initialize lists to store average data and labels
    averages_with_error = []
    folder_labels = []

    for folder_path in csv_directories:
        # Initialize a dictionary to store data for plotting
        plot_data = {}

        # List to store the minimum depth for all files in this folder
        shared_max_depths = []

        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                # Full path to the current CSV file
                file_path = os.path.join(folder_path, filename)

                # Read the CSV file
                data = pd.read_csv(file_path)

                # Ensure tau1, tau2, tau3, A, B, and C columns are numeric
                data[['tau1', 'tau2', 'tau3', 'A', 'B', 'C']] = data[['tau1', 'tau2', 'tau3', 'A', 'B', 'C']].apply(pd.to_numeric, errors='coerce')

                # Add amplitude to a1 if tau is below 300 (PSI)
                data['a1'] = 0.0
                for i, row in data.iterrows():
                    a1_value = 0.0
                    if row['tau1'] < 300:
                        a1_value += row['A']
                    if row['tau2'] < 300:
                        a1_value += row['B']
                    if row['tau3'] < 300:
                        a1_value += row['C']
                    data.at[i, 'a1'] = a1_value

                # Calculate fPSI for each row using the calculated a1
                data['fPSI'] = 1 / ((C / data['a1']) - C + 1)

                # Find the maximum slice number in the current dataset
                max_slice = data['Slice'].max()

                # Convert slices to depth (μm)
                data['Depth'] = (max_slice - data['Slice']) * 0.64

                # Store the maximum depth for this dataset
                shared_max_depths.append(data['Depth'].max())

                # Store data in the dictionary for plotting
                plot_data[filename] = {
                    'Depth': data['Depth'],
                    'fPSI': data['fPSI'],
                }

        # Determine the maximum depth shared across all files in this folder
        max_shared_depth = min(shared_max_depths)

        # Initialize a DataFrame to calculate the average fPSI for this folder
        avg_data = pd.DataFrame()

        for filename, values in plot_data.items():
            depth = values['Depth']
            fPSI = values['fPSI']

            # Truncate data to the maximum shared depth
            truncated_data = pd.DataFrame({'Depth': depth, 'fPSI': fPSI})
            truncated_data = truncated_data[truncated_data['Depth'] <= max_shared_depth]

            # Reset indices for consistent alignment
            truncated_data.reset_index(drop=True, inplace=True)

            # Add the truncated data to avg_data
            if avg_data.empty:
                avg_data['Depth'] = truncated_data['Depth']
            avg_data[filename] = truncated_data['fPSI']

        # Calculate the average and standard deviation of fPSI across datasets for this folder
        avg_data['Mean_fPSI'] = avg_data.iloc[:, 1:].mean(axis=1)
        avg_data['Std_fPSI'] = avg_data.iloc[:, 1:].std(axis=1)

        # Append this folder's average data and label for plotting
        averages_with_error.append(avg_data)
        folder_labels.append(os.path.basename(folder_path))  # Use folder name as label

    # Prepare the plot for averages
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")

    # Plot each folder's average with shaded error bars
    for avg_data, label in zip(averages_with_error, folder_labels):
        plt.plot(avg_data['Depth'], avg_data['Mean_fPSI'], label=f"{label} Average", linewidth=3)
        plt.fill_between(
            avg_data['Depth'],
            avg_data['Mean_fPSI'] - avg_data['Std_fPSI'],
            avg_data['Mean_fPSI'] + avg_data['Std_fPSI'],
            alpha=0.3,
            label=f"{label} ±1 SD",
        )

    # Add plot details
    plt.xlabel('Depth (μm)', fontsize=14)
    plt.ylabel('$\\,fPSI$', fontsize=14)
    plt.title('$\\,fPSI$ vs Depth, averages for two leaves', fontsize=16)
    plt.legend(loc='lower right', fontsize=10, title="Datasets")
    plt.grid(True)

    # Save the consolidated plot
    plot_path = os.path.join(plots_directory_path, "fPSI_vs_depth_averages_with_error.png")
    plt.savefig(plot_path)
    print(f"Plot saved: {plot_path}")
    plt.show()


if __name__ == "__main__":
    # Set the paths to the input directories
    csv_directories = [
        r'your directory to one folder with measurements',
        r'your directory to another folder with measurements',
    ]
    plots_directory = r'your directory to output folder for plot'

    # Call the main processing function
    process_and_plot_folders(csv_directories, plots_directory)
