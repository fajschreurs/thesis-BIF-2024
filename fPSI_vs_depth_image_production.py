# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 08:35:57 2024
@author: Schre082
Code used to produce a figure in which  fPSI vs depth graphs
of different measurements, each in a csv file in the input folder, of a leaf 
is shown.
input has data about the z-slice of the z-stack (multiphoton measurements), 
and the amplitude of PSI (photosystem I) as determined by fitting the data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


def process_csv_files(csv_directory_path, plots_directory_path):
    """
    Processes CSV files to calculate fPSI and generate a plot of fPSI vs depth (μm).

    Parameters:
    csv_directory_path (str): Path to the directory containing CSV files.
    plots_directory_path (str): Path to save the generated plots.
    """
    # Create the plots directory if it doesn't exist
    os.makedirs(plots_directory_path, exist_ok=True)

    # Define the constant C (this depends on microscope detectors)
    C = 6.58

    # Initialize a dictionary to store data for plotting
    plot_data = {}
    
    # Initialize list to store the max slice number of each z-stack
    max_slices = []

    # Loop over each CSV file in the directory
    for filename in os.listdir(csv_directory_path):
        if filename.endswith('.csv'):
            # Full path to the current CSV file
            file_path = os.path.join(csv_directory_path, filename)

            # Read the CSV file
            data = pd.read_csv(file_path)

            # Ensure tau1, tau2, tau3, A, B, and C columns are numeric
            data[['tau1', 'tau2', 'tau3', 'A', 'B', 'C']] = data[['tau1', 'tau2', 'tau3', 'A', 'B', 'C']].apply(pd.to_numeric, errors='coerce')

            # Add amplitude to a1 if tau is below 300 (PSI)
            #where a1 is a combination of all amplitudes belonging to PSI lifetimes
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
            max_slices.append(max_slice)

            # Convert slices to depth (μm)
            data['Depth'] = (max_slice - data['Slice']) * 0.64

            # Store data in the dictionary for plotting
            plot_data[filename] = {
                'Depth': data['Depth'],
                'fPSI': data['fPSI'],
            }

    # Prepare the plot
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    color_palette = sns.color_palette("husl", len(plot_data))  # Generate unique colors

    # Plot data from each file
    for i, (filename, values) in enumerate(plot_data.items()):
        depth = values['Depth']
        fPSI = values['fPSI']

        # Plot the line
        plt.plot(depth, fPSI, label=filename.split('.')[0], color=color_palette[i], linewidth=2)
    
    # Add plot details
    plt.xlabel('Depth (μm)', fontsize=14)
    plt.ylabel('$\\,fPSI$', fontsize=14)
    plt.title('$\\,fPSI$ vs Depth with shifted data', fontsize=16)
    plt.legend(loc='lower right', fontsize=10, title="Datasets")
    plt.grid(True)

    # Save the consolidated plot
    plot_path = os.path.join(plots_directory_path, "nameyourimage.png")
    plt.savefig(plot_path)
    print(f"plot saved: {plot_path}")
    plt.show()


if __name__ == "__main__":
    # Set the paths to directories
    csv_directory = r'your input directory of folder with csv (or multiple csvs)'
    plots_directory = r'your output directory for plot'

    # Call the main processing function
    process_csv_files(csv_directory, plots_directory)