# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 08:35:57 2024
@author: Schre082
takes a number of csv files from a directory folder, calculates
the amplitude of PSI and then determines the ratio fPSI in a graph
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# Set the paths to directories
csv_directory_path = r'C:/Users/Gebruiker/Documents/thesis BIF/resultsfit/onderzijdevsnormaal'
plots_directory_path = r'C:/Users/Gebruiker/Documents/thesis BIF/plotsfit'

# Create the plots directory if it doesn't exist
os.makedirs(plots_directory_path, exist_ok=True)

# Define the constant C (this depends on microscope detectors)
C = 6.58

# Initialize a dictionary to store data for plotting
plot_data = {}
max_slices = []  # To store the max slice value for each dataset

# Loop over each CSV file in the directory
for filename in os.listdir(csv_directory_path):
    if filename.endswith('.csv'):
        # Full path to the current CSV file
        file_path = os.path.join(csv_directory_path, filename)
        
        # Read the CSV file
        data = pd.read_csv(file_path)
        
        # Ensure tau1, tau2, tau3, A, B, and C columns are numeric
        data[['tau1', 'tau2', 'tau3', 'A', 'B', 'C']] = data[['tau1', 'tau2', 'tau3', 'A', 'B', 'C']].apply(pd.to_numeric, errors='coerce')

        # Add amplitude to a1 if tau is below 250 (PSI)
        data['a1'] = 0.0
        for i, row in data.iterrows():
            a1_value = 0.0
            if row['tau1'] < 250:
                a1_value += row['A']
            if row['tau2'] < 250:
                a1_value += row['B']
            if row['tau3'] < 250:
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

# Prepare the consolidated plot
plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid")
color_palette = sns.color_palette("husl", len(plot_data))  # Generate unique colors

# Plot data from each file
for i, (filename, values) in enumerate(plot_data.items()):
    depth = values['Depth']
    fPSI = values['fPSI']
    
    # Calculate a smoothed error band (standard deviation as an example, or adjust as needed)
    fPSI_mean = fPSI.rolling(window=3, center=True).mean()
    fPSI_std = fPSI.rolling(window=3, center=True).std()
    
    # Plot the line with error shading
    plt.plot(depth, fPSI, label=filename.split('.')[0], color=color_palette[i], linewidth=2)
    plt.fill_between(depth, fPSI_mean - fPSI_std, fPSI_mean + fPSI_std, color=color_palette[i], alpha=0.3)

# Add plot details
plt.xlabel('Depth (μm)', fontsize=14)
plt.ylabel('fPSI', fontsize=14)
plt.title('fPSI vs Depth for adaxial vs abaxial', fontsize=16)
plt.legend(loc='upper right', fontsize=10, title="Datasets")
plt.grid(True)

# Save the consolidated plot
consolidated_plot_path = os.path.join(plots_directory_path, "fPSI_vs_depth_onderzijde.png")
plt.savefig(consolidated_plot_path)
print(f"Consolidated plot saved: {consolidated_plot_path}")
plt.show()
