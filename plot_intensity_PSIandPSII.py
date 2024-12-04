# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:52:36 2024

@author: Gebruiker
code that takes csv files with 3 columns (Z-stack number, PSI intensity, PSII intensity)
and produces a figure that includes all csv files in the folder.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the folder containing the CSV files
csv_directory_path = r'C:/Users/Gebruiker/Documents/thesis BIF/intensity/arabidopsis'

# Define the folder to save the output plot
output_directory_path = r'C:/Users/Gebruiker/Documents/thesis BIF/seperateintensityplot'
os.makedirs(output_directory_path, exist_ok=True)  # Create the folder if it doesn't exist

# Name of the output plot file
output_filename = "PSI_PSII_intensities_plot_arabidopsis.png"
output_file_path = os.path.join(output_directory_path, output_filename)

# Initialize the plot
plt.figure(figsize=(10, 6))

# Generate a color palette with one color per file
from itertools import cycle
color_palette = cycle(plt.cm.tab10.colors)  # Use matplotlib's Tab10 colormap for consistent colors

# Loop through each CSV file in the folder
for filename in os.listdir(csv_directory_path):
    if filename.endswith('.csv'):
        # Full path to the CSV file
        file_path = os.path.join(csv_directory_path, filename)
        
        # Read the CSV file
        data = pd.read_csv(file_path)
        
        # Ensure the CSV has the expected structure
        if data.shape[1] != 3:
            print(f"Skipping {filename}: unexpected number of columns.")
            continue
        
        # Extract Z values, PSI intensities, and PSII intensities
        z_values = data.iloc[:, 0]  # First column (Z values)
        PSI_intensity = data.iloc[:, 1]  # Second column (PSI intensity)
        PSII_intensity = data.iloc[:, 2]  # Third column (PSII intensity)
        
        # Convert Z values to Depth (µm)
        max_z = z_values.max()
        depth_values = (max_z - z_values) * 0.64  # Convert Z-stack to depth
        
        # Get a unique color for this file
        color = next(color_palette)
        
        # Plot PSI intensity with the chosen color
        plt.plot(depth_values, PSI_intensity, label=f"{filename.split('.')[0]} - PSI", color=color, linestyle='-')
        
        # Plot PSII intensity with the same color but a different linestyle
        plt.plot(depth_values, PSII_intensity, label=f"{filename.split('.')[0]} - PSII", color=color, linestyle='--')

# Add plot labels, title, and legend
plt.xlabel('Depth (µm)', fontsize=12)
plt.ylabel('Intensity', fontsize=12)
plt.title('PSI and PSII Intensities vs Depth A.thaliana', fontsize=14)
plt.legend(fontsize=10, loc='best')
plt.grid(True)

# Save the plot
plt.tight_layout()
plt.savefig(output_file_path)
print(f"Plot saved as: {output_file_path}")
plt.close()
