# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:15:03 2024

@author: Schre082
code for reading folder of ptu files, calculating average intensity
then saving in a csv file the z-slice number and the intensities of PSI and PSII
"""

import numpy as np
import os
import csv
from readPTU_FLIM import PTUreader
import matplotlib.pyplot as plt

# Define the path to your folder containing the PTU files
path = r"C:/Users/Schre082/OneDrive - Wageningen University & Research/Master thesis BIF/data/24102024_Femke_Schreurs_arabidopsis_1.sptw/"
os.chdir(path)

# Initialize lists to hold the average pixel intensities for each channel
average_intensities_psii = []
average_intensities_psi = []
z_values = []

# Iterate over each PTU file in the folder
for filename in os.listdir(path):
    if filename.endswith('.ptu'):
        ptu_file = PTUreader(os.path.join(path, filename), print_header_data=False)
        data, intensity_image = ptu_file.get_flim_data_stack()
        
        # Separate the data into two 3D arrays, one for each channel
        channel_1_data = data[:, :, 0, :]  # PSII
        channel_2_data = data[:, :, 1, :]  # PSI

        # Calculate the average intensity for each channel
        average_intensity_psii = np.mean(channel_1_data)
        average_intensity_psi = np.mean(channel_2_data)
        
        # Extract the Z value from the filename
        z_value = int(filename.split('z')[-1].split('.')[0])
        
        # Store the results
        average_intensities_psii.append(average_intensity_psii)
        average_intensities_psi.append(average_intensity_psi)
        z_values.append(z_value)

# Sort the data by Z value
sorted_indices = np.argsort(z_values)
z_values = np.array(z_values)[sorted_indices]
average_intensities_psii = np.array(average_intensities_psii)[sorted_indices]
average_intensities_psi = np.array(average_intensities_psi)[sorted_indices]

# Define output directory
output_dir = r"C:/Users/Schre082/Documents/thesis_BIF/intensity_output"
os.makedirs(output_dir, exist_ok=True)

# Generate a name for the output file based on the folder name
folder_name = os.path.basename(os.path.normpath(path))
output_file_name = f"intensity_PSI_PSII_seperate_{folder_name}.csv"
output_path = os.path.join(output_dir, output_file_name)

# Write the data to the CSV file
with open(output_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["z_number", "intensity_PSI", "intensity_PSII"])  # Header
    writer.writerows(zip(z_values, average_intensities_psi, average_intensities_psii))  # Data rows

print(f"Data successfully saved to {output_path}")