 # -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:53:32 2024

This code is used to fit a 3 component function on fluorescence decay
data from FLIM (Fluorescence lifetime measurements).
Input is a .sptw folder with .ptu files. Output is a csv file with z-slice 
number and 3 lifetimes with corresponding amplitudes. 

@author: Schre082

"""

# Import statements
import numpy as np
import os
from scipy.optimize import curve_fit, minimize, LinearConstraint
import matplotlib.pyplot as plt
from readPTU_FLIM_cropped_500 import PTUreader
from scipy import integrate
import re
import csv

output_filepath = 'yourdirectory.csv'


# empty list to store parameters of channel2 to be saved in csv
fit_parameters_channel2 = []

linear_constraint = LinearConstraint([[1, 1, 0, 0]], [1], [1])
# Linear constraint for ampl1 + ampl2 <= 1
linear_constraint_jumpexp = LinearConstraint([[0, 0, 0, 1, 1, 0]], [0], [1])  # Ensures ampl1 + ampl2 <= 1

# Exponential decay function with three components (A + B + C = 1 constraint)
def expdec_3(t, tau1, tau2, tau3, A, B, y0):
    """
    fits data to a 3 component function. A,B,C are the amplitudes of the 
    lifetimes and add up to a total of 1
    
    Parameters:
    t: array of timepoints
    tau1: array of lifetimes, first component
    tau2: array of lifetimes, second component
    tau3: array of lifetimes, third component
    A: amplitude, gives attribution of first component to measured lifetime
    B: amplitude, gives attribution of second component to measured lifetime
    y0: array, background of lifetimes
    
    """
    C = 1 - A - B  # Enforce A + B + C = 1
    exponential_1 = A * np.exp(-t / tau1)
    exponential_2 = B * np.exp(-t / tau2)
    exponential_3 = C * np.exp(-t / tau3)
    y = exponential_1 + exponential_2 + exponential_3 + y0
    return y

# Exponential decay function with three components (A + B + C = 1 constraint)
def mse_expdec_3(p, xdata, ydata):
    """
    calculates mse of expdec3 fit
    
    Parameters:
        p: parameters from expdec3 fit
        xdata: data of x-axis, time array
        ydata: data of y-axis, intensity array of fluorescence
    Returns:
        mse(float): mean squared error of expdec3 fit
    """
    tau1, tau2, tau3, A, B, y0 = p
    C = 1 - A - B  # Enforce A + B + C = 1
    exponential_1 = A * np.exp(-xdata / tau1)
    exponential_2 = B * np.exp(-xdata / tau2)
    exponential_3 = C * np.exp(-xdata / tau3)
    y = exponential_1 + exponential_2 + exponential_3 + y0
    mse = np.sum((y-ydata)**2)
    return mse


def shorten_data(x_time, y_inten):
    """
    Shortens the data to start from the point where intensity is above a certain threshold.
    
    Parameters:
        x_time (array): The time values.
        y_inten (array): The intensity values.
        threshold_ratio (float): The ratio of the maximum intensity to use as the threshold. Default is 1% of max.
        
    Returns:
        np.ndarray: The shortened time and intensity data.
    """
    x_time = np.asarray(x_time)
    y_inten = np.asarray(y_inten)

    if x_time.shape[0] != y_inten.shape[0]:
        raise ValueError("x_time and y_inten must have the same length")

    # Use a threshold to shorten the data
    max_intensity = np.max(y_inten)
    threshold_value = max_intensity

    # Find the first index where intensity is greater than or equal to the threshold
    above_threshold_indices = np.where(y_inten >= threshold_value)[0]
    if len(above_threshold_indices) == 0:
        raise ValueError("No points found above the specified threshold")

    first_index = above_threshold_indices[0]

    # Shorten data starting from that index
    shortened_x_time = x_time[first_index:] - x_time[first_index]
    shortened_y_inten = y_inten[first_index:]

    return np.vstack((shortened_x_time, shortened_y_inten))

def normalize_data(y_inten, method="max"):
    """
    Normalizes intensity data based on the specified method.
    
    Parameters:
        y_inten (array): The intensity values.
        method (str): The method for normalization, either "max" or "first_nonzero".
        
    Returns:
        np.ndarray: The normalized intensity data.
    """
    y_inten = np.asarray(y_inten)
    
    if method == "max":
        normalization_factor = np.max(y_inten)
    elif method == "first_nonzero":
        non_zero_indices = np.nonzero(y_inten)[0]
        if len(non_zero_indices) == 0:
            raise ValueError("No non-zero values found in y_inten")
        normalization_factor = y_inten[non_zero_indices[0]]
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if normalization_factor == 0:
        raise ValueError("Normalization factor is zero, cannot normalize")
    
    return y_inten / normalization_factor

# Function to match data lengths
def match_data_lengths(x_time, y_inten):
    """
    matches lengths of 2 datasets
    
    Parameters:
        x_time: data of x-axis, time array
        y_inten: data of y-axis, intensity array of fluorescence
    Returns:
        x_time: shortened array of time data
        y_inten: shortened array of intensity data
    """
    min_length = min(len(x_time), len(y_inten))
    return x_time[:min_length], y_inten[:min_length]

# Function to match data lengths
def match_data_lengths_3(v, w, x):
    """
    matches lengths of 3 datasets
    
    Parameters:
        v,w,x: 3 datasets to be shortened to length of shortest dataset
    Returns:
        v, w, x: shortened arrays
    """
    min_length = min(len(v), len(w), len(x))
    return v[:min_length], w[:min_length], x[:min_length]

# Adjusted R-squared calculation
def calc_adjusted_R_sq_expdec(data_to_fit, fit_model, popt_now):  
    """
    calculates adjusted R squared of expdec3 fit
    
    Parameters:
        data_to_fit: original data
        fit_model: model fitted by expdec 3 function
        
    Returns:
        adjusted R squared
    """
    try:
        residuals = data_to_fit[:, 1] - fit_model(data_to_fit[:, 0], *popt_now)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((data_to_fit[:, 1] - np.mean(data_to_fit[:, 1])) ** 2)
        R_sq = 1 - (ss_res / ss_tot)
        adjusted_R_sq = 1 - (((1 - R_sq) * (len(data_to_fit[:, 1]) - 1)) / (len(data_to_fit[:, 1]) - len(popt_now) - 1))
        return adjusted_R_sq
    except Exception as e:
        print(f"Error calculating adjusted R-squared: {e}")
        raise
        
# Updated process_file function
def process_file(filepath, z_value):
    """
    Processes FLIM data from a PTU file, filters the intensity, normalizes the decay data,
    and fits an exponential decay model (expdec_3) to the fluorescence decay data for two channels.
    
    Parameters:
        filepath (str): The path to the PTU file containing the FLIM data.
        z_value (float): The z-value associated with the data
        (used for logging or further analysis).
        
    Returns:
        min_coefficients_channel_2 (list): A list of optimized fitting parameters for Channel 2,
        including decay constants (tau1, tau2, tau3) and amplitudes (A, B, C).
        
    """
    ptu_file = PTUreader(filepath, print_header_data=False)
    data, _ = ptu_file.get_flim_data_stack()
    
    print(data.shape)

    # Intensity calculation
    intensity = np.sum(data, axis=(2, 3))
    max_intensity = np.max(intensity)
    threshold = 0.1 * max_intensity
    mask = intensity >= threshold
    filtered_data = np.zeros_like(data)
    filtered_data[mask] = data[mask]

    channel_1 = filtered_data[:, :, 0, :]
    channel_2 = filtered_data[:, :, 1, :]
    decay_data_channel_1 = np.sum(channel_1, axis=(0, 1))
    decay_data_channel_2 = np.sum(channel_2, axis=(0, 1))

    time = 90 * np.arange(0, 133)

    # Normalize decay data
    decay_detector_1_norm = decay_data_channel_1 / np.max(decay_data_channel_1)
    decay_detector_2_norm = decay_data_channel_2 / np.max(decay_data_channel_2)

    # Ensure time and decay data lengths match
    time, decay_detector_1_norm, decay_detector_2_norm = match_data_lengths_3(time, decay_detector_1_norm, decay_detector_2_norm)

    # ----- Step 1: Fit using expdec_3 ----- 
    max_idx_channel_1 = np.argmax(decay_detector_1_norm)
    time_truncated_channel_1 = time[max_idx_channel_1:]
    decay_detector_1_truncated = decay_detector_1_norm[max_idx_channel_1:]

    max_idx_channel_2 = np.argmax(decay_detector_2_norm)
    time_truncated_channel_2 = time[max_idx_channel_2:]
    decay_detector_2_truncated = decay_detector_2_norm[max_idx_channel_2:]

    # Before calling shorten_data, ensure time and decay_data_channel_1 have the same length
    min_length = min(len(time), len(decay_data_channel_1))

    # Truncate both arrays to the same length
    time_truncated = time[:min_length]
    decay_data_channel_1_truncated = decay_data_channel_1[:min_length]
    
    # Before calling shorten_data, ensure time and decay_data_channel_1 have the same length
    min_length = min(len(time), len(decay_data_channel_2))

    # Truncate both arrays to the same length
    time_truncated = time[:min_length]
    decay_data_channel_2_truncated = decay_data_channel_2[:min_length]

    # Shorten data for both channels
    decay_detector_1_from_max = shorten_data(time_truncated, decay_data_channel_1_truncated)
    decay_detector_2_from_max = shorten_data(time_truncated, decay_data_channel_2_truncated)
    
    # Normalize the shortened data
    normalized_decay_detector_1 = normalize_data(decay_detector_1_from_max[1], method="max")
    normalized_decay_detector_2 = normalize_data(decay_detector_2_from_max[1], method="max")
    
    #for the second channel
    cf_coefficients_channel_2, _ = curve_fit(
        expdec_3, 
        decay_detector_2_from_max[0], 
        normalized_decay_detector_2, 
        p0=[100, 500, 1000, 0.2, 0.2, 0],  # Initial guess (tau1, tau2, tau3, A, B, y0)
        bounds=((100, 100, 100, 0, 0, 0), (np.inf, np.inf, np.inf, 1, 1, np.inf))  # Adjust bounds to constrain A + B + C = 1
        )
    
    # Print parameter values from the first fit
    print(f"Curve fit: first fit parameters for Channel 2 (z={z_value}):")
    print(f"tau1: {cf_coefficients_channel_2[0]:.2f}")
    print(f"tau2: {cf_coefficients_channel_2[1]:.2f}")
    print(f"tau3: {cf_coefficients_channel_2[2]:.2f}")
    print(f"A: {cf_coefficients_channel_2[3]:.2f}")
    print(f"B: {cf_coefficients_channel_2[4]:.2f}")
    print(f"C: {1 - cf_coefficients_channel_2[3] - cf_coefficients_channel_2[4]:.2f}")
    print(f"y0: {cf_coefficients_channel_2[5]:.2f}")
    xdata = decay_detector_2_from_max[0]
    ydata = normalized_decay_detector_2
    print(f"MSE:", mse_expdec_3(cf_coefficients_channel_2, xdata, ydata))
    
    linear_constraint = LinearConstraint([[0, 0, 0, 1, 1, 0]], [0], [1]) # Enforce 0 <= A+B <= 1
    res = minimize(mse_expdec_3, 
                   x0 = [100, 500, 1000, 0.2, 0.2, 0], 
                   args=(xdata,ydata), 
                   bounds=((100, np.inf), (100, np.inf), (100, np.inf), (0,1), (0,1), (0,np.inf)),
                   constraints=linear_constraint, tol=1e-6, method="trust-constr")
    print(res.message)
    min_coefficients_channel_2 = res.x

    # Print parameter values from the first fit
    print(f"Minimize: first fit parameters for Channel 2 (z={z_value}):")
    print(f"tau1: {min_coefficients_channel_2[0]:.2f}")
    print(f"tau2: {min_coefficients_channel_2[1]:.2f}")
    print(f"tau3: {min_coefficients_channel_2[2]:.2f}")
    print(f"A: {min_coefficients_channel_2[3]:.2f}")
    print(f"B: {min_coefficients_channel_2[4]:.2f}")
    print(f"C: {1 - min_coefficients_channel_2[3] - min_coefficients_channel_2[4]:.2f}")
    print(f"y0: {min_coefficients_channel_2[5]:.2f}")
    print(f"MSE:", mse_expdec_3(min_coefficients_channel_2, xdata, ydata))

    #add coefficients to list
    fit_parameters_channel2.append(min_coefficients_channel_2)
    #print(fit_parameters_channel2)

    # For now, continue with the minimize results
    
    #coefficients_channel_1 = min_coefficients_channel_1 
    coefficients_channel_2 = min_coefficients_channel_2

    return min_coefficients_channel_2

if __name__ == "__main__":
    # Directory containing the stack of .ptu files
    path = r"your directory of .ptu files.sptw/"
    os.chdir(path)

    # List all PTU files in the directory and sort them by the z-value in the filename
    ptu_files = [f for f in os.listdir(path) if re.match(r'Series009_z\d+\.ptu', f)]# use regex to match ptu file naming
    
    # Sort files based on the z-value (numerical sorting)
    ptu_files.sort(key=lambda x: int(re.search(r'z(\d+)', x).group(1)))
    
    # Process each PTU file
    for file_name in ptu_files:
        z_match = re.search(r'z(\d+)', file_name)  # Extract the z-value from the filename
        if z_match:
            z_value = int(z_match.group(1))
            print(f"Processing file: {file_name} with z={z_value}")
            process_file(os.path.join(path, file_name), z_value)
        else:
            print(f"Could not extract z-value from filename: {file_name}")
            
    # After the loop fitting all slices, write the parameters to a CSV file
    # Open CSV file in append mode at the start
    is_new_file = not os.path.exists(output_filepath)  # Check if file exists
    with open(output_filepath, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
    
    #with open('C:/Users/Schre082/Documents/thesis BIF/fit_parameters.csv', mode='w', newline='') as file:
       #writer = csv.writer(file)
        # Write the header only if the file is new
        if is_new_file:
            csv_writer.writerow(['Slice', 'tau1', 'tau2', 'tau3', 'A', 'B', 'C', 'y0'])
        
        for i, params in enumerate(fit_parameters_channel2):
            tau1, tau2, tau3, A, B, y0 = params
            C = 1 - A - B  # Calculate C
            csv_writer.writerow([i+1, tau1, tau2, tau3, A, B, C, y0])
