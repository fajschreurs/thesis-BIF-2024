 # -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:53:32 2024
code om de amplitude (tau) op te slaan en de bijbehorende z-slice
hier uit channel 2 (700-750nm)
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

output_filepath = 'C:/Users/Schre082/Documents/thesis_BIF/fit_parameters.csv'


# empty list to store parameters of channel2 to be saved in csv
fit_parameters_channel2 = []

linear_constraint = LinearConstraint([[1, 1, 0, 0]], [1], [1])
# Linear constraint for ampl1 + ampl2 <= 1
linear_constraint_jumpexp = LinearConstraint([[0, 0, 0, 1, 1, 0]], [0], [1])  # Ensures ampl1 + ampl2 <= 1

# Exponential decay function with three components (A + B + C = 1 constraint)
def expdec_3(t, tau1, tau2, tau3, A, B, y0):
    C = 1 - A - B  # Enforce A + B + C = 1
    exponential_1 = A * np.exp(-t / tau1)
    exponential_2 = B * np.exp(-t / tau2)
    exponential_3 = C * np.exp(-t / tau3)
    y = exponential_1 + exponential_2 + exponential_3 + y0
    return y

# Exponential decay function with three components (A + B + C = 1 constraint)
def mse_expdec_3(p, xdata, ydata):
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

# # Convolution using FFT
# def Convol(x, h):
#     X = np.fft.fft(x)
#     H = np.fft.fft(h)
#     xch = np.real(np.fft.ifft(X * H))
#     return xch

# Define the model with three exponentials to be convoluted with IRF
#DRdef jumpexpmodel_curve_3(X, tau1, tau2, tau3, ampl1, ampl2, ampl3, y0):
#DR    x, irf = X
#DR    C = 1 - ampl1 - ampl2 - ampl3
#DR    ymodel = ampl1 * np.exp(-x / tau1) + ampl2 * np.exp(-x / tau2) + ampl3 * np.exp(-x / tau3) + C * np.exp(-x / tau3)
#DR    z = Convol(ymodel, irf) + y0
#DR    return z
# def jumpexpmodel_curve_3(X, tau1, tau2, tau3, ampl1, ampl2, y0):
#     x, irf = X
#     ampl3 = 1 - ampl1 - ampl2
#     ymodel = ampl1 * np.exp(-x / tau1) + ampl2 * np.exp(-x / tau2) + ampl3 * np.exp(-x / tau3) + y0
#     z = Convol(ymodel, irf) + y0
#     return z

# # Define the objective function (MSE) for jumpexpmodel
# def mse_jumpexpmodel(p, X, ydata):
#     tau1, tau2, tau3, ampl1, ampl2, y0 = p
#     ymodel = jumpexpmodel_curve_3(X, tau1, tau2, tau3, ampl1, ampl2, y0)
#     mse = np.sum((ydata - ymodel)**2)
#     return mse

# # Function to load and normalize IRF data
# def load_irf(filepath):
#     ptu_file = PTUreader(filepath, print_header_data=False)
#     data, _ = ptu_file.get_flim_data_stack()

#     # Initialize a dictionary to store normalized IRF data for both channels
#     irf_dict = {}
#     time = 90 * np.arange(0, 133)  # Time in picoseconds

#     # Process and normalize the IRF data for both channels
#     for channel in range(2):
#         irf_channel = np.sum(data[:, :, channel, :], axis=(0, 1))
#         irf_channel_norm = irf_channel / np.max(irf_channel)
#         irf_dict[f'Channel_{channel + 1}'] = {'Decay': irf_channel, 'Decay_norm': irf_channel_norm}
    
#     return irf_dict

# Function to match data lengths
def match_data_lengths(x_time, y_inten):
    min_length = min(len(x_time), len(y_inten))
    return x_time[:min_length], y_inten[:min_length]

# Function to match data lengths
def match_data_lengths_3(v, w, x):
    min_length = min(len(v), len(w), len(x))
    return v[:min_length], w[:min_length], x[:min_length]

# Adjusted R-squared calculation
def calc_adjusted_R_sq_expdec(data_to_fit, fit_model, popt_now):    
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
        
# # Adjusted R-squared calculation for jumpexpmodel_curve_3
# def calc_adjusted_R_sq_jumpexp(data_to_fit, irf_data, popt_now):
#     try:
#         # Prepare the X input for jumpexpmodel_curve_3
#         X = (data_to_fit[:, 0], irf_data)
        
#         # Compute the residuals: difference between actual data and model predictions
#         residuals = data_to_fit[:, 1] - jumpexpmodel_curve_3(X, *popt_now)
        
#         # Sum of squares of residuals
#         ss_res = np.sum(residuals ** 2)
        
#         # Total sum of squares
#         ss_tot = np.sum((data_to_fit[:, 1] - np.mean(data_to_fit[:, 1])) ** 2)
        
#         # Calculate R-squared
#         R_sq = 1 - (ss_res / ss_tot)
        
#         # Adjusted R-squared
#         adjusted_R_sq = 1 - (((1 - R_sq) * (len(data_to_fit[:, 1]) - 1)) / (len(data_to_fit[:, 1]) - len(popt_now) - 1))
        
#         return adjusted_R_sq
#     except Exception as e:
#         print(f"Error calculating adjusted R-squared for jumpexpmodel: {e}")
#         raise        

# # Align IRF to data
# def align_and_wrap_irf(time, irf_data, data):
#     irf_peak_idx = np.argmax(irf_data)
#     data_peak_idx = np.argmax(data)
#     time_shift = data_peak_idx - irf_peak_idx
#     irf_data_wrapped = np.roll(irf_data, time_shift)
#     return irf_data_wrapped


# Updated process_file function
def process_file(filepath, z_value):
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

    # irf_channel_1_norm = irf_data['Channel_1']['Decay_norm']
    # irf_channel_2_norm = irf_data['Channel_2']['Decay_norm']

    # # Align IRF with data
    # irf_channel_1_aligned = align_and_wrap_irf(time, irf_channel_1_norm, decay_detector_1_norm)
    # irf_channel_2_aligned = align_and_wrap_irf(time, irf_channel_2_norm, decay_detector_2_norm)

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
    
    # Fit the model to the normalized, shortened data for both channels
    #for the first channel
    # cf_coefficients_channel_1, _ = curve_fit(
    #     expdec_3, 
    #     decay_detector_1_from_max[0], 
    #     normalized_decay_detector_1, 
    #     p0=[100, 500, 1000, 0.2, 0.2, 0],  # Initial guess (tau1, tau2, tau3, A, B, y0)
    #     bounds=((100, 100, 100, 0, 0, 0), (np.inf, np.inf, np.inf, 1, 1, np.inf))  # Adjust bounds to constrain A + B + C = 1
    #     )
    # # Print parameter values from the first fit
    # print(f"Curve fit: first fit parameters for Channel 1 (z={z_value}):")
    # print(f"tau1: {cf_coefficients_channel_1[0]:.2f}")
    # print(f"tau2: {cf_coefficients_channel_1[1]:.2f}")
    # print(f"tau3: {cf_coefficients_channel_1[2]:.2f}")
    # print(f"A: {cf_coefficients_channel_1[3]:.2f}")
    # print(f"B: {cf_coefficients_channel_1[4]:.2f}")
    # print(f"C: {1 - cf_coefficients_channel_1[3] - cf_coefficients_channel_1[4]:.2f}")
    # print(f"y0: {cf_coefficients_channel_1[5]:.2f}")
    # xdata = decay_detector_1_from_max[0]
    # ydata = normalized_decay_detector_1
    # print(f"MSE:", mse_expdec_3(cf_coefficients_channel_1, xdata, ydata))
    
    # linear_constraint = LinearConstraint([[0, 0, 0, 1, 1, 0]], [0], [1]) # Enforce 0 <= A+B <= 1
    # res = minimize(mse_expdec_3, 
    #                x0 = [100, 500, 1000, 0.2, 0.2, 0], 
    #                args=(xdata,ydata), 
    #                bounds=((100, np.inf), (100, np.inf), (100, np.inf), (0,1), (0,1), (0,np.inf)),
    #                constraints=linear_constraint, tol=1e-6, method="trust-constr")
    # print(res.message)
    # min_coefficients_channel_1 = res.x

    # # Print parameter values from the first fit
    # print(f"Minimize: first fit parameters for Channel 1 (z={z_value}):")
    # print(f"tau1: {min_coefficients_channel_1[0]:.2f}")
    # print(f"tau2: {min_coefficients_channel_1[1]:.2f}")
    # print(f"tau3: {min_coefficients_channel_1[2]:.2f}")
    # print(f"A: {min_coefficients_channel_1[3]:.2f}")
    # print(f"B: {min_coefficients_channel_1[4]:.2f}")
    # print(f"C: {1 - min_coefficients_channel_1[3] - min_coefficients_channel_1[4]:.2f}")
    # print(f"y0: {min_coefficients_channel_1[5]:.2f}")
    # print(f"MSE:", mse_expdec_3(min_coefficients_channel_1, xdata, ydata))

    # # plot both fits for channel 1
    # fit_cf  = expdec_3(decay_detector_1_from_max[0], *cf_coefficients_channel_1)
    # fit_min = expdec_3(decay_detector_1_from_max[0], *min_coefficients_channel_1)
    
    # fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    
    # # Channel 1 Plot - First Fit
    # ax[0].plot(decay_detector_1_from_max[0], normalized_decay_detector_1, 'b-', label="Raw Data Channel 1")
    # ax[0].plot(decay_detector_1_from_max[0], fit_cf, 'g--', label="curve_fit (expdec_3) Channel 1")
    # ax[0].legend()
    # ax[0].set_xlabel("Time (ps)")
    # ax[0].set_ylabel("Intensity (Normalized)")
    # ax[0].set_title(f"FLIM Decay and First Fit - Channel 1 (cf, z={z_value})")
    
    # # Channel 1 Plot - First Fit
    # ax[1].plot(decay_detector_1_from_max[0], normalized_decay_detector_1, 'b-', label="Raw Data Channel 2")
    # ax[1].plot(decay_detector_1_from_max[0], fit_min, 'g--', label="minimize (mse_expdec_3) Channel 1")
    # ax[1].legend()
    # ax[1].set_xlabel("Time (ps)")
    # ax[1].set_ylabel("Intensity (Normalized)")
    # ax[1].set_title(f"FLIM Decay and First Fit - Channel 1 (min, z={z_value})")
    
    # plt.tight_layout()
    # plt.show()
    
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

    # # plot both fits for channel2
    # fit_cf  = expdec_3(decay_detector_2_from_max[0], *cf_coefficients_channel_2)
    # fit_min = expdec_3(decay_detector_2_from_max[0], *min_coefficients_channel_2)
    
    # fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    
    # # Channel 2 Plot - First Fit
    # ax[0].plot(decay_detector_2_from_max[0], normalized_decay_detector_2, 'b-', label="Raw Data Channel 2")
    # ax[0].plot(decay_detector_2_from_max[0], fit_cf, 'g--', label="curve_fit (expdec_3) Channel 2")
    # ax[0].legend()
    # ax[0].set_xlabel("Time (ps)")
    # ax[0].set_ylabel("Intensity (Normalized)")
    # ax[0].set_title(f"FLIM Decay and First Fit - Channel 2 (cf, z={z_value})")
    
    # # Channel 2 Plot - First Fit
    # ax[1].plot(decay_detector_2_from_max[0], normalized_decay_detector_2, 'b-', label="Raw Data Channel 2")
    # ax[1].plot(decay_detector_2_from_max[0], fit_min, 'g--', label="minimize (mse_expdec_3) Channel 2")
    # ax[1].legend()
    # ax[1].set_xlabel("Time (ps)")
    # ax[1].set_ylabel("Intensity (Normalized)")
    # ax[1].set_title(f"FLIM Decay and First Fit - Channel 2 (min, z={z_value})")
    
    # plt.tight_layout()
    # plt.show()
      
    # For now, continue with the minimize results
    
    #coefficients_channel_1 = min_coefficients_channel_1 
    coefficients_channel_2 = min_coefficients_channel_2


#    return min_coefficients_final_channel_1, min_coefficients_final_channel_2
    return min_coefficients_channel_2

if __name__ == "__main__":
    # Directory containing the stack of .ptu files
    path = r"C:/Users/Schre082/OneDrive - Wageningen University & Research/Master thesis BIF/data/19062024_Femke_Schreurs_spinazie_onderzijde_2.sptw/"
    os.chdir(path)

    # # Load the IRF data from the provided IRF file path
    # irf_file = r"C:/Users/Schre082/OneDrive - Wageningen University & Research\Master thesis BIF/data/irf_1_ARABIDOPSIS_24OKT.ptu"
    # irf_data = load_irf(irf_file)

    # List all PTU files in the directory and sort them by the z-value in the filename
    ptu_files = [f for f in os.listdir(path) if re.match(r'Series009_z\d+\.ptu', f)]
    #re.match(r'\d+_femkeSchreurs_spinazie\d+_z(\d+)\.ptu', f)]
    #re.match(r'\d+_z(\d+)\.ptu', f)]
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
            
            
           
