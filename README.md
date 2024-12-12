# thesis_BIF_2024 Femke Schreurs

The project it used for data analysis of z-stacks with fluorescence lifetime imaging (FLIM) data. The input should be in ptu format. 
As output it is possible to generate csv files and also generate images subsequently with these files. These include, intensity images for both the 
Photosystem I (PSI)/ photosystem II (PSII) ratio and the intensity of PSI and PSII seperately over the z-stack. Intensity and filtered intensity images can
also be produced. Furthermore the fPSI can be calculated which reflects the PSI and PSII contributions at various depths in the image. Short descriptions of the function
of each file can be found under structure, more elaborate explanations can be found in the files themselves

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Structure](#structure)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/fajschreurs/thesis-BIF-2024
   cd thesis-BIF-2024
   pip install -r requirements.txt

## Features
- data processing
- analyse .ptu files
- Various methods of data visualization
- present fPSI calculations
- calculate ratio PSI/PSII intensities

## Structure
- thesis_BIF_2024/
│
├── PSI_PSII_intensity_ratio_plot.py                            # used for plotting the ratio between PSI and PSII intensities
├── PSI_and_PSII_intensities_plot.py                            # used for plotting PSI and PSII intensities seperately
├── fPSI_vs_depth_image_production.py                           # used for plotting the fPSI vs the depth in the leaf
├── fitting_3_components_on_fluorescence_decay_data.py          # Used for fitting a 3 component decay curve to fluorescence decay data from multiphoton of leaves
├── fitting_3components_and_IRF_convolution.py                  # same as previous but with additional fit of convolution of data with IRF
├── intensity_ratio_PSI_divided by_PSII.py                      #used to produce csv file of ratio between PSI and PSII intensities
├── plotting_average_fPSI_multiple_leafs_with_std.py            #used to plot average fPSI of multiple leaves with standard deviation
├── readPTU_FLIM_cropped_500.py                                 #used to read .ptu files from PicoQuant
├── requirements.txt                                            # Python dependencies
├── seperate_intensities_image_PSI_PSII.py                      #used to image seperate inensities of PSI and PSII
└── README.md                                                   # This file
