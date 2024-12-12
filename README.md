# thesis_BIF_2024 Femke Schreurs

The project it used for data analysis of z-stacks with fluorescence lifetime imaging (FLIM) data. The input should be in ptu format. 
As output it is possible to generate csv files and also generate images subsequently with these files. These include, intensity images for both the 
Photosystem I (PSI)/ photosystem II (PSII) ratio and the intensity of PSI and PSII seperately over the z-stack. Intensity and filtered intensity images can
also be produced. Furthermore the fPSI can be calculated which reflects the PSI and PSII contributions at various depths in the image. 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Acknowledgments](#acknowledgments)

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

## Structure
- thesis_BIF_2024/
│
├── data/                # Raw data files
├── src/                 # Source code for data processing and analysis
├── process_data.py  # Main script for processing data
├── analyze.py       # Script for generating visualizations
├── tests/               # Unit tests for various modules
├── requirements.txt     # Python dependencies
└── README.md            # This file
