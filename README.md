# Deepwavespectra

This repository contains example code and test cases for the machine learning models developed to predict ocean wave conditions using 2D wave spectra. These models are designed to upscale and bias correct 2D wave spectra, improving prediction accuracy for significant wave height and other wave parameters.

## Overview

### Abstract
The accurate prediction of 2D wave spectra is a challenge for the maritime and coastal engineering sectors. This repository includes examples, test cases, and guidelines to:
Develop and compare a range of deep learning approaches for upscaling and bias correcting 2D wave spectra.
Demonstrate the performance of deep learning models against a phase-averaged wave model.
Highlight performance improvements, including a reduction in root mean squared error (RMSE) of up to 33% for significant wave height.
Explore improvements in bias correction across the spectrum, particularly for spectral density in frequency bins ranging from 7 to 28 s period.
While the deep learning approaches showed improved bias correction and reduced RMSE, they also exhibited a tendency to be diffusive (biased towards the mean), suggesting further development of training approaches to reduce diffusion.

Full details are in the following paper: 

## Guide for users

The models provided were trained with years of data at a single location, it is strongly advised that new models be trained for other locations.

## Repository Structure

* examples: Example scripts demonstrating the usage of the ML models.

* data: Sample datasets for training and testing the models.

* models: Pre-trained models and configurations for various scenarios.

## Getting Started

Clone the repository:

git clone https://github.com/DrakonianMight/deepwavespectra.git

Install the required dependencies:

pip install -r requirements.txt

### Requirements

Python
For running the examples a CPU will be OK, but a GPU is recommended for training your own models.
CUDA v10 >

Run the example jupyter notebooks in the /examples directory to get started.

