# Deepwavespectra

This repository contains example code and test cases for the machine learning models developed to predict ocean wave conditions using 1D wave spectra. These models are designed to downscale offshore wave conditions to nearshore locations, improving prediction accuracy for significant wave height and other parameters.

## Overview

Machine Learning (ML) has shown significant potential in enhancing wave condition prediction by leveraging 1D wave spectra. This repository includes examples, test cases, and guidelines to:

Evaluate the impact of feature selection and engineering.

Demonstrate the application of different ML approaches, including Long-Term Short-Term Memory (LSTM).

Highlight performance improvements achieved by using 1D wave spectra compared to integrated parameter-only approaches.

## Repository Structure

/examples: Example scripts demonstrating the usage of the ML models.

/test_cases: Test cases for validating model performance and configuration.

/data: Sample datasets for training and testing the models.

/models: Pre-trained models and configurations for various scenarios.

## Getting Started

Clone the repository:

git clone https://github.com/your-username/ocean-wave-spectra-ml.git

Install the required dependencies:

pip install -r requirements.txt

Run the example scripts in the /examples directory to get started.

### Subsections to Populate

1. Feature Selection and Engineering

Provide detailed explanations and examples of how feature selection and engineering impact model performance. Include guidelines for:

Choosing relevant features from the 1D wave spectra.

Encoding and preprocessing methods.

2. Model Architectures

Add descriptions and configurations for the ML models implemented, including:

LSTM-based approaches.

Comparative analysis with other ML architectures.

3. Performance Metrics

Include examples of performance metrics used to evaluate the models, such as:

Root Mean Squared Error (RMSE)

Mean Absolute Error (MAE)

4. Results and Analysis

Present results obtained from different models and configurations. Highlight key findings, such as:

The 27% RMSE reduction achieved using 1D wave spectra.

Sensitivities observed in the input data.
