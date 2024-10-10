# Anemia Detection Using Logistic Regression

A machine learning project that predicts anemia status using logistic regression based on image pixel data and hemoglobin levels.

# Table of Contents

* Introduction
* Project Overview
* Installation
* Usage
* Data Description
* Project Structure
* Dependencies
* Results
* Contributing
* License
* Contact Information

# Introduction

This project implements a logistic regression model to predict whether a person is anemic based on features extracted from
images and biological data. The model uses pixel percentage values of red, green, and blue colors from images, along with 
hemoglobin levels and sex as input features.

# Project Overview

The main steps involved in the project are:
1. Data Loading: Reading the dataset from a CSV file.
2. Data Preprocessing:
   * Shuffling the dataset.
   * Encoding categorical variables.
   * Scaling numerical features.
3. Model Training:
   * Splitting the data into training and testing sets.
   * Training a logistic regression model.
4. Model Evaluation:
   * Generating a confusion matrix.
   * Calculating evaluation metrics like precision, recall, and F1-score.
   * Plotting the ROC curve and calculating the AUC.
5. Visualization: Displaying the ROC curve to visualize the model's performance.

# Installation

## Prerequisites
* Python 3.x
* pip package manager

## Steps
1. Clone the repository
> git clone https://github.com/sofyaklyuchereva/anemia-detection-logistic-regression.git
> cd anemia-detection-logistic-regression

2. Install required packages
> pip install -r requirements.txt


# Data Description

## Data File
* file_.csv: Contains the dataset used for training and evaluating the model.
## Data Columns
* Sex: Gender of the subject ('M' for male, 'F' for female).
* %Red Pixel: Percentage of red pixels in the image.
* %Green pixel: Percentage of green pixels in the image.
* %Blue pixel: Percentage of blue pixels in the image.
* Hb: Hemoglobin level.
* Anaemic: Anemia status (0 for not anemic, 1 for anemic).


# Project Structure
├── README.md

├── main.py

├── file_.csv

├── requirements.txt

├── Anaemic_Classifier_Model.ipynb


# Dependencies
* pandas
* numpy
* matplotlib
* scikit-learn


# Results
After running the script, you will get:
* Confusion Matrix: Shows the number of true positives, true negatives, false positives, and false negatives.
* Classification Report: Includes precision, recall, F1-score, and support for each class.
* ROC Curve: A plot of the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.
* AUC Score: The Area Under the ROC Curve, indicating the model's ability to distinguish between classes.


# Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository

2. Create a new branch
> git checkout -b feature/your-feature-name

3. Commit your changes
> git commit -am 'Add your message here'

4. Push to the branch
> git push origin feature/your-feature-name

5. Open a Pull Request
