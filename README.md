# Email-sms-Classifier
This project implements a machine learning model to classify SMS messages as spam or ham (not spam). It uses the Naive Bayes algorithm for classification.

## Overview

The goal of this project is to build a model that can accurately distinguish between spam and legitimate SMS messages.

## Key Features

* **Data Loading:** The project loads the SMS Spam Collection dataset from a CSV file named `spam.csv` using Pandas. The file is read with `encoding='latin-1'` to handle specific character encoding. Unnecessary columns ('Unnamed: 2', 'Unnamed: 3', and 'Unnamed: 4') are dropped.
* **Data Preprocessing:**
    * The columns 'v1' and 'v2' are renamed to 'target' and 'text', respectively.
    * The 'target' column is label-encoded, converting 'spam' to 1 and 'ham' to 0.
* **Data Splitting:** The dataset is split into training and testing sets using `train_test_split` from scikit-learn.
* **Model Training:** A Gaussian Naive Bayes model (`GaussianNB`) is trained on the training data.
* **Model Evaluation:** The model's performance is evaluated using:
    * Accuracy score (`accuracy_score`)
    * Confusion matrix (`confusion_matrix`)
    * Precision score (`precision_score`)

## Technologies Used

* Python
* Pandas
* Scikit-learn (`train_test_split`, `GaussianNB`, `accuracy_score`, `confusion_matrix`, `precision_score`)
* NumPy

## Getting Started

1.  **Ensure you have the required libraries installed.** You can install them using pip:

    ```bash
    pip install pandas scikit-learn numpy
    ```

2.  **Locate the dataset file** (`spam.csv`).
3.  **Run the Python script** (`Email as spam or not.ipynb`).

## Results

The project outputs the accuracy, confusion matrix, and precision score of the Gaussian Naive Bayes model on the test set.
