# Detection of Cardiovascular Disease with Predictive Models 

## Project Overview 

This project uses machine learning algorithms to predict cardiovascular disease (CVD) risk based on data from the Behavioral Risk Factor Surveillance System (BRFSS). Several models are implemented to evaluate their performance in binary classification task for CVD risk prediction. The project focuses on the effectiveness of each model in terms of performance metrics scores in order to determine which method best identifies risk factors for coronary heart disease. The goal is to determine the risk of a person in developing a Cardiovascular Disease (CVD), such as heart attacks, based on features of their personal lifestyle and clinical factors.   

## Table of Contents
- [Project Overview](#project-overview)
- [Usage](#usage)
- [Data Description](#data-description)
- [Data Preprocessing and Feature Selection](#data-preprocessing-and-feature-selection)
- [Model Implementation and Training](#model-implementation-and-training)
- [Prediction and test](#results-and-comparative-analysis)
- [Conclusion](#conclusion)
- [References](#references)

## Installation 

Clone this repository, in order to have run the different code

```bash
git clone https://github.com/philippinelch/ML_project1
```

## Contents and Usage 

The following files are available in the repository 
- implementations.py : containing the different machine learning algorithm functions and other functions used in the training (for example function to compute the F1 score);
- Preprocessing_Functions.py : containing all the functions used for the preprocessing of the data; 
- run.ipynb : a notebook containing the pipeline the project : preprocessing, hyperparameter tuning, model training, model testing and predictions.
- helpers.py : containing functions to load the data and to create the final .csv for submissions

Run the script run.ipynb to preprocess the data, implement/train the models and get the final best predictions. *Pay attention to adapt the data_path depending on where the data are store.*

## Data Description 

The data used for this project are from the Behavioral Risk Factor Surveillance System (BRFSS). Data are in three different .csv file : x_train.csv, x_test.csv and y_train.csv.   

x_train and x_test are both composed of 321 columns containing informations (numerical and categorical data) on lifestyle and clinical factors of patients (each row corresponding to a person associated to an ID).  
y_train is composed of two column : ID (associated to patient) and MICHD (1 = patients have a coronary heart disease (MICHD); -1 = the person does not have a MICHD).  

Respondents (participants) were classified as having coronary heart disease (MICHD=1) if they reported having been told by a provider they had MICHD or if they reported having been told they had a heart attacks or angina.

## Data Preprocessing and Feature Selection 

The following steps detailed the data preprocessing and feature selection realized and developed in the file Preprocessing_functions.py : 

- Manual features selection (based on documentation and common knowledge)
- Remove column with to many NaN value
- Standardization of the data
- Hot-encoding for the categorical variable
- Replace the NaN by the mean value 
- Correlation to the output analysis 
- Statistical test relevance (T-test / Chi-Square test)
- Split the data between train set and validation set

## Models Implementation and training

6 different models are implemented, and train. In each models implementation there is a hyperparameters tuning and a performance metrics score analysis on the validation set realized. The goal is to identify the best model regarding our data, and use this model for future predictions. 

The different models (and the associated function) are: 
1. Gradient Descent Method (mean_squared_error_gd(y,tx,initialw,maxiters,gamma))
2. Stochastic Gradient Descent Method (mean_squared_error_sgd(y,tx,initialw,maxiters,gamma))
3. Least Squares Method (least_squares(y,tx))
4. Ridge Regression Method (ridge_regression(y,tx,lambda))
5. Logistic Regression (logistic_regression(y,tx,initialw,maxiters,gamma))
6. Regularized Logistic Regression (reg_logistic_regression(y,tx,lambda,initialw,maxiters,gamma))

Their functions are in the file implementations.py

Hyperparameters, such as learning rate (γ), regularization term (λ), and number of iterations, are tuned to optimize model performance. Tuning is based on F1 scores and validation accuracy to avoid overfitting.
The performance metric scores used are the F1 score, the accuracy and the loss.

## Prediction and Test 

The final step is to predict the output for the dataset x_test. To do so, the model showing the best scores in the training regarding the F1 score, the accuracy and the loss is used to predict the output. 

The performance of the prediction can be assess on the AIcrowd (https://www.aicrowd.com/challenges/epfl-machine-learning-project-1).

## Conclusion 

The project results highlight each predictive model's effectiveness in detecting CVD. Key findings include best practices for CVD prediction, areas for improvement, and further research on handling imbalanced data and refining hyperparameters.

