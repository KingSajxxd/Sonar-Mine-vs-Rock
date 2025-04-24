# Importing required dependencies

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data collection and Data processing

# Csv to a panda dataframe
sonar_data = pd.read_csv('sonar_data.csv', header = None)


# Get the number of rows and columns
def get_shape():
    print("Rows: ", sonar_data.shape[0])
    print("Columns: ", sonar_data.shape[1])


def get_head():
# Head of the panda dataframe(fisrt 5 rows)
    print("\nHead:")
    print(sonar_data.head())

def get_description():
# Description of the panda dataframe
    print("\nDescription:")
    print(sonar_data.describe())

def get_count():
# Count of rocks and mines
# Taking index as 60 because there 61 columns but in python indexing starts from 0
    sonar_data[60].value_counts() 

def get_mean():
# Taking the mean of the datasets, Important for prediction
    sonar_data.groupby(60).mean()



# Seperating data and labels
data = sonar_data.drop(columns=60, axis=1)
label = sonar_data[60]

# Training and test data
data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.1, stratify=label, random_state=1)


def callmodel():
# Training the LogisticRegression model 
    model = LogisticRegression()
    model.fit(data_train, label_train)
    return model


# Checking the models accuracy
def train_data_accuracy(model):
# On training data
    data_train_prediction = model.predict(data_train)
    train_prediction_accuracy = accuracy_score(data_train_prediction, label_train)
    print("Accuracy of training data: ",train_prediction_accuracy)

def test_data_accuracy(model):
# On test data
    data_test_prediction = model.predict(data_test)
    test_prediction_accuracy = accuracy_score(data_test_prediction, label_test)
    print("Accuracy of training data: ",test_prediction_accuracy)


def system(model):
# Making a Predictive System
    input_data = input("Enter the data to predict: ").split(',')

# input_data to a numpy array
    input_data_as_numpy_array = np.asarray([float(i) for i in input_data])

# Reshaping the np array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 'R':
        print("The object is a Rock")
    else:
        print("The object is a Mine")


# System Start
system(callmodel())