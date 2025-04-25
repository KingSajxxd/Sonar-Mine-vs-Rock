# Importing required dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
sonar_data = pd.read_csv('sonar_data.csv', header=None)

# ----- Data Overview Functions -----

def get_shape():
    print("Rows:", sonar_data.shape[0])
    print("Columns:", sonar_data.shape[1])

def get_head():
    print("\nFirst 5 rows of the dataset:")
    print(sonar_data.head())

def get_description():
    print("\nStatistical description:")
    print(sonar_data.describe())

def get_counts():
    print("\nValue counts (R = Rock, M = Mine):")
    print(sonar_data[60].value_counts())

def get_means():
    print("\nMean values grouped by label:")
    print(sonar_data.groupby(60).mean())

# ----- Data Preparation -----
# Separate features and labels
data = sonar_data.drop(columns=60, axis=1)
label = sonar_data[60]

# Train-test split
data_train, data_test, label_train, label_test = train_test_split(
    data, label, test_size=0.1, stratify=label, random_state=1
)

# ----- Model Training Function -----
def train_model():
    model = LogisticRegression(max_iter=1000)
    model.fit(data_train, label_train)
    return model

# ----- Accuracy Functions -----
def train_data_accuracy(model):
    predictions = model.predict(data_train)
    accuracy = accuracy_score(label_train, predictions)
    print("Accuracy score of the training data:", accuracy)

def test_data_accuracy(model):
    predictions = model.predict(data_test)
    accuracy = accuracy_score(label_test, predictions)
    print("Accuracy score of the testing data:", accuracy)

# ----- Prediction System Function -----
def predictive_system(model):
    input_data = input("Enter the data to predict (comma-separated numbers): ").split(',')
    input_data_as_numpy_array = np.asarray([float(i) for i in input_data])

    # Reshape the data for prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = model.predict(input_data_reshaped)
    print("\nPrediction:", prediction)

    if prediction[0] == 'R':
        print("The object is a Rock")
    else:
        print("The object is a Mine")

# ----- Run the System -----
if __name__ == "__main__":
    get_shape()
    get_head()
    get_description()
    get_counts()
    get_means()

    model = train_model()
    train_data_accuracy(model)
    test_data_accuracy(model)
    predictive_system(model)
