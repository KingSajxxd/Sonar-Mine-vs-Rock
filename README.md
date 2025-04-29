# 🎯 Sonar-Mine-vs-Rock

A machine learning project using **Logistic Regression** to classify sonar signals as **rocks or mines**. Includes data preprocessing, model training, evaluation, and an interactive CLI-based prediction system. Built with **Python**, **NumPy**, **Pandas**, and **scikit-learn**.

---

## 📖 Description

This project implements a binary classification system using sonar data to distinguish between underwater **rocks** and **mines**. It demonstrates an end-to-end machine learning pipeline — from data loading and exploration to training, evaluating, and predicting with a Logistic Regression model.

---

## 📁 Dataset

- Source: [UCI Machine Learning Repository – Sonar Dataset](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks))
- 60 numeric features representing sonar energy readings
- Target label:
  - `R` = Rock
  - `M` = Mine

---

## 🛠️ Installation

Install the required packages using `pip`:

```bash
pip install numpy pandas scikit-learn
```

## 🚀 How to Run
Clone the repository:

```bash
git clone https://github.com/KingSajxxd/Sonar-Mine-vs-Rock.git
cd Sonar-Mine-vs-Rock
```
Make sure sonar_data.csv is present in the root directory.

Run the `Python` script:

```bash
python main.py
```
When prompted, enter 60 comma-separated float values (sonar readings) to classify the object.

## 🧠 Model Used
Logistic Regression

Simple and effective binary classification model

Suitable for linearly separable datasets

## 📊 Output
Accuracy on training and test sets

Interactive CLI prediction output:
```
"The object is a Rock" or

"The object is a Mine"
```
## 📂 Project Structure
```bash
├── sonar_data.csv           # Dataset file
├── sonar_classifier.py      # Main Python script
└── README.md                # Project documentation
```
