import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def preprocess_data(data):
    # Define your preprocessing steps here
    return data

# Load your dataset
heart_data = pd.read_csv('Heart_Disease_Prediction.csv')

# Preprocess your data
X = preprocess_data(heart_data.drop(columns='Heart Disease', axis=1))
Y = heart_data['Heart Disease']

# Split the data into train and test sets
X_train , X_test, Y_train , Y_test = train_test_split(X, Y, test_size=0.2 , stratify=Y)

# Initialize and train your model
model = LogisticRegression()
model.fit(X_train , Y_train)

# Save the model
import pickle
pickle.dump(model, open('ml.pkl', 'wb'))
