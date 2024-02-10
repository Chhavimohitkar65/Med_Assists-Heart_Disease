from flask import Flask, render_template, request
import pickle
import numpy as np
from ML import preprocess_data

app = Flask(__name__, template_folder='templates')

# Load the model
try:
    with open('ml.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    print("Error: 'ml.pkl' file not found. Make sure the file exists in the same directory as your script.")
    model = None
except Exception as e:
    print(f"An error occurred while loading the model: {str(e)}")
    model = None

if model is None:
    print("Model is not loaded successfully. Check the model file and loading code.")

# Explanation of medical terms
attribute_explanations = {
    'attribute_1': 'Age (in years)',
    'attribute_2': 'Sex (1: male, 0: female)',
    'attribute_3': 'Chest Pain Type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)',
    'attribute_4': 'Resting Blood Pressure (in mm Hg)',
    'attribute_5': 'Serum Cholesterol (in mg/dl)',
    'attribute_6': 'Fasting Blood Sugar > 120 mg/dl (1: true, 0: false)',
    'attribute_7': 'Resting Electrocardiographic Results (0: normal, 1: having ST-T wave abnormality, 2: showing probable or definite left ventricular hypertrophy)',
    'attribute_8': 'Maximum Heart Rate Achieved',
    'attribute_9': 'Exercise Induced Angina (1: yes, 0: no)',
    'attribute_10': 'ST Depression Induced by Exercise Relative to Rest',
    'attribute_11': 'Peak Exercise ST Segment Slope (0: upsloping, 1: flat, 2: downsloping)',
    'attribute_12': 'Number of Major Vessels (0-3) Colored by Flourosopy',
    'attribute_13': 'Thalassemia (0: normal, 1: fixed defect, 2: reversible defect)'
}


@app.route('/')
def index():
    return render_template('index.html', explanations=attribute_explanations)

@app.route('/predict', methods=['POST'])
def predict():
    # Extracting data from the form and converting to float
    data = [float(request.form[f'attribute_{i}']) for i in range(1, 14)]

    # Preprocess the data
    processed_data = preprocess_data(np.array([data]))

    # Make prediction
    pred = model.predict(processed_data)

    # Render the result template with prediction
    return render_template('after.html', prediction=pred[0])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
