from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

# Load the saved models
loaded_lc_reg = joblib.load('logistic_regression_model.pkl')
loaded_rm_forest = joblib.load('random_forest_model.pkl')
loaded_lc_reg1 = joblib.load('logistic_regression_balanced_model.pkl')
loaded_rm_forest1 = joblib.load('random_forest_balanced_model.pkl')

# Load the saved preprocessor
full_processor = joblib.load('preprocessor_pipeline.pkl')

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    gender = request.form['gender']
    age = float(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    smoking_history = request.form['smoking_history']
    bmi = float(request.form['bmi'])
    HbA1c_level = float(request.form['HbA1c_level'])
    blood_glucose_level = int(request.form['blood_glucose_level'])

    # Create a new DataFrame with the input data
    new_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'smoking_history': [smoking_history],
        'bmi': [bmi],
        'HbA1c_level': [HbA1c_level],
        'blood_glucose_level': [blood_glucose_level]
    })

    # Preprocess the new data
    new_data['age-label'] = pd.cut(new_data['age'], bins=[0, 9, 19, 59, 100], labels=['Child', 'Young_Adult', 'Adult', 'Elderly'])
    new_data.drop(['age'], axis=1, inplace=True)

    # Apply the same transformations as before
    new_data_preprocessed = full_processor.transform(new_data)

    # Make predictions using the loaded models
    logistic_regression_probabilities = loaded_lc_reg.predict_proba(new_data_preprocessed)
    random_forest_probabilities = loaded_rm_forest.predict_proba(new_data_preprocessed)
    logistic_regression_balanced_probabilities = loaded_lc_reg1.predict_proba(new_data_preprocessed)
    random_forest_balanced_probabilities = loaded_rm_forest1.predict_proba(new_data_preprocessed)

    logistic_regression_percentages = logistic_regression_probabilities[:, 1] * 100
    random_forest_percentages = random_forest_probabilities[:, 1] * 100
    logistic_regression_balanced_percentages = logistic_regression_balanced_probabilities[:, 1] * 100
    random_forest_balanced_percentages = random_forest_balanced_probabilities[:, 1] * 100

    # Round the percentages to two decimal places
    logistic_regression_percentages = np.round(logistic_regression_percentages, 2)
    random_forest_percentages = np.round(random_forest_percentages, 2)
    logistic_regression_balanced_percentages = np.round(logistic_regression_balanced_percentages, 2)
    random_forest_balanced_percentages = np.round(random_forest_balanced_percentages, 2)

    # Prepare the prediction result for display
    result_lr = logistic_regression_percentages
    result_rf = random_forest_percentages
    result_lr_balanced = logistic_regression_balanced_percentages
    result_rf_balanced = random_forest_balanced_percentages

    return render_template('result.html', result_lr=result_lr, result_rf=result_rf,
                           result_lr_balanced=result_lr_balanced, result_rf_balanced=result_rf_balanced)
    

if __name__ == '__main__':
    app.run(debug=True)

