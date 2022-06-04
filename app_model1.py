# -*- coding: utf-8 -*-
import pickle

import numpy as np
from flask import Flask, request, render_template
from mlxtend.regressor import StackingCVRegressor
from sklearn.metrics import auc, accuracy_score, f1_score, log_loss

# Create application
app = Flask(__name__)


# Bind home function to URL
@app.route('/')
def home():
    return render_template('Heart Disease Classifier.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Put all form entries values in a list 
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    print(array_features)

    # Load ML model
    meta_model = pickle.load(open('baseModels/MLP_Model.pkl', 'rb'))
    base_model1 = pickle.load(open('baseModels/RFC_Model.pkl', 'rb'))
    base_model2 = pickle.load(open('baseModels/KNN_Model.pkl', 'rb'))
    base_model3 = pickle.load(open('baseModels/ET500_Model.pkl', 'rb'))
    base_model4 = pickle.load(open('baseModels/ET1000_Model.pkl', 'rb'))
    base_model5 = pickle.load(open('baseModels/XGB_Model.pkl', 'rb'))
    base_model6 = pickle.load(open('baseModels/XGB2000_Model.pkl', 'rb'))
    base_model7 = pickle.load(open('baseModels/SGD_Model.pkl', 'rb'))
    base_model8 = pickle.load(open('baseModels/SVCLinear_Model.pkl', 'rb'))
    base_model9 = pickle.load(open('baseModels/ADA_Model.pkl', 'rb'))
    base_model10 = pickle.load(open('baseModels/DT_Model.pkl', 'rb'))
    base_model11 = pickle.load(open('baseModels/LDA_Model.pkl', 'rb'))
    base_model12 = pickle.load(open('baseModels/GBC_Model.pkl', 'rb'))
    stack_model = pickle.load(open('baseModels/stack_model.pkl', 'rb'))

    # selecting list of top performing models to be used in stacked ensemble method
    models = [
        base_model1, base_model2, base_model3, base_model4, base_model5, base_model6, base_model7, base_model8,
        base_model9, base_model10, base_model11,
        base_model12
    ]

    # Predict features
    prediction = meta_model.predict(array_features)
    probability = meta_model.predict_proba(array_features)
    print(probability[0] * 100)

    output = prediction

    # Check the output values and retrive the result with html tag based on the value
    if output == 1:
        return render_template('Heart Disease Classifier.html',
                               result='The patient is not likely to have heart disease!')
    else:
        return render_template('Heart Disease Classifier.html',
                               result='The patient is likely to have heart disease!')


if __name__ == '__main__':
    # Run the application
    app.run()

