# -*- coding: utf-8 -*-

import numpy as np
import time
import pickle
from flask import Flask, request, render_template, flash
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import redirect

import stack_ensemble_nn as stack_nn
import ann_models as ann_models
import cardiologistInfo as cardio_info
import model_one.heartModel as HM
from sklearn.metrics import auc, accuracy_score, f1_score, log_loss


def check_risk_pos(ml, dl):
    if int(ml) > 85 or int(ml) == 85:
        ml_result = 'HIGH RISK'
    if int(dl) > 85 or int(dl) == 85:
        dl_result = 'HIGH RISK'
    if 70 < int(ml) < 85 or int(ml) == 70:
        ml_result = 'MEDIUM RISK'
    if 70 < int(dl) < 85 or int(dl) == 70:
        dl_result = 'MEDIUM RISK'
    if 50 < int(ml) < 70 or int(ml) == 50:
        ml_result = 'RISK'
    if 50 < int(dl) < 70 or int(dl) == 50:
        dl_result = 'RISK'
    if 45 < int(ml) < 50 or int(ml) == 45:
        ml_result = 'LOW RISK'
    if 45 < int(dl) < 50 or int(dl) == 45:
        dl_result = 'LOW RISK'
    return ml_result, dl_result


def check_risk_neg(ml, dl):
    if int(ml) < 50:
        ml_result = 'NO RISK'
    if int(dl) < 50:
        dl_result = 'NO RISK'
    return ml_result, dl_result


def render_resultPage(p_stack_ml, p_stack_dl, prob_ml, prob_dl):
    if p_stack_ml == 1 and p_stack_dl == 1:
        ml_res, dl_res = check_risk_pos(prob_ml, prob_dl)
        final_result = 'a risk'

    if p_stack_ml == 0 and p_stack_dl == 0:
        ml_res, dl_res = check_risk_neg(prob_ml, prob_dl)
        final_result = 'no risk'

    if (p_stack_ml == 0 and p_stack_dl == 1) or (p_stack_ml == 1 and p_stack_dl == 0):
        ml_res = 'uncertain'
        dl_res = 'uncertain'
        final_result = 'uncertain risk'

    return final_result, ml_res, dl_res


def models_load1(new_input):
    # Loading DL models
    n_models = 4

    # Load ML model
    STACK_ML = pickle.load(open('D:/python/Cardio/model_one/baseModels/stack_model_ML.pkl', 'rb'))
    model_name = 'model_one'
    DL_models = ann_models.load_all_models1(4, model_name)

    STACK_DL_meta = pickle.load(open('D:/python/Cardio/model_one/baseModels/meta_model_DL.pkl', 'rb'))

    scaler = pickle.load(open('D:/python/Cardio/model_one/baseModels/scaler1.sav', 'rb'))
    new_array = [[38, 1, 3, 138, 175, 0, 173, 0]]
    scale = scaler.transform(new_input)

    prediction_STACK_ML = STACK_ML.predict(scale)
    probs = STACK_ML.predict_proba(scale)

    probs_dl, prediction_STACK_DL = stack_nn.prediction_realtime(DL_models, STACK_DL_meta, scale)

    print("MODEL ONE predictions")

    print("STACK ML :", probs[0] * 100)
    positive_prob_ml = probs[:, 1] * 100
    print("positive prob ml :", positive_prob_ml)
    print(prediction_STACK_ML)
    print(positive_prob_ml)

    print("STACK DL :", probs_dl[0] * 100)
    positive_prob_dl = probs_dl[:, 1] * 100
    print("positive prob dl :", positive_prob_dl)
    print(prediction_STACK_DL)
    print(int(positive_prob_dl))

    return prediction_STACK_ML, prediction_STACK_DL, positive_prob_ml, positive_prob_dl


def models_load2(new_input):
    # Loading DL models
    n_models = 3

    # Load ML model
    STACK_ML = pickle.load(open('D:/python/Cardio/model_two/baseModels/stack_model_ML2.pkl', 'rb'))
    model_name = 'model_two'
    DL_models = ann_models.load_all_models1(4, model_name)

    STACK_DL_meta = pickle.load(open('D:/python/Cardio/model_two/baseModels/meta_model_DL2.pkl', 'rb'))

    scaler = pickle.load(open('D:/python/Cardio/model_two/baseModels/scaler2.sav', 'rb'))
    scale = scaler.transform(new_input)

    prediction_STACK_ML = STACK_ML.predict(scale)
    probs = STACK_ML.predict_proba(scale)
    probs_dl, prediction_STACK_DL = stack_nn.prediction_realtime(DL_models, STACK_DL_meta, scale)

    print("MODEL THREE predictions")

    print("STACK ML :", probs[0] * 100)
    positive_prob_ml = probs[:, 1] * 100
    print("positive prob ml :", positive_prob_ml)
    print(prediction_STACK_ML)
    print(positive_prob_ml)

    print("STACK DL :", probs_dl[0] * 100)
    positive_prob_dl = probs_dl[:, 1] * 100
    print("positive prob dl :", positive_prob_dl)
    print(prediction_STACK_DL)
    print(int(positive_prob_dl))

    return prediction_STACK_ML, prediction_STACK_DL, positive_prob_ml, positive_prob_dl


def models_load3(new_input):
    # Loading DL models
    n_models = 3

    # Load ML model
    STACK_ML = pickle.load(open('D:/python/Cardio/model_three/baseModels/stack_model_ML3.pkl', 'rb'))
    model_name = 'model_three'
    DL_models = ann_models.load_all_models1(4, model_name)

    STACK_DL_meta = pickle.load(open('D:/python/Cardio/model_three/baseModels/meta_model_DL3_COPY.pkl', 'rb'))

    scaler = pickle.load(open('D:/python/Cardio/model_three/baseModels/scaler3.sav', 'rb'))
    scale = scaler.transform(new_input)

    prediction_STACK_ML = STACK_ML.predict(scale)
    probs = STACK_ML.predict_proba(scale)
    probs_dl, prediction_STACK_DL = stack_nn.prediction_realtime(DL_models, STACK_DL_meta, scale)

    print("MODEL THREE predictions")

    print("STACK ML :", probs[0] * 100)
    positive_prob_ml = probs[:, 1] * 100
    print("positive prob ml :", positive_prob_ml)
    print(prediction_STACK_ML)
    print(positive_prob_ml)

    print("STACK DL :", probs_dl[0] * 100)
    positive_prob_dl = probs_dl[:, 1] * 100
    print("positive prob dl :", positive_prob_dl)
    print(prediction_STACK_DL)
    print(int(positive_prob_dl))

    return prediction_STACK_ML, prediction_STACK_DL, positive_prob_ml, positive_prob_dl


# average the stack ML and stack DL result if both are producing same 'risk type'
def averageResult(ml_res, dl_res, prob_ml, prob_dl):
    if ml_res == dl_res:
        avg_prob = (prob_dl + prob_ml)/2
        avg_risk_status = ml_res
    else:
        avg_prob = '0.0'
        avg_risk_status = 'different risk statuses'
    return avg_risk_status, avg_prob


# Create application
app = Flask(__name__)


# Bind home function to URL
@app.route('/')
def welcome():
    return render_template('welcome.html')


# Bind home function to URL
@app.route('/home')
def home():
    return render_template('home.html')


@app.route("/dashboard", methods=['GET', 'POST'])
def dashboard():
    return render_template("dashboard.html")


# select model out of three models
@app.route('/test_model', methods=['GET', 'POST'])
def test_model():
    model_number = request.form.get('models')
    print(model_number)
    if model_number == '1' or model_number is None:
        return model1()
    if model_number == '2':
        return model2()
    if model_number == '3':
        return model3()
    else:
        return 'error loading model'


@app.route("/result")
def result_page():
    return render_template("heartModel_results.html")


# Bind model 1 function to URL
@app.route('/model1', methods=['GET', 'POST'])
def model1():
    return render_template('heartModel1_form.html')
    # return render_template('heartModel1_form.html')


# Bind model 2 function to URL
@app.route('/model2', methods=['GET', 'POST'])
def model2():
    return render_template('heartModel2_form.html')


# Bind model 3 function to URL
@app.route('/model3', methods=['GET', 'POST'])
def model3():
    return render_template('heartModel3_form.html')


# Bind guideline function to URL
@app.route('/guidelines', methods=['GET', 'POST'])
def guidelines():
    return render_template('guidelines.html')


# Bind guideline function to URL
@app.route('/login', methods=['GET', 'POST'])
def login():
    username = request.form.get('email')
    password = request.form.get('pass')
    if username == 'kama@gmail.com' and password == '1234':
        return render_template('dashboard.html')
    else:
        msg = 'incorrect username or password'
    return render_template('login.html')


@app.route('/predict1', methods=['GET', 'POST'])
def predict1():
    start_time = time.time()
    # Put all form entries values in a list
    features = [float(i) for i in request.form.values()]
    new_list = [int(float(num)) for num in features]
    print(new_list)
    array = [46, 0, 1, 105, 204, 0, 172, 0]
    # Convert features to array
    array_features = np.array([new_list])
#  scale = scaler.fit_transform(array_features)
    print(array_features.reshape(-1, 1))
    pred_STACK_ML, pred_STACK_DL, pos_prob_ml, pos_prob_dl = models_load1(array_features)
    print("prediction completed")

    final_res, ml_result, dl_result = render_resultPage(pred_STACK_ML, pred_STACK_DL, pos_prob_ml, pos_prob_dl)

    average_status, average_prob = averageResult(ml_result, dl_result, pos_prob_ml, pos_prob_dl)

    execution_time = time.time() - start_time
    print(execution_time)

    return render_template('heartModel_results.html', resulting=final_res, result_ML=pos_prob_ml, risk_result_ML=ml_result, result_DL=pos_prob_dl, risk_result_DL=dl_result, result_AVG=average_prob, risk_result_AVG=average_status)


@app.route('/predict2', methods=['GET', 'POST'])
def predict2():
    # Put all form entries values in a list
    features = [float(i) for i in request.form.values()]
    new_list = [int(float(num)) for num in features]
    array_features = np.array([new_list])
    print(array_features)
    pred_STACK_ML, pred_STACK_DL, pos_prob_ml, pos_prob_dl = models_load2(array_features)
    print("prediction completed")
    final_res, ml_result, dl_result = render_resultPage(pred_STACK_ML, pred_STACK_DL, pos_prob_ml, pos_prob_dl)
    print("results calculated")

    average_status, average_prob = averageResult(ml_result, dl_result, pos_prob_ml, pos_prob_dl)

    return render_template('heartModel_results.html', resulting=final_res, result_ML=pos_prob_ml, risk_result_ML=ml_result, result_DL=pos_prob_dl, risk_result_DL=dl_result, result_AVG=average_prob, risk_result_AVG=average_status)


@app.route('/predict3', methods=['GET', 'POST'])
def predict3():
    start_time = time.time()
    # Put all form entries values in a list
    features = [float(i) for i in request.form.values()]
    new_list = [int(float(num)) for num in features]
    array_features = np.array([new_list])
    pred_STACK_ML, pred_STACK_DL, pos_prob_ml, pos_prob_dl = models_load3(array_features)
    print("prediction completed")
    final_res, ml_result, dl_result = render_resultPage(pred_STACK_ML, pred_STACK_DL, pos_prob_ml, pos_prob_dl)
    print("results calculated")

    average_status, average_prob = averageResult(ml_result, dl_result, pos_prob_ml, pos_prob_dl)

    execution_time = time.time() - start_time
    print(execution_time)
    return render_template('heartModel_results.html', resulting=final_res, result_ML=pos_prob_ml, risk_result_ML=ml_result, result_DL=pos_prob_dl, risk_result_DL=dl_result, result_AVG=average_prob, risk_result_AVG=average_status)


@app.route('/cardiologist_info', methods=['GET', 'POST'])
def cardiologist_info():
    df = cardio_info.ShowingResult()
    return render_template('viewCardiologist.html', tables=[df.to_html()])


@app.route('/cardio_map', methods=['GET', 'POST'])
def cardio_map():
    return render_template('map.html')


if __name__ == '__main__':
    app.run()




