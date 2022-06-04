import pickle

import pandas as pd
from matplotlib import pyplot as plt, pyplot
import numpy as np
from scipy.stats import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import auc, accuracy_score, f1_score, log_loss, classification_report, roc_auc_score, \
    precision_score, recall_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn import decomposition, model_selection, metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
import warnings

import base_models as bm
import stack_ensemble as stack
import ann_models as ann_models
import stack_ensemble_nn as stack_nn


warnings.filterwarnings('ignore')


def check_base_models():
    check_model_list = bm.get_models()
    results, names = list(), list()
    for name, model in check_model_list.items():
        scores = evaluate_model(model, features_train, final_status_train)
        results.append(scores)
        names.append(name)
        print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))


def feature_selection(X, y):
    feature_model = ExtraTreesClassifier()
    feature_model.fit(X, y)
    print(feature_model.feature_importances_)
    feat_importances = pd.Series(feature_model.feature_importances_, index=X.columns)
    feat_importances.nlargest(5).plot(kind='barh')
    plt.show()


# evaluate a give model using cross-validation
def evaluate_model(evaluate_models, X, y):
    cv = model_selection.KFold(n_splits=10, random_state=None)
    model_scores = model_selection.cross_val_score(evaluate_models, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return model_scores


# calculating of BMI
def calculate_bmi():
    heart['height_m'] = heart['height'].apply(lambda x: calculate_height(x))

    # calculating bmi
    heart['bmi'] = heart['weight'] / heart['height_m']
    print(heart)


def calculate_height(height):
    height = (height/100) ** 2
    return height


# convert age to years since its given in days
def convert_age_years(agey):
    agey = agey.apply(lambda x: round(x / 365))
    return agey


# checking for duplicates
def check_duplicates(heart_data):
    duplicate_sum = heart_data.duplicated().sum()
    if duplicate_sum:
        print('Duplicates Rows in Dataset are : {}'.format(duplicate_sum))
    else:
        print('Dataset contains no Duplicate Values')
    duplicated = heart_data[heart_data.duplicated(keep=False)]
    duplicated = duplicated.sort_values(by=['gender', 'height', 'weight'], ascending=False)
    print(duplicated.head())


def check_outlier_bp(sys, dia):
    outliers = (sys >= 370) | (sys <= 70) | (dia >= 360) | (dia <= 50)
    return outliers


def check_outlier_age(age):
    age_out = (age > 100) | (age <= 0)
    return age_out


if __name__ == '__main__':
 heart = pd.read_csv("D:/NSBM/NSBM/year 4/research/heart disease prediction/data/cardio_train.csv", sep=';')
 print(heart.head())

# heart status 0= no disease 1= yes disease

# convert age to years since its given in days
 heart['age'] = convert_age_years(heart['age'])

# renaming the features
 heart = heart.rename(
     columns={"ap_hi": "sys_blood_pressure", "ap_lo": "dia_blood_pressure", "gluc": "glucose",
              "cardio": "heart_status"})


# to check whether there any null values
 print(heart.isna().sum())

# identify duplicates
 check_duplicates(heart)

# Outlier detection and Removal
 outlier = check_outlier_bp(heart['sys_blood_pressure'], heart['dia_blood_pressure'])
# print("blood pressure related outliers : " + str(heart[outlier].count()))
 heart = heart[~outlier]

 outlier_age = check_outlier_age(heart['age'])


# getting the BMI cause it is also a contributing factor
# getting the formatted height
 calculate_bmi()
 plt.scatter(heart['heart_status'], pd.to_numeric(heart['sys_blood_pressure']), label='Systolic')
 plt.scatter(heart['heart_status'], heart['bmi'], label='bmi')
 plt.legend()
 plt.show()

# drop unwanted columns
 heart.drop(['height', 'weight', 'id', 'height_m', 'dia_blood_pressure', 'alco', 'active'], axis=1, inplace=True)

# Plotting attrition of employees
 fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(14, 6))

 ax1 = heart['heart_status'].value_counts().plot.pie(x="Heart disease", y='no.of patients', autopct="%1.0f%%", labels=["Heart Disease", "Normal"], startangle=60, ax=ax1)
 ax1.set(title='Percentage of Heart disease patients in Dataset')

 ax2 = heart["heart_status"].value_counts().plot(kind="barh", ax=ax2)
 for i, j in enumerate(heart["heart_status"].value_counts().values):
     ax2.text(.5, i, j, fontsize=12)
     ax2.set(title='No. of Heart disease patients in Dataset')
     plt.show()


# one-hot encoding for categorical features
# heart = pd.get_dummies(heart, columns=['gender', 'smoke'], drop_first=True)

 print(heart.info())
 print(heart.head())

# separating the target column from other features
 final_status = heart['heart_status']
 features = heart.drop(['heart_status'], axis=1)

 feature_selection(features, final_status)

 column_length = len(features.columns)

 print(features.info())

# splitting features and target into train, test sets
 features_train, features_test, final_status_train, final_status_test = train_test_split(features, final_status, test_size=0.2, random_state=42)

 scaler = MinMaxScaler()
 features_train = scaler.fit_transform(features_train)
 features_test = scaler.fit_transform(features_test)

 print('------------Training Set------------------')
 print(features_train.shape)
 print(final_status_train.shape)

 print('------------Test Set------------------')
 print(features_test.shape)
 print(final_status_test.shape)

 x_train = np.reshape(features_train, (features_train.shape[0], 1, features_train.shape[1]))
 x_test = np.reshape(features_test, (features_test.shape[0], 1, features_test.shape[1]))
 print(x_train.shape)
 print(x_test.shape)

 check_base_models()

 svc = SVC(gamma="auto", kernel="linear", probability=True)
 svc.fit(features_train, final_status_train)
 svc_pred = svc.predict(features_test)
 probs_ml_svc = svc.predict_proba(features_test)

 RF = RandomForestClassifier(n_estimators=500, criterion="gini", max_depth=10)
 RF.fit(features_train, final_status_train)
 rf_pred = RF.predict(features_test)
 probs_ml_RF = RF.predict_proba(features_test)

 lr = LogisticRegression()
 lr.fit(features_train, final_status_train)
 lr_pred = lr.predict(features_test)
 probs_ml_lr = lr.predict_proba(features_test)

 SDA = SGDClassifier(max_iter=1000, tol=1e-4)
 SDA.fit(features_train, final_status_train)
 SDA_pred = SDA.predict(features_test)

 ada = AdaBoostClassifier()
 ada.fit(features_train, final_status_train)
 ada_pred = ada.predict(features_test)

 knn12 = KNeighborsClassifier(12)
 knn12.fit(features_train, final_status_train)
 knn12_pred = knn12.predict(features_test)
 probs_ml_knn = knn12.predict_proba(features_test)

 DT = DecisionTreeClassifier(criterion="entropy", max_depth=5)
 DT.fit(features_train, final_status_train)
 DT_pred = DT.predict(features_test)

 LDA = LinearDiscriminantAnalysis()
 LDA.fit(features_train, final_status_train)
 LDA_pred = LDA.predict(features_test)
 probs_ml_LDA = LDA.predict_proba(features_test)

 ET = ExtraTreesClassifier(max_depth=10)
 ET.fit(features_train, final_status_train)
 ET_pred = ET.predict(features_test)

 XGB = xgb.XGBClassifier(use_label_encoder=False)
 XGB.fit(features_train, final_status_train)
 xgb_pred = XGB.predict(features_test)
 probs_ml_XGB = XGB.predict_proba(features_test)

 GBC = GradientBoostingClassifier(n_estimators=200, max_features="sqrt", learning_rate=0.5)
 GBC.fit(features_train, final_status_train)
 GBC_pred = GBC.predict(features_test)
 probs_ml_GBC = GBC.predict_proba(features_test)

 naive = GaussianNB()
 naive.fit(features_train, final_status_train)
 naive_pred = naive.predict(features_test)
 probs_ml_naive = naive.predict_proba(features_test)

# Selected Base Models according to performance
 level0 = list()
 level0.append(('bayes', GaussianNB()))
 level0.append(('lr', LogisticRegression()))
 level0.append(('RF', RandomForestClassifier(n_estimators=500, criterion="gini", max_depth=10, max_features="auto",
                                             min_samples_leaf=0.005, min_samples_split=0.005, n_jobs=-1,
                                             random_state=1000)))
 level0.append(('KNN', KNeighborsClassifier(12)))
 level0.append(('LDA', LinearDiscriminantAnalysis()))
 level0.append(('XGB', xgb.XGBClassifier(use_label_encoder=False)))
 level0.append(('svc', SVC(C=50, degree=1, gamma="auto", kernel="linear", probability=True)))
 level0.append(('gb', GradientBoostingClassifier(n_estimators=200, max_features='sqrt', learning_rate=0.5)))

# define meta learner model
 level1 = LogisticRegression()

 base_ann_models = ann_models.train_ann(features_train, features_test, final_status_train, final_status_test,
                                        column_length)

 stack_nn_model = stack_nn.stack_model_nn(base_ann_models, features_test, final_status_test)

 stack_model = stack.stack_ensemble(features_train, features_test, final_status_train, final_status_test, level0,
                                    level1)

 stack_nn.stacked_prediction(base_ann_models, stack_nn_model, features_test, final_status_test)

# Define parameter grid
 params = {"meta_classifier__kernel": ["linear", "rbf", "poly"],
           "meta_classifier__C": [1, 2],
           "meta_classifier__degree": [3, 4, 5],
           "meta_classifier__probability": [True]}

 Pkl_model_stack = 'stack_model_ML.pkl'
 file_path = 'D:/python/Cardio/model_three/baseModels/'
 with open(file_path + Pkl_model_stack, 'wb') as file:
     pickle.dump(stack_model, file)

 STACK_ML = pickle.load(open('D:/python/Cardio/model_three/baseModels/stack_model_ML.pkl', 'rb'))

 array = scaler.transform([[52, 2, 90, 1, 0, 1, 32.4]])

 print("HeartModelOne Home dataset, new prediction using STACK ML :", STACK_ML.predict(array))
 probs_ml = STACK_ML.predict_proba(array)
 print(probs_ml[0] * 100)
 positive_prob_ml = probs_ml[:, 1]
 print("positive prob ml :", positive_prob_ml)

 probs_dl, prediction = stack_nn.prediction_realtime(base_ann_models, stack_nn_model, array)
 print("HeartModelOne Home dataset, new prediction using STACK DL :", prediction)
 print(probs_dl[0] * 100)
 positive_prob_dl = probs_dl[:, 1]
 print("positive prob dl :", positive_prob_dl)

 avg_prob = (probs_ml + probs_dl) / 2
 print("HeartModelOne Home dataset, average prob", avg_prob[:, 1])







