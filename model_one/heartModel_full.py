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


if __name__ == '__main__':
    heart = pd.read_csv("D:/NSBM/NSBM/year 4/research/heart disease prediction/data/heart_train.csv")
    print(heart.head())

    # heart status 0= no disease 1= yes disease

    heart = heart.rename(
        columns={"cp": "chest_pain", "thalach": "maxHeartRate", "trestbps": "blood_pressure", "fbs": "blood_sugar",
                 "ca": "vessels", "chol": "cholesterol", "sex": "gender", "target": "heart_status",
                 "oldpeak": "st_depression"})

    print("\n")

    print(heart.head())

    # Checking missing entries in the dataset column wise by referring to method in heartDataAnalysis
    print(heart.isna().sum())

    # first checking the shape of the dataset
    print(heart.shape)

    # summary statistics of numerical columns
    print(heart.describe(include=[np.number]))

    # Plotting attrition of employees
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(14, 6))

    ax1 = heart['heart_status'].value_counts().plot.pie(x="Heart disease", y='no.of patients',
                                                        autopct="%1.0f%%", labels=["Heart Disease", "Normal"],
                                                        startangle=60,
                                                        ax=ax1)
    ax1.set(title='Percentage of Heart disease patients in Dataset')

    ax2 = heart["heart_status"].value_counts().plot(kind="barh", ax=ax2)
    for i, j in enumerate(heart["heart_status"].value_counts().values):
        ax2.text(.5, i, j, fontsize=12)
    ax2.set(title='No. of Heart disease patients in Dataset')
    plt.show()

    # filtering numeric features as age , resting bp, cholesterol and max heart rate achieved has outliers as per EDA
    heart_numeric = heart[['age', 'blood_pressure', 'cholesterol', 'maxHeartRate']]
    print(heart_numeric.head())

    # calculating zscore of numeric columns in the dataset
    z = np.abs(stats.zscore(heart_numeric))
    print(z)

    threshold = 3
    print(np.where(z > 3))

    # filtering outliers retaining only those data points which are below threshold
    heart = heart[(z < 3).all(axis=1)]
    print(heart.shape)

    # segregating dataset into features i.e., X and target variables i.e., y
    features = heart.drop(['heart_status'], axis=1)
    final_status = heart['heart_status']

    feature_selection(features, final_status)

    print(features.isna().sum())

    column_length = len(features.columns)

    features_train, features_test, final_status_train, final_status_test = train_test_split(features, final_status,
                                                                                            stratify=final_status,
                                                                                            test_size=0.2, shuffle=True,
                                                                                            random_state=5)
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

    svc = SVC(C=50, degree=1, gamma="auto", kernel="rbf", probability=True)
    svc.fit(features_train, final_status_train)
    svc_pred = svc.predict(features_test)

    MLP = MLPClassifier(activation="relu", alpha=0.1, hidden_layer_sizes=(10, 10, 10), learning_rate="constant", max_iter=2000, random_state=1000)
    MLP.fit(features_train, final_status_train)
    MLP_pred = MLP.predict(features_test)

    RF = RandomForestClassifier(n_estimators=500, criterion="gini", max_depth=10, max_features="auto", min_samples_leaf=0.005, min_samples_split=0.005, n_jobs=-1, random_state=1000)
    RF.fit(features_train, final_status_train)
    rf_pred = RF.predict(features_test)

    lr = LogisticRegression()
    lr.fit(features_train, final_status_train)
    lr_pred = lr.predict(features_test)

    SDA = SGDClassifier(max_iter=1000, tol=1e-4)
    SDA.fit(features_train, final_status_train)
    SDA_pred = SDA.predict(features_test)

    ada = AdaBoostClassifier()
    ada.fit(features_train, final_status_train)
    ada_pred = ada.predict(features_test)

    knn9 = KNeighborsClassifier(9)
    knn9.fit(features_train, final_status_train)
    knn9_pred = knn9.predict(features_test)

    DT = DecisionTreeClassifier(criterion="entropy", max_depth=5)
    DT.fit(features_train, final_status_train)
    DT_pred = DT.predict(features_test)

    LDA = LinearDiscriminantAnalysis()
    LDA.fit(features_train, final_status_train)
    LDA_pred = LDA.predict(features_test)

    ET = ExtraTreesClassifier(max_depth=10)
    ET.fit(features_train, final_status_train)
    ET_pred = ET.predict(features_test)

    XGB500 = xgb.XGBClassifier(n_estimators=500, use_label_encoder=False)
    XGB500.fit(features_train, final_status_train)
    y_pred_rfe = XGB500.predict(features_test)

    GBC = GradientBoostingClassifier(n_estimators=200, max_features='sqrt')
    GBC.fit(features_train, final_status_train)
    GBC_pred = GBC.predict(features_test)

    # Selected Base Models according to performance
    level0 = list()
    level0.append(('MLP', MLPClassifier(activation="relu", alpha=0.1, hidden_layer_sizes=(10, 10, 10),learning_rate="constant", max_iter=2000, random_state=1000)))
    level0.append(('lr', LogisticRegression()))
    level0.append(('RF', RandomForestClassifier(n_estimators=500, criterion="gini", max_depth=10, max_features="auto",min_samples_leaf=0.005, min_samples_split=0.005, n_jobs=-1,random_state=1000)))
    level0.append(('KNN', KNeighborsClassifier(9)))
    level0.append(('ET', ExtraTreesClassifier(max_depth=10)))
    level0.append(('xgb', xgb.XGBClassifier(n_estimators=500, use_label_encoder=False)))
    level0.append(('sgd', SGDClassifier(max_iter=1000, tol=1e-4)))
    level0.append(('svc', SVC(C=50, degree=1, gamma="auto", kernel="rbf", probability=True)))
    level0.append(('ada', AdaBoostClassifier()))
    level0.append(('dt', DecisionTreeClassifier(criterion="entropy", max_depth=5)))
    level0.append(('LDA', LinearDiscriminantAnalysis()))
    level0.append(('gb', GradientBoostingClassifier(n_estimators=200, max_features='sqrt')))

    # define meta learner model
    level1 = LogisticRegression()

    stack_model = stack.stack_ensemble(features_train, features_test, final_status_train, final_status_test, level0, level1)

    base_ann_models = ann_models.train_ann(features_train, features_test, final_status_train, final_status_test, column_length)

    stack_nn_model = stack_nn.stack_model_nn(base_ann_models, features_test, final_status_test)
    stack_nn.stacked_prediction(base_ann_models, stack_nn_model, features_test, final_status_test)

    # Define parameter grid
    params = {"meta_classifier__kernel": ["linear", "rbf", "poly"],
              "meta_classifier__C": [1, 2],
              "meta_classifier__degree": [3, 4, 5],
              "meta_classifier__probability": [True]}

    Pkl_model_stack = 'stack_model_ML.pkl'
    file_path = 'D:/python/Cardio/baseModels/'
    with open(file_path+Pkl_model_stack, 'wb') as file:
        pickle.dump(stack_model, file)

    STACK_ML = pickle.load(open('D:/python/Cardio/baseModels/stack_model_ML.pkl', 'rb'))

    array = scaler.transform([[56, 0, 1, 140, 294, 0, 0, 153, 0, 1.3, 1, 0, 2]])

    print("new prediction using STACK ML :", STACK_ML.predict(array))
    probs_ml = STACK_ML.predict_proba(array)
    print(probs_ml[0] * 100)
    positive_prob_ml = probs_ml[:, 1]
    print("positive prob ml :", positive_prob_ml)

    probs_dl, prediction = stack_nn.prediction_realtime(base_ann_models, stack_nn_model, array)
    print("new prediction using STACK DL :", prediction)
    # probs = STACK_DL.predict_proba(array)
    print(probs_dl[0] * 100)
    positive_prob_dl = probs_dl[:, 1]
    print("positive prob dl :", positive_prob_dl)

    avg_prob = (probs_ml + probs_dl) / 2
    print("average prob", avg_prob[:, 1])

    # model = pickle.load(open('../stack_model.pkl', 'rb'))
    # print("new prediction :", model.predict(scaler.transform([[56,0,1,140,294,0,0,153,0,1.3,1,0,2]])))
    # probs = model.predict_proba(scaler.transform([[56,0,1,140,294,0,0,153,0,1.3,1,0,2]]))
    # print("new prediction :", stack_model.predict(scaler.transform([[56, 0, 1, 140, 294, 0, 0, 153, 0, 1.3, 1, 0, 2]])))
    # probs = stack_model.predict_proba(scaler.transform([[56, 0, 1, 140, 294, 0, 0, 153, 0, 1.3, 1, 0, 2]]))
    # print(probs[0] * 100)


