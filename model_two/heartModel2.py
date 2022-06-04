import pickle

import pandas as pd
from keras.models import load_model
from matplotlib import pyplot as plt, pyplot
import numpy as np
from scipy.stats import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import auc, accuracy_score, f1_score, log_loss, classification_report, roc_auc_score, \
    precision_score, recall_score, matthews_corrcoef, roc_curve
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
import evaluation as xx


warnings.filterwarnings('ignore')


def check_base_models():
    base = bm.BaseModels()
    check_model_list = base.get_models()
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
    heart = pd.read_csv("framingham_train_preprocessed.csv")
    print(heart.head())

    # heart status 0= no disease 1= yes disease

    # to check whether there any null values
    print(heart.isna().sum())

    # identify duplicates
    # check_duplicates(heart)

    heart.drop(['id', 'diaBP', 'education'], axis=1, inplace=True)

    # Plotting attrition of employees
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(14, 6))

    ax1 = heart['heart_status'].value_counts().plot.pie(x="Heart disease", y='no.of patients', autopct="%1.0f%%",
                                                        labels=["Heart Disease", "Normal"], startangle=60, ax=ax1)
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
    features_train, features_test, final_status_train, final_status_test = train_test_split(features, final_status,
                                                                                            test_size=0.2,
                                                                                            random_state=42)
    print(features_train.info())
    print(final_status_test)

    scaler = MinMaxScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.fit_transform(features_test)

    scaler_model2 = 'scaler2.sav'
    file_path = 'D:/python/Cardio/model_two/baseModels/'
    with open(file_path + scaler_model2, 'wb') as file:
        pickle.dump(scaler, file)

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
    #
    # # Selected Base Models according to performance
    level0 = list()
    level0.append(('bayes', GaussianNB()))
    level0.append(('lr', LogisticRegression()))
    level0.append(('RF', RandomForestClassifier(n_estimators=500, criterion="gini", max_depth=10)))
    level0.append(('KNN', KNeighborsClassifier(12)))
    level0.append(('XGB', xgb.XGBClassifier(use_label_encoder=False)))
    level0.append(('svc', SVC(gamma="auto", kernel="linear", probability=True)))
    level0.append(('LDA', LinearDiscriminantAnalysis()))
    level0.append(('gb', GradientBoostingClassifier(n_estimators=200, max_features="sqrt", learning_rate=0.5)))

    # # define meta learner model
    level1 = LogisticRegression()

    base_ann_models = ann_models.train_ann(features_train, features_test, final_status_train, final_status_test, column_length)

    stack_nn_model = stack_nn.stack_model_nn(base_ann_models, features_test, final_status_test)

    stack_model, stack_prob = stack.stack_ensemble(features_train, features_test, final_status_train, final_status_test, level0, level1)

    prediction, proba_dl = stack_nn.stacked_prediction(base_ann_models, stack_nn_model, features_test, final_status_test)

    Pkl_model_stack = 'stack_model_ML2.pkl'
    file_path = 'D:/python/Cardio/model_two/baseModels/'
    with open(file_path + Pkl_model_stack, 'wb') as file:
        pickle.dump(stack_model, file)

    # saving the meta classifier in stacked DL model
    Pkl_DL_LR_model_stack = 'meta_model_DL2.pkl'
    file_path = 'D:/python/Cardio/model_two/baseModels/'
    with open(file_path + Pkl_DL_LR_model_stack, 'wb') as file:
        pickle.dump(stack_nn_model, file)
    #
    STACK_ML = pickle.load(open('D:/python/Cardio/model_two/baseModels/stack_model_ML2.pkl', 'rb'))
    #
    STACK_DL_meta = pickle.load(open('D:/python/Cardio/model_two/baseModels/meta_model_DL2.pkl', 'rb'))
    #
    DL_models = ann_models.load_all_models(3)
    scaler2 = pickle.load(open('D:/python/Cardio/model_two/baseModels/scaler2.sav', 'rb'))
    #

    array = [[1, 51, 1, 43, 0, 0, 0, 0, 207, 126.5, 19.71, 65, 68]]
    array = np.array(array)
    scale = scaler2.transform(array)
    #
    print("HeartModelTwo Home dataset, new prediction using STACK ML :", STACK_ML.predict(scale))
    probs_ml = STACK_ML.predict_proba(array)
    print(probs_ml[0] * 100)
    positive_prob_ml = probs_ml[:, 1]
    print("positive prob ml :", positive_prob_ml)
    #
    probs_dl, prediction = stack_nn.prediction_realtime(DL_models, STACK_DL_meta, scale)
    print("HeartModelTwo Home dataset, new prediction using STACK DL :", prediction)
    # # # probs = STACK_DL.predict_proba(array)
    print(probs_dl[0] * 100)
    positive_prob_dl = probs_dl[:, 1]
    print("positive prob dl :", positive_prob_dl)
    # #
    avg_prob = (probs_ml + probs_dl) / 2
    print("HeartModelTwo Home dataset, average prob", avg_prob[:, 1])
    #

    model_results = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1 Score',
                                          'ROC', 'Log_Loss', 'mathew_corrcoef'])

    data = {'Naive Bayes': naive_pred,
            'Logistic Regression': lr_pred,
            'Random Forest': rf_pred,
            'SVM': svc_pred,
            'KNN': knn12_pred,
            'LDA': LDA_pred,
            'XGBoost': xgb_pred,
            'GBM': GBC_pred}

    models = pd.DataFrame(data)

    for column in models:
        CM = confusion_matrix(final_status_test, models[column])

        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        specificity = TN / (TN + FP)
        loss_log = log_loss(final_status_test, models[column])
        acc = accuracy_score(final_status_test, models[column])
        roc = roc_auc_score(final_status_test, models[column])
        prec = precision_score(final_status_test, models[column])
        rec = recall_score(final_status_test, models[column])
        f1 = f1_score(final_status_test, models[column])

        mathew = matthews_corrcoef(final_status_test, models[column])
        results = pd.DataFrame([[column, acc, prec, rec, specificity, f1, roc, loss_log, mathew]],
                               columns=['Model', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1 Score',
                                        'ROC',
                                        'Log_Loss', 'mathew_corrcoef'])
        model_results = model_results.append(results, ignore_index=True)

    print(model_results)
    model_results.to_csv("model results model 2")

    # roc curve for models
    fpr1, tpr1, thresh1 = roc_curve(final_status_test, probs_ml_lr[:, 1], pos_label=1)
    fpr2, tpr2, thresh2 = roc_curve(final_status_test, probs_ml_naive[:, 1], pos_label=1)
    fpr3, tpr3, thresh3 = roc_curve(final_status_test, probs_ml_svc[:, 1], pos_label=1)
    fpr4, tpr4, thresh4 = roc_curve(final_status_test, probs_ml_RF[:, 1], pos_label=1)
    fpr5, tpr5, thresh5 = roc_curve(final_status_test, probs_ml_LDA[:, 1], pos_label=1)
    fpr6, tpr6, thresh6 = roc_curve(final_status_test, probs_ml_XGB[:, 1], pos_label=1)
    fpr7, tpr7, thresh7 = roc_curve(final_status_test, probs_ml_GBC[:, 1], pos_label=1)
    fpr8, tpr8, thresh8 = roc_curve(final_status_test, probs_ml_knn[:, 1], pos_label=1)
    fpr9, tpr9, thresh9 = roc_curve(final_status_test, stack_prob[:, 1], pos_label=1)

    # roc curve for tpr = fpr
    random_probs = [0 for i in range(len(final_status_test))]
    p_fpr, p_tpr, _ = roc_curve(final_status_test, random_probs, pos_label=1)

    # # auc roc score
    # auc_score1 = roc_auc_score(features_test, probs_ml[:, 1])
    # print("STACK ML MODEL 3 ROC SCORE :", auc_score1)

    # plot roc curves
    plt.plot(fpr1, tpr1, linestyle='--', color='orange', label='LR')
    plt.plot(fpr2, tpr2, linestyle='--', color='green', label='naive')
    plt.plot(fpr3, tpr3, linestyle='--', color='darkcyan', label='svc')
    plt.plot(fpr4, tpr4, linestyle='--', color='yellow', label='RF')
    plt.plot(fpr5, tpr5, linestyle='--', color='pink', label='LDA')
    plt.plot(fpr6, tpr6, linestyle='--', color='purple', label='XGB')
    plt.plot(fpr7, tpr7, linestyle='--', color='brown', label='GBC')
    plt.plot(fpr8, tpr8, linestyle='--', color='blue', label='KNN')
    plt.plot(fpr9, tpr9, linestyle='--', color='darkred', label='STACK ML')
    plt.plot(p_fpr, p_tpr, linestyle='--', color='black')
    # title
    plt.title('ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')

    plt.legend(loc='best')
    plt.savefig('ROC', dpi=300)
    plt.show()

# stack DL and stack ML
    fpr10, tpr10, thresh10 = roc_curve(final_status_test, proba_dl[:, 1], pos_label=1)
    random_probs = [0 for i in range(len(final_status_test))]
    p_fpr1, p_tpr1, _ = roc_curve(final_status_test, random_probs, pos_label=1)

    plt.plot(fpr9, tpr9, linestyle='--', color='orange', label='stack ML')
    plt.plot(fpr10, tpr10, linestyle='--', color='green', label='stack DL')
    plt.plot(p_fpr1, p_tpr1, linestyle='--', color='black')
    # title
    plt.title('ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')

    plt.legend(loc='best')
    plt.savefig('ROC', dpi=300)
    plt.show()


