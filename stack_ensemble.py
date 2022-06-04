
from sklearn.metrics import auc, accuracy_score, f1_score, log_loss, classification_report, roc_auc_score, \
    precision_score, recall_score, matthews_corrcoef, confusion_matrix, roc_curve

import warnings

from sklearn.ensemble import StackingClassifier
import seaborn as sns
from matplotlib import pyplot as plt

# import python scripts
import base_models as bm
warnings.filterwarnings('ignore')


def stack_ensemble(X_train, X_test, y_train, y_test, level0, level1):
    stack = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    stack.fit(X_train, y_train)
    print(X_test.shape)
    pred = stack.predict(X_test)
    print(pred.shape)
    print("predict : ", accuracy_score(y_test, pred))
    probs_ml_stack = stack.predict_proba(X_test)
    check_built_model(y_test, pred, probs_ml_stack)
    print("stack done")
    return stack, probs_ml_stack


def check_built_model(final_status_test, y_pred, predict_proba):
    CM = confusion_matrix(final_status_test, y_pred)
    sns.heatmap(CM, annot=True, fmt='g')
    plt.show()
    check_parameters(CM, final_status_test, y_pred, predict_proba)


def check_parameters(CM, final_status_test, y_pred, predict_proba):
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    print("TP :", str(float(TP)))
    specificity = TN / float(TN + FP)
    sensitivity = TP / float (FN + TP)
    loss_log = log_loss(final_status_test, y_pred)
    acc = accuracy_score(final_status_test, y_pred)
    roc = roc_auc_score(final_status_test, y_pred)
    prec = precision_score(final_status_test, y_pred)
    rec = recall_score(final_status_test, y_pred)
    f1 = f1_score(final_status_test, y_pred)
    print("accuracy of stack ML :" + str(acc))
    print("sensitivity of stack ML :" + str(sensitivity) + " through recall score :" + str(rec))
    print("specificity of stack ML :" + str(specificity))
    print("precision of stack ML :" + str(prec))
    print("F1 score of stack ML :" + str(f1))
    print("roc score of stack ML :" + str(roc))
    draw_roc(final_status_test, predict_proba)
    # roc curve for stack ML


def draw_roc(y_test, probs_ml):
    # roc curve for models
    fpr1, tpr1, thresh1 = roc_curve(y_test, probs_ml[:, 1], pos_label=1)

    # roc curve for tpr = fpr
    random_probs = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

    # auc roc score
    auc_score1 = roc_auc_score(y_test, probs_ml[:, 1])
    print("STACK ML MODEL 1 ROC SCORE :", auc_score1)

    # plot roc curves
    plt.plot(fpr1, tpr1, linestyle='--', color='orange', label='stack ML')
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    # title
    plt.title('ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')

    plt.legend(loc='best')
    plt.savefig('ROC', dpi=300)
    plt.show()










