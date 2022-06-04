from keras.utils.np_utils import to_categorical
from mlxtend.evaluate import confusion_matrix
from numpy import dstack, argmax, array
from sklearn.linear_model import LogisticRegression
from keras.layers import Dense
from keras.layers.merge import concatenate
from keras.models import Model
from keras.utils.vis_utils import plot_model
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, precision_score, recall_score, f1_score, roc_curve
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt


def new_stack_model_meta(members, X, y):
    stacked_model_new = define_stacked_model(members)
    fit_stacked_model(stacked_model_new, X, y)
    # make predictions and evaluate
    final_prediction = predict_stacked_model(stacked_model_new, X)
    final_prediction = argmax(final_prediction, axis=1)
    acc = accuracy_score(y, final_prediction)
    print('Stacked Test Accuracy: %.3f' % acc)


def define_stacked_model(members):
    for i in range(len(members)):
        model = members[i]
        print(model)
        for layer in model.layers:
            layer.trainable = False
            layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
    ensemble_visible = [model.input for model in members]
    ensemble_outputs = [model.output for model in members]
    print("here")
    merge = concatenate(ensemble_outputs)
    hidden = Dense(input_shape=(8,), activation='relu', units=8)(merge)
    output = Dense(1, activation='softmax')(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)
#    plot_model(model, show_shapes=True, to_file='model_graph.png')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# fit a stacked model
def fit_stacked_model(model, inputX, inputy):
    X = [inputX for _ in range(len(model.input))]
    print(inputy.shape)
    model.fit(X, inputy, epochs=200, verbose=0)


# make a prediction with a stacked model
def predict_stacked_model(model, inputX):
    X = [inputX for _ in range(len(model.input))]
    return model.predict(X, verbose=0)


def stack_model_nn(ann, X, y):
    stackedX = stacked_dataset(ann, X)
    print("stackedX", stackedX)
    model = LogisticRegression()  # meta learner
    model.fit(stackedX, y)
    return model


# function to create stack model input dataset
def stacked_dataset(models, X):
    stackX = None
    for model in models:
        y_pred = model.predict(X, verbose=0)
        if stackX is None:
            stackX = y_pred
        else:
            stackX = dstack((stackX, y_pred))
            stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
            print(stackX)
            return stackX


# make a prediction with the stacked DL model
def stacked_prediction(models, stack_model, inputX, y_test):
    stackedX = stacked_dataset(models, inputX)
    final_pred = stack_model.predict(stackedX)
    print(final_pred.shape)
    final_proba = stack_model.predict_proba(stackedX)
    print("predict of stacked DL : ", accuracy_score(y_test, final_pred))
    check_built_model(y_test, final_pred, final_proba)
    return final_pred, final_proba


def check_built_model(final_status_test, y_pred, predict_proba):
    CM = confusion_matrix(final_status_test, y_pred)
    sns.heatmap(CM, annot=True)
    plt.show()
    check_parameters(CM, final_status_test, y_pred, predict_proba)


def prediction_realtime(models, stack_model, inputX):
    print("input X", inputX)
    stackedX = stacked_dataset(models, inputX)
    print(stackedX)
    final_real_pred = stack_model.predict(stackedX)
    probs = stack_model.predict_proba(stackedX)
    return probs, final_real_pred


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
    print("accuracy of stack DL :" + str(acc))
    print("sensitivity of stack DL :" + str(sensitivity) + " through recall score :" + str(rec))
    print("specificity of stack DL :" + str(specificity))
    print("precision of stack DL :" + str(prec))
    print("F1 score of stack DL :" + str(f1))
    print("roc score of stack DL :" + str(roc))
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
    print("STACK DL MODEL 1 ROC SCORE :", auc_score1)

    # plot roc curves
    plt.plot(fpr1, tpr1, linestyle='--', color='orange', label='stack DL')
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
