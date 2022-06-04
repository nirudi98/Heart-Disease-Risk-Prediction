from os import makedirs

from keras import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.models import load_model
from keras.optimizers import Adam
from mlxtend.evaluate import confusion_matrix
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, precision_score, recall_score, f1_score, \
    matthews_corrcoef, roc_curve
import numpy as np
from matplotlib import pyplot as plt


def build_dnn_model1(X_train, X_test, y_train, y_test, col_len):
 dnn_model = Sequential()

# Adding the input layer and the first hidden layer of the DNN
 dnn_model.add(Dense(units=40, activation='relu', input_shape=(col_len,)))

# Add other layers, it is not necessary to pass the shape because there is a layer before
 dnn_model.add(Dense(units=20, kernel_initializer='normal', activation='relu'))
 dnn_model.add(Dropout(rate=0.5))
 dnn_model.add(Dense(units=10, kernel_initializer='normal', activation='relu'))
 dnn_model.add(Dropout(rate=0.5))

# Adding the output layer
 dnn_model.add(Dense(units=1, kernel_initializer='normal', activation='sigmoid'))

# Compiling the DNN
 dnn_model.compile(optimizer=Adam(lr=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

 history = dnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=10, epochs=120)

 _, train_acc = dnn_model.evaluate(X_train, y_train, verbose=0)
 _, test_acc = dnn_model.evaluate(X_test, y_test, verbose=0)
 print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

 plt.subplot(211)
 plt.title('Loss')
 plt.plot(history.history['loss'], label='train')
 plt.plot(history.history['val_loss'], label='test')
 plt.legend()
 plt.subplot(212)
 plt.title('Accuracy')
 plt.plot(history.history['accuracy'], label='train')
 plt.plot(history.history['val_accuracy'], label='test')
 plt.legend()
 plt.show()

 return dnn_model


def build_dnn_model2(X_train, X_test, y_train, y_test, col_len):
 dnn_model = Sequential()

# Adding the input layer and the first hidden layer of the ANN with dropout
 dnn_model.add(Dense(units=32, activation='relu', input_shape=(col_len,)))

# Add other layers, it is not necessary to pass the shape because there is a layer before
 dnn_model.add(Dense(units=16, kernel_initializer='normal', activation='relu'))
 dnn_model.add(Dropout(rate=0.25))
 dnn_model.add(Dense(units=8, kernel_initializer='normal', activation='relu'))
 dnn_model.add(Dropout(rate=0.25))


# Adding the output layer
 dnn_model.add(Dense(units=1, kernel_initializer='normal', activation='sigmoid'))

# Compiling the ANN
 dnn_model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

 history = dnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=10, epochs=120)

 _, train_acc = dnn_model.evaluate(X_train, y_train, verbose=0)
 _, test_acc = dnn_model.evaluate(X_test, y_test, verbose=0)
 print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
 plt.subplot(211)
 plt.title('Loss')
 plt.plot(history.history['loss'], label='train')
 plt.plot(history.history['val_loss'], label='test')
 plt.legend()
 plt.subplot(212)
 plt.title('Accuracy')
 plt.plot(history.history['accuracy'], label='train')
 plt.plot(history.history['val_accuracy'], label='test')
 plt.legend()
 plt.show()

 return dnn_model


def build_lstm1_model(x_train, final_status_train, x_test, y_test, col_len):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=18, dropout=0.02, recurrent_dropout=0.20, return_sequences=True, input_shape=(1, col_len)))
    lstm_model.add(LSTM(units=9, dropout=0.02, recurrent_dropout=0.20, return_sequences=False))
    lstm_model.add(Dense(1, activation='sigmoid'))
    lstm_model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    history = lstm_model.fit(x_train, final_status_train, validation_data=(x_test, y_test), batch_size=10, epochs=150)

    _, train_acc = lstm_model.evaluate(x_train, final_status_train, verbose=0)
    _, test_acc = lstm_model.evaluate(x_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()

    return lstm_model


def build_lstm2_model(x_train, final_status_train, x_test, y_test, col_len):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=16, dropout=0.02, recurrent_dropout=0.20, return_sequences=True, input_shape=(1, col_len)))
    lstm_model.add(LSTM(units=8, dropout=0.02, recurrent_dropout=0.20, return_sequences=False))
    lstm_model.add(Dense(1, activation='sigmoid'))
    lstm_model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    history = lstm_model.fit(x_train, final_status_train, validation_data=(x_test, y_test), batch_size=10, epochs=150)

    _, train_acc = lstm_model.evaluate(x_train, final_status_train, verbose=0)
    _, test_acc = lstm_model.evaluate(x_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()

    return lstm_model


def save_models(model, member, X_train, X_test, y_train, y_test, col_length):
    filename = 'D:/python/Cardio/model_two/baseModels/NNModels/model_' + str(member) + '.h5'
#   filename = 'D:/python/Cardio/model_two/baseModels/NNModels/model_' + str(member) + '.h5'
    model.save(filename)
    print('>Saved %s' % filename)
    ann_prediction = np.round(model.predict(X_test)).astype(int)
    pred = model.predict(X_test)
    ann_pred_proba = model.predict_proba(X_test)
    print('Results for ANN Model' + " " + str(member) + " " + str(accuracy_score(y_test, ann_prediction)))


def train_ann(X_train, X_test, y_train, y_test, col_length):
    n_members = 4

    print("DNN 1  ")
    model = build_dnn_model1(X_train, X_test, y_train, y_test, col_length)
    save_models(model, n_members-3, X_train, X_test, y_train, y_test, col_length)
    # dnn 1 prediction for evaluation
    dnn1_pred = model.predict(X_test, verbose=0)
    dnn1_classes = model.predict_classes(X_test, verbose=0)
    # since it returns two dimensional classes
    dnn1_one_pred = dnn1_pred[:, 0]
    dnn1_one_classes = dnn1_classes[:, 0]
    # evaluating model function
    evaluate_DNN(dnn1_one_classes, y_test)

    print("DNN 2  ")
    model2 = build_dnn_model2(X_train, X_test, y_train, y_test, col_length)
    save_models(model2, n_members-2, X_train, X_test, y_train, y_test, col_length)
    # dnn 2 prediction for evaluation
    dnn2_pred = model2.predict(X_test, verbose=0)
    dnn2_classes = model2.predict_classes(X_test, verbose=0)
    # since it returns two dimensional classes
    dnn2_one_pred = dnn2_pred[:, 0]
    dnn2_one_classes = dnn2_classes[:, 0]
    # evaluating model function
    evaluate_DNN(dnn2_one_classes, y_test)

    x_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    x_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    print("LSTM  1")
    lstm = build_lstm1_model(x_train, y_train, x_test, y_test, col_length)
    lstm_prediction = np.round(lstm.predict(x_test)).astype(int)
    print('Results for LSTM Model', accuracy_score(y_test, lstm_prediction))
    #   lstm_filename = 'D:/python/Cardio/model_two/baseModels/NNModels/model_CHECK' + str(n_members) + '.h5'
    lstm_filename = 'D:/python/Cardio/model_two/baseModels/NNModels/model_' + str(n_members-1) + '.h5'
    lstm.save(lstm_filename)

    # lstm prediction for evaluation
    lstm_pred = lstm.predict(x_test, verbose=0)
    lstm_classes = lstm.predict_classes(x_test, verbose=0)
    # since it returns two dimensional classes
    lstm_one_pred = lstm_pred[:, 0]
    lstm_one_classes = lstm_classes[:, 0]
    # evaluating model function
    evaluate_LSTM(lstm_one_pred, lstm_one_classes, y_test)

    print("LSTM  2")
    lstm1 = build_lstm2_model(x_train, y_train, x_test, y_test, col_length)
    lstm_prediction1 = np.round(lstm1.predict(x_test)).astype(int)
    print('Results for LSTM Model', accuracy_score(y_test, lstm_prediction1))
    #   lstm_filename = 'D:/python/Cardio/model_two/baseModels/NNModels/model_CHECK' + str(n_members) + '.h5'
    lstm1_filename = 'D:/python/Cardio/model_two/baseModels/NNModels/model_' + str(n_members) + '.h5'
    lstm1.save(lstm1_filename)

    # lstm1 prediction for evaluation
    lstm1_pred = lstm1.predict(x_test, verbose=0)
    lstm1_classes = lstm1.predict_classes(x_test, verbose=0)
    # since it returns two dimensional classes
    lstm1_one_pred = lstm1_pred[:, 0]
    lstm1_one_classes = lstm1_classes[:, 0]
    # evaluating model function
    evaluate_LSTM1(lstm1_one_pred, lstm1_one_classes, y_test)
    save_models(lstm, n_members, x_train, x_test, y_train, y_test, col_length)
    members = load_all_models(n_members)
    return members


def load_all_models(n_models):
    print(n_models)
    all_ann_models = list()
    for i in range(n_models):
        print(str(i))
# filename = 'D:/python/Cardio/' + modelName + '/baseModels/NNModels/model_' + str(i + 1) + '.h5'
        filename = 'D:/python/Cardio/model_three/baseModels/NNModels/model_' + str(i + 1) + '.h5'
        model = load_model(filename)
        # Add a list of all the weaker learners
        all_ann_models.append(model)
        print("loop")
        print('>loaded %s' % filename)
    return all_ann_models


def load_all_models1(n_models, modelName):
    print(n_models)
    all_ann_models = list()
    for i in range(n_models):
        print(str(i))
        filename = 'D:/python/Cardio/' + modelName + '/baseModels/NNModels/model_' + str(i + 1) + '.h5'
#        filename = 'D:/python/Cardio/baseModels/NNModels/model_NEW' + str(i + 1) + '.h5'
        model = load_model(filename)
        # Add a list of all the weaker learners
        all_ann_models.append(model)
        print("loop")
        print('>loaded %s' % filename)
    return all_ann_models


def evaluate_DNN(dnn_class, testy):
    accuracy = accuracy_score(testy, dnn_class)
    print("Accuracy of DNN : %f" % accuracy)
    precision = precision_score(testy, dnn_class)
    print("Precision of DNN: %f" % precision)
    recall = recall_score(testy, dnn_class)
    print("Recall of DNN : %f" % recall)
    f1 = f1_score(testy, dnn_class)
    print("F1 score of DNN : %f" % f1)

    # calculating additional metric like roc curve, confusion matrix
    dnn_roc = roc_auc_score(testy, dnn_class)
    print("ROC Score of DNN : %f" % dnn_roc)
    CM_lstm = confusion_matrix(testy, dnn_class)
    print(CM_lstm)

    TN = CM_lstm[0][0]
    FN = CM_lstm[1][0]
    TP = CM_lstm[1][1]
    FP = CM_lstm[0][1]

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    print("sensitivity of DNN : %f" % sensitivity)
    print("specificity of DNN : %f" % specificity)


def evaluate_LSTM(lstm_pred, lstm_class, testy):
    accuracy = accuracy_score(testy, lstm_class)
    print("Accuracy of LSTM : %f" % accuracy)
    precision = precision_score(testy, lstm_class)
    print("Precision of LSTM : %f" % precision)
    recall = recall_score(testy, lstm_class)
    print("Recall of LSTM : %f" % recall)
    f1 = f1_score(testy, lstm_class)
    print("F1 score of LSTM : %f" % f1)

    # calculating additional metric like roc curve, confusion matrix
    lstm_roc = roc_auc_score(testy, lstm_class)
    print("ROC Score of LSTM : %f" % lstm_roc)
    CM_lstm = confusion_matrix(testy, lstm_class)
    print(CM_lstm)

    TN = CM_lstm[0][0]
    FN = CM_lstm[1][0]
    TP = CM_lstm[1][1]
    FP = CM_lstm[0][1]

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    print("sensitivity of LSTM : %f" % sensitivity)
    print("specificity of LSTM : %f" % specificity)


def evaluate_LSTM1(lstm1_pred, lstm1_class, testy):
    accuracy = accuracy_score(testy, lstm1_class)
    print("Accuracy of LSTM : %f" % accuracy)
    precision = precision_score(testy, lstm1_class)
    print("Precision of LSTM : %f" % precision)
    recall = recall_score(testy, lstm1_class)
    print("Recall of LSTM : %f" % recall)
    f1 = f1_score(testy, lstm1_class)
    print("F1 score of LSTM : %f" % f1)

    # calculating additional metric like roc curve, confusion matrix
    lstm_roc = roc_auc_score(testy, lstm1_class)
    print("ROC Score of LSTM : %f" % lstm_roc)
    CM_lstm = confusion_matrix(testy, lstm1_class)
    print(CM_lstm)

    TN = CM_lstm[0][0]
    FN = CM_lstm[1][0]
    TP = CM_lstm[1][1]
    FP = CM_lstm[0][1]

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    print("sensitivity of LSTM : %f" % sensitivity)
    print("specificity of LSTM : %f" % specificity)