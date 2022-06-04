import pandas as pd
import sklearn
from matplotlib import pyplot as plt


# model tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings

from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')


# convert age to years since its given in days
def convert_age_years(agey):
    agey = agey.apply(lambda x: round(x / 365))
    return agey


# calculating the BMI = kg/ m*m feature using height and weight. since a person is overweight
# or obese matters when predicting heart disease presence
# converting height to meters and square
def calculate_height(height):
    height = (height/100) ** 2
    return height


# calculating of BMI
def calculate_bmi():
    heart['height_m'] = heart['height'].apply(lambda x: calculate_height(x))

    # calculating bmi
    heart['bmi'] = heart['weight'] / heart['height_m']
    print(heart)


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


# checking for outliers
# Systolic BP(ap_hi) not greater than 370 and not less than 70
# Diastolic BP(ap_lo) not greater than 360 and not less than 50
# age cannot be more than 100 or less than 0
def check_outlier_bp(sys, dia):
    outliers = (sys >= 370) | (sys <= 70) | (dia >= 360) | (dia <= 50)
    return outliers


def check_outlier_age(age):
    age_out = (age > 100) | (age <= 0)
    return age_out


# overview of heart dataset
def heart_overview(heart_data):
    heart_data.hist(figsize=(16, 20), xlabelsize=8, ylabelsize=8)
    plt.show()


def heart_heatmap(hear):
    # Heat map to check the multi CoLinearity
    corr = hear.corr()
    f, ax1 = plt.subplots(figsize=(15, 15))
    sns.heatmap(corr, annot=True, fmt=".3f", linewidths=0.5, cmap="Blues_r", ax=ax1)
    plt.show()


def cor_selector(X, y, num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature


if __name__ == '__main__':
 heart = pd.read_csv("D:/NSBM/NSBM/year 4/research/heart disease prediction/data/cardio_train.csv", sep=";")
 print(heart.head())

# analyzing dataset
 print(heart.info())
 print(type(heart))
 print(heart.head(5))
 print(heart.describe())

# convert age to years since its given in days
 heart['age'] = convert_age_years(heart['age'])

# renaming the features
 heart = heart.rename(
  columns={"ap_hi": "sys_blood_pressure", "ap_lo": "dia_blood_pressure", "gluc": "glucose", "cardio": "heart_status"})

 print("\n")

# to check whether there any null values
 print(heart.isna().sum())

# identify duplicates
 check_duplicates(heart)

# Outlier detection and Removal
 outlier = check_outlier_bp(heart['sys_blood_pressure'], heart['dia_blood_pressure'])
# print("blood pressure related outliers : " + str(heart[outlier].count()))
 heart = heart[~outlier]

 outlier_age = check_outlier_age(heart['age'])
# print("age related outliers : " + str(heart[outlier_age].count()))

# getting the BMI cause it is also a contributing factor
# getting the formatted height
 calculate_bmi()
 plt.scatter(heart['heart_status'], pd.to_numeric(heart['sys_blood_pressure']), label='Systolic')
 plt.scatter(heart['heart_status'], heart['bmi'], label='bmi')
 plt.legend()
 plt.show()

# Dropping columns ID and Height in meters as they are not important and cardio as it is dependent variable
 heart.drop(['height', 'weight', 'id', 'height_m'], axis=1, inplace=True)
 print(heart)

 cardio = heart["heart_status"]
 print(sns.countplot(cardio))
 heart_patient = heart.heart_status.value_counts()
 print(heart_patient)

# percentages of heart patients according to dataset
 print("Percentage of patience without heart problems: " + str(round(heart_patient[0] * 100 / len(heart), 2)))
 print("Percentage of patience with heart problems: " + str(round(heart_patient[1] * 100 / len(heart), 2)))

# heart.drop(['heart_status'], axis=1, inplace=True)

 X = heart
 y = cardio

 print(X)
 print(y)

# overview distribution of each column - histogram
 heart_overview(heart)

# correlation matrix - heat map
 heart_heatmap(heart)


# Plot to see the chance of getting cardiovascular disease
 plt.figure(figsize=(20, 25), facecolor='white')
 plot_number = 1

 for column in X:
     if plot_number <= 16:
         ax = plt.subplot(4, 4, plot_number)
         sns.stripplot(y, X[column])
     plot_number += 1

 plt.tight_layout()


# visualizing each variable
# 1- gender
 print(heart['gender'].value_counts())
 print(heart.groupby(['gender', 'heart_status'])['gender'].count())
 print(heart['gender'].corr(heart['heart_status']))
 plt.show()

# 2 - age
 print(heart['age'].value_counts())
 print(heart.groupby(['age', 'heart_status'])['age'].count())
 print(heart['age'].corr(heart['heart_status']))
 plt.show()

# 3 - systolic blood pressure
 print(heart['sys_blood_pressure'].value_counts())
 print(heart.groupby(['sys_blood_pressure', 'heart_status'])['sys_blood_pressure'].count())
 print(heart['sys_blood_pressure'].corr(heart['heart_status']))
 plt.show()

# 4 - cholesterol
 print(heart['cholesterol'].value_counts())
 print(heart.groupby(['cholesterol', 'heart_status'])['cholesterol'].count())
 print(heart['cholesterol'].corr(heart['heart_status']))
 plt.show()

# 4 - glucose
 print(heart['glucose'].value_counts())
 print(heart.groupby(['glucose', 'heart_status'])['glucose'].count())
 print(heart['glucose'].corr(heart['heart_status']))
 plt.show()

# 5 - smoke
 print(heart['smoke'].value_counts())
 print(heart.groupby(['smoke', 'heart_status'])['smoke'].count())
 print(heart['smoke'].corr(heart['heart_status']))
 plt.show()

# 6 - bmi
 print(heart['bmi'].value_counts())
 print(heart.groupby(['bmi', 'heart_status'])['bmi'].count())
 print(heart['bmi'].corr(heart['heart_status']))
 plt.show()

# feature selection
cor_support, cor_feature = cor_selector(X, y, num_feats)
print(str(len(cor_feature)), 'selected features')
