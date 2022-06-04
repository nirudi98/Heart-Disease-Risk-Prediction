import statistics

import pandas as pd
import sklearn
from matplotlib import pyplot as plt
import numpy as np


# model tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings

from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')


if __name__ == '__main__':
#  heart = pd.read_csv("D:/NSBM/NSBM/year 4/research/heart disease prediction/data/framingham_train.csv")
#  print(heart.head())
#
# # analyzing dataset
#  print(heart.info())
#  print(type(heart))
#  print(heart.head(5))
#  print(heart.describe())
#
#  heart = heart.rename(
#   columns={"male": "gender", "TenYearCHD": "heart_status"})
# # if gender 1 = male, 0= female
#  missing_values_count = heart.isnull().sum()
#  print(missing_values_count)
#
#  total_cells = np.product(heart.shape)
#  total_missing = missing_values_count.sum()
#
# # percent of data that is missing
#  percent_missing = (total_missing / total_cells) * 100
#  print(percent_missing)
#
#  percent_missing_edu = (heart['education'].isnull().sum() / heart.shape[0]) * 100
#  print(percent_missing_edu)
#
#  edu_missing = heart[heart['education'].isnull()].index
#  print(edu_missing)
#
#  heart = heart.drop(edu_missing)
#  print(heart.isnull().sum())
#
#  cigarette_index = heart[heart['cigsPerDay'].isnull()].index
#  print(cigarette_index)
#
#  current_smoke_status = []
#  for i in cigarette_index:
#      current_smoke_status.append(heart['currentSmoker'][i])
#
#  print(current_smoke_status)
#  smokers = heart[heart['currentSmoker'] == 1].index
#  print(smokers)
#
#  cigarettes_by_smokers = []
#  for i in smokers:
#      if heart['cigsPerDay'][i] != 'nan':
#          cigarettes_by_smokers.append(heart['cigsPerDay'][i])
#
#  print(len(cigarettes_by_smokers))
#
#  smoker_median = statistics.median(cigarettes_by_smokers)
#  print(smoker_median)
#
#  heart['cigsPerDay'] = heart['cigsPerDay'].fillna(smoker_median)
#
#  print(heart.isnull().sum())
#  BP_missing_index = heart[heart['BPMeds'].isnull()].index
#  print(BP_missing_index)
#
#  for i in BP_missing_index:
#      if heart['sysBP'][i] > 140 or heart['diaBP'][i] > 90:
#          heart.loc[i, 'BPMeds'] = 1.0
#      else:
#          heart.loc[i, 'BPMeds'] = 0.0
#
#  print(heart.isnull().sum())
#
#  heart_1 = heart.copy()
#  print(heart_1.head())
#
#  heart_1['totChol'] = heart_1['totChol'].fillna(round(heart_1['totChol'].mean()))
#  heart_1['BMI'] = heart_1['BMI'].fillna(heart_1['BMI'].mean())
#  heart_1['glucose'] = heart_1['glucose'].fillna(round(heart_1['glucose'].mean()))
#  heart_1['heartRate'] = heart_1['heartRate'].fillna(method='bfill', axis=0)
#  print(heart_1.isnull().sum())
#
#  print(heart_1.head(10))
#
#  heart_2 = heart_1.copy()
#  print(heart_2["education"].unique())
#  print(heart_2["education"].value_counts())
#  heart_2["education"] = heart_2["education"].map({1.0: 0, 2.0: 0, 3.0: 1, 4.0: 1})
#  print(heart_2["education"].unique())
#  heart_preprocessed = heart_2.copy()
#  print(heart_2["education"].value_counts())
#  print(heart_2.isnull().sum())
#  heart_preprocessed = heart_2.copy()
#  print(heart_preprocessed.head(10))
#  print(heart_preprocessed.isnull().sum())
#
#  heart_preprocessed.to_csv('framingham_train_preprocessed.csv', index=False)

 heart = pd.read_csv("framingham_train_preprocessed.csv")
 print(heart.head())

 print("\n")
 print(heart.tail())
 print("\n")
 print(heart.shape)

# dropping duplicates
 print(heart[heart.duplicated(keep=False)])
 heart = heart.drop_duplicates(keep='first')

# checking for outliers
 outlier = (heart['age'] > 100) | (heart['age'] <= 0)
 print(heart[outlier].count())

 y = heart["heart_status"]

 print(sns.countplot(y))

 heart_patient = heart.heart_status.value_counts()

 print(heart_patient)

# percentages of heart patients according to dataset
 print("Percentage of patience without heart problems: " + str(round(heart_patient[0] * 100 / 301, 2)))
 print("Percentage of patience with heart problems: " + str(round(heart_patient[1] * 100 / 301, 2)))

 heart.drop(['id'], axis=1, inplace=True)

 heart.corr()
 f, ax = plt.subplots(figsize=(15, 10))
 sns.heatmap(heart.corr(), annot=True, cmap='RdGy', linewidths=.5)
 plt.show()

# visualizing each variable
# 1- gender
 print(heart['gender'].value_counts())

 print(heart.groupby(['gender', 'heart_status'])['gender'].count())

 print(heart['gender'].corr(heart['heart_status']))

# 7- chest pain
sns.countplot(data=heart, x='gender', hue='heart_status')
print(heart['gender'].corr(heart['heart_status']))
plt.show()

# 2- currentSmoker
sns.distplot(heart['currentSmoker'])
f, ax = plt.subplots(figsize=(15, 5))
sns.countplot(data=heart, x=pd.cut(heart['currentSmoker'], 10), hue='heart_status')
plt.show()
print(heart['currentSmoker'].corr(heart['heart_status']))

# 3- age
print(heart['age'].describe())
print(heart['age'].mean())
sns.countplot(data=heart, x='age', hue='heart_status')
print(heart['age'].corr(heart['heart_status']))
plt.show()

# 4- blood pressure
sns.distplot(heart['BPMeds'])
plt.show()

# 5- cholesterol
sns.distplot(heart['prevalentStroke'])
plt.show()

# 6- blood sugar
sns.distplot(heart['prevalentHyp'])
plt.show()

# 7- chest pain
sns.countplot(data=heart, x='diabetes', hue='heart_status')
print(heart['diabetes'].corr(heart['heart_status']))
plt.show()

# 7- chest pain
sns.countplot(data=heart, x='totChol', hue='heart_status')
plt.show()
heart['totChol'].corr(heart['heart_status'])
plt.show()

# 7- chest pain
sns.countplot(data=heart, x='sysBP', hue='heart_status')
heart['diaBP'].corr(heart['heart_status'])
plt.show()

# 7- chest pain
sns.countplot(data=heart, x='BMI', hue='heart_status')
print(heart['BMI'].corr(heart['heart_status']))

# 7- chest pain
sns.countplot(data=heart, x='heartRate', hue='heart_status')
print(heart['heartRate'].corr(heart['heart_status']))

# 7- chest pain
sns.countplot(data=heart, x='glucose', hue='heart_status')
print(heart['glucose'].corr(heart['heart_status']))

# 8- exang
# all features that are be measured without medical examination
sns.pairplot(heart, vars=['age', 'gender', 'education', 'currentSmoker',
                          'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'], hue='heart_status')

plt.show()

# above 8 features can be measured at home, so will be using them to make the early diagnosis
# and see how accurate the result is



