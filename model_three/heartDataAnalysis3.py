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


if __name__ == '__main__':
 heart = pd.read_csv("D:/NSBM/NSBM/year 4/research/heart disease prediction/data/heart_statlog_cleveland_hungary_train_COPY.csv")
 print(heart.head())

# analyzing dataset
 print(heart.info())
 print(type(heart))
 print(heart.head(5))
 print(heart.describe())

 heart = heart.rename(
  columns={"chest pain type": "chest_pain", "max heart rate": "maxHeartRate", "resting bp s": "blood_pressure",
           "fasting blood sugar": "blood_sugar",
           "ca": "vessels", "chol": "cholesterol", "sex": "gender", "target": "heart_status",
           "exercise angina": "exercise_angina"})

 print("\n")

# heart['heart_status'] = ["healthy" if x == 0 else "sick" for x in heart['heart_status']]
 print("\n")

 print(heart["heart_status"].describe())
 print(heart["heart_status"].unique())
 print(heart.info())

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

# checking any null values
 print(heart.isnull())

 y = heart["heart_status"]

 print(sns.countplot(y))

 heart_patient = heart.heart_status.value_counts()

 print(heart_patient)

# percentages of heart patients according to dataset
 print("Percentage of patience without heart problems: " + str(round(heart_patient[0] * 100 / 301, 2)))
 print("Percentage of patience with heart problems: " + str(round(heart_patient[1] * 100 / 301, 2)))

# overview distribution of each column
heart.hist(figsize=(16, 20), xlabelsize=8, ylabelsize=8)
plt.show()

sns.pairplot(heart, hue='heart_status')
plt.show()

sns.pairplot(heart, hue='gender')
plt.show()

heart.corr()
f, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(heart.corr(), annot=True, cmap='RdGy', linewidths=.5)
plt.show()

# visualizing each variable
# 1- gender
# print(heart['gender'].value_counts())
#
# print(heart.groupby(['gender', 'heart_status'])['gender'].count())
#
# print(heart['gender'].corr(heart['heart_status']))
# sns.distplot(heart['gender'])
# f, ax = plt.subplots(figsize=(15, 5))
# sns.countplot(data=heart, x=pd.cut(heart['gender'], 10), hue='heart_status')
# plt.show()

attr_1=heart[heart['heart_status']==1]
attr_0=heart[heart['heart_status']==0]

ax1 = plt.subplot2grid((1,2),(0,1))
sns.countplot(attr_0['gender'], palette='viridis')
plt.title('GENDER DISTRIBUTION OF NORMAL PATIENTS', fontsize=15, weight='bold' )
plt.show()

ax1 = plt.subplot2grid((1,2),(0,1))
sns.countplot(attr_1['gender'], palette='viridis')
plt.title('GENDER DISTRIBUTION OF HEART PATIENTS', fontsize=15, weight='bold' )
plt.show()
#
# # 2- maxHeartRate
# sns.distplot(heart['maxHeartRate'])
# f, ax = plt.subplots(figsize=(15, 5))
# sns.countplot(data=heart, x=pd.cut(heart['maxHeartRate'], 10), hue='heart_status')
# print(heart['maxHeartRate'].corr(heart['heart_status']))
#
#
# # 3- age
# print(heart['age'].describe())
# print(heart['age'].mean())
# sns.distplot(heart['age'])
#
# # 4- blood pressure
# sns.distplot(heart['blood_pressure'])
#
# # 5- cholesterol
# sns.distplot(heart['cholesterol'])
#
# # 6- blood sugar
# sns.distplot(heart['blood_sugar'])
#
# # 7- chest pain
# sns.countplot(data=heart, x='chest_pain', hue='heart_status')
# print(heart['chest_pain'].corr(heart['heart_status']))
#
# # 8- exang
# # all features that are be measured without medical examination
# sns.pairplot(heart, vars=['age', 'gender', 'cholesterol', 'blood_pressure',
#                           'blood_sugar', 'exang', 'maxHeartRate', 'chest_pain'], hue='heart_status')
#
# plt.show()
#
# # above 8 features can be measured at home, so will be using them to make the early diagnosis
# # and see how accurate the result is
