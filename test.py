import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
sns.set_style('whitegrid')
# Get the data

ad_data = pd.read_csv('advertising.csv')

# Check the head of the data

# printInfo = [
#             ad_data.head(),
#             ad_data.info(),
#             ad_data.describe()
#         ]

# for item in printInfo:
#     print(item, end = '\n')


# print(
#         ad_data.head(), end = '\n'
#         )

# print(
#         ad_data.info(), end = '\n'
#         )

# print(
#         ad_data.describe(), end = '\n'
#         )

# Exploratory Data Analysis

# Creates histogram of age
# sns.histplot(data = ad_data, x = 'Age') 
# ad_data['Age'].hist(bins = 30)
# plt.xlabel('Age')
# plt.savefig('./figures/hplot.png')


# Creates jointplot showing area 'Income' vs 'Age'
# sns.jointplot(data = ad_data, x = 'Age', y = 'Area Income')
# plt.savefig('./figures/jplot.png')

# Create a jointplot showing the kde distributions of 'Daily Time spent on Site' vs. 'Age'
# sns.jointplot(data = ad_data,
#               x = 'Age',
#               y = 'Daily Time Spent on Site',
#               kind = 'kde',
#               color = 'red')

# plt.savefig('./figures/jplotkde.png')

# sns.jointplot(
#         data = ad_data,
#         x = 'Daily Time Spent on Site',
#         y = 'Daily Internet Usage',
#         color = 'green'
#         )

# plt.savefig('./figures/jplotgreen.png')

# sns.pairplot(
#         data = ad_data,
#         hue = 'Clicked on Ad'
#         )
# plt.savefig('./figures/pplot.png')

# Logistic Regression

# Train/Test Split

X = ad_data[[
        'Daily Time Spent on Site',
        'Age',
        'Area Income',
        'Daily Internet Usage',
        'Male'
    ]]

y = ad_data['Clicked on Ad'] 

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 101)

lm = LogisticRegression()

lm.fit(X_train,y_train)

predictions = lm.predict(X_test)

# print(
#         classification_report(y_test, predictions)
#         )

print(
        confusion_matrix(y_test, predictions)
        )

plt.show()
