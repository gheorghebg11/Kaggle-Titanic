# -*- coding: utf-8 -*-
"""
Spyder Editor
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from copy import deepcopy
from sklearn.preprocessing import StandardScaler


# open file
titX = pd.read_csv('train.csv')
tit_test = pd.read_csv('test.csv')

# save a variable
nbr_passengers = titX.shape[0]
nbr_passengers_test = tit_test.shape[0]

# function to plot distribution of dead/survived in a given categorical var
# (the input feature)
def bar_chart(feature):
    survived = titX[titX['Survived']==1][feature].value_counts()
    dead = titX[titX['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind = 'bar', stacked = True, figsize = (10,5))

bar_chart('Sex')

# copy output variable
titY = titX.Survived
titX.drop(labels='Survived', axis=1, inplace = True)

# Drop what's not relevant
titX.drop(labels=['PassengerId', 'Name', 'Ticket'], axis=1, inplace = True)
tit_test.drop(labels=['PassengerId', 'Name', 'Ticket'], axis=1, inplace = True)
print('Drop the Name, Name of Reference number of its ticket')

# Put variable sex into dummies
tit_MaleFemale = pd.get_dummies(titX['Sex'])
titX = pd.concat([titX,tit_MaleFemale], axis=1)
titX.drop(labels=['Sex'], axis=1, inplace = True)

tit_test_MaleFemale = pd.get_dummies(tit_test['Sex'])
tit_test = pd.concat([tit_test,tit_test_MaleFemale], axis=1)
tit_test.drop(labels=['Sex'], axis=1, inplace = True)

# Replace missing age by the mean
mean = titX['Age'].mean()
print('Replace {} missing Age values with the mean {}'.format(nbr_passengers - titX['Age'].count(), mean))
titX['Age'] = titX['Age'].replace(np.NaN, mean)

mean_test = titX['Age'].append(tit_test['Age']).mean()
print('On test data, replace {} missing Age values with the total mean {}'.format(nbr_passengers_test - tit_test['Age'].count(), mean_test))
tit_test['Age'] = tit_test['Age'].replace(np.NaN, mean_test)

# Replace missing Fare by the mean
fare_mean_test = (titX['Fare'] + tit_test['Fare']).mean()
print('On test replace {} missing Fare values with the mean {}'.format(nbr_passengers_test - tit_test['Fare'].count(), fare_mean_test))
tit_test['Fare'] = tit_test['Fare'].replace(np.NaN, fare_mean_test)

# Replace missing port by most common one and put them into dummies
most_common_port = titX['Embarked'].value_counts().index[0]
print('Replace {} missing age Embarked values with the most common one : {}'.format(nbr_passengers - titX['Embarked'].count(),most_common_port))
titX['Embarked'] = titX['Embarked'].replace(np.NaN, most_common_port)
tit_Embarked = pd.get_dummies(titX['Embarked'])
titX = pd.concat([titX, tit_Embarked], axis = 1)

most_common_port_test = titX['Embarked'].append(tit_test['Embarked']).value_counts().index[0]
print('On test, replace {} missing age Embarked values with the most common one : {}'.format(nbr_passengers_test - tit_test['Embarked'].count(),most_common_port_test))
tit_test['Embarked'] = tit_test['Embarked'].replace(np.NaN, most_common_port_test)
tit_test_Embarked = pd.get_dummies(tit_test['Embarked'])
tit_test = pd.concat([tit_test, tit_test_Embarked], axis = 1)

titX.drop(labels='Embarked', axis = 1, inplace = True)
tit_test.drop(labels='Embarked', axis = 1, inplace = True)

# Drop Cabin ?
print('For now, drop Cabin, as there are only {} out of {}'.format(titX['Cabin'].count(),nbr_passengers))
titX.drop(labels=['Cabin'], axis=1, inplace = True)
tit_test.drop(labels=['Cabin'], axis=1, inplace = True)


##################################################################
########    Now let's try to regress this data!
##################################################################

# Loading the answer!
titY_test = pd.read_csv('gender_submission.csv')
titY_test.drop(labels='PassengerId', axis=1, inplace = True)

#########################################################
################  Try first a linear model
#########################################################
lm = LinearRegression()
lm.fit(titX, titY)
titY_predict = lm.predict(titX)

best_nbr_wrongs = nbr_passengers
threshold_for_best_accuracy = 0

for i in range(30,70,2):
    treshold = i / 100
    titY_predict_temp = deepcopy(titY_predict)
    titY_predict_temp[titY_predict_temp > treshold] = 1
    titY_predict_temp[titY_predict_temp < treshold] = 0
    nbr_wrongs = (titY-titY_predict_temp).abs().sum()
    if nbr_wrongs < best_nbr_wrongs:
        threshold_for_best_accuracy = treshold
        best_nbr_wrongs = nbr_wrongs

print('The accuracy of Linear Regression on the training data is of {} for (best) threshold {}'.format((nbr_passengers - best_nbr_wrongs) / nbr_passengers, threshold_for_best_accuracy))

titY_test_predict = lm.predict(tit_test)
titY_test_predict[titY_test_predict > threshold_for_best_accuracy] = 1
titY_test_predict[titY_test_predict < threshold_for_best_accuracy] = 0

nbr_wrongs_test = (titY_test.Survived - titY_test_predict).abs().sum()

print('The accuracy of Linear Regression on the test data is of {}'.format((nbr_passengers_test - nbr_wrongs_test) / nbr_passengers_test))

# export the data in the required format
passID = np.arange(892, 892 + nbr_passengers_test)
dfpassID = pd.DataFrame(data=passID[:], columns=['Passengerid'])

titY_test_predict = titY_test_predict.astype(int)
titY_test_predict_toexport = pd.DataFrame(data=titY_test_predict[:], columns=['Survived'])

pd.concat([dfpassID, titY_test_predict_toexport], axis = 1).to_csv('export.csv', index=False)

#print(titY_test_predict_toexport)
#titY_test_predict.to_csv('output')

#########################################################       
################  Try first a linear model
#########################################################
degree = 2
pr = PolynomialFeatures(degree=degree)
titX_polly = pr.fit_transform(titX)
titX_test_polly = pr.fit_transform(tit_test)
SCALE = StandardScaler()
SCALE.fit(titX_polly)
titX_polly_scaled = SCALE.transform(titX_polly)

lm.fit(titX_polly, titY)
titY_polly_predict = lm.predict(titX_polly)

best_nbr_wrongs = nbr_passengers
threshold_for_best_accuracy = 0

for i in range(30,70,2):
    treshold = i / 100
    titY_polly_predict_temp = deepcopy(titY_polly_predict)
    titY_polly_predict_temp[titY_polly_predict_temp > treshold] = 1
    titY_polly_predict_temp[titY_polly_predict_temp < treshold] = 0
    nbr_wrongs = (titY-titY_polly_predict_temp).abs().sum()
    if nbr_wrongs < best_nbr_wrongs:
        threshold_for_best_accuracy = treshold
        best_nbr_wrongs = nbr_wrongs

print('The accuracy of Polynomial Regression of degree {} on the training data is of {} for (best) threshold {}'.format(degree, (nbr_passengers - best_nbr_wrongs) / nbr_passengers, threshold_for_best_accuracy))

titY_polly_test_predict = lm.predict(titX_test_polly)
titY_polly_test_predict[titY_polly_test_predict > threshold_for_best_accuracy] = 1
titY_polly_test_predict[titY_polly_test_predict < threshold_for_best_accuracy] = 0

nbr_wrongs_test = (titY_test.Survived - titY_polly_test_predict).abs().sum()

print('The accuracy of Polynomial Regression of degree {} on the test data is of {}'.format(degree, (nbr_passengers_test - nbr_wrongs_test) / nbr_passengers_test))



#print(titX.describe(include = 'all'))
#print(titX.head())