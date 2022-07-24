# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn import tree
# Similarly LGBMRegressor can also be imported for a regression model.
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import xgboost
# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# create regressor object
regressor = RandomForestRegressor(n_estimators=100, random_state=0)


def RandomForestClassification():
    # df = pd.DataFrame(pd.read_excel("Complete-Dataset.xlsx"))
    df = pd.DataFrame(pd.read_excel("DataSet1.xlsx"))

    # print(df.head())
    # print(df.columns)

    X = df.iloc[:, 4:]
    y = df.iloc[:, 3]

    # Spliting arrays or matrices into random train and test subsets

    # i.e. 70 % training dataset and 30 % test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    # importing random forest classifier from assemble module
    from sklearn.ensemble import RandomForestClassifier

    # creating a RF classifier
    clf = RandomForestClassifier(n_estimators=100)

    # Training the model on the training dataset
    # fit function is used to train the model using the training sets as parameters
    clf.fit(X_train, y_train)

    # performing predictions on the test dataset
    predictions = clf.predict(X_test)

    # metrics are used to find accuracy or error
    from sklearn import metrics

    # using metrics module for accuracy calculation
    print("ACCURACY OF THE MODEL RANDOM FOREST: ", metrics.accuracy_score(y_test, predictions))

    # print("Predictions for the model Random Forest: \n", predictions)
    print("Confusion Matrix for Random Forest: \n", confusion_matrix(y_test, predictions))

    #Calculate Kappa
    #x = confusion_matrix(y_test, predictions)
    k = getKappaValue(confusion_matrix(y_test, predictions))
    print('Kappa Value of Random Forest:', k*100, '%')

    print("Classification Report for Random Forest: \n", classification_report(y_test, predictions))

    # Creating an object for model and fitting it on training data set
    model = GradientBoostingClassifier()
    # model = LGBMClassifier()
    model.fit(X_train, y_train)

    # Predicting the Target variable
    pred = model.predict(X_test)
    # print("Predictions for the model GBM: \n", pred)
    accuracy = model.score(X_test, y_test)
    print("Accuracy for the model GBM: ", accuracy)
    print("Confusion Matrix for GBM:\n", confusion_matrix(y_test, pred))
    print("Classification Report for GBM: \n", classification_report(y_test, pred))
    #calculate Kappa
    k = getKappaValue(confusion_matrix(y_test, pred))
    print('Kappa Value of GBM:', k * 100, '%')


    '''le = LabelEncoder()
    #y_train = le.fit_transform(y_train)
    xgb_clf = XGBClassifier(n_estimators = 400, learning_rate = 0.1, max_depth = 3)
    xgb_clf.fit(X_train, le.fit_transform(y_train))

    predForXGBoost = xgb_clf.predict(X_test)
    #print("Predictions for the model XGBoost: \n", predForXGBoost)
    accuracy = xgb_clf.score(X_test, le.fit_transform(y_test))
    print("Accuracy for the model XGBoost: ", accuracy)
    print("Confusion Matrix for XGBoost:\n", confusion_matrix(le.fit_transform(y_test), predForXGBoost))
    print("Classification Report for XGBoost: \n", classification_report(le.fit_transform(y_test), predForXGBoost))'''


def getKappaValue(x):
    sumOf_a = 0
    overAllTotal = 0
    rowTotalMulColTotal = [0, 0, 0]
    rowTotalMulColTotalDivOverall = [0, 0, 0]
    rowTotalMulColTotalIndex = 0
    sumOfEf = 0
    for i in range(len(x)):
        for j in range(len(x[i])):
            overAllTotal += x[i][j]
            if (i == j):
                sumOf_a += x[i][j]
                sumOfRow = 0
                sumOFCol = 0
                for r in range(3):
                    sumOfRow += x[i][r]
                for c in range(3):
                    sumOFCol += x[c][i]

                rowTotalMulColTotal[rowTotalMulColTotalIndex] = sumOfRow * sumOFCol
                rowTotalMulColTotalIndex += 1
    for i in range(len(rowTotalMulColTotal)):
        rowTotalMulColTotalDivOverall[i] = rowTotalMulColTotal[i] / overAllTotal
        sumOfEf += rowTotalMulColTotalDivOverall[i]
    '''print(sumOf_a)
    print(overAllTotal)
    print(rowTotalMulColTotalDivOverall)
    print(sumOfEf)'''
    k = (sumOf_a - sumOfEf) / (overAllTotal - sumOfEf)

    return k



if __name__ == '__main__':
    RandomForestClassification()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
