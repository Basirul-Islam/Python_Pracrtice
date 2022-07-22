# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

# create regressor object
regressor = RandomForestRegressor(n_estimators=100, random_state=0)



def print_hi(name):
    iris = datasets.load_iris()
    # dividing the datasets into two parts i.e. training datasets and test datasets
    X, y = datasets.load_iris(return_X_y=True)

    # Spliting arrays or matrices into random train and test subsets
    from sklearn.model_selection import train_test_split
    # i.e. 70 % training dataset and 30 % test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


    # importing random forest classifier from assemble module
    from sklearn.ensemble import RandomForestClassifier

    # creating dataframe of IRIS dataset
    data = pd.DataFrame({'sepallength': iris.data[:, 0], 'sepalwidth': iris.data[:, 1],
    'petallength': iris.data[:, 2], 'petalwidth': iris.data[:, 3],
    'species': iris.target})


    # printing the top 5 datasets in iris dataset
    print(data.head())

    # creating a RF classifier
    clf = RandomForestClassifier(n_estimators=100)

    # Training the model on the training dataset
    # fit function is used to train the model using the training sets as parameters
    clf.fit(X_train, y_train)

    # performing predictions on the test dataset
    y_pred = clf.predict(X_test)

    # metrics are used to find accuracy or error
    from sklearn import metrics
    print()

    # using metrics module for accuracy calculation
    print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

    # predicting which type of flower it is.
    prediction = clf.predict([[2, 1, 4, 5]])
    print(iris.target_names[prediction])

if __name__ == '__main__':
    #print("hello")
    #df = pd.read_csv('YoutubeSpamMergedData.csv')

    #print(df.to_string())

    print_hi('Bashir')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
