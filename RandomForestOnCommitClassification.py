# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd


# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
# create regressor object
regressor = RandomForestRegressor(n_estimators=100, random_state=0)



def RandomForestClassification():
    df = pd.DataFrame(pd.read_excel("DataSet1.xlsx"))
    print(df.head())
    print(df.columns)

    X = df.iloc[: , 4:]
    y = df.iloc[:, 3]


    # Spliting arrays or matrices into random train and test subsets

    # i.e. 70 % training dataset and 30 % test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)


    # importing random forest classifier from assemble module
    from sklearn.ensemble import RandomForestClassifier

    # creating a RF classifier
    clf = RandomForestClassifier(n_estimators=100)

    # Training the model on the training dataset
    # fit function is used to train the model using the training sets as parameters
    clf.fit(X_train, y_train)

    # performing predictions on the test dataset
    y_pred = clf.predict(X_test)

    # metrics are used to find accuracy or error
    from sklearn import metrics

    # using metrics module for accuracy calculation
    print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

    # predicting which type of flower it is.
    prediction = clf.predict([[1, 1, 0, 0, 1, 2, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 1, 0, 1, 1, 1, 2, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                               0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 0, 1, 0, 1, 0, 2, 1, 1, 0, 0, 0, 0, 1, 0]])
    print(prediction)


if __name__ == '__main__':

    RandomForestClassification()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
