# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn.ensemble import RandomForestRegressor

# create regressor object
regressor = RandomForestRegressor(n_estimators=100, random_state=0)



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    df = pd.read_csv('Salary.csv')

    #wine_df.columns = ['Position', 'Level', 'Salary']

    #print(df.)
    x= df.iloc[:, 2].values.reshape(-1, 1)
    #x= df.iloc [:, : -1] # ” : ” means it will select all rows,    “: -1 ” means that it will ignore last column
    y= df.iloc [:, -1 :].values.reshape(-1, 1).ravel() # ” : ” means it will select all rows,    “-1 : ” means that it will ignore all columns except the last one
# Press the green button in the gutter to run the script.
    print(x);

    # fit the regressor with x and y data
    regressor.fit(x, y)
    Y_pred = regressor.predict(np.array([6.5]).reshape(1, 1))  # test the output by changing values
    print("Level: ", np.array([6.5]).reshape(1, 1), "Presdiction: ", Y_pred)

    # Visualising the Random Forest Regression results

    # arrange for creating a range of values
    # from min value of x to max
    # value of x with a difference of 0.01
    # between two consecutive values
    X_grid = np.arange(min(x), max(x), 0.01)

    # reshape for reshaping the data into a len(X_grid)*1 array,
    # i.e. to make a column out of the X_grid value
    X_grid = X_grid.reshape((len(X_grid), 1))

    # Scatter plot for original data
    plt.scatter(x, y, color='blue')

    # plot predicted data
    plt.plot(X_grid, regressor.predict(X_grid),
             color='green')
    plt.title('Random Forest Regression')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()

if __name__ == '__main__':
    #print("hello")
    #df = pd.read_csv('YoutubeSpamMergedData.csv')

    #print(df.to_string())

    print_hi('Bashir')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
