from tkinter import messagebox   #which only shows the message
from tkinter import *           #importing tkinter module, Python when combined with Tkinter provides a fast and easy way to create GUI applications. 
from tkinter import simpledialog  #which takes user input as pressing button
import tkinter                      #importing tkinter module
from sklearn.tree import DecisionTreeRegressor       #importing decision tree from sklearn.Scikit-learn (Sklearn) is the most useful and robust library for machine learning in Python
import matplotlib.pyplot as plt       #Used for plotting,instead of matplotlib we use plt for every function.matplotlib work like MATLAB.
import seaborn as sns            #Seaborn is a library that uses Matplotlib underneath to plot graphs.
from sklearn.metrics import mean_squared_error  #The sklearn.metrics module implements functions assessing prediction error for specific purposes,measures the average of error squares i.e. the average squared difference between the estimated values and true value.
from math import sqrt               #from math library importing squareroot
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.linear_model import LinearRegression  #LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation
from sklearn import linear_model  #performing machine learning with linear models
import pandas as pd              # used for data science/data analysis and machine learning tasks.data manipulation library 
import numpy as np		#used for working with arrays, contains a multi-dimensional array and matrix data structures.


global filename
global rmse,rmse1,rmse2   #Root Mean Square Error 


main = tkinter.Tk()
main.title("Data analysis for manufacturers") #designing main screen
main.geometry("500x600")

def Decision():
    text.delete('1.0', END)  #function to clear the input text
    global rmse
    dataset = pd.read_csv('Dataset/vaccinations.csv') # taking data set,a new DataFrame with the data and labels from the file data. csv
    text.insert(END,dataset.head())                   # used to get the first n rows.
    dataset['date'] = pd.to_datetime(dataset['date']).dt.strftime('%Y-%m-%d')    #Convert your date strings into Pandas DateTime formats,Conver DataTime to Different Format
    dataFrame = pd.pivot_table(data=dataset, values='total_vaccinations', index='vaccine', columns='date', aggfunc='sum', fill_value=0)  #create a spreadsheet-style pivot table as a DataFrame.


    year_wise_sale = dataset.groupby(['vaccine'])['total_vaccinations'].sum()
    year_wise_sale.plot(figsize=(15, 6))
    

    x_train = 8
    y_train = 1
    y_test = 8

    dataset = dataFrame.values
    time_periods = dataset.shape[1]
    lag_loops = time_periods + 1 - x_train - y_train - y_test

    training = []
    for i in range(lag_loops):
        value = dataset[:,i:i+x_train+y_train]
        training.append(value)
    training = np.vstack(training)
    Xtrain, Ytrain = np.split(training,[x_train],axis=1)
    
    max_column_test = time_periods - x_train - y_train + 1
    testing = []
    for i in range(lag_loops,max_column_test):
        testing.append(dataset[:,i:i+x_train+y_train])
    testing = np.vstack(testing)
    Xtest, Ytest = np.split(testing,[x_train],axis=1)
    
    if y_train == 1:
        Ytrain = Ytrain.ravel()
        Ytest = Ytest.ravel()

    tree = DecisionTreeRegressor() 
    tree.fit(Xtrain,Ytrain)

    prediction = tree.predict(Xtest) 

    actual = []
    forecast = []
    i = len(Ytest)-1
    index = 0
    while i > 0:
        actual.append(Ytest[i])
        forecast.append(prediction[i])
        text.insert(END,'\n\nDay=%d, Forecasted=%f, Actual=%f' % (index+1, prediction[i], Ytest[i]))
        index = index + 1
        i = i - 1
        if len(actual) > 30:
            break

    rmse = sqrt(mean_squared_error(Ytest,prediction))
    text.insert(END,'\n\nRMSE : ',round(rmse,1))



    plt.figure(figsize=(10,6))  #figsize is a tuple of the width and height of the figure in inches
    plt.grid(True)
    plt.xlabel('Actual & Forecast Vaccines Manufacturing')
    plt.ylabel('Manufacturing Count')
    plt.plot(actual, 'ro-', color = 'blue')  # actual plot on the graph
    plt.plot(forecast, 'ro-', color = 'green')   # forecast plot on the graph
    plt.legend(['Required Vaccines', 'Forecasted Vaccines'], loc='upper left') #is a container of legend handles and legend texts,loc denotes location to place the legend,automatically creates a legend for any labeled plot elements
    plt.title('Decision Tree  Forecasting Graph')
    plt.show()

def Linear():
    global rmse1
    text.delete('1.0', END)
    dataset = pd.read_csv('Dataset/vaccinations.csv')
    text.insert(END,dataset.head())
    dataset['date'] = pd.to_datetime(dataset['date']).dt.strftime('%Y-%m-%d') 
    dataFrame = pd.pivot_table(data=dataset, values='total_vaccinations', index='vaccine', columns='date', aggfunc='sum', fill_value=0)


    year_wise_sale = dataset.groupby(['vaccine'])['total_vaccinations'].sum()
    year_wise_sale.plot(figsize=(15, 6))
    

    x_train = 8
    y_train = 1
    y_test = 8

    dataset = dataFrame.values
    time_periods = dataset.shape[1]
    lag_loops = time_periods + 1 - x_train - y_train - y_test

    training = []
    for i in range(lag_loops):
        value = dataset[:,i:i+x_train+y_train]
        training.append(value)
    training = np.vstack(training)
    Xtrain, Ytrain = np.split(training,[x_train],axis=1)
    
    max_column_test = time_periods - x_train - y_train + 1
    testing = []
    for i in range(lag_loops,max_column_test):
        testing.append(dataset[:,i:i+x_train+y_train])
    testing = np.vstack(testing)
    Xtest, Ytest = np.split(testing,[x_train],axis=1)
    
    if y_train == 1:
        Ytrain = Ytrain.ravel()
        Ytest = Ytest.ravel()

    tree = LinearRegression() 
    tree.fit(Xtrain,Ytrain)

    prediction = tree.predict(Xtest) 

    actual = []
    forecast = []
    i = len(Ytest)-1
    index = 0
    while i > 0:
        actual.append(Ytest[i])
        forecast.append(prediction[i])
        text.insert(END,'\n\nDay=%d, Forecasted=%f, Actual=%f' % (index+1, prediction[i], Ytest[i]))
        index = index + 1
        i = i - 1
        if len(actual) > 30:
            break

    rmse1 = sqrt(mean_squared_error(Ytest,prediction))
    text.insert(END,'\n\nRMSE : ',round(rmse1,1))



    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Actual & Forecast Vaccines Manufacturing')
    plt.ylabel('Manufacturing Count')
    plt.plot(actual, 'ro-', color = 'blue')
    plt.plot(forecast, 'ro-', color = 'green')
    plt.legend(['Required Vaccines', 'Forecasted Vaccines'], loc='upper left')
    #plt.xticks(wordloss.index)
    plt.title('Covid19 Vaccines Forecasting Graph')
    plt.show()

def Lasso():
    global rmse2
    text.delete('1.0', END)
    dataset = pd.read_csv('Dataset/vaccinations.csv')
    text.insert(END,dataset.head())
    dataset['date'] = pd.to_datetime(dataset['date']).dt.strftime('%Y-%m-%d') 
    dataFrame = pd.pivot_table(data=dataset, values='total_vaccinations', index='vaccine', columns='date', aggfunc='sum', fill_value=0)


    year_wise_sale = dataset.groupby(['vaccine'])['total_vaccinations'].sum()
    year_wise_sale.plot(figsize=(15, 6))
    

    x_train = 8
    y_train = 1
    y_test = 8

    dataset = dataFrame.values
    time_periods = dataset.shape[1]
    lag_loops = time_periods + 1 - x_train - y_train - y_test

    training = []
    for i in range(lag_loops):
        value = dataset[:,i:i+x_train+y_train]
        training.append(value)
    training = np.vstack(training)
    Xtrain, Ytrain = np.split(training,[x_train],axis=1)
    
    max_column_test = time_periods - x_train - y_train + 1
    testing = []
    for i in range(lag_loops,max_column_test):
        testing.append(dataset[:,i:i+x_train+y_train])
    testing = np.vstack(testing)
    Xtest, Ytest = np.split(testing,[x_train],axis=1)
    
    if y_train == 1:
        Ytrain = Ytrain.ravel()
        Ytest = Ytest.ravel()

    tree = linear_model.Lasso(alpha=0.1) 
    tree.fit(Xtrain,Ytrain)

    prediction = tree.predict(Xtest) 

    actual = []
    forecast = []
    i = len(Ytest)-1
    index = 0
    while i > 0:
        actual.append(Ytest[i])
        forecast.append(prediction[i])
        text.insert(END,'\n\nDay=%d, Forecasted=%f, Actual=%f' % (index+1, prediction[i], Ytest[i]))
        index = index + 1
        i = i - 1
        if len(actual) > 30:
            break

    rmse2 = sqrt(mean_squared_error(Ytest,prediction))
    text.insert(END,'\n\nRMSE : ',round(rmse2,1))



    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Actual & Forecast Vaccines Manufacturing')
    plt.ylabel('Manufacturing Count')
    plt.plot(actual, 'ro-', color = 'blue')
    plt.plot(forecast, 'ro-', color = 'green')
    plt.legend(['Required Vaccines', 'Forecasted Vaccines'], loc='upper left')
    plt.title('Lasso Regression Forecasting Graph')
    plt.show()


    

def graph():
    bars = ('Decision Tree','Logistic Regression','Lasso Regression')
    plt.title("Error Rates of the Algorithms")
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, [rmse,rmse1,rmse2])
    plt.xticks(y_pos, bars)
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Data Analysis')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=23,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Run Decision Tree", command=Decision)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

preButton = Button(main, text="Run Linear Regression", command=Linear)
preButton.place(x=380,y=100)
preButton.config(font=font1) 

featureButton = Button(main, text="Run Lasso Regression", command=Lasso)
featureButton.place(x=680,y=100)
featureButton.config(font=font1) 

lstmButton = Button(main, text="Comparison graph", command=graph)
lstmButton.place(x=50,y=150)
lstmButton.config(font=font1) 




main.config(bg='OliveDrab2')
main.mainloop()
