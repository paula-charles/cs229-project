import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

##FOR THE TED TALK DATA SET##

def data_import_ted(path):
    '''
    Create training data set, validation data set and test data set.
    The training data set represents 80% of the data, the other two 10%.
    They are split randomly.
    '''
    df = pd.read_csv(path, delimiter=',')
    training_data, remaining_data = train_test_split(df, test_size=0.2, random_state=13)
    validation_data,test_data = train_test_split(remaining_data,test_size=0.5, random_state = 11)

    return training_data, validation_data, test_data

def get_stats_and_plots_ted(train_set, valid_set, test_set):

    names = train_set['title']

    number_comments = train_set['comments']
    log_number_comments = np.log(number_comments)
    number_views = train_set['views']
    log_number_views = np.log(number_views)

    print(np.average(number_comments))

    train_plot = plt.scatter(log_number_comments,log_number_views)
    val_plot = plt.scatter(np.log(valid_set['comments']),np.log(valid_set['views']))
    test_plot = plt.scatter(np.log(test_set['comments']),np.log(test_set['views']))
    plt.title("TED talks, log-values")

    plt.legend((train_plot, val_plot, test_plot),
            ('Training set', 'Validation set', 'Test set'))

    plt.show()

def vanilla_linear_regression_ted(train_set, valid_set):
    number_comments = train_set['comments']
    log_number_comments = np.log(number_comments)
    number_views = train_set['views']
    log_number_views = np.log(number_views)
    X = np.array(number_comments).reshape((-1, 1))
    y=number_views

    #Constant estimator
    print("Using constant estimate :")
    print("Best constant estimate for views = ",np.average(number_views))

    print("Standard prediction error = ",np.sqrt(np.average((np.average(train_set['views'])-valid_set['views'])**2)))

    #Linear estimator
    print("Using linear estimate :")
    reg = LinearRegression().fit(X,y)
    print("Estimated Views = ",reg.intercept_,"+ Comments * ",reg.coef_)
    valid_predict = reg.predict(np.array(valid_set['comments']).reshape((-1, 1)))
    print("Standard prediction error = ",np.sqrt(np.average((valid_predict-valid_set['views'])**2)))

    ### Regression with log-transformed data for views and comments
    X = np.array(log_number_comments).reshape((-1, 1))
    y=log_number_views

    reg = LinearRegression().fit(X, y)

    #Linear estimator
    print("Using linear estimate :")
    reg = LinearRegression().fit(X,y)
    print("log(Estimated Views) = ",reg.intercept_,"+ log(Comments) * ",reg.coef_)
    print("Estimated Views = ",np.exp(reg.intercept_),"* Comments^",reg.coef_)

    valid_predict = reg.predict(np.array(valid_set['comments']).reshape((-1, 1)))
    #print(np.sqrt(np.average((np.exp(valid_predict)-valid_set['views'])**2)))

    print("Standard prediction error = ",np.sqrt(np.average((valid_predict-valid_set['views'])**2)))

##FOR THE YOUTUBE VIDEOS DATA SET

def data_import_youtube(path):
    df = pd.read_csv(path, delimiter=',')
    df= df[df['comment_count']!=0]
    df = df[df['view_count']!=0]
    training_data, remaining_data = train_test_split(df, test_size=0.2)
    validation_data,test_data = train_test_split(remaining_data,test_size=0.5, random_state = 11)
    return training_data, validation_data, test_data

def get_stats_and_plots_yt(train_set, valid_set, test_set):
    names = train_set['title']

    number_comments = train_set['comment_count']
    log_number_comments = np.log(number_comments)
    number_views = train_set['view_count']
    log_number_views = np.log(number_views)

    print(np.average(number_comments))
    plt.clf()
    train_plot = plt.scatter(log_number_comments,log_number_views)
    val_plot = plt.scatter(np.log(valid_set['comment_count']),np.log(valid_set['view_count']))
    test_plot = plt.scatter(np.log(test_set['comment_count']),np.log(test_set['view_count']))
    plt.title("US Youtube videos, log-values")

    plt.legend((train_plot, val_plot, test_plot),
               ('Training set', 'Validation set', 'Test set'))

    plt.show()

def vanilla_linear_regression_yt(train_set, valid_set):
    names = train_set['title']

    number_comments = train_set['comment_count']
    log_number_comments = np.log(number_comments)
    number_views = train_set['view_count']
    log_number_views = np.log(number_views)

    X = np.array(number_comments).reshape((-1, 1))
    y=number_views

    #Constant estimator
    print("Using constant estimate :")
    print("Best constant estimate for views = ",np.average(number_views))

    print("Standard prediction error = ",np.sqrt(np.average((np.average(train_set['view_count'])-valid_set['view_count'])**2)))

    #Linear estimator
    print("Using linear estimate :")
    reg = LinearRegression().fit(X,y)
    print("Estimated Views = ",reg.intercept_,"+ Comments * ",reg.coef_)
    valid_predict = reg.predict(np.array(valid_set['comment_count']).reshape((-1, 1)))
    print("Standard prediction error = ",np.sqrt(np.average((valid_predict-valid_set['view_count'])**2)))

    ### Linear regression with log-transformed inputs
    X = np.array(log_number_comments).reshape((-1, 1))
    y=log_number_views

    print("Using constant estimate :")
    print("Best constant estimate for views = ",np.exp(np.average(log_number_views)))

    print("Standard prediction error = ",np.sqrt(np.average((np.exp(np.average(log_number_views))-valid_set['view_count'])**2)))

    print("log-based prediction error = ",np.sqrt(np.average((np.average(log_number_views)-np.log(valid_set['view_count']))**2)))
    
    print("Using linear estimate :")
    reg = LinearRegression().fit(X,y)
    print("Estimated log(Views) = ",reg.intercept_,"+ log(Comments) * ",reg.coef_)
    valid_log_predict = reg.predict(np.array(np.log(valid_set['comment_count'])).reshape((-1, 1)))
    print("Standard prediction error = ",np.sqrt(np.average((np.exp(valid_log_predict)-valid_set['view_count'])**2)))
    print("Log-based prediction error = ",np.sqrt(np.average((valid_log_predict-np.log(valid_set['view_count']))**2)))





