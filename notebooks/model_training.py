import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

Number_trials = 3

# Function to train k-Nearest Neighbors
def train_kNN(X, y):
    n_neighbors_tuning = (2, 6)
    score_train = []
    score_test = []
    
    for seed in range(Number_trials):
        training_mse = []
        test_mse = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        for n_neighbors in n_neighbors_tuning:
            start_time = time.time()  # Time the training process
            knn = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X_train, y_train)
            
            # Calculate Mean Squared Error as a performance metric
            training_mse.append(mean_squared_error(y_train, knn.predict(X_train)))
            test_mse.append(mean_squared_error(y_test, knn.predict(X_test)))
            
            run_time = time.time() - start_time
        
        score_train.append(training_mse)
        score_test.append(test_mse)

    train_score = np.mean(np.sqrt(score_train), axis=0)
    test_score = np.mean(np.sqrt(score_test), axis=0)
    best_index = np.argmin(test_score)
    
    return ['kNN Regressor', train_score[best_index], test_score[best_index], 
            'n_neighbors = {0}'.format(n_neighbors_tuning[best_index]), 'N/A', run_time]

# Function to train Logistic Regression
def train_logistic(X, y, reg):
    C = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
    score_train = []
    score_test = []
    weighted_coefs = []
    
    # Start timing the process
    start_time = time.time()  # Initialize start_time here
    
    for seed in range(Number_trials):
        training_accuracy = []
        test_accuracy = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        for alpha_run in C:
            if reg == 'l1':
                lr = LogisticRegression(C=alpha_run, penalty=reg, solver='liblinear', fit_intercept=True).fit(X_train, y_train)
            elif reg == 'l2':
                lr = LogisticRegression(C=alpha_run, penalty=reg, dual=False, fit_intercept=True).fit(X_train, y_train)
            
            training_accuracy.append(lr.score(X_train, y_train))
            test_accuracy.append(lr.score(X_test, y_test))
            coefs = lr.coef_[0]
            weighted_coefs.append(coefs)

        score_train.append(training_accuracy)
        score_test.append(test_accuracy)

    mean_coefs = np.mean(weighted_coefs, axis=0)
    score = np.mean(score_test, axis=0)
    train_score = np.mean(score_train, axis=0)
    
    top_predictor = X.columns[np.argmax(np.abs(mean_coefs))]
    abs_mean_coefs = np.abs(mean_coefs)
    coefs_count = len(abs_mean_coefs)
    
    fig, ax = plt.subplots(figsize=(3, 15))
    ax.barh(np.arange(coefs_count), sorted(abs_mean_coefs))
    ax.set_yticks(np.arange(coefs_count))
    ax.set_yticklabels(X.columns[np.argsort(abs_mean_coefs)])

    run_time = (time.time() - start_time)  # Calculate runtime after the training process
    return ['Logistic ({0})'.format(reg), np.amax(train_score), np.amax(score), \
            'C = {0}'.format(C[np.argmax(score)]), top_predictor, run_time]

# Function to train Random Forest
def train_RF(X, y):
    max_depth_tuning = [2, 3, 4, 5]
    score_train = []
    score_test = []
    weighted_coefs = []

    # Start timing the process
    start_time = time.time()  # Initialize start_time here
    
    for seed in range(Number_trials):
        training_accuracy = []
        test_accuracy = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        for max_depth_run in max_depth_tuning:
            rf = RandomForestClassifier(n_estimators=100, max_depth=max_depth_run).fit(X_train, y_train)
            training_accuracy.append(rf.score(X_train, y_train))
            test_accuracy.append(rf.score(X_test, y_test))
            coefs = rf.feature_importances_
            weighted_coefs.append(coefs)
                
        score_train.append(training_accuracy)
        score_test.append(test_accuracy)
    
    mean_coefs = np.mean(weighted_coefs, axis=0)
    score = np.mean(score_test, axis=0)
    train_score = np.mean(score_train, axis=0)

    top_predictor = X.columns[np.argmax(np.abs(mean_coefs))]
    abs_mean_coefs = np.abs(mean_coefs[:])
    
    fig, ax = plt.subplots(figsize=(3, 7))
    ax.barh(np.arange(len(abs_mean_coefs)), sorted(abs_mean_coefs))
    ax.set_yticks(np.arange(len(abs_mean_coefs)))
    ax.set_yticklabels(X.columns[np.argsort(abs_mean_coefs)])

    run_time = (time.time() - start_time)
    return ['Random Forest', np.amax(train_score), np.amax(score), \
            'Max_depth = {0}'.format(max_depth_tuning[np.argmax(score)]), top_predictor, run_time]

# Function to train Gradient Boosting Method
def train_GBM(X, y):
    max_depth_tuning = [2, 3, 4, 5]
    score_train = []
    score_test = []
    weighted_coefs = []

    # Start timing the process
    start_time = time.time()  # Initialize start_time here
    
    for seed in range(Number_trials):
        training_accuracy = []
        test_accuracy = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        for max_depth_run in max_depth_tuning:
            gbm = GradientBoostingClassifier(n_estimators=100, max_depth=max_depth_run).fit(X_train, y_train)
            training_accuracy.append(gbm.score(X_train, y_train))
            test_accuracy.append(gbm.score(X_test, y_test))
            coefs = gbm.feature_importances_
            weighted_coefs.append(coefs)

        score_train.append(training_accuracy)
        score_test.append(test_accuracy)
    
    mean_coefs = np.mean(weighted_coefs, axis=0)
    score = np.mean(score_test, axis=0)
    train_score = np.mean(score_train, axis=0)

    top_predictor = X.columns[np.argmax(np.abs(mean_coefs))]
    abs_mean_coefs = np.abs(mean_coefs[:])

    fig, ax = plt.subplots(figsize=(3, 7))
    ax.barh(np.arange(len(abs_mean_coefs)), sorted(abs_mean_coefs))
    ax.set_yticks(np.arange(len(abs_mean_coefs)))
    ax.set_yticklabels(X.columns[np.argsort(abs_mean_coefs)])

    run_time = (time.time() - start_time)
    return ['Gradient Boosting Method', np.amax(train_score), np.amax(score), \
            'Max_depth = {0}'.format(max_depth_tuning[np.argmax(score)]), top_predictor, run_time]

