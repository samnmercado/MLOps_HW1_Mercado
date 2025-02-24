from solids.imports import *
import mlflow
import mlflow.sklearn

# Function to train k-Nearest Neighbors
@op
def train_kNN(X, y):
    Number_trials = 3
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
    
    best_knn = KNeighborsRegressor(n_neighbors=n_neighbors_tuning[best_index]).fit(X_train, y_train)
    
    # Log the model and metrics to MLflow
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(best_knn, "kNN_model")
        mlflow.log_param("n_neighbors", n_neighbors_tuning[best_index])
        mlflow.log_metric("train_score", train_score[best_index])
        mlflow.log_metric("test_score", test_score[best_index])
        mlflow.log_metric("run_time", run_time)
    
    return ['kNN Regressor', train_score[best_index], test_score[best_index], 
            'n_neighbors = {0}'.format(n_neighbors_tuning[best_index]), 'N/A', run_time]

# Function to train Gradient Boosting Machine
@op
def train_GBM(X, y):
    Number_trials = 3
    n_estimators_tuning = (50, 100)
    score_train = []
    score_test = []
    
    for seed in range(Number_trials):
        training_mse = []
        test_mse = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        for n_estimators in n_estimators_tuning:
            start_time = time.time()  # Time the training process
            gbm = GradientBoostingRegressor(n_estimators=n_estimators).fit(X_train, y_train)
            
            # Calculate Mean Squared Error as a performance metric
            training_mse.append(mean_squared_error(y_train, gbm.predict(X_train)))
            test_mse.append(mean_squared_error(y_test, gbm.predict(X_test)))
            
            run_time = time.time() - start_time
        
        score_train.append(training_mse)
        score_test.append(test_mse)

    train_score = np.mean(np.sqrt(score_train), axis=0)
    test_score = np.mean(np.sqrt(score_test), axis=0)
    best_index = np.argmin(test_score)
    
    best_gbm = GradientBoostingRegressor(n_estimators=n_estimators_tuning[best_index]).fit(X_train, y_train)
    
    # Log the model and metrics to MLflow
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(best_gbm, "GBM_model")
        mlflow.log_param("n_estimators", n_estimators_tuning[best_index])
        mlflow.log_metric("train_score", train_score[best_index])
        mlflow.log_metric("test_score", test_score[best_index])
        mlflow.log_metric("run_time", run_time)
    
    return ['GBM Regressor', train_score[best_index], test_score[best_index], 
            'n_estimators = {0}'.format(n_estimators_tuning[best_index]), 'N/A', run_time]

# Function to train Random Forest
@op
def train_RF(X, y):
    Number_trials = 3
    n_estimators_tuning = (50, 100)
    score_train = []
    score_test = []
    
    for seed in range(Number_trials):
        training_mse = []
        test_mse = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        for n_estimators in n_estimators_tuning:
            start_time = time.time()  # Time the training process
            rf = RandomForestRegressor(n_estimators=n_estimators).fit(X_train, y_train)
            
            # Calculate Mean Squared Error as a performance metric
            training_mse.append(mean_squared_error(y_train, rf.predict(X_train)))
            test_mse.append(mean_squared_error(y_test, rf.predict(X_test)))
            
            run_time = time.time() - start_time
        
        score_train.append(training_mse)
        score_test.append(test_mse)

    train_score = np.mean(np.sqrt(score_train), axis=0)
    test_score = np.mean(np.sqrt(score_test), axis=0)
    best_index = np.argmin(test_score)
    
    best_rf = RandomForestRegressor(n_estimators=n_estimators_tuning[best_index]).fit(X_train, y_train)
    
    # Log the model and metrics to MLflow
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(best_rf, "RF_model")
        mlflow.log_param("n_estimators", n_estimators_tuning[best_index])
        mlflow.log_metric("train_score", train_score[best_index])
        mlflow.log_metric("test_score", test_score[best_index])
        mlflow.log_metric("run_time", run_time)
    
    return ['RF Regressor', train_score[best_index], test_score[best_index], 
            'n_estimators = {0}'.format(n_estimators_tuning[best_index]), 'N/A', run_time]