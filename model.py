
import pandas as pd
import numpy as np

from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import RFE

def evaluate_arima_model(train, test, target, arima_order):
    """ Evaluates an ARIMA model based on arima_order argument, train set, test set, and target. 
    Outputs error, actual test values, and predictions for every timestep in test"""
    train_target = train[target]
    test_target = test[target]
    history = [x for x in train_target]

    # Make predictions
    predictions = []
    for t in range(len(test_target)):
        print(f"\tTesting {arima_order} {t}/{len(test_target)}", end="\r")
        model = ARIMA(history, order = arima_order)
        model_fit = model.fit()
        # Forecast returns forecast value, standard error, and confidence interval - only need forecast value ([0])
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        # Adds the latest test value to history so it can be used to train
        history.append(test_target[t])
    error = mean_squared_error(test_target, predictions)
    print("\n")
    return error, test_target, predictions

def evaluate_models(train, test, target, p_values, d_values, q_values):
    """ Evaluates an ARIMA model per the inputted p, d, and q values. Returns a pandas dataframe with the results from the model"""
    mses=[]
    prediction_list=[]
    actual_test = []
    orders = []
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                orders.append(order)
                try:
                    mse, test_target, predictions = evaluate_arima_model(train, test, target, order)
                    mses.append(mse)
                    prediction_list.append(predictions)
                    actual_test.append(test_target)
                except KeyboardInterrupt:
                    print("Keyboard interrupt")
                    raise
                except:
                    print(f"{order} didn't work, continuing with next order")
                    continue
    results_df = pd.DataFrame.from_records(orders, columns = ['p','d','q'])
    results_df["mse"] = mses
    results_df["test_predictions"] = prediction_list
    results_df["test_actual"] = actual_test
    return results_df

def scale_datasets(train, validate, test, target, features_to_use, features_to_scale):
    """Returns split dataframes with scaled features (and those features that did not need scaling)"""
    
    # Segment out features into individual dataframes
    X_train = train[features_to_use]
    X_validate = validate[features_to_use]
    X_test = test[features_to_use]

    # Segment out target into individual dataframe
    y_train = train[[target]]
    y_validate = validate[[target]]
    y_test = test[[target]]

    # Will scale features using StandardScaler as sigmas are orders of magnitude different from log_ret
    scaler = StandardScaler()

    # Fit scaler to train. Transform validate and test based on fitted scaler. Concatenate to df with non-scaled features.
    X_train_scaled = pd.concat([X_train.drop(columns = features_to_scale),
                                pd.DataFrame(data = scaler.fit_transform(X_train[features_to_scale]), 
                                             columns = features_to_scale, index = X_train.index)], 
                               axis=1)
    X_validate_scaled = pd.concat([X_validate.drop(columns = features_to_scale),
                                pd.DataFrame(data = scaler.transform(X_validate[features_to_scale]), 
                                             columns = features_to_scale, index = X_validate.index)], 
                               axis=1)
    X_test_scaled = pd.concat([X_test.drop(columns = features_to_scale),
                                pd.DataFrame(data = scaler.transform(X_test[features_to_scale]), 
                                             columns = features_to_scale, index = X_test.index)], 
                               axis=1)
                               
    return X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate, y_test

def get_top_features(X_train_scaled, y_train, model, target, n_features):
    """ Performs recursive feature elimination using the inputted model. Returns the top n_features"""
    lm = model
    
    rfe = RFE(lm, n_features_to_select= n_features)

    rfe.fit(X_train_scaled, y_train[[target]])

    # Get mask of the columns selected
    feature_mask = rfe.support_

    # Get list of column names
    rfe_feature = X_train_scaled.iloc[:,feature_mask].columns.tolist()

    # view list of columns and their ranking

    # get the ranks
    var_ranks = rfe.ranking_
    # get the variable names
    var_names = X_train_scaled.columns.tolist()
    # combine ranks and names into a df for clean viewing
    rfe_ranks_df = pd.DataFrame({'Var': var_names, 'Rank': var_ranks})
    # sort the df by rank
    rfe_ranks_df.sort_values('Rank')
    
    return rfe_feature

def predict_regression(models, train, validate, test, features_to_use, features_to_scale, perform_feature_selection, num_features):
    """Fits and predicts using inputted list of models. Outputs RMSE results, y_train, and y_validate with individual model results"""
    target = 'fwd_log_ret'
    
    X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate, y_test = scale_datasets(train, 
                                                                                                   validate, 
                                                                                                   test, 
                                                                                                   target, 
                                                                                                   features_to_use, 
                                                                                                   features_to_scale)
    
    rmses_train = {}
    rmses_validate = {}
    
    # iterate through each model
    for reg_model in models:
        
        # Gets a string name for the model
        model_name = reg_model.__repr__().split('()')[0]

#         print(model_name)
        
        # Whether to use recursive feature elimination
        if perform_feature_selection:

            top_features = get_top_features(X_train_scaled, y_train, reg_model,target, num_features)

            X_train_scaled_featured = X_train_scaled[top_features]
            X_validate_scaled_featured  = X_validate_scaled[top_features]
            X_test_scaled_featured = X_test_scaled[top_features]
            
        else:
            
            X_train_scaled_featured = X_train_scaled.copy()
            X_validate_scaled_featured = X_validate_scaled.copy()
            X_test_scaled_featured = X_test_scaled.copy()
        
        # Fit model to the training data
        reg_model.fit(X_train_scaled_featured, y_train.fwd_log_ret)
        
        # Predict on train and add results to y_train
        y_train[model_name] = reg_model.predict(X_train_scaled_featured)
        
        # Get RMSE metric for train
        rmse_train = mean_squared_error(y_train.fwd_log_ret, y_train[model_name], squared=False)
        
        # Predict on validate
        y_validate[model_name] = reg_model.predict(X_validate_scaled_featured)

        # Get RMSE metric for validate
        rmse_validate = mean_squared_error(y_validate.fwd_log_ret, y_validate[model_name], squared=False)

        # Print RMSE results for train and validate
        # print(f"RMSE for {model_name}\nTraining/In-Sample: ", rmse_train, 
              # "\nValidation/Out-of-Sample: ", rmse_validate)

        rmses_validate[model_name] = rmse_validate
        rmses_train[model_name] = rmse_train
        
    return rmses_train, rmses_validate, y_train, y_validate

def calculate_regression_results(models, rmses_train, rmses_validate, validate, y_validate):
    """Generates average trade and RMSE dataframe from results of regression modeling"""
    
    validate_results = pd.DataFrame()
    # Get names of each model
    model_names = [m.__repr__().split('()')[0] for m in models]
    # Add close prices to y_validate to enable calculated trade return
    validate_results["close"] = validate.close
    validate_results["next_day_close"] = validate.close.shift(-1)

    model_rmse_validate = []
    model_rmse_train = []
    model_average_trade_returns = []
    model_average_pct_trade_returns = []

    # Iterate through each model
    for mod in model_names:
        # Create a column saying whether we would go long or not (short)
        validate_results[mod+"_long"] = y_validate[mod]>0
        # Calculate the return that day (assumes always goes long or short every day)
        validate_results[mod+"_ret"] = np.where(validate_results[mod+"_long"], validate_results.next_day_close-validate_results.close, validate_results.close-validate_results.next_day_close)
        # Calculate the pct return 
        validate_results[mod+"_pct_ret"] = validate_results[mod+"_ret"]/validate_results.close
        
        # print(mod,round(y_validate[mod+"_ret"].mean(),2))
        model_average_trade_returns.append(validate_results[mod+"_ret"].mean())
        model_average_pct_trade_returns.append(validate_results[mod+"_pct_ret"].mean())
        model_rmse_validate.append(rmses_validate[mod])
        model_rmse_train.append(rmses_train[mod])
        
    # Add forward return column to y_validate for baseline comparison
    validate_results['daily_return'] = validate.fwd_ret
    validate_results['pct_daily_return'] = validate.fwd_pct_chg

    # Create dataframe of avg trade and rmse values
    avg_trade_model = pd.DataFrame(data = {'avg_trade':model_average_trade_returns,
                                           'pct_avg_trade':model_average_pct_trade_returns,
                                           'validate_rmse':model_rmse_validate,
                                          'train_rmse':model_rmse_train}, index = model_names)

    # Concatenate to the avg_trade_model df the avg trade we'd get if just bought every day and sold next (close to close)
    # This is a baseline. RMSE not defined for this case.
    avg_trade_model= pd.concat([avg_trade_model,pd.DataFrame({'avg_trade':validate_results.daily_return.mean(),
                                                              'pct_avg_trade':validate_results.pct_daily_return.mean(),
                                                              'validate_rmse':np.nan,
                                                              'train_rmse':np.nan}, index = ['baseline_reg'])])

    return avg_trade_model.sort_values(by='pct_avg_trade',ascending=False), validate_results

def get_rolling_predictions(train, validate, test, model_under_test, target, features_to_use, features_to_scale, perform_feature_selection):
    """Predicts target for each day in validate based on rolling window of previous n days"""
    
    X_train_scaled, X_validate_scaled, X_test_scaled, y_train_rolled, y_validate, y_test = scale_datasets(train, 
                                                                                                   validate, 
                                                                                                   test, 
                                                                                                   target, 
                                                                                                   features_to_use, 
                                                                                                   features_to_scale)            
    # Whether to use recursive feature elimination
    if perform_feature_selection:

        top_features = get_top_features(X_train_scaled, y_train_rolled, model_under_test,target, n_features=10)

        X_train_scaled_featured_rolled = X_train_scaled[top_features]
        X_validate_scaled_featured_rolled  = X_validate_scaled[top_features]
        X_test_scaled_featured_rolled = X_test_scaled[top_features]

    else:

        X_train_scaled_featured_rolled = X_train_scaled.copy()
        X_validate_scaled_featured_rolled = X_validate_scaled.copy()
        X_test_scaled_featured_rolled = X_test_scaled.copy()
        
    # Create empty lists to hold predictions. Actuals included here for easier bookkeeping
    train_rolling_predictions = []
    train_rolling_actuals = []
    validate_rolling_predictions = []
    validate_rolling_actuals = []
    
    # Iterate through each row in validate
    for validate_row in range(len(X_validate_scaled_featured_rolled)):
        
        # Print out which row we're on 
        print(f"{model_under_test} {validate_row+1}/{len(X_validate_scaled_featured_rolled)} Train X range: {X_train_scaled_featured_rolled.index.min().date()} - {X_train_scaled_featured_rolled.index.max().date()}",end="\r")
        # print(f"\nTrain X range: {X_train_scaled_featured_rolled.index.min().date()} - {X_train_scaled_featured_rolled.index.max().date()}",end="\r")

        # Fit the model to the training data
        # print(f"Fitting to train, X: {X_train_scaled_featured_rolled.index.min().date()} - {X_train_scaled_featured_rolled.index.max().date()}, y: {y_train_rolled.index.min().date()} - {y_train_rolled.index.max().date()}") 
        model_under_test.fit(X_train_scaled_featured_rolled, y_train_rolled[target])

        # Predict on Train
        train_prediction = model_under_test.predict(X_train_scaled_featured_rolled)
        train_actual = y_train_rolled[target]

        # Append train results to list
        # print(f"First train prediction {train_prediction[0]} vs actual {train_actual[0]}")
        train_rolling_predictions.append(train_prediction)
        train_rolling_actuals.append(train_actual)

        # Predict on validate, only for one row at a time
        validate_rolling_predictions.append(model_under_test.predict(X_validate_scaled_featured_rolled.iloc[validate_row].array.reshape(1,-1)))
        validate_rolling_actuals.append(y_validate.iloc[validate_row][target])

        # Remove the first row from X train and y train
        X_train_scaled_featured_rolled = X_train_scaled_featured_rolled.iloc[1:]
        y_train_rolled= y_train_rolled.iloc[1:]

        # Add the latest row from X validate and y validate
        X_train_scaled_featured_rolled = X_train_scaled_featured_rolled.append(X_validate_scaled_featured_rolled.iloc[validate_row])
        y_train_rolled = y_train_rolled.append(y_validate.iloc[validate_row])
    
    return train_rolling_predictions, train_rolling_actuals, validate_rolling_predictions, validate_rolling_actuals

def predict_classification(models, train, validate, test, features_to_use, features_to_scale, perform_feature_selection):
    """Fits and predicts using inputted list of classification models. Outputs Classification results, y_train, and y_validate with individual model results"""
    
    target = 'fwd_close_positive'
    
    X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate, y_test = scale_datasets(train, 
                                                                                                   validate, 
                                                                                                   test, 
                                                                                                   target, 
                                                                                                   features_to_use, 
                                                                                                   features_to_scale)
    
    accuracies_train = {}
    accuracies_validate = {}
    
    # iterate through each model
    for class_model in models:
        
        # Gets a string name for the model
        model_name = class_model.__repr__().split('()')[0]
        
        # Whether to use recursive feature elimination
        if perform_feature_selection:

            top_features = get_top_features(X_train_scaled, y_train, class_model,target, 16)

            X_train_scaled_featured = X_train_scaled[top_features]
            X_validate_scaled_featured  = X_validate_scaled[top_features]
            X_test_scaled_featured = X_test_scaled[top_features]
            
        else:
            
            X_train_scaled_featured = X_train_scaled.copy()
            X_validate_scaled_featured = X_validate_scaled.copy()
            X_test_scaled_featured = X_test_scaled.copy()
        
        # Fit model to the training data
        class_model.fit(X_train_scaled_featured, y_train.fwd_close_positive)
        
        # Predict on train and add results to y_train
        y_train[model_name] = class_model.predict(X_train_scaled_featured)
        
        # Get Accuracy metric for train
        accuracy_train = accuracy_score(y_train.fwd_close_positive, y_train[model_name])
        
        # Predict on validate
        y_validate[model_name] = class_model.predict(X_validate_scaled_featured)
        

        # Get RMSE metric for validate
        accuracy_validate = accuracy_score(y_validate.fwd_close_positive, y_validate[model_name])

        accuracies_validate[model_name] = accuracy_validate
        accuracies_train[model_name] = accuracy_train
        accuracies_train['baseline'] = y_train[target].mean()
        accuracies_validate['baseline'] = y_validate[target].mean()
        
    return accuracies_train, accuracies_validate, y_train, y_validate

def calculate_classification_results(models, accuracies_train, accuracies_validate, validate, y_validate):
    """Generates average trade and Accuracy dataframe from results of classification modeling"""
    validate_results = pd.DataFrame()
    # Get names of each model
    model_names = [m.__repr__().split('()')[0] for m in models]
    # Add close prices to y_validate to enable calculated trade return
    validate_results["close"] = validate.close
    validate_results["next_day_close"] = validate.close.shift(-1)

    model_accuracy_validate = []
    model_accuracy_train = []
    model_average_trade_returns = []
    model_average_pct_trade_returns = []

    # Iterate through each model
    for mod in model_names:
        # Skip baseline model
        if mod == 'baseline':
            continue
        # Create a column saying whether we would go long based on the the model prediction
        validate_results[mod+"_long"] = y_validate[mod]>0
        # Calculate the return that day (assumes always goes long or short every day)
        validate_results[mod+"_ret"] = np.where(validate_results[mod+"_long"], 
                                          validate_results.next_day_close-validate_results.close, 
                                          validate_results.close-validate_results.next_day_close)
        # Calculate the pct return 
        validate_results[mod+"_pct_ret"] = validate_results[mod+"_ret"]/validate_results.close
        # print(mod,round(y_validate[mod+"_ret"].mean(),2))
        model_average_trade_returns.append(validate_results[mod+"_ret"].mean())
        model_average_pct_trade_returns.append(validate_results[mod+"_pct_ret"].mean())
        model_accuracy_validate.append(accuracies_validate[mod])
        model_accuracy_train.append(accuracies_train[mod])
        
    # Add forward return column to y_validate for baseline comparison
    validate_results['daily_return'] = validate.fwd_ret
    validate_results['pct_daily_return'] = validate.fwd_pct_chg

    # Create dataframe of avg trade and rmse values
    avg_trade_model = pd.DataFrame(data = {'avg_trade':model_average_trade_returns,
                                           'pct_avg_trade':model_average_pct_trade_returns,
                                           'validate_accuracy':model_accuracy_validate,
                                          'train_accuracy':model_accuracy_train}, index = model_names)

    # Concatenate to the avg_trade_model df the avg trade we'd get if just bought every day and sold next (close to close)
    # This is a baseline. RMSE not defined for this case.
    avg_trade_model= pd.concat([avg_trade_model,pd.DataFrame({'avg_trade':validate_results.daily_return.mean(),
                                                              'pct_avg_trade':validate_results.pct_daily_return.mean(),
                                                              'validate_accuracy':accuracies_validate['baseline'],
                                                              'train_accuracy':accuracies_train['baseline']}, index = ['baseline_class'])])

    return avg_trade_model.sort_values(by='pct_avg_trade',ascending=False), validate_results

def conventional_split(split_data, reg_models, class_models, features_to_use, features_to_scale):
    
    # Trading strategy results - dictionary to hold key for each cryptocurrency
    avg_trade_model_results = {}
    # Predictions from validate
    class_validate_results = {}
    reg_validate_results = {}

    for k in split_data.keys():
        print(f"Train/validate: {k}")
        train, validate, test = split_data[k]
        ### Iterate through regression models, uses existing train, validate, test split
        # Specify regression models to test. Feature selection using recursive feature elimination is also available.

        # Fits model using train and gets predictions for validate
        rmses_train, rmses_validate, y_train, y_validate = predict_regression(reg_models, 
                                                                                    train, 
                                                                                    validate, 
                                                                                    test, 
                                                                                    features_to_use, 
                                                                                    features_to_scale,
                                                                                    perform_feature_selection=True,
                                                                                    num_features = 5)
        # Consolidates results into dataframe. Outputs average trade information for each model.
        reg_avg_trade_model_results, reg_v_results =  calculate_regression_results(reg_models, rmses_train, rmses_validate, validate, y_validate)

        reg_validate_results[k] = reg_v_results

        # Fits model using train and gets predictions for validate
        accuracies_train, accuracies_validate, y_train, y_validate = predict_classification(class_models, 
                                                                                    train, 
                                                                                    validate, 
                                                                                    test, 
                                                                                    features_to_use, 
                                                                                    features_to_scale,
                                                                                    perform_feature_selection=False)
        # Put results into dataframe
        class_avg_trade_model_results, class_v_results  =  calculate_classification_results(class_models, accuracies_train, accuracies_validate, validate, y_validate)

        # Add classification results from standard data split
        avg_trade_model_results[k]= reg_avg_trade_model_results.append(class_avg_trade_model_results).sort_values(by='pct_avg_trade',ascending=False)

        class_validate_results[k] = class_v_results

    conventional_split_model_results = pd.DataFrame()
    for k in avg_trade_model_results.keys():
        results = avg_trade_model_results[k]
        results.index= [i+'_'+k for i in results.index]
        conventional_split_model_results = pd.concat([conventional_split_model_results, results])

    # Calculate the percent dropoff in accuracy or RMSE from train to validate. Lower RMSE and higher accuracy are better but generally trend opposite
    conventional_split_model_results["dropoff"] = np.select([conventional_split_model_results.train_rmse.isna(),conventional_split_model_results.train_accuracy.isna()],
                                             [(conventional_split_model_results.validate_accuracy - conventional_split_model_results.train_accuracy)/conventional_split_model_results.train_accuracy,
                                              (conventional_split_model_results.validate_rmse - conventional_split_model_results.train_rmse)/conventional_split_model_results.train_rmse])
    conventional_split_model_results.sort_values(by=['pct_avg_trade','dropoff'], ascending=False).head(10)[['pct_avg_trade', 'avg_trade', 'train_rmse','validate_rmse',
           'train_accuracy','validate_accuracy','dropoff']]
    
    return reg_validate_results, class_validate_results, conventional_split_model_results

def rolling_reg_models(split_data, reg_models, target, features_to_use, features_to_scale):
    """Performs rolling regression fit/test from split_data and for models indicated in reg_models """
    # target = 'fwd_log_ret'
    all_product_results = {}

    # Specify regression models to test. Feature selection using recursive feature elimination is also available.
    # reg_models = [SVR(kernel='linear',gamma=0.1)]#, LinearRegression(), TweedieRegressor(), LassoLars(), DecisionTreeRegressor()]

    for k in split_data.keys():
        reg_model_results = {}
        train, validate, test = split_data[k]
        for model_under_test in reg_models:

            print("testing",k,model_under_test)

            model_name = model_under_test.__repr__().split('()')[0]

            # Perform rolling predictions
            train_rolling_predictions, train_rolling_actuals, validate_rolling_predictions, validate_rolling_actuals = get_rolling_predictions(train, 
                                                                    validate, 
                                                                    test, 
                                                                    model_under_test,
                                                                    target, 
                                                                    features_to_use, 
                                                                    features_to_scale,
                                                                    True)
            # Calculate the mean of all train RMSEs
            train_rmse = np.mean([mean_squared_error(train_rolling_actuals[i],train_rolling_predictions[i],squared=False) for i in range(len(train_rolling_actuals))])

            # Calculate validate RMSE
            validate_rmse = mean_squared_error(validate_rolling_actuals, [v[0] for v in validate_rolling_predictions], squared=False)

            print(model_name,"avg validate rmse",validate_rmse)

            # Create a dataframe with actual validate log returns, predictions, close prices, next day close prices
            validate_res = pd.DataFrame()
            validate_res['actual'] = validate_rolling_actuals
            validate_res['predictions'] = [v[0] for v in validate_rolling_predictions]

            # Set result index to validate's index
            validate_res.index = validate.index

            # Transfer close values over to results dataframe to allow for return calculation
            validate_res["close"] = validate.close
            validate_res["next_day_close"] = validate.close.shift(-1)
            # Create a column for whether we would go long or not (short) based on the sign of the predictions value
            validate_res["go_long"] = validate_res['predictions']>0
            # Calculate the return that day (assumes always goes long or short every day)
            validate_res["ret"] = np.where(validate_res["go_long"], validate_res.next_day_close-validate_res.close, validate_res.close-validate_res.next_day_close)
            validate_res["pct_ret"] = validate_res["ret"]/validate_res.close

            # Store validate results in dictionary
            reg_model_results[model_name] = validate_res
            reg_model_results[model_name+"_validate_rmse"] = validate_rmse
            reg_model_results[model_name+"_train_rmse"] = train_rmse

        # Append to all products dictionary 
        all_product_results[k] = reg_model_results
        
    return all_product_results
        
def consolidate_rolling_reg(all_product_results):
    """ Create dataframe of results """
    avg_trades = []
    avg_pct_trade = []
    train_rmses = []
    validate_rmses = []
    indices = []
    # iterate through each key in regression model results dictionary
    for k in all_product_results.keys():
        for key in all_product_results[k]:

            # Store RMSE and avg trade data
            if '_train_rmse' in key:
                train_rmses.append(all_product_results[k][key])
            elif '_validate_rmse' in key:
                validate_rmses.append(all_product_results[k][key])
            else:
                # Add average trade, average percent trade, and index name
                avg_trades.append(all_product_results[k][key].ret.mean())
                avg_pct_trade.append(all_product_results[k][key].pct_ret.mean())
                indices.append(key+"_"+k+"_single_step")

    reg_model_results_df = pd.DataFrame(data={'avg_trade':avg_trades,
                                              'pct_avg_trade':avg_pct_trade,
                                              'train_rmse':train_rmses,
                                              'validate_rmse':validate_rmses}, index = indices)
    reg_model_results_df["dropoff"] = (reg_model_results_df.validate_rmse - reg_model_results_df.train_rmse)/reg_model_results_df.train_rmse


    return reg_model_results_df


def rolling_class_models(split_data, class_models, target, features_to_use, features_to_scale):
    
    # class_models = [LogisticRegression(C=10), 
    #             DecisionTreeClassifier(max_depth=None),
    #             KNeighborsClassifier(n_neighbors=10), 
    #             KNeighborsClassifier(n_neighbors=100), 
    #             KNeighborsClassifier(n_neighbors=1000)]
    class_model_results = {}
    all_product_class_results = {}
    # target = 'fwd_close_positive'

    for k in split_data.keys():
        train, validate, test = split_data[k]
        for model_under_test in class_models:

            model_name = model_under_test.__repr__().split('()')[0]

            train_rolling_predictions, train_rolling_actuals, validate_rolling_predictions, validate_rolling_actuals = get_rolling_predictions(train,validate,test, model_under_test, target, features_to_use, features_to_scale, False)

            # Calculate validate accuracy
            validate_accuracy = accuracy_score(validate_rolling_actuals, [v[0] for v in validate_rolling_predictions])

            print(k, model_name,"validate accuracy",validate_accuracy)

            # Create a dataframe with actual validate log returns, predictions, close prices, next day close prices
            validate_res = pd.DataFrame()
            validate_res['actual'] = validate_rolling_actuals
            validate_res['predictions'] = [v[0] for v in validate_rolling_predictions]

            validate_res.index = validate.index

            validate_res["close"] = validate.close
            validate_res["next_day_close"] = validate.close.shift(-1)
            # Create a column saying whether we would go long or not (short) based on the 
            validate_res["go_long"] = validate_res['predictions']>0
            # Calculate the return that day (assumes goes long or short every day)
            validate_res["ret"] = np.where(validate_res["go_long"], validate_res.next_day_close-validate_res.close, validate_res.close-validate_res.next_day_close)
            validate_res["pct_ret"] = validate_res["ret"]/validate_res.close

            class_model_results[model_name] = validate_res
            class_model_results[model_name+"_accuracy"] = validate_accuracy

                # Create baseline dataframe
            baseline = pd.DataFrame(index = validate.index)
            baseline["close"] = validate.close
            baseline["next_day_close"] = validate.close.shift(-1)
            # Just predict most common value
            baseline["predictions"] = train.fwd_close_positive.mode()[0]
            # Where prediction is true, go long
            baseline["go_long"] = baseline["predictions"]
            # Calculate the return that day (assumes always goes long or short every day)
            baseline["ret"] = np.where(baseline["go_long"], baseline.next_day_close-baseline.close, baseline.close-baseline.next_day_close)

            class_model_results["baseline"] = baseline
            
        all_product_class_results[k] = class_model_results

    return all_product_class_results

def consolidate_rolling_class(all_product_class_results):
    """ Create dataframe of rolling classification results """
    avg_trades = []
    accuracies = []
    indices = []
    for k in all_product_class_results.keys():
        for key in all_product_class_results[k]:

            if '_accuracy' in key:
                accuracies.append(all_product_class_results[k][key])
            elif key == 'baseline':
                
                # Calculate baseline accuracy
                if class_res_df[k][key].predictions.mode()[0]:
                    # If mode is true (baseline True)
                    baseline_accuracy = (class_res_df[[k][key].ret>0).mean()
                else:
                    # If mode is False (baseline False)
                    baseline_accuracy = 1-(class_res_df[k][key].ret>0).mean()

                accuracies.append(baseline_accuracy)
                avg_trades.append(all_product_class_results[k][key].ret.mean())
                indices.append(key+"_single_step")
            else:
                avg_trades.append(all_product_class_results[k][key].ret.mean())
                indices.append(key+"_"+k+"_single_step")

    # Single step classification results
    class_model_results_df = pd.DataFrame(data={'avg_trade':avg_trades,'accuracy':accuracies}, index = indices)

    # avg_trade_model_results = avg_trade_model_results.append(class_model_results_df)

    return class_model_results_df