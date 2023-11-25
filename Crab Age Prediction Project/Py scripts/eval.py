# this script evaluates the accuracy of the crab prediction model 

def evaluate(y_pred, y_true):
    """
    Evaluates the accuracy of the random forest regression model built in the "initial_analysis.ipynb" notebook
    Parameters:
        y_pred (1d array): predicted ages 
        y_true (1d array): actual ages

    Returns:
        mse, rmse, mae, r2
    """
    mse = mean_squared_error(y_true,y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true,y_pred)
    
    return mse, rmse, mae, r2

if __name__ == '__main__':

    df_path = sys.stdin
    
    #load test data from file
    df = pd.read_csv(df_path)

    print('The total numer of testing records: {}\n'.format(len(df)))
    
    y_true = df['Age']
    y_pred = df['Predicted Age']
    evaluate(y_true,y_pred)

