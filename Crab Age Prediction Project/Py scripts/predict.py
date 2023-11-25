import pandas as pd 
import numpy as np 
import pickle  
import sys 

def clean_data(df):
    """
    Perform data cleaning operations outlined in the notebook initial_analysis.ipynb 
    
    Expected parameter:
        df (pd.DataFrame): the raw data

    Returned:
        Cleaned df
    """
    
    # check for expected columns and return error message if mismatch

    expected_cols = ['Sex', 'Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Age']
    assert (df.columns == expected_cols).all(), 'Error: mismatch in the column names, check input matches expected input'

    df.dropna(inplace = True) # remove missing rows

    numerical_cols = expected_cols
    numerical_cols.remove('Sex') # new list with names of numerical columns 
    df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
    
      # df[df.drop('Sex', axis=1, inplace=True)] = df[df.drop('Sex', axis=1, inplace=True)].apply(pd.to_numeric, error='coerce')

    # use One Hot Encoding to encode the categorical feature ('Sex')
    cf = pd.get_dummies(df['Sex'])
    df = pd.concat([df, cf], axis=1) # join encoded sex cols back to original df
    df.drop('Sex', axis=1, inplace=True) # remove original 'Sex' column

    return df

def predict(df, model_file_path):
    """
    Parameters:
        df (pd.DataFrame): the raw data
        model_file_path (str): file path to saved ml model 

    Returns:
        Array of predicted values 
    """
    model = pickle.load(open(model_file_path, 'rb')) # load the model using pickle
    y_pred = model.predict(df)

    return y_pred

if __name__ == '__main__':
    df_path = sys.stdin
    
    sys.stderr.write('Reading model & csv file\n')   
    #df_path = '.\data_clean.csv'
    model_path = './best_xgb.dat'
    df = pd.read_csv(df_path)
    
    #clean the data
    sys.stderr.write('Cleaning dataset\n')
    clean_df = clean_data(df) 
    
    sys.stderr.write('Make Prediction\n')
    y_pred = predict(clean_df.drop('Age',axis = 1), model_path)
    
    df['Age'] = df['Age'].astype(int)
    df['Predicted Age'] = np.round(y_pred).astype(int)
    
    df.to_csv(sys.stdout, index = False)
    



