# Flask is the overall web framework
from flask import Flask, request, render_template, jsonify
# joblib is used to unpickle the model
import joblib
# json is used to prepare the result
import json
# import pandas and numpy
import pandas as pd
import numpy as np

# Libraries to prepocess data before prediction
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as skl_Pipeline
import xgboost as xgb

# create new flask app here
app = Flask(__name__)

# helper functions here

#Function to load the model from the file system (for lazy loading)
def load_model():
    with open("xgb_base_model.pkl", "rb") as f:
        return joblib.load(f)
#Alternate function to load the xgb model (currently using this function)
def load_xgb():
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("xgb_base_model.ubj")
    return xgb_model  # Return the loaded model

#Function to load the pipeline from the file system (for lazy loading)
def load_pipeline():
    with open("pipeline.pkl", "rb") as f:
        return joblib.load(f)

#Function for binning specific columns
def bin_data(df, column, bins, labels):
    df[f'{column}_BINNED'] = pd.cut(df[column], bins=bins, labels=labels, right=True).astype(str)
    return df

#Function for preprocessing and predicting 
def cancer_risk_pred_score(FIPS, X_URBSTAT, X_AGEG5YR, SEXVAR, GENHLTH, PHYSHLTH, MENTHLTH, 
                          HLTHPLN1, EXERANY2, SLEPTIM1, CVDINFR4, CVDCRHD4, CVDSTRK3, 
                          ASTHMA3, CHCCOPD2, HAVARTH4, ADDEPEV3, CHCKDNY2, DIABETE4, 
                          LASTDEN4, RMVTETH4, MARITAL, EDUCA, RENTHOM1, VETERAN3, EMPLOY1, 
                          CHILDREN, INCOME2, WEIGHT2, HTIN4, DEAF, BLIND, SEATBELT, HIVTST7, 
                          HIVRISK5, X_LLCPWT, HADEXAM, SMOKDRINK):
    """
    Given the variables, predict the probability risk of an individual having cancer
    """

    # Bin the PHYSHLTH, MENTHLTH, SLEPTIM1, and  CHILDREN columns 

    # Construct the 2D matrix of values that .predict is expecting
    X = [[FIPS, X_URBSTAT, X_AGEG5YR, SEXVAR, GENHLTH, PHYSHLTH, MENTHLTH, HLTHPLN1, EXERANY2, SLEPTIM1, 
          CVDINFR4, CVDCRHD4, CVDSTRK3, ASTHMA3, CHCCOPD2, HAVARTH4, ADDEPEV3, CHCKDNY2, DIABETE4, 
          LASTDEN4, RMVTETH4, MARITAL, EDUCA, RENTHOM1, VETERAN3, EMPLOY1, CHILDREN, INCOME2, WEIGHT2, 
          HTIN4, DEAF, BLIND, SEATBELT, HIVTST7, HIVRISK5, X_LLCPWT, HADEXAM, SMOKDRINK]]
   
    cols = ['FIPS', 'X_URBSTAT', 'X_AGEG5YR', 'SEXVAR', 'GENHLTH', 'PHYSHLTH', 'MENTHLTH', 
            'HLTHPLN1', 'EXERANY2', 'SLEPTIM1', 'CVDINFR4', 'CVDCRHD4', 'CVDSTRK3', 
            'ASTHMA3', 'CHCCOPD2', 'HAVARTH4', 'ADDEPEV3', 'CHCKDNY2', 'DIABETE4', 
            'LASTDEN4', 'RMVTETH4', 'MARITAL', 'EDUCA', 'RENTHOM1', 'VETERAN3', 'EMPLOY1', 
            'CHILDREN', 'INCOME2', 'WEIGHT2', 'HTIN4', 'DEAF', 'BLIND', 'SEATBELT', 'HIVTST7', 
            'HIVRISK5', 'X_LLCPWT', 'HADEXAM', 'SMOKDRINK']
    
    # Create a one row dataframe based on user input data
    df = pd.DataFrame(X, columns=cols)

    # Define your bins and labels here (as you did previously)
    physhlth_bins = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    physhlth_bin_labels = ['1-3', '4-6', '7-9', '10-12', '13-15', '16-18', '19-21', '22-24', '25-27', '28-30']

    sleeptim_bins = [0, 3, 6, 9, 12, 15, 18, 21, 24]
    sleeptim_bin_labels = ['1-3', '4-6', '7-9', '10-12', '13-15', '16-18', '19-21', '22-24']

    children_bins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 87]
    children_bin_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11+']

    df = bin_data(df, 'PHYSHLTH', physhlth_bins, physhlth_bin_labels)
    df = bin_data(df, 'MENTHLTH', physhlth_bins, physhlth_bin_labels)
    df = bin_data(df, 'SLEPTIM1', sleeptim_bins, sleeptim_bin_labels)
    df = bin_data(df, 'CHILDREN', children_bins, children_bin_labels)

    #Drop initial columns
    df = df.drop(columns= ['PHYSHLTH', 'MENTHLTH',  'SLEPTIM1', 'CHILDREN'])

    # Load pipeline
    pipe = load_pipeline()
    # Accessing the preprocessor step and transforming the data
    preprocessor = pipe.named_steps['preprocessor']
    df_transformed = preprocessor.transform(df)

    #Converting back to dataframe
    df = pd.DataFrame(df_transformed.toarray(), columns=pipe.get_feature_names_out())

    #Drop 'X_LLCPWT' column since it was not used to train the xgb model
    df = df.drop(columns= ['num__X_LLCPWT'])

    #Load model
    model = load_xgb()
    # Get the probabilities
    probabilities = model.predict_proba(df)
    probability_of_cancer = probabilities[0][1]

    return {"probability_of_cancer": probability_of_cancer}

# defining routes

@app.route('/', methods=['GET'])
def index():
    return render_template('index_3.html')

@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info(request.form)
    data = request.json
    try:
        FIPS = int(data['FIPS'])
        X_URBSTAT = int(data['X_URBSTAT'])
        X_AGEG5YR = int(data['X_AGEG5YR'])
        SEXVAR = int(data['SEXVAR'])
        GENHLTH = int(data['GENHLTH'])
        PHYSHLTH = int(data['PHYSHLTH'])
        MENTHLTH = int(data['MENTHLTH'])
        HLTHPLN1 = int(data['HLTHPLN1'])
        EXERANY2 = int(data['EXERANY2'])
        SLEPTIM1 = int(data['SLEPTIM1'])
        CVDINFR4 = int(data['CVDINFR4'])
        CVDCRHD4 = int(data['CVDCRHD4'])
        CVDSTRK3 = int(data['CVDSTRK3'])
        ASTHMA3 = int(data['ASTHMA3'])
        CHCCOPD2 = int(data['CHCCOPD2'])
        HAVARTH4 = int(data['HAVARTH4'])
        ADDEPEV3 = int(data['ADDEPEV3'])
        CHCKDNY2 = int(data['CHCKDNY2'])
        DIABETE4 = int(data['DIABETE4'])
        LASTDEN4 = int(data['LASTDEN4'])
        RMVTETH4 = int(data['RMVTETH4'])
        MARITAL = int(data['MARITAL'])
        EDUCA = int(data['EDUCA'])
        RENTHOM1 = int(data['RENTHOM1'])
        VETERAN3 = int(data['VETERAN3'])
        EMPLOY1 = int(data['EMPLOY1'])
        CHILDREN = int(data['CHILDREN'])
        INCOME2 = int(data['INCOME2'])
        WEIGHT2 = int(data['WEIGHT2'])
        HTIN4 = int(data['HTIN4'])
        DEAF = int(data['DEAF'])
        BLIND = int(data['BLIND'])
        SEATBELT = int(data['SEATBELT'])
        HIVTST7 = int(data['HIVTST7'])
        HIVRISK5 = int(data['HIVRISK5'])
        HADEXAM = int(data['HADEXAM'])
        SMOKDRINK = int(data['SMOKDRINK'])

        # Set a default value for X_LLCPWT if it is not provided
        X_LLCPWT = float(data.get('X_LLCPWT', 646.55))

    except KeyError as e:
        return jsonify({'error': f'Missing field: {e.args[0]}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {e}'}), 400

    result = cancer_risk_pred_score(FIPS, X_URBSTAT, X_AGEG5YR, SEXVAR, GENHLTH, PHYSHLTH, MENTHLTH, HLTHPLN1, EXERANY2, SLEPTIM1, 
                                    CVDINFR4, CVDCRHD4, CVDSTRK3, ASTHMA3, CHCCOPD2, HAVARTH4, ADDEPEV3, CHCKDNY2, DIABETE4, 
                                    LASTDEN4, RMVTETH4, MARITAL, EDUCA, RENTHOM1, VETERAN3, EMPLOY1, CHILDREN, INCOME2, WEIGHT2, 
                                    HTIN4, DEAF, BLIND, SEATBELT, HIVTST7, HIVRISK5, X_LLCPWT, HADEXAM, SMOKDRINK)

    # Return the complement of the cancer probability as a percentage, 
    # since class 1 represents 'No Cancer' and class 0 represents 'Has Cancer' 
    # as encoded by the label encoder during the training of the XGB model.
    return jsonify({'probability_of_cancer': round((1 - float(result['probability_of_cancer'])) * 100, 2)})


if __name__ == '__main__':
    app.run(debug=False)