    # -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 20:22:44 2024

@author: sk sunil
"""

import numpy as np
import pickle
import streamlit as st


# loading saved model

loaded_model = pickle.load(open('C:/Users/Priti Jadhav/Downloads/Model Deploy/trained_model.sav','rb'))

# creating the function for prediction

def fraudulent_prediction(ipt_df):
    

    # coverting input data into numpy array
    ipt_df_to_np_array = np.asarray(ipt_df)

    # resharing array as we are predicting one instance
    ipt_df_reshaped = ipt_df_to_np_array.reshape(1,-1)

    pred = loaded_model.predict(ipt_df_reshaped)
    print(pred)

    if (pred[0]==0):
        return'No Fraud'
    else:
        return'Fraud'


def main():
    
    # create streamlite app
    st.title('Fraud Prediction App')

    # input fields for features values on the main screen
    st.header('Enter Transaction Details')
    step = st.number_input('step')
    type = st.selectbox("type",('PAYMENT','CASH_IN','CASH_OUT','TRANSFER','DEBIT'))
    amount = st.number_input('amount')
    oldbalanceOrg = st.number_input('oldbalanceOrg')
    newbalanceOrig = st.number_input('newbalanceOrig')
    oldbalanceDest = st.number_input('oldbalanceDest')
    newbalanceDest = st.number_input('newbalanceDest')
    
    label_mapping={'PAYMENT':1,'CASH_IN':2,'CASH_OUT':3,'TRANSFER':4,'DEBIT':5}
    
    type=label_mapping[type]
        
    # code for prediction
    predictions = ''
    
        
    # creating button for prediction
    
    if st.button('Prediction Result'):
        predictions = fraudulent_prediction([step,type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest])
        
    st.success(predictions)   




if __name__ == '__main__':
    main()   
