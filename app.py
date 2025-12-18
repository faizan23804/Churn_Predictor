import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
from tensorflow.keras.models import load_model



model=load_model('model.keras',compile=False)

with open('ohe_preprocessor.pkl','rb') as file:
    ohe_preprocessor=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

def preprocess_input(data):
    data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
    return data

ohe = ohe_preprocessor.named_transformers_["geo"]

## streamlit app
st.title("CUSTOMBER CHURN PREDICTION")

# User input
geography = st.selectbox('Geography', ohe.categories_[0])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_df = pd.DataFrame({
    'Geography': [geography],
    'CreditScore': [credit_score],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

input_df = preprocess_input(input_df)
input_processed = ohe_preprocessor.transform(input_df)
input_df_scaled = scaler.transform(input_processed)


prediction = model.predict(input_df_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')

