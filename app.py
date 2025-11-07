import streamlit as st 
import tensorflow as tf 
import pandas as pd 
import numpy as np 
import pickle
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from tensorflow.keras.models import load_model


model = load_model('notebook/model.h5')

with open('notebook/preprocessing/label_enc.pkl','rb') as obj: 
    lb_enc = pickle.load(obj)

with open('notebook/preprocessing/ohe_enc.pkl','rb') as obj: 
    ohe_enc = pickle.load(obj)

with open('notebook/preprocessing/scaler.pkl','rb') as obj: 
    scaler = pickle.load(obj)



st.title("Churn Prediction System")




CreditScore=st.number_input("Credit Score")
geography= st.selectbox("Geography",ohe_enc.categories_[0])
gender = st.selectbox("Gender", lb_enc.classes_)
age = st.slider("Age",18,90)
tenure = st.slider("Tenure",0,10)
balance = st.number_input("Balance")
nbrprod = st.slider("Number of products",1,4)
crdit_card =st.selectbox("Has Credit Card",[0,1])
isactivemem =st.selectbox("Is Active Member",[0,1])
salary = st.number_input("Estimated salary")

#  convert input_data into Dataframe 
input_data = pd.DataFrame(
    {
        'CreditScore':[CreditScore],
        'Geography':[geography],
        'Gender':[gender],
        'Age':[age],
        'Tenure':[tenure],
        'Balance':[balance],
        'NumOfProducts':[nbrprod],
        'HasCrCard':[crdit_card],
        'IsActiveMember':[isactivemem],
        'EstimatedSalary':[salary]

    }
)

# convert categorial data into numerical data
input_data['Gender'] = lb_enc.transform(input_data['Gender'])
geo_enc = ohe_enc.transform([input_data['Geography'],]).toarray()
geo_enc_df = pd.DataFrame(geo_enc,columns = ohe_enc.get_feature_names_out())



#combine 
input_data = pd.concat([input_data.drop('Geography',axis=1),geo_enc_df],axis=1)


# scaling the data 
test_data = scaler.transform(input_data)

#churn prediction 
prediction = model.predict(test_data)
pred_prob = prediction[0][0]

if st.button('Submit'):
    if pred_prob > 0.5:
        st.write(f"The Employer is likely to churn and its probability to churn {pred_prob*100}%")
    else:
        st.write(f"The Employer is not likely to churn and its probability to churn {pred_prob*100}%")