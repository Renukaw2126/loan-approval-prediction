import streamlit as st       # to create a simple web application
import pandas as pd
import pickle as pk

model = pk.load(open('model.pkl','rb'))          # read binary access
scaler = pk.load(open('scaler.pkl','rb'))    

st.title = ('**Smart Loan Eligibility Checker**')
st.title
no_of_dep = st.slider('How many dependents do you have?',0,5)
graduation = st.selectbox('Education',["",'Graduated','Not Graduated'])
self_emp = st.selectbox('Are you Self-Employed ?',["",'Yes','No'])
Annual_income= st.slider('Choose Annual Income',0,10000000)
Loan_amount= st.slider('Choose Loan Amount',0,10000000)
Loan_Dur= st.slider('Choose Loan Duration',0,20)
cibil= st.slider('Choose cibil score',0,900)
Assets = st.slider('Choose Assets',0,10000000)

if graduation == 'Graduated':
    graduation_s = 0
else :
    graduation_s = 1

if self_emp == 'No':
    self_emp_s = 0
else :
    self_emp_s = 1

if st.button("predict"):
     pred_data = pd.DataFrame([[ no_of_dep, graduation_s,self_emp_s,Annual_income,Loan_amount,Loan_Dur,cibil,Assets]],columns=['no_of_dependents','education','self_employed','income_annum','loan_amount','loan_term','cibil_score','Assets'])
     pred_data= scaler.transform(pred_data)
     predict = model.predict(pred_data)

     if predict[0] == 1:
         st.markdown('**Loan Is Approved!!**')
     else:
        st.markdown('**Loan Is Rejected!!**')
         

