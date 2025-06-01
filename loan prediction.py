import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
# import dataset
data = pd.read_csv("loan_approval_dataset.csv")
print(data.head())
data.drop(columns=['loan_id'], inplace=True) #drop the loan_id column which is not needed

data.columns=data.columns.str.strip() #remove the space of columns which is before the column name 
data.columns
#combine all assets to one column 

data['Assets'] = data.residential_assets_value + data. commercial_assets_value + data. luxury_assets_value + data. bank_asset_value 
# then drop the columns after combining

data.drop(columns= ['residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value'], inplace=True)

data.isnull().sum() #To check any null value in data

data.dropna(inplace=True)
print(data)
data.education.unique() # to remove space before column education
def clean_data(st):
    st = st.strip()
    return st
clean_data('Graduate')
data.education= data.education.apply(clean_data)
data.education.unique
data['education']= data['education'].replace(['Graduate','Not Graduate'],[1,0]) #to make string into integer to improve the performance of model
data.self_employed= data.self_employed.apply(clean_data)
data['self_employed']= data['self_employed'].replace(['Yes','No'],[1,0])
data.loan_status= data.loan_status.apply(clean_data)
data.loan_status.unique
data['loan_status']= data['loan_status'].replace(['Approved','Rejected'],[1,0])


from sklearn.model_selection import train_test_split
input_data = data.drop(columns=['loan_status'])
output_data = data['loan_status']
print(input_data)
print(output_data)
# splitting dataset
x_train,x_test,y_train,y_test = train_test_split(input_data,output_data,test_size=0.2)

print(x_train.shape,x_test.shape,y_train.shape,y_test.shape ) #to check the after splitting size of dataset
print("x_test shape:", x_test.shape)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train_scaled,y_train)
model.score(x_test_scaled,y_test)
pred_data = pd.DataFrame([[ 2, 1,0,9600000,29900000,12,778,50700000]],columns=['no_of_dependents','education','self_employed','income_annum','loan_amount','loan_term','cibil_score','Assets'])
pred_data=scaler.transform(pred_data)
print("Test Accuracy:", model.score(x_test_scaled, y_test))
print(model.predict(pred_data))
import pickle as pk
pk.dump(model,open ('model.pkl','wb'))
pk.dump(scaler ,open ('scaler.pkl','wb'))






