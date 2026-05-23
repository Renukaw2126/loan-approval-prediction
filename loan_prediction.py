import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import pickle as pk

# import dataset
data = pd.read_csv("loan_approval_dataset.csv")
print(data.head())

# drop the loan_id column which is not needed
data.drop(columns=['loan_id'], inplace=True)

# remove the space before column names
data.columns = data.columns.str.strip()

# combine all assets to one column
data['Assets'] = data.residential_assets_value + data.commercial_assets_value + data.luxury_assets_value + data.bank_asset_value

# drop the individual asset columns after combining
data.drop(columns=['residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'], inplace=True)

# check for null values
data.isnull().sum()
data.dropna(inplace=True)

# clean string columns
def clean_data(st):
    return st.strip()

data.education = data.education.apply(clean_data)
data['education'] = data['education'].replace(['Graduate', 'Not Graduate'], [1, 0])

data.self_employed = data.self_employed.apply(clean_data)
data['self_employed'] = data['self_employed'].replace(['Yes', 'No'], [1, 0])

data.loan_status = data.loan_status.apply(clean_data)
data['loan_status'] = data['loan_status'].replace(['Approved', 'Rejected'], [1, 0])
data['loan_status'] = data['loan_status'].astype(int)

# split into input and output
input_data = data.drop(columns=['loan_status'])
output_data = data['loan_status']

# split dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# train Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train_scaled, y_train)

# print accuracy
print("Test Accuracy:", round(model.score(x_test_scaled, y_test) * 100, 2), "%")

# test prediction
pred_data = pd.DataFrame([[2, 1, 0, 9600000, 29900000, 12, 778, 50700000]],
                          columns=['no_of_dependents', 'education', 'self_employed',
                                   'income_annum', 'loan_amount', 'loan_term',
                                   'cibil_score', 'Assets'])
pred_data = scaler.transform(pred_data)
print("Sample Prediction:", model.predict(pred_data))

# save model and scaler
pk.dump(model, open('model.pkl', 'wb'))
pk.dump(scaler, open('scaler.pkl', 'wb'))
print("Model and scaler saved successfully!")
