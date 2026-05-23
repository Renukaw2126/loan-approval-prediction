# Loan Approval Prediction System

A Machine Learning based Loan Approval Prediction System that predicts whether a loan application will be approved or rejected based on applicant information such as income, assets, credit score, employment details, and loan amount.

## Project Overview

This project uses Machine Learning techniques to automate loan approval prediction and reduce manual evaluation efforts. The model analyzes applicant data and provides loan approval decisions through an interactive web application built with Streamlit.

## Features

✔ Loan approval prediction using Machine Learning  
✔ Data preprocessing and feature scaling  
✔ User-friendly Streamlit interface  
✔ Real-time prediction system  
✔ Model serialization using Pickle  

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Pickle

## Project Structure

```bash
loan-approval-prediction/
│── app.py
│── loan_prediction.py
│── model.pkl
│── scaler.pkl
│── requirements.txt
│── README.md
```

## Dataset Features

The model uses applicant information such as:

- Annual Income
- Loan Amount
- Credit Score
- Total Assets Value
- Residential Assets Value
- Commercial Assets Value
- Luxury Assets Value
- Bank Asset Value
- Employment Status
- Education Level

## Installation

Clone the repository:

```bash
git clone https://github.com/Renukaw2126/loan-approval-prediction.git
```

Move into project directory:

```bash
cd loan-approval-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Usage

1. Enter applicant details
2. Provide financial information
3. Click Predict
4. View loan approval result



## Future Improvements

- Improve prediction accuracy
- Add more financial indicators
- Deploy using cloud services
- Add visualization dashboard

## Author

Renuka Wagh

GitHub:
https://github.com/Renukaw2126
 
 ## Live Demo

Check out the live app here: [Loan Approval Prediction System/Smart Loan Eligibility Checker](https://loan-approval-prediction-ltmjcqq4ghdirfazo5qrhj.streamlit.app/)

