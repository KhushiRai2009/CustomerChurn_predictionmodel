import os
import streamlit as st
import pandas as pd
import pickle

# Loading Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR,"..","models","churn_model.pkl") #".." means concat

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found at: {path}")
        st.stop()

    with open(path, "rb") as f:
        return pickle.load(f)
    
model_package = load_model(MODEL_PATH)

model = model_package["model"]
preprocessor = model_package["preprocessor"]
svd = model_package["svd"]

# UI Design

st.set_page_config(page_title="Churn Prediction",layout="centered")

st.title("Customer Churn Prediction")
st.markdown("--")
st.subheader("Provide customer details to predict churn.")

# Inputing Features

with st.form("prediction_form"):
    col1,col2 = st.columns(2)

    with col1:
        gender=st.selectbox("Gender",["Male","Female"])
        SeniorCitizen=st.selectbox("Senior Citizen",[0,1])
        Partner=st.selectbox("Partner",["Yes","No"])
        Dependents=st.selectbox("Dependents",["Yes","No"])
        tenure=st.slider("Tenure(months)",0,72,12)
        PhoneService=st.selectbox("Phone Service",["Yes","No"])
        MultipleLines=st.selectbox("Multiple Lines",["No","Yes","No phone service"])

    with col2:
        InternetService=st.selectbox("Internet Service",["DSL","Fiber optic","No"])
        OnlineSecurity=st.selectbox("Online Security",["Yes","No","No internet service"])
        OnlineBackup=st.selectbox("Online Backup",["Yes","No","No internet service"])
        DeviceProtection=st.selectbox("Device Protection",["Yes","No","No internet service"])
        TechSupport=st.selectbox("Tech Support",["Yes","No","No internet service"])
        StreamingTV=st.selectbox("Streaming TV",["Yes","No","No internet service"])
        StreamingMovies=st.selectbox("Streaming Movies",["Yes","No","No internet service"])

    Contract=st.selectbox("Contract",["Month-to-month","One year","Two year"])
    PaperlessBilling=st.selectbox("Paperless Billing",["Yes","No"])
    PaymentMethod=st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer(automatic)",
            "Credit card(automatic)"
        ]
    )

    MonthlyCharges=st.number_input("Monthly charges",0.0,200.0,70.0)
    TotalCharges=st.number_input("Total Charges",0.0,10000.0,1000.0)

    submit=st.form_submit_button("Predict")

    # Model Prediction
   
    if submit:
        input_data=pd.DataFrame([{
            "gender":gender,
            "SeniorCitizen":SeniorCitizen,
            "Partner":Partner,
            "Dependents":Dependents,
            "tenure":tenure,
            "PhoneService":PhoneService,
            "MultipleLines":MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges
        }])

        try:
            processed_data = preprocessor.transform(input_data)

            if svd is not None:
                processed_data = svd.transform(processed_data)

            prediction = model.predict(processed_data)[0]

            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(processed_data)[0][1]
            else:
                probability = None

            st.subheader("Prediction Result")

            if prediction == 1:
                st.error("Customer is likely to Churn")
            else:
                st.success("Customer is NOT likely to Churn")

            if probability is not None:
                st.info(f"Churn Probability: {probability:.2f}")

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

        
