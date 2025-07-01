import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open("logistic_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🚢 Titanic Survival Prediction")

# User Inputs
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 0, 100, 25)
SibSp = st.number_input("Siblings/Spouses Aboard", min_value=0)
Parch = st.number_input("Parents/Children Aboard", min_value=0)
Fare = st.number_input("Fare", min_value=0.0)
Embarked_Q = st.selectbox("Embarked at Queenstown (Q)?", [0, 1])
Embarked_S = st.selectbox("Embarked at Southampton (S)?", [0, 1])

# Convert sex to numeric
Sex = 0 if Sex == "male" else 1

# Predict
if st.button("Predict"):
    input_data = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_Q, Embarked_S]])
    prediction = model.predict(input_data)[0]
    result = "Survived 🟢" if prediction == 1 else "Did Not Survive 🔴"
    st.success(f"The model predicts: {result}")
