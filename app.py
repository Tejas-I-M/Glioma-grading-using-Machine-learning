import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load(r"C:\Users\Tejas\OneDrive\Desktop\glioma_grading\data\glicomaxl\rf_model.pkl")

# Get expected features (try different methods to retrieve them)
try:
    expected_features = model.feature_names_in_  # For scikit-learn >= 1.0
except AttributeError:
    # Manually specify all features your model expects (from the error message)
    expected_features = ['Age', 'Gender', 'IDH1',
        'ATRX', 'BCOR', 'CIC', 'CSMD3', 'EGFR',		'TP53',		'PTEN',	'MUC16',	'PIK3CA',	'NF1',	'PIK3R1',	'FUBP1',	'RB1',	'NOTCH1','SMARCA4',	'GRIN2A',	'IDH2',	'FAT4',	'PDGFRA'
    ]

st.title("Glioma Grade Predictor")

# Input form
st.sidebar.header("Patient Data")
age = st.sidebar.number_input("Age", min_value=1, max_value=100)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
idh1 = st.sidebar.selectbox("IDH1 Mutation", ["MUTATED", "NOT_MUTATED"])

#Add inputs for other important features (example)
atrx = st.sidebar.selectbox("ATRX Status", ["Mutated", "Wildtype"])
bcor = st.sidebar.number_input("BCOR Expression", min_value=0.0, max_value=10.0)

# Add all other features your model needs

# Predict
if st.sidebar.button("Predict"):
    # Create dictionary with all expected features, defaulting missing ones to 0
    input_dict = {feature: 0 for feature in expected_features}
    
    # Update with actual values we collected
    input_dict.update({
        'Age': age,
        'Gender': 1 if gender == "Male" else 0,
        'IDH1': 1 if idh1 == "MUTATED" else 0,
        # Update with other features:
        # 'ATRX': 1 if atrx == "Mutated" else 0,
        # 'BCOR': bcor,
    })
    
    # Create DataFrame with features in correct order
    input_df = pd.DataFrame([input_dict])[expected_features]
    
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Grade: {prediction}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.write("Model expects these features:", expected_features)