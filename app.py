import pandas as pd
import numpy as np
import pickle
import streamlit as st
import os

# --- Set the Working Directory ---
target_dir = "C:/Users/alexg/Ambiente de Trabalho/Mestrado Data Science NOVA IMS/1st semester/Machine Learning/Project_github/ML_Project_Group52/project/Deployment of the model"
os.chdir(target_dir)

# --- Load the Pipeline ---
with open("trained_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# --- Reverse Target Mapping for Predictions ---
reverse_target_mapping = {
    0: '1. CANCELLED',
    1: '2. NON-COMP',
    2: '3. MED ONLY',
    3: '4. TEMPORARY',
    4: '5. PPD SCH LOSS',
    5: '6. PPD NSL',
    6: '7. PTD',
    7: '8. DEATH'
}

# --- Define Selected Features for Input ---
selected_features = [
    'Age at Injury', 'Average Weekly Wage', 'Gender', 'Age_Group',
    'County of Injury', 'Carrier Name', 'District Name', 'Zip Code',
    'Number of Dependents', 'Cause Injury Category',
    'Nature of Injury Category', 'Body Part Category'
]

# --- Streamlit App ---
def main():
    st.title("Insurance Claim Prediction App")
    st.write("Enter the details below to predict the claim outcome using the trained model.")

    # --- Collect User Inputs ---
    user_inputs = {}

    # Numeric inputs
    st.header("Numeric Inputs")
    user_inputs['Age at Injury'] = st.number_input("Enter Age at Injury", min_value=0.0, value=30.0)
    user_inputs['Average Weekly Wage'] = st.number_input("Enter Average Weekly Wage", min_value=0.0, value=1000.0)
    user_inputs['Number of Dependents'] = st.number_input("Enter Number of Dependents", min_value=0, step=1)

    # Categorical inputs
    st.header("Categorical Inputs")
    user_inputs['Gender'] = st.selectbox("Select Gender", ['M', 'F', 'Other'])
    user_inputs['County of Injury'] = st.text_input("Enter County of Injury", value="Unknown")
    user_inputs['Carrier Name'] = st.text_input("Enter Carrier Name", value="Unknown")
    user_inputs['District Name'] = st.text_input("Enter District Name", value="Unknown")
    user_inputs['Zip Code'] = st.text_input("Enter Zip Code", value="00000")
    user_inputs['Cause Injury Category'] = st.text_input("Enter Cause Injury Category", value="Unknown")
    user_inputs['Nature of Injury Category'] = st.text_input("Enter Nature of Injury Category", value="Unknown")
    user_inputs['Body Part Category'] = st.text_input("Enter Body Part Category", value="Unknown")

    # Ordinal input
    st.header("Ordinal Input")
    user_inputs['Age_Group'] = st.selectbox("Select Age Group", 
                                           ['Teen', 'Young Adult', 'Adult', 'Middle-Aged Adult', 'Older Adult', 'Senior'])

    # --- Submit Button ---
    if st.button("Predict Outcome"):
        try:
            # Convert user inputs to DataFrame
            input_df = pd.DataFrame([user_inputs])

            # Ensure selected features are in order
            input_df = input_df[selected_features]

            # Convert numeric columns explicitly to float
            numeric_columns = ['Age at Injury', 'Average Weekly Wage', 'Number of Dependents']
            input_df[numeric_columns] = input_df[numeric_columns].astype(float)

            # Debug: Show input data before transformation
            st.write("Input DataFrame:", input_df)

            # --- Preprocess and Predict in One Step ---
            prediction = pipeline.predict(input_df)[0]

            # Map the prediction to the target label
            predicted_label = reverse_target_mapping.get(prediction, "Unknown")

            # Display the prediction
            st.success(f"Predicted Outcome: {predicted_label}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- Run the App ---
if __name__ == "__main__":
    main()
