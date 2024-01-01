import streamlit as st
import pandas as pd
from selectboxinputs import symptom_selection
from model import AImodel
from pycaret.classification import setup, compare_models, pull, save_model

df = pd.read_csv("dataset.csv")


with st.sidebar:
    st.title("Disease prediction prototype")
    choice = st.radio("Navigation", ["Pycaret - Finding best model", "Disease Prediction"])

if choice == "Pycaret - Finding best model":
    st.title("Pycaret - Finding best model")
    chosen_target = st.selectbox("Choose Target", df.columns)
    if st.button('Run trial'):
        setup(df, target=chosen_target, use_gpu=True)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')


if choice == "Disease Prediction":
    symptoms_df = pd.read_csv("Symptom-severity.csv")
    symptoms_list = symptoms_df["Symptom"].tolist()

    # Use the symptom_selection function from the package
    selected_symptoms, selected_symptoms_set = symptom_selection(symptoms_list), set()

    # Create a button to submit
    submit_button = st.button("Submit")
    warning_displayed = False  # Flag to track if the warning has been displayed

   # Check if every select box input is 200
    if all(symptom == 200 for symptom in selected_symptoms):
        st.warning(f"Select at least 1 symptom lol")
        submit_button = False
    else:
        # Check if a symptom is selected more than once
        warning_displayed = False
        for symptom in selected_symptoms:
            if selected_symptoms.count(symptom) > 1 and symptom != 200:  # Assuming 200 is a placeholder, adjust as needed
                if not warning_displayed:
                    st.warning(f"Symptom '{symptom}' is selected more than once. Please select each symptom only once.")
                    warning_displayed = True  # Set the flag to indicate the warning has been displayed
                submit_button = False



    if submit_button:
        st.empty()
        selected_symptoms_text = selected_symptoms
        styled_text = f"<p style='font-family: Arial, sans-serif; font-size: 18px;'><span style='color: #FF5733;'>Selected Symptoms:</span> {selected_symptoms_text}</p>"
        st.markdown(styled_text, unsafe_allow_html=True)

        symptom_integers = [int(symptom) for symptom in selected_symptoms[:17]]  # Convert the first 17 symptoms to integers

        # Unpack the first 17 elements of symptom_integers into the AImodel function
        AImodel(*symptom_integers)
