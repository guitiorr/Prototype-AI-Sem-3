import streamlit as st

def symptom_selection(symptoms_list):
    # Add a placeholder option that is not typically used as a symptom
    placeholder_option = "Select a symptom"

    # Create a list to store the selected symptoms as floats
    selected_symptoms: float = []

    select_symptomps_styled_text = f"<p style='font-family: Arial, sans-serif; font-size: 18px;'><span style='color: #FF5733;'>Select Symptoms:</span>"
    st.write(select_symptomps_styled_text, unsafe_allow_html=True)

    # Helper function to get the index of a symptom in the list as a float
    def get_symptom_index(symptom):
        if symptom in symptoms_list:
            index = symptoms_list.index(symptom)
            return float(index)
        else:
            return None
        
    for i in range(1, 18):  # Create up to 17 select boxes
        symptom_input = st.selectbox(f"Select {i} symptom", [placeholder_option] + symptoms_list)
        if symptom_input != placeholder_option:
            selected_symptoms.append(float(get_symptom_index(symptom_input)))
        elif symptom_input == placeholder_option:
            selected_symptoms.append(float(200))

    return selected_symptoms
