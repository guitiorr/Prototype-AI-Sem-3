# Disease prediction AI using Streamlit

This is my final project for my Artificial Intelligence course in my 3rd semester, credits to kaggle dataset


# How to run
run using "streamlit main.py" in the terminal

# First glances
![image](https://github.com/user-attachments/assets/9c0c51cd-4c85-4721-beaf-1a81cdc39f90)


# Finding best model using PyCaret
I implemented PyCaret to determine which AI Model suits the best for the dataset, the target colum should be "disease"

# Disease prediction page
In the dataset, there are up to 18 rows of symptoms for one disease, so you can select from 1 up to 18 symptoms
![image](https://github.com/user-attachments/assets/eecd34db-2f2f-4a41-95e4-e6b0f4a37942)


# Symptom selection
Every symptom in the dataset is inside the dropdown list
![image](https://github.com/user-attachments/assets/a9a74e83-5a56-4339-89e7-b5b4435b1aff)

# Runnning the prediction
After selecting symptoms, press the submit button and the model will generate a prediction.
The lists of outputs are :
- Selected symptoms
- Disease name
- Disease description
- Accuracy, F1 Score
- Prediction confidence
- List of recommended things to do at home
![image](https://github.com/user-attachments/assets/732a9e05-e98e-4d87-af9d-cecdb941b1ec)

