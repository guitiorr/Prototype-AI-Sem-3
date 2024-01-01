import joblib
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix,classification_report,plot_confusion_matrix,plot_roc_curve,precision_score,roc_curve
import seaborn as sns
from sklearn.utils import shuffle
# from pandas_profiling import ProfileReport
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import streamlit as st

def AImodel(s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17):

    #Read dataset and shuffling
    df = pd.read_csv('dataset.csv')
    df = shuffle(df,random_state=42)
    #Removing Hyphen from strings
    for col in df.columns:    
        df[col] = df[col].str.replace('_',' ')

    #Check of null values
    # null_checker = df.apply(lambda x: sum(x.isnull())).to_frame(name='count')
    # print(null_checker)
    # plt.figure(figsize=(10,5))
    # plt.plot(null_checker.index, null_checker['count'])
    # plt.xticks(null_checker.index, null_checker.index, rotation=45,
    # horizontalalignment='right')
    # plt.title('Before removing Null values')
    # plt.xlabel('column names')
    # plt.margins(0.1)
    # plt.show()

    #Remove the trailing space from the symptom columns
    cols = df.columns
    data = df[cols].values.flatten()
    s = pd.Series(data)
    s = s.str.strip()
    s = s.values.reshape(df.shape)
    df = pd.DataFrame(s, columns=df.columns)

    #Fill null values with 0
    df = df.fillna(0)

    #Symptom severity rank
    df1 = pd.read_csv('Symptom-severity.csv')
    df1['Symptom'] = df1['Symptom'].str.replace('_',' ')
    df1['Symptom'].unique()
    vals = df.values
    symptoms = df1['Symptom'].unique()
    for i in range(len(symptoms)):
        vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]    
    d = pd.DataFrame(vals, columns=cols)

    #Assing symptoms with no rank to 0
    d = d.replace('dischromic  patches', 0)
    d = d.replace('spotting  urination',0)
    df = d.replace('foul smell of urine',0)

    df['Disease'].unique()
    data = df.iloc[:,1:].values
    labels = df['Disease'].values

    #Splitting dataset
    x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size = 0.8,random_state=42)

    #Random forest model
    rfc=RandomForestClassifier(random_state=42)
    rnd_forest = RandomForestClassifier(random_state=42, max_features='sqrt', n_estimators= 500, max_depth=13)
    rnd_forest.fit(x_train,y_train)
    preds=rnd_forest.predict(x_test)
    conf_mat = confusion_matrix(y_test, preds)
    df_cm = pd.DataFrame(conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())

    #Manually test model
    discrp = pd.read_csv('symptom_Description.csv')
    ektra7at = pd.read_csv('symptom_precaution.csv')
    joblib.dump(rfc, "random_forest.joblib")
    loaded_rf = joblib.load("random_forest.joblib")
    
    
    def predd(x, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17):
        psymptoms = [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17]
        a = np.array(df1["Symptom"])
        b = np.array(df1["weight"])
        
        # Convert symptoms to their corresponding weights
        for j in range(len(psymptoms)):
            for k in range(len(a)):
                if psymptoms[j] == a[k]:
                    psymptoms[j] = b[k]
        
        # Make prediction and get probabilities
        psy = [psymptoms]
        pred_probs = x.predict_proba(psy)
        pred_class = x.predict(psy)
        
        # Get disease description and precautions
        disp = discrp[discrp['Disease'] == pred_class[0]]
        disp = disp.values[0][1]
        
        recomnd = ektra7at[ektra7at['Disease'] == pred_class[0]]
        c = np.where(ektra7at['Disease'] == pred_class[0])[0][0]
        precuation_list = []
        for i in range(1, len(ektra7at.iloc[c])):
            precuation_list.append(ektra7at.iloc[c, i])
        
        # Display prediction details including confidence
        st.write("The Disease Name: ", pred_class[0])
        st.write("The Disease Description: ", disp)
        st.write("Confidence: ", pred_probs.max())  # Maximum confidence among predicted classes
        st.write("Recommended Things to do at home: ")
        for i in precuation_list:
            st.write(i)


    indexlist = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17]
    indexlist.sort()

    s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17 = indexlist

    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    st.write(f"Accuracy: {accuracy}, F1 Score: {f1}")

    sympList=df1["Symptom"].to_list()

    symList = {}  # Assuming symList is a dictionary mapping values

    if s1 == 200:
        sym1 = 0
    elif s1 != 200:
        sym1 = sympList[s1]

    if s2 == 200:
        sym2 = 0
    elif s2 != 200:
        sym2 = sympList[s2]

    if s3 == 200:
        sym3 = 0
    elif s3 != 200:
        sym3 = sympList[s3]
    
    if s4 == 200:
        sym4 = 0
    elif s4 != 200:
        sym4 = sympList[s4]
    
    if s5 == 200:
        sym5 = 0
    elif s5 != 200:
        sym5 = sympList[s5]
    
    if s6 == 200:
        sym6 = 0
    elif s6 != 200:
        sym6 = sympList[s6]
    
    if s7 == 200:
        sym7 = 0
    elif s7 != 200:
        sym7 = sympList[s7]
    
    if s8 == 200:
        sym8 = 0
    elif s8 != 200:
        sym8 = sympList[s8]
    
    if s9 == 200:
        sym9 = 0
    elif s9 != 200:
        sym9 = sympList[s9]
    
    if s10 == 200:
        sym10 = 0
    elif s10 != 200:
        sym10 = sympList[s10]
    
    if s11 == 200:
        sym11 = 0
    elif s11 != 200:
        sym11 = sympList[s11]
    
    if s12 == 200:
        sym12 = 0
    elif s12 != 200:
        sym12 = sympList[s12]
    
    if s13 == 200:
        sym13 = 0
    elif s13 != 200:
        sym13 = sympList[s13]
    
    if s14 == 200:
        sym14 = 0
    elif s14 != 200:
        sym14 = sympList[s14]
    
    if s15 == 200:
        sym15 = 0
    elif s15 != 200:
        sym15 = sympList[s15]
    
    if s16 == 200:
        sym16 = 0
    elif s16 != 200:
        sym16 = sympList[s16]
    
    if s17 == 200:
        sym17 = 0
    elif s17 != 200:
        sym17 = sympList[s17]


    predd(rnd_forest, sym1, sym2, sym3, sym4, sym5, sym6, sym7, sym8, sym9, sym10, sym11, sym12, sym13, sym14, sym15, sym16, sym17)