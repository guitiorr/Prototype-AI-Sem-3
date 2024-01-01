def AImodel(s1,s2,s3):
    import joblib
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
    from sklearn.svm import SVC
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix,classification_report,plot_confusion_matrix,plot_roc_curve,precision_score,roc_curve
    import seaborn as sns
    from sklearn.utils import shuffle
    # from pandas_profiling import ProfileReport
    from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
    from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier 
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import metrics

    # %%
    df = pd.read_csv('dataset.csv')
    df = shuffle(df,random_state=42)
    df.head()

    # %%
    for col in df.columns:
        
        df[col] = df[col].str.replace('_',' ')
    df.head()

    # %%
    df.describe()

    # %%
    null_checker = df.apply(lambda x: sum(x.isnull())).to_frame(name='count')
    print(null_checker)

    # %%
    plt.figure(figsize=(10,5))
    plt.plot(null_checker.index, null_checker['count'])
    plt.xticks(null_checker.index, null_checker.index, rotation=45,
    horizontalalignment='right')
    plt.title('Before removing Null values')
    plt.xlabel('column names')
    plt.margins(0.1)
    plt.show()

    # %%
    cols = df.columns
    data = df[cols].values.flatten()

    s = pd.Series(data)
    s = s.str.strip()
    s = s.values.reshape(df.shape)

    df = pd.DataFrame(s, columns=df.columns)
    df.head()

    # %%


    df = df.fillna(0)
    df.head()



    # %%
    df1 = pd.read_csv('Symptom-severity.csv')
    df1['Symptom'] = df1['Symptom'].str.replace('_',' ')
    df1.head()

    # %%
    df1['Symptom'].unique()

    # %%
    vals = df.values
    symptoms = df1['Symptom'].unique()

    for i in range(len(symptoms)):
        vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]
        
    d = pd.DataFrame(vals, columns=cols)
    d.head()

    # %%
    d = d.replace('dischromic  patches', 0)
    d = d.replace('spotting  urination',0)
    df = d.replace('foul smell of urine',0)
    df.head(10)

    # %%
    null_checker = df.apply(lambda x: sum(x.isnull())).to_frame(name='count')
    print(null_checker)

    # %%
    plt.figure(figsize=(10,5))
    plt.plot(null_checker.index, null_checker['count'])
    plt.xticks(null_checker.index, null_checker.index, rotation=45,
    horizontalalignment='right')
    plt.title('After removing Null values')
    plt.xlabel('column names')
    plt.margins(0.01)
    plt.show()

    # %%


    print("Number of symptoms used to identify the disease ",len(df1['Symptom'].unique()))
    print("Number of diseases that can be identified ",len(df['Disease'].unique()))



    # %%
    df['Disease'].unique()

    # %%
    data = df.iloc[:,1:].values
    labels = df['Disease'].values

    # %%
    x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size = 0.8,random_state=42)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # %%


    tree =DecisionTreeClassifier(criterion='gini',random_state=42,max_depth=13)
    tree.fit(x_train, y_train)
    preds=tree.predict(x_test)
    conf_mat = confusion_matrix(y_test, preds)
    df_cm = pd.DataFrame(conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())
    print('F1-score% =', f1_score(y_test, preds, average='macro')*100, '|', 'Accuracy% =', accuracy_score(y_test, preds)*100)
    sns.heatmap(df_cm)



    # %%
    kfold = KFold(n_splits=10,shuffle=True,random_state=42)
    DS_train =cross_val_score(tree, x_train, y_train, cv=kfold, scoring='accuracy')
    pd.DataFrame(DS_train,columns=['Scores'])
    print("Mean Accuracy: %.3f%%, Standard Deviation: (%.2f%%)" % (DS_train.mean()*100.0, DS_train.std()*100.0))

    # %%
    kfold = KFold(n_splits=10,shuffle=True,random_state=42)
    DS_test =cross_val_score(tree, x_test, y_test, cv=kfold, scoring='accuracy')
    pd.DataFrame(DS_test,columns=['Scores'])
    print("Mean Accuracy: %.3f%%, Standard Deviation: (%.2f%%)" % (DS_test.mean()*100.0, DS_test.std()*100.0))

    # %%
    rfc=RandomForestClassifier(random_state=42)

    # %%


    rnd_forest = RandomForestClassifier(random_state=42, max_features='sqrt', n_estimators= 500, max_depth=13)
    rnd_forest.fit(x_train,y_train)
    preds=rnd_forest.predict(x_test)
    print(x_test[0])
    print(preds[0])
    conf_mat = confusion_matrix(y_test, preds)
    df_cm = pd.DataFrame(conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())
    print('F1-score% =', f1_score(y_test, preds, average='macro')*100, '|', 'Accuracy% =', accuracy_score(y_test, preds)*100)
    sns.heatmap(df_cm)



    # %%
    kfold = KFold(n_splits=10,shuffle=True,random_state=42)
    rnd_forest_train =cross_val_score(rnd_forest, x_train, y_train, cv=kfold, scoring='accuracy')
    pd.DataFrame(rnd_forest_train,columns=['Scores'])
    print("Mean Accuracy: %.3f%%, Standard Deviation: (%.2f%%)" % (rnd_forest_train.mean()*100.0, rnd_forest_train.std()*100.0))

    # %%
    kfold = KFold(n_splits=10,shuffle=True,random_state=42)
    rnd_forest_test =cross_val_score(rnd_forest, x_test, y_test, cv=kfold, scoring='accuracy')
    pd.DataFrame(rnd_forest_test,columns=['Scores'])
    print("Mean Accuracy: %.3f%%, Standard Deviation: (%.2f%%)" % (rnd_forest_test.mean()*100.0, rnd_forest_test.std()*100.0))

    # %%
    discrp = pd.read_csv("symptom_Description.csv")

    # %%


    discrp.head()



    # %%
    ektra7at = pd.read_csv("symptom_precaution.csv")

    # %%
    ektra7at.head()

    # %%
    # save
    joblib.dump(rfc, "random_forest.joblib")

    # %%
    # load, no need to initialize the loaded_rf
    loaded_rf = joblib.load("random_forest.joblib")

    # %%
    def predd(x,S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15,S16,S17):
        psymptoms = [S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15,S16,S17]
        #print(psymptoms)
        a = np.array(df1["Symptom"])
        b = np.array(df1["weight"])
        for j in range(len(psymptoms)):
            for k in range(len(a)):
                if psymptoms[j]==a[k]:
                    psymptoms[j]=b[k]
        psy = [psymptoms]
        pred2 = x.predict(psy)
        disp= discrp[discrp['Disease']==pred2[0]]
        disp = disp.values[0][1]
        recomnd = ektra7at[ektra7at['Disease']==pred2[0]]
        c=np.where(ektra7at['Disease']==pred2[0])[0][0]
        precuation_list=[]
        for i in range(1,len(ektra7at.iloc[c])):
            precuation_list.append(ektra7at.iloc[c,i])
        print("The Disease Name: ",pred2[0])
        print("The Disease Discription: ",disp)
        print("Recommended Things to do at home: ")
        for i in precuation_list:
            print(i)

    # %%
    n_groups = 2
    algorithms = ('Decision Tree', 'Random Forest')
    train_accuracy = (DS_train.mean()*100.0,
                    rnd_forest_train.mean()*100.0,)

    # %%
    test_accuracy = (DS_test.mean()*100.0,
                    rnd_forest_test.mean()*100.0)

    # %%
    Standard_Deviation=(DS_test.std()*100.0,     
                    rnd_forest_test.std()*100.0)

    # %%
    # create plot
    fig, ax = plt.subplots(figsize=(15, 10))
    index = np.arange(n_groups)
    bar_width = 0.3
    opacity = 1
    rects1 = plt.bar(index, train_accuracy, bar_width, alpha = opacity, color='Cornflowerblue', label='Train')
    rects2 = plt.bar(index + bar_width, test_accuracy, bar_width, alpha = opacity, color='Teal', label='Test')
    rects3 = plt.bar(index + bar_width, Standard_Deviation, bar_width, alpha = opacity, color='red', label='Standard Deviation')
    plt.xlabel('Algorithm') # x axis label
    plt.ylabel('Accuracy (%)') # y axis label
    plt.ylim(0, 115)
    plt.title('Comparison of Algorithm Accuracies') # plot title
    plt.xticks(index + bar_width * 0.5, algorithms) # x axis data labels
    plt.legend(loc = 'upper right') # show legend
    for index, data in enumerate(train_accuracy):
        plt.text(x = index - 0.035, y = data + 1, s = round(data, 2), fontdict = dict(fontsize = 8))
    for index, data in enumerate(test_accuracy):
        plt.text(x = index + 0.25, y = data + 1, s = round(data, 2), fontdict = dict(fontsize = 8))
    for index, data in enumerate(Standard_Deviation):
        plt.text(x = index + 0.25, y = data + 1, s = round(data, 2), fontdict = dict(fontsize = 8))

    # %%


    # %%
    sympList = df1["Symptom"].to_list()
    symptom1 = s1
    symptom2 = s2
    symptom3 = s3

    # print("Symptoms inputted:")
    # print(f"Symptom 7: {symptom7}")
    # print(f"Symptom 5: {symptom5}")
    # print(f"Symptom 2: {symptom2}")
    # print(f"Symptom 80: {symptom80}")

    # Now call the predd function with these symptoms
    predd(rnd_forest, s1, s2, s3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    # %% %% %% %% %% %% %%`
