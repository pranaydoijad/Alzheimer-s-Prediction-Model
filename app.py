# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 16:30:25 2023

@author: excel
"""

import pickle
import streamlit as st
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
     confusion_matrix
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt

# open model in read binary mode
load = open('pipe.pkl','rb')
model = pickle.load(load)


# Define predict function 
def predict(SleepQuality,MMSE,FunctionalAssessment,MemoryComplaints,BehavioralProblems,ADL):
    prediction = model.predict([[SleepQuality,MMSE,FunctionalAssessment,MemoryComplaints,BehavioralProblems,ADL]]) 
    # pass arguments in 2 dimensions using 2 square brackets
    return prediction
#'SleepQuality', 'MMSE', 'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 'ADL'
def main():
    st.title('Alzheimer’s Disease Classification')
    tab1, tab2 = st.tabs(["📊 Model Info","🔍 Prediction"])
    df = pd.read_csv("alzheimers_disease_data.csv")
    X = df.drop("Diagnosis", axis=1)
    y = df["Diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42  # use same random_state as training!
        )
    X_train=X_train[['SleepQuality', 'MMSE', 'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 'ADL']]
    X_test=X_test[['SleepQuality', 'MMSE', 'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 'ADL']]

#########################################################################################################################   
    with tab1:
        st.title('Alzheimer’s Disease Classification Model Info')
        model.fit(X_train, y_train)
        ypred=model.predict(X_test)
        # Accuracies
        st.write("Train Accuracy:", model.score(X_train, y_train)*100)
        st.write("Test Accuracy:", model.score(X_test, y_test)*100)

        # Classification Report
        
        

        report = classification_report(y_test, ypred, output_dict=True)  # ← add output_dict=True
        st.dataframe(pd.DataFrame(report).transpose())

        # Confusion Matrix
        
        cm = confusion_matrix(y_test, ypred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        selector = model.named_steps['feature_selection']
        # Get boolean mask of selected features
        selected_mask = selector.get_support()
        # Get selected feature names
        selected_features = X_train.columns[selected_mask].tolist()
        print(f"Number of features selected: {len(selected_features)}")
        #print(f"Selected features: {selected_features}")
        st.write(f"Number of features selected: {len(selected_features)}")
        #st.write(f"Selected features: {selected_features}")
        
        selector = model.named_steps['feature_selection']
        # Get feature scores and selected mask
        feature_scores = pd.DataFrame({
            'feature': X_train.columns,
            'score': selector.scores_,
            'selected': selector.get_support()
            }).sort_values('score', ascending=False)
        print(feature_scores)
        st.dataframe(feature_scores)
        # Get only selected feature names
        selected_features = X_train.columns[selector.get_support()].tolist()
        print("\nSelected features:", selected_features)
        st.success(f"✅ Selected Features: {selected_features}")
#########################################################################################################################        
    with tab2:
        st.title('Alzheimer’s Disease Prediction')
        # Accept input values from user through browser
        SleepQuality = st.number_input('SleepQuality: ')
        MMSE = st.number_input('Mini-Mental State Examination score : ')    
        FunctionalAssessment = st.number_input('FunctionalAssessment: ')
        MemoryComplaints = st.number_input('MemoryComplaints: ')
        BehavioralProblems = st.number_input('BehavioralProblems: ')
        ADL = st.number_input('Activities of Daily Living score : ')




        if st.button('Predict'): # if predict button is clicked then execute below code
             result = predict(SleepQuality,MMSE,FunctionalAssessment,MemoryComplaints,BehavioralProblems,ADL)
             if result==0:
                 st.success("Non-Alzheimer’s Person")
             else:
                    st.success("Alzheimer’s Person")


# When we create app.py file in the backend python will give default global name to this python script file as "__name__"
# 
if __name__ == '__main__': # it will always be true so calls the main() function
    main()
        
        

        
        
        
        
        
        
        
