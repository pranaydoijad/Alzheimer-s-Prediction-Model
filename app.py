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
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
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
    X = df.drop(['PatientID', 'Diagnosis', 'DoctorInCharge'], axis=1)
    y = df['Diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y
)
#########################################################################################################################   
    with tab1:
        st.title('Alzheimer’s Disease Classification Model Info')
        pipe2 = Pipeline([
            ('feature_selection', SelectKBest(score_func=f_classif)),
            ('scaler', StandardScaler()),
            ('model', GradientBoostingClassifier(
                # --- Core ---
                n_estimators=100,        # Number of boosting stages (trees)
                learning_rate=0.1,       # Shrinks contribution of each tree
                random_state=42,

                # --- Regularization ---
                max_depth=3,             # Depth of each tree (main regularizer)
                min_samples_split=2,     # Min samples to split an internal node
                min_samples_leaf=1,      # Min samples required at a leaf node
                max_features='sqrt',     # Features considered per split ('sqrt', 'log2', float)
                subsample=0.8,           # Fraction of samples per tree (< 1.0 = stochastic GB)
                max_leaf_nodes=None,     # Limit number of leaves per tree
                
                # --- Loss ---
                loss='log_loss',         # 'log_loss' or 'exponential' (AdaBoost-like)
                
                # --- Early Stopping ---
                n_iter_no_change=10,     # Stop if no improvement for N rounds
                validation_fraction=0.1, # Data fraction used for early stopping
                tol=1e-4,                # Tolerance for early stopping
                
                # --- Speed ---
                warm_start=False,        # Reuse previous fit (add more trees incrementally)
                ))
            ])
        selector = pipe2.named_steps['feature_selection']
        # Get boolean mask of selected features
        selected_mask = selector.get_support()
        # Get selected feature names
        selected_features = X_train.columns[selected_mask].tolist()
        print(f"Number of features selected: {len(selected_features)}")
        print(f"Selected features: {selected_features}")
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
        
        

        
        
        
        
        
        
        