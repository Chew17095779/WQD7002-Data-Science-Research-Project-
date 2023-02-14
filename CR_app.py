#!/usr/bin/env python
# coding: utf-8

# In[226]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import Lasso


# In[227]:


st.write ("""
# AACVPR Risk Level Prediction App

This app predicts the AACVPR Risk Level of cardiac patients.

Data obtained from UMMC

""")



# In[228]:


st.sidebar.header('User Input Features')


# In[229]:


def user_input_features():
    
    Gender =st.sidebar.selectbox('Gender',('Male','Female'))
    
    AgeGroup =st.sidebar.selectbox('Age Group',('Adult','Elderly'))
    
    Education_level=st.sidebar.selectbox('Education level',('Lower than pre-university','HE Qualification'))
    
    Race =st.sidebar.selectbox('Race',('Malay','Chinese','Indian','Others'))
    
    Patient_occupation =st.sidebar.selectbox('Patient occupation',('Government servant','Self-employed (excludes housewives)','Private employment'))
    
    Health_funding =st.sidebar.selectbox('Health funding',('Fully Funded','Semi-Funded','Self funded'))
    
    Exercise_intensity =st.sidebar.selectbox('Exercise intensity',('1','2','3','4','5','6','7','8'))
    
    Prescribed_Sessions =st.sidebar.selectbox('Prescribed sessions',('8 weeks','> 8 weeks'))
        
    Pre_Tobacco =st.sidebar.selectbox('Smoking status before CR',('Former smoker','Never smoked','Current smoker'))
    
    Post_Tobacco =st.sidebar.selectbox('Smoking status after CR',('Abstaining','Unknown','Not Abstaining'))
    
    Pre_METs_range =st.sidebar.selectbox('METs before exercise',('Light Intensity','Moderate Intensity','Vigorous Intensity'))
    
    Post_Peak_METs_range =st.sidebar.selectbox('Peak METs after exercise',('Light Intensity','Moderate Intensity','Vigorous Intensity'))
    
    Pre_Peak_Heart_Rate_range =st.sidebar.selectbox('Peak heart rate before exercise',('Very Light','Light','Moderate','Hard','Very Hard'))
    
    Post_Peak_Heart_Rate_range =st.sidebar.selectbox('Peak heart rate after exercise',('Very Light','Light','Moderate','Hard','Very Hard'))
    
    Pre_BMI_range =st.sidebar.selectbox('BMI before CR',('Underweight','Normal','Overweight','Obesity'))
    
    Post_BMI_range =st.sidebar.selectbox('BMI after CR',('Underweight','Normal','Overweight','Obesity'))
    
    Pre_BP_cat =st.sidebar.selectbox('Blood pressure before CR',('Optimal','Normal','At Risk',
                                                                 'Isolated Systolic Hypertension','Hypertension Stage 1',
                                                                 'Hypertension Stage 2','Hypertension Stage 3'))
    
    CR_BP_cat =st.sidebar.selectbox('Blood pressure during CR',('Optimal','Normal','At Risk',
                                                                'Isolated Systolic Hypertension','Hypertension Stage 1',
                                                                'Hypertension Stage 2','Hypertension Stage 3'))
    
    LDL_cat =st.sidebar.selectbox('LDL cholesterol',('Low risk','Intermediate risk','High risk'))
    
    HDL_cat =st.sidebar.selectbox('HDL cholesterol',('Low risk','Intermediate risk','High risk'))
    
    HbA1c_cat =st.sidebar.selectbox('HbA1c',('Normal','Prediabetes','Diabetes'))
    
    Triglyceride_cat=st.sidebar.selectbox('Triglyceride',('Normal','Borderline high','High','Very High'))
    
    Pre_Left_Ventricle_EF =st.sidebar.selectbox('Left Ventricle EF',('less then 40%','between 40 to 50%','more then 50% with no failure symptoms'))
    
    Depression_Scores =st.sidebar.selectbox('Depression Scores',('Low','Medium','High'))
    
    Risk_Factors_Hypertension =st.sidebar.selectbox('Hypertension',('1','0'))
                                                
    Total_Risk_Factors =st.sidebar.selectbox('Total_Risk_Factors',('0','1','2','3'))                                               

    Past_CV_PrevCABG =st.sidebar.selectbox('Past cardiac event-CABG',('1','0'))
    
    Past_CV_PrevMI =st.sidebar.selectbox('Past cardiac event-MI',('1','0'))
    
    Past_CV_Unknown =st.sidebar.selectbox('Past cardiac event-Unknown',('1','0'))
    
    Past_CV_Noneoftheabove =st.sidebar.selectbox('No past cardiac event',('1','0'))
    
    data= {'Gender': Gender,'Education_level': Education_level,'Race': Race,
           'Patient_occupation': Patient_occupation,'Health_funding': Health_funding,
           'Exercise_intensity': Exercise_intensity,'Prescribed_Sessions': Prescribed_Sessions,'Pre_Tobacco': Pre_Tobacco,
           'Post_Tobacco': Post_Tobacco,'AgeGroup': AgeGroup,'Pre_METs_range': Pre_METs_range,'Post_Peak_METs_range': Post_Peak_METs_range,
           'Pre_Peak_Heart_Rate_range': Pre_Peak_Heart_Rate_range,'Post_Peak_Heart_Rate_range': Post_Peak_Heart_Rate_range,
           'Pre_BMI_range': Pre_BMI_range,'Post_BMI_range': Post_BMI_range,'Pre_BP_cat': Pre_BP_cat,
           'CR_BP_cat': CR_BP_cat,'LDL_cat': LDL_cat,'HDL_cat': HDL_cat,'HbA1c_cat': HbA1c_cat,
           'Triglyceride_cat': Triglyceride_cat,'Pre_Left_Ventricle_EF': Pre_Left_Ventricle_EF,
           'Depression_Scores': Depression_Scores,'Risk_Factors_Hypertension': Risk_Factors_Hypertension,
           'Total_Risk_Factors': Total_Risk_Factors,'Past_CV_PrevCABG': Past_CV_PrevCABG,'Past_CV_PrevMI': Past_CV_PrevMI,
           'Past_CV_Unknown': Past_CV_Unknown,'Past_CV_Noneoftheabove': Past_CV_Noneoftheabove}
    
    features= pd.DataFrame(data, index=[0])
    
    
    return features

input_df =user_input_features()


# In[230]:


# Combines user input features with entire CR dataset

CR_raw=pd.read_csv('C:/Users/chewc/Desktop/CR/CR_recat.csv')


# In[231]:


CR= CR_raw.drop(columns=['AACVPR_Risk_Category'])

df= pd.concat([input_df, CR], axis=0, ignore_index=True)


# In[232]:


df[[ 'Past_CV_Noneoftheabove', 'Past_CV_PrevCABG']] = df[['Past_CV_Noneoftheabove','Past_CV_PrevCABG']].astype(str)

df[[ 'Past_CV_Unknown', 'Past_CV_PrevMI']] = df[['Past_CV_Unknown','Past_CV_PrevMI']].astype(str)

df[[ 'Risk_Factors_Hypertension','HDL_cat']] = df[['Risk_Factors_Hypertension','HDL_cat']].astype(str)

df[[ 'Exercise_intensity','Total_Risk_Factors']] = df[['Exercise_intensity','Total_Risk_Factors']].astype(str)


# In[233]:


# Encoding of ordinal features

encode=['Gender',"AgeGroup","Education_level","Race",
        "Pre_BP_cat",'Patient_occupation',
        'Health_funding',"Exercise_intensity",
        "Pre_Peak_Heart_Rate_range","CR_BP_cat",'Prescribed_Sessions',
        'Total_Risk_Factors','Pre_BMI_range',
        'Post_Peak_Heart_Rate_range','Post_Peak_METs_range','LDL_cat',
        'Past_CV_PrevCABG','HDL_cat',
        'Pre_Left_Ventricle_EF','Depression_Scores',
        'HbA1c_cat','Post_BMI_range','Pre_Tobacco',
        'Pre_METs_range','Post_Tobacco','Past_CV_Unknown',
        'Past_CV_PrevMI','Triglyceride_cat','Risk_Factors_Hypertension','Past_CV_Noneoftheabove']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1]


# In[234]:


# Displays the user input features
st.subheader('User Input Features')
st.write(df)


# In[235]:


# Read in saved model

load_clf=pickle.load(open('CR.pkl','rb'))                       


# In[236]:


# Apply model to make predictions
prediction =load_clf.predict(df)

st.subheader('Prediction')
# CR_AACVPR_Risk_Category =np.array(['0','1','2']).astype(int)

# target_mapper= {'Low':0,'Intermediate':1,'High':2}


# In[237]:


st.write(prediction)

st.subheader('Your ACCVPR risk level is:')
if prediction > 1.5:
    st.write('High Risk')

elif prediction > 0.5:
    st.write('Intermediate Risk')

else: 
    st.write('Low Risk')

st.subheader('Advice on Exercise Intensity in Cardiac Rehabilitation Program')
if prediction > 1.5:
    st.write('High intensity exercise: Not recommended')
    st.write('Intermediate intensity exercise: Exercise should be performed with Medical Clearance')
    st.write('Low intensity exercise: Recommended')

elif prediction > 0.5:
    st.write('High intensity exercise: Exercise should be performed with Medical Clearance')
    st.write('Intermediate intensity exercise: Recommended')
    st.write('Low intensity exercise: Recommended')

else: 
    st.write('High and Moderate intensity exercise are both recommended')
 


