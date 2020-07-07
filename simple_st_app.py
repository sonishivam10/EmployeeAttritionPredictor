###################### Import Necessary Packages, Libraries ###########################
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import PIL
from PIL import Image


###################### Functions ###########################
@st.cache
def load_data():
    DATA_URL = ('emp_attrition.csv')
    data=pd.read_csv(DATA_URL)
    return data

def prediction():
    st.write( f'<div style="text-align: center; color:black;"><h5>Enter the parameters and click submit</h5></div>', unsafe_allow_html=True)
    imp_ind=['monthlyincome', 'overtime', 'age', 'monthlyrate', 'distancefromhome', 'dailyrate', 'totalworkingyears', 'yearsatcompany', 'hourlyrate', 'yearswithcurrmanager']
    monthlyincome= st.number_input('Monthly Income ($)?( Ex: 20000)', min_value=0.0)
    overtime=st.selectbox('OverTime? (Yes/No)', ('Yes','No'))
    age=st.slider('Employee Age?', 18, 100, 18)
    distancefromhome=int(st.number_input('Distance to office from home? (Kms)', min_value=0.0 ))
    totalworkingyears=st.slider('Total Working Experience? (In Years)', 1,100,10)
    yearsatcompany=st.slider('Working Years at this company? (In Years)', 1,100,8)
    yearswithcurrmanager=st.slider('Working Years with Current Manager?', 1,100,5)
    return None

def load_home(Home):
    if Home==0:
        st.image(logo, use_column_width=True)

def analysis():
    if st.checkbox("Data Description"):
        st.write(data.describe())
    if st.checkbox("Features Histograms"):
        data.hist()
        st.pyplot()
###################### App Page ###########################
logo=Image.open('REALAI.jpg')
st.sidebar.image(logo, use_column_width=True)

data=load_data()

home=0
if st.sidebar.checkbox("Show Employee Past Dataframe"):
    st.write(f'<h3 style="color:white; text-align:center; background-color:rgba(53, 52, 159, 1);">Employee DataFrame</h3>', unsafe_allow_html=True)
    #test_data
    data_load_state = st.text('Loading data...')
    # Load data into the dataframe.
    st.write(data)
    #st.write(data_set[['monthlyincome', 'overtime', 'age', 'monthlyrate', 'distancefromhome', 'dailyrate', 'totalworkingyears', 'yearsatcompany', 'hourlyrate', 'yearswithcurrmanager']])
    # Notify the reader that the data was successfully loaded.
    data_load_state.text('Done!')
    home=1

if st.sidebar.checkbox('Predict New Employee Attrition Rate'):
    st.write(f'<h3 style="color:white; text-align:center; background-color:rgba(53, 52, 159, 1);">Employee Attrition Prediction App</h3>', unsafe_allow_html=True)
    prediction()
    home=1
if st.sidebar.checkbox('Show Scatter plot'):
    st.write(f'<h3 style="color:white; text-align:center; background-color:rgba(53, 52, 159, 1);">Analysis of Employee Data</h3>', unsafe_allow_html=True)

    analysis()
    
    #e_attrition = st.multiselect('Show attrition Yes/No?', ['Yes', 'No'], ['Yes'])
    #col1 = st.selectbox('Which feature on x?', data.columns, )
    #col2 = st.selectbox('Which feature on y?', data.columns)
    #new_df = data[(data['Attrition'].isin(e_attrition))]
    #st.write(new_df)
    #fig = px.scatter(new_df, x =col1,y=col2, color='Attrition')
    #st.plotly_chart(fig)
    
    home=1

load_home(home)
