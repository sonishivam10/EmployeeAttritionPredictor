###################### Import Necessary Packages, Libraries ###########################
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import PIL
from PIL import Image
import plotly.graph_objs as go
import base64
import pickle
from random import randint as rnum
from plotly.subplots import make_subplots
import plotly.graph_objects as go

###################### Functions ###########################

def load_data():
    old_data=pd.read_csv('HR_comma_sep.csv')
    #old_data[old_data['left']==0].drop('left', axis=1).to_csv('Emp_data.csv', index=False)
    Emp_data=pd.read_csv('Emp_data.csv')
    return old_data, Emp_data
def high_risk_analysis(emp_dataset):
    objects = []
    data=emp_dataset.copy()
    
    with open('final_model.pkl', 'rb') as f:
        objects.append(pickle.load(f))
    # Create indicator features
    data=data.drop(['EmpName'], axis=1)
    
    data['underperformer'] = (data.last_evaluation < 0.6).astype(int)
    data['unhappy'] = (data.satisfaction_level < 0.2).astype(int)
    data['overachiever'] = ((data.last_evaluation > 0.8) & (data.satisfaction_level > 0.7)).astype(int)
    cate_columns = data.select_dtypes(include=['object']).columns
    btable = pd.get_dummies(data, columns=cate_columns)
    pred = objects[0].predict_proba(btable)
    pred = [p[1] for p in pred]
    
    data=emp_dataset.copy()
    data["TurnOverRate"]=pred
    st.subheader("Employees at High Risk of Leaving")
    st.table(data[['EmpName', 'TurnOverRate']].set_index('EmpName').sort_values(by=['TurnOverRate'], ascending=False).head(15))

def data_update(EmpName, ESL, NOP, TWC, Promotion, Department, Salary, WA, AMH, LE, Update, Emp_data):
    objects = []
    with open('final_model.pkl', 'rb') as f:
        objects.append(pickle.load(f))
    if float(LE) < 0.6:
        underperformer = 1
    else:
        underperformer = 0

    if float(ESL)/10 < 0.4:
        unhappy = 1
    else:
        unhappy = 0

    if float(LE) > 0.8 and float(ESL)/10 > 0.7:
        overachiever = 1
    else:
        overachiever = 0

    if Salary == 'high':
        salary_high = 1
        salary_medium = 0
        salary_low = 0
    elif Salary == 'medium':
        salary_high = 0
        salary_medium = 1
        salary_low = 0
    elif Salary == 'low':
        salary_high = 0
        salary_medium = 0
        salary_low = 1

    if Department == 'Sales':
        Department_IT = 0
        Department_RandD = 0
        Department_accounting = 0
        Department_hr = 0
        Department_management = 0
        Department_marketing = 0
        Department_product_mng = 0
        Department_sales = 1
        Department_support = 0
        Department_technical = 0
    elif Department == 'Accounting':
        Department_IT = 0
        Department_RandD = 0
        Department_accounting = 1
        Department_hr = 0
        Department_management = 0
        Department_marketing = 0
        Department_product_mng = 0
        Department_sales = 0
        Department_support = 0
        Department_technical = 0
    elif Department == 'IT':
        Department_IT = 1
        Department_RandD = 0
        Department_accounting = 0
        Department_hr = 0
        Department_management = 0
        Department_marketing = 0
        Department_product_mng = 0
        Department_sales = 0
        Department_support = 0
        Department_technical = 0
    elif Department == 'R&D':
        Department_IT = 0
        Department_RandD = 1
        Department_accounting = 0
        Department_hr = 0
        Department_management = 0
        Department_marketing = 0
        Department_product_mng = 0
        Department_sales = 0
        Department_support = 0
        Department_technical = 0
    elif Department == 'HR':
        Department_IT = 0
        Department_RandD = 0
        Department_accounting = 0
        Department_hr = 1
        Department_management = 0
        Department_marketing = 0
        Department_product_mng = 0
        Department_sales = 0
        Department_support = 0
        Department_technical = 0
    elif Department == 'Management':
        Department_IT = 0
        Department_RandD = 0
        Department_accounting = 0
        Department_hr = 0
        Department_management = 1
        Department_marketing = 0
        Department_product_mng = 0
        Department_sales = 0
        Department_support = 0
        Department_technical = 0
    elif Department == 'Marketing':
        Department_IT = 0
        Department_RandD = 0
        Department_accounting = 0
        Department_hr = 0
        Department_management = 0
        Department_marketing = 1
        Department_product_mng = 0
        Department_sales = 0
        Department_support = 0
        Department_technical = 0
    elif Department == 'Product_Management':
        Department_IT = 0
        Department_RandD = 0
        Department_accounting = 0
        Department_hr = 0
        Department_management = 0
        Department_marketing = 0
        Department_product_mng = 1
        Department_sales = 0
        Department_support = 0
        Department_technical = 0
    elif Department == 'Support':
        Department_IT = 0
        Department_RandD = 0
        Department_accounting = 0
        Department_hr = 0
        Department_management = 0
        Department_marketing = 0
        Department_product_mng = 0
        Department_sales = 0
        Department_support = 1
        Department_technical = 0
    elif Department == 'Technical':
        Department_IT = 0
        Department_RandD = 0
        Department_accounting = 0
        Department_hr = 0
        Department_management = 0
        Department_marketing = 0
        Department_product_mng = 0
        Department_sales = 0
        Department_support = 0
        Department_technical = 1
    if WA=='Yes':
        WA=1
    else:
        WA=0

    if Promotion=='Yes':
        Promotion=1
    else:
        Promotion=0
    pred_data = pd.DataFrame(
            {'satisfaction_level':[ESL/10], 'last_evaluation':[LE],
                'number_project':[NOP], 'average_monthly_hours':[AMH],
                'time_spend_company':[TWC], 'Work_accident':[WA],
                'promotion_last_5years':[Promotion], 'underperformer':[underperformer], 'unhappy':[unhappy],
                'overachiever':[overachiever], 'Department_IT':[Department_IT], 'Department_RandD':[Department_RandD],
                'Department_accounting':[Department_accounting], 'Department_hr':[Department_hr], 'Department_management':[Department_management], 'Department_marketing':[Department_marketing],
    'Department_product_mng':[Department_product_mng], 'Department_sales':[Department_sales], 'Department_support':[Department_support],
    'Department_technical':[Department_technical], 'salary_high':[salary_high], 'salary_low':[salary_low], 'salary_medium':[salary_medium]}
    )
    
    pred = objects[0].predict_proba(pred_data)
    attrition_risk = [p[1] for p in pred][0] * 100
    attrition_safe = [p[0] for p in pred][0] * 100

    if Department!='IT' or Department!='RandD':
        Department=Department.lower()
    if Update==1 :
        if EmpName not in Emp_data['EmpName'].values:
            Emp_data.loc[len(Emp_data)]=[EmpName, ESL/10, LE , NOP,AMH,  TWC, WA, Promotion, Department, Salary]
        else:
            Emp_data.drop(Emp_data[Emp_data['EmpName'] == EmpName].index, inplace=True)
            Emp_data.loc[len(Emp_data)]=[EmpName, ESL/10, LE , NOP,AMH,  TWC, WA, Promotion, Department, Salary]

        Emp_data.to_csv('Emp_data.csv', index=False)
        
    figure = go.Figure(
            data = [
                go.Bar(
                    y = ['Attrition Risk'],
                    x = [attrition_risk],
                    marker = dict(color='red'),
                    name = 'Risk',
                    orientation = 'h'
                ),

                go.Bar(
                    y = ['Attrition Safe'],
                    x = [attrition_safe],
                    marker = dict(color='green'),
                    name = 'Safe',
                    orientation = 'h'
                )
            ],

            layout = dict(
                    title = "Attrition Probability {}%".format(int(attrition_risk)),
                    yaxis = dict(
                                showgrid=False,
                                showline=False,
                                showticklabels=True,
                                domain = [0,1]
                    ),
                    hovermode = 'closest',
                    width = 540, height = 275
            )
    )
    return figure

def prediction():
    if st.checkbox("New Employee TurnOver Prediction"):
        st.write( f'<div style="text-align: center; color:black;"><h5>Enter the parameters and click submit</h5></div>', unsafe_allow_html=True)
        if st.checkbox('Check to add/update this employee data to dataset.'):
            Update=1
        else:
            Update=0
        EmpName=st.text_input('Employee Full Name')
        ESL = st.slider('Employee Satisfaction Level?', 1, 10, 3)
        NOP=st.number_input('Number of Projects?', min_value=1.0)
        TWC=st.number_input('Time With Company?(In Years)', min_value=0.0)
        Promotion=st.selectbox( 'Promotion in Last 5 years?', ('Yes', 'No'))
        Department=st.selectbox( 'Department?', ('Sales', 'Accounting', 'IT', 'RandD', 'HR', 'Management', 'Marketing', 'Product_Management', 'Support', 'Technical'))
        Salary=st.selectbox( 'Salary?', ('low', 'medium','high'))
        WA=st.selectbox( 'Work Accident?', ('Yes', 'No'))
        AMH=st.number_input('Average Monthly Hours?', min_value=1.0)
        LE=st.number_input('Last Evaluation Score? ( 0.75- Enter decimal %) ', min_value=0.01, max_value=1.0)

       
        if st.button("Attrition Rate"):
            result=data_update(EmpName, ESL, NOP, TWC, Promotion, Department, Salary, WA, AMH, LE, Update, Emp_data)
            st.plotly_chart(result)
    if st.checkbox("List of Employee on High Risk of TurnOver"):
        high_risk_analysis(Emp_data)
    return None

def load_home(Home):
    if Home==0:
        dash=Image.open('hr_dash.jpg')
        st.image(logo, use_column_width=True)
        st.title("HR-Anaytics Project")
        st.header("HR, Say Goodbye to Spreadsheets!")
        st.markdown("PeopleInsight is a workforce analytics platform delivered as a managed service. We connect all your people data and deliver beautiful dashboards and analytics so you can skip the spreadsheets."
        " Spend your time where it matters most... on HR")
        st.image(dash, use_column_width=True)
def turnover_analysis():    
    # Types of colors
    color_types = ['#78C850','#F08030','#6890F0','#A8B820','#A8A878','#A040A0','#F8D030',  
                    '#E0C068','#EE99AC','#C03028','#F85888','#B8A038','#705898','#98D8D8','#7038F8']

    fig = px.histogram(dataset, x='Department', color='left')
    st.plotly_chart(fig)
    st.markdown("The  **sales, technical**, and **support** department were the top 3 **departments** to have employee turnover. "
    "The management department had the smallest amount of turnover")
    fig = px.box(dataset, x="number_project", y="average_montly_hours", color="left")
    st.plotly_chart(fig)
    st.markdown("The average employees who **stayed** worked about **200hours/month**. Those that had a **turnover** worked about **250hours/month** and **150hours/month**")


###################### App Page ###########################
logo=Image.open('REALAI.jpg')
st.sidebar.image(logo, use_column_width=True)
st.sidebar.title("Employee TurnOver App")

st.sidebar.markdown("This application is used to analyze and predict Employee TurnOver from the Company:")  
dataset, Emp_data=load_data()


home=0
if st.sidebar.checkbox("Past Employee TurnOver- Dataframe"):
    st.write(f'<h3 style="color:white; text-align:center; background-color:rgba(53, 52, 159, 1);">Employee DataFrame</h3>', unsafe_allow_html=True)
    #test_data
    data_load_state = st.text('Loading data...')
    # Load data into the dataframe.
    emp_left=dataset[dataset['left']==1].drop('left', axis=1).set_index('EmpName')
    st.write(emp_left)
    # Notify the reader that the data was successfully loaded.
    data_load_state.text('Employee who left in past had these stats!')
    home=1

if st.sidebar.checkbox('TurnOver Prediction'):
    st.write(f'<h3 style="color:white; text-align:center; background-color:rgba(53, 52, 159, 1);">Employee Attrition Prediction App</h3>', unsafe_allow_html=True)
    prediction()
    home=1

if st.sidebar.checkbox('TurnOver Analysis'):   
    turnover_analysis()
    home=1

load_home(home)
