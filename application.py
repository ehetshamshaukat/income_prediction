import streamlit as st
from src.pipeline.prediction_pipeline import Features, Prediction

st.header("income price prediction")
gender_list = ['Male', 'Female']
education_list = ['Bachelors', 'college', 'Masters', 'school', 'Associate',
                  'Doctorate', 'bootcamp', 'Preschool']
marital_status_list = ['Never-married', 'Married-civ-spouse', 'Divorced',
                       'Married-spouse-absent', 'Separated', 'Married-AF-spouse',
                       'Widowed']
race_list = ['White', 'Black', 'asian_american', 'Other']
native_country_list = ['United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
                       'South', 'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany',
                       'Iran', 'Philippines', 'Italy', 'Poland', 'Columbia', 'Cambodia',
                       'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
                       'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
                       'China', 'Japan', 'Yugoslavia', 'Peru',
                       'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
                       'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
                       'Holand-Netherlands']
workclass_list = ['Private', 'gov-emp', 'self-emp', 'training', 'Never-worked']
occupation_list = ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
                   'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair',
                   'Transport-moving', 'Farming-fishing', 'Machine-op-inspct',
                   'Tech-support', 'Protective-serv', 'Armed-Forces',
                   'Priv-house-serv']

gender = st.selectbox("please select gender", gender_list)
education = st.selectbox("please select education", education_list)
marital_status = st.selectbox("please enter your marital status", marital_status_list)
race = st.selectbox("please enter your race", race_list)
native_country = st.selectbox("please enter your native_country", native_country_list)
workclass = st.selectbox("please enter your work class", workclass_list)

occupation = st.selectbox("please enter your occupation", occupation_list)
age = st.number_input("please enter your age", value=0, max_value=100)
hours_per_week = st.number_input("please enter hours per week", value=0, max_value=100)
ok = st.button("predict")

if __name__ == "__main__":
    f = Features(age=age, workclass=workclass, education=education, marital_status=marital_status,
                 occupation=occupation, race=race,
                 gender=gender, hours_per_week=hours_per_week, native_country=native_country)
    feature_as_df = f.to_dataframe()
    pred = Prediction()
    op = pred.initiate_prediction(feature_as_df)
    if op == 1:
        st.subheader("greater than 50k")
    else:
        st.subheader("less than 50k")


# <=50 == 0
# >50 == 1