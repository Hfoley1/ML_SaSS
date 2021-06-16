import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

st.set_page_config(page_title='The SS&C Machine Learning Advice App',
                   layout='wide')

st.write("""
# The SS&C Machine Learning App

In this app you can import a dataset to train the model and then input data via the UI to predict suitability for an ISA 
with probability scoring. 

This uses a linear regression with a confusion matrix for prediction and probability.


""")
# load the data
#with st.sidebar.header('1. Upload your CSV data'):
 #   uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
  #  st.sidebar.markdown("""
#[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
#""")
#if uploaded_file is not None:
 #   df = pd.read_csv(uploaded_file)
  #  st.markdown('**1.1. Glimpse of dataset**')
   # st.write(df)


train_df = pd.read_csv("data/MOCK_DATA_ISA.csv")
#print(train_df.head())
st.write(train_df)
# convert the data

def manipulate_df(df):
    df['gender'] = df['gender'].map(lambda x: 0 if x == 'M' else 1)
    #df['has_ISA'] = df['has_ISA'].map(lambda x: 0 if x == 'True' else 1)
    #df['recommend_isa'] = df['recommend_isa'].map(lambda x: 0 if x == 'True' else 1)
    df = df[['age', 'gender', 'has_ISA', 'retirement_age', 'ATR', 'region', 'recommend_isa']]
    return df

train_df = manipulate_df(train_df)
features = train_df[['age', 'gender', 'has_ISA', 'retirement_age', 'ATR', 'region']]
recommendation = train_df['recommend_isa']
X_train, X_test, y_train, y_test = train_test_split(features, recommendation, test_size=0.2)

scaler = StandardScaler()
train_features = scaler.fit_transform(X_train)
test_features = scaler.transform(X_test)

model = LogisticRegression()
model.fit(train_features , y_train)
train_score = model.score(train_features,y_train)
test_score = model.score(test_features,y_test)
y_predict = model.predict(test_features)

st.title("Should investor be recommended and ISA?")
st.subheader("This model will predict if an investor should have an ISA based on input")
st.table(train_df.head(5))

confusion = confusion_matrix(y_test, y_predict)
FN = confusion[1][0]
TN = confusion[0][0]
TP = confusion[1][1]
FP = confusion[0][1]

st.subheader("Train Set Score: {}".format ( round(train_score,3)))
st.subheader("Test Set Score: {}".format(round(test_score,3)))

plt.bar(['False Negative' , 'True Negative' , 'True Positive' , 'False Positive'],[FN,TN,TP,FP])

st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()


name = st.text_input("Name of Investor ")
gender = st.selectbox("gender",options=['Male', 'Female'])
age = st.slider("Age", 18, 100)
ATR = st.slider("ATR", 1, 5)
has_ISA = st.selectbox("Has ISA?", options = ['Yes', 'No'])
retirement_age = st.slider("Retirement Age", 55, 78)
region = st.selectbox("Region", options=['England', 'Ireland', 'Scotland', 'Wales'])

gender = 0 if gender == 'Male' else 1
has_ISA = 0 if has_ISA == 'No' else 1
region = 0
if region == 'England':
    region = 1
elif region == 'Ireland':
    region = 2
elif region == 'Scotland':
    region = 3
else:
    region = 4

input_data = scaler.transform([[gender , age, ATR, has_ISA, region, retirement_age]])
prediction = model.predict(input_data)
predict_probability = model.predict_proba(input_data)

if prediction[0] == 1:
	st.subheader('Investor {} should be recommended an ISA with a probability of {}%'.format(name , round(predict_probability[0][1]*100 , 3)))
else:
	st.subheader('Investor {} should not be recommended an ISA with a probability of {}%'.format(name, round(predict_probability[0][0]*100 , 3)))