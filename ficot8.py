import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib

st.markdown('# FICO Default Risk Prediction')

# Load model
model_names = ['RF', 'SVMrbf','KNN']

def load_model(model_names):
    models = []
    for model_name in ['RF', 'SVMrbf','KNN']:
        filename = model_name + '.pkl'
        model = joblib.load(filename)
        models.append(model)
    return models

models = load_model(model_names)
RF,  SVMrbf, KNN = models

# Select model
selected_model = st.selectbox('Model: ',
                              ['Random Forest',
                               'Support Vectors Machine(rbf)','K-nearest neighbors'])


if selected_model == 'Random Forest':
    model = RF
elif selected_model == 'Support Vectors Machine(rbf)':
    model = SVMrbf
elif selected_model == 'K-nearest neighbors':
    model = KNN


# Fill values
st.sidebar.markdown("**Variables:**")
X = {}

X['ExternalRiskEstimate'] = st.sidebar.slider('ExternalRiskEstimate', 0, 94, 67)
X['MSinceOldestTradeOpen'] = st.sidebar.slider('MSinceOldestTradeOpen', -8, 803, 184)
X['MSinceMostRecentTradeOpen'] = st.sidebar.slider('MSinceMostRecentTradeOpen', 0, 227, 8)
X['AverageMInFile'] = st.sidebar.slider('AverageMInFile', 4, 322, 73)
X['NumSatisfactoryTrades'] = st.sidebar.slider('NumSatisfactoryTrades', 0, 79, 19)
X['NumTrades60Ever2DerogPubRec'] = st.sidebar.slider('NumTrades60Ever2DerogPubRec', 0, 19, 0)
X['NumTrades90Ever2DerogPubRec'] = st.sidebar.slider('NumTrades90Ever2DerogPubRec', 0, 19, 0)
X['PercentTradesNeverDelq'] = st.sidebar.slider('PercentTradesNeverDelq', 0, 100, 86)
X['MSinceMostRecentDelq'] = st.sidebar.slider('MSinceMostRecentDelq', -8, 83, 6)
X['MaxDelq2PublicRecLast12M'] = st.sidebar.slider('MaxDelq2PublicRecLast12M', 0, 9, 5)
X['MaxDelqEver'] = st.sidebar.slider('MaxDelqEver', 2, 8, 5)
X['NumTotalTrades'] = st.sidebar.slider('NumTotalTrades', 0, 104, 20)
X['NumTradesOpeninLast12M'] = st.sidebar.slider('NumTradesOpeninLast12M', 0, 19, 1)
X['PercentInstallTrades'] = st.sidebar.slider('PercentInstallTrades', 0, 100, 32)
X['MSinceMostRecentInqexcl7days'] = st.sidebar.slider('MSinceMostRecentInqexcl7days', -8, 24, 0)
X['NumInqLast6M'] = st.sidebar.slider('NumInqLast6M', 0, 66, 0)
X['NumInqLast6Mexcl7days'] = st.sidebar.slider('NumInqLast6Mexcl7days', 0, 66, 0)
X['NetFractionRevolvingBurden'] = st.sidebar.slider('NetFractionRevolvingBurden', -8, 232, 32)
X['NetFractionInstallBurden'] = st.sidebar.slider('NetFractionInstallBurden', -8, 471, 39)
X['NumRevolvingTradesWBalance'] = st.sidebar.slider('NumRevolvingTradesWBalance', -8, 32, 3)
X['NumInstallTradesWBalance'] = st.sidebar.slider('NumInstallTradesWBalance', -8, 23, 1)
X['NumBank2NatlTradesWHighUtilization'] = st.sidebar.slider('NumBank2NatlTradesWHighUtilization', -8, 18, 0)
X['PercentTradesWBalance'] = st.sidebar.slider('PercentTradesWBalance', -8, 100, 62)

# Convert X to DataFrame
X = pd.DataFrame(X, index=[0])


# Show prediction
def show_prediction():
    if st.button("Run Model"):
        pred = model.predict(X)
        if pred == 1:
            st.markdown('## Risk Performance: Good')
        else:
            st.markdown('## Risk Performance: Bad')

show_prediction()

# Show model's evaluation
def show_evaluation():
    if selected_model == 'Random Forest':
        st.write('**Accuracy:** 71.29%')
    elif selected_model == 'Support Vectors Machine(rbf)':
        st.write('**Accuracy:** 73.52%')
    elif selected_model == 'K-nearest neighbors':
        st.write('**Accuracy:** 72.14%')


show_evaluation()
