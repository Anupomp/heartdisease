import streamlit as st
import pandas as pd

from sklearn.tree       import DecisionTreeClassifier
from sklearn.impute     import SimpleImputer

@st.cache_data
def load_and_train():
    df = pd.read_csv('heart_disease_uci (1).csv')
    df.drop(columns=['id','dataset'], inplace=True)

    # Keep only the columns we care about + target
    cols = ['age','sex','cp','trestbps','chol','restecg','thalch','num']
    df = df[cols].dropna()

    # Map sex
    df['sex'] = df['sex'].map({'Female':0,'Male':1})

    # Split
    X = df.drop('num', axis=1)
    y = df['num']

    # Impute numerics
    num_cols = ['age','sex','trestbps','chol','thalch']
    X[num_cols] = SimpleImputer(strategy='median').fit_transform(X[num_cols])

    # One-hot encode all categoricals at once
    X = pd.get_dummies(X, drop_first=True)

    # Save feature set
    feature_cols = X.columns.tolist()

    # Train
    model = DecisionTreeClassifier(random_state=0)
    model.fit(X, y)

    return model, num_cols, feature_cols

model, num_cols, feature_cols = load_and_train()

st.title("UCI Heart Disease Predictor")

with st.form("input_form"):
    age      = st.number_input("Age",      28, 77, 54)
    sex      = st.radio("Sex",      (0,1), format_func=lambda x: "Female" if x==0 else "Male")
    cp_str   = st.radio("Chest Pain Type",
                        ["typical angina","atypical angina","non-anginal pain","asymptomatic"])
    trestbps = st.number_input("Resting BP", 0, 200, 130)
    chol     = st.number_input("Cholesterol",0, 603, 246)
    restecg_str = st.radio("Rest ECG", ["normal","stt abnormality","lv hypertrophy"])
    thalch   = st.number_input("Max Heart Rate", 60, 202, 150)
    submitted= st.form_submit_button("Predict")

if submitted:
    # Map back to numeric codes
    cp_map = {
        "typical angina":       1,
        "atypical angina":      2,
        "non-anginal pain":     3,
        "asymptomatic":         4
    }
    restecg_map = {
        "normal":              0,
        "stt abnormality":     1,
        "lv hypertrophy":      2
    }

    # Build the row
    row = {
        'age':      age,
        'sex':      sex,
        'trestbps': trestbps,
        'chol':     chol,
        'thalch':   thalch,
        'cp':       cp_map[cp_str],
        'restecg':  restecg_map[restecg_str]
    }
    df_row = pd.DataFrame([row])

    # Impute numerics
    df_row[num_cols] = SimpleImputer(strategy='median').fit_transform(df_row[num_cols])

    # One-hot encode row
    df_row = pd.get_dummies(df_row, drop_first=True)

    # Align to training features
    df_row = df_row.reindex(columns=feature_cols, fill_value=0)

    # Predict
    pred = model.predict(df_row)[0]
    st.success(f"Predicted class: **{pred}** (0 = no disease, 1–4 increasing severity)")
