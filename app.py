import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from xgboost import XGBClassifier

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lung Cancer Risk Predictor",
    page_icon="🫁",
    layout="centered",
)

st.title("🫁 Lung Cancer Risk Predictor")
st.caption("⚠️ For educational/research purposes only. Not a substitute for clinical diagnosis.")

# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("Patient Symptoms")

gender = st.sidebar.selectbox("GENDER", ["Male", "Female"])
age    = st.sidebar.number_input("AGE", min_value=1, max_value=120, value=50)

def yn(label):
    return st.sidebar.selectbox(label, ["Yes", "No"])

smoking             = yn("SMOKING")
yellow_fingers      = yn("YELLOW_FINGERS")
anxiety             = yn("ANXIETY")
peer_pressure       = yn("PEER_PRESSURE")
chronic_disease     = yn("CHRONIC DISEASE")
fatigue             = yn("FATIGUE")
allergy             = yn("ALLERGY")
wheezing            = yn("WHEEZING")
alcohol_consuming   = yn("ALCOHOL CONSUMING")
coughing            = yn("COUGHING")
shortness_of_breath = yn("SHORTNESS OF BREATH")
swallowing_diff     = yn("SWALLOWING DIFFICULTY")
chest_pain          = yn("CHEST PAIN")

# ── Helper ────────────────────────────────────────────────────────────────────
def encode_yn(val):
    return 2 if val == "Yes" else 1   # matches dataset encoding (1=No, 2=Yes)

# ── Model training ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on dataset…")
def load_model():
    try:
        df = pd.read_csv("lung_cancer_survey.csv")
    except FileNotFoundError:
        return None, None, None

    df = df.drop_duplicates()

    le_gender = LabelEncoder()
    le_target = LabelEncoder()
    df["GENDER"]      = le_gender.fit_transform(df["GENDER"])          # F=0, M=1
    df["LUNG_CANCER"] = le_target.fit_transform(df["LUNG_CANCER"])     # NO=0, YES=1

    # Interaction feature
    df["ANXYELFIN"] = df["ANXIETY"] * df["YELLOW_FINGERS"]

    X = df.drop("LUNG_CANCER", axis=1)
    y = df["LUNG_CANCER"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    adasyn = ADASYN(random_state=42)
    X_res, y_res = adasyn.fit_resample(X_scaled, y)

    best_params = {
        "n_estimators":    200,
        "max_depth":       4,
        "learning_rate":   0.1,
        "subsample":       0.8,
        "colsample_bytree":0.8,
        "use_label_encoder": False,
        "eval_metric":     "logloss",
        "random_state":    42,
    }
    model = XGBClassifier(**best_params)
    model.fit(X_res, y_res)

    return model, scaler, le_gender

model, scaler, le_gender = load_model()

# ── Predict button ────────────────────────────────────────────────────────────
st.divider()
if st.button("🔍 Assess Risk", use_container_width=True, type="primary"):
    if model is None:
        st.error(
            "**Dataset not found.** Place `lung_cancer_survey.csv` in the repo root.  \n"
            "Download from: https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer"
        )
    else:
        gender_enc = 1 if gender == "Male" else 0   # M=1, F=0 after LabelEncoder

        anx  = encode_yn(anxiety)
        yelf = encode_yn(yellow_fingers)

        input_data = pd.DataFrame([{
            "GENDER":               gender_enc,
            "AGE":                  age,
            "SMOKING":              encode_yn(smoking),
            "YELLOW_FINGERS":       yelf,
            "ANXIETY":              anx,
            "PEER_PRESSURE":        encode_yn(peer_pressure),
            "CHRONIC DISEASE":      encode_yn(chronic_disease),
            "FATIGUE ":             encode_yn(fatigue),        # note trailing space matches CSV
            "ALLERGY ":             encode_yn(allergy),        # note trailing space matches CSV
            "WHEEZING":             encode_yn(wheezing),
            "ALCOHOL CONSUMING":    encode_yn(alcohol_consuming),
            "COUGHING":             encode_yn(coughing),
            "SHORTNESS OF BREATH":  encode_yn(shortness_of_breath),
            "SWALLOWING DIFFICULTY":encode_yn(swallowing_diff),
            "CHEST PAIN":           encode_yn(chest_pain),
            "ANXYELFIN":            anx * yelf,
        }])

        input_scaled = scaler.transform(input_data)
        prob   = model.predict_proba(input_scaled)[0][1]   # probability of HIGH risk
        pred   = model.predict(input_scaled)[0]

        st.subheader("Result")
        if pred == 1:
            st.error(f"### High Risk ⚠️")
        else:
            st.success(f"### Low Risk ✅")

        st.write(f"**Prediction confidence:** {prob*100:.1f}%")
        st.progress(float(prob))
        st.caption(
            "This prediction is based on symptomatic survey data only. "
            "Consult a qualified medical professional for diagnosis."
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "Published at **INDIACom 2025** | "
    "Paper: [Read here](https://bvicam.in/INDIACom/news/INDIACom%202025%20Proceedings/Main/papers/984.pdf)",
    unsafe_allow_html=False,
)
