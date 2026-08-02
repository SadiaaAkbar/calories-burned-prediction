import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Calories Burned Predictor", page_icon="🔥", layout="centered")

@st.cache_resource
def load_model():
    model = joblib.load("models/best_model.joblib")
    num_cols = joblib.load("models/feature_cols_num.joblib")
    cat_cols = joblib.load("models/feature_cols_cat.joblib")
    with open("models/metadata.json") as f:
        meta = json.load(f)
    return model, num_cols, cat_cols, meta

@st.cache_data
def load_reference_data():
    return pd.read_csv("data/gym_data.csv")

model, num_cols, cat_cols, meta = load_model()
ref_df = load_reference_data()

st.title("🔥 Calories Burned Predictor")
st.caption(
    f"Predicts calories burned in a gym session from your stats and workout details. "
    f"Powered by a **{meta['best_model']}** model (test R² = {meta['metrics']['test_r2']:.2f}, "
    f"avg. error ≈ {meta['metrics']['test_mae']:.0f} kcal)."
)

with st.sidebar:
    st.header("About")
    st.write(
        "This app was trained on gym session data (age, body stats, heart rate, "
        "workout type/duration) to predict total calories burned per session. "
        "Multiple models were compared during training; the best performer was "
        "selected automatically."
    )
    st.write("**Model comparison:**")
    st.dataframe(pd.read_csv("models/model_comparison.csv")[["model", "test_r2", "test_mae"]], hide_index=True)

st.subheader("Enter your session details")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 15, 80, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    weight = st.slider("Weight (kg)", 40.0, 130.0, 70.0, step=0.5)
    height = st.slider("Height (m)", 1.40, 2.10, 1.70, step=0.01)
    experience = st.select_slider("Experience level", options=[1, 2, 3],
                                   value=2, format_func=lambda x: {1: "Beginner", 2: "Intermediate", 3: "Advanced"}[x])
    frequency = st.slider("Workout frequency (days/week)", 1, 7, 3)

with col2:
    workout_type = st.selectbox("Workout type", ["Cardio", "Strength", "HIIT", "Yoga"])
    duration = st.slider("Session duration (hours)", 0.25, 2.5, 1.0, step=0.05)
    resting_bpm = st.slider("Resting heart rate (BPM)", 45, 100, 65)
    max_bpm = st.slider("Max heart rate during session (BPM)", 120, 210, 180)
    avg_bpm = st.slider("Average heart rate during session (BPM)", resting_bpm + 5, max_bpm - 1, min(resting_bpm + 60, max_bpm - 2))
    fat_pct = st.slider("Body fat %", 5.0, 45.0, 22.0, step=0.5)
    water = st.slider("Water intake during workout (liters)", 0.5, 5.0, 2.0, step=0.1)

bmi = round(weight / (height ** 2), 2)
hr_reserve = max_bpm - resting_bpm
hr_intensity = (avg_bpm - resting_bpm) / hr_reserve if hr_reserve > 0 else 0
weight_x_duration = weight * duration

st.metric("Your BMI", bmi)

input_row = pd.DataFrame([{
    "Age": age, "Weight (kg)": weight, "Height (m)": height,
    "Max_BPM": max_bpm, "Avg_BPM": avg_bpm, "Resting_BPM": resting_bpm,
    "Session_Duration (hours)": duration, "Fat_Percentage": fat_pct,
    "Water_Intake (liters)": water, "Workout_Frequency (days/week)": frequency,
    "Experience_Level": experience, "BMI": bmi,
    "HR_Reserve": hr_reserve, "HR_Intensity": hr_intensity,
    "Weight_x_Duration": weight_x_duration,
    "Gender": gender, "Workout_Type": workout_type,
}])

if st.button("Predict calories burned", type="primary", use_container_width=True):
    pred = model.predict(input_row)[0]
    st.success(f"### Estimated calories burned: **{pred:,.0f} kcal**")

    # Compare to the reference population doing the same workout type
    peer = ref_df[ref_df["Workout_Type"] == workout_type]["Calories_Burned"]
    percentile = (peer < pred).mean() * 100
    st.write(
        f"That's higher than **{percentile:.0f}%** of {workout_type.lower()} sessions "
        f"in the reference dataset (avg: {peer.mean():.0f} kcal)."
    )
    st.bar_chart(pd.DataFrame({"kcal": [pred, peer.mean()]}, index=["Your session", f"Avg {workout_type}"]))

st.divider()
st.caption("Built as an end-to-end ML portfolio project: data cleaning → EDA → feature engineering → model comparison → deployment.")
