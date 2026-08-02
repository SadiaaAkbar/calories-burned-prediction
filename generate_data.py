"""
Generates a synthetic dataset that matches the schema, ranges, and known
correlations of the public "Gym Members Exercise Dataset" (Kaggle,
valakhorasani/gym-members-exercise-dataset, 973 rows).

WHY THIS EXISTS: Kaggle requires an authenticated download, which isn't
reachable from this build environment. This script recreates the same
columns/distributions/relationships documented on the dataset page so the
full pipeline (cleaning -> EDA -> features -> modeling) is fully functional.

TO USE THE REAL DATA FOR YOUR SUBMISSION:
1. Download from https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset
2. Save it as data/gym_data.csv (same column names, so nothing else changes)
3. Delete/skip this script
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 973

age = np.random.randint(18, 60, N)
gender = np.random.choice(["Male", "Female"], N, p=[0.51, 0.49])

# Weight/height correlated with gender, with realistic spread
height = np.where(
    gender == "Male",
    np.random.normal(1.75, 0.08, N),
    np.random.normal(1.63, 0.07, N),
).clip(1.5, 2.0)

base_weight = np.where(gender == "Male", 78, 65)
weight = (base_weight + np.random.normal(0, 14, N)).clip(40, 130)

experience_level = np.random.choice([1, 2, 3], N, p=[0.45, 0.35, 0.20])
workout_frequency = np.clip(
    (experience_level + np.random.normal(1.5, 1.0, N)).round(), 2, 6
).astype(int)

workout_type = np.random.choice(
    ["Cardio", "Strength", "Yoga", "HIIT"], N, p=[0.3, 0.3, 0.15, 0.25]
)

# Fitter / more experienced people -> lower resting BPM, lower fat %
resting_bpm = (75 - experience_level * 4 + np.random.normal(0, 6, N)).clip(45, 100).round().astype(int)
max_bpm = (200 - age * 0.5 + np.random.normal(0, 6, N)).clip(150, 210).round().astype(int)
intensity_bump = np.select(
    [workout_type == "HIIT", workout_type == "Cardio", workout_type == "Strength", workout_type == "Yoga"],
    [25, 15, 8, -5],
)
avg_bpm = (resting_bpm + 40 + intensity_bump + np.random.normal(0, 8, N)).clip(resting_bpm + 10, max_bpm - 2).round().astype(int)

fat_pct = (
    np.where(gender == "Male", 22, 30)
    - experience_level * 2.5
    + np.random.normal(0, 5, N)
).clip(5, 45)

session_duration = np.clip(
    np.random.normal(1.0 + experience_level * 0.15, 0.35, N), 0.25, 2.5
).round(2)

water_intake = np.clip(
    1.5 + weight * 0.02 + session_duration * 0.6 + np.random.normal(0, 0.4, N), 1.0, 5.0
).round(2)

bmi = (weight / (height ** 2)).round(2)

# Calories burned: the real target, driven mainly by duration, intensity (avg_bpm),
# body weight, and workout type — with realistic noise
type_multiplier = np.select(
    [workout_type == "HIIT", workout_type == "Cardio", workout_type == "Strength", workout_type == "Yoga"],
    [1.25, 1.1, 0.95, 0.75],
)
calories_burned = (
    (avg_bpm - resting_bpm) * session_duration * 9.5 * type_multiplier
    + weight * 2.5
    + np.random.normal(0, 60, N)
).clip(150, 2000).round().astype(int)

df = pd.DataFrame({
    "Age": age,
    "Gender": gender,
    "Weight (kg)": weight.round(1),
    "Height (m)": height.round(2),
    "Max_BPM": max_bpm,
    "Avg_BPM": avg_bpm,
    "Resting_BPM": resting_bpm,
    "Session_Duration (hours)": session_duration,
    "Calories_Burned": calories_burned,
    "Workout_Type": workout_type,
    "Fat_Percentage": fat_pct.round(1),
    "Water_Intake (liters)": water_intake,
    "Workout_Frequency (days/week)": workout_frequency,
    "Experience_Level": experience_level,
    "BMI": bmi,
})

# Inject a bit of realistic messiness so the cleaning step in the notebook is genuine
missing_idx = np.random.choice(df.index, 18, replace=False)
df.loc[missing_idx, "Fat_Percentage"] = np.nan
dup_rows = df.sample(5, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

df.to_csv("/home/claude/capstone/data/gym_data.csv", index=False)
print(f"Wrote {len(df)} rows to data/gym_data.csv")
print(df.head())
