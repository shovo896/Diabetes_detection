import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🩺", layout="wide")


@st.cache_data
def load_data(csv_path: str = "kaggle_diabetes.csv") -> pd.DataFrame:
    """Load and clean the diabetes dataset."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"DiabetesPedigreeFunction": "DPF"})

    df = df.copy()
    df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = (
        df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.nan)
    )
    df["Glucose"] = df["Glucose"].fillna(df["Glucose"].mean())
    df["BloodPressure"] = df["BloodPressure"].fillna(df["BloodPressure"].mean())
    df["SkinThickness"] = df["SkinThickness"].fillna(df["SkinThickness"].median())
    df["Insulin"] = df["Insulin"].fillna(df["Insulin"].median())
    df["BMI"] = df["BMI"].fillna(df["BMI"].median())
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    """Train a simple pipeline and return model plus validation accuracy."""
    X = df.drop(columns="Outcome")
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=200, random_state=0)),
        ]
    )

    pipeline.fit(X_train, y_train)
    val_accuracy = pipeline.score(X_test, y_test)
    return pipeline, val_accuracy


def user_input_form(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar form for user inputs and return as single-row DataFrame."""
    st.sidebar.header("Patient Measurements")
    feature_cols = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DPF",
        "Age",
    ]

    stats = df[feature_cols].describe()
    defaults = stats.loc["mean"]

    inputs = {}
    for col in feature_cols:
        step = 0.1 if col in {"BMI", "DPF", "Glucose", "BloodPressure"} else 1.0
        # Keep data-driven bounds for most features, but allow unrestricted BMI and Age
        number_kwargs = {
            "label": col,
            "value": float(defaults[col]),
            "step": step,
        }
        if col not in {"BMI", "Age"}:
            number_kwargs["min_value"] = float(stats.loc["min", col])
            number_kwargs["max_value"] = float(stats.loc["max", col])

        inputs[col] = st.sidebar.number_input(**number_kwargs)

    return pd.DataFrame([inputs], columns=feature_cols)


def main():
    st.title("Diabetes Risk Prediction")
    st.write("Enter patient measurements to estimate diabetes risk.")

    df = load_data()
    model, _ = train_model(df)
    user_df = user_input_form(df)

    st.subheader("Prediction")
    if st.button("Predict diabetes risk"):
        proba = model.predict_proba(user_df)[0][1]
        pred = model.predict(user_df)[0]
        st.write(f"**Predicted class:** {'Positive' if pred == 1 else 'Negative'}")
        st.progress(float(proba))
        st.caption(f"Estimated probability of diabetes: {proba*100:.2f}%")
    else:
        st.info("Adjust measurements in the sidebar, then click Predict.")


if __name__ == "__main__":
    main()
