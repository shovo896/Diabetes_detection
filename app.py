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
    mins = stats.loc["min"]
    maxs = stats.loc["max"]

    inputs = {}
    for col in feature_cols:
        step = 0.1 if col in {"BMI", "DPF", "Glucose", "BloodPressure"} else 1.0
        inputs[col] = st.sidebar.number_input(
            col,
            min_value=float(mins[col]),
            max_value=float(maxs[col]),
            value=float(defaults[col]),
            step=step,
        )

    return pd.DataFrame([inputs], columns=feature_cols)


def main():
    st.title("Diabetes Risk Prediction")
    st.write(
        "Train a quick model from the Kaggle diabetes dataset and run individual predictions "
        "to estimate diabetes risk."
    )

    df = load_data()
    model, val_acc = train_model(df)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
    with col2:
        st.subheader("Validation Accuracy")
        st.metric(label="Accuracy (holdout)", value=f"{val_acc*100:.2f}%")

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

    with st.expander("Feature notes"):
        st.markdown(
            "- Pregnancies: Number of times pregnant.\n"
            "- Glucose: Plasma glucose concentration.\n"
            "- BloodPressure: Diastolic blood pressure (mm Hg).\n"
            "- SkinThickness: Triceps skin fold thickness (mm).\n"
            "- Insulin: 2-Hour serum insulin (mu U/ml).\n"
            "- BMI: Body mass index (weight/height^2).\n"
            "- DPF: Diabetes pedigree function.\n"
            "- Age: Age in years."
        )


if __name__ == "__main__":
    main()
