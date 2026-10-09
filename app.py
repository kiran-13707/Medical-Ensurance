
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Medical Insurance Prediction",
    page_icon="📊",
    layout="wide"
)

# ---------------- HIDE HEADING LINK ICONS ----------------
st.markdown(
    """
    <style>
    [data-testid="stHeaderActionElements"],
    [data-testid="stMarkdownContainer"] a.anchor-link,
    a[data-testid="stHeaderActionElements"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- MODEL FILE PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent


# ---------------- LOAD MODEL FILES ----------------
@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "decision_tree_regresion.pkl")
    scaler = joblib.load(BASE_DIR / "scal.pkl")
    columns = joblib.load(BASE_DIR / "column.pkl")
    return model, scaler, columns


# ---------------- SIDEBAR ----------------
st.sidebar.title("InsurePredict")
st.sidebar.caption("Medical Insurance Analytics")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["Predict Charges", "About Project"]
)

st.sidebar.divider()
st.sidebar.subheader("Model Information")
st.sidebar.write("Algorithm: Decision Tree Regressor")
st.sidebar.write("Problem Type: Regression")


# ---------------- MAIN HEADER ----------------
st.title("Medical Insurance Charges Prediction")

st.write(
    "Estimate medical insurance charges using a trained "
    "machine learning model."
)

st.divider()


# ---------------- ABOUT PROJECT PAGE ----------------
if page == "About Project":

    st.header("About the Project")

    st.write(
        "This application predicts estimated medical insurance "
        "charges using customer demographic and lifestyle information."
    )

    st.subheader("Input Features")

    st.markdown("""
    - Age
    - Sex
    - BMI
    - Number of Children
    - Smoking Status
    - Residential Region
    """)

    st.subheader("Technologies Used")

    st.markdown("""
    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - Joblib
    """)

    st.info(
        "This is an educational machine learning project. "
        "Predictions are estimates, not official insurance quotes."
    )

    st.stop()


# ---------------- PREDICTION PAGE ----------------
st.header("Customer Information")

st.write(
    "Enter the details below to calculate the estimated charges."
)

try:
    model, scaler, feature_columns = load_model()

except Exception as error:
    st.error(f"Unable to load model files: {error}")
    st.stop()


# ---------------- INPUT FORM ----------------
with st.form("insurance_form"):

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=30,
            step=1
        )

        sex = st.selectbox(
            "Sex",
            ["Female", "Male"]
        )

        bmi = st.number_input(
            "Body Mass Index (BMI)",
            min_value=10.0,
            max_value=80.0,
            value=27.5,
            step=0.1,
            help="Enter the customer's BMI value."
        )

    with col2:

        children = st.number_input(
            "Number of Children",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

        smoker = st.selectbox(
            "Smoking Status",
            ["No", "Yes"]
        )

        region = st.selectbox(
            "Residential Region",
            [
                "Northeast",
                "Northwest",
                "Southeast",
                "Southwest"
            ]
        )

    submitted = st.form_submit_button(
        "Calculate Insurance Charges",
        type="primary",
        use_container_width=True
    )


# ---------------- PREDICTION LOGIC ----------------
if submitted:

    try:

        sex_mapping = {
            "Female": 0,
            "Male": 1
        }

        smoker_mapping = {
            "No": 0,
            "Yes": 1
        }

        region_mapping = {
            "Northeast": 0,
            "Northwest": 1,
            "Southeast": 2,
            "Southwest": 3
        }

        input_values = {
            "age": age,
            "sex": sex_mapping[sex],
            "bmi": bmi,
            "children": children,
            "smoker": smoker_mapping[smoker],
            "region": region_mapping[region]
        }

        if set(feature_columns) != set(input_values):

            st.error(
                "Feature columns do not match the saved model. "
                "Please check column.pkl."
            )

            st.stop()

        input_data = pd.DataFrame(
            [
                [
                    input_values[column]
                    for column in feature_columns
                ]
            ],
            columns=feature_columns
        )

        scaled_data = scaler.transform(input_data)

        prediction = float(
            model.predict(scaled_data)[0]
        )

        if not pd.notna(prediction):
            raise ValueError("Invalid prediction received.")

        # ---------------- DISPLAY RESULT ----------------
        st.divider()

        st.header("Prediction Result")

        st.metric(
            label="Estimated Insurance Charges",
            value=f"${prediction:,.2f}"
        )

        st.subheader("Customer Summary")

        result_col1, result_col2, result_col3 = st.columns(3)

        result_col1.metric(
            "Age",
            f"{age} years"
        )

        result_col2.metric(
            "BMI",
            f"{bmi:.1f}"
        )

        result_col3.metric(
            "Children",
            children
        )

        st.write(f"**Sex:** {sex}")
        st.write(f"**Smoking Status:** {smoker}")
        st.write(f"**Region:** {region}")

        st.caption(
            "This is a machine learning estimate in USD, "
            "not an official insurance quotation."
        )

    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )


# ---------------- FOOTER ----------------
st.divider()

st.caption(
    "Medical Insurance Prediction | Built with Streamlit"
)