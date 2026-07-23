from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "registration_model_pipeline.joblib"

FEATURE_COLUMNS = [
    "time_on_page_seconds",
    "scroll_depth_percent",
    "previous_visits",
    "started_registration_form",
    "traffic_source",
]

TRAFFIC_SOURCE_OPTIONS = [
    "Paid Ad",
    "Google Search",
    "LinkedIn",
    "YouTube",
    "Organic / Direct",
    "Email",
    "Unknown",
]


st.set_page_config(
    page_title="Registration Probability Predictor",
    page_icon=None,
    layout="centered",
)


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #071018 0%, #101820 52%, #071018 100%);
        color: #f4f7fb;
    }

    .block-container {
        max-width: 820px;
        padding-top: 3.5rem;
        padding-bottom: 3.5rem;
    }

    h1, h2, h3, p, label, span {
        color: #f4f7fb;
    }

    .app-shell {
        border: 1px solid rgba(116, 185, 210, 0.24);
        background: rgba(10, 21, 32, 0.82);
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.36);
        border-radius: 8px;
        padding: 2rem;
    }

    .app-kicker {
        color: #77d7c8;
        font-size: 0.84rem;
        font-weight: 700;
        letter-spacing: 0.08rem;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .app-title {
        font-size: 2.25rem;
        line-height: 1.15;
        font-weight: 760;
        margin-bottom: 0.6rem;
    }

    .app-subtitle {
        color: #b8c6d1;
        font-size: 1rem;
        line-height: 1.55;
        margin-bottom: 1.45rem;
    }

    .result-panel {
        border: 1px solid rgba(119, 215, 200, 0.34);
        background: rgba(8, 31, 42, 0.82);
        border-radius: 8px;
        padding: 1.1rem 1.25rem;
        margin-top: 1.25rem;
    }

    .result-label {
        color: #9fb1c0;
        font-size: 0.86rem;
        margin-bottom: 0.2rem;
    }

    .result-value {
        color: #77d7c8;
        font-size: 2rem;
        font-weight: 780;
        margin-bottom: 0.55rem;
    }

    .recommendation {
        color: #f4f7fb;
        font-size: 1rem;
        line-height: 1.55;
    }

    div.stButton > button {
        width: 100%;
        border: 1px solid rgba(119, 215, 200, 0.55);
        background: linear-gradient(90deg, #1f7a8c 0%, #2f9f8f 100%);
        color: #ffffff;
        font-weight: 700;
        border-radius: 6px;
        padding: 0.7rem 1rem;
    }

    div.stButton > button:hover {
        border-color: rgba(119, 215, 200, 0.9);
        background: linear-gradient(90deg, #258ca1 0%, #35b09e 100%);
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """Load the fitted preprocessing plus model pipeline once per app session."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def get_probability_band(probability):
    """Translate the model probability into a simple action band."""
    if probability < 0.40:
        return (
            "Low",
            "This visitor may need help. Trigger a chat widget or FAQ prompt to answer questions and reduce friction.",
        )
    if probability < 0.70:
        return (
            "Medium",
            "This visitor is showing some intent. Show reassurance, such as a student success story or short trust-building message.",
        )
    return (
        "High",
        "This visitor looks likely to register. Avoid interruption and keep the call-to-action visible.",
    )


def build_input_dataframe(
    traffic_source,
    time_on_page_seconds,
    scroll_depth_percent,
    previous_visits,
    started_registration_form,
):
    """Create the single-row input with the exact feature names used in training."""
    return pd.DataFrame(
        [
            {
                "time_on_page_seconds": float(time_on_page_seconds),
                "scroll_depth_percent": float(scroll_depth_percent),
                "previous_visits": int(previous_visits),
                "started_registration_form": int(started_registration_form),
                "traffic_source": traffic_source,
            }
        ],
        columns=FEATURE_COLUMNS,
    )


st.markdown('<div class="app-shell">', unsafe_allow_html=True)
st.markdown('<div class="app-kicker">Data Science Career Session</div>', unsafe_allow_html=True)
st.markdown('<div class="app-title">Registration probability predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Enter visitor session details to estimate the chance that the visitor will register.</div>',
    unsafe_allow_html=True,
)

try:
    model = load_model()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

with st.form("registration_prediction_form"):
    traffic_source = st.selectbox(
        "Traffic source",
        TRAFFIC_SOURCE_OPTIONS,
        index=0,
    )
    time_on_page_seconds = st.number_input(
        "Time on page, seconds",
        min_value=0,
        max_value=600,
        value=120,
        step=5,
    )
    scroll_depth_percent = st.slider(
        "Scroll depth percent",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=0.5,
        format="%.1f%%",
    )
    previous_visits = st.number_input(
        "Previous visits",
        min_value=0,
        max_value=50,
        value=1,
        step=1,
    )
    started_registration_form_label = st.radio(
        "Did the visitor start the registration form?",
        ["No", "Yes"],
        horizontal=True,
    )

    submitted = st.form_submit_button("Predict registration probability")

if submitted:
    started_registration_form = 1 if started_registration_form_label == "Yes" else 0
    input_df = build_input_dataframe(
        traffic_source=traffic_source,
        time_on_page_seconds=time_on_page_seconds,
        scroll_depth_percent=scroll_depth_percent,
        previous_visits=previous_visits,
        started_registration_form=started_registration_form,
    )

    # The saved pipeline applies the same preprocessing used during training.
    probability = model.predict_proba(input_df)[0, 1]
    band, recommendation = get_probability_band(probability)

    st.markdown(
        f"""
        <div class="result-panel">
            <div class="result-label">Predicted probability of registration</div>
            <div class="result-value">{probability:.0%}</div>
            <div class="recommendation"><strong>{band} probability:</strong> {recommendation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)
