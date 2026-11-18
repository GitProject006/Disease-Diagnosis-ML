import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="Disease Diagnosis AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# VISUAL DESIGN
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
      radial-gradient(ellipse at 85% 8%, rgba(14,165,190,.18), transparent 34%),
      radial-gradient(ellipse at 8% 30%, rgba(124,91,220,.16), transparent 35%),
      #0b1425;
    color: #edf3ff;
}
.block-container { max-width: 1500px; padding-top: 1.5rem; padding-bottom: 2rem; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111d33 0%, #17213b 100%);
    border-right: 1px solid #2e4161;
}
section[data-testid="stSidebar"] * { color: #e8efff; }

h1, h2, h3 { letter-spacing: -0.5px; }
h1 { color: #ffffff !important; font-weight: 800 !important; }
h2 { color: #b8a7ff !important; font-weight: 750 !important; }
h3 { color: #77dce8 !important; font-weight: 700 !important; }
p, label { color: #cbd7e9; }

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1c3150, #172841);
    border: 1px solid #34516e;
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 8px 24px rgba(0,0,0,.14);
}
[data-testid="stMetricLabel"] { color: #b9c9df !important; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 800; }

div.stButton > button {
    background: linear-gradient(100deg, #7658df, #238eae);
    color: white;
    border: 0;
    border-radius: 10px;
    padding: .65rem 1rem;
    font-weight: 700;
}
div.stButton > button:hover {
    color: white;
    border: 1px solid #9be8ee;
    filter: brightness(1.12);
}
div[data-baseweb="select"] > div,
.stTextInput input {
    background: #172943;
    color: #ffffff;
    border: 1px solid #3b5878;
    border-radius: 10px;
}
[data-testid="stExpander"] {
    background: #172943;
    border: 1px solid #344d6b;
    border-radius: 12px;
}
hr { border-color: #30435f; }

.hero {
    min-height: 300px;
    padding: 32px;
    border-radius: 22px;
    background: linear-gradient(115deg, #293b68 0%, #315b79 54%, #176d78 100%);
    border: 1px solid #557b9d;
    box-shadow: 0 16px 35px rgba(0,0,0,.2);
}
.eyebrow {
    color: #8ce8e8;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
}
.hero h1 {
    font-size: 38px;
    line-height: 1.18;
    margin: 15px 0 12px 0;
}
.hero p { color: #e0ebf7; font-size: 15px; line-height: 1.8; max-width: 600px; }

.visual-panel {
    min-height: 300px;
    height: 100%;
    padding: 22px;
    border-radius: 22px;
    background: linear-gradient(145deg, #172b49, #13243d);
    border: 1px solid #3a5a7d;
    box-shadow: 0 16px 35px rgba(0,0,0,.2);
}
.visual-heading { color: #dceaff; font-size: 15px; font-weight: 700; }
.visual-sub { color: #8fa9c8; font-size: 11px; }
.scan-circle {
    margin: 22px auto 14px auto;
    width: 112px; height: 112px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 48px;
    background: radial-gradient(circle, #315d86, #1b3154 70%);
    border: 2px solid #46c8d4;
    box-shadow: 0 0 0 10px rgba(52,191,207,.08), 0 0 28px rgba(52,191,207,.25);
}
.signal-row { display:flex; gap:7px; justify-content:center; margin-top:18px; }
.signal { height:6px; border-radius:10px; background:#2d4868; flex:1; }
.signal:nth-child(1) { background:#8d72ef; }
.signal:nth-child(2) { background:#32c4d0; }
.signal:nth-child(3) { background:#e8a45d; }
.signal:nth-child(4) { background:#58c99e; }
.mini-stat {
    background: rgba(11,24,43,.55);
    border: 1px solid #304e70;
    border-radius: 12px;
    padding: 10px;
    text-align: center;
}
.mini-stat strong { display:block; color:#fff; font-size:18px; }
.mini-stat span { color:#9fb5d0; font-size:10px; }

.feature {
    background: linear-gradient(145deg, #1b2d49, #17253e);
    border: 1px solid #344d6b;
    border-radius: 15px;
    padding: 19px;
    min-height: 145px;
}
.feature-icon { font-size: 27px; }
.feature-title { color:#fff; font-size:16px; font-weight:750; margin-top:10px; }
.feature-copy { color:#aebed3; font-size:12px; line-height:1.6; margin-top:6px; }

.result {
    background: linear-gradient(110deg, #164c61, #176c70);
    border: 1px solid #45bdb5;
    border-radius: 16px;
    padding: 22px;
    margin: 12px 0 18px 0;
}
.result small { color:#b8f2e8; font-weight:800; letter-spacing:1.5px; }
.result div { color:white; font-size:28px; font-weight:800; margin-top:8px; }

.footer { text-align:center; color:#8298b5; font-size:11px; padding:18px 0 4px 0; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD TRAINED MODEL BUNDLE
# Expected file: artifacts/disease_model.pkl
# =========================================================
@st.cache_resource
def load_bundle():
    return joblib.load("artifacts/disease_model.pkl")

try:
    bundle = load_bundle()
    model = bundle["model"]
    mlb = bundle["mlb"]
    diseases = sorted([str(x) for x in bundle["eligible_classes"]])
    symptoms = sorted([str(x) for x in mlb.classes_])
except Exception as exc:
    st.error("The trained model could not be loaded.")
    st.markdown(
        "Check that **artifacts/disease_model.pkl** exists in your GitHub repository "
        "and that the file contains the trained model and symptom encoder."
    )
    st.code(str(exc))
    st.stop()

def as_series(value, name):
    """Convert saved counts (dict, Series, or list) into a Series."""
    if isinstance(value, pd.Series):
        return value
    if isinstance(value, dict):
        return pd.Series(value, dtype="float64")
    if isinstance(value, list):
        return pd.Series(value, name=name)
    return pd.Series(dtype="float64")

disease_counts = as_series(bundle.get("disease_counts", {}), "Records")
symptom_counts = as_series(bundle.get("symptom_counts", {}), "Frequency")

# =========================================================
# SIDEBAR NAVIGATION
# Note: no manual session_state assignment after radio.
# This avoids StreamlitWidgetAlreadyInstantiatedError.
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding:12px 0 20px 0">
          <div style="font-size:38px">🩺</div>
          <div style="font-size:22px;font-weight:800;color:#c0adff">
            Disease Diagnosis AI
          </div>
          <div style="font-size:11px;letter-spacing:1.5px;color:#8fa9c8;margin-top:5px">
            SMART HEALTH INSIGHTS
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("### WORKSPACE")
    page = st.radio(
        "Navigate",
        ["Dashboard", "Symptom Analysis", "Analytics", "Model Insights", "About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Symptom patterns  ·  Data analytics  ·  Machine learning")

# =========================================================
# DASHBOARD
# =========================================================
if page == "Dashboard":
    left, right = st.columns([1.15, .85], gap="large")

    with left:
        st.markdown(
            """
            <div class="hero">
              <div class="eyebrow">✦ INTELLIGENT HEALTH DATA</div>
              <h1>Explore health patterns<br>with machine learning</h1>
              <p>
                Explore symptom combinations, view model-generated category
                predictions and understand patterns in the dataset through
                a clean interactive workspace.
              </p>
              <div style="margin-top:22px;color:#b8f4ef;font-size:13px;font-weight:700">
                ✦ Symptom analysis &nbsp;&nbsp; ✦ Data insights &nbsp;&nbsp; ✦ Model evaluation
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        st.markdown(
            """
            <div class="visual-panel">
              <div class="visual-heading">AI HEALTH ANALYTICS</div>
              <div class="visual-sub">SYMPTOM PATTERN EXPLORER</div>
              <div class="scan-circle">🧬</div>
              <div style="text-align:center;color:#e5f4ff;font-weight:700">
                Pattern recognition
              </div>
              <div style="text-align:center;color:#8fa9c8;font-size:11px;margin-top:4px">
                Machine learning · Classification · Insights
              </div>
              <div class="signal-row">
                <div class="signal"></div><div class="signal"></div>
                <div class="signal"></div><div class="signal"></div>
              </div>
              <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:20px">
                <div class="mini-stat"><strong>01</strong><span>INPUT</span></div>
                <div class="mini-stat"><strong>02</strong><span>MODEL</span></div>
                <div class="mini-stat"><strong>03</strong><span>OUTPUT</span></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### WORKSPACE OVERVIEW")

    c1, c2, c3 = st.columns(3)
    c1.metric("Disease categories", len(diseases))
    c2.metric("Available symptoms", len(symptoms))
    c3.metric("Models evaluated", len(bundle.get("model_results", [])))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Explore your workspace")

    cards = st.columns(3)
    card_data = [
        ("🔎", "Symptom Analysis", "Select symptoms and view the trained model output.", "#8b72ed"),
        ("📊", "Analytics", "Explore disease distribution and frequent symptoms.", "#25b9c8"),
        ("🤖", "Model Insights", "Compare model evaluation results.", "#e5a45f")
    ]
    for col, (icon, title, desc, accent) in zip(cards, card_data):
        with col:
            st.markdown(
                f"""
                <div class="feature" style="border-top:3px solid {accent}">
                  <div class="feature-icon">{icon}</div>
                  <div class="feature-title">{title}</div>
                  <div class="feature-copy">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.info("Model predictions are not diagnoses and must not guide healthcare decisions.")

# =========================================================
# SYMPTOM ANALYSIS
# =========================================================
elif page == "Symptom Analysis":
    st.markdown("### INTERACTIVE ANALYSIS")
    st.title("Symptom Analysis")
    st.write("Select symptoms that appear in the dataset to view the classification model's output.")

    selected_symptoms = st.multiselect(
        "Select symptoms",
        options=symptoms,
        placeholder="Search and select one or more symptoms..."
    )

    st.caption(f"{len(selected_symptoms)} symptom(s) selected")

    if st.button("Run model analysis", use_container_width=True):
        if not selected_symptoms:
            st.warning("Select at least one symptom before running the model.")
        else:
            try:
                input_data = mlb.transform([selected_symptoms])
                prediction = model.predict(input_data)[0]

                st.markdown(
                    f"""
                    <div class="result">
                      <small>MODEL-GENERATED CATEGORY</small>
                      <div>{prediction}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("#### Input summary")
                st.write(", ".join(selected_symptoms))

                if hasattr(model, "predict_proba"):
                    scores = model.predict_proba(input_data)[0]
                    classes = model.classes_
                    top_idx = np.argsort(scores)[::-1][:3]
                    score_df = pd.DataFrame({
                        "Model category": [str(classes[i]) for i in top_idx],
                        "Model score (%)": [round(float(scores[i]) * 100, 2) for i in top_idx]
                    })
                    st.markdown("#### Highest model scores")
                    st.dataframe(score_df, use_container_width=True, hide_index=True)
                    st.caption(
                        "Scores describe the model output only. They are not medical probabilities."
                    )
                else:
                    st.caption("This trained model does not provide probability scores.")

                st.warning(
                    "Model predictions are not diagnoses and must not guide healthcare decisions."
                )
            except Exception as exc:
                st.error("The model could not process this selection.")
                st.code(str(exc))

# =========================================================
# ANALYTICS
# =========================================================
elif page == "Analytics":
    st.markdown("### DATA EXPLORATION")
    st.title("Dataset Analytics")
    st.write("Explore the distribution of categories and the frequency of symptom features.")

    if disease_counts.empty and symptom_counts.empty:
        st.warning("Saved dataset summary counts were not found in the model bundle.")
    else:
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("#### Disease category distribution")
            if not disease_counts.empty:
                plot_data = disease_counts.sort_values(ascending=False).head(20)
                fig, ax = plt.subplots(figsize=(8, 5))
                fig.patch.set_facecolor("#0f1d32")
                ax.set_facecolor("#172943")
                plot_data.sort_values().plot(kind="barh", ax=ax, color="#987df0")
                ax.set_xlabel("Records", color="white")
                ax.tick_params(colors="#dce7f5")
                for spine in ax.spines.values():
                    spine.set_color("#385371")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        with right:
            st.markdown("#### Most frequent symptoms")
            if not symptom_counts.empty:
                plot_data = symptom_counts.sort_values(ascending=False).head(15)
                fig, ax = plt.subplots(figsize=(8, 5))
                fig.patch.set_facecolor("#0f1d32")
                ax.set_facecolor("#172943")
                plot_data.sort_values().plot(kind="barh", ax=ax, color="#29bdc9")
                ax.set_xlabel("Frequency", color="white")
                ax.tick_params(colors="#dce7f5")
                for spine in ax.spines.values():
                    spine.set_color("#385371")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        if not disease_counts.empty:
            st.markdown("#### Category records")
            counts_df = disease_counts.rename("Record count").rename_axis("Category").reset_index()
            st.dataframe(counts_df, use_container_width=True, hide_index=True)

# =========================================================
# MODEL INSIGHTS
# =========================================================
elif page == "Model Insights":
    st.markdown("### MODEL EVALUATION")
    st.title("Model Insights")
    st.write("Review the classification models and evaluation metrics saved during training.")

    raw_results = bundle.get("model_results", [])
    results = pd.DataFrame(raw_results)

    if results.empty:
        st.warning("Model comparison results were not found in the saved bundle.")
    else:
        st.markdown("#### Model comparison")
        st.dataframe(results.round(4), use_container_width=True, hide_index=True)

        if "Accuracy" in results.columns and "Model" in results.columns:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            fig.patch.set_facecolor("#0f1d32")
            ax.set_facecolor("#172943")
            palette = ["#987df0", "#29bdc9", "#e6a45f", "#55c69a", "#e27e9c", "#6e9de2"]
            ax.bar(results["Model"].astype(str), results["Accuracy"], color=palette[:len(results)])
            ax.set_ylabel("Accuracy", color="white")
            ax.set_ylim(0, 1.05)
            ax.tick_params(axis="x", rotation=25, colors="#dce7f5")
            ax.tick_params(axis="y", colors="#dce7f5")
            for spine in ax.spines.values():
                spine.set_color("#385371")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.markdown("#### Selected final model")
    st.success(str(bundle.get("model_name", "Not specified")))

    metrics = bundle.get("metrics", {})
    if metrics:
        st.markdown("#### Final evaluation metrics")
        metric_cols = st.columns(min(4, len(metrics)))
        for idx, (key, value) in enumerate(metrics.items()):
            metric_cols[idx % len(metric_cols)].metric(str(key), f"{float(value):.3f}" if isinstance(value, (int, float, np.number)) else str(value))
    else:
        st.caption("No final metrics were saved in the model bundle.")

# =========================================================
# ABOUT
# =========================================================
elif page == "About":
    st.markdown("### PLATFORM INFORMATION")
    st.title("About")
    st.write(
        "This application uses a trained machine-learning classifier to explore "
        "relationships between symptom features and disease-category labels in a dataset."
    )

    a, b = st.columns(2)
    with a:
        st.markdown(
            """
            <div class="feature">
              <div class="feature-icon">🧠</div>
              <div class="feature-title">Machine learning</div>
              <div class="feature-copy">
                A trained classification model processes selected symptom features
                and returns a dataset category.
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with b:
        st.markdown(
            """
            <div class="feature">
              <div class="feature-icon">📈</div>
              <div class="feature-title">Data exploration</div>
              <div class="feature-copy">
                Charts and model comparisons help explore the dataset and
                understand the model's evaluation results.
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("#### Technology")
    st.write("Python · Pandas · NumPy · Scikit-learn · Matplotlib · Streamlit")

    st.markdown("#### Dataset limitations")
    st.write(
        "The dataset contains repeated records and may not represent real-world clinical populations. "
        "The model only predicts labels represented in its training data."
    )

    st.info("Model predictions are not diagnoses and must not guide healthcare decisions.")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    '<div class="footer">Disease Diagnosis AI &nbsp;·&nbsp; '
    'Model predictions are not diagnoses and must not guide healthcare decisions.</div>',
    unsafe_allow_html=True
)
