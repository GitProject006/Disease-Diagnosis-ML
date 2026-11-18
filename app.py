
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Disease Diagnosis AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 2. CUSTOM DESIGN
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 5% 5%, #29345B 0%, transparent 32%),
        radial-gradient(ellipse at 95% 20%, #174452 0%, transparent 30%),
        linear-gradient(135deg, #101B30, #17243C 55%, #101D32);
    color: #F5F7FF;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1 {
    color: #FFFFFF !important;
    font-weight: 800 !important;
}

h2 {
    color: #BDAAFF !important;
    font-weight: 700 !important;
}

h3 {
    color: #79DFE9 !important;
    font-weight: 700 !important;
}

p, label {
    color: #D5DDED;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111B30, #202744);
    border-right: 1px solid #394665;
}

section[data-testid="stSidebar"] h1 {
    color: #BDAAFF !important;
}

/* METRIC CARDS */

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #293B5D, #1C2D48);
    border: 1px solid #465B7A;
    padding: 23px;
    border-radius: 18px;
    box-shadow: 0 8px 22px rgba(0,0,0,.16);
}

[data-testid="stMetricLabel"] {
    color: #C4D0E2 !important;
}

[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-weight: 800;
}

/* BUTTONS */

div.stButton > button {
    background: linear-gradient(100deg, #8064E8, #A16BEA);
    color: white;
    border: none;
    border-radius: 11px;
    padding: 11px 16px;
    font-weight: 700;
    transition: .2s ease;
}

div.stButton > button:hover {
    background: linear-gradient(100deg, #A16BEA, #13AFC1);
    color: white;
    transform: translateY(-2px);
}

/* INPUTS */

div[data-baseweb="select"] > div {
    background: #202F4B;
    border: 1px solid #596A91;
    border-radius: 10px;
}

.stTextInput input {
    background: #202F4B;
    color: white;
    border-radius: 10px;
}

/* HERO */

.hero {
    background:
        radial-gradient(circle at 90% 10%, #60478E 0%, transparent 42%),
        linear-gradient(120deg, #263C62, #3B4F7B 55%, #286174);
    border: 1px solid #6479A7;
    border-radius: 23px;
    padding: 32px;
    min-height: 320px;
    box-shadow: 0 12px 30px rgba(0,0,0,.16);
}

.hero-tag {
    color: #85E3E8;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 2px;
}

.hero-title {
    color: #FFFFFF;
    font-size: 35px;
    line-height: 1.25;
    font-weight: 800;
    margin-top: 15px;
}

.hero-description {
    color: #E0E7F4;
    font-size: 15px;
    line-height: 1.8;
    margin-top: 14px;
}

/* AI ILLUSTRATION */

.ai-visual {
    background:
        radial-gradient(circle at center, #416D9B 0%, #25446C 35%,
        #202D50 68%, #17243D 100%);
    border: 1px solid #6479A7;
    border-radius: 23px;
    min-height: 320px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 20px;
}

.ai-symbol {
    font-size: 90px;
    line-height: 1.4;
}

.ai-visual-title {
    font-size: 23px;
    font-weight: 800;
    color: white;
}

.ai-visual-subtitle {
    color: #B9E6EA;
    font-size: 13px;
    margin-top: 8px;
}

/* FEATURE CARDS */

.feature-card {
    background: linear-gradient(145deg, #293B5D, #202D49);
    border: 1px solid #465A7B;
    border-radius: 17px;
    padding: 22px;
    min-height: 155px;
    margin-bottom: 12px;
}

.feature-icon {
    font-size: 32px;
    margin-bottom: 12px;
}

.feature-title {
    color: #FFFFFF;
    font-size: 18px;
    font-weight: 800;
}

.feature-description {
    color: #C2CEE0;
    font-size: 13px;
    line-height: 1.7;
    margin-top: 8px;
}

/* RESULT CARD */

.result-card {
    background: linear-gradient(115deg, #174F65, #176E72);
    border: 1px solid #54BDB5;
    border-radius: 18px;
    padding: 25px;
    margin-top: 15px;
}

.result-label {
    color: #B8F0E7;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 1px;
}

.result-value {
    color: white;
    font-size: 29px;
    font-weight: 800;
    margin-top: 8px;
}

/* OTHER ELEMENTS */

[data-testid="stExpander"] {
    background: #202F4B;
    border: 1px solid #465A7B;
    border-radius: 12px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #465A7B;
    border-radius: 12px;
}

hr {
    border-color: #465675;
}

.footer {
    text-align: center;
    color: #AAB8CF;
    font-size: 12px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD TRAINED MODEL
# =====================================================

@st.cache_resource
def load_bundle():
    return joblib.load("artifacts/disease_model.pkl")


try:
    bundle = load_bundle()

    model = bundle["model"]
    mlb = bundle["mlb"]

    diseases = sorted(list(bundle["eligible_classes"]))
    symptoms = sorted(list(mlb.classes_))

except Exception as e:
    st.error("Unable to load the trained model.")
    st.write("Check the model file and folder structure.")
    st.code(str(e))
    st.stop()


# =====================================================
# 4. NAVIGATION
# =====================================================

page_options = [
    "Home",
    "Disease Prediction",
    "Disease Library",
    "EDA Dashboard",
    "Model Performance",
    "About"
]

if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = "Home"

with st.sidebar:

    st.markdown("""
    <div style="padding:12px 0 22px 0;">
        <div style="font-size:38px;">🩺</div>
        <div style="font-size:23px;font-weight:800;color:#BDAAFF;">
            Disease Diagnosis AI
        </div>
        <div style="font-size:12px;color:#9DB6D1;margin-top:6px;">
            SMART HEALTH INSIGHTS
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Explore")

    page = st.radio(
        "Navigation",
        page_options,
        key="selected_page",
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("""
    <div style="color:#AAB8CF;font-size:12px;line-height:2;">
        <b style="color:#79DFE9;">DISEASE DIAGNOSIS AI</b><br>
        Symptom insights<br>
        Data exploration<br>
        Machine learning
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# 5. HOME PAGE
# =====================================================

if page == "Home":

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:

        st.markdown("""
        <div class="hero">
            <div class="hero-tag">✦ SMART HEALTH INSIGHTS</div>
            <div class="hero-title">
                Discover patterns.<br>
                Explore possibilities.
            </div>
            <div class="hero-description">
                Explore symptom combinations, discover
                disease categories and understand the
                power of machine learning through
                interactive data insights.
            </div>
            <br>
            <div style="color:#9AE9E8;font-size:14px;">
                ✦ Symptom Explorer &nbsp; ✦ AI Insights
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right:

        st.markdown("""
        <div class="ai-visual">
            <div class="ai-symbol">🧬</div>
            <div class="ai-visual-title">AI + Healthcare</div>
            <div class="ai-visual-subtitle">
                Connecting symptoms with data
            </div>
            <div style="font-size:35px;margin-top:18px;">
                ✚ &nbsp; ◉ &nbsp; ✧
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Your platform at a glance")

    c1, c2, c3 = st.columns(3)

    c1.metric("Disease Categories", len(diseases))
    c2.metric("Available Symptoms", len(symptoms))
    c3.metric("ML Algorithms", len(bundle["model_results"]))

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("## Explore the experience")
    st.caption("Choose an area to explore.")

    features = [
        (
            "🔎",
            "Symptom Explorer",
            "Explore symptom combinations and model predictions.",
            "Disease Prediction",
            "#8064E8"
        ),
        (
            "🧬",
            "Disease Explorer",
            "Browse categories represented in the dataset.",
            "Disease Library",
            "#13AFC1"
        ),
        (
            "📊",
            "Data Insights",
            "Discover patterns and explore interactive charts.",
            "EDA Dashboard",
            "#E58A69"
        ),
        (
            "🤖",
            "AI Performance",
            "Explore model comparisons and evaluation scores.",
            "Model Performance",
            "#45B995"
        )
    ]

    for start in range(0, len(features), 2):

        col1, col2 = st.columns(2)

        for col, item in zip(
            [col1, col2],
            features[start:start + 2]
        ):

            with col:

                st.markdown(f"""
                <div style="
                    background:linear-gradient(145deg,#293B5D,#202D49);
                    border:1px solid #465A7B;
                    border-left:5px solid {item[4]};
                    border-radius:17px;
                    padding:22px;
                    min-height:150px;
                    margin-bottom:12px;">
                    <div style="font-size:30px;">{item[0]}</div>
                    <div style="font-size:18px;font-weight:800;
                    color:white;margin-top:10px;">
                        {item[1]}
                    </div>
                    <div style="font-size:13px;color:#C4D0E2;
                    line-height:1.6;margin-top:8px;">
                        {item[2]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button(
                    "Explore →",
                    key="home_" + item[3],
                    use_container_width=True
                ):
                    st.session_state["selected_page"] = item[3]
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background:linear-gradient(110deg,#263C62,#315D77);
        border:1px solid #526F99;
        padding:24px;
        border-radius:17px;">
        <div style="font-size:20px;font-weight:800;color:white;">
            A smarter way to explore data
        </div>
        <div style="font-size:14px;color:#D4E5F1;margin-top:8px;">
            Discover how machine learning can identify patterns
            in structured symptom datasets.
        </div>
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# 6. DISEASE PREDICTION
# =====================================================

elif page == "Disease Prediction":

    st.markdown("### ✦ INTERACTIVE TOOL")
    st.title("Symptom Analysis")

    st.write(
        "Choose symptoms from the available dataset to "
        "explore the trained machine-learning model's output."
    )

    st.warning(
        "Model predictions are not diagnoses and must not "
        "guide healthcare decisions."
    )

    selected_symptoms = st.multiselect(
        "Search and select symptoms",
        options=symptoms,
        placeholder="Start typing to find symptoms..."
    )

    st.caption(f"{len(selected_symptoms)} symptom(s) selected")

    if selected_symptoms:

        st.markdown("### Selected symptoms")

        st.write(", ".join(selected_symptoms))

    if st.button(
        "Analyze Selected Symptoms",
        use_container_width=True
    ):

        if not selected_symptoms:

            st.warning("Please select at least one symptom.")

        else:

            input_data = mlb.transform([selected_symptoms])

            prediction = model.predict(input_data)[0]

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">MODEL PREDICTION</div>
                <div class="result-value">{prediction}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Top model scores")

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(input_data)[0]
                classes = model.classes_

                top_indices = np.argsort(probabilities)[::-1][:3]

                top_results = pd.DataFrame({
                    "Predicted Category": classes[top_indices],
                    "Model Score (%)": (
                        probabilities[top_indices] * 100
                    ).round(2)
                })

                st.dataframe(
                    top_results,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "Model scores are not medical probabilities "
                    "or estimates of the likelihood of an illness."
                )

            else:

                st.info(
                    "This model does not provide probability scores."
                )


# =====================================================
# 7. DISEASE LIBRARY
# =====================================================

elif page == "Disease Library":

    st.markdown("### ✦ EXPLORE CATEGORIES")
    st.title("Disease Library")

    st.write(
        "Browse the disease categories represented "
        "in the classification dataset."
    )

    search = st.text_input(
        "Search categories",
        placeholder="Type a category name..."
    )

    filtered = [
        d for d in diseases
        if search.lower() in d.lower()
    ]

    st.caption(f"{len(filtered)} categories found")

    if filtered:

        cols = st.columns(2)

        for index, disease in enumerate(filtered):

            with cols[index % 2]:

                st.markdown(f"""
                <div style="
                    background:linear-gradient(145deg,#293B5D,#202D49);
                    border:1px solid #465A7B;
                    border-left:4px solid #13AFC1;
                    border-radius:14px;
                    padding:18px;
                    margin-bottom:12px;">
                    <div style="font-size:18px;font-weight:800;
                    color:white;">🧬 {disease}</div>
                    <div style="font-size:13px;color:#C4D0E2;
                    margin-top:8px;">
                        Included in the symptom classification dataset.
                    </div>
                </div>
                """, unsafe_allow_html=True)

    else:

        st.info("No matching category found.")


# =====================================================
# 8. EDA DASHBOARD
# =====================================================

elif page == "EDA Dashboard":

    st.markdown("### ✦ DATA EXPLORATION")
    st.title("Data Insights")

    st.write(
        "Explore disease distribution and symptom "
        "frequency patterns in the dataset."
    )

    disease_counts = pd.Series(
        bundle["disease_counts"]
    ).sort_values(ascending=False)

    symptom_counts = pd.Series(
        bundle["symptom_counts"]
    ).sort_values(ascending=False).head(15)

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("### Disease Distribution")

        fig, ax = plt.subplots(figsize=(9, 6))

        fig.patch.set_facecolor("#17243C")
        ax.set_facecolor("#202F4B")

        disease_counts.plot(
            kind="bar",
            ax=ax,
            color="#9B82F2"
        )

        ax.set_ylabel("Number of Records", color="white")
        ax.tick_params(axis="x", rotation=90, colors="white")
        ax.tick_params(axis="y", colors="white")

        for spine in ax.spines.values():
            spine.set_color("#52627F")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with c2:

        st.markdown("### Most Frequent Symptoms")

        fig, ax = plt.subplots(figsize=(9, 6))

        fig.patch.set_facecolor("#17243C")
        ax.set_facecolor("#202F4B")

        symptom_counts.sort_values().plot(
            kind="barh",
            ax=ax,
            color="#27BED0"
        )

        ax.set_xlabel("Frequency", color="white")
        ax.tick_params(colors="white")

        for spine in ax.spines.values():
            spine.set_color("#52627F")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("### Dataset Summary")

    summary = pd.DataFrame({
        "Disease Category": disease_counts.index,
        "Record Count": disease_counts.values
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# 9. MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.markdown("### ✦ MODEL EVALUATION")
    st.title("AI Model Performance")

    results = pd.DataFrame(bundle["model_results"])

    st.write(
        "Compare the classification algorithms evaluated "
        "during model development."
    )

    st.markdown("### Model comparison")

    st.dataframe(
        results.round(4),
        use_container_width=True,
        hide_index=True
    )

    if "Accuracy" in results.columns:

        fig, ax = plt.subplots(figsize=(11, 5))

        fig.patch.set_facecolor("#17243C")
        ax.set_facecolor("#202F4B")

        chart_colors = [
            "#9B82F2",
            "#27BED0",
            "#E58A69",
            "#45B995",
            "#E7B95D",
            "#E47F9A"
        ]

        ax.bar(
            results["Model"],
            results["Accuracy"],
            color=chart_colors[:len(results)],
            width=0.6
        )

        ax.set_title(
            "Model Accuracy Comparison",
            color="white",
            fontsize=14
        )

        ax.set_ylabel("Accuracy", color="white")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=25, colors="white")
        ax.tick_params(axis="y", colors="white")

        for spine in ax.spines.values():
            spine.set_color("#52627F")

        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("### Selected final model")

    st.success(bundle["model_name"])

    st.markdown("### Evaluation metrics")

    st.json(bundle["metrics"])


# =====================================================
# 10. ABOUT
# =====================================================

elif page == "About":

    st.markdown("### ✦ PLATFORM INFORMATION")
    st.title("About Disease Diagnosis AI")

    st.write(
        "Disease Diagnosis AI is an interactive application "
        "that explores symptom-based classification patterns "
        "using machine learning."
    )

    st.markdown("### Platform features")

    a, b = st.columns(2)

    with a:

        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔎</div>
            <div class="feature-title">Symptom Classification</div>
            <div class="feature-description">
                Explore the output of a trained model
                using selected dataset symptoms.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with b:

        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <div class="feature-title">Data and Evaluation</div>
            <div class="feature-description">
                Understand dataset patterns and compare
                classification algorithms.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Technologies used")

    technologies = [
        "Python",
        "Pandas",
        "NumPy",
        "Scikit-learn",
        "Matplotlib",
        "Streamlit"
    ]

    cols = st.columns(3)

    for i, tech in enumerate(technologies):

        cols[i % 3].markdown(f"""
        <div style="
            background:#293B5D;
            border:1px solid #52658A;
            border-radius:10px;
            padding:13px;
            text-align:center;
            margin-bottom:10px;
            color:white;
            font-weight:600;">
            {tech}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Dataset limitations")

    st.write(
        "The dataset contains repeated records and may not "
        "represent real-world clinical populations. Some "
        "disease categories may have been excluded from "
        "model evaluation because of insufficient unique records."
    )


# =====================================================
# 11. FOOTER
# =====================================================

st.markdown("---")

st.markdown("""
<div class="footer">
    Disease Diagnosis AI
    <br><br>
    Model predictions are not diagnoses and must not guide
    healthcare decisions.
</div>
""", unsafe_allow_html=True)
