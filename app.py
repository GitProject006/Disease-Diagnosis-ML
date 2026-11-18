
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Disease Diagnosis AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM STYLING
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #081525 0%, #102B43 55%, #0B2034 100%);
    color: #F4F8FC;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #68D9CB !important;
    font-weight: 700 !important;
}

p, label, .stMarkdown {
    color: #D9E5EF;
}

section[data-testid="stSidebar"] {
    background: #0D2236;
    border-right: 1px solid #24445C;
}

section[data-testid="stSidebar"] h1 {
    color: #68D9CB !important;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #C5D5E3;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #173B54, #16334B);
    border: 1px solid #285773;
    padding: 22px;
    border-radius: 17px;
    box-shadow: 0 5px 18px rgba(0,0,0,0.12);
}

[data-testid="stMetricLabel"] {
    color: #B9D0DF !important;
}

[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-weight: 700;
}

div.stButton > button {
    background: linear-gradient(90deg, #16B9A6, #35C9B7);
    color: #062333;
    border: none;
    border-radius: 11px;
    padding: 0.65rem 1.2rem;
    font-weight: 700;
    width: 100%;
    transition: 0.2s ease;
}

div.stButton > button:hover {
    background: #65DCCB;
    color: #062333;
    border: none;
    transform: translateY(-2px);
}

div[data-baseweb="select"] > div {
    background-color: #17364F;
    border-color: #37627C;
    border-radius: 10px;
}

.stTextInput input {
    background-color: #17364F;
    color: white;
    border-radius: 10px;
}

[data-testid="stExpander"] {
    background: #142F46;
    border: 1px solid #2B526A;
    border-radius: 12px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #2B526A;
    border-radius: 12px;
}

hr {
    border-color: #29485E;
}

.hero {
    background: linear-gradient(115deg, #123650, #14536A 65%, #167C80);
    border: 1px solid #337487;
    border-radius: 22px;
    padding: 34px;
    margin-bottom: 25px;
}

.hero-tag {
    color: #75E6D6;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero-title {
    color: #FFFFFF;
    font-size: 38px;
    line-height: 1.2;
    font-weight: 800;
    margin-top: 12px;
}

.hero-description {
    color: #D4E9F0;
    font-size: 16px;
    line-height: 1.7;
    margin-top: 12px;
}

.section-label {
    color: #68D9CB;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.feature-card {
    background: linear-gradient(145deg, #173A53, #122E45);
    border: 1px solid #2A536B;
    border-radius: 17px;
    padding: 22px;
    min-height: 155px;
    margin-bottom: 12px;
}

.feature-icon {
    font-size: 30px;
    margin-bottom: 12px;
}

.feature-title {
    color: #FFFFFF;
    font-size: 18px;
    font-weight: 700;
}

.feature-description {
    color: #B8CCDA;
    font-size: 13px;
    line-height: 1.6;
    margin-top: 8px;
}

.result-card {
    background: linear-gradient(120deg, #124A54, #176B6C);
    border: 1px solid #3B9D94;
    border-radius: 18px;
    padding: 25px;
    margin-top: 15px;
}

.result-label {
    color: #B6EDE3;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.result-value {
    color: white;
    font-size: 29px;
    font-weight: 800;
    margin-top: 8px;
}

.footer {
    text-align: center;
    color: #91AABC;
    font-size: 12px;
    padding: 20px 0 5px 0;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD TRAINED MODEL
# =====================================================

@st.cache_resource
def load_bundle():
    return joblib.load("artifacts/disease_model.pkl")


try:
    bundle = load_bundle()
except Exception as e:
    st.error("Unable to load the trained model.")
    st.code(str(e))
    st.info(
        "Please check that artifacts/disease_model.pkl "
        "exists in your GitHub repository."
    )
    st.stop()

model = bundle["model"]
mlb = bundle["mlb"]

diseases = sorted(list(bundle["eligible_classes"]))
symptoms = sorted(list(mlb.classes_))


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding: 12px 0 22px 0;">
            <div style="font-size:34px;">🩺</div>
            <div style="font-size:23px;font-weight:800;color:#68D9CB;">
                Disease Diagnosis AI
            </div>
            <div style="font-size:12px;color:#91ADBF;margin-top:5px;">
                SMART HEALTH INSIGHTS
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Explore")

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Disease Prediction",
            "Disease Library",
            "EDA Dashboard",
            "Model Performance",
            "About"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="color:#8EA9BC;font-size:12px;line-height:2;">
            <b style="color:#68D9CB;">DISEASE DIAGNOSIS AI</b><br>
            Symptom insights<br>
            Data exploration<br>
            Machine learning
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# HOME PAGE
# =====================================================

if page == "Home":

    left, right = st.columns([1.35, 1], gap="large")

    with left:

        st.markdown(
            """
            <div class="hero">
                <div class="hero-tag">INTELLIGENT HEALTH INSIGHTS</div>
                <div class="hero-title">
                    Explore symptom patterns with AI
                </div>
                <div class="hero-description">
                    Discover symptom combinations, explore
                    health-related dataset patterns and view
                    machine-learning classification results
                    through an interactive experience.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style="display:flex;gap:10px;flex-wrap:wrap;">
                <span style="background:#164B55;padding:8px 13px;
                border-radius:20px;color:#8CE8D8;font-size:12px;">
                ✦ Symptom analysis</span>
                <span style="background:#164B55;padding:8px 13px;
                border-radius:20px;color:#8CE8D8;font-size:12px;">
                ✦ Data insights</span>
                <span style="background:#164B55;padding:8px 13px;
                border-radius:20px;color:#8CE8D8;font-size:12px;">
                ✦ ML predictions</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:

        st.image(
            "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d"
            "?auto=format&fit=crop&w=1000&q=85",
            use_container_width=True
        )

    st.markdown("")

    st.markdown(
        '<div class="section-label">OVERVIEW</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Disease Categories", len(diseases))
    c2.metric("Available Symptoms", len(symptoms))
    c3.metric("ML Algorithms", len(bundle["model_results"]))

    st.markdown("")

    st.markdown("## Explore the platform")

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🔎</div>
                <div class="feature-title">Symptom Analysis</div>
                <div class="feature-description">
                    Select symptoms and explore the
                    classification model's predicted category.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with f2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Health Data Insights</div>
                <div class="feature-description">
                    Explore disease distributions, symptom
                    frequencies and dataset patterns.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with f3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <div class="feature-title">Model Intelligence</div>
                <div class="feature-description">
                    Compare machine-learning algorithms
                    and explore their evaluation metrics.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("## Discover more")

    img1, img2, img3 = st.columns(3)

    with img1:
        st.image(
            "https://images.unsplash.com/photo-1559757175-0eb30cd8c063"
            "?auto=format&fit=crop&w=700&q=80",
            use_container_width=True
        )
        st.markdown("**Health and science**")
        st.caption("Explore information through data.")

    with img2:
        st.image(
            "https://images.unsplash.com/photo-1551288049-bebda4e38f71"
            "?auto=format&fit=crop&w=700&q=80",
            use_container_width=True
        )
        st.markdown("**Data analytics**")
        st.caption("Discover patterns in the dataset.")

    with img3:
        st.image(
            "https://images.unsplash.com/photo-1532938911079-1b06ac7ceec7"
            "?auto=format&fit=crop&w=700&q=80",
            use_container_width=True
        )
        st.markdown("**Digital healthcare**")
        st.caption("Explore technology and health insights.")


# =====================================================
# DISEASE PREDICTION
# =====================================================

elif page == "Disease Prediction":

    st.markdown(
        '<div class="section-label">INTERACTIVE TOOL</div>',
        unsafe_allow_html=True
    )

    st.title("Symptom Analysis")

    st.write(
        "Choose symptoms from the available dataset to "
        "explore the trained machine-learning model's output."
    )

    st.markdown("### Select your symptoms")

    selected_symptoms = st.multiselect(
        "Search and select symptoms",
        options=symptoms,
        placeholder="Start typing to find symptoms...",
        help="Select one or more symptoms from the dataset."
    )

    st.caption(
        f"{len(selected_symptoms)} symptom(s) selected"
    )

    if selected_symptoms:

        st.markdown("**Selected symptoms**")

        for i in range(0, len(selected_symptoms), 3):
            cols = st.columns(3)

            for j, symptom in enumerate(
                selected_symptoms[i:i + 3]
            ):
                cols[j].markdown(
                    f"""
                    <div style="background:#17465A;
                    border:1px solid #347287;
                    border-radius:9px;padding:9px;
                    margin-bottom:7px;color:#DDF8F2;">
                    ✓ {symptom}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    if st.button("Analyze Selected Symptoms", use_container_width=True):

        if not selected_symptoms:

            st.warning("Please select at least one symptom.")

        else:

            input_data = mlb.transform([selected_symptoms])

            prediction = model.predict(input_data)[0]

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        MODEL PREDICTION
                    </div>
                    <div class="result-value">
                        {prediction}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### Selected symptom summary")

            st.write(", ".join(selected_symptoms))

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

                st.markdown("### Top model scores")

                st.dataframe(
                    top_results,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "Model scores are not medical probabilities "
                    "or estimates of the likelihood of an illness."
                )


# =====================================================
# DISEASE LIBRARY
# =====================================================

elif page == "Disease Library":

    st.markdown(
        '<div class="section-label">EXPLORE CATEGORIES</div>',
        unsafe_allow_html=True
    )

    st.title("Disease Library")

    st.write(
        "Browse the disease categories represented in "
        "the classification dataset."
    )

    search = st.text_input(
        "Search categories",
        placeholder="Type a disease category..."
    )

    filtered = [
        d for d in diseases
        if search.lower() in d.lower()
    ]

    st.markdown(
        f"**{len(filtered)} categories found**"
    )

    if filtered:

        cols = st.columns(2)

        for index, disease in enumerate(filtered):

            with cols[index % 2]:

                with st.container(border=True):

                    st.markdown("### 🩺 " + disease)

                    st.write(
                        "This category is represented in "
                        "the symptom classification dataset."
                    )

                    st.caption(
                        "Dataset category information only."
                    )

    else:

        st.info("No matching category found.")


# =====================================================
# EDA DASHBOARD
# =====================================================

elif page == "EDA Dashboard":

    st.markdown(
        '<div class="section-label">DATA EXPLORATION</div>',
        unsafe_allow_html=True
    )

    st.title("Exploratory Data Analysis")

    st.write(
        "Visualize disease distribution and symptom "
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

        fig.patch.set_facecolor("#122B40")
        ax.set_facecolor("#122B40")

        disease_counts.plot(
            kind="bar",
            ax=ax,
            color="#31C6B4"
        )

        ax.set_ylabel("Number of Records", color="white")
        ax.tick_params(axis="x", rotation=90, colors="white")
        ax.tick_params(axis="y", colors="white")
        ax.spines["bottom"].set_color("#557287")
        ax.spines["left"].set_color("#557287")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with c2:

        st.markdown("### Most Frequent Symptoms")

        fig, ax = plt.subplots(figsize=(9, 6))

        fig.patch.set_facecolor("#122B40")
        ax.set_facecolor("#122B40")

        symptom_counts.sort_values().plot(
            kind="barh",
            ax=ax,
            color="#4AABD7"
        )

        ax.set_xlabel("Frequency", color="white")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#557287")
        ax.spines["left"].set_color("#557287")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

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
# MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="section-label">MODEL EVALUATION</div>',
        unsafe_allow_html=True
    )

    st.title("Machine Learning Performance")

    results = pd.DataFrame(bundle["model_results"])

    st.write(
        "Compare the performance of the classification "
        "algorithms evaluated during model development."
    )

    st.markdown("### Model comparison")

    st.dataframe(
        results.round(4),
        use_container_width=True,
        hide_index=True
    )

    if "Accuracy" in results.columns:

        fig, ax = plt.subplots(figsize=(11, 5))

        fig.patch.set_facecolor("#122B40")
        ax.set_facecolor("#122B40")

        ax.bar(
            results["Model"],
            results["Accuracy"],
            color="#25BFAE",
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

        ax.spines["bottom"].set_color("#557287")
        ax.spines["left"].set_color("#557287")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("### Selected final model")

    st.success(bundle["model_name"])

    st.markdown("### Evaluation metrics")

    st.json(bundle["metrics"])


# =====================================================
# ABOUT
# =====================================================

elif page == "About":

    st.markdown(
        '<div class="section-label">PLATFORM INFORMATION</div>',
        unsafe_allow_html=True
    )

    st.title("About Disease Diagnosis AI")

    st.write(
        "Disease Diagnosis AI is an interactive application "
        "that uses machine learning to explore symptom-based "
        "classification patterns."
    )

    st.markdown("### What this platform offers")

    a, b = st.columns(2)

    with a:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🔎</div>
                <div class="feature-title">Symptom Classification</div>
                <div class="feature-description">
                    Explore the output of a trained model
                    using selected dataset symptoms.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with b:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📈</div>
                <div class="feature-title">Data and Evaluation</div>
                <div class="feature-description">
                    Understand dataset patterns and
                    compare classification algorithms.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Technologies used")

    techs = [
        "Python", "Pandas", "NumPy", "Scikit-learn",
        "Matplotlib", "Seaborn", "Streamlit"
    ]

    cols = st.columns(4)

    for i, tech in enumerate(techs):
        cols[i % 4].markdown(
            f"""
            <div style="background:#173B54;border:1px solid #2B586F;
            border-radius:10px;padding:13px;text-align:center;
            margin-bottom:10px;color:#DDF8F2;font-weight:600;">
            {tech}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Dataset limitations")

    st.write(
        "The dataset contains repeated records and may not "
        "represent real-world clinical populations. Some "
        "disease categories may have been excluded from "
        "model evaluation because of insufficient unique records."
    )


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        Disease Diagnosis AI
        <br><br>
        Model predictions are not diagnoses and must not guide
        healthcare decisions.
    </div>
    """,
    unsafe_allow_html=True
)
