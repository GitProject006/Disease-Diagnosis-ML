
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Disease Diagnosis AI",
    page_icon="🩺",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #081526, #102B43);
    color: #F4F8FC;
}

h1, h2, h3 {
    color: #65D6C5 !important;
}

[data-testid="stMetric"] {
    background-color: #17364F;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #28546C;
}

div.stButton > button {
    background-color: #19B8A6;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: bold;
    width: 100%;
    padding: 10px;
}

div.stButton > button:hover {
    background-color: #11998B;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #102438;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_bundle():
    return joblib.load("artifacts/disease_model.pkl")

try:
    bundle = load_bundle()
except Exception as e:
    st.error("Model file not found. Check artifacts/disease_model.pkl")
    st.stop()

model = bundle["model"]
mlb = bundle["mlb"]

diseases = list(bundle["eligible_classes"])
symptoms = list(mlb.classes_)

# ---------------- SIDEBAR ----------------

st.sidebar.title("🩺 Disease Diagnosis AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Disease Prediction",
        "Disease Library",
        "EDA Dashboard",
        "Model Performance",
        "About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Machine Learning Academic Project")
st.sidebar.caption("Educational use only")


# ---------------- HOME ----------------

if page == "Home":

    st.title("🩺 Disease Diagnosis AI")

    st.subheader(
        "Machine Learning Based Symptom Classification"
    )

    st.image(
        "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d",
        use_container_width=True
    )

    st.write(
        "An educational machine learning project that "
        "explores the classification of disease categories "
        "using symptom-based dataset features."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Disease Classes", len(diseases))
    c2.metric("Available Symptoms", len(symptoms))
    c3.metric("ML Algorithms", len(bundle["model_results"]))

    st.info(
        "This application is for academic demonstration only. "
        "It is not a medical diagnostic tool."
    )

    st.markdown("### Project Features")

    a, b, c = st.columns(3)

    a.markdown("### 🔎")
    a.write("Symptom-based prediction")

    b.markdown("### 📊")
    b.write("Dataset analytics")

    c.markdown("### 🤖")
    c.write("ML model comparison")


# ---------------- DISEASE PREDICTION ----------------

elif page == "Disease Prediction":

    st.title("🔎 Symptom Checker")

    st.write(
        "Select symptoms from the dataset to explore "
        "the output of the trained classification model."
    )

    st.warning(
        "Educational demonstration only. Model predictions "
        "are not diagnoses and must not guide healthcare decisions."
    )

    selected_symptoms = st.multiselect(
        "Search and select symptoms",
        options=symptoms,
        placeholder="Choose symptoms..."
    )

    if st.button("Predict Disease Category"):

        if len(selected_symptoms) == 0:
            st.warning("Please select at least one symptom.")

        else:

            input_data = mlb.transform([selected_symptoms])

            prediction = model.predict(input_data)[0]

            st.success("Model prediction completed!")

            st.markdown("## Predicted Category")

            st.markdown(
                f"### 🩺 {prediction}"
            )

            st.write("Selected symptoms:")

            st.write(", ".join(selected_symptoms))

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(input_data)[0]

                classes = model.classes_

                top_indices = np.argsort(probabilities)[::-1][:3]

                top_results = pd.DataFrame({
                    "Disease Category": classes[top_indices],
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
                    "These are model scores, not medical probabilities "
                    "or estimates of the likelihood of a real illness."
                )


# ---------------- DISEASE LIBRARY ----------------

elif page == "Disease Library":

    st.title("📚 Disease Category Library")

    st.write(
        "Browse the disease categories represented in the "
        "dataset used for model training."
    )

    search = st.text_input("Search disease category")

    filtered = [
        d for d in diseases
        if search.lower() in d.lower()
    ]

    for disease in filtered:

        with st.expander("🩺 " + disease):

            st.write("Disease category:", disease)

            st.write(
                "This category is included in the project's "
                "symptom classification dataset."
            )

            st.caption(
                "This dataset association does not establish "
                "clinical symptoms, diagnosis or treatment."
            )


# ---------------- EDA DASHBOARD ----------------

elif page == "EDA Dashboard":

    st.title("📊 Exploratory Data Analysis")

    disease_counts = pd.Series(
        bundle["disease_counts"]
    ).sort_values(ascending=False)

    symptom_counts = pd.Series(
        bundle["symptom_counts"]
    ).sort_values(ascending=False).head(15)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Disease Distribution")

        fig, ax = plt.subplots(figsize=(8, 6))

        disease_counts.plot(
            kind="bar",
            ax=ax,
            color="#19B8A6"
        )

        ax.set_ylabel("Records")
        ax.tick_params(axis="x", rotation=90)

        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.subheader("Most Frequent Symptoms")

        fig, ax = plt.subplots(figsize=(8, 6))

        symptom_counts.sort_values().plot(
            kind="barh",
            ax=ax,
            color="#3A9BD9"
        )

        ax.set_xlabel("Frequency")

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Dataset Summary")

    st.write(
        "Unique disease and symptom records used for "
        "the exploratory analysis."
    )

    st.dataframe(
        pd.DataFrame({
            "Disease": disease_counts.index,
            "Record Count": disease_counts.values
        }),
        use_container_width=True,
        hide_index=True
    )


# ---------------- MODEL PERFORMANCE ----------------

elif page == "Model Performance":

    st.title("🤖 Machine Learning Performance")

    results = pd.DataFrame(bundle["model_results"])

    st.subheader("Comparison of Six Classification Models")

    st.dataframe(
        results.round(4),
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        results["Model"],
        results["Accuracy"],
        color="#19B8A6"
    )

    ax.set_title("Model Accuracy Comparison")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)

    plt.xticks(rotation=25)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Selected Final Model")

    st.success(bundle["model_name"])

    st.write("Evaluation metrics:")

    st.json(bundle["metrics"])


# ---------------- ABOUT PROJECT ----------------

elif page == "About Project":

    st.title("ℹ️ About the Project")

    st.subheader("Problem Statement")

    st.write(
        "To study and develop a supervised machine learning "
        "classification system using symptom-based data "
        "to predict disease category labels."
    )

    st.subheader("Objectives")

    st.markdown("""
    - Explore a symptom-based dataset.
    - Perform data cleaning and preprocessing.
    - Conduct exploratory data analysis.
    - Implement six classification algorithms.
    - Perform hyperparameter tuning.
    - Compare model performance.
    - Develop an interactive academic demonstration.
    """)

    st.subheader("Technology Stack")

    st.write(
        "Python, Pandas, NumPy, Matplotlib, Seaborn, "
        "Scikit-learn and Streamlit."
    )

    st.subheader("Limitations")

    st.write(
        "The dataset contains repeated records and "
        "may not represent real-world clinical populations. "
        "Some disease classes may be excluded from model "
        "evaluation because of insufficient unique records."
    )

    st.info(
        "The application is an educational project and must "
        "not be used for diagnosis, treatment or medical decisions."
    )

# ---------------- FOOTER ----------------

st.markdown("---")

st.caption(
    "Disease Diagnosis AI | Machine Learning Academic Project"
)
