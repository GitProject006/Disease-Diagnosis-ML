import base64
import os

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------------------
st.set_page_config(
    page_title="MediPredict | Disease Diagnosis AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# IMAGES
# Put your own photos in an "assets" folder (hero.jpg, doctor.jpg, lab.jpg)
# and they will be used automatically. Otherwise the web URLs below are used,
# and if those fail the gradient background still looks good.
# ------------------------------------------------------------------
IMAGE_URLS = {
    "hero": "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=1200&q=80",
    "doctor": "https://images.unsplash.com/photo-1559757148-5c350d0d3c56?auto=format&fit=crop&w=900&q=80",
    "lab": "https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?auto=format&fit=crop&w=900&q=80",
}


def image_src(key):
    """Use a local file from assets/ if present, else the web URL."""
    for ext in ("jpg", "jpeg", "png", "webp"):
        path = f"assets/{key}.{ext}"
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            mime = "jpeg" if ext in ("jpg", "jpeg") else ext
            return f"data:image/{mime};base64,{data}"
    return IMAGE_URLS[key]


HERO_IMG, DOCTOR_IMG, LAB_IMG = image_src("hero"), image_src("doctor"), image_src("lab")

# ------------------------------------------------------------------
# STYLE
# ------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.stApp { background: linear-gradient(180deg, #f2faf8 0%, #eef4fb 100%); color: #1b2733; }
.block-container { max-width: 1280px; padding-top: 1.4rem; padding-bottom: 2rem; }
header[data-testid="stHeader"] { background: transparent; }
h1, h2, h3, h4 { color: #0b2e3f !important; letter-spacing: -0.4px; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(190deg, #06353f 0%, #0a5560 55%, #0f6e56 100%);
}
section[data-testid="stSidebar"] * { color: #e7fbf6 !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.18); }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(255,255,255,.07); border-radius: 12px; padding: 9px 14px;
    margin-bottom: 7px; transition: all .25s ease; width: 100%;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,.18); transform: translateX(4px);
}
.brand { display:flex; align-items:center; gap:12px; padding: 8px 0 18px 0; }
.brand-logo {
    width: 46px; height: 46px; border-radius: 14px; display:flex; align-items:center;
    justify-content:center; font-size: 24px;
    background: linear-gradient(135deg, #34d3b0, #1a9bb0);
    box-shadow: 0 6px 18px rgba(0,0,0,.25);
}
.brand-name { font-size: 20px; font-weight: 700; line-height: 1.1; }
.brand-sub { font-size: 11px; letter-spacing: 1.6px; opacity: .75; }

/* ---------- hero ---------- */
.hero {
    position: relative; overflow: hidden; border-radius: 26px; min-height: 360px;
    display: grid; grid-template-columns: 1.15fr .85fr; align-items: center;
    background: linear-gradient(120deg, #06353f 0%, #0f6e56 60%, #1aa6a0 100%);
    box-shadow: 0 22px 50px rgba(6,53,63,.28); margin-bottom: 26px;
}
.hero-text { padding: 44px 46px; z-index: 2; }
.pill {
    display:inline-block; background: rgba(255,255,255,.16); color:#d6fff3;
    border: 1px solid rgba(255,255,255,.28); padding: 6px 14px; border-radius: 30px;
    font-size: 12px; font-weight: 600; letter-spacing: 1px;
}
.hero h1 { color:#fff !important; font-size: 42px; font-weight: 800; line-height: 1.15; margin: 18px 0 14px 0; }
.hero h1 span { color: #7ff0d0; }
.hero p { color:#d3efe9; font-size: 15px; line-height: 1.8; max-width: 520px; margin: 0; }
.hero-photo {
    height: 100%; min-height: 360px; background-size: cover; background-position: center;
    clip-path: polygon(14% 0, 100% 0, 100% 100%, 0 100%);
    position: relative;
}
.hero-photo::after {
    content:""; position:absolute; inset:0;
    background: linear-gradient(90deg, rgba(6,53,63,.55), rgba(6,53,63,0) 60%);
}
.float-badge {
    position:absolute; z-index:3; background:#fff; border-radius:16px; padding: 11px 16px;
    box-shadow: 0 12px 30px rgba(0,0,0,.25); font-size: 12px; color:#496070;
    animation: floaty 4s ease-in-out infinite;
}
.float-badge b { display:block; color:#0b2e3f; font-size: 17px; }
.fb1 { right: 34px; top: 38px; }
.fb2 { right: 150px; bottom: 34px; animation-delay: 1.4s; }
@keyframes floaty { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-9px); } }

.pulse-line { margin-top: 26px; width: 260px; height: 46px; }
.pulse-line path {
    stroke: #7ff0d0; stroke-width: 3; fill: none; stroke-linecap: round; stroke-linejoin: round;
    stroke-dasharray: 420; stroke-dashoffset: 420; animation: draw 3s linear infinite;
}
@keyframes draw { to { stroke-dashoffset: 0; } }

/* ---------- cards ---------- */
.stat {
    background:#fff; border-radius: 20px; padding: 20px 22px; border: 1px solid #e2ecf0;
    box-shadow: 0 8px 22px rgba(16,60,80,.06); transition: all .25s ease; position: relative; overflow: hidden;
}
.stat:hover { transform: translateY(-5px); box-shadow: 0 16px 32px rgba(16,60,80,.12); }
.stat .ico { font-size: 26px; }
.stat .num { font-size: 30px; font-weight: 800; color:#0b2e3f; margin-top: 6px; }
.stat .lbl { font-size: 12px; color:#6a7f8c; font-weight: 500; }
.stat::before { content:""; position:absolute; left:0; top:0; width:100%; height:4px; background: var(--c); }

.step {
    background:#fff; border-radius: 20px; overflow:hidden; border:1px solid #e2ecf0; height:100%;
    box-shadow: 0 8px 22px rgba(16,60,80,.06); transition: all .25s ease;
}
.step:hover { transform: translateY(-6px); box-shadow: 0 18px 36px rgba(16,60,80,.14); }
.step .img { height: 130px; background-size: cover; background-position: center; background-color:#cfeee6; }
.step .body { padding: 18px 20px 20px 20px; }
.step .no { color:#0f8a6e; font-size: 12px; font-weight: 700; letter-spacing: 1.5px; }
.step h4 { margin: 4px 0 6px 0; font-size: 17px; }
.step p { color:#62768a; font-size: 13px; line-height: 1.7; margin:0; }

.banner {
    border-radius: 22px; padding: 28px 32px; color:#fff; margin-bottom: 20px;
    background: linear-gradient(100deg, rgba(6,53,63,.92), rgba(15,110,86,.82)), var(--img);
    background-size: cover; background-position: center;
    box-shadow: 0 16px 36px rgba(6,53,63,.22);
}
.banner h1 { color:#fff !important; margin:0 0 6px 0; font-size: 32px; }
.banner p { color:#d3efe9; margin:0; font-size: 14px; }

.panel {
    background:#fff; border:1px solid #e2ecf0; border-radius: 20px; padding: 22px 24px;
    box-shadow: 0 8px 22px rgba(16,60,80,.06);
}
.chip {
    display:inline-block; background: linear-gradient(135deg, #d9f7ee, #d3eefb); color:#0a4d57;
    font-size: 12.5px; font-weight: 600; padding: 6px 14px; border-radius: 20px; margin: 4px 6px 4px 0;
    border: 1px solid #bfe6dc;
}

/* ---------- result ---------- */
.result {
    border-radius: 22px; padding: 26px 30px; color:#fff; display:flex; align-items:center; gap: 28px;
    background: linear-gradient(120deg, #0a5560, #0f8a6e); box-shadow: 0 18px 38px rgba(15,110,86,.3);
}
.result small { letter-spacing: 2px; font-weight: 600; color:#b8fbe6; }
.result .name { font-size: 36px; font-weight: 800; line-height: 1.15; margin-top: 4px; }
.result .sub { font-size: 13px; color:#c9f3e8; margin-top: 6px; }
.ring-wrap { position: relative; width: 110px; height: 110px; flex-shrink: 0; }
.ring-wrap .txt {
    position:absolute; inset:0; display:flex; flex-direction:column; align-items:center;
    justify-content:center; font-size: 22px; font-weight: 800;
}
.ring-wrap .txt span { font-size: 10px; font-weight: 500; opacity: .85; }

.bar-row { display:grid; grid-template-columns: 150px 1fr 56px; align-items:center; gap: 14px; margin: 11px 0; }
.bar-name { font-size: 14px; font-weight: 600; color:#0b2e3f; }
.bar-track { background:#e8f1f4; height: 12px; border-radius: 10px; overflow:hidden; }
.bar-fill { height: 100%; border-radius: 10px; animation: grow 1s ease-out; }
@keyframes grow { from { width: 0 !important; } }
.bar-val { font-size: 13px; font-weight: 700; color:#37566a; text-align:right; }

.disclaimer {
    background:#fff7e6; border:1px solid #f2d9a3; color:#7a5200; border-radius: 14px;
    padding: 12px 16px; font-size: 13px; margin-top: 18px;
}
.footer { text-align:center; color:#7d92a1; font-size: 12px; padding: 28px 0 4px 0; }

/* ---------- widgets ---------- */
div.stButton > button {
    background: linear-gradient(100deg, #0f8a6e, #1a9bb0); color:#fff; border:0; border-radius: 14px;
    padding: .7rem 1.2rem; font-weight: 600; font-size: 15px; transition: all .2s ease;
    box-shadow: 0 8px 20px rgba(15,138,110,.3);
}
div.stButton > button:hover { transform: translateY(-2px); filter: brightness(1.08); color:#fff; }
div[data-baseweb="select"] > div { border-radius: 14px; border-color: #cfe3e8; background:#fff; }
[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; }
@media (max-width: 900px) { .hero { grid-template-columns: 1fr; } .hero-photo { display:none; } }
</style>
""",
    unsafe_allow_html=True,
)

DISCLAIMER = (
    "Educational project only. Predictions come from a machine-learning model and are "
    "not a medical diagnosis. Please consult a qualified doctor."
)
PALETTE = ["#0f8a6e", "#1a9bb0", "#7c6cf0", "#e9a23b", "#e5677f"]


# ------------------------------------------------------------------
# LOAD MODEL BUNDLE
# ------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    return joblib.load("artifacts/disease_model.pkl")


def as_series(value):
    if isinstance(value, pd.Series):
        return value
    if isinstance(value, dict):
        return pd.Series(value, dtype="float64")
    return pd.Series(dtype="float64")


def pretty(text):
    return str(text).replace("_", " ").strip().title()


try:
    bundle = load_bundle()
    model = bundle["model"]
    mlb = bundle["mlb"]
    symptoms = sorted(str(s) for s in mlb.classes_)
    diseases = sorted(str(d) for d in bundle["eligible_classes"])
except Exception as exc:
    st.error("Could not load artifacts/disease_model.pkl. Check that the file is in your GitHub repo.")
    st.code(str(exc))
    st.stop()

disease_counts = as_series(bundle.get("disease_counts", {}))
symptom_counts = as_series(bundle.get("symptom_counts", {}))
results_df = pd.DataFrame(bundle.get("model_results", []))
metrics = bundle.get("metrics", {})
best_params = bundle.get("best_params", {})


def bar_chart(series, color, title, top=15):
    df = series.sort_values(ascending=False).head(top).rename("value").rename_axis("name").reset_index()
    df["name"] = df["name"].map(pretty)
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=8, color=color)
        .encode(
            x=alt.X("value:Q", title=None),
            y=alt.Y("name:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
            tooltip=["name", "value"],
        )
        .properties(height=30 * len(df) + 10, title=title)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False, labelFontSize=12)
        .configure_title(fontSize=16, anchor="start", color="#0b2e3f")
    )
    st.altair_chart(chart, use_container_width=True)


# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
          <div class="brand-logo">🩺</div>
          <div><div class="brand-name">MediPredict</div>
          <div class="brand-sub">AI DISEASE DIAGNOSIS</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigate",
        ["🏠  Home", "🔍  Diagnose", "📊  Data explorer", "🤖  Model comparison", "ℹ️  About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(f"🧬 {len(diseases)} diseases   ·   🩸 {len(symptoms)} symptoms")
    st.caption("Built with Python, Scikit-learn and Streamlit")

page = page.split("  ", 1)[1]

# ------------------------------------------------------------------
# HOME
# ------------------------------------------------------------------
if page == "Home":
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-text">
            <span class="pill">✦ AI POWERED HEALTH INSIGHTS</span>
            <h1>Understand your symptoms with <span>machine learning</span></h1>
            <p>Select the symptoms you feel and our trained classifier ranks the most
            likely diseases in seconds, with a confidence score for each prediction.</p>
            <svg class="pulse-line" viewBox="0 0 260 46">
              <path d="M0 24 H70 L82 6 L98 40 L112 14 L124 24 H260"/>
            </svg>
          </div>
          <div class="hero-photo" style="background-image:url('{HERO_IMG}')">
            <div class="float-badge fb1">Diseases covered<b>{len(diseases)}+</b></div>
            <div class="float-badge fb2">Symptoms known<b>{len(symptoms)}+</b></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    acc = metrics.get("Accuracy") or metrics.get("accuracy")
    stats = [
        ("🧬", len(diseases), "Diseases covered", "#0f8a6e"),
        ("🩸", len(symptoms), "Symptoms available", "#1a9bb0"),
        ("🤖", len(results_df) if not results_df.empty else "-", "Models compared", "#7c6cf0"),
        ("🎯", f"{float(acc) * 100:.1f}%" if acc is not None else "-", "Final accuracy", "#e9a23b"),
    ]
    for col, (ico, num, lbl, c) in zip(st.columns(4), stats):
        col.markdown(
            f'<div class="stat" style="--c:{c}"><div class="ico">{ico}</div>'
            f'<div class="num">{num}</div><div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### How it works")
    steps = [
        ("STEP 01", "Select symptoms", "Search and pick every symptom you are experiencing.", DOCTOR_IMG),
        ("STEP 02", "Model analyses", "The tuned classifier scores each disease from the pattern.", LAB_IMG),
        ("STEP 03", "Get ranked results", "See the top predictions with confidence bars.", HERO_IMG),
    ]
    for col, (no, t, d, img) in zip(st.columns(3, gap="large"), steps):
        col.markdown(
            f'<div class="step"><div class="img" style="background-image:url(\'{img}\')"></div>'
            f'<div class="body"><div class="no">{no}</div><h4>{t}</h4><p>{d}</p></div></div>',
            unsafe_allow_html=True,
        )
    st.markdown(f'<div class="disclaimer">⚠️ {DISCLAIMER}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# DIAGNOSE
# ------------------------------------------------------------------
elif page == "Diagnose":
    st.markdown(
        f"""
        <div class="banner" style="--img:url('{DOCTOR_IMG}')">
          <h1>Symptom checker</h1>
          <p>Pick every symptom you have. More symptoms usually means a better prediction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "sel" not in st.session_state:
        st.session_state["sel"] = []

    def add_symptom(sym):
        if sym not in st.session_state["sel"]:
            st.session_state["sel"] = st.session_state["sel"] + [sym]

    def clear_all():
        st.session_state["sel"] = []

    if not symptom_counts.empty:
        quick = [str(s) for s in symptom_counts.sort_values(ascending=False).index[:6] if str(s) in symptoms]
        if quick:
            st.caption("Quick add common symptoms")
            for col, s in zip(st.columns(len(quick)), quick):
                col.button(pretty(s), key=f"q_{s}", on_click=add_symptom, args=(s,), use_container_width=True)

    selected = st.multiselect(
        "Symptoms",
        options=symptoms,
        key="sel",
        format_func=pretty,
        placeholder="Type to search, e.g. fever, cough, headache",
    )

    if selected:
        st.markdown(
            "".join(f'<span class="chip">{pretty(s)}</span>' for s in selected), unsafe_allow_html=True
        )

    b1, b2 = st.columns([4, 1])
    run = b1.button("🔬  Predict condition", use_container_width=True)
    b2.button("Clear", on_click=clear_all, use_container_width=True)

    if run:
        if not selected:
            st.warning("Select at least one symptom first.")
        else:
            try:
                x = mlb.transform([selected])
                pred = model.predict(x)[0]
                has_proba = hasattr(model, "predict_proba")
                probs = model.predict_proba(x)[0] if has_proba else None
                conf = float(probs.max()) * 100 if has_proba else None

                ring = ""
                if conf is not None:
                    dash = conf / 100 * 263.9
                    ring = (
                        '<div class="ring-wrap"><svg width="110" height="110" viewBox="0 0 110 110">'
                        '<circle cx="55" cy="55" r="42" stroke="rgba(255,255,255,.2)" stroke-width="10" fill="none"/>'
                        f'<circle cx="55" cy="55" r="42" stroke="#7ff0d0" stroke-width="10" fill="none" '
                        f'stroke-linecap="round" stroke-dasharray="{dash:.1f} 263.9" transform="rotate(-90 55 55)"/>'
                        f'</svg><div class="txt">{conf:.0f}%<span>confidence</span></div></div>'
                    )
                st.markdown(
                    f'<div class="result">{ring}<div><small>MOST LIKELY CONDITION</small>'
                    f'<div class="name">{pretty(pred)}</div>'
                    f'<div class="sub">Based on {len(selected)} selected symptom(s)</div></div></div>',
                    unsafe_allow_html=True,
                )

                if has_proba:
                    top = np.argsort(probs)[::-1][:5]
                    rows = ""
                    for rank, i in enumerate(top):
                        pct = float(probs[i]) * 100
                        rows += (
                            f'<div class="bar-row"><div class="bar-name">{pretty(model.classes_[i])}</div>'
                            f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;'
                            f'background:{PALETTE[rank % len(PALETTE)]}"></div></div>'
                            f'<div class="bar-val">{pct:.1f}%</div></div>'
                        )
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f'<div class="panel"><h4>Top 5 predictions</h4>{rows}</div>', unsafe_allow_html=True)

                    report = pd.DataFrame(
                        {
                            "Condition": [pretty(model.classes_[i]) for i in top],
                            "Confidence %": [round(float(probs[i]) * 100, 2) for i in top],
                        }
                    )
                    st.download_button(
                        "⬇️  Download result (CSV)",
                        report.to_csv(index=False).encode(),
                        file_name="prediction_result.csv",
                        mime="text/csv",
                    )
                st.markdown(f'<div class="disclaimer">⚠️ {DISCLAIMER}</div>', unsafe_allow_html=True)
            except Exception as exc:
                st.error("The model could not process this selection.")
                st.code(str(exc))

# ------------------------------------------------------------------
# DATA EXPLORER
# ------------------------------------------------------------------
elif page == "Data explorer":
    st.markdown(
        f"""
        <div class="banner" style="--img:url('{LAB_IMG}')">
          <h1>Exploratory data analysis</h1>
          <p>Understand the dataset before modelling: class balance and symptom frequency.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if disease_counts.empty and symptom_counts.empty:
        st.warning("Dataset counts were not saved in the model bundle.")
    else:
        a, b = st.columns(2, gap="large")
        with a:
            if not disease_counts.empty:
                bar_chart(disease_counts, "#0f8a6e", "Records per disease (top 15)")
        with b:
            if not symptom_counts.empty:
                bar_chart(symptom_counts, "#7c6cf0", "Most frequent symptoms (top 15)")

        if not disease_counts.empty:
            st.markdown("### Search a disease")
            q = st.text_input("Disease name", placeholder="Type part of a disease name")
            table = disease_counts.rename("Records").rename_axis("Disease").reset_index()
            if q:
                table = table[table["Disease"].astype(str).str.contains(q, case=False)]
            st.dataframe(table, hide_index=True, use_container_width=True)

# ------------------------------------------------------------------
# MODEL COMPARISON
# ------------------------------------------------------------------
elif page == "Model comparison":
    st.markdown(
        f"""
        <div class="banner" style="--img:url('{HERO_IMG}')">
          <h1>Model comparison</h1>
          <p>All classifiers tried, with hyper-parameter tuning, and the final selected model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if results_df.empty:
        st.warning("Model comparison results were not found in the bundle.")
    else:
        num_cols = results_df.select_dtypes("number").columns.tolist()
        st.dataframe(
            results_df.style.format({c: "{:.4f}" for c in num_cols}).highlight_max(
                subset=num_cols, color="#c9efe2"
            ),
            hide_index=True,
            use_container_width=True,
        )
        if "Model" in results_df.columns and num_cols:
            metric = st.selectbox("Chart metric", num_cols, index=0)
            best = results_df[metric].max()
            chart = (
                alt.Chart(results_df)
                .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
                .encode(
                    x=alt.X("Model:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-20)),
                    y=alt.Y(f"{metric}:Q", scale=alt.Scale(domain=[0, 1.05])),
                    color=alt.condition(
                        alt.datum[metric] == best, alt.value("#0f8a6e"), alt.value("#a9c9d6")
                    ),
                    tooltip=["Model", metric],
                )
                .properties(height=340)
                .configure_view(strokeWidth=0)
                .configure_axis(grid=False)
            )
            st.altair_chart(chart, use_container_width=True)

    st.success(f"🏆 Selected final model: {bundle.get('model_name', 'Not specified')}")

    if best_params:
        st.markdown("#### Best hyper-parameters")
        st.json({str(k): (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in best_params.items()})

    if metrics:
        st.markdown("#### Final test metrics")
        cols = st.columns(min(4, len(metrics)))
        for i, (k, v) in enumerate(metrics.items()):
            val = f"{float(v):.3f}" if isinstance(v, (int, float, np.number)) else str(v)
            cols[i % len(cols)].markdown(
                f'<div class="stat" style="--c:{PALETTE[i % 5]}"><div class="lbl">{k}</div>'
                f'<div class="num">{val}</div></div>',
                unsafe_allow_html=True,
            )

# ------------------------------------------------------------------
# ABOUT
# ------------------------------------------------------------------
else:
    st.markdown(
        f"""
        <div class="banner" style="--img:url('{DOCTOR_IMG}')">
          <h1>About the project</h1>
          <p>Disease Diagnosis: a multi-class classification project built for the final lab evaluation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    blocks = [
        ("🎯", "Problem", "Predict the disease from a set of reported symptoms."),
        ("⚙️", "Pipeline", "Cleaning, EDA, encoding, model training, tuning and evaluation."),
        ("🧰", "Tech stack", "Python, Pandas, NumPy, Scikit-learn, Altair and Streamlit."),
    ]
    for col, (ico, t, d) in zip(st.columns(3, gap="large"), blocks):
        col.markdown(
            f'<div class="panel"><div style="font-size:28px">{ico}</div><h4>{t}</h4>'
            f'<p style="color:#62768a;font-size:13px;line-height:1.7">{d}</p></div>',
            unsafe_allow_html=True,
        )
    st.markdown("### Limitations")
    st.write(
        "The dataset can contain repeated records and may not represent real patients. "
        "The model only predicts diseases present in its training data."
    )
    st.markdown(f'<div class="disclaimer">⚠️ {DISCLAIMER}</div>', unsafe_allow_html=True)

st.markdown(f'<div class="footer">MediPredict · {DISCLAIMER}</div>', unsafe_allow_html=True)
