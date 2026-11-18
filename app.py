import base64
import os

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="MediPredict | Symptom Checker",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# IMAGES
# Your own photos: create an "assets" folder in the repo and add
# hero.jpg, doctor.jpg, lab.jpg. They are used automatically.
# Otherwise the web photos below are used (illustrations always show).
# ------------------------------------------------------------------
IMAGE_URLS = {
    "hero": "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=1200&q=80",
    "doctor": "https://images.unsplash.com/photo-1559757148-5c350d0d3c56?auto=format&fit=crop&w=900&q=80",
    "lab": "https://images.unsplash.com/photo-1532187863486-abf9dbad1b69?auto=format&fit=crop&w=900&q=80",
}


def image_src(key):
    for ext in ("jpg", "jpeg", "png", "webp"):
        path = f"assets/{key}.{ext}"
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            return f"data:image/{'jpeg' if ext in ('jpg', 'jpeg') else ext};base64,{data}"
    return IMAGE_URLS[key]


HERO_IMG, DOCTOR_IMG, LAB_IMG = image_src("hero"), image_src("doctor"), image_src("lab")

# Illustrations drawn in SVG so they always load
SVG_SELECT = """<svg viewBox="0 0 160 110" xmlns="http://www.w3.org/2000/svg"><rect width="160" height="110" fill="#e3f6f0"/>
<circle cx="30" cy="85" r="26" fill="#cdeee3"/><circle cx="138" cy="20" r="22" fill="#d3eefb"/>
<rect x="50" y="14" width="60" height="82" rx="9" fill="#fff" stroke="#0f8a6e" stroke-width="3"/>
<rect x="68" y="8" width="24" height="12" rx="5" fill="#0f8a6e"/>
<path d="M60 40l4 4 8-9M60 58l4 4 8-9M60 76l4 4 8-9" stroke="#0f8a6e" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
<rect x="80" y="38" width="22" height="5" rx="2.5" fill="#bfe6dc"/><rect x="80" y="56" width="22" height="5" rx="2.5" fill="#bfe6dc"/><rect x="80" y="74" width="16" height="5" rx="2.5" fill="#bfe6dc"/></svg>"""

SVG_ANALYSE = """<svg viewBox="0 0 160 110" xmlns="http://www.w3.org/2000/svg"><rect width="160" height="110" fill="#e6effb"/>
<circle cx="132" cy="90" r="24" fill="#d6e4f8"/><circle cx="24" cy="24" r="18" fill="#d9f2f6"/>
<circle cx="74" cy="52" r="30" fill="#fff" stroke="#1a9bb0" stroke-width="5"/>
<path d="M96 74l24 24" stroke="#1a9bb0" stroke-width="9" stroke-linecap="round"/>
<path d="M52 54h12l5-14 8 28 6-14h10" stroke="#e5677f" stroke-width="3.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>"""

SVG_RESULT = """<svg viewBox="0 0 160 110" xmlns="http://www.w3.org/2000/svg"><rect width="160" height="110" fill="#efeafd"/>
<circle cx="24" cy="90" r="22" fill="#e2dafa"/><circle cx="136" cy="22" r="20" fill="#d9f2f6"/>
<rect x="34" y="22" width="92" height="68" rx="10" fill="#fff" stroke="#7c6cf0" stroke-width="3"/>
<rect x="46" y="62" width="14" height="20" rx="3" fill="#bdb3f7"/><rect x="68" y="46" width="14" height="36" rx="3" fill="#7c6cf0"/><rect x="90" y="54" width="14" height="28" rx="3" fill="#bdb3f7"/>
<circle cx="112" cy="36" r="11" fill="#0f8a6e"/><path d="M107 36l4 4 7-8" stroke="#fff" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>"""

SVG_HERO = """<svg viewBox="0 0 400 360" xmlns="http://www.w3.org/2000/svg" style="position:absolute;inset:0;width:100%;height:100%">
<circle cx="260" cy="170" r="130" fill="rgba(255,255,255,.08)"/><circle cx="260" cy="170" r="90" fill="rgba(255,255,255,.08)"/>
<path d="M260 235C190 190 170 150 195 120c18-22 50-14 65 8 15-22 47-30 65-8 25 30 5 70-65 115z" fill="#7ff0d0" opacity=".9"/>
<path d="M150 175h60l12-36 20 72 16-36h70" stroke="#06353f" stroke-width="7" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>"""

# ------------------------------------------------------------------
# STYLE
# ------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.stApp { background: linear-gradient(180deg, #f2faf8 0%, #eef4fb 100%); color: #1b2733; }
.block-container { max-width: 1240px; padding-top: 1.4rem; padding-bottom: 2rem; }
header[data-testid="stHeader"] { background: transparent; }
h1, h2, h3, h4 { color: #0b2e3f !important; letter-spacing: -0.4px; }

section[data-testid="stSidebar"] { background: linear-gradient(190deg, #06353f 0%, #0a5560 55%, #0f6e56 100%); }
section[data-testid="stSidebar"] * { color: #e7fbf6 !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.18); }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(255,255,255,.07); border-radius: 12px; padding: 9px 14px; margin-bottom: 7px;
    transition: all .25s ease; width: 100%;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: rgba(255,255,255,.18); transform: translateX(4px); }
.brand { display:flex; align-items:center; gap:12px; padding: 8px 0 18px 0; }
.brand-logo { width:46px; height:46px; border-radius:14px; display:flex; align-items:center; justify-content:center;
    font-size:24px; background: linear-gradient(135deg, #34d3b0, #1a9bb0); box-shadow: 0 6px 18px rgba(0,0,0,.25); }
.brand-name { font-size: 20px; font-weight: 700; line-height: 1.1; }
.brand-sub { font-size: 11px; letter-spacing: 1.6px; opacity: .75; }

.hero { position: relative; overflow: hidden; border-radius: 26px; min-height: 380px; display: grid;
    grid-template-columns: 1.1fr .9fr; align-items: center;
    background: linear-gradient(120deg, #06353f 0%, #0f6e56 60%, #1aa6a0 100%);
    box-shadow: 0 22px 50px rgba(6,53,63,.28); margin-bottom: 18px; }
.hero-text { padding: 46px 48px; z-index: 2; }
.pill { display:inline-block; background: rgba(255,255,255,.16); color:#d6fff3; border:1px solid rgba(255,255,255,.28);
    padding: 6px 14px; border-radius: 30px; font-size: 12px; font-weight: 600; letter-spacing: 1px; }
.hero h1 { color:#fff !important; font-size: 44px; font-weight: 800; line-height: 1.12; margin: 18px 0 14px 0; }
.hero h1 span { color:#7ff0d0; }
.hero p { color:#d3efe9; font-size: 15.5px; line-height: 1.8; max-width: 500px; margin:0; }
.hero-art { position: relative; height: 100%; min-height: 380px; }
.hero-photo { position:absolute; inset:0; z-index:2; background-size: cover; background-position: center;
    clip-path: polygon(14% 0, 100% 0, 100% 100%, 0 100%); }
.hero-photo::after { content:""; position:absolute; inset:0; background: linear-gradient(90deg, rgba(6,53,63,.55), rgba(6,53,63,0) 60%); }
.float-badge { position:absolute; z-index:3; background:#fff; border-radius:16px; padding: 11px 16px;
    box-shadow: 0 12px 30px rgba(0,0,0,.25); font-size: 12px; color:#496070; animation: floaty 4s ease-in-out infinite; }
.float-badge b { display:block; color:#0b2e3f; font-size: 15px; }
.fb1 { right: 30px; top: 36px; } .fb2 { right: 140px; bottom: 34px; animation-delay: 1.4s; }
@keyframes floaty { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-9px); } }

.section-title { font-size: 26px; font-weight: 700; margin: 26px 0 4px 0; color:#0b2e3f; }
.section-sub { color:#62768a; font-size: 14px; margin-bottom: 14px; }

.step { background:#fff; border-radius: 20px; overflow:hidden; border:1px solid #e2ecf0; height:100%;
    box-shadow: 0 8px 22px rgba(16,60,80,.06); transition: all .25s ease; }
.step:hover { transform: translateY(-6px); box-shadow: 0 18px 36px rgba(16,60,80,.14); }
.step svg { display:block; width:100%; height:auto; }
.step .body { padding: 16px 20px 20px 20px; }
.step .no { color:#0f8a6e; font-size: 12px; font-weight: 700; letter-spacing: 1.5px; }
.step h4 { margin: 4px 0 6px 0; font-size: 17px; }
.step p { color:#62768a; font-size: 13px; line-height: 1.7; margin:0; }

.photo-strip { display:grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 8px; }
.photo-card { position:relative; border-radius: 22px; overflow:hidden; min-height: 210px; color:#fff;
    background: linear-gradient(120deg, #0a5560, #0f8a6e); background-size: cover; background-position: center;
    box-shadow: 0 14px 30px rgba(6,53,63,.2); }
.photo-card .in { position:absolute; inset:0; padding: 26px 28px; display:flex; flex-direction:column; justify-content:flex-end;
    background: linear-gradient(0deg, rgba(6,53,63,.88), rgba(6,53,63,.1) 75%); }
.photo-card h3 { color:#fff !important; margin:0 0 4px 0; font-size: 20px; }
.photo-card p { margin:0; font-size: 13px; color:#d3efe9; }

.banner { border-radius: 22px; padding: 28px 32px; color:#fff; margin-bottom: 20px;
    background: linear-gradient(100deg, rgba(6,53,63,.92), rgba(15,110,86,.82)), var(--img);
    background-size: cover; background-position: center; box-shadow: 0 16px 36px rgba(6,53,63,.22); }
.banner h1 { color:#fff !important; margin:0 0 6px 0; font-size: 32px; }
.banner p { color:#d3efe9; margin:0; font-size: 14px; }

.panel { background:#fff; border:1px solid #e2ecf0; border-radius: 20px; padding: 20px 24px;
    box-shadow: 0 8px 22px rgba(16,60,80,.06); }
.panel h4 { margin: 0 0 8px 0; }
.chip { display:inline-block; background: linear-gradient(135deg, #d9f7ee, #d3eefb); color:#0a4d57; font-size: 12.5px;
    font-weight: 600; padding: 6px 14px; border-radius: 20px; margin: 4px 6px 4px 0; border: 1px solid #bfe6dc; }
.muted { color:#7d92a1; font-size: 13px; }

.result { border-radius: 22px; padding: 26px 30px; color:#fff; display:flex; align-items:center; gap: 28px;
    background: linear-gradient(120deg, #0a5560, #0f8a6e); box-shadow: 0 18px 38px rgba(15,110,86,.3); }
.result small { letter-spacing: 2px; font-weight: 600; color:#b8fbe6; }
.result .name { font-size: 36px; font-weight: 800; line-height: 1.15; margin-top: 4px; }
.result .sub { font-size: 13px; color:#c9f3e8; margin-top: 6px; }
.ring-wrap { position: relative; width: 110px; height: 110px; flex-shrink: 0; }
.ring-wrap .txt { position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center;
    font-size: 22px; font-weight: 800; }
.ring-wrap .txt span { font-size: 10px; font-weight: 500; opacity: .85; }
.bar-row { display:grid; grid-template-columns: 170px 1fr 56px; align-items:center; gap: 14px; margin: 11px 0; }
.bar-name { font-size: 14px; font-weight: 600; color:#0b2e3f; }
.bar-track { background:#e8f1f4; height: 12px; border-radius: 10px; overflow:hidden; }
.bar-fill { height: 100%; border-radius: 10px; animation: grow 1s ease-out; }
@keyframes grow { from { width: 0 !important; } }
.bar-val { font-size: 13px; font-weight: 700; color:#37566a; text-align:right; }
.next { background:#fff; border:1px solid #e2ecf0; border-radius: 16px; padding: 16px 18px; height:100%; font-size: 13px; color:#4b6272; line-height:1.6; }
.next b { display:block; color:#0b2e3f; font-size: 14px; margin-bottom: 4px; }
.notice { background:#fff7e6; border:1px solid #f2d9a3; color:#7a5200; border-radius: 14px; padding: 12px 16px; font-size: 13px; margin-top: 18px; }

.dgrid { display:grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 14px; }
.dcard { background:#fff; border:1px solid #e2ecf0; border-radius: 16px; padding: 14px 16px; display:flex; align-items:center; gap: 12px;
    transition: all .2s ease; box-shadow: 0 6px 16px rgba(16,60,80,.05); }
.dcard:hover { transform: translateY(-3px); box-shadow: 0 12px 26px rgba(16,60,80,.12); }
.dbadge { width: 40px; height: 40px; border-radius: 12px; display:flex; align-items:center; justify-content:center;
    color:#fff; font-weight: 700; font-size: 16px; flex-shrink: 0; }
.dname { font-size: 13.5px; font-weight: 600; color:#0b2e3f; line-height: 1.3; }

.stat { background:#fff; border-radius: 18px; padding: 16px 20px; border:1px solid #e2ecf0; box-shadow: 0 8px 22px rgba(16,60,80,.06);
    position: relative; overflow:hidden; }
.stat::before { content:""; position:absolute; left:0; top:0; width:100%; height:4px; background: var(--c); }
.stat .num { font-size: 26px; font-weight: 800; color:#0b2e3f; }
.stat .lbl { font-size: 12px; color:#6a7f8c; font-weight: 500; }

.footer { text-align:center; color:#8aa0ad; font-size: 12px; padding: 30px 0 4px 0; }

div.stButton > button { background: linear-gradient(100deg, #0f8a6e, #1a9bb0); color:#fff; border:0; border-radius: 14px;
    padding: .7rem 1.2rem; font-weight: 600; font-size: 15px; transition: all .2s ease; box-shadow: 0 8px 20px rgba(15,138,110,.3); }
div.stButton > button:hover { transform: translateY(-2px); filter: brightness(1.08); color:#fff; }
div[data-baseweb="select"] > div { border-radius: 14px; border-color: #cfe3e8; background:#fff; }
.stTabs [data-baseweb="tab"] { font-weight: 600; }
@media (max-width: 900px) { .hero { grid-template-columns: 1fr; } .hero-art { display:none; } .photo-strip { grid-template-columns: 1fr; } }
</style>
""",
    unsafe_allow_html=True,
)

PALETTE = ["#0f8a6e", "#1a9bb0", "#7c6cf0", "#e9a23b", "#e5677f"]
NAV = ["🏠  Home", "🔍  Check symptoms", "🧬  Diseases", "📑  Project report"]


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


def banner(title, sub, img):
    st.markdown(
        f'<div class="banner" style="--img:url(\'{img}\')"><h1>{title}</h1><p>{sub}</p></div>',
        unsafe_allow_html=True,
    )


def go_check():
    st.session_state["nav"] = NAV[1]


# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """<div class="brand"><div class="brand-logo">🩺</div>
        <div><div class="brand-name">MediPredict</div><div class="brand-sub">SYMPTOM CHECKER</div></div></div>""",
        unsafe_allow_html=True,
    )
    choice = st.radio("Navigate", NAV, key="nav", label_visibility="collapsed")

page = choice.split("  ", 1)[1]

# ------------------------------------------------------------------
# HOME
# ------------------------------------------------------------------
if page == "Home":
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-text">
            <span class="pill">✦ YOUR SMART HEALTH COMPANION</span>
            <h1>Not feeling well? <span>Check your symptoms</span></h1>
            <p>Tell us what you are experiencing and get a ranked list of the conditions
            that match, in a few seconds.</p>
          </div>
          <div class="hero-art">
            {SVG_HERO}
            <div class="hero-photo" style="background-image:url('{HERO_IMG}')"></div>
            <div class="float-badge fb1">Quick<b>Under a minute</b></div>
            <div class="float-badge fb2">Private<b>Nothing is stored</b></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1, 1, 1])
    c2.button("Start symptom check  →", on_click=go_check, use_container_width=True)

    st.markdown('<div class="section-title">How it works</div><div class="section-sub">Three simple steps</div>', unsafe_allow_html=True)
    steps = [
        ("STEP 01", "Choose your symptoms", "Search and pick everything you are feeling.", SVG_SELECT),
        ("STEP 02", "We analyse the pattern", "Your combination of symptoms is matched against known cases.", SVG_ANALYSE),
        ("STEP 03", "See your results", "View the most likely conditions with a match score.", SVG_RESULT),
    ]
    for col, (no, t, d, svg) in zip(st.columns(3, gap="large"), steps):
        col.markdown(
            f'<div class="step">{svg}<div class="body"><div class="no">{no}</div><h4>{t}</h4><p>{d}</p></div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="photo-strip">
          <div class="photo-card" style="background-image:url('{DOCTOR_IMG}')">
            <div class="in"><h3>Know what to ask your doctor</h3><p>Go to your appointment with a clearer picture of your symptoms.</p></div>
          </div>
          <div class="photo-card" style="background-image:url('{LAB_IMG}')">
            <div class="in"><h3>Built on real case data</h3><p>Predictions come from patterns learned from thousands of recorded cases.</p></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# CHECK SYMPTOMS
# ------------------------------------------------------------------
elif page == "Check symptoms":
    banner("Symptom checker", "Select everything you are feeling. The more you add, the better the match.", DOCTOR_IMG)

    if "sel" not in st.session_state:
        st.session_state["sel"] = []

    def add_symptom(sym):
        if sym not in st.session_state["sel"]:
            st.session_state["sel"] = st.session_state["sel"] + [sym]

    def clear_all():
        st.session_state["sel"] = []

    left, right = st.columns([1.5, 1], gap="large")
    with left:
        if not symptom_counts.empty:
            quick = [str(s) for s in symptom_counts.sort_values(ascending=False).index[:6] if str(s) in symptoms]
            if quick:
                st.caption("Common symptoms, tap to add")
                for row in (quick[:3], quick[3:6]):
                    for col, s in zip(st.columns(3), row):
                        col.button(pretty(s), key=f"q_{s}", on_click=add_symptom, args=(s,), use_container_width=True)

        selected = st.multiselect(
            "Search all symptoms",
            options=symptoms,
            key="sel",
            format_func=pretty,
            placeholder="Type a symptom, e.g. fever, cough, headache",
        )
        b1, b2 = st.columns([3, 1])
        run = b1.button("🔬  Check my symptoms", use_container_width=True)
        b2.button("Clear", on_click=clear_all, use_container_width=True)

    with right:
        chips = "".join(f'<span class="chip">{pretty(s)}</span>' for s in selected)
        body = chips if selected else '<span class="muted">Nothing selected yet.</span>'
        st.markdown(f'<div class="panel"><h4>Your symptoms ({len(selected)})</h4>{body}</div>', unsafe_allow_html=True)

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

                st.markdown("<br>", unsafe_allow_html=True)
                ring = ""
                if conf is not None:
                    dash = conf / 100 * 263.9
                    ring = (
                        '<div class="ring-wrap"><svg width="110" height="110" viewBox="0 0 110 110">'
                        '<circle cx="55" cy="55" r="42" stroke="rgba(255,255,255,.2)" stroke-width="10" fill="none"/>'
                        f'<circle cx="55" cy="55" r="42" stroke="#7ff0d0" stroke-width="10" fill="none" '
                        f'stroke-linecap="round" stroke-dasharray="{dash:.1f} 263.9" transform="rotate(-90 55 55)"/>'
                        f'</svg><div class="txt">{conf:.0f}%<span>match</span></div></div>'
                    )
                st.markdown(
                    f'<div class="result">{ring}<div><small>MOST LIKELY CONDITION</small>'
                    f'<div class="name">{pretty(pred)}</div>'
                    f'<div class="sub">Based on {len(selected)} symptom(s) you selected</div></div></div>',
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
                    st.markdown(f'<div class="panel"><h4>Other possible matches</h4>{rows}</div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                n1, n2, n3 = st.columns(3, gap="medium")
                n1.markdown('<div class="next"><b>Talk to a doctor</b>Share this list with a healthcare professional for a proper diagnosis.</div>', unsafe_allow_html=True)
                n2.markdown('<div class="next"><b>Track your symptoms</b>Note when they started and whether they get better or worse.</div>', unsafe_allow_html=True)
                n3.markdown('<div class="next"><b>Know the warning signs</b>If symptoms are severe or sudden, seek urgent medical care.</div>', unsafe_allow_html=True)

                if has_proba:
                    report = pd.DataFrame(
                        {
                            "Condition": [pretty(model.classes_[i]) for i in top],
                            "Match %": [round(float(probs[i]) * 100, 2) for i in top],
                        }
                    )
                    st.download_button("⬇️  Download results", report.to_csv(index=False).encode(), "my_results.csv", "text/csv")

                st.markdown(
                    '<div class="notice">This tool gives suggestions only and is not a medical diagnosis. '
                    'Please consult a qualified doctor.</div>',
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.error("Something went wrong while checking these symptoms.")
                st.code(str(exc))

# ------------------------------------------------------------------
# DISEASES
# ------------------------------------------------------------------
elif page == "Diseases":
    banner("Conditions we can recognise", "Browse the conditions covered by the checker.", LAB_IMG)
    q = st.text_input("Search", placeholder="Search a condition, e.g. diabetes")
    shown = [d for d in diseases if q.lower() in d.lower()] if q else diseases
    cards = "".join(
        f'<div class="dcard"><div class="dbadge" style="background:{PALETTE[i % 5]}">{pretty(d)[:1]}</div>'
        f'<div class="dname">{pretty(d)}</div></div>'
        for i, d in enumerate(shown)
    )
    st.markdown(f'<div class="dgrid">{cards}</div>' if shown else '<p class="muted">No matching condition.</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# PROJECT REPORT  (techniques required for the lab evaluation)
# ------------------------------------------------------------------
else:
    banner("Project report", "Disease Diagnosis: dataset analysis, model comparison and results.", HERO_IMG)
    t1, t2, t3, t4 = st.tabs(["Overview", "Dataset analysis", "Model comparison", "Tuning and results"])

    with t1:
        st.markdown("#### Problem statement")
        st.write("Predict the most likely disease from a set of symptoms, treated as a multi-class classification problem.")
        st.markdown("#### Approach")
        pipe = [
            ("1", "Exploratory data analysis", "Class balance, symptom frequency, duplicates and missing values."),
            ("2", "Preprocessing", "Symptoms encoded as a binary matrix with a multi-label binarizer."),
            ("3", "Model training", "Several classification algorithms trained and compared on the same split."),
            ("4", "Hyper-parameter tuning", "Cross-validated search to improve the best models."),
            ("5", "Evaluation", "Accuracy, precision, recall and F1 on held-out test data."),
        ]
        for col, (n, t, d) in zip(st.columns(5), pipe):
            col.markdown(f'<div class="next"><b>{n}. {t}</b>{d}</div>', unsafe_allow_html=True)
        st.markdown("#### Limitations")
        st.write(
            "The dataset can contain repeated records and may not represent real patients, "
            "and the model only predicts diseases present in its training data."
        )

    with t2:
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
                st.dataframe(
                    disease_counts.rename("Records").rename_axis("Disease").reset_index(),
                    hide_index=True,
                    use_container_width=True,
                )

    with t3:
        if results_df.empty:
            st.warning("Model comparison results were not found in the bundle.")
        else:
            num_cols = results_df.select_dtypes("number").columns.tolist()
            st.dataframe(
                results_df.style.format({c: "{:.4f}" for c in num_cols}).highlight_max(subset=num_cols, color="#c9efe2"),
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
                        color=alt.condition(alt.datum[metric] == best, alt.value("#0f8a6e"), alt.value("#a9c9d6")),
                        tooltip=["Model", metric],
                    )
                    .properties(height=340)
                    .configure_view(strokeWidth=0)
                    .configure_axis(grid=False)
                )
                st.altair_chart(chart, use_container_width=True)

    with t4:
        st.success(f"Selected final model: {bundle.get('model_name', 'Not specified')}")
        if best_params:
            st.markdown("#### Best hyper-parameters")
            st.json({str(k): (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in best_params.items()})
        if metrics:
            st.markdown("#### Final test metrics")
            cols = st.columns(min(4, len(metrics)))
            for i, (k, v) in enumerate(metrics.items()):
                val = f"{float(v):.3f}" if isinstance(v, (int, float, np.number)) else str(v)
                cols[i % len(cols)].markdown(
                    f'<div class="stat" style="--c:{PALETTE[i % 5]}"><div class="lbl">{k}</div><div class="num">{val}</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No final metrics were saved in the model bundle.")

st.markdown('<div class="footer">MediPredict · Student project · Not a substitute for professional medical advice</div>', unsafe_allow_html=True)
