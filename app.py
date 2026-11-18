
# =====================================================
# PREMIUM MULTI-COLOR UI
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 5% 5%, #25365A 0%, transparent 32%),
        radial-gradient(ellipse at 95% 20%, #173E50 0%, transparent 30%),
        linear-gradient(135deg, #101B30, #14243B 55%, #101D32);
    color: #F5F7FF;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1 {
    color: #F4F5FF !important;
    font-weight: 800 !important;
}

h2 {
    color: #B9A7FF !important;
    font-weight: 700 !important;
}

h3 {
    color: #73DDEB !important;
    font-weight: 700 !important;
}

p, label, .stMarkdown {
    color: #D5DDED;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111B30, #1B2440);
    border-right: 1px solid #394665;
}

section[data-testid="stSidebar"] h1 {
    color: #B9A7FF !important;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #CBD4E7;
}

/* METRIC CARDS */

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #263958, #1C2D48);
    border: 1px solid #465B7A;
    padding: 23px;
    border-radius: 18px;
    box-shadow: 0 8px 22px rgba(0,0,0,.16);
}

[data-testid="stMetricLabel"] {
    color: #BFCDE1 !important;
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
    background: linear-gradient(100deg, #A16BEA, #C17FEA);
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
        radial-gradient(circle at 90% 10%, #57418D 0%, transparent 42%),
        linear-gradient(120deg, #233C61, #344B78 55%, #385E76);
    border: 1px solid #6479A7;
    border-radius: 23px;
    padding: 34px;
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
    font-size: 36px;
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

/* FEATURE CARDS */

.feature-card {
    background: linear-gradient(145deg, #263958, #1B2C47);
    border: 1px solid #455A7D;
    border-radius: 17px;
    padding: 23px;
    min-height: 165px;
    margin-bottom: 12px;
    transition: .2s ease;
}

.feature-card:hover {
    border-color: #A58BFF;
    transform: translateY(-3px);
}

.feature-icon {
    font-size: 32px;
    margin-bottom: 13px;
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

/* ALTERNATING FEATURE ACCENTS */

div[data-testid="stVerticalBlock"] .feature-card:nth-child(1) {
    border-top: 3px solid #9A7BFF;
}

/* RESULT */

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

/* EXPANDERS AND TABLES */

[data-testid="stExpander"] {
    background: #202F4B;
    border: 1px solid #465A7A;
    border-radius: 12px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #465A7A;
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
