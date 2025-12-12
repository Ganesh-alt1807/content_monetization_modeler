import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
import urllib.parse

# Page config
st.set_page_config(page_title="YouTube Revenue Predictor", page_icon=":money_with_wings:", layout="centered")

# load assets
assets_dir = Path(__file__).parent / "assets"
bg_svg = (assets_dir / "stars-bg.svg").read_text() if (assets_dir / "stars-bg.svg").exists() else ""
bg_data = urllib.parse.quote(bg_svg)
icon_svg = (assets_dir / "youtube-icon.svg").read_text() if (assets_dir / "youtube-icon.svg").exists() else ""
icon_data = urllib.parse.quote(icon_svg)

# CSS
css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [data-testid='stAppViewContainer'] {
    height: 100%;
    font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, rgba(11,17,29,0.85) 0%, rgba(27,16,61,0.85) 100%);
    background-image: url("data:image/svg+xml;utf8," + bg_data + ");
    background-size: cover;
    background-position: center;
}
.app-header {
    display:flex; align-items:center; gap:18px; margin-bottom:8px;
}
.app-header h1{ margin:0; color:#fff; font-size:28px; letter-spacing:0.2px;}
.app-subtitle{ color:rgba(255,255,255,0.85); margin:6px 0 18px 0; }
.glass-panel{
    margin: 28px auto;
    max-width: 1100px;
    background: linear-gradient(180deg, rgba(255,255,255,0.12), rgba(255,255,255,0.06));
    border-radius:20px; padding:28px; backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
    box-shadow: 0 12px 40px rgba(3,10,25,0.45); border: 1px solid rgba(255,255,255,0.06);
}
.inputs-row{ display:flex; gap:20px; }
.inputs-col{ flex:1; }
.stButton>button{ background: linear-gradient(90deg,#ff3b30,#e62117) !important; color:#fff !important; border-radius:10px !important; padding:10px 18px !important; font-weight:700 !important; box-shadow:0 8px 20px rgba(230,59,48,0.18) !important; }
.result-card{ background: linear-gradient(180deg,#061226,#0b0f1a); padding:18px; border-radius:12px; color:#fff; box-shadow: 0 8px 30px rgba(2,6,23,0.5); margin-top:18px; }
.muted{ color: rgba(255,255,255,0.7); font-size:14px; }
.small{ font-size:13px; color:rgba(255,255,255,0.8); }

</style>
"""

st.markdown(css, unsafe_allow_html=True)

model = None
model_loaded = False
model_loaded_time = None
model_path = Path(__file__).parent / "yt_revenue_model.pkl"
if model_path.exists():
    try:
        model = joblib.load(str(model_path))
        model_loaded = True
        model_loaded_time = datetime.now()
    except Exception as e:
        model = None
        model_loaded = False
        st.error(f"Failed to load model at startup: {e}")



def _ensure_model_inputs(df: pd.DataFrame, model) -> pd.DataFrame:
    """Ensure DataFrame contains all columns expected by the model's preprocessor.
    Fills missing numeric columns with 0 and missing categorical columns with the first
    known category from the trained encoder (if available).
    """
    if model is None or not hasattr(model, 'named_steps') or 'preprocessor' not in model.named_steps:
        return df

    pre = model.named_steps['preprocessor']
    # gather required columns from transformers
    required = []
    num_cols = []
    cat_cols = []
    cat_categories = {}
    for name, transformer, cols in pre.transformers_:
        try:
            cols_list = list(cols)
        except Exception:
            cols_list = []
        required.extend(cols_list)
        if name == 'num':
            num_cols.extend(cols_list)
        if name == 'cat':
            cat_cols.extend(cols_list)
            try:
                cats = transformer.categories_
                for i, col in enumerate(cols_list):
                    if i < len(cats):
                        cat_categories[col] = list(cats[i])
            except Exception:
                pass

    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"Adding missing columns with defaults so model can predict: {missing}")
    for c in missing:
        if c in num_cols:
            df[c] = 0
        elif c in cat_cols:
            # default to first known category if available
            default = cat_categories.get(c, [None])[0]
            df[c] = default if default is not None else ''
        else:
            df[c] = 0

    # Ensure numeric columns are proper dtypes
    for c in num_cols:
        if c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            except Exception:
                df[c] = 0

    return df

st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

with st.form('main_form'):
    st.markdown(
        f'<div style="display:flex;align-items:center;justify-content:space-between"><div style="display:flex;align-items:center;gap:18px"><img src="data:image/svg+xml;utf8,{icon_data}" style="width:56px;height:56px;border-radius:10px;box-shadow:0 8px 20px rgba(0,0,0,0.3);"/><div><h1>YouTube Revenue Predictor</h1></div></div></div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1,1])
    with col1:
        views = st.number_input("Views", min_value=0, value=1000)
        likes = st.number_input("Likes", min_value=0, value=100)
        comments = st.number_input("Comments", min_value=0, value=10)
        subscribers = st.number_input("Subscribers", min_value=0, value=1000)
    with col2:
        watch_time_minutes = st.number_input("Watch time (min)", min_value=0, value=4)
        video_length_minutes = st.number_input("Video length (min)", min_value=0, value=10)
        category = st.selectbox("Category", ["Entertainment", "Education", "Gaming", "Lifestyle", "Music", "Tech"])
        device = st.selectbox("Device", ["Mobile", "Desktop", "Tablet", "TV"])
        country = st.selectbox("Country", ["US", "UK", "CA", "AU", "IN", "DE"])

    submitted = st.form_submit_button("Predict Revenue")

    if submitted:
        # feature engineering
        engagement_rate = (likes + comments) / (views if views else 1)
        watch_efficiency = watch_time_minutes / (video_length_minutes if video_length_minutes else 1)
        data = pd.DataFrame([{
            'views': views, 'likes': likes, 'comments': comments,
            'watch_time_minutes': watch_time_minutes, 'video_length_minutes': video_length_minutes,
            'subscribers': subscribers, 'category': category, 'device': device, 'country': country,
            'engagement_rate': engagement_rate, 'watch_efficiency': watch_efficiency
        }])
        # If model wasn't loaded at startup, try to load it now (useful when user adds the file while app is running)
        if model is None:
            try:
                model = joblib.load(str(model_path))
                model_loaded = True
                model_loaded_time = datetime.now()
            except Exception as e:
                st.exception(e)

        if model is not None:
            try:
                # Ensure our data has all columns expected by the model
                data = _ensure_model_inputs(data, model)
                pred = model.predict(data)[0]
                st.markdown(f'<div class="result-card"><div style="font-size:20px;font-weight:800">Estimated Revenue: <span style="color:#ffdf7a">${pred:.2f}</span></div><div class="small muted">Model output — use with caution</div></div>', unsafe_allow_html=True)
            except Exception as e:
                st.exception(e)


st.markdown('</div>', unsafe_allow_html=True)