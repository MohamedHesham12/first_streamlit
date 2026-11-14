import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

MODEL_PATH = Path(__file__).parent / "random.pkl"

st.set_page_config(page_title="Delivery Delay Predictor", page_icon="🛵", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()
pre = model.named_steps["preprocessor"]
cats = dict(zip(pre.named_transformers_["cat"].feature_names_in_,
                pre.named_transformers_["cat"].categories_))
clf = model.steps[-1][1]
positive_class = clf.classes_[1]


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600;700&family=Nunito:wght@500;700&display=swap');
:root { --navy:#0F3D4C; --sun:#FFC93C; --chili:#E4572E; --mint:#2BB673; --aqua:#DDF0F0; }
html, body, .stApp, [class*="st-"] { font-family: 'Nunito', sans-serif; }
.stApp { background: var(--aqua); }
.block-container { max-width: 760px; padding-top: 2rem; }
header[data-testid="stHeader"] { background: transparent; }

.hero { position: relative; background: var(--sun); border: 4px solid var(--navy);
  border-radius: 28px; padding: 28px 28px 24px; box-shadow: 8px 8px 0 var(--navy); margin-bottom: 34px; }
.hero h1 { font-family: 'Fredoka', sans-serif; font-weight: 700; font-size: 2.5rem;
  line-height: 1.05; margin: 0 0 8px; padding: 0; color: var(--navy); }
.hero p { margin: 0; font-size: 1.05rem; font-weight: 700; color: var(--navy); max-width: 30ch; }
.hero .badge { position: absolute; right: 22px; top: -22px; font-size: 3.4rem; background: #fff;
  border: 4px solid var(--navy); border-radius: 50%; width: 92px; height: 92px; display: flex;
  align-items: center; justify-content: center; transform: rotate(10deg); box-shadow: 4px 4px 0 var(--navy); }
.hero .mini { position: absolute; right: 120px; bottom: 14px; font-size: 2rem; transform: rotate(-14deg); }
.hero .mini2 { position: absolute; right: 26px; bottom: 12px; font-size: 2.2rem; transform: rotate(8deg); }

[data-testid="stForm"] { background: #fff; border: 4px solid var(--navy); border-radius: 24px;
  padding: 26px; box-shadow: 8px 8px 0 var(--navy); }
.tag { display: inline-flex; align-items: center; gap: 10px; background: var(--aqua);
  border: 3px solid var(--navy); border-radius: 999px; padding: 4px 16px 4px 6px; margin: 14px 0 6px;
  font-family: 'Fredoka', sans-serif; font-weight: 600; font-size: 1.15rem; color: var(--navy);
  transform: rotate(-1.5deg); }
.tag .ic { background: #fff; border: 3px solid var(--navy); border-radius: 50%; width: 36px; height: 36px;
  display: inline-flex; align-items: center; justify-content: center; font-size: 1.1rem; }
.tag:first-of-type { margin-top: 0; }

[data-testid="stForm"] label p { font-weight: 700; color: var(--navy); }
[data-baseweb="select"] > div, [data-baseweb="input"], [data-testid="stNumberInputContainer"] {
  border-radius: 14px !important; }

[data-testid="stFormSubmitButton"] button { width: 100%; margin-top: 18px; background: var(--sun);
  color: var(--navy); border: 3px solid var(--navy); border-radius: 18px; padding: 12px 0;
  box-shadow: 5px 5px 0 var(--navy); transition: transform .08s, box-shadow .08s; }
[data-testid="stFormSubmitButton"] button p { font-family: 'Fredoka', sans-serif; font-weight: 700; font-size: 1.25rem; }
[data-testid="stFormSubmitButton"] button:hover { background: #FFD766; color: var(--navy); border-color: var(--navy); }
[data-testid="stFormSubmitButton"] button:active { transform: translate(4px,4px); box-shadow: 1px 1px 0 var(--navy); }

.result { margin-top: 36px; text-align: center; padding: 30px 22px 26px; background: #fff;
  border: 5px solid var(--navy); border-radius: 30px; outline: 10px solid #fff;
  box-shadow: 0 0 0 14px var(--navy), 14px 18px 0 14px rgba(15,61,76,.25); transform: rotate(-1.5deg); }
.result .big { font-size: 5rem; line-height: 1; }
.result h2 { font-family: 'Fredoka', sans-serif; font-weight: 700; font-size: 2.2rem; margin: 10px 0 6px; padding: 0; }
.result p { margin: 0; font-size: 1.1rem; font-weight: 700; }
.result.late { background: #FFE3DA; } .result.late h2 { color: var(--chili); }
.result.ok { background: #D9F5E6; } .result.ok h2 { color: #17894F; }

@media (max-width: 600px) {
  .hero h1 { font-size: 1.9rem; } .hero .mini { display: none; }
  .hero .badge { width: 70px; height: 70px; font-size: 2.4rem; right: 14px; top: -18px; }
  .hero p { max-width: 22ch; }
}
</style>
""", unsafe_allow_html=True)


def sticker(icon, text):
    st.markdown(f'<div class="tag"><span class="ic">{icon}</span>{text}</div>', unsafe_allow_html=True)


st.markdown("""
<div class="hero">
  <div class="badge">🛵</div>
  <h1>Will it arrive on time?</h1>
  <p>Fill in the order and we'll tell you if it's running late.</p>
  <span class="mini">🍕</span><span class="mini2">🍔</span>
</div>
""", unsafe_allow_html=True)

with st.form("order"):
    sticker("⏰", "When")
    c1, c2 = st.columns(2)
    hour = c1.slider("Hour of day", 0, 23, 19)
    day_name = c2.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday",
                                    "Friday", "Saturday", "Sunday"], index=4)
    c1, c2 = st.columns(2)
    is_weekend = c1.checkbox("Weekend", value=day_name in ("Friday", "Saturday"))
    is_rush_hour = c2.checkbox("Rush hour", value=hour in range(18, 22))

    sticker("📍", "Where")
    c1, c2 = st.columns(2)
    city = c1.selectbox("City", list(cats["city"]))
    zone = c2.selectbox("Zone", list(cats["zone"]))

    sticker("🍔", "The order")
    c1, c2 = st.columns(2)
    restaurant = c1.selectbox("Restaurant", list(cats["restaurant"]))
    cuisine = c2.selectbox("Cuisine", list(cats["cuisine"]))
    c1, c2, c3 = st.columns(3)
    items_count = c1.number_input("Items", 1, 30, 3)
    order_value_egp = c2.number_input("Order value (EGP)", 0.0, 5000.0, 250.0, step=10.0)
    discount_egp = c3.number_input("Discount (EGP)", 0.0, 2000.0, 0.0, step=5.0)

    sticker("🛵", "The delivery")
    vehicle_type = st.selectbox("Vehicle", list(cats["vehicle_type"]))
    c1, c2, c3 = st.columns(3)
    distance_km = c1.number_input("Distance (km)", 0.0, 60.0, 4.0, step=0.5)
    prep_minutes = c2.number_input("Prep time (min)", 0.0, 120.0, 15.0, step=1.0)

    submitted = st.form_submit_button("Check this delivery 🛵", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "hour": hour, "day_name": day_name,
        "is_weekend": int(is_weekend), "is_rush_hour": int(is_rush_hour),
        "city": city, "zone": zone, "restaurant": restaurant, "cuisine": cuisine,
        "vehicle_type": vehicle_type,
        "items_count": items_count, "order_value_egp": order_value_egp,
        "discount_egp": discount_egp, "distance_km": distance_km,
        "prep_minutes": prep_minutes, "promised_minutes": promised_minutes,
    }])
    # Stopgap: the current model was trained WITH payment_method. An unseen value makes its
    # one-hot columns all zero (handle_unknown="ignore"). Retrain without the column to
    # remove it for real; this line then does nothing.
    if "payment_method" in model.feature_names_in_:
        row["payment_method"] = "none"
    row = row[list(model.feature_names_in_)]

    pred = int(model.predict(row)[0])

    if pred == positive_class:
        st.markdown("""<div class="result late"><div class="big">🐢⏰</div>
<h2>Running late</h2><p>This delivery will likely be late.</p></div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="result ok"><div class="big">🎉🛵</div>
<h2>On time!</h2><p>This delivery will likely arrive on time.</p></div>""", unsafe_allow_html=True)
