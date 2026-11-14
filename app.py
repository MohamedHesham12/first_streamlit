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

st.title("🛵 Delivery Delay Predictor")
st.caption("Enter the order details to estimate the probability of the positive class "
           "(assumed: delivery is late).")

with st.form("order"):
    st.subheader("When")
    c1, c2 = st.columns(2)
    hour = c1.slider("Hour of day", 0, 23, 19)
    day_name = c2.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday",
                                    "Friday", "Saturday", "Sunday"], index=4)
    c1, c2 = st.columns(2)
    is_weekend = c1.checkbox("Weekend", value=day_name in ("Friday", "Saturday"))
    is_rush_hour = c2.checkbox("Rush hour", value=hour in range(18, 22))

    st.subheader("Where")
    c1, c2 = st.columns(2)
    city = c1.selectbox("City", list(cats["city"]))
    zone = c2.selectbox("Zone", list(cats["zone"]))

    st.subheader("Order")
    c1, c2 = st.columns(2)
    restaurant = c1.selectbox("Restaurant", list(cats["restaurant"]))
    cuisine = c2.selectbox("Cuisine", list(cats["cuisine"]))
    c1, c2, c3 = st.columns(3)
    items_count = c1.number_input("Items", 1, 30, 3)
    order_value_egp = c2.number_input("Order value (EGP)", 0.0, 5000.0, 250.0, step=10.0)
    discount_egp = c3.number_input("Discount (EGP)", 0.0, 2000.0, 0.0, step=5.0)

    st.subheader("Delivery")
    c1, c2 = st.columns(2)
    vehicle_type = c1.selectbox("Vehicle", list(cats["vehicle_type"]))
    payment_method = c2.selectbox("Payment", list(cats["payment_method"]))
    c1, c2, c3 = st.columns(3)
    distance_km = c1.number_input("Distance (km)", 0.0, 60.0, 4.0, step=0.5)
    prep_minutes = c2.number_input("Prep time (min)", 0.0, 120.0, 15.0, step=1.0)
    promised_minutes = c3.number_input("Promised time (min)", 5.0, 180.0, 30.0, step=1.0)

    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "hour": hour, "day_name": day_name,
        "is_weekend": int(is_weekend), "is_rush_hour": int(is_rush_hour),
        "city": city, "zone": zone, "restaurant": restaurant, "cuisine": cuisine,
        "vehicle_type": vehicle_type, "payment_method": payment_method,
        "items_count": items_count, "order_value_egp": order_value_egp,
        "discount_egp": discount_egp, "distance_km": distance_km,
        "prep_minutes": prep_minutes, "promised_minutes": promised_minutes,
    }])[list(model.feature_names_in_)]

    pred = int(model.predict(row)[0])

    st.divider()
    if pred == positive_class:
        st.error("Prediction: this delivery will likely be late.")
    else:
        st.success("Prediction: this delivery will likely be on time.")
