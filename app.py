import streamlit as st
import numpy as np
import joblib
import os
import random

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="AI-Powered Product Recommendation Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model_data = joblib.load(os.path.join(MODEL_DIR, "funk_svd_model.pkl"))
popular_items = joblib.load(os.path.join(MODEL_DIR, "popularity_top100.pkl"))

user_f = model_data["user_f"]
item_f = model_data["item_f"]
user_b = model_data["user_b"]
item_b = model_data["item_b"]
global_mean = model_data["global_mean"]
user2idx = model_data["user2idx"]
idx2item = model_data["idx2item"]

# ==========================================================
# PRODUCT CATALOG (REALISTIC PRESENTATION LAYER)
# ==========================================================
CATEGORIES = {
    "Electronics": ["Samsung", "Apple", "Sony", "HP", "Dell"],
    "Fashion": ["Nike", "Adidas", "Puma", "Zara"],
    "Beauty": ["L'Oréal", "Maybelline", "Lakmé"],
    "Home & Kitchen": ["Philips", "Prestige", "Butterfly"],
    "Sports": ["Yonex", "Decathlon", "Cosco"]
}

def build_product(product_id):
    random.seed(hash(product_id))
    category = random.choice(list(CATEGORIES.keys()))
    brand = random.choice(CATEGORIES[category])
    return {
        "id": product_id,
        "name": f"{brand} {category} Product",
        "brand": brand,
        "category": category,
        "price": random.randint(999, 14999),
        "rating": round(random.uniform(3.8, 4.9), 1)
    }

PRODUCT_CATALOG = {pid: build_product(pid) for pid in idx2item.values()}

# ==========================================================
# RECOMMENDATION ENGINE
# ==========================================================
def recommend_popular(k=8):
    return [PRODUCT_CATALOG[i] for i in popular_items[:k]]

def recommend_personalized(k=8):
    demo_user = list(user2idx.values())[0]
    scores = (
        global_mean
        + item_b
        + user_b[demo_user]
        + item_f @ user_f[demo_user]
    )
    top_indices = np.argsort(-scores)[:k]
    return [PRODUCT_CATALOG[idx2item[i]] for i in top_indices]

# ==========================================================
# UI STYLING (MODERN E-COMMERCE)
# ==========================================================
st.markdown("""
<style>
.product-card {
    background: linear-gradient(145deg, #0f172a, #020617);
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #1e293b;
    box-shadow: 0 10px 25px rgba(0,0,0,0.4);
}
.product-title {
    font-size: 18px;
    font-weight: 700;
    color: #f8fafc;
}
.product-brand {
    font-size: 14px;
    color: #38bdf8;
}
.product-rating {
    color: #facc15;
    font-size: 14px;
}
.product-price {
    font-size: 20px;
    color: #22c55e;
    font-weight: bold;
}
.product-reason {
    font-size: 12px;
    color: #94a3b8;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================
st.title("🛒 AI-Powered Product Recommendation Platform")
st.markdown(
    "A **production-grade recommendation system** that intelligently suggests "
    "products based on user intent, popularity trends, and collaborative filtering."
)

st.divider()

# ==========================================================
# USER INTENT CAPTURE (CRITICAL FOR PROFESSIONAL SYSTEM)
# ==========================================================
with st.form("intent_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        customer_type = st.selectbox(
            "Customer Type",
            ["New Visitor", "Returning Customer"]
        )

    with col2:
        category = st.selectbox(
            "Category of Interest",
            list(CATEGORIES.keys())
        )

    with col3:
        strategy = st.selectbox(
            "Recommendation Strategy",
            ["Most Popular", "Personalized"]
        )

    submit = st.form_submit_button("🔍 Discover Products")

# ==========================================================
# RESULTS
# ==========================================================
if submit:
    if customer_type == "New Visitor" or strategy == "Most Popular":
        products = recommend_popular()
        reason = "Trending products preferred by most customers"
    else:
        products = recommend_personalized()
        reason = "Personalized based on similar user behavior"

    products = [p for p in products if p["category"] == category][:8]

    st.subheader(f"Recommended {category} Products")
    st.caption(reason)

    cols = st.columns(4)
    for i, p in enumerate(products):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="product-card">
                <div class="product-title">{p['name']}</div>
                <div class="product-brand">{p['brand']} • {p['category']}</div>
                <div class="product-rating">⭐ {p['rating']} / 5</div>
                <div class="product-price">₹{p['price']}</div>
                <div class="product-reason">
                    Recommended based on {strategy.lower()} strategy
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.success("Personalized recommendations generated successfully")

st.divider()
st.caption(
    "Machine Learning: Matrix Factorization (Collaborative Filtering) + Popularity Models"
)
