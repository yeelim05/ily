import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import gc

# ============================================================================
# UI CONFIGURATION & STYLING
# ============================================================================
st.set_page_config(page_title="Housing Price Predictor", layout="wide")

st.markdown("""
    <style>
    /* Dark Header from Prototype */
    .main-header {
        background-color: #1e2130;
        padding: 20px;
        color: white;
        text-align: center;
        border-radius: 8px;
        margin-bottom: 25px;
    }
  /* Section Headers */
    .section-title {
        color: #00d4ff; /* Electric Blue (Change to #27ae60 for Mint Green) */
        border-bottom: 2px solid #00d4ff;
        padding-bottom: 5px;
        margin-top: 25px;
        margin-bottom: 15px;
        font-weight: bold;
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    /* Result Display Header */
    .result-header {
        background-color: #f1f3f9;
        padding: 10px;
        border-radius: 5px 5px 0 0;
        border-left: 10px solid #27ae60;
        font-weight: bold;
        color: #1e2130;
        margin-top: 20px;
    }
    .result-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 0 0 5px 5px;
        border: 1px solid #f1f3f9;
        border-left: 10px solid #27ae60;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    </style>
    <div class="main-header">
        <h1>🏠 HOUSING PRICE IN MALAYSIA </h1>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA & MODEL LOADING
# ============================================================================
@st.cache_resource
def load_app_core():
    try:
        from datasets import load_dataset
        ds = load_dataset("jienweng/housing-prices-malaysia-2025")
        df = ds['train'].to_pandas()
    except:
        # Fallback dataset for structure verification
        df = pd.DataFrame({
            'township': ['1 Bukit Utama', 'Alor Setar'], 
            'area': ['Petaling Jaya', 'Kota Setar'], 
            'state': ['Selangor', 'Kedah'], 
            'type': ['Condo', 'Terrace'],
            'tenure': ['Freehold', 'Freehold'], 
            'median_psf': [800, 300], 
            'transactions': [10, 5], 
            'median_price': [1200000, 450000]
        })

    cols = df.columns.tolist()
    def find_col(keywords):
        return next((c for c in cols if any(k in c.lower() for k in keywords)), None)

    col_map = {
        'town': find_col(['township', 'town']),
        'area': find_col(['area', 'district']),
        'state': find_col(['state']),
        'type': find_col(['type', 'property']),
        'tenure': find_col(['tenure']),
        'psf': find_col(['psf']),
        'trans': find_col(['trans']),
        'price': find_col(['price', 'median_price'])
    }

    df = df.dropna(subset=[col_map['state'], col_map['area'], col_map['town']])
    
    encoders = {}
    for feat in ['town', 'area', 'state', 'type', 'tenure']:
        le = LabelEncoder()
        df[f'{feat}_enc'] = le.fit_transform(df[col_map[feat]].astype(str))
        encoders[feat] = le

    df['psf_val'] = pd.to_numeric(df[col_map['psf']], errors='coerce').fillna(0)
    df['trans_val'] = pd.to_numeric(df[col_map['trans']], errors='coerce').fillna(0)
    
    feature_cols = ['psf_val', 'trans_val', 'town_enc', 'area_enc', 'state_enc', 'type_enc', 'tenure_enc']
    model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=1, random_state=42)
    model.fit(df[feature_cols], df[col_map['price']])
    
    gc.collect()
    return df, model, encoders, col_map

df, model, encoders, col_map = load_app_core()

def predict_price(town, area, state, p_type, tenure, psf, trans, adjustment):
    input_data = [
        psf, trans,
        encoders['town'].transform([town])[0],
        encoders['area'].transform([area])[0],
        encoders['state'].transform([state])[0],
        encoders['type'].transform([p_type])[0],
        encoders['tenure'].transform([tenure])[0]
    ]
    return model.predict([input_data])[0] + adjustment


# ============================================================================
# MAIN INTERFACE
# ============================================================================
mode = st.sidebar.radio("Navigation", ["🔮 Price Predictor", "📊 Comparison Mode"])

if mode == "🔮 Price Predictor":
    st.markdown('<p class="section-title">Step 1: Location </p>', unsafe_allow_html=True)
loc_c1, loc_c2, loc_c3 = st.columns(3)

with loc_c1:
    state_list = sorted(df[col_map['state']].unique())
    state_choice = st.selectbox("1. Select State", state_list)

with loc_c2:
    # Filter areas based on chosen state
    filtered_areas = sorted(df[df[col_map['state']] == state_choice][col_map['area']].unique())
    area_choice = st.selectbox("2. Select Area (District)", filtered_areas)

with loc_c3:
    # Filter townships based on chosen area
    filtered_towns = sorted(df[df[col_map['area']] == area_choice][col_map['town']].unique())
    town_choice = st.selectbox("3. Select Township", filtered_towns)

st.markdown('<p class="section-title">Step 2: Property Specifics</p>', unsafe_allow_html=True)
spec_c1, spec_c2, spec_c3, spec_c4 = st.columns(4)

with spec_c1:
    prop_type = st.selectbox("Type", sorted(df[col_map['type']].unique()))
with spec_c2:
    tenure_type = st.selectbox("Tenure", sorted(df[col_map['tenure']].unique()))
with spec_c3:
    avg_psf = df[df[col_map['town']] == town_choice]['psf_val'].mean()
    # Setting a floor of 10.0 (or whatever minimum you prefer)
    psf_in = st.number_input("Median PSF (RM)", value=float(avg_psf) if avg_psf > 10 else 100.0, min_value=0.0)   
    
    MIN_PSF_THRESHOLD = 10.0  # Set your logical limit here
    is_valid_psf = psf_in >= MIN_PSF_THRESHOLD

    if not is_valid_psf:
       st.markdown(f'''
         <div style="background-color: #ff4b4b; color: white; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
            ⚠️ <b>Invalid Input:</b> Median PSF must be at least <b>RM {MIN_PSF_THRESHOLD}</b> to perform a calculation.
        </div>
    ''', unsafe_allow_html=True)

with spec_c4:
    trans_in = st.number_input("Transactions", value=10)

st.markdown("---")
adj = st.number_input("Customer Price Adjustment (± RM):", value=0.0, step=100.0)

if st.button("CALCULATE PREDICTED PRICE", type="primary", use_container_width=True):
    if psf_in <= 0:
        st.error("Error: Calculation failed. Median PSF must be greater than RM 0.")
    else:
        # Encoding for prediction
        input_data = [
            psf_in, trans_in,
            encoders['town'].transform([town_choice])[0],
            encoders['area'].transform([area_choice])[0],
            encoders['state'].transform([state_choice])[0],
            encoders['type'].transform([prop_type])[0],
            encoders['tenure'].transform([tenure_type])[0]
        ]
        
        raw_pred = model.predict([input_data])[0]
        final_price = raw_pred + adj
            
    # RESULT SECTION WITH NEW HEADERS
    st.markdown('<div class="result-header" style="font-size: 24px; padding: 15px;">📊 PREDICTED PRICE RESULTS</div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="result-card">
            <span style="color: #666; font-size: 0.9em;">ESTIMATED MARKET VALUE</span>
            <h2 style="color: #27ae60; margin: 0;">RM {final_price:,.2f}</h2>
            <hr style="margin: 15px 0; border: 0; border-top: 1px solid #eee;">
           <p style="margin: 5px 0; color: #1e2130;"><b>Location:</b> {town_choice}, {area_choice}, {state_choice}</p>
	   <p style="margin: 5px 0; color: #1e2130;"><b>Property:</b> {prop_type} ({tenure_type})</p>
            <small style="color: #999;">Statistical Baseline: RM {raw_pred:,.2f} | Adjustment: RM {adj:,.2f}</small>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

pass

# ============================================================================
# PART 2: COMPARISON MODE 
# ============================================================================
elif mode == "📊 Comparison Mode":
    st.markdown('<p class="section-title">Property Comparison Tool</p>', unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    
    # Setup inputs for two different locations
    with col_a:
        st.subheader("📍 Location A")
        state_a = st.selectbox("Select State", sorted(df[col_map['state']].unique()), key="s_a")
        area_a = st.selectbox("Select Area", sorted(df[df[col_map['state']] == state_a][col_map['area']].unique()), key="a_a")
        town_a = st.selectbox("Select Township", sorted(df[df[col_map['area']] == area_a][col_map['town']].unique()), key="t_a")
        psf_a = st.number_input("Median PSF (A)", value=500.0, key="p_a")

    with col_b:
        st.subheader("📍 Location B")
        state_b = st.selectbox("Select State", sorted(df[col_map['state']].unique()), key="s_b")
        area_b = st.selectbox("Select Area", sorted(df[df[col_map['state']] == state_b][col_map['area']].unique()), key="a_b")
        town_b = st.selectbox("Select Township", sorted(df[df[col_map['area']] == area_b][col_map['town']].unique()), key="t_b")
        psf_b = st.number_input("Median PSF (B)", value=500.0, key="p_b")

    st.markdown("---")
    # Common shared attributes for comparison
    c1, c2, c3 = st.columns(3)
    with c1: comp_type = st.selectbox("Property Type", sorted(df[col_map['type']].unique()), key="c_type")
    with c2: comp_tenure = st.selectbox("Tenure", sorted(df[col_map['tenure']].unique()), key="c_tenure")
    with c3: comp_trans = st.number_input("Transactions", value=10, key="c_trans")

    if st.button("RUN SIDE-BY-SIDE COMPARISON", type="primary", use_container_width=True):
        price_a = predict_price(town_a, area_a, state_a, comp_type, comp_tenure, psf_a, comp_trans, 0)
        price_b = predict_price(town_b, area_b, state_b, comp_type, comp_tenure, psf_b, comp_trans, 0)
        
        diff = price_b - price_a
        percent_diff = (diff / price_a) * 100 if price_a != 0 else 0

        # Aesthetic Comparison Results
        res_a, res_b = st.columns(2)
        with res_a:
            st.metric(label=f"Value in {town_a}", value=f"RM {price_a:,.2f}")
        with res_b:
            st.metric(label=f"Value in {town_b}", value=f"RM {price_b:,.2f}", delta=f"{diff:,.2f} ({percent_diff:.1f}%)")

        # Visualization Chart
        chart_data = pd.DataFrame({
            'Location': [town_a, town_b],
            'Predicted Price': [price_a, price_b]
        })
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x='Location', y='Predicted Price', data=chart_data, palette=['#1e2130', '#27ae60'], ax=ax)
        st.pyplot(fig)

	pass