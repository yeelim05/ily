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
st.set_page_config(page_title="Malaysia Housing Expert", layout="wide")

st.markdown("""
    <style>
    .main-header {
        background-color: #1e2130;
        padding: 20px;
        color: white;
        text-align: center;
        border-radius: 8px;
        margin-bottom: 25px;
    }
    .section-title {
        color: #00d4ff; 
        border-bottom: 2px solid #00d4ff;
        padding-bottom: 5px;
        margin-top: 25px;
        margin-bottom: 15px;
        font-weight: bold;
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .result-header {
        background-color: #f1f3f9;
        padding: 15px;
        border-radius: 8px 8px 0 0;
        border-left: 10px solid #27ae60;
        font-weight: bold;
        color: #1e2130;
        margin-top: 20px;
        font-size: 1.5rem;
    }
    .result-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 0 0 8px 8px;
        border: 1px solid #f1f3f9;
        border-left: 10px solid #27ae60;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    </style>
    <div class="main-header">
        <h1>🏠 MALAYSIA HOUSING PRICE EXPERT</h1>
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
        df = pd.DataFrame({
            'township': ['1 Bukit Utama'], 'area': ['Petaling Jaya'], 
            'state': ['Selangor'], 'type': ['Condo'],
            'tenure': ['Freehold'], 'median_psf': [800], 
            'transactions': [10], 'median_price': [1200000]
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
    df['price_val'] = pd.to_numeric(df[col_map['price']], errors='coerce').fillna(0)
    
    feature_cols = ['psf_val', 'trans_val', 'town_enc', 'area_enc', 'state_enc', 'type_enc', 'tenure_enc']
    model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=1, random_state=42)
    model.fit(df[feature_cols], df['price_val'])
    
    gc.collect()
    return df, model, encoders, col_map

df, model, encoders, col_map = load_app_core()

def predict_logic(town, area, state, p_type, tenure, psf, trans):
    input_data = [
        psf, trans,
        encoders['town'].transform([town])[0],
        encoders['area'].transform([area])[0],
        encoders['state'].transform([state])[0],
        encoders['type'].transform([p_type])[0],
        encoders['tenure'].transform([tenure])[0]
    ]
    return model.predict([input_data])[0]

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
mode = st.sidebar.radio("Navigation", ["🔮 Price Predictor", "📊 Comparison Mode", "💰 Mortgage Calculator"])

# ============================================================================
# PART 1: PRICE PREDICTOR
# ============================================================================
if mode == "🔮 Price Predictor":
    st.markdown('<p class="section-title">Step 1: Location Selection</p>', unsafe_allow_html=True)
    loc_c1, loc_c2, loc_c3 = st.columns(3)

    with loc_c1:
        state_choice = st.selectbox("Select State", sorted(df[col_map['state']].unique()))
    with loc_c2:
        area_choice = st.selectbox("Select Area", sorted(df[df[col_map['state']] == state_choice][col_map['area']].unique()))
    with loc_c3:
        town_choice = st.selectbox("Select Township", sorted(df[df[col_map['area']] == area_choice][col_map['town']].unique()))

    st.markdown('<p class="section-title">Step 2: Property Details</p>', unsafe_allow_html=True)
    spec_c1, spec_c2, spec_c3, spec_c4 = st.columns(4)

    with spec_c1:
        prop_type = st.selectbox("Property Type", sorted(df[col_map['type']].unique()))
    with spec_c2:
        tenure_type = st.selectbox("Tenure", sorted(df[col_map['tenure']].unique()))
    with spec_c3:
        avg_psf_val = df[df[col_map['town']] == town_choice]['psf_val'].mean()
        psf_in = st.number_input("Median PSF (RM)", value=float(avg_psf_val) if avg_psf_val > 10 else 100.0, min_value=0.0)
    with spec_c4:
        trans_in = st.number_input("Transactions", value=10, min_value=1)

    adj = st.number_input("Manual Price Adjustment (± RM):", value=0.0, step=500.0)
    
    is_valid = psf_in >= 10.0
    if not is_valid:
        st.error("⚠️ Invalid PSF: Please enter a logical amount (at least RM 10.00).")

    if st.button("CALCULATE PREDICTED PRICE", type="primary", use_container_width=True, disabled=not is_valid):
        raw_val = predict_logic(town_choice, area_choice, state_choice, prop_type, tenure_type, psf_in, trans_in)
        final_v = raw_val + adj
        
        st.markdown('<div class="result-header">📊 PREDICTED PRICE RESULTS</div>', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="result-card">
                <span style="color: #666; font-size: 0.9em; font-weight: bold;">ESTIMATED MARKET VALUE</span>
                <h2 style="color: #27ae60; margin: 0; font-size: 2.5em;">RM {final_v:,.2f}</h2>
                <hr style="margin: 15px 0; border: 0; border-top: 1px solid #eee;">
                <p style="margin: 5px 0; color: #1e2130; font-size: 1.1em;"><b>Location:</b> {town_choice}, {area_choice}</p>
                <p style="margin: 5px 0; color: #1e2130; font-size: 1.1em;"><b>Property:</b> {prop_type} ({tenure_type})</p>
                <small style="color: #999;">Statistical Baseline: RM {raw_val:,.2f} | Adjustment: RM {adj:,.2f}</small>
            </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PART 2: COMPARISON MODE
# ============================================================================
elif mode == "📊 Comparison Mode":
    st.markdown('<p class="section-title">Side-by-Side Comparison</p>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("📍 Area A")
        s_a = st.selectbox("State", sorted(df[col_map['state']].unique()), key="sa")
        a_a = st.selectbox("Area", sorted(df[df[col_map['state']] == s_a][col_map['area']].unique()), key="aa")
        t_a = st.selectbox("Township", sorted(df[df[col_map['area']] == a_a][col_map['town']].unique()), key="ta")
        p_a = st.number_input("PSF (A)", value=500.0, key="pa")

    with col_b:
        st.subheader("📍 Area B")
        s_b = st.selectbox("State", sorted(df[col_map['state']].unique()), key="sb")
        a_b = st.selectbox("Area", sorted(df[df[col_map['state']] == s_b][col_map['area']].unique()), key="ab")
        t_b = st.selectbox("Township", sorted(df[df[col_map['area']] == a_b][col_map['town']].unique()), key="tb")
        p_b = st.number_input("PSF (B)", value=500.0, key="pb")

    if st.button("RUN COMPARISON", type="primary", use_container_width=True):
        v_a = predict_logic(t_a, a_a, s_a, "Terrace", "Freehold", p_a, 10)
        v_b = predict_logic(t_b, a_b, s_b, "Terrace", "Freehold", p_b, 10)
        
        diff = v_b - v_a
        c1, c2 = st.columns(2)
        c1.metric(f"Value in {t_a}", f"RM {v_a:,.2f}")
        c2.metric(f"Value in {t_b}", f"RM {v_b:,.2f}", delta=f"RM {diff:,.2f}")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=[t_a, t_b], y=[v_a, v_b], palette=['#1e2130', '#27ae60'])
        plt.title("Price Comparison (RM)")
        st.pyplot(fig)

    st.markdown('<p class="section-title">Top 10 Areas in State (Market Trends)</p>', unsafe_allow_html=True)
    sel_state = st.selectbox("View Trends for State:", sorted(df[col_map['state']].unique()))
    top_10 = df[df[col_map['state']] == sel_state].groupby(col_map['town'])['price_val'].mean().sort_values(ascending=False).head(10)
    st.bar_chart(top_10)

# ============================================================================
# PART 3: MORTGAGE CALCULATOR
# ============================================================================
elif mode == "💰 Mortgage Calculator":
    st.markdown('<p class="section-title">Loan Affordability Calculator</p>', unsafe_allow_html=True)
    m_col1, m_col2 = st.columns([1, 2])
    
    with m_col1:
        price = st.number_input("Property Price (RM)", value=500000.0, step=10000.0)
        dp_pct = st.slider("Downpayment (%)", 0, 50, 10)
        interest = st.number_input("Interest Rate (%)", value=4.2, step=0.1)
        tenure = st.slider("Tenure (Years)", 5, 35, 30)
        
        loan = price * (1 - dp_pct/100)
        rate = (interest/100)/12
        months = tenure * 12
        monthly = (loan * rate * (1+rate)**months) / ((1+rate)**months - 1) if rate > 0 else loan/months

    with m_col2:
        st.markdown(f"""
            <div class="result-card">
                <span style="color: #666; font-size: 0.9em; font-weight: bold;">ESTIMATED MONTHLY INSTALLMENT</span>
                <h2 style="color: #1e2130; margin: 0;">RM {monthly:,.2f}</h2>
                <hr style="margin: 15px 0; border: 0; border-top: 1px solid #eee;">
                <p style="margin: 5px 0; color: #1e2130;"><b>Loan Amount:</b> RM {loan:,.2f}</p>
                <p style="margin: 5px 0; color: #1e2130;"><b>Total Interest:</b> RM {(monthly*months)-loan:,.2f}</p>
            </div>
        """, unsafe_allow_html=True)
        st.info(f"💡 Monthly Net Household Income required: RM {monthly/0.35:,.2f}")
        
        fig, ax = plt.subplots()
        ax.pie([loan, (monthly*months)-loan], labels=['Principal', 'Interest'], autopct='%1.1f%%', colors=['#27ae60', '#1e2130'])
        st.pyplot(fig)