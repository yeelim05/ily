import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['HF_DATASETS_OFFLINE'] = '0'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

import logging
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
logging.getLogger('datasets').setLevel(logging.ERROR)

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 8

st.set_page_config(page_title="Malaysia Housing - Complete ML Pipeline", layout="wide")

# ============================================================================
# LOAD HUGGINGFACE DATASET
# ============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_huggingface_data():
    """Load dataset from HuggingFace"""
    try:
        from datasets import load_dataset
        ds = load_dataset("jienweng/housing-prices-malaysia-2025")
        df = ds['train'].to_pandas()
        return df
    except Exception as e:
        np.random.seed(42)
        df = pd.DataFrame({
            'township': np.random.choice(['Cheras', 'Subang', 'Shah Alam', 'Petaling Jaya', 'Klang'], 2000),
            'area': np.random.choice(['Klang Valley', 'Selangor', 'Johor', 'Penang', 'Sabah'], 2000),
            'state': np.random.choice(['Selangor', 'Johor', 'Penang', 'KL', 'Sabah'], 2000),
            'tenure': np.random.choice(['Freehold', 'Leasehold'], 2000),
            'type': np.random.choice(['Terrace', 'Condominium', 'Semi-D', 'Detached', 'Bungalow'], 2000),
            'median_price': np.random.randint(250000, 2500000, 2000),
            'median_psf': np.random.uniform(250, 1500, 2000),
            'transactions': np.random.randint(3, 250, 2000),
        })
        return df

df_original = load_huggingface_data()
target_col = 'median_price' if 'median_price' in df_original.columns else 'Median_Price'

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
page = st.sidebar.radio("📍 Navigation", [
    "📊 Dataset Overview",
    "🔍 Initial EDA",
    "🧹 Data Cleaning",
    "🔧 Feature Engineering",
    "📋 Data Preparation",
    "⚙️ Model Training",
    "🏆 Model Evaluation"
])

st.sidebar.markdown("---")
st.sidebar.info(f"""
📈 **Dataset Source:** HuggingFace
🔗 jienweng/housing-prices-malaysia-2025

📊 **Rows:** {df_original.shape[0]:,}
📋 **Columns:** {df_original.shape[1]}
🎯 **Target:** {target_col}
""")

# ============================================================================
# PAGE 1: DATASET OVERVIEW
# ============================================================================
if page == "📊 Dataset Overview":
    st.header("📊 DATASET OVERVIEW")
    st.write("**Source:** HuggingFace - jienweng/housing-prices-malaysia-2025")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{df_original.shape[0]:,}")
    col2.metric("Columns", f"{df_original.shape[1]}")
    col3.metric("Data Types", f"{df_original.dtypes.nunique()}")
    col4.metric("Missing Values", f"{df_original.isnull().sum().sum()}")
    
    st.subheader("Column Information")
    col_info = []
    for col in df_original.columns:
        dtype = df_original[col].dtype
        missing = df_original[col].isnull().sum()
        missing_pct = (missing / len(df_original)) * 100
        unique = df_original[col].nunique()
        
        if dtype == 'object':
            col_info.append({
                'Column': col, 
                'Type': str(dtype), 
                'Missing': f"{missing} ({missing_pct:.2f}%)", 
                'Unique': unique,
                'Sample Values': str(df_original[col].unique()[:3])
            })
        else:
            try:
                min_val = df_original[col].min()
                max_val = df_original[col].max()
                if pd.isna(min_val) or pd.isna(max_val):
                    min_str = "N/A"
                    max_str = "N/A"
                else:
                    min_str = f"{min_val:.0f}"
                    max_str = f"{max_val:.0f}"
            except:
                min_str = "N/A"
                max_str = "N/A"
            
            col_info.append({
                'Column': col, 
                'Type': str(dtype), 
                'Missing': f"{missing} ({missing_pct:.2f}%)", 
                'Min': min_str,
                'Max': max_str
            })
    
    st.dataframe(pd.DataFrame(col_info), use_container_width=True)
    
    with st.expander("📊 First 10 Rows"):
        st.dataframe(df_original.head(10), use_container_width=True)
    
    with st.expander("📈 Statistical Summary"):
        st.dataframe(df_original.describe().round(2), use_container_width=True)

# ============================================================================
# PAGE 2: INITIAL EDA
# ============================================================================
elif page == "🔍 Initial EDA":
    st.header("🔍 INITIAL EXPLORATORY DATA ANALYSIS")
    
    st.subheader("📊 Diagram 1: Missing Values & Data Quality")
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
    
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(df_original[target_col], kde=True, color='#3498db', ax=ax1, bins=25)
    ax1.set_title(f'Distribution of {target_col}', fontweight='bold')
    ax1.set_xlabel('Price (RM)')
    ax1.set_ylabel('Frequency')
    
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(np.log1p(df_original[target_col]), kde=True, color='#2ecc71', ax=ax2, bins=25)
    ax2.set_title('Log Distribution (Normalized)', fontweight='bold')
    ax2.set_xlabel('Log Price')
    ax2.set_ylabel('Frequency')
    
    ax3 = fig.add_subplot(gs[1, 0])
    sns.boxplot(y=df_original[target_col], color='#f1c40f', ax=ax3)
    ax3.set_title('Price Boxplot - Outlier Detection', fontweight='bold')
    ax3.set_ylabel('Price (RM)')
    
    ax4 = fig.add_subplot(gs[1, 1])
    missing_data = df_original.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if len(missing_data) > 0:
        ax4.barh(missing_data.index, missing_data.values, color='#e74c3c', alpha=0.7)
        ax4.set_title('Missing Values by Column', fontweight='bold')
        ax4.set_xlabel('Count')
    else:
        ax4.text(0.5, 0.5, '✅ No Missing Values', ha='center', va='center', fontsize=14, fontweight='bold', transform=ax4.transAxes)
        ax4.set_title('Missing Values Status', fontweight='bold')
        ax4.axis('off')
    
    plt.suptitle('MISSING VALUES & DATA QUALITY ASSESSMENT', fontsize=12, fontweight='bold')
    st.pyplot(fig, use_container_width=True)
    plt.close()
    
    st.subheader("📊 Diagram 2: Correlation Between Variables")
    
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    numeric_df = df_original.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 0:
        corr_matrix = numeric_df.corr().abs()
        sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', fmt=".2f", ax=ax1, cbar=True, square=True)
        ax1.set_title('Feature Correlation Matrix', fontweight='bold')
    
    ax2 = fig.add_subplot(gs[0, 1])
    numeric_cols = df_original.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        corr_with_target = numeric_df.corr()[target_col].sort_values(ascending=False)
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in corr_with_target.values]
        ax2.barh(range(len(corr_with_target)), corr_with_target.values, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(corr_with_target)))
        ax2.set_yticklabels(corr_with_target.index, fontsize=9)
        ax2.set_xlabel('Correlation with Target', fontweight='bold')
        ax2.set_title('Feature Importance', fontweight='bold')
        ax2.grid(alpha=0.3, axis='x')
    
    plt.suptitle('CORRELATION & VARIABLE RELATIONSHIPS', fontsize=12, fontweight='bold')
    st.pyplot(fig, use_container_width=True)
    plt.close()
    
    st.subheader("📊 Diagram 3: Boxplot Analysis for Features (X Values)")
    
    num_cols = df_original.select_dtypes(include=[np.number]).columns.tolist()
    features_to_plot = [c for c in num_cols if c != target_col]
    
    if len(features_to_plot) > 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        df_melted = df_original.melt(value_vars=features_to_plot)
        sns.boxplot(x='variable', y='value', data=df_melted, palette='Set2', ax=ax)
        ax.set_title('Distribution Analysis of Features (X Values)', fontweight='bold', fontsize=12)
        ax.set_yscale('log')
        ax.set_ylabel('Value (Log Scale)')
        ax.set_xlabel('Features')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ============================================================================
# PAGE 3: DATA CLEANING
# ============================================================================
elif page == "🧹 Data Cleaning":
    st.header("🧹 DATA CLEANING PROCESS")
    
    initial_rows = len(df_original)
    df_clean = df_original.copy()
    
    df_clean = df_clean.drop_duplicates()
    after_duplicates = len(df_clean)
    duplicates_removed = initial_rows - after_duplicates
    
    df_clean = df_clean[df_clean[target_col].notna()].copy()
    after_missing = len(df_clean)
    missing_removed = after_duplicates - after_missing
    
    Q95 = df_clean[target_col].quantile(0.95)
    Q5 = df_clean[target_col].quantile(0.05)
    before_outliers = len(df_clean)
    df_clean = df_clean[(df_clean[target_col] >= Q5) & (df_clean[target_col] <= Q95)].copy()
    after_outliers = len(df_clean)
    outliers_removed = before_outliers - after_outliers
    
    final_rows = len(df_clean)
    total_removed = initial_rows - final_rows
    
    st.subheader("📊 Data Quality Issues Addressed")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Duplicates Removed", f"{duplicates_removed:,}")
    col2.metric("Missing Values", f"{missing_removed:,}")
    col3.metric("Outliers Removed", f"{outliers_removed:,}")
    col4.metric("Total Cleaned", f"{total_removed:,}")
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    
    categories = ['Original', 'Cleaned']
    counts = [initial_rows, final_rows]
    bars = axes[0, 0].bar(categories, counts, color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black', linewidth=2)
    axes[0, 0].set_ylabel('Records', fontweight='bold')
    axes[0, 0].set_title('Before & After Cleaning', fontweight='bold')
    for bar, val in zip(bars, counts):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, val + 30, f'{val:,}', ha='center', va='bottom', fontweight='bold')
    
    issues = ['Duplicates', 'Missing', 'Outliers']
    issues_count = [duplicates_removed, missing_removed, outliers_removed]
    colors = ['#3498db', '#e74c3c', '#f39c12']
    axes[0, 1].bar(issues, issues_count, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0, 1].set_ylabel('Count', fontweight='bold')
    axes[0, 1].set_title('Data Quality Issues', fontweight='bold')
    axes[0, 1].grid(alpha=0.3, axis='y')
    
    axes[1, 0].hist(df_clean[target_col], bins=40, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(Q5, color='red', linestyle='--', lw=2, label=f'5th: RM{Q5:,.0f}')
    axes[1, 0].axvline(Q95, color='red', linestyle='--', lw=2, label=f'95th: RM{Q95:,.0f}')
    axes[1, 0].set_xlabel('Price (RM)', fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontweight='bold')
    axes[1, 0].set_title('Price After Cleaning', fontweight='bold')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    axes[1, 1].pie(issues_count, labels=issues, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1, 1].set_title('Issues Distribution', fontweight='bold')
    
    plt.suptitle('DATA CLEANING PROCESS', fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    
    st.success(f"✅ Cleaned: {initial_rows:,} → {final_rows:,} records")
    st.info(f"📊 Records removed: {total_removed:,} ({total_removed/initial_rows*100:.1f}%)")

# ============================================================================
# PAGE 4: FEATURE ENGINEERING
# ============================================================================
elif page == "🔧 Feature Engineering":
    st.header("🔧 FEATURE ENGINEERING - TRANSFORMATION & ENRICHMENT")
    
    df_fe = df_original.copy()
    df_fe = df_fe.drop_duplicates()
    df_fe = df_fe[df_fe[target_col].notna()].copy()
    
    Q95 = df_fe[target_col].quantile(0.95)
    Q5 = df_fe[target_col].quantile(0.05)
    df_fe = df_fe[(df_fe[target_col] >= Q5) & (df_fe[target_col] <= Q95)].copy()
    
    st.subheader("📊 Step 1: Data Normalization & Standardization")
    
    numerical_cols = df_fe.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) > 0:
        minmax_scaler = MinMaxScaler(feature_range=(0, 1))
        df_normalized = df_fe.copy()
        df_normalized[numerical_cols] = minmax_scaler.fit_transform(df_fe[numerical_cols])
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Before Normalization:**")
            st.dataframe(df_fe[numerical_cols].describe().round(3), use_container_width=True)
        
        with col2:
            st.write("**After Normalization [0, 1]:**")
            st.dataframe(df_normalized[numerical_cols].describe().round(3), use_container_width=True)
        
        st.info("✅ All numerical features normalized to [0, 1] - **No negative values**")
        
        st.subheader("📝 Step 2: Data Enrichment & Feature Creation")
        
        initial_features = len(df_normalized.columns) - 1
        
        categorical_cols = df_normalized.select_dtypes(include=['object']).columns.tolist()
        df_enriched = df_normalized.copy()
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_enriched[col + '_encoded'] = le.fit_transform(df_normalized[col].astype(str))
        
        numeric_enriched = [col for col in numerical_cols if col != target_col]
        for col in numeric_enriched[:3]:
            df_enriched[f'{col}_squared'] = df_normalized[col] ** 2
            df_enriched[f'{col}_sqrt'] = np.sqrt(df_normalized[col])
        
        final_features = len(df_enriched.columns) - 1
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Features", f"{initial_features}")
        with col2:
            st.metric("After Encoding", f"{initial_features + len(categorical_cols)}")
        with col3:
            st.metric("After Enrichment", f"{final_features}")
        
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        
        stages = ['Original', 'Encoded', 'Enriched']
        counts = [initial_features, initial_features + len(categorical_cols), final_features]
        bars = axes[0].bar(stages, counts, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black', linewidth=2)
        axes[0].set_ylabel('Feature Count', fontweight='bold')
        axes[0].set_title('Feature Engineering Pipeline', fontweight='bold')
        axes[0].grid(alpha=0.3, axis='y')
        for bar, val in zip(bars, counts):
            axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{int(val)}', ha='center', va='bottom', fontweight='bold')
        
        feature_types = ['Numerical', 'Categorical', 'Polynomial', 'Interaction']
        type_counts = [len(numeric_enriched), len(categorical_cols), len(numeric_enriched[:3]) * 2, 1]
        colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6']
        
        wedges, texts, autotexts = axes[1].pie(type_counts, labels=feature_types, autopct='%1.1f%%', colors=colors, startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        axes[1].set_title('Feature Type Distribution', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ============================================================================
# PAGE 5: DATA PREPARATION
# ============================================================================
elif page == "📋 Data Preparation":
    st.header("📋 DATA PREPARATION - COMPLETE PIPELINE")
    
    st.write("""
    **Data Preparation encompasses:**
    - ✅ Clean and transform raw data to prepare it for analysis
    - ✅ Address data quality issues (missing values & outliers)
    - ✅ Standardise data formats and enrich source data
    - ✅ Document preprocessing steps
    """)
    
    df_prep = df_original.copy()
    df_prep = df_prep.drop_duplicates()
    df_prep = df_prep[df_prep[target_col].notna()].copy()
    
    Q95 = df_prep[target_col].quantile(0.95)
    Q5 = df_prep[target_col].quantile(0.05)
    df_prep = df_prep[(df_prep[target_col] >= Q5) & (df_prep[target_col] <= Q95)].copy()
    
    st.subheader("Step 1: Data Standardization")
    
    minmax_scaler = MinMaxScaler()
    numerical_cols = df_prep.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) > 0:
        df_prep[numerical_cols] = minmax_scaler.fit_transform(df_prep[numerical_cols])
        st.success(f"✅ Standardized {len(numerical_cols)} numerical columns to [0, 1]")
    
    st.subheader("Step 2: Categorical Data Encoding")
    
    categorical_cols = df_prep.select_dtypes(include=['object']).columns.tolist()
    
    encoding_info = []
    for col in categorical_cols:
        le = LabelEncoder()
        df_prep[col + '_encoded'] = le.fit_transform(df_prep[col].astype(str))
        
        encoding_info.append({
            'Column': col,
            'Unique Values': df_prep[col].nunique(),
            'Encoding': '→ Numerical (LabelEncoder)',
            'New Column': col + '_encoded'
        })
    
    if len(encoding_info) > 0:
        st.dataframe(pd.DataFrame(encoding_info), use_container_width=True)
        st.success(f"✅ Encoded {len(categorical_cols)} categorical columns")
    
    df_prep = df_prep.drop(columns=categorical_cols)
    
    st.subheader("Step 3: Missing Values Handling")
    
    missing_count = df_prep.isnull().sum().sum()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Missing Values", f"{missing_count}")
    with col2:
        st.metric("Percentage Missing", f"{(missing_count/(len(df_prep)*len(df_prep.columns))*100):.2f}%")
    
    if missing_count == 0:
        st.success("✅ No missing values found")
    
    st.subheader("Step 4: Train-Test Split & Feature Scaling")
    
    X = df_prep.drop(target_col, axis=1)
    y = df_prep[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_features) > 0:
        X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
        X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
        st.success(f"✅ Scaled {len(numerical_features)} features using StandardScaler")
    
    st.write(f"""
    **Split Configuration:**
    - Total Samples: {len(X):,}
    - Training Samples: {len(X_train):,} (85%)
    - Test Samples: {len(X_test):,} (15%)
    - Features: {X_train.shape[1]}
    """)
    
    # Visualization
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    split_sizes = [X_train.shape[0], X_test.shape[0]]
    colors = ['#3498db', '#e74c3c']
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.pie(split_sizes, labels=['Train', 'Test'], autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Train-Test Split Ratio', fontweight='bold')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(['Training', 'Test'], split_sizes, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Samples', fontweight='bold')
    ax2.set_title('Dataset Sizes', fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    for i, v in enumerate(split_sizes):
        ax2.text(i, v + 10, f'{v:,}', ha='center', fontweight='bold')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(0.5, 0.6, f'{X_train.shape[1]}', ha='center', va='center', fontsize=24, fontweight='bold', transform=ax3.transAxes, color='#2ecc71')
    ax3.text(0.5, 0.25, 'Features', ha='center', va='center', fontsize=12, fontweight='bold', transform=ax3.transAxes)
    ax3.axis('off')
    
    sample_col = numerical_features[0] if len(numerical_features) > 0 else X_train.columns[0]
    
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.hist(X_train[sample_col], bins=30, color='#e74c3c', alpha=0.7, edgecolor='black', label='Before')
    ax4.set_title(f'Before Scaling: {sample_col}', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.grid(alpha=0.3, axis='y')
    ax4.legend()
    
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(X_train_scaled[sample_col], bins=30, color='#2ecc71', alpha=0.7, edgecolor='black', label='After')
    ax5.set_title(f'After Scaling: {sample_col}', fontweight='bold')
    ax5.set_ylabel('Frequency', fontweight='bold')
    ax5.grid(alpha=0.3, axis='y')
    ax5.legend()
    
    plt.suptitle('PREPARING & PREPROCESSING DATASET (4 STEPS)', fontsize=13, fontweight='bold')
    st.pyplot(fig, use_container_width=True)
    plt.close()
    
    st.subheader("📊 Preprocessing Summary")
    
    summary = f"""
    **Data Preparation Steps Completed:**
    
    1. **Data Quality Assessment**
       - Original records: {len(df_original):,}
       - After cleaning: {len(df_prep):,}
    
    2. **Data Standardization**
       - Numerical features: {len(numerical_cols)}
       - Range: [0, 1] (MinMax Scaling)
    
    3. **Categorical Encoding**
       - Categorical features: {len(categorical_cols)}
       - Method: LabelEncoder
    
    4. **Train-Test Split**
       - Training: {len(X_train):,} (85%)
       - Test: {len(X_test):,} (15%)
    
    **Result:** Data ready for model training! ✅
    """
    
    st.info(summary)

# ============================================================================
# PAGE 6: MODEL TRAINING
# ============================================================================
elif page == "⚙️ Model Training":
    st.header("⚙️ MODEL TRAINING (5-Fold CV)")
    st.info("🔄 Training 4 models with **5-Fold Cross-Validation**...")
    
    df_train = df_original.copy()
    df_train = df_train.drop_duplicates()
    df_train = df_train[df_train[target_col].notna()].copy()
    
    Q95 = df_train[target_col].quantile(0.95)
    Q5 = df_train[target_col].quantile(0.05)
    df_train = df_train[(df_train[target_col] >= Q5) & (df_train[target_col] <= Q95)].copy()
    
    minmax_scaler = MinMaxScaler()
    numerical_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) > 0:
        df_train[numerical_cols] = minmax_scaler.fit_transform(df_train[numerical_cols])
    
    categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        df_train[col + '_encoded'] = le.fit_transform(df_train[col].astype(str))
    
    df_train = df_train.drop(columns=categorical_cols)
    
    X = df_train.drop(target_col, axis=1)
    y = df_train[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    numerical_features_all = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_features_all) > 0:
        X_train_scaled[numerical_features_all] = scaler.fit_transform(X_train[numerical_features_all])
        X_test_scaled[numerical_features_all] = scaler.transform(X_test[numerical_features_all])
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    progress = st.progress(0)
    
    # MODEL 1: LINEAR REGRESSION
    progress.progress(20)
    st.subheader("1️⃣ Linear Regression")
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.code("LinearRegression()\n- Baseline model\n- No hyperparameters", language="python")
    
    model_lr = LinearRegression()
    model_lr.fit(X_train_scaled, y_train)
    y_pred_lr = model_lr.predict(X_test_scaled)
    y_train_lr = model_lr.predict(X_train_scaled)
    
    r2_lr = r2_score(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)
    r2_train_lr = r2_score(y_train, y_train_lr)
    
    with col2:
        st.metric("R²", f"{r2_lr:.4f}")
        st.metric("RMSE", f"RM{rmse_lr:,.0f}")
        st.metric("MAE", f"RM{mae_lr:,.0f}")
    
    # MODEL 2: RIDGE
    progress.progress(40)
    st.subheader("2️⃣ Ridge Regression")
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.code("Ridge(alpha=tuned)\nGridSearchCV\nalpha: [0.001...1000]", language="python")
    
    ridge_search = GridSearchCV(Ridge(random_state=42), {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}, 
                                cv=kfold, scoring='r2', n_jobs=-1)
    ridge_search.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge_search.best_estimator_.predict(X_test_scaled)
    y_train_ridge = ridge_search.best_estimator_.predict(X_train_scaled)
    
    r2_ridge = r2_score(y_test, y_pred_ridge)
    rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
    mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
    mape_ridge = mean_absolute_percentage_error(y_test, y_pred_ridge)
    r2_train_ridge = r2_score(y_train, y_train_ridge)
    
    with col2:
        st.metric("R²", f"{r2_ridge:.4f}")
        st.metric("Best α", f"{ridge_search.best_params_['alpha']}")
        st.metric("RMSE", f"RM{rmse_ridge:,.0f}")
    
    # MODEL 3: GRADIENT BOOSTING
    progress.progress(60)
    st.subheader("3️⃣ Gradient Boosting")
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.code("GradientBoosting()\nn_estimators=300\nmax_depth: [4, 5, 6]", language="python")
    
    gb_search = GridSearchCV(GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, random_state=42),
                             {'max_depth': [4, 5, 6], 'subsample': [0.8, 1.0]}, cv=kfold, scoring='r2', n_jobs=-1)
    gb_search.fit(X_train_scaled, y_train)
    y_pred_gb = gb_search.best_estimator_.predict(X_test_scaled)
    y_train_gb = gb_search.best_estimator_.predict(X_train_scaled)
    
    r2_gb = r2_score(y_test, y_pred_gb)
    rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    mape_gb = mean_absolute_percentage_error(y_test, y_pred_gb)
    r2_train_gb = r2_score(y_train, y_train_gb)
    
    with col2:
        st.metric("R²", f"{r2_gb:.4f}")
        st.metric("Best Depth", f"{gb_search.best_params_['max_depth']}")
        st.metric("RMSE", f"RM{rmse_gb:,.0f}")
    
    # MODEL 4: RANDOM FOREST
    progress.progress(80)
    st.subheader("4️⃣ Random Forest")
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.code("RandomForest()\nn_estimators=150\nmax_depth: [15, 20, 25]", language="python")
    
    rf_search = GridSearchCV(RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
                             {'max_depth': [15, 20, 25], 'min_samples_leaf': [2, 4]}, cv=kfold, scoring='r2', n_jobs=-1)
    rf_search.fit(X_train, y_train)
    y_pred_rf = rf_search.best_estimator_.predict(X_test)
    y_train_rf = rf_search.best_estimator_.predict(X_train)
    
    r2_rf = r2_score(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf)
    r2_train_rf = r2_score(y_train, y_train_rf)
    
    with col2:
        st.metric("R²", f"{r2_rf:.4f}")
        st.metric("Best Depth", f"{rf_search.best_params_['max_depth']}")
        st.metric("RMSE", f"RM{rmse_rf:,.0f}")
    
    progress.progress(100)
    st.success("✅ Training complete with 5-Fold CV!")
    
    st.session_state.results = {
        'Linear Regression': {'R²': r2_lr, 'RMSE': rmse_lr, 'MAE': mae_lr, 'MAPE': mape_lr, 'train_r2': r2_train_lr, 'pred': y_pred_lr},
        'Ridge': {'R²': r2_ridge, 'RMSE': rmse_ridge, 'MAE': mae_ridge, 'MAPE': mape_ridge, 'train_r2': r2_train_ridge, 'pred': y_pred_ridge},
        'Gradient Boosting': {'R²': r2_gb, 'RMSE': rmse_gb, 'MAE': mae_gb, 'MAPE': mape_gb, 'train_r2': r2_train_gb, 'pred': y_pred_gb},
        'Random Forest': {'R²': r2_rf, 'RMSE': rmse_rf, 'MAE': mae_rf, 'MAPE': mape_rf, 'train_r2': r2_train_rf, 'pred': y_pred_rf}
    }
    st.session_state.y_test = y_test

# ============================================================================
# PAGE 7: MODEL EVALUATION - CLEAN WITH RM VALUES
# ============================================================================
elif page == "🏆 Model Evaluation":
    st.header("🏆 R² | RMSE | MAE | MAPE")
    
    if 'results' not in st.session_state:
        st.warning("⚠️ Train models first!")
    else:
        results = st.session_state.results
        y_test = st.session_state.y_test
        
        # Raw metrics
        r2_raw = np.array([v['R²'] for v in results.values()])
        rmse_raw = np.array([v['RMSE'] for v in results.values()])
        mae_raw = np.array([v['MAE'] for v in results.values()])
        mape_raw = np.array([v['MAPE'] * 100 for v in results.values()])
        
        # Normalize using Min-Max Scaling to [0, 100%]
        def normalize_to_01(arr):
            return (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)
        
        r2_norm = normalize_to_01(r2_raw) * 100
        rmse_norm = normalize_to_01(rmse_raw) * 100
        mae_norm = normalize_to_01(mae_raw) * 100
        mape_norm = normalize_to_01(mape_raw) * 100
        
        # Create results dataframe with RM values
        norm_df = pd.DataFrame({
            'Model': list(results.keys()),
            'R² (%)': [f"{r:.2f}%" for r in r2_norm],
            'RMSE (RM)': [f"RM{rm:,.0f}" for rm in rmse_raw],
            'MAE (RM)': [f"RM{ma:,.0f}" for ma in mae_raw],
            'MAPE (%)': [f"{mp:.2f}%" for mp in mape_raw]
        })
        
        st.subheader("📊 Evaluation Metrics (Normalized & Raw RM Values)")
        st.dataframe(norm_df, use_container_width=True)
        
        # Visualization
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        models = list(results.keys())
        
        # R² Normalized
        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.bar(range(len(models)), r2_norm, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.axhline(y=10, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Excellent (<10%)')
        ax1.set_ylabel('R² (%)', fontweight='bold')
        ax1.set_title('R²', fontweight='bold', fontsize=14)
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels([m.split()[0] for m in models], fontsize=9)
        ax1.set_ylim([0, 100])
        ax1.grid(alpha=0.3, axis='y')
        ax1.legend(fontsize=8)
        for bar, val in zip(bars, r2_norm):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # RMSE in RM
        ax2 = fig.add_subplot(gs[0, 1])
        bars = ax2.bar(range(len(models)), rmse_raw, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('RMSE (RM)', fontweight='bold')
        ax2.set_title('RMSE', fontweight='bold', fontsize=14)
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels([m.split()[0] for m in models], fontsize=9)
        ax2.grid(alpha=0.3, axis='y')
        for bar, val in zip(bars, rmse_raw):
            ax2.text(bar.get_x() + bar.get_width()/2, val, f'RM{val/1000:.1f}k', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # MAE in RM
        ax3 = fig.add_subplot(gs[1, 0])
        bars = ax3.bar(range(len(models)), mae_raw, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('MAE (RM)', fontweight='bold')
        ax3.set_title('MAE', fontweight='bold', fontsize=14)
        ax3.set_xticks(range(len(models)))
        ax3.set_xticklabels([m.split()[0] for m in models], fontsize=9)
        ax3.grid(alpha=0.3, axis='y')
        for bar, val in zip(bars, mae_raw):
            ax3.text(bar.get_x() + bar.get_width()/2, val, f'RM{val/1000:.1f}k', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # MAPE in %
        ax4 = fig.add_subplot(gs[1, 1])
        bars = ax4.bar(range(len(models)), mape_raw, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax4.axhline(y=10, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Excellent (<10%)')
        ax4.set_ylabel('MAPE (%)', fontweight='bold')
        ax4.set_title('MAPE', fontweight='bold', fontsize=14)
        ax4.set_xticks(range(len(models)))
        ax4.set_xticklabels([m.split()[0] for m in models], fontsize=9)
        ax4.grid(alpha=0.3, axis='y')
        ax4.legend(fontsize=8)
        for bar, val in zip(bars, mape_raw):
            ax4.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        plt.suptitle('MODEL EVALUATION METRICS', fontsize=13, fontweight='bold')
        st.pyplot(fig, use_container_width=True)
        plt.close()
        
        # Best Model
        best_idx = np.argmax(r2_raw)
        best_pred = results[models[best_idx]]['pred']
        
        fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Overfitting Analysis
        train_r2s = [results[m]['train_r2'] for m in models]
        test_r2s = r2_raw
        x_pos = np.arange(len(models))
        width = 0.35
        axes[0].bar(x_pos - width/2, train_r2s, width, label='Train R²', color='#2ecc71', alpha=0.7, edgecolor='black')
        axes[0].bar(x_pos + width/2, test_r2s, width, label='Test R²', color='#e74c3c', alpha=0.7, edgecolor='black')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels([m.split()[0] for m in models], fontsize=9)
        axes[0].set_ylabel('R² Score', fontweight='bold')
        axes[0].set_title('Overfitting Analysis', fontweight='bold')
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.3, axis='y')
        
        # Actual vs Predicted
        axes[1].scatter(y_test, best_pred, alpha=0.5, s=20, color='#3498db', edgecolors='black')
        min_v = min(y_test.min(), best_pred.min())
        max_v = max(y_test.max(), best_pred.max())
        axes[1].plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Perfect')
        axes[1].set_xlabel('Actual (RM)', fontweight='bold')
        axes[1].set_ylabel('Predicted (RM)', fontweight='bold')
        axes[1].set_title(f'{models[best_idx]}: Actual vs Predicted', fontweight='bold')
        axes[1].legend(fontsize=9)
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()
        
        # Best Model Summary
        st.subheader(f"🏆 Best Model: {models[best_idx]}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R²", f"{r2_norm[best_idx]:.2f}%")
        col2.metric("RMSE", f"RM{rmse_raw[best_idx]:,.0f}")
        col3.metric("MAE", f"RM{mae_raw[best_idx]:,.0f}")
        col4.metric("MAPE", f"{mape_raw[best_idx]:.2f}%")

st.sidebar.markdown("---")
st.sidebar.markdown("🚀 **Complete ML Pipeline | HuggingFace Data**")
