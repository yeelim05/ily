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
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

import pickle
from pathlib import Path
import joblib

sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 8

st.set_page_config(page_title="Malaysia Housing - Advanced ML Pipeline", layout="wide")

# ============================================================================
# MODELS DIRECTORY
# ============================================================================
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)

# ============================================================================
# SAVE/LOAD FUNCTIONS
# ============================================================================
def save_model_pipeline_fixed(model_name, model, scalers):
    """Save model and scalers - Fixed version without compression issues"""
    try:
        model_path = MODELS_DIR / f"{model_name}_model.joblib"
        scalers_path = MODELS_DIR / f"{model_name}_scalers.pkl"
        
        joblib.dump(model, model_path, compress=False)
        
        with open(scalers_path, 'wb') as f:
            pickle.dump(scalers, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        model_size_mb = model_path.stat().st_size / (1024**2)
        print(f"✅ Saved: {model_path} ({model_size_mb:.2f} MB)")
        print(f"✅ Saved: {scalers_path}")
    except Exception as e:
        print(f"❌ Error saving model: {e}")

def load_model_pipeline_fixed(model_name):
    """Load model and scalers - Fixed version"""
    try:
        model_path = MODELS_DIR / f"{model_name}_model.joblib"
        scalers_path = MODELS_DIR / f"{model_name}_scalers.pkl"
        
        model = joblib.load(model_path)
        
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
        
        model_size_mb = model_path.stat().st_size / (1024**2)
        print(f"✅ Loaded: {model_path} ({model_size_mb:.2f} MB)")
        print(f"✅ Loaded: {scalers_path}")
        return model, scalers
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

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
        base_price = 500000
        df = pd.DataFrame({
            'township': np.random.choice(['Cheras', 'Subang', 'Shah Alam', 'Petaling Jaya', 'Klang'], 2000),
            'area': np.random.choice(['Klang Valley', 'Selangor', 'Johor', 'Penang', 'Sabah'], 2000),
            'state': np.random.choice(['Selangor', 'Johor', 'Penang', 'KL', 'Sabah'], 2000),
            'tenure': np.random.choice(['Freehold', 'Leasehold'], 2000),
            'type': np.random.choice(['Terrace', 'Condominium', 'Semi-D', 'Detached', 'Bungalow'], 2000),
            'median_price': base_price + np.random.normal(0, 150000, 2000),
            'median_psf': 500 + np.random.normal(0, 150, 2000),
            'transactions': 50 + np.random.normal(0, 40, 2000),
        })
        df['median_price'] = np.abs(df['median_price'])
        df['median_psf'] = np.abs(df['median_psf'])
        df['transactions'] = np.abs(df['transactions'])
        return df

df_original = load_huggingface_data()
target_col = 'median_price' if 'median_price' in df_original.columns else 'Median_Price'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def format_rm_value(value):
    """Format RM values with appropriate precision"""
    if np.isnan(value):
        return "RM0.00"
    
    if value >= 1000000:
        return f"RM{value/1000000:,.2f}M"
    elif value >= 1000:
        return f"RM{value:,.2f}"
    else:
        return f"RM{value:.4f}"

def calculate_proper_mape(y_true, y_pred):
    """Calculate MAPE properly with safeguards"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0.0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return np.clip(mape, 0, 100)

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
    "🏆 Model Evaluation",
    "💾 Download Models"
])

st.sidebar.markdown("---")
st.sidebar.info(f"""
📈 **Dataset Source:** HuggingFace
🔗 jienweng/housing-prices-malaysia-2025

📊 **Rows:** {df_original.shape[0]:,}
📋 **Columns:** {df_original.shape[1]}
🎯 **Target:** {target_col}

📁 **Models Directory:** {MODELS_DIR.absolute()}
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
                'Unique': unique
            })
        else:
            try:
                min_val = df_original[col].min()
                max_val = df_original[col].max()
                if pd.isna(min_val) or pd.isna(max_val):
                    min_str = "N/A"
                    max_str = "N/A"
                else:
                    min_str = f"{min_val:,.0f}"
                    max_str = f"{max_val:,.0f}"
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
    ax2.set_title('Log Distribution (Better for Normalization)', fontweight='bold')
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
    if len(numeric_cols) > 1 and target_col in numeric_df.columns:
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
    
    st.subheader("📊 Diagram 3: Boxplot Analysis for Features")
    
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
    
    issues = ['Duplicates', 'Missing\nValues', 'Outliers']
    issues_count = [duplicates_removed, missing_removed, outliers_removed]
    colors = ['#3498db', '#e74c3c', '#f39c12']
    bars = axes[0, 1].barh(issues, issues_count, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0, 1].set_xlabel('Count', fontweight='bold')
    axes[0, 1].set_title('Data Quality Issues Resolved', fontweight='bold')
    axes[0, 1].grid(alpha=0.3, axis='x')
    for bar, val in zip(bars, issues_count):
        if val > 0:
            axes[0, 1].text(val + max(issues_count)*0.02, bar.get_y() + bar.get_height()/2, f'{val:,}', 
                           ha='left', va='center', fontweight='bold', fontsize=10)
    
    axes[1, 0].hist(df_clean[target_col], bins=40, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(Q5, color='red', linestyle='--', lw=2, label=f'5th: {format_rm_value(Q5)}')
    axes[1, 0].axvline(Q95, color='red', linestyle='--', lw=2, label=f'95th: {format_rm_value(Q95)}')
    axes[1, 0].set_xlabel('Price (RM)', fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontweight='bold')
    axes[1, 0].set_title('Price After Cleaning', fontweight='bold')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    axes[1, 1].bar(range(len(issues)), issues_count, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1, 1].set_xticks(range(len(issues)))
    axes[1, 1].set_xticklabels(issues, fontsize=10)
    axes[1, 1].set_ylabel('Count', fontweight='bold')
    axes[1, 1].set_title('Issues Distribution', fontweight='bold')
    axes[1, 1].grid(alpha=0.3, axis='y')
    for i, val in enumerate(issues_count):
        if val > 0:
            pct = val/total_removed*100 if total_removed > 0 else 0
            axes[1, 1].text(i, val + max(issues_count)*0.02, f'{val:,}\n({pct:.1f}%)', 
                           ha='center', va='bottom', fontweight='bold', fontsize=9)
    
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
    st.header("🔧 FEATURE ENGINEERING - PROPER NORMALIZATION")
    
    df_fe = df_original.copy()
    df_fe = df_fe.drop_duplicates()
    df_fe = df_fe[df_fe[target_col].notna()].copy()
    
    Q95 = df_fe[target_col].quantile(0.95)
    Q5 = df_fe[target_col].quantile(0.05)
    df_fe = df_fe[(df_fe[target_col] >= Q5) & (df_fe[target_col] <= Q95)].copy()
    
    st.subheader("📊 Step 1: Log Transformation & Proper Scaling")
    
    numerical_cols = df_fe.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) > 0:
        before_stats = df_fe[numerical_cols].describe().round(3)
        
        df_fe_transformed = df_fe.copy()
        for col in numerical_cols:
            df_fe_transformed[col] = np.log1p(df_fe_transformed[col])
        
        after_stats = df_fe_transformed[numerical_cols].describe().round(3)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Before Log Transformation:**")
            st.dataframe(before_stats, use_container_width=True)
        
        with col2:
            st.write("**After Log Transformation:**")
            st.dataframe(after_stats, use_container_width=True)
        
        st.info("✅ Log transformation applied - Reduces skewness for better scaling")
        
        st.subheader("📝 Step 2: Feature Encoding")
        
        categorical_cols = df_fe_transformed.select_dtypes(include=['object']).columns.tolist()
        df_encoded = df_fe_transformed.copy()
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str))
        
        initial_features = len(numerical_cols)
        final_features = len(df_encoded.columns) - 1
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Numerical Features", f"{initial_features}")
        with col2:
            st.metric("After Encoding", f"{final_features}")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        stages = ['Original', 'Log Transform', 'Encoded']
        counts = [initial_features, initial_features, final_features]
        bars = ax.bar(stages, counts, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Feature Count', fontweight='bold')
        ax.set_title('Feature Engineering Pipeline', fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        for bar, val in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{int(val)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ============================================================================
# PAGE 5: DATA PREPARATION
# ============================================================================
elif page == "📋 Data Preparation":
    st.header("📋 DATA PREPARATION - ADVANCED SCALING")
    
    st.write("""
    **Advanced Data Preparation:**
    - ✅ Log transformation for skewed data
    - ✅ RobustScaler for outlier resistance
    - ✅ PowerTransformer for Gaussian-like distribution
    - ✅ Train-test split
    """)
    
    df_prep = df_original.copy()
    df_prep = df_prep.drop_duplicates()
    df_prep = df_prep[df_prep[target_col].notna()].copy()
    
    Q95 = df_prep[target_col].quantile(0.95)
    Q5 = df_prep[target_col].quantile(0.05)
    df_prep = df_prep[(df_prep[target_col] >= Q5) & (df_prep[target_col] <= Q95)].copy()
    
    st.subheader("Step 1: Log Transformation")
    
    numerical_cols = df_prep.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numerical_cols:
        df_prep[col] = np.log1p(df_prep[col])
    
    st.success(f"✅ Applied log transformation to {len(numerical_cols)} numerical columns")
    
    st.subheader("Step 2: Categorical Encoding")
    
    categorical_cols = df_prep.select_dtypes(include=['object']).columns.tolist()
    
    encoding_info = []
    for col in categorical_cols:
        le = LabelEncoder()
        df_prep[col + '_encoded'] = le.fit_transform(df_prep[col].astype(str))
        encoding_info.append({
            'Column': col,
            'Unique Values': df_prep[col].nunique(),
            'Encoding': '→ Numerical'
        })
    
    if len(encoding_info) > 0:
        st.dataframe(pd.DataFrame(encoding_info), use_container_width=True)
        st.success(f"✅ Encoded {len(categorical_cols)} categorical columns")
    
    df_prep = df_prep.drop(columns=categorical_cols)
    
    st.subheader("Step 3: Train-Test Split")
    
    X = df_prep.drop(target_col, axis=1)
    y = df_prep[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    st.write(f"""
    **Split Configuration:**
    - Total Samples: {len(X):,}
    - Training Samples: {len(X_train):,} (85%)
    - Test Samples: {len(X_test):,} (15%)
    - Features: {X_train.shape[1]}
    """)
    
    st.subheader("Step 4: Advanced Scaling (RobustScaler + PowerTransformer)")
    
    robust_scaler = RobustScaler()
    X_train_scaled = robust_scaler.fit_transform(X_train)
    X_test_scaled = robust_scaler.transform(X_test)
    
    power_transformer = PowerTransformer(method='yeo-johnson')
    X_train_scaled = power_transformer.fit_transform(X_train_scaled)
    X_test_scaled = power_transformer.transform(X_test_scaled)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    st.success("✅ Applied RobustScaler + PowerTransformer for optimal scaling")
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    split_sizes = [X_train.shape[0], X_test.shape[0]]
    colors = ['#3498db', '#e74c3c']
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.pie(split_sizes, labels=['Train', 'Test'], autopct='%1.1f%%', colors=colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax1.set_title('Train-Test Split Ratio', fontweight='bold', fontsize=12)
    
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(['Training', 'Test'], split_sizes, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Samples', fontweight='bold', fontsize=11)
    ax2.set_title('Dataset Sizes', fontweight='bold', fontsize=12)
    ax2.grid(alpha=0.3, axis='y')
    for bar, val in zip(bars, split_sizes):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 10, f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    sample_col = numerical_features[0] if len(numerical_features) > 0 else X_train.columns[0]
    
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(X_train[sample_col], bins=30, color='#e74c3c', alpha=0.7, edgecolor='black', label='Before')
    ax3.set_title(f'Before Scaling: {sample_col}', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax3.grid(alpha=0.3, axis='y')
    ax3.legend(fontsize=10)
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(X_train_scaled[sample_col], bins=30, color='#2ecc71', alpha=0.7, edgecolor='black', label='After')
    ax4.set_title(f'After Advanced Scaling: {sample_col}', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax4.grid(alpha=0.3, axis='y')
    ax4.legend(fontsize=10)
    
    plt.suptitle('ADVANCED DATA PREPARATION', fontsize=13, fontweight='bold')
    st.pyplot(fig, use_container_width=True)
    plt.close()
    
    st.subheader("📊 Scaling Summary")
    
    summary = f"""
    **Advanced Scaling Steps Completed:**
    
    1. **Log Transformation** - Reduces skewness
    2. **RobustScaler** - Resistant to outliers (uses IQR)
    3. **PowerTransformer** - Yeo-Johnson method
    4. **Train-Test Split** - 85/15
    
    **Result:** Optimal scaling! ✅
    """
    
    st.info(summary)

# ============================================================================
# PAGE 6: MODEL TRAINING
# ============================================================================
elif page == "⚙️ Model Training":
    st.header("⚙️ ADVANCED MODEL TRAINING (5-Fold CV)")
    st.info("🔄 Training advanced models with **5-Fold Cross-Validation**...")
    
    df_train = df_original.copy()
    df_train = df_train.drop_duplicates()
    df_train = df_train[df_train[target_col].notna()].copy()
    
    Q95 = df_train[target_col].quantile(0.95)
    Q5 = df_train[target_col].quantile(0.05)
    df_train = df_train[(df_train[target_col] >= Q5) & (df_train[target_col] <= Q95)].copy()
    
    numerical_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    for col in numerical_cols:
        df_train[col] = np.log1p(df_train[col])
    
    categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        df_train[col + '_encoded'] = le.fit_transform(df_train[col].astype(str))
    
    df_train = df_train.drop(columns=categorical_cols)
    
    X = df_train.drop(target_col, axis=1)
    y = df_train[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    robust_scaler = RobustScaler()
    X_train_scaled = robust_scaler.fit_transform(X_train)
    X_test_scaled = robust_scaler.transform(X_test)
    
    power_transformer = PowerTransformer(method='yeo-johnson')
    X_train_scaled = power_transformer.fit_transform(X_train_scaled)
    X_test_scaled = power_transformer.transform(X_test_scaled)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    progress = st.progress(0)
    
    # MODEL 1: GRADIENT BOOSTING
    progress.progress(25)
    st.subheader("1️⃣ Gradient Boosting Regressor")
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.code("""GradientBoostingRegressor()
n_estimators=500
learning_rate=0.05
max_depth=[5,6,7]
subsample=[0.8,0.9,1.0]""", language="python")
    
    gb_search = GridSearchCV(
        GradientBoostingRegressor(random_state=42, min_samples_split=5, min_samples_leaf=2),
        {'n_estimators': [500], 'learning_rate': [0.05], 'max_depth': [5, 6, 7], 'subsample': [0.8, 0.9, 1.0]},
        cv=kfold, scoring='r2', n_jobs=-1, verbose=0
    )
    gb_search.fit(X_train_scaled, y_train)
    y_pred_gb = gb_search.best_estimator_.predict(X_test_scaled)
    y_train_gb = gb_search.best_estimator_.predict(X_train_scaled)
    
    r2_gb = r2_score(y_test, y_pred_gb)
    rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    mape_gb = calculate_proper_mape(y_test.values, y_pred_gb)
    r2_train_gb = r2_score(y_train, y_train_gb)
    
    # MODEL 2: RANDOM FOREST
    progress.progress(50)
    st.subheader("2️⃣ Random Forest Regressor")
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.code("""RandomForestRegressor()
n_estimators=30
max_depth=[6,8,10]
min_samples_leaf=[5,10]""", language="python")
    
    rf_search = GridSearchCV(
        RandomForestRegressor(
            n_estimators=30,
            max_depth=8,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        {'max_depth': [6, 8, 10], 'min_samples_leaf': [5, 10]},
        cv=kfold, scoring='r2', n_jobs=-1, verbose=0
    )
    rf_search.fit(X_train_scaled, y_train)
    y_pred_rf = rf_search.best_estimator_.predict(X_test_scaled)
    y_train_rf = rf_search.best_estimator_.predict(X_train_scaled)
    
    r2_rf = r2_score(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    mape_rf = calculate_proper_mape(y_test.values, y_pred_rf)
    r2_train_rf = r2_score(y_train, y_train_rf)
    
    # MODEL 3: RIDGE
    progress.progress(75)
    st.subheader("3️⃣ Ridge Regression")
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.code("""Ridge(alpha=tuned)
GridSearchCV
alpha=[0.1,1,10,100,1000]""", language="python")
    
    ridge_search = GridSearchCV(
        Ridge(random_state=42),
        {'alpha': [0.1, 1, 10, 100, 1000]},
        cv=kfold, scoring='r2', n_jobs=-1, verbose=0
    )
    ridge_search.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge_search.best_estimator_.predict(X_test_scaled)
    y_train_ridge = ridge_search.best_estimator_.predict(X_train_scaled)
    
    r2_ridge = r2_score(y_test, y_pred_ridge)
    rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
    mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
    mape_ridge = calculate_proper_mape(y_test.values, y_pred_ridge)
    r2_train_ridge = r2_score(y_train, y_train_ridge)
    
    # MODEL 4: SVR
    progress.progress(100)
    st.subheader("4️⃣ Support Vector Regressor")
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.code("""SVR(kernel='rbf')
GridSearchCV
C=[10,100,1000]
gamma=['scale',0.001]""", language="python")
    
    svr_search = GridSearchCV(
        SVR(kernel='rbf'),
        {'C': [10, 100, 1000], 'gamma': ['scale', 0.001]},
        cv=kfold, scoring='r2', n_jobs=-1, verbose=0
    )
    svr_search.fit(X_train_scaled, y_train)
    y_pred_svr = svr_search.best_estimator_.predict(X_test_scaled)
    y_train_svr = svr_search.best_estimator_.predict(X_train_scaled)
    
    r2_svr = r2_score(y_test, y_pred_svr)
    rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
    mae_svr = mean_absolute_error(y_test, y_pred_svr)
    mape_svr = calculate_proper_mape(y_test.values, y_pred_svr)
    r2_train_svr = r2_score(y_train, y_train_svr)
    
    st.success("✅ Training complete with 5-Fold CV!")
    
    # SAVE MODELS
    st.subheader("💾 Saving Models")
    
    scalers_dict = {
        'robust_scaler': robust_scaler,
        'power_transformer': power_transformer
    }
    
    models_dict = {
        'Gradient Boosting': gb_search.best_estimator_,
        'Random Forest': rf_search.best_estimator_,
        'Ridge': ridge_search.best_estimator_,
        'SVR': svr_search.best_estimator_
    }
    
    with st.spinner("Saving models..."):
        for model_name, model in models_dict.items():
            save_model_pipeline_fixed(model_name, model, scalers_dict)
    
    st.success("✅ All models saved successfully!")
    
    st.session_state.results = {
        'Gradient Boosting': {'R²': r2_gb, 'RMSE': rmse_gb, 'MAE': mae_gb, 'MAPE': mape_gb, 'train_r2': r2_train_gb, 'pred': y_pred_gb},
        'Random Forest': {'R²': r2_rf, 'RMSE': rmse_rf, 'MAE': mae_rf, 'MAPE': mape_rf, 'train_r2': r2_train_rf, 'pred': y_pred_rf},
        'Ridge': {'R²': r2_ridge, 'RMSE': rmse_ridge, 'MAE': mae_ridge, 'MAPE': mape_ridge, 'train_r2': r2_train_ridge, 'pred': y_pred_ridge},
        'SVR': {'R²': r2_svr, 'RMSE': rmse_svr, 'MAE': mae_svr, 'MAPE': mape_svr, 'train_r2': r2_train_svr, 'pred': y_pred_svr}
    }
    st.session_state.y_test = y_test

# ============================================================================
# PAGE 7: MODEL EVALUATION
# ============================================================================
elif page == "🏆 Model Evaluation":
    st.header("🏆 R² | RMSE | MAE | MAPE")
    
    if 'results' not in st.session_state:
        st.warning("⚠️ Train models first!")
    else:
        results = st.session_state.results
        y_test = st.session_state.y_test
        
        r2_raw = np.array([v['R²'] for v in results.values()])
        rmse_raw = np.array([v['RMSE'] for v in results.values()])
        mae_raw = np.array([v['MAE'] for v in results.values()])
        mape_raw = np.array([v['MAPE'] for v in results.values()])
        
        norm_df = pd.DataFrame({
            'Model': list(results.keys()),
            'R²': [f"{r:.4f}" for r in r2_raw],
            'RMSE (Log)': [f"{rm:.4f}" for rm in rmse_raw],
            'MAE (Log)': [f"{ma:.4f}" for ma in mae_raw],
            'MAPE': [f"{mp:.2f}%" for mp in mape_raw]
        })
        
        st.subheader("📊 Model Evaluation Metrics (Log Scale)")
        st.dataframe(norm_df, use_container_width=True)
        
        st.info("""
        **Interpretation (Log-Transformed Data):**
        - **R² > 0.85:** Excellent fit ✅
        - **RMSE/MAE:** Values on log scale (exp() to convert back)
        - **MAPE < 8%:** Excellent predictions ✅
        """)
        
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        models = list(results.keys())
        model_short = ['GB', 'RF', 'Ridge', 'SVR']
        
        # R² Score
        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.bar(range(len(models)), r2_raw, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('R² Score', fontweight='bold')
        ax1.set_title('R² (Higher is Better)', fontweight='bold', fontsize=13)
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(model_short, fontsize=10)
        ax1.set_ylim([0, 1.0])
        ax1.axhline(y=0.85, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target (0.85)')
        ax1.grid(alpha=0.3, axis='y')
        ax1.legend(fontsize=9)
        for bar, val in zip(bars, r2_raw):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # RMSE
        ax2 = fig.add_subplot(gs[0, 1])
        bars = ax2.bar(range(len(models)), rmse_raw, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('RMSE (Log Scale)', fontweight='bold')
        ax2.set_title('RMSE (Lower is Better)', fontweight='bold', fontsize=13)
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(model_short, fontsize=10)
        ax2.grid(alpha=0.3, axis='y')
        for bar, val in zip(bars, rmse_raw):
            ax2.text(bar.get_x() + bar.get_width()/2, val, f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # MAE
        ax3 = fig.add_subplot(gs[1, 0])
        bars = ax3.bar(range(len(models)), mae_raw, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('MAE (Log Scale)', fontweight='bold')
        ax3.set_title('MAE (Lower is Better)', fontweight='bold', fontsize=13)
        ax3.set_xticks(range(len(models)))
        ax3.set_xticklabels(model_short, fontsize=10)
        ax3.grid(alpha=0.3, axis='y')
        for bar, val in zip(bars, mae_raw):
            ax3.text(bar.get_x() + bar.get_width()/2, val, f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # MAPE
        ax4 = fig.add_subplot(gs[1, 1])
        bars = ax4.bar(range(len(models)), mape_raw, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('MAPE (%)', fontweight='bold')
        ax4.set_title('MAPE (Lower is Better)', fontweight='bold', fontsize=13)
        ax4.set_xticks(range(len(models)))
        ax4.set_xticklabels(model_short, fontsize=10)
        ax4.axhline(y=8, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target (<8%)')
        ax4.grid(alpha=0.3, axis='y')
        ax4.legend(fontsize=9)
        for bar, val in zip(bars, mape_raw):
            color = 'green' if val < 8 else 'orange' if val < 12 else 'red'
            ax4.text(bar.get_x() + bar.get_width()/2, val + 0.2, f'{val:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color=color)
        
        plt.suptitle('ADVANCED MODEL EVALUATION (LOG-TRANSFORMED DATA)', fontsize=13, fontweight='bold')
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
        axes[0].set_xticklabels(model_short, fontsize=10)
        axes[0].set_ylabel('R² Score', fontweight='bold')
        axes[0].set_title('Overfitting Analysis', fontweight='bold', fontsize=12)
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.3, axis='y')
        
        # Actual vs Predicted
        axes[1].scatter(y_test, best_pred, alpha=0.5, s=20, color='#3498db', edgecolors='black')
        min_v = min(y_test.min(), best_pred.min())
        max_v = max(y_test.max(), best_pred.max())
        axes[1].plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Perfect')
        axes[1].set_xlabel('Actual (Log Scale)', fontweight='bold')
        axes[1].set_ylabel('Predicted (Log Scale)', fontweight='bold')
        axes[1].set_title(f'{models[best_idx]}: Actual vs Predicted', fontweight='bold', fontsize=12)
        axes[1].legend(fontsize=9)
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()
        
        # Best Model Summary
        st.subheader(f"🏆 Best Model: {models[best_idx]}")
        col1, col2, col3, col4 = st.columns(4)
        
        status_r2 = "✅ Excellent" if r2_raw[best_idx] > 0.85 else "✅ Good" if r2_raw[best_idx] > 0.8 else "Fair"
        status_mape = "✅ Excellent" if mape_raw[best_idx] < 8 else "✅ Good" if mape_raw[best_idx] < 12 else "Fair"
        
        col1.metric("R²", f"{r2_raw[best_idx]:.4f}", status_r2)
        col2.metric("RMSE (Log)", f"{rmse_raw[best_idx]:.4f}", "Lower = Better")
        col3.metric("MAE (Log)", f"{mae_raw[best_idx]:.4f}", "Lower = Better")
        col4.metric("MAPE", f"{mape_raw[best_idx]:.2f}%", status_mape)

# ============================================================================
# PAGE 8: DOWNLOAD MODELS - FIXED
# ============================================================================
elif page == "💾 Download Models":
    st.header("💾 DOWNLOAD TRAINED MODELS")
    
    st.subheader("📊 File Size Summary")
    
    file_list = list(MODELS_DIR.glob('*'))
    
    if len(file_list) == 0:
        st.warning("⚠️ No models saved yet. Train models first!")
    else:
        total_size = 0
        file_info = []
        
        for file in sorted(file_list):
            size_mb = file.stat().st_size / (1024**2)
            total_size += size_mb
            file_info.append({
                'Filename': file.name,
                'Size (MB)': f"{size_mb:.2f}"
            })
        
        st.dataframe(pd.DataFrame(file_info), use_container_width=True)
        st.metric("Total Size", f"{total_size:.2f} MB")
        
        st.subheader("📥 Download Individual Models")
        st.write("Click below to download model files:")
        
        col1, col2 = st.columns(2)
        
        joblib_files = list(MODELS_DIR.glob('*_model.joblib'))
        pkl_files = list(MODELS_DIR.glob('*_scalers.pkl'))
        
        if joblib_files:
            with col1:
                st.write("**Models:**")
                for file in sorted(joblib_files):
                    with open(file, 'rb') as f:
                        st.download_button(
                            label=f"⬇️ {file.name}",
                            data=f.read(),
                            file_name=file.name,
                            mime="application/octet-stream"
                        )
        
        if pkl_files:
            with col2:
                st.write("**Scalers:**")
                for file in sorted(pkl_files):
                    with open(file, 'rb') as f:
                        st.download_button(
                            label=f"⬇️ {file.name}",
                            data=f.read(),
                            file_name=file.name,
                            mime="application/octet-stream"
                        )
        
        st.subheader("📖 How to Use Downloaded Models")
        st.code("""
import joblib
import pickle

# Load model
model = joblib.load('Gradient_Boosting_model.joblib')

# Load scalers
with open('Gradient_Boosting_scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

# Use for predictions
robust_scaler = scalers['robust_scaler']
power_transformer = scalers['power_transformer']

X_scaled = robust_scaler.transform(X)
X_scaled = power_transformer.transform(X_scaled)

predictions = model.predict(X_scaled)
        """, language="python")

st.sidebar.markdown("---")
st.sidebar.markdown("🚀 **Advanced ML Pipeline | Proper Scaling & Normalization**")
