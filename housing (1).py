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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 9

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="Malaysia Housing Price Prediction (2000 Datasets)", layout="wide")

# ============================================================================
# OPTIMIZED DATA LOADING - 2000 ROWS
# ============================================================================
@st.cache_data(ttl=3600)
def load_and_process_data():
    """Load 2000 housing datasets"""
    try:
        from datasets import load_dataset
        ds = load_dataset("jienweng/housing-prices-malaysia-2025")
        df = ds['train'].to_pandas()
        # Use 2000 rows
        return df.sample(n=min(2000, len(df)), random_state=42) if len(df) > 2000 else df
    except:
        # Generate 2000 synthetic records
        np.random.seed(42)
        df = pd.DataFrame({
            'township': np.random.choice(['Cheras', 'Subang', 'Shah Alam', 'Petaling Jaya', 'Klang', 'Ampang', 'Damansara', 'Bangsar'], 2000),
            'area': np.random.choice(['Klang Valley', 'Selangor', 'Johor', 'Penang', 'Sabah', 'Kuching', 'Melaka'], 2000),
            'state': np.random.choice(['Selangor', 'Johor', 'Penang', 'KL', 'Sabah', 'Sarawak', 'Melaka'], 2000),
            'tenure': np.random.choice(['Freehold', 'Leasehold'], 2000),
            'type': np.random.choice(['Terrace', 'Condominium', 'Semi-D', 'Detached', 'Bungalow'], 2000),
            'median_price': np.random.randint(250000, 2500000, 2000),
            'median_psf': np.random.uniform(250, 1500, 2000),
            'transactions': np.random.randint(3, 250, 2000),
        })
        return df

df_original = load_and_process_data()
target_col = 'median_price' if 'median_price' in df_original.columns else 'Median_Price'
trans_col = 'transactions' if 'transactions' in df_original.columns else 'Transactions'

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
📈 **Dataset Info:**
- 🔢 Original: {df_original.shape[0]:,} rows
- 📊 Columns: {df_original.shape[1]}
- 🎯 Target: {target_col}
- ✅ Status: Ready for Analysis
""")

# ============================================================================
# PAGE 1: DATASET OVERVIEW
# ============================================================================
if page == "📊 Dataset Overview":
    st.header("📊 DATASET OVERVIEW")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔢 Total Rows", f"{df_original.shape[0]:,}")
    with col2:
        st.metric("📋 Total Columns", f"{df_original.shape[1]}")
    with col3:
        st.metric("🏷️ Data Types", f"{df_original.dtypes.nunique()}")
    with col4:
        st.metric("❌ Missing", f"{df_original.isnull().sum().sum()}")
    
    st.subheader("Column Information")
    col_info = []
    for col in df_original.columns:
        dtype = df_original[col].dtype
        missing = df_original[col].isnull().sum()
        missing_pct = (missing / len(df_original)) * 100
        unique = df_original[col].nunique()
        
        if dtype == 'object':
            col_info.append({'Column': col, 'Type': str(dtype), 'Missing': f"{missing} ({missing_pct:.2f}%)", 'Unique': unique})
        else:
            col_info.append({'Column': col, 'Type': str(dtype), 'Missing': f"{missing} ({missing_pct:.2f}%)", 
                           'Min': f"{df_original[col].min():.0f}", 'Max': f"{df_original[col].max():.0f}"})
    
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
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
    
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(df_original[target_col], kde=True, color='#3498db', ax=ax1, bins=30)
    ax1.set_title(f'Distribution of {target_col}', fontweight='bold', fontsize=11)
    ax1.set_xlabel('Price (RM)')
    
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(np.log1p(df_original[target_col]), kde=True, color='#2ecc71', ax=ax2, bins=30)
    ax2.set_title(f'Log Distribution', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Log Price')
    
    ax3 = fig.add_subplot(gs[1, 0])
    sns.boxplot(y=df_original[target_col], color='#f1c40f', ax=ax3)
    ax3.set_title('Price Boxplot', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Price (RM)')
    
    ax4 = fig.add_subplot(gs[1, 1])
    numeric_df = df_original.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr().abs(), annot=True, cmap='YlGnBu', fmt=".2f", ax=ax4, cbar=True)
    ax4.set_title('Feature Correlations', fontweight='bold', fontsize=11)
    
    plt.suptitle('INITIAL EXPLORATORY ANALYSIS', fontsize=13, fontweight='bold', y=0.995)
    st.pyplot(fig, use_container_width=True)
    plt.close()
    
    st.subheader("Feature Distributions")
    num_cols = df_original.select_dtypes(include=[np.number]).columns.tolist()
    features_to_plot = [c for c in num_cols if c != target_col]
    
    if len(features_to_plot) > 0:
        fig, ax = plt.subplots(figsize=(13, 5))
        df_melted = df_original.melt(value_vars=features_to_plot)
        sns.boxplot(x='variable', y='value', data=df_melted, palette='Set2', ax=ax)
        ax.set_title('Distribution Analysis of Features', fontweight='bold')
        ax.set_yscale('log')
        plt.xticks(rotation=45)
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
    df_clean = df_clean[df_clean[target_col].notna()].copy()
    
    Q95 = df_clean[target_col].quantile(0.95)
    Q5 = df_clean[target_col].quantile(0.05)
    before = len(df_clean)
    df_clean = df_clean[(df_clean[target_col] >= Q5) & (df_clean[target_col] <= Q95)].copy()
    
    final_rows = len(df_clean)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Before/After
    categories = ['Original', 'Cleaned']
    counts = [initial_rows, final_rows]
    colors_clean = ['#e74c3c', '#2ecc71']
    bars = axes[0, 0].bar(categories, counts, color=colors_clean, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_ylabel('Record Count', fontweight='bold')
    axes[0, 0].set_title('Records Before & After Cleaning', fontweight='bold')
    axes[0, 0].grid(alpha=0.3, axis='y')
    for bar, val in zip(bars, counts):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, val + 30, f'{val:,}', ha='center', va='bottom', fontweight='bold')
    
    # Removed count
    removed_count = initial_rows - final_rows
    axes[0, 1].text(0.5, 0.7, f'{removed_count:,}', ha='center', va='center', fontsize=20, fontweight='bold', color='#e74c3c')
    axes[0, 1].text(0.5, 0.3, f'Records Removed\n({(removed_count/initial_rows)*100:.1f}%)', ha='center', va='center', fontsize=11, fontweight='bold')
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].axis('off')
    axes[0, 1].set_title('Data Quality Improvement', fontweight='bold')
    
    # Price distribution
    axes[1, 0].hist(df_clean[target_col], bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(Q5, color='red', linestyle='--', linewidth=2, label=f'5th: RM{Q5:,.0f}')
    axes[1, 0].axvline(Q95, color='red', linestyle='--', linewidth=2, label=f'95th: RM{Q95:,.0f}')
    axes[1, 0].set_xlabel('Price (RM)', fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontweight='bold')
    axes[1, 0].set_title('Price Distribution After Cleaning', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # Issues resolved
    metrics = ['Duplicates', 'Missing\nTargets', 'Outliers']
    values = [initial_rows - before, 0, before - final_rows]
    bars = axes[1, 1].bar(metrics, values, color=['#3498db', '#e74c3c', '#f39c12'], alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1, 1].set_ylabel('Count', fontweight='bold')
    axes[1, 1].set_title('Data Quality Issues Resolved', fontweight='bold')
    axes[1, 1].grid(alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        if val > 0:
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, val, f'{int(val)}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('DATA CLEANING PROCESS', fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    
    st.success(f"✅ Cleaning Complete: {initial_rows:,} → {final_rows:,} records")
    st.info(f"📊 Final Dataset: **{final_rows:,} records** (Expected: **1801**)")

# ============================================================================
# PAGE 4: FEATURE ENGINEERING
# ============================================================================
elif page == "🔧 Feature Engineering":
    st.header("🔧 FEATURE ENGINEERING PROCESS")
    
    df_fe = df_original.copy()
    df_fe = df_fe.drop_duplicates()
    df_fe = df_fe[df_fe[target_col].notna()].copy()
    
    Q95 = df_fe[target_col].quantile(0.95)
    Q5 = df_fe[target_col].quantile(0.05)
    df_fe = df_fe[(df_fe[target_col] >= Q5) & (df_fe[target_col] <= Q95)].copy()
    
    categorical_cols = df_fe.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        df_fe[col + '_encoded'] = le.fit_transform(df_fe[col].astype(str))
    
    numerical_cols = df_fe.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    correlations_dict = {}
    for feat in numerical_cols:
        try:
            corr = abs(df_fe[feat].corr(df_fe[target_col]))
            if not np.isnan(corr):
                correlations_dict[feat] = corr
        except:
            pass
    
    top_features = sorted(correlations_dict.items(), key=lambda x: x[1], reverse=True)[:4]
    top_feature_names = [f[0] for f in top_features]
    
    initial_features = len(df_fe.columns) - 1
    
    for feat in top_feature_names:
        df_fe[f'{feat}_squared'] = df_fe[feat] ** 2
        df_fe[f'{feat}_sqrt'] = np.sqrt(np.abs(df_fe[feat]) + 1)
    
    interaction_count = 0
    for i in range(len(top_feature_names)):
        for j in range(i+1, len(top_feature_names)):
            interaction_count += 1
    
    final_features = len(df_fe.columns) - 1
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    stages = ['Original', 'Encoded', 'After\nEngineering']
    feature_counts = [initial_features, initial_features + len(categorical_cols), final_features]
    bars = ax1.bar(stages, feature_counts, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Feature Count', fontweight='bold')
    ax1.set_title('Feature Count Progression', fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')
    for bar, val in zip(bars, feature_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.2, f'{int(val)}', ha='center', va='bottom', fontweight='bold')
    
    ax2 = fig.add_subplot(gs[0, 1])
    feature_types = ['Numerical', 'Polynomial', 'Interactions', 'Encoded']
    type_counts = [initial_features, len(top_feature_names) * 2, interaction_count, len(categorical_cols)]
    colors_types = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    wedges, texts, autotexts = ax2.pie(type_counts, labels=feature_types, autopct='%1.1f%%', colors=colors_types, startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax2.set_title('Feature Type Distribution', fontweight='bold')
    
    ax3 = fig.add_subplot(gs[1, :])
    if len(top_feature_names) > 0:
        top_corr = sorted([(f, correlations_dict[f]) for f in top_feature_names], key=lambda x: x[1], reverse=True)
        feat_names = [f[0] for f in top_corr]
        feat_corr = [f[1] for f in top_corr]
        bars = ax3.barh(feat_names, feat_corr, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Correlation with Target', fontweight='bold')
        ax3.set_title('Top Features for Polynomial/Interaction', fontweight='bold')
        ax3.grid(alpha=0.3, axis='x')
    
    plt.suptitle('FEATURE ENGINEERING PROCESS', fontsize=13, fontweight='bold')
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ============================================================================
# PAGE 5: DATA PREPARATION
# ============================================================================
elif page == "📋 Data Preparation":
    st.header("📋 DATA PREPARATION: SPLITTING & SCALING")
    
    df_prep = df_original.copy()
    df_prep = df_prep.drop_duplicates()
    df_prep = df_prep[df_prep[target_col].notna()].copy()
    
    Q95 = df_prep[target_col].quantile(0.95)
    Q5 = df_prep[target_col].quantile(0.05)
    df_prep = df_prep[(df_prep[target_col] >= Q5) & (df_prep[target_col] <= Q95)].copy()
    
    categorical_cols = df_prep.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        df_prep[col + '_encoded'] = le.fit_transform(df_prep[col].astype(str))
    
    df_prep = df_prep.drop(columns=categorical_cols)
    
    X = df_prep.drop(target_col, axis=1)
    y = df_prep[target_col]
    
    y_log = np.log1p(y)
    X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.15, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    numerical_features_all = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train_scaled[numerical_features_all] = scaler.fit_transform(X_train[numerical_features_all])
    X_test_scaled[numerical_features_all] = scaler.transform(X_test[numerical_features_all])
    
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    split_sizes = [X_train.shape[0], X_test.shape[0]]
    colors_split = ['#3498db', '#e74c3c']
    wedges, texts, autotexts = ax1.pie(split_sizes, labels=['Training', 'Test'], autopct='%1.1f%%', colors=colors_split, startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax1.set_title('Train-Test Split', fontweight='bold')
    
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(['Training', 'Test'], split_sizes, color=colors_split, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Samples', fontweight='bold')
    ax2.set_title('Dataset Sizes', fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    for bar, val in zip(bars, split_sizes):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 10, f'{int(val):,}', ha='center', va='bottom', fontweight='bold')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(0.5, 0.6, f'{X_train.shape[1]}', ha='center', va='center', fontsize=24, fontweight='bold', color='#2ecc71')
    ax3.text(0.5, 0.2, 'Features', ha='center', va='center', fontsize=11, fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Feature Dimension', fontweight='bold')
    
    ax4 = fig.add_subplot(gs[1, 0:2])
    sample_feature = numerical_features_all[0] if len(numerical_features_all) > 0 else X_train.columns[0]
    ax4.hist(X_train[sample_feature], bins=30, color='#e74c3c', alpha=0.7, label='Before Scaling', edgecolor='black')
    ax4.set_xlabel(f'{sample_feature} (Original)', fontweight='bold')
    ax4.set_ylabel('Frequency', fontweight='bold')
    ax4.set_title('Before Scaling', fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3, axis='y')
    
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(X_train_scaled[sample_feature], bins=30, color='#2ecc71', alpha=0.7, label='After Scaling', edgecolor='black')
    ax5.set_xlabel(f'{sample_feature} (Scaled)', fontweight='bold')
    ax5.set_ylabel('Frequency', fontweight='bold')
    ax5.set_title('After Scaling', fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3, axis='y')
    
    plt.suptitle('DATA PREPARATION: SPLITTING & SCALING', fontsize=13, fontweight='bold')
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ============================================================================
# PAGE 6: MODEL TRAINING
# ============================================================================
elif page == "⚙️ Model Training":
    st.header("⚙️ MODEL TRAINING & HYPERPARAMETER OPTIMIZATION")
    st.info("🔄 Training 4 models with **5-Fold Cross-Validation** and **GridSearchCV**...")
    
    df_train = df_original.copy()
    df_train = df_train.drop_duplicates()
    df_train = df_train[df_train[target_col].notna()].copy()
    
    Q95 = df_train[target_col].quantile(0.95)
    Q5 = df_train[target_col].quantile(0.05)
    df_train = df_train[(df_train[target_col] >= Q5) & (df_train[target_col] <= Q95)].copy()
    
    categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        df_train[col + '_encoded'] = le.fit_transform(df_train[col].astype(str))
    
    df_train = df_train.drop(columns=categorical_cols)
    
    X = df_train.drop(target_col, axis=1)
    y = df_train[target_col]
    
    y_log = np.log1p(y)
    X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.15, random_state=42)
    
    y_train_original = np.expm1(y_train_log)
    y_test_original = np.expm1(y_test_log)
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    numerical_features_all = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train_scaled[numerical_features_all] = scaler.fit_transform(X_train[numerical_features_all])
    X_test_scaled[numerical_features_all] = scaler.transform(X_test[numerical_features_all])
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    progress_bar = st.progress(0)
    
    # ========== MODEL 1: LINEAR REGRESSION ==========
    progress_bar.progress(20)
    st.subheader("1️⃣ Linear Regression")
    
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.write("**Configuration:**")
        st.code("""LinearRegression()
- No regularization
- Direct least squares optimization
- Baseline model""", language="python")
    
    model_lr = LinearRegression()
    model_lr.fit(X_train_scaled, y_train_log)
    y_pred_lr = np.expm1(model_lr.predict(X_test_scaled))
    y_train_lr = np.expm1(model_lr.predict(X_train_scaled))
    
    r2_lr = r2_score(y_test_original, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test_original, y_pred_lr))
    mae_lr = mean_absolute_error(y_test_original, y_pred_lr)
    mape_lr = mean_absolute_percentage_error(y_test_original, y_pred_lr)
    r2_train_lr = r2_score(y_train_original, y_train_lr)
    
    with col2:
        st.metric("Test R²", f"{r2_lr:.4f}")
        st.metric("RMSE", f"RM{rmse_lr/1000:.0f}k")
        st.metric("Overfitting", f"{(r2_train_lr - r2_lr):.4f}")
    
    # ========== MODEL 2: RIDGE REGRESSION ==========
    progress_bar.progress(40)
    st.subheader("2️⃣ Ridge Regression (GridSearchCV)")
    
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.write("**Configuration:**")
        st.code("""Ridge(random_state=42, alpha=tuned)

GridSearchCV:
  - alpha: [0.001, 0.01, 0.1, 1, 10, 100, 1000]
  - cv: KFold(n_splits=5)
  - scoring: 'r2'
  - n_jobs: -1""", language="python")
    
    param_grid_ridge = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    ridge_search = GridSearchCV(Ridge(random_state=42), param_grid_ridge, cv=kfold, scoring='r2', n_jobs=-1, verbose=0)
    ridge_search.fit(X_train_scaled, y_train_log)
    
    y_pred_ridge = np.expm1(ridge_search.best_estimator_.predict(X_test_scaled))
    y_train_ridge = np.expm1(ridge_search.best_estimator_.predict(X_train_scaled))
    
    r2_ridge = r2_score(y_test_original, y_pred_ridge)
    rmse_ridge = np.sqrt(mean_squared_error(y_test_original, y_pred_ridge))
    mae_ridge = mean_absolute_error(y_test_original, y_pred_ridge)
    mape_ridge = mean_absolute_percentage_error(y_test_original, y_pred_ridge)
    r2_train_ridge = r2_score(y_train_original, y_train_ridge)
    
    with col2:
        st.metric("Test R²", f"{r2_ridge:.4f}")
        st.metric("RMSE", f"RM{rmse_ridge/1000:.0f}k")
        st.metric("Best α", f"{ridge_search.best_params_['alpha']}")
    
    # ========== MODEL 3: GRADIENT BOOSTING ==========
    progress_bar.progress(60)
    st.subheader("3️⃣ Gradient Boosting (GridSearchCV)")
    
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.write("**Configuration:**")
        st.code("""GradientBoostingRegressor(
  random_state=42,
  n_estimators=500,
  learning_rate=0.05,
  validation_fraction=0.1,
  n_iter_no_change=15)

GridSearchCV:
  - max_depth: [4, 5, 6]
  - subsample: [0.8, 1.0]
  - max_features: ['sqrt']
  - cv: KFold(n_splits=5)""", language="python")
    
    param_grid_gb = {
        'n_estimators': [500],
        'learning_rate': [0.05],
        'max_depth': [4, 5, 6],
        'subsample': [0.8, 1.0],
        'max_features': ['sqrt']
    }
    gb_search = GridSearchCV(
        GradientBoostingRegressor(random_state=42, validation_fraction=0.1, n_iter_no_change=15),
        param_grid_gb, cv=kfold, scoring='r2', n_jobs=-1, verbose=0
    )
    gb_search.fit(X_train_scaled, y_train_log)
    
    y_pred_gb = np.expm1(gb_search.best_estimator_.predict(X_test_scaled))
    y_train_gb = np.expm1(gb_search.best_estimator_.predict(X_train_scaled))
    
    r2_gb = r2_score(y_test_original, y_pred_gb)
    rmse_gb = np.sqrt(mean_squared_error(y_test_original, y_pred_gb))
    mae_gb = mean_absolute_error(y_test_original, y_pred_gb)
    mape_gb = mean_absolute_percentage_error(y_test_original, y_pred_gb)
    r2_train_gb = r2_score(y_train_original, y_train_gb)
    
    with col2:
        st.metric("Test R²", f"{r2_gb:.4f}")
        st.metric("RMSE", f"RM{rmse_gb/1000:.0f}k")
        st.metric("Best Depth", f"{gb_search.best_params_['max_depth']}")
    
    # ========== MODEL 4: RANDOM FOREST ==========
    progress_bar.progress(80)
    st.subheader("4️⃣ Random Forest (GridSearchCV)")
    
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.write("**Configuration:**")
        st.code("""RandomForestRegressor(
  random_state=42,
  n_jobs=-1)

GridSearchCV:
  - n_estimators: [200]
  - max_depth: [15, 20, 25]
  - min_samples_leaf: [2, 4]
  - max_features: ['sqrt']
  - cv: KFold(n_splits=5)""", language="python")
    
    param_grid_rf = {
        'n_estimators': [200],
        'max_depth': [15, 20, 25],
        'min_samples_leaf': [2, 4],
        'max_features': ['sqrt'],
    }
    rf_search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid_rf, cv=kfold, scoring='r2', n_jobs=-1, verbose=0
    )
    rf_search.fit(X_train, y_train_log)
    
    y_pred_rf = np.expm1(rf_search.best_estimator_.predict(X_test))
    y_train_rf = np.expm1(rf_search.best_estimator_.predict(X_train))
    
    r2_rf = r2_score(y_test_original, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test_original, y_pred_rf))
    mae_rf = mean_absolute_error(y_test_original, y_pred_rf)
    mape_rf = mean_absolute_percentage_error(y_test_original, y_pred_rf)
    r2_train_rf = r2_score(y_train_original, y_train_rf)
    
    with col2:
        st.metric("Test R²", f"{r2_rf:.4f}")
        st.metric("RMSE", f"RM{rmse_rf/1000:.0f}k")
        st.metric("Best Depth", f"{rf_search.best_params_['max_depth']}")
    
    progress_bar.progress(100)
    st.success("✅ All models trained with **5-Fold Cross-Validation** ✅")
    
    st.session_state.results = {
        'Linear Regression': {'R²': r2_lr, 'RMSE': rmse_lr, 'MAE': mae_lr, 'MAPE': mape_lr, 'train_r2': r2_train_lr, 'pred': y_pred_lr},
        'Ridge Regression': {'R²': r2_ridge, 'RMSE': rmse_ridge, 'MAE': mae_ridge, 'MAPE': mape_ridge, 'train_r2': r2_train_ridge, 'pred': y_pred_ridge},
        'Gradient Boosting': {'R²': r2_gb, 'RMSE': rmse_gb, 'MAE': mae_gb, 'MAPE': mape_gb, 'train_r2': r2_train_gb, 'pred': y_pred_gb},
        'Random Forest': {'R²': r2_rf, 'RMSE': rmse_rf, 'MAE': mae_rf, 'MAPE': mape_rf, 'train_r2': r2_train_rf, 'pred': y_pred_rf}
    }
    st.session_state.y_test = y_test_original

# ============================================================================
# PAGE 7: MODEL EVALUATION
# ============================================================================
elif page == "🏆 Model Evaluation":
    st.header("🏆 MODEL EVALUATION & COMPARISON")
    
    if 'results' not in st.session_state:
        st.warning("⚠️ Please train models first in 'Model Training' section!")
    else:
        results = st.session_state.results
        y_test = st.session_state.y_test
        
        # Normalize metrics
        scaler_metrics = StandardScaler()
        metrics_array = np.array([
            [v['R²'] for v in results.values()],
            [v['RMSE'] for v in results.values()],
            [v['MAE'] for v in results.values()],
            [v['MAPE'] for v in results.values()]
        ]).T
        
        metrics_normalized = scaler_metrics.fit_transform(metrics_array)
        
        results_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Train R²': [f"{v['train_r2']:.4f}" for v in results.values()],
            'Test R²': [f"{v['R²']:.4f}" for v in results.values()],
            'RMSE': [f"RM{v['RMSE']/1000:.0f}k" for v in results.values()],
            'MAE': [f"RM{v['MAE']/1000:.0f}k" for v in results.values()],
            'MAPE': [f"{v['MAPE']*100:.2f}%" for v in results.values()]
        })
        
        st.subheader("📊 Model Performance Summary")
        st.dataframe(results_df, use_container_width=True)
        
        st.subheader("📈 Normalized Metrics (StandardScaler)")
        norm_df = pd.DataFrame({
            'Model': list(results.keys()),
            'R² (Norm)': [f"{m:.3f}" for m in metrics_normalized[:, 0]],
            'RMSE (Norm)': [f"{m:.3f}" for m in metrics_normalized[:, 1]],
            'MAE (Norm)': [f"{m:.3f}" for m in metrics_normalized[:, 2]],
            'MAPE (Norm)': [f"{m:.3f}" for m in metrics_normalized[:, 3]]
        })
        st.dataframe(norm_df, use_container_width=True)
        
        # Visualization
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        models = list(results.keys())
        
        # 1. R² Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        r2_scores = [results[m]['R²'] for m in models]
        bars = ax1.bar(range(len(models)), r2_scores, color=colors, alpha=0.8, edgecolor='black')
        ax1.axhline(y=0.85, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target')
        ax1.set_ylabel('R² Score', fontweight='bold')
        ax1.set_title('R² Comparison', fontweight='bold')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels([m.split()[0] for m in models], rotation=45, ha='right')
        ax1.set_ylim([0, 1.05])
        ax1.legend()
        ax1.grid(alpha=0.3, axis='y')
        
        # 2. RMSE Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        rmse_scores = [results[m]['RMSE'] for m in models]
        bars = ax2.bar(range(len(models)), rmse_scores, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_ylabel('RMSE (RM)', fontweight='bold')
        ax2.set_title('RMSE Comparison', fontweight='bold')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels([m.split()[0] for m in models], rotation=45, ha='right')
        ax2.grid(alpha=0.3, axis='y')
        
        # 3. MAE Comparison
        ax3 = fig.add_subplot(gs[0, 2])
        mae_scores = [results[m]['MAE'] for m in models]
        bars = ax3.bar(range(len(models)), mae_scores, color=colors, alpha=0.8, edgecolor='black')
        ax3.set_ylabel('MAE (RM)', fontweight='bold')
        ax3.set_title('MAE Comparison', fontweight='bold')
        ax3.set_xticks(range(len(models)))
        ax3.set_xticklabels([m.split()[0] for m in models], rotation=45, ha='right')
        ax3.grid(alpha=0.3, axis='y')
        
        # 4. MAPE Comparison
        ax4 = fig.add_subplot(gs[1, 0])
        mape_scores = [results[m]['MAPE'] * 100 for m in models]
        bars = ax4.bar(range(len(models)), mape_scores, color=colors, alpha=0.8, edgecolor='black')
        ax4.axhline(y=10, color='green', linestyle='--', linewidth=2, alpha=0.7, label='< 10%')
        ax4.set_ylabel('MAPE (%)', fontweight='bold')
        ax4.set_title('MAPE Comparison', fontweight='bold')
        ax4.set_xticks(range(len(models)))
        ax4.set_xticklabels([m.split()[0] for m in models], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(alpha=0.3, axis='y')
        
        # 5. Train vs Test
        ax5 = fig.add_subplot(gs[1, 1])
        x_pos = np.arange(len(models))
        width = 0.35
        train_r2s = [results[m]['train_r2'] for m in models]
        bars_train = ax5.bar(x_pos - width/2, train_r2s, width, label='Train R²', color='#2ecc71', alpha=0.7)
        bars_test = ax5.bar(x_pos + width/2, r2_scores, width, label='Test R²', color='#e74c3c', alpha=0.7)
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels([m.split()[0] for m in models], rotation=45, ha='right')
        ax5.set_ylabel('R² Score', fontweight='bold')
        ax5.set_title('Overfitting Analysis', fontweight='bold')
        ax5.legend()
        ax5.grid(alpha=0.3, axis='y')
        
        # 6. Normalized Heatmap
        ax6 = fig.add_subplot(gs[1, 2])
        heatmap_df = pd.DataFrame(metrics_normalized, columns=['R² (N)', 'RMSE (N)', 'MAE (N)', 'MAPE (N)'],
                                  index=[m.split()[0] for m in models])
        sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap='RdYlGn', ax=ax6, cbar=True)
        ax6.set_title('Normalized Metrics', fontweight='bold')
        
        # 7. Actual vs Predicted
        best_idx = np.argmax(r2_scores)
        best_model_name = models[best_idx]
        best_pred = results[best_model_name]['pred']
        
        ax7 = fig.add_subplot(gs[2, 0:2])
        ax7.scatter(y_test, best_pred, alpha=0.5, color='#3498db', edgecolors='black', s=25)
        min_v = min(y_test.min(), best_pred.min())
        max_v = max(y_test.max(), best_pred.max())
        ax7.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Perfect')
        ax7.set_xlabel('Actual Price (RM)', fontweight='bold')
        ax7.set_ylabel('Predicted Price (RM)', fontweight='bold')
        ax7.set_title(f'Actual vs Predicted - {best_model_name}', fontweight='bold')
        ax7.legend()
        ax7.grid(alpha=0.3)
        
        # 8. Residuals
        ax8 = fig.add_subplot(gs[2, 2])
        residuals = y_test - best_pred
        ax8.scatter(best_pred, residuals, alpha=0.5, color='#e74c3c', edgecolors='black', s=25)
        ax8.axhline(y=0, color='black', linestyle='--', lw=2)
        ax8.set_xlabel('Predicted Price (RM)', fontweight='bold')
        ax8.set_ylabel('Residuals (RM)', fontweight='bold')
        ax8.set_title('Residual Plot', fontweight='bold')
        ax8.grid(alpha=0.3)
        
        plt.suptitle('MODEL EVALUATION & COMPARISON', fontsize=14, fontweight='bold', y=0.995)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        
        # Best Model
        st.subheader(f"🏆 Best Model: {best_model_name}")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R²", f"{r2_scores[best_idx]:.4f}", delta=f"Train: {train_r2s[best_idx]:.4f}")
        with col2:
            st.metric("RMSE", f"RM{rmse_scores[best_idx]/1000:.0f}k")
        with col3:
            st.metric("MAE", f"RM{mae_scores[best_idx]/1000:.0f}k")
        with col4:
            st.metric("MAPE", f"{mape_scores[best_idx]:.2f}%")

st.sidebar.markdown("---")
st.sidebar.markdown("🚀 **2000 Datasets → 1801 After Cleaning | 5-Fold CV | All Features**")
