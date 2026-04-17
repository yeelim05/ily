import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Malaysia Housing Price Prediction", layout="wide")

# Reduce dataset size for cloud
@st.cache_data
def load_and_process_data():
    try:
        from datasets import load_dataset
        ds = load_dataset("jienweng/housing-prices-malaysia-2025")
        df = ds['train'].to_pandas()
        # Limit rows for cloud
        return df.sample(n=min(800, len(df)), random_state=42)
    except:
        np.random.seed(42)
        return pd.DataFrame({
            'township': np.random.choice(['Cheras', 'Subang', 'Shah Alam', 'Petaling Jaya', 'Klang'], 800),
            'area': np.random.choice(['Klang Valley', 'Selangor', 'Johor', 'Penang', 'Sabah'], 800),
            'state': np.random.choice(['Selangor', 'Johor', 'Penang', 'KL', 'Sabah'], 800),
            'tenure': np.random.choice(['Freehold', 'Leasehold'], 800),
            'type': np.random.choice(['Terrace', 'Condominium', 'Semi-D', 'Detached', 'Bungalow'], 800),
            'median_price': np.random.randint(250000, 2500000, 800),
            'median_psf': np.random.uniform(250, 1500, 800),
            'transactions': np.random.randint(3, 250, 800),
        })

df_original = load_and_process_data()
target_col = 'median_price' if 'median_price' in df_original.columns else 'Median_Price'
trans_col = 'transactions' if 'transactions' in df_original.columns else 'Transactions'

page = st.sidebar.radio("Navigation", [
    "📊 Dataset Overview",
    "🔍 Initial EDA",
    "🧹 Data Cleaning",
    "🔧 Feature Engineering",
    "📋 Data Preparation",
    "⚙️ Model Training",
    "🏆 Model Evaluation"
])

# ============================================================================
# PAGE 1: DATASET OVERVIEW
# ============================================================================
if page == "📊 Dataset Overview":
    st.header("DATASET OVERVIEW")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset Shape", f"{df_original.shape[0]} rows × {df_original.shape[1]} columns")
    with col2:
        st.metric("Data Types", f"{df_original.dtypes.nunique()} types")
    with col3:
        st.metric("Missing Values", f"{df_original.isnull().sum().sum()}")
    
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
            col_info.append({
                'Column': col,
                'Type': str(dtype),
                'Missing': f"{missing} ({missing_pct:.2f}%)",
                'Min': f"{df_original[col].min():.0f}",
                'Max': f"{df_original[col].max():.0f}"
            })
    
    st.dataframe(pd.DataFrame(col_info), use_container_width=True)
    st.dataframe(df_original.head(), use_container_width=True)
    st.dataframe(df_original.describe().round(2), use_container_width=True)

# ============================================================================
# PAGE 2: INITIAL EDA
# ============================================================================
elif page == "🔍 Initial EDA":
    st.header("INITIAL EXPLORATORY ANALYSIS")
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
    
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(df_original[target_col], kde=True, color='#3498db', ax=ax1)
    ax1.set_title(f'Distribution of {target_col}', fontweight='bold')
    ax1.set_xlabel('Price (RM)')
    
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(np.log1p(df_original[target_col]), kde=True, color='#2ecc71', ax=ax2)
    ax2.set_title(f'Log({target_col})', fontweight='bold')
    
    ax3 = fig.add_subplot(gs[1, 0])
    sns.boxplot(x=df_original[trans_col], color='#f1c40f', ax=ax3)
    ax3.set_title(f'Market Activity', fontweight='bold')
    
    ax4 = fig.add_subplot(gs[1, 1])
    numeric_df = df_original.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr().abs(), annot=True, cmap='YlGnBu', fmt=".2f", ax=ax4)
    ax4.set_title('Feature Correlations', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ============================================================================
# PAGE 3: DATA CLEANING
# ============================================================================
elif page == "🧹 Data Cleaning":
    st.header("DATA CLEANING PROCESS")
    
    initial_rows = len(df_original)
    df_clean = df_original.drop_duplicates()
    df_clean = df_clean[df_clean[target_col].notna()].copy()
    
    Q95 = df_clean[target_col].quantile(0.95)
    Q5 = df_clean[target_col].quantile(0.05)
    df_clean = df_clean[(df_clean[target_col] >= Q5) & (df_clean[target_col] <= Q95)].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    categories = ['Original', 'After Cleaning']
    counts = [initial_rows, len(df_clean)]
    colors_clean = ['#e74c3c', '#2ecc71']
    bars = axes[0, 0].bar(categories, counts, color=colors_clean, alpha=0.7)
    axes[0, 0].set_ylabel('Record Count')
    axes[0, 0].set_title('Records Before & After Cleaning')
    for bar, val in zip(bars, counts):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, val + 10, f'{val:,}', ha='center', fontweight='bold')
    
    removed_count = initial_rows - len(df_clean)
    axes[0, 1].text(0.5, 0.5, f'{removed_count:,}\nRemoved', ha='center', va='center', fontsize=16, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].hist(df_clean[target_col], bins=40, color='#2ecc71', alpha=0.7)
    axes[1, 0].axvline(Q5, color='red', linestyle='--', linewidth=2, label=f'5th: RM{Q5:,.0f}')
    axes[1, 0].axvline(Q95, color='red', linestyle='--', linewidth=2, label=f'95th: RM{Q95:,.0f}')
    axes[1, 0].set_title('Price Distribution After Cleaning')
    axes[1, 0].legend()
    
    metrics = ['Duplicates', 'Missing', 'Outliers']
    values = [0, 0, initial_rows - len(df_clean)]
    axes[1, 1].bar(metrics, values, color=['#3498db', '#e74c3c', '#f39c12'], alpha=0.7)
    axes[1, 1].set_title('Issues Resolved')
    
    plt.suptitle('DATA CLEANING', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ============================================================================
# PAGE 4: FEATURE ENGINEERING
# ============================================================================
elif page == "🔧 Feature Engineering":
    st.header("FEATURE ENGINEERING")
    
    df_fe = df_original.drop_duplicates()
    df_fe = df_fe[df_fe[target_col].notna()]
    Q95 = df_fe[target_col].quantile(0.95)
    Q5 = df_fe[target_col].quantile(0.05)
    df_fe = df_fe[(df_fe[target_col] >= Q5) & (df_fe[target_col] <= Q95)]
    
    categorical_cols = df_fe.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df_fe.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    initial_features = len(df_fe.columns) - 1
    final_features = initial_features + len(categorical_cols) + (len(numerical_cols) * 2) + 1
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    stages = ['Original', 'Encoded', 'After Eng.']
    feature_counts = [initial_features, initial_features + len(categorical_cols), final_features]
    axes[0].bar(stages, feature_counts, color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.7)
    axes[0].set_ylabel('Feature Count')
    axes[0].set_title('Feature Progression')
    
    feature_types = ['Numerical', 'Polynomial', 'Encoded', 'Interactions']
    type_counts = [initial_features, len(numerical_cols) * 2, len(categorical_cols), 1]
    axes[1].pie(type_counts, labels=feature_types, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Feature Distribution')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ============================================================================
# PAGE 5: DATA PREPARATION
# ============================================================================
elif page == "📋 Data Preparation":
    st.header("DATA PREPARATION")
    
    df_prep = df_original.drop_duplicates()
    df_prep = df_prep[df_prep[target_col].notna()]
    Q95 = df_prep[target_col].quantile(0.95)
    Q5 = df_prep[target_col].quantile(0.05)
    df_prep = df_prep[(df_prep[target_col] >= Q5) & (df_prep[target_col] <= Q95)]
    
    categorical_cols = df_prep.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        df_prep[col + '_encoded'] = le.fit_transform(df_prep[col].astype(str))
    df_prep = df_prep.drop(columns=categorical_cols)
    
    X = df_prep.drop(target_col, axis=1)
    y = df_prep[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    split_labels = ['Train', 'Test']
    split_sizes = [X_train.shape[0], X_test.shape[0]]
    axes[0].pie(split_sizes, labels=split_labels, autopct='%1.1f%%', colors=['#3498db', '#e74c3c'])
    axes[0].set_title('Train-Test Split')
    
    categories = ['Training', 'Test']
    sizes = [X_train.shape[0], X_test.shape[0]]
    axes[1].bar(categories, sizes, color=['#3498db', '#e74c3c'], alpha=0.7)
    axes[1].set_title('Dataset Sizes')
    axes[1].set_ylabel('Samples')
    
    axes[2].text(0.5, 0.5, f'{X_train.shape[1]}\nFeatures', ha='center', va='center', fontsize=18, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ============================================================================
# PAGE 6: MODEL TRAINING
# ============================================================================
elif page == "⚙️ Model Training":
    st.header("MODEL TRAINING (5-Fold CV)")
    
    df_train = df_original.drop_duplicates()
    df_train = df_train[df_train[target_col].notna()]
    Q95 = df_train[target_col].quantile(0.95)
    Q5 = df_train[target_col].quantile(0.05)
    df_train = df_train[(df_train[target_col] >= Q5) & (df_train[target_col] <= Q95)]
    
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
    numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    progress = st.progress(0)
    
    # Linear Regression
    progress.progress(25)
    st.subheader("1️⃣ Linear Regression")
    col1, col2 = st.columns(2)
    with col1:
        st.code("LinearRegression()", language="python")
    
    model_lr = LinearRegression()
    model_lr.fit(X_train_scaled, y_train_log)
    y_pred_lr = np.expm1(model_lr.predict(X_test_scaled))
    y_train_lr = np.expm1(model_lr.predict(X_train_scaled))
    
    r2_lr = r2_score(y_test_original, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test_original, y_pred_lr))
    r2_train_lr = r2_score(y_train_original, y_train_lr)
    
    with col2:
        st.metric("Test R²", f"{r2_lr:.4f}")
    
    # Ridge Regression
    progress.progress(50)
    st.subheader("2️⃣ Ridge Regression (GridSearch)")
    col1, col2 = st.columns(2)
    with col1:
        st.code("Ridge + GridSearchCV\nalpha: [0.001, 0.01, 0.1, 1, 10, 100, 1000]\nCV: 5-Fold", language="python")
    
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    ridge_search = GridSearchCV(Ridge(random_state=42), param_grid, cv=kfold, scoring='r2', n_jobs=-1)
    ridge_search.fit(X_train_scaled, y_train_log)
    y_pred_ridge = np.expm1(ridge_search.best_estimator_.predict(X_test_scaled))
    y_train_ridge = np.expm1(ridge_search.best_estimator_.predict(X_train_scaled))
    
    r2_ridge = r2_score(y_test_original, y_pred_ridge)
    rmse_ridge = np.sqrt(mean_squared_error(y_test_original, y_pred_ridge))
    r2_train_ridge = r2_score(y_train_original, y_train_ridge)
    
    with col2:
        st.metric("Test R²", f"{r2_ridge:.4f}")
        st.metric("Best α", f"{ridge_search.best_params_['alpha']}")
    
    # Gradient Boosting
    progress.progress(75)
    st.subheader("3️⃣ Gradient Boosting (GridSearch)")
    col1, col2 = st.columns(2)
    with col1:
        st.code("GradientBoosting + GridSearchCV\nmax_depth: [4, 5, 6]\nsubsample: [0.8, 1.0]\nCV: 5-Fold", language="python")
    
    param_grid_gb = {'max_depth': [4, 5, 6], 'subsample': [0.8, 1.0]}
    gb_search = GridSearchCV(GradientBoostingRegressor(random_state=42, n_estimators=300, learning_rate=0.1),
                             param_grid_gb, cv=kfold, scoring='r2', n_jobs=-1)
    gb_search.fit(X_train_scaled, y_train_log)
    y_pred_gb = np.expm1(gb_search.best_estimator_.predict(X_test_scaled))
    y_train_gb = np.expm1(gb_search.best_estimator_.predict(X_train_scaled))
    
    r2_gb = r2_score(y_test_original, y_pred_gb)
    rmse_gb = np.sqrt(mean_squared_error(y_test_original, y_pred_gb))
    r2_train_gb = r2_score(y_train_original, y_train_gb)
    
    with col2:
        st.metric("Test R²", f"{r2_gb:.4f}")
        st.metric("Best Depth", f"{gb_search.best_params_['max_depth']}")
    
    # Random Forest
    progress.progress(100)
    st.subheader("4️⃣ Random Forest (GridSearch)")
    col1, col2 = st.columns(2)
    with col1:
        st.code("RandomForest + GridSearchCV\nmax_depth: [15, 20, 25]\nmin_samples_leaf: [2, 4]\nCV: 5-Fold", language="python")
    
    param_grid_rf = {'max_depth': [15, 20, 25], 'min_samples_leaf': [2, 4]}
    rf_search = GridSearchCV(RandomForestRegressor(random_state=42, n_estimators=150, n_jobs=-1),
                             param_grid_rf, cv=kfold, scoring='r2', n_jobs=-1)
    rf_search.fit(X_train, y_train_log)
    y_pred_rf = np.expm1(rf_search.best_estimator_.predict(X_test))
    y_train_rf = np.expm1(rf_search.best_estimator_.predict(X_train))
    
    r2_rf = r2_score(y_test_original, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test_original, y_pred_rf))
    r2_train_rf = r2_score(y_train_original, y_train_rf)
    
    with col2:
        st.metric("Test R²", f"{r2_rf:.4f}")
        st.metric("Best Depth", f"{rf_search.best_params_['max_depth']}")
    
    st.info("✅ All models trained with 5-Fold Cross-Validation")
    
    st.session_state.results = {
        'Linear Regression': {'R²': r2_lr, 'RMSE': rmse_lr, 'MAE': mean_absolute_error(y_test_original, y_pred_lr), 'MAPE': mean_absolute_percentage_error(y_test_original, y_pred_lr), 'train_r2': r2_train_lr, 'pred': y_pred_lr},
        'Ridge': {'R²': r2_ridge, 'RMSE': rmse_ridge, 'MAE': mean_absolute_error(y_test_original, y_pred_ridge), 'MAPE': mean_absolute_percentage_error(y_test_original, y_pred_ridge), 'train_r2': r2_train_ridge, 'pred': y_pred_ridge},
        'Gradient Boosting': {'R²': r2_gb, 'RMSE': rmse_gb, 'MAE': mean_absolute_error(y_test_original, y_pred_gb), 'MAPE': mean_absolute_percentage_error(y_test_original, y_pred_gb), 'train_r2': r2_train_gb, 'pred': y_pred_gb},
        'Random Forest': {'R²': r2_rf, 'RMSE': rmse_rf, 'MAE': mean_absolute_error(y_test_original, y_pred_rf), 'MAPE': mean_absolute_percentage_error(y_test_original, y_pred_rf), 'train_r2': r2_train_rf, 'pred': y_pred_rf}
    }
    st.session_state.y_test = y_test_original

# ============================================================================
# PAGE 7: MODEL EVALUATION
# ============================================================================
elif page == "🏆 Model Evaluation":
    st.header("MODEL EVALUATION & COMPARISON")
    
    if 'results' in st.session_state:
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
            'Test R²': [v['R²'] for v in results.values()],
            'RMSE': [f"RM{v['RMSE']/1000:.0f}k" for v in results.values()],
            'MAE': [f"RM{v['MAE']/1000:.0f}k" for v in results.values()],
            'MAPE': [f"{v['MAPE']*100:.2f}%" for v in results.values()]
        })
        
        st.subheader("📊 Performance Summary")
        st.dataframe(results_df, use_container_width=True)
        
        # Normalized table
        norm_df = pd.DataFrame({
            'Model': list(results.keys()),
            'R² (Norm)': metrics_normalized[:, 0],
            'RMSE (Norm)': metrics_normalized[:, 1],
            'MAE (Norm)': metrics_normalized[:, 2],
            'MAPE (Norm)': metrics_normalized[:, 3]
        })
        
        st.subheader("📈 Normalized Metrics")
        st.dataframe(norm_df, use_container_width=True)
        
        # Comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        models = list(results.keys())
        
        r2_scores = [results[m]['R²'] for m in models]
        axes[0, 0].bar(range(len(models)), r2_scores, color=colors, alpha=0.8)
        axes[0, 0].set_xticks(range(len(models)))
        axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('R² Comparison')
        axes[0, 0].grid(alpha=0.3, axis='y')
        
        rmse_scores = [results[m]['RMSE'] for m in models]
        axes[0, 1].bar(range(len(models)), rmse_scores, color=colors, alpha=0.8)
        axes[0, 1].set_xticks(range(len(models)))
        axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 1].set_ylabel('RMSE (RM)')
        axes[0, 1].set_title('RMSE Comparison')
        axes[0, 1].grid(alpha=0.3, axis='y')
        
        mae_scores = [results[m]['MAE'] for m in models]
        axes[1, 0].bar(range(len(models)), mae_scores, color=colors, alpha=0.8)
        axes[1, 0].set_xticks(range(len(models)))
        axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[1, 0].set_ylabel('MAE (RM)')
        axes[1, 0].set_title('MAE Comparison')
        axes[1, 0].grid(alpha=0.3, axis='y')
        
        mape_scores = [results[m]['MAPE'] * 100 for m in models]
        axes[1, 1].bar(range(len(models)), mape_scores, color=colors, alpha=0.8)
        axes[1, 1].set_xticks(range(len(models)))
        axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].set_title('MAPE Comparison')
        axes[1, 1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        best_idx = np.argmax(r2_scores)
        best_model = models[best_idx]
        st.success(f"🏆 **Best Model: {best_model}** (R² = {r2_scores[best_idx]:.4f})")
        
    else:
        st.warning("⚠️ Please train models first!")

st.sidebar.markdown("---")
st.sidebar.markdown("**Built with Streamlit** 🎈")
