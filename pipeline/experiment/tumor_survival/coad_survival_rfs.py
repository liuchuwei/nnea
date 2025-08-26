import numpy as np
import pandas as pd
import warnings
import toml
import os
import pickle
import random
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sksurv.ensemble import RandomSurvivalForest
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
import seaborn as sns
import nnea as na

warnings.filterwarnings('ignore')

print("🚀 Starting Random Survival Forest experiment...")

# Read configuration file
print("⚙️ Reading configuration file...")
try:
    config = toml.load("./config.toml")
    print("✅ Configuration file read successfully")
except Exception as e:
    print(f"❌ Configuration file reading failed: {e}")
    exit(1)


# Set all random seeds
def set_all_seeds(seed=42):
    """设置所有随机种子以确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# Set global random seed
print("🔧 Setting global random seed...")
set_all_seeds(config['global']['seed'])
print("✅ Global random seed set successfully")

# Create output directory
output_dir = config['global']['outdir']
os.makedirs(output_dir, exist_ok=True)
print(f"📁 Output directory created: {output_dir}")

# Data loading
print("📂 Loading data...")
try:
    nadata = na.nadata()
    nadata.load(filepath="./datasets/tumor_survival/TCGA_Colon_Cancer_survival.pkl")
    print("✅ Preprocessed nadata object loaded successfully, data shape:", nadata.X.shape)
except Exception as e:
    print(f"❌ Data loading failed: {e}")
    exit(1)

# Data preprocessing
print("🔧 Data preprocessing...")
X = nadata.X

# Use na.pp.fillna to handle missing values
if np.isnan(X).any():
    print("⚠️ NaN values detected in X, performing fill operation...")
    X = na.pp.fillna(X, method=config['data']['handle_missing'])
    print(f"   NaN count after filling: {np.isnan(X).sum()}")
else:
    print("✅ No NaN values detected in X")

# Update X in nadata
nadata.X = X

# Process survival data labels
print("🏷️ Processing survival data labels...")
try:
    nadata = na.pp.process_survival_data(
        nadata,
        os_col='OS',
        os_time_col='OS.time',
        time_unit='auto'
    )
    print("✅ Survival data processing completed")
except Exception as e:
    print(f"❌ Survival data processing failed: {e}")
    exit(1)

# Feature selection
if config['data']['feature_selection']:
    print("🔍 Performing feature selection...")
    try:
        nadata = na.fs.apply_feature_selection(
            nadata,
            method=config['data']['selection_method'],
            n_features=config['data']['n_features'],
            target_col='OS'  # Use survival event as target
        )
        print(f"✅ Feature selection completed, selected {nadata.X.shape[1]} features")
    except Exception as e:
        print(f"⚠️ Feature selection failed: {e}, using all features")

# Data splitting
print("✂️ Performing data splitting...")
try:
    nadata = na.pp.split_data(
        nadata,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state'],
        strategy="stratified" if config['data']['stratify'] else "random"
    )
    print("✅ Data splitting completed")

    # Get train and test indices
    train_indices = nadata.Model.get_indices('train')
    test_indices = nadata.Model.get_indices('test')

    # Get data based on indices
    X = nadata.X
    Meta = nadata.Meta

    # Get train data
    X_train = X[train_indices]
    Meta_train = Meta.iloc[train_indices]

    # Get test data
    X_test = X[test_indices]
    Meta_test = Meta.iloc[test_indices]

    print(f"📊 Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"📊 Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

except Exception as e:
    print(f"❌ Data splitting failed: {e}")
    exit(1)

# Prepare survival data for Random Survival Forest
print("🔄 Preparing survival data for Random Survival Forest...")


# For Random Survival Forest, we need to create a structured array with survival times and events
def prepare_survival_data(X, Meta, time_col, event_col):
    """准备生存分析数据"""
    # Get survival times and events
    times = Meta[time_col].values
    events = Meta[event_col].values

    # For sksurv RandomSurvivalForest, we need a structured array
    # Create structured array with dtype for survival data
    y = np.empty(len(times), dtype=[('Status', bool), ('Time', float)])
    y['Status'] = events.astype(bool)
    y['Time'] = times.astype(float)
    
    return X, y


# Prepare training data
X_train_rsf, y_train_rsf = prepare_survival_data(
    X_train, Meta_train,
    config['data']['time_col'],
    config['data']['event_col']
)

# Prepare test data
X_test_rsf, y_test_rsf = prepare_survival_data(
    X_test, Meta_test,
    config['data']['time_col'],
    config['data']['event_col']
)

print("✅ Survival data preparation completed")

# Hyperparameter tuning
print("🔍 Performing hyperparameter tuning...")
try:
    # Define parameter grid
    param_grid = {
        'n_estimators': config['rsf_model']['n_estimators_values'],
        'max_depth': config['rsf_model']['max_depth_values'] + [None],
        'min_samples_split': config['rsf_model']['min_samples_split_values'],
        'min_samples_leaf': config['rsf_model']['min_samples_leaf_values'],
        'max_features': config['rsf_model']['max_features_values'] + [None]
    }

    # Create base Random Survival Forest model
    base_rsf = RandomSurvivalForest(
        random_state=config['rsf_model']['random_state'],
        n_jobs=config['rsf_model']['n_jobs'],
        bootstrap=config['rsf_model']['bootstrap']
    )

    # Custom scoring function for survival analysis
    def c_index_scorer(estimator, X, y):
        """计算C-index评分"""
        try:
            # For sksurv RandomSurvivalForest, y is already a structured array
            # Get risk scores
            risk_scores = estimator.predict(X)
            # Calculate C-index using lifelines
            c_idx = concordance_index(y['Time'], risk_scores, y['Status'])
            return c_idx
        except:
            return 0.0

    # Grid search with cross-validation
    grid_search = GridSearchCV(
        base_rsf,
        param_grid,
        cv=config['rsf_model']['cv_folds'],
        scoring=c_index_scorer,
        n_jobs=config['rsf_model']['n_jobs'],
        verbose=1
    )

    # Fit grid search with structured array
    grid_search.fit(X_train_rsf, y_train_rsf)

    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best cross-validation score: {grid_search.best_score_:.4f}")

    # Get best model
    best_rsf = grid_search.best_estimator_

except Exception as e:
    print(f"⚠️ Hyperparameter tuning failed: {e}")
    print("🔄 Using default parameters...")

    # Use default parameters
    best_rsf = RandomSurvivalForest(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=config['rsf_model']['random_state'],
        n_jobs=config['rsf_model']['n_jobs'],
        bootstrap=config['rsf_model']['bootstrap']
    )

    # Fit model with structured array
    best_rsf.fit(X_train_rsf, y_train_rsf)

# Model evaluation
print("📊 Evaluating model performance...")

# Get predictions
train_risk_scores = best_rsf.predict(X_train_rsf)
test_risk_scores = best_rsf.predict(X_test_rsf)

# Calculate C-index
train_c_index = concordance_index(y_train_rsf['Time'], train_risk_scores, y_train_rsf['Status'])
test_c_index = concordance_index(y_test_rsf['Time'], test_risk_scores, y_test_rsf['Status'])

print(f"✅ Train C-index: {train_c_index:.4f}")
print(f"✅ Test C-index: {test_c_index:.4f}")

# Risk stratification
print("📈 Performing risk stratification...")

def perform_risk_stratification(risk_scores, times, events, method='median', n_groups=2):
    """执行风险分层"""
    # 验证输入数据
    if len(risk_scores) == 0 or len(times) == 0 or len(events) == 0:
        print("⚠️ Warning: Empty arrays detected in risk stratification")
        return None
    
    # 确保数据类型正确
    risk_scores = np.asarray(risk_scores, dtype=float)
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    
    # 检查是否有有效的风险分数差异
    if np.std(risk_scores) < 1e-10:
        print("⚠️ Warning: Risk scores have no variance, using random stratification")
        # 如果风险分数没有差异，使用随机分层
        np.random.seed(42)
        high_risk = np.random.choice([True, False], size=len(risk_scores), p=[0.5, 0.5])
        threshold = np.median(risk_scores)  # 设置默认阈值
    else:
        if method == 'median':
            threshold = np.median(risk_scores)
            high_risk = risk_scores > threshold
        elif method == 'quartile':
            threshold = np.percentile(risk_scores, 75)
            high_risk = risk_scores > threshold
        else:
            # Optimal cutpoint (simplified)
            threshold = np.median(risk_scores)
            high_risk = risk_scores > threshold

    # 检查分层结果
    if np.sum(high_risk) == 0 or np.sum(~high_risk) == 0:
        print("⚠️ Warning: All samples in one group, adjusting stratification")
        # 如果所有样本都在一组，调整分层
        high_risk = np.arange(len(risk_scores)) >= len(risk_scores) // 2

    # 获取各组的生存数据
    low_risk_times = times[~high_risk]
    low_risk_events = events[~high_risk]
    high_risk_times = times[high_risk]
    high_risk_events = events[high_risk]

    # 验证各组数据
    if len(low_risk_times) == 0 or len(high_risk_times) == 0:
        print("⚠️ Warning: One or both risk groups are empty")
        return None

    # Log-rank test
    try:
        lr_test = logrank_test(low_risk_times, high_risk_times,
                               event_observed_A=low_risk_events,
                               event_observed_B=high_risk_events)
        logrank_pvalue = lr_test.p_value
        logrank_statistic = lr_test.test_statistic
    except Exception as e:
        print(f"⚠️ Warning: Log-rank test failed: {e}")
        logrank_pvalue = np.nan
        logrank_statistic = np.nan

    return {
        'threshold': threshold,
        'high_risk': high_risk,
        'low_risk_times': low_risk_times,
        'low_risk_events': low_risk_events,
        'high_risk_times': high_risk_times,
        'high_risk_events': high_risk_events,
        'logrank_pvalue': logrank_pvalue,
        'logrank_statistic': logrank_statistic
    }

# Perform risk stratification on test set
stratification_results = perform_risk_stratification(
    test_risk_scores,
    y_test_rsf['Time'],
    y_test_rsf['Status'],
    method=config['evaluation']['stratification_method'],
    n_groups=config['evaluation']['n_groups']
)

if stratification_results is None:
    print("❌ Risk stratification failed")
    exit(1)

print(f"✅ Log-rank test p-value: {stratification_results['logrank_pvalue']:.4e}")
print(f"✅ Log-rank test statistic: {stratification_results['logrank_statistic']:.4f}")

# Visualization
print("🎨 Creating visualizations...")

# Set up plotting style
plt.style.use(config['visualization']['style'])
plt.rcParams['font.family'] = config['visualization']['font_family']
plt.rcParams['font.size'] = config['visualization']['font_size']

# 1. Survival curves
if config['evaluation']['plot_km_curves']:
    print("📊 Plotting Kaplan-Meier survival curves...")

    # 验证数据
    low_risk_times = stratification_results['low_risk_times']
    low_risk_events = stratification_results['low_risk_events']
    high_risk_times = stratification_results['high_risk_times']
    high_risk_events = stratification_results['high_risk_events']

    # 检查数据有效性
    if (len(low_risk_times) > 0 and len(high_risk_times) > 0 and 
        np.all(np.isfinite(low_risk_times)) and np.all(np.isfinite(high_risk_times))):
        
        fig, ax = plt.subplots(figsize=tuple(config['visualization']['figsize']))

        # Low risk group
        kmf_low = KaplanMeierFitter()
        kmf_low.fit(
            low_risk_times,
            low_risk_events,
            label='Low Risk'
        )
        kmf_low.plot(ax=ax, ci_show=config['evaluation']['confidence_intervals'])

        # High risk group
        kmf_high = KaplanMeierFitter()
        kmf_high.fit(
            high_risk_times,
            high_risk_events,
            label='High Risk'
        )
        kmf_high.plot(ax=ax, ci_show=config['evaluation']['confidence_intervals'])

        # Add log-rank test result
        p_value = stratification_results['logrank_pvalue']
        if np.isnan(p_value):
            p_text = "p = N/A"
        elif p_value < 0.0001:
            p_text = "p < 0.0001"
        else:
            p_text = f"p = {p_value:.4f}"

        ax.text(0.6, 0.3, f'Log-rank test\n{p_text}',
                transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Kaplan-Meier Survival Curves by Risk Group')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'survival_curves.png'),
                    dpi=config['visualization']['save_dpi'], bbox_inches='tight')
        plt.close()
        print("✅ Survival curves saved")
    else:
        print("⚠️ Warning: Invalid data for survival curves, skipping plot")

# 2. Risk score distribution
if config['evaluation']['plot_risk_distribution']:
    print("📊 Plotting risk score distribution...")

    # 验证风险分数数据
    if len(test_risk_scores) > 0 and np.all(np.isfinite(test_risk_scores)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Overall distribution
        ax1.hist(test_risk_scores, bins=30, alpha=0.7, color=config['visualization']['colors'][0])
        ax1.axvline(stratification_results['threshold'], color='red', linestyle='--',
                    label=f'Threshold: {stratification_results["threshold"]:.2f}')
        ax1.set_xlabel('Risk Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Risk Score Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Distribution by risk group
        low_risk_scores = test_risk_scores[~stratification_results['high_risk']]
        high_risk_scores = test_risk_scores[stratification_results['high_risk']]

        if len(low_risk_scores) > 0:
            ax2.hist(low_risk_scores, bins=20, alpha=0.7, label='Low Risk',
                     color=config['visualization']['colors'][1])
        if len(high_risk_scores) > 0:
            ax2.hist(high_risk_scores, bins=20, alpha=0.7, label='High Risk',
                     color=config['visualization']['colors'][2])
        
        ax2.set_xlabel('Risk Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Risk Score Distribution by Risk Group')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'risk_distribution.png'),
                    dpi=config['visualization']['save_dpi'], bbox_inches='tight')
        plt.close()
        print("✅ Risk distribution plot saved")
    else:
        print("⚠️ Warning: Invalid risk scores for distribution plot, skipping")

# 3. Feature importance
if config['evaluation']['plot_feature_importance']:
    print("📊 Plotting feature importance...")

    # sksurv RandomSurvivalForest不支持feature_importances_属性
    print("⚠️ Warning: sksurv RandomSurvivalForest does not support feature_importances_, skipping feature importance plot")
    print("   Note: Consider using sklearn.ensemble.RandomForestRegressor for feature importance analysis")

# 4. Model performance summary
print("📊 Creating model performance summary...")

# 验证数据用于绘图
valid_data = (len(stratification_results['low_risk_times']) > 0 and 
              len(stratification_results['high_risk_times']) > 0 and
              not np.isnan(train_c_index) and not np.isnan(test_c_index))

if valid_data:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # C-index comparison
    metrics = ['Train', 'Test']
    c_indices = [train_c_index, test_c_index]
    bars = ax1.bar(metrics, c_indices, color=config['visualization']['colors'][:2])
    ax1.set_ylabel('C-index')
    ax1.set_title('Model Performance (C-index)')
    ax1.set_ylim(0, 1)
    for bar, value in zip(bars, c_indices):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom')

    # Risk group sizes
    risk_groups = ['Low Risk', 'High Risk']
    group_sizes = [len(stratification_results['low_risk_times']),
                   len(stratification_results['high_risk_times'])]
    bars = ax2.bar(risk_groups, group_sizes, color=config['visualization']['colors'][1:3])
    ax2.set_ylabel('Number of Patients')
    ax2.set_title('Risk Group Distribution')
    for bar, value in zip(bars, group_sizes):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 str(value), ha='center', va='bottom')

    # Survival times by risk group
    ax3.boxplot([stratification_results['low_risk_times'],
                 stratification_results['high_risk_times']],
                labels=['Low Risk', 'High Risk'])
    ax3.set_ylabel('Survival Time (months)')
    ax3.set_title('Survival Time Distribution by Risk Group')

    # Event rates by risk group
    low_risk_event_rate = np.mean(stratification_results['low_risk_events'])
    high_risk_event_rate = np.mean(stratification_results['high_risk_events'])
    event_rates = [low_risk_event_rate, high_risk_event_rate]
    bars = ax4.bar(risk_groups, event_rates, color=config['visualization']['colors'][2:4])
    ax4.set_ylabel('Event Rate')
    ax4.set_title('Event Rate by Risk Group')
    ax4.set_ylim(0, 1)
    for bar, value in zip(bars, event_rates):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_performance_summary.png'),
                dpi=config['visualization']['save_dpi'], bbox_inches='tight')
    plt.close()
    print("✅ Model performance summary saved")
else:
    print("⚠️ Warning: Insufficient data for performance summary plot, skipping")

# Save results
print("💾 Saving results...")

# Create results dictionary
results = {
    'model': best_rsf,
    'config': config,
    'train_c_index': train_c_index,
    'test_c_index': test_c_index,
    'stratification_results': stratification_results,
    'train_risk_scores': train_risk_scores,
    'test_risk_scores': test_risk_scores,
    'X_train': X_train,
    'X_test': X_test,
    'y_train_rsf': y_train_rsf,
    'y_test_rsf': y_test_rsf
}

# Save model and results
if config['training']['save_model']:
    model_path = os.path.join(output_dir, 'rsf_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_rsf, f)
    print(f"✅ Model saved to: {model_path}")

# Save results
results_path = os.path.join(output_dir, 'experiment_results.pkl')
with open(results_path, 'wb') as f:
    pickle.dump(results, f)
print(f"✅ Results saved to: {results_path}")

# Save summary statistics
summary_stats = {
    'experiment_name': config['global']['experiment_name'],
    'model_type': 'Random Survival Forest',
    'train_samples': X_train.shape[0],
    'test_samples': X_test.shape[0],
    'n_features': X_train.shape[1],
    'train_c_index': train_c_index,
    'test_c_index': test_c_index,
    'logrank_pvalue': stratification_results['logrank_pvalue'],
    'logrank_statistic': stratification_results['logrank_statistic'],
    'best_parameters': getattr(grid_search, 'best_params_', 'Default parameters used') if 'grid_search' in locals() else 'Default parameters used'
}

summary_path = os.path.join(output_dir, 'summary_statistics.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("Random Survival Forest Experiment Summary\n")
    f.write("=" * 50 + "\n\n")
    for key, value in summary_stats.items():
        f.write(f"{key}: {value}\n")

print(f"✅ Summary statistics saved to: {summary_path}")

print("\n🎉 Random Survival Forest experiment completed successfully!")
print(f"📁 All results saved to: {output_dir}")
print(f"📊 Test C-index: {test_c_index:.4f}")
print(f"📊 Log-rank p-value: {stratification_results['logrank_pvalue']:.4e}")
