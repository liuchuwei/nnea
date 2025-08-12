from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import nnea as na
import numpy as np
import torch
import os
import warnings
import toml  # for reading toml files

warnings.filterwarnings('ignore')

# Read RBFSVM configuration file
try:
    config = toml.load("./config/experiment/ccRCC_imm_RBFSVM.toml")
except Exception as e:
    print(f"❌ Configuration file reading failed: {e}")
    exit(1)

# Create output directory
output_dir = config['global']['outdir']
os.makedirs(output_dir, exist_ok=True)

# Set up logging output to output directory
log_file = os.path.join(output_dir, "rbf_svm_experiment.log")
na.setup_logging(log_file=log_file, experiment_name="rbf_svm")
logger = na.get_logger(__name__)

logger.info("⚙️ Reading RBFSVM configuration file...")
logger.info("✅ Configuration file read successfully")
logger.info(f"📁 Creating output directory: {output_dir}")
logger.info(f"📝 Log file set to: {log_file}")

# Set global random seed (before data loading)
logger.info("🔧 Setting global random seed...")
na.set_global_seed(config['global']['seed'])
logger.info("✅ Global random seed set successfully")

# Data loading
logger.info("📂 Loading data...")
try:
    nadata = na.nadata()
    nadata.load(filepath="experiment/tumor_imm/ccRCC_imm_exp.pkl")
    logger.info(f"✅ Preprocessed nadata object loaded successfully, data shape: {nadata.X.shape}")
except Exception as e:
    logger.error(f"❌ Data loading failed: {e}")
    exit(1)

# Data preprocessing
logger.info("🔧 Data preprocessing...")
X = nadata.X

# Use preprocessing settings from configuration
preprocessing_config = config['rbf_svm']['preprocessing']

# Use na.pp.fillna to handle missing values
if preprocessing_config['fill_na'] and np.isnan(X).any():
    logger.warning("⚠️ NaN values detected in X, performing fill operation...")
    X = na.pp.fillna(X, method=preprocessing_config['fill_method'])
    logger.info(f"   NaN count after filling: {np.isnan(X).sum()}")
else:
    logger.info("✅ No NaN values detected in X")

# Use na.pp.scale for standardization
if preprocessing_config['scale_data']:
    X = na.pp.scale(X, method=preprocessing_config['scale_method'])
    logger.info("✅ Data standardization completed")

nadata.X = X

# Process labels
logger.info("🏷️ Processing labels...")
y = nadata.Meta['response_NR']
y = y.map({'N': 0, 'R': 1})
nadata.Meta['target'] = y  # Model uses 'target' by default

# Feature selection
if config['rbf_svm']['feature_selection']:
    logger.info("🔍 Feature selection...")
    nadata = na.fs.apply_feature_selection(
        nadata,
        method=config['rbf_svm']['selection_method'],
        n_features=config['rbf_svm']['n_features'],
        target_col='target',  # Use default 'target' column
        alpha=config['rbf_svm']['selection_alpha']
    )
    logger.info(f"✅ Feature selection completed, selected features: {config['rbf_svm']['n_features']}")

# Data splitting
logger.info("✂️ Performing data splitting...")
try:
    nadata = na.pp.split_data(
        nadata,
        test_size=config['dataset']['test_size'],
        random_state=config['dataset']['random_state'],
        strategy="stratified"
    )
    logger.info("✅ Data splitting completed")
except Exception as e:
    logger.error(f"❌ Data splitting failed: {e}")

# Use na.pp.x_train_test and na.pp.y_train_test to get train/test sets
X_train, X_test = na.pp.x_train_test(X, nadata)
y_train, y_test = na.pp.y_train_test(y, nadata)

logger.info(f"Training set feature shape: {X_train.shape}")
logger.info(f"Test set feature shape: {X_test.shape}")
logger.info(f"Training set label shape: {y_train.shape}")
logger.info(f"Test set label shape: {y_test.shape}")

# Build parameter grid from configuration file
param_grid = {
    'C': config['rbf_svm']['C'],
    'gamma': config['rbf_svm']['gamma'],
    'kernel': config['rbf_svm']['kernel']
}

# Build SVC model (RBF kernel)
rbf_svm = SVC(
    random_state=config['rbf_svm']['random_state'],
    class_weight=config['rbf_svm']['class_weight'],
    probability=config['rbf_svm']['probability'],
    tol=config['rbf_svm']['tol'],
    max_iter=config['rbf_svm']['max_iter']
)

# Grid search cross-validation
grid = GridSearchCV(
    rbf_svm,
    param_grid,
    cv=StratifiedKFold(
        n_splits=config['rbf_svm']['cv_folds'], 
        shuffle=True, 
        random_state=config['rbf_svm']['random_state']
    ),
    scoring=config['rbf_svm']['cv_scoring'],
    n_jobs=config['rbf_svm']['n_jobs'],
    verbose=config['training']['verbose']
)

logger.info("🚀 Starting grid search training...")
grid.fit(X_train, y_train)

logger.info(f"Best parameters: {grid.best_params_}")
logger.info(f"Best AUC score: {grid.best_score_}")

# Evaluate on test set
y_pred = grid.predict(X_test)
y_proba = grid.predict_proba(X_test)[:, 1]
logger.info(f"Test set classification report:\n{classification_report(y_test, y_pred)}")
logger.info(f"Test set AUC: {roc_auc_score(y_test, y_proba)}")

# Build RBFSVM result dictionary
rbf_svm_result = {
    "best_params": grid.best_params_,
    "best_cv_auc": grid.best_score_,
    "test_auc": roc_auc_score(y_test, y_proba),
    "test_report": classification_report(y_test, y_pred, output_dict=True),
    "test_pred": y_pred,
    "test_proba": y_proba,
    "test_true": y_test.values,
    "best_model": grid.best_estimator_  # Save best model
}

# Save to nadata object
if not hasattr(nadata, "Model"):
    nadata.Model = {}

nadata.Model["RBFSVM"] = rbf_svm_result

# Save nadata object to configured output directory
output_file = os.path.join(output_dir, "ccRCC_imm_RBFSVM.pkl")
nadata.save(output_file, format=config['training']['save_format'], save_data=config['training']['save_data'])
logger.info(f"✅ RBF SVM model training completed and saved to: {output_file}")

# Save configuration information
config_file = os.path.join(output_dir, "rbf_svm_config.toml")
with open(config_file, 'w', encoding='utf-8') as f:
    toml.dump(config, f)
logger.info(f"✅ Configuration file saved to: {config_file}")

# Save training result summary
summary_file = os.path.join(output_dir, "training_summary.txt")
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("RBFSVM Training Result Summary\n")
    f.write("=" * 50 + "\n")
    f.write(f"Best parameters: {grid.best_params_}\n")
    f.write(f"Best cross-validation AUC: {grid.best_score_:.4f}\n")
    f.write(f"Test set AUC: {roc_auc_score(y_test, y_proba):.4f}\n")
    f.write(f"Training set shape: {X_train.shape}\n")
    f.write(f"Test set shape: {X_test.shape}\n")
    f.write("\nClassification report:\n")
    f.write(classification_report(y_test, y_pred))

logger.info(f"✅ Training result summary saved to: {summary_file}")
logger.info("🎉 Experiment completed!") 