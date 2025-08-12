from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

import nnea as na
import numpy as np
import os
import warnings
import toml  # For reading toml files
import random
import torch

warnings.filterwarnings('ignore')

# Read LinearSVM configuration file
try:
    config = toml.load("config.toml")
except Exception as e:
    print(f"❌ Configuration file reading failed: {e}")
    exit(1)

# Set all random seeds
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Call before data loading
set_all_seeds(config['global']['seed'])

# Create output directory
output_dir = config['global']['outdir']
os.makedirs(output_dir, exist_ok=True)

# Set log output to output directory
log_file = os.path.join(output_dir, "linear_svm_experiment.log")
na.setup_logging(log_file=log_file, experiment_name="linear_svm")
logger = na.get_logger(__name__)

logger.info("⚙️ Reading LinearSVM configuration file...")
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
    nadata.load(filepath=config['global']['inputfl'])
    logger.info(f"✅ Preprocessed nadata object loaded successfully, data shape: {nadata.X.shape}")
except Exception as e:
    logger.error(f"❌ Data loading failed: {e}")
    exit(1)

# Data preprocessing
logger.info("🔧 Data preprocessing...")
X = nadata.X

# Use preprocessing settings from configuration
preprocessing_config = config['linear_svm']['preprocessing']

# Use na.pp.fillna to handle missing values
if preprocessing_config['fill_na'] and np.isnan(X).any():
    logger.warning("⚠️ NaN values detected in X, performing fill operation...")
    X = na.pp.fillna(X, method=preprocessing_config['fill_method'])
    logger.info(f"   NaN count after filling: {np.isnan(X).sum()}")
else:
    logger.info("✅ No NaN values detected in X")

# Use na.pp.scale for standardization - Linear SVM is sensitive to feature scale
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
if config['linear_svm']['feature_selection']:
    logger.info("🔍 Feature selection...")
    nadata = na.fs.apply_feature_selection(
        nadata,
        method=config['linear_svm']['selection_method'],
        n_features=config['linear_svm']['n_features'],
        target_col='target',  # Use default 'target' column
        alpha=config['linear_svm']['selection_alpha']
    )
    logger.info(f"✅ Feature selection completed, selected features: {config['linear_svm']['n_features']}")

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

# Use na.pp.x_train_test and na.pp.y_train_test to get training and test sets
X_train, X_test = na.pp.x_train_test(X, nadata)
y_train, y_test = na.pp.y_train_test(y, nadata)

logger.info(f"Training set feature shape: {X_train.shape}")
logger.info(f"Test set feature shape: {X_test.shape}")
logger.info(f"Training set label shape: {y_train.shape}")
logger.info(f"Test set label shape: {y_test.shape}")

# Build parameter grid from configuration
param_grid = {
    'C': config['linear_svm']['C'],
    'loss': config['linear_svm']['loss']
}

# Build LinearSVC model
linear_svc = LinearSVC(
    random_state=config['linear_svm']['random_state'],
    class_weight=config['linear_svm']['class_weight'],
    max_iter=config['linear_svm']['max_iter'],
    dual=False  # For large datasets, primal form is faster
)

# Grid search cross-validation
grid = GridSearchCV(
    linear_svc,
    param_grid,
    cv=StratifiedKFold(
        n_splits=config['linear_svm']['cv_folds'],
        shuffle=True,
        random_state=config['linear_svm']['random_state']
    ),
    scoring=config['linear_svm']['cv_scoring'],
    n_jobs=config['linear_svm']['n_jobs'],
    verbose=config['training']['verbose']
)

logger.info("🚀 Starting grid search training...")
grid.fit(X_train, y_train)

logger.info(f"Best parameters: {grid.best_params_}")
logger.info(f"Best cross-validation score: {grid.best_score_}")

# LinearSVC does not directly provide probabilities, calibration is needed
logger.info("🔧 Performing probability calibration...")
calibrated_model = CalibratedClassifierCV(
    grid.best_estimator_,
    cv=3,
    method='sigmoid'
)
calibrated_model.fit(X_train, y_train)

# Evaluate on the test set
y_pred = calibrated_model.predict(X_test)
y_proba = calibrated_model.predict_proba(X_test)[:, 1]

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
# Calculate and record F1, recall, precision, and accuracy
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

logger.info(f"Test set F1 score: {f1:.4f}")
logger.info(f"Test set recall: {recall:.4f}")
logger.info(f"Test set precision: {precision:.4f}")
logger.info(f"Test set accuracy: {acc:.4f}")
logger.info(f"Test set AUC: {auc:.4f}")
logger.info(f"Test set classification report:\n{classification_report(y_test, y_pred)}")

# Build LinearSVM result dictionary
linear_svm_result = {
    "best_params": grid.best_params_,
    "best_cv_score": grid.best_score_,
    "test_auc": auc,
    "test_report": classification_report(y_test, y_pred, output_dict=True),
    "test_pred": y_pred,
    "test_proba": y_proba,
    "test_true": y_test.values,
    "best_model": grid.best_estimator_,  # Save best model
    "calibrated_model": calibrated_model  # Save calibrated model
}

# Save to nadata object
if not hasattr(nadata, "Model"):
    nadata.Model = {}

nadata.Model["LinearSVM"] = linear_svm_result

# Save nadata object to configured output directory
output_file = os.path.join(output_dir, config['global']['outputfl'])
nadata.save(output_file, format=config['training']['save_format'], save_data=config['training']['save_data'])
logger.info(f"✅ LinearSVM model training completed and saved to: {output_file}")

# Save configuration information
config_file = os.path.join(output_dir, "linear_svm_config.toml")
with open(config_file, 'w', encoding='utf-8') as f:
    toml.dump(config, f)
logger.info(f"✅ Configuration file saved to: {config_file}")

# Save training results summary
summary_file = os.path.join(output_dir, "training_summary.txt")
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("LinearSVM Training Results Summary\n")
    f.write("=" * 50 + "\n")
    f.write(f"Best parameters: {grid.best_params_}\n")
    f.write(f"Best cross-validation score: {grid.best_score_:.4f}\n")
    f.write(f"Test set AUC: {auc:.4f}\n")
    f.write(f"Test set F1 score: {f1:.4f}\n")
    f.write(f"Test set recall: {recall:.4f}\n")
    f.write(f"Test set precision: {precision:.4f}\n")
    f.write(f"Test set accuracy: {acc:.4f}\n")
    f.write(f"Training set shape: {X_train.shape}\n")
    f.write(f"Test set shape: {X_test.shape}\n")
    f.write("\nClassification report:\n")
    f.write(classification_report(y_test, y_pred))

logger.info(f"✅ Training results summary saved to: {summary_file}")
logger.info("🎉 LinearSVM experiment completed!")
