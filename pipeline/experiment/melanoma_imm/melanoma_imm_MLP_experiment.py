from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import nnea as na
import numpy as np
import os
import warnings
import toml  # For reading toml files
import random
import torch

warnings.filterwarnings('ignore')

# Read MLPClassifier configuration file
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
# Ensure output directory is an absolute path relative to current working directory
if not os.path.isabs(output_dir):
    output_dir = os.path.join(os.getcwd(), output_dir)
os.makedirs(output_dir, exist_ok=True)

# Set log output to output directory
log_file = os.path.join(output_dir, "mlp_experiment.log")
na.setup_logging(log_file=log_file, experiment_name="mlp_classifier")
logger = na.get_logger(__name__)

logger.info("⚙️ Reading MLPClassifier configuration file...")
logger.info("✅ Configuration file read successfully")
logger.info(f"📁 Creating output directory: {output_dir}")
logger.info(f"📝 Log file set to: {log_file}")

# Set global random seed (before data loading)
logger.info("🔧 Setting global random seed...")
na.set_global_seed(config['global']['seed'])
logger.info("✅ Global random seed set successfully")

# Data loading
logger.info("📂 Loading data...")
input_file = config['global']['inputfl']
if not os.path.exists(input_file):
    logger.error(f"❌ Input file does not exist: {input_file}")
    exit(1)

try:
    nadata = na.nadata()
    nadata.load(filepath=input_file)
    logger.info(f"✅ Preprocessed nadata object loaded successfully, data shape: {nadata.X.shape}")
except Exception as e:
    logger.error(f"❌ Data loading failed: {e}")
    exit(1)

# Data preprocessing
logger.info("🔧 Data preprocessing...")
X = nadata.X

# Use preprocessing settings from configuration
preprocessing_config = config['mlp']['preprocessing']

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
if 'response_NR' not in nadata.Meta.columns:
    logger.error("❌ Missing 'response_NR' column in data")
    exit(1)

y = nadata.Meta['response_NR']
# Check unique values of labels
unique_labels = y.unique()
logger.info(f"Unique labels: {unique_labels}")

if len(unique_labels) != 2:
    logger.error(f"❌ Labels should only have 2 categories, but found {len(unique_labels)}: {unique_labels}")
    exit(1)

y = y.map({'N': 0, 'R': 1})
nadata.Meta['target'] = y  # Model uses target by default

# Check label distribution
logger.info(f"Label distribution: {y.value_counts().to_dict()}")

# Feature selection
if config['mlp']['feature_selection']:
    logger.info("🔍 Feature selection...")
    nadata = na.fs.apply_feature_selection(
        nadata,
        method=config['mlp']['selection_method'],
        n_features=config['mlp']['n_features'],
        target_col='target',  # Use default target column
        alpha=config['mlp']['selection_alpha']
    )
    logger.info(f"✅ Feature selection completed, selected features: {config['mlp']['n_features']}")

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
    'hidden_layer_sizes': config['mlp']['hidden_layer_sizes'],
    'activation': config['mlp']['activation'],
    'solver': config['mlp']['solver'],
    'alpha': config['mlp']['alpha'],
    'learning_rate': config['mlp']['learning_rate']
}

# Build MLPClassifier model
mlp = MLPClassifier(
    max_iter=config['mlp']['max_iter'],
    random_state=config['mlp']['random_state'],
    early_stopping=config['mlp']['early_stopping'],
    validation_fraction=config['mlp']['validation_fraction']
)

# Grid search cross-validation
grid = GridSearchCV(
    mlp,
    param_grid,
    cv=StratifiedKFold(
        n_splits=config['mlp']['cv_folds'],
        shuffle=True,
        random_state=config['mlp']['random_state']
    ),
    scoring=config['mlp']['cv_scoring'],
    n_jobs=config['mlp']['n_jobs'],
    verbose=config['training']['verbose']
)

logger.info("🚀 Starting grid search training...")
logger.info(f"Parameter grid size: {len(param_grid['hidden_layer_sizes']) * len(param_grid['activation']) * len(param_grid['solver']) * len(param_grid['alpha']) * len(param_grid['learning_rate'])} combinations")

# Record training start time
import time
start_time = time.time()

grid.fit(X_train, y_train)

# Record training end time
end_time = time.time()
training_time = end_time - start_time

logger.info(f"✅ Training completed, time: {training_time:.2f} seconds")
logger.info(f"Best parameters: {grid.best_params_}")
logger.info(f"Best AUC score: {grid.best_score_:.4f}")

# Evaluate on test set
y_pred = grid.predict(X_test)
y_proba = grid.predict_proba(X_test)[:, 1]
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

# Build MLPClassifier result dictionary
mlp_result = {
    "best_params": grid.best_params_,
    "best_cv_auc": grid.best_score_,
    "test_auc": auc,
    "test_report": classification_report(y_test, y_pred, output_dict=True),
    "test_pred": y_pred,
    "test_proba": y_proba,
    "test_true": y_test.values,
    "best_model": grid.best_estimator_  # Save best model
}

# Save to nadata object
if not hasattr(nadata, "Model"):
    nadata.Model = {}

nadata.Model["MLPClassifier"] = mlp_result

# Save nadata object to configured output directory
output_file = os.path.join(output_dir, config['global']['outputfl'])
nadata.save(output_file, format=config['training']['save_format'], save_data=config['training']['save_data'])
logger.info(f"✅ MLPClassifier model training completed and saved to: {output_file}")

# Save configuration information
config_file = os.path.join(output_dir, "mlp_config.toml")
with open(config_file, 'w', encoding='utf-8') as f:
    toml.dump(config, f)
logger.info(f"✅ Configuration file saved to: {config_file}")

# Save training results summary
summary_file = os.path.join(output_dir, "training_summary.txt")
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("MLPClassifier Training Results Summary\n")
    f.write("=" * 50 + "\n")
    f.write(f"Best parameters: {grid.best_params_}\n")
    f.write(f"Best cross-validation AUC: {grid.best_score_:.4f}\n")
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

# Save detailed training results
results_file = os.path.join(output_dir, "detailed_results.txt")
with open(results_file, 'w', encoding='utf-8') as f:
    f.write("MLPClassifier Detailed Training Results\n")
    f.write("=" * 60 + "\n")
    f.write(f"Training time: {training_time:.2f} seconds\n")
    f.write(f"Best parameters: {grid.best_params_}\n")
    f.write(f"Best cross-validation AUC: {grid.best_score_:.4f}\n")
    f.write(f"Test set AUC: {auc:.4f}\n")
    f.write(f"Test set F1 score: {f1:.4f}\n")
    f.write(f"Test set recall: {recall:.4f}\n")
    f.write(f"Test set precision: {precision:.4f}\n")
    f.write(f"Test set accuracy: {acc:.4f}\n")
    f.write(f"Training set shape: {X_train.shape}\n")
    f.write(f"Test set shape: {X_test.shape}\n")
    f.write(f"Feature count: {X_train.shape[1]}\n")
    f.write(f"Sample count: {X_train.shape[0] + X_test.shape[0]}\n")
    f.write("\nAll cross-validation results:\n")
    for i, score in enumerate(grid.cv_results_['mean_test_score']):
        f.write(f"Combination {i+1}: {score:.4f}\n")
    f.write("\nClassification report:\n")
    f.write(classification_report(y_test, y_pred))

logger.info(f"✅ Detailed results saved to: {results_file}")
logger.info("🎉 Experiment completed!")
