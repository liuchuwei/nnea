from sklearn.metrics import roc_auc_score, classification_report

import nnea as na
import numpy as np
import torch
import os
import warnings
import toml  # For reading toml files

warnings.filterwarnings('ignore')

print("🚀 Starting nnea model experiment...")

# Read nnea configuration file
print("⚙️ Reading nnea configuration file...")
try:
    nnea_config = toml.load("./config.toml")
    print("✅ Configuration file read successfully")
except Exception as e:
    print(f"❌ Configuration file reading failed: {e}")
    exit(1)

# Set global random seed (before data loading)
print("🔧 Setting global random seed...")
na.set_global_seed(nnea_config['global']['seed'])
print("✅ Global random seed set successfully")

# Data loading
print("📂 Loading data...")
try:
    nadata = na.nadata()
    nadata.load(filepath="./datasets/cell_gender/lymphocyte_gender_classification.pkl")
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
    X = na.pp.fillna(X, method="mean")
    print(f"   NaN count after filling: {np.isnan(X).sum()}")
else:
    print("✅ No NaN values detected in X")

# Update X in nadata
nadata.X = X

# Process labels
print("🏷️ Processing labels...")
y = nadata.Meta['sex']
y = y.map({'Female': 0, 'Male': 1})
nadata.Meta['target'] = y  # Model uses 'target' by default

# Feature selection
if nnea_config['dataset']['feature_selection']:
    nadata = na.fs.apply_feature_selection(
        nadata,
        method=nnea_config['dataset']['selection_method'],
        n_features=nnea_config['dataset']['n_features'],
        target_col='target',  # Use default target column
    )


# Data splitting
print("✂️ Performing data splitting...")
try:
    nadata = na.pp.split_data(
        nadata,
        test_size=nnea_config['dataset']['test_size'],
        random_state=nnea_config['dataset']['random_state'],
        strategy="stratified"
    )
    print("✅ Data splitting completed")
except Exception as e:
    print(f"❌ Data splitting failed: {e}")

# Handle device configuration
if nnea_config['global']['device'] == 'auto':
    nnea_config['global']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"🖥️ Using device: {nnea_config['global']['device']}")

# Set nadata configuration (will automatically create output directory and configure logging)
try:
    nadata.Model.set_config(nnea_config)
    print("✅ Model configuration set successfully")
    print(f"📁 Output directory: {nnea_config['global']['outdir']}")
except Exception as e:
    print(f"❌ Model configuration setup failed: {e}")
    exit(1)

# Build model
print("🔧 Building nnea model...")
try:
    na.build(nadata)
    print("✅ Model built successfully")
except Exception as e:
    print(f"❌ Model building failed: {e}")
    exit(1)

# Train model
print("🚀 Starting nnea model training...")
try:
    # Check if tailor strategy is enabled
    training_config = nnea_config.get('training', {})
    tailor_enabled = training_config.get('tailor', False)

    if tailor_enabled:
        print(
            f"✂️ Enabling tailor strategy: tailor_epoch={training_config.get('tailor_epoch', 20)}, tailor_geneset={training_config.get('tailor_geneset', 2)}")

    train_results = na.train(nadata, verbose=2)
    print("✅ Model training completed")
    print(f"📊 Training results: {train_results}")

    # If tailor strategy was used, display pruning information
    if tailor_enabled and isinstance(train_results, dict) and 'tailor_info' in train_results:
        tailor_info = train_results['tailor_info']
        print(f"✂️ Loop Tailor strategy information:")
        print(f"   - Pruning epoch interval: {tailor_info['tailor_epoch']}")
        print(f"   - Number of genesets pruned per iteration: {tailor_info['tailor_geneset']}")
        print(f"   - Total training stages: {tailor_info['total_stages']}")
        print(f"   - Final number of genesets: {tailor_info['final_geneset_count']}")
except Exception as e:
    print(f"❌ Model training failed: {e}")
    print(f"   Error type: {type(e).__name__}")
    print(f"   Error details: {str(e)}")

# Evaluate model
print("📈 Evaluating nnea model...")
try:
    eval_results = na.eval(nadata, split='test')
    print("✅ Model evaluation completed")
    print(f"📊 Evaluation results: {eval_results}")
except Exception as e:
    print(f"❌ Model evaluation failed: {e}")
    eval_results = {}

print("✅ Model training and evaluation completed!")

# Get model prediction results
print("🔮 Performing model prediction...")
try:
    # Use nnea package's predict function
    from nnea import predict

    prediction_results = predict(nadata, split='test')

    # Check prediction results
    if prediction_results.get('error'):
        print(f"❌ Prediction failed: {prediction_results['error']}")
        y_test = None
        y_pred = None
        y_proba = None
    else:
        y_test = prediction_results['y_test']
        y_pred = prediction_results['y_pred']
        y_proba = prediction_results['y_proba']
        print("✅ Model prediction completed")

except Exception as e:
    print(f"❌ Model prediction failed: {e}")
    y_test = None
    y_pred = None
    y_proba = None

# Build nnea result dictionary
print("💾 Saving experiment results...")
nnea_result = {
    "model_config": nnea_config,
    "train_results": train_results,
    "eval_results": eval_results,
    "test_auc": roc_auc_score(y_test, y_proba) if y_test is not None and y_proba is not None else None,
    "test_report": classification_report(y_test, y_pred,
                                         output_dict=True) if y_test is not None and y_pred is not None else None,
    "test_pred": y_pred,
    "test_proba": y_proba,
    "test_true": y_test
}

# Save to nadata object
if not hasattr(nadata, "Model"):
    nadata.Model = {}

nadata.Model["nnea_result"] = nnea_result

# Save nadata object to file (using output directory from config)
try:
    save_path = os.path.join(nnea_config['global']['outdir'], "~lym_gender.pkl")
    nadata.save(save_path, format="pickle", save_data=True)
    print(f"✅ nnea model training completed and saved to: {save_path}")
except Exception as e:
    print(f"❌ Saving failed: {e}")

# Reload nadata object
print("🔄 Reloading nadata object...")
try:
    nadata_reloaded = na.nadata()
    load_path = os.path.join(nnea_config['global']['outdir'], "~lym_gender.pkl")
    nadata_reloaded.load(filepath=load_path)
    print(f"✅ Data reloaded successfully: {load_path}")
except Exception as e:
    print(f"❌ Data reloaded failed: {e}")
    exit(1)

# Get saved nnea results
nnea_result_reloaded = nadata_reloaded.Model.get(nnea_config['global']['model'], None)

# Model interpretability analysis
print("🔍 Performing model interpretability analysis...")
try:
    # Use nnea's explain function
    nnea_result_reloaded.explain(nadata_reloaded, method='importance')
    print("✅ Feature importance analysis completed")

except Exception as e:
    print(f"⚠️ Error during model interpretability analysis: {e}")

# Get model summary
print("📋 Getting model summary...")
try:
    summary = na.get_summary(nadata_reloaded)
    print("📊 Model summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
except Exception as e:
    print(f"⚠️ Error getting model summary: {e}")

print("🎉 nnea model experiment completed!")
print(f"📁 Results saved to: {nnea_config['global']['outdir']}")
print(f"�� Log file saved to: {os.path.join(nnea_config['global']['outdir'], 'logs')}")

