from sklearn.metrics import roc_auc_score, classification_report

import nnea as na
import numpy as np
import torch
import os
import warnings
import toml  # for reading toml files

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
    X = na.pp.fillna(X, method="mean")
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

# Data splitting
print("✂️ Performing data splitting...")
try:
    nadata = na.pp.split_data(
        nadata,
        test_size=0.2,
        random_state=42,
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
    print("✅ Model building completed")
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
            f"✂️ Tailor strategy enabled: tailor_epoch={training_config.get('tailor_epoch', 20)}, tailor_geneset={training_config.get('tailor_geneset', 2)}")


    train_results = na.train(nadata, verbose=2)
    print("✅ Model training completed")
    print(f"📊 Training results: {train_results}")

    # If tailor strategy is used, display pruning information
    if tailor_enabled and isinstance(train_results, dict) and 'tailor_info' in train_results:
        tailor_info = train_results['tailor_info']
        print(f"✂️ Cyclic Tailor strategy information:")
        print(f"   - Pruning epoch interval: {tailor_info['tailor_epoch']}")
        print(f"   - Geneset count pruned each time: {tailor_info['tailor_geneset']}")
        print(f"   - Total training stages: {tailor_info['total_stages']}")
        print(f"   - Final geneset count: {tailor_info['final_geneset_count']}")

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
    # Use predict function from nnea package
    from nnea import predict

    prediction_results = predict(nadata, split='test')

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
    "test_results": prediction_results
}

# Save to nadata object
if not hasattr(nadata, "Model"):
    nadata.Model = {}

nadata.Model["nnea_model"] = nnea_result

# Save nadata object to file (using output directory from configuration)
try:
    save_path = os.path.join(nnea_config['global']['outdir'], "melanoma_imm.pkl")
    nadata.save(save_path, format="pickle", save_data=True)
    print(f"✅ Nnea model training completed and saved to: {save_path}")
except Exception as e:
    print(f"❌ Save failed: {e}")

# Reload nadata object
print("🔄 Reloading nadata object...")
try:
    nadata_reloaded = na.nadata()
    load_path = os.path.join(nnea_config['global']['outdir'], "melanoma_imm.pkl")
    nadata_reloaded.load(filepath=load_path)
    print(f"✅ Data reload successful: {load_path}")
except Exception as e:
    print(f"❌ Data reload failed: {e}")
    exit(1)

# Get saved nnea results
nnea_result_reloaded = nadata_reloaded.Model.get("nnea_model", None)
if nnea_result_reloaded is None:
    print("⚠️ Nnea model results not found in nadata object")
else:
    print("📊 Reloaded model results:")
    print(f"Training results: {nnea_result_reloaded.get('train_results', {})}")
    print(f"Evaluation results: {nnea_result_reloaded.get('eval_results', {})}")

# Model interpretability analysis
print("🔍 Performing model interpretability analysis...")
try:
    # Use nnea's explain functionality
    na.explain(nadata_reloaded, method='importance', model_name="nnea")
    print("✅ Feature importance analysis completed")

except Exception as e:
    print(f"⚠️ Error occurred during model interpretability analysis: {e}")

# Get model summary
print("📋 Getting model summary...")
try:
    summary = na.get_summary(nadata_reloaded)
    print("📊 Model summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
except Exception as e:
    print(f"⚠️ Error occurred while getting model summary: {e}")

print("🎉 Nnea model experiment completed!")
print(f"📁 Results saved to: {nnea_config['global']['outdir']}")
print(f"📊 Log files saved to: {os.path.join(nnea_config['global']['outdir'], 'logs')}")

