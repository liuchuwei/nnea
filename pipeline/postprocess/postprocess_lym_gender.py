import os

import nnea as na
import toml

# Read nnea configuration file
print("⚙️ Reading nnea configuration file...")
try:
    nnea_config = toml.load("./config.toml")
    print("✅ Configuration file read successfully")
except Exception as e:
    print(f"❌ Configuration file reading failed: {e}")
    exit(1)

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