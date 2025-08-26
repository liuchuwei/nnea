#!/usr/bin/env python3
"""
Runner script for RBF SVM experiment with lym_gender data.
This script copies the configuration file to the correct location and runs the experiment.
"""

import os
import shutil
import subprocess
import sys

def main():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Configuration file path
    config_file = os.path.join(current_dir, "lym_gender_RBFSVM.toml")
    experiment_file = os.path.join(current_dir, "lym_gender_RBFSVM_experiment.py")
    
    # Check if files exist
    if not os.path.exists(config_file):
        print(f"❌ Configuration file not found: {config_file}")
        sys.exit(1)
    
    if not os.path.exists(experiment_file):
        print(f"❌ Experiment file not found: {experiment_file}")
        sys.exit(1)
    
    # Copy configuration file to the correct location (same directory as experiment)
    target_config = os.path.join(current_dir, "config.toml")
    shutil.copy2(config_file, target_config)
    print(f"✅ Configuration file copied to: {target_config}")
    
    # Change to the experiment directory
    os.chdir(current_dir)
    
    # Run the experiment
    print("🚀 Starting RBF SVM experiment...")
    try:
        result = subprocess.run([sys.executable, experiment_file], 
                              capture_output=False, 
                              text=True, 
                              check=True)
        print("✅ RBF SVM experiment completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment failed with error code: {e.returncode}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
    
    # Clean up the temporary config file
    if os.path.exists(target_config):
        os.remove(target_config)
        print("✅ Temporary configuration file cleaned up")

if __name__ == "__main__":
    main()
