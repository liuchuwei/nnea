import numpy as np
import pandas as pd
import warnings
import toml
import os
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import nnea as na

warnings.filterwarnings('ignore')

print("🚀 Starting Cox proportional hazards model experiment...")

# Read configuration file
print("⚙️ Reading configuration file...")
try:
    config = toml.load("./config.toml")
    print("✅ Configuration file read successfully")
except Exception as e:
    print(f"❌ Configuration file reading failed: {e}")
    exit(1)

# Set global random seed
print("🔧 Setting global random seed...")
np.random.seed(config['global']['seed'])
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

    # Get train and test indices using Model.get_indices() method (参考nnea_survival.py的实现)
    train_indices = nadata.Model.get_indices('train')
    test_indices = nadata.Model.get_indices('test')
    val_indices = nadata.Model.get_indices('val')  # 可能为None

    # Get data based on indices
    X = nadata.X
    Meta = nadata.Meta

    # Get train data
    X_train = X[train_indices]
    Meta_train = Meta.iloc[train_indices]

    # Get test data
    X_test = X[test_indices]
    Meta_test = Meta.iloc[test_indices]

    # Get validation data (if exists)
    X_val = None
    Meta_val = None
    if val_indices is not None:
        X_val = X[val_indices]
        Meta_val = Meta.iloc[val_indices]
        print(f"   Validation set shape: {Meta_val.shape}")

    print(f"   Training set shape: {Meta_train.shape}")
    print(f"   Test set shape: {Meta_test.shape}")

    # Ensure column names are correct for survival data
    time_col = 'Time'
    event_col = 'Event'

    # Check if required columns exist
    if time_col not in Meta_train.columns or event_col not in Meta_train.columns:
        print(f"❌ Missing required columns: {time_col} or {event_col}")
        print(f"   Available columns: {list(Meta_train.columns)}")
        exit(1)

except Exception as e:
    print(f"❌ Data splitting failed: {e}")
    exit(1)

# Hyperparameter tuning
print("🔍 Starting hyperparameter tuning...")
if config['experiment']['hyperparameter_tuning']:
    tuning_method = config['experiment']['tuning_method']

    if tuning_method == "grid_search":
        print("🔍 Using grid search for hyperparameter tuning...")

        # Build parameter grid
        param_grid = {
            'penalizer': config['cox_model']['penalizer_values'],
            'l1_ratio': config['cox_model']['l1_ratio_values'],
            'max_iter': config['cox_model']['max_iter_values']
        }

        print(f"   Parameter grid: {param_grid}")

        # Create cross-validation object
        cv = StratifiedKFold(
            n_splits=config['experiment']['cv_folds'],
            shuffle=True,
            random_state=config['global']['seed']
        )

        best_score = -1
        best_params = None
        best_model = None
        all_results = []

        # Grid search
        total_combinations = len(param_grid['penalizer']) * len(param_grid['l1_ratio']) * len(param_grid['max_iter'])
        current_combination = 0

        for penalizer in param_grid['penalizer']:
            for l1_ratio in param_grid['l1_ratio']:
                for max_iter in param_grid['max_iter']:
                    current_combination += 1
                    print(f"   Progress: {current_combination}/{total_combinations}")
                    print(f"   Testing parameters: penalizer={penalizer}, l1_ratio={l1_ratio}, max_iter={max_iter}")

                    # Create model
                    model = CoxPHFitter(
                        penalizer=penalizer,
                        l1_ratio=l1_ratio
                    )

                    # Cross-validation
                    cv_scores = []
                    for train_idx, val_idx in cv.split(Meta_train, Meta_train['OS']):
                        train_fold = Meta_train.iloc[train_idx].copy()
                        val_fold = Meta_train.iloc[val_idx].copy()

                        # Clean data: keep only required survival columns
                        required_cols = ['Time', 'Event']

                        # Ensure required columns exist
                        missing_cols = [col for col in required_cols if col not in train_fold.columns]
                        if missing_cols:
                            print(f"   ⚠️ Missing required columns: {missing_cols}")
                            cv_scores.append(0.0)
                            continue

                        train_fold_clean = train_fold[required_cols]
                        val_fold_clean = val_fold[required_cols]

                        try:
                            # Train model
                            model.fit(
                                train_fold_clean,
                                duration_col='Time',
                                event_col='Event',
                                show_progress=False
                            )

                            # Predict risk scores
                            val_risk_scores = model.predict_partial_hazard(val_fold_clean)

                            # Calculate C-index
                            c_index = concordance_index(
                                val_fold_clean['Time'],
                                val_risk_scores,
                                val_fold_clean['Event']
                            )
                            cv_scores.append(c_index)

                        except Exception as e:
                            print(f"   ⚠️ Parameter combination failed: {e}")
                            cv_scores.append(0.0)

                    # Calculate mean C-index
                    mean_c_index = np.mean(cv_scores)
                    std_c_index = np.std(cv_scores)
                    print(f"   Mean C-index: {mean_c_index:.4f} ± {std_c_index:.4f}")

                    # Record results
                    result = {
                        'penalizer': penalizer,
                        'l1_ratio': l1_ratio,
                        'max_iter': max_iter,
                        'mean_c_index': mean_c_index,
                        'std_c_index': std_c_index,
                        'cv_scores': cv_scores
                    }
                    all_results.append(result)

                    # Update best parameters
                    if mean_c_index > best_score:
                        best_score = mean_c_index
                        best_params = {
                            'penalizer': penalizer,
                            'l1_ratio': l1_ratio,
                            'max_iter': max_iter
                        }
                        best_model = CoxPHFitter(
                            penalizer=penalizer,
                            l1_ratio=l1_ratio
                        )

        print(f"✅ Best parameters: {best_params}")
        print(f"✅ Best cross-validation C-index: {best_score:.4f}")

        # Save tuning results
        tuning_df = pd.DataFrame(all_results)
        os.makedirs(config['global']['outdir'], exist_ok=True)
        tuning_df.to_csv(f"{config['global']['outdir']}/hyperparameter_tuning_results.csv", index=False)
        print(
            f"✅ Hyperparameter tuning results saved to: {config['global']['outdir']}/hyperparameter_tuning_results.csv")

    else:
        print(f"⚠️ Unknown tuning method: {tuning_method}, using default parameters")
        best_params = {
            'penalizer': config['cox_model']['penalizer'],
            'l1_ratio': config['cox_model']['l1_ratio']
        }
        best_model = CoxPHFitter(
            penalizer=best_params['penalizer'],
            l1_ratio=best_params['l1_ratio']
        )
else:
    print("✅ Hyperparameter tuning skipped, using default parameters")
    best_params = {
        'penalizer': config['cox_model']['penalizer'],
        'l1_ratio': config['cox_model']['l1_ratio']
    }
    best_model = CoxPHFitter(
        penalizer=best_params['penalizer'],
        l1_ratio=best_params['l1_ratio']
    )

# Train model
print("🚀 Starting Cox model training...")
try:
    # Clean training data: keep only required survival columns
    required_cols = ['Time', 'Event']

    # Ensure required columns exist
    missing_cols = [col for col in required_cols if col not in Meta_train.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"   Available columns: {list(Meta_train.columns)}")
        exit(1)

    Meta_train_clean = Meta_train[required_cols]
    print(f"✅ Cleaned training data shape: {Meta_train_clean.shape}")
    print(f"   Columns kept: {list(Meta_train_clean.columns)}")

    best_model.fit(
        Meta_train_clean,
        duration_col='Time',
        event_col='Event',
        show_progress=False
    )
    print("✅ Model training completed")

    # Print model summary
    print("📊 Model summary:")
    print(best_model.print_summary())

except Exception as e:
    print(f"❌ Model training failed: {e}")
    exit(1)

# Evaluate model
print("📈 Evaluating Cox model...")
try:
    # Clean validation and test data
    if Meta_val is not None:
        Meta_val_clean = Meta_val[required_cols]
    Meta_test_clean = Meta_test[required_cols]

    # Calculate training set C-index
    train_risk_scores = best_model.predict_partial_hazard(Meta_train_clean)
    train_c_index = concordance_index(
        Meta_train_clean['Time'],
        train_risk_scores,
        Meta_train_clean['Event']
    )

    # Calculate validation set C-index (if validation set exists)
    val_c_index = None
    val_risk_scores = None
    if Meta_val is not None:
        val_risk_scores = best_model.predict_partial_hazard(Meta_val_clean)
        val_c_index = concordance_index(
            Meta_val_clean['Time'],
            val_risk_scores,
            Meta_val_clean['Event']
        )

    # Calculate test set C-index
    test_risk_scores = best_model.predict_partial_hazard(Meta_test_clean)
    test_c_index = concordance_index(
        Meta_test_clean['Time'],
        test_risk_scores,
        Meta_test_clean['Event']
    )

    print("✅ Model evaluation completed")
    print(f"   Training set C-index: {train_c_index:.4f}")
    if val_c_index is not None:
        print(f"   Validation set C-index: {val_c_index:.4f}")
    print(f"   Test set C-index: {test_c_index:.4f}")

except Exception as e:
    print(f"❌ Model evaluation failed: {e}")
    train_c_index = 0.0
    test_c_index = 0.0
    train_risk_scores = None
    test_risk_scores = None

# Risk stratification analysis
print("🎯 Risk stratification analysis...")
try:
    if test_risk_scores is not None:
        # Stratify based on risk scores
        median_risk = np.median(test_risk_scores)
        high_risk_mask = test_risk_scores > median_risk
        low_risk_mask = test_risk_scores <= median_risk

        # Extract high and low risk group data
        high_risk_data = Meta_test[high_risk_mask].copy()
        low_risk_data = Meta_test[low_risk_mask].copy()

        # Perform log-rank test
        logrank_result = logrank_test(
            high_risk_data['Time'],
            low_risk_data['Time'],
            high_risk_data['Event'],
            low_risk_data['Event']
        )

        print("✅ Risk stratification completed")
        print(f"   High risk group count: {len(high_risk_data)}")
        print(f"   Low risk group count: {len(low_risk_data)}")
        print(f"   Log-rank test p-value: {logrank_result.p_value:.4e}")
    else:
        print("⚠️ Risk stratification skipped due to missing risk scores")
        logrank_result = None

except Exception as e:
    print(f"❌ Risk stratification failed: {e}")
    logrank_result = None

# Create output directory
os.makedirs(config['global']['outdir'], exist_ok=True)

# Save results
print("💾 Saving experiment results...")
try:
    # Save model using pickle
    with open(f"{config['global']['outdir']}/cox_model.pkl", 'wb') as f:
        pickle.dump(best_model, f)

    # Save results data
    results_data = {
        'metric': ['train_c_index', 'test_c_index', 'logrank_p_value'],
        'value': [
            train_c_index,
            test_c_index,
            logrank_result.p_value if logrank_result else None
        ]
    }

    # Add validation results if validation set exists
    if val_c_index is not None:
        results_data['metric'].append('val_c_index')
        results_data['value'].append(val_c_index)

    # Add hyperparameter tuning information if performed
    if config['experiment']['hyperparameter_tuning']:
        results_data['metric'].extend(['best_penalizer', 'best_l1_ratio'])
        results_data['value'].extend([
            best_params['penalizer'],
            best_params['l1_ratio']
        ])

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(f"{config['global']['outdir']}/results.csv", index=False)

    # Save risk scores
    if test_risk_scores is not None:
        test_risk_df = pd.DataFrame({
            'risk_score': test_risk_scores,
            'OS': Meta_test['Event'],
            'OS_time': Meta_test['Time']
        })
        test_risk_df.to_csv(f"{config['global']['outdir']}/test_risk_scores.csv", index=False)
        print("✅ Risk scores saved")
    else:
        print("⚠️ Risk scores not saved due to missing data")

    # Save data indices for reproducibility
    indices_data = {
        'train_indices': train_indices.tolist() if hasattr(train_indices, 'tolist') else list(train_indices),
        'test_indices': test_indices.tolist() if hasattr(test_indices, 'tolist') else list(test_indices),
        'val_indices': val_indices.tolist() if val_indices is not None and hasattr(val_indices, 'tolist') else (
            list(val_indices) if val_indices is not None else [])
    }
    import json

    with open(f"{config['global']['outdir']}/data_indices.json", 'w') as f:
        json.dump(indices_data, f)
    print("✅ Data indices saved for reproducibility")

    print(f"✅ Results saved to: {config['global']['outdir']}")

except Exception as e:
    print(f"❌ Save failed: {e}")
