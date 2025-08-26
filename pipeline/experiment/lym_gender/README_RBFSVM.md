# RBF SVM Experiment for Lymphocyte Gender Classification

This directory contains the RBF SVM experiment for lymphocyte gender classification using the NNEA framework.

## Files

- `lym_gender_RBFSVM.toml` - Configuration file for the RBF SVM experiment
- `lym_gender_RBFSVM_experiment.py` - Main experiment script
- `run_rbf_svm_experiment.py` - Runner script to execute the experiment
- `README_RBFSVM.md` - This documentation file

## Configuration

The experiment uses the following key parameters:

### Model Parameters
- **C**: Regularization parameter [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
- **gamma**: Kernel coefficient [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
- **kernel**: RBF kernel
- **class_weight**: Balanced class weights
- **probability**: True (for probability estimates)

### Feature Selection
- **method**: Lasso-based feature selection
- **n_features**: 7000 features
- **alpha**: 0.01

### Cross-validation
- **cv_folds**: 5-fold stratified cross-validation
- **scoring**: ROC AUC

## Usage

### Method 1: Using the runner script (Recommended)
```bash
cd scientist/nnea/pipeline/experiment/lym_gender
python run_rbf_svm_experiment.py
```

### Method 2: Manual execution
```bash
cd scientist/nnea/pipeline/experiment/lym_gender
cp lym_gender_RBFSVM.toml config.toml
python lym_gender_RBFSVM_experiment.py
```

## Output

The experiment will create the following outputs in the `experiment/lym_gender_RBFSVM/` directory:

- `lym_gender_RBFSVM.pkl` - Saved model and results
- `rbf_svm_experiment.log` - Detailed experiment log
- `rbf_svm_config.toml` - Saved configuration
- `training_summary.txt` - Summary of results

## Expected Results

The experiment will output:
- Best hyperparameters found through grid search
- Cross-validation AUC score
- Test set performance metrics (AUC, F1, precision, recall, accuracy)
- Classification report
- Model predictions and probabilities

## Dependencies

- nnea framework
- scikit-learn
- numpy
- torch
- toml

## Notes

- The experiment uses stratified sampling to maintain class balance
- Data is preprocessed with standardization for RBF SVM
- Feature selection is applied to reduce dimensionality
- The model supports probability estimation for ROC AUC calculation
