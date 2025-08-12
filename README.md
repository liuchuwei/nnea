# NNEA (Neural Network with Explainable Architecture)

NNEA is an interpretable neural network package specifically designed for biological research, particularly suitable for transcriptomics data analysis. The package provides a complete workflow from data loading, preprocessing, model training to result interpretation.

## Features

- 🧬 **Biology-oriented**: Specifically designed for gene expression data analysis
- 🔍 **Interpretability**: Provides model interpretation and feature importance analysis
- 📊 **Visualization**: Rich charts and result display functionality
- 🚀 **Ease of use**: Simple API interface, quick to get started
- 📈 **High performance**: Efficient implementation based on PyTorch

## Dependencies

The NNEA package requires the following Python dependencies:

### Core Dependencies
- **torch** >= 1.9.0 - PyTorch deep learning framework
- **numpy** >= 1.21.0 - Numerical computing library
- **pandas** >= 1.3.0 - Data processing library
- **scikit-learn** >= 1.0.0 - Machine learning library

### Data Science and Visualization
- **matplotlib** >= 3.5.0 - Plotting library
- **seaborn** >= 0.11.0 - Statistical visualization
- **plotly** >= 5.0.0 - Interactive charts
- **umap-learn** >= 0.5.0 - Dimensionality reduction algorithm

### Model Interpretation
- **shap** >= 0.40.0 - SHAP value calculation
- **lime** >= 0.2.0 - LIME interpreter

### Data Processing
- **scipy** >= 1.7.0 - Scientific computing
- **h5py** >= 3.1.0 - HDF5 file support
- **imbalanced-learn** >= 0.8.0 - Imbalanced data processing

### Other Tools
- **toml** >= 0.10.0 - Configuration file parsing
- **networkx** >= 2.6.0 - Network analysis
- **pyyaml** >= 5.4.0 - YAML file support

## Installation

### Method 1: Install from source (recommended)

```bash
# Clone repository
git clone https://github.com/liuchuwei/nnea.git
cd nnea

# Install dependencies
pip install -r requirements.txt

# Install nnea package
pip install -e .
```

### Method 2: Direct installation

```bash
# Install using pip
pip install nnea
```

### Method 3: Using conda (if available)

```bash
conda install -c conda-forge nnea
```

## Quick Start

### Basic Usage

```python
import nnea as na

# Create NNEA project
nadata = na.io.CreateNNEA(config="config/your_config.toml")

# Build model
na.build(nadata)

# Train model
na.train(nadata, verbose=2)

# Evaluate model
na.eval(nadata, split='test')

# Model interpretation
na.explain(nadata, method='importance')

# Save model
na.save_model(nadata, "results/model.pt")
```

### Immunotherapy Response Prediction Example

```python
import nnea as na
import numpy as np

# Set random seed
np.random.seed(42)

# Create project
nadata = na.io.CreateNNEA(config="config/ccRCC_imm.toml")

# Complete training workflow
na.build(nadata)
na.train(nadata, verbose=2)
na.eval(nadata, split='test')
na.explain(nadata, method='importance')

# Save results
na.save_model(nadata, "results/nnea_ccrcc_model.pt")
na.get_summary(nadata)
```

## Main Function Modules

### 1. Data Loading (`nnea.io`)
- `CreateNNEA()`: Create NNEA project
- `nadata`: Data container object

### 2. Model Building (`nnea.model`)
- `build()`: Build neural network model
- `train()`: Train model
- `eval()`: Evaluate model performance
- `explain()`: Model interpretation

### 3. Data Preprocessing (`nnea.data_factory`)
- `pp`: Data preprocessing tools

### 4. Visualization (`nnea.plot`)
- Training curves
- Feature importance
- Geneset networks

### 5. Utility Functions (`nnea.utils`)
- Helper functions
- Evaluation metrics

## Configuration

NNEA uses TOML format configuration files, main configuration items include:

```toml
[data]
input_path = "data/your_data.csv"
output_path = "results/"

[model]
model_type = "transformer"
hidden_size = 256
num_layers = 4

[training]
epochs = 100
batch_size = 32
learning_rate = 0.001
```

## Supported Biological Applications

- 🧬 **Gene Expression Analysis**: Transcriptomics data processing
- 🎯 **Immunotherapy Prediction**: Patient response prediction
- 📊 **Survival Analysis**: Patient prognosis prediction
- 🔬 **Drug Sensitivity**: Drug response prediction
- 🧪 **Single-cell Analysis**: Single-cell transcriptomics data

## Contributing

Welcome to contribute code! Please follow these steps:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- Author: Chuwei Liu
- Email: liuchw26@mail.sysu.edu.cn
- Project link: https://github.com/liuchuwei/nnea

## Changelog

### v0.1.2
- Added new model interpretation functionality
- Improved data preprocessing workflow
- Optimized visualization functionality

### v0.1.0
- Initial version release
- Basic functionality implementation