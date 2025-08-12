import torch
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Union
import h5py
import pickle
import os
from datetime import datetime


class nadata(object):
    """
    NNEA's core data class for storing data
    
    Refactored concise data structure design:
    1. **Expression Matrix Data (X)**: Rows are genes, columns are samples, supports sparse matrix format
    2. **Phenotype Data (Meta)**: Rows are samples, columns are sample features, includes train/test/val indices
    3. **Gene Data (Var)**: Rows are genes, columns are gene features, including gene names, types, importance, etc.
    4. **Prior Knowledge (Prior)**: Geneset 0,1 sparse matrix, representing whether genes are in genesets
    5. **Model Container (Model)**: Stores all models, configurations, training history, etc.
    """

    def __init__(self, X=None, Meta=None, Var=None, Prior=None, uns=None):
        """
        Initialize nadata object
        
        Parameters:
        -----------
        X : Optional[Union[np.ndarray, torch.Tensor, pd.DataFrame]]
            Expression matrix, shape (genes, samples)
        Meta : Optional[Union[np.ndarray, pd.DataFrame]]
            Phenotype data, shape (samples, features), includes train/test/val indices
        Var : Optional[Union[np.ndarray, pd.DataFrame]]
            Gene data, shape (genes, features)
        Prior : Optional[Union[np.ndarray, torch.Tensor]]
            Prior knowledge matrix, shape (genesets, genes)
        uns : Optional[Dict[str, Any]]
            Dictionary for storing additional information, such as PCA data, dataset info, etc.
        """
        # Core data
        self.X = X          # Expression matrix
        self.Meta = Meta    # Phenotype data (includes indices)
        self.Var = Var      # Gene data
        self.Prior = Prior  # Prior knowledge
        self.uns = uns if uns is not None else {}  # Additional information dictionary
        
        # Model container - contains all model-related content
        self.Model = ModelContainer(self)
        # Set ModelContainer's reference to nadata
        self.Model._nadata = self

    def save(self, filepath: str, format: str = 'pt', save_data: bool = True):
        """
        Save nadata object
        
        Parameters:
        -----------
        filepath : str
            Save path
        format : str
            Save format, supports 'pt', 'h5', 'pickle'
        save_data : bool
            Whether to save data, if False only save models and configurations
        """
        if format == 'pt':
            # Use new save function
            from ._save import save_project
            save_project(self, filepath, save_data=save_data)
        elif format == 'h5':
            with h5py.File(filepath, 'w') as f:
                # Save expression matrix
                if self.X is not None:
                    if isinstance(self.X, torch.Tensor):
                        f.create_dataset('X', data=self.X.cpu().numpy())
                    else:
                        f.create_dataset('X', data=self.X)
                
                # Save phenotype data
                if self.Meta is not None:
                    if isinstance(self.Meta, pd.DataFrame):
                        f.create_dataset('Meta', data=self.Meta.values)
                        f.attrs['Meta_columns'] = self.Meta.columns.tolist()
                    else:
                        f.create_dataset('Meta', data=self.Meta)
                
                # Save gene data
                if self.Var is not None:
                    if isinstance(self.Var, pd.DataFrame):
                        f.create_dataset('Var', data=self.Var.values)
                        f.attrs['Var_columns'] = self.Var.columns.tolist()
                    else:
                        f.create_dataset('Var', data=self.Var)
                
                # Save prior knowledge
                if self.Prior is not None:
                    if isinstance(self.Prior, torch.Tensor):
                        f.create_dataset('Prior', data=self.Prior.cpu().numpy())
                    else:
                        f.create_dataset('Prior', data=self.Prior)
                
                # Save uns dictionary
                if hasattr(self, 'uns') and self.uns:
                    # Convert uns dictionary to JSON string for storage
                    import json
                    uns_json = json.dumps(self.uns, default=str)
                    f.attrs['uns'] = uns_json
                
                # 保存模型容器
                if self.Model:
                    f.attrs['Model'] = str(self.Model)
                
        elif format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def load(self, filepath: str):
        """
        Load nadata object
        
        Parameters:
        -----------
        filepath : str
            File path
        """
        if filepath.endswith('.h5'):
            with h5py.File(filepath, 'r') as f:
                # Load expression matrix
                if 'X' in f:
                    self.X = f['X'][:]
                
                # Load phenotype data
                if 'Meta' in f:
                    meta_data = f['Meta'][:]
                    if 'Meta_columns' in f.attrs:
                        meta_cols = f.attrs['Meta_columns']
                        self.Meta = pd.DataFrame(meta_data, columns=meta_cols)
                    else:
                        self.Meta = meta_data
                
                # Load gene data
                if 'Var' in f:
                    var_data = f['Var'][:]
                    if 'Var_columns' in f.attrs:
                        var_cols = f.attrs['Var_columns']
                        self.Var = pd.DataFrame(var_data, columns=var_cols)
                    else:
                        self.Var = var_data
                
                # Load prior knowledge
                if 'Prior' in f:
                    self.Prior = f['Prior'][:]
                
                # Load uns dictionary
                if 'uns' in f.attrs:
                    import json
                    uns_json = f.attrs['uns']
                    self.uns = json.loads(uns_json)
                else:
                    self.uns = {}
                
                # Load model container
                if 'Model' in f.attrs:
                    # Model container loading logic needs to be implemented here
                    pass
                    
        elif filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                loaded_data = pickle.load(f)
                self.__dict__.update(loaded_data.__dict__)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

    def print(self, module: Optional[str] = None):
        """
        Print basic information of the class, supports printing specific modules
        
        Parameters:
        -----------
        module : Optional[str]
            Module name to print, if None print all information
        """
        if module is None:
            print("=== NNEA Data Object ===")
            print(f"Expression matrix (X): {self.X.shape if self.X is not None else 'None'}")
            print(f"Phenotype data (Meta): {self.Meta.shape if self.Meta is not None else 'None'}")
            print(f"Gene data (Var): {self.Var.shape if self.Var is not None else 'None'}")
            print(f"Prior knowledge (Prior): {self.Prior.shape if self.Prior is not None else 'None'}")
            print(f"Additional info (uns): {len(self.uns) if hasattr(self, 'uns') and self.uns else 0} keys")
            print(f"Model container: {self.Model}")
        elif module == 'X':
            print(f"Expression matrix shape: {self.X.shape if self.X is not None else 'None'}")
        elif module == 'Meta':
            print(f"Phenotype data shape: {self.Meta.shape if self.Meta is not None else 'None'}")
            if self.Meta is not None and hasattr(self.Meta, 'columns'):
                print(f"Meta columns: {list(self.Meta.columns)}")
        elif module == 'Var':
            print(f"Gene data shape: {self.Var.shape if self.Var is not None else 'None'}")
        elif module == 'Prior':
            print(f"Prior knowledge shape: {self.Prior.shape if self.Prior is not None else 'None'}")
        elif module == 'Model':
            print(f"Model container: {self.Model}")
        elif module == 'uns':
            print(f"Additional info (uns): {len(self.uns) if hasattr(self, 'uns') and self.uns else 0} keys")
            if hasattr(self, 'uns') and self.uns:
                for key, value in self.uns.items():
                    if isinstance(value, (list, np.ndarray)):
                        print(f"  {key}: {type(value).__name__} with shape {getattr(value, 'shape', len(value))}")
                    else:
                        print(f"  {key}: {value}")
        else:
            print(f"Unknown module: {module}")

    def copy(self):
        """
        Deep copy nadata object
        
        Returns:
        --------
        nadata
            Copied nadata object
        """
        import copy
        return copy.deepcopy(self)

    def subset(self, samples: Optional[list] = None, genes: Optional[list] = None):
        """
        Subset selection
        
        Parameters:
        -----------
        samples : Optional[list]
            Sample index list
        genes : Optional[list]
            Gene index list
            
        Returns:
        --------
        nadata
            Subset nadata object
        """
        new_nadata = self.copy()
        
        if genes is not None:
            if self.X is not None:
                new_nadata.X = self.X[genes, :] if self.X is not None else None
            if self.Var is not None:
                new_nadata.Var = self.Var.iloc[genes] if isinstance(self.Var, pd.DataFrame) else self.Var[genes]
            if self.Prior is not None:
                new_nadata.Prior = self.Prior[:, genes]
        
        if samples is not None:
            if self.X is not None:
                new_nadata.X = self.X[:, samples] if self.X is not None else None
            if self.Meta is not None:
                new_nadata.Meta = self.Meta.iloc[samples] if isinstance(self.Meta, pd.DataFrame) else self.Meta[samples]
        
        return new_nadata

    def merge(self, other: 'nadata'):
        """
        Merge two nadata objects
        
        Parameters:
        -----------
        other : nadata
            nadata object to merge
        """
        # Merge expression matrix
        if self.X is not None and other.X is not None:
            self.X = np.concatenate([self.X, other.X], axis=1)
        
        # Merge phenotype data
        if self.Meta is not None and other.Meta is not None:
            if isinstance(self.Meta, pd.DataFrame) and isinstance(other.Meta, pd.DataFrame):
                self.Meta = pd.concat([self.Meta, other.Meta], axis=0, ignore_index=True)
            else:
                self.Meta = np.concatenate([self.Meta, other.Meta], axis=0)
        
        # Merge gene data
        if self.Var is not None and other.Var is not None:
            if isinstance(self.Var, pd.DataFrame) and isinstance(other.Var, pd.DataFrame):
                self.Var = pd.concat([self.Var, other.Var], axis=0, ignore_index=True)
            else:
                self.Var = np.concatenate([self.Var, other.Var], axis=0)
        
        # Merge prior knowledge
        if self.Prior is not None and other.Prior is not None:
            self.Prior = np.concatenate([self.Prior, other.Prior], axis=1)
        
        # Merge uns dictionary
        if hasattr(self, 'uns') and hasattr(other, 'uns'):
            if self.uns is None:
                self.uns = {}
            if other.uns is not None:
                self.uns.update(other.uns)
        
        # Merge model container
        self.Model.merge(other.Model)

    def build(self):
        """
        Build model, model is placed in nadata's Model container
        """
        from ..model import build
        build(self)

    def train(self, verbose: int = 1):
        """
        Train model, supports verbose parameter
        
        Parameters:
        -----------
        verbose : int
            Verbosity level: 0-only show progress bar, 1-show training details, 2-show debug information
        """
        from ..model import train
        train(self, verbose=verbose)

    def evaluate(self):
        """
        Evaluate model
        """
        from ..model import eval
        eval(self)

    def explain(self, verbose: int = 1):
        """
        Model explanation, supports verbose parameter
        
        Parameters:
        -----------
        verbose : int
            Verbosity level: 0-only show progress bar, 1-show explanation details, 2-show debug information
        """
        from ..model import explain
        explain(self, verbose=verbose)

    def compare_baseline_models(self, save_path="results", verbose: int = 1):
        """
        Compare baseline model performance
        
        Parameters:
        -----------
        save_path : str
            Results save path
        verbose : int
            Verbosity level: 0-only show progress bar, 1-show basic information, 2-show detailed results
            
        Returns:
        --------
        dict
            Comparison results summary
        """
        import logging
        import os
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import SVC, LinearSVC
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        
        logger = logging.getLogger(__name__)
        
        if verbose >= 1:
            logger.info("Starting baseline model comparison experiment...")
        
        # Create results directory
        os.makedirs(save_path, exist_ok=True)
        
        # Get data indices
        train_indices = self.Model.get_indices('train')
        test_indices = self.Model.get_indices('test')
        
        if train_indices is None or test_indices is None:
            logger.warning("Data indices not set, will automatically split data...")
            # Manually set training and test indices
            n_samples = self.X.shape[1]  # Number of samples
            indices = list(range(n_samples))
            
            # Get split parameters from configuration
            config = self.Model.get_config()
            test_size = config.get('dataset', {}).get('test_size', 0.2)
            random_state = config.get('global', {}).get('seed', 42)
            
            # Stratified sampling to split data
            target_column = config.get('dataset', {}).get('target_column', 'class')
            y = self.Meta[target_column].values
            
            train_indices, test_indices = train_test_split(
                indices, 
                test_size=test_size, 
                stratify=y, 
                random_state=random_state
            )
            
            # Set indices to Model container
            self.Model.set_indices(train_idx=train_indices, test_idx=test_indices)
        
        # Ensure indices are integer type
        train_indices = [int(i) for i in train_indices]
        test_indices = [int(i) for i in test_indices]
        
        # Get target column name
        target_column = self.Model.get_config().get('dataset', {}).get('target_column', 'class')
        
        # Get training and test data
        X_train = self.X[:, train_indices].T  # Transpose to (num_samples, num_features)
        X_test = self.X[:, test_indices].T    # Transpose to (num_samples, num_features)
        y_train = self.Meta.iloc[train_indices][target_column].values
        y_test = self.Meta.iloc[test_indices][target_column].values
        
        if verbose >= 1:
            logger.info(f"Training set size: {X_train.shape}")
            logger.info(f"Test set size: {X_test.shape}")
            logger.info(f"Class distribution - Training set: {np.bincount(y_train)}")
            logger.info(f"Class distribution - Test set: {np.bincount(y_test)}")
        
        # Data preprocessing
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define baseline models
        models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'MLPClassifier': MLPClassifier(hidden_layer_sizes=(100,), random_state=42, max_iter=1000),
            'LinearSVM': LinearSVC(random_state=42, max_iter=1000),
            'RBFSVM': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        # Train and evaluate models
        results = {}
        if verbose >= 1:
            logger.info("Starting training and evaluating baseline models...")
        
        for name, model in models.items():
            if verbose >= 1:
                logger.info(f"Training {name}...")
            
            try:
                # Train model
                if name == 'LinearSVM':
                    # LinearSVC does not support probability prediction, use SVC instead
                    model = SVC(kernel='linear', probability=True, random_state=42)
                
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.0
                
                results[name] = {
                    'accuracy': accuracy,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'auc': auc,
                    'model': model
                }
                
                if verbose >= 1:
                    logger.info(f"  {name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
                    
            except Exception as e:
                if verbose >= 1:
                    logger.warning(f"  {name} Training failed: {e}")
                continue
        
        # Save results to Model container
        self.Model.add_metadata('baseline_results', results)
        
        # Create comparison plot
        if results:
            # Create performance comparison plot
            metrics = ['accuracy', 'f1', 'precision', 'recall', 'auc']
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, metric in enumerate(metrics):
                if i < len(axes):
                    values = [results[name][metric] for name in results.keys()]
                    names = list(results.keys())
                    
                    axes[i].bar(names, values)
                    axes[i].set_title(f'{metric.upper()} Comparison')
                    axes[i].set_ylabel(metric.upper())
                    axes[i].tick_params(axis='x', rotation=45)
            
            # Hide extra subplots
            for i in range(len(metrics), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'baseline_model_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create ROC curves
            plt.figure(figsize=(10, 8))
            for name, result in results.items():
                if result['auc'] > 0:
                    y_pred_proba = result['model'].predict_proba(X_test_scaled)[:, 1]
                    from sklearn.metrics import roc_curve
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    plt.plot(fpr, tpr, label=f'{name} (AUC = {result["auc"]:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves Comparison')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_path, 'roc_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save results table
            results_df = pd.DataFrame([
                {
                    'Model': name,
                    'Accuracy': result['accuracy'],
                    'F1_Score': result['f1'],
                    'Precision': result['precision'],
                    'Recall': result['recall'],
                    'AUC': result['auc']
                }
                for name, result in results.items()
            ])
            
            results_df.to_csv(os.path.join(save_path, 'baseline_model_results.csv'), index=False)
            
            # Get best model
            best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
            best_auc = results[best_model_name]['auc']
            
            if verbose >= 1:
                logger.info(f"Best baseline model: {best_model_name}")
                logger.info(f"Best AUC: {best_auc:.4f}")
            
            # Save detailed report
            with open(os.path.join(save_path, 'detailed_report.txt'), 'w', encoding='utf-8') as f:
                f.write("Baseline Model Comparison Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Dataset size: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples\n")
                f.write(f"Number of features: {X_train.shape[1]}\n")
                f.write(f"Class distribution - Training set: {np.bincount(y_train)}\n")
                f.write(f"Class distribution - Test set: {np.bincount(y_test)}\n\n")
                
                f.write("Model Performance Comparison:\n")
                f.write("-" * 30 + "\n")
                for name, result in results.items():
                    f.write(f"{name}:\n")
                    f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
                    f.write(f"  F1 Score: {result['f1']:.4f}\n")
                    f.write(f"  Precision: {result['precision']:.4f}\n")
                    f.write(f"  Recall: {result['recall']:.4f}\n")
                    f.write(f"  AUC: {result['auc']:.4f}\n\n")
                
                f.write(f"Best Model: {best_model_name}\n")
                f.write(f"Best AUC: {best_auc:.4f}\n")
            
            return {
                'best_model': best_model_name,
                'best_auc': best_auc,
                'results': results,
                'summary': results_df
            }
        
        else:
            logger.error("No models trained successfully")
            return None


class ModelContainer:
    """
    Model container class, used to manage all model-related content
    Including models, configuration, training history, data indices, etc.
    """
    
    def __init__(self, nadata_obj=None):
        """
        Initialize model container
        
        Parameters:
        -----------
        nadata_obj : Optional[nadata]
            Associated nadata object
        """
        # Model dictionary
        self.models = {}
        
        # Configuration information
        self.config = {}
        
        # Training history
        self.train_results = {}
        
        # Data indices (train/test/val)
        self.indices = {
            'train': None,
            'test': None,
            'val': None
        }
        
        # Other metadata
        self.metadata = {}
        
        # Associated nadata object
        self._nadata = nadata_obj
    
    def add_model(self, name: str, model):
        """
        Add model
        
        Parameters:
        -----------
        name : str
            Model name
        model : Any
            Model object
        """
        self.models[name] = model
    
    def get_model(self, name: str):
        """
        Get model
        
        Parameters:
        -----------
        name : str
            Model name
            
        Returns:
        --------
        Any
            Model object
        """
        return self.models.get(name)
    
    def has_model(self, name: str) -> bool:
        """
        Check if specified model exists
        
        Parameters:
        -----------
        name : str
            Model name
            
        Returns:
        --------
        bool
            Whether it exists
        """
        return name in self.models
    
    def list_models(self) -> list:
        """
        List all model names
        
        Returns:
        --------
        list
            List of model names
        """
        return list(self.models.keys())
    
    def _print_config_details(self, config: dict, indent: str = ""):
        """
        Recursively print configuration details
        
        Parameters:
        -----------
        config : dict
            配置字典
        indent : str
            缩进字符串
        """
        import logging
        logger = logging.getLogger(__name__)
        
        for key, value in config.items():
            if isinstance(value, dict):
                logger.info(f"{indent}📁 {key}:")
                self._print_config_details(value, indent + "  ")
            elif isinstance(value, list):
                logger.info(f"{indent}📋 {key}: {value}")
            elif isinstance(value, bool):
                status = "✅" if value else "❌"
                logger.info(f"{indent}{status} {key}: {value}")
            elif isinstance(value, (int, float)):
                logger.info(f"{indent}🔢 {key}: {value}")
            else:
                logger.info(f"{indent}📄 {key}: {value}")

    def set_config(self, config: dict):
        """
        Set configuration
        
        Parameters:
        -----------
        config : dict
            配置字典
        """
        self.config = config
        
        # Create output directory
        import os
        outdir = config.get('global', {}).get('outdir', 'experiment/test')
        os.makedirs(outdir, exist_ok=True)
        
        # Set logging to specified directory
        from ..logging_utils import setup_logging
        import logging
        
        # Create log subdirectory
        log_dir = os.path.join(outdir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Reconfigure logging to save log files to outdir/logs directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")
        
        # Reconfigure logging
        setup_logging(log_file=log_file)
        
        # Log configuration setting information
        logger = logging.getLogger(__name__)
        logger.info(f"Configuration set, output directory: {outdir}")
        logger.info(f"Log file: {log_file}")
        
        # Detailed print configuration parameters
        logger.info("=" * 60)
        logger.info("📋 NNEA Configuration File Detailed Parameters:")
        logger.info("=" * 60)
        self._print_config_details(config)
        logger.info("=" * 60)
        
        # Store output directory information in config for other modules to use
        self.outdir = outdir
    
    def get_config(self) -> dict:
        """
        Get configuration
        
        Returns:
        --------
        dict
            配置字典
        """
        return self.config
    
    def set_train_results(self, results: dict):
        """
        Set training results
        
        Parameters:
        -----------
        results : dict
            训练结果字典
        """
        self.train_results = results
    
    def get_train_results(self) -> dict:
        """
        Get training results
        
        Returns:
        --------
        dict
            训练结果字典
        """
        return self.train_results
    
    def set_indices(self, train_idx=None, test_idx=None, val_idx=None):
        """
        Set data indices to Model container's indices
        
        Parameters:
        -----------
        train_idx : Optional[list]
            训练集索引
        test_idx : Optional[list]
            测试集索引
        val_idx : Optional[list]
            验证集索引
        """
        # Directly store in Model container's indices property
        if train_idx is not None:
            self.indices['train'] = train_idx
        if test_idx is not None:
            self.indices['test'] = test_idx
        if val_idx is not None:
            self.indices['val'] = val_idx
        else:
            # If val_idx is None, remove val index from storage
            self.indices['val'] = None
    
    def get_indices(self, split: str = None):
        """
        Get data indices
        
        Parameters:
        -----------
        split : Optional[str]
            分割类型（'train', 'test', 'val'），如果为None则返回所有索引
            
        Returns:
        --------
        Union[list, dict]
            索引列表或字典
        """
        # Directly get from Model container's indices property
        if split is None:
            return self.indices
        return self.indices.get(split)
    
    def get_var_indices(self, split: str = None):
        """
        Get data indices from nadata.Var
        
        Parameters:
        -----------
        split : Optional[str]
            分割类型（'train', 'test', 'val'），如果为None则返回所有索引
            
        Returns:
        --------
        Union[list, dict]
            索引列表或字典
        """
        if hasattr(self, '_nadata') and self._nadata is not None:
            try:
                import pandas as pd
                if self._nadata.Var is not None and 'indices' in self._nadata.Var.columns and len(self._nadata.Var) > 0:
                    indices_data = self._nadata.Var.loc[0, 'indices']
                    if split is None:
                        return indices_data
                    return indices_data.get(split) if isinstance(indices_data, dict) else None
            except ImportError:
                # If pandas is not available, get from _indices property
                if hasattr(self._nadata, '_indices') and self._nadata._indices:
                    if split is None:
                        return self._nadata._indices
                    return self._nadata._indices.get(split)
        return None
    
    def add_metadata(self, key: str, value):
        """
        Add metadata
        
        Parameters:
        -----------
        key : str
            键名
        value : Any
            值
        """
        self.metadata[key] = value
    
    def get_metadata(self, key: str = None):
        """
        Get metadata
        
        Parameters:
        -----------
        key : Optional[str]
            键名，如果为None则返回所有元数据
            
        Returns:
        --------
        Any
            元数据值或字典
        """
        if key is None:
            return self.metadata
        return self.metadata.get(key)
    
    def merge(self, other: 'ModelContainer'):
        """
        Merge another model container
        
        Parameters:
        -----------
        other : ModelContainer
            要合并的模型容器
        """
        # Merge models
        self.models.update(other.models)
        
        # Merge configuration (based on other)
        if other.config:
            self.config = other.config
        
        # Merge training results
        if other.train_results:
            self.train_results.update(other.train_results)
        
        # Merge indices
        for key in ['train', 'test', 'val']:
            if other.indices[key] is not None:
                self.indices[key] = other.indices[key]
        
        # Merge metadata
        self.metadata.update(other.metadata)
    
    def __str__(self):
        """
        String representation
        """
        return f"ModelContainer(models={list(self.models.keys())}, config_keys={list(self.config.keys())}, train_results_keys={list(self.train_results.keys())})"
    
    def __repr__(self):
        """
        Detailed string representation
        """
        return self.__str__()
    
    def __setitem__(self, key, value):
        """
        Support dictionary assignment, store value in models dictionary
        
        Parameters:
        -----------
        key : str
            键名
        value : Any
            要存储的值
        """
        self.models[key] = value
    
    def __getitem__(self, key):
        """
        Support dictionary access, get value from models dictionary
        
        Parameters:
        -----------
        key : str
            键名
            
        Returns:
        --------
        Any
            存储的值
        """
        return self.models[key]
    
    def __contains__(self, key):
        """
        Support in operator, check if key exists in models dictionary
        
        Parameters:
        -----------
        key : str
            键名
            
        Returns:
        --------
        bool
            是否存在
        """
        return key in self.models
    
    def get(self, key, default=None):
        """
        Get value, return default if key does not exist
        
        Parameters:
        -----------
        key : str
            键名
        default : Any
            默认值
            
        Returns:
        --------
        Any
            存储的值或默认值
        """
        return self.models.get(key, default)