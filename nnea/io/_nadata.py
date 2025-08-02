import torch
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Union
import h5py
import pickle


class nadata(object):
    """
    NNEA的核心数据类，用来储存数据
    
    重构后的简洁数据结构设计：
    1. **表达矩阵数据（X）**: 行是基因数，列是样本数，支持稀疏矩阵格式
    2. **表型数据（Meta）**: 行是样本数，列是样本的特征，包含train/test/val索引
    3. **基因数据（Var）**: 行是基因数，列是基因特征，包括基因名称、类型、重要性等
    4. **先验知识（Prior）**: 基因集的0，1稀疏矩阵，代表基因是否在基因集合里
    5. **模型容器（Model）**: 储存所有模型、配置、训练历史等
    """

    def __init__(self, X=None, Meta=None, Var=None, Prior=None):
        """
        初始化nadata对象
        
        Parameters:
        -----------
        X : Optional[Union[np.ndarray, torch.Tensor, pd.DataFrame]]
            表达矩阵，形状为(基因数, 样本数)
        Meta : Optional[Union[np.ndarray, pd.DataFrame]]
            表型数据，形状为(样本数, 特征数)，包含train/test/val索引
        Var : Optional[Union[np.ndarray, pd.DataFrame]]
            基因数据，形状为(基因数, 特征数)
        Prior : Optional[Union[np.ndarray, torch.Tensor]]
            先验知识矩阵，形状为(基因集数, 基因数)
        """
        # 核心数据
        self.X = X          # 表达矩阵
        self.Meta = Meta    # 表型数据（包含索引）
        self.Var = Var      # 基因数据
        self.Prior = Prior  # 先验知识
        
        # 模型容器 - 包含所有模型相关的内容
        self.Model = ModelContainer()
        # 设置ModelContainer对nadata的引用
        self.Model._nadata = self

    def save(self, filepath: str, format: str = 'pt', save_data: bool = True):
        """
        保存nadata对象
        
        Parameters:
        -----------
        filepath : str
            保存路径
        format : str
            保存格式，支持'pt', 'h5', 'pickle'
        save_data : bool
            是否保存数据，如果为False只保存模型和配置
        """
        if format == 'pt':
            # 使用新的保存函数
            from ._save import save_project
            save_project(self, filepath, save_data=save_data)
        elif format == 'h5':
            with h5py.File(filepath, 'w') as f:
                # 保存表达矩阵
                if self.X is not None:
                    if isinstance(self.X, torch.Tensor):
                        f.create_dataset('X', data=self.X.cpu().numpy())
                    else:
                        f.create_dataset('X', data=self.X)
                
                # 保存表型数据
                if self.Meta is not None:
                    if isinstance(self.Meta, pd.DataFrame):
                        f.create_dataset('Meta', data=self.Meta.values)
                        f.attrs['Meta_columns'] = self.Meta.columns.tolist()
                    else:
                        f.create_dataset('Meta', data=self.Meta)
                
                # 保存基因数据
                if self.Var is not None:
                    if isinstance(self.Var, pd.DataFrame):
                        f.create_dataset('Var', data=self.Var.values)
                        f.attrs['Var_columns'] = self.Var.columns.tolist()
                    else:
                        f.create_dataset('Var', data=self.Var)
                
                # 保存先验知识
                if self.Prior is not None:
                    if isinstance(self.Prior, torch.Tensor):
                        f.create_dataset('Prior', data=self.Prior.cpu().numpy())
                    else:
                        f.create_dataset('Prior', data=self.Prior)
                
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
        加载nadata对象
        
        Parameters:
        -----------
        filepath : str
            文件路径
        """
        if filepath.endswith('.h5'):
            with h5py.File(filepath, 'r') as f:
                # 加载表达矩阵
                if 'X' in f:
                    self.X = f['X'][:]
                
                # 加载表型数据
                if 'Meta' in f:
                    meta_data = f['Meta'][:]
                    if 'Meta_columns' in f.attrs:
                        meta_cols = f.attrs['Meta_columns']
                        self.Meta = pd.DataFrame(meta_data, columns=meta_cols)
                    else:
                        self.Meta = meta_data
                
                # 加载基因数据
                if 'Var' in f:
                    var_data = f['Var'][:]
                    if 'Var_columns' in f.attrs:
                        var_cols = f.attrs['Var_columns']
                        self.Var = pd.DataFrame(var_data, columns=var_cols)
                    else:
                        self.Var = var_data
                
                # 加载先验知识
                if 'Prior' in f:
                    self.Prior = f['Prior'][:]
                
                # 加载模型容器
                if 'Model' in f.attrs:
                    # 这里需要实现模型容器的加载逻辑
                    pass
                    
        elif filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                loaded_data = pickle.load(f)
                self.__dict__.update(loaded_data.__dict__)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

    def print(self, module: Optional[str] = None):
        """
        打印类的基本信息，支持打印特定模块
        
        Parameters:
        -----------
        module : Optional[str]
            要打印的模块名称，如果为None则打印所有信息
        """
        if module is None:
            print("=== NNEA Data Object ===")
            print(f"Expression matrix (X): {self.X.shape if self.X is not None else 'None'}")
            print(f"Phenotype data (Meta): {self.Meta.shape if self.Meta is not None else 'None'}")
            print(f"Gene data (Var): {self.Var.shape if self.Var is not None else 'None'}")
            print(f"Prior knowledge (Prior): {self.Prior.shape if self.Prior is not None else 'None'}")
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
        else:
            print(f"Unknown module: {module}")

    def copy(self):
        """
        深拷贝nadata对象
        
        Returns:
        --------
        nadata
            拷贝的nadata对象
        """
        import copy
        return copy.deepcopy(self)

    def subset(self, samples: Optional[list] = None, genes: Optional[list] = None):
        """
        子集选择
        
        Parameters:
        -----------
        samples : Optional[list]
            样本索引列表
        genes : Optional[list]
            基因索引列表
            
        Returns:
        --------
        nadata
            子集nadata对象
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
        合并两个nadata对象
        
        Parameters:
        -----------
        other : nadata
            要合并的nadata对象
        """
        # 合并表达矩阵
        if self.X is not None and other.X is not None:
            self.X = np.concatenate([self.X, other.X], axis=1)
        
        # 合并表型数据
        if self.Meta is not None and other.Meta is not None:
            if isinstance(self.Meta, pd.DataFrame) and isinstance(other.Meta, pd.DataFrame):
                self.Meta = pd.concat([self.Meta, other.Meta], axis=0, ignore_index=True)
            else:
                self.Meta = np.concatenate([self.Meta, other.Meta], axis=0)
        
        # 合并基因数据
        if self.Var is not None and other.Var is not None:
            if isinstance(self.Var, pd.DataFrame) and isinstance(other.Var, pd.DataFrame):
                self.Var = pd.concat([self.Var, other.Var], axis=0, ignore_index=True)
            else:
                self.Var = np.concatenate([self.Var, other.Var], axis=0)
        
        # 合并先验知识
        if self.Prior is not None and other.Prior is not None:
            self.Prior = np.concatenate([self.Prior, other.Prior], axis=1)
        
        # 合并模型容器
        self.Model.merge(other.Model)

    def build(self):
        """
        构建模型，模型放入nadata的Model容器中
        """
        from ..model import build
        build(self)

    def train(self, verbose: int = 1):
        """
        训练模型，支持verbose参数
        
        Parameters:
        -----------
        verbose : int
            详细程度：0-只显示进度条，1-显示训练详情，2-显示调试信息
        """
        from ..model import train
        train(self, verbose=verbose)

    def evaluate(self):
        """
        评估模型
        """
        from ..model import eval
        eval(self)

    def explain(self, verbose: int = 1):
        """
        模型解释，支持verbose参数
        
        Parameters:
        -----------
        verbose : int
            详细程度：0-只显示进度条，1-显示解释详情，2-显示调试信息
        """
        from ..model import explain
        explain(self, verbose=verbose)

    def compare_baseline_models(self, save_path="results", verbose: int = 1):
        """
        比较基线模型性能
        
        Parameters:
        -----------
        save_path : str
            结果保存路径
        verbose : int
            详细程度：0-只显示进度条，1-显示基本信息，2-显示详细结果
            
        Returns:
        --------
        dict
            比较结果摘要
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
            logger.info("开始基线模型比较实验...")
        
        # 创建结果目录
        os.makedirs(save_path, exist_ok=True)
        
        # 获取数据索引
        train_indices = self.Model.get_indices('train')
        test_indices = self.Model.get_indices('test')
        
        if train_indices is None or test_indices is None:
            logger.warning("数据索引未设置，将自动分割数据...")
            # 手动设置训练和测试索引
            n_samples = self.X.shape[1]  # 样本数
            indices = list(range(n_samples))
            
            # 获取配置中的分割参数
            config = self.Model.get_config()
            test_size = config.get('dataset', {}).get('test_size', 0.2)
            random_state = config.get('global', {}).get('seed', 42)
            
            # 分层抽样分割数据
            target_column = config.get('dataset', {}).get('target_column', 'class')
            y = self.Meta[target_column].values
            
            train_indices, test_indices = train_test_split(
                indices, 
                test_size=test_size, 
                stratify=y, 
                random_state=random_state
            )
            
            # 设置索引到Model容器
            self.Model.set_indices(train_idx=train_indices, test_idx=test_indices)
        
        # 确保索引是整数类型
        train_indices = [int(i) for i in train_indices]
        test_indices = [int(i) for i in test_indices]
        
        # 获取目标列名
        target_column = self.Model.get_config().get('dataset', {}).get('target_column', 'class')
        
        # 获取训练和测试数据
        X_train = self.X[:, train_indices].T  # 转置为(样本数, 特征数)
        X_test = self.X[:, test_indices].T    # 转置为(样本数, 特征数)
        y_train = self.Meta.iloc[train_indices][target_column].values
        y_test = self.Meta.iloc[test_indices][target_column].values
        
        if verbose >= 1:
            logger.info(f"训练集大小: {X_train.shape}")
            logger.info(f"测试集大小: {X_test.shape}")
            logger.info(f"类别分布 - 训练集: {np.bincount(y_train)}")
            logger.info(f"类别分布 - 测试集: {np.bincount(y_test)}")
        
        # 数据预处理
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 定义基线模型
        models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'DecisionTree': DecisionTreeClassifier(random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'MLPClassifier': MLPClassifier(hidden_layer_sizes=(100,), random_state=42, max_iter=1000),
            'LinearSVM': LinearSVC(random_state=42, max_iter=1000),
            'RBFSVM': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        # 训练和评估模型
        results = {}
        if verbose >= 1:
            logger.info("开始训练和评估基线模型...")
        
        for name, model in models.items():
            if verbose >= 1:
                logger.info(f"训练 {name}...")
            
            try:
                # 训练模型
                if name == 'LinearSVM':
                    # LinearSVC不支持概率预测，使用SVC替代
                    model = SVC(kernel='linear', probability=True, random_state=42)
                
                model.fit(X_train_scaled, y_train)
                
                # 预测
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # 计算指标
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
                    logger.info(f"  {name} - 准确率: {accuracy:.4f}, AUC: {auc:.4f}")
                    
            except Exception as e:
                if verbose >= 1:
                    logger.warning(f"  {name} 训练失败: {e}")
                continue
        
        # 保存结果到Model容器
        self.Model.add_metadata('baseline_results', results)
        
        # 创建比较图
        if results:
            # 创建性能比较图
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
            
            # 隐藏多余的子图
            for i in range(len(metrics), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'baseline_model_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 创建ROC曲线
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
            
            # 保存结果表格
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
            
            # 获取最佳模型
            best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
            best_auc = results[best_model_name]['auc']
            
            if verbose >= 1:
                logger.info(f"最佳基线模型: {best_model_name}")
                logger.info(f"最佳AUC: {best_auc:.4f}")
            
            # 保存详细报告
            with open(os.path.join(save_path, 'detailed_report.txt'), 'w', encoding='utf-8') as f:
                f.write("基线模型比较实验报告\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"数据集大小: {X_train.shape[0]} 训练样本, {X_test.shape[0]} 测试样本\n")
                f.write(f"特征数量: {X_train.shape[1]}\n")
                f.write(f"类别分布 - 训练集: {np.bincount(y_train)}\n")
                f.write(f"类别分布 - 测试集: {np.bincount(y_test)}\n\n")
                
                f.write("模型性能比较:\n")
                f.write("-" * 30 + "\n")
                for name, result in results.items():
                    f.write(f"{name}:\n")
                    f.write(f"  准确率: {result['accuracy']:.4f}\n")
                    f.write(f"  F1分数: {result['f1']:.4f}\n")
                    f.write(f"  精确率: {result['precision']:.4f}\n")
                    f.write(f"  召回率: {result['recall']:.4f}\n")
                    f.write(f"  AUC: {result['auc']:.4f}\n\n")
                
                f.write(f"最佳模型: {best_model_name}\n")
                f.write(f"最佳AUC: {best_auc:.4f}\n")
            
            return {
                'best_model': best_model_name,
                'best_auc': best_auc,
                'results': results,
                'summary': results_df
            }
        
        else:
            logger.error("没有成功训练的模型")
            return None


class ModelContainer:
    """
    模型容器类，用于管理所有模型相关的内容
    包括模型、配置、训练历史、数据索引等
    """
    
    def __init__(self):
        """
        初始化模型容器
        """
        # 模型字典
        self.models = {}
        
        # 配置信息
        self.config = {}
        
        # 训练历史
        self.train_results = {}
        
        # 数据索引（train/test/val）
        self.indices = {
            'train': None,
            'test': None,
            'val': None
        }
        
        # 其他元数据
        self.metadata = {}
    
    def add_model(self, name: str, model):
        """
        添加模型
        
        Parameters:
        -----------
        name : str
            模型名称
        model : Any
            模型对象
        """
        self.models[name] = model
    
    def get_model(self, name: str):
        """
        获取模型
        
        Parameters:
        -----------
        name : str
            模型名称
            
        Returns:
        --------
        Any
            模型对象
        """
        return self.models.get(name)
    
    def has_model(self, name: str) -> bool:
        """
        检查是否有指定模型
        
        Parameters:
        -----------
        name : str
            模型名称
            
        Returns:
        --------
        bool
            是否存在
        """
        return name in self.models
    
    def list_models(self) -> list:
        """
        列出所有模型名称
        
        Returns:
        --------
        list
            模型名称列表
        """
        return list(self.models.keys())
    
    def set_config(self, config: dict):
        """
        设置配置
        
        Parameters:
        -----------
        config : dict
            配置字典
        """
        self.config = config
    
    def get_config(self) -> dict:
        """
        获取配置
        
        Returns:
        --------
        dict
            配置字典
        """
        return self.config
    
    def set_train_results(self, results: dict):
        """
        设置训练结果
        
        Parameters:
        -----------
        results : dict
            训练结果字典
        """
        self.train_results = results
    
    def get_train_results(self) -> dict:
        """
        获取训练结果
        
        Returns:
        --------
        dict
            训练结果字典
        """
        return self.train_results
    
    def set_indices(self, train_idx=None, test_idx=None, val_idx=None):
        """
        设置数据索引到nadata.Var中
        
        Parameters:
        -----------
        train_idx : Optional[list]
            训练集索引
        test_idx : Optional[list]
            测试集索引
        val_idx : Optional[list]
            验证集索引
        """
        # 创建索引DataFrame
        indices_data = {}
        if train_idx is not None:
            indices_data['train'] = train_idx
        if test_idx is not None:
            indices_data['test'] = test_idx
        if val_idx is not None:
            indices_data['val'] = val_idx
        
        # 将索引信息存储到nadata.Var中
        if hasattr(self, '_nadata') and self._nadata is not None:
            try:
                import pandas as pd
                if self._nadata.Var is None:
                    # 如果Var不存在，创建一个空的DataFrame
                    self._nadata.Var = pd.DataFrame()
                
                # 添加indices列，将字典存储为单个值
                if len(self._nadata.Var) == 0:
                    # 如果DataFrame为空，先添加一行
                    self._nadata.Var = pd.DataFrame([{'indices': indices_data}])
                else:
                    # 如果DataFrame不为空，在索引0处设置值
                    # 确保indices列存在
                    if 'indices' not in self._nadata.Var.columns:
                        self._nadata.Var['indices'] = None
                    
                    # 检查DataFrame是否有索引0，如果没有则添加一行
                    if 0 not in self._nadata.Var.index:
                        # 添加一行到DataFrame
                        new_row = pd.DataFrame([{'indices': indices_data}], index=[0])
                        self._nadata.Var = pd.concat([self._nadata.Var, new_row])
                    else:
                        # 使用at方法设置单个值，避免Series对齐问题
                        self._nadata.Var.at[0, 'indices'] = indices_data
            except ImportError:
                # 如果没有pandas，使用简单的字典存储
                if not hasattr(self._nadata, '_indices'):
                    self._nadata._indices = {}
                self._nadata._indices = indices_data
        
        # 同时保持原有的Model容器索引存储（向后兼容）
        if train_idx is not None:
            self.indices['train'] = train_idx
        if test_idx is not None:
            self.indices['test'] = test_idx
        if val_idx is not None:
            self.indices['val'] = val_idx
    
    def get_indices(self, split: str = None):
        """
        获取数据索引
        
        Parameters:
        -----------
        split : Optional[str]
            分割类型（'train', 'test', 'val'），如果为None则返回所有索引
            
        Returns:
        --------
        Union[list, dict]
            索引列表或字典
        """
        # 优先从nadata.Var中获取索引信息
        if hasattr(self, '_nadata') and self._nadata is not None:
            try:
                import pandas as pd
                if self._nadata.Var is not None and 'indices' in self._nadata.Var.columns and len(self._nadata.Var) > 0:
                    indices_data = self._nadata.Var.loc[0, 'indices']
                    if split is None:
                        return indices_data
                    return indices_data.get(split) if isinstance(indices_data, dict) else None
            except ImportError:
                # 如果没有pandas，从_indices属性获取
                if hasattr(self._nadata, '_indices') and self._nadata._indices:
                    if split is None:
                        return self._nadata._indices
                    return self._nadata._indices.get(split)
        
        # 如果Var中没有索引信息，则从Model容器中获取（向后兼容）
        if split is None:
            return self.indices
        return self.indices.get(split)
    
    def get_var_indices(self, split: str = None):
        """
        从nadata.Var中获取数据索引
        
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
                # 如果没有pandas，从_indices属性获取
                if hasattr(self._nadata, '_indices') and self._nadata._indices:
                    if split is None:
                        return self._nadata._indices
                    return self._nadata._indices.get(split)
        return None
    
    def add_metadata(self, key: str, value):
        """
        添加元数据
        
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
        获取元数据
        
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
        合并另一个模型容器
        
        Parameters:
        -----------
        other : ModelContainer
            要合并的模型容器
        """
        # 合并模型
        self.models.update(other.models)
        
        # 合并配置（以other为准）
        if other.config:
            self.config = other.config
        
        # 合并训练结果
        if other.train_results:
            self.train_results.update(other.train_results)
        
        # 合并索引
        for key in ['train', 'test', 'val']:
            if other.indices[key] is not None:
                self.indices[key] = other.indices[key]
        
        # 合并元数据
        self.metadata.update(other.metadata)
    
    def __str__(self):
        """
        字符串表示
        """
        return f"ModelContainer(models={list(self.models.keys())}, config_keys={list(self.config.keys())}, train_results_keys={list(self.train_results.keys())})"
    
    def __repr__(self):
        """
        详细字符串表示
        """
        return self.__str__()