import nnea as na
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
import threading
import time
import logging
import os
import sys
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
warnings.filterwarnings('ignore')

# 设置日志记录
def setup_logging(log_dir="experiment/tumor_imm/ccRCC_imm/logs"):
    """设置日志记录系统"""
    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"✅ 创建日志目录: {log_dir}")
    
    # 创建日志文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.log")
    
    # 清除之前的日志配置
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8', mode='w'),
            logging.StreamHandler()
        ],
        force=True
    )
    
    logger = logging.getLogger(__name__)
    
    # 记录日志文件路径
    print(f"📝 日志文件保存路径: {os.path.abspath(log_file)}")
    logger.info(f"日志文件保存路径: {os.path.abspath(log_file)}")
    
    return logger

# 线程安全的日志记录器
class ThreadSafeLogger:
    def __init__(self, logger):
        self.logger = logger
        self.lock = threading.Lock()
    
    def info(self, message):
        with self.lock:
            self.logger.info(message)
    
    def warning(self, message):
        with self.lock:
            self.logger.warning(message)
    
    def error(self, message):
        with self.lock:
            self.logger.error(message)
    
    def debug(self, message):
        with self.lock:
            self.logger.debug(message)

# 训练单个模型的函数
def train_single_model(name, classifier, param_grid, X_train, y_train, X_test, y_test, logger, progress_queue):
    """训练单个模型的函数"""
    try:
        logger.info(f"开始训练 {name}")
        logger.info(f"训练数据形状: X_train={X_train.shape}, y_train={y_train.shape}")
        logger.info(f"测试数据形状: X_test={X_test.shape}, y_test={y_test.shape}")
        logger.info(f"参数网格大小: {len([k for k in param_grid.keys()])} 个参数")
        
        # 网格搜索交叉验证
        grid = GridSearchCV(
            classifier,
            param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=1,  # 单个线程内不使用多进程
            verbose=0
        )
        
        logger.info(f"开始网格搜索，交叉验证折数: 5")
        # 训练模型
        grid.fit(X_train, y_train)
        
        logger.info(f"✅ {name} 训练完成")
        logger.info(f"最优参数：{grid.best_params_}")
        logger.info(f"最佳交叉验证AUC得分：{grid.best_score_:.4f}")
        
        # 在测试集上评估
        logger.info(f"开始在测试集上评估 {name}")
        y_pred = grid.predict(X_test)
        
        # 获取预测概率
        if hasattr(grid.best_estimator_, "predict_proba"):
            y_proba = grid.predict_proba(X_test)[:, 1]
            logger.info(f"使用 predict_proba 方法获取预测概率")
        elif hasattr(grid.best_estimator_, "decision_function"):
            # 对于SVM，使用decision_function
            y_proba = grid.decision_function(X_test)
            # 确保概率值在[0,1]范围内
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
            logger.info(f"使用 decision_function 方法获取预测概率")
        else:
            # 如果没有概率预测方法，使用全0
            y_proba = np.zeros_like(y_pred, dtype=float)
            logger.info(f"模型不支持概率预测，使用零概率")
        
        # 计算AUC
        test_auc = roc_auc_score(y_test, y_proba)
        
        logger.info(f"测试集AUC：{test_auc:.4f}")
        logger.info(f"预测准确率：{(y_pred == y_test).mean():.4f}")
        
        # 构建结果字典
        result = {
            "best_params": grid.best_params_,
            "best_cv_auc": grid.best_score_,
            "test_auc": test_auc,
            "test_report": classification_report(y_test, y_pred, output_dict=True),
            "test_pred": y_pred,
            "test_proba": y_proba,
            "test_true": y_test.values,
            "best_model": grid.best_estimator_
        }
        
        # 通知进度更新
        progress_queue.put(("success", name, result))
        logger.info(f"✅ {name} 结果已保存")
        
        return name, result
        
    except Exception as e:
        error_msg = f"❌ {name} 训练失败: {str(e)}"
        logger.error(error_msg)
        progress_queue.put(("error", name, str(e)))
        return name, None

# 主函数
def main():
    # 设置日志记录
    logger = setup_logging()
    thread_logger = ThreadSafeLogger(logger)
    
    logger.info("🚀 开始多线程模型训练实验")
    logger.info("=" * 60)
    logger.info(f"实验开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"工作目录: {os.getcwd()}")
    logger.info("=" * 60)
    
    # 加载预处理后保存的ccRCC免疫治疗数据
    nadata = na.nadata()
    nadata.load(filepath="experiment/tumor_imm/ccRCC_imm_exp.pkl")
    logger.info(f"✅ 预处理后的nadata对象加载完成，数据形状: {nadata.X.shape}")
    
    # 获取特征和标签
    X = nadata.X
    
    # 使用na.pp.fillna处理缺失值
    if np.isnan(X).any():
        logger.warning("⚠️ 检测到X中存在NaN值，正在进行填充处理...")
        X = na.pp.fillna(X, method="mean")
    else:
        logger.info("✅ X中未检测到NaN值")
    
    # 使用na.pp.scale进行标准化处理
    X = na.pp.scale(X, method="standard")
    
    y = nadata.Meta['response_NR']
    # 将y中的'N'映射为0，'R'映射为1
    y = y.map({'N': 0, 'R': 1})
    
    # 使用na.pp.x_train_test和na.pp.y_train_test获取训练测试集
    X_train, X_test = na.pp.x_train_test(X, nadata)
    y_train, y_test = na.pp.y_train_test(y, nadata)
    
    logger.info(f"训练集特征形状: {X_train.shape}")
    logger.info(f"测试集特征形状: {X_test.shape}")
    logger.info(f"训练集标签形状: {y_train.shape}")
    logger.info(f"测试集标签形状: {y_test.shape}")
    
    # 定义所有分类器的超参数搜索空间
    param_grids = {
        'LogisticRegression': {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000]
        },
        'DecisionTreeClassifier': {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [None, 3, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2']
        },
        'RandomForestClassifier': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        },
        'AdaBoostClassifier': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.5, 1.0],
            'algorithm': ['SAMME', 'SAMME.R']
        },
        'MLPClassifier': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [1000]
        },
        'LinearSVM': {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'loss': ['hinge', 'squared_hinge'],
            'max_iter': [1000]
        },
        'RBFSVM': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf']
        }
    }
    
    # 定义所有分类器
    classifiers = {
        'LogisticRegression': LogisticRegression(random_state=42),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
        'RandomForestClassifier': RandomForestClassifier(random_state=42),
        'AdaBoostClassifier': AdaBoostClassifier(random_state=42),
        'MLPClassifier': MLPClassifier(random_state=42, max_iter=1000),
        'LinearSVM': LinearSVC(random_state=42, max_iter=1000),
        'RBFSVM': SVC(random_state=42, probability=True)
    }
    
    # 初始化结果存储
    if not hasattr(nadata, "Model"):
        nadata.Model = {}
    
    # 创建进度队列
    progress_queue = queue.Queue()
    
    # 创建进度条
    total_models = len(classifiers)
    pbar = tqdm(total=total_models, desc="训练进度", unit="模型")
    
    # 存储所有任务的结果
    results = {}
    
    # 使用线程池进行多线程训练
    logger.info(f"开始多线程训练，线程池大小: 4")
    logger.info(f"待训练模型数量: {len(classifiers)}")
    logger.info("模型列表: " + ", ".join(classifiers.keys()))
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        # 提交所有训练任务
        future_to_name = {}
        for name, classifier in classifiers.items():
            param_grid = param_grids[name]
            logger.info(f"提交训练任务: {name}")
            future = executor.submit(
                train_single_model,
                name, classifier, param_grid,
                X_train, y_train, X_test, y_test,
                thread_logger, progress_queue
            )
            future_to_name[future] = name
        
        # 监控进度
        completed = 0
        while completed < total_models:
            try:
                # 检查进度队列
                status, name, result = progress_queue.get(timeout=1)
                if status == "success":
                    results[name] = result
                    nadata.Model[name] = result
                    pbar.update(1)
                    completed += 1
                    pbar.set_postfix({"当前": name, "完成": f"{completed}/{total_models}"})
                elif status == "error":
                    pbar.update(1)
                    completed += 1
                    pbar.set_postfix({"当前": name, "完成": f"{completed}/{total_models}", "状态": "失败"})
            except queue.Empty:
                # 检查是否有任务完成
                for future in as_completed(future_to_name):
                    name = future_to_name[future]
                    if name not in results:
                        # 如果队列中没有结果，说明任务失败
                        pbar.update(1)
                        completed += 1
                        pbar.set_postfix({"当前": name, "完成": f"{completed}/{total_models}", "状态": "失败"})
                continue
    
    pbar.close()
    
    # 保存nadata对象到文件
    save_path = "experiment/tumor_imm/ccRCC_imm_exp.pkl"
    logger.info(f"保存实验结果到: {os.path.abspath(save_path)}")
    nadata.save(save_path, format="pickle", save_data=True)
    logger.info("=" * 60)
    logger.info("所有分类器训练完成，结果已保存到nadata对象中")
    logger.info(f"实验结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # 打印所有模型的性能比较
    logger.info("\n📊 所有模型性能比较:")
    logger.info("-" * 80)
    logger.info(f"{'模型名称':<20} {'CV AUC':<10} {'测试AUC':<10}")
    logger.info("-" * 80)
    
    for name in classifiers.keys():
        if name in nadata.Model:
            cv_auc = nadata.Model[name]["best_cv_auc"]
            test_auc = nadata.Model[name]["test_auc"]
            logger.info(f"{name:<20} {cv_auc:<10.4f} {test_auc:<10.4f}")
    
    logger.info("-" * 80)
    logger.info("✅ 所有分类器训练和评估完成！")
    
    return nadata

if __name__ == "__main__":
    main() 