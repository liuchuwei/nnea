from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import nnea as na
import numpy as np
import torch
import os
import warnings
import toml  # 用于读取toml文件

warnings.filterwarnings('ignore')

# 读取AdaBoostClassifier配置文件
try:
    config = toml.load("./config/experiment/ccRCC_imm_AdaBoost.toml")
except Exception as e:
    print(f"❌ 配置文件读取失败: {e}")
    exit(1)

# 创建输出目录
output_dir = config['global']['outdir']
os.makedirs(output_dir, exist_ok=True)

# 设置日志输出到输出目录
log_file = os.path.join(output_dir, "adaboost_experiment.log")
na.setup_logging(log_file=log_file, experiment_name="adaboost")
logger = na.get_logger(__name__)

logger.info("⚙️ 读取AdaBoostClassifier配置文件...")
logger.info("✅ 配置文件读取成功")
logger.info(f"📁 创建输出目录: {output_dir}")
logger.info(f"📝 日志文件已设置到: {log_file}")

# 设置全局随机种子（在数据加载之前）
logger.info("🔧 设置全局随机种子...")
na.set_global_seed(config['global']['seed'])
logger.info("✅ 全局随机种子设置完成")

# 数据加载
logger.info("📂 加载数据...")
try:
    nadata = na.nadata()
    nadata.load(filepath="experiment/tumor_imm/ccRCC_imm_exp.pkl")
    logger.info(f"✅ 预处理后的nadata对象加载完成，数据形状: {nadata.X.shape}")
except Exception as e:
    logger.error(f"❌ 数据加载失败: {e}")
    exit(1)

# 数据预处理
logger.info("🔧 数据预处理...")
X = nadata.X

# 使用配置中的预处理设置
preprocessing_config = config['adaboost']['preprocessing']

# 使用na.pp.fillna处理缺失值
if preprocessing_config['fill_na'] and np.isnan(X).any():
    logger.warning("⚠️ 检测到X中存在NaN值，正在进行填充处理...")
    X = na.pp.fillna(X, method=preprocessing_config['fill_method'])
    logger.info(f"   填充后NaN值数量: {np.isnan(X).sum()}")
else:
    logger.info("✅ X中未检测到NaN值")

# 使用na.pp.scale进行标准化处理
if preprocessing_config['scale_data']:
    X = na.pp.scale(X, method=preprocessing_config['scale_method'])
    logger.info("✅ 数据标准化完成")

nadata.X = X

# 处理标签
logger.info("🏷️ 处理标签...")
y = nadata.Meta['response_NR']
y = y.map({'N': 0, 'R': 1})
nadata.Meta['target'] = y  # 模型默认使用target

# 特征选择
if config['adaboost']['feature_selection']:
    logger.info("🔍 特征选择...")
    nadata = na.fs.apply_feature_selection(
        nadata,
        method=config['adaboost']['selection_method'],
        n_features=config['adaboost']['n_features'],
        target_col='target',  # 使用默认的target列
        alpha=config['adaboost']['selection_alpha']
    )
    logger.info(f"✅ 特征选择完成，选择特征数: {config['adaboost']['n_features']}")

# 数据分割
logger.info("✂️ 进行数据分割...")
try:
    nadata = na.pp.split_data(
        nadata,
        test_size=config['dataset']['test_size'],
        random_state=config['dataset']['random_state'],
        strategy="stratified"
    )
    logger.info("✅ 数据分割完成")
except Exception as e:
    logger.error(f"❌ 数据分割失败: {e}")

# 使用na.pp.x_train_test和na.pp.y_train_test获取训练测试集
X_train, X_test = na.pp.x_train_test(X, nadata)
y_train, y_test = na.pp.y_train_test(y, nadata)

logger.info(f"训练集特征形状: {X_train.shape}")
logger.info(f"测试集特征形状: {X_test.shape}")
logger.info(f"训练集标签形状: {y_train.shape}")
logger.info(f"测试集标签形状: {y_test.shape}")

# 从配置文件构建参数网格
param_grid = {
    'n_estimators': config['adaboost']['n_estimators'],
    'learning_rate': config['adaboost']['learning_rate'],
    'algorithm': config['adaboost']['algorithm']
}

# 构建AdaBoostClassifier模型
adaboost = AdaBoostClassifier(
    random_state=config['adaboost']['random_state']
)

# 网格搜索交叉验证
grid = GridSearchCV(
    adaboost,
    param_grid,
    cv=StratifiedKFold(
        n_splits=config['adaboost']['cv_folds'], 
        shuffle=True, 
        random_state=config['adaboost']['random_state']
    ),
    scoring=config['adaboost']['cv_scoring'],
    n_jobs=config['adaboost']['n_jobs'],
    verbose=config['training']['verbose']
)

logger.info("🚀 开始网格搜索训练...")
grid.fit(X_train, y_train)

logger.info(f"最优参数: {grid.best_params_}")
logger.info(f"最佳AUC得分: {grid.best_score_}")

# 在测试集上评估
y_pred = grid.predict(X_test)
y_proba = grid.predict_proba(X_test)[:, 1]
logger.info(f"测试集分类报告:\n{classification_report(y_test, y_pred)}")
logger.info(f"测试集AUC: {roc_auc_score(y_test, y_proba)}")

# 构建AdaBoostClassifier结果字典
adaboost_result = {
    "best_params": grid.best_params_,
    "best_cv_auc": grid.best_score_,
    "test_auc": roc_auc_score(y_test, y_proba),
    "test_report": classification_report(y_test, y_pred, output_dict=True),
    "test_pred": y_pred,
    "test_proba": y_proba,
    "test_true": y_test.values,
    "best_model": grid.best_estimator_  # 保存最佳模型
}

# 保存到nadata对象
if not hasattr(nadata, "Model"):
    nadata.Model = {}

nadata.Model["AdaBoostClassifier"] = adaboost_result

# 保存nadata对象到配置的输出目录
output_file = os.path.join(output_dir, "ccRCC_imm_AdaBoost.pkl")
nadata.save(output_file, format=config['training']['save_format'], save_data=config['training']['save_data'])
logger.info(f"✅ 已完成adaboost模型训练，并保存到: {output_file}")

# 保存配置信息
config_file = os.path.join(output_dir, "adaboost_config.toml")
with open(config_file, 'w', encoding='utf-8') as f:
    toml.dump(config, f)
logger.info(f"✅ 配置文件已保存到: {config_file}")

# 保存训练结果摘要
summary_file = os.path.join(output_dir, "training_summary.txt")
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("AdaBoostClassifier 训练结果摘要\n")
    f.write("=" * 50 + "\n")
    f.write(f"最优参数: {grid.best_params_}\n")
    f.write(f"最佳交叉验证AUC: {grid.best_score_:.4f}\n")
    f.write(f"测试集AUC: {roc_auc_score(y_test, y_proba):.4f}\n")
    f.write(f"训练集形状: {X_train.shape}\n")
    f.write(f"测试集形状: {X_test.shape}\n")
    f.write("\n分类报告:\n")
    f.write(classification_report(y_test, y_pred))

logger.info(f"✅ 训练结果摘要已保存到: {summary_file}")
logger.info("🎉 实验完成！") 