import nnea as na
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
import numpy as np

# 加载预处理后保存的ccRCC免疫治疗数据
nadata = na.nadata()
nadata.load(filepath ="tumor_imm/ccRCC_imm_exp.pkl")
print("✅ 预处理后的nadata对象加载完成，数据形状:", nadata.X.shape)

# 构建LogisticRegression超参数模型训练
# 获取特征和标签
X = nadata.X

# 使用na.pp.fillna处理缺失值
if np.isnan(X).any():
    print("⚠️ 检测到X中存在NaN值，正在进行填充处理...")
    X = na.pp.fillna(X, method="mean")
else:
    print("✅ X中未检测到NaN值")

# 使用na.pp.scale进行标准化处理
X = na.pp.scale(X, method="standard")

y = nadata.Meta['response_NR']
# 将y中的'N'映射为0，'R'映射为1
y = y.map({'N': 0, 'R': 1})

# 使用na.pp.x_train_test和na.pp.y_train_test获取训练测试集
X_train, X_test = na.pp.x_train_test(X, nadata)
y_train, y_test = na.pp.y_train_test(y, nadata)

print(f"训练集特征形状: {X_train.shape}")
print(f"测试集特征形状: {X_test.shape}")
print(f"训练集标签形状: {y_train.shape}")
print(f"测试集标签形状: {y_test.shape}")

# 定义常用的完整LogisticRegression超参数搜索空间
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200, 500]
}

# 构建LogisticRegression模型
logreg = LogisticRegression()

# 网格搜索交叉验证
grid = GridSearchCV(
    logreg,
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=-16,
    verbose=1
)

grid.fit(X_train, y_train)

print("最优参数：", grid.best_params_)
print("最佳AUC得分：", grid.best_score_)

# 在测试集上评估
y_pred = grid.predict(X_test)
y_proba = grid.predict_proba(X_test)[:, 1]
print("测试集分类报告：\n", classification_report(y_test, y_pred))
print("测试集AUC：", roc_auc_score(y_test, y_proba))

# 将最佳模型和相关结果保存到nadata对象中，并进行保存，创建LogisticRegression字典对象

# 构建LogisticRegression结果字典
logreg_result = {
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

nadata.Model["LogisticRegression"] = logreg_result

# 保存nadata对象到文件
nadata.save("experiment/tumor_imm/ccRCC_imm_exp.pkl", format="pickle", save_data=True)
print("已完成logistic regression模型训练，并保存到nadata对象中")

# 重新加载nadata对象
nadata_reloaded = na.nadata()
nadata_reloaded.load(filepath="tumor_imm/ccRCC_imm_exp.pkl")

# 获取保存的LogisticRegression结果
logreg_result_reloaded = nadata_reloaded.Model.get("LogisticRegression", None)
if logreg_result_reloaded is None:
    raise ValueError("未在nadata对象中找到LogisticRegression结果，请先训练并保存模型。")

# 重新加载最佳LogisticRegression模型
logreg_reload = logreg_result_reloaded["best_model"]

# 重新获取测试集数据
# 假设nadata_reloaded中X和Meta与原始一致
X = nadata_reloaded.X

# 使用na.pp.fillna处理缺失值
if np.isnan(X).any():
    print("⚠️ 检测到X中存在NaN值，正在进行填充处理...")
    X = na.pp.fillna(X, method="mean")
else:
    print("✅ X中未检测到NaN值")

# 使用na.pp.scale进行标准化处理
X = na.pp.scale(X, method="standard")

y = nadata_reloaded.Meta['response_NR']
y = y.map({'N': 0, 'R': 1})

# 使用na.pp.x_train_test和na.pp.y_train_test获取训练测试集
X_train, X_test = na.pp.x_train_test(X, nadata_reloaded)
y_train, y_test = na.pp.y_train_test(y, nadata_reloaded)

# 在测试集上验证
y_pred_reload = logreg_reload.predict(X_test)
y_proba_reload = logreg_reload.predict_proba(X_test)[:, 1]

print("【重加载模型】测试集分类报告：\n", classification_report(y_test, y_pred_reload))
print("【重加载模型】测试集AUC：", roc_auc_score(y_test, y_proba_reload))

