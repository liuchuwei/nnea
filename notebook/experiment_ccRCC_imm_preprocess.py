import nnea as na
import numpy as np

# 加载ccRCC免疫治疗数据为nadata类
nadata = na.nadata()
nadata.load(filepath ="../datasets/tumor_imm/ccRCC_immunotherapy.pkl")
print("✅ nadata对象加载完成，数据形状:", nadata.X.shape)

# 假设nadata.X为特征，nadata.Meta为元信息（含标签）
n_samples = nadata.X.shape[0]
indices = np.arange(n_samples)
np.random.seed(42)  # 固定随机种子，保证可复现
np.random.shuffle(indices)

# 80% 训练集，20% 测试集
train_size = int(0.8 * n_samples)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

nadata.Model.set_indices(train_idx=train_idx, test_idx=test_idx)

# 保存为pickle格式
nadata.save(filepath ="tumor_imm/ccRCC_imm_exp.pkl", format='pickle', save_data=True)





