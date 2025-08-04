# NNEA (Neural Network with Explainable Architecture)

NNEA是一个专门为生物学研究设计的可解释性神经网络包，特别适用于转录组学数据分析。该包提供了从数据加载、预处理、模型训练到结果解释的完整流程。

## 功能特点

- 🧬 **生物学导向**: 专门为基因表达数据分析设计
- 🔍 **可解释性**: 提供模型解释和特征重要性分析
- 📊 **可视化**: 丰富的图表和结果展示功能
- 🚀 **易用性**: 简洁的API接口，快速上手
- 📈 **高性能**: 基于PyTorch的高效实现

## 依赖包

NNEA包需要以下Python依赖包：

### 核心依赖
- **torch** >= 1.9.0 - PyTorch深度学习框架
- **numpy** >= 1.21.0 - 数值计算库
- **pandas** >= 1.3.0 - 数据处理库
- **scikit-learn** >= 1.0.0 - 机器学习库

### 数据科学和可视化
- **matplotlib** >= 3.5.0 - 绘图库
- **seaborn** >= 0.11.0 - 统计可视化
- **plotly** >= 5.0.0 - 交互式图表
- **umap-learn** >= 0.5.0 - 降维算法

### 模型解释
- **shap** >= 0.40.0 - SHAP值计算
- **lime** >= 0.2.0 - LIME解释器

### 数据处理
- **scipy** >= 1.7.0 - 科学计算
- **h5py** >= 3.1.0 - HDF5文件支持
- **imbalanced-learn** >= 0.8.0 - 不平衡数据处理

### 其他工具
- **toml** >= 0.10.0 - 配置文件解析
- **networkx** >= 2.6.0 - 网络分析
- **pyyaml** >= 5.4.0 - YAML文件支持

## 安装方法

### 方法1: 从源码安装（推荐）

```bash
# 克隆仓库
git clone https://github.com/liuchuwei/nnea.git
cd nnea

# 安装依赖
pip install -r requirements.txt

# 安装nnea包
pip install -e .
```

### 方法2: 直接安装

```bash
# 使用pip安装
pip install nnea
```

### 方法3: 使用conda（如果可用）

```bash
conda install -c conda-forge nnea
```

## 快速开始

### 基本使用

```python
import nnea as na

# 创建NNEA项目
nadata = na.io.CreateNNEA(config="config/your_config.toml")

# 构建模型
na.build(nadata)

# 训练模型
na.train(nadata, verbose=2)

# 评估模型
na.eval(nadata, split='test')

# 模型解释
na.explain(nadata, method='importance')

# 保存模型
na.save_model(nadata, "results/model.pt")
```

### 免疫治疗响应预测示例

```python
import nnea as na
import numpy as np

# 设置随机种子
np.random.seed(42)

# 创建项目
nadata = na.io.CreateNNEA(config="config/ccRCC_imm.toml")

# 完整训练流程
na.build(nadata)
na.train(nadata, verbose=2)
na.eval(nadata, split='test')
na.explain(nadata, method='importance')

# 保存结果
na.save_model(nadata, "results/nnea_ccrcc_model.pt")
na.get_summary(nadata)
```

## 主要功能模块

### 1. 数据加载 (`nnea.io`)
- `CreateNNEA()`: 创建NNEA项目
- `nadata`: 数据容器对象

### 2. 模型构建 (`nnea.model`)
- `build()`: 构建神经网络模型
- `train()`: 训练模型
- `eval()`: 评估模型性能
- `explain()`: 模型解释

### 3. 数据预处理 (`nnea.data_factory`)
- `pp`: 数据预处理工具

### 4. 可视化 (`nnea.plot`)
- 训练曲线
- 特征重要性
- 基因集网络

### 5. 工具函数 (`nnea.utils`)
- 辅助函数
- 评估指标

## 配置说明

NNEA使用TOML格式的配置文件，主要配置项包括：

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

## 支持的生物学应用

- 🧬 **基因表达分析**: 转录组数据处理
- 🎯 **免疫治疗预测**: 患者响应预测
- 📊 **生存分析**: 患者预后预测
- 🔬 **药物敏感性**: 药物响应预测
- 🧪 **单细胞分析**: 单细胞转录组数据

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 联系方式

- 作者: Chuwei Liu
- 邮箱: liuchw26@mail.sysu.edu.cn
- 项目链接: https://github.com/liuchuwei/nnea

## 更新日志

### v0.1.2
- 添加了新的模型解释功能
- 改进了数据预处理流程
- 优化了可视化功能

### v0.1.0
- 初始版本发布
- 基础功能实现