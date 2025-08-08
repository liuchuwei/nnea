from sklearn.metrics import roc_auc_score, classification_report

import nnea as na
import numpy as np
import torch
import os
import warnings
import toml  # 用于读取toml文件

warnings.filterwarnings('ignore')

print("🚀 开始nnea模型实验...")

# 读取nnea配置文件
print("⚙️ 读取nnea配置文件...")
try:
    nnea_config = toml.load("./config/nnea_classification_config.toml")
    print("✅ 配置文件读取成功")
except Exception as e:
    print(f"❌ 配置文件读取失败: {e}")
    exit(1)

# 设置全局随机种子（在数据加载之前）
print("🔧 设置全局随机种子...")
na.set_global_seed(nnea_config['global']['seed'])
print("✅ 全局随机种子设置完成")

# 数据加载
print("📂 加载数据...")
try:
    nadata = na.nadata()
    nadata.load(filepath="./datasets/tumor_imm/melanoma_immunotherapy.pkl")
    print("✅ 预处理后的nadata对象加载完成，数据形状:", nadata.X.shape)
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    exit(1)

# 数据预处理
print("🔧 数据预处理...")
X = nadata.X

# 使用na.pp.fillna处理缺失值
if np.isnan(X).any():
    print("⚠️ 检测到X中存在NaN值，正在进行填充处理...")
    X = na.pp.fillna(X, method="mean")
    print(f"   填充后NaN值数量: {np.isnan(X).sum()}")
else:
    print("✅ X中未检测到NaN值")

# 更新nadata中的X
nadata.X = X

# 处理标签
print("🏷️ 处理标签...")
y = nadata.Meta['response_NR']
y = y.map({'N': 0, 'R': 1})
nadata.Meta['target'] = y  # 模型默认使用target

# 数据分割
print("✂️ 进行数据分割...")
try:
    nadata = na.pp.split_data(
        nadata,
        test_size=0.2,
        random_state=42,
        strategy="stratified"
    )
    print("✅ 数据分割完成")
except Exception as e:
    print(f"❌ 数据分割失败: {e}")

# 处理设备配置
if nnea_config['global']['device'] == 'auto':
    nnea_config['global']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"🖥️ 使用设备: {nnea_config['global']['device']}")

# 设置nadata的配置（会自动创建输出目录和配置日志）
try:
    nadata.Model.set_config(nnea_config)
    print("✅ 模型配置设置完成")
    print(f"📁 输出目录: {nnea_config['global']['outdir']}")
except Exception as e:
    print(f"❌ 模型配置设置失败: {e}")
    exit(1)

# 构建模型
print("🔧 构建nnea模型...")
try:
    na.build(nadata)
    print("✅ 模型构建完成")
except Exception as e:
    print(f"❌ 模型构建失败: {e}")
    exit(1)

# 训练模型
print("🚀 开始训练nnea模型...")
try:
    # 检查是否启用tailor策略
    training_config = nnea_config.get('training', {})
    tailor_enabled = training_config.get('tailor', False)
    
    if tailor_enabled:
        print(f"✂️ 启用tailor策略: tailor_epoch={training_config.get('tailor_epoch', 20)}, tailor_geneset={training_config.get('tailor_geneset', 2)}")
    
    train_results = na.train(nadata, verbose=2)
    print("✅ 模型训练完成")
    print(f"📊 训练结果: {train_results}")
    
    # 如果使用了tailor策略，显示裁剪信息
    if tailor_enabled and isinstance(train_results, dict) and 'tailor_info' in train_results:
        tailor_info = train_results['tailor_info']
        print(f"✂️ 循环Tailor策略信息:")
        print(f"   - 裁剪epoch间隔: {tailor_info['tailor_epoch']}")
        print(f"   - 每次裁剪基因集数量: {tailor_info['tailor_geneset']}")
        print(f"   - 总训练阶段数: {tailor_info['total_stages']}")
        print(f"   - 最终基因集数量: {tailor_info['final_geneset_count']}")
        
        # 显示每个阶段的裁剪历史
        if 'tailor_history' in train_results:
            print(f"   - 裁剪历史:")
            for i, history in enumerate(train_results['tailor_history']):
                print(f"     阶段{i+1}: epoch {history['epoch']}, 移除基因集 {history['removed_genesets']}, 保留基因集数量 {history['num_genesets_after']}")
        
except Exception as e:
    print(f"❌ 模型训练失败: {e}")
    print(f"   错误类型: {type(e).__name__}")
    print(f"   错误详情: {str(e)}")

# 评估模型
print("📈 评估nnea模型...")
try:
    eval_results = na.eval(nadata, split='test')
    print("✅ 模型评估完成")
    print(f"📊 评估结果: {eval_results}")
except Exception as e:
    print(f"❌ 模型评估失败: {e}")
    eval_results = {}

print("✅ 模型训练和评估完成!")

# 获取模型预测结果
print("🔮 进行模型预测...")
try:
    # 使用nnea包内的predict函数
    from nnea import predict
    prediction_results = predict(nadata, split='test')
    
    # 检查预测结果
    if prediction_results.get('error'):
        print(f"❌ 预测失败: {prediction_results['error']}")
        y_test = None
        y_pred = None
        y_proba = None
    else:
        y_test = prediction_results['y_test']
        y_pred = prediction_results['y_pred']
        y_proba = prediction_results['y_proba']
        print("✅ 模型预测完成")
        
except Exception as e:
    print(f"❌ 模型预测失败: {e}")
    y_test = None
    y_pred = None
    y_proba = None

# 构建nnea结果字典
print("💾 保存实验结果...")
nnea_result = {
    "model_config": nnea_config,
    "train_results": train_results,
    "eval_results": eval_results,
    "test_auc": roc_auc_score(y_test, y_proba) if y_test is not None and y_proba is not None else None,
    "test_report": classification_report(y_test, y_pred,
                                         output_dict=True) if y_test is not None and y_pred is not None else None,
    "test_pred": y_pred,
    "test_proba": y_proba,
    "test_true": y_test
}

# 保存到nadata对象
if not hasattr(nadata, "Model"):
    nadata.Model = {}

nadata.Model["nnea_model"] = nnea_result

# 保存nadata对象到文件（使用配置中的输出目录）
try:
    save_path = os.path.join(nnea_config['global']['outdir'], "melanoma_imm.pkl")
    nadata.save(save_path, format="pickle", save_data=True)
    print(f"✅ 已完成nnea模型训练，并保存到: {save_path}")
except Exception as e:
    print(f"❌ 保存失败: {e}")

# 重新加载nadata对象
print("🔄 重新加载nadata对象...")
try:
    nadata_reloaded = na.nadata()
    load_path = os.path.join(nnea_config['global']['outdir'], "melanoma_imm.pkl")
    nadata_reloaded.load(filepath=load_path)
    print(f"✅ 数据重加载成功: {load_path}")
except Exception as e:
    print(f"❌ 数据重加载失败: {e}")
    exit(1)

# 获取保存的nnea结果
nnea_result_reloaded = nadata_reloaded.Model.get("nnea_model", None)
if nnea_result_reloaded is None:
    print("⚠️ 未在nadata对象中找到nnea模型结果")
else:
    print("📊 重加载的模型结果:")
    print(f"训练结果: {nnea_result_reloaded.get('train_results', {})}")
    print(f"评估结果: {nnea_result_reloaded.get('eval_results', {})}")

# 模型解释性分析
print("🔍 进行模型解释性分析...")
try:
    # 使用nnea的explain功能
    na.explain(nadata_reloaded, method='importance', model_name="nnea")
    print("✅ 特征重要性分析完成")

except Exception as e:
    print(f"⚠️ 模型解释性分析时出现错误: {e}")

# 获取模型摘要
print("📋 获取模型摘要...")
try:
    summary = na.get_summary(nadata_reloaded)
    print("📊 模型摘要:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
except Exception as e:
    print(f"⚠️ 获取模型摘要时出现错误: {e}")

print("🎉 nnea模型实验完成!")
print(f"📁 结果已保存到: {nnea_config['global']['outdir']}")
print(f"📊 日志文件保存在: {os.path.join(nnea_config['global']['outdir'], 'logs')}")

