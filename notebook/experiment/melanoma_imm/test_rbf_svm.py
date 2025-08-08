#!/usr/bin/env python3
"""
RBF SVM实验测试脚本
用于验证代码是否能正常运行
"""

import os
import sys
import toml

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_config_loading():
    """测试配置文件加载"""
    print("🔧 测试配置文件加载...")
    try:
        config = toml.load("melanoma_imm_RBFSVM_config.toml")
        print("✅ 配置文件加载成功")
        print(f"   模型类型: {config['global']['model']}")
        print(f"   输入文件: {config['global']['inputfl']}")
        print(f"   输出目录: {config['global']['outdir']}")
        return True
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return False

def test_data_file():
    """测试数据文件是否存在"""
    print("\n📂 测试数据文件...")
    config = toml.load("melanoma_imm_RBFSVM_config.toml")
    data_file = config['global']['inputfl']
    
    if os.path.exists(data_file):
        print(f"✅ 数据文件存在: {data_file}")
        return True
    else:
        print(f"❌ 数据文件不存在: {data_file}")
        return False

def test_imports():
    """测试必要的导入"""
    print("\n📦 测试模块导入...")
    try:
        import nnea as na
        print("✅ nnea模块导入成功")
        
        from sklearn.svm import SVC
        from sklearn.metrics import roc_auc_score, classification_report
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        print("✅ sklearn模块导入成功")
        
        import numpy as np
        import torch
        print("✅ numpy和torch模块导入成功")
        
        return True
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_output_directory():
    """测试输出目录创建"""
    print("\n📁 测试输出目录...")
    config = toml.load("melanoma_imm_RBFSVM_config.toml")
    output_dir = config['global']['outdir']
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"✅ 输出目录创建成功: {output_dir}")
        return True
    except Exception as e:
        print(f"❌ 输出目录创建失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始RBF SVM实验测试...\n")
    
    tests = [
        test_config_loading,
        test_data_file,
        test_imports,
        test_output_directory
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！可以运行RBF SVM实验了。")
        print("\n💡 运行命令:")
        print("   python melanoma_imm_RBFSVM_experiment.py")
    else:
        print("⚠️ 部分测试失败，请检查配置和依赖。")

if __name__ == "__main__":
    main() 