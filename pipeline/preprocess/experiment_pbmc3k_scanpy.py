import os
# 设置环境变量禁用 numba JIT 编译，避免兼容性问题
# os.environ['NUMBA_DISABLE_JIT'] = '1'

# 设置matplotlib后端，确保在Windows环境下正确工作
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合保存图片

# 尝试导入必要的库
try:
    import pandas as pd
    import numpy as np
    import scanpy as sc
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    print("所有库导入成功，matplotlib后端已设置为'Agg'，适用于图片保存")
except ImportError as e:
    print(f"导入库时出错: {e}")
    print("请检查是否安装了所有必要的包")
    exit(1)


# 设置Scanpy参数
sc.settings.verbosity = 2  # 详细输出
# 设置matplotlib参数，确保图片正确保存
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

# 创建结果目录
results_file = '../datasets/sc_pbmc3k/pbmc3k_processed.h5ad'
Path('../../datasets/sc_pbmc3k').mkdir(parents=True, exist_ok=True)

# 检查数据目录是否存在
data_path = "../../data/single_cell/pbmc3k"
if not os.path.exists(data_path):
    print(f"错误：数据目录 {data_path} 不存在")
    exit(1)

print("=== 单细胞测序数据分析流程 ===")
print("基于Scanpy官方教程的完整分析流程")

# 1. 数据读取
print("\n1. 读取数据...")
try:
    adata = sc.read_10x_mtx(
        data_path,  # 包含.mtx文件的目录
        var_names="gene_symbols",  # 使用基因符号作为变量名
        cache=False,  # 禁用缓存以避免缓存文件损坏问题
    )
    print(f"成功读取数据，形状: {adata.shape}")
    print(f"基因数量: {adata.n_vars}")
    print(f"细胞数量: {adata.n_obs}")
    
except Exception as e:
    print(f"读取数据时出错: {e}")
    print("请检查数据目录是否正确，或者尝试重新下载数据")
    exit(1)

sc.pp.filter_cells(adata, min_genes=200)  # this does nothing, in this specific case
sc.pp.filter_genes(adata, min_cells=3)

# 2. 质量控制和线粒体基因注释
print("\n2. 质量控制和线粒体基因注释...")
# 注释线粒体基因
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
)


print(f"线粒体基因数量: {adata.var['mt'].sum()}")

# 4. 过滤低质量细胞
print("\n4. 过滤低质量细胞...")
print(f"过滤前的数据形状: {adata.shape}")
adata = adata[
    (adata.obs.n_genes_by_counts < 2500)
    & (adata.obs.n_genes_by_counts > 200)
    & (adata.obs.pct_counts_mt < 5),
    :,
].copy()
adata.layers["counts"] = adata.X.copy()
print(f"过滤后的数据形状: {adata.shape}")

# 5. 数据标准化
print("\n5. 数据标准化...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# 6. 高变异基因识别
print("\n6. 识别高变异基因...")
sc.pp.highly_variable_genes(
    adata,
    layer="counts",
    n_top_genes=2000,
    min_mean=0.0125,
    max_mean=3,
    min_disp=0.5,
    flavor="seurat_v3",
)
print(f"高变异基因数量: {sum(adata.var.highly_variable)}")

# 绘制高变异基因
print("绘制高变异基因图...")
fig, ax = plt.subplots(figsize=(10, 6))
sc.pl.highly_variable_genes(adata)
plt.tight_layout()
plt.savefig('./datasets/sc_pbmc3k/highly_variable_genes.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("高变异基因图保存完成")

# 7. 数据缩放
print("\n7. 数据缩放...")
adata.layers["scaled"] = adata.X.toarray()
sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"], layer="scaled")
sc.pp.scale(adata, max_value=10, layer="scaled")

# 8. 主成分分析
print("\n8. 主成分分析...")
sc.pp.pca(adata, layer="scaled", svd_solver="arpack")
print("绘制PCA方差比例图...")
fig, ax = plt.subplots(figsize=(10, 6))
sc.pl.pca_variance_ratio(adata, log=True)
plt.tight_layout()
plt.savefig('./datasets/sc_pbmc3k/pca_variance_ratio.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("PCA方差比例图保存完成")

# 9. 计算邻居图
print("\n9. 计算邻居图...")
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=10)

# 10. UMAP降维
print("\n10. UMAP降维...")
sc.tl.umap(adata)
print("绘制UMAP质量控制图...")
fig, ax = plt.subplots(figsize=(12, 5))
sc.pl.umap(adata, color=['total_counts', 'n_genes_by_counts'], 
           use_raw=False, ncols=2)
plt.tight_layout()
plt.savefig('./datasets/sc_pbmc3k/umap_qc.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("UMAP质量控制图保存完成")

# 11. 聚类分析
print("\n11. 聚类分析...")


print("开始执行Leiden聚类...")
sc.tl.leiden(
    adata,
    resolution=0.7,
    random_state=42,
    flavor="igraph",
    n_iterations=2,
    directed=False,
)
adata.obs["leiden"] = adata.obs["leiden"].copy()
adata.uns["leiden"] = adata.uns["leiden"].copy()
adata.obsm["X_umap"] = adata.obsm["X_umap"].copy()  
print("Leiden聚类完成")
print(f"聚类数量: {len(adata.obs['leiden'].cat.categories)}")
print(f"聚类分布: {adata.obs['leiden'].value_counts()}")


print("绘制聚类结果...")
try:
    fig, ax = plt.subplots(figsize=(12, 4))
    sc.pl.umap(adata, color=["leiden", "CD14", "NKG7"], ncols=3)
    plt.tight_layout()
    plt.savefig('./datasets/sc_pbmc3k/umap_clusters.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("聚类图保存完成")
except Exception as e:
    print(f"绘制聚类图时出错: {e}")

# 12. 差异表达分析
print("\n12. 差异表达分析...")
try:
    print("开始差异表达分析...")
    sc.tl.rank_genes_groups(adata, "leiden", mask_var="highly_variable", method="t-test")
    print("绘制差异表达基因图...")
    fig, ax = plt.subplots(figsize=(12, 8))
    sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)
    plt.tight_layout()
    plt.savefig('./datasets/sc_pbmc3k/rank_genes_groups.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("差异表达分析完成")

    # 显示每个聚类的前10个标记基因
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    marker_genes_df = pd.DataFrame(
        {f"{group}_{key[:1]}": result[key][group]
         for group in groups
         for key in ['names', 'pvals']}
    ).head(10)

    print("\n每个聚类的前10个标记基因:")
    print(marker_genes_df)

    # 保存标记基因结果
    marker_genes_df.to_csv('./datasets/sc_pbmc3k/marker_genes.csv')
    print("标记基因结果已保存")
except Exception as e:
    print(f"差异表达分析时出错: {e}")
    print("跳过差异表达分析...")

# 13. 可视化标记基因
print("\n13. 可视化标记基因...")
try:
    # 定义已知的标记基因
    marker_genes = [
        "IL7R", "CD79A", "MS4A1", "CD8A", "CD8B", "LYZ", "CD14",
        "LGALS3", "S100A8", "GNLY", "NKG7", "KLRB1",
        "FCGR3A", "MS4A7", "FCER1A", "CST3", "PPBP"
    ]

    # 检查哪些基因在数据中存在
    available_genes = [gene for gene in marker_genes if gene in adata.var_names]
    print(f"可用的标记基因: {available_genes}")

    if len(available_genes) > 0:
        # 点图
        print("绘制标记基因点图...")
        fig, ax = plt.subplots(figsize=(12, 8))
        sc.pl.dotplot(adata, available_genes, groupby='leiden')
        plt.tight_layout()
        plt.savefig('./datasets/sc_pbmc3k/marker_genes_dotplot.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print("标记基因点图保存完成")

        # 堆叠小提琴图
        print("绘制标记基因小提琴图...")
        fig, ax = plt.subplots(figsize=(12, 8))
        sc.pl.stacked_violin(adata, available_genes, groupby='leiden')
        plt.tight_layout()
        plt.savefig('./datasets/sc_pbmc3k/marker_genes_violin.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print("标记基因小提琴图保存完成")
    else:
        print("没有找到可用的标记基因")
except Exception as e:
    print(f"可视化标记基因时出错: {e}")

# 14. 细胞类型注释
print("\n14. 细胞类型注释...")
try:
    # 基于标记基因进行细胞类型注释
    new_cluster_names = [
        "CD8 T", "CD4 T", "B", "CD14+ Monocytes",
        "NK", "FCGR3A+ Monocytes", "Dendritic", "Megakaryocytes"
    ]

    # 确保聚类数量匹配
    if len(new_cluster_names) >= len(adata.obs['leiden'].cat.categories):
        adata.rename_categories('leiden', new_cluster_names[:len(adata.obs['leiden'].cat.categories)])
        print("细胞类型注释完成")
    else:
        print(f"警告：聚类数量({len(adata.obs['leiden'].cat.categories)})超过预定义细胞类型数量({len(new_cluster_names)})")

    # 绘制注释后的UMAP
    print("绘制注释后的UMAP图...")
    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.umap(adata, color='leiden', legend_loc='on data', title='', frameon=False)
    plt.tight_layout()
    plt.savefig('./datasets/sc_pbmc3k/umap_annotated.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("注释后的UMAP图保存完成")
except Exception as e:
    print(f"细胞类型注释时出错: {e}")

# 15. 保存结果
print("\n15. 保存结果...")
try:
    adata.write(results_file, compression='gzip')
    print(f"结果已保存到: {results_file}")
except Exception as e:
    print(f"保存结果时出错: {e}")

# 16. 生成分析报告
print("\n16. 生成分析报告...")
try:
    report = f"""
=== 单细胞测序数据分析报告 ===

数据概览:
- 原始细胞数量: {adata.n_obs}
- 原始基因数量: {adata.n_vars}
- 过滤后细胞数量: {adata.n_obs}
- 过滤后基因数量: {adata.n_vars}
- 高变异基因数量: {sum(adata.var.highly_variable)}

质量控制:
- 线粒体基因数量: {adata.var['mt'].sum()}
- 平均线粒体基因比例: {adata.obs['pct_counts_mt'].mean():.2f}%

聚类结果:
- 聚类数量: {len(adata.obs['leiden'].cat.categories)}
- 聚类分布:
{adata.obs['leiden'].value_counts().to_string()}

分析完成！
结果文件: {results_file}
图片文件: ./datasets/sc_pbmc3k/
标记基因文件: ./datasets/sc_pbmc3k/marker_genes.csv
"""

    print(report)

    # 保存报告
    with open('../../datasets/sc_pbmc3k/analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("分析报告已保存")
except Exception as e:
    print(f"生成分析报告时出错: {e}")

print("\n=== 分析完成！ ===")

