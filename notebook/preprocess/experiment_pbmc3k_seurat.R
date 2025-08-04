# PBMC3K 单细胞RNA-seq分析
# 参考: https://satijalab.org/seurat/articles/pbmc3k_tutorial

# 加载必要的包
library(dplyr)
library(Seurat)
library(patchwork)
library(ggplot2)

# 设置工作目录和输出路径
data_dir <- "data/single_cell/pbmc3k"
output_dir <- "datasets/sc_pbmc3k"

# 创建输出目录
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# 1. 数据加载和Seurat对象创建
cat("正在加载PBMC3K数据...\n")
pbmc.data <- Read10X(data.dir = data_dir)

# 创建Seurat对象
pbmc <- CreateSeuratObject(counts = pbmc.data, 
                          project = "pbmc3k", 
                          min.cells = 3, 
                          min.features = 200)

cat("Seurat对象创建完成\n")
print(pbmc)

# 2. 质控指标计算
cat("正在计算质控指标...\n")
# 计算线粒体基因比例
pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT-")

# 查看质控指标
cat("质控指标统计:\n")
print(head(pbmc@meta.data, 5))

# 3. 质控可视化
cat("正在生成质控图表...\n")
# 质控指标分布图
qc_plot <- VlnPlot(pbmc, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), 
                   ncol = 3, pt.size = 0.1)

# 保存质控图
ggsave(filename = file.path(output_dir, "qc_metrics.pdf"), 
       plot = qc_plot, width = 12, height = 4)

# 特征相关性图
feature_correlation <- FeatureScatter(pbmc, feature1 = "nCount_RNA", feature2 = "percent.mt")
ggsave(filename = file.path(output_dir, "feature_correlation.pdf"), 
       plot = feature_correlation, width = 8, height = 6)

# 4. 细胞过滤
cat("正在过滤低质量细胞...\n")
pbmc <- subset(pbmc, subset = nFeature_RNA > 200 & 
               nFeature_RNA < 2500 & 
               percent.mt < 5)

cat("过滤后的细胞数量:", ncol(pbmc), "\n")

# 5. 数据标准化
cat("正在进行数据标准化...\n")
pbmc <- NormalizeData(pbmc, normalization.method = "LogNormalize", scale.factor = 10000)

# 6. 识别高变基因
cat("正在识别高变基因...\n")
pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 2000)

# 查看前10个高变基因
top10 <- head(VariableFeatures(pbmc), 10)
cat("前10个高变基因:\n")
print(top10)

# 高变基因可视化
plot1 <- VariableFeaturePlot(pbmc)
plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
var_feature_plot = plot1 + plot2
ggsave(filename = file.path(output_dir, "variable_features.pdf"), 
       plot = var_feature_plot, width = 10, height = 6)

# 7. 数据缩放
cat("正在进行数据缩放...\n")
all.genes <- rownames(pbmc)
pbmc <- ScaleData(pbmc, features = all.genes)

# 8. 线性降维 (PCA)
cat("正在进行PCA分析...\n")
pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc))

# PCA结果可视化
pca_plot <- VizDimLoadings(pbmc, dims = 1:2, reduction = "pca")
ggsave(filename = file.path(output_dir, "pca_loadings.pdf"), 
       plot = pca_plot, width = 10, height = 6)

# PCA热图
pca_heatmap <- DimHeatmap(pbmc, dims = 1:15, cells = 500, balanced = TRUE)
ggsave(filename = file.path(output_dir, "pca_heatmap.pdf"), 
       plot = pca_heatmap, width = 12, height = 8)

# 9. 确定主成分数量
cat("正在确定主成分数量...\n")
# Elbow图
elbow_plot <- ElbowPlot(pbmc, ndims = 50)
ggsave(filename = file.path(output_dir, "elbow_plot.pdf"), 
       plot = elbow_plot, width = 8, height = 6)

# 10. 聚类分析
cat("正在进行聚类分析...\n")
# 使用前20个主成分进行聚类
pbmc <- FindNeighbors(pbmc, dims = 1:10)
pbmc <- FindClusters(pbmc, resolution = 0.5)

# 11. 非线性降维 (UMAP)
cat("正在进行UMAP降维...\n")
pbmc <- RunUMAP(pbmc, dims = 1:10)

# UMAP聚类可视化
umap_cluster_plot <- DimPlot(pbmc, reduction = "umap", label = TRUE, pt.size = 0.5)
ggsave(filename = file.path(output_dir, "umap_clusters.pdf"), 
       plot = umap_cluster_plot, width = 10, height = 8)

# 12. 差异表达分析
cat("正在进行差异表达分析...\n")
# 找到每个cluster的marker基因
cluster_markers <- FindAllMarkers(pbmc, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)

# 保存marker基因结果
write.csv(cluster_markers, file.path(output_dir, "cluster_markers.csv"), row.names = FALSE)

# 查看每个cluster的top marker基因
top_markers <- cluster_markers %>%
  group_by(cluster) %>%
  slice_max(n = 10, order_by = avg_log2FC)

cat("每个cluster的top marker基因:\n")
print(top_markers)

# 13. 细胞类型注释
cat("正在进行细胞类型注释...\n")
# 基于已知marker基因进行细胞类型注释
new.cluster.ids <- c("Naive CD4 T", "CD14+ Mono", "Memory CD4 T", "B", "CD8 T", 
                     "FCGR3A+ Mono", "NK", "DC", "Platelet")
names(new.cluster.ids) <- levels(pbmc)
pbmc <- RenameIdents(pbmc, new.cluster.ids)

# 注释后的UMAP图
umap_annotated_plot <- DimPlot(pbmc, reduction = "umap", label = TRUE, pt.size = 0.5) + 
  NoLegend() +
  ggtitle("PBMC3K细胞类型注释")

ggsave(filename = file.path(output_dir, "umap_annotated.pdf"), 
       plot = umap_annotated_plot, width = 10, height = 8)

# 14. 特征基因可视化
cat("正在生成特征基因可视化...\n")
# 选择一些重要的marker基因进行可视化
feature_genes <- c("IL7R", "CCR7", "CD14", "LYZ", "IL7R", "S100A4", "MS4A1", "CD8A", "FCGR3A", "MS4A7",
                   "GNLY", "NKG7", "FCER1A", "CST3", "PPBP")

# 特征基因热图
feature_heatmap <- DoHeatmap(pbmc, features = feature_genes) + NoLegend()
ggsave(filename = file.path(output_dir, "feature_heatmap.pdf"), 
       plot = feature_heatmap, width = 12, height = 8)

# 特征基因小提琴图
feature_violin <- VlnPlot(pbmc, features = feature_genes, stack = TRUE, flip = TRUE) + 
  NoLegend()
ggsave(filename = file.path(output_dir, "feature_violin.pdf"), 
       plot = feature_violin, width = 12, height = 8)

# 15. 保存结果
cat("正在保存分析结果...\n")
# 保存Seurat对象
saveRDS(pbmc, file.path(output_dir, "pbmc3k_final.rds"))

# 保存细胞类型注释信息
cell_types <- data.frame(
  cluster = 0:(length(levels(pbmc))-1),
  cell_type = new.cluster.ids
)
write.csv(cell_types, file.path(output_dir, "cell_type_annotations.csv"), row.names = FALSE)

# 16. 生成分析报告
cat("正在生成分析报告...\n")
report <- paste0(
  "PBMC3K单细胞RNA-seq分析报告\n",
  "================================\n\n",
  "数据概览:\n",
  "- 原始细胞数: ", ncol(pbmc.data), "\n",
  "- 过滤后细胞数: ", ncol(pbmc), "\n",
  "- 基因数: ", nrow(pbmc), "\n",
  "- 聚类数: ", length(levels(pbmc)), "\n\n",
  "细胞类型分布:\n"
)

# 添加细胞类型分布
cell_counts <- table(Idents(pbmc))
for (i in 1:length(cell_counts)) {
  report <- paste0(report, "- ", names(cell_counts)[i], ": ", cell_counts[i], " cells\n")
}

# 保存报告
writeLines(report, file.path(output_dir, "analysis_report.txt"))

cat("分析完成！结果保存在:", output_dir, "\n")
cat("主要输出文件:\n")
cat("- pbmc3k_final.rds: 最终的Seurat对象\n")
cat("- cluster_markers.csv: 聚类marker基因\n")
cat("- cell_type_annotations.csv: 细胞类型注释\n")
cat("- 各种可视化图表(.pdf格式)\n")
cat("- analysis_report.txt: 分析报告\n")
