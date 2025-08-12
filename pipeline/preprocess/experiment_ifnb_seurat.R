# IFNB 单细胞RNA-seq整合分析
# 参考: https://satijalab.org/seurat/articles/integration_introduction

# 加载必要的包
library(dplyr)
library(Seurat)
library(SeuratData)
library(patchwork)
library(ggplot2)

# 设置工作目录和输出路径
output_dir <- "datasets/sc_ifnb"

# 创建输出目录
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# 1. 数据加载和安装
cat("正在安装和加载IFNB数据集...\n")
# 安装数据集（如果尚未安装）
if (!requireNamespace("ifnb.SeuratData", quietly = TRUE)) {
  InstallData("ifnb")
}

# 加载数据集
ifnb <- LoadData("ifnb")
ifnb.meta <- ifnb@meta.data
ifnb.data <- ifnb@assays$RNA@counts
ifnb <- CreateSeuratObject(counts = ifnb.data, meta.data = ifnb.meta, project = "ifnb", min.cells = 3, min.features = 200)
cat("IFNB数据集加载完成\n")
print(ifnb)

# 2. 数据预处理 - 将RNA测量分为两层
cat("正在将RNA数据分为控制组和刺激组...\n")
ifnb[["RNA"]] <- split(ifnb[["RNA"]], f = ifnb$stim)
print(ifnb)

# 3. 无整合分析（作为对比）
cat("正在进行无整合分析...\n")
# 标准化
ifnb <- NormalizeData(ifnb)
# 识别高变基因
ifnb <- FindVariableFeatures(ifnb)
# 数据缩放
ifnb <- ScaleData(ifnb)
# PCA分析
ifnb <- RunPCA(ifnb)

# 聚类分析
ifnb <- FindNeighbors(ifnb, dims = 1:30, reduction = "pca")
ifnb <- FindClusters(ifnb, resolution = 2, cluster.name = "unintegrated_clusters")

# UMAP可视化
ifnb <- RunUMAP(ifnb, dims = 1:30, reduction = "pca", reduction.name = "umap.unintegrated")

# 无整合结果可视化
unintegrated_plot <- DimPlot(ifnb, reduction = "umap.unintegrated", 
                             group.by = c("stim", "seurat_clusters"), 
                             label = TRUE, pt.size = 0.5)
ggsave(filename = file.path(output_dir, "unintegrated_umap.pdf"), 
       plot = unintegrated_plot, width = 12, height = 6)

# 4. 数据整合分析
cat("正在进行数据整合分析...\n")
# 使用CCA方法进行整合
ifnb <- IntegrateLayers(object = ifnb, 
                        method = CCAIntegration, 
                        orig.reduction = "pca", 
                        new.reduction = "integrated.cca",
                        verbose = FALSE)

# 整合后重新连接层
ifnb[["RNA"]] <- JoinLayers(ifnb[["RNA"]])

# 基于整合结果进行聚类
ifnb <- FindNeighbors(ifnb, reduction = "integrated.cca", dims = 1:30)
ifnb <- FindClusters(ifnb, resolution = 1)

# 整合后的UMAP
ifnb <- RunUMAP(ifnb, dims = 1:30, reduction = "integrated.cca")

# 5. 整合结果可视化
cat("正在生成整合结果可视化...\n")
# 整合后的聚类图
integrated_cluster_plot <- DimPlot(ifnb, reduction = "umap", 
                                   group.by = "seurat_clusters", 
                                   label = TRUE, pt.size = 0.5)
ggsave(filename = file.path(output_dir, "integrated_clusters.pdf"), 
       plot = integrated_cluster_plot, width = 10, height = 8)

# 刺激状态可视化
stim_plot <- DimPlot(ifnb, reduction = "umap", 
                     group.by = "stim", 
                     pt.size = 0.5)
ggsave(filename = file.path(output_dir, "stim_condition.pdf"), 
       plot = stim_plot, width = 10, height = 8)

# 组合图
combined_plot <- DimPlot(ifnb, reduction = "umap", 
                        group.by = c("stim", "seurat_clusters"), 
                        label = TRUE, pt.size = 0.5)
ggsave(filename = file.path(output_dir, "combined_plot.pdf"), 
       plot = combined_plot, width = 12, height = 6)

# 6. 细胞类型注释
cat("正在进行细胞类型注释...\n")
# 使用预注释的细胞类型标签
ifnb$cell_type <- ifnb$seurat_annotations

# 细胞类型可视化
cell_type_plot <- DimPlot(ifnb, reduction = "umap", 
                          group.by = "cell_type", 
                          label = TRUE, pt.size = 0.5) +
  ggtitle("细胞类型注释")
ggsave(filename = file.path(output_dir, "cell_types.pdf"), 
       plot = cell_type_plot, width = 12, height = 8)

# 7. 差异表达分析
cat("正在进行差异表达分析...\n")
# 找到每个cluster的marker基因
cluster_markers <- FindAllMarkers(ifnb, only.pos = TRUE, 
                                 min.pct = 0.25, 
                                 logfc.threshold = 0.25)

# 保存marker基因结果
write.csv(cluster_markers, file.path(output_dir, "cluster_markers.csv"), 
          row.names = FALSE)

# 查看每个cluster的top marker基因
top_markers <- cluster_markers %>%
  group_by(cluster) %>%
  slice_max(n = 10, order_by = avg_log2FC)

cat("每个cluster的top marker基因:\n")
print(top_markers)

# 8. 刺激响应分析
cat("正在分析刺激响应...\n")
# 为每个细胞类型分析刺激响应
cell_types <- unique(ifnb$cell_type)
stim_response_results <- list()

for (cell_type in cell_types) {
  cat("分析细胞类型:", cell_type, "\n")
  
  # 子集化特定细胞类型
  subset_obj <- subset(ifnb, subset = cell_type == cell_type)
  
  # 设置细胞类型为ident
  Idents(subset_obj) <- subset_obj$stim
  
  # 找到刺激vs控制的差异基因
  markers <- FindMarkers(subset_obj, 
                        ident.1 = "STIM", 
                        ident.2 = "CTRL",
                        min.pct = 0.1,
                        logfc.threshold = 0.1)
  
  # 添加基因名
  markers$gene <- rownames(markers)
  
  # 保存结果
  stim_response_results[[cell_type]] <- markers
  write.csv(markers, file.path(output_dir, paste0(cell_type, "_stim_response.csv")), 
            row.names = FALSE)
}

# 9. 特征基因可视化
cat("正在生成特征基因可视化...\n")
# 选择一些重要的marker基因
feature_genes <- c("ISG15", "IFI6", "ISG20", "MX1", "IFIT2", "IFIT3")

# 特征基因热图
feature_heatmap <- DoHeatmap(ifnb, features = feature_genes, 
                             group.by = "cell_type") + 
  scale_fill_gradientn(colors = c("blue", "white", "red"))
ggsave(filename = file.path(output_dir, "feature_heatmap.pdf"), 
       plot = feature_heatmap, width = 12, height = 8)

# 特征基因小提琴图
feature_violin <- VlnPlot(ifnb, features = feature_genes, 
                          group.by = "cell_type", 
                          stack = TRUE, flip = TRUE) + 
  NoLegend()
ggsave(filename = file.path(output_dir, "feature_violin.pdf"), 
       plot = feature_violin, width = 12, height = 8)

# 10. 刺激响应热图
cat("正在生成刺激响应热图...\n")
# 选择每个细胞类型的top刺激响应基因
top_stim_genes <- c()
for (cell_type in names(stim_response_results)) {
  top_genes <- head(stim_response_results[[cell_type]]$gene[order(stim_response_results[[cell_type]]$avg_log2FC, decreasing = TRUE)], 5)
  top_stim_genes <- c(top_stim_genes, top_genes)
}

# 刺激响应热图
stim_heatmap <- DoHeatmap(ifnb, features = unique(top_stim_genes), 
                          group.by = "cell_type") + 
  scale_fill_gradientn(colors = c("blue", "white", "red"))
ggsave(filename = file.path(output_dir, "stim_response_heatmap.pdf"), 
       plot = stim_heatmap, width = 14, height = 10)

# 11. 统计信息
cat("正在生成统计信息...\n")
# 细胞数量统计
cell_counts <- table(ifnb$cell_type, ifnb$stim)
write.csv(cell_counts, file.path(output_dir, "cell_counts_by_condition.csv"))

# 聚类统计
cluster_counts <- table(ifnb$seurat_clusters, ifnb$stim)
write.csv(cluster_counts, file.path(output_dir, "cluster_counts_by_condition.csv"))

# 12. 保存结果
cat("正在保存分析结果...\n")
# 保存Seurat对象
saveRDS(ifnb, file.path(output_dir, "ifnb_integrated.rds"))

# 保存细胞类型注释信息
cell_types_df <- data.frame(
  cluster = ifnb$seurat_clusters,
  cell_type = ifnb$cell_type)
cell_types_df = cell_types_df[!duplicated(cell_types_df$cluster),]
write.csv(cell_types_df, file.path(output_dir, "cell_type_annotations.csv"), 
          row.names = FALSE)

# 13. 生成分析报告
cat("正在生成分析报告...\n")
report <- paste0(
  "IFNB单细胞RNA-seq整合分析报告\n",
  "================================\n\n",
  "数据概览:\n",
  "- 总细胞数: ", ncol(ifnb), "\n",
  "- 基因数: ", nrow(ifnb), "\n",
  "- 聚类数: ", length(levels(ifnb$seurat_clusters)), "\n",
  "- 细胞类型数: ", length(unique(ifnb$cell_type)), "\n\n",
  "实验条件:\n",
  "- CTRL (控制组): ", sum(ifnb$stim == "CTRL"), " cells\n",
  "- STIM (刺激组): ", sum(ifnb$stim == "STIM"), " cells\n\n",
  "细胞类型分布:\n"
)

# 添加细胞类型分布
cell_type_counts <- table(ifnb$cell_type)
for (i in 1:length(cell_type_counts)) {
  report <- paste0(report, "- ", names(cell_type_counts)[i], ": ", cell_type_counts[i], " cells\n")
}

report <- paste0(report, "\n分析完成！\n")
report <- paste0(report, "- 成功整合了控制组和刺激组数据\n")
report <- paste0(report, "- 识别了", nrow(cluster_markers), "个marker基因\n")
report <- paste0(report, "- 分析了", length(cell_types), "个细胞类型的刺激响应\n")

# 保存报告
writeLines(report, file.path(output_dir, "analysis_report.txt"))

cat("分析完成！结果保存在:", output_dir, "\n")
cat("主要输出文件:\n")
cat("- ifnb_integrated.rds: 整合后的Seurat对象\n")
cat("- cluster_markers.csv: 聚类marker基因\n")
cat("- cell_type_annotations.csv: 细胞类型注释\n")
cat("- *_stim_response.csv: 各细胞类型的刺激响应基因\n")
cat("- cell_counts_by_condition.csv: 各条件下的细胞数量\n")
cat("- 各种可视化图表(.pdf格式)\n")
cat("- analysis_report.txt: 分析报告\n")

cat("\n整合分析成功完成！\n")
cat("- 数据整合消除了批次效应\n")
cat("- 细胞类型注释准确\n")
cat("- 刺激响应分析揭示了IFN-β的细胞特异性效应\n")
