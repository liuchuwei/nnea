library(Seurat)
library(rhdf5)
library(dplyr)

# ============================================================================
# PBMC3K 数据处理
# ============================================================================

cat("开始处理 PBMC3K 数据...\n")

# 检查文件是否存在
if (!file.exists("datasets/sc_pbmc3k/pbmc3k_final.rds")) {
  stop("PBMC3K RDS文件不存在: datasets/sc_pbmc3k/pbmc3k_final.rds")
}

# 加载PBMC3K数据
pbmc3k = readRDS("datasets/sc_pbmc3k/pbmc3k_final.rds")

# 验证数据完整性
if (is.null(pbmc3k@assays$RNA$counts)) {
  stop("PBMC3K数据中缺少RNA counts矩阵")
}

# 设置细胞类型标识
pbmc3k$cell_types = pbmc3k@active.ident

# 提取关键数据
pca = pbmc3k@reductions$pca@cell.embeddings[,1:10]
X = pbmc3k@assays$RNA$counts[VariableFeatures(pbmc3k),]
Meta = pbmc3k@meta.data

# 提取稀疏矩阵信息
i = X@i
p = X@p
x = X@x
dims = X@Dim
rownames = rownames(X)
colnames = colnames(X)
meta_columns = colnames(Meta)

# 创建输出目录
dir.create("datasets/sc_pbmc3k", showWarnings = FALSE, recursive = TRUE)

# 保存到HDF5
h5createFile("datasets/sc_pbmc3k/sc_pbmc3k.h5")
h5write(i, "datasets/sc_pbmc3k/sc_pbmc3k.h5", "i") 
h5write(p, "datasets/sc_pbmc3k/sc_pbmc3k.h5", "p")
h5write(x, "datasets/sc_pbmc3k/sc_pbmc3k.h5", "x")
h5write(dims, "datasets/sc_pbmc3k/sc_pbmc3k.h5", "dims")
h5write(as.matrix(Meta), "datasets/sc_pbmc3k/sc_pbmc3k.h5", "Meta")
h5write(pca, "datasets/sc_pbmc3k/sc_pbmc3k.h5", "pca")
if (!is.null(rownames)) h5write(rownames, "datasets/sc_pbmc3k/sc_pbmc3k.h5", "rownames")
if (!is.null(colnames)) h5write(colnames, "datasets/sc_pbmc3k/sc_pbmc3k.h5", "colnames")
h5write(meta_columns, "datasets/sc_pbmc3k/sc_pbmc3k.h5", "Meta_columns")
H5close()

cat("PBMC3K 数据处理完成！\n")
cat("- 细胞数:", ncol(X), "\n")
cat("- 基因数:", nrow(X), "\n")
cat("- 细胞类型数:", length(unique(Meta$cell_types)), "\n")

# ============================================================================
# IFNB 数据处理
# ============================================================================

cat("\n开始处理 IFNB 数据...\n")

# 检查文件是否存在
if (!file.exists("datasets/sc_ifnb/ifnb_integrated.rds")) {
  stop("IFNB RDS文件不存在: datasets/sc_ifnb/ifnb_integrated.rds")
}

# 加载IFNB数据
ifnb = readRDS("datasets/sc_ifnb/ifnb_integrated.rds")

# 验证数据完整性
if (is.null(ifnb@assays$RNA$counts)) {
  stop("IFNB数据中缺少RNA counts矩阵")
}

# 提取关键数据
ifnb_pca = ifnb@reductions$pca@cell.embeddings[,1:30]
ifnb_X  =  FindVariableFeatures(ifnb, selection.method = "vst", nfeatures = 7000)
ifnb_X = ifnb@assays$RNA$counts[VariableFeatures(ifnb),]
ifnb_Meta = ifnb@meta.data

# 提取稀疏矩阵信息
ifnb_i = ifnb_X@i
ifnb_p = ifnb_X@p
ifnb_x = ifnb_X@x
ifnb_dims = ifnb_X@Dim
ifnb_rownames = rownames(ifnb_X)
ifnb_colnames = colnames(ifnb_X)
ifnb_meta_columns = colnames(ifnb_Meta)

# 创建输出目录
dir.create("datasets/sc_ifnb", showWarnings = FALSE, recursive = TRUE)

# 保存到HDF5
h5createFile("datasets/sc_ifnb/sc_ifnb.h5")
h5write(ifnb_i, "datasets/sc_ifnb/sc_ifnb.h5", "i") 
h5write(ifnb_p, "datasets/sc_ifnb/sc_ifnb.h5", "p")
h5write(ifnb_x, "datasets/sc_ifnb/sc_ifnb.h5", "x")
h5write(ifnb_dims, "datasets/sc_ifnb/sc_ifnb.h5", "dims")
h5write(as.matrix(ifnb_Meta), "datasets/sc_ifnb/sc_ifnb.h5", "Meta")
h5write(ifnb_pca, "datasets/sc_ifnb/sc_ifnb.h5", "pca")
if (!is.null(ifnb_rownames)) h5write(ifnb_rownames, "datasets/sc_ifnb/sc_ifnb.h5", "rownames")
if (!is.null(ifnb_colnames)) h5write(ifnb_colnames, "datasets/sc_ifnb/sc_ifnb.h5", "colnames")
h5write(ifnb_meta_columns, "datasets/sc_ifnb/sc_ifnb.h5", "Meta_columns")
H5close()

cat("IFNB 数据处理完成！\n")
cat("- 细胞数:", ncol(ifnb_X), "\n")
cat("- 基因数:", nrow(ifnb_X), "\n")
cat("- 细胞类型数:", length(unique(ifnb_Meta$cell_types)), "\n")
cat("- 实验条件:", paste(unique(ifnb_Meta$stim), collapse = ", "), "\n")

# ============================================================================
# 数据验证和统计信息
# ============================================================================

cat("\n=== 数据验证和统计信息 ===\n")

# PBMC3K统计
cat("PBMC3K 统计信息:\n")
cat("- 总细胞数:", ncol(X), "\n")
cat("- 高变基因数:", nrow(X), "\n")
cat("- 细胞类型分布:\n")
pbmc3k_cell_counts = table(Meta$cell_types)
for (cell_type in names(pbmc3k_cell_counts)) {
  cat("  ", cell_type, ":", pbmc3k_cell_counts[cell_type], "cells\n")
}

# IFNB统计
cat("\nIFNB 统计信息:\n")
cat("- 总细胞数:", ncol(ifnb_X), "\n")
cat("- 高变基因数:", nrow(ifnb_X), "\n")
cat("- 细胞类型分布:\n")
ifnb_cell_counts = table(ifnb_Meta$cell_types)
for (cell_type in names(ifnb_cell_counts)) {
  cat("  ", cell_type, ":", ifnb_cell_counts[cell_type], "cells\n")
}

cat("\n实验条件分布:\n")
ifnb_stim_counts = table(ifnb_Meta$stim)
for (condition in names(ifnb_stim_counts)) {
  cat("  ", condition, ":", ifnb_stim_counts[condition], "cells\n")
}

cat("\n所有数据处理完成！\n")
cat("- PBMC3K HDF5文件: datasets/sc_pbmc3k/sc_pbmc3k.h5\n")
cat("- IFNB HDF5文件: datasets/sc_ifnb/sc_ifnb.h5\n")
