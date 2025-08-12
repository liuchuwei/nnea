# Lymphocyte male-female differential expression analysis and enrichment analysis
# =============================================================================

# Load necessary R packages
library(data.table)
library(tibble)
library(stringr)
library(limma)
library(edgeR)
library(ggplot2)
library(pheatmap)
library(VennDiagram)
library(ggrepel)
library(dplyr)
library(clusterProfiler)
library(enrichplot)
library(org.Hs.eg.db)

# Set working directory and output path
output_dir <- "results/lym_gender_analysis"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

### 1. Data loading and preprocessing ----
print("=== Loading data ===")
cell_exp = data.table::fread("data/cell_line/OmicsExpressionProteinCodingGenesTPMLogp1.csv")
hugo_geneset = data.table::fread("data/hugo_2025.txt")
cell_ann = data.table::fread("data/cell_line/Celligner_info.csv")

print(paste("Expression data dimensions:", paste(dim(cell_exp), collapse = " x ")))
print(paste("Cell annotation dimensions:", paste(dim(cell_ann), collapse = " x ")))

## Organize data
cell_exp = tibble::column_to_rownames(cell_exp, var = "V1")
colnames(cell_exp) = stringr::str_split(colnames(cell_exp), " ", simplify = T)[,1]
cell_ann = subset(cell_ann, type == "CL")

## Extract lymphocyte data
print("=== Extracting lymphocyte data ===")
table(cell_ann$lineage)
lym_dat = subset(cell_ann, lineage == "lymphocyte")
lym_dat = subset(lym_dat, sex != "Unknown")

print(paste("Number of lymphocyte samples:", nrow(lym_dat)))
print("Gender distribution:")
print(table(lym_dat$sex))

# Get common samples
comm_sample = intersect(lym_dat$sampleID, row.names(cell_exp))
lym_dat = lym_dat[match(comm_sample, lym_dat$sampleID),]
lym_sex = lym_dat$sex
lym_exp = cell_exp[comm_sample,]

print(paste("Final analysis sample count:", length(comm_sample)))

# Filter protein-coding genes
hugo_gene = subset(hugo_geneset, locus_group == "protein-coding gene")
lym_exp = lym_exp[,colnames(lym_exp) %in% hugo_gene$symbol]

print(paste("Number of protein-coding genes:", ncol(lym_exp)))

# Transpose expression matrix (genes as rows, samples as columns)
lym_exp_mat = t(lym_exp)
colnames(lym_exp_mat) = comm_sample

# Create grouping information
lym_sex_df = data.frame(
  sample = comm_sample,
  sex = lym_sex,
  group = ifelse(lym_sex == "Male", "Male", "Female"),
  row.names = comm_sample
)

### 2. Data quality control and filtering ----
print("=== Data quality control ===")

# Filter low-expression genes (expressed in at least 50% of samples with value > 1)
# min_samples = ceiling(ncol(lym_exp_mat) * 0.5)
min_samples = 0
expressed_genes = rowSums(lym_exp_mat > 1) >= min_samples
lym_exp_filtered = lym_exp_mat[expressed_genes, ]

print(paste("Number of genes before filtering:", nrow(lym_exp_mat)))
print(paste("Number of genes after filtering:", nrow(lym_exp_filtered)))

# Perform log2 transformation (data is already log1p transformed TPM)
# Check data distribution
summary_stats = summary(as.vector(lym_exp_filtered))
print("Expression data distribution statistics:")
print(summary_stats)

### 3. Differential expression analysis (Limma) ----
print("=== Differential expression analysis ===")

# Create design matrix
design = model.matrix(~ 0 + group, data = lym_sex_df)
colnames(design) = c("Female", "Male")
print("Design matrix:")
print(head(design))

# Create contrast matrix
contrast_matrix = makeContrasts(
  Male_vs_Female = Male - Female,
  levels = design
)
print("Contrast matrix:")
print(contrast_matrix)

# Fit linear model
fit = lmFit(lym_exp_filtered, design)
fit2 = contrasts.fit(fit, contrast_matrix)
fit2 = eBayes(fit2)

# Get differential expression results
de_results = topTable(fit2, coef = "Male_vs_Female", number = Inf, sort.by = "P")
de_results$gene = rownames(de_results)

print(paste("Total number of genes:", nrow(de_results)))
print(paste("Number of significantly different genes (P < 0.05):", sum(de_results$P.Value < 0.05)))
print(paste("Number of significantly different genes (adj.P < 0.05):", sum(de_results$adj.P.Val < 0.05)))

# Add significance labels
de_results$significance = ifelse(de_results$adj.P.Val < 0.05 & abs(de_results$logFC) > 0.5, 
                                ifelse(de_results$logFC > 0.5, "Up_in_Male", "Up_in_Female"), 
                                "Not_significant")

# Statistically significant genes
sig_summary = table(de_results$significance)
print("Statistically significant genes summary:")
print(sig_summary)

# Save differential expression results
write.csv(de_results, file.path(output_dir, "differential_expression_results.csv"), row.names = FALSE)
saveRDS(de_results, file = "results/lym_gender_analysis/de_results.rds")

### 5. Gene set enrichment analysis (Gene Set Enrichment Analysis) ----
print("=== Gene set enrichment analysis ===")

# 5.1 Prepare GSEA data
# Gene list sorted by logFC
gsea_gene_list = de_results$logFC
names(gsea_gene_list) = de_results$gene
gsea_gene_list = sort(gsea_gene_list, decreasing = TRUE)

print(paste("GSEA gene list length:", length(gsea_gene_list)))

# 5.2 Read c1.all.v2025.1.Hs.symbols.gmt gene sets (chromosome-localized gene sets)
gmt_file = "data/genesets/c1.all.v2025.1.Hs.symbols.gmt"

# Read GMT file
c1_genesets = read.gmt(gmt_file)
print(paste("Number of c1 chromosome gene sets:", length(unique(c1_genesets$term))))

# Perform GSEA analysis
gsea_c1_results = GSEA(gsea_gene_list, 
                      TERM2GENE = c1_genesets,
                      verbose = FALSE,
                      pvalueCutoff = 0.05,
                      pAdjustMethod = "BH")

# Save GSEA results
write.csv(gsea_c1_results@result, file.path(output_dir, "gsea_c1_chromosome_results.csv"), row.names = FALSE)
saveRDS(gsea_c1_results, file = "results/lym_gender_analysis/gsea_cl_results.rds")
