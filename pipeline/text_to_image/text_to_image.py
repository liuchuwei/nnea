# Basic Usage of OmiCLIP Model
import os
import pandas as pd
import numpy as np
import scanpy as sc
import anndata
from PIL import Image

import loki.utils
import loki.preprocess

data_dir = 'Loki/data/basic_usage/'

# Load OmiCLIP Model
model_path = os.path.join(data_dir, 'checkpoint.pt')
device = 'cpu'
model, preprocess, tokenizer = loki.utils.load_model(model_path, device)
model.eval()

# Encode Image
image_path = os.path.join(data_dir, 'demo_data', 'TUM-TCGA-TLSHWGSQ.tif')
image = Image.open(image_path)
image.show()
image_embeddings = loki.utils.encode_images(model, preprocess, [image_path], device)
image_embeddings.shape


# Encode Text
text = ['TP53 EPCAM KRAS EGFR DEFA5 DEFA6 CEACAM5 CEA KRT18 KRT8 KRT19 CDH17 CK20 MYO6 TP53BP2 PLA2G2A CLDN7 TJP1 PKP3 DSP']
text_embeddings = loki.utils.encode_texts(model, tokenizer, text, device)
text_embeddings.shape

# Calculate Similarity
dot_similarity = image_embeddings @ text_embeddings.T
dot_similarity

# Examples of preprocessing ST data, scRNA-seq data, bulk RNA-seq data, and whole image
# Encode transcriptome from AnnData object (ST data or scRNA-seq data)
ad_path = os.path.join(data_dir, 'demo_data', 'RZ_GT_P2.h5ad')
ad = sc.read_h5ad(ad_path)
ad

house_keeping_genes = pd.read_csv(os.path.join(data_dir, 'demo_data', 'housekeeping_genes.csv'), index_col = 0)
top_k_genes_str = loki.preprocess.generate_gene_df(ad, house_keeping_genes)
top_k_genes_str

text_embeddings = loki.utils.encode_text_df(model, tokenizer, top_k_genes_str, 'label', device)
text_embeddings.shape

# Encode transcriptome from GCT object (bulk RNA-seq data)
sc_data_path = os.path.join(data_dir, 'demo_data', 'bulk_fibroblasts.gct')
gct_data = loki.preprocess.read_gct(sc_data_path)
gct_data

bulk_ad = anndata.AnnData(pd.DataFrame(gct_data.iloc[:, 3:].mean(axis=1)).T)
bulk_ad.var.index = gct_data['Description']
bulk_text_feature = loki.preprocess.generate_gene_df(bulk_ad, house_keeping_genes, todense=False)
bulk_text_feature

text_embeddings = loki.utils.encode_text_df(model, tokenizer, bulk_text_feature, 'label', device)
text_embeddings.shape

# Encode patches from a whole image
coord = pd.read_csv(os.path.join(data_dir, 'demo_data', 'coord.csv'), index_col=0)
coord

img = Image.open(os.path.join(data_dir, 'demo_data', 'whole_img.png'))
img.show()

img_array = np.asarray(img)
patch_dir = os.path.join(data_dir, 'demo_data', 'patch')
loki.preprocess.segment_patches(img_array, coord, patch_dir)

img_list = os.listdir(patch_dir)
patch_paths = [os.path.join(patch_dir, fn) for fn in img_list]

image_embeddings = loki.utils.encode_images(model, preprocess, patch_paths, device)
image_embeddings.shape
