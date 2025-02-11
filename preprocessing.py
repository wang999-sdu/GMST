# preprocessing.py
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance_matrix
from config import device

def preprocess_data(adata):
    # 筛选基因
    gene_nonzero_proportion = np.count_nonzero(adata.X.toarray(), axis=0) / adata.n_obs
    gene_total_counts = np.sum(adata.X.toarray(), axis=0)
    selected_genes = (gene_nonzero_proportion >= 0.01) & (gene_total_counts >= 200)
    adata = adata[:, selected_genes]

    # 归一化和对数变换
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    data = pd.DataFrame(adata.X.toarray())

    # 标准化数据
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # 转换为PyTorch张量
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32).to(device)

    # 计算邻接矩阵
    coordinates = pd.DataFrame(adata.obsm['spatial'])
    dist_matrix = distance_matrix(coordinates, coordinates)
    k = 15
    adj_matrix = np.zeros_like(dist_matrix)
    for i in range(dist_matrix.shape[0]):
        nearest_neighbors = np.argsort(dist_matrix[i])[1:k + 1]
        adj_matrix[i, nearest_neighbors] = 1
        adj_matrix[nearest_neighbors, i] = 1

    adj_matrix_tensor = torch.tensor(adj_matrix, dtype=torch.float32).to(device)

    return data_tensor, adj_matrix_tensor, adata