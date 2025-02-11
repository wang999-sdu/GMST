# evaluation.py
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
import libpysal as lp
import esda.moran as esda_moran
import esda.geary as esda_geary

def evaluate_model(encoded_data_np, adata, num_clusters):
    gmm = GaussianMixture(n_components=num_clusters, covariance_type='full', random_state=49)
    clusters = gmm.fit_predict(encoded_data_np)

    # 计算评估指标
    true_labels = np.array(adata.obs['GT'])
    predicted_labels = clusters
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    fmi = fowlkes_mallows_score(true_labels, predicted_labels)

    # 空间自相关分析
    w = lp.weights.KNN.from_array(adata.obsm['spatial'], k=15)
    moran = esda_moran.Moran(clusters, w)
    geary = esda_geary.Geary(clusters, w)

    return ari, nmi, fmi, moran, geary