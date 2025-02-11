# visualization.py
import matplotlib.pyplot as plt
import cv2
import simplejson
import scanpy as sc

def visualize_clusters(adata, clusters, image_path, scalefactors_path):
    with open(scalefactors_path, 'r') as f:
        scalefactors = simplejson.load(f)
    tissue_hires_scalef = scalefactors["tissue_hires_scalef"]
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fx = 1 / tissue_hires_scalef
    fy = 1 / tissue_hires_scalef
    image_rgb = cv2.resize(image_rgb, None, fx=fx, fy=fy)

    spot_positions = np.array(adata.obsm['spatial'])
    for position, spot_class in zip(spot_positions, clusters):
        x, y = position
        color = np.array(plt.cm.get_cmap('tab20')(spot_class)[:3]) * 255
        cv2.circle(image_rgb, (x, y), radius=50, color=color, thickness=-1)

    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.savefig('figures/clusterfig/DLPFC151507_cluster.png')

def paga_analysis(adata):
    sc.pp.neighbors(adata)
    sc.tl.paga(adata, groups='cluster')
    sc.pl.paga(adata, save='figures/DLPFC151507_PAGA.png')

def umap_visualization(adata):
    sc.tl.umap(adata)
    sc.pl.umap(adata, color='cluster_result', save='figures/DLPFC151507_UMAP.png')