# main.py
from data_loading import load_data
from preprocessing import preprocess_data
from models import MaskedAutoencoder
from training import train_model
from evaluation import evaluate_model
from visualization import visualize_clusters, paga_analysis, umap_visualization
from config import device, hidden_dim, num_clusters

# 数据路径
data_path = "/home/wangwm/GFT_VAE/20241017/#1ST_data/0.DLPFC/151507"
image_path = f"{data_path}/spatial/tissue_hires_image.png"
scalefactors_path = f"{data_path}/spatial/scalefactors_json.json"


def main():
    # 加载数据
    adata = load_data()

    # 数据预处理
    data_tensor, adj_matrix_tensor, adata = preprocess_data(adata)

    # 初始化模型
    model = MaskedAutoencoder(data_tensor.shape[1], hidden_dim).to(device)
    model.apply(weights_init)

    # 训练模型
    train_model(model, data_tensor, adj_matrix_tensor)

    # 使用编码器对数据进行编码
    with torch.no_grad():
        gcn_output = model.gcn(data_tensor, data_tensor, adj_matrix_tensor)
        encoded_data = model.fc_encoder(gcn_output)
        encoded_data_np = encoded_data.cpu().numpy()

    # 评估模型
    ari, nmi, fmi, moran, geary = evaluate_model(encoded_data_np, adata, num_clusters)

    # 可视化结果
    visualize_clusters(adata, clusters, image_path, scalefactors_path)
    paga_analysis(adata)
    umap_visualization(adata)

if __name__ == "__main__":
    main()