# config.py
import torch

# 设置随机种子
torch.manual_seed(2583)
np.random.seed(2583)

# 设备配置
device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() >= 2 else "cpu")


# 超参数
input_dim = None  # 将在数据加载后设置
hidden_dim = 64
num_clusters = 7
batch_size = 1024
epoch_num = 390
learning_rate = 0.001