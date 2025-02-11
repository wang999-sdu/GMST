# training.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from models import MaskedAutoencoder
from config import device, hidden_dim, batch_size, epoch_num, learning_rate

def train_model(model, data_tensor, adj_matrix_tensor):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epoch_num):
        permutation = torch.randperm(data_tensor.size()[0])
        epoch_loss = 0
        count = 0
        for i in range(0, data_tensor.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x = data_tensor[indices]
            batch_adj = adj_matrix_tensor[indices][:, indices]

            optimizer.zero_grad()
            recon_x = model(batch_x, batch_adj)
            loss = F.mse_loss(recon_x, batch_x, reduction='sum')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            count += 1

        print(f'Epoch [{epoch + 1}/{epoch_num}], Loss: {epoch_loss/count:.4f}')