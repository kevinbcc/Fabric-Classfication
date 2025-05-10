import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)  # 限制在 0~1
        x = x / x.sum(dim=1, keepdim=True)  # 確保輸出相加不超過 1
        return x

class MLP2(nn.Module):
    def __init__(self, input_dim, hidden_dim1,hidden_dim2,hidden_dim3,output_dim):
        super(MLP2, self).__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim1)
        self.hidden2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.hidden3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.output_layer = nn.Linear(hidden_dim3, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu(self.hidden2(x))
        x = self.hidden3(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        x = x / x.sum(dim=1, keepdim=True)
        return x




class SimpleCNN_MLP(nn.Module):
    def __init__(self, input_channels, input_dim, hidden_dim, output_dim):
        super(SimpleCNN_MLP, self).__init__()

        # 第一層 1D 卷積層 (kernel_size=3, padding=1 保持輸出大小不變)
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=7, padding=3)
        # 第二層 1D 卷積層 (kernel_size=3, padding=1 保持輸出大小不變)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

        # Fully Connected Layers
        self.fc1 = nn.Linear(32 * input_dim, hidden_dim)  # 調整為 32 * input_dim
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

        # 確保輸出符合 非負 & 和等於 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1D CNN
        x = self.relu(self.conv1(x))  # 第一層 1D 卷積
        x = self.relu(self.conv2(x))  # 第二層 1D 卷積

        x = x.view(x.size(0), -1)  # 展平 (flatten)

        # Fully Connected Layers
        x = self.relu(self.fc1(x))  # 第一層全連接層
        x = self.relu(self.fc2(x))  # 第二層全連接層
        x = self.fc3(x)  # 輸出層

        # 非負條件 & 和為 1
        x = self.sigmoid(x)
        x = x / x.sum(dim=1, keepdim=True)

        return x


class ResidualBlock1D(nn.Module):
    """一維殘差塊：Conv→BN→ReLU→Conv→BN + 残差連接"""
    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2   = nn.BatchNorm1d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)

class MultiScaleConv1D(nn.Module):
    """多尺度卷積：同時用不同 kernel_size，最後 concat"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.bn    = nn.BatchNorm1d(out_channels * 3)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        y1 = self.conv3(x)
        y2 = self.conv5(x)
        y3 = self.conv7(x)
        out = torch.cat([y1, y2, y3], dim=1)   # concat channel
        out = self.bn(out)
        return self.relu(out)

class ImprovedCNN1DClassifier(nn.Module):
    def __init__(self, input_channels: int, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        # multi-scale + downsample
        self.ms1 = MultiScaleConv1D(input_channels, 16)   # out channels = 16*3 = 48
        self.pool1 = nn.MaxPool1d(kernel_size=2)          # length //=2

        self.res1 = ResidualBlock1D(48, kernel_size=3, padding=1)
        self.ms2  = MultiScaleConv1D(48, 32)               # out channels = 32*3 = 96
        self.pool2 = nn.MaxPool1d(kernel_size=2)           # length //=2

        self.res2 = ResidualBlock1D(96, kernel_size=3, padding=1)

        # 全局平均池化到 (batch, channels, 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),                                  # (batch, 96*1)
            nn.Linear(96, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)        # raw logits
        )

    def forward(self, x):
        # x: (batch, 1, input_dim)
        x = self.ms1(x)       # -> (batch, 48, input_dim)
        x = self.pool1(x)     # -> (batch, 48, input_dim//2)
        x = self.res1(x)      # -> (batch, 48, input_dim//2)

        x = self.ms2(x)       # -> (batch, 96, input_dim//2)
        x = self.pool2(x)     # -> (batch, 96, input_dim//4)
        x = self.res2(x)      # -> (batch, 96, input_dim//4)

        x = self.global_pool(x)  # -> (batch, 96, 1)
        logits = self.classifier(x)  # -> (batch, output_dim)
        return logits



# class SimpleCNN_MLP_classfication(nn.Module):
#     def __init__(self, input_channels, input_dim, hidden_dim, output_dim):
#         super(SimpleCNN_MLP_classfication, self).__init__()
#
#         self.features = nn.Sequential(
#             nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=7, padding=3),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2),
#             nn.Dropout(0.3),
#         )
#
#         conv_out_dim = input_dim // 2  # 因為 MaxPool1d(kernel_size=2)
#         self.classifier = nn.Sequential(
#             nn.Linear(64 * conv_out_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim // 2, output_dim)  # raw logits
#         )
#
#     def forward(self, x):
#         x = self.features(x)  # [B, 64, D/2]
#         x = x.view(x.size(0), -1)  # Flatten
#         x = self.classifier(x)
#         return x




# class SimpleCNN_MLP(nn.Module):
#     def __init__(self, input_channels, input_dim, hidden_dim, output_dim):
#         super(SimpleCNN_MLP, self).__init__()
#
#         # 只保留一層 1D 卷積層
#         self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=7, padding=3)
#         self.relu = nn.ReLU()
#
#         # Fully Connected Layers (調整輸入維度為 16 * input_dim)
#         self.fc1 = nn.Linear(16 * input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
#         self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.relu(self.conv1(x))  # 一層卷積
#         x = x.view(x.size(0), -1)     # 展平
#
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#
#         x = self.sigmoid(x)
#         x = x / x.sum(dim=1, keepdim=True)  # normalize to sum=1
#         return x


# 定義改進的SimpleCNN_MLP模型
class ImprovedSimpleCNN_MLP(nn.Module):
    def __init__(self, input_channels, input_dim, hidden_dim, output_dim):
        super(ImprovedSimpleCNN_MLP, self).__init__()
        # 卷積層
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # 池化層減小特徵尺寸
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # 計算卷積層後的展平尺寸
        conv_output_dim = input_dim // 4  # 兩次池化層將尺寸減小4倍
        self.fc1 = nn.Linear(64 * conv_output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)  # 輸出概率分佈
        return x

# 定義自定義數據集
class CustomDataset(Dataset):
    def __init__(self, num_samples, input_channels, input_dim, num_classes):
        self.data = torch.randn(num_samples, input_channels, input_dim)  # 模擬數據
        self.labels = torch.randint(0, num_classes, (num_samples,))  # 隨機標籤

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]




def FCLS(M, r1, delta):

    A = M
    numloop = A.shape[1]
    e = delta
    eA = e * A
    E = np.vstack((np.ones((1, numloop)), eA))
    EtE = np.dot(E.T, E)
    m = EtE.shape[0]
    One = np.ones((m, 1))
    iEtE = np.linalg.inv(EtE)
    iEtEOne = np.dot(iEtE, One)
    sumiEtEOne = np.sum(iEtEOne)
    weights = np.diag(iEtE)

    sample = r1
    er = e * sample
    f = np.vstack((np.ones((1, 1)), er.reshape(-1, 1)))
    Etf = np.dot(E.T, f)

    tol = 1e-7

    # lamdiv2 calculation
    ls = np.dot(iEtE, Etf)
    lamdiv2 = -(1 - np.dot(ls.T, One)) / sumiEtEOne
    x2 = ls - lamdiv2 * iEtEOne
    x2old = x2.copy()

    if np.any(x2 < -tol):
        Z = np.zeros((m, 1))
        iter = 0
        while np.any(x2 < -tol) and iter < m:
            Z[x2 < -tol] = 1
            zz = np.where(Z)[0]
            x2 = x2old.copy()  # Reset x2
            L = iEtE[zz[:, None], zz]
            ab = zz.shape
            lastrow = ab[0]
            lastcol = lastrow
            L = np.pad(L, ((0, 1), (0, 1)), mode='constant', constant_values=0)
            L[lastrow, :ab[0]] = np.dot(iEtE[:, zz].T, One).flatten()
            L[:ab[0], lastcol] = iEtEOne[zz].flatten()
            L[lastrow, lastcol] = sumiEtEOne
            xerow = x2[zz].flatten()
            xerow = np.append(xerow, 0)
            lagra = np.linalg.solve(L, xerow)

            while np.any(lagra[:ab[0]] > 0):  # Reset Lagrange multipliers
                maxneg = weights[zz] * lagra[:ab[0]]
                iz = np.argmax(maxneg)
                Z[zz[iz]] = 0
                zz = np.where(Z)[0]  # Will always be at least one (prove)
                L = iEtE[zz[:, None], zz]
                ab = zz.shape
                lastrow = ab[0]
                lastcol = lastrow
                L = np.pad(L, ((0, 1), (0, 1)), mode='constant', constant_values=0)
                L[lastrow, :ab[0]] = np.dot(iEtE[:, zz].T, One).flatten()
                L[:ab[0], lastcol] = iEtEOne[zz].flatten()
                L[lastrow, lastcol] = sumiEtEOne
                xerow = x2[zz].flatten()
                xerow = np.append(xerow, 0)
                lagra = np.linalg.solve(L, xerow)

            if zz.size > 0:
                x2 -= np.dot(iEtE[:, zz], lagra[:ab[0]].reshape(-1, 1)) + lagra[lastrow] * iEtEOne

            iter += 1

    abundance = x2
    error_vector = np.dot(A, abundance) - r1

    return abundance.flatten(), error_vector.flatten()

# def FCLS(M, r1, delta, device='cuda'):
#     A = torch.tensor(M, dtype=torch.float32, device=device)
#     sample = torch.tensor(r1, dtype=torch.float32, device=device)
#
#     numloop = A.shape[1]
#     e = delta
#     eA = e * A
#     E = torch.cat((torch.ones((1, numloop), device=device), eA), dim=0)  # shape: [bands+1, numloop]
#     EtE = torch.matmul(E.T, E)
#     m = EtE.shape[0]
#     One = torch.ones((m, 1), device=device)
#
#     iEtE = torch.linalg.inv(EtE)
#     iEtEOne = torch.matmul(iEtE, One)
#     sumiEtEOne = torch.sum(iEtEOne)
#     weights = torch.diag(iEtE)
#
#     er = e * sample
#     f = torch.cat((torch.ones((1,), device=device), er)).view(-1, 1)
#     Etf = torch.matmul(E.T, f)
#
#     tol = 1e-7
#
#     # lamdiv2 calculation
#     ls = torch.matmul(iEtE, Etf)
#     lamdiv2 = -(1 - torch.matmul(ls.T, One)) / sumiEtEOne
#     x2 = ls - lamdiv2 * iEtEOne
#     x2old = x2.clone()
#
#     if torch.any(x2 < -tol):
#         Z = torch.zeros((m, 1), dtype=torch.bool, device=device)
#         iter = 0
#         while torch.any(x2 < -tol) and iter < m:
#             Z[x2 < -tol] = True
#             zz = torch.nonzero(Z, as_tuple=False).view(-1)
#             x2 = x2old.clone()
#
#             ab0 = zz.shape[0]
#             L_pad = torch.zeros((ab0 + 1, ab0 + 1), device=device)
#             if ab0 > 0:
#                 L = iEtE[zz][:, zz]
#                 L_pad[:ab0, :ab0] = L
#                 L_pad[ab0, :ab0] = (iEtE[zz].T @ One).flatten()
#                 L_pad[:ab0, ab0] = iEtEOne[zz].flatten()
#             L_pad[ab0, ab0] = sumiEtEOne
#
#             xerow = x2[zz].flatten()
#             xerow = torch.cat((xerow, torch.zeros(1, device=device)))
#             lagra = torch.linalg.solve(L_pad, xerow)
#
#             while ab0 > 0 and torch.any(lagra[:ab0] > 0):
#                 maxneg = weights[zz] * lagra[:ab0]
#                 iz = torch.argmax(maxneg)
#                 Z[zz[iz]] = False
#                 zz = torch.nonzero(Z, as_tuple=False).view(-1)
#
#                 ab0 = zz.shape[0]
#                 L_pad = torch.zeros((ab0 + 1, ab0 + 1), device=device)
#                 if ab0 > 0:
#                     L = iEtE[zz][:, zz]
#                     L_pad[:ab0, :ab0] = L
#                     L_pad[ab0, :ab0] = (iEtE[zz].T @ One).flatten()
#                     L_pad[:ab0, ab0] = iEtEOne[zz].flatten()
#                 L_pad[ab0, ab0] = sumiEtEOne
#
#                 xerow = x2[zz].flatten()
#                 xerow = torch.cat((xerow, torch.zeros(1, device=device)))
#                 lagra = torch.linalg.solve(L_pad, xerow)
#
#             if zz.numel() > 0:
#                 x2 -= iEtE[:, zz] @ lagra[:ab0].view(-1, 1) + lagra[ab0] * iEtEOne
#
#             iter += 1
#
#     abundance = x2
#     error_vector = torch.matmul(A, abundance).flatten() - sample
#
#     return abundance.flatten().cpu().numpy(), error_vector.cpu().numpy()





# # 訓練函數
# def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
#     for epoch in range(num_epochs):
#         # 訓練階段
#         model.train()
#         train_loss = 0.0
#         for data, target in train_loader:
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
#             output = model(data)
#             loss = criterion(output, target)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#
#         train_loss /= len(train_loader)
#
#         # 驗證階段
#         model.eval()
#         val_loss = 0.0
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for data, target in val_loader:
#                 data, target = data.to(device), target.to(device)
#                 output = model(data)
#                 loss = criterion(output, target)
#                 val_loss += loss.item()
#                 _, predicted = torch.max(output, 1)
#                 total += target.size(0)
#                 correct += (predicted == target).sum().item()
#
#         val_loss /= len(val_loader)
#         val_accuracy = 100 * correct / total
#         scheduler.step(val_loss)
#
#         print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, '
#               f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')