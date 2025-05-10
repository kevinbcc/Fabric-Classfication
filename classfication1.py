from src.train import train_model, select_train_test_samples_classification
from src.utils import save_predictions_by_source_classification, calculate_rmse_by_source, print_avg_predicted_ratios
from src.module import SimpleCNN_MLP, ImprovedCNN1DClassifier
from src.preprocessing import Preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import torch
import torch.optim as optim
import torch.nn as nn
import glob
import os
import matplotlib.pyplot as plt



# ==== 前處理參數設定 ====
orignal_data_path = "data"
preprocessin_data_path = "preprocessing_data"
block_size = 20
band_num = 224
# mode = "PCA_BANDSELECT"
mode = "SNV"
amplification_factor = 2.0

# ==== 訓練參數 ====
epochs = 100
Learning_Rate = 0.01

# ==== 數據前處理 ====
file_paths = sorted(glob.glob(f'./{orignal_data_path}/*.npy'))
preprocessing_paths = [os.path.join(f'./{preprocessin_data_path}', os.path.basename(path)) for path in file_paths]
preprocessor_pca = Preprocessing(n_components=band_num, mode=mode, amplification_factor=amplification_factor)
preprocessor_pca.preprocess(file_paths, preprocessing_paths)

# ==== 分類標籤對應表（3 類）====
g_mapping = {
    "./preprocessing_data/COMPACT100C_RT_roi_25x500.npy": 0,
    "./preprocessing_data/COMPACT5050_RT_roi_25x500.npy": 0,
    "./preprocessing_data/COMPACT100P_RT_roi_25x500.npy": 0,
    "./preprocessing_data/MVS100C_RT_roi_25x500.npy": 1,
    "./preprocessing_data/MVS5050_RT_roi_25x500.npy": 1,
    "./preprocessing_data/MVS100P_RT_roi_25x500.npy": 1,
    "./preprocessing_data/OE100C_RT_roi_25x500.npy": 2,
    "./preprocessing_data/OE5050_RT_roi_25x500.npy": 2,
    "./preprocessing_data/OE100P_RT_roi_25x500.npy": 2,
}


proportion = 0.7  # 表示訓練資料比例 70%
train_X, train_Y, test_X, test_Y, test_sources = select_train_test_samples_classification(
    proportion=proportion,
    g_mapping=g_mapping
)
print(train_X.shape, test_X.shape)

# ==== 模型定義（3 類分類輸出）====
output_dim = 3  # 改為 3 類
model = ImprovedCNN1DClassifier(input_channels=1, input_dim=band_num, hidden_dim=128, output_dim=output_dim)

# ==== 分類損失函數與優化器 ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=Learning_Rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# ==== 訓練模型 ====
model, best_loss, final_lr = train_model(
    model,
    train_X,
    train_Y.long(),  # CrossEntropyLoss 需要 long tensor 類別標籤
    epochs=epochs,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    batch_size=32
)

torch.save(model.state_dict(), "./weight/SimpleCNN_MLP_final.pt")
print(f'Best Loss: {best_loss:.4f}, Final Learning Rate: {final_lr:.6f}')

# ==== 測試模型 ====
with torch.no_grad():
    pred_logits = model(test_X)  # [batch, 3]
    pred_classes = torch.argmax(pred_logits, dim=1)  # 預測的類別 [batch]



# ==== 計算分類準確率 ====
correct = (pred_classes == test_Y).sum().item()
total = test_Y.size(0)
accuracy = correct / total * 100

print(f"Test Accuracy: {accuracy:.2f}%")

# ==== 儲存預測結果 ====
results_by_source = save_predictions_by_source_classification(test_sources, pred_classes, test_Y)



# 假設你已經有這些：
# pred_classes: 預測類別 (tensor or list)
# test_Y: 實際類別 (tensor or list)

# 將 tensor 轉為 numpy array（如果還不是的話）
y_pred = pred_classes.cpu().numpy()
y_true = test_Y.cpu().numpy()

# 計算混淆矩陣
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
print(classification_report(y_true, y_pred, target_names=["Compact","MVS","OE"]))

# 顯示混淆矩陣
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Compact", "MVS", "OE"])
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix")
plt.show()
