from src.train import train_model,select_train_test_samples
from src.utils import save_predictions_by_source,calculate_rmse_by_source, print_avg_predicted_ratios
from src.module import SimpleCNN_MLP,MLP2, ImprovedSimpleCNN_MLP
from src.preprocessing import Preprocessing
import torch
import torch.optim as optim
import torch.nn as nn
import glob
import os

#前處理參數
orignal_data_path = "data"
preprocessin_data_path = "preprocessing_data"
block_size =20
# band_num =22 #best
# band_num =15
band_num = 100
# mode = "PCA_BANDSELECT_AMPLIFY_ERROR"
mode = "PCA_BANDSELECT"
# mode  = "SNV"
# mode = "DERIVATIVE"
# mode = "NMF_AMPLIFY_ERROR"
# mode = "NMF" # 這個會慢很多
# mode = "NMF_AMPLIFY_ERROR"
amplification_factor = 2.0

#訓練參數
epochs = 20
# epochs =500
proportion_mode=(0.1, "train")
Learning_Rate=0.01

# ======= 加載並預處理數據 =======
file_paths = sorted(glob.glob(f'./{orignal_data_path}/*.npy'))
preprocessing_paths = [os.path.join(f'./{preprocessin_data_path}', os.path.basename(path)) for path in file_paths]
# PCA 降維
preprocessor_pca = Preprocessing( n_components=band_num, mode=mode,amplification_factor=amplification_factor)
preprocessor_pca.preprocess(file_paths,preprocessing_paths)


g_mapping = {
    f"./{preprocessin_data_path}/COMPACT100C_RT_roi_25x500.npy": (1.0, 0.0),
    f"./{preprocessin_data_path}/COMPACT100P_RT_roi_25x500.npy": (0.0, 1.0),
    f"./{preprocessin_data_path}/COMPACT5050_RT_roi_25x500.npy": (0.5, 0.5),
    f"./{preprocessin_data_path}/MVS100C_RT_roi_25x500.npy": (1.0, 0.0),
    f"./{preprocessin_data_path}/MVS100P_RT_roi_25x500.npy": (0.0, 1.0),
    f"./{preprocessin_data_path}/MVS5050_RT_roi_25x500.npy": (0.5, 0.5),
    f"./{preprocessin_data_path}/OE100C_RT_roi_25x500.npy": (1.0, 0.0),
    f"./{preprocessin_data_path}/OE100P_RT_roi_25x500.npy": (0.0, 1.0),
    f"./{preprocessin_data_path}/OE5050_RT_roi_25x500.npy": (0.5, 0.5),
}



# 將 "preprocessing_data" 資料夾中的 `.npy` 檔案處理成區塊，區塊大小為 10
# blocks = process_npy_to_blocks(folder=preprocessin_data_path, block_size=block_size)


train_X, train_Y, test_X, test_Y, test_sources = select_train_test_samples(
    # blocks=blocks,
    proportion_mode=proportion_mode,
    g_mapping=g_mapping
)

# ==================================
# PCA_BANDSELECT_amplify_error  n_components= 22  block_size=10,test_sample_size=100 epochs = 300
# RMSE在0.1
# ======= 設定模型參數 =======
output_dim = 2  # 輸出維度固定
model = SimpleCNN_MLP(input_channels=1, input_dim=band_num, hidden_dim=32, output_dim=output_dim) #best
# model = MLP2(input_dim=band_num, hidden_dim1=band_num*1,hidden_dim2=band_num*3,hidden_dim3=band_num*1,output_dim=output_dim)

# ======= 設定損失函數與優化器 =======
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=Learning_Rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)# 設定學習率調度器：根據 loss 變化調整學習率

# 訓練模型並獲取回傳的參數
model, best_loss, final_lr = train_model(
    model,
    train_X,
    train_Y,
    epochs=50,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    batch_size=64  # 可以自由調整 batch size
)

torch.save(model.state_dict(), "./weight/SimpleCNN_MLP_final.pt")
# 輸出訓練結果
print(f'Best Loss: {best_loss:.4f}, Final Learning Rate: {final_lr:.6f}')

# ======= 測試模型 =======
with torch.no_grad():
    pred_Y = model(test_X)

# 儲存預測結果並分類
results_by_source = save_predictions_by_source(test_sources, pred_Y, test_Y)

# 計算並顯示/儲存 RMSE
rmse_by_source = calculate_rmse_by_source(results_by_source, save_csv_path="result/sourcewise_rmse.csv")

# 顯示每一種紗種的真實成分平均比例（Cotton / Poly）
print_avg_predicted_ratios(results_by_source)




