import numpy as np
import torch
import glob
import os
from src.utils import save_predictions_by_source, calculate_rmse_by_source, print_avg_predicted_ratios
from src.preprocessing import Preprocessing
from src.train import select_train_test_samples
from src.module import FCLS  # 確保這是你定義的 FCLS 函式

# 前處理參數
# ======= 前處理參數 =======
orignal_data_path = "data"
preprocessin_data_path = "preprocessing_data"
block_size = 20
band_num = 20
mode = "PCA_BANDSELECT"
# mode = "SNV"
amplification_factor = 2.0

# ======= 載入與預處理資料 =======
file_paths = sorted(glob.glob(f'./{orignal_data_path}/*.npy'))
preprocessing_paths = [os.path.join(f'./{preprocessin_data_path}', os.path.basename(path)) for path in file_paths]

preprocessor_pca = Preprocessing(n_components=band_num, mode=mode, amplification_factor=amplification_factor)
preprocessor_pca.preprocess(file_paths, preprocessing_paths)

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

proportion_mode = (0.1, "train")

train_X, train_Y, test_X, test_Y, test_sources = select_train_test_samples(
    proportion_mode=proportion_mode,
    g_mapping=g_mapping
)

# ======= 建立端元矩陣 M（你需要自己提供這個）=======
# 假設你有兩個材料的 endmember 向量，經過相同 PCA 處理後得到：
# M 的 shape 為 [bands, 2]
# 請根據你的應用場景從原始樣本中取出 pure pixel 並套用 PCA

# 定義每個測試來源對應的 cotton 和 polyester 純樣本檔案
source_to_endmember_paths = ({
    "COMPACT100C_RT_roi_25x500.npy": ("COMPACT100C_RT_roi_25x500.npy", "COMPACT100P_RT_roi_25x500.npy"),
    "COMPACT100P_RT_roi_25x500.npy": ("COMPACT100C_RT_roi_25x500.npy", "COMPACT100P_RT_roi_25x500.npy"),
    "COMPACT5050_RT_roi_25x500.npy": ("COMPACT100C_RT_roi_25x500.npy", "COMPACT100P_RT_roi_25x500.npy"),
    "MVS100C_RT_roi_25x500.npy": ("MVS100C_RT_roi_25x500.npy", "MVS100P_RT_roi_25x500.npy"),
    "MVS100P_RT_roi_25x500.npy": ("MVS100C_RT_roi_25x500.npy", "MVS100P_RT_roi_25x500.npy"),
    "MVS5050_RT_roi_25x500.npy": ("MVS100C_RT_roi_25x500.npy", "MVS100P_RT_roi_25x500.npy"),
    "OE100C_RT_roi_25x500.npy": ("OE100C_RT_roi_25x500.npy", "OE100P_RT_roi_25x500.npy"),
    "OE100P_RT_roi_25x500.npy": ("OE100C_RT_roi_25x500.npy", "OE100P_RT_roi_25x500.npy"),
    "OE5050_RT_roi_25x500.npy": ("OE100C_RT_roi_25x500.npy", "OE100P_RT_roi_25x500.npy"),
})

# # 定義每個測試來源對應的 cotton 和 polyester 純樣本檔案
# source_to_endmember_paths = ({
#     "COMPACT100C_RT_roi_spectra.npy": ("COMPACT100C_RT_roi_spectra.npy", "COMPACT100P_RT_roi_spectra.npy"),
#     "COMPACT100P_RT_roi_spectra.npy": ("COMPACT100C_RT_roi_spectra.npy", "COMPACT100P_RT_roi_spectra.npy"),
#     "COMPACT5050_RT_roi_spectra.npy": ("COMPACT100C_RT_roi_spectra.npy", "COMPACT100P_RT_roi_spectra.npy"),
#     "MVS100C_RT_roi_spectra.npy": ("MVS100C_RT_roi_spectra.npy", "MVS100P_RT_roi_spectra.npy"),
#     "MVS100P_RT_roi_spectra.npy": ("MVS100C_RT_roi_spectra.npy", "MVS100P_RT_roi_spectra.npy"),
#     "MVS5050_RT_roi_spectra.npy": ("MVS100C_RT_roi_spectra.npy", "MVS100P_RT_roi_spectra.npy"),
#     "OE100C_RT_roi_spectra.npy": ("OE100C_RT_roi_spectra.npy", "OE100P_RT_roi_spectra.npy"),
#     "OE100P_RT_roi_spectra.npy": ("OE100C_RT_roi_spectra.npy", "OE100P_RT_roi_spectra.npy"),
#     "OE5050_RT_roi_spectra.npy": ("OE100C_RT_roi_spectra.npy", "OE100P_RT_roi_spectra.npy"),
# })


test_X_np = test_X.squeeze(1).cpu().numpy()
test_X_np = test_X_np  # 只保留第 20–200 波段

pred_Y = []

for i, r1 in enumerate(test_X_np):
    source_filename = os.path.basename(test_sources[i])
    cotton_file, poly_file = source_to_endmember_paths[source_filename]

    cotton_path = os.path.join(preprocessin_data_path, cotton_file)
    poly_path = os.path.join(preprocessin_data_path, poly_file)

    #確保 shape 是 [num_pixel, bands]
    m1 = np.load(cotton_path).reshape(-1, r1.shape[-1]).mean(axis=0)
    m2 = np.load(poly_path).reshape(-1, r1.shape[-1]).mean(axis=0)

    # def load_mean_band_range(path, start=10, end=225):
    #     arr = np.load(path)
    #     if arr.ndim == 1:
    #         return arr[start:end]
    #     elif arr.ndim == 2:
    #         return arr[:, start:end].mean(axis=0)
    #     else:
    #         raise ValueError(f"Unsupported array shape: {arr.shape}")


    # m1 = load_mean_band_range(cotton_path)
    # m2 = load_mean_band_range(poly_path)
    M = np.stack([m1, m2], axis=1)  # shape: [bands, 2]

    delta = 1 / (10 * np.linalg.norm(M, ord='fro'))
    abund, _ = FCLS(M, r1, delta)
    pred_Y.append(abund)

pred_Y = torch.from_numpy(np.array(pred_Y, dtype=np.float32))


# ======= 儲存與評估 =======
results_by_source = save_predictions_by_source(test_sources, pred_Y, test_Y)
rmse_by_source = calculate_rmse_by_source(results_by_source, save_csv_path="result/sourcewise_rmse.csv")

# 顯示每一種紗種的真實成分平均比例（Cotton / Poly）
print_avg_predicted_ratios(results_by_source)