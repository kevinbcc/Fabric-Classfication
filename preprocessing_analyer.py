import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# 把專案根目錄加入 path
sys.path.append(os.getcwd())
from src.preprocessing import Preprocessing

# 路徑設定
original_data_path = "data"
preprocessing_data_path = "preprocessing_data"
os.makedirs(preprocessing_data_path, exist_ok=True)

# 參數
band_num = 50
mode = "SNV"
# mode = "PCA_BANDSELECT"
# mode = "DERIVATIVE"
amplification_factor = 3.0

# 找檔 & 產生儲存路徑
file_paths = sorted(glob.glob(f"./{original_data_path}/*.npy"))
file_names = [os.path.splitext(os.path.basename(p))[0] for p in file_paths]
preprocessing_paths = [os.path.join(preprocessing_data_path, f"{n}.npy") for n in file_names]

# ==== 診斷輸入資料 ====
print("原始檔案：", file_paths)
for p in file_paths:
    arr = np.load(p)
    print(f"  {os.path.basename(p)} shape = {arr.shape}")

# 執行前處理
preprocessor = Preprocessing(n_components=band_num, mode=mode, amplification_factor=amplification_factor)
preprocessor.preprocess(file_paths, preprocessing_paths)

# ==== 診斷輸出資料 ====
print("預處理後檔案：", preprocessing_paths)
for p in preprocessing_paths:
    arr = np.load(p)
    print(f"  {os.path.basename(p)} shape = {arr.shape}")

# 計算平均光譜
orig_spectra = [ np.mean(np.load(p), axis=0) for p in file_paths ]
print(orig_spectra[0].shape)
proc_spectra = [ np.mean(np.load(p), axis=0) for p in preprocessing_paths ]
print(proc_spectra[0].shape)

# 畫圖——原始
plt.figure(figsize=(10,6))
for spec, lbl in zip(orig_spectra, file_names):
    plt.plot(spec, label=lbl)
plt.title("Mean Spectral Reflectance (Original)")
plt.xlabel("Band"); plt.ylabel("Reflectance"); plt.legend()
plt.tight_layout()
plt.savefig("original_spectra.png")
# 畫圖——處理後
plt.figure(figsize=(10,6))
for spec, lbl in zip(proc_spectra, file_names):
    plt.plot(spec, label=lbl)
plt.title(f"Mean Spectral Reflectance ({mode})")
plt.xlabel("Band"); plt.ylabel("Reflectance"); plt.legend()
plt.tight_layout()
plt.savefig(f"processed_spectra_{mode}.png")

# 最後再顯示
plt.show(block=True)
