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
orig_spectra = [np.mean(np.load(p), axis=0) for p in file_paths]
proc_spectra = [np.mean(np.load(p), axis=0) for p in preprocessing_paths]

# === 分群繪圖設定 ===
groups = {
    "MVS": [],
    "OE": [],
    "Compact": [],
}
group_names = {
    "MVS": [],
    "OE": [],
    "Compact": [],
}
proc_groups = {
    "MVS": [],
    "OE": [],
    "Compact": [],
}

# 根據檔名分類（請根據你的檔名實際格式調整）
for orig_spec, proc_spec, name in zip(orig_spectra, proc_spectra, file_names):
    lname = name.lower()
    if "mvs" in lname:
        groups["MVS"].append(orig_spec)
        group_names["MVS"].append(name)
        proc_groups["MVS"].append(proc_spec)
    elif "oe" in lname:
        groups["OE"].append(orig_spec)
        group_names["OE"].append(name)
        proc_groups["OE"].append(proc_spec)
    elif "compact" in lname or "環錠" in lname:
        groups["Compact"].append(orig_spec)
        group_names["Compact"].append(name)
        proc_groups["Compact"].append(proc_spec)
    else:
        print(f"⚠️ 無法分類的檔案：{name}")

# === 畫圖：原始光譜 ===
for g in groups:
    plt.figure(figsize=(10,6))
    for spec, name in zip(groups[g], group_names[g]):
        plt.plot(spec, label=name)
    plt.title(f"{g} - Mean Spectral Reflectance (Original)")
    plt.xlabel("Band")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{g.lower()}_original_spectra.png")

# === 畫圖：處理後光譜 ===
for g in proc_groups:
    plt.figure(figsize=(10,6))
    for spec, name in zip(proc_groups[g], group_names[g]):
        plt.plot(spec, label=name)
    plt.title(f"{g} - Mean Spectral Reflectance ({mode})")
    plt.xlabel("Band")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{g.lower()}_processed_spectra_{mode}.png")

# 顯示所有圖表
plt.show(block=True)
