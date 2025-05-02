# Homework

---

### Question 1
#### Bitmap graphics vs Vector graphics 

1. **可縮放性**
   - **點陣圖**: 放大時會失真  
   - **向量圖**: 縮放時不會失真  
2. **細節**
   - **點陣圖**: 適合展示細節複雜的圖片  
   - **向量圖**: 適合簡單的有明確邊緣的圖片  
3. **文件大小**
   - **點陣圖**: 解析度高時檔案較大  
   - **向量圖**: 通常以數學公式儲存，檔案較小  

---

### Question 2
#### One twenty-fifth of a second? 其他物理意義

1. **視覺皮層影格速率** 低於 48 影格/秒會導致眼睛疲勞，因此影格速率從 24 提升到 25 以上，每秒 25 張影像是最低可接受的影片流暢度。  
2. **PAL / NTSC / SECAM 電視規格**  
   - PAL/SECAM: 每秒 **25 格**（交流電頻率 50Hz）  
   - NTSC: 每秒 **30 格**（交流電頻率 60Hz）  

---

### Question 3
#### Vector vs Raster Displays? 失真?

1. **向量圖 (Vector)** 放大後仍然是線性，不會失真。  
2. **點陣圖 (Raster)** 放大後會有方格狀像素 (pixelation)。  
3. **儲存與編輯**：
   - Raster 儲存檔案較大，但能表現較細膩的顏色細節。  
   - Raster 易於使用軟體編輯，但不適合用於印刷。  

**比較圖：**  
![比較圖](a.png)

---

### Question 4
#### 什麼格式會失真、不會失真？類別？

1. **點陣圖 (Raster) 放大會失真，格式：**  
   - **有損壓縮**: `JPEG`  
   - **無損壓縮**: `TIFF`、`PNG`、`GIF`、`WEBP`  
   - **無壓縮**: `BMP`、`RAW`  

2. **向量圖 (Vector) 放大不失真，格式：**  
   - `SVG`

---

### Question 5
#### 這句話你可以得到多少資訊？

- 音頻或視訊可以是 **raw bitstream**，但通常會使用 **container format** 或 **audio data format** 來儲存。  
- 訊號可能會經過處理後儲存，這可能會造成 **失真**。  

---

### Question 6
#### 串流平台上也想要達到 Lossless，有哪些新技術？

- **無損編解碼技術 (Lossless Codec)**：
  - `FLAC`、`ALAC`、`Hi-Res Audio`
  - `MQA`、`WAV`、`DSD`（Direct Stream Digital）、`AIFF`
- **DSD（Direct Stream Digital）**：
  - Sony 和 Philips 專利技術，利用 **脈衝密度調變 (PDM, Pulse-Density Modulation)**  
  - 直接將模擬音樂訊號波形轉換為數位訊號，由於高採樣率，檔案非常大。  

---

### Question 7
#### 有最新的技術嗎？

1. **AV2**
   - `AV1` 的後續編碼技術，目前規範尚未確定。  

2. **H.266 / VVC**
   - **高效率視訊編碼 (HEVC / H.265)** 的後續技術。  
   - H.266 相較於 H.265 **壓縮效率提高 49%**。  

3. **LCEVC（低複雜度增強視訊編碼）**
   - 結合 **AVC、HEVC 等編碼技術**，提升視訊品質，降低壓縮負擔。  

4. **EVC（Essential Video Coding）**
   - 於 2020 年推出，主打更高壓縮效能。  

**相關技術示意圖：**  
![MPEG 規範](mpeg.png)

5. **FMP4（Fragmented MP4）**
   - 結合 **TS 與 MP4** 格式優點，檔案損毀時仍可讀取前後影像。  
   - 相容性高，支援多種播放器與上傳平台。  
