# 新訓練數據

## 訓練集: 1150302 - 1150307 (data) 其中最小的5筆轉至測試集

## 測試集: 1150309 - 1150410 (test_data_old)

### Other有部分是自主蒐集

### 有做增強數據，可以嘗試第三類別(Other)

### 161 Hz BNO086

---

資料夾內容

data             原始 notTired 與 Tired 數據

aug_data         data 經小增強後的數據

other            data 經大增強後的數據

other_app        各種亂揮數據


test_data_old    原始測試數據

test_data_other  test_data_old經大增強後的數據，作為Other類測試使用


dm_data          Diffusion Models增強數據 notTired & Tired 各1000筆

---

### bak1 增強參數

NOISE_STD = 2.0、SCALE_FACTOR = 1.2

原始數據小增強，NOISE_STD = 0.3、SCALE_FACTOR = 1.1

加入dm_data所有數據

---

### bak2 增強參數

參數與bak1一致

惟測試集不同

---

### bak3 增強參數

NOISE_STD = 2.0、SCALE_FACTOR = 1.2

原始數據小增強，NOISE_STD = 0.3、SCALE_FACTOR = 1.1

僅加入 dm_data notTired 數據

測試集為橫嶺山&合歡山數據

