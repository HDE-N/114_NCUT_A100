"""
精簡版：
1. 移除數值 Clip (保留原始精度)。
2. 移除分類資料夾 (所有檔案輸出至同一目錄)。
3. 僅執行滑動視窗切割 (Window=300, Stride=150)。
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ======== 參數設定 ========
IN_DIR = Path("data")
OUT_DIR = Path("train_data_pre")
WINDOW = 300
STRIDE = 150

# ======== 主程式 ========
def main():
    # 建立輸出資料夾 (不分類，直接建立一層)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(IN_DIR.glob("*.csv"))
    if not csv_files:
        print(f"[WARN] {IN_DIR} 內無 CSV 檔案")
        return

    for csv_path in tqdm(csv_files, desc="Processing"):
        # 讀取檔案
        df = pd.read_csv(csv_path)
        
        # 僅保留數值欄位 (避免含有非數值的文字欄位報錯)
        df = df.select_dtypes(include=['number'])

        n_rows = len(df)
        base_name = csv_path.stem

        # 滑動視窗切割
        # range(start, stop, step) 確保視窗不會超出範圍
        for start in range(0, n_rows - WINDOW + 1, STRIDE):
            end = start + WINDOW
            window_df = df.iloc[start:end]
            
            # 存檔：保留原始檔名並加上起始索引
            out_name = f"{base_name}_s{str(start).zfill(6)}.csv"
            window_df.to_csv(OUT_DIR / out_name, index=False)

    print(f"完成！檔案已輸出至 {OUT_DIR}/")

if __name__ == "__main__":
    main()