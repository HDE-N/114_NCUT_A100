import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

# === 設定區 ===
# 原始資料夾名稱 (請確認這裡與您的資料夾名稱一致)
SRC_DIR_NAME = "train_data_pre" 
OUT_DIR = Path("K_Fold")             
K_FOLDS = 5
SEED = 42

def main():
    # 1. 設定路徑 (修復 UnboundLocalError)
    # 先建立一個 Path 物件
    source_path = Path(SRC_DIR_NAME)

    # 檢查路徑是否存在，若不存在則找上一層
    if not source_path.exists():
        # 嘗試往上一層找
        alt_path = Path(f"../{SRC_DIR_NAME}")
        if alt_path.exists():
            source_path = alt_path
        else:
            # 都找不到就報錯
            raise FileNotFoundError(f"找不到資料夾: '{SRC_DIR_NAME}' 或 '{alt_path}'，請確認路徑。")

    print(f"正在掃描資料夾: {source_path} ...")
    
    file_list = []
    # 使用確認過存在的 source_path 來掃描
    for p in source_path.rglob("*.csv"):
        filename = p.name
        
        # === 針對您的新格式解析 ===
        # 檔名: 1754550957366_user_B_esp32_v8_notTired_pre_s000000.csv
        
        # 1. 抓取 Group (時間戳記): 用 "_" 切割，取第 0 個
        group_id = filename.split("_")[0]
        
        # 2. 抓取 Label (標籤): 判斷檔名內是否包含 "notTired"
        if "notTired" in filename:
            label = "notTired"
        else:
            label = "Tired" 

        file_list.append({
            "path": p,
            "label": label,
            "group": group_id
        })

    if not file_list:
        raise ValueError("找不到任何 csv 檔案，請檢查路徑或副檔名。")

    df = pd.DataFrame(file_list)
    
    # 隨機打亂列表 (Shuffle)，但 GroupKFold 會負責保持同一組在一起
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"共找到 {len(df)} 個片段，來自 {df['group'].nunique()} 個原始錄製時段 (Groups)。")
    # 顯示前幾個 Group ID 確保抓對了
    print(f"Group ID 範例: {df['group'].unique()[:3]}")  

    # 2. 開始 Group K-Fold 切分
    gkf = GroupKFold(n_splits=K_FOLDS)
    
    # 清空並重建輸出目錄
    if OUT_DIR.exists(): shutil.rmtree(OUT_DIR)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df["group"]), start=1):
        print(f"\n=== 處理 Fold {fold}/{K_FOLDS} ===")
        
        # 定義這個 Fold 的資料集
        splits = {
            "train": df.iloc[train_idx],
            "test": df.iloc[val_idx]
        }

        # 複製檔案
        for split_name, split_df in splits.items():
            for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Copying {split_name}", leave=False):
                # 目標路徑: K_Fold/fold_1/train/Tired/xxx.csv
                target_dir = OUT_DIR / f"fold_{fold}" / split_name / row["label"]
                target_dir.mkdir(parents=True, exist_ok=True)
                
                shutil.copy2(row["path"], target_dir / row["path"].name)

    print(f"\n✅ K-Fold 完成！輸出資料夾: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()