import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

# ======== 參數設定 (依要求調整) ========
IN_DIR = Path("data")           # 原始資料存放處
OUT_ROOT = Path("K_Fold")       # 最終輸出目錄
WINDOW = 960
STRIDE = 480
K_FOLDS = 5
SEED = 42

def main():
    # 1. 取得原始檔案列表並標記 Group (時間戳記)
    raw_files = sorted(list(IN_DIR.glob("*.csv")))
    if not raw_files:
        print(f"[Error] {IN_DIR} 內找不到 CSV 檔案"); return

    file_meta = []
    for p in raw_files:
        # 依照您的邏輯：取第一個底線前的內容作為 Group ID (如時間戳)
        group_id = p.name.split("_")[0]
        label = "notTired" if "notTired" in p.name else "Tired"
        file_meta.append({"path": p, "label": label, "group": group_id})

    df_meta = pd.DataFrame(file_meta)
    print(f"原始檔案：{len(df_meta)} 個，包含 {df_meta['group'].nunique()} 個獨立 Group。")

    # 2. 準備交叉驗證 (GroupKFold)
    gkf = GroupKFold(n_splits=K_FOLDS)
    
    # 清空輸出目錄
    if OUT_ROOT.exists(): shutil.rmtree(OUT_ROOT)

    # 3. 開始執行 Fold 循環
    # 這裡 split(數據, 標籤, 分組依據)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_meta, groups=df_meta["group"]), start=1):
        print(f"\n=== 處理 Fold {fold}/{K_FOLDS} ===")
        
        # 定義這個 Fold 的任務內容
        job_config = {
            "train": df_meta.iloc[train_idx],
            "test":  df_meta.iloc[val_idx]
        }

        for split_name, split_df in job_config.items():
            for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Processing {split_name}", leave=False):
                # 讀取並切割數據
                df_raw = pd.read_csv(row["path"]).select_dtypes(include=['number'])
                base_name = row["path"].stem
                
                # 滑動視窗切割與儲存
                for start in range(0, len(df_raw) - WINDOW + 1, STRIDE):
                    chunk = df_raw.iloc[start : start + WINDOW]
                    out_filename = f"{base_name}_s{start:06d}.csv"
                    
                    # 建立格式化目錄：K_Fold/fold_1/train/Tired/xxx.csv
                    target_dir = OUT_ROOT / f"fold_{fold}" / split_name / row["label"]
                    target_dir.mkdir(parents=True, exist_ok=True)
                    
                    chunk.to_csv(target_dir / out_filename, index=False)

    print(f"\n✅ 任務完成！輸出資料夾: {OUT_ROOT.resolve()}")

if __name__ == "__main__":
    main()