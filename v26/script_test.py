import subprocess
import time
import itertools
import os
import glob
import pandas as pd
from datetime import datetime

# ================= 設定區 =================
dic = {
    'layer': [2, 3, 4],
    'hidden': [16, 32, 64],
    'data_version': [47],
    'lr': [0.003, 0.001, 0.03, 0.01],
    'columns': ['acceleration_Y'],
    'folds': [1, 2, 3, 4, 5],
}

# 測試程式名稱 (請確保檔名正確)
TEST_SCRIPT = "test_GRU_v26.py" 
MODEL_ROOT = "./models"
TEST_LOG_DIR = "./test_log" # 統一輸出目錄
MAX_CONCURRENT_JOBS = 5
# =========================================

def get_combinations(params):
    keys = list(params.keys())
    values = list(params.values())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

def find_latest_model_dir(dv, fold, layer, hidden):
    pattern = f"dv{dv}_fold{fold}_L{layer}_H{hidden}_*"
    search_path = os.path.join(MODEL_ROOT, pattern)
    candidates = glob.glob(search_path)
    if not candidates:
        return None
    return sorted(candidates)[-1]

def aggregate_results():
    """
    掃描 TEST_LOG_DIR 下所有的 *_summary.csv，
    合併並依照 mAP 排序，產出 test_all.csv
    """
    print("正在統整所有測試結果...")
    all_files = glob.glob(os.path.join(TEST_LOG_DIR, "*_summary.csv"))
    
    # 排除之前的 test_all.csv 以免重複讀取
    all_files = [f for f in all_files if "test_all.csv" not in f]
    
    if not all_files:
        print("[Warning] 沒有找到任何 summary csv 檔案。")
        return

    df_list = []
    for f in all_files:
        try:
            df_list.append(pd.read_csv(f))
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        # 依照 mAP 由大到小排序
        final_df = final_df.sort_values(by="mAP", ascending=False)
        
        out_path = os.path.join(TEST_LOG_DIR, "test_all.csv")
        final_df.to_csv(out_path, index=False)
        print(f"統整完成！結果已儲存至: {out_path}")
        print(f"Top 5 Models:\n{final_df.head(5)[['model_folder', 'ckpt', 'mAP']]}")
    else:
        print("沒有讀取到有效數據。")

def main():
    start_time = datetime.now()
    print(f"開始測試流程: {start_time}")
    
    # 確保輸出目錄存在，避免 script 報錯
    os.makedirs(TEST_LOG_DIR, exist_ok=True)
    
    combinations = list(get_combinations(dic))
    print(f"預計掃描 {len(combinations)} 組參數配置...")

    running_processes = []
    
    for i, p in enumerate(combinations):
        model_dir = find_latest_model_dir(
            dv=p['data_version'],
            fold=p['folds'],
            layer=p['layer'],
            hidden=p['hidden']
        )
        
        if not model_dir:
            # print(f"[{i+1}] [Skip] Not found: L={p['layer']}, H={p['hidden']}, F={p['folds']}")
            continue

        # 檢查該模型是否已經跑過 (檢查對應的 summary 是否存在)
        model_name = os.path.basename(model_dir)
        summary_file = os.path.join(TEST_LOG_DIR, f"{model_name}_summary.csv")
        if os.path.exists(summary_file):
            print(f"[{i+1}] [Skip] 已存在: {model_name}")
            continue

        cmd = [
            'python3', TEST_SCRIPT,
            f'--data_version={p["data_version"]}',
            f'--column={p["columns"]}',
            f'--layer={p["layer"]}',
            f'--hidden={p["hidden"]}',
            f'--lr={p["lr"]}',
            f'--fold={p["folds"]}',
            f'--model_dir={model_dir}',
            f'--out_dir={TEST_LOG_DIR}'
        ]
        
        print(f"[{i+1}/{len(combinations)}] 啟動測試: {model_name}")
        
        # 啟動並隱藏 stdout 以免洗版 (若要除錯可拿掉 stdout=subprocess.DEVNULL)
        proc = subprocess.Popen(cmd) 
        running_processes.append(proc)

        while len(running_processes) >= MAX_CONCURRENT_JOBS:
            running_processes = [p for p in running_processes if p.poll() is None]
            if len(running_processes) >= MAX_CONCURRENT_JOBS:
                time.sleep(1)

    # 等待剩餘任務
    for p in running_processes:
        p.wait()

    end_time = datetime.now()
    print(f"所有測試任務結束。耗時: {end_time - start_time}")
    
    # === 最後執行統整 ===
    aggregate_results()

if __name__ == "__main__":
    main()