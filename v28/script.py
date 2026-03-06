import subprocess
import time
import itertools
import os
import pandas as pd
import glob
from datetime import datetime

# ================= 設定區 =================
dic = {
    'batch_size': [1000],
    'epoch': [500],
    'layer': [1, 2, 3, 4, 5],
    'hidden': [16, 32, 64],
    'data_version': [48],
    'lr': [0.003, 0.001, 0.03, 0.01],
    'column': ['acceleration_X,acceleration_Y,acceleration_Z', 
               'gyroscope_X,gyroscope_Y,gyroscope_Z', 
               'acceleration_X,acceleration_Y,acceleration_Z,gyroscope_X,gyroscope_Y,gyroscope_Z'],
    'folds': [1, 2, 3, 4, 5],
}

MAX_CONCURRENT_JOBS = 10  # 訓練時的併發數
TRAIN_SCRIPT = "train_GRU_v28.py"
TEST_SCRIPT = "test_GRU_v28.py"
# =========================================

def get_combinations(params):
    keys = list(params.keys())
    values = list(params.values())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

def run_phase(phase_name, script_name, combinations, max_jobs):
    print(f"\n=== 開始執行階段: {phase_name} ===")
    total_jobs = len(combinations)
    running_processes = []
    
    for i, p in enumerate(combinations):
        # 準備指令
        cmd = [
            'python3', script_name,
            f'--batch_size={p["batch_size"]}',
            f'--epoch={p["epoch"]}',
            f'--layer={p["layer"]}',
            f'--hidden={p["hidden"]}',
            f'--data_version={p["data_version"]}',
            f'--lr={p["lr"]}',
            f'--column={p["column"]}',
            f'--fold={p["folds"]}'
        ]
        
        # 啟動程序
        # 注意：train.py 和 test.py 內部都有 "Skip if exists" 的檢查，所以這裡可以直接呼叫
        # 為了避免畫面太亂，我們將 stdout 導向 null，除非你想看 log
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        running_processes.append(proc)
        
        # 進度顯示
        if i % 10 == 0:
            print(f"[{phase_name}] 進度: {i}/{total_jobs} (Running: {len(running_processes)})")

        # 控管併發數
        while len(running_processes) >= max_jobs:
            running_processes = [p for p in running_processes if p.poll() is None]
            if len(running_processes) >= max_jobs:
                time.sleep(1)

    # 等待最後一批完成
    for p in running_processes:
        p.wait()
    print(f"=== {phase_name} 階段完成 ===\n")

def collect_results():
    print("=== 正在彙整測試報告 ===")
    # 搜尋所有 models 資料夾底下的 test_result.csv
    result_files = glob.glob("models/*/test_result.csv")
    
    if not result_files:
        print("尚未產生任何測試結果。")
        return

    all_dfs = []
    for f in result_files:
        try:
            df = pd.read_csv(f)
            all_dfs.append(df)
        except: pass
    
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df = final_df.sort_values(by="f1", ascending=False)
        
        out_name = "Final_Report_540.csv"
        final_df.to_csv(out_name, index=False)
        print(f"報告已產出: {out_name}")
        print("Top 3 Models:")
        print(final_df.head(3))
    else:
        print("無法合併結果。")

def main():
    start_time = datetime.now()
    combinations = list(get_combinations(dic))
    
    # 階段 1: 訓練
    # 訓練很吃 GPU/CPU，併發數依照您的設定 (10)
    run_phase("Training", TRAIN_SCRIPT, combinations, max_jobs=MAX_CONCURRENT_JOBS)
    
    # 階段 2: 測試
    # 測試比較快，可以稍微開多一點併發，或者維持一樣
    run_phase("Testing", TEST_SCRIPT, combinations, max_jobs=MAX_CONCURRENT_JOBS)
    
    # 階段 3: 彙整
    collect_results()

    end_time = datetime.now()
    print(f"總耗時: {end_time - start_time}")

if __name__ == "__main__":
    main()