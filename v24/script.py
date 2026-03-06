import subprocess
import time
import itertools
import os
from datetime import datetime

# ================= 設定區 =================
dic = {
    'batch_size': [1500],
    'epoch': [500],
    'layer': [2, 3, 4],
    'hidden': [16, 32, 64],
    'data_version': [46],
    'lr': [0.003, 0.001, 0.03, 0.01],
    'columns': ['acceleration_Y'],
    'folds': [1, 2, 3, 4, 5],
}

# [重要] 同時執行最大任務數
# 您希望提供 10核/程式，並同時跑 4 個，總共約佔用 40~50 核心 (4 x 10核運算 + workers)
# 這在 100 核心的機器上非常安全且餘裕。
MAX_CONCURRENT_JOBS = 4

PYTHON_SCRIPT = "train_GRU_v24.py" 
# =========================================

def get_combinations(params):
    keys = list(params.keys())
    values = list(params.values())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

def main():
    start_time = datetime.now()
    print(f"開始時間: {start_time}")
    print(f"最大平行任務數: {MAX_CONCURRENT_JOBS} (每任務 10 核心)")

    combinations = list(get_combinations(dic))
    total_jobs = len(combinations)
    print(f"總共需要執行 {total_jobs} 個訓練任務。")

    running_processes = []
    
    for i, p in enumerate(combinations):
        # Resume 機制：檢查 last_model.pth 是否存在
        model_sub_dir = f"fold{p['folds']}_layer_{p['layer']}_hidden_{p['hidden']}_lr_{p['lr']}_dv_{p['data_version']}_col_{p['columns']}"
        check_path = os.path.join('./model', model_sub_dir, 'last_model.pth')
        
        if os.path.exists(check_path):
            print(f"[{i+1}/{total_jobs}] [Skip] 已存在: {model_sub_dir}")
            continue

        cmd = [
            'python3', PYTHON_SCRIPT,
            f'--batch_size={p["batch_size"]}',
            f'--epoch={p["epoch"]}',
            f'--layer={p["layer"]}',
            f'--hidden={p["hidden"]}',
            f'--data_version={p["data_version"]}',
            f'--lr={p["lr"]}',
            f'--column={p["columns"]}',
            f'--fold={p["folds"]}'
        ]
        
        print(f"[{i+1}/{total_jobs}] 啟動: {model_sub_dir}")
        
        # 啟動並隱藏輸出 (log 已寫入 csv)
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        running_processes.append(proc)

        # 監控並維持任務數量在 MAX_CONCURRENT_JOBS
        while len(running_processes) >= MAX_CONCURRENT_JOBS:
            # 移除已結束的
            running_processes = [p for p in running_processes if p.poll() is None]
            
            # 若仍滿載，暫停 1 秒後再檢查
            if len(running_processes) >= MAX_CONCURRENT_JOBS:
                time.sleep(1)

    print("所有任務已分發，等待剩餘任務完成...")
    for p in running_processes:
        p.wait()

    end_time = datetime.now()
    print(f"結束時間: {end_time}")
    print(f"總耗時: {end_time - start_time}")

if __name__ == "__main__":
    main()
