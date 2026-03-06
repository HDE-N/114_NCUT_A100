import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, average_precision_score

# 參數設置
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--fold", type=int, default=1)
parser.add_argument("--column", type=str, default='acceleration_Y')
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--layer", type=int, default=2)
parser.add_argument("--patience", type=int, default=30)
parser.add_argument("--data_version", type=str, default='46')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Logger 類別：用於同時將輸出顯示在螢幕並寫入檔案 (Optional)
# 或者直接重導向 stdout。這裡採用直接重導向，確保"所有"輸出都在檔案中。
# -----------------------------------------------------------------------------
class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # 這是為了滿足 python 3 的緩衝需求
        self.terminal.flush()
        self.log.flush()

# -----------------------------------------------------------------------------
# Dataset & Model
# -----------------------------------------------------------------------------
def infer_label(fname: str) -> int:
    f = fname.lower()
    if "nottired" in f: return 0
    if "tired" in f: return 1
    return 1

class TestDataset(Dataset):
    def __init__(self, folder, column_names, seq_len=300):
        self.items = []
        if not os.path.exists(folder): return
        files = glob.glob(os.path.join(folder, "**/*.csv"), recursive=True)
        
        for p in files:
            try:
                df = pd.read_csv(p, usecols=column_names, dtype=np.float32)
                if not all(col in df.columns for col in column_names): continue
                x = df[column_names].values
                if len(x) < seq_len: continue
                for s in range(0, len(x)-seq_len+1, seq_len):
                    seg = x[s:s+seq_len, :] 
                    y = infer_label(os.path.basename(p))
                    self.items.append((seg, y, os.path.basename(p)))
            except: pass

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): 
        seg, y, fname = self.items[idx]
        return torch.tensor(seg), torch.tensor(y), fname

class GRUModel(nn.Module):
    def __init__(self, input_dim=1, hidden=128, layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        self.gru.flatten_parameters()
        _, h = self.gru(x)
        return self.fc(h[-1]).squeeze()

def main():
    target_columns = [c.strip() for c in args.column.split(',')]
    input_dim = len(target_columns)
    
    # 建立 run_name
    run_name = f"fold{args.fold}_layer_{args.layer}_hidden_{args.hidden}_lr_{args.lr}_dv_{args.data_version}_col_{args.column}"
    
    # ========================== [修改 1] 設定輸出目錄與路徑 ==========================
    output_dir = "./test_log"
    os.makedirs(output_dir, exist_ok=True)
    
    # 定義輸出的 CSV 路徑 (改為存到 ./test_log，並加上 run_name 以區分不同參數)
    result_csv_path = os.path.join(output_dir, f"{run_name}_result.csv")
    
    # 定義輸出的 Log 文字檔路徑
    log_file_path = os.path.join(output_dir, f"{run_name}_log.txt")
    
    # 重導向 print 輸出至檔案 (同時保留螢幕輸出)
    sys.stdout = DualLogger(log_file_path)
    # ==============================================================================

    # 檢查是否已經測過了 (檢查 ./test_log 下的檔案)
    if os.path.exists(result_csv_path):
        print(f"[Skip] Test result exists in {output_dir}: {run_name}")
        return

    # 尋找模型路徑 (保持原樣，從 models 資料夾讀取)
    model_dir = f"models/{run_name}"
    ckpt_path = os.path.join(model_dir, "best.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(model_dir, "last.pth")
        if not os.path.exists(ckpt_path):
            print(f"[Error] No model weights found for {run_name} in {model_dir}")
            return

    print(f"Testing: {run_name}")
    print(f"Loading weights from: {ckpt_path}")

    # 準備測試資料
    test_data_path = f"../data_v{args.data_version}/test_data"    # ====================================測試集路徑========================================
    test_ds = TestDataset(test_data_path, target_columns, seq_len=300)
    if len(test_ds) == 0:
        print("[Error] No test data found.")
        return
    loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=4)

    # 載入模型
    model = GRUModel(input_dim=input_dim, hidden=args.hidden, layers=args.layer).to(device)
    
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        new_state = {}
        for k, v in state.items():
            if k.startswith("_orig_mod."): new_state[k[10:]] = v
            else: new_state[k] = v
        model.load_state_dict(new_state)
    except Exception as e:
        print(f"[Error] Load failed: {e}")
        return

    model.eval()
    file_metrics = {}

    with torch.no_grad():
        for segs, y, fnames in loader:
            segs = segs.to(device)
            probs = torch.sigmoid(model(segs)).cpu().numpy()
            y = y.numpy()
            for i, fname in enumerate(fnames):
                if fname not in file_metrics: file_metrics[fname] = {'probs': [], 'label': int(y[i])}
                file_metrics[fname]['probs'].append(float(probs[i]))

    # 計算結果
    final_y_true, final_y_prob = [], []
    for fname, data in file_metrics.items():
        avg_prob = np.mean(data['probs'])
        final_y_true.append(data['label'])
        final_y_prob.append(avg_prob)

    if len(set(final_y_true)) > 1: mAP = average_precision_score(final_y_true, final_y_prob)
    else: mAP = 0.0
    
    final_preds = [int(p >= 0.5) for p in final_y_prob]
    acc = accuracy_score(final_y_true, final_preds)
    f1  = f1_score(final_y_true, final_preds)

    # 儲存結果
    df = pd.DataFrame([{
        "run_name": run_name,
        "f1": f1,
        "acc": acc,
        "mAP": mAP,
        "layer": args.layer,
        "hidden": args.hidden,
        "lr": args.lr,
        "column": args.column
    }])
    
    # ========================== [修改 2] 儲存至新的路徑 ==========================
    df.to_csv(result_csv_path, index=False)
    print(f"Saved result to {result_csv_path}")
    print("-" * 30)

if __name__ == "__main__":
    main()