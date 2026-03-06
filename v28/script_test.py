import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, average_precision_score

# ---------------- Args ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_version", type=str, default="46", help="Data version to locate test_data")
# [修改] 支援多欄位輸入，例如 "acceleration_Y,acceleration_Z"
parser.add_argument("--column", type=str, default="acceleration_Y")
parser.add_argument("--layer", type=int, default=2)
parser.add_argument("--hidden", type=int, default=128)
# [移除] 已移除 lr 與 fold 參數
parser.add_argument("--model_dir", type=str, required=True, help="Path to the directory containing model checkpoints")
parser.add_argument("--out_dir", type=str, default="./test_log", help="Output directory for logs")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [設定] 測試數據固定路徑
test_dir = f'../data_v{args.data_version}/test_data'

os.makedirs(args.out_dir, exist_ok=True)
model_folder_name = os.path.basename(os.path.normpath(args.model_dir))

# ---------------- Model ----------------
class GRUModel(nn.Module):
    def __init__(self, input_dim=1, hidden=128, layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        self.gru.flatten_parameters()
        _, h = self.gru(x)
        return self.fc(h[-1]).squeeze()

# ---------------- Dataset ----------------
def infer_label(fname: str) -> int:
    # 根據檔名判斷標籤 (請依實際情況調整關鍵字)
    f = fname.lower()
    if "nottired" in f: return 0
    if "tired" in f: return 1
    return 1 # Default fallback

class TestDataset(Dataset):
    def __init__(self, folder, column_names, seq_len=300):
        self.items = []
        
        # 檢查路徑
        if not os.path.exists(folder):
            print(f"[Error] Test folder not found: {folder}")
            return
            
        # 搜尋所有 csv (包含子資料夾)
        files = sorted(glob.glob(os.path.join(folder, "**/*.csv"), recursive=True))
        
        if not files:
            print(f"[Warning] No CSV files found in {folder}")
            return

        print(f"Loading {len(files)} test files from: {folder}")

        for p in files:
            try:
                # 讀取指定欄位
                df = pd.read_csv(p, usecols=column_names, dtype=np.float32)
                
                # 確保欄位都存在
                if not all(col in df.columns for col in column_names):
                    continue

                # 取值 (Rows, Channels)
                x = df[column_names].values
                
                if len(x) < seq_len: continue
                
                # 切割片段 (Sliding Window / Non-overlapping)
                # 這裡使用無重疊切割 (stride = seq_len)
                for s in range(0, len(x)-seq_len+1, seq_len):
                    seg = x[s:s+seq_len, :] 
                    y = infer_label(os.path.basename(p))
                    # 紀錄: (片段數據, 標籤, 原始檔名)
                    self.items.append((seg, y, os.path.basename(p)))
            except Exception as e:
                print(f"Error reading {p}: {e}")

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): 
        seg, y, fname = self.items[idx]
        return torch.tensor(seg, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), fname

# ---------------- Helper ----------------
def load_weight_safe(model, ckpt_path):
    # weights_only=True 增加安全性 (PyTorch 建議)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    new_state_dict = {}
    for k, v in state_dict.items():
        # 處理可能的 DataParallel 前綴
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return model

# ---------------- Main ----------------
def main():
    # 1. 處理欄位
    target_columns = [c.strip() for c in args.column.split(',')]
    input_dim = len(target_columns)
    print(f"Target Columns: {target_columns} | Input Dim: {input_dim}")

    # 2. 準備 Dataset
    test_ds = TestDataset(test_dir, target_columns, seq_len=300)
    
    if len(test_ds) == 0:
        print(f"[Error] No valid data loaded from {test_dir}. Exiting.")
        return

    loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)

    # 3. 搜尋 Checkpoints
    if not os.path.exists(args.model_dir):
         print(f"[Error] Model directory not found: {args.model_dir}")
         return

    ckpts = glob.glob(os.path.join(args.model_dir, "*.pth"))
    ckpts = sorted(ckpts)
        
    if not ckpts:
        print(f"[Error] No .pth checkpoints found in {args.model_dir}")
        return

    results = []
    
    # 4. 迴圈測試每個模型檔
    for ck in ckpts:
        ckpt_name = os.path.splitext(os.path.basename(ck))[0]
        if "optimizer" in ckpt_name: continue # 跳過優化器存檔

        print(f"Testing checkpoint: {ckpt_name} ...")
        
        # 初始化模型
        model = GRUModel(input_dim=input_dim, hidden=args.hidden, layers=args.layer).to(device)
        
        try:
            model = load_weight_safe(model, ck)
        except Exception as e:
            print(f"  [Error] Failed to load {ckpt_name}: {e}")
            continue
            
        model.eval()

        # 儲存預測結果 {filename: {'probs': [], 'label': int}}
        file_metrics = {}
        
        with torch.no_grad():
            for segs, y, fnames in loader:
                segs = segs.to(device)
                
                outputs = model(segs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                y = y.numpy()
                
                for i, fname in enumerate(fnames):
                    if fname not in file_metrics:
                        file_metrics[fname] = {'probs': [], 'label': int(y[i])}
                    
                    file_metrics[fname]['probs'].append(float(probs[i]))

        # === 聚合 (Aggregation) ===
        final_y_true = []
        final_y_prob = []
        rows = [] 

        for fname, data in file_metrics.items():
            # Soft Voting (平均機率)
            avg_prob = np.mean(data['probs'])
            true_label = data['label']
            
            final_y_true.append(true_label)
            final_y_prob.append(avg_prob)
            
            rows.append({
                "file": fname,
                "label": true_label,
                "avg_prob": avg_prob,        
                "pred": int(avg_prob >= 0.5), 
                "segment_count": len(data['probs']),
                "min_prob": np.min(data['probs']),
                "max_prob": np.max(data['probs'])
            })

        # === 計算指標 ===
        if len(set(final_y_true)) > 1:
            mAP = average_precision_score(final_y_true, final_y_prob)
        else:
            mAP = 0.0 # 只有單一類別時無法計算 mAP
            
        final_preds = [int(p >= 0.5) for p in final_y_prob]
        acc = accuracy_score(final_y_true, final_preds)
        f1  = f1_score(final_y_true, final_preds)

        print(f"  -> mAP: {mAP:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

        results.append({
            "model_folder": model_folder_name,
            "ckpt": ckpt_name,
            "mAP": mAP, 
            "acc": acc, 
            "f1": f1,
            "layer": args.layer,
            "hidden": args.hidden,
            "columns": args.column
        })
        
        # 輸出單檔預測明細
        full_display_name = f"{model_folder_name}_{ckpt_name}"
        pred_csv_name = f"{full_display_name}_pred.csv"
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, pred_csv_name), index=False)

    # 5. 輸出總表 (Summary)
    if results:
        summary_name = f"{model_folder_name}_summary.csv"
        df = pd.DataFrame(results)
        # 調整欄位順序 (移除 fold, lr)
        cols = ["model_folder", "ckpt", "mAP", "acc", "f1", "layer", "hidden", "columns"]
        df = df[cols]
        
        out_path = os.path.join(args.out_dir, summary_name)
        df.to_csv(out_path, index=False)
        print(f"\n[Done] Summary saved to: {out_path}")

if __name__=="__main__":
    main()