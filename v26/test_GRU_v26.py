import os, re, glob, argparse
import numpy as np, pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, average_precision_score

# ---------------- Args ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_version", type=str, default="46")
parser.add_argument("--column", type=str, default="acceleration_Y")
parser.add_argument("--layer", type=int, default=2)
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--fold", type=int, default=1)
parser.add_argument("--model_dir", type=str, required=True, help="Specific model directory path")
parser.add_argument("--out_dir", type=str, default="./test_log", help="Output directory for logs")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 測試數據路徑
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
    f = fname.lower()
    if "nottired" in f: return 0
    if "tired" in f: return 1
    return 1

class TestDataset(Dataset):
    def __init__(self, folder, col, seq_len=300):
        self.items = []
        if not os.path.exists(folder):
            print(f"[Error] Test folder not found: {folder}")
            return
            
        files = sorted(glob.glob(os.path.join(folder, "**/*.csv"), recursive=True))
        if not files:
            files = sorted(glob.glob(os.path.join(folder, "*.csv")))

        for p in files:
            try:
                df = pd.read_csv(p)
                if col not in df.columns: continue
                x = df[col].to_numpy(dtype=np.float32)
                
                if len(x) < seq_len: continue
                
                # 切段
                for s in range(0, len(x)-seq_len+1, seq_len):
                    seg = x[s:s+seq_len].reshape(seq_len, 1)
                    y = infer_label(os.path.basename(p))
                    # 這裡多傳回一個 unique_id (檔名) 用於聚合
                    self.items.append((seg, y, os.path.basename(p)))
            except Exception as e:
                print(f"Error reading {p}: {e}")

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): 
        seg, y, fname = self.items[idx]
        return torch.tensor(seg), torch.tensor(y), fname

# ---------------- Helper ----------------
def load_weight_safe(model, ckpt_path):
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return model

# ---------------- Main ----------------
def main():
    test_ds = TestDataset(test_dir, args.column, seq_len=300)
    if len(test_ds) == 0:
        print("[Error] No test data found.")
        return

    loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)

    ckpts = glob.glob(os.path.join(args.model_dir, "*.pth"))
    ckpts = sorted(ckpts)
        
    if not ckpts:
        print(f"[Error] No checkpoints found in {args.model_dir}")
        return

    results = []
    
    for ck in ckpts:
        ckpt_name = os.path.splitext(os.path.basename(ck))[0]
        full_display_name = f"{model_folder_name}_{ckpt_name}"
        
        model = GRUModel(input_dim=1, hidden=args.hidden, layers=args.layer).to(device)
        try:
            model = load_weight_safe(model, ck)
        except Exception as e:
            print(f"  [Error] Failed to load {ckpt_name}: {e}")
            continue
            
        model.eval()

        # 用字典來儲存每個檔案的預測結果 {filename: {'probs': [], 'label': int}}
        file_metrics = {}
        
        with torch.no_grad():
            for segs, y, fnames in loader:
                segs = segs.to(device)
                outputs = model(segs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                y = y.numpy()
                
                # 將 batch 裡的每一個片段歸戶到對應的檔名
                for i, fname in enumerate(fnames):
                    if fname not in file_metrics:
                        file_metrics[fname] = {'probs': [], 'label': int(y[i])}
                    
                    file_metrics[fname]['probs'].append(float(probs[i]))

        # === 聚合階段 (Aggregation) ===
        final_y_true = []
        final_y_prob = []
        rows = [] # 用於存 CSV

        for fname, data in file_metrics.items():
            # 策略：平均機率 (Soft Voting)
            # 這能容忍少數片段判錯，只要平均值傾向正確方向即可
            avg_prob = np.mean(data['probs'])
            true_label = data['label']
            
            final_y_true.append(true_label)
            final_y_prob.append(avg_prob)
            
            rows.append({
                "file": fname,
                "label": true_label,
                "avg_prob": avg_prob,        # 該檔案的平均預測機率
                "pred": int(avg_prob >= 0.5), # 最終判定
                "segment_count": len(data['probs']), # 該檔案切了幾段
                "min_prob": np.min(data['probs']),   # 參考用：最低分片段
                "max_prob": np.max(data['probs'])    # 參考用：最高分片段
            })

        # === 計算 File-Level 指標 ===
        if len(set(final_y_true)) > 1:
            mAP = average_precision_score(final_y_true, final_y_prob)
        else:
            mAP = 0.0
            
        final_preds = [int(p >= 0.5) for p in final_y_prob]
        acc = accuracy_score(final_y_true, final_preds)
        f1  = f1_score(final_y_true, final_preds)

        # 紀錄 Summary
        results.append({
            "model_folder": model_folder_name,
            "ckpt": ckpt_name,
            "mAP": mAP, 
            "acc": acc, 
            "f1": f1,
            "layer": args.layer,
            "hidden": args.hidden,
            "lr": args.lr,
            "fold": args.fold
        })
        
        # 輸出 "檔案級" 的詳細預測 (現在一行代表一個檔案，而不是一個片段)
        pred_csv_name = f"{full_display_name}_pred.csv"
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, pred_csv_name), index=False)

    # 輸出 Summary
    if results:
        summary_name = f"{model_folder_name}_summary.csv"
        df = pd.DataFrame(results)
        cols = ["model_folder", "ckpt", "mAP", "acc", "f1", "layer", "hidden", "lr", "fold"]
        df = df[cols]
        df.to_csv(os.path.join(args.out_dir, summary_name), index=False)

if __name__=="__main__":
    main()