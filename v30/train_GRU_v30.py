import os
import time
import glob
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

plt.switch_backend('Agg')
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

# -----------------------------------------------------------------------------
# 參數設定
# -----------------------------------------------------------------------------
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
# Dataset & Model
# -----------------------------------------------------------------------------
class FastCSVDataset(Dataset):
    def __init__(self, folder_path, label, column_names, seq_len=300):
        self.data = []
        self.labels = []
        if not os.path.exists(folder_path): return
        files = glob.glob(os.path.join(folder_path, "*.csv"))
        if not files: return

        # print(f"Loading {len(files)} files from {folder_path}...")

        def read_file(f):
            try:
                df = pd.read_csv(f, usecols=column_names, dtype=np.float32)
                val = df[column_names].values 
                if len(val) >= seq_len:
                    return val[:seq_len, :], label
            except: pass
            return None

        with ThreadPoolExecutor(max_workers=8) as ex: # 稍微降低worker避免搶佔
            results = list(ex.map(read_file, files))
        
        valid = [r for r in results if r is not None]
        if valid:
            data_list, label_list = zip(*valid)
            self.data = torch.tensor(np.array(data_list), dtype=torch.float32)
            self.labels = torch.tensor(np.array(label_list), dtype=torch.float32)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

class GRUModel(nn.Module):
    def __init__(self, input_dim=1, hidden=128, layers=2):
        super().__init__()
        # 1. 開啟雙向設定 (bidirectional=True)
        self.gru = nn.GRU(input_dim, hidden, layers, batch_first=True, dropout=0.2, bidirectional=True)
        # 2. 全連接層的輸入維度必須乘以 2
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        self.gru.flatten_parameters()
        _, h = self.gru(x)
        
        # h 的 shape 為 (num_layers * 2, batch_size, hidden_size)
        # 3. 提取最後一層的正向 (h[-2]) 與反向 (h[-1]) 隱藏狀態並拼接
        h_out = torch.cat((h[-2], h[-1]), dim=1)
        
        return self.fc(h_out).squeeze()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    target_columns = [c.strip() for c in args.column.split(',')]
    input_dim = len(target_columns)
    
    # 路徑準備
    base_path = f"../data_v{args.data_version}/K_Fold/fold_{args.fold}"
    if not os.path.exists(base_path):
        print(f"[Error] Path not found: {base_path}")
        return

    # [關鍵] 統一命名邏輯
    run_name = f"fold{args.fold}_layer_{args.layer}_hidden_{args.hidden}_lr_{args.lr}_dv_{args.data_version}_col_{args.column}"
    
    log_dir = "train_logs"
    model_dir = f"models/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 檢查是否已完成 (有 last.pth 就當作跑完了)
    if os.path.exists(os.path.join(model_dir, "last.pth")):
        print(f"[Skip] Already exists: {run_name}")
        return

    print(f"Start Training: {run_name}")

    # 資料載入
    train_t = FastCSVDataset(f"{base_path}/train/Tired", 1, target_columns)
    train_n = FastCSVDataset(f"{base_path}/train/notTired", 0, target_columns)
    val_t = FastCSVDataset(f"{base_path}/test/Tired", 1, target_columns)
    val_n = FastCSVDataset(f"{base_path}/test/notTired", 0, target_columns)

    if len(train_t) == 0 and len(train_n) == 0: return

    train_ds = ConcatDataset([d for d in [train_t, train_n] if len(d) > 0])
    val_ds = ConcatDataset([d for d in [val_t, val_n] if len(d) > 0])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = GRUModel(input_dim=input_dim, hidden=args.hidden, layers=args.layer).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 計算權重
    n_pos = len(train_t)
    n_neg = len(train_n)
    pos_weight = torch.tensor(n_neg / max(n_pos, 1)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')
    early_stop_cnt = 0
    csv_log_path = os.path.join(log_dir, f"{run_name}.csv")

    with open(csv_log_path, 'w') as f: f.write("Epoch,TrainLoss,ValLoss,ValAcc,ValF1\n")

    for ep in range(1, args.epoch + 1):
        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item()
        
        avg_tr_loss = tr_loss / len(train_loader)

        # Validation
        avg_val_loss, acc, f1 = 0.0, 0.0, 0.0
        model.eval()
        val_loss_sum, tp, fp, fn, correct, total = 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast('cuda'):
                    out = model(x)
                    val_loss_sum += criterion(out, y).item()
                preds = (out >= 0).float()
                correct += (preds == y).sum().item()
                total += len(y)
                tp += ((preds == 1) & (y == 1)).sum().item()
                fp += ((preds == 1) & (y == 0)).sum().item()
                fn += ((preds == 0) & (y == 1)).sum().item()
        
        avg_val_loss = val_loss_sum / len(val_loader)
        acc = correct / max(total, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1e-8)

        with open(csv_log_path, 'a') as f:
            f.write(f"{ep},{avg_tr_loss:.4f},{avg_val_loss:.4f},{acc:.4f},{f1:.4f}\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_cnt = 0
            torch.save(model.state_dict(), f"{model_dir}/best.pth")
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= args.patience:
                break

    torch.save(model.state_dict(), f"{model_dir}/last.pth")
    # 這裡順便存一個 info 檔，方便 script 檢查
    with open(f"{model_dir}/train_done.flag", "w") as f: f.write("done")

if __name__ == "__main__":
    main()