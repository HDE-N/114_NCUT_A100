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
from sklearn.metrics import f1_score, accuracy_score

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
    def __init__(self, folder_path, label, column_names, seq_len=960):
        self.data = []
        self.labels = []
        if not os.path.exists(folder_path): return
        files = glob.glob(os.path.join(folder_path, "*.csv"))
        if not files: return

        def read_file(f):
            try:
                # 只讀取需要的欄位
                df = pd.read_csv(f, usecols=column_names, dtype=np.float32)
                
                # 確保欄位順序正確
                val = df[column_names].values 
                if len(val) >= seq_len:
                    return val[:seq_len, :], label
            except: pass
            return None

        with ThreadPoolExecutor(max_workers=8) as ex: 
            results = list(ex.map(read_file, files))

        valid = [r for r in results if r is not None]
        if valid:
            data_list, label_list = zip(*valid)
            self.data = torch.tensor(np.array(data_list), dtype=torch.float32)
            # 多類別分類，Label 需要是 long 型別
            self.labels = torch.tensor(np.array(label_list), dtype=torch.long)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

class GRUModel(nn.Module):
    def __init__(self, input_dim=1, hidden=128, layers=2, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, layers, batch_first=True, dropout=0.2)
        # 輸出維度改為 3 (notTired=0, Tired=1, Other=2)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        self.gru.flatten_parameters()
        _, h = self.gru(x)
        # h[-1] 是最後一層隱藏狀態
        out = self.fc(h[-1]).squeeze()
        return out

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    target_columns = [c.strip() for c in args.column.split(',')]
    input_dim = len(target_columns)
    
    # 根據 target_columns 生成 Other 對應的 aug_columns
    aug_columns = [f"{c}_aug" for c in target_columns]
    
    # 路徑準備
    base_path = f"../data_v{args.data_version}/K_Fold/fold_{args.fold}"
    if not os.path.exists(base_path):
        print(f"[Error] Path not found: {base_path}")
        return

    # 統一命名邏輯 (加入 3class 標示)
    run_name = f"fold{args.fold}_layer_{args.layer}_hidden_{args.hidden}_lr_{args.lr}_dv_{args.data_version}_col_{args.column}_3class"
    
    log_dir = "train_logs"
    model_dir = f"models/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 檢查是否已完成
    if os.path.exists(os.path.join(model_dir, "last.pth")):
        print(f"[Skip] Already exists: {run_name}")
        return

    print(f"Start Training: {run_name}")

    # ===== 資料載入 =====
    # Label 0: notTired, Label 1: Tired, Label 2: Other (Augmented data)
    
    # 訓練集
    train_t = FastCSVDataset(f"{base_path}/train/Tired", label=1, column_names=target_columns)
    train_n = FastCSVDataset(f"{base_path}/train/notTired", label=0, column_names=target_columns)
    
    # 讀取相同資料夾，但指定欄位為 aug_columns，Label 設為 2
    train_other_t = FastCSVDataset(f"{base_path}/train/Tired", label=2, column_names=aug_columns)
    train_other_n = FastCSVDataset(f"{base_path}/train/notTired", label=2, column_names=aug_columns)

    # 驗證集
    val_t = FastCSVDataset(f"{base_path}/test/Tired", label=1, column_names=target_columns)
    val_n = FastCSVDataset(f"{base_path}/test/notTired", label=0, column_names=target_columns)
    
    val_other_t = FastCSVDataset(f"{base_path}/test/Tired", label=2, column_names=aug_columns)
    val_other_n = FastCSVDataset(f"{base_path}/test/notTired", label=2, column_names=aug_columns)

    train_datasets = [d for d in [train_t, train_n, train_other_t, train_other_n] if len(d) > 0]
    val_datasets = [d for d in [val_t, val_n, val_other_t, val_other_n] if len(d) > 0]

    if not train_datasets: return

    train_ds = ConcatDataset(train_datasets)
    val_ds = ConcatDataset(val_datasets)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = GRUModel(input_dim=input_dim, hidden=args.hidden, layers=args.layer, num_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # ===== 計算 3 個類別的權重 (Class Weights) =====
    num_notTired = len(train_n)
    num_Tired = len(train_t)
    num_Other = len(train_other_t) + len(train_other_n) # 理論上是前兩者相加
    total_samples = num_notTired + num_Tired + num_Other
    
    counts = [num_notTired, num_Tired, num_Other]
    
    # 處理分母可能為 0 的情況
    weights = []
    for count in counts:
        if count == 0:
            weights.append(0.0)
        else:
            weights.append(total_samples / (3.0 * count))
            
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"Data Counts - notTired: {num_notTired}, Tired: {num_Tired}, Other: {num_Other}")
    print(f"Class Weights: {class_weights.cpu().numpy()}")

    # 改用 CrossEntropyLoss (支援多類別與權重平衡)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')
    early_stop_cnt = 0
    csv_log_path = os.path.join(log_dir, f"{run_name}.csv")

    with open(csv_log_path, 'w') as f: 
        f.write("Epoch,TrainLoss,ValLoss,ValAcc,ValMacroF1\n")

    for ep in range(1, args.epoch + 1):
        # ---------- Train ----------
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

        # ---------- Validation ----------
        model.eval()
        val_loss_sum = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast('cuda'):
                    out = model(x)
                    loss = criterion(out, y)
                    val_loss_sum += loss.item()
                
                # 取最大機率的索引作為預測類別
                preds = torch.argmax(out, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        avg_val_loss = val_loss_sum / len(val_loader)
        
        # 多類別指標計算
        acc = accuracy_score(all_targets, all_preds)
        # 使用 macro F1-score 來平衡評估三個類別的表現
        macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

        with open(csv_log_path, 'a') as f:
            f.write(f"{ep},{avg_tr_loss:.4f},{avg_val_loss:.4f},{acc:.4f},{macro_f1:.4f}\n")

        # ---------- Early Stopping & Save Model ----------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_cnt = 0
            torch.save(model.state_dict(), f"{model_dir}/best.pth")
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= args.patience:
                print(f"Early stopping at epoch {ep}")
                break

    torch.save(model.state_dict(), f"{model_dir}/last.pth")
    with open(f"{model_dir}/train_done.flag", "w") as f: f.write("done")
    print(f"Training Complete. Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()