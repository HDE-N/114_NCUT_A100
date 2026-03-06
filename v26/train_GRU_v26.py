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

# 設定無介面繪圖模式
plt.switch_backend('Agg')

# -----------------------------------------------------------------------------
# PyTorch 2.x 效能優化設定
# -----------------------------------------------------------------------------
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

# -----------------------------------------------------------------------------
# 參數設定
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--fold", type=int, default=1, help="K-Fold number (1-5)")
parser.add_argument("--column", type=str, default='acceleration_Y')
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--layer", type=int, default=2)
parser.add_argument("--patience", type=int, default=30)
# [補回] 資料版本參數
parser.add_argument("--data_version", type=str, default='46', help="Data version folder name")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# 極速資料讀取 (Parallel IO)
# -----------------------------------------------------------------------------
class FastCSVDataset(Dataset):
    def __init__(self, folder_path, label, column_name, seq_len=300):
        self.data = []
        self.labels = []
        
        # 容錯：確保路徑存在才搜尋
        if not os.path.exists(folder_path):
            # print(f"[Warn] Path not found: {folder_path}") 
            return

        files = glob.glob(os.path.join(folder_path, "*.csv"))
        if not files:
            return

        print(f"Loading {len(files)} files from {folder_path}...")

        def read_file(f):
            try:
                # usecols 僅讀取需要的欄位
                val = pd.read_csv(f, usecols=[column_name], dtype=np.float32)[column_name].values
                if len(val) >= seq_len:
                    return val[:seq_len], label
            except:
                pass
            return None

        # 平行讀取
        with ThreadPoolExecutor(max_workers=16) as ex:
            results = list(ex.map(read_file, files))
        
        valid = [r for r in results if r is not None]
        if valid:
            data_list, label_list = zip(*valid)
            self.data = torch.tensor(np.array(data_list), dtype=torch.float32).unsqueeze(-1)
            self.labels = torch.tensor(np.array(label_list), dtype=torch.float32)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

# -----------------------------------------------------------------------------
# 模型定義
# -----------------------------------------------------------------------------
class GRUModel(nn.Module):
    def __init__(self, input_dim=1, hidden=128, layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        self.gru.flatten_parameters()
        _, h = self.gru(x)
        return self.fc(h[-1]).squeeze()

# -----------------------------------------------------------------------------
# 主程式
# -----------------------------------------------------------------------------
def main():
    # 1. 路徑準備
    base_path = f"../data_v{args.data_version}/K_Fold/fold_{args.fold}"
    
    if not os.path.exists(base_path):
        print(f"[Error] Path not found: {base_path}")
        print("Please check if data_version matches your folder structure.")
        return

    # 建立 Log 與 Model 存放路徑
    pid = os.getpid()
    timestamp = time.strftime("%Y%m%d_%H%M")
    
    run_name = f"dv{args.data_version}_fold{args.fold}_L{args.layer}_H{args.hidden}_{timestamp}_{pid}"
    
    log_dir = "train_logs"
    model_dir = f"models/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    csv_log_path = os.path.join(log_dir, f"{run_name}.csv")
    with open(csv_log_path, 'w') as f: 
        f.write("Epoch,TrainLoss,ValLoss,ValAcc,ValF1\n")

    # 2. 載入資料
    print(f"[{pid}] Loading Data v{args.data_version} | Fold {args.fold}...")
    
    # 訓練集
    train_t = FastCSVDataset(f"{base_path}/train/Tired", 1, args.column)
    train_n = FastCSVDataset(f"{base_path}/train/notTired", 0, args.column)
    
    if len(train_t) == 0 and len(train_n) == 0:
        print("[Error] No training data found."); return

    train_ds = ConcatDataset([d for d in [train_t, train_n] if len(d) > 0])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=10, pin_memory=True, persistent_workers=True)

    # 驗證集
    val_t = FastCSVDataset(f"{base_path}/test/Tired", 1, args.column)
    val_n = FastCSVDataset(f"{base_path}/test/notTired", 0, args.column)
    val_ds = ConcatDataset([d for d in [val_t, val_n] if len(d) > 0])
    
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=8, pin_memory=True) if len(val_ds) > 0 else None

    # 3. 初始化模型與編譯
    model = GRUModel(hidden=args.hidden, layers=args.layer).to(device)
    
    print(f"[{pid}] Compiling model...")
    try:
        model = torch.compile(model, mode="default")
    except Exception as e:
        print(f"[Warning] Compile failed: {e}")

    # 4. 優化器與 Loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 動態計算權重
    n_pos = len(train_t)
    n_neg = len(train_n)
    pos_weight_val = n_neg / max(n_pos, 1)
    pos_weight = torch.tensor(pos_weight_val).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler('cuda')

    # 5. 訓練迴圈
    best_val_loss = float('inf')
    early_stop_cnt = 0

    print(f"[{pid}] Start Training...")
    
    for ep in range(1, args.epoch + 1):
        # --- Train ---
        model.train()
        tr_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                out = model(x)
                loss = criterion(out, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item()
        
        avg_tr_loss = tr_loss / len(train_loader)

        # --- Validation ---
        avg_val_loss, acc, f1 = 0.0, 0.0, 0.0
        
        if val_loader:
            model.eval()
            val_loss_sum = 0
            tp, fp, fn, correct, total = 0, 0, 0, 0, 0
            
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    
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

        # --- Log & Save ---
        print(f"Ep {ep:3d} | TrLoss: {avg_tr_loss:.4f} | ValLoss: {avg_val_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
        
        with open(csv_log_path, 'a') as f:
            f.write(f"{ep},{avg_tr_loss:.4f},{avg_val_loss:.4f},{acc:.4f},{f1:.4f}\n")

        # ------------------- [修改點 1: 每 10 Epoch 存檔] -------------------
        if ep % 10 == 0:
            save_path = f"{model_dir}/model_epoch_{ep}.pth"
            torch.save(model.state_dict(), save_path)
            # print(f"Saved checkpoint: {save_path}") # 若怕 log 太多可註解

        # Checkpoint & Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_cnt = 0
            torch.save(model.state_dict(), f"{model_dir}/best.pth")
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= args.patience:
                print(f"[{pid}] Early stopping at epoch {ep}")
                break

    # ------------------- [修改點 2: 迴圈結束後存 Last] -------------------
    # 無論是跑滿 Epoch 自然結束，還是 Early Stopping 跳出，都會執行到這裡
    torch.save(model.state_dict(), f"{model_dir}/last.pth")
    print(f"[{pid}] Training Finished. Last model saved to {model_dir}/last.pth")

    # 6. 繪圖
    try:
        df = pd.read_csv(csv_log_path)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(df['TrainLoss'], label='Train')
        plt.plot(df['ValLoss'], label='Val')
        plt.title('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(df['ValAcc'], label='Acc')
        plt.plot(df['ValF1'], label='F1')
        plt.title('Metrics')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{log_dir}/{run_name}.png")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()