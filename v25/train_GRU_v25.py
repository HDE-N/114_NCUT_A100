import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import argparse
import glob
from concurrent.futures import ThreadPoolExecutor

# [繪圖套件]
import matplotlib.pyplot as plt
# 設定無介面繪圖模式 (避免在無螢幕的 Server 上報錯)
plt.switch_backend('Agg')

# -----------------------------------------------------------------------------
# 參數設定
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="GRU Training Script")
parser.add_argument("--batch_size", type=int, default=300)
parser.add_argument("--epoch", type=int, default=500)
parser.add_argument("--layer", type=int, default=2)
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--data_version", type=str, default='46')
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--column", type=str, default='acceleration_Y')
parser.add_argument("--fold", type=int, default=1)
parser.add_argument("--patience", type=int, default=100, help="Early Stopping Patience")
args = parser.parse_args()

# [資源控制] 限制 CPU 核心數
torch.set_num_threads(10)

# [GPU 加速]
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Logging 與路徑準備
# -----------------------------------------------------------------------------
t = time.localtime()
formatted_time = f"{t.tm_year-1911}{t.tm_mon:02d}{t.tm_mday:02d}_{t.tm_hour:02d}{t.tm_min:02d}"
pid = os.getpid()

# 建立 Log 資料夾
log_dir = './train_loss_log'
os.makedirs(log_dir, exist_ok=True)
csv_filename = f"{formatted_time}_fold{args.fold}_L{args.layer}_H{args.hidden}_lr{args.lr}_dv{args.data_version}_{args.column}_{pid}.csv"
csv_path = os.path.join(log_dir, csv_filename)

with open(csv_path, 'w', encoding='utf-8') as f:
    f.write("Epoch,TrainLoss,ValLoss,ValAcc,ValF1\n")

# 建立模型儲存資料夾
model_dir = './model'
model_sub_dir = f"fold{args.fold}_layer_{args.layer}_hidden_{args.hidden}_lr_{args.lr}_dv_{args.data_version}_col_{args.column}"
model_path = os.path.join(model_dir, model_sub_dir)
os.makedirs(model_path, exist_ok=True)

# -----------------------------------------------------------------------------
# 繪圖函數
# -----------------------------------------------------------------------------
def plot_metrics(csv_file_path):
    """
    讀取訓練 Log CSV 並繪製 Loss 與 Accuracy/F1 折線圖
    """
    print(f"Generating plot for {csv_file_path}...")
    try:
        df = pd.read_csv(csv_file_path)
        if df.empty:
            return

        plt.figure(figsize=(15, 6))
        
        # Loss 曲線
        plt.subplot(1, 2, 1)
        plt.plot(df['Epoch'], df['TrainLoss'], label='Train Loss', color='blue', alpha=0.7)
        plt.plot(df['Epoch'], df['ValLoss'], label='Val Loss', color='red', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Metrics 曲線
        plt.subplot(1, 2, 2)
        plt.plot(df['Epoch'], df['ValAcc'], label='Val Accuracy', color='green', alpha=0.7)
        plt.plot(df['Epoch'], df['ValF1'], label='Val F1 Score', color='orange', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Validation Metrics')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plot_path = csv_file_path.replace('.csv', '.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved: {plot_path}")
        
    except Exception as e:
        print(f"[Error] Failed to plot metrics: {e}")

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class CSVDataset(Dataset):
    def __init__(self, folder_path, label=None, seq_len=500, column_name='XXX'):
        self.data = []
        self.labels = []
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        def process_file(file_path):
            try:
                df = pd.read_csv(file_path, usecols=[column_name])
                seq = df[[column_name]].values
                if len(seq) >= seq_len:
                    curr_label = label if label is not None else (1 if "Tired" in file_path else 0)
                    return seq[:seq_len], curr_label
            except Exception:
                return None
            return None

        print(f"Loading {len(csv_files)} files from {folder_path}...")
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(process_file, csv_files))
            
        for res in results:
            if res is not None:
                self.data.append(res[0])
                self.labels.append(res[1])
        print(f"Loaded {len(self.data)} samples.")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1]).squeeze(dim=1)

# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------
def main():
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 4,  
        "pin_memory": True,
        "persistent_workers": True, 
        "prefetch_factor": 2
    }

    path_prefix = f"../data_v{args.data_version}/K_Fold"
    
    try:
        train_tired = CSVDataset(f"{path_prefix}/train_data_pre_{args.fold}/Tired", label=1, column_name=args.column)
        train_not = CSVDataset(f"{path_prefix}/train_data_pre_{args.fold}/notTired", label=0, column_name=args.column)
        val_dataset = CSVDataset(f"{path_prefix}/val_data_pre_{args.fold}", label=None, column_name=args.column)
    except Exception as e:
        print(f"[Error] Dataset loading failed: {e}")
        return

    if len(train_tired) == 0 and len(train_not) == 0:
        print(f"[Error] No training data found.")
        return

    train_loader = DataLoader(ConcatDataset([train_tired, train_not]), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = GRUClassifier(hidden_dim=args.hidden, num_layers=args.layer).to(device)
    
    try:
        model = torch.compile(model)
    except Exception:
        pass

    # Loss & Optimizer
    pos_len = len(train_tired) if len(train_tired) > 0 else 1
    pos_weight = torch.tensor(len(train_not) / pos_len, dtype=torch.float32).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')
    no_improve_count = 0

    print(f"Start Training... (PID {pid})")

    for epoch in range(1, args.epoch + 1):
        # --- Training ---
        model.train()
        total_train_loss = 0.0
        
        for X, y in train_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.float16):
                output = model(X)
                loss = criterion(output, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        total_val_loss = 0.0
        tp, fp, fn, correct, total = 0, 0, 0, 0, 0
        
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device, non_blocking=True), y_val.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    val_output = model(X_val)
                    val_loss = criterion(val_output, y_val)
                total_val_loss += val_loss.item()
                preds = (val_output >= 0).float()
                correct += (preds == y_val).sum().item()
                total += len(y_val)
                tp += ((preds == 1) & (y_val == 1)).sum().item()
                fp += ((preds == 1) & (y_val == 0)).sum().item()
                fn += ((preds == 0) & (y_val == 1)).sum().item()

        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        val_acc = correct / total if total > 0 else 0.0
        val_f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

        # Log
        with open(csv_path, 'a', encoding='utf-8') as f:
            f.write(f"{epoch},{avg_train_loss:.4f},{avg_val_loss:.4f},{val_acc:.4f},{val_f1:.4f}\n")

        # [新增] 每 10 個 Epoch 強制存檔
        if epoch % 10 == 0:
            periodic_path = os.path.join(model_path, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), periodic_path)
            # print(f"Saved: {periodic_path}") # 若嫌 log 太多可註解掉

        # Early Stopping & Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_count = 0
            torch.save(model.state_dict(), os.path.join(model_path, "best_model.pth"))
        else:
            no_improve_count += 1
            if no_improve_count >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break
    
    torch.save(model.state_dict(), os.path.join(model_path, "last_model.pth"))
    print(f"Finished. Models saved in {model_path}")
    plot_metrics(csv_path)

if __name__ == "__main__":
    main()
