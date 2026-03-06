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

# [資源控制] 限制每個訓練任務只使用 10 個 CPU 核心進行矩陣運算
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
# 檔名加入 PID 防止平行寫入衝突
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
# Dataset (平行讀取優化)
# -----------------------------------------------------------------------------
class CSVDataset(Dataset):
    def __init__(self, folder_path, label=None, seq_len=500, column_name='XXX'):
        self.data = []
        self.labels = []
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        # 定義單檔處理函數
        def process_file(file_path):
            try:
                # usecols 加速 IO，只讀需要的欄位
                df = pd.read_csv(file_path, usecols=[column_name])
                seq = df[[column_name]].values
                if len(seq) >= seq_len:
                    # 依檔名或參數決定 Label
                    curr_label = label if label is not None else (1 if "Tired" in file_path else 0)
                    return seq[:seq_len], curr_label
            except Exception:
                return None
            return None

        print(f"Loading {len(csv_files)} files from {folder_path}...")
        
        # 使用 ThreadPoolExecutor 多執行緒讀檔 (IO Bound 加速)
        # 16 個執行緒讀取速度通常是單執行緒的 5-10 倍
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(process_file, csv_files))
            
        for res in results:
            if res is not None:
                self.data.append(res[0])
                self.labels.append(res[1])
        
        print(f"Loaded {len(self.data)} samples.")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        # 轉 Tensor
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# -----------------------------------------------------------------------------
# Model 定義
# -----------------------------------------------------------------------------
class GRUClassifier(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        _, h_n = self.gru(x)
        # 取最後一層的 hidden state
        return self.fc(h_n[-1]).squeeze(dim=1)

# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------
def main():
    # DataLoader 優化參數
    # num_workers=4: 配合 10 核運算，分配 4 個 subprocess 搬運資料
    # persistent_workers=True: 避免每個 epoch 重建 subprocess，對小模型訓練速度提升巨大
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 4,  
        "pin_memory": True,
        "persistent_workers": True, 
        "prefetch_factor": 2
    }

    # 載入資料集
    path_prefix = f"../data_v{args.data_version}/K_Fold"
    
    # 建立 Dataset (加入例外處理避免路徑錯誤崩潰)
    try:
        train_tired = CSVDataset(f"{path_prefix}/train_data_pre_{args.fold}/Tired", label=1, column_name=args.column)
        train_not = CSVDataset(f"{path_prefix}/train_data_pre_{args.fold}/notTired", label=0, column_name=args.column)
        val_dataset = CSVDataset(f"{path_prefix}/val_data_pre_{args.fold}", label=None, column_name=args.column)
    except Exception as e:
        print(f"[Error] Dataset loading failed: {e}")
        return

    # 若無資料則退出
    if len(train_tired) == 0 and len(train_not) == 0:
        print(f"[Error] No training data found in {path_prefix}. Check path or data_version.")
        return

    train_loader = DataLoader(ConcatDataset([train_tired, train_not]), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    # 初始化模型
    model = GRUClassifier(hidden_dim=args.hidden, num_layers=args.layer).to(device)
    
    # Linux PyTorch 2.0+ Compile 加速
    try:
        print("Compiling model...")
        model = torch.compile(model)
    except Exception as e:
        print(f"Warning: torch.compile not supported or failed: {e}")

    # Loss & Optimizer
    # 計算正樣本權重處理不平衡
    pos_len = len(train_tired) if len(train_tired) > 0 else 1
    pos_weight = torch.tensor(len(train_not) / pos_len, dtype=torch.float32).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # [新版] 混合精度 Scaler
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')
    no_improve_count = 0

    print(f"Start Training (PID {pid}): Fold{args.fold} | Layer{args.layer} | Hidden{args.hidden} | LR{args.lr}")

    for epoch in range(1, args.epoch + 1):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0
        
        for X, y in train_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            # [新版] 混合精度 Context Manager
            with torch.amp.autocast('cuda', dtype=torch.float16):
                output = model(X)
                loss = criterion(output, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation Phase ---
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

        # Log 寫入 CSV
        with open(csv_path, 'a', encoding='utf-8') as f:
            f.write(f"{epoch},{avg_train_loss:.4f},{avg_val_loss:.4f},{val_acc:.4f},{val_f1:.4f}\n")

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
    
    # 訓練結束，儲存最後一個模型
    torch.save(model.state_dict(), os.path.join(model_path, "last_model.pth"))
    print(f"Finished: {model_sub_dir}")

if __name__ == "__main__":
    main()