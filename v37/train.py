import os
import sys
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
parser.add_argument("--patience", type=int, default=40)
parser.add_argument("--data_version", type=str, default='46')
parser.add_argument("--lambda_dann", type=float, default=1.0, help="保留參數槽以相容自動化腳本")
parser.add_argument("--target_version", type=str, default='46/special_data', help="新使用者前5分鐘數據的資料夾路徑")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Dataset 核心組件
# -----------------------------------------------------------------------------
class FastCSVDataset(Dataset):
    def __init__(self, folder_path, label, column_names, seq_len=960):
        self.data = []
        self.labels = []
        if not os.path.exists(folder_path): return
        files = glob.glob(os.path.join(folder_path, "*.csv"))
        if not files: return

        all_segments = []

        def read_file(f):
            try:
                df = pd.read_csv(f, usecols=column_names, dtype=np.float32)
                val = df[column_names].values 
                
                local_segs = []
                if len(val) >= seq_len:
                    for s in range(0, len(val) - seq_len + 1, seq_len):
                        local_segs.append(val[s:s+seq_len, :])
                return local_segs
            except: 
                return None

        with ThreadPoolExecutor(max_workers=8) as ex: 
            results = list(ex.map(read_file, files))

        for res in results:
            if res:
                all_segments.extend(res)

        if all_segments:
            self.data = torch.tensor(np.array(all_segments), dtype=torch.float32)
            self.labels = torch.tensor([label] * len(all_segments), dtype=torch.long)
            print(f"Loaded folder {os.path.basename(folder_path)}: Extracted {len(self.data)} segments.")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]


# -----------------------------------------------------------------------------
# 方案 A 核心：特徵標準化對齊映射器 (Feature Standardization Alignment)
# -----------------------------------------------------------------------------
class FeatureStandardizationMapper:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        # W 存放縮放矩陣 (Scaling Matrix)，b 存放平移向量 (Offset Vector)
        self.W = torch.eye(input_dim).to(device)
        self.b = torch.zeros(input_dim).to(device)

    def fit(self, source_class0_dataset, target_dataset):
        """
        利用兩端 Class 0 (notTired) 的均值與標準差，計算出 Z-Score 轉換矩陣 W 與 b。
        這樣能同時校正使用者的「感測器佩戴角度(均值)」與「動作幅度(方差)」。
        """
        print("====== 正在計算特徵標準化對齊 (Z-score Alignment) ======")
        
        # 1. 提取 Source 端「只有 Class 0 (notTired)」的全部數據點
        src_list = [source_class0_dataset[i][0] for i in range(len(source_class0_dataset))]
        src_data = torch.cat(src_list, dim=0).to(device)  # shape: [N*960, input_dim]
        
        # 2. 提取 Target 端「前 5 分鐘 (全是 Class 0)」的全部數據點
        tgt_list = [target_dataset[i][0] for i in range(len(target_dataset))]
        tgt_data = torch.cat(tgt_list, dim=0).to(device)  # shape: [M*960, input_dim]

        # 3. 計算各個通道的均值 (Mean) 與標準差 (Std)
        mu_src = torch.mean(src_data, dim=0)
        std_src = torch.std(src_data, dim=0) + 1e-5  # 避免除以 0
        
        mu_tgt = torch.mean(tgt_data, dim=0)
        std_tgt = torch.std(tgt_data, dim=0) + 1e-5

        # 4. 展開 Z-score 數學式：
        # 目標: X_aligned = ((X_tgt - mu_tgt) / std_tgt) * std_src + mu_src
        # 展開成: X_aligned = X_tgt * W + b
        # 其中: W = std_src / std_tgt,  b = mu_src - (mu_tgt * W)
        
        scaling_factor = std_src / std_tgt
        self.W = torch.diag(scaling_factor)  # 轉為對角矩陣，這樣矩陣乘法 X @ W 時各通道獨立縮放，不會互相污染
        self.b = mu_src - (mu_tgt @ self.W)
        
        print(f"-> Z-score 特徵標準化對齊參數計算成功！")
        print(f"   各通道縮放比例 (Scaling W): {scaling_factor.cpu().numpy()}")
        print(f"   各通道偏移量 (Offset b): {self.b.cpu().numpy()}")

    def forward(self, x):
        # 測試時呼叫：獨立對各通道進行縮放與平移
        return x @ self.W + self.b


# -----------------------------------------------------------------------------
# 核心模型 - 標準 GRU 網路本體
# -----------------------------------------------------------------------------
class GRUModel(nn.Module):
    def __init__(self, input_dim=1, hidden=128, layers=2, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        self.gru.flatten_parameters()
        _, h = self.gru(x)
        return self.fc(h[-1]).squeeze()


# -----------------------------------------------------------------------------
# 繪圖輔助函數
# -----------------------------------------------------------------------------
def plot_training_history(csv_path, save_path, title_name):
    try:
        df = pd.read_csv(csv_path)
        if df.empty: return
        epochs = df["Epoch"].values
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"Model Training Analysis: {title_name}", fontsize=14, fontweight='bold')
        
        ax1.plot(epochs, df["TrainTotalLoss"], label="Train Loss", color="tab:blue", linewidth=2)
        ax1.plot(epochs, df["ValLoss"], label="Val Loss", color="tab:red", linewidth=2)
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.set_title("Standard GRU Convergence")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend()
        
        ax2.plot(epochs, df["ValAcc"], label="Val Accuracy", color="tab:green", linewidth=2)
        ax2.plot(epochs, df["ValMacroF1"], label="Val Macro F1", color="tab:purple", linewidth=2)
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Score (0.0 ~ 1.0)")
        ax2.set_ylim(0.0, 1.05)
        ax2.set_title("Validation Metrics Evaluation")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="lower right")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"[Warning] Failed to generate training plot: {e}")


# -----------------------------------------------------------------------------
# Main 流程
# -----------------------------------------------------------------------------
def main():
    target_columns = [c.strip() for c in args.column.split(',')]
    input_dim = len(target_columns)
    
    base_path = f"../data_v{args.data_version}/K_Fold/fold_{args.fold}"
    target_base_path = f"../data_v{args.target_version}"

    # 命名標籤
    run_name = f"fold{args.fold}_layer_{args.layer}_hidden_{args.hidden}_lr_{args.lr}_dv_{args.data_version}_col_{args.column}_CORAL"
    
    log_dir = "train_logs"
    model_dir = f"models/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    if os.path.exists(os.path.join(model_dir, "last.pth")):
        print(f"[Skip] Already exists: {run_name}")
        return

    print(f"Start Z-Score FSA Training: {run_name}")

    # ===== 資料載入 =====
    train_n = FastCSVDataset(f"{base_path}/train/notTired", label=0, column_names=target_columns)
    train_t = FastCSVDataset(f"{base_path}/train/Tired", label=1, column_names=target_columns)
    train_o = FastCSVDataset(f"{base_path}/train/other", label=2, column_names=target_columns)
    
    # Target 端只取前五分鐘的資料 (做為個人基準校正用)
    train_target = FastCSVDataset(target_base_path, label=0, column_names=target_columns)

    val_n = FastCSVDataset(f"{base_path}/val/notTired", label=0, column_names=target_columns)
    val_t = FastCSVDataset(f"{base_path}/val/Tired", label=1, column_names=target_columns)
    val_o = FastCSVDataset(f"{base_path}/val/other", label=2, column_names=target_columns)

    source_datasets = [d for d in [train_n, train_t, train_o] if len(d) > 0]
    val_datasets = [d for d in [val_n, val_t, val_o] if len(d) > 0]

    if not source_datasets or len(train_target) == 0: 
        print(f"[Error] Missing Source or Target training data.")
        return

    source_ds = ConcatDataset(source_datasets)
    val_ds = ConcatDataset(val_datasets)
    
    # ==== ⚙️ 核心步驟：提取 source 的 notTired 與 Target 的前五分鐘，進行 Z-score 標準化對齊矩陣解算 ====
    alignment_mapper = FeatureStandardizationMapper(input_dim=input_dim)
    alignment_mapper.fit(train_n, train_target)

    # 訓練時只需要 Source Loader，模型只需學習標準的分類邊界
    source_loader = DataLoader(source_ds, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)

    # 初始化標準模型
    model = GRUModel(input_dim=input_dim, hidden=args.hidden, layers=args.layer, num_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 類別權重計算 (處理資料不平衡)
    num_notTired = len(train_n)
    num_Tired = len(train_t)
    num_Other = len(train_o)
    total_samples = num_notTired + num_Tired + num_Other
    weights = [total_samples / (3.0 * count) if count > 0 else 0.0 for count in [num_notTired, num_Tired, num_Other]]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')
    early_stop_cnt = 0
    csv_log_path = os.path.join(log_dir, f"{run_name}.csv")

    with open(csv_log_path, 'w') as f: 
        # 維持 7 欄位以相容你的外層腳本
        f.write("Epoch,TrainTotalLoss,TrainClassLoss,TrainDomainLoss,ValLoss,ValAcc,ValMacroF1\n")

    for ep in range(1, args.epoch + 1):
        # ---------- Train ----------
        model.train()
        tr_loss = 0.0
        
        # 乾淨的訓練迴圈：讓 GRU 在標準 Source 空間學出純淨的分類邊界
        for x_src, y_src in source_loader:
            x_src, y_src = x_src.to(device), y_src.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out_src = model(x_src)
                loss_src = criterion(out_src, y_src)
                total_loss = loss_src

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += total_loss.item()

        avg_tr_loss = tr_loss / len(source_loader)

        # ---------- Validation ----------
        model.eval()
        val_loss_sum = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast('cuda'):
                    class_out = model(x)
                    loss = criterion(class_out, y)
                    val_loss_sum += loss.item()
                
                preds = torch.argmax(class_out, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        avg_val_loss = val_loss_sum / len(val_loader)
        acc = accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

        with open(csv_log_path, 'a') as f:
            # DomainLoss 欄位填寫 0.0000 保持外層腳本相容
            f.write(f"{ep},{avg_tr_loss:.4f},{avg_tr_loss:.4f},0.0000,{avg_val_loss:.4f},{acc:.4f},{macro_f1:.4f}\n")

        # ---------- Early Stopping & Save Model ----------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_cnt = 0
            torch.save(model.state_dict(), f"{model_dir}/best.pth")
            
            # 同步保存解算出來的變換矩陣 W (縮放) 與 b (平移)
            # 推論端 (test.py) 載入後執行 X @ W + b 即可無縫完成 Z-Score 標準化對齊
            torch.save({'W': alignment_mapper.W, 'b': alignment_mapper.b}, f"{model_dir}/coral_weights.pth")
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= args.patience:
                print(f"Early stopping at epoch {ep}")
                break

    torch.save(model.state_dict(), f"{model_dir}/last.pth")
    with open(f"{model_dir}/train_done.flag", "w") as f: f.write("done")
    print(f"Z-Score FSA Training Complete. Best Val Loss: {best_val_loss:.4f}")

    img_save_path = os.path.join(model_dir, "training_metrics.png")
    plot_training_history(csv_path=csv_log_path, save_path=img_save_path, title_name=run_name)

if __name__ == "__main__":
    main()