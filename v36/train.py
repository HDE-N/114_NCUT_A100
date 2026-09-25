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
parser.add_argument("--patience", type=int, default=40)
parser.add_argument("--data_version", type=str, default='46')
# ==== 配合自動化網格腳本，保留參數名稱但功能轉向統計映射 ====
parser.add_argument("--lambda_dann", type=float, default=1.0, help="保留參數槽以相容自動化腳本")
parser.add_argument("--target_version", type=str, default='46/special_data', help="新使用者前5分鐘數據的資料夾路徑")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Dataset 核心組件 (滑動視窗自動切割)
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
# 統計學前置映射小模型 (CORAL Alignment Network)
# -----------------------------------------------------------------------------
class CORALMappingNetwork:
    def __init__(self, input_dim=6):
        self.input_dim = input_dim
        self.W = torch.eye(input_dim).to(device)
        self.b = torch.zeros(input_dim).to(device)

    def fit(self, source_dataset, target_dataset):
        """
        利用線性代數閉式解，直接計算出將 Target 分佈拉向 Source 分佈的變換矩陣 W 與 b
        """
        print("====== 正在優化統計學前置映射網路 (CORAL) ======")
        # 1. 提取並平移 Source 數據
        src_list = [source_dataset[i][0] for i in range(len(source_dataset))]
        src_data = torch.cat(src_list, dim=0).to(device) # [N*960, 6]
        mu_src = torch.mean(src_data, dim=0)
        src_centered = src_data - mu_src
        cov_src = (src_centered.T @ src_centered) / (src_data.size(0) - 1) + torch.eye(self.input_dim).to(device) * 1e-5

        # 2. 提取並平移 Target (新使用者前 5 分鐘) 數據
        tgt_list = [target_dataset[i][0] for i in range(len(target_dataset))]
        tgt_data = torch.cat(tgt_list, dim=0).to(device) # [M*960, 6]
        mu_tgt = torch.mean(tgt_data, dim=0)
        tgt_centered = tgt_data - mu_tgt
        cov_tgt = (tgt_centered.T @ tgt_centered) / (tgt_data.size(0) - 1) + torch.eye(self.input_dim).to(device) * 1e-5

        # 3. 核心數學對齊：利用奇異值分解 (SVD) 進行白化與著色變換
        try:
            Ut, St, Vt = torch.linalg.svd(cov_tgt)
            cov_tgt_sqrt_inv = Ut @ torch.diag(1.0 / torch.sqrt(St)) @ Vt
            
            Us, Ss, Vs = torch.linalg.svd(cov_src)
            cov_src_sqrt = Us @ torch.diag(torch.sqrt(Ss)) @ Vs
            
            self.W = cov_tgt_sqrt_inv @ cov_source_sqrt
            self.b = mu_src - (mu_tgt @ self.W)
            print("-> CORAL 映射矩陣解算成功！")
        except:
            # 防呆退路：若矩陣退化，退回標準標準差縮放
            print("-> 觸發防呆機制：採用標準方差縮放對齊")
            std_src = torch.sqrt(torch.diagonal(cov_src))
            std_tgt = torch.sqrt(torch.diagonal(cov_tgt))
            self.W = torch.diag(std_src / (std_tgt + 1e-5))
            self.b = mu_src - (mu_tgt @ self.W)

    def forward(self, x):
        # 對齊最後的 6 通道維度: X_new = X @ W + b
        return x @ self.W + self.b


# -----------------------------------------------------------------------------
# 核心核心模型 - 標準 GRU 網路本體 (功耗極低、部署極速)
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
        fig.suptitle(f"CORAL Alignment Training Analysis: {title_name}", fontsize=14, fontweight='bold')
        
        ax1.plot(epochs, df["TrainTotalLoss"], label="Train Class Loss", color="tab:blue", linewidth=2)
        ax1.plot(epochs, df["ValLoss"], label="Val Fatigue Loss", color="tab:red", linewidth=2)
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

    # 修改命名後綴，標記為 CORAL 映射模式
    run_name = f"fold{args.fold}_layer_{args.layer}_hidden_{args.hidden}_lr_{args.lr}_dv_{args.data_version}_col_{args.column}_CORAL"
    
    log_dir = "train_logs"
    model_dir = f"models/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    if os.path.exists(os.path.join(model_dir, "last.pth")):
        print(f"[Skip] Already exists: {run_name}")
        return

    print(f"Start CORAL Mapping Training: {run_name}")

    # ===== 資料載入 =====
    train_n = FastCSVDataset(f"{base_path}/train/notTired", label=0, column_names=target_columns)
    train_t = FastCSVDataset(f"{base_path}/train/Tired", label=1, column_names=target_columns)
    train_o = FastCSVDataset(f"{base_path}/train/other", label=2, column_names=target_columns)
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
    
    # ==== ⚙️ 核心步驟：在訓練前直接完成前置映射小模型的優化 ====
    coral_mapper = CORALMappingNetwork(input_dim=input_dim)
    coral_mapper.fit(source_ds, train_target)

    source_loader = DataLoader(source_ds, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=False)
    target_loader = DataLoader(train_target, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)

    # 初始化標準模型
    model = GRUModel(input_dim=input_dim, hidden=args.hidden, layers=args.layer, num_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
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
        f.write("Epoch,TrainTotalLoss,TrainClassLoss,TrainDomainLoss,ValLoss,ValAcc,ValMacroF1\n")

    for ep in range(1, args.epoch + 1):
        # ---------- Train ----------
        model.train()
        tr_loss = 0.0
        
        iter_target = iter(target_loader)
        for x_src, y_src in source_loader:
            try:
                x_tgt, _ = next(iter_target)
            except StopIteration:
                iter_target = iter(target_loader)
                x_tgt, _ = next(iter_target)
            
            x_src, y_src = x_src.to(device), y_src.to(device)
            x_tgt = x_tgt.to(device)
            
            # 💡 【核心插刀位置】：將 Target 數據通過小模型映射，偽裝成標準分布
            x_tgt_aligned = coral_mapper.forward(x_tgt)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                # 混合訓練：核心模型同時學習原始數據與對齊後的新人數據
                out_src = model(x_src)
                loss_src = criterion(out_src, y_src)
                
                # 新人數據此時只有不累(Label 0)
                out_tgt = model(x_tgt_aligned)
                y_tgt_dummy = torch.zeros(x_tgt_aligned.size(0), dtype=torch.long).to(device)
                loss_tgt = criterion(out_tgt, y_tgt_dummy)
                
                # 總聯合損失
                total_loss = loss_src + loss_tgt

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
            # 配合大網格腳本欄位，維持 7 欄位寫入 (把無用的 Slot 填入 0)
            f.write(f"{ep},{avg_tr_loss:.4f},{avg_tr_loss:.4f},0.0000,{avg_val_loss:.4f},{acc:.4f},{macro_f1:.4f}\n")

        # ---------- Early Stopping & Save Model ----------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_cnt = 0
            torch.save(model.state_dict(), f"{model_dir}/best.pth")
            # 同步保存解算出來的變換矩陣，以便推論端加載
            torch.save({'W': coral_mapper.W, 'b': coral_mapper.b}, f"{model_dir}/coral_weights.pth")
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= args.patience:
                print(f"Early stopping at epoch {ep}")
                break

    torch.save(model.state_dict(), f"{model_dir}/last.pth")
    with open(f"{model_dir}/train_done.flag", "w") as f: f.write("done")
    print(f"CORAL GRU Training Complete. Best Val Loss: {best_val_loss:.4f}")

    img_save_path = os.path.join(model_dir, "training_metrics.png")
    plot_training_history(csv_path=csv_log_path, save_path=img_save_path, title_name=run_name)

if __name__ == "__main__":
    main()