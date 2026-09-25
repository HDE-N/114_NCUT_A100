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
# ==== DANN 新增參數 ====
parser.add_argument("--lambda_dann", type=float, default=1.0, help="對抗損失的權重強度")
parser.add_argument("--target_version", type=str, default='46/special_data', help="DANN 目標域資料夾路徑")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Dataset & DANN 核心組件
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


# 1. 梯度反轉層 (Gradient Reversal Layer) 實作
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)


# 2. 修改後的 DANN GRU 模型
class DANNGRUModel(nn.Module):
    def __init__(self, input_dim=1, hidden=128, layers=2, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, layers, batch_first=True, dropout=0.2)
        self.class_classifier = nn.Linear(hidden, num_classes)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 2)
        )

    def forward(self, x, alpha=1.0):
        self.gru.flatten_parameters()
        _, h = self.gru(x)
        feature = h[-1]
        
        class_output = self.class_classifier(feature).squeeze()
        
        reverse_feature = grad_reverse(feature, alpha)
        domain_output = self.domain_classifier(reverse_feature).squeeze()
        
        return class_output, domain_output

# -----------------------------------------------------------------------------
# 繪圖輔助函數
# -----------------------------------------------------------------------------
def plot_training_history(csv_path, save_path, title_name):
    """
    自日誌 CSV 檔案自動繪製 DANN 訓練曲線圖
    """
    try:
        df = pd.read_csv(csv_path)
        if df.empty: return
        
        epochs = df["Epoch"].values
        
        # 建立雙子圖畫布 (1行2列)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"Training Analysis: {title_name}", fontsize=14, fontweight='bold')
        
        # 圖一：損失函數曲線
        ax1.plot(epochs, df["TrainTotalLoss"], label="Train Total Loss", color="tab:gray", alpha=0.6, linestyle="--")
        ax1.plot(epochs, df["TrainClassLoss"], label="Train Class Loss", color="tab:blue", linewidth=2)
        ax1.plot(epochs, df["TrainDomainLoss"], label="Train Domain Loss", color="tab:orange", linewidth=2)
        ax1.plot(epochs, df["ValLoss"], label="Val Fatigue Loss", color="tab:red", linewidth=2)
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.set_title("DANN Convergence (Min-Max Game)")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend()
        
        # 圖二：驗證集指標曲線
        ax2.plot(epochs, df["ValAcc"], label="Val Accuracy", color="tab:green", linewidth=2)
        ax2.plot(epochs, df["ValMacroF1"], label="Val Macro F1", color="tab:purple", linewidth=2)
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Score (0.0 ~ 1.0)")
        ax2.set_ylim(0.0, 1.05)
        ax2.set_title("Validation Evaluation")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="lower right")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"--> Performance metric plot successfully saved to: {save_path}")
    except Exception as e:
        print(f"[Warning] Failed to generate training plot: {e}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    target_columns = [c.strip() for c in args.column.split(',')]
    input_dim = len(target_columns)
    
    # 原始 1、2 人的 K_Fold 標準路徑
    base_path = f"../data_v{args.data_version}/K_Fold/fold_{args.fold}"
    
    # 精準對接：直接指向大自動化腳本傳進來的 target_version 路徑 (即 ../data_v58/special_data)
    target_base_path = f"../data_v{args.target_version}"

    # 統一命名邏輯
    run_name = f"fold{args.fold}_layer_{args.layer}_hidden_{args.hidden}_lr_{args.lr}_dv_{args.data_version}_col_{args.column}_DANN"
    
    log_dir = "train_logs"
    model_dir = f"models/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    if os.path.exists(os.path.join(model_dir, "last.pth")):
        print(f"[Skip] Already exists: {run_name}")
        return

    print(f"Start DANN Training: {run_name}")
    print(f"Source Path: {base_path}")
    print(f"Target Path (攤平目錄): {target_base_path}")

    # ===== 資料載入 =====
    train_n = FastCSVDataset(f"{base_path}/train/notTired", label=0, column_names=target_columns)
    train_t = FastCSVDataset(f"{base_path}/train/Tired", label=1, column_names=target_columns)
    train_o = FastCSVDataset(f"{base_path}/train/other", label=2, column_names=target_columns)

    # Target Domain：直接讀取 special_data 資料夾下的所有 CSV
    train_target = FastCSVDataset(target_base_path, label=0, column_names=target_columns)

    # 驗證集
    val_n = FastCSVDataset(f"{base_path}/val/notTired", label=0, column_names=target_columns)
    val_t = FastCSVDataset(f"{base_path}/val/Tired", label=1, column_names=target_columns)
    val_o = FastCSVDataset(f"{base_path}/val/other", label=2, column_names=target_columns)

    source_datasets = [d for d in [train_n, train_t, train_o] if len(d) > 0]
    val_datasets = [d for d in [val_n, val_t, val_o] if len(d) > 0]

    if not source_datasets or len(train_target) == 0: 
        print(f"[Error] Missing Source or Target training data.")
        print(f"-> Source dataset segments found: {[len(d) for d in source_datasets]}")
        print(f"-> Target dataset path checked: {target_base_path}")
        print(f"-> Target dataset segments found: {len(train_target)}")
        return

    source_ds = ConcatDataset(source_datasets)
    val_ds = ConcatDataset(val_datasets)
    
    source_loader = DataLoader(source_ds, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=False)
    target_loader = DataLoader(train_target, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)

    # 初始化模型與優化器
    model = DANNGRUModel(input_dim=input_dim, hidden=args.hidden, layers=args.layer, num_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    num_notTired = len(train_n)
    num_Tired = len(train_t)
    num_Other = len(train_o)
    total_samples = num_notTired + num_Tired + num_Other
    weights = [total_samples / (3.0 * count) if count > 0 else 0.0 for count in [num_notTired, num_Tired, num_Other]]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion_class = nn.CrossEntropyLoss(weight=class_weights)
    criterion_domain = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')
    early_stop_cnt = 0
    csv_log_path = os.path.join(log_dir, f"{run_name}.csv")

    with open(csv_log_path, 'w') as f: 
        f.write("Epoch,TrainTotalLoss,TrainClassLoss,TrainDomainLoss,ValLoss,ValAcc,ValMacroF1\n")

    len_dataloader = min(len(source_loader), len(target_loader))
    total_steps = args.epoch * len_dataloader

    current_step = 0
    for ep in range(1, args.epoch + 1):
        # ---------- Train ----------
        model.train()
        tr_total_loss = 0.0
        tr_class_loss = 0.0
        tr_domain_loss = 0.0
        
        iter_target = iter(target_loader)
        for x_src, y_src in source_loader:
            try:
                x_tgt, _ = next(iter_target)
            except StopIteration:
                iter_target = iter(target_loader)
                x_tgt, _ = next(iter_target)
            
            p = float(current_step) / total_steps
            alpha = (2.0 / (1.0 + np.exp(-10 * p)) - 1.0) * args.lambda_dann
            current_step += 1

            x_src, y_src = x_src.to(device), y_src.to(device)
            x_tgt = x_tgt.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                # (1) Source Domain
                src_class_out, src_domain_out = model(x_src, alpha)
                loss_src_class = criterion_class(src_class_out, y_src)
                
                domain_label_src = torch.zeros(src_domain_out.size(0), dtype=torch.long).to(device)
                loss_src_domain = criterion_domain(src_domain_out, domain_label_src)
                
                # (2) Target Domain
                _, tgt_domain_out = model(x_tgt, alpha)
                
                domain_label_tgt = torch.ones(tgt_domain_out.size(0), dtype=torch.long).to(device)
                loss_tgt_domain = criterion_domain(tgt_domain_out, domain_label_tgt)
                
                loss_domain = loss_src_domain + loss_tgt_domain
                total_loss = loss_src_class + loss_domain

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            tr_total_loss += total_loss.item()
            tr_class_loss += loss_src_class.item()
            tr_domain_loss += loss_domain.item()

        avg_tr_total_loss = tr_total_loss / len_dataloader
        avg_tr_class_loss = tr_class_loss / len_dataloader
        avg_tr_domain_loss = tr_domain_loss / len_dataloader

        # ---------- Validation ----------
        model.eval()
        val_loss_sum = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast('cuda'):
                    class_out, _ = model(x, alpha=0.0)
                    loss = criterion_class(class_out, y)
                    val_loss_sum += loss.item()
                
                preds = torch.argmax(class_out, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        avg_val_loss = val_loss_sum / len(val_loader)
        acc = accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

        with open(csv_log_path, 'a') as f:
            f.write(f"{ep},{avg_tr_total_loss:.4f},{avg_tr_class_loss:.4f},{avg_tr_domain_loss:.4f},{avg_val_loss:.4f},{acc:.4f},{macro_f1:.4f}\n")

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
    print(f"DANN Training Complete. Best Val Loss: {best_val_loss:.4f}")

    # ==== ⚙️ 新增核心功能：訓練完成後立即自動繪圖 ====
    img_save_path = os.path.join(model_dir, "training_metrics.png")
    plot_training_history(csv_path=csv_log_path, save_path=img_save_path, title_name=run_name)

if __name__ == "__main__":
    main()