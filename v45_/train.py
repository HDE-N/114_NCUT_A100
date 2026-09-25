import os
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

# 設定 matplotlib 在無圖形介面環境下運作
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
parser.add_argument("--column", type=str, default='acceleration_Y, gyro_Z') 
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--layer", type=int, default=2)
parser.add_argument("--patience", type=int, default=40)
parser.add_argument("--data_version", type=str, default='46')
parser.add_argument("--lambda_dann", type=float, default=1.0, help="DANN 領域對抗 Loss 權重")
parser.add_argument("--target_version", type=str, default='46/special_data', help="用於對齊的邊緣端無標籤數據路徑")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Dataset 核心組件 (包含無標籤 Target 載入器)
# -----------------------------------------------------------------------------
class FastCSVDataset(Dataset):
    def __init__(self, folder_path, label, column_names, seq_len=960, is_target=False):
        self.data = None
        self.labels = None
        if not os.path.exists(folder_path): return
        
        # 支援 DANN 的 Target 模式（遞迴搜尋所有 CSV）
        if is_target:
            files = sorted(glob.glob(os.path.join(folder_path, "**/*.csv"), recursive=True))
        else:
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
            if res: all_segments.extend(res)

        if all_segments:
            self.data = torch.tensor(np.array(all_segments), dtype=torch.float32)
            self.labels = torch.tensor([label] * len(all_segments), dtype=torch.long)
            mode_str = "Target (Unlabeled)" if is_target else f"Source Folder {os.path.basename(folder_path)}"
            print(f"Loaded {mode_str}: Extracted {len(self.data)} segments.")

    def __len__(self): 
        return len(self.data) if self.data is not None else 0
        
    def __getitem__(self, idx): 
        return self.data[idx], self.labels[idx]

# -----------------------------------------------------------------------------
# DANN 核心：梯度反轉層 (Gradient Reversal Layer)
# -----------------------------------------------------------------------------
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

# -----------------------------------------------------------------------------
# 核心模型 - DANN GRU (結合 Average Pooling 與領域鑑別器)
# -----------------------------------------------------------------------------
class DANNGRUModel(nn.Module):
    def __init__(self, input_dim=1, hidden=128, layers=2, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, num_classes)
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 2) 
        )

    def forward(self, x, alpha=1.0):
        self.gru.flatten_parameters()
        out_seq, _ = self.gru(x)
        
        feature = torch.mean(out_seq, dim=1) 
        class_output = self.fc(feature)
        
        reversed_feature = GradientReversalFunction.apply(feature, alpha)
        domain_output = self.domain_classifier(reversed_feature)
        
        return class_output, feature, domain_output

# -----------------------------------------------------------------------------
# Main 流程
# -----------------------------------------------------------------------------
def main():
    target_columns = [c.strip() for c in args.column.split(',')]
    input_dim = len(target_columns)
    
    base_path = f"../data_v{args.data_version}/K_Fold/fold_{args.fold}"
    target_path = f"../data_v{args.target_version}"
    run_name = f"fold{args.fold}_layer_{args.layer}_hidden_{args.hidden}_lr_{args.lr}_DANN_AvgPool"
    
    log_dir = "train_logs"
    model_dir = f"models/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Start DANN + Average Pooling Training: {run_name}")

    # ===== 1. 資料載入 =====
    train_n = FastCSVDataset(f"{base_path}/train/notTired", label=0, column_names=target_columns)
    train_t = FastCSVDataset(f"{base_path}/train/Tired", label=1, column_names=target_columns)
    train_o = FastCSVDataset(f"{base_path}/train/other", label=2, column_names=target_columns)

    val_n = FastCSVDataset(f"{base_path}/val/notTired", label=0, column_names=target_columns)
    val_t = FastCSVDataset(f"{base_path}/val/Tired", label=1, column_names=target_columns)
    val_o = FastCSVDataset(f"{base_path}/val/other", label=2, column_names=target_columns)

    train_target = FastCSVDataset(target_path, label=0, column_names=target_columns, is_target=True)

    # ===== 2. 計算全域 Z-Score 正規化參數 =====
    train_datasets = [d for d in [train_n, train_t, train_o] if len(d) > 0]
    if not train_datasets or len(train_target) == 0:
        print("[Error] Missing training data or target adjustment data.")
        return

    all_train_tensors = [d.data.view(-1, input_dim) for d in train_datasets]
    all_train_flat = torch.cat(all_train_tensors, dim=0)
    global_mu = all_train_flat.mean(dim=0)
    global_std = all_train_flat.std(dim=0) + 1e-8 

    for d in [train_n, train_t, train_o, val_n, val_t, val_o, train_target]:
        if len(d) > 0: d.data = (d.data - global_mu) / global_std

    source_ds = ConcatDataset(train_datasets)
    val_ds = ConcatDataset([d for d in [val_n, val_t, val_o] if len(d) > 0])
    
    # 核心修正：避免小樣本校正集 drop_last 造成真空崩潰
    source_loader = DataLoader(source_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    target_loader = DataLoader(train_target, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ===== 3. 模型與權重初始化 =====
    model = DANNGRUModel(input_dim=input_dim, hidden=args.hidden, layers=args.layer, num_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    weights = [len(source_ds) / (3.0 * len(d)) if len(d) > 0 else 0.0 for d in [train_n, train_t, train_o]]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion_class = nn.CrossEntropyLoss(weight=class_weights)
    criterion_domain = nn.CrossEntropyLoss()
    
    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # ===== 4. 訓練迴圈 =====
    best_val_loss = float('inf')
    early_stop_cnt = 0
    
    # 初始化歷程容器
    history_tr_loss = []
    history_val_loss = []
    history_val_acc = []
    history_val_f1 = []

    for ep in range(1, args.epoch + 1):
        model.train()
        tr_loss = 0.0
        p = float(ep) / args.epoch
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        
        target_iter = iter(target_loader)
        
        for x_src, y_src in source_loader:
            x_src, y_src = x_src.to(device), y_src.to(device)
            
            try:
                x_tgt, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                x_tgt, _ = next(target_iter)
            x_tgt = x_tgt.to(device)
            
            optimizer.zero_grad()
            domain_y_src = torch.zeros(len(x_src), dtype=torch.long).to(device)
            domain_y_tgt = torch.ones(len(x_tgt), dtype=torch.long).to(device)
            
            if use_amp:
                with torch.amp.autocast(device_type=device.type):
                    out_src, _, d_out_src = model(x_src, alpha=alpha)
                    loss_cls = criterion_class(out_src, y_src)
                    loss_dom_src = criterion_domain(d_out_src, domain_y_src)
                    
                    _, _, d_out_tgt = model(x_tgt, alpha=alpha)
                    loss_dom_tgt = criterion_domain(d_out_tgt, domain_y_tgt)
                    
                    total_loss = loss_cls + args.lambda_dann * (loss_dom_src + loss_dom_tgt)
                
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_src, _, d_out_src = model(x_src, alpha=alpha)
                loss_cls = criterion_class(out_src, y_src)
                loss_dom_src = criterion_domain(d_out_src, domain_y_src)
                
                _, _, d_out_tgt = model(x_tgt, alpha=alpha)
                loss_dom_tgt = criterion_domain(d_out_tgt, domain_y_tgt)
                
                total_loss = loss_cls + args.lambda_dann * (loss_dom_src + loss_dom_tgt)
                total_loss.backward()
                optimizer.step()
                
            tr_loss += total_loss.item()

        # Validation 評估
        model.eval()
        val_loss_sum = 0.0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if use_amp:
                    with torch.amp.autocast(device_type=device.type):
                        class_out, _, _ = model(x)
                        loss = criterion_class(class_out, y)
                else:
                    class_out, _, _ = model(x)
                    loss = criterion_class(class_out, y)
                    
                val_loss_sum += loss.item()
                preds = torch.argmax(class_out, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        avg_tr_loss = tr_loss / len(source_loader)
        avg_val_loss = val_loss_sum / len(val_loader)
        val_acc = accuracy_score(all_targets, all_preds)
        val_f1 = f1_score(all_targets, all_preds, average='macro')

        # ✨ 關鍵修正一：將每輪數據老老實實裝進容器
        history_tr_loss.append(avg_tr_loss)
        history_val_loss.append(avg_val_loss)
        history_val_acc.append(val_acc)
        history_val_f1.append(val_f1)

        print(f"Epoch [{ep:03d}/{args.epoch}] (alpha={alpha:.2f}) - Total Loss: {avg_tr_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_cnt = 0
            torch.save(model.state_dict(), f"{model_dir}/best_gru.pth")
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= args.patience:
                print(f"Early stopping at epoch {ep}")
                break

    # ✨ 關鍵修正二：將歷程數據匯出成 CSV 與 趨勢圖表，防範 DEVNULL 吞掉數據
    print(f"\n====== 正在匯出訓練歷程與圖表至 {log_dir} ======")
    history_df = pd.DataFrame({
        'epoch': range(1, len(history_tr_loss) + 1),
        'train_total_loss': history_tr_loss,
        'val_loss': history_val_loss,
        'val_acc': history_val_acc,
        'val_f1': history_val_f1
    })
    history_df.to_csv(os.path.join(log_dir, f"{run_name}_history.csv"), index=False)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_tr_loss, label='Train Total Loss', color='blue')
    plt.plot(history_val_loss, label='Val Loss', color='orange')
    plt.title('DANN Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(history_val_acc, label='Val Accuracy', color='green')
    plt.plot(history_val_f1, label='Val F1 (Macro)', color='red')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plot_filepath = os.path.join(log_dir, f"{run_name}_metrics.png")
    plt.savefig(plot_filepath, dpi=300)
    plt.close()
    print(f"-> 歷程紀錄與圖表匯出完成。")

    # ===== 5. 核心提取：打包部署檔 =====
    print("\n====== 正在提取 DANN 對齊後的特徵基準統計量 ======")
    if len(source_ds) > 0:
        model.load_state_dict(torch.load(f"{model_dir}/best_gru.pth"))
        model.eval() 
        
        extract_loader = DataLoader(source_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        all_features = []
        
        with torch.no_grad():
            for x_batch, _ in extract_loader:
                x_batch = x_batch.to(device)
                _, feat, _ = model(x_batch)
                all_features.append(feat.cpu())
        
        src_all_feat = torch.cat(all_features, dim=0).to(device)
        feat_dim = src_all_feat.size(1) 
        
        coral_mu_src = src_all_feat.mean(dim=0)
        src_centered = src_all_feat - coral_mu_src
        coral_cov_src = (src_centered.T @ src_centered) / (src_all_feat.size(0) - 1) + torch.eye(feat_dim).to(device) * 1e-1
        
        Us, Ss, Vs = torch.linalg.svd(coral_cov_src)
        coral_cov_src_sqrt = Us @ torch.diag(torch.sqrt(torch.clamp(Ss, min=1e-6))) @ Vs
        
        edge_deployment_package = {
            'global_mu': global_mu.cpu(),
            'global_std': global_std.cpu(),
            'coral_mu_src': coral_mu_src.cpu(),
            'coral_cov_src_sqrt': coral_cov_src_sqrt.cpu()
        }
        torch.save(edge_deployment_package, f"{model_dir}/edge_deployment_stats.pth")
        print("-> 已成功匯出 edge_deployment_stats.pth")

if __name__ == "__main__":
    main()