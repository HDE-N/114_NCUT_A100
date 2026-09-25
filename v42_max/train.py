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
parser.add_argument("--column", type=str, default='acceleration_Y, gyro_Z') # 範例通道
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--layer", type=int, default=2)
parser.add_argument("--patience", type=int, default=40)
parser.add_argument("--data_version", type=str, default='46')
parser.add_argument("--lambda_dann", type=float, default=1.0, help="保留網格參數槽以防報錯")
parser.add_argument("--target_version", type=str, default='46/special_data', help="保留網格參數槽以防報錯")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Dataset 核心組件
# -----------------------------------------------------------------------------
class FastCSVDataset(Dataset):
    def __init__(self, folder_path, label, column_names, seq_len=960):
        self.data = None
        self.labels = None
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

    def __len__(self): 
        return len(self.data) if self.data is not None else 0
        
    def __getitem__(self, idx): 
        return self.data[idx], self.labels[idx]

# -----------------------------------------------------------------------------
# 核心模型 - 標準 GRU (同時輸出分類與高維特徵)
# -----------------------------------------------------------------------------
class GRUModel(nn.Module):
    def __init__(self, input_dim=1, hidden=128, layers=2, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        self.gru.flatten_parameters()
        # out 的 Shape: [batch_size, seq_len, hidden] -> 包含所有時間步的特徵
        out_seq, _ = self.gru(x)
        
        # 沿著時間軸 (dim=1) 取最大值，強制作保留整個序列中最強烈的特徵
        # torch.max 回傳 (values, indices)，我們只需要 values
        feature, _ = torch.max(out_seq, dim=1) 
        
        # 分類器輸出
        out = self.fc(feature).squeeze()
        return out, feature

# -----------------------------------------------------------------------------
# Main 流程
# -----------------------------------------------------------------------------
def main():
    target_columns = [c.strip() for c in args.column.split(',')]
    input_dim = len(target_columns)
    
    base_path = f"../data_v{args.data_version}/K_Fold/fold_{args.fold}"
    run_name = f"fold{args.fold}_layer_{args.layer}_hidden_{args.hidden}_lr_{args.lr}_ServerMaster"
    
    log_dir = "train_logs"
    model_dir = f"models/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Start Server Master Model Training: {run_name}")

    # ===== 1. 資料載入 =====
    train_n = FastCSVDataset(f"{base_path}/train/notTired", label=0, column_names=target_columns)
    train_t = FastCSVDataset(f"{base_path}/train/Tired", label=1, column_names=target_columns)
    train_o = FastCSVDataset(f"{base_path}/train/other", label=2, column_names=target_columns)

    val_n = FastCSVDataset(f"{base_path}/val/notTired", label=0, column_names=target_columns)
    val_t = FastCSVDataset(f"{base_path}/val/Tired", label=1, column_names=target_columns)
    val_o = FastCSVDataset(f"{base_path}/val/other", label=2, column_names=target_columns)

    # ===== 2. 計算全域 Z-Score 正規化參數 (對原始輸入資料) =====
    print("====== 計算並套用全域 Z-Score 正規化 ======")
    train_datasets = [d for d in [train_n, train_t, train_o] if len(d) > 0]
    if not train_datasets:
        print("[Error] Missing Source training data.")
        return

    all_train_tensors = [d.data.view(-1, input_dim) for d in train_datasets]
    all_train_flat = torch.cat(all_train_tensors, dim=0)
    
    global_mu = all_train_flat.mean(dim=0)
    global_std = all_train_flat.std(dim=0) + 1e-8 

    for d in [train_n, train_t, train_o, val_n, val_t, val_o]:
        if len(d) > 0:
            d.data = (d.data - global_mu) / global_std

    source_ds = ConcatDataset(train_datasets)
    val_ds = ConcatDataset([d for d in [val_n, val_t, val_o] if len(d) > 0])
    
    source_loader = DataLoader(source_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ===== 3. 模型初始化與類別權重 =====
    model = GRUModel(input_dim=input_dim, hidden=args.hidden, layers=args.layer, num_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    num_notTired = len(train_n)
    num_Tired = len(train_t)
    num_Other = len(train_o)
    total_samples = num_notTired + num_Tired + num_Other
    weights = [total_samples / (3.0 * count) if count > 0 else 0.0 for count in [num_notTired, num_Tired, num_Other]]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # ===== 4. 訓練迴圈 =====
    best_val_loss = float('inf')
    early_stop_cnt = 0

    history_tr_loss = []
    history_val_loss = []
    history_val_acc = []
    history_val_f1 = []

    for ep in range(1, args.epoch + 1):
        model.train()
        tr_loss = 0.0
        
        for x_src, y_src in source_loader:
            x_src, y_src = x_src.to(device), y_src.to(device)
            optimizer.zero_grad()
            
            if use_amp:
                with torch.amp.autocast(device.type):
                    out_src, _ = model(x_src) 
                    loss_src = criterion(out_src, y_src)
                scaler.scale(loss_src).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_src, _ = model(x_src)
                loss_src = criterion(out_src, y_src)
                loss_src.backward()
                optimizer.step()
                
            tr_loss += loss_src.item()

        # Validation
        model.eval()
        val_loss_sum = 0.0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if use_amp:
                    with torch.amp.autocast(device.type):
                        class_out, _ = model(x)
                        loss = criterion(class_out, y)
                else:
                    class_out, _ = model(x)
                    loss = criterion(class_out, y)
                    
                val_loss_sum += loss.item()
                preds = torch.argmax(class_out, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        avg_tr_loss = tr_loss / len(source_loader)
        avg_val_loss = val_loss_sum / len(val_loader)
        val_acc = accuracy_score(all_targets, all_preds)
        val_f1 = f1_score(all_targets, all_preds, average='macro')

        history_tr_loss.append(avg_tr_loss)
        history_val_loss.append(avg_val_loss)
        history_val_acc.append(val_acc)
        history_val_f1.append(val_f1)

        print(f"Epoch [{ep:03d}/{args.epoch}] - Tr Loss: {avg_tr_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_cnt = 0
            torch.save(model.state_dict(), f"{model_dir}/best_gru.pth")
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= args.patience:
                print(f"Early stopping at epoch {ep}")
                break

    # ===== 繪製並匯出訓練圖表 =====
    print(f"\n====== 正在生成訓練圖表至 {log_dir} ======")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_tr_loss, label='Train Loss', color='blue')
    plt.plot(history_val_loss, label='Val Loss', color='orange')
    plt.title('Training & Validation Loss')
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
    print(f"-> 已成功儲存圖表：{plot_filepath}")

    # ===== 5. 核心：提取並打包邊緣端部署所需的所有參數 (改為全域骨架特齊 + 均值對齊) =====
    print("\n====== 正在從已訓練完畢之 GRU 提取全域骨架（全類別）的特徵基準統計量 ======")
    if len(source_ds) > 0:
        # 載入最優權重
        model.load_state_dict(torch.load(f"{model_dir}/best_gru.pth"))
        model.eval() 
        
        # 關鍵修正：改用包含 notTired, Tired, other 的全體訓練集數據提取統計量
        extract_loader = DataLoader(source_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        all_features = []
        
        print("-> 正在將全類別原始訊號經由 GRU 轉換為總體高維特徵骨架...")
        with torch.no_grad():
            for x_batch, _ in extract_loader:
                x_batch = x_batch.to(device)
                _, feat = model(x_batch)  # 取得 [batch_size, hidden] 的特徵
                all_features.append(feat.cpu())
        
        src_all_feat = torch.cat(all_features, dim=0).to(device)
        feat_dim = src_all_feat.size(1) 
        
        # 1. 計算全域特徵均值向量 (作為後續邊緣端「均值對齊」的黃金中心點)
        coral_mu_src = src_all_feat.mean(dim=0)
        src_centered = src_all_feat - coral_mu_src
        
        # 2. 計算全域特徵共變異數矩陣 (保留您設定的 1e-1 穩定常數)
        coral_cov_src = (src_centered.T @ src_centered) / (src_all_feat.size(0) - 1) + torch.eye(feat_dim).to(device) * 1e-1
        
        # 3. 奇異值分解 (SVD) 計算特徵根號矩陣
        Us, Ss, Vs = torch.linalg.svd(coral_cov_src)
        coral_cov_src_sqrt = Us @ torch.diag(torch.sqrt(torch.clamp(Ss, min=1e-6))) @ Vs
        
        # 打包部署檔 (包含全域輸入正規化、全域特徵幾何中心、全域特徵共變異數)
        edge_deployment_package = {
            'global_mu': global_mu.cpu(),
            'global_std': global_std.cpu(),
            'coral_mu_src': coral_mu_src.cpu(),         # 全類別特徵均值
            'coral_cov_src_sqrt': coral_cov_src_sqrt.cpu() # 全類別特徵共變異數根號
        }
        torch.save(edge_deployment_package, f"{model_dir}/edge_deployment_stats.pth")
        
        print("-> 已成功匯出基於全域特徵空間（分佈+均值）的 edge_deployment_stats.pth")
        print(f"-> 目前黃金統計量特徵維度大小 (Size) 為: [{feat_dim}, {feat_dim}]")
        print("-> 請將 best_gru.pth 與此統計檔一同部署至設備。")

    print(f"Server Master Training Complete. Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()