import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # [新增] 載入 LDA
import matplotlib.pyplot as plt
import seaborn as sns

plt.switch_backend('Agg')

# -----------------------------------------------------------------------------
# 參數設置
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--fold", type=int, default=1)
parser.add_argument("--column", type=str, default='acceleration_Y, gyro_Z')
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--layer", type=int, default=2)
parser.add_argument("--data_version", type=str, default='46')
parser.add_argument("--target_version", type=str, default='46/special_data', help="使用者校正數據(5分鐘)路徑")
parser.add_argument("--test_version", type=str, default='46/test_data_2', help="實際測試數據路徑")

parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epoch", type=int, default=1000)
parser.add_argument("--lambda_dann", type=float, default=1.0)

parser.add_argument("--alpha", type=float, default=0.05, help="動態閾值的寬容常數")
parser.add_argument("--ceiling", type=float, default=0.80, help="notTired 的最高保底門檻")
parser.add_argument("--floor", type=float, default=0.40, help="Tired 的最低觸發門檻")

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Logger 類別
# -----------------------------------------------------------------------------
class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# -----------------------------------------------------------------------------
# Dataset & Model
# -----------------------------------------------------------------------------
def infer_label(fname: str) -> int:
    f = fname.lower()
    if "nottired" in f: return 0
    elif "tired" in f: return 1
    elif "other" in f: return 2
    return -1

class TestDataset(Dataset):
    def __init__(self, folder, column_names, seq_len=960, only_notTired=False, calibration_mode=False):
        self.items = []
        if not os.path.exists(folder): return
            
        files = sorted(glob.glob(os.path.join(folder, "**/*.csv"), recursive=True))
        for p in files:
            try:
                if calibration_mode:
                    y_label = 0
                elif only_notTired:
                    y_label = 0
                else:
                    y_label = infer_label(os.path.basename(p))
                    if y_label == -1: continue
                    
                df = pd.read_csv(p, usecols=lambda x: x in column_names, dtype=np.float32)
                if all(col in df.columns for col in column_names):
                    x_data = df[column_names].values
                    if len(x_data) >= seq_len:
                        for s in range(0, len(x_data)-seq_len+1, seq_len):
                            seg = x_data[s:s+seq_len, :] 
                            self.items.append((seg, y_label, os.path.basename(p)))
            except Exception as e:
                pass

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): 
        seg, y, fname = self.items[idx]
        return torch.tensor(seg, dtype=torch.float32), torch.tensor(y, dtype=torch.long), fname

class GRUModel(nn.Module):
    def __init__(self, input_dim=1, hidden=128, layers=2, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        self.gru.flatten_parameters()
        _, h = self.gru(x)
        feature = h[-1]
        out = self.fc(feature).squeeze()
        return out, feature

# -----------------------------------------------------------------------------
# 繪圖 Helpers
# -----------------------------------------------------------------------------
def plot_cm(y_true, y_pred, labels, target_names, save_path, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, annot_kws={"size": 14})
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confidence_chart(all_y_true, all_preds, all_probs, target_names, save_path):
    try:
        true_class_probs = all_probs[np.arange(len(all_y_true)), all_y_true]
        bins = np.arange(0.0, 1.1, 0.1)
        bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
        color_map = {0: '#2ecc71', 1: '#e74c3c', 2: '#3498db'}
        pred_names = ['Pred: notTired', 'Pred: Tired', 'Pred: Other']
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        for c_idx, ax in enumerate(axes):
            class_mask = (all_y_true == c_idx)
            c_probs = true_class_probs[class_mask]
            c_preds = all_preds[class_mask]
            total_class_samples = len(c_probs)
            
            df_hist = pd.DataFrame(0.0, index=bin_labels, columns=[0, 1, 2])
            df_count = pd.DataFrame(0, index=bin_labels, columns=[0, 1, 2])
            
            if total_class_samples > 0:
                bin_indices = np.digitize(c_probs, bins) - 1
                bin_indices[bin_indices == 10] = 9
                for b_i in range(10):
                    for p_i in range(3):
                        match_count = np.sum((bin_indices == b_i) & (c_preds == p_i))
                        df_count.iloc[b_i, p_i] = match_count
                        df_hist.iloc[b_i, p_i] = (match_count / total_class_samples) * 100
            
            df_hist.plot(kind='bar', stacked=True, ax=ax, color=[color_map[0], color_map[1], color_map[2]], edgecolor='black', linewidth=0.5, width=0.7)
            
            flatten_props = df_hist.to_numpy().flatten()
            flatten_counts = df_count.to_numpy().flatten()
            for idx, rect in enumerate(ax.patches):
                height, width, x, y = rect.get_height(), rect.get_width(), rect.get_x(), rect.get_y()
                num_rows = len(bin_labels)
                col_ordered_idx = (idx // num_rows) + (idx % num_rows) * 3
                prop_val, count_val = flatten_props[col_ordered_idx], flatten_counts[col_ordered_idx]
                if prop_val > 1.0:
                    ax.text(x + width/2., y + height/2., f"{prop_val:.1f}%\n({int(count_val)})", ha='center', va='center', color='white', fontweight='bold', fontsize=8.5, bbox=dict(facecolor='black', alpha=0.3, boxstyle='round,pad=0.1', edgecolor='none'))
            
            ax.set_title(f"True Label: {target_names[c_idx]} (Total Samples: {total_class_samples})", fontsize=13, fontweight='bold', loc='left')
            ax.set_ylabel("Proportion (%)", fontsize=11, fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.set_ylim(0, 106)
            ax.set_yticks(np.arange(0, 101, 20))
            if c_idx == 0: ax.legend(pred_names, loc='upper left', fontsize=11)
            else: ax.get_legend().remove()

        plt.xlabel('Model Confidence Interval (Probability of True Class)', fontsize=12, fontweight='bold', labelpad=10)
        plt.xticks(rotation=0, ha='center', fontsize=11) 
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"[Warning] Failed to generate proportion plot: {e}")

# -----------------------------------------------------------------------------
# Main 推論測試
# -----------------------------------------------------------------------------
def main():
    target_columns = [c.strip() for c in args.column.split(',')]
    input_dim = len(target_columns)
    
    run_name = f"fold{args.fold}_layer_{args.layer}_hidden_{args.hidden}_lr_{args.lr}_ServerMaster"
    output_dir = "./test_log"
    os.makedirs(output_dir, exist_ok=True)
    
    test_run_name = f"{run_name}_EdgeAdapted"
    result_csv_path = os.path.join(output_dir, f"{test_run_name}_result.csv")
    log_file_path = os.path.join(output_dir, f"{test_run_name}_log.txt")
    sys.stdout = DualLogger(log_file_path)

    model_dir = f"models/{run_name}"
    ckpt_path = os.path.join(model_dir, "best_gru.pth")
    stats_path = os.path.join(model_dir, "edge_deployment_stats.pth")
    
    if not os.path.exists(ckpt_path) or not os.path.exists(stats_path):
        print(f"[Error] Missing model weights or deployment stats in {model_dir}")
        return

    print(f"Testing Model: {test_run_name}")

    # =========================================================
    # 階段 1：載入出廠參數
    # =========================================================
    model = GRUModel(input_dim=input_dim, hidden=args.hidden, layers=args.layer, num_classes=3).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    stats = torch.load(stats_path, map_location=device)
    global_mu = stats['global_mu'].to(device)
    global_std = stats['global_std'].to(device)
    coral_mu_src = stats['coral_mu_src'].to(device)
    coral_cov_src_sqrt = stats['coral_cov_src_sqrt'].to(device)

    # =========================================================
    # 階段 2：邊緣端個人化校正
    # =========================================================
    print("\n====== 正在執行邊緣端個人化校正 ======")
    calib_data_path = f"../data_v{args.target_version}"
    calib_ds = TestDataset(calib_data_path, target_columns, seq_len=960, calibration_mode=True)
    
    active_th = args.ceiling 
    base_active_th = args.ceiling 
    fallback_floor = 0.20 

    if len(calib_ds) == 0:
        print(f"[Warning] 找不到校正數據，放棄 CORAL 與動態閾值，使用基礎值 (門檻={active_th})")
        W_align = torch.eye(args.hidden).to(device)
        b_align = torch.zeros(args.hidden).to(device)
    else:
        calib_segs = torch.cat([calib_ds[i][0].unsqueeze(0) for i in range(len(calib_ds))], dim=0).to(device)
        calib_segs_norm = (calib_segs - global_mu) / global_std
        
        with torch.no_grad():
            base_calib_logits, calib_feat = model(calib_segs_norm)
            if base_calib_logits.dim() == 1: base_calib_logits = base_calib_logits.unsqueeze(0)
        
        base_calib_probs = torch.softmax(base_calib_logits, dim=1).cpu().numpy()
        base_notTired_probs = base_calib_probs[:, 0]
        base_p5 = np.percentile(base_notTired_probs, 5)
        base_dynamic_th = base_p5 - args.alpha
        base_active_th = max(min(base_dynamic_th, args.ceiling), fallback_floor)
        
        mu_tgt = calib_feat.mean(dim=0)
        tgt_centered = calib_feat - mu_tgt
        cov_tgt = (tgt_centered.T @ tgt_centered) / (calib_feat.size(0) - 1) + torch.eye(args.hidden).to(device) * 1e-5
        
        Ut, St, Vt = torch.linalg.svd(cov_tgt)
        cov_tgt_sqrt_inv = Ut @ torch.diag(1.0 / torch.sqrt(torch.clamp(St, min=1e-6))) @ Vt
        
        W_align = cov_tgt_sqrt_inv @ coral_cov_src_sqrt  
        b_align = coral_mu_src - (mu_tgt @ W_align)       
        
        calib_feat_aligned = calib_feat @ W_align + b_align
        with torch.no_grad():
            calib_logits = model.fc(calib_feat_aligned)
            if calib_logits.dim() == 1: calib_logits = calib_logits.unsqueeze(0)
        
        calib_probs = torch.softmax(calib_logits, dim=1).cpu().numpy()
        baseline_notTired_probs = calib_probs[:, 0]
        p5 = np.percentile(baseline_notTired_probs, 5)
        dynamic_th = p5 - args.alpha
        active_th = max(min(dynamic_th, args.ceiling), fallback_floor)

    # =========================================================
    # 階段 3：正式推論期 (記錄 4 種狀態消融對比)
    # =========================================================
    print("\n====== 開始正式推論評估 ======")
    test_data_path = f"../data_v{args.test_version}"
    test_ds = TestDataset(test_data_path, target_columns, seq_len=960, only_notTired=False)
    loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=1)

    all_y_true = []
    
    all_probs_base, all_preds_base = [], []         
    all_preds_base_adapt = []                       
    all_probs_align, all_preds_coral = [], []       
    all_preds_adapt = []                            

    # [新增] 用來儲存 GRU 的高維度特徵
    all_gru_feat_base = []
    all_gru_feat_aligned = []

    with torch.no_grad():
        for segs, y, _ in loader:
            segs = segs.to(device)
            
            segs_norm = (segs - global_mu) / global_std
            logits_base, feat_base = model(segs_norm)   
            if logits_base.dim() == 1: logits_base = logits_base.unsqueeze(0)
            
            batch_probs_base = torch.softmax(logits_base, dim=1).cpu().numpy()
            batch_preds_base = torch.argmax(logits_base, dim=1).cpu().numpy()
            
            for p, raw in zip(batch_probs_base, batch_preds_base):
                if raw == 2: base_adapted = 2
                elif (p[0] < base_active_th) and (p[1] > args.floor): base_adapted = 1
                else: base_adapted = 0
                all_preds_base_adapt.append(base_adapted)

            feat_aligned = feat_base @ W_align + b_align
            logits_align = model.fc(feat_aligned)       
            if logits_align.dim() == 1: logits_align = logits_align.unsqueeze(0)
            
            batch_probs_align = torch.softmax(logits_align, dim=1).cpu().numpy()
            batch_preds_coral = torch.argmax(logits_align, dim=1).cpu().numpy() 
            
            for p, raw in zip(batch_probs_align, batch_preds_coral):
                if raw == 2: adapted = 2
                elif (p[0] < active_th) and (p[1] > args.floor): adapted = 1
                else: adapted = 0
                all_preds_adapt.append(adapted)
            
            all_y_true.extend(y.numpy().tolist())
            all_probs_base.extend(batch_probs_base.tolist())
            all_preds_base.extend(batch_preds_base.tolist())
            all_probs_align.extend(batch_probs_align.tolist())
            all_preds_coral.extend(batch_preds_coral.tolist())

            # [新增] 收集 GRU 特徵 (轉回 CPU Numpy)
            all_gru_feat_base.append(feat_base.cpu().numpy())
            all_gru_feat_aligned.append(feat_aligned.cpu().numpy())

    all_y_true = np.array(all_y_true)
    all_preds_base = np.array(all_preds_base)
    all_probs_base = np.array(all_probs_base)
    all_preds_base_adapt = np.array(all_preds_base_adapt)
    all_preds_coral = np.array(all_preds_coral)
    all_probs_align = np.array(all_probs_align)
    all_preds_adapt = np.array(all_preds_adapt)
    
    # [新增] 拼接所有批次的特徵
    feat_base_np = np.vstack(all_gru_feat_base)
    feat_aligned_np = np.vstack(all_gru_feat_aligned)

    # =========================================================
    # [優化階段]：GRU 特徵之「等量平衡下採樣」LDA 分析與繪圖 (固定軸範圍版)
    # =========================================================
    print("\n====== 正在執行 GRU 特徵之「等量平衡」LDA 判別分析 (固定軸範圍) ======")
    try:
        # 1. 篩選出 Tired (1) 與 notTired (0) 的資料
        mask_01 = (all_y_true == 0) | (all_y_true == 1)
        
        y_filtered = all_y_true[mask_01]
        X_base_filtered = feat_base_np[mask_01]
        X_aligned_filtered = feat_aligned_np[mask_01]

        if len(np.unique(y_filtered)) > 1:
            # 2. 建立臨時的 DataFrame 進行結構化隨機抽樣
            df_temp = pd.DataFrame({
                'label': y_filtered,
                'idx': np.arange(len(y_filtered))
            })
            
            # 計算兩類各自的筆數，並找出最小筆數
            class_counts = df_temp['label'].value_counts()
            min_samples = class_counts.min()
            
            print(f"   📊 原始筆數 -> notTired(0): {class_counts[0]} 筆 | Tired(1): {class_counts[1]} 筆")
            print(f"   ⚖️ 啟動平衡機制：兩類各隨機抽取 {min_samples} 筆進行分析...")
            
            balanced_df = df_temp.groupby('label').sample(n=min_samples, random_state=42)
            balanced_indices = balanced_df['idx'].values
            
            # 3. 提取平衡後的特徵與標籤
            y_balanced = y_filtered[balanced_indices]
            X_base_balanced = X_base_filtered[balanced_indices]
            X_aligned_balanced = X_aligned_filtered[balanced_indices]

            # 4. 建立「唯一共享」的 LDA 模型
            lda_shared = LinearDiscriminantAnalysis(n_components=1)

            # 5. 用「平衡且對齊後 (CORAL)」的特徵建立黃金標準邊界
            lda_out_aligned = lda_shared.fit_transform(X_aligned_balanced, y_balanced)

            # 6. 用同一個模型轉換「平衡但未對齊 (Base)」的特徵
            lda_out_base = lda_shared.transform(X_base_balanced)

            # 將結果整理成繪圖用 DataFrame
            df_plot_lda = pd.DataFrame({
                'LDA_Base': lda_out_base[:, 0],
                'LDA_Aligned': lda_out_aligned[:, 0],
                'State': ['Tired' if label == 1 else 'notTired' for label in y_balanced]
            })

            # 7. 繪製對比圖
            fig_lda, axes_lda = plt.subplots(1, 2, figsize=(15, 6), sharey=True, sharex=True)
            custom_palette = {'notTired': '#1f77b4', 'Tired': '#d62728'}

            # 左圖：未對齊特徵 (平衡後)
            sns.kdeplot(data=df_plot_lda, x='LDA_Base', hue='State', palette=custom_palette, 
                        fill=True, alpha=0.4, linewidth=2, ax=axes_lda[0])
            axes_lda[0].set_title("Balanced Base Features (Unaligned)", fontsize=13, fontweight='bold')
            axes_lda[0].set_xlabel("Shared LDA Axis (1D)", fontsize=11)
            axes_lda[0].set_ylabel("Density", fontsize=11)

            # 右圖：經 CORAL 對齊特徵 (平衡後)
            sns.kdeplot(data=df_plot_lda, x='LDA_Aligned', hue='State', palette=custom_palette, 
                        fill=True, alpha=0.4, linewidth=2, ax=axes_lda[1])
            axes_lda[1].set_title("Balanced Aligned Features (CORAL)", fontsize=13, fontweight='bold')
            axes_lda[1].set_xlabel("Shared LDA Axis (1D)", fontsize=11)

            # 💡 【關鍵修改】: 強制依據要求鎖定 X 軸 與 Y 軸的顯示範圍
            axes_lda[0].set_xlim(-10, 10)
            axes_lda[1].set_xlim(-10, 10)
            axes_lda[0].set_ylim(0, 0.45)
            axes_lda[1].set_ylim(0, 0.45)

            plt.tight_layout()
            lda_save_path = os.path.join(output_dir, f"{test_run_name}_GRU_LDA_Balanced_Density.png")
            plt.savefig(lda_save_path, dpi=300)
            plt.close()
            print(f"🖼️ ✅ 嚴謹固定軸版（X: -4~10, Y: 0~0.45）LDA 對比圖已儲存至: {lda_save_path}")
        else:
            print("[Warning] 測試集中缺乏多種類別，無法執行 LDA。")
    except Exception as e:
        print(f"[Error] GRU 特徵 LDA 平衡繪圖失敗: {e}")

    # =========================================================
    # [新增階段]：2D 特徵空間視覺化 (僅顯示未對齊 Base 特徵)
    # =========================================================
    print("\n====== 正在繪製 2D 特徵散佈圖 (僅顯示轉換前) ======")
    try:
        from sklearn.decomposition import PCA

        # 1. 篩選出 Tired (1) 與 notTired (0) 的資料
        mask_01_2d = (all_y_true == 0) | (all_y_true == 1)
        y_2d = all_y_true[mask_01_2d]
        X_base_128d = feat_base_np[mask_01_2d]       # 僅取轉換前 B 的特徵

        if len(np.unique(y_2d)) > 1:
            # 2. 隨機下採樣 (各抽 2000 點最美觀，避免點太多變成一團黑影)
            df_temp_2d = pd.DataFrame({'label': y_2d, 'idx': np.arange(len(y_2d))})
            min_samples_2d = min(df_temp_2d['label'].value_counts().min(), 2000)
            
            balanced_indices_2d = df_temp_2d.groupby('label').sample(n=min_samples_2d, random_state=42)['idx'].values

            y_2d_plot = y_2d[balanced_indices_2d]
            X_base_plot = X_base_128d[balanced_indices_2d]

            # 3. 建立 2D 投影螢幕 (PCA)
            # 針對未對齊特徵尋找其最大變異方向
            pca_2d = PCA(n_components=2)
            X_base_2d = pca_2d.fit_transform(X_base_plot)

            # 4. 準備繪製散佈圖的 DataFrame
            df_base_2d = pd.DataFrame({
                'PC1': X_base_2d[:, 0], 'PC2': X_base_2d[:, 1], 
                'State': ['Tired' if l==1 else 'notTired' for l in y_2d_plot]
            })

            # 5. 繪製單張散佈圖 (Scatter Plot)
            plt.figure(figsize=(8, 7))
            custom_palette = {'notTired': '#1f77b4', 'Tired': '#d62728'}

            # 繪製未對齊的 B 特徵
            sns.scatterplot(data=df_base_2d, x='PC1', y='PC2', hue='State', 
                            palette=custom_palette, alpha=0.6, s=30, edgecolor=None)
            
            plt.title("Target Domain Base Features (Unaligned)", fontsize=15, fontweight='bold', pad=15)
            plt.xlabel("Principal Component 1 (2D)", fontsize=12, fontweight='bold')
            plt.ylabel("Principal Component 2 (2D)", fontsize=12, fontweight='bold')
            
            # 微調圖例位置與大小
            plt.legend(title='State', title_fontsize='11', fontsize='10', loc='best')

            # 💡 【除錯修正】：單張圖直接使用 plt.xlim 與 plt.ylim
            plt.xlim(-4, 2)
            plt.ylim(-2, 2.5)
            
            plt.tight_layout()
            # 儲存為獨立的新檔名，避免覆蓋掉之前的圖
            pca_save_path = os.path.join(output_dir, f"{test_run_name}_GRU_PCA2D_BaseOnly.png")
            plt.savefig(pca_save_path, dpi=300)
            plt.close()
            print(f"🖼️ ✅ 單張「轉換前」2D 特徵散佈圖已成功儲存至: {pca_save_path}")
            
    except Exception as e:
        print(f"[Error] 2D 特徵單張視覺化失敗: {e}")

    print("\n====== 正在繪製 2D 特徵散佈圖 (僅顯示轉換前) ======")
    try:
        from sklearn.decomposition import PCA

        # 1. 篩選出 Tired (1) 與 notTired (0) 的資料
        mask_01_2d = (all_y_true == 0) | (all_y_true == 1)
        y_2d = all_y_true[mask_01_2d]
        X_base_128d = feat_base_np[mask_01_2d]       # 僅取轉換前 B 的特徵

        if len(np.unique(y_2d)) > 1:
            # 2. 隨機下採樣 (各抽 2000 點最美觀，避免點太多變成一團黑影)
            df_temp_2d = pd.DataFrame({'label': y_2d, 'idx': np.arange(len(y_2d))})
            min_samples_2d = min(df_temp_2d['label'].value_counts().min(), 2000)
            
            balanced_indices_2d = df_temp_2d.groupby('label').sample(n=min_samples_2d, random_state=42)['idx'].values

            y_2d_plot = y_2d[balanced_indices_2d]
            X_base_plot = X_base_128d[balanced_indices_2d]

            # 3. 建立 2D 投影螢幕 (PCA)
            # 針對未對齊特徵尋找其最大變異方向
            pca_2d = PCA(n_components=2)
            X_base_2d = pca_2d.fit_transform(X_base_plot)

            # 4. 準備繪製散佈圖的 DataFrame
            df_base_2d = pd.DataFrame({
                'PC1': X_base_2d[:, 0], 'PC2': X_base_2d[:, 1], 
                'State': ['Tired' if l==1 else 'notTired' for l in y_2d_plot]
            })

            # 5. 繪製單張散佈圖 (Scatter Plot)
            plt.figure(figsize=(8, 7))
            custom_palette = {'notTired': '#1f77b4', 'Tired': '#d62728'}

            # 繪製未對齊的 B 特徵
            sns.scatterplot(data=df_base_2d, x='PC1', y='PC2', hue='State', 
                            palette=custom_palette, alpha=0.6, s=30, edgecolor=None)
            
            plt.title("Target Domain Base Features (Unaligned)", fontsize=15, fontweight='bold', pad=15)
            plt.xlabel("Principal Component 1 (2D)", fontsize=12, fontweight='bold')
            plt.ylabel("Principal Component 2 (2D)", fontsize=12, fontweight='bold')
            
            # 微調圖例位置與大小
            plt.legend(title='State', title_fontsize='11', fontsize='10', loc='best')

            # 💡 【除錯修正】：單張圖直接使用 plt.xlim 與 plt.ylim
            plt.xlim(-4, 2)
            plt.ylim(-2, 2.5)
            
            plt.tight_layout()
            # 儲存為獨立的新檔名，避免覆蓋掉之前的圖
            pca_save_path = os.path.join(output_dir, f"{test_run_name}_GRU_PCA2D_BaseOnly2.png")
            plt.savefig(pca_save_path, dpi=300)
            plt.close()
            print(f"🖼️ ✅ 單張「轉換前」2D 特徵散佈圖已成功儲存至: {pca_save_path}")
            
    except Exception as e:
        print(f"[Error] 2D 特徵單張視覺化失敗: {e}")
        
    # =========================================================
    # 指標評估與 CSV 紀錄設計 (維持原樣)
    # =========================================================
    classes = [0, 1, 2]
    target_names = ['notTired (0)', 'Tired (1)', 'Other (2)']
    y_true_binarized = label_binarize(all_y_true, classes=classes)

    def eval_and_get_dict(y_pred, y_probs, mode_name, prefix):
        acc = accuracy_score(all_y_true, y_pred)
        macro_f1 = f1_score(all_y_true, y_pred, average='macro', zero_division=0)
        report_dict = classification_report(all_y_true, y_pred, labels=classes, target_names=target_names, zero_division=0, output_dict=True)
        
        try:
            ap_per_class = average_precision_score(y_true_binarized, y_probs, average=None)
            mAP = np.mean(ap_per_class)
            ap_notTired = ap_per_class[0] if len(ap_per_class) > 0 else 0.0
            ap_Tired = ap_per_class[1] if len(ap_per_class) > 1 else 0.0
            mAP_notTired_Tired = np.nanmean([ap_notTired, ap_Tired]) if len(ap_per_class) >= 2 else 0.0
        except ValueError:
            ap_notTired = ap_Tired = mAP = mAP_notTired_Tired = 0.0

        print(f"\n================ [ {mode_name} ] ================")
        print(f"Overall Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f} | Overall mAP: {mAP:.4f}")
        for i, name in enumerate(target_names):
            p = report_dict[name]['precision']
            r = report_dict[name]['recall']
            f1 = report_dict[name]['f1-score']
            ap = ap_per_class[i] if i < len(ap_per_class) else 0.0
            print(f"[{i}] {name.split()[0]:<8} -> P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f} | AP: {ap:.4f}")
        
        return {
            f"{prefix}acc": acc, f"{prefix}macro_f1": macro_f1, f"{prefix}mAP": mAP, f"{prefix}mAP_nOT_T": mAP_notTired_Tired,
            f"{prefix}p_notTired": report_dict['notTired (0)']['precision'], f"{prefix}r_notTired": report_dict['notTired (0)']['recall'], f"{prefix}f1_notTired": report_dict['notTired (0)']['f1-score'], f"{prefix}ap_notTired": ap_notTired,
            f"{prefix}p_Tired": report_dict['Tired (1)']['precision'], f"{prefix}r_Tired": report_dict['Tired (1)']['recall'], f"{prefix}f1_Tired": report_dict['Tired (1)']['f1-score'], f"{prefix}ap_Tired": ap_Tired,
            f"{prefix}p_Other": report_dict['Other (2)']['precision'], f"{prefix}r_Other": report_dict['Other (2)']['recall'], f"{prefix}f1_Other": report_dict['Other (2)']['f1-score']
        }

    dict_base = eval_and_get_dict(all_preds_base, all_probs_base, "1. BASE (無自適應)", "base_")
    dict_base_adapt = eval_and_get_dict(all_preds_base_adapt, all_probs_base, f"2. BASE_ADAPT (僅做信心自適應_Th={base_active_th:.2f})", "base_adapt_")
    dict_coral = eval_and_get_dict(all_preds_coral, all_probs_align, "3. CORAL (純特徵對齊_Argmax)", "coral_")
    dict_adapt = eval_and_get_dict(all_preds_adapt, all_probs_align, f"4. ADAPT (特徵對齊+動態閾值_Th={active_th:.2f})", "adapt_")

    print("\n====== 正在產出 4 組圖表 ======")
    cm_base_path = os.path.join(output_dir, f"{test_run_name}_1_CM_BASE.png")
    cm_base_adapt_path = os.path.join(output_dir, f"{test_run_name}_2_CM_BASE_ADAPT.png")
    cm_coral_path = os.path.join(output_dir, f"{test_run_name}_3_CM_CORAL.png")
    cm_adapt_path = os.path.join(output_dir, f"{test_run_name}_4_CM_ADAPT.png")
    
    plot_cm(all_y_true, all_preds_base, classes, target_names, cm_base_path, "1. Base Predictions")
    plot_cm(all_y_true, all_preds_base_adapt, classes, target_names, cm_base_adapt_path, f"2. Base Adapted Predictions (Th={base_active_th:.2f})")
    plot_cm(all_y_true, all_preds_coral, classes, target_names, cm_coral_path, "3. CORAL Only Predictions")
    plot_cm(all_y_true, all_preds_adapt, classes, target_names, cm_adapt_path, f"4. Adapted Predictions (Th={active_th:.2f})")
    
    conf_base_path = os.path.join(output_dir, f"{test_run_name}_1_Conf_BASE.png")
    conf_base_adapt_path = os.path.join(output_dir, f"{test_run_name}_2_Conf_BASE_ADAPT.png")
    conf_coral_path = os.path.join(output_dir, f"{test_run_name}_3_Conf_CORAL.png")
    conf_adapt_path = os.path.join(output_dir, f"{test_run_name}_4_Conf_ADAPT.png")
    
    plot_confidence_chart(all_y_true, all_preds_base, all_probs_base, target_names, conf_base_path)
    plot_confidence_chart(all_y_true, all_preds_base_adapt, all_probs_base, target_names, conf_base_adapt_path)
    plot_confidence_chart(all_y_true, all_preds_coral, all_probs_align, target_names, conf_coral_path)
    plot_confidence_chart(all_y_true, all_preds_adapt, all_probs_align, target_names, conf_adapt_path)
    
    print(f"完成！圖片已儲存於 {output_dir}")

    final_result_dict = {**dict_base, **dict_base_adapt, **dict_coral, **dict_adapt}
    final_result_dict.update({
        "run_name": test_run_name,
        "num_segments": len(all_y_true),
        "base_threshold_used": base_active_th,
        "adapt_threshold_used": active_th
    })
    
    df = pd.DataFrame([final_result_dict])
    meta_cols = ['run_name', 'num_segments', 'base_threshold_used', 'adapt_threshold_used']
    other_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + other_cols]
    
    file_exists = os.path.isfile(result_csv_path)
    df.to_csv(result_csv_path, mode='a', header=not file_exists, index=False)
    
    print(f"Saved complete multi-stage result to {result_csv_path}")
    print("-" * 50)

if __name__ == "__main__":
    main()