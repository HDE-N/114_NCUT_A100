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
                    # 校正模式：不需真實標籤，預設給 0 且不跳過任何檔案，用來完整估計 Target 領域骨架
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

# =============================================================================
# 關鍵修正：適配新版 Temporal Attention（時間注意力機制）網路模型 
# =============================================================================
class GRUModel(nn.Module):
    def __init__(self, input_dim=1, hidden=128, layers=2, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, layers, batch_first=True, dropout=0.2)
        
        # 新增：用於動態評估時間步重要性的輕量注意力線性網路
        self.attention_net = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1, bias=False)
        )
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        self.gru.flatten_parameters()
        # out_seq Shape: [batch_size, seq_len, hidden] -> 包含全時段的隱藏狀態
        out_seq, _ = self.gru(x)
        
        # 1. 計算每個時間步的注意力原始分數 -> [batch_size, seq_len, 1]
        attn_scores = self.attention_net(out_seq)
        
        # 2. 透過 Softmax 歸一化權重，使 960 個時間點的權重分佈加總等於 1 -> [batch_size, seq_len, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # 3. 執行矩陣加權總和 (Weighted Sum)，智能過濾無關雜訊並壓平時間軸 -> [batch_size, hidden]
        feature = torch.sum(out_seq * attn_weights, dim=1)
        
        # 穩定輸出：保持二維 [batch_size, num_classes]，防止邊緣端單筆推推論出錯
        out = self.fc(feature)
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
    coral_mu_src = stats['coral_mu_src'].to(device)         # 全類別特徵中心
    coral_cov_src_sqrt = stats['coral_cov_src_sqrt'].to(device) # 全類別特徵共變異數根號矩陣

    # =========================================================
    # 階段 2：邊緣端個人化校正 (已完美相容 Attention 特徵雲)
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
        # 載入校正原始資料並進行 Z-Score 標準化
        calib_segs = torch.cat([calib_ds[i][0].unsqueeze(0) for i in range(len(calib_ds))], dim=0).to(device)
        calib_segs_norm = (calib_segs - global_mu) / global_std
        
        # 1. 透過目前模型提取校正資料之 GRU 空間總體特徵 (此處已自動套用 Temporal Attention)
        print("-> 正在經由 GRU 提取邊緣端 Target 領域總體特徵空間...")
        with torch.no_grad():
            base_calib_logits, calib_feat = model(calib_segs_norm)
            if base_calib_logits.dim() == 1: base_calib_logits = base_calib_logits.unsqueeze(0)
        
        # 2. 計算 Base 版的動態閾值 (無特徵對齊)
        base_calib_probs = torch.softmax(base_calib_logits, dim=1).cpu().numpy()
        base_notTired_probs = base_calib_probs[:, 0]
        base_p5 = np.percentile(base_notTired_probs, 5)
        base_dynamic_th = base_p5 - args.alpha
        base_active_th = max(min(base_dynamic_th, args.ceiling), fallback_floor)
        print(f"-> Base專屬閾值校正完成 | P5: {base_p5:.3f} | 動態計算值: {base_dynamic_th:.3f} | 實際啟用門檻: {base_active_th:.3f}")

        # 3. 在 GRU 特徵空間層面計算轉移對齊矩陣 (分佈 + 均值雙重對齊)
        mu_tgt = calib_feat.mean(dim=0) 
        tgt_centered = calib_feat - mu_tgt
        cov_tgt = (tgt_centered.T @ tgt_centered) / (calib_feat.size(0) - 1) + torch.eye(args.hidden).to(device) * 1e-5
        
        Ut, St, Vt = torch.linalg.svd(cov_tgt)
        cov_tgt_sqrt_inv = Ut @ torch.diag(1.0 / torch.sqrt(torch.clamp(St, min=1e-6))) @ Vt
        
        W_align = cov_tgt_sqrt_inv @ coral_cov_src_sqrt  
        b_align = coral_mu_src - (mu_tgt @ W_align)       
        print(f"-> 全域骨架特徵與均值對齊矩陣計算完成 ({len(calib_ds)} 片段，特徵維度: {args.hidden})")
        
        # 4. 計算 CORAL 版的動態閾值 (特徵經雙重對齊轉換後送入全連接層)
        calib_feat_aligned = calib_feat @ W_align + b_align
        with torch.no_grad():
            calib_logits = model.fc(calib_feat_aligned)
            if calib_logits.dim() == 1: calib_logits = calib_logits.unsqueeze(0)
        
        calib_probs = torch.softmax(calib_logits, dim=1).cpu().numpy()
        baseline_notTired_probs = calib_probs[:, 0]
        p5 = np.percentile(baseline_notTired_probs, 5)
        dynamic_th = p5 - args.alpha
        active_th = max(min(dynamic_th, args.ceiling), fallback_floor)
        print(f"-> CORAL閾值校正完成 | P5: {p5:.3f} | 動態計算值: {dynamic_th:.3f} | 實際啟用門檻: {active_th:.3f}")

    # =========================================================
    # 階段 3：正式推論期 (記錄 4 種狀態消融對比)
    # =========================================================
    print("\n====== 開始正式推論評估 (產出 4 組對比) ======")
    test_data_path = f"../data_v{args.test_version}"
    test_ds = TestDataset(test_data_path, target_columns, seq_len=960, only_notTired=False)
    loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=1)

    all_y_true = []
    
    # 消融實驗 4 大狀態
    all_probs_base, all_preds_base = [], []         # 狀態 1: Base (無自適應)
    all_preds_base_adapt = []                       # 狀態 2: Base Adapt (僅做信心自適應)
    all_probs_align, all_preds_coral = [], []       # 狀態 3: CORAL (純高維特徵對齊)
    all_preds_adapt = []                            # 狀態 4: Adapt (特徵對齊 + 動態閾值完全體)

    with torch.no_grad():
        for segs, y, _ in loader:
            segs = segs.to(device)
            
            # 先跑常規的 Z-Score 並進入 GRU 網路層提取注意力加權後的特徵
            segs_norm = (segs - global_mu) / global_std
            logits_base, feat_base = model(segs_norm)   
            if logits_base.dim() == 1: logits_base = logits_base.unsqueeze(0)
            
            batch_probs_base = torch.softmax(logits_base, dim=1).cpu().numpy()
            batch_preds_base = torch.argmax(logits_base, dim=1).cpu().numpy()
            
            # --- 狀態 2: 僅作信心自適應 (使用 Base 專屬閾值) ---
            for p, raw in zip(batch_probs_base, batch_preds_base):
                if raw == 2:
                    base_adapted = 2
                elif (p[0] < base_active_th) and (p[1] > args.floor):
                    base_adapted = 1
                else:
                    base_adapted = 0
                all_preds_base_adapt.append(base_adapted)

            # --- 狀態 3 & 4: 高維特徵空間雙重對齊與預測 ---
            feat_aligned = feat_base @ W_align + b_align
            logits_align = model.fc(feat_aligned)       
            if logits_align.dim() == 1: logits_align = logits_align.unsqueeze(0)
            
            batch_probs_align = torch.softmax(logits_align, dim=1).cpu().numpy()
            batch_preds_coral = torch.argmax(logits_align, dim=1).cpu().numpy() # 狀態 3
            
            # 狀態 4: 特徵雙重對齊 + 結合新版 CORAL 自適應閾值
            for p, raw in zip(batch_probs_align, batch_preds_coral):
                if raw == 2:
                    adapted = 2
                elif (p[0] < active_th) and (p[1] > args.floor):
                    adapted = 1
                else:
                    adapted = 0
                all_preds_adapt.append(adapted)
            
            # 資料快取聚合
            all_y_true.extend(y.numpy().tolist())
            all_probs_base.extend(batch_probs_base.tolist())
            all_preds_base.extend(batch_preds_base.tolist())
            all_probs_align.extend(batch_probs_align.tolist())
            all_preds_coral.extend(batch_preds_coral.tolist())

    all_y_true = np.array(all_y_true)
    all_preds_base = np.array(all_preds_base)
    all_probs_base = np.array(all_probs_base)
    all_preds_base_adapt = np.array(all_preds_base_adapt)
    all_preds_coral = np.array(all_preds_coral)
    all_probs_align = np.array(all_probs_align)
    all_preds_adapt = np.array(all_preds_adapt)

    # =========================================================
    # 指標評估與 CSV 紀錄設計
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

    # ---------------------------------------------------------
    # 生成 4 組圖表 (共 8 張圖片: 4 CM + 4 Conf)
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 將所有資訊整合儲存至 CSV (橫向擴充)
    # ---------------------------------------------------------
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