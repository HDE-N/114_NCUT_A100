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

# 設定 matplotlib 後端以避免在無 GUI 環境下報錯
plt.switch_backend('Agg')

# 參數設置
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--fold", type=int, default=1)
parser.add_argument("--column", type=str, default='acceleration_Y')
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--layer", type=int, default=2)
parser.add_argument("--patience", type=int, default=30)
parser.add_argument("--data_version", type=str, default='46')

# 保留參數槽，確保相容網格搜尋大腳本
parser.add_argument("--lambda_dann", type=float, default=1.0, help="保留參數槽以相容自動化腳本")
parser.add_argument("--target_version", type=str, default='46/special_data', help="DANN 目標域資料夾路徑")
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
    if "nottired" in f: 
        return 0
    elif "tired" in f: 
        return 1
    elif "other" in f: 
        return 2
    return -1

class TestDataset(Dataset):
    def __init__(self, folder, column_names, seq_len=960):
        self.items = []
        if not os.path.exists(folder): 
            return
            
        files = glob.glob(os.path.join(folder, "**/*.csv"), recursive=True)
        
        for p in files:
            try:
                df = pd.read_csv(p, usecols=lambda x: x in column_names, dtype=np.float32)
                if all(col in df.columns for col in column_names):
                    y_label = infer_label(os.path.basename(p))
                    if y_label == -1:
                        continue
                        
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
        return self.fc(h[-1]).squeeze()

# -----------------------------------------------------------------------------
# Main 推論測試
# -----------------------------------------------------------------------------
def main():
    target_columns = [c.strip() for c in args.column.split(',')]
    input_dim = len(target_columns)
    
    # Run Name
    run_name = f"fold{args.fold}_layer_{args.layer}_hidden_{args.hidden}_lr_{args.lr}_dv_{args.data_version}_col_{args.column}_CORAL"
    
    output_dir = "./test_log"
    os.makedirs(output_dir, exist_ok=True)
    
    result_csv_path = os.path.join(output_dir, f"{run_name}_result.csv")
    log_file_path = os.path.join(output_dir, f"{run_name}_log.txt")
    cm_img_path = os.path.join(output_dir, f"{run_name}_cm.png") 
    conf_img_path = os.path.join(output_dir, f"{run_name}_confidence.png") 
    
    sys.stdout = DualLogger(log_file_path)

    # 權重與對齊矩陣路徑
    model_dir = f"models/{run_name}"
    ckpt_path = os.path.join(model_dir, "best.pth")
    coral_path = os.path.join(model_dir, "coral_weights.pth")
    
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(model_dir, "last.pth")
        if not os.path.exists(ckpt_path):
            print(f"[Error] No weights found for {run_name}")
            return

    print(f"Testing CORAL Optimization Model (Evaluation Unit: Segment): {run_name}")
    print(f"Loading weights from: {ckpt_path}")

    # 測試集資料夾路徑
    test_data_path = f"../data_v{args.data_version}/test_data_2"
    test_ds = TestDataset(test_data_path, target_columns, seq_len=960)
    if len(test_ds) == 0:
        print(f"[Error] No test data found in path: {test_data_path}")
        return
    loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=1)

    # 模型載入 (標準 GRU)
    model = GRUModel(input_dim=input_dim, hidden=args.hidden, layers=args.layer, num_classes=3).to(device)
    
    # 加載核心 GRU 權重
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print("-> GRU 核心模型權重載入成功！")
    except Exception as e:
        print(f"[Error] Model load failed: {e}")
        return

    # 自動加載解算出來的 CORAL 變換矩陣 W 與 b
    if os.path.exists(coral_path):
        coral_weights = torch.load(coral_path, map_location=device)
        W_coral = coral_weights['W'].to(device)
        b_coral = coral_weights['b'].to(device)
        print("-> CORAL 空間變換矩陣載入成功！(推論時將自動執行即時分佈整容)")
        has_coral = True
    else:
        print("[Warning] 找不到 coral_weights.pth，將使用原始資料直接進行推論。")
        has_coral = False

    model.eval()

    all_y_true = []
    all_preds = []
    all_probs = []  

    with torch.no_grad():
        for segs, y, _ in loader:
            segs = segs.to(device)
            
            # 如果存在變換矩陣，推論前先將測試集訊號映射至標準空間中
            if has_coral:
                segs = segs @ W_coral + b_coral
                
            logits = model(segs)
            
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
                
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_y_true.extend(y.numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    # 轉為 numpy array 計算指標
    all_y_true = np.array(all_y_true)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # ---------------------------------------------------------
    # 指標核心計算與擴充
    # ---------------------------------------------------------
    classes = [0, 1, 2]
    target_names = ['notTired (0)', 'Tired (1)', 'Other (2)']
    
    # 1. 基礎整體指標
    acc = accuracy_score(all_y_true, all_preds)
    macro_f1 = f1_score(all_y_true, all_preds, average='macro', zero_division=0)
    
    # 2. 透過 classification_report 完整抓取各類別的 P, R, F1 與整體 Macro/Weighted 平均
    report_dict = classification_report(
        all_y_true, all_preds, 
        labels=classes, 
        target_names=target_names, 
        zero_division=0, 
        output_dict=True
    )
    
    # 抽取各類別細部 PRF1 指標
    p_notTired = report_dict['notTired (0)']['precision']
    r_notTired = report_dict['notTired (0)']['recall']
    f1_notTired = report_dict['notTired (0)']['f1-score']

    p_Tired = report_dict['Tired (1)']['precision']
    r_Tired = report_dict['Tired (1)']['recall']
    f1_Tired = report_dict['Tired (1)']['f1-score']

    p_Other = report_dict['Other (2)']['precision']
    r_Other = report_dict['Other (2)']['recall']
    f1_Other = report_dict['Other (2)']['f1-score']
    
    macro_p = report_dict['macro avg']['precision']
    macro_r = report_dict['macro avg']['recall']
    weighted_f1 = report_dict['weighted avg']['f1-score']

    # 3. 各類別 AP (Average Precision) 計算
    y_true_binarized = label_binarize(all_y_true, classes=classes)
    try:
        ap_per_class = average_precision_score(y_true_binarized, all_probs, average=None)
        mAP = np.mean(ap_per_class)
        
        ap_notTired = ap_per_class[0] if len(ap_per_class) > 0 else 0.0
        ap_Tired = ap_per_class[1] if len(ap_per_class) > 1 else 0.0
        ap_Other = ap_per_class[2] if len(ap_per_class) > 2 else 0.0
        
        if len(ap_per_class) >= 2:
            mAP_notTired_Tired = np.nanmean([ap_notTired, ap_Tired])
        else:
            mAP_notTired_Tired = 0.0
    except ValueError:
        ap_notTired = ap_Tired = ap_Other = mAP = mAP_notTired_Tired = 0.0

    # 終端機日誌輸出
    print(f"\n================ Results (Per-Segment) ================")
    print(f"Total Segments Evaluated: {len(all_y_true)}")
    print(f"Overall Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f} | Overall mAP: {mAP:.4f} | mAP (notTired & Tired): {mAP_notTired_Tired:.4f}")
    print(f"--- Detailed Per-Class Performance ---")
    print(f"[0] notTired -> P: {p_notTired:.4f} | R: {r_notTired:.4f} | F1: {f1_notTired:.4f} | AP: {ap_notTired:.4f}")
    print(f"[1] Tired    -> P: {p_Tired:.4f} | R: {r_Tired:.4f} | F1: {f1_Tired:.4f} | AP: {ap_Tired:.4f}")
    print(f"[2] Other    -> P: {p_Other:.4f} | R: {r_Other:.4f} | F1: {f1_Other:.4f} | AP: {ap_Other:.4f}")
    
    print("\nDetailed Classification Report:")
    unique_labels = np.unique(all_y_true)
    print(classification_report(all_y_true, all_preds, labels=unique_labels, target_names=[target_names[i] for i in unique_labels], zero_division=0))
    print("=======================================================\n")

    # ---------------------------------------------------------
    # 混淆矩陣繪製 (調整字體並移除標題)
    # ---------------------------------------------------------
    cm = confusion_matrix(all_y_true, all_preds, labels=classes)
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=target_names, 
        yticklabels=target_names,
        annot_kws={"size": 14}
    )
    
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0)
    
    plt.tight_layout()
    plt.savefig(cm_img_path, dpi=300)
    plt.close()
    print(f"Confusion Matrix saved to: {cm_img_path}")

    # ---------------------------------------------------------
    # 測試片段信心度分布圖繪製 (獨立子圖 + Y軸固定 0-100% + X軸精簡版)
    # ---------------------------------------------------------
    try:
        # 1. 提取模型預測該片段「真正標籤」對應通道的機率值（信心度）
        true_class_probs = all_probs[np.arange(len(all_y_true)), all_y_true]
        
        # 2. 定義 bins 與精簡後的區間標籤 (0.0-0.1, ..., 0.9-1.0)
        bins = np.arange(0.0, 1.1, 0.1)
        bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
        
        # 設定顏色與標籤順序
        color_map = {0: '#2ecc71', 1: '#e74c3c', 2: '#3498db'} # 綠、紅、藍
        pred_names = ['Pred: notTired', 'Pred: Tired', 'Pred: Other']
        
        # 建立 3x1 的畫布
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        for c_idx, ax in enumerate(axes):
            # 篩選出屬於當前真實類別的數據
            class_mask = (all_y_true == c_idx)
            c_probs = true_class_probs[class_mask]
            c_preds = all_preds[class_mask]
            total_class_samples = len(c_probs)
            
            # 初始化 DataFrame 用來儲存當前類別的分佈統計
            df_hist = pd.DataFrame(0.0, index=bin_labels, columns=[0, 1, 2])
            df_count = pd.DataFrame(0, index=bin_labels, columns=[0, 1, 2])
            
            if total_class_samples > 0:
                # 統計每個訊號落在哪個 bin
                bin_indices = np.digitize(c_probs, bins) - 1
                # 邊界處理：把機率剛好等於 1.0 的歸到最後一個 bin
                bin_indices[bin_indices == 10] = 9
                
                for b_i in range(10):
                    for p_i in range(3):
                        match_count = np.sum((bin_indices == b_i) & (c_preds == p_i))
                        df_count.iloc[b_i, p_i] = match_count
                        # 核心歸一化：佔該真實類別總樣本的比例 (%)
                        df_hist.iloc[b_i, p_i] = (match_count / total_class_samples) * 100
            
            # 繪製當前類別的堆疊長條圖
            df_hist.plot(
                kind='bar',
                stacked=True,
                ax=ax,
                color=[color_map[0], color_map[1], color_map[2]],
                edgecolor='black',
                linewidth=0.5,
                width=0.7
            )
            
            # 在長條圖上標註「百分比」與「(絕對件數)」
            flatten_props = df_hist.to_numpy().flatten()
            flatten_counts = df_count.to_numpy().flatten()
            
            for idx, rect in enumerate(ax.patches):
                height = rect.get_height()
                width = rect.get_width()
                x = rect.get_x()
                y = rect.get_y()
                
                num_rows = len(bin_labels)
                col_ordered_idx = (idx // num_rows) + (idx % num_rows) * 3
                
                prop_val = flatten_props[col_ordered_idx]
                count_val = flatten_counts[col_ordered_idx]
                
                # 比例大於 1% 才顯示標籤
                if prop_val > 1.0:
                    # 如果標籤太靠頂部，稍微調整文字顏色或位置避免出界
                    ax.text(
                        x + width/2.,
                        y + height/2.,
                        f"{prop_val:.1f}%\n({int(count_val)})",
                        ha='center',
                        va='center',
                        color='white',
                        fontweight='bold',
                        fontsize=8.5,
                        bbox=dict(facecolor='black', alpha=0.3, boxstyle='round,pad=0.1', edgecolor='none')
                    )
            
            # 子圖美化
            ax.set_title(f"True Label: {target_names[c_idx]} (Total Samples: {total_class_samples})", fontsize=13, fontweight='bold', loc='left')
            ax.set_ylabel("Proportion (%)", fontsize=11, fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            
            # 【關鍵修改】：強行將 Y 軸極限固定在 0 - 106% (留 6% 給頂部標籤空間，避免頂滿切字)
            ax.set_ylim(0, 106)
            ax.set_yticks(np.arange(0, 101, 20))
            ax.tick_params(axis='y', labelsize=10)
            
            # 處理圖例
            if c_idx == 0:
                ax.legend(pred_names, loc='upper left', fontsize=11)
            else:
                ax.get_legend().remove()

        # 最下方子圖的 X 軸設定
        plt.xlabel('Model Confidence Interval (Probability of True Class)', fontsize=12, fontweight='bold', labelpad=10)
        plt.xticks(rotation=0, ha='center', fontsize=11) 
        plt.tight_layout()
        
        normalized_conf_img_path = conf_img_path.replace(".png", "_normalized_proportion.png")
        plt.savefig(normalized_conf_img_path, dpi=300)
        plt.close()
        
        print(f"Normalized Proportion Plot saved to: {normalized_conf_img_path}")
        
    except Exception as e:
        print(f"[Warning] Failed to generate normalized proportion plot: {e}")

# ---------------------------------------------------------
# 儲存擴充後的結果 CSV
# ---------------------------------------------------------
    df = pd.DataFrame([{
        "run_name": run_name,
        "acc": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mAP": mAP,
        "mAP_notTired_Tired": mAP_notTired_Tired,  
        
        # 類別 0: notTired
        "p_notTired": p_notTired,
        "r_notTired": r_notTired,
        "f1_notTired": f1_notTired,
        "ap_notTired": ap_notTired,
        
        # 類別 1: Tired
        "p_Tired": p_Tired,
        "r_Tired": r_Tired,
        "f1_Tired": f1_Tired,
        "ap_Tired": ap_Tired,
        
        # 類別 2: Other
        "p_Other": p_Other,
        "r_Other": r_Other,
        "f1_Other": f1_Other,
        "ap_Other": ap_Other,
        
        "num_segments": len(all_y_true),
        "layer": args.layer,
        "hidden": args.hidden,
        "lr": args.lr,
        "column": args.column
    }])
    
    df.to_csv(result_csv_path, index=False)
    print(f"Saved result to {result_csv_path}")
    print("-" * 50)

if __name__ == "__main__":
    main()