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
    if "tired" in f: return 1
    return 1 # 預設

class TestDataset(Dataset):
    def __init__(self, folder, column_names, seq_len=960):
        self.items = []
        if not os.path.exists(folder): return
        files = glob.glob(os.path.join(folder, "**/*.csv"), recursive=True)
        
        # 自動產生 Other 類別的對應增強欄位名稱
        aug_columns = [f"{c}_aug" for c in column_names]
        
        for p in files:
            try:
                # 為了避免讀取不必要的龐大數據，只讀取我們需要的兩組欄位 (正常+增強)
                use_cols = column_names + aug_columns
                df = pd.read_csv(p, usecols=lambda x: x in use_cols, dtype=np.float32)
                
                # --- 處理正常的 0 (notTired) 或 1 (Tired) 數據 ---
                if all(col in df.columns for col in column_names):
                    x_normal = df[column_names].values
                    y_normal = infer_label(os.path.basename(p))
                    
                    if len(x_normal) >= seq_len:
                        for s in range(0, len(x_normal)-seq_len+1, seq_len):
                            seg = x_normal[s:s+seq_len, :] 
                            self.items.append((seg, y_normal, os.path.basename(p), "normal"))
                
                # --- 處理 Other (Label=2) 的增強數據 ---
                if all(col in df.columns for col in aug_columns):
                    x_aug = df[aug_columns].values
                    y_aug = 2 # 強制將增強數據標記為 Other(2)
                    
                    if len(x_aug) >= seq_len:
                        for s in range(0, len(x_aug)-seq_len+1, seq_len):
                            seg = x_aug[s:s+seq_len, :] 
                            self.items.append((seg, y_aug, os.path.basename(p), "augmented"))
            except Exception as e:
                pass

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): 
        seg, y, fname, data_type = self.items[idx]
        return torch.tensor(seg, dtype=torch.float32), torch.tensor(y, dtype=torch.long), fname

class GRUModel(nn.Module):
    def __init__(self, input_dim=1, hidden=128, layers=2, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, layers, batch_first=True, dropout=0.2)
        # 輸出改為 3 維
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        self.gru.flatten_parameters()
        _, h = self.gru(x)
        return self.fc(h[-1]).squeeze()

def main():
    target_columns = [c.strip() for c in args.column.split(',')]
    input_dim = len(target_columns)
    
    # [關鍵] 確保 run_name 與訓練時的三類別命名一致
    run_name = f"fold{args.fold}_layer_{args.layer}_hidden_{args.hidden}_lr_{args.lr}_dv_{args.data_version}_col_{args.column}_3class"
    
    output_dir = "./test_log"
    os.makedirs(output_dir, exist_ok=True)
    
    result_csv_path = os.path.join(output_dir, f"{run_name}_result.csv")
    log_file_path = os.path.join(output_dir, f"{run_name}_log.txt")
    cm_img_path = os.path.join(output_dir, f"{run_name}_cm.png")  # 混淆矩陣圖片路徑
    
    # 重導向輸出
    sys.stdout = DualLogger(log_file_path)

    # 檢查是否測過
    if os.path.exists(result_csv_path):
        print(f"[Skip] Test result exists: {run_name}")
        return

    # 權重路徑
    model_dir = f"models/{run_name}"
    ckpt_path = os.path.join(model_dir, "best.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(model_dir, "last.pth")
        if not os.path.exists(ckpt_path):
            print(f"[Error] No weights found for {run_name}")
            return

    print(f"Testing 3-Class Model (Evaluation Unit: Segment): {run_name}")
    print(f"Loading weights from: {ckpt_path}")

    # 測試集
    test_data_path = f"../data_v{args.data_version}/test_data"
    test_ds = TestDataset(test_data_path, target_columns, seq_len=300)
    if len(test_ds) == 0:
        print("[Error] No test data found.")
        return
    loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=4)

    # 模型載入 (num_classes=3)
    model = GRUModel(input_dim=input_dim, hidden=args.hidden, layers=args.layer, num_classes=3).to(device)
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        new_state = {}
        for k, v in state.items():
            if k.startswith("_orig_mod."): new_state[k[10:]] = v
            else: new_state[k] = v
        model.load_state_dict(new_state)
    except Exception as e:
        print(f"[Error] Load failed: {e}")
        return

    model.eval()

    all_y_true = []
    all_preds = []
    all_probs = []  # 用於儲存 mAP 所需的機率值

    with torch.no_grad():
        for segs, y, _ in loader:
            segs = segs.to(device)
            logits = model(segs)
            
            # 確保維度一致 (處理 batch_size=1 的情況)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
                
            # 計算機率分佈 (Softmax)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            # 三類別預測: 取 logits 中最大值的 index (0, 1, 或是 2)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_y_true.extend(y.numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    # 轉為 numpy array 計算指標
    all_y_true = np.array(all_y_true)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # ---------------------------------------------------------
    # 指標計算
    # ---------------------------------------------------------
    acc = accuracy_score(all_y_true, all_preds)
    macro_f1  = f1_score(all_y_true, all_preds, average='macro', zero_division=0)
    
    # 獨立類別 F1
    f1_scores_per_class = f1_score(all_y_true, all_preds, average=None, zero_division=0)
    f1_notTired = f1_scores_per_class[0] if len(f1_scores_per_class) > 0 else 0
    f1_Tired = f1_scores_per_class[1] if len(f1_scores_per_class) > 1 else 0
    f1_Other = f1_scores_per_class[2] if len(f1_scores_per_class) > 2 else 0

    # 計算多類別 mAP
    classes = [0, 1, 2]
    y_true_binarized = label_binarize(all_y_true, classes=classes)
    
    # 防止若測試集中某個 label 完全缺失導致 average_precision_score 報錯
    try:
        mAP = average_precision_score(y_true_binarized, all_probs, average='macro')
    except ValueError:
        mAP = 0.0

    print(f"\n================ Results (Per-Segment) ================")
    print(f"Total Segments Evaluated: {len(all_y_true)}")
    print(f"Overall Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f} | mAP: {mAP:.4f}")
    print(f"--- F1 Scores per class ---")
    print(f"[0] notTired F1 : {f1_notTired:.4f}")
    print(f"[1] Tired F1    : {f1_Tired:.4f}")
    print(f"[2] Other F1    : {f1_Other:.4f}")
    
    # 印出詳細的混淆報告
    print("\nDetailed Classification Report:")
    target_names = ['notTired (0)', 'Tired (1)', 'Other (2)']
    unique_labels = np.unique(all_y_true)
    print(classification_report(all_y_true, all_preds, labels=unique_labels, target_names=[target_names[i] for i in unique_labels], zero_division=0))
    print("=======================================================\n")

    # ---------------------------------------------------------
    # 繪製並儲存混淆矩陣 (Confusion Matrix)
    # ---------------------------------------------------------
    cm = confusion_matrix(all_y_true, all_preds, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix\n{run_name}', fontsize=12)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(cm_img_path, dpi=300)
    plt.close()
    print(f"Confusion Matrix saved to: {cm_img_path}")

    # ---------------------------------------------------------
    # 儲存結果 CSV
    # ---------------------------------------------------------
    df = pd.DataFrame([{
        "run_name": run_name,
        "acc": acc,
        "macro_f1": macro_f1,
        "mAP": mAP,                # 新增 mAP 欄位
        "f1_notTired": f1_notTired,
        "f1_Tired": f1_Tired,
        "f1_Other": f1_Other,
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