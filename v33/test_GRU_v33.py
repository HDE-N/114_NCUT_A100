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
    """
    依照順序檢查: notTired -> Tired -> other
    """
    f = fname.lower()
    if "nottired" in f: 
        return 0
    elif "tired" in f: 
        return 1
    elif "other" in f: 
        return 2
    return -1  # 防呆機制，理論上不會發生

class TestDataset(Dataset):
    def __init__(self, folder, column_names, seq_len=960):
        self.items = []
        if not os.path.exists(folder): 
            return
            
        files = glob.glob(os.path.join(folder, "**/*.csv"), recursive=True)
        
        for p in files:
            try:
                # 只讀取標準特徵欄位，不再區分 _aug 欄位
                df = pd.read_csv(p, usecols=lambda x: x in column_names, dtype=np.float32)
                
                # 確保檔案內包含所需的全部欄位
                if all(col in df.columns for col in column_names):
                    # 依據檔名判斷 Label
                    y_label = infer_label(os.path.basename(p))
                    
                    if y_label == -1:
                        continue  # 若檔名不符合任何標籤則跳過
                        
                    x_data = df[column_names].values
                    
                    # 切割序列
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
    all_probs = []  

    with torch.no_grad():
        for segs, y, _ in loader:
            segs = segs.to(device)
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
    
    try:
        ap_per_class = average_precision_score(y_true_binarized, all_probs, average=None)
        mAP = np.mean(ap_per_class)
        
        # 新增指標：僅考慮 notTired (0) 與 Tired (1) 的 mAP 
        if len(ap_per_class) >= 2:
            mAP_notTired_Tired = np.nanmean([ap_per_class[0], ap_per_class[1]])
        else:
            mAP_notTired_Tired = 0.0
            
    except ValueError:
        mAP = 0.0
        mAP_notTired_Tired = 0.0

    print(f"\n================ Results (Per-Segment) ================")
    print(f"Total Segments Evaluated: {len(all_y_true)}")
    print(f"Overall Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f} | Overall mAP: {mAP:.4f} | mAP (notTired & Tired): {mAP_notTired_Tired:.4f}")
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
    # 繪製並儲存混淆矩陣
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
        "mAP": mAP,
        "mAP_notTired_Tired": mAP_notTired_Tired,  
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