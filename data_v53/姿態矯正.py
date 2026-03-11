import os
import glob
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

def process_attitude_correction(input_dir='./test_data', output_dir='./test_data2'):
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 搜尋輸入目錄下的所有 CSV 檔案
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    
    if not csv_files:
        print(f"在 {input_dir} 中找不到任何 CSV 檔案。")
        return

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"正在處理: {file_name}")
        
        try:
            df = pd.read_csv(file_path)
            
            # 提取 6 軸數據 (N, 3)
            accel_body = df[['acceleration_X', 'acceleration_Y', 'acceleration_Z']].values
            gyro_body = df[['gyroscope_X', 'gyroscope_Y', 'gyroscope_Z']].values
            
            # 提取四元數 (Scipy 預設順序為 x, y, z, w)
            quats = df[['quaternion_x', 'quaternion_y', 'quaternion_z', 'quaternion_w']].values
            
            # 直接使用四元數建立旋轉物件
            rotations = R.from_quat(quats)
            
            # 進行姿態矯正 (Body Frame -> World Frame)
            accel_world = rotations.apply(accel_body)
            gyro_world = rotations.apply(gyro_body)
            
            # 將結果寫回 DataFrame 新欄位
            df['acceleration_X_corrected'] = accel_world[:, 0]
            df['acceleration_Y_corrected'] = accel_world[:, 1]
            df['acceleration_Z_corrected'] = accel_world[:, 2]
            
            df['gyroscope_X_corrected'] = gyro_world[:, 0]
            df['gyroscope_Y_corrected'] = gyro_world[:, 1]
            df['gyroscope_Z_corrected'] = gyro_world[:, 2]
            
            # 若原始資料包含磁力計，也一併進行旋轉矯正
            if all(col in df.columns for col in ['magnetometer_X', 'magnetometer_Y', 'magnetometer_Z']):
                mag_body = df[['magnetometer_X', 'magnetometer_Y', 'magnetometer_Z']].values
                mag_world = rotations.apply(mag_body)
                df['magnetometer_X_corrected'] = mag_world[:, 0]
                df['magnetometer_Y_corrected'] = mag_world[:, 1]
                df['magnetometer_Z_corrected'] = mag_world[:, 2]
            
            # 儲存結果到 output3
            output_path = os.path.join(output_dir, file_name)
            df.to_csv(output_path, index=False)
            print(f"已儲存矯正後資料至: {output_path}")
            
        except Exception as e:
            print(f"處理 {file_name} 時發生錯誤: {e}")

if __name__ == "__main__":
    process_attitude_correction()
    print("所有檔案處理完成！")