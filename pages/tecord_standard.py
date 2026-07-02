import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
from pose_utils import get_full_body_angles

class AICoach:
    def __init__(self, model_path='pose_landmarker.task'):
        """
        初始化 MediaPipe Pose Landmarker
        """
        self.model_path = model_path
        self.last_valid_data = None
        
        # 設定 MediaPipe 選項
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        self.options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO
        )

    def record(self, video_path, start_frame, end_frame, output_csv="standard_up.csv"):
        """
        執行教練數據提取主流程
        """
        if not os.path.exists(video_path):
            print(f"❌ 錯誤: 找不到影片檔案 {video_path}")
            return

        raw_records = []
        self.last_valid_data = None  # 重置快取

        print(f"🚀 開始提取教練數據: {start_frame} -> {end_frame} 影格")

        with mp.tasks.vision.PoseLandmarker.create_from_options(self.options) as landmarker:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 僅處理指定區間
                if start_frame <= frame_idx <= end_frame:
                    # 轉換圖片格式
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    timestamp_ms = int((frame_idx / fps) * 1000)
                    
                    # 偵測骨架
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    
                    frame_data = {}
                    if result.pose_landmarks:
                        landmarks = result.pose_landmarks[0]
                        coord_dict = {i: (landmarks[i].x, landmarks[i].y) for i in range(33)}
                        
                        # 1. 取得角度特徵 (來自你的 pose_utils)
                        try:
                            frame_data = get_full_body_angles(coord_dict)
                        except Exception as e:
                            print(f"⚠️ 影格 {frame_idx} 角度計算失敗: {e}")
                        
                        # 2. 取得座標特徵 (供 FastDTW 計算速度與高度使用)
                        for i in range(11, 33):
                            frame_data[f"{i}_x"] = landmarks[i].x
                            frame_data[f"{i}_y"] = landmarks[i].y
                        
                        frame_data['original_frame'] = frame_idx
                        self.last_valid_data = frame_data.copy()
                    
                    else:
                        # 💡 關鍵補點邏輯：偵測失敗時沿用前一影格，確保時間軸連續
                        if self.last_valid_data is not None:
                            frame_data = self.last_valid_data.copy()
                            frame_data['original_frame'] = frame_idx
                            # print(f"⚠️ 影格 {frame_idx} 偵測失敗，已使用補點")
                    
                    if frame_data:
                        raw_records.append(frame_data)
                
                frame_idx += 1
                if frame_idx > end_frame:
                    break 

            cap.release()

        # 處理與儲存數據
        if raw_records:
            self._save_to_csv(raw_records, output_csv)
        else:
            print("❌ 提取失敗：區間內未偵測到任何有效骨架。")

    def _save_to_csv(self, records, output_csv):
        """
        內部方法：處理 DataFrame 平滑化並存檔
        """
        df = pd.DataFrame(records)
        
        # 僅對數值欄位進行平滑化
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # 使用 window=5 的滑動平均來過濾噪點
        df[numeric_cols] = df[numeric_cols].rolling(window=5, min_periods=1, center=True).mean()

        df.to_csv(output_csv, index=False)
        print("-" * 30)
        print(f"✅ 教練標準數據已生成！")
        print(f"📏 片段總長: {len(df)} 影格")
        print(f"💾 檔案位置: {output_csv}")


# --- 執行區 ---
if __name__ == "__main__":
    # 實例化類別
    recorder = AICoach(model_path='pose_landmarker.task')
    
    # 執行提取任務
    recorder.record(
        video_path="IMG_0287.mov",
        start_frame=20, 
        end_frame=80,
        output_csv="standard_up.csv"
    )