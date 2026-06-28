import cv2
import os
import subprocess
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from ai_coach import AICoach
from pages1.pose_utils import get_full_body_angles
from ai_coach import AICoach

class PoseProcessor:

    def __init__(self):
        self.CONNECTIONS = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28)
        ]
        self.coach = AICoach()

    # =========================
    # 空間對齊（雲端穩健增強版）
    # =========================
    def _get_center_and_scale(self, df_row):
        """計算單影格的中心點與骨架長度，並嚴格處理雲端資料異常"""
        try:
            # 雲端資料常有 NaN 或 None，先取出並給予安全預設值
            def get_val(col, default=0.5):
                val = df_row.get(col, default)
                return default if pd.isna(val) or val == 0 else float(val)

            hip_l = np.array([get_val('23_x'), get_val('23_y')])
            hip_r = np.array([get_val('24_x'), get_val('24_y')])
            center = (hip_l + hip_r) / 2

            sh_l = np.array([get_val('11_x'), get_val('11_y')])
            sh_r = np.array([get_val('12_x'), get_val('12_y')])

            # 計算肩膀到髖關節的軀幹長度作為 Scale 基準
            scale = (np.linalg.norm(sh_l - hip_l) + np.linalg.norm(sh_r - hip_r)) / 2

            # 如果算出來的值異常或遺失，給予安全兜底
            if scale < 0.001 or np.isnan(scale):
                return center, 0.2
            return center, scale

        except:
            # 雲端防崩潰安全鎖
            return np.array([0.5, 0.5]), 0.2

    def align_to_user_space(self, coach_row, user_row, current_scale_ratio):
        """將教練的骨架對齊到使用者的實際畫面空間（使用平滑後的縮放比）"""
        c_center, _ = self._get_center_and_scale(coach_row)
        u_center, _ = self._get_center_and_scale(user_row)

        # 直接沿用外部傳入、經過全局濾波後的平滑穩定 scale_ratio
        out = coach_row.copy()

        for i in range(11, 33):
            x_col, y_col = f"{i}_x", f"{i}_y"
            if x_col in out and y_col in out:
                val_x = out[x_col]
                val_y = out[y_col]
                # 排除 NaN 與無效偵測點
                if pd.isna(val_x) or pd.isna(val_y):
                    continue
                
                out[x_col] = (float(val_x) - c_center[0]) * current_scale_ratio + u_center[0]
                out[y_col] = (float(val_y) - c_center[1]) * current_scale_ratio + u_center[1]

        return out

    # =========================
    # 特徵提取 (加入強健數據清洗)
    # =========================
    def extract_features(self, df):
        """提取特徵並清洗遺失值，防止 DTW 計算崩潰"""
        if df.empty:
            return np.zeros((1, 3))
            
        # ⚠️ 強健修正：防止傳入的 DataFrame 帶有開頭或中間的 NaN 導致計算出 NaN 特徵
        df_clean = df.ffill().bfill()
        feats = []
 
        for i in range(len(df_clean)):
            row = df_clean.iloc[i]
 
            try:
                s = np.array([float(row["12_x"]), float(row["12_y"])])
                e = np.array([float(row["14_x"]), float(row["14_y"])])
                w = np.array([float(row["16_x"]), float(row["16_y"])])
 
                v1, v2 = s - e, w - e
 
                denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                angle = np.arccos(np.clip(np.dot(v1, v2) / denom, -1, 1))
 
                if i > 0:
                    prev = df_clean.iloc[i - 1]
                    speed = np.linalg.norm([
                        float(row["16_x"]) - float(prev["16_x"]),
                        float(row["16_y"]) - float(prev["16_y"])
                    ])
                else:
                    speed = 0
 
                height = 1.0 - float(row["16_y"])
 
                feats.append([
                    angle / np.pi,
                    np.tanh(speed * 5),
                    height
                ])
 
            except:
                feats.append([0, 0, 0])
 
        return np.array(feats)
 
    # =========================
    # 相似度評分 (DTW 優化版)
    # =========================
    def calculate_auto_similarity(self, df_std, df_usr):
        if df_std.empty or df_usr.empty:
            return 0.0, {"feedback": "未偵測到有效動作數據", "penalty": 0}
 
        feat_std = self.extract_features(df_std)
        feat_usr = self.extract_features(df_usr)
 
        # 使用 fastdtw 計算最佳匹配路徑
        _, path = fastdtw(feat_std, feat_usr, dist=euclidean)
 
        joint_weights = {0: 1.0, 1: 1.5, 2: 1.2}
        path_scores = []
 
        for s, u in path:
            # 邊界保護防止索引溢出
            s_idx = min(s, len(feat_std) - 1)
            u_idx = min(u, len(feat_usr) - 1)
            
            diff = np.abs(feat_std[s_idx] - feat_usr[u_idx])
            weighted_error = (
                diff[0] * joint_weights[0] +
                diff[1] * joint_weights[1] +
                diff[2] * joint_weights[2]
            )
            score = 100 * np.exp(-2.0 * weighted_error)
            path_scores.append(score)
 
        path_scores = np.array(path_scores)
        n = len(path_scores)
        
        # 動態截剪保護
        if n > 20:
            trim = int(n * 0.05)
            path_scores = path_scores[trim:n - trim]
 
        # ⚠️ 防止路徑為空時計算均值出錯
        if len(path_scores) == 0:
            return 0.0, {"feedback": "動作比對路徑無效", "penalty": 0}

                # =========================
        # FINAL SCORE
        # =========================
        mean = np.mean(path_scores)
        p50 = np.percentile(path_scores, 50)
        p25 = np.percentile(path_scores, 25)
        worst = np.min(path_scores)
        std = np.std(path_scores)
 
        final_score = (
            mean * 0.7 +
            p50 * 0.25 +
            p25 * 0.10 +
            worst * 0.15
        )
 
        final_score *= 1.2
 
        if mean > 85:
            final_score += 5
        elif mean > 75:
            final_score += 1
 
 
        # =========================
        # AICoach penalty
        # =========================
        feedback, overall, penalty = self.coach.generate_feedback(
            feat_std, feat_usr, path, None, final_score
        )
        
        final_score = final_score - penalty
        final_score = float(np.clip(final_score, 0, 100))
 
        return final_score, {
            "mean_path_score": float(mean),
            "p50": float(p50),
            "p25": float(p25),
            "min": float(worst),
            "std": float(std),
            "path_length": len(path),
            "feedback": feedback,
            "overall": overall,
            "penalty": penalty
        }
 
    # =========================
    # 計算得分曲線
    # =========================
    def compute_similarity_curve(self, df_std, df_usr):
        if df_std.empty or df_usr.empty:
            return [0] * max(1, len(df_usr))
            
        feat_std = self.extract_features(df_std)
        feat_usr = self.extract_features(df_usr)
 
        _, path = fastdtw(feat_std, feat_usr, dist=euclidean)
 
        scores = np.zeros(len(df_usr))
        counts = np.zeros(len(df_usr))
 
        for s, u in path:
            s_idx = min(s, len(feat_std) - 1)
            u_idx = min(u, len(df_usr) - 1)
            
            d = np.linalg.norm(feat_std[s_idx] - feat_usr[u_idx])
            scores[u_idx] += 100 * np.exp(-0.5 * d)
            counts[u_idx] += 1
 
        # ⚠️ 使用 numpy.divide 的安全除法機制防止除以零
        curve = np.divide(
            scores,
            counts,
            out=np.zeros_like(scores),
            where=counts != 0
        )
 
        return np.convolve(curve, np.ones(3) / 3, mode='same').tolist()
 
    # =========================
    # 骨架疊加影片 (🔥 雲端優化平滑版)
    # =========================
    def generate_auto_overlay(self, video_path, df_std, df_usr, start_idx, output_path):
        if df_std.empty or df_usr.empty:
            return
            
        feat_std = self.extract_features(df_std)
        feat_usr = self.extract_features(df_usr)
 
        _, path = fastdtw(feat_std, feat_usr, dist=euclidean)
        u_to_s_map = {u: s for s, u in path}
 
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
        # 先寫入臨時檔案，最後透過 FFmpeg 編碼成無損標準格式
        tmp_path = output_path.replace(".mp4", "_tmp.mp4")
        out = cv2.VideoWriter(
            tmp_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (w, h)
        )
 
        # ─── 💡 全局 Scale 平滑化 ───
        raw_ratios = []
        for u_idx in range(len(df_usr)):
            if u_idx in u_to_s_map:
                s_idx = min(u_to_s_map[u_idx], len(df_std) - 1)
                _, c_scale = self._get_center_and_scale(df_std.iloc[s_idx])
                _, u_scale = self._get_center_and_scale(df_usr.iloc[u_idx])
                raw_ratios.append(u_scale / max(0.001, c_scale))
            else:
                raw_ratios.append(1.0)
        
        raw_ratios = np.array(raw_ratios, dtype=np.float32)
        window_size = 9
        if len(raw_ratios) > window_size:
            smoothed_ratios = np.convolve(raw_ratios, np.ones(window_size)/window_size, mode='same')
            half = window_size // 2
            smoothed_ratios[:half] = raw_ratios[:half]
            smoothed_ratios[-half:] = raw_ratios[-half:]
        else:
            smoothed_ratios = raw_ratios
        # ────────────────────────────
 
        f_idx = 0
 
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
 
            rel_idx = f_idx - start_idx
 
            if 0 <= rel_idx < len(df_usr):
                u_row = df_usr.iloc[rel_idx]
                # 繪製使用者骨架 (紅色)
                self.draw_skeleton(frame, u_row, (0, 0, 255), 2, w, h)
 
                if rel_idx in u_to_s_map:
                    s_idx = min(u_to_s_map[rel_idx], len(df_std) - 1)
                    current_ratio = float(smoothed_ratios[rel_idx])
                    
                    # 將教練數據進行平移與平滑縮放對齊
                    c_row = self.align_to_user_space(df_std.iloc[s_idx], u_row, current_ratio)
                    
                    # 繪製教練骨架 (藍色)
                    self.draw_skeleton(frame, c_row, (255, 0, 0), 3, w, h)
 
            out.write(frame)
            f_idx += 1
 
        cap.release()
        out.release()
 
        # ─── 🔥 H.264 影音轉碼 ───
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", tmp_path,
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                output_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except:
            if os.path.exists(tmp_path):
                os.rename(tmp_path, output_path)
 
    # =========================
    # 繪製骨架基礎函數
    # =========================
    def draw_skeleton(self, frame, row, color, thickness, w, h):
        pts = {}
 
        for i in range(11, 33):
            x_col, y_col = f"{i}_x", f"{i}_y"
            if x_col in row and y_col in row:
                val_x = row[x_col]
                val_y = row[y_col]
                
                if pd.isna(val_x) or pd.isna(val_y) or (val_x == 0 and val_y == 0):
                    continue
                    
                pts[i] = (
                    int(np.clip(float(val_x), 0, 1) * w),
                    int(np.clip(float(val_y), 0, 1) * h)
                )
 
        for a, b in self.CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], color, thickness, cv2.LINE_AA)
 
        for p in pts.values():
            cv2.circle(frame, p, thickness + 1, color, -1, cv2.LINE_AA)
 
    # =========================
    # 揮拍區間識別 (手腕速度回溯法)
    # =========================
    def detect_action_range(self, df):
        if df.empty or len(df) < 5:
            return 0, max(1, len(df)//2), max(2, len(df)-1)
            
        try:
            # 填補極端缺失狀況
            df_filled = df.ffill().bfill()
            wx = df_filled['16_x'].values
            wy = df_filled['16_y'].values
 
            # 計算手腕一階差分移動速度
            speed = np.hypot(
                np.diff(wx, prepend=wx[0]),
                np.diff(wy, prepend=wy[0])
            )
 
            peak = int(np.argmax(speed))
            start_th = np.percentile(speed, 15)
            end_th = np.percentile(speed, 10)
 
            start = 0
            for i in range(peak, 0, -1):
                if speed[i] < start_th:
                    start = i
                    break
 
            end = len(df_filled) - 1
            for i in range(peak, len(df_filled)):
                if speed[i] < end_th:
                    end = i
                    break
 
            # 前後緩衝幀配置
            start = max(0, start - 15)
            end = min(len(df_filled) - 1, end + 35)
 
            # ⚠️ 核心防禦：如果揮拍區間過窄，進行硬性區間寬度保底
            if (end - start) < 45:
                start = max(0, peak - 20)
                end = min(len(df_filled) - 1, peak + 40)
            
            # ⚠️ 強效防護：確保物理時間順序 start < peak < end，100% 避免除以零
            if peak <= start: 
                peak = start + 1
            if end <= peak: 
                end = peak + 1
 
            return int(start), int(peak), int(end)
 
        except:
            # 萬一發生未知例外，執行全局安全兜底
            p = len(df) // 2
            s = max(0, p - 20)
            e = min(len(df) - 1, p + 40)
            if p <= s: p = s + 1
            if e <= p: e = p + 1
            return int(s), int(p), int(e)