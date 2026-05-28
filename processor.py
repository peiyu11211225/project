import numpy as np
import pandas as pd
import cv2
import os
import subprocess
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
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

        # 這裡直接沿用外部傳入、經過全局濾波後的平滑穩定 scale_ratio
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
    # feature extraction
    # =========================
    def extract_features(self, df):
        feats = []
 
        for i in range(len(df)):
            row = df.iloc[i]
 
            try:
                s = np.array([row["12_x"], row["12_y"]])
                e = np.array([row["14_x"], row["14_y"]])
                w = np.array([row["16_x"], row["16_y"]])
 
                v1, v2 = s - e, w - e
 
                denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                angle = np.arccos(np.clip(np.dot(v1, v2) / denom, -1, 1))
 
                if i > 0:
                    prev = df.iloc[i - 1]
                    speed = np.linalg.norm([
                        row["16_x"] - prev["16_x"],
                        row["16_y"] - prev["16_y"]
                    ])
                else:
                    speed = 0
 
                height = 1.0 - row["16_y"]
 
                feats.append([
                    angle / np.pi,
                    np.tanh(speed * 5),
                    height
                ])
 
            except:
                feats.append([0, 0, 0])
 
        return np.array(feats)
 
    # =========================
    # similarity score
    # =========================
    def calculate_auto_similarity(self, df_std, df_usr):
 
        feat_std = self.extract_features(df_std)
        feat_usr = self.extract_features(df_usr)
 
        _, path = fastdtw(feat_std, feat_usr, dist=euclidean)
 
        joint_weights = {0: 1.0, 1: 1.5, 2: 1.2}
        path_scores = []
 
        for s, u in path:
            diff = np.abs(feat_std[s] - feat_usr[u])
            weighted_error = (
                diff[0] * joint_weights[0] +
                diff[1] * joint_weights[1] +
                diff[2] * joint_weights[2]
            )
            score = 100 * np.exp(-2.0 * weighted_error)
            path_scores.append(score)
 
        path_scores = np.array(path_scores)
 
        n = len(path_scores)
        trim = int(n * 0.05)
 
        if n > 20:
            path_scores = path_scores[trim:n - trim]
 
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
        final_score -= std * 0.05
 
        if mean > 85:
            final_score += 5
        elif mean > 75:
            final_score += 1
 
        if worst < 40:
            final_score -= 5
 
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
    # curve
    # =========================
    def compute_similarity_curve(self, df_std, df_usr):
 
        feat_std = self.extract_features(df_std)
        feat_usr = self.extract_features(df_usr)
 
        _, path = fastdtw(feat_std, feat_usr, dist=euclidean)
 
        scores = np.zeros(len(df_usr))
        counts = np.zeros(len(df_usr))
 
        for s, u in path:
            d = np.linalg.norm(feat_std[s] - feat_usr[u])
            scores[u] += 100 * np.exp(-0.5 * d)
            counts[u] += 1
 
        curve = np.divide(
            scores,
            counts,
            out=np.zeros_like(scores),
            where=counts != 0
        )
 
        return np.convolve(curve, np.ones(3) / 3, mode='same').tolist()
 
    # =========================
    # overlay video (🔥 雲端優化平滑版)
    # =========================
    def generate_auto_overlay(self, video_path, df_std, df_usr, start_idx, output_path):
 
        feat_std = self.extract_features(df_std)
        feat_usr = self.extract_features(df_usr)
 
        _, path = fastdtw(feat_std, feat_usr, dist=euclidean)
        u_to_s_map = {u: s for s, u in path}
 
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
        # 雲端常因編碼器缺失產生 0kb 損壞影片，故先寫入臨時檔，最後以 FFmpeg 統一轉碼
        tmp_path = output_path.replace(".mp4", "_tmp.mp4")
        out = cv2.VideoWriter(
            tmp_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (w, h)
        )
 
        # ─── 🔥 全局 Scale 平滑化 ───
        raw_ratios = []
        for u_idx in range(len(df_usr)):
            if u_idx in u_to_s_map:
                s_idx = u_to_s_map[u_idx]
                _, c_scale = self._get_center_and_scale(df_std.iloc[s_idx])
                _, u_scale = self._get_center_and_scale(df_usr.iloc[u_idx])
                raw_ratios.append(u_scale / c_scale)
            else:
                raw_ratios.append(1.0)
        
        raw_ratios = np.array(raw_ratios, dtype=np.float32)
        # 用 9 影格的時間視窗平滑化比例，消除雲端雜訊引起的忽大忽小
        window_size = 9
        if len(raw_ratios) > window_size:
            smoothed_ratios = np.convolve(raw_ratios, np.ones(window_size)/window_size, mode='same')
            # 邊緣補償修復
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
                # 畫使用者骨架
                self.draw_skeleton(frame, u_row, (0, 0, 255), 2, w, h)
 
                if rel_idx in u_to_s_map:
                    s_idx = u_to_s_map[rel_idx]
                    
                    # 取得當前影格平滑過後的穩定縮放比
                    current_ratio = float(smoothed_ratios[rel_idx])
                    
                    # 將對齊計算與平滑比例打包傳入
                    c_row = self.align_to_user_space(df_std.iloc[s_idx], u_row, current_ratio)
                    
                    # 畫教練骨架
                    self.draw_skeleton(frame, c_row, (255, 0, 0), 3, w, h)
 
            out.write(frame)
            f_idx += 1
 
        cap.release()
        out.release()

        # ─── 🔥 H.264 雲端影音編碼轉換 ───
        # 雲端環境中絕對不可使用 cv2.imshow，必須靜態寫入後交給 FFmpeg 轉換
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
    # draw skeleton
    # =========================
    def draw_skeleton(self, frame, row, color, thickness, w, h):
        pts = {}
 
        for i in range(11, 33):
            x_col, y_col = f"{i}_x", f"{i}_y"
            if x_col in row and y_col in row:
                val_x = row[x_col]
                val_y = row[y_col]
                
                # 嚴格過濾雲端常見的 NaN 或是未偵測到的 0 值
                if pd.isna(val_x) or pd.isna(val_y) or (val_x == 0 and val_y == 0):
                    continue
                    
                pts[i] = (
                    int(np.clip(float(val_x), 0, 1) * w),
                    int(np.clip(float(val_y), 0, 1) * h)
                )
 
        for a, b in self.CONNECTIONS:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], color, thickness)
 
        for p in pts.values():
            cv2.circle(frame, p, thickness + 1, color, -1)
 
    # =========================
    # detect action
    # =========================
    def detect_action_range(self, df):
        try:
            wx = df['16_x'].values
            wy = df['16_y'].values
 
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
 
            end = len(df) - 1
            for i in range(peak, len(df)):
                if speed[i] < end_th:
                    end = i
                    break
 
            start = max(0, start - 15)
            end = min(len(df) - 1, end + 35)
 
            if (end - start) < 45:
                start = max(0, peak - 20)
                end = min(len(df) - 1, peak + 40)
 
            return int(start), int(peak), int(end)
 
        except:
            return 0, len(df) // 2, len(df) - 1