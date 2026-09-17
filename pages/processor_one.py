import cv2
import os
import subprocess
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from ai_coach_one import AICoach
from pages.pose_utils import get_full_body_angles
from ai_coach_one import AICoach

# ▼▼▼ 新增：畫圖用套件（原本檔案沒有，補上）▼▼▼
import matplotlib
matplotlib.use('Agg')  # 純存檔用，不開視窗，避免雲端/無 GUI 環境報錯
import matplotlib.pyplot as plt
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.sans-serif'] = [
    'Microsoft JhengHei', 'PingFang TC', 'Noto Sans CJK TC', 'Noto Sans TC', 'SimHei'
]
# ▲▲▲ 新增結束 ▲▲▲


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
            def get_val(col, default=0.5):
                val = df_row.get(col, default)
                return default if pd.isna(val) or val == 0 else float(val)

            hip_l = np.array([get_val('23_x'), get_val('23_y')])
            hip_r = np.array([get_val('24_x'), get_val('24_y')])
            center = (hip_l + hip_r) / 2

            sh_l = np.array([get_val('11_x'), get_val('11_y')])
            sh_r = np.array([get_val('12_x'), get_val('12_y')])

            scale = (np.linalg.norm(sh_l - hip_l) + np.linalg.norm(sh_r - hip_r)) / 2

            if scale < 0.001 or np.isnan(scale):
                return center, 0.2
            return center, scale

        except:
            return np.array([0.5, 0.5]), 0.2

    def align_to_user_space(self, coach_row, user_row, current_scale_ratio):
        """將教練的骨架對齊到使用者的實際畫面空間（使用平滑後的縮放比）"""
        c_center, _ = self._get_center_and_scale(coach_row)
        u_center, _ = self._get_center_and_scale(user_row)

        out = coach_row.copy()

        for i in range(11, 33):
            x_col, y_col = f"{i}_x", f"{i}_y"
            if x_col in out and y_col in out:
                val_x = out[x_col]
                val_y = out[y_col]
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
            return np.zeros((1, 4))

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

                hip_l = np.array([float(row["23_x"]), float(row["23_y"])])
                hip_r = np.array([float(row["24_x"]), float(row["24_y"])])
                body_center_x = (hip_l[0] + hip_r[0]) / 2.0
                torso_width = np.linalg.norm(hip_l - hip_r) + 1e-6

                direction = np.tanh((w[0] - body_center_x) / torso_width)

                feats.append([
                    angle / np.pi,
                    np.tanh(speed * 5),
                    height,
                    direction
                ])

            except:
                feats.append([0, 0, 0, 0])

        return np.array(feats)

    # =========================
    # 正式最終評分公式（唯一來源）
    # =========================
    def _calculate_final_score_from_path(self, feat_std, feat_usr, path):
        """
        對指定的比對 path 套用「正式分數」完整公式。

        對齊前與對齊後共用這一套公式，唯一不同只有 path：
        - 正式分數：DTW path
        - 對齊前：逐幀硬比對 path [(0,0), (1,1), ...]
        """
        if len(feat_std) == 0 or len(feat_usr) == 0 or len(path) == 0:
            return 0.0, {
                "feedback": "動作比對路徑無效",
                "penalty": 0,
            }

        # ===== 與正式分數完全相同的特徵權重 =====
        joint_weights = {0: 1.0, 1: 1.5, 2: 1.2, 3: 2.0}
        path_scores = []

        for s, u in path:
            s_idx = min(s, len(feat_std) - 1)
            u_idx = min(u, len(feat_usr) - 1)

            diff = np.abs(feat_std[s_idx] - feat_usr[u_idx])
            weighted_error = (
                diff[0] * joint_weights[0] +
                diff[1] * joint_weights[1] +
                diff[2] * joint_weights[2] +
                diff[3] * joint_weights[3]
            )

            score = 100 * np.exp(-2.0 * weighted_error)
            path_scores.append(score)

        path_scores = np.asarray(path_scores, dtype=float)
        n = len(path_scores)

        # ===== 與正式分數完全相同：去除前後 5% 離群值 =====
        if n > 20:
            trim = int(n * 0.05)
            path_scores = path_scores[trim:n - trim]

        if len(path_scores) == 0:
            return 0.0, {
                "feedback": "動作比對路徑無效",
                "penalty": 0,
            }

        # ===== 與正式分數完全相同的統計量 =====
        mean = float(np.mean(path_scores))
        p50 = float(np.percentile(path_scores, 50))
        p25 = float(np.percentile(path_scores, 25))
        worst = float(np.min(path_scores))
        std = float(np.std(path_scores))

        # ===== 正式最終分數完整公式 =====
        final_score = (
            mean * 0.7 +
            p50 * 0.25 +
            p25 * 0.10 +
            worst * 0.15
        )

        final_score *= 1.4

        if mean > 85:
            final_score += 5
        elif mean > 75:
            final_score += 1

        # ===== 與正式分數完全相同：AI Coach penalty =====
        feedback, overall, penalty = self.coach.generate_feedback(
            feat_std,
            feat_usr,
            path,
            None,
            final_score,
        )

        final_score = final_score - penalty
        final_score = float(np.clip(final_score, 0, 100))

        return final_score, {
            "mean_path_score": mean,
            "p50": p50,
            "p25": p25,
            "min": worst,
            "std": std,
            "path_length": len(path),
            "feedback": feedback,
            "overall": overall,
            "penalty": penalty,
        }

    # =========================
    # 相似度評分（正式 DTW 分數）
    # =========================
    def calculate_auto_similarity(self, df_std, df_usr):
        """
        正式評分：
        先用 DTW 找到最佳 path，再套用正式最終評分公式。

        此函式回傳的 final_score，就是使用者畫面上看到的最終分數。
        """
        if df_std.empty or df_usr.empty:
            return 0.0, {
                "feedback": "未偵測到有效動作數據",
                "penalty": 0,
            }

        feat_std = self.extract_features(df_std)
        feat_usr = self.extract_features(df_usr)

        _, path = fastdtw(feat_std, feat_usr, dist=euclidean)

        return self._calculate_final_score_from_path(
            feat_std,
            feat_usr,
            path,
        )

    # =========================================================
    # 不對齊對照組：逐幀硬比對 + 完整正式評分公式
    # =========================================================
    def calculate_naive_similarity(self, df_std, df_usr):
        """
        對齊前（Before Alignment）。

        與 calculate_auto_similarity 共用完全相同的：
        - 特徵
        - 權重
        - 單幀分數公式
        - 離群值處理
        - mean / p50 / p25 / worst 混合
        - *1.4
        - mean 加分
        - AI Coach penalty
        - 0~100 clip

        唯一差別：
        不做 DTW，直接使用 (0,0), (1,1), (2,2)...
        的逐幀硬比對。

        因此紅色「對齊前」與綠色「對齊後」是真正
        apples-to-apples 的比較。
        """
        if df_std.empty or df_usr.empty:
            return 0.0, {
                "feedback": "未偵測到有效動作數據",
                "penalty": 0,
            }

        feat_std = self.extract_features(df_std)
        feat_usr = self.extract_features(df_usr)

        min_len = min(len(feat_std), len(feat_usr))
        path = [(i, i) for i in range(min_len)]

        return self._calculate_final_score_from_path(
            feat_std,
            feat_usr,
            path,
        )

    # =========================================================
    # ▲▲▲ 對齊前 / 對齊後共用同一套完整正式評分公式 ▲▲▲
    # =========================================================

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

        curve = np.divide(
            scores,
            counts,
            out=np.zeros_like(scores),
            where=counts != 0
        )

        return np.convolve(curve, np.ones(3) / 3, mode='same').tolist()

    # =========================
    # 骨架疊加影片 (雲端優化平滑版)
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

        tmp_path = output_path.replace(".mp4", "_tmp.mp4")
        out = cv2.VideoWriter(
            tmp_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (w, h)
        )

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

        f_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rel_idx = f_idx - start_idx

            if 0 <= rel_idx < len(df_usr):
                u_row = df_usr.iloc[rel_idx]
                self.draw_skeleton(frame, u_row, (0, 0, 255), 2, w, h)

                if rel_idx in u_to_s_map:
                    s_idx = min(u_to_s_map[rel_idx], len(df_std) - 1)
                    current_ratio = float(smoothed_ratios[rel_idx])

                    c_row = self.align_to_user_space(df_std.iloc[s_idx], u_row, current_ratio)

                    self.draw_skeleton(frame, c_row, (255, 0, 0), 3, w, h)

            out.write(frame)
            f_idx += 1

        cap.release()
        out.release()

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
            df_filled = df.ffill().bfill()
            wx = df_filled['16_x'].values
            wy = df_filled['16_y'].values

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

            start = max(0, start - 15)
            end = min(len(df_filled) - 1, end + 35)

            if (end - start) < 45:
                start = max(0, peak - 20)
                end = min(len(df_filled) - 1, peak + 40)

            if peak <= start:
                peak = start + 1
            if end <= peak:
                end = peak + 1

            return int(start), int(peak), int(end)

        except:
            p = len(df) // 2
            s = max(0, p - 20)
            e = min(len(df) - 1, p + 40)
            if p <= s: p = s + 1
            if e <= p: e = p + 1
            return int(s), int(p), int(e)

    # =========================================================
    # ▼▼▼ 新增：對齊前 / 對齊後 量化比較圖（實驗數據佐證用）▼▼▼
    # =========================================================
    def plot_alignment_proof(self, df_std, df_usr,
                              output_dir="alignment_proof_output",
                              tag="sample",
                              feature_names=("手肘夾角", "手腕速度", "擊球高度", "揮拍方向")):
        """
        用跟 calculate_auto_similarity 完全相同的特徵抽取方式與 fastdtw 呼叫，
        畫出「對齊前 vs 對齊後」的量化比較圖，並存成 CSV 量化表。

        只存到本機資料夾，不回傳前端，適合當作實驗數據 / 報告佐證。
        """
        if df_std.empty or df_usr.empty:
            print("⚠️ 輸入資料為空，無法繪製對齊比較圖")
            return None

        os.makedirs(output_dir, exist_ok=True)

        feat_std = self.extract_features(df_std)
        feat_usr = self.extract_features(df_usr)

        distance, path = fastdtw(feat_std, feat_usr, dist=euclidean)

        n_feat = feat_std.shape[1]
        fig, axes = plt.subplots(n_feat, 2, figsize=(14, 3.2 * n_feat))
        if n_feat == 1:
            axes = np.array([axes])

        metrics_rows = []

        for f_idx in range(n_feat):
            name = feature_names[f_idx] if f_idx < len(feature_names) else f"feature_{f_idx}"

            raw_std = feat_std[:, f_idx]
            raw_usr = feat_usr[:, f_idx]
            min_len = min(len(raw_std), len(raw_usr))
            rmse_before = float(np.sqrt(np.mean((raw_std[:min_len] - raw_usr[:min_len]) ** 2)))
            corr_before = (
                float(np.corrcoef(raw_std[:min_len], raw_usr[:min_len])[0, 1])
                if min_len > 1 else float("nan")
            )

            ax_before = axes[f_idx, 0]
            ax_before.plot(raw_std, label="教練", color="#1f77b4", linewidth=1.8)
            ax_before.plot(raw_usr, label="使用者", color="#ff7f0e", linewidth=1.8)
            ax_before.set_title(f"{name} - 對齊前 (RMSE={rmse_before:.3f}, r={corr_before:.2f})")
            ax_before.legend(fontsize=8)
            ax_before.grid(alpha=0.3)

            aligned_std = np.array([feat_std[min(s, len(feat_std) - 1), f_idx] for s, u in path])
            aligned_usr = np.array([feat_usr[min(u, len(feat_usr) - 1), f_idx] for s, u in path])
            rmse_after = float(np.sqrt(np.mean((aligned_std - aligned_usr) ** 2)))
            corr_after = (
                float(np.corrcoef(aligned_std, aligned_usr)[0, 1])
                if len(aligned_std) > 1 else float("nan")
            )

            ax_after = axes[f_idx, 1]
            ax_after.plot(aligned_std, label="教練 (對齊後)", color="#1f77b4", linewidth=1.8)
            ax_after.plot(aligned_usr, label="使用者 (對齊後)", color="#ff7f0e", linewidth=1.8)
            ax_after.set_title(f"{name} - 對齊後 (RMSE={rmse_after:.3f}, r={corr_after:.2f})")
            ax_after.legend(fontsize=8)
            ax_after.grid(alpha=0.3)

            metrics_rows.append({
                "feature": name,
                "rmse_before": rmse_before,
                "corr_before": corr_before,
                "rmse_after": rmse_after,
                "corr_after": corr_after,
                "improvement_rmse_pct": (
                    float((rmse_before - rmse_after) / rmse_before * 100)
                    if rmse_before > 0 else float("nan")
                ),
            })

        fig.suptitle(
            f"DTW 對齊前 vs 對齊後 量化比較 [{tag}]\n"
            f"path_length={len(path)}, dtw_distance={distance:.2f}",
            fontsize=13
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        img_path = os.path.join(output_dir, f"{tag}_alignment_proof.png")
        plt.savefig(img_path, dpi=150)
        plt.close(fig)

        metrics_df = pd.DataFrame(metrics_rows)
        csv_path = os.path.join(output_dir, f"{tag}_alignment_metrics.csv")
        metrics_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        print(f"✅ 對齊比較圖已存至: {img_path}")
        print(f"✅ 量化指標表已存至: {csv_path}")
        print(metrics_df.to_string(index=False))

        return metrics_df
    # =========================================================
    # ▲▲▲ 新增區塊結束 ▲▲▲
    # =========================================================

    # =========================================================
    # ▼▼▼ 新增：整體評分系統 對齊前 vs 對齊後 證明 ▼▼▼
    # =========================================================
    def _weighted_score_series(self, feat_std, feat_usr, index_pairs, joint_weights):
        scores = []
        for s_idx, u_idx in index_pairs:
            s_idx = min(s_idx, len(feat_std) - 1)
            u_idx = min(u_idx, len(feat_usr) - 1)
            diff = np.abs(feat_std[s_idx] - feat_usr[u_idx])
            weighted_error = (
                diff[0] * joint_weights[0] +
                diff[1] * joint_weights[1] +
                diff[2] * joint_weights[2] +
                diff[3] * joint_weights[3]
            )
            scores.append(100 * np.exp(-2.0 * weighted_error))
        return np.array(scores)

    def _calculate_original_score_without_formal_formula(self, feat_std, feat_usr, path):
        """
        原本的「基礎分數」：只計算每一個配對點的 similarity score，
        再取平均；不套用正式最終分數的 mean/p50/p25/worst 混合、1.4x、
        bonus 與 AI Coach penalty。
        """
        if len(feat_std) == 0 or len(feat_usr) == 0 or not path:
            return float("nan")

        joint_weights = {0: 1.0, 1: 1.5, 2: 1.2, 3: 2.0}
        scores = []

        for s_idx, u_idx in path:
            s_idx = min(int(s_idx), len(feat_std) - 1)
            u_idx = min(int(u_idx), len(feat_usr) - 1)
            diff = np.abs(feat_std[s_idx] - feat_usr[u_idx])
            weighted_error = (
                diff[0] * joint_weights[0] +
                diff[1] * joint_weights[1] +
                diff[2] * joint_weights[2] +
                diff[3] * joint_weights[3]
            )
            scores.append(100 * np.exp(-2.0 * weighted_error))

        scores = np.asarray(scores, dtype=float)
        if len(scores) == 0:
            return float("nan")

        if len(scores) > 20:
            trim = int(len(scores) * 0.05)
            scores = scores[trim:len(scores) - trim]

        return float(np.mean(scores)) if len(scores) else float("nan")

    def plot_overall_score_proof(self, sample_pairs,
                                  output_dir="alignment_proof_output",
                                  tag="overall_proof"):
        """
        產生一張三層實驗圖：

        第一層：量化指標
            - 四個特徵的 RMSE：對齊前 vs DTW 對齊後
            - 四個特徵的 correlation：對齊前 vs DTW 對齊後

        第二層：原本的基礎分數（未套正式最終評分公式）

        第三層：正式最終分數（完整公式，等於使用者畫面上看到的分數）

        三層全部統一用「紅色＝對齊前 / 綠色＝對齊後」配色。
        """
        os.makedirs(output_dir, exist_ok=True)

        COLOR_BEFORE = "#d62728"  # 紅色：對齊前
        COLOR_AFTER = "#2ca02c"   # 綠色：對齊後

        names = []
        raw_before, raw_after = [], []
        original_score_before, original_score_after = [], []
        formal_before, formal_after = [], []
        feature_metrics = []
        rows = []

        feature_names = ("手肘夾角", "手腕速度", "擊球高度", "揮拍方向")
        feature_keys = ("elbow_angle", "wrist_speed", "hit_height", "swing_direction")

        for name, df_std, df_usr in sample_pairs:
            if df_std.empty or df_usr.empty:
                continue

            feat_std = self.extract_features(df_std)
            feat_usr = self.extract_features(df_usr)
            min_len = min(len(feat_std), len(feat_usr))
            naive_path = [(i, i) for i in range(min_len)]
            _, dtw_path = fastdtw(feat_std, feat_usr, dist=euclidean)

            # ---------- 第一層：原始量化指標 ----------
            before_feature_rmse = []
            after_feature_rmse = []
            before_corr = []
            after_corr = []

            for f_idx in range(min(feat_std.shape[1], feat_usr.shape[1])):
                std_raw = feat_std[:min_len, f_idx]
                usr_raw = feat_usr[:min_len, f_idx]
                rmse_b = float(np.sqrt(np.mean((std_raw - usr_raw) ** 2))) if min_len else float("nan")
                corr_b = (
                    float(np.corrcoef(std_raw, usr_raw)[0, 1])
                    if min_len > 1 and np.std(std_raw) > 0 and np.std(usr_raw) > 0
                    else float("nan")
                )

                aligned_std = np.array([feat_std[min(int(s), len(feat_std)-1), f_idx] for s, _ in dtw_path])
                aligned_usr = np.array([feat_usr[min(int(u), len(feat_usr)-1), f_idx] for _, u in dtw_path])
                rmse_a = float(np.sqrt(np.mean((aligned_std - aligned_usr) ** 2))) if len(aligned_std) else float("nan")
                corr_a = (
                    float(np.corrcoef(aligned_std, aligned_usr)[0, 1])
                    if len(aligned_std) > 1 and np.std(aligned_std) > 0 and np.std(aligned_usr) > 0
                    else float("nan")
                )

                before_feature_rmse.append(rmse_b)
                after_feature_rmse.append(rmse_a)
                before_corr.append(corr_b)
                after_corr.append(corr_a)

            overall_rmse_before = float(np.nanmean(before_feature_rmse)) if before_feature_rmse else float("nan")
            overall_rmse_after = float(np.nanmean(after_feature_rmse)) if after_feature_rmse else float("nan")
            avg_corr_before = float(np.nanmean(before_corr)) if before_corr else float("nan")
            avg_corr_after = float(np.nanmean(after_corr)) if after_corr else float("nan")

            # ---------- 第二層：原本基礎分數 ----------
            base_before = self._calculate_original_score_without_formal_formula(
                feat_std, feat_usr, naive_path
            )
            base_after = self._calculate_original_score_without_formal_formula(
                feat_std, feat_usr, dtw_path
            )

            # ---------- 第三層：正式最終分數 ----------
            formal_before_score, formal_before_info = self.calculate_naive_similarity(df_std, df_usr)
            formal_after_score, formal_after_info = self.calculate_auto_similarity(df_std, df_usr)

            names.append(name)
            raw_before.append(overall_rmse_before)
            raw_after.append(overall_rmse_after)
            original_score_before.append(base_before)
            original_score_after.append(base_after)
            formal_before.append(formal_before_score)
            formal_after.append(formal_after_score)

            row = {
                "sample": name,
                "overall_rmse_before": overall_rmse_before,
                "overall_rmse_after_dtw": overall_rmse_after,
                "overall_rmse_reduction": overall_rmse_before - overall_rmse_after,
                "overall_rmse_reduction_pct": (
                    (overall_rmse_before - overall_rmse_after) / overall_rmse_before * 100
                    if overall_rmse_before > 0 else float("nan")
                ),
                "average_correlation_before": avg_corr_before,
                "average_correlation_after_dtw": avg_corr_after,
                "original_score_before_no_formal_formula": base_before,
                "original_score_after_dtw_no_formal_formula": base_after,
                "original_score_improvement": base_after - base_before,
                "original_score_improvement_pct": (
                    (base_after - base_before) / base_before * 100
                    if base_before > 0 else float("nan")
                ),
                "formal_score_before": formal_before_score,
                "formal_score_after": formal_after_score,
                "formal_score_improvement": formal_after_score - formal_before_score,
                "formal_score_improvement_pct": (
                    (formal_after_score - formal_before_score) / formal_before_score * 100
                    if formal_before_score > 0 else float("nan")
                ),
                "penalty_before": formal_before_info.get("penalty", 0),
                "penalty_after": formal_after_info.get("penalty", 0),
            }

            for i, key in enumerate(feature_keys):
                row[f"{key}_rmse_before"] = before_feature_rmse[i]
                row[f"{key}_rmse_after"] = after_feature_rmse[i]
                row[f"{key}_corr_before"] = before_corr[i]
                row[f"{key}_corr_after"] = after_corr[i]

            rows.append(row)
            feature_metrics.append((before_feature_rmse, after_feature_rmse, before_corr, after_corr))

        if not names:
            print("⚠️ 沒有有效樣本可以比較")
            return None

        x = np.arange(len(names))
        width = 0.35
        fig, axes = plt.subplots(3, 1, figsize=(max(10, len(names) * 1.8), 16))

        # =====================================================
        # 第一層：量化指標（RMSE + correlation）— 紅／綠配色
        # =====================================================
        mean_before_rmse = np.nanmean(np.asarray([m[0] for m in feature_metrics], dtype=float), axis=0)
        mean_after_rmse = np.nanmean(np.asarray([m[1] for m in feature_metrics], dtype=float), axis=0)
        mean_before_corr = np.nanmean(np.asarray([m[2] for m in feature_metrics], dtype=float), axis=0)
        mean_after_corr = np.nanmean(np.asarray([m[3] for m in feature_metrics], dtype=float), axis=0)

        fx = np.arange(len(feature_names))
        fwidth = 0.34
        b1 = axes[0].bar(fx - fwidth/2, mean_before_rmse, fwidth, label="對齊前 RMSE", color=COLOR_BEFORE)
        b2 = axes[0].bar(fx + fwidth/2, mean_after_rmse, fwidth, label="對齊後 RMSE", color=COLOR_AFTER)
        axes[0].set_xticks(fx)
        axes[0].set_xticklabels(feature_names)
        axes[0].set_ylabel("RMSE（越低越好）")
        axes[0].set_title("量化指標：各特徵 RMSE + Correlation")
        axes[0].legend(loc="upper left")
        axes[0].grid(axis="y", alpha=0.3)

        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                axes[0].annotate(f"{h:.3f}", (bar.get_x()+bar.get_width()/2, h),
                                 xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

        corr_text = "  |  ".join(
            f"{feature_names[i]} r: {mean_before_corr[i]:.2f} → {mean_after_corr[i]:.2f}"
            for i in range(len(feature_names))
        )
        avg_rmse_before = float(np.nanmean(raw_before))
        avg_rmse_after = float(np.nanmean(raw_after))
        axes[0].text(
            0.5, -0.24,
            f"Overall RMSE: {avg_rmse_before:.4f} → {avg_rmse_after:.4f}  "
            f"（下降 {(avg_rmse_before-avg_rmse_after):+.4f}）\n{corr_text}",
            transform=axes[0].transAxes, ha="center", fontsize=9
        )

        # =====================================================
        # 第二層：原本基礎分數（未套正式最終公式）— 紅／綠配色
        # =====================================================
        b3 = axes[1].bar(x - width/2, original_score_before, width,
                         label="對齊前（No DTW）", color=COLOR_BEFORE)
        b4 = axes[1].bar(x + width/2, original_score_after, width,
                         label="對齊後（DTW）", color=COLOR_AFTER)
        axes[1].set_ylabel("原本基礎分數（0–100）")
        axes[1].set_ylim(0, 100)
        axes[1].set_title(
            "原本分數：未套正式最終評分公式\n"
            "per-path similarity score 取平均（未加入最終加權、1.4x、bonus、AI penalty）"
        )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=20, ha="right")
        axes[1].legend()
        axes[1].grid(axis="y", alpha=0.3)
        for bars in (b3, b4):
            for bar in bars:
                h = bar.get_height()
                axes[1].annotate(f"{h:.1f}", (bar.get_x()+bar.get_width()/2, h),
                                 xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

        # =====================================================
        # 第三層：正式最終分數 — 紅／綠配色
        # =====================================================
        b5 = axes[2].bar(x - width/2, formal_before, width,
                         label="對齊前（No DTW / 正式公式）", color=COLOR_BEFORE)
        b6 = axes[2].bar(x + width/2, formal_after, width,
                         label="對齊後（DTW / 正式公式）", color=COLOR_AFTER)
        axes[2].set_ylabel("正式最終分數（0–100）")
        axes[2].set_ylim(0, 100)
        axes[2].set_title(
            "正式最終分數：套用完整評分公式\n"
            "mean / p50 / p25 / worst + 1.4x + bonus + AI Coach penalty"
        )
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(names, rotation=20, ha="right")
        axes[2].legend()
        axes[2].grid(axis="y", alpha=0.3)
        for bars in (b5, b6):
            for bar in bars:
                h = bar.get_height()
                axes[2].annotate(f"{h:.1f}", (bar.get_x()+bar.get_width()/2, h),
                                 xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

        avg_base_before = float(np.mean(original_score_before))
        avg_base_after = float(np.mean(original_score_after))
        avg_formal_before = float(np.mean(formal_before))
        avg_formal_after = float(np.mean(formal_after))
        axes[1].text(
            0.5, -0.16,
            f"平均：對齊前={avg_base_before:.2f}  對齊後={avg_base_after:.2f}  "
            f"提升={avg_base_after-avg_base_before:+.2f}",
            transform=axes[1].transAxes, ha="center", fontsize=9
        )
        axes[2].text(
            0.5, -0.16,
            f"平均：對齊前={avg_formal_before:.2f}  對齊後={avg_formal_after:.2f}  "
            f"提升={avg_formal_after-avg_formal_before:+.2f}",
            transform=axes[2].transAxes, ha="center", fontsize=9
        )

        fig.suptitle(
            f"DTW 對齊效益：量化指標 → 原本分數 → 正式分數 [{tag}]",
            fontsize=15
        )
        plt.tight_layout(rect=[0, 0.02, 1, 0.97])

        img_path = os.path.join(output_dir, f"{tag}_overall_score_comparison.png")
        plt.savefig(img_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        metrics_df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, f"{tag}_overall_score_comparison.csv")
        metrics_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        print(f"✅ 三層比較圖已存至: {img_path}")
        print(f"✅ 完整量化指標 + 兩種分數已存至: {csv_path}")
        print(metrics_df.to_string(index=False))

        return metrics_df

    # =========================================================
    # ▲▲▲ 新增區塊結束 ▲▲▲
    # =========================================================


def show_alignment_proof_in_streamlit(processor: "PoseProcessor", df_std, df_usr,
                                       output_dir="alignment_proof_output", tag=None):
    """
    在 Streamlit 頁面裡一行呼叫，就能：
      1. 呼叫 processor.plot_alignment_proof() 產生圖檔 + CSV
      2. 用 st.image() 把圖顯示在頁面上
      3. 提供下載按鈕
      4. 用 st.dataframe() 顯示量化指標表
    """
    import streamlit as st
    from datetime import datetime

    if tag is None:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    metrics_df = processor.plot_alignment_proof(df_std, df_usr, output_dir=output_dir, tag=tag)

    if metrics_df is None:
        st.warning("⚠️ 對齊比較圖產生失敗（輸入的 df_std 或 df_usr 是空的）")
        return None

    img_path = os.path.join(output_dir, f"{tag}_alignment_proof.png")

    st.subheader("對齊前後比較圖")
    st.image(img_path, caption=f"對齊比較圖 - {tag}")

    with open(img_path, "rb") as f:
        st.download_button(
            "下載這張對齊比較圖 (PNG)",
            f,
            file_name=f"{tag}_alignment_proof.png",
            mime="image/png",
            key=f"download_img_{tag}",
        )

    st.subheader("這次的量化指標表")
    st.dataframe(metrics_df)

    csv_path = os.path.join(output_dir, f"{tag}_alignment_metrics.csv")
    with open(csv_path, "rb") as f:
        st.download_button(
            "下載這份量化指標表 (CSV)",
            f,
            file_name=f"{tag}_alignment_metrics.csv",
            mime="text/csv",
            key=f"download_csv_{tag}",
        )

    return metrics_df


def show_alignment_proof_history(output_dir="alignment_proof_output"):
    """列出歷史所有對比圖，各自附下載按鈕"""
    import streamlit as st

    if not os.path.isdir(output_dir):
        st.info("目前還沒有任何對比圖紀錄")
        return

    png_files = sorted(
        [f for f in os.listdir(output_dir) if f.endswith("_alignment_proof.png")],
        reverse=True,
    )

    if not png_files:
        st.info("目前還沒有任何對比圖紀錄")
        return

    st.subheader(f"歷史對比圖紀錄（共 {len(png_files)} 筆）")

    for png_name in png_files:
        tag = png_name.replace("_alignment_proof.png", "")
        img_path = os.path.join(output_dir, png_name)
        csv_path = os.path.join(output_dir, f"{tag}_alignment_metrics.csv")

        with st.expander(f"📊 {tag}"):
            st.image(img_path)
            col1, col2 = st.columns(2)
            with col1:
                with open(img_path, "rb") as f:
                    st.download_button(
                        "下載圖片", f, file_name=png_name,
                        mime="image/png", key=f"hist_img_{tag}",
                    )
            with col2:
                if os.path.exists(csv_path):
                    with open(csv_path, "rb") as f:
                        st.download_button(
                            "下載指標表", f, file_name=f"{tag}_alignment_metrics.csv",
                            mime="text/csv", key=f"hist_csv_{tag}",
                        )


def show_overall_score_proof_in_streamlit(processor: "PoseProcessor", sample_pairs,
                                           output_dir="alignment_proof_output", tag=None):
    """
    在 Streamlit 頁面顯示「整體評分系統：對齊前 vs 對齊後」的三層長條圖比較。

    用法（單一影片測試）：
        show_overall_score_proof_in_streamlit(
            proc,
            [("這次分析", df_std_action, df_usr_action)]
        )
    """
    import streamlit as st
    from datetime import datetime

    if tag is None:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    metrics_df = processor.plot_overall_score_proof(sample_pairs, output_dir=output_dir, tag=tag)

    if metrics_df is None:
        st.warning("⚠️ 沒有有效樣本可以比較")
        return None

    img_path = os.path.join(output_dir, f"{tag}_overall_score_comparison.png")

    st.subheader("📊 整體評分系統：未套公式 + 正式公式，對齊前 vs 對齊後")
    st.image(img_path)

    st.dataframe(metrics_df)

    with open(img_path, "rb") as f:
        st.download_button(
            "下載比較圖 (PNG)", f,
            file_name=f"{tag}_overall_score_comparison.png",
            mime="image/png", key=f"overall_img_{tag}",
        )

    csv_path = os.path.join(output_dir, f"{tag}_overall_score_comparison.csv")
    with open(csv_path, "rb") as f:
        st.download_button(
            "下載比較表 (CSV)", f,
            file_name=f"{tag}_overall_score_comparison.csv",
            mime="text/csv", key=f"overall_csv_{tag}",
        )

    return metrics_df