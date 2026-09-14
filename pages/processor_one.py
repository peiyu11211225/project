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
    # 相似度評分 (DTW 優化版)
    # =========================
    def calculate_auto_similarity(self, df_std, df_usr):
        if df_std.empty or df_usr.empty:
            return 0.0, {"feedback": "未偵測到有效動作數據", "penalty": 0}

        feat_std = self.extract_features(df_std)
        feat_usr = self.extract_features(df_usr)

        _, path = fastdtw(feat_std, feat_usr, dist=euclidean)

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

        path_scores = np.array(path_scores)
        n = len(path_scores)

        if n > 20:
            trim = int(n * 0.05)
            path_scores = path_scores[trim:n - trim]

        if len(path_scores) == 0:
            return 0.0, {"feedback": "動作比對路徑無效", "penalty": 0}

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

        final_score *= 1.4

        if mean > 85:
            final_score += 5
        elif mean > 75:
            final_score += 1

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

        參數:
            df_std: 教練標準動作 DataFrame（跟其他方法輸入格式一致）
            df_usr: 使用者動作 DataFrame
            output_dir: 輸出資料夾（會自動建立）
            tag: 這次比對的名稱，會用在檔名上（例如使用者ID、動作代碼）
            feature_names: extract_features 輸出的四個維度對應的中文名稱

        回傳:
            metrics_df: 每個特徵對齊前後的 RMSE / 對齊後相關係數
            (若失敗則回傳 None)
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
            axes = np.array([axes])  # 確保可用 axes[i, 0] / axes[i, 1] 的方式索引

        metrics_rows = []

        for f_idx in range(n_feat):
            name = feature_names[f_idx] if f_idx < len(feature_names) else f"feature_{f_idx}"

            # ---- 對齊前：直接逐幀比對（取重疊長度，模擬「沒有 DTW」的情況）----
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

            # ---- 對齊後：沿 fastdtw path 取值 ----
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
# ▼▼▼ 新增：Streamlit 頁面用的顯示包裝函式 ▼▼▼
# =========================================================
def show_alignment_proof_in_streamlit(processor: "PoseProcessor", df_std, df_usr,
                                       output_dir="alignment_proof_output", tag=None):
    """
    在 Streamlit 頁面裡一行呼叫，就能：
      1. 呼叫 processor.plot_alignment_proof() 產生圖檔 + CSV
         （若沒給 tag，會自動用「時間戳記」命名，例如 20260915_143022，
         這樣每次跑都會是新檔案，不會把前一次的結果覆蓋掉）
      2. 用 st.image() 把圖顯示在頁面上（因為 Streamlit Cloud 的檔案系統
         本機看不到，要靠這個才能親眼確認圖有沒有畫出來）
      3. 提供下載按鈕，讓你能把這次的圖存回本機
      4. 用 st.dataframe() 顯示量化指標表

    用法（在你的 Streamlit 頁面，例如 pages/xxx.py）：

        from pose_processor import PoseProcessor, show_alignment_proof_in_streamlit

        processor = PoseProcessor()
        show_alignment_proof_in_streamlit(processor, df_std, df_usr)

    注意：這個函式內部才 import streamlit，所以 pose_processor.py
    本身仍然可以在非 Streamlit 環境（例如純命令列腳本）下正常使用，
    不會因為缺少 streamlit 套件而整支檔案 import 失敗。
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

    st.subheader("這次的對齊前後比較圖")
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
    """
    列出 output_dir 裡「過去所有」跑過的對比圖，
    每一張都附下載按鈕 —— 不用重新跑一次也能拿到舊的結果。

    用法（放在頁面任何位置，通常放在最下面當作歷史紀錄區）：

        from pose_processor import show_alignment_proof_history
        show_alignment_proof_history()
    """
    import streamlit as st

    if not os.path.isdir(output_dir):
        st.info("目前還沒有任何對比圖紀錄")
        return

    png_files = sorted(
        [f for f in os.listdir(output_dir) if f.endswith("_alignment_proof.png")],
        reverse=True,  # 最新的排前面
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
# =========================================================
# ▲▲▲ 新增結束 ▲▲▲
# =========================================================