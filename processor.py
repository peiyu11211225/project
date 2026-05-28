import numpy as np
import pandas as pd
import cv2
import os
import subprocess
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from ai_coach import AICoach
from collections import defaultdict

class PoseProcessor:

    def __init__(self):
        self.CONNECTIONS = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28)
        ]
        self.coach = AICoach()
    def _hip_transform(self, row):
        """把 skeleton 轉到 hip-centered + normalized space"""
        l = np.array([row['23_x'], row['23_y']])
        r = np.array([row['24_x'], row['24_y']])

        center = (l + r) / 2
        width = np.linalg.norm(r - l)

        if width < 1e-6:
            width = 1.0

        out = {}
        for i in range(11, 33):
            x = row.get(f"{i}_x", 0.5)
            y = row.get(f"{i}_y", 0.5)

            pt = np.array([x, y])

            pt = (pt - center) / width   # ⭐ normalize

            out[f"{i}_x"] = pt[0]
            out[f"{i}_y"] = pt[1]

        return out


    def align_to_user_space(self, coach_row, user_row):

        # hip center (normalized space)
        c_l = np.array([coach_row['23_x'], coach_row['23_y']])
        c_r = np.array([coach_row['24_x'], coach_row['24_y']])
        c_center = (c_l + c_r) / 2

        u_l = np.array([user_row['23_x'], user_row['23_y']])
        u_r = np.array([user_row['24_x'], user_row['24_y']])
        u_center = (u_l + u_r) / 2

        # hip width scale
        c_w = np.linalg.norm(c_r - c_l)
        u_w = np.linalg.norm(u_r - u_l)

        scale = (u_w / c_w) if c_w > 1e-6 else 1.0

        out = {}

        for i in range(11, 33):

            x = coach_row.get(f"{i}_x")
            y = coach_row.get(f"{i}_y")

            if x is None or y is None:
                continue

            pt = np.array([x, y])

            # normalize to coach hip center
            pt = pt - c_center

            # scale to user size
            pt = pt * scale

            # move to user hip center
            pt = pt + u_center

            out[f"{i}_x"] = float(pt[0])
            out[f"{i}_y"] = float(pt[1])

        return out
    # =========================
    # feature extraction（穩定版）
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
                    np.tanh(speed * 5),  # 正規化速度
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
 
        joint_weights = {
            0: 1.0,
            1: 1.5,
            2: 1.2
        }
 
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
 
        # =========================
        # trim DTW boundary noise
        # =========================
        n = len(path_scores)
        trim = int(n * 0.05)
 
        if n > 20:
            path_scores = path_scores[trim:n - trim]
 
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
        final_score -= std * 0.05
 
        if mean > 85:
            final_score += 5
        elif mean > 75:
            final_score += 1
 
        if worst < 40:
            final_score -= 5
 
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
    # curve（和評分一致）
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
    # overlay video
    # =========================
    def generate_auto_overlay(self, video_path, df_std, df_usr, start_idx, output_path):

        feat_std = self.extract_features(df_std)
        feat_usr = self.extract_features(df_usr)

        _, path = fastdtw(feat_std, feat_usr, dist=euclidean)

        u_to_s_map = defaultdict(list)
        for s, u in path:
            u_to_s_map[u].append(s)

        cap = cv2.VideoCapture(video_path)

        ret, frame0 = cap.read()
        if not ret:
            raise RuntimeError("Cannot read video")

        h, w = frame0.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 1:
            fps = 30

        tmp_path = output_path.replace(".mp4", "_tmp.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))

        if not out.isOpened():
            raise RuntimeError("VideoWriter failed (codec issue)")

        f_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame is None:
                continue

            if frame.shape[-1] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            frame = np.ascontiguousarray(frame, dtype=np.uint8)

            rel_idx = f_idx - start_idx

            if 0 <= rel_idx < len(df_usr):

                u_row = df_usr.iloc[rel_idx]
                self.draw_skeleton(frame, u_row, (0, 0, 255), 2, w, h)

                if rel_idx in u_to_s_map:
                    s_idx = u_to_s_map[rel_idx][0]

                    if s_idx < len(df_std):
                        c_row = df_std.iloc[s_idx]
                        self.draw_skeleton(frame, c_row, (255, 0, 0), 3, w, h)

            out.write(frame)
            f_idx += 1

        cap.release()
        out.release()

        # =========================
        # 🔥 FFmpeg FIX (KEY FIX)
        # =========================
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-i", tmp_path,
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                output_path
            ]

            subprocess.run(cmd, check=True)

            os.remove(tmp_path)

        except Exception as e:
            # fallback：直接用 tmp
            os.rename(tmp_path, output_path)
 
    # =========================
    # draw skeleton
    # =========================
    def draw_skeleton(self, frame, row, color, thickness, w, h):

        pts = {}

        for i in range(11, 33):

            x = row.get(f"{i}_x")
            y = row.get(f"{i}_y")

            if x is None or y is None:
                continue

            # ⚠️ ONLY HERE convert to pixel
            px = int(np.clip(x, 0, 1) * w)
            py = int(np.clip(y, 0, 1) * h)

            pts[i] = (px, py)

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