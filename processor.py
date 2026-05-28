import numpy as np
import pandas as pd
import cv2
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
    # similarity
    # =========================
    def calculate_auto_similarity(self, df_std, df_usr):

        feat_std = self.extract_features(df_std)
        feat_usr = self.extract_features(df_usr)

        _, path = fastdtw(feat_std, feat_usr, dist=euclidean)

        u_to_s_map = defaultdict(list)
        for s, u in path:
            u_to_s_map[u].append(s)

        path_scores = []

        for s, u in path:
            diff = np.abs(feat_std[s] - feat_usr[u])

            weighted_error = (
                diff[0] * 1.0 +
                diff[1] * 1.5 +
                diff[2] * 1.2
            )

            score = 100 * np.exp(-2.0 * weighted_error)
            path_scores.append(score)

        path_scores = np.array(path_scores)

        if len(path_scores) > 20:
            trim = int(len(path_scores) * 0.05)
            path_scores = path_scores[trim:-trim]

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

        final_score -= penalty
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
    # overlay video (FIXED)
    # =========================
    def generate_auto_overlay(self, video_path, df_std, df_usr, start_idx, output_path):

        feat_std = self.extract_features(df_std)
        feat_usr = self.extract_features(df_usr)

        _, path = fastdtw(feat_std, feat_usr, dist=euclidean)

        u_to_s_map = defaultdict(list)
        for s, u in path:
            u_to_s_map[u].append(s)

        cap = cv2.VideoCapture(video_path)

        ret, test_frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot read video")

        h, w = test_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1:
            fps = 30

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (w, h)
        )

        if not out.isOpened():
            raise RuntimeError("VideoWriter failed to open")

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
                    s_idx = u_to_s_map[rel_idx][0]
                    c_row = df_std.iloc[s_idx]

                    self.draw_skeleton(frame, c_row, (255, 0, 0), 3, w, h)

            if frame is None:
                continue

            if frame.shape[-1] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            frame = np.ascontiguousarray(frame, dtype=np.uint8)

            out.write(frame)

            f_idx += 1

        cap.release()
        out.release()

    # =========================
    # draw skeleton
    # =========================
    def draw_skeleton(self, frame, row, color, thickness, w, h):

        pts = {}

        for i in range(11, 33):
            xk, yk = f"{i}_x", f"{i}_y"

            if xk in row and yk in row:
                x, y = row[xk], row[yk]

                if x != 0 and y != 0:
                    pts[i] = (int(x * w), int(y * h))

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
            return 0, len(df)//2, len(df)-1