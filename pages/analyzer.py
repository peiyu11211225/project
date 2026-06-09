import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile
import os

from pose_utils import get_full_body_angles
from processor import PoseProcessor
from ai_coach import AICoach
from fastdtw import fastdtw

# =========================
# Page Header（注意：不要 set_page_config）
# =========================
st.title("🏸 AI 羽球教練 - 揮拍動作分析")

st.markdown("上傳影片後，系統將自動進行動作分析與評分")

# =========================
# Sidebar（保留簡化版）
# =========================
with st.sidebar:
    st.header("操作")

    uploaded_file = st.file_uploader(
        "上傳羽球揮拍影片",
        type=["mp4", "mov", "avi"]
    )

    st.info("建議：全身入鏡、光線充足、動作完整")

# =========================
# 沒上傳影片
# =========================
if not uploaded_file:
    st.stop()

# =========================
# 暫存影片
# =========================
tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
tfile.write(uploaded_file.read())
video_path = tfile.name
tfile.close()

status_box = st.empty()
status_box.info("🔄 MediaPipe 分析中...")

prog = st.progress(0)
records = []

# =========================
# MediaPipe 初始化
# =========================
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=os.path.join(
            os.path.dirname(__file__),
            "pose_landmarker.task"
        ),
        delegate=BaseOptions.Delegate.CPU
    ),
    running_mode=RunningMode.VIDEO
)

# =========================
# Pose 分析
# =========================
with PoseLandmarker.create_from_options(options) as landmarker:

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )

        timestamp_ms = int(i * (1000 / fps))

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        data = {}

        if result and result.pose_landmarks and len(result.pose_landmarks) > 0:

            lms = result.pose_landmarks[0]

            try:
                coord = {k: (lms[k].x, lms[k].y) for k in range(33)}
                data = get_full_body_angles(coord) or {}

            except:
                data = {}

            for j in range(11, 33):
                data[f"{j}_x"] = float(lms[j].x)
                data[f"{j}_y"] = float(lms[j].y)

        else:
            for j in range(11, 33):
                data[f"{j}_x"] = 0.5
                data[f"{j}_y"] = 0.5

        records.append(data)
        i += 1

        if i % 10 == 0 and total_frames > 0:
            prog.progress(min(i / total_frames, 1.0))

    cap.release()

status_box.empty()
prog.empty()

# =========================
# 檢查資料量
# =========================
if len(records) < 10:
    st.error("影片偵測失敗，請使用更清晰的影片")
    st.stop()

# =========================
# Pose Processor
# =========================
proc = PoseProcessor()
coach_ai = AICoach()

df_usr_full = (
    pd.DataFrame(records)
    .ffill()
    .bfill()
    .fillna(0)
)

start_auto, peak, end_auto = proc.detect_action_range(df_usr_full)

start = max(0, int(start_auto))
end = min(len(df_usr_full) - 1, int(end_auto))

if end <= start:
    end = start + 30

df_usr_action = df_usr_full.iloc[start:end + 1].reset_index(drop=True)

# =========================
# 標準動作
# =========================
if not os.path.exists("standard_swing.csv"):
    st.error("缺少 standard_swing.csv")
    st.stop()

df_std_action = pd.read_csv("standard_swing.csv")

score, stats = proc.calculate_auto_similarity(df_std_action, df_usr_action)
curve = proc.compute_similarity_curve(df_std_action, df_usr_action)

feat_std = proc.extract_features(df_std_action)
feat_usr = proc.extract_features(df_usr_action)

distance, path = fastdtw(feat_std, feat_usr)

# =========================
# UI
# =========================
col1, col2 = st.columns([2, 1])

with col1:

    st.subheader("🎥 動作影片")

    output_path = "overlay_output.mp4"

    with st.spinner("生成影片中..."):
        proc.generate_auto_overlay(
            video_path,
            df_std_action,
            df_usr_action,
            start,
            output_path
        )

    if os.path.exists(output_path):
        st.video(output_path)

    st.subheader("📈 相似度曲線")

    if curve:
        st.line_chart(pd.DataFrame(curve, columns=["相似度"]))

with col2:

    st.subheader("📊 分數")

    st.metric("總分", f"{score:.1f}")

    with st.expander("分數細節"):

        mean = stats['mean_path_score']
        p50 = stats['p50']
        p25 = stats['p25']
        worst = stats['min']
        std = stats['std']
        penalty = stats['penalty']

        base = (mean * 0.7 + p50 * 0.25 + p25 * 0.1 + worst * 0.15) * 1.2
        std_pen = std * 0.05
        worst_pen = 5 if worst < 40 else 0

        st.markdown(f"""
        | 項目 | 數值 |
        |------|------|
        | 基礎分 | {base:.1f} |
        | 標準差懲罰 | -{std_pen:.1f} |
        | 最低分懲罰 | -{worst_pen} |
        | AI懲罰 | -{penalty:.1f} |
        | 最終分數 | **{score:.1f}** |
        """)

    st.subheader("🧠 教練回饋")

    if path and len(path) > 0:

        feedback_list, overall, penalty = coach_ai.generate_feedback(
            feat_std,
            feat_usr,
            path,
            curve,
            score
        )

        labels = ["📌 引拍", "💥 擊球", "🔄 收拍"]

        for label, fb in zip(labels, feedback_list):

            st.markdown(f"**{label}**")

            items = fb if isinstance(fb, list) else [fb]

            for item in items:
                st.markdown(f"- {item}")

        st.markdown("### 🏆 整體評語")
        st.markdown(overall)

    else:
        st.warning("無法對齊動作")

# =========================
# cleanup
# =========================
os.remove(video_path)