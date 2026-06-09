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


def run_mediapipe(uploaded_file):
    """MediaPipe 骨架偵測，回傳 records list"""

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    tfile.close()

    status_box = st.empty()
    status_box.info("🔄 MediaPipe 分析中...")
    prog = st.progress(0)

    records = []

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=os.path.join(os.path.dirname(__file__), "pose_landmarker.task"),
            delegate=BaseOptions.Delegate.CPU
        ),
        running_mode=RunningMode.VIDEO
    )

    with PoseLandmarker.create_from_options(options) as landmarker:

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        i = 0

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp_ms = int(i * (1000 / fps))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            data = {}

            if result and result.pose_landmarks and len(result.pose_landmarks) > 0:
                lms = result.pose_landmarks[0]
                try:
                    coord = {k: (lms[k].x, lms[k].y) for k in range(33)}
                    data = get_full_body_angles(coord)
                    if not isinstance(data, dict):
                        data = {}
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

    return video_path, records


def run_analysis(uploaded_file, standard_csv_path, swing_type="高遠球（正拍）"):
    """
    完整分析流程，吃上傳影片 + 標準動作 csv
    swing_type: 用於顯示名稱
    """

    video_path, records = run_mediapipe(uploaded_file)

    if len(records) <= 10:
        st.error("影片過短或偵測失敗，請重新上傳")
        return

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

    df_usr_action = df_usr_full.iloc[start:end+1].reset_index(drop=True)

    if not os.path.exists(standard_csv_path):
        st.error(f"找不到標準動作檔案：{standard_csv_path}")
        return

    df_std_action = pd.read_csv(standard_csv_path)

    score, stats = proc.calculate_auto_similarity(df_std_action, df_usr_action)
    curve = proc.compute_similarity_curve(df_std_action, df_usr_action)
    feat_std = proc.extract_features(df_std_action)
    feat_usr = proc.extract_features(df_usr_action)
    distance, path = fastdtw(feat_std, feat_usr)

    # ── UI ──
    col1, col2 = st.columns([2, 1])

    with col1:

        st.subheader("🎥 動作對照影片")
        output_path = f"overlay_{swing_type}.mp4"

        with st.spinner("生成影片中..."):
            proc.generate_auto_overlay(
                video_path, df_std_action, df_usr_action, start, output_path
            )

        if os.path.exists(output_path):
            st.video(output_path)

        st.subheader("📈 相似度曲線")
        if curve:
            st.line_chart(pd.DataFrame(curve, columns=["相似度"]))

    with col2:

        st.subheader("📊 分析結果")
        st.metric("動作分數", f"{score:.1f}")

        with st.expander("📋 分數組成"):
            mean  = stats['mean_path_score']
            p50   = stats['p50']
            p25   = stats['p25']
            worst = stats['min']
            std   = stats['std']

            base    = (mean * 0.70 + p50 * 0.25 + p25 * 0.10 + worst * 0.15) * 1.2
            std_pen = std * 0.05
            worst_pen = 5 if worst < 40 else 0

            st.markdown(f"""
| 項目 | 數值 |
|------|------|
| 相似度基礎分 | `{base:.1f}` |
| − 穩定度懲罰 | `−{std_pen:.1f}` |
| − 最低分懲罰 | `−{worst_pen}` |
| − AI教練懲罰 | `−{stats['penalty']:.1f}` |
| **最終分數** | **`{score:.1f}`** |
""")

        st.subheader("🧠 教練回饋")

        if path is None or len(path) == 0:
            st.warning("無法對齊動作，請確認影片品質")
        else:
            feedback_list, overall, penalty = coach_ai.generate_feedback(
                feat_std, feat_usr, path, curve, score
            )

            phase_labels = ["📌 引拍", "💥 擊球", "🔄 收拍"]

            if feedback_list:
                for label, f in zip(phase_labels, feedback_list):
                    st.markdown(f"**{label}**")
                    items = f if isinstance(f, list) else [f]
                    for item in items:
                        st.markdown(f"&nbsp;&nbsp;• {item}", unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("### 🏆 整體評語")
                st.markdown(f"**{overall}**")
            else:
                st.info("未產生回饋")

    if os.path.exists(video_path):
        os.remove(video_path)