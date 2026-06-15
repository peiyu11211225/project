import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile
import os

import sys

# 1. 精準取得專案根目錄的絕對路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 2. 【核心修改】直接把根目錄塞到搜尋路徑的最前面（權限最高）
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # 使用 insert(0, ...) 確保優先被看見

# 3. 順便也把 pages 資料夾本身也塞進去，雙重保險
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from pose_utils import get_full_body_angles
from processor import PoseProcessor
from ai_coach import AICoach
from fastdtw import fastdtw

# =========================
# UI 設定
# =========================
st.set_page_config(
    page_title="🏸 AI 羽球分析",
    layout="wide"
)

st.title("🏸 AI羽球教練 : 揮拍動作診斷平台")

# =========================
# CSS
# =========================
st.markdown("""
<style>     
/* 🎯 這行只會拔掉 pages 的檔案名字，不會動到你寫的按鈕和上傳元件 */
[data-testid="stSidebarNav"] {
    display: none !important;
} 
            
video {
    max-width: 800px !important;
    max-height: 600px !important;
    margin: auto;
    display: block;
}

/* sidebar 使用說明按鈕 */
section[data-testid="stSidebar"] .stButton > button {
    width: 140%;
    margin: 0 auto 12px auto;
    display: block;

    background: #a9c7de;

    border: none;
    border-bottom: 4px solid #7395ad;

    border-radius: 10px;

    color: #111111;
    font-weight: 800;
    font-size: 1rem;

    padding: 0.55rem 0;

    cursor: pointer;

    box-shadow: 0 4px 10px rgba(80,110,140,0.25);

    transition:
        background 0.18s ease,
        transform 0.08s ease,
        box-shadow 0.18s ease;
}

/* hover */
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #89afcc;
    box-shadow: 0 5px 14px rgba(80,110,140,0.35);
}

/* 按下 */
section[data-testid="stSidebar"] .stButton > button:active {
    transform: translateY(2px);
    border-bottom: 2px solid #7395ad;
    box-shadow: 0 2px 5px rgba(80,110,140,0.2);
}
</style>
""", unsafe_allow_html=True)

# =========================
# 使用說明 Dialog
# =========================
import os
import streamlit as st
import streamlit.components.v1 as components

@st.dialog("📘 使用說明")
def show_help_dialog():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(current_dir, "IMG_0284.mp4")
    
    if not os.path.exists(video_path):
        st.error(f"找不到影片！預期路徑：{video_path}")
        return

    # 讀取影片並轉為 base64 格式
    import base64
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    video_base64 = base64.b64encode(video_bytes).decode()

    # 優化後的 HTML，加上了 max-height 限制影片高度
    html_code = f"""
    <div style="font-family: sans-serif; display: flex; flex-direction: column; align-items: center;">
        <video id="my-video" style="max-width: 100%; max-height: 350px; border-radius: 8px;" controls autoplay muted>
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            您的瀏覽器不支援此影片格式。
        </video>

        <p style="margin: 15px 0 5px 0; font-size: 14px; color: #555; width: 100%; text-align: left;">📌 <b>快速跳轉至動作要領：</b></p>
        
        <div style="display: flex; justify-content: space-between; width: 100%;">
            <button onclick="seekVideo(1.0)" style="flex: 1; margin: 5px; padding: 12px; background-color: #f0f2f6; border: 1px solid #ddd; border-radius: 6px; cursor: pointer; font-weight: bold;">🎾 引拍預備</button>
            <button onclick="seekVideo(2.5)" style="flex: 1; margin: 5px; padding: 12px; background-color: #f0f2f6; border: 1px solid #ddd; border-radius: 6px; cursor: pointer; font-weight: bold;">💥 擊球</button>
            <button onclick="seekVideo(4.5)" style="flex: 1; margin: 5px; padding: 12px; background-color: #f0f2f6; border: 1px solid #ddd; border-radius: 6px; cursor: pointer; font-weight: bold;">🏁 收拍結尾</button>
        </div>
    </div>

    <script>
    function seekVideo(seconds) {{
        var video = document.getElementById('my-video');
        video.currentTime = seconds;
        video.play();
    }}
    </script>
    """
    
    # 這裡將總高度放大到 500，確保完整容納影片、文字跟按鈕
    components.html(html_code, height=500)

    st.write("---")
    if st.button("關閉"):
        st.rerun()
# Sidebar
# =========================
with st.sidebar:

    if st.button("🏠 返回首頁", use_container_width=False):
        st.switch_page("app.py")

    if st.button("📘 使用說明", use_container_width=False):
        show_help_dialog()

    uploaded_file = st.file_uploader(
        "上傳使用者影片",
        type=['mp4', 'mov', 'avi']
    )

    st.info("⚠️ 注意影片越清晰，分析越準確，建議全身入鏡")

# =========================
# 主流程
# =========================
if uploaded_file:

    # -------------------------
    # 1. 存檔
    # -------------------------
    tfile = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    tfile.write(uploaded_file.read())

    video_path = tfile.name

    tfile.close()

    status_box = st.empty()
    status_box.info("🔄 MediaPipe 分析中...")

    prog = st.progress(0)

    records = []

    # -------------------------
    # 2. MediaPipe
    # -------------------------
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    # ─── 🔥 雲端關鍵優化設定 ───
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=os.path.join(project_root, "pose_landmarker.task"),
            # 強制指定使用 CPU 運算，徹底根除雲端的 GPU/EGL 初始化閃退錯誤
            delegate=BaseOptions.Delegate.CPU
        ),
        running_mode=RunningMode.VIDEO
    )

    with PoseLandmarker.create_from_options(options) as landmarker:

        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        i = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            # 將 OpenCV 的 BGR 轉換為 MediaPipe 影像格式
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame
            )

            timestamp_ms = int(
                i * (1000 / fps)
            )

            # 執行偵測
            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            data = {}

            # ─── 🔥 修正：增強偵測結果的防禦性 ───
            if result and result.pose_landmarks and len(result.pose_landmarks) > 0:

                lms = result.pose_landmarks[0]

                try:
                    coord = {
                        k: (lms[k].x, lms[k].y)
                        for k in range(33)
                    }

                    # 計算角度 (保留你原本的邏輯)
                    data = get_full_body_angles(coord)
                    if not isinstance(data, dict):
                        data = {}

                except:
                    data = {}

                # 寫入每個關鍵點的 x, y 座標
                for j in range(11, 33):
                    data[f"{j}_x"] = float(lms[j].x)
                    data[f"{j}_y"] = float(lms[j].y)

            else:
                # 🔥 雲端防空鎖：如果這一格沒偵測到人，填入 0.5 兜底，並確保所有欄位名稱完整存在
                data = {}
                for j in range(11, 33):
                    data[f"{j}_x"] = 0.5
                    data[f"{j}_y"] = 0.5

            records.append(data)

            i += 1

            # 更新 Streamlit 進度條
            if i % 10 == 0 and total_frames > 0:
                prog.progress(
                    min(i / total_frames, 1.0)
                )

        cap.release()

        status_box.empty()
        prog.empty()
    # =========================
    # 3. 分析
    # =========================
    if len(records) > 10:

        proc = PoseProcessor()

        coach_ai = AICoach()

        df_usr_full = (
            pd.DataFrame(records)
            .ffill()
            .bfill()
            .fillna(0)
        )

        start_auto, peak, end_auto = (
            proc.detect_action_range(df_usr_full)
        )

        start = max(0, int(start_auto))

        end = min(
            len(df_usr_full) - 1,
            int(end_auto)
        )

        if end <= start:
            end = start + 30

        df_usr_action = (
            df_usr_full
            .iloc[start:end+1]
            .reset_index(drop=True)
        )

        # -------------------------
        # 標準動作
        # -------------------------
        csv_path = os.path.join(current_dir, "standard_swing.csv")

        if os.path.exists(csv_path):

            df_std_action = pd.read_csv(csv_path)

            score, stats = (
                proc.calculate_auto_similarity(
                    df_std_action,
                    df_usr_action
                )
            )

            curve = proc.compute_similarity_curve(
                df_std_action,
                df_usr_action
            )

            feat_std = proc.extract_features(
                df_std_action
            )

            feat_usr = proc.extract_features(
                df_usr_action
            )

            distance, path = fastdtw(
                feat_std,
                feat_usr
            )

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

                    st.line_chart(
                        pd.DataFrame(
                            curve,
                            columns=["相似度"]
                        )
                    )

            with col2:

                st.subheader("📊 分析")

                st.metric(
                    "分數",
                    f"{score:.1f}"
                )

                with st.expander("📋 查看分數組成"):

                    mean = stats['mean_path_score']
                    p50 = stats['p50']
                    p25 = stats['p25']
                    worst = stats['min']
                    std = stats['std']
                    penalty = stats['penalty']

                    a = mean * 0.70
                    b = p50 * 0.25
                    c = p25 * 0.10
                    d = worst * 0.15

                    base = (
                        a + b + c + d
                    ) * 1.2

                    std_pen = std * 0.05

                    worst_pen = (
                        5 if worst < 40 else 0
                    )

                    st.markdown(f"""
                    | 項目 | 數值 | 註記 |
                    |------|------|------|
                    | 相似度總計 | `{base:.1f}` | 動作相似程度經專業加權後計算之基礎分 |
                    | − 標準差懲罰 | `−{std_pen:.1f}` | 動作穩定性懲罰 |
                    | − 最低分懲罰 | `−{worst_pen}` | 單一動作片段過差時額外扣分 |
                    | − AI教練懲罰 | `−{penalty:.1f}` | AI 教練動作誤差扣分 |
                    | **最終分數** | **`{score:.1f}`** | 最終成績 |
                    """)

                # =========================
                # AI 教練
                # =========================
                st.subheader("🧠 教練回饋")

                if path is None or len(path) == 0:

                    st.warning(
                        "無法對齊動作，請確認影片品質"
                    )

                else:

                    feedback_list, overall, penalty = (
                        coach_ai.generate_feedback(
                            feat_std,
                            feat_usr,
                            path,
                            curve,
                            score
                        )
                    )

                    phase_labels = [
                        "📌 引拍",
                        "💥 擊球",
                        "🔄 收拍"
                    ]

                    if feedback_list:

                        for label, f in zip(
                            phase_labels,
                            feedback_list
                        ):

                            st.markdown(
                                f"**{label}**"
                            )

                            items = (
                                f if isinstance(f, list)
                                else [f]
                            )

                            for item in items:

                                st.markdown(
                                    f"&nbsp;&nbsp;• {item}",
                                    unsafe_allow_html=True
                                )

                        st.markdown("---")

                        st.markdown(
                            "### 🏆 整體評語"
                        )

                        st.markdown(
                            f"**{overall}**"
                        )

                    else:
                        st.info("未產生回饋")

        else:
            st.error(
                "找不到 standard_swing.csv"
            )

    if os.path.exists(video_path):
        os.remove(video_path)

else:
    st.info("請上傳影片")