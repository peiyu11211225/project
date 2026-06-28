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
section[data-testid="stSidebar"] .stButton:nth-of-type(1) button{
    width:100%;
    background:#f5f5f5;
    color:#333;
    border-radius:8px;
}

/* 第二顆按鈕：教學示範 */
section[data-testid="stSidebar"] .stButton:nth-of-type(2) button{
    width:100%;
    background:#C98989;
    color:white;
    font-size:22px;
    font-weight:bold;
    padding:14px 0;
    border:none;
    border-radius:12px;
}

section[data-testid="stSidebar"] .stButton:nth-of-type(2) button:hover{
    background:#B87777;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 使用說明 Dialog
# =========================
import os
import streamlit as st
import streamlit.components.v1 as components

@st.dialog("📘 正拍高遠球教學", width="large")
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

    # 純前端完美的左右版面、左右按鈕與動態說明文字
    html_code = f"""
    <style>
        .action-btn {{
            display: inline-block;
            padding: 10px 16px;
            background-color: #930000;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            font-size: 15px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.2s ease;
        }}
        
        .action-btn:hover {{
            opacity: 0.85;
            transform: translateY(-1px);
        }}
        
        /* 啟動狀態的按鈕樣式（點擊後加深顏色） */
        .action-btn.active {{
            background-color: #E27E7E;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
        }}

        /* 2. 說明文字顯示框樣式 */
        .content-box {{
            display: none; /* 預設隱藏 */
            width: 100%;
            margin-top: 20px;
            padding: 15px;
            background-color: #f9f9f9;
            border-left: 5px solid #FF9797; /* 粉紅邊條 */
            border-radius: 4px;
            font-size: 16px;
            font-family: sans-serif;
            line-height: 1.6;
            color: #333;
        }}
    </style>

    <div style="font-family: sans-serif; display: flex; flex-direction: row; align-items: flex-start; justify-content: center; gap: 35px; padding: 10px;">
        
        <div style="flex: 1.3; max-width: 420px; display: flex; justify-content: center;">
            <video id="my-video" style="width: 100%; max-height: 580px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);" controls autoplay muted>
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                您的瀏覽器不支援此影片格式。
            </video>
        </div>

        <div style="flex: 1; display: flex; flex-direction: column; justify-content: flex-start; min-height: 500px; padding-top: 10px;">
            <p style="margin: 0 0 20px 0; font-size: 18px; color: #333; font-weight: bold;">📌 點擊下列關鍵點解說：</p>
            
            <div style="display: flex; flex-direction: row; gap: 10px; width: 100%; justify-content: flex-start;">
                <button id="btn-prep" class="action-btn" onclick="seekAndShow(1.0, 'prep')">🎾 引拍預備</button>
                <button id="btn-hit" class="action-btn" onclick="seekAndShow(4.2, 'hit')">💥 擊球瞬間</button>
                <button id="btn-finish" class="action-btn" onclick="seekAndShow(5.4, 'finish')">🏁 收拍結尾</button>

            </div>
            
            <div id="content-container" style="width: 100%;">
                <div id="text-prep" class="content-box">
                1.側身站位：非持拍手朝向球網，非持拍腳腳尖朝前，持拍腳與球網平行。<br>
                2.球拍後引：持拍手肘抬高至接近肩膀高，球拍舉至頭後方。<br>
                3.非持拍手：指向來球，幫助瞄準落點並維持平衡。</div>
                <div id="text-hit" class="content-box">
                1.全身連貫發力：手腕先自然垂落，腰部發力向前轉身，帶動手臂加速揮拍。<br>
                2.擊球點瞬間：擊球瞬間手臂接近伸直，手腕瞬間發力向前壓球。<br>
                3.擊球點：在身體前上方最高點擊球。
</div>
                <div id="text-finish" class="content-box">
                1.自然收拍：擊球後順勢向前下方揮出，不要急停。<br>
                2.身體轉正：持拍腳順勢向前踩，身體轉回正面，保持平衡。</div>
            </div>

            <p style="font-size: 14px; color: #666; margin-top: 20px; line-height: 1.5; max-width: 300px;">
                <b>使用小提示：</b><br>點擊上方鮮紅色按鈕，影片會立刻瞬移到該動作並自動暫停，方便您精準比對姿勢。
            </p>

        </div>

    </div>

    <script>
    function seekAndShow(seconds, stage) {{
        // 1. 控制影片跳轉與暫停
        var video = document.getElementById('my-video');
        video.currentTime = seconds;
        video.pause();

        // 2. 切換按鈕高亮狀態
        document.getElementById('btn-prep').classList.remove('active');
        document.getElementById('btn-hit').classList.remove('active');
        document.getElementById('btn-finish').classList.remove('active');
        document.getElementById('btn-' + stage).classList.add('active');

        // 3. 切換顯示對應的說明文字
        document.getElementById('text-prep').style.display = 'none';
        document.getElementById('text-hit').style.display = 'none';
        document.getElementById('text-finish').style.display = 'none';
        document.getElementById('text-' + stage).style.display = 'block';
    }}
    
    </script>
    """
    
    components.html(html_code, height=610)

    st.write("---")
    if st.button("關閉說明"):
        st.rerun()

# Sidebar
# =========================
with st.sidebar:

    # 左上角
    if st.button("🏠", key="home"):
        st.switch_page("app.py")

    st.write("")

    # 左右置中
    left, center, right = st.columns([1,3,1])

    with center:
        if st.button(
            "📘 教學示範",
            key="help",
            use_container_width=True
        ):
            show_help_dialog()

    st.divider()

    uploaded_file = st.file_uploader(
        "上傳使用者影片",
        type=["mp4","mov","avi"]
    )

    st.info("⚠️ 注意影片越清晰，分析越準確，影片拍法請比對教學示範影片!!")

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

                    st.markdown(f"""
                    | 項目 | 數值 | 註記 |
                    |------|------|------|
                    | 相似度總計 | `{base:.1f}` | 動作相似程度經專業加權後計算之基礎分 |
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