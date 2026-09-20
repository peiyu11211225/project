import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile
import os
import sys
import base64
import hashlib

import streamlit.components.v1 as components
from fastdtw import fastdtw

# =========================================================
# 1. 專案路徑
# =========================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


# =========================================================
# 2. 匯入自己的模組
# =========================================================

from pose_utils import get_full_body_angles
from processor_two import (
    PoseProcessor,
    show_alignment_proof_in_streamlit,
    show_overall_score_proof_in_streamlit
)
from ai_coach_two import AICoach


# =========================================================
# 3. Page 設定
# =========================================================

st.set_page_config(
    page_title="🏸 AI 羽球分析",
    layout="wide"
)

st.title("🏸 AI羽球教練 : 反拍高遠球揮拍動作診斷")


# =========================================================
# 4. CSS
# =========================================================

st.markdown("""
<style>

/* =========================================================
   拔掉 pages 的檔案名字
   ========================================================= */

[data-testid="stSidebarNav"] {
    display: none !important;
}


/* =========================================================
   影片
   ========================================================= */

video {
    max-width: 800px !important;
    max-height: 600px !important;
    margin: auto;
    display: block;
}


/* =========================================================
   回首頁按鈕
   ========================================================= */

.home-btn div button,
.home-btn button {
    width: 100px !important;
    min-width: 100px !important;
    padding: 8px 20px !important;
    border-radius: 8px !important;
    background: #a9c7de !important;
}


/* =========================================================
   教學示範按鈕
   ========================================================= */

section[data-testid="stSidebar"] [data-testid="stColumn"] button,
section[data-testid="stSidebar"] [data-testid="stColumn"] button * {
    background-color: #C97B7B !important;
    color: white !important;
    font-size: 26px !important;
    font-weight: 550 !important;
    letter-spacing: 2px !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 10px rgba(201,123,123,0.3) !important;
}

section[data-testid="stSidebar"] [data-testid="stColumn"] button:hover {
    background-color: #B86A6A !important;
    color: white !important;
}


/* =========================================================
   詳細比對按鈕
   ========================================================= */

.detail-result-btn button {
    font-size: 18px !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 10px !important;
}


/* =========================================================
   分析完成提示
   ========================================================= */

.analysis-done {
    padding: 12px 16px;
    border-radius: 10px;
    background-color: #eef7ee;
    border: 1px solid #b7d7b7;
    color: #285c28;
    font-weight: 600;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 5. 初始化 Session State
# =========================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "uploaded_file_hash" not in st.session_state:
    st.session_state.uploaded_file_hash = None

if "video_path" not in st.session_state:
    st.session_state.video_path = None

if "overlay_output_path" not in st.session_state:
    st.session_state.overlay_output_path = None


# =========================================================
# 6. 使用說明 Dialog
# =========================================================

@st.dialog("📘 反拍高遠球教學", width="large")
def show_help_dialog():

    video_path = os.path.join(
        current_dir,
        "IMG_0285.mp4"
    )

    if not os.path.exists(video_path):
        st.error(
            f"找不到影片！預期路徑：{video_path}"
        )
        return

    with open(video_path, "rb") as f:
        video_bytes = f.read()

    video_base64 = base64.b64encode(
        video_bytes
    ).decode()

    html_code = f"""
    <style>

        .action-btn {{
            display: inline-block;
            padding: 14px 20px;
            background-color: #930000;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            font-size: 18px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.2s ease;
            margin: 0 8px;
        }}

        .action-btn:hover {{
            opacity: 0.85;
            transform: translateY(-1px);
        }}

        .action-btn.active {{
            background-color: #E27E7E;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
        }}

        .content-box {{
            display: none;
            width: 100%;
            margin-top: 60px;
            padding: 15px;
            background-color: #f9f9f9;
            border-left: 5px solid #FF9797;
            border-radius: 4px;
            font-size: 18px;
            font-family: sans-serif;
            font-weight: 650;
            line-height: 1.6;
            color: #333;
        }}

    </style>


    <div style="
        font-family: sans-serif;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        justify-content: center;
        gap: 35px;
        padding: 10px;
    ">

        <div style="
            flex: 1.3;
            max-width: 420px;
            display: flex;
            justify-content: center;
        ">

            <video
                id="my-video"
                style="
                    width: 100%;
                    max-height: 580px;
                    border-radius: 12px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                "
                controls
                autoplay
                muted
            >

                <source
                    src="data:video/mp4;base64,{video_base64}"
                    type="video/mp4"
                >

                您的瀏覽器不支援此影片格式。

            </video>

        </div>


        <div style="
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            min-height: 500px;
            padding-top: 10px;
        ">

            <p style="
                margin: 0 0 20px 0;
                font-size: 18px;
                color: #333;
                font-weight: bold;
            ">
                📌 點擊下列關鍵點解說：
            </p>


            <div style="
                display: flex;
                flex-direction: row;
                gap: 10px;
                width: 100%;
                justify-content: flex-start;
            ">

                <button
                    id="btn-prep"
                    class="action-btn"
                    onclick="seekAndShow(1.3, 'prep')"
                >
                    🎾 引拍預備
                </button>

                <button
                    id="btn-hit"
                    class="action-btn"
                    onclick="seekAndShow(2.5, 'hit')"
                >
                    💥 擊球瞬間
                </button>

                <button
                    id="btn-finish"
                    class="action-btn"
                    onclick="seekAndShow(6.3, 'finish')"
                >
                    🏁 收拍結尾
                </button>

            </div>


            <div id="content-container" style="width: 100%;">

                <div id="text-prep" class="content-box">

                    1.背向側身站位：向後轉並側身，持拍腳向後方踩，背向球網。<br>

                    2.球拍引拍：持拍手腕自然垂落，手肘高舉至肩膀附近，反拍握法，大拇指按在球拍側面。<br>

                    3.非持拍手：自然垂落，保持身體平衡即可。

                </div>


                <div id="text-hit" class="content-box">

                    1.揮拍連貫：持拍手向網子方向前揮。<br>

                    2.擊球點瞬間：以手腕瞬間前壓發力。<br>

                    3.擊球點：在身體前上方高點擊球。

                </div>


                <div id="text-finish" class="content-box">

                    1.身體回正：擊球瞬間後腰部及持拍腳向前自然轉回正面面對球網。<br>

                    2.球拍收拍：球拍自然向前下方收拍至持拍腳右邊。

                </div>

            </div>


            <p style="
                font-size: 14px;
                color: #666;
                line-height: 1.5;
                max-width: 300px;
                margin-top: 40px;
            ">

                <b>使用小提示：</b><br>

                點擊上方鮮紅色按鈕，影片會立刻瞬移到該動作並自動暫停，
                方便您精準比對姿勢。

            </p>

        </div>

    </div>


    <script>

    function seekAndShow(seconds, stage) {{

        var video = document.getElementById('my-video');

        video.currentTime = seconds;

        video.pause();


        document
            .getElementById('btn-prep')
            .classList
            .remove('active');

        document
            .getElementById('btn-hit')
            .classList
            .remove('active');

        document
            .getElementById('btn-finish')
            .classList
            .remove('active');


        document
            .getElementById('btn-' + stage)
            .classList
            .add('active');


        document
            .getElementById('text-prep')
            .style
            .display = 'none';

        document
            .getElementById('text-hit')
            .style
            .display = 'none';

        document
            .getElementById('text-finish')
            .style
            .display = 'none';


        document
            .getElementById('text-' + stage)
            .style
            .display = 'block';

    }}

    </script>
    """

    components.html(
        html_code,
        height=610
    )

    st.write("---")

    if st.button(
        "關閉說明",
        use_container_width=True
    ):
        st.rerun()


# =========================================================
# 7. 詳細比對 Dialog
# =========================================================

@st.dialog("🔍 詳細比對結果", width="large")
def show_comparison_dialog():

    # -----------------------------------------------------
    # 從 Session State 取得已經算好的資料
    # -----------------------------------------------------

    result = st.session_state.analysis_result

    if result is None:

        st.error("目前沒有可顯示的分析結果。")

        if st.button(
            "關閉",
            use_container_width=True
        ):
            st.rerun()

        return


    proc = result["proc"]

    df_std_action = result["df_std_action"]

    df_usr_action = result["df_usr_action"]


    # =====================================================
    # 動作對齊
    # =====================================================

    st.subheader("📐 動作對齊比對")

    show_alignment_proof_in_streamlit(
        proc,
        df_std_action,
        df_usr_action
    )


    st.divider()


    # =====================================================
    # 整體分數
    # =====================================================

    st.subheader("📊 整體分數比對")

    show_overall_score_proof_in_streamlit(
        proc,
        [
            (
                "Current Analysis",
                df_std_action,
                df_usr_action
            )
        ]
    )


    st.divider()


    if st.button(
        "關閉",
        use_container_width=True,
        key="close_comparison_dialog"
    ):
        st.rerun()


# =========================================================
# 8. Sidebar
# =========================================================

with st.sidebar:

    # -----------------------------------------------------
    # 回首頁
    # -----------------------------------------------------

    st.markdown(
        '<div class="home-btn">',
        unsafe_allow_html=True
    )

    if st.button(
        "🏠",
        key="home"
    ):
        st.switch_page("app.py")

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    st.write("")


    # -----------------------------------------------------
    # 教學示範
    # -----------------------------------------------------

    left, center, right = st.columns([1, 3, 1])

    with center:

        st.markdown(
            '<div class="teach-btn">',
            unsafe_allow_html=True
        )

        if st.button(
            "教學示範",
            key="help",
            use_container_width=True
        ):
            show_help_dialog()

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    st.divider()


    # -----------------------------------------------------
    # 上傳影片
    # -----------------------------------------------------

    uploaded_file = st.file_uploader(
        "上傳使用者影片",
        type=[
            "mp4",
            "mov",
            "avi"
        ]
    )


    st.info(
        "⚠️ 拍攝影片時請開啟慢動作240fps，"
        "影片拍法比對教學示範影片!!"
    )


# =========================================================
# 9. 取得影片 Hash
# =========================================================

def get_uploaded_file_hash(uploaded_file):

    if uploaded_file is None:
        return None

    file_bytes = uploaded_file.getvalue()

    return hashlib.md5(
        file_bytes
    ).hexdigest()


# =========================================================
# 10. 建立分析結果
# =========================================================

def run_analysis(uploaded_file):

    # =====================================================
    # 存檔
    # =====================================================

    tfile = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    tfile.write(
        uploaded_file.getvalue()
    )

    video_path = tfile.name

    tfile.close()


    try:

        # =================================================
        # MediaPipe
        # =================================================

        status_box = st.empty()

        status_box.info(
            "🔄 MediaPipe 分析中..."
        )

        prog = st.progress(0)

        records = []


        BaseOptions = mp.tasks.BaseOptions

        PoseLandmarker = (
            mp.tasks.vision.PoseLandmarker
        )

        PoseLandmarkerOptions = (
            mp.tasks.vision.PoseLandmarkerOptions
        )

        RunningMode = (
            mp.tasks.vision.RunningMode
        )


        # -------------------------------------------------
        # CPU 模式
        # -------------------------------------------------

        model_path = os.path.join(
            project_root,
            "pose_landmarker.task"
        )

        if not os.path.exists(model_path):

            st.error(
                f"找不到 MediaPipe 模型：{model_path}"
            )

            return None


        options = PoseLandmarkerOptions(

            base_options=BaseOptions(

                model_asset_path=model_path,

                delegate=BaseOptions.Delegate.CPU

            ),

            running_mode=RunningMode.VIDEO
        )


        # =================================================
        # MediaPipe PoseLandmarker
        # =================================================

        with PoseLandmarker.create_from_options(
            options
        ) as landmarker:

            cap = cv2.VideoCapture(
                video_path
            )


            fps = (
                cap.get(cv2.CAP_PROP_FPS)
                or 30
            )


            total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )


            i = 0


            while cap.isOpened():

                ret, frame = cap.read()


                if not ret:
                    break


                # -----------------------------------------
                # BGR → RGB
                # -----------------------------------------

                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )


                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb
                )


                timestamp_ms = int(
                    i * (1000 / fps)
                )


                # -----------------------------------------
                # Pose Detection
                # -----------------------------------------

                result = (
                    landmarker.detect_for_video(
                        mp_image,
                        timestamp_ms
                    )
                )


                data = {}


                if (
                    result
                    and result.pose_landmarks
                    and len(result.pose_landmarks) > 0
                ):

                    lms = (
                        result.pose_landmarks[0]
                    )


                    try:

                        coord = {

                            k: (
                                lms[k].x,
                                lms[k].y
                            )

                            for k in range(33)

                        }


                        data = (
                            get_full_body_angles(
                                coord
                            )
                        )


                        if not isinstance(
                            data,
                            dict
                        ):

                            data = {}


                    except Exception:

                        data = {}


                    # -------------------------------------
                    # 關鍵點座標
                    # -------------------------------------

                    for j in range(11, 33):

                        data[
                            f"{j}_x"
                        ] = float(
                            lms[j].x
                        )

                        data[
                            f"{j}_y"
                        ] = float(
                            lms[j].y
                        )


                else:

                    # -------------------------------------
                    # 沒偵測到人
                    # -------------------------------------

                    for j in range(11, 33):

                        data[
                            f"{j}_x"
                        ] = 0.5

                        data[
                            f"{j}_y"
                        ] = 0.5


                records.append(data)

                i += 1


                if (
                    i % 10 == 0
                    and total_frames > 0
                ):

                    prog.progress(
                        min(
                            i / total_frames,
                            1.0
                        )
                    )


            cap.release()


        status_box.empty()
        prog.empty()


        # =================================================
        # 檢查資料
        # =================================================

        if len(records) <= 10:

            st.error(
                "影片太短，無法進行分析。"
            )

            return None


        # =================================================
        # PoseProcessor / AI Coach
        # =================================================

        proc = PoseProcessor()

        coach_ai = AICoach()


        # =================================================
        # 使用者完整資料
        # =================================================

        df_usr_full = (

            pd.DataFrame(records)

            .ffill()

            .bfill()

            .fillna(0)

        )


        # =================================================
        # 自動偵測動作區間
        # =================================================

        start_auto, peak, end_auto = (
            proc.detect_action_range(
                df_usr_full
            )
        )


        start = max(
            0,
            int(start_auto)
        )


        end = min(
            len(df_usr_full) - 1,
            int(end_auto)
        )


        if end <= start:

            end = min(
                len(df_usr_full) - 1,
                start + 30
            )


        df_usr_action = (

            df_usr_full

            .iloc[start:end + 1]

            .reset_index(drop=True)

        )


        # =================================================
        # 標準動作（反拍高遠球）
        # =================================================

        csv_path = os.path.join(
            current_dir,
            "standard_backhand_clear.csv"
        )


        if not os.path.exists(csv_path):

            st.error(
                "找不到 standard_backhand_clear.csv"
            )

            return None


        df_std_action = pd.read_csv(
            csv_path
        )


        # =================================================
        # 相似度
        # =================================================

        score, stats = (
            proc.calculate_auto_similarity(
                df_std_action,
                df_usr_action
            )
        )


        # =================================================
        # Similarity Curve
        # =================================================

        curve = (
            proc.compute_similarity_curve(
                df_std_action,
                df_usr_action
            )
        )


        # =================================================
        # Features
        # =================================================

        feat_std = (
            proc.extract_features(
                df_std_action
            )
        )


        feat_usr = (
            proc.extract_features(
                df_usr_action
            )
        )


        # =================================================
        # FastDTW
        # =================================================

        distance, path = fastdtw(
            feat_std,
            feat_usr
        )


        # =================================================
        # AI 教練
        # =================================================

        if path is None or len(path) == 0:

            feedback_list = []

            overall = ""

            penalty = 0

        else:

            (
                feedback_list,
                overall,
                penalty
            ) = coach_ai.generate_feedback(
                feat_std,
                feat_usr,
                path,
                curve,
                score
            )


        # =================================================
        # 產生 Overlay 影片
        # =================================================

        output_path = os.path.join(
            tempfile.gettempdir(),
            f"overlay_{os.getpid()}_{id(uploaded_file)}.mp4"
        )


        with st.spinner(
            "生成影片中..."
        ):

            proc.generate_auto_overlay(
                video_path,
                df_std_action,
                df_usr_action,
                start,
                output_path
            )


        # =================================================
        # 將所有結果存進 Session State
        # =================================================

        result = {

            "proc": proc,

            "df_std_action": df_std_action,

            "df_usr_action": df_usr_action,

            "score": score,

            "stats": stats,

            "curve": curve,

            "feat_std": feat_std,

            "feat_usr": feat_usr,

            "distance": distance,

            "path": path,

            "feedback_list": feedback_list,

            "overall": overall,

            "penalty": penalty,

            "start": start,

            "end": end,

            "video_path": video_path,

            "overlay_output_path": output_path

        }


        return result


    except Exception as e:

        st.error(
            f"分析過程發生錯誤：{e}"
        )

        return None


# =========================================================
# 11. 主流程
# =========================================================

if uploaded_file is not None:

    # =====================================================
    # 判斷是不是新影片
    # =====================================================

    current_file_hash = (
        get_uploaded_file_hash(
            uploaded_file
        )
    )


    # =====================================================
    # 如果換影片
    # =====================================================

    if (
        st.session_state.uploaded_file_hash
        != current_file_hash
    ):

        # -----------------------------------------------
        # 清除舊資料
        # -----------------------------------------------

        old_video_path = (
            st.session_state.video_path
        )

        old_overlay_path = (
            st.session_state.overlay_output_path
        )


        if (
            old_video_path
            and os.path.exists(old_video_path)
        ):

            try:
                os.remove(old_video_path)
            except Exception:
                pass


        if (
            old_overlay_path
            and os.path.exists(old_overlay_path)
        ):

            try:
                os.remove(old_overlay_path)
            except Exception:
                pass


        # -----------------------------------------------
        # 清除 Session State
        # -----------------------------------------------

        st.session_state.analysis_result = None

        st.session_state.video_path = None

        st.session_state.overlay_output_path = None

        st.session_state.uploaded_file_hash = (
            current_file_hash
        )


    # =====================================================
    # 如果還沒有分析結果
    # =====================================================

    if st.session_state.analysis_result is None:

        result = run_analysis(
            uploaded_file
        )


        if result is not None:

            st.session_state.analysis_result = (
                result
            )

            st.session_state.video_path = (
                result["video_path"]
            )

            st.session_state.overlay_output_path = (
                result["overlay_output_path"]
            )

            # -------------------------------------------------
            # 這裡不要 st.rerun()
            #
            # 直接繼續往下顯示結果
            # -------------------------------------------------


    # =====================================================
    # 取得已經分析好的結果
    # =====================================================

    result = (
        st.session_state.analysis_result
    )


    if result is not None:

        proc = result["proc"]

        df_std_action = (
            result["df_std_action"]
        )

        df_usr_action = (
            result["df_usr_action"]
        )

        score = result["score"]

        stats = result["stats"]

        curve = result["curve"]

        feat_std = result["feat_std"]

        feat_usr = result["feat_usr"]

        path = result["path"]

        feedback_list = (
            result["feedback_list"]
        )

        overall = result["overall"]

        penalty = result["penalty"]

        output_path = (
            result["overlay_output_path"]
        )


        # =================================================
        # 分析完成提示
        # =================================================

        st.markdown(
            """
            <div class="analysis-done">
                ✅ 分析完成！以下結果已暫存在本次 Session。
            </div>
            """,
            unsafe_allow_html=True
        )


        # =================================================
        # UI
        # =================================================

        col1, col2 = st.columns(
            [2, 1]
        )


        # =================================================
        # 左邊：影片
        # =================================================

        with col1:

            st.subheader(
                "🎥 動作影片"
            )


            if (
                output_path
                and os.path.exists(output_path)
            ):

                st.video(
                    output_path
                )

            else:

                st.warning(
                    "找不到分析後影片。"
                )


        # =================================================
        # 右邊：分析
        # =================================================

        with col2:

            st.subheader(
                "📊 分析"
            )


            st.metric(
                "分數",
                f"{score:.1f}"
            )


            # =================================================
            # 分數組成（反拍公式：0.6 / 0.2 / 0.1 / 0.15、*1.1）
            # =================================================

            with st.expander(
                "📋 查看分數組成"
            ):

                mean = stats[
                    "mean_path_score"
                ]

                p50 = stats[
                    "p50"
                ]

                p25 = stats[
                    "p25"
                ]

                worst = stats[
                    "min"
                ]

                std = stats[
                    "std"
                ]

                penalty_stats = stats[
                    "penalty"
                ]


                a = mean * 0.6

                b = p50 * 0.2

                c = p25 * 0.1

                d = worst * 0.15


                base = (
                    a + b + c + d
                ) * 1.1


                st.markdown(
                    f"""
                    | 項目 | 數值 | 註記 |
                    |------|------|------|
                    | 相似度總計 | `{base:.1f}` | 動作相似程度經專業加權後計算之基礎分 |
                    | − AI教練懲罰 | `−{penalty:.1f}` | AI 教練動作誤差扣分 |
                    | **最終分數** | **`{score:.1f}`** | 最終成績 |
                    """
                )


            # =================================================
            # AI 教練
            # =================================================

            st.subheader(
                "🧠 教練回饋"
            )


            if (
                path is None
                or len(path) == 0
            ):

                st.warning(
                    "無法對齊動作，請確認影片品質"
                )


            else:

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
                            f
                            if isinstance(f, list)
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


                    # =================================================
                    # 詳細比對結果
                    #
                    # ⚠️ 這裡按下去雖然 Streamlit 仍然會 rerun，
                    # 但因為 analysis_result 已經存在，
                    # 所以不會重新 MediaPipe / DTW / AI / Overlay。
                    # =================================================

                    st.markdown(
                        '<div class="detail-result-btn">',
                        unsafe_allow_html=True
                    )


                    if st.button(
                        "🔍 查看詳細比對結果",
                        use_container_width=True,
                        key="show_comparison"
                    ):

                        show_comparison_dialog()


                    st.markdown(
                        '</div>',
                        unsafe_allow_html=True
                    )


                else:

                    st.info(
                        "未產生回饋"
                    )


# =========================================================
# 12. 尚未上傳影片
# =========================================================

else:

    # -----------------------------------------------------
    # 如果使用者把影片移除
    # -----------------------------------------------------

    if (
        st.session_state.analysis_result
        is not None
    ):

        old_video_path = (
            st.session_state.video_path
        )

        old_overlay_path = (
            st.session_state.overlay_output_path
        )


        if (
            old_video_path
            and os.path.exists(old_video_path)
        ):

            try:
                os.remove(old_video_path)
            except Exception:
                pass


        if (
            old_overlay_path
            and os.path.exists(old_overlay_path)
        ):

            try:
                os.remove(old_overlay_path)
            except Exception:
                pass


        st.session_state.analysis_result = None

        st.session_state.uploaded_file_hash = None

        st.session_state.video_path = None

        st.session_state.overlay_output_path = None


    st.info(
        "請上傳影片"
    )