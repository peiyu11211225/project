import os
import streamlit as st

st.set_page_config(
    page_title="🏸 AI 羽球教練 : 揮拍動作診斷平台",
    layout="wide"
)

st.markdown("""
<style>
/* ===== 隱藏側邊欄 ===== */
[data-testid="stSidebarNav"],
[data-testid="stSidebar"] {
    display: none !important;
}

/* ===== 整體白色背景 ===== */
.stApp {
    background: #ffffff;
}
.block-container {
    padding-top: 3rem;
    max-width: 1200px;
}

/* ===== 標題 ===== */
.main-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 800;
    color: #1a3a4f;
    margin-bottom: 0.3rem;
    letter-spacing: 2px;
}
.sub-title {
    text-align: center;
    font-size: 1.3rem;
    font-weight: 500;
    color: #5a7d94;
    margin-bottom: 1rem;
}

/* ===== 按鈕（同時相容新舊版 Streamlit 的欄位名稱） ===== */
div[data-testid="column"] .stButton > button,
div[data-testid="stColumn"] .stButton > button {
    width: 100%;
    height: 220px;
    border-radius: 24px;
    border: 2px solid #b8d8ee;
    background: linear-gradient(180deg, #eaf5fd 0%, #d3e9f8 100%);
    color: #1a3a4f;
    cursor: pointer;
    transition: all 0.2s ease;
    /* 立體按鈕感：外陰影 + 底部厚度 */
    box-shadow:
        0 8px 0 #a9cde6,
        0 14px 24px rgba(80, 130, 170, 0.25);
    margin-bottom: 10px;
}

/* 按鈕內的文字（Streamlit 文字在 p 裡面，要另外指定） */
div[data-testid="column"] .stButton > button p,
div[data-testid="stColumn"] .stButton > button p {
    font-size: 1.9rem;
    font-weight: 800;
    line-height: 1.8;
    white-space: pre-wrap;
    text-align: center;
}

/* 滑過 */
div[data-testid="column"] .stButton > button:hover,
div[data-testid="stColumn"] .stButton > button:hover {
    background: linear-gradient(180deg, #d9edfb 0%, #bfdff5 100%);
    border-color: #8fc2e3;
    color: #0f2a3b;
    transform: translateY(-4px);
    box-shadow:
        0 12px 0 #98c3de,
        0 20px 30px rgba(80, 130, 170, 0.32);
}

/* 按下去 */
div[data-testid="column"] .stButton > button:active,
div[data-testid="stColumn"] .stButton > button:active {
    transform: translateY(6px);
    box-shadow:
        0 2px 0 #a9cde6,
        0 4px 10px rgba(80, 130, 170, 0.2);
}

/* 點擊後的 focus 外框 */
div[data-testid="column"] .stButton > button:focus:not(:active),
div[data-testid="stColumn"] .stButton > button:focus:not(:active) {
    border-color: #8fc2e3;
    color: #1a3a4f;
}

/* ===== 分隔線與說明文字 ===== */
hr {
    border-color: #e3eef6 !important;
    margin: 1.8rem 0 !important;
}
.footer-note {
    text-align: center;
    font-size: 1.05rem;
    color: #6f8da1;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🏸 AI 羽球教練 : 揮拍動作診斷平台</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>請選擇要練習的揮拍項目</div>", unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3, col4 = st.columns(4, gap="large")

with col1:
    if st.button("🏸\n高遠球\n正拍", use_container_width=True):
        st.switch_page("pages/1_analyzer.py")

with col2:
    if st.button("🏸\n高遠球\n反拍", use_container_width=True):
        st.switch_page("pages/2_bnalyzer.py")

with col3:
    if st.button("🏸\n挑球\n正拍", use_container_width=True):
        st.switch_page("pages/3_cnalyzer.py")

with col4:
    if st.button("🏸\n挑球\n反拍", use_container_width=True):
        st.switch_page("pages/4_dnalyzer.py")

st.markdown("---")
st.markdown("<div class='footer-note'>上傳影片後，AI 將自動分析揮拍動作並給予教練回饋</div>", unsafe_allow_html=True)