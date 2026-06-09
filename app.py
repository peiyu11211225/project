import os
import streamlit as st
    
st.set_page_config(
    page_title="🏸 AI 羽球教練",
    layout="wide",
    initial_sidebar_state="collapsed"  # 預設讓側邊欄保持收合狀態
)

# =========================
# CSS 美化與側邊欄拔除
# =========================
st.markdown("""
<style>     
/* ─── 🛑 徹底移除首頁的側邊欄與相關按鈕 ─── */
[data-testid="stSidebar"], section[data-testid="stSidebar"] {
    display: none !important;
    width: 0px !important;
}
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}
.stMainBlockContainer {
    max-width: 1200px !important; /* 讓內容在正中間聚集，視覺更舒適 */
    margin: 0 auto;
    padding-top: 4rem !important;
}

/* ─── ✨ 高級感卡片按鈕美化 ─── */
div[data-testid="column"] .stButton > button {
    width: 100%;
    height: 160px; /* 稍微加高，更有分量感 */
    border-radius: 20px; /* 更圓潤的現代幾何感 */
    border: 1px solid #d0e1fd;
    background: linear-gradient(145deg, #f3f8fc, #e6f1f9); /* 微漸層質感 */
    color: #1e3d59;
    font-size: 1.2rem;
    font-weight: 700;
    line-height: 1.7;
    cursor: pointer;
    box-shadow: 0 6px 15px rgba(80, 130, 170, 0.08);
    white-space: pre-wrap;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

/* 懸停發光與上浮效果 */
div[data-testid="column"] .stButton > button:hover {
    background: linear-gradient(145deg, #e6f2fc, #d4e8f7);
    border-color: #a3cbf0;
    color: #17b978; /* 懸停時文字變為羽球動感綠 */
    box-shadow: 0 12px 24px rgba(23, 185, 120, 0.15); /* 帶有綠色微光的科技感陰影 */
    transform: translateY(-5px); /* 向上漂浮感 */
}

/* 按下時的反饋 */
div[data-testid="column"] .stButton > button:active {
    transform: translateY(1px);
    box-shadow: 0 3px 8px rgba(80, 130, 170, 0.1);
}

/* 調整大標題樣式 */
.main-title {
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    color: #1e3d59;
    margin-bottom: 5px;
}
.sub-title {
    color: #17b978 !important; /* 精緻的羽球綠主題色 */
    font-weight: 600 !important;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 主頁 UI 渲染
# =========================
# 使用 HTML 自訂樣式達到更精美的前端效果
st.markdown('<h1 class="main-title">🏸 AI 羽球教練</h1>', unsafe_allow_html=True)
st.markdown('<h4 class="sub-title">🤖 智慧揮拍動作診斷與技術優化平台</h4>', unsafe_allow_html=True)
st.markdown("##### 💡 請選擇今日要練習的揮拍項目：")

# 加上稍微寬鬆的間距
st.write("")

col1, col2, col3, col4 = st.columns(4, gap="large")

with col1:
    if st.button("🏹\n\n高遠球 · 正拍"):
        st.switch_page("pages/1_analyzer.py")

with col2:
    if st.button("🔄\n\n高遠球 · 反拍"):
        st.switch_page("pages/2_overhead_clear_bh.py")

with col3:
    if st.button("⬆️\n\n挑球 · 正拍"):
        st.switch_page("pages/3_lift_fh.py")

with col4:
    if st.button("↩️\n\n挑球 · 反拍"):
        st.switch_page("pages/4_lift_bh.py")

st.write("")
st.write("")
st.markdown("---")
st.caption("✨ 點選項目並上傳你的揮拍影片，AI 將自動與標準動作進行跨時序對齊（DTW），並精準計算關節角度給予專業回饋。")