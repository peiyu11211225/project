import os
import streamlit as st

st.set_page_config(
    page_title="🏸 AI 羽球教練 : 揮拍動作診斷平台",
    layout="wide"
)

st.markdown("""
<style>     
[data-testid="stSidebarNav"] {
    display: none !important;
}

[data-testid="stSidebar"] {
    display: none !important;
}

div[data-testid="column"] .stButton > button {
    width: 100%;
    height: 140px;
    border-radius: 16px;
    border: 2px solid #c9dcea;
    background: #f0f7fc;
    color: #1a3a4f;
    font-size: 1.1rem;
    font-weight: 700;
    line-height: 1.6;
    cursor: pointer;
    transition: background 0.18s, transform 0.08s, box-shadow 0.18s;
    box-shadow: 0 3px 10px rgba(80,130,170,0.12);
    white-space: pre-wrap;
}
div[data-testid="column"] .stButton > button:hover {
    background: #d6eaf7;
    box-shadow: 0 6px 18px rgba(80,130,170,0.22);
    transform: translateY(-2px);
}
div[data-testid="column"] .stButton > button:active {
    transform: translateY(1px);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🏸 AI 羽球教練 : 揮拍動作診斷平台</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>請選擇要練習的揮拍項目</h4>", unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("\n高遠球\n正拍"):
        st.switch_page("pages/1_analyzer.py")

with col2:
    if st.button("\n高遠球\n反拍"):
        st.switch_page("pages/2_bnalyzer.py")

with col3:
    if st.button("\n挑球\n正拍"):
        st.switch_page("pages/3_cnalyzer.py")

with col4:
    if st.button("\n挑球\n反拍"):
        st.switch_page("pages/4_dnalyzer.py")

st.markdown("---")
st.caption("上傳影片後，AI 將自動分析揮拍動作並給予教練回饋")