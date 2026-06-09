import os
import streamlit as st

st.write("cwd =", os.getcwd())

if os.path.exists("pages"):
    st.write("pages內容：", os.listdir("pages"))
else:
    st.write("沒有 pages 資料夾")

import streamlit as st

st.set_page_config(
    page_title="🏸 AI 羽球教練",
    layout="wide"
)

# =========================
# CSS
# =========================
st.markdown("""
<style>
/* 卡片按鈕 */
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

# =========================
# 主頁
# =========================
st.title("🏸 AI 羽球教練")
st.markdown("#### 請選擇要練習的揮拍項目")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🏹\n高遠球\n正拍"):
        st.switch_page("pages/analyzer")

with col2:
    if st.button("🔄\n高遠球\n反拍"):
        st.switch_page("pages/2_overhead_clear_bh.py")

with col3:
    if st.button("⬆️\n挑球\n正拍"):
        st.switch_page("pages/3_lift_fh.py")

with col4:
    if st.button("↩️\n挑球\n反拍"):
        st.switch_page("pages/4_lift_bh.py")

st.markdown("---")
st.caption("上傳影片後，AI 將自動分析揮拍動作並給予教練回饋")