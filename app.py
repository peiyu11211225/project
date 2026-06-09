import os
import streamlit as st
    
st.set_page_config(
    page_title="🏸 AI 羽球教練",
    layout="wide"
)

# =========================
# CSS
# =========================

# =========================
# 主頁
# =========================
st.title("🏸 AI 羽球教練")
st.markdown("#### 請選擇要練習的揮拍項目")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🏹\n高遠球\n正拍"):
        st.switch_page("pages/1_analyzer.py")

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