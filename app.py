import os
import streamlit as st

st.set_page_config(
    page_title="🏸 AI 羽球教練 : 揮拍動作診斷平台",
    page_icon="🏸",
    layout="wide"
)

st.markdown("""
<style>
/* ===== 隱藏側邊欄 ===== */
[data-testid="stSidebarNav"],
[data-testid="stSidebar"],
[data-testid="collapsedControl"] {
    display: none !important;
}

/* ===== 整體白色背景、版面拉寬 ===== */
.stApp {
    background: #ffffff;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    padding-left: 3rem;
    padding-right: 3rem;
    max-width: 1600px;
}

/* ===== 頂部橫幅 ===== */
.hero {
    text-align: center;
    padding: 3.2rem 2rem 2.6rem 2rem;
    border-radius: 32px;
    background: linear-gradient(135deg, #f4faff 0%, #dcedfa 100%);
    border: 2px solid #d3e7f5;
    box-shadow: 0 10px 30px rgba(80, 130, 170, 0.12);
    margin-bottom: 2.5rem;
}
.hero-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    background: #ffffff;
    border: 1.5px solid #b8d8ee;
    color: #3f7ea6;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 3.4rem;
    font-weight: 800;
    color: #1a3a4f;
    letter-spacing: 3px;
    margin: 0.2rem 0 0.6rem 0;
}
.hero-sub {
    font-size: 1.4rem;
    color: #5a7d94;
    font-weight: 500;
    margin: 0;
}

/* ===== 區塊標題 ===== */
.section-title {
    text-align: center;
    font-size: 2rem;
    font-weight: 800;
    color: #1a3a4f;
    margin: 2.5rem 0 0.4rem 0;
}
.section-sub {
    text-align: center;
    font-size: 1.1rem;
    color: #7a97aa;
    margin-bottom: 1.8rem;
}

/* ===== 三步驟流程卡片 ===== */
.step-card {
    text-align: center;
    padding: 1.8rem 1.2rem;
    border-radius: 24px;
    background: #ffffff;
    border: 2px solid #e1eef7;
    box-shadow: 0 6px 18px rgba(80, 130, 170, 0.08);
    height: 100%;
}
.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 46px;
    height: 46px;
    border-radius: 50%;
    background: #d3e9f8;
    color: #1a3a4f;
    font-size: 1.3rem;
    font-weight: 800;
    margin-bottom: 0.6rem;
}
.step-icon {
    font-size: 2.6rem;
    margin: 0.2rem 0;
}
.step-title {
    font-size: 1.4rem;
    font-weight: 800;
    color: #1a3a4f;
    margin: 0.3rem 0;
}
.step-desc {
    font-size: 1.05rem;
    color: #6f8da1;
    line-height: 1.6;
}

/* ===== 項目卡上方標籤 ===== */
.tag-row {
    text-align: center;
    margin-bottom: 0.7rem;
}
.tag {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    background: #eaf5fd;
    border: 1.5px solid #c4dff1;
    color: #3f7ea6;
    font-size: 0.95rem;
    font-weight: 700;
}

/* ===== 項目按鈕（相容新舊版 Streamlit） ===== */
div[data-testid="column"] .stButton > button,
div[data-testid="stColumn"] .stButton > button {
    width: 100%;
    height: 300px;
    border-radius: 28px;
    border: 2px solid #b8d8ee;
    background: linear-gradient(180deg, #eaf5fd 0%, #d3e9f8 100%);
    color: #1a3a4f;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow:
        0 10px 0 #a9cde6,
        0 18px 30px rgba(80, 130, 170, 0.25);
    margin-bottom: 14px;
}
div[data-testid="column"] .stButton > button p,
div[data-testid="stColumn"] .stButton > button p {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.9;
    white-space: pre-wrap;
    text-align: center;
}
div[data-testid="column"] .stButton > button:hover,
div[data-testid="stColumn"] .stButton > button:hover {
    background: linear-gradient(180deg, #d9edfb 0%, #bfdff5 100%);
    border-color: #8fc2e3;
    color: #0f2a3b;
    transform: translateY(-6px);
    box-shadow:
        0 16px 0 #98c3de,
        0 26px 38px rgba(80, 130, 170, 0.32);
}
div[data-testid="column"] .stButton > button:active,
div[data-testid="stColumn"] .stButton > button:active {
    transform: translateY(8px);
    box-shadow:
        0 2px 0 #a9cde6,
        0 4px 10px rgba(80, 130, 170, 0.2);
}
div[data-testid="column"] .stButton > button:focus:not(:active),
div[data-testid="stColumn"] .stButton > button:focus:not(:active) {
    border-color: #8fc2e3;
    color: #1a3a4f;
}

/* ===== 按鈕下方說明 ===== */
.card-desc {
    text-align: center;
    font-size: 1.05rem;
    color: #6f8da1;
    line-height: 1.7;
    padding: 0 0.5rem;
}

/* ===== 小提醒卡片 ===== */
.tip-card {
    padding: 1.4rem 1.6rem;
    border-radius: 20px;
    background: #f7fbfe;
    border-left: 6px solid #8fc2e3;
    box-shadow: 0 4px 14px rgba(80, 130, 170, 0.07);
    height: 100%;
}
.tip-title {
    font-size: 1.2rem;
    font-weight: 800;
    color: #1a3a4f;
    margin-bottom: 0.4rem;
}
.tip-desc {
    font-size: 1.02rem;
    color: #6f8da1;
    line-height: 1.7;
}

/* ===== 分隔線與頁尾 ===== */
hr {
    border-color: #e3eef6 !important;
    margin: 2.5rem 0 !important;
}
.footer-note {
    text-align: center;
    font-size: 1.1rem;
    color: #6f8da1;
    margin-top: 0.5rem;
}
.footer-small {
    text-align: center;
    font-size: 0.95rem;
    color: #a3b8c6;
    margin-top: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

# ===== 頂部橫幅 =====
st.markdown("""
<div class='hero'>
    <div class='hero-badge'>🤖 AI 動作分析 · 即時教練回饋</div>
    <div class='hero-title'>🏸 AI 羽球教練</div>
    <p class='hero-sub'>揮拍動作診斷平台｜上傳影片，讓 AI 幫你找出動作問題</p>
</div>
""", unsafe_allow_html=True)

# ===== 使用流程 =====
st.markdown("<div class='section-title'>📋 使用流程</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>只要三個步驟，就能獲得專屬的揮拍建議</div>", unsafe_allow_html=True)

s1, s2, s3 = st.columns(3, gap="large")
with s1:
    st.markdown("""
    <div class='step-card'>
        <div class='step-num'>1</div>
        <div class='step-icon'>🎯</div>
        <div class='step-title'>選擇項目</div>
        <div class='step-desc'>從下方挑選想練習的球種與拍面</div>
    </div>
    """, unsafe_allow_html=True)
with s2:
    st.markdown("""
    <div class='step-card'>
        <div class='step-num'>2</div>
        <div class='step-icon'>🎥</div>
        <div class='step-title'>上傳影片</div>
        <div class='step-desc'>上傳你的揮拍影片，AI 自動辨識骨架動作</div>
    </div>
    """, unsafe_allow_html=True)
with s3:
    st.markdown("""
    <div class='step-card'>
        <div class='step-num'>3</div>
        <div class='step-icon'>📈</div>
        <div class='step-title'>取得回饋</div>
        <div class='step-desc'>查看評分與教練建議，針對弱點加強練習</div>
    </div>
    """, unsafe_allow_html=True)

# ===== 揮拍項目選擇 =====
st.markdown("<div class='section-title'>🏸 請選擇要練習的揮拍項目</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>點擊下方按鈕，進入對應的動作分析頁面</div>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4, gap="large")

with col1:
    st.markdown("<div class='tag-row'><span class='tag'>後場 · 進攻</span></div>", unsafe_allow_html=True)
    if st.button("🏸\n高遠球\n正拍", use_container_width=True, key="btn1"):
        st.switch_page("pages/1_analyzer.py")
    st.markdown("<div class='card-desc'>後場基本功，分析引拍、擊球點與收拍</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='tag-row'><span class='tag'>後場 · 防守</span></div>", unsafe_allow_html=True)
    if st.button("🏸\n高遠球\n反拍", use_container_width=True, key="btn2"):
        st.switch_page("pages/2_bnalyzer.py")
    st.markdown("<div class='card-desc'>反手後場回擊，檢查轉身與發力順序</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='tag-row'><span class='tag'>前場 · 防守</span></div>", unsafe_allow_html=True)
    if st.button("🏸\n挑球\n正拍", use_container_width=True, key="btn3"):
        st.switch_page("pages/3_cnalyzer.py")
    st.markdown("<div class='card-desc'>網前挑高球，分析步伐與手腕控制</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='tag-row'><span class='tag'>前場 · 防守</span></div>", unsafe_allow_html=True)
    if st.button("🏸\n挑球\n反拍", use_container_width=True, key="btn4"):
        st.switch_page("pages/4_dnalyzer.py")
    st.markdown("<div class='card-desc'>反手網前挑球，檢查握拍與擊球角度</div>", unsafe_allow_html=True)

# ===== 拍攝小提醒 =====
st.markdown("<div class='section-title'>💡 拍攝小提醒</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>影片拍得好，分析結果更準確</div>", unsafe_allow_html=True)

t1, t2, t3 = st.columns(3, gap="large")
with t1:
    st.markdown("""
    <div class='tip-card'>
        <div class='tip-title'>📐 拍攝角度</div>
        <div class='tip-desc'>建議從側面或斜後方拍攝，讓全身與球拍都入鏡。</div>
    </div>
    """, unsafe_allow_html=True)
with t2:
    st.markdown("""
    <div class='tip-card'>
        <div class='tip-title'>💡 光線與背景</div>
        <div class='tip-desc'>光線充足、背景單純，能讓骨架偵測更穩定。</div>
    </div>
    """, unsafe_allow_html=True)
with t3:
    st.markdown("""
    <div class='tip-card'>
        <div class='tip-title'>🎬 影片內容</div>
        <div class='tip-desc'>一支影片錄一次完整揮拍，並固定手機位置不要移動。</div>
    </div>
    """, unsafe_allow_html=True)

# ===== 頁尾 =====
st.markdown("---")
st.markdown("<div class='footer-note'>上傳影片後，AI 將自動分析揮拍動作並給予教練回饋</div>", unsafe_allow_html=True)
st.markdown("<div class='footer-small'>🏸 AI 羽球教練 · 揮拍動作診斷平台</div>", unsafe_allow_html=True)