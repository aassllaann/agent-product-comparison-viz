import streamlit as st
from multi_agent import MultiCategoryAgent
from category_detector import CategoryDetector
import os

# 1. Page Configuration
st.set_page_config(
      page_title="AI 智能选购助手", 
      layout="wide", 
      page_icon="◉"
)

# 2. Premium Design System CSS
if os.path.exists("style.css"):
    with open("style.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("style.css not found. Please ensure it exists in the same directory.")

# 3. 初始化状态
if 'agent' not in st.session_state:
    st.session_state.agent = MultiCategoryAgent()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'last_results' not in st.session_state:
    st.session_state.last_results = None
if 'last_charts' not in st.session_state:
    st.session_state.last_charts = None
if 'last_analyses' not in st.session_state:
    st.session_state.last_analyses = None
if 'last_reasons' not in st.session_state:
    st.session_state.last_reasons = None
if 'current_category' not in st.session_state:
    st.session_state.current_category = None
if 'last_eco_suggestion' not in st.session_state:
    st.session_state.last_eco_suggestion = None

# 初始化检测器
detector = CategoryDetector()

# --- PROCESSING LOGIC (MOVED TO TOP) ---
if 'current_prompt' in st.session_state and st.session_state.current_prompt:
    prompt = st.session_state.current_prompt
    del st.session_state.current_prompt
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    cat_key, cat_name = detector.detect_category(prompt)
    st.session_state.current_category = cat_name
    
    with st.spinner("正在分析您的需求并检索数据..."):
        try:
            history = [{"role": m["role"], "content": str(m["content"])} for m in st.session_state.messages[-6:]]
            reply, charts, results, analyses, eco_suggestion = st.session_state.agent.handle_chat(prompt, history=history)
            
            st.session_state.last_results = results
            st.session_state.last_charts = charts
            st.session_state.last_analyses = analyses
            st.session_state.last_reasons = reply if isinstance(reply, list) else None
            st.session_state.last_eco_suggestion = eco_suggestion
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(f"Error executing agent: {e}")

# 品类标签
CATEGORY_LABELS = {
    "camera": "相机", "phone": "手机", "headphone": "耳机", 
    "laptop": "笔记本", "tablet": "平板", "smartwatch": "智能手表", 
    "bluetooth_speaker": "音箱", "monitor": "显示器", 
    "gaming_console": "主机", "gpu": "显卡"
}

def render_product_card(product, reason=None):
    """渲染商品卡片"""
    is_dict = isinstance(product, dict)
    if is_dict:
        brand = product.get('brand', '')
        model = product.get('model', '')
        price = product.get('price', 0)
        scores = product.get('scores', {})
        specs = product.get('specs', {})
    else:
        brand = getattr(product, 'Brand', '')
        model = getattr(product, 'Model', '')
        price = getattr(product, 'Price', 0)
        class_name = type(product).__name__
        specs = {}
        scores = {}
        for k in dir(product):
            if k.endswith('_Score') and getattr(product, k):
                scores[k.replace('_Score', '')] = getattr(product, k)

        if class_name == 'Phone':
            specs = {'处理器': getattr(product, 'Processor', '-'), '内存': f"{getattr(product, 'RAM_GB', 0)}G+{getattr(product, 'Storage_GB', 0)}G", '电池': f"{getattr(product, 'Battery_mAh', 0)}mAh", '主摄': f"{getattr(product, 'Camera_MP', 0)}MP"}
        elif class_name == 'Laptop':
            specs = {'CPU': getattr(product, 'CPU', '-'), '显卡': getattr(product, 'GPU', '-'), '内存': f"{getattr(product, 'RAM_GB', 0)}G", '屏幕': f"{getattr(product, 'Screen_Size_in', 0)}英寸"}
        elif class_name == 'Headphone':
            specs = {'类型': getattr(product, 'Type', '-'), '降噪': '是' if getattr(product, 'ANC', False) else '否', '续航': f"{getattr(product, 'Battery_Hours', 0)}h"}
        elif class_name == 'Camera':
            specs = {'像素': f"{getattr(product, 'Total_megapixels', 0)}MP", 'ISO': getattr(product, 'Max_ISO', 0), '重量': f"{getattr(product, 'Weight_g', 0)}g", '4K': '是' if getattr(product, 'Supports_4K', False) else '否'}
        elif class_name == 'Tablet':
            specs = {'处理器': getattr(product, 'Processor', '-'), '屏幕': f"{getattr(product, 'Screen_Size_in', 0)}英寸", '手写笔': '是' if getattr(product, 'Stylus_Support', False) else '否'}
        elif class_name in ['BluetoothSpeaker', 'Bluetooth_Speaker']:
            specs = {'功率': f"{getattr(product, 'Power_W', 0)}W", '续航': f"{getattr(product, 'Battery_Hours', 0)}h", '防水': getattr(product, 'Waterproof_Rating', '-')}
        elif class_name == 'Smartwatch':
            specs = {'续航': f"{getattr(product, 'Battery_Days', 0)}天", '系统': getattr(product, 'OS', '-'), '防水': getattr(product, 'Waterproof_Rating', '-')}
        elif class_name == 'Monitor':
            specs = {'尺寸': f"{getattr(product, 'Screen_Size_in', 0)}英寸", '分辨率': getattr(product, 'Resolution', '-'), '刷新率': f"{getattr(product, 'Refresh_Rate_Hz', 0)}Hz"}
        elif class_name in ['GamingConsole', 'Gaming_Console']:
            specs = {'分辨率': getattr(product, 'Max_Resolution', '-'), '存储': f"{getattr(product, 'Storage_GB', 0)}GB", '独占': f"{getattr(product, 'Exclusive_Games_Count', 0)}款"}
        elif class_name == 'GPU':
            specs = {'显存': f"{getattr(product, 'VRAM_GB', 0)}GB", '芯片': getattr(product, 'Chip_Manufacturer', '-'), '功耗': f"{getattr(product, 'TDP_W', 0)}W"}
        else:
            specs = {'ID': getattr(product, 'id', '-')}

    score_html = ""
    if scores:
        top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        score_html = f"<div style='margin: 12px 0;'>{''.join([f'<span class=score-tag>{k} {v}</span>' for k, v in top_scores])}</div>"
    
    spec_html_str = f"<div style='margin-bottom:12px;'>{''.join([f'<span class=spec-tag><b>{k}</b> {v}</span>' for k, v in specs.items()])}</div>" if specs else ""
    
    reason_html = ""
    if reason:
        import re
        r_text = reason.replace('\n', '<br>')
        r_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', r_text)
        reason_html = f"<div class='reason-box'>{r_text}</div>"

    st.markdown(f"""
    <div class='product-card'>
        <div class='product-title'>{brand} {model}</div>
        <div class='product-price'><span class='currency'>¥</span> {int(price):,}</div>
        {score_html}
        {spec_html_str}
        {reason_html}
    </div>
    """, unsafe_allow_html=True)

# --- DASHBOARD LAYOUT ---
results = st.session_state.last_results
reasons = st.session_state.last_reasons
charts = st.session_state.last_charts
analyses = st.session_state.last_analyses

if not results:
    # START PAGE (CENTRIC VIEW)
    # Inject temporary CSS to strictly remove top padding for the hero view
    st.markdown("""
        <style>
            .block-container {
                padding-top: 3rem !important;
                max-width: 1100px;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='home-container'>", unsafe_allow_html=True)
    st.markdown("<h1 class='home-title'>Your Professional Digital<br>Shopping Consultant</h1>", unsafe_allow_html=True)
    st.markdown("<p class='home-subtitle'>基于大模型深度分析，为您从海量数据中精准匹配并生成专属的量化选购仪表盘</p>", unsafe_allow_html=True)
    
    # Center the form horizontally using Streamlit columns
    _, center_col, _ = st.columns([1, 6, 1])
    with center_col:
        st.markdown("<div class='home-form-container'>", unsafe_allow_html=True)
        # We use border=False to eliminate the ugly outer line, letting our CSS handle it
        with st.form("home_consultation_form", border=False):
            c1, c2 = st.columns([4, 1])
            with c1:
                prompt_input = st.text_input("请输入您的选购需求", placeholder="例如：推荐一款 5000 元左右的适合 Vlog 的轻便相机...", label_visibility="collapsed")
            with c2:
                submitted = st.form_submit_button("开始智能分析", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='home-intro'>
        <div class='home-features'>
            <div class='home-feature-item'>🎯 <b>智能对标</b><br><span style='font-size:0.85rem; opacity:0.8;'>精准匹配预算与用途</span></div>
            <div class='home-feature-item'>📊 <b>量化分析</b><br><span style='font-size:0.85rem; opacity:0.8;'>实时生成能力雷达图</span></div>
            <div class='home-feature-item'>🔗 <b>生态协同</b><br><span style='font-size:0.85rem; opacity:0.8;'>跨品类场景化搭配建议</span></div>
            <div class='home-feature-item'>⚠️ <b>风险避雷</b><br><span style='font-size:0.85rem; opacity:0.8;'>智能识别产品潜在短板</span></div>
        </div>
        <div style='margin-top: 40px; display: flex; flex-direction: column; align-items: center;'>
            <b style='font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 12px;'>当前支持品类</b>
            <div class='category-badge-list'>
                <span class='category-badge'>📷 相机</span>
                <span class='category-badge'>📱 手机</span>
                <span class='category-badge'>🎧 耳机</span>
                <span class='category-badge'>💻 笔记本</span>
                <span class='category-badge'>📒 平板</span>
                <span class='category-badge'>⌚ 智能手表</span>
                <span class='category-badge'>🔊 音箱</span>
                <span class='category-badge'>🖥️ 显示器</span>
                <span class='category-badge'>🎮 主机</span>
                <span class='category-badge'>⚡ 显卡</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if submitted and prompt_input:
        st.session_state.current_prompt = prompt_input
        st.rerun()

else:
    # RESULTS DASHBOARD VIEW
    with st.container():
        with st.form("consultation_form"):
            st.markdown("<div class='section-header'>咨询中心 · CONSULTATION CENTER</div>", unsafe_allow_html=True)
            c1, c2 = st.columns([5, 1])
            with c1:
                prompt_input = st.text_input("请输入您的选购需求", placeholder="例如：推荐一款 5000 元左右的适合 Vlog 的相机...", label_visibility="collapsed")
            with c2:
                submitted = st.form_submit_button("开始咨询", use_container_width=True)

        if submitted and prompt_input:
            st.session_state.current_prompt = prompt_input
            st.rerun()

    col_main, col_side = st.columns([4, 1], gap="medium") 

    with col_main:
        cat_name = st.session_state.current_category or "商品"
        with st.container():
            st.markdown(f"<div class='section-header'>选购建议 · {cat_name} 推荐结果</div>", unsafe_allow_html=True)
            cols = st.columns(min(len(results), 3))
            for idx, (col, product) in enumerate(zip(cols, results[:3])):
                with col:
                    reason = reasons[idx] if reasons and idx < len(reasons) else None
                    render_product_card(product, reason)
            
            if charts and any(charts):
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<div class='chart-section-title'>深度对比分析报告</div>", unsafe_allow_html=True)
                valid_charts = [c for c in charts if c is not None]
                if valid_charts:
                    chart_cols = st.columns(len(valid_charts))
                    titles = ["综合能力雷达", "核心得分分布", "性价比指数"]
                    for i, (col, chart) in enumerate(zip(chart_cols, valid_charts)):
                        with col:
                            with st.container(border=True):
                                title = titles[i] if i < len(titles) else "分析视图"
                                st.markdown(f"<div class='chart-title' style='text-align:center;'>{title}</div>", unsafe_allow_html=True)
                                st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False})
                                if analyses and i < len(analyses):
                                    st.markdown(f"<div class='analysis-box'>{analyses[i].replace('📊 ', '').replace('🏆 ', '').replace('💰 ', '')}</div>", unsafe_allow_html=True)
        
        eco_suggestion = st.session_state.last_eco_suggestion
        if eco_suggestion:
            with st.container():
                st.markdown(f"""
                    <div class='eco-box-title'>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path></svg>
                        全场景生态链建议
                    </div>
                    <div class='eco-box-content'>{eco_suggestion}</div>
                """, unsafe_allow_html=True)

    with col_side:
        with st.container():
            st.markdown("<div class='section-header'>咨询记录 · ANALYST NOTES</div>", unsafe_allow_html=True)
            if not st.session_state.messages:
                st.caption("暂无记录")
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(f"生成建议 {len(msg['content'])} 项" if isinstance(msg["content"], list) else msg["content"])
            
            if st.button("清空仪表盘", use_container_width=True, type="secondary"):
                st.session_state.messages, st.session_state.last_results, st.session_state.last_charts, st.session_state.current_category = [], None, None, None
                if 'current_prompt' in st.session_state: del st.session_state.current_prompt
                st.rerun()