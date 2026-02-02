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
# Check for 'current_prompt' from the form submission.
# This runs BEFORE the layout logic below, so 'last_results' will be populated when referenced.
if 'current_prompt' in st.session_state and st.session_state.current_prompt:
    prompt = st.session_state.current_prompt
    # Consume prompt immediately
    del st.session_state.current_prompt
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    cat_key, cat_name = detector.detect_category(prompt)
    st.session_state.current_category = cat_name
    
    # Use spinner at the top or a placeholder if preferred, 
    # but top-level spinner is fine for this flow.
    with st.spinner("正在分析您的需求并检索数据..."):
        try:
            history = [{"role": m["role"], "content": str(m["content"])} for m in st.session_state.messages[-6:] if m["role"] != "user" or m["content"] != prompt] 
            # Note: history filtering above is just a safety, usually [-6:] is fine. 
            # Simplified history fetching:
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


# 品类标签 (保持极简设计)
CATEGORY_LABELS = {
    "camera": "相机", "phone": "手机", "headphone": "耳机", 
    "laptop": "笔记本", "tablet": "平板", "smartwatch": "智能手表", 
    "bluetooth_speaker": "音箱", "monitor": "显示器", 
    "gaming_console": "主机", "gpu": "显卡"
}

def get_label(cat):
    return CATEGORY_LABELS.get(cat, "商品")

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
        
        # 动态提取规格和评分
        class_name = type(product).__name__
        specs = {}
        scores = {}
        
        # 提取评分
        for k in dir(product):
            if k.endswith('_Score') and getattr(product, k):
                scores[k.replace('_Score', '')] = getattr(product, k)

        # 提取特定规格
        if class_name == 'Phone':
            specs = {
                '处理器': getattr(product, 'Processor', '-'),
                '内存': f"{getattr(product, 'RAM_GB', 0)}G+{getattr(product, 'Storage_GB', 0)}G",
                '电池': f"{getattr(product, 'Battery_mAh', 0)}mAh",
                '主摄': f"{getattr(product, 'Camera_MP', 0)}MP"
            }
        elif class_name == 'Laptop':
            specs = {
                'CPU': getattr(product, 'CPU', '-'),
                '显卡': getattr(product, 'GPU', '-'),
                '内存': f"{getattr(product, 'RAM_GB', 0)}G",
                '屏幕': f"{getattr(product, 'Screen_Size_in', 0)}英寸"
            }
        elif class_name == 'Headphone':
            specs = {
                '类型': getattr(product, 'Type', '-'),
                '降噪': '是' if getattr(product, 'ANC', False) else '否',
                '续航': f"{getattr(product, 'Battery_Hours', 0)}h"
            }
        elif class_name == 'Camera':
            specs = {
                '像素': f"{getattr(product, 'Total_megapixels', 0)}MP",
                'ISO': getattr(product, 'Max_ISO', 0),
                '重量': f"{getattr(product, 'Weight_g', 0)}g", 
                '4K': '是' if getattr(product, 'Supports_4K', False) else '否'
            }
        elif class_name == 'Tablet':
             specs = {
                '处理器': getattr(product, 'Processor', '-'),
                '屏幕': f"{getattr(product, 'Screen_Size_in', 0)}英寸",
                '手写笔': '是' if getattr(product, 'Stylus_Support', False) else '否'
             }
        elif class_name == 'BluetoothSpeaker' or class_name == 'Bluetooth_Speaker':
            specs = {
                '功率': f"{getattr(product, 'Power_W', 0)}W",
                '续航': f"{getattr(product, 'Battery_Hours', 0)}h",
                '防水': getattr(product, 'Waterproof_Rating', '-')
            }
        elif class_name == 'Smartwatch':
            specs = {
                '续航': f"{getattr(product, 'Battery_Days', 0)}天",
                '系统': getattr(product, 'OS', '-'),
                '防水': getattr(product, 'Waterproof_Rating', '-')
            }
        elif class_name == 'Monitor':
            specs = {
                '尺寸': f"{getattr(product, 'Screen_Size_in', 0)}英寸",
                '分辨率': getattr(product, 'Resolution', '-'),
                '刷新率': f"{getattr(product, 'Refresh_Rate_Hz', 0)}Hz"
            }
        elif class_name == 'GamingConsole' or class_name == 'Gaming_Console':
            specs = {
                '分辨率': getattr(product, 'Max_Resolution', '-'),
                '存储': f"{getattr(product, 'Storage_GB', 0)}GB",
                '独占': f"{getattr(product, 'Exclusive_Games_Count', 0)}款"
            }
        elif class_name == 'GPU':
            specs = {
                '显存': f"{getattr(product, 'VRAM_GB', 0)}GB",
                '芯片': getattr(product, 'Chip_Manufacturer', '-'),
                '功耗': f"{getattr(product, 'TDP_W', 0)}W"
            }
        else:
             # 默认/Fallback
             specs = {'ID': getattr(product, 'id', '-')}

    # Build the complete HTML string
    score_html = ""
    if scores:
        top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        score_html = f"<div style='margin: 12px 0;'>{''.join([f'<span class=score-tag>{k} {v}</span>' for k, v in top_scores])}</div>"
    
    spec_html_str = ""
    if specs:
        spec_html_str = f"<div style='margin-bottom:12px;'>{''.join([f'<span class=spec-tag><b>{k}</b> {v}</span>' for k, v in specs.items()])}</div>"
    
    reason_html = ""
    if reason:
        import re
        r_text = reason.replace('\n', '<br>')
        r_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', r_text)
        reason_html = f"<div class='reason-box'>{r_text}</div>"

    card_html = f"""
    <div class='product-card'>
        <div class='product-title'>{brand} {model}</div>
        <div class='product-price'><span class='currency'>¥</span> {int(price):,}</div>
        {score_html}
        {spec_html_str}
        {reason_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

# --- DASHBOARD LAYOUT ---
# 1. HERO SECTION: REQUIREMENT INPUT
hero_container = st.container()
with hero_container:
    # 使用 st.form 配合 CSS [data-testid="stForm"] 自动获得玻璃拟态样式
    with st.form("consultation_form"):
        st.markdown("<div class='section-header'>咨询中心 · CONSULTATION CENTER</div>", unsafe_allow_html=True)
        # 使用 columns 让输入框和按钮在一行或协调布局
        c1, c2 = st.columns([5, 1])
        with c1:
            prompt_input = st.text_input("请输入您的选购需求", placeholder="例如：推荐一款 5000 元左右的适合 Vlog 的相机...", label_visibility="collapsed")
        with c2:
            submitted = st.form_submit_button("开始咨询", use_container_width=True)

    if submitted and prompt_input:
        st.session_state.current_prompt = prompt_input
        st.rerun()

# 2. MAIN DASHBOARD AREA
col_main, col_side = st.columns([4, 1], gap="medium") 

# --- LEFT COLUMN: RESULTS & ANALYSIS ---
with col_main:
    cat_name = st.session_state.current_category or "商品"
    main_container = st.container()
    with main_container:
        st.markdown(f"<div class='section-header'>选购建议 · {cat_name} 推荐结果</div>", unsafe_allow_html=True)
        
        results = st.session_state.last_results
        reasons = st.session_state.last_reasons
        charts = st.session_state.last_charts
        analyses = st.session_state.last_analyses
        
        if results:
            # 商品卡格
            cols = st.columns(3 if len(results) >= 3 else len(results))
            for idx, (col, product) in enumerate(zip(cols, results[:3])):
                with col:
                    reason = reasons[idx] if reasons and idx < len(reasons) else None
                    render_product_card(product, reason)
            
            # 图表与深度分析
            if charts and any(charts):
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown("<div class='chart-section-title'>深度对比分析报告</div>", unsafe_allow_html=True)
                
                valid_charts = [c for c in charts if c is not None]
                if valid_charts:
                    chart_cols = st.columns(len(valid_charts))
                    titles = ["综合能力雷达", "核心得分分布", "性价比指数"]
                    
                    for i, (col, chart) in enumerate(zip(chart_cols, valid_charts)):
                        with col:
                            # Use Streamlit native container with border for the card effect
                            with st.container(border=True):
                                title = titles[i] if i < len(titles) else "分析视图"
                                st.markdown(f"<div class='chart-title' style='text-align:center;'>{title}</div>", unsafe_allow_html=True)
                                st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False})
                                
                                if analyses and i < len(analyses):
                                    clean_text = analyses[i].replace("📊 ", "").replace("🏆 ", "").replace("💰 ", "")
                                    st.markdown(f"<div class='analysis-box'>{clean_text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='hero-card'>
<h3 style='margin-bottom: 12px; color: var(--color-accent-purple);'>👋 您的专业数码选购顾问</h3>
<p style='color: var(--text-secondary); font-size: 0.95rem;'>请在上方输入您的选购需求，系统将为您生成深度个性化对标仪表盘。</p>
<div style='background: rgba(255,255,255,0.6); border-radius: 12px; padding: 20px; margin-top: 15px; border: 1px solid rgba(159, 122, 234, 0.15);'>
<p style='margin-bottom: 10px;'><b style='color: var(--color-accent-pink);'>✨ 核心功能亮点：</b></p>
<ul style='font-size: 0.85rem; line-height: 1.7; color: var(--text-primary); margin-left: -15px;'>
<li><b>🎯 智能对标推荐</b>：基于预算与核心用途，从海量数据库中深度筛选匹配产品。</li>
<li><b>📊 多维量化分析</b>：实时生成能力雷达图、关键参数对比及客观性价比指数。</li>
<li><b>🔗 生态链协同</b>：独家提供跨品类（如：相机 + 镜头 + 稳定器）的场景化搭配建议。</li>
<li><b>⚠️ 实时风险避雷</b>：智能识别并标注产品的潜在短板，大幅降低您的决策溢价。</li>
</ul>
</div>
<div style='margin-top: 20px;'>
<p style='font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 8px;'><b>当前支持品类：</b></p>
<div style='display: flex; flex-wrap: wrap; gap: 8px; font-size: 0.8rem; color: var(--color-accent-deep);'>
<span>📷 相机</span> | <span>📱 手机</span> | <span>🎧 耳机</span> | <span>💻 笔记本</span> | 
<span>📒 平板</span> | <span>⌚ 智能手表</span> | <span>🔊 音箱</span> | <span>🖥️ 显示器</span> | 
<span>🎮 主机</span> | <span>⚡ 显卡</span>
</div>
</div>
</div>""", unsafe_allow_html=True)
        
    # 生态系统建议 (独立容器)
    eco_suggestion = st.session_state.last_eco_suggestion
    if eco_suggestion:
        eco_container = st.container()
        with eco_container:
            st.markdown(f"""
                <div class='eco-box-title'>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path></svg>
                    全场景生态链建议
                </div>
                <div class='eco-box-content'>{eco_suggestion}</div>
            """, unsafe_allow_html=True)

# --- RIGHT COLUMN: CONSULTANT LOGS ---
with col_side:
    side_container = st.container()
    with side_container:
        st.markdown("<div class='section-header'>咨询记录 · ANALYST NOTES</div>", unsafe_allow_html=True)
        
        # 对话内容
        if not st.session_state.messages:
            st.caption("暂无记录")
        
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]
            with st.chat_message(role):
                if isinstance(content, list):
                    st.write(f"生成建议 {len(content)} 项")
                else:
                    st.write(content)
        
        if st.button("清空仪表盘", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.session_state.last_results = None
            st.session_state.last_charts = None
            st.session_state.current_category = None
            if 'current_prompt' in st.session_state:
                del st.session_state.current_prompt
            st.rerun()

# 处理输入逻辑
# Logic moved to top