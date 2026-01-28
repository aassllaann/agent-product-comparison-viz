import streamlit as st
from multi_agent import MultiCategoryAgent
from category_detector import CategoryDetector
import os

# 1. 页面配置
st.set_page_config(
    page_title="AI 智能选购助手", 
    layout="wide", 
    page_icon="🛒"
)

# 2. 自定义 CSS
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.4rem; color: #008080; }
    .main .block-container { padding: 0.5rem 2rem; max-width: 100%; }
    h1 { margin-top: -1.5rem; padding-top: 0; }
    .chat-container { 
        height: 400px; 
        overflow-y: auto; 
        border: 1px solid #e0e0e0; 
        border-radius: 8px; 
        padding: 1rem;
        background: #fafafa;
    }
    .product-card {
        background: #fff;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    .reason-box {
        background: linear-gradient(90deg, #e8f4f8 0%, #f8f9fa 100%);
        border-radius: 8px;
        padding: 12px;
        margin-top: 0.5em;
        font-size: 0.9em;
    }
    .section-header {
        font-size: 1.2em;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.8em;
        padding-bottom: 0.5em;
        border-bottom: 2px solid #008080;
    }
    </style>
""", unsafe_allow_html=True)

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

# 品类图标
CATEGORY_ICONS = {
    "camera": "📷", "phone": "📱", "headphone": "🎧", "laptop": "💻", "tablet": "📟",
    "smartwatch": "⌚", "bluetooth_speaker": "🔊", "monitor": "🖥️", 
    "gaming_console": "🎮", "gpu": "🎨",
    "skincare": "🧴", "cosmetics": "💄", "stationery": "✏️", "office": "🖨️",
    "appliance": "🔌", "sports": "👟", "book": "📚"
}

def get_icon(cat):
    return CATEGORY_ICONS.get(cat, "🛒")

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
                '耳机类型': getattr(product, 'Type', '-'),
                '主动降噪': '✅' if getattr(product, 'ANC', False) else '❌',
                '续航时间': f"{getattr(product, 'Battery_Hours', 0)}h"
            }
        elif class_name == 'Camera':
            specs = {
                '像素': f"{getattr(product, 'Total_megapixels', 0)}MP",
                'ISO': getattr(product, 'Max_ISO', 0),
                '重量': f"{getattr(product, 'Weight_g', 0)}g", 
                '4K': '✅' if getattr(product, 'Supports_4K', False) else '❌'
            }
        elif class_name == 'Tablet':
             specs = {
                '处理器': getattr(product, 'Processor', '-'),
                '屏幕': f"{getattr(product, 'Screen_Size_in', 0)}英寸",
                '手写笔': '✅' if getattr(product, 'Stylus_Support', False) else '❌'
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

    st.markdown(f"<div class='product-card'>", unsafe_allow_html=True)
    st.markdown(f"**{brand} {model}**")
    st.markdown(f"<div style='color:#008080;font-weight:bold;font-size:1.1em'>💰 ¥{int(price)}</div>", unsafe_allow_html=True)
    
    # 评分展示
    if scores:
        # 选取前3个关键评分
        top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        score_tags = "".join([f"<span style='background:#e0f2f1;color:#00695c;padding:2px 8px;border-radius:12px;margin-right:4px;font-size:0.8em'>{k} {v}</span>" for k, v in top_scores])
        st.markdown(f"<div style='margin: 8px 0;'>{score_tags}</div>", unsafe_allow_html=True)
    
    # 规格展示 (不折叠，使用 Tag 样式)
    if specs:
        st.markdown("<div style='margin-bottom:8px;display:flex;flex-wrap:wrap;gap:4px;'>", unsafe_allow_html=True)
        for k, v in specs.items():
            st.markdown(f"<span style='background:#f5f5f5;padding:2px 6px;border-radius:4px;font-size:0.85em;color:#555'><b>{k}</b>: {v}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # 推荐理由
    if reason:
        import re
        html_reason = reason.replace('\n', '<br>')
        html_reason = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', html_reason)
        st.markdown(f"<div class='reason-box'>💡 {html_reason}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- 主界面：左右分栏 ---
st.title("🛒 数码产品智能导购 · 生态推荐")
st.caption("💡 支持 10 大品类：📷 相机 | 📱 手机 | 💻 笔记本 | 🎧 耳机 | 📟 平板 | ⌚ 智能手表 | 🔊 音箱 | 🖥️ 显示器 | 🎮 主机 | 🎨 显卡")

# 左侧：对话区 | 右侧：推荐结果区
left_col, right_col = st.columns([0.6, 2.2]) # 调整比例，对话栏再窄一些

# 左侧：对话区
with left_col:
    st.markdown("<div class='section-header'>💬 智能咨询</div>", unsafe_allow_html=True)
    
    # 对话容器
    chat_container = st.container(height=500) # 使用固定高度容器
    
    with chat_container:
        if not st.session_state.messages:
            st.info("""👋 你好！我是你的**数码生态选购助手**。
            
**🎯 核心功能：**
• 智能推荐最适合你的产品
• 提供生态化搭配建议，让设备发挥更大价值
• 支持多轮对话，理解你的真实场景

**💬 快速开始：**
- *推荐一款适合Vlog的相机，预算5000左右*
- *经常出差，需要轻薄本和配套耳机*
- *想玩《黑神话》，帮我选主机和显示器*

支持品类：📷 相机 | 📱 手机 | 💻 笔记本 | 🎧 耳机 | 📟 平板 | ⌚ 智能手表 | 🔊 音箱 | 🖥️ 显示器 | 🎮 主机 | 🎨 显卡

输入你的需求，开始智能选购吧！""")
            
        # 显示历史消息
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                st.chat_message("user").write(content)
            else:
                # 助手消息只显示简短摘要
                with st.chat_message("assistant"):
                    if isinstance(content, list):
                        st.write(f"已为您推荐 {len(content)} 款商品，请看右侧详情 👉")
                    else:
                        st.write(content)
    
    # 输入区域
    prompt = st.chat_input("输入需求，如：推荐一款游戏本、想买个降噪耳机...")
    
    # 底部工具栏
    col_tools, col_empty = st.columns([1, 2])
    with col_tools:
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_results = None
            st.session_state.last_charts = None
            st.rerun()
    
    if prompt:
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 识别品类
        cat_key, cat_name = detector.detect_category(prompt)
        # 强制修正为电子产品范围（防止 detector 返回其他）
        if cat_key not in ['camera', 'phone', 'laptop', 'headphone', 'tablet']:
             # 实际上 DynamicAgent 还是会处理，但这里为了前端展示，我们可以更新 current_category
             pass

        st.session_state.current_category = cat_name
        
        # 获取推荐
        with st.spinner(f"正在分析需求并搜索 {cat_name}..."):
            history = [{"role": m["role"], "content": str(m["content"])} for m in st.session_state.messages[-6:]]
            reply, charts, results, analyses, eco_suggestion = st.session_state.agent.handle_chat(prompt, history=history)
            
            # 保存结果
            st.session_state.last_results = results
            st.session_state.last_charts = charts
            st.session_state.last_analyses = analyses
            st.session_state.last_reasons = reply if isinstance(reply, list) else None
            st.session_state.last_eco_suggestion = eco_suggestion
            
            # 添加助手消息
            st.session_state.messages.append({"role": "assistant", "content": reply})
        
        st.rerun()

# 右侧：推荐结果区
with right_col:
    cat_name = st.session_state.current_category or "商品"
    st.markdown(f"<div class='section-header'>🎯 {cat_name}推荐结果</div>", unsafe_allow_html=True)
    
    results = st.session_state.last_results
    reasons = st.session_state.last_reasons
    charts = st.session_state.last_charts
    analyses = st.session_state.last_analyses
    
    if results:
        # 商品卡片网格
        cols = st.columns(3 if len(results) >= 3 else len(results))
        
        for idx, (col, product) in enumerate(zip(cols, results[:3])):
            with col:
                reason = reasons[idx] if reasons and idx < len(reasons) else None
                render_product_card(product, reason)
        
        # 生态系统搭配建议板块
        eco_suggestion = st.session_state.last_eco_suggestion
        if eco_suggestion:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%); 
                            border-left: 5px solid #008080; 
                            padding: 15px; 
                            border-radius: 8px; 
                            margin: 20px 0;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.05);'>
                    <div style='color: #008080; font-weight: bold; font-size: 1.1em; margin-bottom: 5px;'>
                        🔗 全场景生态搭配建议
                    </div>
                    <div style='color: #334e68; font-size: 1em; line-height: 1.5;'>
                        {eco_suggestion}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # 图表区
        if charts and any(charts):
            st.markdown("---")
            st.markdown("<div style='font-size:1.1em;font-weight:600;margin-bottom:1em;'>📊 深度对比分析</div>", unsafe_allow_html=True)
            
            valid_charts = [c for c in charts if c is not None]
            if valid_charts:
                # 调整为3列显示3个图表
                chart_cols = st.columns(len(valid_charts))
                titles = ["综合能力雷达图", "核心得分对比", "价格性价比分布"]
                
                for i, (col, chart) in enumerate(zip(chart_cols, valid_charts)):
                    with col:
                        title = titles[i] if i < len(titles) else "分析图表"
                        st.markdown(f"<p style='text-align:center;font-weight:500;'>{title}</p>", unsafe_allow_html=True)
                        st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False})
                        
                        # 在图表下方显示对应分析
                        if analyses and i < len(analyses):
                            analysis_text = analyses[i]
                            # 移除开头的 emoji 以保持整洁，已通过样式美化
                            clean_text = analysis_text.replace("📊 ", "").replace("🏆 ", "").replace("💰 ", "")
                            st.markdown(f"""
                                <div style='background:#f8f9fa;border-radius:8px;padding:10px;font-size:0.9em;color:#555;'>
                                    💡 {clean_text}
                                </div>
                            """, unsafe_allow_html=True)
        
        # 移除底部的统一分析文本循环
    else:
        st.info("👈 在左侧输入您的需求，开始智能选购！\n\n支持手机、笔记本、相机、耳机、显卡等 10 大数码品类，为您提供全场景生态搭配建议。")