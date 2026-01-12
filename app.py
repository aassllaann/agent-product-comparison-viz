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
    .main .block-container { padding: 1rem 2rem; max-width: 100%; }
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

# 品类图标
CATEGORY_ICONS = {
    "camera": "📷", "phone": "📱", "headphone": "🎧", "laptop": "💻", "tablet": "📟",
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
        scores = {'便携': getattr(product, 'Portability_Score', 0), '低光': getattr(product, 'LowLight_Score', 0)}
        specs = {'重量': f"{getattr(product, 'Weight_g', 0)}g", 'ISO': getattr(product, 'Max_ISO', 0)}
    
    st.markdown(f"<div class='product-card'>", unsafe_allow_html=True)
    st.markdown(f"**{brand} {model}**")
    st.markdown(f"💰 ¥{int(price)}")
    
    # 评分
    if scores:
        score_str = " | ".join([f"{k.replace('_Score','')}: {v}" for k, v in list(scores.items())[:3]])
        st.caption(score_str)
    
    # 规格
    if specs:
        with st.expander("规格详情", expanded=False):
            for k, v in list(specs.items())[:5]:
                st.write(f"• {k}: {v}")
    
    # 推荐理由
    if reason:
        # 手动处理 Markdown 转 HTML 以确保在 div 中样式生效
        import re
        html_reason = reason.replace('\n', '<br>')
        html_reason = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', html_reason)
        st.markdown(f"<div class='reason-box'>💡 {html_reason}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- 侧边栏 ---
with st.sidebar:
    st.header("🛒 支持品类")
    detector = CategoryDetector()
    
    groups = {
        "数码": ["camera", "phone", "headphone", "laptop"],
        "美妆": ["skincare", "cosmetics"],
        "办公": ["stationery", "office"],
        "生活": ["appliance", "sports", "book"],
    }
    
    for group, keys in groups.items():
        cats = [f"{get_icon(k)} {detector.KNOWN_CATEGORIES[k]['name']}" for k in keys if k in detector.KNOWN_CATEGORIES]
        st.write(f"**{group}**: " + ", ".join(cats))
    
    st.markdown("---")
    st.info("""
    💡 **数据说明**
    - **所有品类**: 统一使用电商API
    - **数据来源**: 京东联盟API（Mock）
    - **后续**: 可接入真实API密钥
    """)
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.session_state.last_results = None
        st.session_state.last_charts = None
        st.rerun()

# --- 主界面：左右分栏 ---
st.title("🛒 AI 智能选购助手")

# 左侧：对话区 | 右侧：推荐结果区
left_col, right_col = st.columns([1, 3])

# 左侧：对话区
with left_col:
    st.markdown("<div class='section-header'>💬 对话</div>", unsafe_allow_html=True)
    
    # 对话容器
    chat_container = st.container()
    
    with chat_container:
        # 显示历史消息
        for msg in st.session_state.messages[-10:]:  # 只显示最近10条
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                st.markdown(f"**🧑 你**: {content}")
            else:
                # 助手消息只显示简短摘要
                if isinstance(content, list):
                    st.markdown(f"**🤖 助手**: 已为您推荐 {len(content)} 款商品")
                else:
                    st.markdown(f"**🤖 助手**: {str(content)[:100]}...")
    
    st.markdown("---")
    
    # 输入框
    prompt = st.chat_input("输入需求，如：推荐护肤精华、想买耳机...")
    
    if prompt:
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 识别品类
        cat_key, cat_name = detector.detect_category(prompt)
        st.session_state.current_category = cat_name
        
        # 获取推荐
        with st.spinner(f"搜索 {cat_name}..."):
            history = [{"role": m["role"], "content": str(m["content"])} for m in st.session_state.messages[-6:]]
            reply, charts, results, analyses = st.session_state.agent.handle_chat(prompt, history=history)
            
            # 保存结果
            st.session_state.last_results = results
            st.session_state.last_charts = charts
            st.session_state.last_analyses = analyses
            st.session_state.last_reasons = reply if isinstance(reply, list) else None
            
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
        st.info("👈 在左侧输入您的需求，开始智能选购！\n\n支持：相机、手机、耳机、护肤品、文具、小家电等 12 个品类")