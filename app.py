import streamlit as st
from main_agent import CameraAgent
import os

st.set_page_config(page_title="高级AI相机选购助手", layout="wide")

if 'agent' not in st.session_state:
    st.session_state.agent = CameraAgent()

st.title("📸 全维度 AI 智能选购专家")

# --- 侧边栏/表单收集需求 ---
with st.sidebar:
    st.header("🎯 您的需求偏好")
    
    usage = st.multiselect("您将如何使用本相机？", 
        ["记录旅行", "拍摄校园/公司活动", "拍摄体育比赛", "自然风景", "人像摄影", "天文摄影"])
    
    features = st.multiselect("您看中哪些额外功能？", 
        ["内置闪光灯", "录制视频", "蓝牙传输", "触控屏", "内置GPS", "防水"])
    
    weight_tolerance = st.radio("您对重量的接受程度？",
        ["专业负重：3kg也没问题", "平衡型：尽可能避免负重", "轻便型：必须轻便"])
    
    budget_level = st.radio("您的投入预算倾向？",
        ["高端：追求极致性能", "中端：注重性价比", "入门：经济实用款"])
    
    others = st.multiselect("其他要求：", ["尽量购买新型号", "机身材质要好"])
    
    other_text = st.text_input("还有什么想告诉 AI 的？")

# --- 主界面逻辑 ---
if st.button("生成定制化推荐方案"):
    # 将结构化选项拼接成一段描述文本
    full_prompt = f"""
    使用场景：{', '.join(usage)}；
    核心功能要求：{', '.join(features)}；
    重量偏好：{weight_tolerance}；
    预算倾向：{budget_level}；
    额外要求：{', '.join(others)}；
    补充：{other_text}
    """
    
    with st.spinner("AI 正在根据多维需求计算最佳匹配..."):
        results, charts, intent, expert_reason = st.session_state.agent.handle_query(full_prompt)
        
        if results:
            st.success(f"🎯 **AI 诊断结论：** {intent.get('reason_summary', '为您匹配到以下方案')}")
            
            # --- 展示逻辑 (保持之前的布局) ---
            top = results[0]
            st.subheader(f"🏆 最佳匹配：{top.Brand} {top.Model}")
            col1, col2 = st.columns([1, 1.2])
            with col1:
                st.image(charts[0], caption="性能雷达分布")
            with col2:
                st.info("💡 **专家点评：**")
                st.write(expert_reason)
                st.markdown(f"**关键参数：** ¥{top.Price:.0f} | 重量: {top.Weight_g}g | 4K: {'支持' if top.Supports_4K else '不支持'}")
            
            st.divider()
            st.subheader("📊 Top 3 对比分析")
            st.image(charts[1], use_container_width=True)