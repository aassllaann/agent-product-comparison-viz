import streamlit as st
from main_agent import CameraAgent
import os

# 1. 页面配置：设置标题、图标及布局
st.set_page_config(
    page_title="AI 相机选购智能助手", 
    layout="wide", 
    page_icon="📸"
)

# 2. 自定义 CSS 样式：优化数值显示和气泡间距
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.6rem; color: #008080; }
    .stChatMessage { margin-bottom: 20px; }
    .main .block-container { padding-left: 2vw !important; padding-right: 2vw !important; max-width: 98vw; }
    </style>
""", unsafe_allow_html=True)

# 3. 初始化持久化状态
if 'agent' not in st.session_state:
    st.session_state.agent = CameraAgent()
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- 定义渲染函数：确保新消息和历史记录排版一致 ---
def render_assistant_response(reply, charts, results):
    """
    统一渲染 AI 的图文回复组件
    """
    if charts and results:
        # 兼容历史和新结构：results为(results, chart_analyses)或仅results
        if isinstance(results, tuple) and len(results) == 2 and isinstance(results[0], list):
            cam_list, chart_analyses = results
        else:
            cam_list = results
            chart_analyses = None

        # 三机型横向对比卡片
        st.markdown("<h4 style='margin-bottom:0.5em;'>✨ 推荐三款相机对比</h4>", unsafe_allow_html=True)
        cols = st.columns(3, gap="large")
        for idx, (col, cam) in enumerate(zip(cols, cam_list)):
            with col:
                # 型号标签移到卡片外部
                st.markdown(f"<div style='font-weight:700;font-size:1.08em;color:#008080;text-align:center;margin-bottom:0.3em;'>{cam.Brand} {cam.Model}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='background:#fff;border-radius:12px;padding:18px 16px 10px 16px;box-shadow:0 2px 8px #e0eaff;'>", unsafe_allow_html=True)
                img_path = getattr(cam, 'image_file', None)
                if img_path and os.path.exists(img_path):
                    st.image(img_path, width=220, clamp=True)
                else:
                    st.info("🖼️ 暂无实机图片")
                m1, m2 = st.columns(2)
                m1.metric("参考价格", f"¥{cam.Price}")
                m2.metric("机身重量", f"{cam.Weight_g}g")
                st.markdown("""
                                        <ul style='list-style:none;padding-left:0;margin-bottom:8px;'>
                                            <li><b>最大ISO</b>: {iso}</li>
                                            <li><b>屏幕尺寸</b>: {screen} 英寸</li>
                                            <li><b>4K视频</b>: {support4k}</li>
                                            <li><b>总像素</b>: {mp} MP</li>
                                            <li><b>传感器类型</b>: {sensor}</li>
                                            <li><b>上市年份</b>: {year}</li>
                                        </ul>
                                        </div>
                                """.format(
                                        iso=cam.Max_ISO,
                                        screen=cam.Screen_Size_in,
                                        support4k='✅ 支持' if cam.Supports_4K else '❌ 不支持',
                                        mp=cam.Total_megapixels if hasattr(cam, 'Total_megapixels') and cam.Total_megapixels is not None else '-',
                                        sensor=cam.Sensor_type if hasattr(cam, 'Sensor_type') and cam.Sensor_type else '-',
                                        year=cam.Year if hasattr(cam, 'Year') and cam.Year else '-'), unsafe_allow_html=True)
                # 推荐理由（每台）
                if reply and isinstance(reply, list) and idx < len(reply):
                    st.markdown("""
                        <div style='background:linear-gradient(90deg,#e0ecff 0%,#f8f9fa 100%);border-radius:12px;padding:14px 14px 8px 14px;box-shadow:0 2px 8px #e0eaff;margin-top:0.5em;'>
                        <h5 style='margin-top:0;margin-bottom:0.5em;'>🤖 推荐理由</h5>
                    """ + reply[idx] + "</div>", unsafe_allow_html=True)
        st.markdown("<hr style='margin:1.5em 0 1em 0;border:0;border-top:1.5px solid #e0eaff;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight:600;font-size:1.1em;margin-bottom:0.5em;'>📊 性能深度对比分析</div>", unsafe_allow_html=True)
        chart_cols = st.columns(3)
        captions = ["性能画像", "Top3 对比", "核心能力对比"]
        for i, col in enumerate(chart_cols):
            if i < len(charts):
                with col:
                    st.plotly_chart(charts[i], use_container_width=True, config={'displayModeBar': False})
        # 图表下方详细分析
        if chart_analyses:
            for i, analysis in enumerate(chart_analyses):
                st.markdown(f"<div style='margin-top:-0.5em;margin-bottom:1.2em;color:#555;font-size:0.98em;'>🔎 {analysis}</div>", unsafe_allow_html=True)
    else:
        st.markdown(reply)

# --- 主界面 ---
st.title("💬 AI 相机选购智能助手")
st.markdown("---")
st.write("欢迎！您可以直接输入您的摄影需求或预算偏好。")

# 4. 渲染历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "results" in msg:
            render_assistant_response(msg["content"], msg["charts"], msg["results"])
        else:
            st.markdown(msg["content"])

# 5. 聊天输入逻辑
if prompt := st.chat_input("在这里输入您的需求..."):
    # 记录并显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成并渲染 AI 响应
    with st.chat_message("assistant"):
        with st.spinner("AI 正在根据您的需求分析机型..."):
            # 构造历史对话，剔除图表，仅保留role和content
            history_msgs = []
            for m in st.session_state.messages:
                if m["role"] in ["user", "assistant"]:
                    history_msgs.append({"role": m["role"], "content": m["content"]})
            reply, charts, results, chart_analyses = st.session_state.agent.handle_chat(prompt, history=history_msgs)
            # 传递详细分析到渲染函数
            render_assistant_response(reply, charts, (results, chart_analyses))
            # 将结构化数据存入 Session，保证刷新后排版不乱
            msg_data = {"role": "assistant", "content": reply}
            if charts and results:
                msg_data.update({"charts": charts, "results": (results, chart_analyses)})
            st.session_state.messages.append(msg_data)