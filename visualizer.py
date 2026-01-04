import plotly.graph_objects as go
import plotly.express as px
import os

def _ensure_dir():
    if not os.path.exists("charts"):
        os.makedirs("charts")

def draw_radar(cameras):
    # 雷达图数据维度
    labels = ['便携', '画质', '视频', '高感', '操控']
    colors = ['#008080', '#FF8C00', '#4682B4']
    
    fig = go.Figure()

    for i, cam in enumerate(cameras):
        # 归一化/处理数据
        values = [cam.Portability_Score, cam.LowLight_Score, cam.Video_Score, 
                  min(cam.Max_ISO/25600*100, 100), 85] # 操控暂时固定85，实际可拓展
        # 闭合
        values += values[:1]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels + labels[:1],
            fill='toself',
            name=f"{cam.Model}",
            line_color=colors[i % len(colors)],
            fillcolor=colors[i % len(colors)],
            opacity=0.6 if i==0 else 0.3, # 第一款稍微深一点
            hovertemplate='<b>%{theta}</b>: %{r:.1f}<br>(%{data.name})<extra></extra>'
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=8), gridcolor='rgba(0,0,0,0.1)'),
            angularaxis=dict(tickfont=dict(size=10, color='#333'), gridcolor='rgba(0,0,0,0.1)')
        ),
        margin=dict(l=40, r=40, t=20, b=20),
        height=300,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5), # 图例放到底部
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def draw_comparison(cameras, field_name):
    # 此函数保持不变，因为柱状图本身就是对比
    names = [f"{c.Model[:8]}.." for c in cameras]
    full_names = [f"{c.Brand} {c.Model}" for c in cameras]
    scores = [getattr(c, field_name) for c in cameras]
    colors = ['#008080', '#FF8C00', '#4682B4']
    
    fig = go.Figure(data=[go.Bar(
        x=names,
        y=scores,
        marker_color=colors[:len(cameras)],
        text=scores,
        textposition='auto',
        hovertext=full_names,
        hovertemplate='<b>%{hovertext}</b><br>得分: %{y}<extra></extra>'
    )])

    fig.update_layout(
        title=dict(text=f"{field_name} 对比", font=dict(size=14), x=0.5, y=0.95),
        margin=dict(l=20, r=20, t=40, b=20),
        height=280,
        yaxis=dict(range=[0, 105], showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
        xaxis=dict(tickangle=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def draw_multi_dimension_compare(cameras):
    # 多维能力对比（分组柱状图）
    dims = ['便携性', '低光画质', '视频能力']
    colors = ['#008080', '#FF8C00', '#4682B4'] # 三款相机三种颜色
    
    fig = go.Figure()

    for i, cam in enumerate(cameras):
        # 提取该相机的三个维度得分
        scores = [cam.Portability_Score, cam.LowLight_Score, cam.Video_Score]
        
        fig.add_trace(go.Bar(
            name=f"{cam.Model}",
            x=dims,
            y=scores,
            marker_color=colors[i % len(colors)],
            text=scores,
            textposition='auto',
            hovertemplate='<b>%{x}</b>: %{y}<br>(%{data.name})<extra></extra>'
        ))

    fig.update_layout(
        title=dict(text="核心能力多维对比", font=dict(size=14), x=0.5, y=0.95),
        barmode='group', # 分组显示
        xaxis=dict(tickfont=dict(size=12)),
        yaxis=dict(range=[0, 105], title='评分', gridcolor='rgba(0,0,0,0.1)'),
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig