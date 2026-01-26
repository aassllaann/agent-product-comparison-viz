import plotly.graph_objects as go
import plotly.express as px
import os

def _ensure_dir():
    if not os.path.exists("charts"):
        os.makedirs("charts")

def draw_radar(products, dimensions=None):
    """
    绘制雷达图
    
    Args:
        products: 产品列表
        dimensions: (可选) 维度列表，格式为 [(label, field_name, max_val), ...]
                   如果不传，默认使用相机的配置
    """
    fig = go.Figure()
    
    # 默认配置（兼容 Camera）
    if not dimensions:
        labels = ['便携', '画质', '视频', '高感', '操控']
        # 字段映射逻辑在循环内处理
    else:
        labels = [d[0] for d in dimensions]
        
    colors = ['#008080', '#2E9CCA', '#20C997', '#F4A261', '#9B51E0']

    for i, p in enumerate(products):
        values = []
        if not dimensions:
            # 兼容旧逻辑
            values = [
                getattr(p, 'Portability_Score', 0), 
                getattr(p, 'LowLight_Score', 0), 
                getattr(p, 'Video_Score', 0), 
                min(getattr(p, 'Max_ISO', 0)/25600*100, 100), 
                85
            ]
        else:
            # 通用逻辑
            for _, field, max_val in dimensions:
                val = getattr(p, field, 0)
                if val is None: val = 0
                if max_val:
                    val = min(val / max_val * 100, 100)
                values.append(val)
                
        # 闭合
        values += values[:1]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels + labels[:1],
            fill='toself',
            name=f"{p.Model}",
            line_color=colors[i % len(colors)],
            fillcolor=colors[i % len(colors)],
            opacity=0.4, # 统一透明度，避免叠加过于厚重
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

def draw_comparison(products, field_name):
    # 柱状图对比
    names = [f"{c.Model[:8]}.." for c in products]
    full_names = [f"{c.Brand} {c.Model}" for c in products]
    
    # 获取分数，支持缺省
    scores = []
    real_field_name = field_name
    
    # 尝试找到对应的字段名（因为传入的可能是 "Value_Score" 但模型只有 "Performance_Score"）
    # 这里不做复杂猜测，假设 caller 传对了
    
    for c in products:
        val = getattr(c, field_name, 0)
        if val is None: val = 0
        scores.append(val)
        
    colors = ['#008080', '#2E9CCA', '#20C997', '#F4A261', '#9B51E0']
    
    fig = go.Figure(data=[go.Bar(
        x=names,
        y=scores,
        marker_color=colors[:len(products)],
        text=scores,
        textposition='auto',
        hovertext=full_names,
        hovertemplate='<b>%{hovertext}</b><br>得分: %{y}<extra></extra>'
    )])

    safe_name = field_name.replace('_Score', '评分').replace('_', ' ')
    
    fig.update_layout(
        title=dict(text=f"{safe_name} 对比", font=dict(size=14), x=0.5, y=0.95),
        margin=dict(l=20, r=20, t=40, b=20),
        height=280,
        yaxis=dict(range=[0, 105], showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
        xaxis=dict(tickangle=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def draw_multi_dimension_compare(products, dimensions=None):
    # 多维能力对比（分组柱状图）
    
    if not dimensions:
        # 兼容相机
        dims_conf = [('便携性', 'Portability_Score'), ('低光画质', 'LowLight_Score'), ('视频能力', 'Video_Score')]
    else:
        dims_conf = dimensions
        
    labels = [d[0] for d in dims_conf]
    fields = [d[1] for d in dims_conf]
    
    colors = ['#008080', '#2E9CCA', '#20C997', '#F4A261', '#9B51E0']
    
    fig = go.Figure()

    for i, p in enumerate(products):
        scores = []
        for f in fields:
            val = getattr(p, f, 0)
            if val is None: val = 0
            scores.append(val)
        
        fig.add_trace(go.Bar(
            name=f"{p.Model}",
            x=labels,
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