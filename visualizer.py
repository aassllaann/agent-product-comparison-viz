import plotly.graph_objects as go
import plotly.express as px
import os

def _ensure_dir():
    if not os.path.exists("charts"):
        os.makedirs("charts")

# Theme Palette (Quieter & Refined)
THEME_COLORS = [
    '#5D89B1',  # Muted Blue
    '#F1C40F',  # Flat Gold
    '#E67E22',  # Carrot Orange
]

def _apply_theme(fig):
    """应用主题配置"""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font=dict(family="Inter, sans-serif", color="#3A5A78"),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(color='#3A5A78')
        ),
        yaxis=dict(
            gridcolor='rgba(58, 90, 120, 0.1)',
            zerolinecolor='rgba(58, 90, 120, 0.1)',
            tickfont=dict(color='#3A5A78')
        )
    )
    return fig

def draw_radar(products, dimensions=None):
    """
    绘制雷达图
    """
    fig = go.Figure()
    
    if not dimensions:
        labels = ['便携', '画质', '视频', '高感', '操控']
    else:
        labels = [d[0] for d in dimensions]

    for i, p in enumerate(products):
        values = []
        if not dimensions:
            values = [
                getattr(p, 'Portability_Score', 0), 
                getattr(p, 'LowLight_Score', 0), 
                getattr(p, 'Video_Score', 0), 
                min(getattr(p, 'Max_ISO', 0)/25600*100, 100), 
                85
            ]
        else:
            for _, field, max_val in dimensions:
                val = getattr(p, field, 0)
                if val is None: val = 0
                if max_val:
                    val = min(val / max_val * 100, 100)
                values.append(val)
                
        # 闭合
        values += values[:1]
        
        color = THEME_COLORS[i % len(THEME_COLORS)]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels + labels[:1],
            fill='toself',
            name=f"{p.Model}",
            line_color=color,
            fillcolor=color,
            opacity=0.2, 
            hovertemplate='<b>%{theta}</b>: %{r:.1f}<br>(%{data.name})<extra></extra>'
        ))

    _apply_theme(fig)
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9, color='#3A5A78'), gridcolor='rgba(0, 115, 203, 0.1)'),
            angularaxis=dict(tickfont=dict(size=11, color='#3A5A78'), gridcolor='rgba(0, 115, 203, 0.1)'),
            bgcolor='#FFFFFF'
        ),
        height=320,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(color="#3A5A78"))
    )
    return fig

def draw_comparison(products, field_name):
    # 柱状图对比
    names = [f"{c.Model[:8]}.." for c in products]
    full_names = [f"{c.Brand} {c.Model}" for c in products]
    
    scores = []
    for c in products:
        val = getattr(c, field_name, 0)
        if val is None: val = 0
        scores.append(val)
        
    fig = go.Figure(data=[go.Bar(
        x=names,
        y=scores,
        marker_color=THEME_COLORS[:len(products)],
        text=scores,
        textposition='auto',
        hovertext=full_names,
        hovertemplate='<b>%{hovertext}</b><br>得分: %{y}<extra></extra>'
    )])

    safe_name = field_name.replace('_Score', '评分').replace('_', ' ')
    
    _apply_theme(fig)
    fig.update_layout(
        title=dict(text=f"{safe_name} 对比", font=dict(size=14, color="#3A5A78"), x=0.5, y=0.95),
        height=300,
        yaxis=dict(range=[0, 110])
    )
    return fig

def draw_multi_dimension_compare(products, dimensions=None):
    # 多维能力对比
    if not dimensions:
        dims_conf = [('便携性', 'Portability_Score'), ('低光画质', 'LowLight_Score'), ('视频能力', 'Video_Score')]
    else:
        dims_conf = dimensions
        
    labels = [d[0] for d in dims_conf]
    fields = [d[1] for d in dims_conf]
    
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
            marker_color=THEME_COLORS[i % len(THEME_COLORS)],
            text=scores,
            textposition='auto',
            hovertemplate='<b>%{x}</b>: %{y}<br>(%{data.name})<extra></extra>'
        ))

    _apply_theme(fig)
    
    fig.update_layout(
        title=dict(text="核心能力多维对比", font=dict(size=14, color="#3A5A78"), x=0.5, y=0.95),
        barmode='group',
        height=320,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(color="#3A5A78")),
        yaxis=dict(title='评分', range=[0, 110])
    )
    return fig
