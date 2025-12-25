import matplotlib.pyplot as plt
import numpy as np
import os

# 样式美化
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def _ensure_dir():
    if not os.path.exists("charts"):
        os.makedirs("charts")

def draw_radar(camera):
    _ensure_dir()
    labels = ['便携', '画质', '视频', '高感', '操控']
    values = [camera.Portability_Score, camera.LowLight_Score, camera.Video_Score, 
              min(camera.Max_ISO/25600*100, 100), 85]
    
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='#008080', alpha=0.3)
    ax.plot(angles, values, color='#008080', linewidth=1.5)
    ax.set_yticklabels([])
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)
    
    path = f"charts/radar_{camera.id}.png"
    plt.savefig(path, transparent=True, bbox_inches='tight')
    plt.close()
    return path

def draw_comparison(cameras, field_name):
    _ensure_dir()
    names = [f"{c.Model[:6]}.." for c in cameras]
    scores = [getattr(c, field_name) for c in cameras]
    
    fig, ax = plt.subplots(figsize=(3, 2.5))
    ax.bar(names, scores, color=['#008080', '#FF8C00', '#4682B4'])
    ax.set_title(f"{field_name} 对比", fontsize=9)
    ax.tick_params(labelsize=7)
    
    path = "charts/compare_bar.png"
    plt.savefig(path, transparent=True, bbox_inches='tight')
    plt.close()
    return path

def draw_price_performance(cameras, all_cameras):
    _ensure_dir()
    all_prices = [c.Price for c in all_cameras]
    all_scores = [(c.LowLight_Score + c.Video_Score)/2 for c in all_cameras]
    
    fig, ax = plt.subplots(figsize=(3, 2.5))
    ax.scatter(all_prices, all_scores, c='gray', alpha=0.1, s=5)
    
    rec_prices = [c.Price for c in cameras]
    rec_scores = [(c.LowLight_Score + c.Video_Score)/2 for c in cameras]
    ax.scatter(rec_prices, rec_scores, c='#E63946', s=40, edgecolors='white')
    ax.set_title("性价比分布", fontsize=9)
    ax.tick_params(labelsize=7)
    
    path = "charts/price_perf.png"
    plt.savefig(path, transparent=True, bbox_inches='tight')
    plt.close()
    return path