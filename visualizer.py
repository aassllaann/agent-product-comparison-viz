import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def draw_radar(camera):
    if not os.path.exists("charts"): os.makedirs("charts")
    labels = ['便携性', '弱光性能', '视频能力', '高感潜力', '屏幕体验']
    # 模拟五个维度的 0-100 分
    values = [camera.Portability_Score, camera.LowLight_Score, camera.Video_Score, 
              min(camera.Max_ISO/25600*100, 100), min(camera.Screen_Size_in/3.5*100, 100)]
    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist() + [0]
    
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='teal', alpha=0.3)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    path = f"charts/radar_{camera.id}.png"
    plt.savefig(path, bbox_inches='tight'); plt.close()
    return path

def draw_comparison(cameras, field):
    names = [c.Model for c in cameras]
    scores = [getattr(c, field) for c in cameras]
    plt.figure(figsize=(6, 4))
    plt.bar(names, scores, color='skyblue')
    plt.ylim(0, 110); plt.title("性能评分对比")
    path = "charts/compare_bar.png"
    plt.savefig(path); plt.close()
    return path