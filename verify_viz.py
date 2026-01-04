import visualizer
from models import Camera
import plotly.graph_objects as go
import os

# Create dummy camera data
cam = Camera(
    id=1, Brand="Test", Model="Cam", Price=5000,
    Portability_Score=80, LowLight_Score=70, Video_Score=60, Max_ISO=12800
)

print("Testing draw_radar...")
fig1 = visualizer.draw_radar(cam)
print(f"Type: {type(fig1)}")
if not isinstance(fig1, go.Figure):
    raise ValueError("draw_radar did not return a Plotly Figure")
print("draw_radar passed.")

print("Testing draw_comparison...")
fig2 = visualizer.draw_comparison([cam], "Video_Score")
print(f"Type: {type(fig2)}")
if not isinstance(fig2, go.Figure):
    raise ValueError("draw_comparison did not return a Plotly Figure")
print("draw_comparison passed.")

print("Testing draw_price_performance...")
fig3 = visualizer.draw_price_performance([cam], [cam])
print(f"Type: {type(fig3)}")
if not isinstance(fig3, go.Figure):
    raise ValueError("draw_price_performance did not return a Plotly Figure")
print("draw_price_performance passed.")

print("All visualization tests passed!")
