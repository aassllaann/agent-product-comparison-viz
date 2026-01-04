import visualizer
from models import Camera
import plotly.graph_objects as go
import os

# Create dummy camera data
cams = [
    Camera(id=1, Brand="Test", Model="Cam1", Price=5000, Portability_Score=80, LowLight_Score=70, Video_Score=60, Max_ISO=12800),
    Camera(id=2, Brand="Test", Model="Cam2", Price=6000, Portability_Score=70, LowLight_Score=80, Video_Score=70, Max_ISO=25600),
    Camera(id=3, Brand="Test", Model="Cam3", Price=7000, Portability_Score=60, LowLight_Score=90, Video_Score=80, Max_ISO=6400)
]

print("Testing draw_radar (multi)...")
fig1 = visualizer.draw_radar(cams)
print(f"Type: {type(fig1)}")
# Check if it has 3 traces + background or just 3 traces?
# implementation has loop for cams, so should have 3 traces.
print(f"Num traces: {len(fig1.data)}")
if len(fig1.data) != 3:
    raise ValueError(f"Expected 3 traces, got {len(fig1.data)}")
print("draw_radar passed.")

print("Testing draw_price_performance...")
fig3 = visualizer.draw_price_performance(cams, cams) # passing cams as all_cams for simplicity
print(f"Type: {type(fig3)}")
# Implementation: Background trace + 3 recommendation traces = 4 traces
print(f"Num traces: {len(fig3.data)}")
if len(fig3.data) != 4:
    raise ValueError(f"Expected 4 traces, got {len(fig3.data)}")
print("draw_price_performance passed.")
