import visualizer
from models import Camera
import plotly.graph_objects as go

# Create dummy camera data
cams = [
    Camera(id=1, Brand="Test", Model="Cam1", Price=5000, Portability_Score=80, LowLight_Score=70, Video_Score=60),
    Camera(id=2, Brand="Test", Model="Cam2", Price=6000, Portability_Score=70, LowLight_Score=80, Video_Score=70),
    Camera(id=3, Brand="Test", Model="Cam3", Price=7000, Portability_Score=60, LowLight_Score=90, Video_Score=80)
]

print("Testing draw_multi_dimension_compare...")
# Note: implementation changed, now takes only cameras list
try:
    fig = visualizer.draw_multi_dimension_compare(cams)
except TypeError as e:
    print(f"Error: {e}")
    print("Likely signature mismatch. Checking visualizer.py...")
    # Rethrow to fail test
    raise e

print(f"Type: {type(fig)}")
# Expecting 3 traces (one bar group per camera)
print(f"Num traces: {len(fig.data)}")
if len(fig.data) != 3:
    raise ValueError(f"Expected 3 traces, got {len(fig.data)}")

# Verifying layout title
if fig.layout.title.text != "核心能力多维对比":
    raise ValueError(f"Unexpected title: {fig.layout.title.text}")

print("draw_multi_dimension_compare passed!")
