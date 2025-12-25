# 推荐：用环境变量管理敏感信息和配置
import os

# PostgreSQL 配置
DB_URI = os.getenv("DB_URI", "postgresql://postgres:727304@localhost:5432/camera_db")

# LLM 配置（API Key、模型名）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-a2c6636594fa497f9b6053b93ea5ff8d")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v3.2")

# --- 其他配置 ---
CHART_DIR = "charts"