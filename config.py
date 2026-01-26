# 推荐：用环境变量管理敏感信息和配置
import os

# PostgreSQL 配置（保留用于相机数据兼容）
DB_URI = os.getenv("DB_URI", "postgresql://postgres:727304@localhost:5432/camera_db")

# MongoDB 配置（多品类商品数据）
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "product_recommendation")

# LLM 配置（API Key、模型名）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-a2c6636594fa497f9b6053b93ea5ff8d")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v3.2")

# 淘宝（OneBound）API 配置
# 京东联盟 API 配置
JD_APP_KEY = os.getenv("JD_APP_KEY", "666ac91c8c065ed90ed854276d2c0c3f")
JD_APP_SECRET = os.getenv("JD_APP_SECRET", "18265b805c584f78869d38217b765da8")
JD_API_URL = "https://api.jd.com/routerjson"

# --- 其他配置 ---
CHART_DIR = "charts"

# --- 品类配置 ---
# Tier 1 热门品类（预置数据）
TIER1_CATEGORIES = ["camera", "phone", "headphone", "laptop"]
# 每个品类预置商品数量
TIER1_PRODUCT_LIMIT = 100