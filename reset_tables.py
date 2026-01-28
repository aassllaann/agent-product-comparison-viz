from models import engine, Base
from sqlalchemy import text

def reset_tables():
    print("正在重置 GamingConsole 和 GPU 表...")
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS gaming_consoles CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS gpus CASCADE"))
        conn.commit()
    print("表已删除。")

if __name__ == "__main__":
    reset_tables()
