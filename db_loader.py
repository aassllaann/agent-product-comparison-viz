import pandas as pd
import random
from models import engine, Base, SessionLocal, Camera
import os

# 配置 CSV 文件路径
CSV_PATH = os.path.join("data", "camera_data_clean3.csv")

def load_data():
    # 1. 检查文件是否存在
    if not os.path.exists(CSV_PATH):
        print(f"❌ 错误：在路径 {CSV_PATH} 未找到 CSV 文件。")
        return

    # 2. 在数据库中创建表 (如果表已存在则忽略)
    print("正在初始化数据库表结构...")
    Base.metadata.create_all(bind=engine)

    # 3. 读取 CSV 数据
    print(f"正在读取数据：{CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    # 将 Pandas 的 NaN (空值) 转换为 Python 的 None，否则写入数据库会报错
    df = df.where(pd.notnull(df), None)

    # 4. 写入数据库
    session = SessionLocal()
    try:
        # 清空旧数据 (可选，防止重复导入)
        session.query(Camera).delete()
        
        print(f"正在导入 {len(df)} 条数据到 PostgreSQL...")
        
        for _, row in df.iterrows():
            # 💡 关键：由于原始 CSV 可能没有 Price 列，我们在这里随机生成价格用于演示
            # 实际项目中你应该使用真实的价格数据
            simulated_price = random.randint(30, 250) * 100 
            
            camera = Camera(
                Brand=row.get('Brand'),
                Model=row.get('Model'),
                Price=simulated_price,  # 演示用价格
                Weight_g=row.get('Weight_g'),
                Max_ISO=row.get('Max_ISO'),
                Min_Aperture_F=row.get('Min_Aperture_F'),
                Supports_4K=bool(row.get('Supports_4K')),
                Portability_Score=row.get('Portability_Score'),
                LowLight_Score=row.get('LowLight_Score'),
                Video_Score=row.get('Video_Score'),
                Screen_Size_in=row.get('Screen_Size_in')
            )
            session.add(camera)
        
        session.commit()
        print("✅ 数据导入成功！")
        
    except Exception as e:
        session.rollback()
        print(f"❌ 导入失败，原因：{e}")
    finally:
        session.close()

if __name__ == "__main__":
    load_data()