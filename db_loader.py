import pandas as pd
import random
from models import engine, Base, SessionLocal, Camera
import os

# 配置 CSV 文件路径
CSV_PATH = os.path.join("data", "camera_data_clean4.csv")

def load_data():
    # 1. 检查文件是否存在
    if not os.path.exists(CSV_PATH):
        print(f"❌ 错误：在路径 {CSV_PATH} 未找到 CSV 文件。")
        return

    # 2. 删除并重建数据库表结构
    print("正在重建数据库表结构...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    # 3. 读取 CSV 数据
    print(f"正在读取数据：{CSV_PATH}")
    df = pd.read_csv(CSV_PATH, encoding='utf-8')
    df = df.where(pd.notnull(df), None)

    # 4. 写入数据库
    session = SessionLocal()
    try:
        print(f"正在导入 {len(df)} 条数据到 PostgreSQL...")
        for _, row in df.iterrows():
            # 读取 Alias
            alias = row.get('Alias')
            if pd.isna(alias):
                # 尝试从 'Also known as' 读取 (如果 update_data.py 没有重命名列)
                alias = row.get('Also known as')
            
            # 读取 Price (不再随机生成，除非为空)
            price = row.get('Price')
            if pd.isna(price) or price == 0 or price == '':
                 price = random.randint(30, 250) * 100
            
            camera = Camera(
                Brand=row.get('Brand'),
                Model=row.get('Model'),
                Alias=alias,
                Price=price,
                Year=row.get('Year'),
                image_file=row.get('image_file'),
                Total_megapixels=row.get('Total megapixels'),
                Sensor_type=row.get('Sensor type'),
                Weight_g=row.get('Weight_g'),
                Max_ISO=row.get('Max_ISO'),
                Screen_Size_in=row.get('Screen_Size_in'),
                Supports_4K=bool(row.get('Supports_4K')),
                Portability_Score=row.get('Portability_Score'),
                LowLight_Score=row.get('LowLight_Score'),
                Video_Score=row.get('Video_Score')
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