import pandas as pd
import random
from models import engine, Base, SessionLocal, Camera, Phone, Laptop, Headphone, Tablet
import os

def load_generic_data(file_name, model_class, table_name_cn):
    """通用的数据加载函数"""
    csv_path = os.path.join("data", file_name)
    
    # 1. 检查文件
    if not os.path.exists(csv_path):
        print(f"❌ [{table_name_cn}] 跳过：未找到文件 {csv_path}")
        return

    # 3. 读取 CSV 数据
    print(f"正在读取 {table_name_cn} 数据：{csv_path}")
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        df = df.where(pd.notnull(df), None)
    except Exception as e:
         try:
            df = pd.read_csv(csv_path, encoding='gbk')
            df = df.where(pd.notnull(df), None)
         except:
            print(f"❌ 读取 CSV 失败: {e}")
            return

    # 4. 写入数据库
    session = SessionLocal()
    try:
        print(f"[{table_name_cn}] 正在导入 {len(df)} 条数据...")
        
        # 获取模型的所有字段名
        model_columns = model_class.__table__.columns.keys()
        
        for _, row in df.iterrows():
            # 动态构建参数字典
            data = {}
            for col in model_columns:
                if col == 'id': continue
                
                # 尝试从 CSV 获取对应列
                val = row.get(col)
                
                # 特殊处理 Alias
                if col == 'Alias' and pd.isna(val):
                     val = row.get('Also known as')
                
                # 简单类型转换
                if pd.isna(val) or val == '':
                    val = None
                
                # Boolean转换
                if val is not None and model_class.__table__.columns[col].type.python_type == bool:
                    val = str(val).lower() in ['true', '1', 'yes', '是']
                
                data[col] = val
            
            # 创建实例
            item = model_class(**data)
            session.add(item)
            
        session.commit()
        print(f"✅ [{table_name_cn}] 导入成功！")
    except Exception as e:
        session.rollback()
        print(f"❌ [{table_name_cn}] 导入失败，原因：{e}")
    finally:
        session.close()

def load_all_data():
    # 重建所有表
    print("正在重建数据库表结构...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    # 加载各品类
    load_generic_data("camera_data_clean4.csv", Camera, "相机")
    load_generic_data("phone_data.csv", Phone, "手机")
    load_generic_data("laptop_data.csv", Laptop, "笔记本")
    load_generic_data("headphone_data.csv", Headphone, "耳机")
    load_generic_data("tablet_data.csv", Tablet, "平板")

if __name__ == "__main__":
    load_all_data()