"""
批量加载CSV数据到SQLite数据库

支持5个新增品类的数据导入
"""
import pandas as pd
from models import engine, Base, Smartwatch, BluetoothSpeaker, Monitor, GamingConsole, GPU
from sqlalchemy.orm import sessionmaker

# 创建会话
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def load_csv_to_db(csv_path, model_class, table_name):
    """
    将CSV数据加载到数据库
    
    Args:
        csv_path: CSV文件路径
        model_class: ORM模型类
        table_name: 表名
    """
    print(f"\n正在加载 {table_name}...")
    
    # 读取CSV
    df = pd.read_csv(csv_path)
    print(f"读取到 {len(df)} 条记录")
    
    # 创建数据库会话
    db = SessionLocal()
    
    try:
        #  插入数据
        inserted = 0
        for _, row in df.iterrows():
            # 将DataFrame行转为字典，过滤掉NaN值
            data = row.to_dict()
            data = {k: v for k, v in data.items() if pd.notna(v)}
            
            # 创建模型实例
            try:
                # 特殊处理 GPU 的 3DMark_Score 字段映射
                if model_class == GPU and '3DMark_Score' in data:
                    data['ThreeDMark_Score'] = data.pop('3DMark_Score')
                
                item = model_class(**data)
                db.add(item)
                inserted += 1
            except Exception as e:
                print(f"插入失败: {data.get('Brand', '')} {data.get('Model', '')} - {e}")
        
        # 提交
        db.commit()
        print(f"✅ 成功插入 {inserted} 条记录到 {table_name}")
        
    except Exception as e:
        print(f"❌ 加载 {table_name} 失败: {e}")
        db.rollback()
    finally:
        db.close()


def main():
    """主函数"""
    print("=" * 60)
    print("开始导入新品类数据")
    print("=" * 60)
    
    # 创建所有表
    try:
        # 强制删除已有的游戏主机表以应用新的字段类型 (Integer -> String)
        GPU.__table__.drop(engine, checkfirst=True) # 显卡也重导一次确保干净
        GamingConsole.__table__.drop(engine, checkfirst=True)
    except:
        pass
        
    Base.metadata.create_all(bind=engine)
    print("✅ 数据库表创建/重置完成\n")
    
    # 定义要加载的数据
    datasets = [
        # ("data/smartwatch_data.csv", Smartwatch, "smartwatches"),
        # ("data/bluetooth_speaker_data.csv", BluetoothSpeaker, "bluetooth_speakers"),
        # ("data/monitor_data.csv", Monitor, "monitors"),
        ("data/gaming_console_data.csv", GamingConsole, "gaming_consoles"),
        ("data/gpu_data.csv", GPU, "gpus"),
    ]
    
    # 逐个加载
    for csv_path, model_class, table_name in datasets:
        try:
            load_csv_to_db(csv_path, model_class, table_name)
        except Exception as e:
            print(f"❌ 加载 {csv_path} 时出错: {e}")
    
    print("\n" + "=" * 60)
    print("数据导入完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
