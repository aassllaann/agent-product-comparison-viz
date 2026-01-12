"""
相机数据迁移脚本

将现有 PostgreSQL 中的相机数据迁移到 MongoDB。
"""

import sys
from datetime import datetime
from models import SessionLocal, Camera
from mongo_db import MongoProductDB
import config


def migrate_cameras_to_mongo():
    """
    将 PostgreSQL 中的相机数据迁移到 MongoDB
    """
    print("=" * 50)
    print("开始迁移相机数据到 MongoDB")
    print("=" * 50)
    
    # 1. 连接数据库
    pg_session = SessionLocal()
    mongo_db = MongoProductDB(
        uri=config.MONGO_URI,
        db_name=config.MONGO_DB_NAME
    )
    
    try:
        # 2. 从 PostgreSQL 读取所有相机数据
        cameras = pg_session.query(Camera).all()
        print(f"从 PostgreSQL 读取到 {len(cameras)} 条相机数据")
        
        if not cameras:
            print("没有数据需要迁移")
            return
        
        # 3. 转换为 MongoDB 文档格式
        mongo_docs = []
        for cam in cameras:
            doc = {
                "category": "camera",
                "brand": cam.Brand,
                "model": cam.Model,
                "price": cam.Price,
                "source": "postgresql_migration",
                "source_id": f"camera_{cam.id}",
                "url": "",
                "image_url": cam.image_file or "",
                "scores": {
                    "Portability_Score": cam.Portability_Score or 0,
                    "LowLight_Score": cam.LowLight_Score or 0,
                    "Video_Score": cam.Video_Score or 0,
                },
                "specs": {
                    "year": cam.Year,
                    "total_megapixels": cam.Total_megapixels,
                    "sensor_type": cam.Sensor_type,
                    "weight_g": cam.Weight_g,
                    "max_iso": cam.Max_ISO,
                    "screen_size_in": cam.Screen_Size_in,
                    "supports_4k": cam.Supports_4K,
                    "alias": cam.Alias,
                },
                "crawled_at": datetime.now(),
                "updated_at": datetime.now()
            }
            mongo_docs.append(doc)
        
        # 4. 先清空现有相机数据（避免重复）
        deleted_count = mongo_db.delete_by_category("camera")
        if deleted_count > 0:
            print(f"已清空 MongoDB 中现有的 {deleted_count} 条相机数据")
        
        # 5. 批量插入到 MongoDB
        inserted_ids = mongo_db.insert_products(mongo_docs)
        print(f"成功插入 {len(inserted_ids)} 条相机数据到 MongoDB")
        
        # 6. 创建索引
        mongo_db.ensure_indexes()
        print("已创建 MongoDB 索引")
        
        # 7. 验证
        count = mongo_db.count_by_category("camera")
        print(f"\n验证：MongoDB 中现有 {count} 条相机数据")
        
        # 显示部分数据
        sample = mongo_db.find_by_category("camera", limit=3)
        print("\n示例数据：")
        for doc in sample:
            print(f"  - {doc['brand']} {doc['model']}: ¥{doc['price']}")
        
        print("\n" + "=" * 50)
        print("迁移完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"迁移失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        pg_session.close()
        mongo_db.close()
    
    return True


def verify_mongo_data():
    """验证 MongoDB 中的数据"""
    mongo_db = MongoProductDB(
        uri=config.MONGO_URI,
        db_name=config.MONGO_DB_NAME
    )
    
    try:
        # 统计
        categories = mongo_db.get_categories()
        print(f"已有品类: {categories}")
        
        for cat in categories:
            count = mongo_db.count_by_category(cat)
            print(f"  {cat}: {count} 条")
        
        # 按价格查询测试
        print("\n测试查询：价格 5000-10000 的相机（按 LowLight_Score 排序）")
        results = mongo_db.find_by_price_range(
            category="camera",
            min_price=5000,
            max_price=10000,
            sort_by="LowLight_Score",
            limit=5
        )
        for doc in results:
            score = doc.get('scores', {}).get('LowLight_Score', 0)
            print(f"  - {doc['brand']} {doc['model']}: ¥{doc['price']} (低光评分: {score})")
            
    finally:
        mongo_db.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        verify_mongo_data()
    else:
        migrate_cameras_to_mongo()
