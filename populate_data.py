"""
预置热门品类数据脚本

从京东/模拟数据源获取 Tier 1 品类的热门商品并存入 MongoDB。
"""

import sys
from datetime import datetime
from mongo_db import MongoProductDB
from scrapers import JDScraper
import config


def populate_category(category: str, limit: int = 100):
    """
    为指定品类预置热门商品数据
    
    Args:
        category: 品类标识
        limit: 商品数量
    """
    print(f"\n{'='*50}")
    print(f"正在预置 {category} 品类数据...")
    print(f"{'='*50}")
    
    # 1. 初始化爬虫和数据库
    scraper = JDScraper(category)
    mongo_db = MongoProductDB(
        uri=config.MONGO_URI,
        db_name=config.MONGO_DB_NAME
    )
    
    try:
        # 2. 获取热门商品
        products = scraper.get_hot_products(limit=limit)
        print(f"获取到 {len(products)} 条商品数据")
        
        if not products:
            print(f"警告：未获取到 {category} 商品数据")
            return 0
        
        # 3. 转换为 MongoDB 文档并插入
        inserted_count = 0
        for product in products:
            doc = product.to_mongo_doc()
            try:
                mongo_db.upsert_product(doc)
                inserted_count += 1
            except Exception as e:
                print(f"插入失败: {product.brand} {product.model} - {e}")
        
        print(f"成功插入/更新 {inserted_count} 条 {category} 商品")
        
        # 4. 验证
        count = mongo_db.count_by_category(category)
        print(f"MongoDB 中 {category} 品类共有 {count} 条数据")
        
        return inserted_count
        
    except Exception as e:
        print(f"预置 {category} 数据失败: {e}")
        import traceback
        traceback.print_exc()
        return 0
    finally:
        mongo_db.close()


def populate_all_tier1():
    """预置所有 Tier 1 品类数据"""
    print("=" * 60)
    print("开始预置 Tier 1 热门品类数据")
    print(f"品类: {config.TIER1_CATEGORIES}")
    print(f"每品类数量: {config.TIER1_PRODUCT_LIMIT}")
    print("=" * 60)
    
    results = {}
    
    for category in config.TIER1_CATEGORIES:
        count = populate_category(category, config.TIER1_PRODUCT_LIMIT)
        results[category] = count
    
    # 汇总报告
    print("\n" + "=" * 60)
    print("预置完成汇总")
    print("=" * 60)
    
    mongo_db = MongoProductDB(
        uri=config.MONGO_URI,
        db_name=config.MONGO_DB_NAME
    )
    
    try:
        for category in config.TIER1_CATEGORIES:
            count = mongo_db.count_by_category(category)
            status = "✅" if count >= config.TIER1_PRODUCT_LIMIT * 0.8 else "⚠️"
            print(f"  {status} {category}: {count} 条")
    finally:
        mongo_db.close()
    
    print("\n提示：当前使用模拟数据。生产环境需实现真实爬虫。")


def show_sample_data(category: str = None):
    """显示示例数据"""
    mongo_db = MongoProductDB(
        uri=config.MONGO_URI,
        db_name=config.MONGO_DB_NAME
    )
    
    try:
        categories = [category] if category else config.TIER1_CATEGORIES
        
        for cat in categories:
            print(f"\n{'='*40}")
            print(f"{cat} 品类示例数据")
            print(f"{'='*40}")
            
            docs = mongo_db.find_by_category(cat, limit=5)
            
            if not docs:
                print("  (无数据)")
                continue
            
            for doc in docs:
                scores = doc.get('scores', {})
                main_score = list(scores.values())[0] if scores else 0
                print(f"  {doc['brand']} {doc['model']}")
                print(f"    价格: ¥{doc['price']}")
                print(f"    评分: {scores}")
    finally:
        mongo_db.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "show":
            category = sys.argv[2] if len(sys.argv) > 2 else None
            show_sample_data(category)
        elif cmd in config.TIER1_CATEGORIES:
            populate_category(cmd, config.TIER1_PRODUCT_LIMIT)
        else:
            print(f"未知命令: {cmd}")
            print("用法:")
            print("  python populate_data.py         # 预置所有 Tier 1 品类")
            print("  python populate_data.py camera  # 只预置相机品类")
            print("  python populate_data.py show    # 显示示例数据")
    else:
        populate_all_tier1()
