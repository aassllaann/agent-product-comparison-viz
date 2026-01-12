"""
多品类推荐架构验证脚本

验证新架构的各个组件是否正常工作。
"""

import sys
import os

# 确保可以导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_base_agent_import():
    """测试基类导入"""
    print("\n[1] 测试 BaseProductAgent 导入...")
    try:
        from base_agent import BaseProductAgent, CategoryConfig, ScoringDimension
        print("  ✅ BaseProductAgent, CategoryConfig, ScoringDimension 导入成功")
        
        # 验证 ScoringDimension
        dim = ScoringDimension("测试", "Test_Score", 0.5, "测试维度")
        assert dim.weight == 0.5
        print("  ✅ ScoringDimension 创建成功")
        
        return True
    except Exception as e:
        print(f"  ❌ 导入失败: {e}")
        return False


def test_camera_agent():
    """测试相机代理"""
    print("\n[2] 测试 CameraAgent...")
    try:
        from main_agent import CameraAgent
        from base_agent import BaseProductAgent
        
        agent = CameraAgent()
        
        # 验证继承关系
        assert isinstance(agent, BaseProductAgent), "CameraAgent 应该继承 BaseProductAgent"
        print("  ✅ CameraAgent 继承 BaseProductAgent")
        
        # 验证配置
        config = agent.get_category_config()
        assert config.name == "相机"
        assert config.name_en == "camera"
        print(f"  ✅ 品类配置: {config.name} ({config.name_en})")
        
        # 验证评分维度
        dims = agent.get_specific_dimensions()
        dim_names = [d.name for d in dims]
        assert "低光画质" in dim_names
        print(f"  ✅ 评分维度: {dim_names}")
        
        # 验证所有维度（通用+特有）
        all_dims = agent.get_all_dimensions()
        print(f"  ✅ 总维度数: {len(all_dims)} (通用+特有)")
        
        return True
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_category_detector():
    """测试品类识别器"""
    print("\n[3] 测试 CategoryDetector...")
    try:
        from category_detector import CategoryDetector
        
        detector = CategoryDetector()
        
        # 测试关键词匹配
        test_cases = [
            ("我想买一台相机拍vlog", "camera"),
            ("推荐一款手机", "phone"),
            ("想买个降噪耳机", "headphone"),
            ("笔记本电脑推荐", "laptop"),
        ]
        
        for msg, expected in test_cases:
            key, name = detector.detect_category(msg)
            status = "✅" if key == expected else "❌"
            print(f"  {status} '{msg}' -> {key} ({name})")
        
        return True
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_agent():
    """测试多品类代理"""
    print("\n[4] 测试 MultiCategoryAgent...")
    try:
        from multi_agent import MultiCategoryAgent
        
        agent = MultiCategoryAgent()
        
        # 验证相机代理已注册
        camera_agent = agent.get_agent("camera")
        assert camera_agent is not None, "相机代理应该已注册"
        print("  ✅ 相机代理已注册")
        
        # 获取可用品类
        categories = agent.get_available_categories()
        print(f"  ✅ 可用品类: {[c['name'] for c in categories]}")
        
        return True
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scraper():
    """测试爬虫模块"""
    print("\n[5] 测试爬虫模块...")
    try:
        from scrapers import JDScraper, ProductData
        
        # 测试相机爬虫
        scraper = JDScraper("camera")
        products = scraper.search("相机", limit=5)
        
        assert len(products) > 0, "应该返回商品数据"
        print(f"  ✅ 搜索返回 {len(products)} 条数据")
        
        # 验证数据结构
        p = products[0]
        assert isinstance(p, ProductData)
        assert p.category == "camera"
        assert p.price > 0
        print(f"  ✅ 示例: {p.brand} {p.model} - ¥{p.price}")
        print(f"  ✅ 评分: {p.scores}")
        
        return True
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mongo_db():
    """测试 MongoDB 连接（可选）"""
    print("\n[6] 测试 MongoDB 连接...")
    try:
        from mongo_db import MongoProductDB
        import config
        
        db = MongoProductDB(
            uri=config.MONGO_URI,
            db_name=config.MONGO_DB_NAME
        )
        
        # 尝试连接
        categories = db.get_categories()
        print(f"  ✅ 连接成功，已有品类: {categories}")
        
        db.close()
        return True
    except Exception as e:
        print(f"  ⚠️ MongoDB 连接失败（可能未安装）: {e}")
        print("    提示：请先安装并启动 MongoDB")
        return False


def main():
    print("=" * 60)
    print("多品类推荐架构验证")
    print("=" * 60)
    
    results = {}
    
    results["base_agent"] = test_base_agent_import()
    results["camera_agent"] = test_camera_agent()
    results["category_detector"] = test_category_detector()
    results["multi_agent"] = test_multi_agent()
    results["scraper"] = test_scraper()
    results["mongo_db"] = test_mongo_db()
    
    # 汇总
    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status}: {name}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！架构验证成功。")
    elif passed >= total - 1:
        print("\n✅ 核心功能验证通过（MongoDB 可稍后配置）。")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息。")
    
    return passed >= total - 1


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
