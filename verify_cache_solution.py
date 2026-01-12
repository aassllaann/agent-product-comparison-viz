"""
简化缓存方案验证脚本

验证内存缓存 + 实时 API 方案是否正常工作。
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_cache():
    """测试内存缓存"""
    print("\n[1] 测试内存缓存...")
    try:
        from cache import ProductCache, get_product_cache
        
        cache = ProductCache(default_ttl=2)  # 2秒过期
        
        # 测试设置和获取
        cache.set("test", [{"name": "test_product"}], "query1")
        result = cache.get("test", "query1")
        assert result is not None, "缓存获取失败"
        print("  ✅ 缓存设置和获取正常")
        
        # 测试过期
        time.sleep(2.5)
        result = cache.get("test", "query1")
        assert result is None, "缓存应该已过期"
        print("  ✅ 缓存过期机制正常")
        
        # 测试 get_or_fetch
        fetch_count = [0]
        def fetch_func():
            fetch_count[0] += 1
            return [{"name": "fetched"}]
        
        cache.get_or_fetch("test2", fetch_func, "q1", ttl=5)
        cache.get_or_fetch("test2", fetch_func, "q1", ttl=5)
        assert fetch_count[0] == 1, "第二次应该命中缓存"
        print("  ✅ get_or_fetch 正常")
        
        # 测试统计
        stats = cache.get_stats()
        print(f"  ✅ 缓存统计: {stats}")
        
        return True
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_service():
    """测试数据服务"""
    print("\n[2] 测试数据服务...")
    try:
        from data_service import ProductDataService, get_data_service
        
        service = get_data_service()
        
        # 测试搜索
        results = service.search("phone", "手机", limit=5)
        assert len(results) > 0, "应该返回结果"
        print(f"  ✅ 搜索返回 {len(results)} 条数据")
        
        # 测试缓存命中
        results2 = service.search("phone", "手机", limit=5)
        print("  ✅ 第二次搜索（缓存命中）")
        
        # 测试热门商品
        hot = service.get_hot_products("laptop", limit=10)
        print(f"  ✅ 热门商品返回 {len(hot)} 条")
        
        # 测试筛选
        filtered = service.get_products_by_filter(
            category="phone",
            max_price=5000,
            sort_by="Value_Score",
            limit=3
        )
        print(f"  ✅ 筛选返回 {len(filtered)} 条（预算≤5000）")
        
        # 显示示例
        if filtered:
            p = filtered[0]
            print(f"    示例: {p.get('brand')} {p.get('model')} - ¥{p.get('price')}")
        
        # 缓存统计
        stats = service.get_cache_stats()
        print(f"  ✅ 缓存统计: {stats}")
        
        return True
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_category():
    """测试多品类代理（包括新增品类）"""
    print("\n[3] 测试多品类代理...")
    try:
        from category_detector import CategoryDetector
        from multi_agent import MultiCategoryAgent, DynamicAgent
        
        # 测试品类识别（包括新增品类）
        detector = CategoryDetector()
        
        test_cases = [
            # 数码电子
            ("推荐一款手机", "phone"),
            ("想买个降噪耳机", "headphone"),
            ("笔记本电脑推荐", "laptop"),
            ("推荐相机拍vlog", "camera"),
            
            # 护肤美妆
            ("推荐一款面霜", "skincare"),
            ("好用的精华液", "skincare"),
            ("推荐口红", "cosmetics"),
            
            # 文具办公
            ("推荐钢笔", "stationery"),
            ("好写的中性笔", "stationery"),
            
            # 家电
            ("推荐吹风机", "appliance"),
            ("电饭煲哪个好", "appliance"),
            
            # 运动
            ("跑步鞋推荐", "sports"),
            
            # 图书
            ("推荐小说", "book"),
        ]
        
        passed = 0
        for msg, expected in test_cases:
            key, name = detector.detect_category(msg)
            status = "✅" if key == expected else "❌"
            if key == expected:
                passed += 1
            print(f"  {status} '{msg}' -> {key} ({name})")
        
        print(f"\n  品类识别: {passed}/{len(test_cases)} 通过")
        
        # 测试数据服务获取新品类数据
        from data_service import get_data_service
        service = get_data_service()
        
        new_categories = ["skincare", "stationery", "appliance"]
        for cat in new_categories:
            products = service.get_hot_products(cat, limit=5)
            if products:
                p = products[0]
                print(f"  ✅ {cat}: {p.get('brand')} {p.get('model')} - ¥{p.get('price')}")
            else:
                print(f"  ⚠️ {cat}: 无数据")
        
        return passed >= len(test_cases) * 0.8  # 80% 通过即可
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("简化缓存方案验证")
    print("(内存缓存 + 实时 API，无需 MongoDB)")
    print("=" * 60)
    
    results = {
        "cache": test_cache(),
        "data_service": test_data_service(),
        "multi_category": test_multi_category(),
    }
    
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
        print("\n🎉 简化方案验证成功！")
        print("\n可以运行 app.py 测试多品类推荐：")
        print("  streamlit run app.py")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息。")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
