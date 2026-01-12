"""
商品数据服务

整合缓存和爬虫，提供统一的数据获取接口。
所有品类都通过实时 API + 内存缓存获取数据。
"""

from typing import Dict, List, Optional, Any
from cache import ProductCache, get_product_cache
from scrapers import JDScraper, ProductData
from scrapers.base_scraper import BaseScraper


class ProductDataService:
    """
    商品数据服务
    
    提供统一的商品数据获取接口：
    1. 先查缓存
    2. 缓存未命中则调用爬虫
    3. 结果存入缓存
    """
    
    # 缓存过期时间配置（秒）
    CACHE_TTL = {
        "default": 600,      # 默认 10 分钟
        "search": 300,       # 搜索结果 5 分钟
        "hot": 1800,         # 热门榜单 30 分钟
    }
    
    def __init__(self, cache: ProductCache = None):
        """
        Args:
            cache: 缓存实例，默认使用全局缓存
        """
        self.cache = cache or get_product_cache()
        self._scrapers: Dict[str, BaseScraper] = {}
    
    def _get_scraper(self, category: str) -> BaseScraper:
        """获取或创建品类爬虫"""
        if category not in self._scrapers:
            # 目前所有品类都使用 JDScraper
            self._scrapers[category] = JDScraper(category)
        return self._scrapers[category]
    
    def search(
        self, 
        category: str, 
        query: str, 
        limit: int = 50,
        use_cache: bool = True
    ) -> List[dict]:
        """
        搜索商品
        
        Args:
            category: 品类标识
            query: 搜索关键词
            limit: 返回数量
            use_cache: 是否使用缓存
            
        Returns:
            商品数据列表（dict 格式）
        """
        cache_key = f"search:{query}:{limit}"
        
        # 尝试从缓存获取
        if use_cache:
            cached = self.cache.get(category, cache_key)
            if cached is not None:
                print(f"[DataService] 缓存命中: {category}/{query}")
                return cached
        
        # 调用爬虫
        print(f"[DataService] 调用 API: {category}/{query}")
        scraper = self._get_scraper(category)
        products = scraper.search(query, limit)
        
        # 转换为 dict 并缓存
        result = [self._product_to_dict(p) for p in products]
        
        if result:
            self.cache.set(
                category, 
                result, 
                cache_key, 
                self.CACHE_TTL["search"]
            )
        
        return result
    
    def get_hot_products(
        self, 
        category: str, 
        limit: int = 100,
        use_cache: bool = True
    ) -> List[dict]:
        """
        获取热门商品
        
        Args:
            category: 品类标识
            limit: 返回数量
            use_cache: 是否使用缓存
            
        Returns:
            热门商品列表
        """
        cache_key = f"hot:{limit}"
        
        if use_cache:
            cached = self.cache.get(category, cache_key)
            if cached is not None:
                print(f"[DataService] 热门缓存命中: {category}")
                return cached
        
        print(f"[DataService] 获取热门商品: {category}")
        scraper = self._get_scraper(category)
        products = scraper.get_hot_products(limit)
        
        result = [self._product_to_dict(p) for p in products]
        
        if result:
            self.cache.set(
                category, 
                result, 
                cache_key, 
                self.CACHE_TTL["hot"]
            )
        
        return result
    
    def get_products_by_filter(
        self,
        category: str,
        max_price: float = None,
        min_price: float = None,
        sort_by: str = None,
        limit: int = 50
    ) -> List[dict]:
        """
        按条件筛选商品
        
        先获取热门商品，然后在内存中筛选。
        """
        # 获取更多商品用于筛选
        products = self.get_hot_products(category, limit=200)
        
        # 价格筛选
        if max_price is not None:
            products = [p for p in products if p.get("price", 0) <= max_price]
        if min_price is not None:
            products = [p for p in products if p.get("price", 0) >= min_price]
        
        # 排序
        if sort_by and products:
            products.sort(
                key=lambda x: x.get("scores", {}).get(sort_by, 0),
                reverse=True
            )
        
        return products[:limit]
    
    def refresh_category(self, category: str):
        """
        刷新某品类的缓存
        
        强制重新获取数据。
        """
        print(f"[DataService] 刷新品类缓存: {category}")
        self.cache.invalidate(category)
        # 预热缓存
        self.get_hot_products(category, use_cache=False)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return self.cache.get_stats()
    
    def _product_to_dict(self, product: ProductData) -> dict:
        """将 ProductData 转换为 dict"""
        return {
            "category": product.category,
            "brand": product.brand,
            "model": product.model,
            "price": product.price,
            "source": product.source,
            "source_id": product.source_id,
            "url": product.url,
            "image_url": product.image_url,
            "scores": product.scores,
            "specs": product.specs,
        }


# 全局服务实例
_data_service: Optional[ProductDataService] = None


def get_data_service() -> ProductDataService:
    """获取全局数据服务实例"""
    global _data_service
    if _data_service is None:
        _data_service = ProductDataService()
    return _data_service
