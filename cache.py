"""
内存缓存模块

提供带 TTL 的内存缓存，用于缓存实时 API 获取的商品数据。
无需 MongoDB，重启后缓存自动清空。
"""

import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class CacheEntry:
    """缓存条目"""
    data: Any
    created_at: float
    ttl_seconds: float
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        return time.time() - self.created_at > self.ttl_seconds


class ProductCache:
    """
    商品数据内存缓存
    
    特性：
    - 带 TTL 自动过期
    - 线程安全
    - 按品类分组存储
    """
    
    # 默认缓存时间（秒）
    DEFAULT_TTL = 600  # 10 分钟
    
    def __init__(self, default_ttl: float = None):
        """
        Args:
            default_ttl: 默认缓存过期时间（秒）
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = Lock()
        self.default_ttl = default_ttl or self.DEFAULT_TTL
    
    def _make_key(self, category: str, query: str = None) -> str:
        """生成缓存 key"""
        if query:
            return f"{category}:{query.lower()}"
        return f"{category}:__all__"
    
    def get(self, category: str, query: str = None) -> Optional[List[dict]]:
        """
        获取缓存数据
        
        Args:
            category: 品类标识
            query: 搜索关键词（可选）
            
        Returns:
            缓存的商品数据，如果不存在或过期返回 None
        """
        key = self._make_key(category, query)
        
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                return None
            
            if entry.is_expired():
                del self._cache[key]
                return None
            
            return entry.data
    
    def set(
        self, 
        category: str, 
        data: List[dict], 
        query: str = None,
        ttl: float = None
    ):
        """
        设置缓存数据
        
        Args:
            category: 品类标识
            data: 商品数据列表
            query: 搜索关键词（可选）
            ttl: 过期时间（秒），默认使用 default_ttl
        """
        key = self._make_key(category, query)
        
        with self._lock:
            self._cache[key] = CacheEntry(
                data=data,
                created_at=time.time(),
                ttl_seconds=ttl or self.default_ttl
            )
    
    def get_or_fetch(
        self,
        category: str,
        fetch_func: Callable[[], List[dict]],
        query: str = None,
        ttl: float = None
    ) -> List[dict]:
        """
        获取缓存，如果不存在则调用 fetch_func 获取并缓存
        
        Args:
            category: 品类标识
            fetch_func: 获取数据的函数
            query: 搜索关键词（可选）
            ttl: 过期时间（秒）
            
        Returns:
            商品数据列表
        """
        # 先尝试从缓存获取
        cached = self.get(category, query)
        if cached is not None:
            return cached
        
        # 缓存未命中，调用 fetch_func
        data = fetch_func()
        
        # 存入缓存
        if data:
            self.set(category, data, query, ttl)
        
        return data
    
    def invalidate(self, category: str = None, query: str = None):
        """
        使缓存失效
        
        Args:
            category: 品类标识，如果为 None 则清空所有
            query: 搜索关键词，如果为 None 则清空该品类所有
        """
        with self._lock:
            if category is None:
                self._cache.clear()
                return
            
            if query is not None:
                key = self._make_key(category, query)
                self._cache.pop(key, None)
            else:
                # 删除该品类的所有缓存
                keys_to_delete = [
                    k for k in self._cache.keys() 
                    if k.startswith(f"{category}:")
                ]
                for k in keys_to_delete:
                    del self._cache[k]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total = len(self._cache)
            expired = sum(1 for e in self._cache.values() if e.is_expired())
            
            categories = {}
            for key in self._cache.keys():
                cat = key.split(":")[0]
                categories[cat] = categories.get(cat, 0) + 1
            
            return {
                "total_entries": total,
                "expired_entries": expired,
                "active_entries": total - expired,
                "categories": categories
            }
    
    def cleanup_expired(self):
        """清理过期条目"""
        with self._lock:
            expired_keys = [
                k for k, v in self._cache.items() 
                if v.is_expired()
            ]
            for k in expired_keys:
                del self._cache[k]
            
            return len(expired_keys)


# 全局缓存实例
_product_cache: Optional[ProductCache] = None


def get_product_cache() -> ProductCache:
    """获取全局缓存实例"""
    global _product_cache
    if _product_cache is None:
        _product_cache = ProductCache()
    return _product_cache
