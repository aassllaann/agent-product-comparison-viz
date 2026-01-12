"""
MongoDB 数据访问层

提供多品类商品数据的存储和查询能力。
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pymongo import MongoClient, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database
import config


class MongoProductDB:
    """
    MongoDB 商品数据库操作类
    
    使用单一 collection 存储所有品类商品，
    通过 category 字段区分不同品类。
    """
    
    def __init__(
        self, 
        uri: str = None,
        db_name: str = "product_recommendation"
    ):
        """
        Args:
            uri: MongoDB 连接 URI，默认使用 config 中的配置
            db_name: 数据库名称
        """
        self.uri = uri or getattr(config, 'MONGO_URI', 'mongodb://localhost:27017')
        self.db_name = db_name
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
    
    @property
    def client(self) -> MongoClient:
        """懒加载 MongoDB 客户端"""
        if self._client is None:
            self._client = MongoClient(self.uri)
        return self._client
    
    @property
    def db(self) -> Database:
        """获取数据库实例"""
        if self._db is None:
            self._db = self.client[self.db_name]
        return self._db
    
    def get_collection(self, name: str = "products") -> Collection:
        """获取集合"""
        return self.db[name]
    
    # ==================== 商品 CRUD 操作 ====================
    
    def insert_product(self, product: dict) -> str:
        """
        插入单个商品
        
        Args:
            product: 商品文档
            
        Returns:
            插入的文档 ID
        """
        product["updated_at"] = datetime.now()
        if "crawled_at" not in product:
            product["crawled_at"] = datetime.now()
        
        result = self.get_collection().insert_one(product)
        return str(result.inserted_id)
    
    def insert_products(self, products: List[dict]) -> List[str]:
        """
        批量插入商品
        
        Args:
            products: 商品文档列表
            
        Returns:
            插入的文档 ID 列表
        """
        now = datetime.now()
        for p in products:
            p["updated_at"] = now
            if "crawled_at" not in p:
                p["crawled_at"] = now
        
        result = self.get_collection().insert_many(products)
        return [str(id) for id in result.inserted_ids]
    
    def upsert_product(self, product: dict, key_fields: List[str] = None) -> str:
        """
        更新或插入商品（根据 source + source_id 判断）
        
        Args:
            product: 商品文档
            key_fields: 用于判断唯一性的字段，默认 ["source", "source_id"]
            
        Returns:
            文档 ID
        """
        if key_fields is None:
            key_fields = ["source", "source_id"]
        
        filter_dict = {k: product.get(k) for k in key_fields}
        product["updated_at"] = datetime.now()
        
        result = self.get_collection().update_one(
            filter_dict,
            {"$set": product, "$setOnInsert": {"crawled_at": datetime.now()}},
            upsert=True
        )
        
        if result.upserted_id:
            return str(result.upserted_id)
        
        # 返回已存在的文档 ID
        doc = self.get_collection().find_one(filter_dict)
        return str(doc["_id"]) if doc else ""
    
    def find_by_category(
        self, 
        category: str,
        filters: Dict[str, Any] = None,
        sort_by: str = None,
        limit: int = 100
    ) -> List[dict]:
        """
        按品类查询商品
        
        Args:
            category: 品类标识
            filters: 额外过滤条件
            sort_by: 排序字段（评分字段名）
            limit: 返回数量限制
            
        Returns:
            商品文档列表
        """
        query = {"category": category}
        if filters:
            query.update(filters)
        
        cursor = self.get_collection().find(query)
        
        if sort_by:
            # 支持按评分排序，评分存储在 scores 字典中
            cursor = cursor.sort(f"scores.{sort_by}", DESCENDING)
        
        return list(cursor.limit(limit))
    
    def find_by_price_range(
        self,
        category: str,
        min_price: float = 0,
        max_price: float = float('inf'),
        sort_by: str = None,
        limit: int = 100
    ) -> List[dict]:
        """
        按价格范围查询
        
        Args:
            category: 品类标识
            min_price: 最低价格
            max_price: 最高价格
            sort_by: 排序字段
            limit: 返回数量
            
        Returns:
            商品文档列表
        """
        query = {
            "category": category,
            "price": {"$gte": min_price, "$lte": max_price}
        }
        
        cursor = self.get_collection().find(query)
        
        if sort_by:
            cursor = cursor.sort(f"scores.{sort_by}", DESCENDING)
        
        return list(cursor.limit(limit))
    
    def search_products(
        self,
        category: str,
        keyword: str,
        limit: int = 50
    ) -> List[dict]:
        """
        关键词搜索
        
        Args:
            category: 品类标识
            keyword: 搜索关键词
            limit: 返回数量
            
        Returns:
            匹配的商品列表
        """
        query = {
            "category": category,
            "$or": [
                {"brand": {"$regex": keyword, "$options": "i"}},
                {"model": {"$regex": keyword, "$options": "i"}}
            ]
        }
        
        return list(self.get_collection().find(query).limit(limit))
    
    def get_categories(self) -> List[str]:
        """获取所有已有品类"""
        return self.get_collection().distinct("category")
    
    def count_by_category(self, category: str) -> int:
        """统计某品类的商品数量"""
        return self.get_collection().count_documents({"category": category})
    
    def delete_by_category(self, category: str) -> int:
        """删除某品类的所有商品"""
        result = self.get_collection().delete_many({"category": category})
        return result.deleted_count
    
    # ==================== 索引管理 ====================
    
    def ensure_indexes(self):
        """创建必要的索引"""
        collection = self.get_collection()
        
        # 品类索引
        collection.create_index("category")
        
        # 品类+价格复合索引
        collection.create_index([("category", 1), ("price", 1)])
        
        # 来源唯一索引
        collection.create_index(
            [("source", 1), ("source_id", 1)],
            unique=True
        )
        
        # 文本搜索索引
        collection.create_index([("brand", "text"), ("model", "text")])
    
    def close(self):
        """关闭连接"""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None


# 全局实例（可选）
_mongo_db: Optional[MongoProductDB] = None


def get_mongo_db() -> MongoProductDB:
    """获取全局 MongoDB 实例"""
    global _mongo_db
    if _mongo_db is None:
        _mongo_db = MongoProductDB()
        _mongo_db.ensure_indexes()
    return _mongo_db
