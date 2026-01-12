"""
爬虫基类

定义商品数据爬取的通用接口和流程。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ProductData:
    """
    通用商品数据结构
    
    适用于所有品类，使用灵活的 specs 字段存储品类特有属性。
    """
    # 基础信息
    category: str               # 品类 (camera, phone, headphone, laptop)
    brand: str                  # 品牌
    model: str                  # 型号
    price: float                # 价格
    
    # 元信息
    source: str                 # 数据来源 (jd, taobao, official)
    source_id: str              # 来源平台的商品ID
    url: str                    # 商品链接
    image_url: str              # 主图链接
    
    # 评分（由 LLM 或规则计算）
    scores: Dict[str, float]    # {"Performance_Score": 85, "Value_Score": 78, ...}
    
    # 品类特有规格（灵活存储）
    specs: Dict[str, Any]       # {"sensor_type": "CMOS", "weight_g": 450, ...}
    
    # 时间戳
    crawled_at: datetime = None
    updated_at: datetime = None
    
    def to_mongo_doc(self) -> dict:
        """转换为 MongoDB 文档格式"""
        return {
            "category": self.category,
            "brand": self.brand,
            "model": self.model,
            "price": self.price,
            "source": self.source,
            "source_id": self.source_id,
            "url": self.url,
            "image_url": self.image_url,
            "scores": self.scores,
            "specs": self.specs,
            "crawled_at": self.crawled_at or datetime.now(),
            "updated_at": self.updated_at or datetime.now()
        }
    
    @classmethod
    def from_mongo_doc(cls, doc: dict) -> 'ProductData':
        """从 MongoDB 文档构建对象"""
        return cls(
            category=doc.get("category", ""),
            brand=doc.get("brand", ""),
            model=doc.get("model", ""),
            price=doc.get("price", 0),
            source=doc.get("source", ""),
            source_id=doc.get("source_id", ""),
            url=doc.get("url", ""),
            image_url=doc.get("image_url", ""),
            scores=doc.get("scores", {}),
            specs=doc.get("specs", {}),
            crawled_at=doc.get("crawled_at"),
            updated_at=doc.get("updated_at")
        )


class BaseScraper(ABC):
    """
    商品数据爬虫基类
    
    所有品类的爬虫都应继承此类，实现抽象方法。
    """
    
    def __init__(self, category: str):
        """
        Args:
            category: 品类标识 (如 "camera", "phone")
        """
        self.category = category
    
    @abstractmethod
    def search(self, query: str, limit: int = 50) -> List[ProductData]:
        """
        搜索商品
        
        Args:
            query: 搜索关键词
            limit: 返回数量限制
            
        Returns:
            商品数据列表
        """
        pass
    
    @abstractmethod
    def get_details(self, product_id: str) -> Optional[ProductData]:
        """
        获取商品详情
        
        Args:
            product_id: 商品ID（来源平台的ID）
            
        Returns:
            商品详情，如果不存在返回 None
        """
        pass
    
    @abstractmethod
    def get_hot_products(self, limit: int = 100) -> List[ProductData]:
        """
        获取热门商品列表
        
        用于预置热门品类数据。
        
        Args:
            limit: 返回数量
            
        Returns:
            热门商品列表
        """
        pass
    
    def calculate_scores(self, product: ProductData) -> Dict[str, float]:
        """
        计算商品评分
        
        子类可覆盖以实现品类特有的评分逻辑。
        默认返回空字典，由 LLM 或其他方式计算。
        
        Args:
            product: 商品数据
            
        Returns:
            评分字典 {"Score_Name": value, ...}
        """
        return {}
    
    def normalize_data(self, raw_data: dict) -> ProductData:
        """
        规范化原始数据
        
        将不同来源的数据转换为统一的 ProductData 格式。
        子类应覆盖此方法。
        
        Args:
            raw_data: 原始爬取数据
            
        Returns:
            规范化的商品数据
        """
        return ProductData(
            category=self.category,
            brand=raw_data.get("brand", "未知"),
            model=raw_data.get("model", "未知"),
            price=float(raw_data.get("price", 0)),
            source="unknown",
            source_id=str(raw_data.get("id", "")),
            url=raw_data.get("url", ""),
            image_url=raw_data.get("image", ""),
            scores={},
            specs=raw_data
        )
