import re
import json
import random
from typing import List, Optional, Dict, Any
from datetime import datetime

import config
from .base_scraper import BaseScraper, ProductData
from .taobao_api_client import TaobaoAPIClient


class TaobaoScraper(BaseScraper):
    """
    淘宝商品爬虫（基于API）
    
    使用淘宝联盟API获取商品数据。
    """
    
    def __init__(self, category: str, app_key: str = None, app_secret: str = None):
        super().__init__(category)
        self.api = TaobaoAPIClient(
            app_key=app_key or config.TAOBAO_APP_KEY,
            app_secret=app_secret or config.TAOBAO_APP_SECRET
        )
    
    def search(self, query: str, limit: int = 50) -> List[ProductData]:
        """淘宝API搜索"""
        print(f"[TaobaoScraper] API搜索: {query}, 限制: {limit}")
        
        products = self.api.search_products(
            keyword=query,
            page_size=min(limit, 100)
        )
        
        result = []
        for tb_product in products[:limit]:
            product_data = self._convert_to_product_data(tb_product)
            result.append(product_data)
        
        return result
    
    def get_details(self, product_id: str) -> Optional[ProductData]:
        """获取商品详情"""
        print(f"[TaobaoScraper] 获取详情: {product_id}")
        
        tb_product = self.api.get_product_detail(product_id)
        if not tb_product:
            return None
        
        return self._convert_to_product_data(tb_product)
    
    def get_hot_products(self, limit: int = 100) -> List[ProductData]:
        """获取热门商品"""
        category_keywords = {
            "camera": ["相机", "数码相机"],
            "phone": ["手机", "智能手机"],
            "headphone": ["耳机", "无线耳机"],
            "laptop": ["笔记本电脑", "超极本"],
            "tablet": ["平板电脑"],
            "skincare": ["护肤品套装", "精华液"],
            "cosmetics": ["彩妆", "口红"],
            "stationery": ["文具", "书写工具"],
            "office": ["办公用品"],
            "appliance": ["小家电", "生活电器"],
            "sports": ["运动装备", "健身器材"],
            "book": ["畅销书"],
        }
        
        keywords = category_keywords.get(self.category, [self.category])
        all_products = []
        per_keyword_limit = (limit // len(keywords)) + 1
        
        for keyword in keywords:
            products = self.search(keyword, per_keyword_limit)
            all_products.extend(products)
        
        # 去重
        seen = set()
        unique_products = []
        for p in all_products:
            if p.source_id not in seen:
                seen.add(p.source_id)
                unique_products.append(p)
                if len(unique_products) >= limit:
                    break
        
        return unique_products
    
    def _convert_to_product_data(self, tb_product) -> ProductData:
        """将淘宝API数据转换为标准格式"""
        
        # 从标题提取品牌和型号 (简单模拟)
        title = tb_product.title
        brand = tb_product.nick.replace("旗舰店", "").replace("官方", "")
        if not brand:
             # 尝试从标题提取头部
             brand = title.split(" ")[0] if " " in title else title[:4]
             
        model = self._extract_model(title, brand)
        
        # 淘宝API通常不直接返回详细规格JSON，这里做模拟
        specs = self._build_specs(tb_product)
        
        product = ProductData(
            category=self.category,
            brand=brand,
            model=model,
            price=tb_product.zk_final_price,
            source="taobao_api",
            source_id=tb_product.num_iid,
            url=tb_product.item_url,
            image_url=tb_product.pict_url,
            scores={},
            specs=specs,
            crawled_at=datetime.now()
        )
        
        product.scores = self.calculate_scores(product)
        return product
    
    def _extract_model(self, title: str, brand: str) -> str:
        model = title.replace(brand, "").strip()
        for suffix in ["现货", "正品", "包邮", "新款"]:
            model = model.replace(suffix, "")
        return model[:20].strip()
    
    def _build_specs(self, tb_product) -> Dict[str, Any]:
        """构建规格数据"""
        specs = {}
        
        # 通用规格基于销量生成模拟好评
        specs["rating"] = round(4.5 + random.uniform(-0.5, 0.4), 1)
        specs["comment_count"] = tb_product.volume  # 用月销量代替评论数
        specs["sales_30d"] = tb_product.volume
        specs["shop_name"] = tb_product.shop_title
        
        # 已知品类特定规格 (模拟)
        if self.category == "camera":
            specs["weight_g"] = random.randint(300, 900)
            specs["megapixels"] = random.randint(20, 61)
            specs["sensor_type"] = random.choice(["Full Frame", "APS-C"])
        
        # ... 可以添加其他品类逻辑，类似于 JDScraper
        
        return specs
    
    def calculate_scores(self, product: ProductData) -> Dict[str, float]:
        """计算评分"""
        scores = {}
        
        # 复用 JDScraper 的评分配置（如果需要）
        # 这里使用通用逻辑
        
        base_score = product.specs.get("rating", 4.5) * 20
        scores["Performance_Score"] = round(min(100, base_score + random.randint(-5, 10)), 1)
        scores["Quality_Score"] = round(min(100, base_score + random.randint(-5, 5)), 1)
        
        # 销量加权
        volume = product.specs.get("sales_30d", 0)
        pop_score = min(100, volume / 100) # 假设10000销量满分 -> 调整为 /100 
        scores["Popularity_Score"] = round(pop_score, 1)

        if product.price > 0:
            avg = sum(scores.values()) / len(scores) if scores else 50
            scores["Value_Score"] = round(avg * 10000 / max(product.price, 1), 1)
            scores["Value_Score"] = min(100, scores["Value_Score"])
            
        return scores
