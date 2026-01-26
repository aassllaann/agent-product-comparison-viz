"""
拼多多商品爬虫（基于API客户端）

使用 pdd_api_client 获取商品数据并标准化。
"""

import re
import json
import random
from typing import List, Optional, Dict, Any
from datetime import datetime

import config
from .base_scraper import BaseScraper, ProductData
from .pdd_api_client import PddAPIClient


class PddScraper(BaseScraper):
    """
    拼多多商品爬虫（基于API）
    
    使用拼多多开放平台API获取商品数据。
    """
    
    def __init__(self, category: str, client_id: str = None, client_secret: str = None):
        super().__init__(category)
        self.api = PddAPIClient(
            client_id=client_id or config.PDD_CLIENT_ID,
            client_secret=client_secret or config.PDD_CLIENT_SECRET
        )
    
    def search(self, query: str, limit: int = 50) -> List[ProductData]:
        """PDD API搜索"""
        print(f"[PddScraper] API搜索: {query}, 限制: {limit}")
        
        products = self.api.search_products(
            keyword=query,
            page_size=min(limit, 100)
        )
        
        result = []
        for pdd_product in products[:limit]:
            product_data = self._convert_to_product_data(pdd_product)
            result.append(product_data)
        
        return result
    
    def get_details(self, product_id: str) -> Optional[ProductData]:
        """获取商品详情 (传入 goods_sign)"""
        print(f"[PddScraper] 获取详情: {product_id}")
        
        pdd_product = self.api.get_product_detail(product_id)
        if not pdd_product:
            return None
        
        return self._convert_to_product_data(pdd_product)
    
    def get_hot_products(self, limit: int = 100) -> List[ProductData]:
        """获取热门商品"""
        category_keywords = {
            "camera": ["相机", "单反相机"],
            "phone": ["手机", "智能手机"],
            "headphone": ["耳机", "蓝牙耳机"],
            "laptop": ["笔记本电脑", "游戏本"],
            "tablet": ["平板电脑"],
            "skincare": ["护肤套装", "补水保湿"],
            "cosmetics": ["彩妆", "口红"],
            "stationery": ["文具", "考试用笔"],
            "office": ["办公用品"],
            "appliance": ["小家电", "网红电器"],
            "sports": ["运动鞋", "运动器材"],
            "book": ["畅销书榜"],
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
    
    def _convert_to_product_data(self, pdd_product) -> ProductData:
        """将PDD API数据转换为标准格式"""
        
        title = pdd_product.goods_name
        brand = pdd_product.brand_name
        if not brand:
             brand = title.split(" ")[0] if " " in title else "未知品牌"
             
        model = self._extract_model(title, brand)
        
        specs = self._build_specs(pdd_product)
        
        # 价格转换（分 -> 元）
        price = pdd_product.min_group_price / 100.0
        
        # 构造链接 (Mock: 使用 mobile web 链接格式)
        url = f"https://mobile.yangkeduo.com/goods.html?goods_id={pdd_product.goods_id}"
        
        product = ProductData(
            category=self.category,
            brand=brand,
            model=model,
            price=price,
            source="pdd_api",
            source_id=str(pdd_product.goods_sign), # 使用 goods_sign 作为 ID 以便获取详情
            url=url,
            image_url=pdd_product.goods_image_url,
            scores={},
            specs=specs,
            crawled_at=datetime.now()
        )
        
        product.scores = self.calculate_scores(product)
        return product
    
    def _extract_model(self, title: str, brand: str) -> str:
        model = title.replace(brand, "").strip()
        for suffix in ["拼购", "大促", "包邮", "正品"]:
            model = model.replace(suffix, "")
        return model[:20].strip()
    
    def _parse_sales_tip(self, tip: str) -> int:
        """解析销量文本 (e.g. '10万+' -> 100000)"""
        if "万" in tip:
            try:
                num = float(re.findall(r"[\d\.]+", tip)[0])
                return int(num * 10000)
            except:
                return 10000
        try:
            return int(re.findall(r"\d+", tip)[0])
        except:
            return 0

    def _build_specs(self, pdd_product) -> Dict[str, Any]:
        """构建规格数据"""
        specs = {}
        
        volume = self._parse_sales_tip(pdd_product.sales_tip)
        
        specs["rating"] = round(4.5 + random.uniform(-0.5, 0.4), 1)
        specs["comment_count"] = volume # 拼多多销量即热度
        specs["sales_tip"] = pdd_product.sales_tip
        specs["shop_name"] = pdd_product.mall_name
        
        # 已知品类特定规格 (模拟)
        if self.category == "camera":
            specs["weight_g"] = random.randint(300, 900)
            specs["megapixels"] = random.randint(20, 61)
            specs["sensor_type"] = random.choice(["Full Frame", "APS-C"])
        
        return specs
    
    def calculate_scores(self, product: ProductData) -> Dict[str, float]:
        """计算评分"""
        scores = {}
        
        # 通用逻辑
        base_score = product.specs.get("rating", 4.5) * 20
        scores["Performance_Score"] = round(min(100, base_score + random.randint(-5, 10)), 1)
        scores["Quality_Score"] = round(min(100, base_score + random.randint(-5, 5)), 1)
        
        # 销量加权
        volume_text = product.specs.get("sales_tip", "0")
        volume = self._parse_sales_tip(volume_text)
        pop_score = min(100, volume / 100) 
        scores["Popularity_Score"] = round(pop_score, 1)

        if product.price > 0:
            avg = sum(scores.values()) / len(scores) if scores else 50
            # 拼多多价格通常较低，性价比分权重可以调高
            scores["Value_Score"] = round(avg * 10000 / max(product.price, 1) * 1.2, 1) # 1.2x 系数
            scores["Value_Score"] = min(100, scores["Value_Score"])
            
        return scores
