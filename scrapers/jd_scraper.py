"""
京东商品爬虫（基于API客户端）

使用 jd_api_client 获取商品数据并标准化。
"""

import re
import json
import random
from typing import List, Optional, Dict, Any
from datetime import datetime

from .base_scraper import BaseScraper, ProductData
from .jd_api_client import JDAPIClient
import config

# 品类搜索关键词映射 (部分 key 需与 JD 搜索习惯对齐)
CATEGORY_KEYWORDS = {
    "camera": ["数码相机"],
    "phone": ["手机"],
    "headphone": ["耳机"],
    "laptop": ["笔记本电脑"],
    "tablet": ["平板电脑"],
    "skincare": ["护肤品"],
    "cosmetics": ["化妆品"],
    "stationery": ["文具"],
    "office": ["办公用品"],
    "appliance": ["小家电"],
    "sports": ["运动装备"],
    "book": ["畅销书"],
}

class JDScraper(BaseScraper):
    """
    京东商品爬虫（基于API）
    
    使用京东联盟API获取商品数据。
    """
    
    def __init__(self, category: str, app_key: str = None, app_secret: str = None):
        super().__init__(category)
        self.api = JDAPIClient(
            app_key=app_key or config.JD_APP_KEY,
            app_secret=app_secret or config.JD_APP_SECRET
        )
    
    def search(self, query: str, limit: int = 50) -> List[ProductData]:
        """
        搜索商品
        """
        print(f"[JDScraper] API搜索: {query}, 限制: {limit}")
        
        products = self.api.search_products(
            keyword=query,
            page_size=min(limit, 100),
            sort_type="bfs" # 综合排序
        )
        
        result = []
        for jd_product in products[:limit]:
            product_data = self._convert_to_product_data(jd_product)
            result.append(product_data)
        
        return result
    
    def get_details(self, product_id: str) -> Optional[ProductData]:
        """获取商品详情 (Big Field)"""
        print(f"[JDScraper] 获取详情: {product_id}")
        
        jd_product = self.api.get_product_detail(product_id)
        if not jd_product:
            return None
        
        # 详情接口返回的通常信息更全，尤其是大字段
        return self._convert_to_product_data(jd_product)
    
    def get_hot_products(self, limit: int = 100) -> List[ProductData]:
        """
        获取热门商品
        """
        keywords = CATEGORY_KEYWORDS.get(self.category, [self.category])
        all_products = []
        
        # 平均分配每个关键词的额度
        per_keyword_limit = max(10, (limit // len(keywords)) + 1)
        
        for keyword in keywords:
            products = self.search(keyword, per_keyword_limit)
            all_products.extend(products)
            if len(all_products) >= limit * 1.5: # 稍微多抓一点用于去重
                break
        
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
    
    def _convert_to_product_data(self, jd_product) -> ProductData:
        """将京东API商品数据转换为标准格式"""
        
        # 如果有大字段信息，优先从大字段提取
        big_field = jd_product.big_field_info or {}
        
        if big_field:
            cat_info = big_field.get("categoryInfo", {})
            owner = big_field.get("owner", "")
            brand_name = cat_info.get("cid3Name", "未知品牌") # 大字段里 brand 可能不直观
            # User example does not have brandName explicit in bigField
            # Try to get from param_json if available
        else:
            brand_name = jd_product.brand_name
            
        # 尝试从参数JSON提取更准确的品牌/型号
        try:
            params = json.loads(jd_product.param_json) if jd_product.param_json else {}
        except:
            params = {}
            
        real_brand = params.get("品牌") or brand_name or "京东商品"
        # 简单清洗品牌名
        if "（" in real_brand: 
            real_brand = real_brand.split("（")[0]
            
        # 提取型号
        sku_name = jd_product.sku_name or big_field.get("skuName", "") 
        model = self._extract_model(sku_name, real_brand)
        if "型号" in params:
             model = params["型号"]

        # 规格构建
        specs = self._build_specs(params, jd_product, big_field)
        
        product = ProductData(
            category=self.category,
            brand=real_brand,
            model=model,
            price=jd_product.price, # 注意：bigfield 接口可能返 0，如果是这种情况可能需要由调用方（search结果）补全，或者再次查价
            source="jd_api",
            source_id=jd_product.sku_id,
            url=f"https://item.jd.com/{jd_product.sku_id}.html",
            image_url=jd_product.img_url,
            scores={},
            specs=specs,
            crawled_at=datetime.now()
        )
        
        product.scores = self.calculate_scores(product)
        return product
    
    def _extract_model(self, sku_name: str, brand: str) -> str:
        model = sku_name.replace(brand, "").strip()
        for suffix in ["官方旗舰店", "正品", "自营", "京东配送"]:
            model = model.replace(suffix, "")
        return model[:30].strip()
    
    def _build_specs(self, params: Dict, jd_product, big_field: Dict) -> Dict[str, Any]:
        """构建规格数据"""
        specs = {}
        
        # 基础数据
        specs["rating"] = round(jd_product.good_comments_share * 5, 1) # e.g. 0.98 -> 4.9
        specs["comment_count"] = jd_product.comments
        specs["shop_name"] = jd_product.shop_name or big_field.get("owner", "")
        
        # 整合 big_field 数据
        if big_field:
            base_info = big_field.get("baseBigFieldInfo", {})
            try:
                # propGroups 有时是 JSON 串
                 prop_groups = base_info.get("propGroups")
                 if prop_groups:
                     # 尝试解析 propGroups
                     # User example says "JSON串"
                     pass 
            except:
                pass
                
            video_info = big_field.get("videoBigFieldInfo", {})
            if video_info:
                specs["product_features"] = video_info.get("productFeatures", "")
                
        # 尝试从 params 提取通用属性
        if "商品毛重" in params:
            specs["weight_info"] = params["商品毛重"]
        if "产地" in params:
            specs["origin"] = params["产地"]
            
        # 能够提取到的额外参数都放进去
        for k, v in params.items():
            if k not in ["品牌", "型号"]:
                specs[k] = v
                
        return specs
        
