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


class JDScraper(BaseScraper):
    """
    京东商品爬虫（基于API）
    
    使用京东联盟API获取商品数据。
    当前为mock实现，真实API接入只需更新 jd_api_client.py
    """
    
    def __init__(self, category: str, app_key: str = None, app_secret: str = None):
        super().__init__(category)
        self.api = JDAPIClient(
            app_key=app_key or "mock_key",
            app_secret=app_secret or "mock_secret"
        )
    
    def search(self, query: str, limit: int = 50) -> List[ProductData]:
        """
        搜索商品
        
        Args:
            query: 搜索关键词
            limit: 返回数量限制
        
        Returns:
            标准化的商品数据列表
        """
        print(f"[JDScraper] API搜索: {query}, 限制: {limit}")
        
        # 调用API
        products = self.api.search_products(
            keyword=query,
            page_size=min(limit, 100)  # API单页最多100条
        )
        
        # 转换为标准格式
        result = []
        for jd_product in products[:limit]:
            product_data = self._convert_to_product_data(jd_product)
            result.append(product_data)
        
        return result
    
    def get_details(self, product_id: str) -> Optional[ProductData]:
        """获取商品详情"""
        print(f"[JDScraper] 获取详情: {product_id}")
        
        jd_product = self.api.get_product_detail(product_id)
        if not jd_product:
            return None
        
        return self._convert_to_product_data(jd_product)
    
    def get_hot_products(self, limit: int = 100) -> List[ProductData]:
        """
        获取热门商品
        
        通过搜索品类关键词获取热门商品
        """
        # 品类关键词映射
        category_keywords = {
            "camera": ["相机", "数码相机"],
            "phone": ["手机", "智能手机"],
            "headphone": ["耳机", "降噪耳机"],
            "laptop": ["笔记本电脑", "游戏本"],
            "tablet": ["平板电脑", "iPad"],
            "skincare": ["护肤品", "精华"],
            "cosmetics": ["化妆品", "口红"],
            "stationery": ["钢笔", "中性笔"],
            "office": ["打印机"],
            "appliance": ["吹风机", "电饭煲"],
            "sports": ["跑步鞋", "运动鞋"],
            "book": ["畅销书", "小说"],
        }
        
        keywords = category_keywords.get(self.category, [self.category])
        all_products = []
        
        per_keyword_limit = (limit // len(keywords)) + 1
        
        for keyword in keywords:
            products = self.search(keyword, per_keyword_limit)
            all_products.extend(products)
        
        # 去重并限制数量
        seen = set()
        unique_products = []
        for p in all_products:
            key = f"{p.brand}_{p.model}"
            if key not in seen:
                seen.add(key)
                unique_products.append(p)
                if len(unique_products) >= limit:
                    break
        
        return unique_products
    
    def _convert_to_product_data(self, jd_product) -> ProductData:
        """将京东API商品数据转换为标准格式"""
        # 解析参数JSON
        try:
            params = json.loads(jd_product.param_json)
        except:
            params = {}
        
        # 提取品牌和型号
        brand = jd_product.brand_name or params.get("品牌", "")
        model = self._extract_model(jd_product.sku_name, brand)
        
        # 构建规格数据
        specs = self._build_specs(params, jd_product)
        
        # 创建商品对象
        product = ProductData(
            category=self.category,
            brand=brand,
            model=model,
            price=jd_product.price,
            source="jd_api",
            source_id=jd_product.sku_id,
            url=f"https://item.jd.com/{jd_product.sku_id}.html",
            image_url=jd_product.img_url,
            scores={},
            specs=specs,
            crawled_at=datetime.now()
        )
        
        # 计算评分
        product.scores = self.calculate_scores(product)
        
        return product
    
    def _extract_model(self, sku_name: str, brand: str) -> str:
        """从商品名称提取型号"""
        # 移除品牌名
        model = sku_name.replace(brand, "").strip()
        
        # 移除常见后缀
        for suffix in ["官方旗舰店", "正品", "国行", "全新"]:
            model = model.replace(suffix, "")
        
        # 取前20个字符作为型号
        return model[:20].strip()
    
    def _build_specs(self, params: Dict, jd_product) -> Dict[str, Any]:
        """根据品类构建规格数据"""
        specs = {}
        
        # 通用规格
        specs["rating"] = jd_product.good_comments_share * 5  # 转为1-5分
        specs["comment_count"] = jd_product.comments
        
        # 尝试从参数JSON中提取常用规格
        if "商品毛重" in params:
            specs["weight_info"] = params["商品毛重"]
        if "产地" in params:
            specs["origin"] = params["产地"]
            
        # 已知品类特定规格
        if self.category == "camera":
            specs["weight_g"] = self._extract_number(params.get("重量", ""), default=random.randint(300, 900))
            specs["max_iso"] = random.choice([25600, 51200, 102400, 204800])
            specs["supports_4k"] = random.choice([True, True, True, False])
            specs["sensor_type"] = random.choice(["Full Frame", "APS-C", "M4/3"])
            specs["megapixels"] = random.randint(20, 61)
        
        # ... (保留其他已知品类)
        
        return specs
    
    def calculate_scores(self, product: ProductData) -> Dict[str, float]:
        """根据商品规格计算评分"""
        scores = {}
        
        # 尝试获取已知品类的特定字段配置
        from .jd_scraper import CATEGORY_SCORE_FIELDS
        score_config = CATEGORY_SCORE_FIELDS.get(self.category, {})
        
        if score_config:
            # 已知品类：按规则计算
            specs_to_scores = score_config.get("specs_to_scores", {})
            for spec_field, (score_field, calc_func) in specs_to_scores.items():
                if spec_field in product.specs:
                    value = product.specs[spec_field]
                    if value is not None:
                        try:
                            scores[score_field] = round(calc_func(value), 1)
                        except:
                            pass
        else:
            # 未知品类：生成通用评分
            # 基于价格、销量(评论数)、好评率生成模拟分数
            base_score = product.specs.get("rating", 4.5) * 20  # 基础分 (0-100)
            
            scores["Performance_Score"] = round(min(100, base_score + random.randint(-5, 10)), 1)
            scores["Quality_Score"] = round(min(100, base_score + random.randint(-5, 5)), 1)
            scores["Popularity_Score"] = min(100, round(product.specs.get("comment_count", 0) / 1000 * 10, 1))
        
        # 添加通用评分
        if product.price > 0:
            avg_score = sum(scores.values()) / len(scores) if scores else 50
            scores["Value_Score"] = round(avg_score * 10000 / max(product.price, 1), 1)
            scores["Value_Score"] = min(100, scores["Value_Score"])
        
        return scores
        
        return scores



# 品类搜索关键词映射
CATEGORY_KEYWORDS = {
    # 数码电子
    "camera": ["数码相机", "微单相机", "单反相机"],
    "phone": ["手机", "智能手机"],
    "headphone": ["耳机", "蓝牙耳机", "降噪耳机"],
    "laptop": ["笔记本电脑", "游戏本", "轻薄本"],
    "tablet": ["平板电脑", "iPad"],
    
    # 护肤美妆
    "skincare": ["护肤品", "面霜", "精华液", "防晒霜"],
    "cosmetics": ["化妆品", "口红", "粉底液"],
    
    # 文具办公
    "stationery": ["钢笔", "中性笔", "文具套装"],
    "office": ["办公用品", "打印机"],
    
    # 家电
    "appliance": ["小家电", "吹风机", "电饭煲"],
    
    # 运动户外
    "sports": ["运动鞋", "跑步鞋", "运动装备"],
    
    # 图书
    "book": ["畅销书", "小说", "教材"],
}

# 品类评分维度映射
CATEGORY_SCORE_FIELDS = {
    "camera": {
        "specs_to_scores": {
            "weight_g": ("Portability_Score", lambda x: max(0, 100 - x / 10)),
            "max_iso": ("LowLight_Score", lambda x: min(100, x / 2000)),
            "supports_4k": ("Video_Score", lambda x: 90 if x else 50),
        }
    },
    "phone": {
        "specs_to_scores": {
            "battery_mah": ("Battery_Score", lambda x: min(100, x / 50)),
            "screen_size": ("Display_Score", lambda x: min(100, x * 15)),
            "ram_gb": ("Performance_Score", lambda x: min(100, x * 8)),
        }
    },
    "headphone": {
        "specs_to_scores": {
            "battery_hours": ("Battery_Score", lambda x: min(100, x * 3)),
            "has_anc": ("Feature_Score", lambda x: 90 if x else 50),
            "weight_g": ("Comfort_Score", lambda x: max(0, 100 - x / 3)),
        }
    },
    "laptop": {
        "specs_to_scores": {
            "weight_kg": ("Portability_Score", lambda x: max(0, 100 - x * 20)),
            "ram_gb": ("Performance_Score", lambda x: min(100, x * 5)),
            "battery_hours": ("Battery_Score", lambda x: min(100, x * 8)),
        }
    },
    "skincare": {
        "specs_to_scores": {
            "volume_ml": ("Value_Score", lambda x: min(100, x / 2)),
            "rating": ("Effect_Score", lambda x: x * 20),
            "is_organic": ("Natural_Score", lambda x: 90 if x else 60),
        }
    },
    "cosmetics": {
        "specs_to_scores": {
            "color_count": ("Variety_Score", lambda x: min(100, x * 10)),
            "rating": ("Quality_Score", lambda x: x * 20),
            "is_waterproof": ("Durability_Score", lambda x: 85 if x else 50),
        }
    },
    "stationery": {
        "specs_to_scores": {
            "tip_size_mm": ("Precision_Score", lambda x: 100 - x * 10),
            "ink_capacity_ml": ("Durability_Score", lambda x: min(100, x * 50)),
            "is_refillable": ("Value_Score", lambda x: 80 if x else 50),
        }
    },
    "appliance": {
        "specs_to_scores": {
            "power_w": ("Performance_Score", lambda x: min(100, x / 20)),
            "warranty_years": ("Reliability_Score", lambda x: min(100, x * 30)),
            "noise_db": ("Comfort_Score", lambda x: max(0, 100 - x)),
        }
    },
    "sports": {
        "specs_to_scores": {
            "weight_g": ("Comfort_Score", lambda x: max(0, 100 - x / 5)),
            "cushion_level": ("Comfort_Score", lambda x: x * 20),
            "is_waterproof": ("Durability_Score", lambda x: 85 if x else 60),
        }
    },
    "book": {
        "specs_to_scores": {
            "rating": ("Quality_Score", lambda x: x * 20),
            "pages": ("Content_Score", lambda x: min(100, x / 5)),
            "year": ("Freshness_Score", lambda x: max(0, 100 - (2026 - x) * 5)),
        }
    },
}


class JDScraper(BaseScraper):
    """
    京东商品爬虫
    
    注意：这是一个示例实现，实际使用时：
    1. 需要处理反爬机制
    2. 建议使用京东开放平台 API
    3. 遵守爬虫协议和法律法规
    """
    
    BASE_URL = "https://search.jd.com/Search"
    DETAIL_URL = "https://item.jd.com/{}.html"
    
    def __init__(self, category: str):
        super().__init__(category)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    
    def search(self, query: str, limit: int = 50) -> List[ProductData]:
        """
        搜索商品
        
        注意：此方法需要实际实现爬取逻辑，
        目前返回模拟数据用于开发测试。
        """
        # TODO: 实现实际爬取逻辑
        # 这里返回模拟数据用于开发
        print(f"[JDScraper] 搜索: {query}, 限制: {limit}")
        return self._get_mock_data(query, limit)
    
    def get_details(self, product_id: str) -> Optional[ProductData]:
        """获取商品详情"""
        # TODO: 实现实际爬取逻辑
        print(f"[JDScraper] 获取详情: {product_id}")
        return None
    
    def get_hot_products(self, limit: int = 100) -> List[ProductData]:
        """
        获取热门商品
        
        用于预置 Tier 1 品类的 100 条数据。
        """
        keywords = CATEGORY_KEYWORDS.get(self.category, [self.category])
        all_products = []
        
        per_keyword_limit = limit // len(keywords) + 1
        
        for keyword in keywords:
            products = self.search(keyword, per_keyword_limit)
            all_products.extend(products)
        
        # 去重并限制数量
        seen = set()
        unique_products = []
        for p in all_products:
            key = f"{p.brand}_{p.model}"
            if key not in seen:
                seen.add(key)
                unique_products.append(p)
                if len(unique_products) >= limit:
                    break
        
        return unique_products
    
    def calculate_scores(self, product: ProductData) -> Dict[str, float]:
        """
        根据商品规格计算评分
        """
        scores = {}
        score_config = CATEGORY_SCORE_FIELDS.get(self.category, {})
        specs_to_scores = score_config.get("specs_to_scores", {})
        
        for spec_field, (score_field, calc_func) in specs_to_scores.items():
            if spec_field in product.specs:
                value = product.specs[spec_field]
                if value is not None:
                    try:
                        scores[score_field] = round(calc_func(value), 1)
                    except:
                        pass
        
        # 添加通用评分
        if product.price > 0:
            # 性价比评分（简化计算）
            avg_score = sum(scores.values()) / len(scores) if scores else 50
            scores["Value_Score"] = round(avg_score * 10000 / max(product.price, 1), 1)
            scores["Value_Score"] = min(100, scores["Value_Score"])
        
        return scores
    
    def _get_mock_data(self, query: str, limit: int) -> List[ProductData]:
        """
        生成模拟数据用于开发测试
        
        实际项目中应替换为真实爬取逻辑。
        """
        mock_products = []
        
        # 品类品牌和型号映射
        MOCK_DATA = {
            "camera": {
                "brands": ["Sony", "Canon", "Nikon", "Fujifilm", "Panasonic"],
                "models": ["A7 IV", "R6 II", "Z6 III", "X-T5", "GH6", "A6700", "R8", "Zf"],
                "price_range": (3000, 30000),
            },
            "phone": {
                "brands": ["Apple", "Samsung", "Xiaomi", "Huawei", "OPPO", "vivo"],
                "models": ["iPhone 16", "Galaxy S25", "15 Pro", "Mate 70", "Find X8", "X200"],
                "price_range": (2000, 15000),
            },
            "headphone": {
                "brands": ["Sony", "Apple", "Bose", "Sennheiser", "JBL"],
                "models": ["WH-1000XM5", "AirPods Pro 2", "QC Ultra", "Momentum 4", "Tour Pro 2"],
                "price_range": (200, 4000),
            },
            "laptop": {
                "brands": ["Apple", "Lenovo", "Dell", "ASUS", "HP", "Huawei"],
                "models": ["MacBook Pro 14", "ThinkPad X1", "XPS 15", "ROG Zephyrus", "MateBook X"],
                "price_range": (4000, 25000),
            },
            "skincare": {
                "brands": ["兰蔻", "雅诗兰黛", "SK-II", "资生堂", "欧莱雅", "珀莱雅"],
                "models": ["小黑瓶精华", "小棕瓶精华", "神仙水", "红腰子", "玻色因面霜", "双抗精华"],
                "price_range": (100, 2000),
            },
            "cosmetics": {
                "brands": ["MAC", "阿玛尼", "YSL", "迪奥", "香奈儿", "完美日记"],
                "models": ["子弹头口红", "红管唇釉", "黑管口红", "999口红", "丝绒唇膏", "小细跟"],
                "price_range": (50, 500),
            },
            "stationery": {
                "brands": ["百乐", "斑马", "三菱", "凌美", "派克", "晨光"],
                "models": ["P500中性笔", "JJ15按动笔", "uni-ball签字笔", "狩猎者钢笔", "威雅钢笔", "优品中性笔"],
                "price_range": (5, 500),
            },
            "office": {
                "brands": ["惠普", "爱普生", "兄弟", "佳能", "得力", "齐心"],
                "models": ["激光打印机", "喷墨打印机", "彩色打印机", "碎纸机", "装订机", "扫描仪"],
                "price_range": (200, 5000),
            },
            "appliance": {
                "brands": ["戴森", "松下", "美的", "小米", "苏泊尔", "九阳"],
                "models": ["吹风机HD08", "电饭煲SR", "空气炸锅", "破壁机", "电磁炉", "电热水壶"],
                "price_range": (100, 4000),
            },
            "sports": {
                "brands": ["耐克", "阿迪达斯", "亚瑟士", "新百伦", "李宁", "安踏"],
                "models": ["Air Max跑鞋", "Ultra Boost", "GEL-KAYANO", "990v6", "超轻20", "C202"],
                "price_range": (300, 2000),
            },
            "book": {
                "brands": ["人民文学出版社", "中信出版社", "机械工业出版社", "电子工业出版社", "清华大学出版社"],
                "models": ["三体", "活着", "深度学习", "Python编程", "经济学原理", "人类简史"],
                "price_range": (20, 200),
            },
        }
        
        # 获取品类数据配置
        data_config = MOCK_DATA.get(self.category, {
            "brands": ["品牌A", "品牌B", "品牌C"],
            "models": ["型号1", "型号2", "型号3"],
            "price_range": (100, 1000),
        })
        
        brands = data_config["brands"]
        models = data_config["models"]
        price_min, price_max = data_config["price_range"]
        
        for i in range(min(limit, 20)):
            brand = brands[i % len(brands)]
            model = models[i % len(models)]
            
            # 生成随机价格和规格
            price = random.randint(price_min, price_max)
            
            specs = self._generate_mock_specs()
            
            product = ProductData(
                category=self.category,
                brand=brand,
                model=f"{model} ({i+1})",
                price=price,
                source="jd_mock",
                source_id=f"mock_{self.category}_{i}",
                url=f"https://item.jd.com/mock_{i}.html",
                image_url=f"https://via.placeholder.com/300?text={brand}+{model}",
                scores={},
                specs=specs,
                crawled_at=datetime.now()
            )
            
            # 计算评分
            product.scores = self.calculate_scores(product)
            
            mock_products.append(product)
        
        return mock_products
    
    def _generate_mock_specs(self) -> Dict[str, Any]:
        """生成模拟规格数据"""
        SPECS_CONFIG = {
            "camera": {
                "weight_g": lambda: random.randint(300, 900),
                "max_iso": lambda: random.choice([25600, 51200, 102400, 204800]),
                "supports_4k": lambda: random.choice([True, True, True, False]),
                "sensor_type": lambda: random.choice(["Full Frame", "APS-C", "M4/3"]),
                "megapixels": lambda: random.randint(20, 61),
            },
            "phone": {
                "ram_gb": lambda: random.choice([8, 12, 16]),
                "storage_gb": lambda: random.choice([128, 256, 512, 1024]),
                "battery_mah": lambda: random.randint(4000, 6000),
                "screen_size": lambda: round(random.uniform(6.1, 6.9), 1),
            },
            "headphone": {
                "weight_g": lambda: random.randint(200, 350),
                "battery_hours": lambda: random.randint(20, 60),
                "has_anc": lambda: random.choice([True, True, False]),
                "driver_mm": lambda: random.choice([30, 40, 50]),
            },
            "laptop": {
                "weight_kg": lambda: round(random.uniform(1.2, 2.5), 1),
                "ram_gb": lambda: random.choice([8, 16, 32, 64]),
                "storage_gb": lambda: random.choice([256, 512, 1024, 2048]),
                "battery_hours": lambda: random.randint(6, 18),
                "screen_size": lambda: random.choice([13.3, 14, 15.6, 16]),
            },
            "skincare": {
                "volume_ml": lambda: random.choice([30, 50, 75, 100, 150]),
                "rating": lambda: round(random.uniform(4.0, 5.0), 1),
                "is_organic": lambda: random.choice([True, False]),
                "skin_type": lambda: random.choice(["干性", "油性", "混合", "敏感", "中性"]),
            },
            "cosmetics": {
                "color_count": lambda: random.randint(1, 12),
                "rating": lambda: round(random.uniform(4.0, 5.0), 1),
                "is_waterproof": lambda: random.choice([True, False]),
                "finish": lambda: random.choice(["哑光", "滋润", "丝绒", "水光"]),
            },
            "stationery": {
                "tip_size_mm": lambda: random.choice([0.38, 0.5, 0.7, 1.0]),
                "ink_capacity_ml": lambda: round(random.uniform(0.5, 2.0), 1),
                "is_refillable": lambda: random.choice([True, False]),
                "color": lambda: random.choice(["黑色", "蓝色", "红色", "彩色"]),
            },
            "appliance": {
                "power_w": lambda: random.choice([500, 800, 1200, 1600, 2000]),
                "warranty_years": lambda: random.choice([1, 2, 3]),
                "noise_db": lambda: random.randint(40, 70),
                "capacity": lambda: random.choice(["1L", "2L", "3L", "5L"]),
            },
            "sports": {
                "weight_g": lambda: random.randint(200, 400),
                "cushion_level": lambda: random.randint(3, 5),
                "is_waterproof": lambda: random.choice([True, False]),
                "size_range": lambda: random.choice(["36-44", "38-46", "35-45"]),
            },
            "book": {
                "rating": lambda: round(random.uniform(4.0, 5.0), 1),
                "pages": lambda: random.randint(200, 800),
                "year": lambda: random.randint(2020, 2026),
                "format": lambda: random.choice(["精装", "平装", "电子书"]),
            },
        }
        
        config = SPECS_CONFIG.get(self.category, {})
        return {k: v() for k, v in config.items()}

