"""
京东联盟API客户端（Mock实现）

此模块模拟京东联盟API的响应格式，便于开发测试。
真实API接入时，只需替换 _mock_request 为真实HTTP请求。

京东联盟API文档: https://union.jd.com/openplatform
"""

import json
import hashlib
import time
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class JDProduct:
    """京东商品数据模型"""
    sku_id: str
    sku_name: str
    price: float
    img_url: str
    shop_name: str
    comments: int
    good_comments_share: float
    category_name: str
    brand_name: str
    # 扩展属性
    param_json: str  # 商品参数JSON字符串


class JDAPIClient:
    """京东联盟API客户端"""
    
    def __init__(self, app_key: str = "mock_key", app_secret: str = "mock_secret"):
        self.app_key = app_key
        self.app_secret = app_secret
        self.base_url = "https://api.jd.com/routerjson"
        
    def search_products(
        self, 
        keyword: str, 
        page: int = 1, 
        page_size: int = 20,
        price_from: Optional[int] = None,
        price_to: Optional[int] = None,
        sort_type: str = "综合"  # 综合/价格/销量/评论数
    ) -> List[JDProduct]:
        """
        搜索商品
        
        Args:
            keyword: 搜索关键词
            page: 页码
            page_size: 每页数量
            price_from: 最低价格
            price_to: 最高价格
            sort_type: 排序方式
        
        Returns:
            商品列表
        """
        params = {
            "method": "jd.union.open.goods.query",
            "app_key": self.app_key,
            "timestamp": self._get_timestamp(),
            "format": "json",
            "v": "1.0",
            "sign_method": "md5",
            "param_json": json.dumps({
                "keyword": keyword,
                "pageIndex": page,
                "pageSize": page_size,
                "priceFrom": price_from,
                "priceTo": price_to,
                "sortName": sort_type
            })
        }
        
        # 签名
        params["sign"] = self._sign(params)
        
        # Mock实现 - 真实实现需替换为HTTP请求
        response = self._mock_request(params)
        
        return self._parse_products(response)
    
    def get_product_detail(self, sku_id: str) -> Optional[JDProduct]:
        """获取商品详情"""
        params = {
            "method": "jd.union.open.goods.promotiongoodsinfo.query",
            "app_key": self.app_key,
            "timestamp": self._get_timestamp(),
            "format": "json",
            "v": "1.0",
            "sign_method": "md5",
            "param_json": json.dumps({
                "skuIds": [sku_id]
            })
        }
        
        params["sign"] = self._sign(params)
        response = self._mock_request(params)
        
        products = self._parse_products(response)
        return products[0] if products else None
    
    def _sign(self, params: Dict[str, Any]) -> str:
        """生成API签名"""
        # 按字母序排序
        sorted_params = sorted(params.items())
        
        # 拼接字符串
        sign_str = self.app_secret
        for k, v in sorted_params:
            if k != "sign":
                sign_str += f"{k}{v}"
        sign_str += self.app_secret
        
        # MD5加密
        return hashlib.md5(sign_str.encode()).hexdigest().upper()
    
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        return time.strftime("%Y-%m-%d %H:%M:%S")
    
    def _mock_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        模拟API请求
        
        真实实现应替换为:
        import requests
        response = requests.post(self.base_url, data=params)
        return response.json()
        """
        keyword = json.loads(params.get("param_json", "{}")).get("keyword", "")
        page_size = json.loads(params.get("param_json", "{}")).get("pageSize", 20)
        
        # 根据关键词生成模拟数据
        mock_data = self._generate_mock_data(keyword, page_size)
        
        return {
            "code": "0",
            "message": "success",
            "data": mock_data
        }
    
    def _generate_mock_data(self, keyword: str, count: int) -> List[Dict]:
        """生成模拟商品数据"""
        # 根据关键词决定品类，优先匹配已知品类，否则使用通用逻辑
        category_map = {
            "相机": ("相机", ["Sony", "Canon", "Nikon", "Fujifilm"], 
                    ["A7 IV", "R6 II", "Z6 III", "X-T5"]),
            "手机": ("手机", ["Apple", "Samsung", "Xiaomi", "Huawei"], 
                    ["iPhone 16", "Galaxy S25", "15 Pro", "Mate 70"]),
            "耳机": ("耳机", ["Sony", "Apple", "Bose", "Sennheiser"], 
                    ["WH-1000XM5", "AirPods Pro 2", "QC Ultra", "Momentum 4"]),
            "护肤": ("护肤品", ["兰蔻", "雅诗兰黛", "SK-II", "资生堂"], 
                    ["小黑瓶", "小棕瓶", "神仙水", "红腰子"]),
            "化妆": ("化妆品", ["Dior", "Chanel", "YSL", "MAC"], 
                  ["999", "58", "小金条", "子弹头"]),
            "笔": ("文具", ["百乐", "斑马", "三菱", "晨光"], 
                  ["P500", "JJ15", "UM-151", "优品"]),
        }
        
        # 匹配品类
        matched = False
        category_name, brands, models = (keyword, ["品牌A", "品牌B", "品牌C"], ["型号Pro", "型号Max", "型号Ultra"])
        
        for key, data in category_map.items():
            if key in keyword:
                category_name, brands, models = data
                matched = True
                break
        
        # 如果未匹配到已知品类，生成通用数据
        if not matched:
            category_name = keyword
            if len(keyword) > 8: # 关键词太长截断
                 category_name = keyword[:8]
            brands = ["知名品牌", "热销品牌", "进口品牌", "国产优选"]
            models = ["旗舰款", "升级版", "经典款", "专业版", "家用款"]
        
        products = []
        for i in range(min(count, 20)):
            brand = brands[i % len(brands)]
            model = models[i % len(models)]
            
            # 增加一些随机性
            if not matched:
                sku_name = f"{brand} {category_name} {model} {random.randint(100, 999)}"
            else:
                sku_name = f"{brand} {model} {i+1}"
            
            sku_id = f"10{random.randint(10000000, 99999999)}"
            price = random.randint(50, 5000) # 通用价格范围
            
            products.append({
                "skuId": sku_id,
                "skuName": sku_name,
                "price": price,
                "imageUrl": f"https://img.jd.com/{sku_id}.jpg",
                "shopName": f"{brand}官方旗舰店",
                "comments": random.randint(100, 50000),
                "goodCommentsShare": round(random.uniform(0.90, 0.99), 2),
                "categoryInfo": {
                    "categoryName": category_name
                },
                "brandName": brand,
                "paramJson": json.dumps({
                    "品牌": brand,
                    "型号": model,
                    "商品毛重": f"{random.randint(100, 2000)}g",
                    "产地": "中国大陆"
                }, ensure_ascii=False)
            })
        
        return products
    
    def _parse_products(self, response: Dict[str, Any]) -> List[JDProduct]:
        """解析API响应为商品对象"""
        if response.get("code") != "0":
            return []
        
        products = []
        for item in response.get("data", []):
            product = JDProduct(
                sku_id=item.get("skuId", ""),
                sku_name=item.get("skuName", ""),
                price=float(item.get("price", 0)),
                img_url=item.get("imageUrl", ""),
                shop_name=item.get("shopName", ""),
                comments=item.get("comments", 0),
                good_comments_share=item.get("goodCommentsShare", 0.95),
                category_name=item.get("categoryInfo", {}).get("categoryName", ""),
                brand_name=item.get("brandName", ""),
                param_json=item.get("paramJson", "{}")
            )
            products.append(product)
        
        return products


# 测试代码
if __name__ == "__main__":
    client = JDAPIClient()
    
    # 测试搜索
    products = client.search_products("相机", page_size=5)
    print(f"找到 {len(products)} 个商品：")
    for p in products:
        print(f"- {p.sku_name}: ¥{p.price}")
