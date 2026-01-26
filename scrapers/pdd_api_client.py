"""
拼多多（多多进宝）API客户端（Mock实现）

此模块模拟拼多多开放平台API的响应格式。
真实API接入时，需替换 _mock_request 为真实HTTP请求。

拼多多API: pdd.ddk.goods.search
"""

import json
import hashlib
import time
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PddProduct:
    """拼多多商品数据模型"""
    goods_id: int
    goods_name: str
    min_group_price: int  # 最小拼团价（单位：分）
    min_normal_price: int # 最小单买价（单位：分）
    goods_image_url: str
    mall_name: str
    sales_tip: str        # 销量（如 "10万+"）
    goods_sign: str       # 商品加密ID
    category_name: str
    brand_name: str
    # 扩展属性
    goods_desc: str


class PddAPIClient:
    """拼多多开放平台API客户端"""
    
    def __init__(self, client_id: str = "mock_pdd_id", client_secret: str = "mock_pdd_secret"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://gw-api.pinduoduo.com/api/router"
        
    def search_products(
        self, 
        keyword: str, 
        page: int = 1, 
        page_size: int = 20,
        sort_type: int = 0  # 0-综合排序; 1-按佣金比率升序; 2-按佣金比率降序; 3-按价格升序; 4-按价格降序; 5-按销量升序; 6-按销量降序
    ) -> List[PddProduct]:
        """
        搜索商品
        """
        params = {
            "type": "pdd.ddk.goods.search",
            "client_id": self.client_id,
            "timestamp": str(int(time.time())),
            "data_type": "JSON",
            "version": "V1",
            "keyword": keyword,
            "page": page,
            "page_size": page_size,
            "sort_type": sort_type,
            "with_coupon": "false"  # 示例参数
        }
        
        # 签名
        params["sign"] = self._sign(params)
        
        # Mock实现
        response = self._mock_request(params)
        
        return self._parse_products(response)
    
    def get_product_detail(self, goods_sign: str) -> Optional[PddProduct]:
        """获取商品详情 (Mock:复用搜索接口模拟)"""
        # pdd.ddk.goods.detail
        params = {
            "type": "pdd.ddk.goods.detail",
            "client_id": self.client_id,
            "timestamp": str(int(time.time())),
            "goods_sign_list": f"[{goods_sign}]"
        }
        
        params["sign"] = self._sign(params)
        response = self._mock_request(params)
        
        products = self._parse_products(response)
        return products[0] if products else None
    
    def _sign(self, params: Dict[str, Any]) -> str:
        """生成API签名"""
        # 拼多多签名规则：secret + key1value1key2value2... + secret -> MD5 -> Upper
        sorted_params = sorted(params.items())
        sign_str = self.client_secret
        for k, v in sorted_params:
            if k != "sign":
                sign_str += f"{k}{v}"
        sign_str += self.client_secret
        return hashlib.md5(sign_str.encode()).hexdigest().upper()
    
    def _mock_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """模拟API请求"""
        method = params.get("type")
        
        if method == "pdd.ddk.goods.detail":
            # 模拟详情响应
            # 这里简单起见，mock一个商品
            mock_data = self._generate_mock_data("商品", 1)
            return {
                "goods_detail_response": {
                    "goods_list": mock_data
                }
            }
            
        else:
            # 模拟搜索响应
            keyword = params.get("keyword", "")
            page_size = int(params.get("page_size", 20))
            mock_data = self._generate_mock_data(keyword, page_size)
            
            return {
                "goods_search_response": {
                    "goods_list": mock_data,
                    "total_count": 1000
                }
            }
    
    def _generate_mock_data(self, keyword: str, count: int) -> List[Dict]:
        """生成模拟商品数据 (拼多多格式)"""
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
        
        matched = False
        category_name, brands, models = (keyword, ["品牌A", "品牌B"], ["型号Lite", "型号Pro"])
        
        for key, data in category_map.items():
            if key in keyword:
                category_name, brands, models = data
                matched = True
                break
        
        if not matched:
            category_name = keyword[:8]
            brands = ["拼多多精选", "百亿补贴", "品牌好货", "工厂直供"]
            models = ["爆款", "新款", "热销", "推荐"]

        products = []
        for i in range(min(count, 20)):
            brand = brands[i % len(brands)]
            model = models[i % len(models)]
            
            if not matched:
                title = f"{brand} {category_name} {model} {random.randint(100,999)}"
            else:
                title = f"{brand} {model} {i+1} 国行正品"
            
            # PDD 特有的 goods_id (数字)
            goods_id = random.randint(10000000000, 99999999999)
            goods_sign = f"c9{random.randint(1000,9999)}..." # 模拟加密ID
            
            price_yuan = random.randint(50, 5000)
            min_group_price = price_yuan * 100 # 分
            min_normal_price = int(price_yuan * 1.2 * 100)
            
            sales_num = random.randint(100, 100000)
            sales_tip = f"{sales_num}" if sales_num < 10000 else f"{sales_num//10000}万+"
            
            products.append({
                "goods_id": goods_id,
                "goods_name": title,
                "goods_image_url": f"https://t00img.yangkeduo.com/goods/images/{random.randint(2023,2025)}/{random.randint(1,12)}/{random.randint(1,30)}/{goods_id}.jpg",
                "min_group_price": min_group_price,
                "min_normal_price": min_normal_price,
                "mall_name": f"{brand}官方旗舰店",
                "sales_tip": sales_tip,
                "goods_sign": goods_sign,
                "category_name": category_name,
                "brand_name": brand,
                "goods_desc": f"{brand} {model} 现货速发 正品保障 假一赔十"
            })
        
        return products
    
    def _parse_products(self, response: Dict[str, Any]) -> List[PddProduct]:
        """解析API响应"""
        products = []
        
        # 处理搜索结果
        if "goods_search_response" in response:
            data = response["goods_search_response"].get("goods_list", [])
        # 处理详情结果
        elif "goods_detail_response" in response:
            data = response["goods_detail_response"].get("goods_list", [])
        else:
            data = []
            
        for item in data:
            product = PddProduct(
                goods_id=item.get("goods_id", 0),
                goods_name=item.get("goods_name", ""),
                min_group_price=int(item.get("min_group_price", 0)),
                min_normal_price=int(item.get("min_normal_price", 0)),
                goods_image_url=item.get("goods_image_url", ""),
                mall_name=item.get("mall_name", ""),
                sales_tip=item.get("sales_tip", "0"),
                goods_sign=item.get("goods_sign", ""),
                category_name=item.get("category_name", ""),
                brand_name=item.get("brand_name", ""),
                goods_desc=item.get("goods_desc", "")
            )
            products.append(product)
            
        return products

if __name__ == "__main__":
    client = PddAPIClient()
    products = client.search_products("相机", page_size=5)
    print(f"找到 {len(products)} 个商品：")
    for p in products:
        print(f"- {p.goods_name}: ¥{p.min_group_price/100:.2f} (销量:{p.sales_tip})")
