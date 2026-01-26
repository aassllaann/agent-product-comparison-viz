"""
京东联盟API客户端

京东联盟API文档: https://union.jd.com/openplatform
"""

import json
import hashlib
import time
import requests
import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import config

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
    big_field_info: Optional[Dict] = None # 大字段信息

class JDAPIClient:
    """京东联盟API客户端"""
    
    def __init__(self, app_key: str = None, app_secret: str = None):
        self.app_key = app_key or config.JD_APP_KEY
        self.app_secret = app_secret or config.JD_APP_SECRET
        self.base_url = config.JD_API_URL
        
    def search_products(
        self, 
        keyword: str, 
        page: int = 1, 
        page_size: int = 20,
        price_from: Optional[int] = None,
        price_to: Optional[int] = None,
        sort_type: str = "bfs"  # 排序：desc-倒序,asc-升序; 示例：price_asc, price_desc, commission_asc, commission_desc, sale_desc(30天成交), bfs(综合), new_desc(新品), good_comments(好评)
    ) -> List[JDProduct]:
        """
        搜索商品 (使用 jd.union.open.goods.query)
        """
        # 参数构建
        param_json = {
            "goodsReqDTO": {
                "keyword": keyword,
                "pageIndex": page,
                "pageSize": page_size,
                "sortName": sort_type,
            }
        }
        
        if price_from:
            param_json["goodsReqDTO"]["priceFrom"] = price_from
        if price_to:
            param_json["goodsReqDTO"]["priceTo"] = price_to

        params = {
            "method": "jd.union.open.goods.query",
            "app_key": self.app_key,
            "timestamp": self._get_timestamp(),
            "format": "json",
            "v": "1.0",
            "sign_method": "md5",
            "param_json": json.dumps(param_json)
        }
        
        # 签名并请求
        params["sign"] = self._sign(params)
        response = self._execute_request(params)
        
        return self._parse_search_response(response)

    def get_product_detail(self, sku_id: str) -> Optional[JDProduct]:
        """
        获取商品详情 (使用 jd.union.open.goods.bigfield.query)
        """
        # 尝试简化参数，移除 fields，仅保留 skuIds
        param_json = {
            "goodsReq": {
                "skuIds": [int(sku_id)]
            }
        }
        
        params = {
            "method": "jd.union.open.goods.bigfield.query",
            "app_key": self.app_key,
            "timestamp": self._get_timestamp(),
            "format": "json",
            "v": "1.0",
            "sign_method": "md5",
            "param_json": json.dumps(param_json)
        }
        
        # print(f"[DEBUG] Request Params: {params}") # Debug
        params["sign"] = self._sign(params)
        response = self._execute_request(params)
        
        return self._parse_detail_response(response, sku_id)
    
    def _sign(self, params: Dict[str, Any]) -> str:
        """生成API签名"""
        sorted_params = sorted(params.items())
        
        sign_str = self.app_secret
        for k, v in sorted_params:
            if k != "sign" and v is not None:
                sign_str += f"{k}{v}"
        sign_str += self.app_secret
        
        return hashlib.md5(sign_str.encode("utf-8")).hexdigest().upper()
    
    def _get_timestamp(self) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _execute_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行HTTP请求"""
        try:
            # requests 会自动进行 urlencode
            response = requests.post(self.base_url, data=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[JDAPI] Request Failed: {e}")
            return {}

    def _parse_search_response(self, response: Dict[str, Any]) -> List[JDProduct]:
        """解析搜索响应"""
        # 结构: jd_union_open_goods_query_responce -> queryResult -> data -> [ ... ]
        try:
            # 兼容可能的 key 拼写（responce vs response）如果不确定，通常是 response
            # 但用户提供的示例是 `jd_union_open_goods_bigfield_query_responce` (结尾 ce)
            # 京东官方通常是 response，但有时会有 typo。我们先尝试 response
            root = response.get("jd_union_open_goods_query_response") 
            if not root:
                # 尝试 ce 结尾
                root = response.get("jd_union_open_goods_query_responce")
                
            if not root:
                print(f"[JDAPI] Search response format error: {response.keys()}")
                return []
                
            result = json.loads(root.get("queryResult", "{}"))
            # 注意：result 可能是 string 也可能是 dict，取决于 API 返回
            # 如果是 string 则 load，如果是 dict 则直接用 (requests.json() 可能会自动解析部分，但 queryResult 通常是 JSON string)
            # 不过根据 common behavior, queryResult 是一个 JSON string
            
            if result.get("code") != 200:
                print(f"[JDAPI] Search API error: {result.get('message')}")
                return []
                
            data = result.get("data", [])
            products = []
            if not data:
                return []

            for item in data:
                # 提取图片
                image_info = item.get("imageInfo", {})
                img_list = image_info.get("imageList", [])
                img_url = img_list[0].get("url") if img_list else ""
                
                # 价格信息
                price_info = item.get("priceInfo", {})
                price = price_info.get("price", 0)

                product = JDProduct(
                    sku_id=str(item.get("skuId", "")),
                    sku_name=item.get("skuName", ""),
                    price=float(price),
                    img_url=img_url,
                    shop_name=item.get("shopInfo", {}).get("shopName", ""),
                    comments=item.get("comments", 0),
                    good_comments_share=item.get("goodCommentsShare", 0),
                    category_name=item.get("categoryInfo", {}).get("cid3Name", ""), # 使用三级类目
                    brand_name=item.get("brandCode", ""), # 注意：search 接口可能只返回 brandCode，需要详情或 lookup
                    param_json="{}", # 搜索列表通常不含参数
                )
                products.append(product)
            return products
            
        except Exception as e:
            print(f"[JDAPI] Parse search error: {e}")
            return []

    def _parse_detail_response(self, response: Dict[str, Any], sku_id: str) -> Optional[JDProduct]:
        """解析详情响应 (Big Field)"""
        # 结构: jd_union_open_goods_bigfield_query_responce -> queryResult -> data -> [ ... ]
        try:
            root = response.get("jd_union_open_goods_bigfield_query_response")
            if not root:
                root = response.get("jd_union_open_goods_bigfield_query_responce") # User example spelling
            
            if not root:
                print(f"[JDAPI] Detail response format error: {response.keys()}")
                return None
                
            result_str = root.get("queryResult", "{}")
            if isinstance(result_str, str): # Handle if it's a string
                 result = json.loads(result_str)
            else:
                 result = result_str

            if str(result.get("code")) != "200":
                print(f"[JDAPI] Detail API error: {result.get('message')}")
                return None
            
            data = result.get("data", [])
            if not data:
                return None
                
            # bigfield 接口返回的是列表，通常只有一个（如果我们只查一个SKU）
            # 每个 item 包含 bigFieldGoodsResp 等
            # 根据用户示例: data -> bigFieldGoodsResp -> ...
            # 但用户示例 data 是一个对象。
            # 如果我们传入的是 skuIds 数组，返回的 data 应该是一个列表。
            # 让我们做一些鲁棒性处理。
            
            target_item = None
            if isinstance(data, list):
                if len(data) > 0:
                    target_item = data[0]
            elif isinstance(data, dict):
                 target_item = data # 用户示例情况

            if not target_item:
                return None
                
            # 解析 bigFieldGoodsResp
            big_field = target_item.get("bigFieldGoodsResp", {})
            if not big_field:
                 # 也许直接在 item 里 (depend on API version)
                 big_field = target_item

            # 提取信息
            # User example:
            # "categoryInfo": { "cid1Name":..., "cid3Name":... }
            # "imageInfo": { "imageList": ... }
            # "baseBigFieldInfo": { "propGroups": ... }
            
            cat_info = big_field.get("categoryInfo", {})
            img_info = big_field.get("imageInfo", {})
            
            # Extract Image
            img_url = ""
            img_list = img_info.get("imageList", [])
            if isinstance(img_list, dict): # User example: imageList: { urlInfo: { url: ... } } ?? No, example has imageList containing urlInfo
                 # User example structure: 
                 # "imageList": { "urlInfo": { "url": "..." } } 
                 # Wait, usually imageList is a list. Let's look closely at user example.
                 # "imageList": { "urlInfo": { "url": "..." } } -> This looks like a single object not list.
                 # Let's handle both.
                 url_info = img_list.get("urlInfo", {})
                 img_url = url_info.get("url", "")
            elif isinstance(img_list, list) and len(img_list) > 0:
                 img_url = img_list[0].get("url", "")

            if not img_url:
                img_url = img_info.get("whiteImage", "")

            # Extract Params
            base_info = big_field.get("baseBigFieldInfo", {})
            param_json = base_info.get("propGroups", "{}") # User example says "JSON串"
            
            product = JDProduct(
                sku_id=str(big_field.get("skuId", sku_id)),
                sku_name=big_field.get("skuName", ""),
                price=0.0, # Big field usually doesn't have real-time price, need query interface for that or already have it
                img_url=img_url,
                shop_name=big_field.get("owner", ""), # "g" ? maybe not shop name
                comments=0, # Need other interface
                good_comments_share=0.0,
                category_name=cat_info.get("cid3Name", ""),
                brand_name="", # specific mapping needed
                param_json=param_json,
                big_field_info=big_field
            )
            return product

        except Exception as e:
            print(f"[JDAPI] Parse detail error: {e}")
            return None

if __name__ == "__main__":
    client = JDAPIClient()
    # Test
    # print(client.search_products("手机"))
