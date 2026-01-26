"""
淘宝联盟API客户端 (OneBound API Impl)

此模块使用 OneBound 提供的 Taobao API 接口。
Compatible for python2.x and python3.x
"""

import json
import time
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class TaobaoProduct:
    """淘宝商品数据模型"""
    num_iid: str
    title: str
    reserve_price: float  # 原价
    zk_final_price: float # 折扣价
    pict_url: str
    shop_title: str
    volume: int           # 30天销量
    item_url: str
    category_name: str
    nick: str             # 卖家昵称
    # 扩展属性
    item_description: str  # 简短描述


class TaobaoAPIClient:
    """OneBound Taobao API客户端"""
    
    def __init__(self, app_key: str = "t8840868711", app_secret: str = "87119902"):
        self.app_key = app_key
        self.app_secret = app_secret
        self.base_url = "https://api-gw.onebound.cn/taobao/"
        
    def search_products(
        self, 
        keyword: str, 
        page: int = 1, 
        page_size: int = 20,
        sort: str = "total_sales_des"  # total_sales_des, price_asc, price_des
    ) -> List[TaobaoProduct]:
        """
        搜索商品 (使用 item_search API)
        """
        # Map internal sort keys to API sort keys
        # _sale: 销量 desc (default implicit?)
        # bid2: 价格 asc
        # _bid2: 价格 desc
        sort_map = {
            "total_sales_des": "_sale",
            "price_asc": "bid2",
            "price_des": "_bid2",
            "tk_rate_des": "" # Not supported directly or unknown
        }
        
        api_sort = sort_map.get(sort, "")
        
        params = {
            "key": self.app_key,
            "secret": self.app_secret,
            "q": keyword,
            "page": page,
            "page_size": page_size,
            "sort": api_sort,
            "cat": "0", # all categories
            "is_promotion": "1" # Assume we want promo info
        }
        
        try:
            response = self._request("item_search", params)
            return self._parse_search_response(response)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_product_detail(self, num_iid: str) -> Optional[TaobaoProduct]:
        """
        获取商品详情 (使用 item_get API)
        """
        params = {
            "key": self.app_key,
            "secret": self.app_secret,
            "num_iid": num_iid,
            "is_promotion": "1"
        }
        
        try:
            response = self._request("item_get", params)
            item_data = response.get("item")
            if item_data:
                return self._parse_single_item(item_data)
            return None
        except Exception as e:
            logger.error(f"Get details failed: {e}")
            return None
    
    def _request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """发送API请求"""
        url = f"{self.base_url}{endpoint}/"
        headers = {
            "Accept-Encoding": "gzip",
            "Connection": "close"
        }
        
        logger.info(f"Requesting {url} with params keys: {list(params.keys())}")
        
        r = requests.get(url, params=params, headers=headers)
        r.raise_for_status()
        data = r.json()
        
        # Log response status
        if data.get("error"):
            logger.error(f"API Error: {data.get('error')} - Reason: {data.get('reason')}")
        
        return data
    
    def _parse_search_response(self, response: Dict[str, Any]) -> List[TaobaoProduct]:
        """解析搜索结果"""
        products = []
        # OneBound item_search can return items in several structures
        # Structure 1: {"items": {"item": [...]}}
        items_wrapper = response.get("items", {})
        if isinstance(items_wrapper, dict):
            items = items_wrapper.get("item", [])
        elif isinstance(items_wrapper, list):
            # Structure 2: {"items": [...]}
            items = items_wrapper
        else:
            items = []
            
        if not items:
            logger.info(f"No items found in response keys: {list(response.keys())}")
            if "items" in response:
                logger.debug(f"Items content type: {type(response['items'])}")
            return []
            
        for item in items:
            product = self._parse_single_item(item)
            if product:
                products.append(product)
                
        return products

    def _parse_single_item(self, item: Dict[str, Any]) -> TaobaoProduct:
        """解析单个商品数据"""
        # API fields:
        # title, pic_url, price, orginal_price, num_iid, sales, nick, detail_url
        
        try:
            # Price handling
            price = float(item.get("price", 0) or 0)
            org_price = float(item.get("orginal_price", 0) or price)
            
            # Volume handling
            # detailed response has 'sales', 'total_sold', 'num'
            sales_str = item.get("sales", "0")
            if not sales_str and "total_sold" in item:
                sales_str = item["total_sold"]
            
            # Clean sales string (e.g. "1000+" -> 1000)
            sales = 0
            if sales_str:
                import re
                nums = re.findall(r'\d+', str(sales_str))
                if nums:
                    sales = int(nums[0])
            
            return TaobaoProduct(
                num_iid=str(item.get("num_iid", "")),
                title=item.get("title", ""),
                reserve_price=org_price,
                zk_final_price=price,
                pict_url=item.get("pic_url", "") or item.get("pic", ""),
                shop_title=item.get("shop_name") or item.get("nick", ""), # item_search uses nick, item_get has shopinfo
                volume=sales,
                item_url=item.get("detail_url", f"https://item.taobao.com/item.htm?id={item.get('num_iid')}"),
                category_name=item.get("rootCatId", ""), # API returns ID mainly
                nick=item.get("nick", ""),
                item_description=item.get("desc_short", "")
            )
        except Exception as e:
            logger.warning(f"Failed to parse item: {e}")
            return None

if __name__ == "__main__":
    # Test
    client = TaobaoAPIClient()
    # print("Testing Details...")
    # Using the ID from user example
    detail = client.get_product_detail("652874751412") 
    if detail:
        print(f"Title: {detail.title}")
        print(f"Price: {detail.zk_final_price}")
        print(f"Shop: {detail.shop_title}")
    else:
        print("Failed to get details (Note: Public keys might be expired/limited)")
