"""
商品数据爬虫模块

提供多品类商品数据的获取能力：
- 预置热门品类数据
- 实时搜索刷新
- 缓存管理
"""

from .base_scraper import BaseScraper, ProductData
from .jd_scraper import JDScraper

__all__ = ["BaseScraper", "ProductData", "JDScraper"]
