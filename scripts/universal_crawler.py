import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# 通用商品爬虫，支持京东、天猫等平台
# 可扩展品类：手机、相机、耳机等

class ProductCrawler:
    def __init__(self, keyword, platform="jd", max_pages=3):
        self.keyword = keyword
        self.platform = platform
        self.max_pages = max_pages
        self.results = []

    def crawl_jd(self):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        }
        for page in range(1, self.max_pages + 1):
            url = f"https://search.jd.com/Search?keyword={self.keyword}&page={page}"
            resp = requests.get(url, headers=headers)
            soup = BeautifulSoup(resp.text, "html.parser")
            items = soup.select(".gl-item")
            for item in items:
                name = item.select_one(".p-name em")
                price = item.select_one(".p-price strong")
                link = item.select_one(".p-name a")
                if name and price and link:
                    self.results.append({
                        "name": name.text.strip(),
                        "price": price.text.strip(),
                        "url": "https:" + link['href']
                    })
            time.sleep(1)

    def crawl_tm(self):
        # 天猫爬虫示例（可扩展）
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        }
        for page in range(1, self.max_pages + 1):
            url = f"https://list.tmall.com/search_product.htm?q={self.keyword}&page={page}"
            resp = requests.get(url, headers=headers)
            soup = BeautifulSoup(resp.text, "html.parser")
            items = soup.select(".product")
            for item in items:
                name = item.select_one(".productTitle")
                price = item.select_one(".productPrice")
                link = item.select_one(".productTitle a")
                if name and price and link:
                    self.results.append({
                        "name": name.text.strip(),
                        "price": price.text.strip(),
                        "url": "https:" + link['href']
                    })
            time.sleep(1)

    def run(self):
        if self.platform == "jd":
            self.crawl_jd()
        elif self.platform == "tm":
            self.crawl_tm()
        else:
            raise ValueError("Unsupported platform")
        return self.results

if __name__ == "__main__":
    # 示例：抓取手机、相机、耳机
    keywords = ["手机", "相机", "耳机", "笔记本", "显示器"]
    all_results = []
    for kw in keywords:
        crawler = ProductCrawler(keyword=kw, platform="jd", max_pages=2)
        results = crawler.run()
        for r in results:
            r["category"] = kw
        all_results.extend(results)
    df = pd.DataFrame(all_results)
    df.to_csv("data/new_products.csv", index=False, encoding="utf-8-sig")
    print("已保存最新在售商品数据到 data/new_products.csv")
