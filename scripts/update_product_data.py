import pandas as pd
import os

# 数据处理脚本：筛选、合并、更新各品类商品数据
# 1. 剔除已停产/仅二手商品（根据新爬取数据）
# 2. 合并新数据，补充新款
# 3. 更新本地 CSV 文件

def filter_and_update(category_csv, new_csv, output_csv):
    # 读取原始数据
    df_old = pd.read_csv(category_csv, encoding='gbk')
    # 读取新爬取数据
    df_new = pd.read_csv(new_csv, encoding='utf-8-sig')
    # 只保留新数据中在售商品
    new_names = set(df_new['name'])
    df_filtered = df_old[df_old['名称'].isin(new_names)]
    # 合并新数据（补充新款）
    df_merged = pd.concat([df_filtered, df_new], ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset=['name'], keep='last')
    # 保存
    df_merged.to_csv(output_csv, index=False, encoding='gbk')
    print(f"已更新数据文件：{output_csv}")

if __name__ == "__main__":
    # 示例：批量处理所有品类
    categories = [
        ('data/phone_data.csv', 'data/new_products.csv', 'data/phone_data_updated.csv'),
        ('data/camera_data_clean4.csv', 'data/new_products.csv', 'data/camera_data_updated.csv'),
        ('data/headphone_data.csv', 'data/new_products.csv', 'data/headphone_data_updated.csv'),
        ('data/laptop_data.csv', 'data/new_products.csv', 'data/laptop_data_updated.csv'),
        ('data/monitor_data.csv', 'data/new_products.csv', 'data/monitor_data_updated.csv'),
    ]
    for cat_csv, new_csv, out_csv in categories:
        if os.path.exists(cat_csv) and os.path.exists(new_csv):
            filter_and_update(cat_csv, new_csv, out_csv)
