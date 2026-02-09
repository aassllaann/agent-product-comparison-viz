import os
import csv

data_dir = r'd:\毕业设计\代码\data'
print(f"{'Filename':<30} | {'Count':<5} | {'Years Range'}")
print("-" * 60)

for filename in os.listdir(data_dir):
    if filename.endswith('.csv'):
        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                years = [int(r['Year']) for r in rows if r.get('Year') and r['Year'].isdigit()]
                year_range = f"{min(years)}-{max(years)}" if years else "N/A"
                print(f"{filename:<30} | {len(rows):<5} | {year_range}")
        except Exception as e:
            print(f"{filename:<30} | Error: {e}")
