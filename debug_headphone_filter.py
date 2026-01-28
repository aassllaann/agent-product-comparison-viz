"""
Debug 脚本：分析为什么耳机推荐在场景2中失败
"""
from electronics_agents import HeadphoneAgent
from models import Headphone
import json

def debug_filter():
    agent = HeadphoneAgent()
    
    # 模拟场景2的上下文
    history = [
        {"role": "user", "content": "经常出差，推荐个轻薄本"},
        {"role": "assistant", "content": "为您推荐了 Huawei MateBook X Pro..."}
    ]
    user_msg = "再推荐个耳机搭配着用"
    
    # 获取意图
    config = agent.get_category_config()
    fields_desc = ", ".join([f"{d.field}({d.name})" for d in config.scoring_dimensions])
    intent = agent._parse_intent_generic(user_msg, config.name, fields_desc, history)
    
    print("=== 意图解析 ===")
    print(json.dumps(intent, indent=2, ensure_ascii=False))
    
    # 场景匹配
    keyword_map = agent.SCENARIO_KEYWORDS
    usage = intent.get('usage', '').lower()
    summary = intent.get('summary', '').lower()
    target_scenario = None
    
    print(f"\nUsage: {usage}")
    print(f"Summary: {summary}")
    print("\n=== 场景匹配过程 ===")
    
    for scenario, keywords in keyword_map.items():
        matched = []
        for k in keywords:
            if k in usage or k in summary:
                matched.append(k)
        if matched:
            print(f"{scenario}: 匹配到 {matched}")
            if not target_scenario:
                target_scenario = scenario
    
    print(f"\n✅ 最终匹配场景: {target_scenario}")
    
    # 获取预设产品
    if target_scenario:
        candidates = agent._get_preset_products(target_scenario, Headphone)
        print(f"\n=== 预设候选 ({len(candidates)} 款) ===")
        for c in candidates[:5]:
            print(f"  {c.Brand} {c.Model} - ￥{c.Price} - {c.Type}")
        
        # 进行过滤
        print(f"\n=== 过滤过程 (max_price={intent.get('max_price')}) ===")
        filtered = agent._filter_and_sort(candidates, intent, Headphone)
        
        print(f"过滤后剩余: {len(filtered)} 款")
        for f in filtered[:3]:
            print(f"  {f.Brand} {f.Model} - ￥{f.Price}")
        
        if len(filtered) == 0 and len(candidates) > 0:
            print("\n⚠️ 警告：所有候选都被过滤了！")
            print("检查过滤条件...")
            
            # 检查价格过滤
            max_price = intent.get('max_price', 20000)
            over_budget = [c for c in candidates if c.Price and c.Price > max_price]
            print(f"  价格超预算的: {len(over_budget)} 款")
            
            # 检查类型过滤
            product_type = intent.get('product_type')
            if product_type and product_type.lower() != "null":
                print(f"  要求的 product_type: {product_type}")
                type_mismatch = []
                for c in candidates:
                    if hasattr(c, 'Type') and c.Type:
                        if product_type not in c.Type:
                            type_mismatch.append(f"{c.Model}({c.Type})")
                print(f"  类型不匹配的: {len(type_mismatch)} 款")
                if type_mismatch:
                    print(f"    示例: {type_mismatch[:3]}")
    
    # 尝试兜底搜索
    print("\n=== 兜底搜索 ===")
    fallback_results = agent._fallback_search(intent, Headphone, [])
    print(f"兜底找到: {len(fallback_results)} 款")
    for r in fallback_results[:3]:
        print(f"  {r.Brand} {r.Model} - ￥{r.Price}")

if __name__ == "__main__":
    debug_filter()
