
from electronics_agents import HeadphoneAgent
import json

def debug_intent():
    agent = HeadphoneAgent()
    
    # 模拟 Scenario 2 的上下文
    history = [
        {"role": "user", "content": "经常出差，推荐个轻薄本"},
        {"role": "assistant", "content": "为您推荐了 Huawei MateBook X Pro..."}
    ]
    user_msg = "再推荐个耳机搭配着用"
    
    # 调用解析
    print(f"User: {user_msg}")
    print("Parsing intent...")
    
    # 获取 prompt 中使用的 fields_desc
    config = agent.config
    fields_desc = ", ".join([f"{d.field}({d.name})" for d in config.scoring_dimensions])
    
    intent = agent._parse_intent_generic(user_msg, config.name, fields_desc, history)
    
    print("\nParsed Intent:")
    print(json.dumps(intent, indent=2, ensure_ascii=False))
    
    # 模拟 handle_chat 的部分逻辑
    keyword_map = agent.SCENARIO_KEYWORDS
    usage = intent.get('usage', '').lower()
    summary = intent.get('summary', '').lower()
    target_scenario = None
    
    for scenario, keywords in keyword_map.items():
        if any(k in usage for k in keywords) or any(k in summary for k in keywords):
            target_scenario = scenario
            break
            
    print(f"\nTarget Scenario: {target_scenario}")

    if target_scenario:
        candidates = agent._get_preset_products(target_scenario, agent.get_model_class())
        print(f"Preset Candidates: {len(candidates)}")
    else:
        print("No scenario matched.")
        
    # 模拟 Fallback
    results = []
    if not target_scenario or True: # Force check fallback logic
        print("\nChecking Fallback Search...")
        # 模拟 electronics_agents.py 中的 fallback 逻辑
        # specifically checking product_type filter
        model_class = agent.get_model_class()
        product_type = intent.get('product_type')
        print(f"Filter Product Type: {product_type}")
        
        query = agent.db.query(model_class)
        # Type Check
        if product_type and product_type.lower() != "null":
             if hasattr(model_class, 'Type'):
                 print("Applying Type filter...")
                 # query = query.filter(model_class.Type.ilike(f"%{product_type}%"))
                 # We just want to see if it Would match anything
                 all_types = [h.Type for h in agent.db.query(model_class).all()]
                 print(f"Available Types in DB: {set(all_types)}")
                 matches = [t for t in all_types if product_type.lower() in t.lower()]
                 print(f"Matches for '{product_type}': {matches}")

if __name__ == "__main__":
    debug_intent()
