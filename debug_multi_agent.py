"""
测试 MultiCategoryAgent 完整多轮对话流程
"""
from multi_agent import MultiCategoryAgent

def test_multi_agent():
    agent = MultiCategoryAgent()
    
    print("=== 第1轮：推荐轻薄本 ===")
    history1 = []
    msg1 = "经常出差，推荐个轻薄本"
    
    result1 = agent.handle_chat(msg1, history1)
    print(f"返回值数量: {len(result1)}")
    reasons1, charts1, results1, analyses1, eco1 = result1
    
    print(f"results1 类型: {type(results1)}")
    if results1:
        print(f"results1 长度: {len(results1)}")
        print(f"产品: {[f'{r.Brand} {r.Model}' for r in results1[:2]]}")
        
        # 构建历史
        assistant_content = f"为您推荐了 {len(results1)} 款产品: " + ", ".join([f"{r.Brand} {r.Model}" for r in results1])
        history2 = [
            {"role": "user", "content": msg1},
            {"role": "assistant", "content": assistant_content}
        ]
        
        print(f"\n=== 第2轮：推荐耳机 ===")
        msg2 = "再推荐个耳机搭配着用"
        
        result2 = agent.handle_chat(msg2, history2)
        print(f"返回值数量: {len(result2)}")
        reasons2, charts2, results2, analyses2, eco2 = result2
        
        print(f"\nreasons2 类型: {type(reasons2)}")
        print(f"results2 类型: {type(results2)}")
        
        if isinstance(reasons2, str):
            print(f"⚠️ reasons2 是字符串: {reasons2[:100]}")
        elif isinstance(reasons2, list):
            print(f"✅ reasons2 是列表，长度: {len(reasons2)}")
            
        if results2:
            print(f"✅ results2 有数据，长度: {len(results2)}")
            print(f"产品: {[f'{r.Brand} {r.Model}' for r in results2[:2]]}")
        else:
            print(f"❌ results2 为空: {results2}")
            print(f"   这将导致测试脚本显示'未找到匹配商品'")
    else:
        print("第1轮就失败了")

if __name__ == "__main__":
    test_multi_agent()
