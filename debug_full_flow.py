"""
完整端到端测试：模拟真实的多轮对话调用
"""
from electronics_agents import HeadphoneAgent

def test_headphone_full_flow():
    agent = HeadphoneAgent()
    
    # 模拟场景2的完整上下文
    history = [
        {"role": "user", "content": "经常出差，推荐个轻薄本"},
        {"role": "assistant", "content": "为您推荐了 2 款产品: Huawei MateBook X Pro 2025, Huawei MateBook X Pro 2024"}
    ]
    user_msg = "再推荐个耳机搭配着用"
    
    print("=== 完整调用 handle_chat ===")
    print(f"用户输入: {user_msg}")
    print(f"历史记录: {len(history)} 条\n")
    
    # 调用完整流程
    result = agent.handle_chat(user_msg, history)
    
    print(f"\n=== 返回值分析 ===")
    print(f"返回值类型: {type(result)}")
    print(f"返回值长度: {len(result) if isinstance(result, (list, tuple)) else 'N/A'}")
    
    if isinstance(result, tuple):
        reasons, charts, results, analyses = result
        
        print(f"\n1. reasons 类型: {type(reasons)}")
        if isinstance(reasons, list):
            print(f"   reasons 长度: {len(reasons)}")
            if reasons:
                print(f"   第一个: {reasons[0][:100] if len(reasons[0]) > 100 else reasons[0]}")
        else:
            print(f"   reasons 内容: {reasons}")
        
        print(f"\n2. charts: {charts is not None}")
        
        print(f"\n3. results 类型: {type(results)}")
        if results:
            print(f"   results 长度: {len(results)}")
            print(f"   产品列表:")
            for r in results[:3]:
                print(f"     - {r.Brand} {r.Model} (￥{r.Price})")
        else:
            print(f"   ⚠️ results 为空或 None！")
            
        print(f"\n4. analyses 类型: {type(analyses)}")
        if analyses:
            print(f"   analyses 长度: {len(analyses)}")
    else:
        print(f"⚠️ 返回值不是 tuple: {result}")

if __name__ == "__main__":
    test_headphone_full_flow()
