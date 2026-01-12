import sys
import os

# 确保能导入模块
sys.path.append(os.getcwd())

from multi_agent import MultiCategoryAgent

def test_new_category():
    print("=== 测试新品类支持 ===")
    agent = MultiCategoryAgent()
    
    # 测试一个完全全新的品类
    user_msg = "推荐一款好用的钓鱼竿"
    print(f"\n用户输入: {user_msg}")
    
    # 1. 识别
    key, name = agent.detector.detect_category(user_msg)
    print(f"识别结果: key={key}, name={name}")
    
    # 2. 对话处理
    print("正在调用代理处理...")
    try:
        reply, charts, results, analyses = agent.handle_chat(user_msg)
        
        print("\n--- 处理结果 ---")
        if isinstance(reply, list):
            print(f"推荐理由数量: {len(reply)}")
            print(f"第一条理由预览: {reply[0][:50]}...")
        else:
            print(f"回复: {reply}")
            
        print(f"\n商品数量: {len(results) if results else 0}")
        if results:
            first = results[0]
            print(f"第一款商品: {first.get('brand')} {first.get('model')} - ¥{first.get('price')}")
            print(f"规格示例: {first.get('specs')}")
            print(f"评分示例: {first.get('scores')}")
            
        print(f"\n图表数量: {len(charts) if charts else 0}")
        print(f"分析数量: {len(analyses) if analyses else 0}")
        if analyses:
            print(f"分析示例: {analyses[0]}")
            
        if results and len(results) > 0:
            print("\n✅ 测试通过：成功处理新品类")
        else:
            print("\n❌ 测试失败：无结果")
            
    except Exception as e:
        print(f"\n❌ 发生异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_new_category()
