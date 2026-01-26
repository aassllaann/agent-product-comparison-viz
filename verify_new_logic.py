
import sys
import os

# 把当前目录加入 path，以便导入模块
sys.path.append(os.getcwd())

from multi_agent import MultiCategoryAgent
from electronics_agents import PhoneAgent, LaptopAgent

def test_multi_agent():
    print("=== 初始化 MultiCategoryAgent ===")
    agent = MultiCategoryAgent()
    
    print("\n[测试 0] 识别相机品类 (新 CameraAgent)")
    msg = "适合拍Vlog的相机，预算1万"
    replies, charts, results, analyses = agent.handle_chat(msg)
    if results:
        print(f"推荐数量: {len(results)}")
        for i, r in enumerate(results):
            print(f"{i+1}. {r.Brand} {r.Model} (￥{r.Price})")
            if i < len(replies):
                print(f"   理由: {replies[i][:50]}...")
        
        print("\n[图表分析]:")
        for ana in analyses:
            print(f"- {ana}")
    else:
        print("❌ Camera 未返回推荐结果")

    print("\n[测试 1] 识别手机品类")
    msg = "推荐一款拍照好的手机，预算6000左右"
    # 直接调用 parse 看看
    cat, name = agent.detector.detect_category(msg)
    print(f"识别结果: {cat} ({name}) (预期: phone)")
    
    print("\n[测试 2] 调用手机代理推荐 (验证通用分析文案)")
    replies, charts, results, analyses = agent.handle_chat(msg)
    if results:
        print(f"推荐数量: {len(results)}")
        for r in results:
            print(f"- {r.Brand} {r.Model} (￥{r.Price})")
        
        print("\n[图表分析]:")
        for ana in analyses:
            print(f"- {ana}")
    else:
        print("❌ 未返回推荐结果")

    print("\n[测试 3] 识别笔记本品类")
    msg = "适合程序员的笔记本，要轻便"
    replies, charts, results, analyses = agent.handle_chat(msg)
    if results:
        print(f"推荐数量: {len(results)}")
        for r in results:
            print(f"- {r.Brand} {r.Model} (￥{r.Price})")
    else:
        print("❌ 未返回推荐结果")

    print("\n[测试 4] 测试未知品类（动态代理 - 无数据模式）")
    msg = "推荐一款好用的洗面奶"
    # DynamicAgent 应该返回建议文本，results 为空
    advice, charts, results, analyses = agent.handle_chat(msg)
    print(f"返回建议: {advice[:100]}...")
    print(f"Results (预期为空): {len(results) if results else 0}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_multi_agent()
