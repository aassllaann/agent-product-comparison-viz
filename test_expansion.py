"""
扩展集成测试脚本：验证产品品类扩展、边界条件及复杂场景建议
"""
from multi_agent import MultiCategoryAgent
import json

def test_recommendation(query, test_name):
    print(f"\n{'='*20} 测试: {test_name} {'='*20}")
    print(f"用户输入: {query}")
    
    agent = MultiCategoryAgent()
    try:
        reasons, charts, results, analyses, eco_suggestion = agent.handle_chat(query)
        
        if results:
            # 识别品类名称
            print(f"✅ 匹配品类: {results[0].__class__.__name__}")
            print(f"✅ 推荐产品: {[f'{r.Brand} {r.Model} (￥{r.Price})' for r in results]}")
            
            # 检查生态系统建议
            if eco_suggestion:
                print("✅ 发现生态系统建议")
                print(f"   详情: {eco_suggestion}")
            else:
                print("ℹ️ 未发现生态系统建议")
        else:
            print("❓ 未能识别品类或数据库无匹配数据")
            
    except Exception as e:
        print(f"❌ 测试运行出错: {e}")

def main():
    # 1. 基础扩展品类验证 (之前已测，快速回顾)
    test_recommendation("推荐几款适合跑步的智能手表", "1. 基础功能测试 (手表)")
    
    # 2. 严苛预算边界测试
    test_recommendation("推荐一款500元以内的智能手表，要有心率监测", "2. 低预算边界测试")
    
    # 3. 极高性能与品牌偏好
    test_recommendation("我就要性能最强的英伟达显卡，不差钱", "3. 极致性能与特定品牌")
    
    # 4. 复杂多场景组合测试 (出差 + 办公 + 娱乐)
    test_recommendation("作为程序员经常要出差，推荐一些便携的办公和听歌装备", "4. 复杂多场景测试")
    
    # 5. 模糊需求与多品类重叠 (屏幕/视频场景)
    test_recommendation("给推荐个屏幕好的设备，平时想在床上看电影用", "5. 模糊需求测试")
    
    # 6. 具体配套硬件驱动场景
    test_recommendation("刚买了PS5，推荐个能完美适配的显示器，预算3000左右", "6. 配套硬件驱动测试")
    
    # 7. 负面/冲突测试 (在音箱上要求潜水功能)
    test_recommendation("推荐一款能潜水50米的蓝牙音箱", "7. 奇葩/冲突需求测试")
    
    # 8. 原有品类回归测试 (确保老品类也能触发生态)
    test_recommendation("推荐一款拍照好看的手机，平时喜欢去旅游", "8. 回归测试 + 生态触发")

if __name__ == "__main__":
    main()
