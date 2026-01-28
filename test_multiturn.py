"""
多轮对话集成测试脚本
验证系统是否能根据历史上下文（Chat History）进行跨品类的生态化推荐
"""
from multi_agent import MultiCategoryAgent
import time

def print_separator(title):
    print(f"\n{'='*20} {title} {'='*20}")

def simulate_conversation(agent, inputs, test_name):
    print_separator(f"测试场景: {test_name}")
    history = []
    
    for i, user_input in enumerate(inputs):
        print(f"\n[第 {i+1} 轮对话] 用户: {user_input}")
        
        # 调用 Agent
        reasons, charts, results, analyses, eco_suggestion = agent.handle_chat(user_input, history)
        
        # 1. 识别并打印主要推荐
        if results:
            category_name = results[0].__class__.__name__
            print(f"🤖 Agent 响应 ({category_name}):")
            for idx, r in enumerate(results[:2]): # 只打前两个
                # 尝试打印关键规格以验证场景（如刷新率、分辨率）
                specs = ""
                if hasattr(r, "Refresh_Rate_Hz"):
                    specs += f" | {r.Refresh_Rate_Hz}Hz"
                if hasattr(r, "Resolution"):
                    specs += f" | {r.Resolution}"
                if hasattr(r, "TDP_W"):
                    specs += f" | {r.TDP_W}W"
                
                print(f"   {idx+1}. {r.Brand} {r.Model} (￥{r.Price}){specs}")
        else:
            print(f"🤖 Agent 响应: 未找到匹配商品或回复文本: {reasons if isinstance(reasons, str) else 'No data'}")

        # 2. 打印生态建议
        if eco_suggestion:
            print(f"🔗 生态建议: {eco_suggestion}")
        
        # 3. 更新历史记录 (模拟 App 行为)
        # 注意：这里我们简单地将推荐理由列表转为字符串存入历史，真实 App 可能会更复杂
        assistant_content = ""
        if isinstance(reasons, list):
            assistant_content = f"为您推荐了 {len(results)} 款产品: " + ", ".join([f"{r.Brand} {r.Model}" for r in results])
        elif isinstance(reasons, str):
            assistant_content = reasons
            
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": assistant_content})
        
        # 简单的停顿
        time.sleep(0.5)

def main():
    agent = MultiCategoryAgent()

    # --- 场景 1: PS5 游戏生态 ---
    # 预期：
    # Round 1: 推荐游戏主机
    # Round 2: 推荐显示器 -> 应该是高刷、4K、游戏显示器 (识别到 Gaming 上下文)
    conversation1 = [
        "想买个这种游戏主机玩黑神话悟空",
        "给我推荐个配套的显示器"
    ]
    simulate_conversation(agent, conversation1, "1. 游戏主机 -> 游戏显示器联动")

    # --- 场景 2: 商务办公生态 ---
    # 预期：
    # Round 1: 推荐轻薄本
    # Round 2: 推荐耳机 -> 应该是降噪、通话好的耳机 (识别到 Office/Commute 上下文)
    conversation2 = [
        "经常出差，推荐个轻薄本",
        "再推荐个耳机搭配着用"
    ]
    simulate_conversation(agent, conversation2, "2. 商务轻薄本 -> 降噪耳机联动")

    # --- 场景 3: 没有任何上下文的对比 ---
    # 预期：没有上下文时，"推荐个显示器" 应该推荐综合或办公类，或者询问需求
    conversation3 = [
        "推荐个显示器"
    ]
    simulate_conversation(agent, conversation3, "3. 无上下文基准测试")

if __name__ == "__main__":
    main()
