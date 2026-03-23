import re

file_path = r"d:\毕业设计\代码\文章part\全文latex存档.md"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

additions = {
    "通用意图解析算法实现": "base_agent.py",
    "基于LLM的动态评分体系构建算法": "category_detector.py",
    "异构商品专有评分维度定义抽象": "electronics_agents.py",
    "双层场景化推荐检索策略算法": "base_agent.py",
    "跨品类生态环境定义与关联结构": "multi_agent.py",
    "系统状态初始化机制": "app.py",
    "大语言模型客户端初始化架构": "base_agent.py",
    "推荐代理基类抽象接口定义": "base_agent.py",
    "核心商品推荐流转控制序列": "electronics_agents.py",
    "具体领域代理（相机）实现示例": "electronics_agents.py",
    "多品类中央路由分发算法": "multi_agent.py",
    "关系型数据库表的结构化表示": "models.py",
    "事实数据约束与动态检索过滤": "base_agent.py",
    "基于状态驱动的视图组件渲染机制": "app.py",
    "多维性能比较雷达图与柱状图渲染算法": "visualizer.py"
}

count = 0
def replacer(match):
    global count
    caption_content = match.group(1)
    if caption_content in additions:
        count += 1
        return f"caption={{{caption_content}（对应文件：{additions[caption_content]}）}}"
    return match.group(0)

new_content = re.sub(r"caption=\{([^\}]+)\}", replacer, content)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(new_content)

with open(r"d:\毕业设计\代码\script_out.txt", "w", encoding="utf-8") as f:
    f.write(f"Updated {count} captions.")
