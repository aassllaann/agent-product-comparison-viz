import os

file_path = r"d:\毕业设计\代码\文章part\全文latex存档.md"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

replacements = {
    "caption={通用意图解析算法实现}": r"caption={通用意图解析算法实现（对应文件：base\_agent.py）}",
    "caption={基于LLM的动态评分体系构建算法}": r"caption={基于LLM的动态评分体系构建算法（对应文件：category\_detector.py）}",
    "caption={异构商品专有评分维度定义抽象}": r"caption={异构商品专有评分维度定义抽象（对应文件：electronics\_agents.py）}",
    "caption={双层场景化推荐检索策略算法}": r"caption={双层场景化推荐检索策略算法（对应文件：base\_agent.py）}",
    "caption={跨品类生态环境定义与关联结构}": r"caption={跨品类生态环境定义与关联结构（对应文件：multi\_agent.py）}",
    "caption={系统状态初始化机制}": r"caption={系统状态初始化机制（对应文件：app.py）}",
    "caption={大语言模型客户端初始化架构}": r"caption={大语言模型客户端初始化架构（对应文件：base\_agent.py）}",
    "caption={推荐代理基类抽象接口定义}": r"caption={推荐代理基类抽象接口定义（对应文件：base\_agent.py）}",
    "caption={核心商品推荐流转控制序列}": r"caption={核心商品推荐流转控制序列（对应文件：electronics\_agents.py）}",
    "caption={具体领域代理（相机）实现示例}": r"caption={具体领域代理（相机）实现示例（对应文件：electronics\_agents.py）}",
    "caption={多品类中央路由分发算法}": r"caption={多品类中央路由分发算法（对应文件：multi\_agent.py）}",
    "caption={事实数据约束与动态检索过滤}": r"caption={事实数据约束与动态检索过滤（对应文件：base\_agent.py）}",
    "caption={关系型数据库表的结构化表示}": r"caption={关系型数据库表的结构化表示（对应文件：models.py）}",
    "caption={基于状态驱动的视图组件渲染机制}": r"caption={基于状态驱动的视图组件渲染机制（对应文件：app.py）}",
    "caption={多维性能比较雷达图与柱状图渲染算法}": r"caption={多维性能比较雷达图与柱状图渲染算法（对应文件：visualizer.py）}"
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Captions updated successfully.")
