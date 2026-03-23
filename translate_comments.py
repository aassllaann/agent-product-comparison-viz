import codecs
import re

file_path = r"d:\毕业设计\代码\文章part\全文latex存档.md"

with codecs.open(file_path, "r", "utf-8") as f:
    text = f.read()

replacements = [
    (r"// e\.g\., \"Portability\"", "// 例如: \"Portability\""),
    (r"// e\.g\., \"Portability_Score\"", "// 例如: \"Portability_Score\""),
    (r"// e\.g\., 0\.3", "// 例如: 0.3"),
    (r"// Dimension explanation", "// 维度解释说明"),
    (r"// First Layer: Expert Knowledge Scenario Recall", "// 第一层：专家知识场景召回"),
    (r"// Second Layer: Fallback Database Retrieval", "// 第二层：全库降级检索兜底"),
    (r"// Ensure critical components persist across interactions", "// 确保关键组件在多次交互中持久存在"),
    (r"// Setup connection to the Large Language Model service", "// 初始化与大语言模型服务的连接"),
    (r"// Verify connection status", "// 验证连接状态"),
    (r"// Shared universal dimensions for all products", "// 所有商品共享的通用评分维度"),
    (r"// Abstract methods to be implemented by specific category agents", "// 需由具体品类代理实现的抽象方法"),
    (r"// Concrete flow pipeline template", "// 具体的推荐流转管道模板"),
    (r"// Step 1: Semantic Intent Extraction", "// 步骤 1: 语义意图提取"),
    (r"// Step 2: Extract hard constraints and soft preferences", "// 步骤 2: 提取硬性约束与软性偏好"),
    (r"// Step 3: Candidate Generation & Ranking", "// 步骤 3: 候选商品集生成与排序"),
    (r"// Step 4: Fallback Mechanism \(if results < 3\)", "// 步骤 4: 兜底机制 (若结果少于3款)"),
    (r"// Step 5: Explainability & Visualization Generation", "// 步骤 5: 可解释性生成与可视化构建"),
    (r"// Brand identifier", "// 品牌标识"),
    (r"// Product model name", "// 产品型号名称"),
    (r"// Current market price", "// 当前市场价格"),
    (r"// Release year", "// 发布年份"),
    (r"// Path to product image", "// 产品图片路径"),
    (r"// Sensor resolution", "// 传感器分辨率"),
    (r"// Full Frame, APS-C, etc\.", "// 画幅类型（全画幅、APS-C等）"),
    (r"// Device weight", "// 设备重量"),
    (r"// 4K video recording capability", "// 4K视频录制支持情况"),
    (r"\(Calculated offline\)", "(离线计算得到)"),
    (r"// Construct base query", "// 构建基础查询"),
    (r"// Apply hard constraint: Budget Barrier", "// 应用硬性约束：预算上限"),
    (r"// Apply exclusion filter", "// 应用排除过滤（剔除已推荐项）"),
    (r"// Apply dynamic sorting based on LLM extracted intention", "// 依据大模型提取的意图应用动态排序"),
    (r"// Extract properties", "// 提取商品属性"),
    (r"// Generate specialized parameter UI", "// 生成专属参数UI组件"),
    (r"// Construct Interactive Element", "// 构建交互式页面DOM元素"),
    (r"// Inject LLM generated reason", "// 注入大模型生成的推荐理由"),
    (r"// 1\. Render Radar Chart \(Holistic Evaluation\)", "// 1. 渲染雷达图（用于全局整体评估）"),
    (r"// Close polygon for radar", "// 闭合雷达图多边形边界"),
    (r"// 2\. Render Bar Chart \(Relative Competitiveness\)", "// 2. 渲染柱状图（用于相对竞争力对比）")
]

for old_pat, new_str in replacements:
    text = re.sub(old_pat, new_str, text, count=0)

with codecs.open(file_path, "w", "utf-8") as f:
    f.write(text)

with codecs.open(r"d:\毕业设计\代码\success_flag.txt", "w", "utf-8") as f:
    f.write("DONE")
