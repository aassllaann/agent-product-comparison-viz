import sys
import re

path = r"d:\毕业设计\代码\文章part\全文latex存档.md"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Make a backup just in case
with open(path + ".bak", "w", encoding="utf-8") as f:
    f.write(content)

# Title replacements
content = re.sub(r'\\section\{需求分析与方法设计\}', r'\\section{智能商品推荐系统}', content)
content = re.sub(r'\\subsection\{需求层面分析\}', r'\\subsection{需求分析}', content)

# Insert 方法设计
content = re.sub(r'\\subsection\{系统总体框架设计\}', r'\\subsection{方法设计}\n\n\\subsubsection{系统总体框架设计}', content)

# Downgrade 总体框架设计 subsubsections
content = re.sub(r'\\subsubsection\{系统整体运行流程\}', r'\\paragraph{系统整体运行流程}', content)
content = re.sub(r'\\subsubsection\{四层模块架构设计\}', r'\\paragraph{四层模块架构设计}', content)
content = re.sub(r'\\subsubsection\{动态领域扩展机制\}', r'\\paragraph{动态领域扩展机制}', content)

# Remove old summary 1
s1_regex = r'\\subsection\{本章小结\}\s*本章首先通过问卷调研.*?基础。'
content = re.sub(s1_regex, '', content, flags=re.DOTALL)

# Delete section 2 and downgrade its subsections
content = re.sub(r'\\section\{核心算法设计与系统设计\}\s*', '', content)
content = re.sub(r'\\subsection\{用户需求语义解析与参数提取\}', r'\\subsubsection{用户需求语义解析与参数提取}', content)
content = re.sub(r'\\subsection\{异构商品多维评分模型\}', r'\\subsubsection{异构商品多维评分模型}', content)
content = re.sub(r'\\subsection\{双层场景化推荐引擎\}', r'\\subsubsection{双层场景化推荐引擎}', content)
content = re.sub(r'\\subsection\{跨品类生态推荐算法\}', r'\\subsubsection{跨品类生态推荐算法}', content)

# Remove old summary 2
s2_regex = r'\\subsection\{本章总结\}\s*本章围绕系统核心算法设计.*?支撑。'
content = re.sub(s2_regex, '', content, flags=re.DOTALL)

# Change section 3 to subsection
content = re.sub(r'\\section\{系统架构实现与代码落地\}', r'\\subsection{系统实现}', content)
content = re.sub(r'\\subsection\{全栈基础架构与运行环境实现\}', r'\\subsubsection{全栈基础架构与运行环境实现}', content)
content = re.sub(r'\\subsection\{多 Agent 推荐模块实现\}', r'\\subsubsection{多 Agent 推荐模块实现}', content)
content = re.sub(r'\\subsection\{商品数据层实现与数据清洗\}', r'\\subsubsection{商品数据层实现与数据清洗}', content)
content = re.sub(r'\\subsection\{用户界面与交互式可视化实现\}', r'\\subsubsection{用户界面与交互式可视化实现}', content)

# Replace old summary 3
s3_regex = r'\\subsection\{本章小结\}\s*本章从系统工程实现角度.*?有效实现。'
new_s3 = r'\\subsection{本章小结}\n\n本章围绕智能商品推荐系统进行了完整的论述。首先通过问卷调研进行了需求分析，明确了核心痛点与系统功能目标；随后完成了系统的方法设计，包括总体架构框架设计、需求语义解析、多维评分模型、场景化推荐以及生态推荐算法；最后介绍了系统的具体落地实现，详细说明了全栈环境、多Agent推荐模块、数据层建设与交互式可视化界面的实现方案。总体而言，本章实现了从理论设计到工程实现的完整闭环，验证了系统的可行性与实用性。'
content = re.sub(s3_regex, new_s3, content, flags=re.DOTALL)

# Clean up multiple newlines
content = re.sub(r'\n{3,}', '\n\n', content)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("Modification complete.")
