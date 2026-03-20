\section{核心算法设计与系统设计}

\subsection{用户需求语义解析与参数提取}

在智能商品选购系统中，用户需求的准确理解是推荐流程的起点。现实场景下，消费者通常通过自然语言表达需求，而这些表达往往具有明显的口语化特征，例如“我想买一台不差钱、能打游戏还能做后期修图的电脑”。此类描述同时包含主观评价和多个使用场景，因此难以直接转换为数据库查询条件。

虽然大语言模型（Large Language Model，LLM）在语义理解方面具有显著优势，但在长文本推理过程中仍可能出现注意力漂移或语义扩散问题。如果缺乏约束机制，模型在解析需求时可能生成不稳定甚至虚构的参数，从而影响推荐系统的可靠性。

为此，系统设计了一种基于思维链（Chain of Thought, CoT）的提示工程方法，并结合结构化 JSON Schema 对模型输出进行约束。在单轮交互中，模型需要完成关键参数的结构化提取。首先，系统识别用户输入中的核心使用场景，并提取为字段 \texttt{usage}，用于匹配系统内部的场景知识库（Golden Sets）。其次，将带有主观含义的价格描述转换为可计算的预算参数，例如将“不差钱”等表达映射为系统允许的最高预算值 \texttt{max\_price = 999999}。随后，模型根据用户需求推断其关注的性能维度，并生成排序字段 \texttt{sort\_field}。最后，对用户需求进行简要语义归纳，生成字段 \texttt{summary} 作为推荐解释文本。

\begin{lstlisting}[language=Python, caption={通用意图解析系统提示词构建}, label={code:intent_prompt}]
def _parse_intent_generic(self, user_msg: str, category_name: str, fields_desc: str) -> dict:
    system_rules = f"""
    你是一个{category_name}导购专家，负责从对话中提取结构化信息。

    【待提取维度（用户均可缺省）】：
    1. usage (用途): 使用场景关键词
    2. budget_level (投入): 预算描述
    3. max_price (预算): 数字，单位元，默认 100000 (无限)
    4. sort_field (排序字段): 基于用户需求选择最合适的评分字段 ({fields_desc})
    5. summary (摘要): 解析出的核心诉求
    6. product_type (类型): 显式要求的特定类型
    7. brand (品牌): 用户明确指出想要购买的品牌名
    8. owned_items (已购设备): 用户提到已经拥有的产品品类或型号

    【推理规则】：
    - 预算缺省默认为 20000。
    - 如果用户提到"不差钱"、"旗舰"、"顶配"，max_price 设为 999999。

    输出严格 JSON：{{"max_price": 数字, "sort_field": "字段名", "summary": "核心诉求", "usage": "场景关键词", "product_type": "...", "brand": "品牌", "owned_items": []}}
    """
    # 调用 LLM 进行 JSON 结构化提取
\end{lstlisting}

此外，为解决系统预设商品类别有限的问题，系统设计了动态品类补全机制（Dynamic Category Scoring System）。当核心识别模块 \texttt{CategoryDetector} 检测到用户需求涉及未在系统预定义类别中的商品时，系统将进入知识推理模式，由模型自动生成该品类的评价维度。例如在厨房电器场景中，系统可能生成“功率”“容量”“温控能力”等评价指标，从而构建新的评分体系。该机制使系统能够在无需预先配置数据结构的情况下扩展新的商品类别。

\begin{lstlisting}[language=Python, caption={基于大语言模型的动态评分体系构建}, label={code:dynamic_scoring}]
def build_scoring_system(self, category_key: str, category_name: str) -> Dict:
    system_prompt = f"""
    你是一个{category_name}领域专家。请为该品类设计一套评分体系和使用场景。
    
    要求：
    1. 设计 3-5 个核心评分维度（如性能、续航、便携性等）
    2. 每个维度需要：中文名称、英文字段名(xxx_Score格式)、权重、描述
    3. 设计 3-5 个典型使用场景及对应的推荐关键词
    
    输出 JSON 格式：
    {{
        "dimensions": [
            {{"name": "性能", "field": "Performance_Score", "weight": 0.3, "description": "..."}}
        ],
        "scenarios": {{
            "gaming": {{"keywords": ["游戏", "电竞"], "presets": ["推荐品牌关键词"]}}
        }},
        "default_sort_field": "主排序字段名"
    }}
    """
    # 调用 LLM 生成并缓存动态体系
\end{lstlisting}

\subsection{异构商品多维评分模型}

在完成用户需求解析后，系统需要对数据库中的商品进行统一的性能评价。然而，不同类别电子产品的参数体系差异较大，例如手机关注摄像头和电池容量，而耳机更强调频响范围和降噪能力。因此，不同类别之间难以直接进行性能比较。

为解决该问题，系统构建了双层评价维度体系。第一层为通用评价维度（Universal Dimension），用于描述所有商品共享的基础指标，例如价格水平和综合性价比（Value Score）。所有参与推荐排序的商品均需要具备该层评价数据，从而形成统一比较基准。

第二层为专有性能维度（Specific Dimension），用于描述不同产品类别的核心性能指标。例如在相机代理模块 \texttt{CameraAgent} 中，系统建立了“便携性评分（Portability Score）”“弱光性能（LowLight Score）”和“视频能力（Video Score）”等指标；在显卡代理 \texttt{GPUAgent} 中，则使用“游戏性能（Gaming Score）”和“散热表现（Thermal Score）”等指标。

\begin{lstlisting}[language=Python, caption={异构商品专有评分维度定义}, label={code:specific_dimensions}]
class CameraAgent(BaseDbAgent):
    def get_specific_dimensions(self) -> List[ScoringDimension]:
        return [
            ScoringDimension("便携性", "Portability_Score", 0.3, "如重量体积"),
            ScoringDimension("低光画质", "LowLight_Score", 0.4, "暗光表现"),
            ScoringDimension("视频能力", "Video_Score", 0.3, "视频拍摄能力")
        ]

class GPUAgent(BaseDbAgent):
    def get_category_config(self) -> CategoryConfig:
        return CategoryConfig(
            # ...
            scoring_dimensions=[
                ScoringDimension("游戏性能", "Gaming_Score", 0.35, "游戏帧率表现"),
                ScoringDimension("创作性能", "Creative_Score", 0.25, "渲染与AI加速"),
                ScoringDimension("功耗散热", "Thermal_Score", 0.2, "温度与噪音"),
                ScoringDimension("性价比", "Value_Score", 0.2, "综合性价比")
            ],
            # ...
        )
\end{lstlisting}

由于商品数据来源多样，不同参数的量纲和格式存在差异，系统在数据处理中引入了多类型归一化策略（Normalization Strategy）。对于布尔型参数，系统采用固定权值映射；对于离散枚举变量，根据性能等级设置分档权重；对于连续数值变量，则采用 Min-Max 归一化方法进行标准化处理。

归一化后，各类指标通过加权求和（Weighted Sum）方式计算综合得分，并统一映射至 $0$--$100$ 的评分区间。该评分不仅用于推荐排序，也为前端可视化模块提供稳定的数据结构，使系统能够生成雷达图等多维性能展示图形。

\subsection{双层场景化推荐引擎}

在商品评分体系建立之后，推荐系统需要从大量商品中召回最符合用户需求的候选结果。如果仅依赖数值排序，往往难以体现数码产品选购中的经验知识；而完全依赖固定规则，又可能在预算约束较严格时出现无推荐结果的情况。

为提高系统稳定性，本研究设计了双层场景化推荐引擎。第一层为专家知识预设机制（Golden Sets）。系统将常见消费场景与优质机型建立映射关系，例如“电竞游戏”“户外 Vlog 拍摄”“高保真音乐欣赏”等。当用户需求中的字段 \texttt{usage} 与这些场景匹配时，系统优先在对应候选机型集合中进行筛选，从而利用领域经验快速缩小搜索范围。

然而，由于市场价格变化或预算限制，预设机型可能被全部排除。为避免推荐结果为空，系统设计了降级检索机制（Fallback Search）。当预设集合在预算过滤后剩余商品数量不足时，系统自动触发全库检索。此时推荐算法不再依赖场景规则，而是根据用户最关注的排序字段 \texttt{sort\_field} 对数据库商品进行排序，并结合预算条件筛选新的候选商品。

最终，系统通过集合并集与去重策略整合两类结果，确保每次推荐均能输出 $3$ 至 $5$ 款不同商品，从而在保证推荐质量的同时提高系统鲁棒性。

\begin{lstlisting}[language=Python, caption={双层场景化推荐检索策略}, label={code:two_layer_search}]
def handle_chat_generic(self, user_msg, history, model_class, keyword_map):
    # 1. 解析意图 (获取 max_price, sort_field, usage 等)
    intent = self._parse_intent_generic(user_msg, config.name, fields_desc, history)
    
    # 2. 匹配预设场景 target_scenario
    # ...
    
    # 3. 第一层：专家知识场景召回 (Golden Sets)
    results = []
    if target_scenario:
        candidates = self._get_preset_products(target_scenario, model_class)
        results = self._filter_and_sort(candidates, intent, model_class)[:3]
        
    # 4. 第二层：降级全库检索兜底 (Fallback Search)
    if len(results) < 3:
        exclude = [i.id for i in results]
        more = self._fallback_search(intent, model_class, exclude)
        results.extend(more)
        results = results[:3]
        
    return results
\end{lstlisting}

\subsection{跨品类生态推荐算法}

随着智能设备种类的增加，用户在购买单一设备时往往同时关注设备之间的协同体验。例如，在选择笔记本电脑时，用户可能还需要显示器、蓝牙音箱或无线鼠标等配套设备。因此，系统在核心推荐流程之外设计了跨品类生态推荐模块（Eco-system Suggestion）。

该模块首先构建若干设备生态主题集合，例如“移动办公生态”和“家庭影音娱乐生态”。每个主题包含一个核心设备类别以及若干关联设备类别。例如在移动办公场景中，笔记本电脑作为核心设备，而显示器和蓝牙扬声器等设备作为扩展组件。

在算法实现上，系统采用关键词权重评分机制对用户需求进行生态分类。例如与办公场景高度相关的关键词（如“编程”“出差”）被赋予较高权重，而“文档”等通用词汇则赋予较低权重。系统根据关键词累计得分判断用户所属的设备生态类别。

在确定生态类别后，系统结合用户已有设备列表 \texttt{owned\_items} 进行过滤，以避免推荐重复设备。例如当用户已拥有显示器时，系统将优先推荐其他辅助设备。最终，系统通过语言模型生成简要推荐说明，并附加在主推荐结果之后，从而形成完整的设备组合建议。

\begin{lstlisting}[language=Python, caption={跨品类生态环境定义与关联规则}, label={code:ecosystem_config}]
ECOSYSTEMS = {
    "移动办公": {
        "name_en": "mobile_office",
        "description": "适合移动办公、远程协作的数码产品组合",
        "core_categories": ["laptop"],  # 核心设备
        "related_categories": ["monitor", "bluetooth_speaker", "smartwatch"],  # 关联外设
        "scenarios": ["办公", "会议", "出差", "远程工作"],
        "keywords": {
            "high": ["办公", "工作", "商务", "出差", "会议", "远程"],  # 高权重关键词
            "medium": ["文档", "表格", "演示", "ppt"]  # 中权重关键词
        }
    },
    "专业游戏": {
        "name_en": "professional_gaming",
        "core_categories": ["laptop", "gaming_console", "gpu"],
        "related_categories": ["monitor", "headphone"],
        "scenarios": ["3A游戏", "电竞", "主机游戏", "PC游戏"],
        # ...
    }
}
\end{lstlisting}

\subsection{本章总结}

本章围绕系统核心算法设计，对需求语义解析、商品评分模型、推荐引擎以及跨品类生态推荐等关键模块进行了系统分析。通过引入思维链提示工程与结构化约束机制，系统能够稳定地将用户自然语言需求转换为结构化参数。通过构建通用维度与专有维度相结合的评分体系，解决了不同类别商品难以统一评价的问题。

在推荐策略方面，系统提出了结合专家知识与降级检索的双层推荐架构，从而在保证推荐质量的同时提升系统稳定性。最后，通过跨品类生态推荐机制，系统在单一商品推荐的基础上扩展了设备组合建议能力，为后续系统实现与前端展示提供了重要支撑。