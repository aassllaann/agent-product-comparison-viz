
% =============================
\section{绪论}
% =============================
随着电子商务的快速发展，网络购物已深度融入人们的日常生活。据统计，中国网络零售市场规模已突破十万亿元，平台上在售商品数量动辄以亿计。商品数量的爆炸式增长虽然带来了更丰富的选择，却也让消费者在面对庞大的商品列表、繁杂的规格参数以及质量不一的用户评价时，常常难以迅速判断哪些商品真正符合自己的需求。研究显示，当信息量过大时，用户的认知负担会显著增加，决策效率随之下降，甚至可能出现决策回避或购买后的懊悔情绪。心理学上的"选择悖论"（Paradox of Choice）早已表明：选项越多，幸福感反而可能越低，决策质量也未必提升。换句话说，信息越多，决策反而越难。

这一问题在数码品类中尤为突出。以智能手机为例，消费者在选购时需要综合比较处理器架构、屏幕刷新率、相机传感器尺寸、电池容量与快充协议、系统生态等多个维度，而每个维度本身又有大量行业术语与品牌话术的包装，进一步提高了普通用户的理解门槛。如果用户同时关注多个价位段的商品，所需处理的信息量将以指数级增长，决策周期往往从数小时延长至数天，甚至因无所适从而放弃购买。

现有的推荐方式难以破解这一困境。\textbf{第一}，传统协同过滤依赖历史行为数据，对用户当下的即时意图理解不足，且在新用户、新商品场景下面临严峻的冷启动挑战；其推荐逻辑本质上是"你和相似的人买了什么"，而非"这款商品是否真的适合你现在的需求"。\textbf{第二}，基于大语言模型的纯对话式推荐虽然能理解自然语言意图，但由于缺乏与实时、权威商品库的深度绑定，容易产生"幻觉"——推荐出并不存在的型号、过时的参数或无法核实的性能描述，反而增加了用户的信任成本。\textbf{第三}，内容型测评平台（如知乎、什么值得买）信息丰富，创作者的深度测评具有较高参考价值，但此类内容依赖个体的主观经验，覆盖品类和更新频率均受限，用户仍需投入大量时间自行阅读、筛选与归纳，且不同测评之间的评价标准并不统一，横向比较困难。\textbf{第四}，平台自带的筛选与排序功能虽然支持按参数过滤，却无法理解诸如"拍照好用、续航够用、价格不超过三千"这类融合偏好与约束的自然语言表达，用户依然需要手动将模糊需求转化为精确筛选条件。

上述四类方式分别在"理解自然语言即时需求""提供可靠结构化事实依据"和"以可视化方式辅助多维决策"三项关键能力上存在不同程度的缺失，且至今尚无一套成熟方案能够将这三项能力有机整合。这一空缺，正是本研究所要填补的核心技术挑战。

在上述背景下，本文提出并实现了一套基于多Agent架构的智能商品选购推荐系统。系统的核心设计思路在于：将大语言模型对自然语言的理解能力，与结构化商品知识库中明确、可验证的商品数据深度结合，再通过三级分工的Agent路由体系，使用户一句自然语言的需求能够被准确拆解、精准匹配，并最终以清晰直观的可视化方式呈现推荐结果。系统支持10大数码品类的精准推荐与未知品类的动态扩展，每次推荐给出三款候选商品并附带推荐理由与优缺点分析；同时通过多维雷达图与分组柱状图对候选商品进行可视化对比，将抽象的规格数据转化为直观的差异图景，有效降低用户的认知负担与决策门槛。

\textbf{各章概述}

第二章对推荐系统、多 Agent 技术以及可解释性人机交互的相关研究进行综述；第三章依据问卷调研结果开展需求分析，并给出包含数据采集流程在内的系统总体架构；第四章围绕双层推荐引擎、多维度评分机制和生态化推荐方法等核心算法展开说明；第五章介绍系统的具体工程实现；第六章通过功能测试与用户评价对系统进行验证；第七章对全文工作进行总结，并提出后续可能的优化方向。

% =============================
\section{相关工作}
% =============================

\subsection{推荐系统研究现状}

\subsubsection{传统推荐算法的原理与局限}

现有的推荐方法大致可以分为三类，各有其形成背景与适用范围。

协同过滤（Collaborative Filtering，CF）是推荐系统中最早成型、也最具代表性的一类方法。无论是基于用户还是基于物品的实现，它们都建立在同一个朴素但有效的假设上：相似的用户会喜欢相似的物品\cite{Schafer2007CollaborativeFiltering}。矩阵分解方法的引入显著提升了协同过滤在大规模数据上的表现，SVD++ 等模型在工业实践中得到广泛应用\cite{Koren2009MatrixFactorization}，让这一路线在工业界的实践相对成熟，尤其是在数据规模足够大、用户偏好稳定时，往往能给出相当可靠的结果。然而，这类方法本质上依赖历史交互，因此在数据稀疏的情况下往往力不从心；当新用户或新商品刚进入系统、缺乏足够行为记录时，“冷启动”问题会格外明显\cite{Adomavicius2005RecommenderSurvey}。此外，协同过滤并不善于理解用户当下的即时意图。例如，两位用户都可能浏览过同一类笔记本电脑，但其中一位正在寻找轻薄便携款，另一位关注的是游戏性能，这类需求差异不会自然地体现在历史行为里，也超出了协同过滤的辨别能力。

基于内容的过滤（Content-Based Filtering）试图绕开对其他用户行为的依赖，通过分析物品属性和用户过往偏好建立匹配关系\cite{Pazzani2007ContentBasedRS}，因此在用户规模较小时也能保持稳定性能。它的挑战在于对特征工程的高度依赖：人工构建的特征往往难以覆盖商品的全部关键要素，且维护成本高；系统很容易陷入“越推越窄”的过滤气泡，难以拓展用户的潜在兴趣。此外，面对自然语言中隐含的偏好描述，例如“想要一个适合旅行用、充电久一点的相机”，传统内容过滤缺乏足够的语义理解能力，往往无法有效处理，因此常被质疑“听不懂人话”。

基于知识的推荐（Knowledge-Based Recommendation）则通过领域规则或知识图谱来约束匹配空间，在如数码产品、房产、汽车等高价值、低频决策场景中表现尤为稳健\cite{Burke2000KnowledgeBasedRS}\cite{Burke2002HybridRS}。它的优势在于决策链条清晰、可解释性强，可以明确指出某个推荐结果背后的逻辑依据。但这类方法的构建门槛较高：规则需要密集的领域知识支撑，维护难度也随商品更新而增加；同时，系统面对开放式自然语言的表达缺乏弹性，扩展性相对不足。总体而言，它适合结构化、逻辑性明确的任务，却难以应对用户真实咨询场景中那种复杂、多变、带有模糊需求的对话式偏好表述。

\subsubsection{基于大语言模型的推荐研究进展}

近年来，大语言模型（LLM）的出现为推荐系统带来了一条全新的技术路线。以 TALLRec、LLM4Rec 等模型为代表的一系列研究尝试将 LLM 置于推荐流程的“语义中枢”位置\cite{Bao2023TALLRec}\cite{Zhang2023CollaborativeLLM4Rec}，让用户以自然语言表达的需求不再需要被手工拆解成标签或关键词，而是能够直接转化为更具语义结构的查询条件。例如，用户说“想买一台适合剪视频、预算八千左右的笔记本电脑”，模型可以自动从中提取预算、性能需求、用途场景等信息，进一步对接检索流程。与此同时，当缺乏用户历史行为时，LLM 还能通过语义推断生成合理的初步偏好解释，为缓解冷启动提供了新的技术可能\cite{Bao2023TALLRec}\cite{Zhang2023CollaborativeLLM4Rec}，使得推荐不再完全依赖已有交互记录。

然而，将 LLM 直接引入推荐并非没有代价。最典型的问题是幻觉生成：模型可能会编造不存在的商品型号，误述产品规格\cite{Zhang2023CollaborativeLLM4Rec}，甚至在品类属性之间做出错误归纳。在数码消费等高度依赖真实参数和可验证信息的场景中，这些错误往往会导致明显的决策风险\cite{Ji2023HallucinationSurvey}，用户难以容忍。此外，LLM 给出的推荐理由虽然往往听起来流畅自然，但并不总能清晰映射回可信的数据来源，用户无法判断其逻辑是否基于真实商品信息还是模型的“想当然”，导致实际应用上难以完全依赖。

\subsubsection{现有研究的空白}

综合现有工作，可以看到几个尚未被充分解决的问题。其一，许多方法仍停留在“大模型接入推荐流程”的宏观层面，缺乏对具体使用情境的细致设计，难以让领域知识真正融入推荐链路，导致在专业类场景（如数码、家电、户外装备等）表现不稳定。其二，跨品类的一致化路由机制仍不成熟：不同类别对商品属性的表达方式、用户关注点、约束条件差异显著，系统很难保持统一的理解与响应能力，导致推荐质量随品类波动。其三，缺少能够有效依托结构化数据的稳健机制，用以约束模型的生成边界、抑制幻觉，并确保每一次推荐都能基于真实、可验证的商品信息。

本文提出的架构正是围绕这些空白展开：通过在语义理解、品类路由和数据约束之间建立更紧密的联动机制，希望在真实使用场景中构建一个既能发挥大模型表达力，又具备可验证性与稳定性的推荐系统框架。

\subsection{多 Agent 系统研究}

\subsubsection{Agent 的基本概念与理论基础}

在人工智能研究中，Agent 通常被描述为能够感知环境、自主作出判断并执行行动的软件实体\cite{RussellNorvigAIMA4}。这一概念的核心在于 Agent 通过感知、推理与行动三个环节形成一个持续闭环：感知负责收集外部环境的状态变化；推理阶段整合内部模型与经验，对输入信息进行判断；执行环节将推理结果落实为具体动作，从而影响环境并进入下一轮循环。基于这一框架，研究者逐渐形成了不同的 Agent 设计思路。

反应式 Agent 强调对环境变化的快速响应，通常基于预设规则或简单的刺激–反应机制，在要求实时性高但情境较为明确的场景中表现稳定。相对而言，慎思式 Agent 更倾向于先构建内部模型，再通过规划、策略选择等步骤做出决策，适用于任务复杂、目标多元的情境。这两类 Agent 在灵活性、复杂度以及可解释性上呈现不同取向，也构成了后续多 Agent 系统设计的基本背景。

\subsubsection{多 Agent 系统的分布式协作机制}

多 Agent 系统（Multi-Agent System, MAS）通过让多个具有自主能力的 Agent 协同工作，为处理复杂、动态或跨领域任务提供了一种可扩展的解决方案\cite{Wooldridge2009MultiAgentSystems}。在这一框架下，系统通常采用分布式任务拆分和角色划分的方式，让不同 Agent 在相对独立的职责范围内协作完成整体任务。这样不仅提升了系统在规模扩张时的可扩展性，也增强了面对局部故障或信息不确定性时的鲁棒性。

在众多协作模式中，层级式结构尤为常见。上层的协调 Agent 负责理解任务、判断意图并进行路由，而下层的领域 Agent 则依据专业知识处理更细粒度的问题。由于职责边界清晰，这一架构便于集成与扩展\cite{Stone2000MASSurvey}，与本文采用的“MultiCategoryAgent + 品类专家 Agent”模式天然吻合。

随着大语言模型的发展，面向多 Agent 协作的工程化框架也逐渐成熟。例如 LangChain、LangGraph 等系统通过路由链（Router Chain）、计划–执行结构以及工具调用接口，使开发者能够更方便地构建多 Agent 之间的信息流动和任务依赖。这些框架强调模块化与可组合性，让不同 Agent 能够在统一协议下协同处理复杂任务，为大型推荐系统的构建提供了实际可行的工程路径。

\subsubsection{多 Agent 架构在推荐系统中的应用潜力}

近年来，随着对话式推荐、个性化推荐与可解释推荐的发展，多 Agent 架构开始在推荐系统领域展现出明显的优势。对话式场景中，用户的完整需求往往在多次对话中慢慢形成，多 Agent 系统中的对话 Agent、意图识别 Agent 和领域专家 Agent 能组建分工明确的协作链路，使用户的语义表达能够被不断细化并映射到更清晰的商品偏好结构。

在跨品类推荐中，传统单体模型往往难以同时处理数码、家电、户外装备等类别之间差异巨大的属性体系，而多 Agent 架构能够让每个品类由相应的专家 Agent 负责处理，使推荐逻辑更具一致性和可解释性。同时，工具调用能力让 Agent 可以在对话过程中实时访问结构化数据库、知识图谱或外部 API，从而有效弥补大模型在知识时效性与事实准确性上的不足。

一系列工作证明了多 Agent 在复杂情境下的潜力。Park 等人在 UIST 2023 提出的 Generative Agents 框架展示了具备记忆、反思与社交行为的类人 Agent 如何形成稳定协作\cite{Park2023GenerativeAgents}；而 MetaGPT\cite{Hong2023MetaGPT}通过角色分配与任务流水线的方式，实现了更偏工程化的大规模协作能力。这些成果说明，在需要多轮推理、跨领域知识整合或实时信息检索的任务中，多 Agent 具备良好的适应性。这些特性为构建更稳健、可解释、可扩展的推荐系统提供了重要的参考方向，也为本文提出的系统架构奠定了理论与工程基础。

\subsection{可视化辅助的 LLM 人机交互}

随着大语言模型逐渐被应用于推荐与决策支持场景，单纯依赖文本生成的交互方式逐渐暴露出一定局限。一方面，模型的推理过程对用户而言难以观察，推荐结果缺乏透明度；另一方面，当系统返回多个候选方案时，用户需要在不同属性之间进行比较，也容易产生较高的认知负担。同时，对于需求尚不明确的用户，传统对话界面往往缺乏有效的引导机制。近年来的研究开始尝试将信息可视化方法引入 LLM 交互界面，通过图形化方式辅助理解模型推理过程、比较推荐结果，并在交互过程中提供适当引导。围绕这一思路，相关工作大体可以从三个方面展开：推理过程的解释性呈现、推荐结果的可视化比较以及交互过程中的视觉引导机制。

\subsubsection{推理过程的可解释性可视化}

对于普通用户而言，大语言模型的推理过程通常难以直接观察，推荐结论往往以最终文本形式呈现。这种“黑盒式”输出在一定程度上影响了用户对系统结果的信任。可解释性人工智能（Explainable AI, XAI）研究正是试图缓解这一问题，其中一种常见思路是通过可视化方式呈现模型决策的依据或推理路径\cite{Guidotti2018SurveyXAI}，使用户能够理解推荐结论形成的基本逻辑。

Wang 等人在对话推荐系统研究中发现，如果系统能够以结构化步骤展示推荐理由，例如“因用户提及‘轻巧’，过滤掉重量超过 1.5kg 的机型”，用户对系统的信任度指标（Trust Score）明显高于仅给出文本推荐的情况\cite{Wang2018RippleNet}。Strobelt 等人在 CHI 会议提出的注意力权重热力图可视化工具，则通过标注模型在输入文本中的关注区域，使非专业用户也能够直观地看到模型关注的关键词位置\cite{Strobelt2018LSTMVis}，这为 LLM 推荐系统的解释性设计提供了新的思路。另一类研究尝试将思维链（Chain-of-Thought, CoT）推理过程进行图形化表达\cite{Wei2022ChainOfThought}，例如将逐步推理转换为流程节点或关系结构，从而使模型的推理过程能够以更直观的方式呈现。这类可视化形式不仅有助于系统开发者调试模型，也为用户理解与监督模型行为提供了可能。

\subsubsection{多维推荐结果的可视化比较}

在实际推荐场景中，系统往往会返回多款候选产品。用户需要在价格、性能、重量、续航等多个维度之间进行比较，如果仅通过文本描述呈现信息，往往难以快速形成整体判断。信息可视化研究表明，合适的图表形式能够降低用户在多属性信息处理过程中的认知负荷（Cognitive Load），并有助于提升决策效率\cite{Amar2005AnalyticActivity}。

Amar 等人\cite{Amar2005AnalyticActivity}在信息可视化研究中总结了用户分析多维数据时常见的认知任务，包括数值检索、条件筛选、趋势识别以及异常值判断等。这些任务为面向决策的图表设计提供了重要参考。在多属性比较场景中，**极坐标雷达图（Radar Chart）**常用于展示不同对象在多个指标上的整体表现，它能够在同一视图中呈现各产品在不同维度上的相对优势与劣势，从而帮助用户快速获得整体印象\cite{Zhang2023RadarChartImprove}。相比之下，当分析重点集中在某一具体维度时，**分组柱状图（Grouped Bar Chart）**通常能够提供更清晰的数值对比效果\cite{Amar2005AnalyticActivity}。在对话式推荐系统中，这类图表如果与 LLM 生成的推荐理由结合使用，可以同时提供数据层面的客观比较与语言层面的解释说明，从而形成更加完整的决策支持信息结构。


\subsubsection{对话交互中的视觉引导机制}

在许多实际使用情境中，用户在初次提出问题时往往并未形成明确需求。如果界面仅提供自由文本输入，用户可能难以准确表达自己的意图，进而导致多轮低效对话。为改善这一问题，人机交互（HCI）领域开始尝试在对话界面中加入可视化交互元素，通过轻量级的视觉提示帮助用户逐步明确需求。

Amershi 等人\cite{Amershi2014InteractiveML}在 IUI 会议的研究指出，在对话界面中加入结构化视觉提示，例如约束条件标签或意图确认卡片，可以在交互过程中帮助用户逐步细化需求，从而减少不必要的对话轮次。Cai 等人\cite{Cai2024GenerativeUI}提出的生成式 UI（Generative UI）概念则进一步扩展了这一思路，即根据当前对话上下文动态生成不同的界面组件，例如筛选条件、参数滑块或结果卡片。这种动态界面形式在探索型信息检索任务中表现出较高的用户满意度和任务完成率。与此同时，界面视觉设计本身也会影响用户对系统的初始感知。例如“玻璃拟态（Glassmorphism）”等现代界面风格以及平滑的过渡动效，已被相关可用性研究证明能够在一定程度上降低用户接触 AI 系统时的心理门槛，并提升感知可用性（Perceived Usability）与整体交互体验\cite{Baylor2002AgentPersona}。


\section{智能商品推荐系统}


\subsection{需求分析与方法设计}

\subsubsection{需求层面分析}

在设计智能商品推荐系统之前，有必要首先了解用户在电商选购过程中所面临的实际问题。为此，本文围绕数码电子产品选购体验开展了初步问卷调研，共回收有效问卷20份，调研内容主要涵盖信息获取、产品比较以及决策判断等环节的使用体验。

调研结果显示，用户在购买数码产品时普遍需要同时参考多个信息渠道。约85\%的受访者表示，在查询商品信息时通常需要在京东、B站、小红书等平台之间来回切换，由于各平台的信息呈现方式不统一，用户往往不得不手动整理参数并自行完成横向对比，这一过程显著拉长了决策周期。与此同时，约80\%的受访者认为网络评测内容普遍带有一定主观倾向，部分文章或视频夹杂个人偏好甚至商业推广因素，使得用户难以从中提取真正客观的产品信息。此外，部分技术参数本身具有一定的理解门槛——"传感器单像素尺寸""阻抗"等专业术语对普通用户而言较难直观把握，约40\%的受访者明确表示希望系统能够通过图表等可视化方式辅助理解关键参数的差异。

上述问题共同指向三类功能需求。在输入层面，系统需要能够理解用户较为口语化的需求表达，例如"3000元拍视频的相机"这类描述实际上同时包含预算范围、使用场景与产品类型等信息，系统须借助大语言模型完成语义解析与槽位填充，从自然语言中提取结构化参数。在处理层面，调研显示大多数用户希望系统在一次交互中给出3至5款重点产品并附带简要推荐理由，因此系统需要基于客观参数完成数据筛选，并生成具有解释性的推荐说明，而非简单罗列商品列表。在输出层面，系统应提供可视化展示以帮助用户理解产品间的性能差异；考虑到数码产品日益形成生态化使用趋势，系统还应能够提供跨品类设备的搭配建议。综合而言，系统设计需要兼顾通用性与客观性，并以关系型数据库作为底层事实来源，以减少大模型在推荐过程中可能产生的幻觉问题。


\subsubsection{系统总体框架设计}

在明确上述需求之后，本文进一步构建了支撑这些功能的整体架构。系统以用户输入自然语言请求为起点，通过多Agent协同机制完成全流程任务处理。

用户输入首先由前置模块进行语义识别，通过关键词检测或调用大语言模型完成场景分类；识别出具体商品类别后，中央调度模块结合当前会话上下文将任务路由至对应的领域专家Agent。领域Agent随后解析用户需求，提取预算范围、使用场景等关键信息，并通过ORM访问底层关系型数据库，先按硬性参数筛选，再结合场景规则匹配，生成候选商品列表。在确定候选产品之后，系统调用大语言模型生成推荐理由，同时将评分数据传递至可视化模块，用于生成雷达图或分组柱状图，最终以图文结合的形式呈现于前端界面。

在软件实现层面，系统整体架构划分为四个相对独立的模块层级。最外层为表象展示与会话层，负责前端界面展示和用户会话管理，动态加载图表组件与商品信息卡片；第二层为路由与意图解析层，将自然语言输入转换为结构化参数，通过对模型输出格式加以约束，保证业务层能够直接读取相关数据；第三层为业务决策与领域代理层，执行场景匹配、评分计算与候选产品排序等核心推荐逻辑，并生成最终的推荐说明；最底层为实体映射与事实数据层，存储结构化商品参数数据，目前覆盖多个数码产品类别并包含数千条参数记录，严格的数据结构设计有效约束了模型产生幻觉的空间。

此外，架构中还引入了动态领域扩展机制以提升系统的可扩展能力。当系统检测到用户需求属于尚未预置的新商品类别时，可调用大语言模型对该类别的评价维度进行初步推断，临时构建相应的评分体系，并自动切换至咨询模式以保证用户仍能获得基础信息服务。这一机制使系统能够在不修改既有代码的情况下扩展所支持的商品类别范围。


\subsection{核心算法设计}

\subsubsection{用户需求语义解析与参数提取}

在智能商品选购系统中，用户需求的准确理解是推荐流程的起点。现实场景下，消费者通常通过自然语言表达需求，而这些表达往往具有明显的口语化特征，例如“我想买一台不差钱、能打游戏还能做后期修图的电脑”。此类描述同时包含主观评价和多个使用场景，因此难以直接转换为数据库查询条件。

虽然大语言模型在语义理解方面具有显著优势，但在长文本推理过程中仍可能出现注意力漂移或语义扩散问题。如果缺乏约束机制，模型在解析需求时可能生成不稳定甚至虚构的参数，从而影响推荐系统的可靠性。

为此，系统设计了一种基于提示工程的方法，通过结构化提示词约束模型输出。在单轮交互中，模型需完成 8 个维度的关键参数提取：1. \texttt{usage}（用途）；2. \texttt{budget\_level}（预算描述）；3. \texttt{max\_price}（数值预算，检测到“不差钱”等表达时映射为最高限额 \texttt{999999}，缺省默认为 \texttt{20000}）；4. \texttt{sort\_field}（依据用户偏好推断的排序权重）；5. \texttt{summary}（核心诉求归纳）；6. \texttt{product\_type}（特定类型约束）；7. \texttt{brand}（品牌偏好）；8. \texttt{owned\_items}（用户已购设备，用于生态联动）。提取出的 \texttt{owned\_items} 字段是实现跨品类关联逻辑的核心，使系统能够识别用户已有的产品并推荐具有强协同效应的新型号。

\begin{lstlisting}[language=Python, caption={通用意图解析系统提示词构建（对应文件：base\_agent.py）}, label={code:intent_prompt}]
def _parse_intent_generic(self, user_msg: str, category_name: str, fields_desc: str) -> dict:
    system_rules = f"""
    你是一个{category_name}导购专家，负责从对话中提取结构化信息。
    【待提取维度】：
    1. usage (用途): 使用场景关键词
    2. budget_level (投入): 预算描述
    3. max_price (预算): 数字，默认 20000，"不差钱"设为 999999
    4. sort_field (排序字段): 匹配最合适的评分字段 ({fields_desc})
    5. summary (摘要): 核心诉求归纳
    6. product_type (类型): 显式要求的特定产品类型
    7. brand (品牌): 用户明确想要购买的品牌（排除已拥有设备品牌）
    8. owned_items (已购设备): 用户提到的已拥有产品型号/品类

    输出严格 JSON: {{"max_price": int, "sort_field": str, "summary": str, ...}}
    """
    # 调用 LLM 接口进行 JSON 提取
    return self.call_llm_json(system_rules, user_msg)
\end{lstlisting}

此外，为解决系统预设商品类别有限的问题，系统设计了动态品类补全机制（Dynamic Category Scoring System）。当核心识别模块 \texttt{CategoryDetector} 检测到用户需求涉及未在系统预定义类别中的商品时，系统将进入知识推理模式，由模型自动生成该品类的评价维度。例如在厨房电器场景中，系统可能生成“功率”“容量”“温控能力”等评价指标，从而构建新的评分体系。该机制使系统能够在无需预先配置数据结构的情况下扩展新的商品类别。

\begin{lstlisting}}[language=Python, caption={基于LLM的动态评分体系构建算法（对应文件：category\_detector.py）}, label={code:intent_prompt}]
    def build_scoring_system(
        self, 
        category_key: str, 
        category_name: str,
        user_context: str = ""
    ) -> Dict:
        """
        LLM 动态构建该品类的评分体系
        
        Args:
            category_key: 品类英文标识
            category_name: 品类中文名称
            user_context: 用户上下文（可选）
            
        Returns:
            包含评分维度和场景预设的字典
        """
        system_prompt = f"""
        你是一个{category_name}领域专家。请为该品类设计一套评分体系和使用场景。
        
        要求：
        1. 设计 3-5 个核心评分维度（如性能、续航、便携性等）
        2. 每个维度需要：中文名称、英文字段名(xxx_Score格式)、权重(0-1,总和=1)、描述
        3. 设计 3-5 个典型使用场景及对应的推荐关键词
        
        输出 JSON 格式：
        {{
            "dimensions": [
                {{"name": "性能", "field": "Performance_Score", "weight": 0.3, "description": "..."}}
            ],
            "scenarios": {{
                "gaming": {{"keywords": ["游戏", "电竞"], "presets": ["推荐品牌/型号关键词"]}}
            }},
            "default_sort_field": "主排序字段名"
        }}
        """
\end{lstlisting}


\subsubsection{异构商品多维评分模型}

在完成用户需求解析后，系统需要对数据库中的商品进行统一的性能评价。然而，不同类别电子产品的参数体系差异较大，例如手机关注摄像头和电池容量，而耳机更强调频响范围和降噪能力。因此，不同类别之间难以直接进行性能比较。

为解决该问题，系统构建了双层评价维度体系。第一层为通用评价维度（Universal Dimension），用于描述所有商品共享的基础指标，涵盖了综合性价比（\texttt{Value\_Score}）、品牌信誉（\texttt{Brand\_Score}）以及用户真实评价评分（\texttt{User\_Rating}）。这类通用维度为不同类别的商品提供了统一的价值比较基准。

第二层为专有性能维度（Specific Dimension），用于描述垂直领域产品的核心技术性能指标。例如在相机代理模块 \texttt{CameraAgent} 中，系统定义了“便携性评分（\texttt{Portability\_Score}）”“低光画质（\texttt{LowLight\_Score}）”和“视频能力（\texttt{Video\_Score}）”；而在显卡代理 \texttt{GPUAgent} 中，则聚焦于“游戏性能（\texttt{Gaming\_Score}）”“创作性能（\texttt{Creative\_Score}）”及“功耗散热（\texttt{Thermal\_Score}）”等指标。

\begin{algorithm}
\caption{异构商品专有评分维度定义抽象（对应文件：base\_agent.py）}
\label{alg:specific_dimensions}
\begin{algorithmic}[1]
\STATE \textbf{Data Class ScoringDimension:}
\STATE \quad name: String \COMMENT{例如: ``便携性''}
\STATE \quad field: String \COMMENT{例如: ``Portability\_Score''}
\STATE \quad weight: Float \COMMENT{权重区间 [0, 1]}
\STATE \quad description: String \COMMENT{维度功能描述}
\end{algorithmic}
\end{algorithm}

\begin{algorithm}
\caption{CameraAgent 专有评分维度示例}
\label{alg:camera_dimensions}
\begin{algorithmic}[1]
\STATE \textbf{Class CameraAgent Inherits BaseDbAgent:}
\STATE \textbf{Function GetSpecificDimensions() $\rightarrow$ List<ScoringDimension>:}
\STATE \quad \textbf{Return} [
\STATE \quad \quad ScoringDimension(``便携性'', ``Portability\_Score'', 0.3, ``\dots''),
\STATE \quad \quad ScoringDimension(``低光画质'', ``LowLight\_Score'', 0.4, ``\dots''),
\STATE \quad \quad ScoringDimension(``视频能力'', ``Video\_Score'', 0.3, ``\dots'')
\STATE \quad ]
\end{algorithmic}
\end{algorithm}

\begin{algorithm}
\caption{GPUAgent 专有评分维度示例}
\label{alg:gpu_dimensions}
\begin{algorithmic}[1]
\STATE \textbf{Class GPUAgent Inherits BaseDbAgent:}
\STATE \textbf{Function GetCategoryConfig() $\rightarrow$ CategoryConfig:}
\STATE \quad \textbf{Return} CategoryConfig(
\STATE \quad \quad ScoringDimensions = [
\STATE \quad \quad \quad ScoringDimension(``游戏性能'', ``Gaming\_Score'', 0.35, ``\dots''),
\STATE \quad \quad \quad ScoringDimension(``创作性能'', ``Creative\_Score'', 0.25, ``\dots''),
\STATE \quad \quad \quad ScoringDimension(``功耗散热'', ``Thermal\_Score'', 0.20, ``\dots''),
\STATE \quad \quad \quad ScoringDimension(``性价比'', ``Value\_Score'', 0.20, ``\dots'')
\STATE \quad \quad ]
\STATE \quad )
\end{algorithmic}
\end{algorithm}

由于商品原始数据来源多样，参数量纲与格式存在显著差异，系统在数据处理阶段引入了多类型归一化策略。对于布尔型参数，系统采用固定权值映射；对于离散枚举变量，根据性能等级设置分档权重；对于连续数值变量，则采用 Min-Max 归一化方法进行标准化处理。

归一化后的各项指标依据预设权重通过加权求和（Weighted Sum）计算综合得分，并映射至 $0$--$100$ 的评分空间。该结果不仅用于最终推荐排序，也为前端 Plotly 可视化模块提供了稳定的数据结构，支持雷达图与多维能力对比图的动态呈现。

\subsubsection{双层场景化推荐引擎}

在商品评分体系建立之后，推荐系统需要从大量商品中召回最符合用户需求的候选结果。如果仅依赖数值排序，往往难以体现数码产品选购中的经验知识；而完全依赖固定规则，又可能在预算约束较严格时出现无推荐结果的情况。

为提高系统稳定性，本研究设计了双层场景化推荐引擎。第一层为专家知识预设机制（Golden Sets）。系统将常见消费场景与优质机型建立映射关系，例如“电竞游戏”“户外 Vlog 拍摄”“高保真音乐欣赏”等。当用户需求中的字段 \texttt{usage} 与这些场景匹配时，系统优先在对应候选机型集合中进行筛选，从而利用领域经验快速缩小搜索范围。

然而，由于市场价格变化或预算限制，预设机型可能被全部排除。为避免推荐结果为空，系统设计了降级检索机制（Fallback Search）。当预设集合在预算过滤后剩余商品数量不足时，系统自动触发全库检索。此时推荐算法不再依赖场景规则，而是根据用户最关注的排序字段 \texttt{sort\_field} 对数据库商品进行排序，并结合预算条件筛选新的候选商品。

最终，系统通过集合并集与去重策略整合两类结果，确保每次推荐均能输出 $3$ 至 $5$ 款不同商品，从而在保证推荐质量的同时提高系统鲁棒性。

\begin{algorithm}
\caption{双层场景化推荐检索策略算法（对应文件：base\_agent.py）}
\label{code:two_layer_search}
\begin{algorithmic}[1]
\REQUIRE user\_message (String), category\_agent (Object)
\ENSURE candidate\_products (List)

\STATE intent\_data $\gets$ \texttt{ParseIntentGeneric}(user\_message, category\_agent.metadata)
\STATE target\_scenario $\gets$ \texttt{MatchScenarioRules}(intent\_data.usage, intent\_data.summary)
\STATE candidate\_products $\gets$ \textsc{EmptyList}()

\COMMENT{第一层：专家知识场景召回}
\IF{target\_scenario $\neq$ \textsc{Null}}
    \STATE preset\_pool $\gets$ \texttt{GetPresetProductsFromDatabase}(target\_scenario)
    \STATE filtered\_pool $\gets$ \texttt{FilterByBudgetAndBrand}(preset\_pool, intent\_data.max\_price, intent\_data.brand)
    \STATE candidate\_products $\gets$ \texttt{SortByDimension}(filtered\_pool, intent\_data.sort\_field)
    \STATE candidate\_products $\gets$ \texttt{Limit}(candidate\_products, \textsc{Top}=3)
\ENDIF

\COMMENT{第二层：全库降级检索兜底}
\IF{\texttt{Length}(candidate\_products) $<$ 3}
    \STATE exclude\_ids $\gets$ \texttt{GetProductIDs}(candidate\_products)
    \STATE fallback\_pool $\gets$ \texttt{QueryDatabase}(
        \STATE \quad \textsc{Price} $\le$ intent\_data.max\_price,
        \STATE \quad \textsc{Brand} matches intent\_data.brand,
        \STATE \quad \textsc{ExcludeIDs} in exclude\_ids,
        \STATE \quad \textsc{SortBy} = intent\_data.sort\_field,
        \STATE \quad \textsc{Limit} = 3
    \STATE )
    \STATE candidate\_products $\gets$ \texttt{Merge}(candidate\_products, fallback\_pool)
    \STATE candidate\_products $\gets$ \texttt{Limit}(candidate\_products, 3)
\ENDIF

\RETURN candidate\_products
\end{algorithmic}
\end{algorithm}

\subsubsection{跨品类生态推荐算法}

随着智能设备种类的增加，用户在购买单一设备时往往同时关注设备之间的协同体验。例如，在选择笔记本电脑时，用户可能还需要显示器、蓝牙音箱或无线鼠标等配套设备。因此，系统在核心推荐流程之外设计了跨品类生态推荐模块（Eco-system Suggestion）。

该模块首先构建若干设备生态主题集合，例如“移动办公生态”和“家庭影音娱乐生态”。每个主题包含一个核心设备类别以及若干关联设备类别。例如在移动办公场景中，笔记本电脑作为核心设备，而显示器和蓝牙扬声器等设备作为扩展组件。

在算法实现上，系统采用关键词权重评分机制对用户需求进行生态分类。例如与办公场景高度相关的关键词（如“编程”“出差”）被赋予较高权重，而“文档”等通用词汇则赋予较低权重。系统根据关键词累计得分判断用户所属的设备生态类别。

在确定生态类别后，系统结合用户已有设备列表 \texttt{owned\_items} 进行过滤，以避免推荐重复设备。例如当用户已拥有显示器时，系统将优先推荐其他辅助设备。最终，系统通过语言模型生成简要推荐说明，并附加在主推荐结果之后，从而形成完整的设备组合建议。

\begin{lstlisting}[language=Python, caption={跨品类生态环境定义与关联规则（对应文件：category_detector.py）}, label={code:ecosystem_config}]
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
    ……
}
\end{lstlisting}



\subsection{系统架构实现}

\subsubsection{全栈基础架构与运行环境实现}

在第四章完成系统总体架构设计之后，本章进一步从工程实现角度说明系统的具体落地过程。系统实现阶段的核心目标是在保证功能完整性的前提下，使系统具备良好的可维护性与扩展能力。因此，在技术选型和架构实现过程中，本研究重点考虑开发效率、系统稳定性以及后期扩展需求。

在整体技术架构方面，系统采用 Python 作为主要开发语言，并构建了一套轻量化的全栈应用环境。其中，前端界面与交互逻辑基于 Streamlit 框架实现。Streamlit 可以通过 Python 脚本直接构建 Web 界面，从而显著降低前端开发复杂度。同时，该框架提供了基于状态驱动的界面刷新机制，使系统能够在用户输入发生变化时自动更新页面内容。

在系统交互过程中，用户的对话记录以及部分运行状态信息通过 

st.session\_state进行统一管理。

如下所示的代码片段展示了应用入口状态的初始化逻辑，该机制能够在多轮交互中保持数据一致性，从而保证推荐流程能够持续依据用户输入进行动态更新：

\begin{algorithm}
\caption{系统状态初始化机制（对应文件：app.py）}
\label{alg:init_session}
\begin{algorithmic}[1]
\COMMENT{确保关键组件在多次交互中持久存在}
\IF{‘central\_router\_agent’ 不存在于 session}
    \STATE session[‘central\_router\_agent’] $\gets$ \texttt{Instantiate(MultiCategoryAgent)}
\ENDIF

\IF{‘conversation\_history’ 不存在于 session}
    \STATE session[‘conversation\_history’] $\gets$ \textsc{EmptyList()}
\ENDIF

\IF{‘current\_recommendation\_results’ 不存在于 session}
    \STATE session[‘current\_recommendation\_results’] $\gets$ \textsc{Null}
\ENDIF

\IF{‘active\_visualization\_charts’ 不存在于 session}
    \STATE session[‘active\_visualization\_charts’] $\gets$ \textsc{Null}
\ENDIF
\end{algorithmic}
\end{algorithm}


在智能推理部分，系统通过标准 API 接口接入大语言模型服务。该接口遵循 OpenAI API 的调用规范，并通过配置文件统一管理模型参数，例如 config.LLM\_MODEL和自定义 baseUrl 等。具体实现见如下代码：

\begin{algorithm}
\caption{大语言模型客户端初始化架构（对应文件：base\_agent.py）}
\label{alg:init_model_client}
\begin{algorithmic}[1]
\COMMENT{初始化与大语言模型服务的连接}
\STATE self.llm\_interface $\gets$ \texttt{Instantiate LLM\_Client With Parameters:}
\STATE \quad API\_Key = \texttt{RetrieveFromConfig("API\_KEY")}
\STATE \quad Base\_Endpoint = \texttt{"https://api.model-provider.com/v1"}
\STATE \quad Timeout\_Seconds = 30
\STATE \quad Max\_Retries = 3

\COMMENT{验证连接状态}
\IF{\texttt{ConnectionFails}(self.llm\_interface)}
    \STATE \texttt{LogWarning("Model service unavailable. Fallback protocols enabled.")}
\ENDIF
\end{algorithmic}
\end{algorithm}

通过这种方式，模型调用逻辑与系统业务代码保持相对独立，当需要升级或替换底层模型时，只需修改配置文件即可完成迁移。

在数据可视化方面，系统采用 Plotly 作为主要图表绘制工具。Plotly 支持浏览器端交互式图表展示，能够实现动态缩放、悬停提示等功能，使用户可以更加直观地观察产品参数差异和评分结果。

系统的数据存储采用 SQLite 关系型数据库，并通过 SQLAlchemy ORM 框架进行管理。ORM 技术能够将数据库表结构映射为 Python 类对象，从而避免直接编写 SQL 语句所带来的安全隐患，同时也提高了代码的可读性和可维护性。

在项目结构组织方面，系统入口文件为 \texttt{app.py}，主要负责前端界面逻辑与会话控制；商品类别识别模块位于 \texttt{category\_detector.py}；而各类商品推荐逻辑则集中在 \texttt{electronics\_agents.py} 中实现。通过这样的模块划分，系统各组件之间保持较低耦合度，从而提升整体架构的清晰性。

\subsubsection{多 Agent 推荐模块实现}

在实际电商场景中，不同类别商品的评价标准往往存在较大差异。例如，相机产品通常更加关注画质表现和视频能力，而手机产品则更强调性能与续航能力。如果所有推荐逻辑集中在同一模块中，不仅代码结构复杂，也不利于系统后期扩展。

为了解决这一问题，系统在实现阶段采用了面向对象编程思想，并构建了多 Agent 推荐模块。系统首先在base\_agent.py文件中定义了抽象基类 BaseProductAgent，用于描述所有商品推荐代理的共同行为。该基类通过 @abstractmethod 声明若干必须由子类实现的方法，例如获取数据库模型类型以及返回评分维度信息等，其核心结构如下：

\begin{algorithm}
\caption{推荐代理基类抽象接口定义（对应文件：base\_agent.py）}
\label{alg:base_agent}
\begin{algorithmic}[1]
\STATE \textbf{Abstract Class BaseProductAgent:}
\STATE \quad \COMMENT{所有商品共享的通用评分维度}
\STATE \quad \textbf{Constant} UNIVERSAL\_DIMENSIONS $\gets$ [ Value\_Score, Brand\_Score, User\_Rating ]
\STATE
\STATE \quad \COMMENT{需由具体品类代理实现的抽象方法}
\STATE \quad \textbf{Abstract Function} GetCategoryConfig() $\rightarrow$ CategoryConfig
\STATE \quad \textbf{Abstract Function} GetSpecificDimensions() $\rightarrow$ List$<$ScoringDimension$>$
\STATE \quad \textbf{Abstract Function} GetDatabaseSchema() $\rightarrow$ DatabaseModel
\STATE \quad \textbf{Abstract Function} FormatProductInformation(product) $\rightarrow$ String
\STATE
\STATE \quad \COMMENT{具体的推荐流转管道模板}
\STATE \quad \textbf{Function} ExecuteRecommendationPipeline(user\_input, intent) $\rightarrow$ Result:
\STATE \quad \quad candidates $\gets$ FetchCandidates(intent)
\STATE \quad \quad filtered $\gets$ FilterCandidates(candidates)
\STATE \quad \quad sorted $\gets$ SortCandidates(filtered, intent.sort\_field)
\STATE \quad \quad \textbf{Return} sorted
\end{algorithmic}
\end{algorithm}

在此基础上，系统在具备数据处理能力的基类 \texttt{BaseDbAgent} 中实现了统一的推荐流程方法 \texttt{handle\_chat\_generic}。该方法采用模板方法模式，将完整推荐流程划分为以下核心代码呈现的多个步骤：

\begin{algorithm}
\caption{核心商品推荐流转控制序列（对应文件：base\_agent.py）}
\label{alg:process_recommendation}
\begin{algorithmic}[1]
\REQUIRE self, user\_message, history, db\_model
\ENSURE (justifications, visual\_charts, ranked\_results, analytical\_text)

\COMMENT{步骤 1: 语义意图提取}
\STATE intent\_data $\gets$ \texttt{Execute(parse\_intent\_generic, user\_message)}

\COMMENT{步骤 2: 提取硬性约束与软性偏好}
\STATE budget\_limit $\gets$ intent\_data.max\_price
\STATE target\_scenario $\gets$ \texttt{MatchScenario(intent\_data.usage)}

\COMMENT{步骤 3: 候选商品集生成与排序}
\STATE candidate\_products $\gets$ \texttt{GetPreconfiguredSet(target\_scenario)}
\STATE ranked\_results $\gets$ \texttt{ApplyFilteringAndSorting(candidate\_products, intent\_data)}

\COMMENT{步骤 4: 兜底机制 (若结果少于3款)}
\IF{\texttt{Count(ranked\_results)} $<$ 3}
    \STATE ranked\_results $\gets$ \texttt{Execute(fallback\_full\_database\_search, intent\_data)}
\ENDIF

\COMMENT{步骤 5: 可解释性生成与可视化构建}
\STATE justifications $\gets$ \texttt{GenerateExplanationsWithLLM(ranked\_results, user\_message, intent\_data)}
\STATE visual\_charts, analytical\_text $\gets$ \texttt{GenerateVisualizations(ranked\_results, intent\_data)}

\RETURN (justifications, visual\_charts, ranked\_results, analytical\_text)
\end{algorithmic}
\end{algorithm}

通过这种方式，不同商品类别在实现时只需关注自身特有的评分逻辑，而无需重复编写完整流程代码。

在具体实现中，系统定义了多个继承自基类的子类。例如 \texttt{CameraAgent} 主要负责相机类产品推荐，而 \texttt{PhoneAgent} 则用于手机产品推荐。这些子类通过重写方法定义各自的评价指标与专属设置，例如 \texttt{CameraAgent} 定义的代码如下：

\begin{algorithm}
\caption{相机代理类实现（对应文件：electronics\_agents.py）}
\label{alg:camera_agent_impl}
\begin{algorithmic}[1]
\STATE \textbf{Class} CameraAgent \textbf{Inherits} BaseDbAgent:
\STATE
\STATE \quad \COMMENT{相机推荐代理}
\STATE
\STATE \quad \textbf{Properties:}
\STATE \quad SCENARIO\_PRESETS: \{
\STATE \quad \quad "vlog": ["ZV-E10", "G7 X", "Pocket", "Action", "Z30"],
\STATE \quad \quad "travel": ["X100", "GR III", "a6400", "Z fc", "X-T30", "X-S10"],
\STATE \quad \quad "street": ["GR III", "X100", "Leica", "Pen-F", "X-E4"],
\STATE \quad \quad "portrait": ["A7", "R6", "R5", "Z6", "Z5", "5D"],
\STATE \quad \quad "landscape": ["A7 R", "Z7", "D850", "GFX"],
\STATE \quad \quad "beginner": ["R50", "Z30", "M50", "200D", "D3500", "a6000"]
\STATE \quad \}
\STATE
\STATE \quad SCENARIO\_KEYWORDS: \{
\STATE \quad \quad "vlog": ["vlog", "视频", "拍片", "直播", "up主"],
\STATE \quad \quad "travel": ["travel", "旅行", "旅游"],
\STATE \quad \quad "street": ["street", "街拍", "人文", "扫街"],
\STATE \quad \quad "portrait": ["portrait", "人像", "写真"],
\STATE \quad \quad "landscape": ["landscape", "风光", "风景", "大片"],
\STATE \quad \quad "beginner": ["beginner", "新手", "入门", "小白", "学生"]
\STATE \quad \}
\STATE
\STATE \quad \textbf{Function} get\_category\_config() $\rightarrow$ CategoryConfig:
\STATE \quad \quad \textbf{Return} CategoryConfig(
\STATE \quad \quad \quad name = "相机",
\STATE \quad \quad \quad name\_en = "camera",
\STATE \quad \quad \quad table\_name = "cameras",
\STATE \quad \quad \quad scoring\_dimensions = self.get\_specific\_dimensions(),
\STATE \quad \quad \quad scenario\_presets = self.SCENARIO\_PRESETS,
\STATE \quad \quad \quad scenario\_keywords = self.SCENARIO\_KEYWORDS,
\STATE \quad \quad \quad default\_sort\_field = "LowLight\_Score",
\STATE \quad \quad \quad display\_fields = ["Brand", "Model", "Price", "LowLight\_Score", "Video\_Score", "Portability\_Score"]
\STATE \quad \quad )
\STATE
\STATE \quad \textbf{Function} get\_specific\_dimensions() $\rightarrow$ List$<$ScoringDimension$>$:
\STATE \quad \quad \textbf{Return} [
\STATE \quad \quad \quad ScoringDimension("便携性", "Portability\_Score", 0.3, "如重量体积"),
\STATE \quad \quad \quad ScoringDimension("低光画质", "LowLight\_Score", 0.4, "暗光表现"),
\STATE \quad \quad \quad ScoringDimension("视频能力", "Video\_Score", 0.3, "视频拍摄能力")
\STATE \quad \quad ]
\STATE
\STATE \quad \textbf{Function} get\_model\_class() $\rightarrow$ Model:
\STATE \quad \quad \textbf{Return} Camera
\STATE
\STATE \quad \textbf{Function} get\_product\_info\_text(p: Camera) $\rightarrow$ String:
\STATE \quad \quad \textbf{Return} Concatenate(
\STATE \quad \quad \quad "型号:", p.Brand, " ", p.Model,
\STATE \quad \quad \quad ", 价格:", p.Price,
\STATE \quad \quad \quad ", 低光:", p.LowLight\_Score,
\STATE \quad \quad \quad ", 视频:", p.Video\_Score,
\STATE \quad \quad \quad ", 便携:", p.Portability\_Score
\STATE \quad \quad )
\STATE
\STATE \quad \textbf{Function} handle\_chat(user\_msg, history = None) $\rightarrow$ Result:
\STATE \quad \quad \textbf{Return} self.handle\_chat\_generic(user\_msg, history, Camera, self.SCENARIO\_KEYWORDS)
\end{algorithmic}
\end{algorithm}

此处定义的维度直接映射到系统对相机进行综合打分的策略中，而手机类则会相似地关注“性能表现”和“续航能力”等指标并为其分配比重。



为了实现不同商品类别之间的统一调度，系统在 \texttt{multi\_agent.py} 中实现了中央调度模块 \texttt{MultiCategoryAgent}。该模块作为全系统的单一访问入口，负责将非结构化的用户查询路由至最合适的领域专家代理。其核心调度逻辑（如算法 \ref{alg:multi_category_dispatcher} 所示）涵盖了品类识别、代理调度、动态代理构建以及跨品类生态建议生成等关键环节。

当系统接收到用户消息后，首先由 \texttt{CategoryDetector} 识别目标品类（Category Key）。若该品类已预置了专家代理（如 \texttt{CameraAgent}），则直接进行任务分发；若属于长尾品类，系统将激活动态代理构建机制（\texttt{create\_dynamic\_agent}），利用大语言模型推演该品类的评分维度并生成通用选购建议，从而保证了系统对未知品类的泛化处理能力。

此外，该模块还承载了系统的“生态系统增强”功能。在获取主品类的推荐结果后，系统会分析用户意图中的已购设备（\texttt{owned\_items}）并调用生态配置模块，识别用户当前的潜在应用场景（如“移动办公”或“游戏发烧”）。通过大语言模型生成自然的生态搭配建议，系统能够主动推荐具有强互补性的次选品类，将原本单一的商品筛选过程升华为场景化的整体解决方案构建。

\begin{algorithm}
\caption{多品类中央路由分发算法（对应文件：multi\_agent.py）}
\label{alg:multi_category_dispatcher}
\begin{algorithmic}[1]
\REQUIRE 用户消息 user\_message (String), 对话历史 conversation\_history (List)
\ENSURE 建议原因 reasons, 可视化图表 charts, 商品列表 products, 数据分析 analyses, 生态建议 eco\_suggestion

\STATE (category\_key, category\_name) $\gets$ detector.\texttt{detect\_category}(user\_message)

\STATE target\_agent $\gets$ \texttt{GetRegisteredAgent}(category\_key)
\IF{target\_agent = \textsc{Null}}
    \STATE target\_agent $\gets$ \texttt{\_create\_dynamic\_agent}(category\_key, category\_name, user\_message)
    \IF{target\_agent = \textsc{Null}}
        \RETURN \texttt{Error}("Unable to build handling agent")
    \ENDIF
\ENDIF

\STATE (reasons, charts, products, analyses) $\gets$ target\_agent.\texttt{handle\_chat}(user\_message, history)

\COMMENT{生态系统增强逻辑}
\STATE eco\_info $\gets$ \texttt{get\_ecosystem\_recommendations}(user\_message, category\_key)
\IF{eco\_info 匹配到关联品类}
    \STATE \texttt{owned\_items} $\gets$ \texttt{ExtractFromAgentIntent}(target\_agent)
    \STATE eco\_suggestion $\gets$ \texttt{\_generate\_eco\_suggestion\_llm}(..., owned\_items)
\ENDIF

\RETURN (reasons, charts, products, analyses, eco\_suggestion)
\end{algorithmic}
\end{algorithm}

通过这种方式，系统能够灵活支持多种商品类型，并且在需要扩展新类别时只需新增对应 Agent 并注册，无需修改总体处理链路。

\subsubsection{商品数据层实现与数据清洗}

推荐系统的有效性在很大程度上依赖于底层数据质量。因此，在系统实现过程中，本研究构建了结构规范的商品数据库，并通过数据清洗保证数据的准确性和一致性。

系统数据库通过 SQLAlchemy ORM 框架进行管理，所有数据模型统一定义在 \texttt{models.py} 文件中。各类商品数据表通过继承 \texttt{Base} 类建立实体对象，例如 \texttt{Camera}、\texttt{Laptop} 等类分别对应数据库中的商品数据表结构。

设计数据模型时，每个类别都包含基础属性如品牌 (Brand)、型号 (Model) 和价格 (Price)。同时针对不同商品类别，系统还设计了专门的参数字段和评分维度，例如通过 SQLAlchemy 定义的 \texttt{Camera} 模型：

\begin{algorithm}
\caption{关系型数据库表的结构化表示（对应文件：models.py）}
\label{alg:camera_table}
\begin{algorithmic}[1]
\STATE \textbf{Database Schema Definition: Table "Cameras"}
\STATE
\STATE PrimaryKey: id (Integer)
\STATE
\STATE \textbf{Standard\_Attributes:}
\STATE \quad Brand (String) \COMMENT{品牌标识}
\STATE \quad Model (String) \COMMENT{产品型号名称}
\STATE \quad Price (Float) \COMMENT{当前市场价格}
\STATE \quad Year (Integer) \COMMENT{发布年份}
\STATE \quad Image\_File (String) \COMMENT{产品图片路径}
\STATE
\STATE \textbf{Technical\_Specifications:}
\STATE \quad Total\_Megapixels (Float) \COMMENT{传感器分辨率}
\STATE \quad Sensor\_Type (String) \COMMENT{画幅类型（全画幅、APS-C等）}
\STATE \quad Weight\_g (Float) \COMMENT{设备重量}
\STATE \quad Supports\_4K (Boolean) \COMMENT{4K视频录制支持情况}
\STATE
\STATE \textbf{Scoring\_Metrics (离线计算得到):}
\STATE \quad Portability\_Score (Float) [0-100]
\STATE \quad LowLight\_Score (Float) [0-100]
\STATE \quad Video\_Score (Float) [0-100]
\end{algorithmic}
\end{algorithm}

如上所示，手机数据表也会有自己的电池容量（Battery\_mAh）等指标，这些结构化数据与评分参数能够为推荐算法在生成策略时提供更加细粒度的参考信息。

当专家知识库中的场景预案无法满足用户的预算约束或特定偏好时，系统会激活兜底搜索机制。该机制通过 \texttt{BaseProductAgent} 类中的 \texttt{fallback\_search} 方法实现，利用 SQLAlchemy 提供的 ORM 接口对完整事实数据库执行带约束的动态检索。

如算法 \ref{alg:constrained_search} 所示，检索过程分为三个阶段。首先是应用硬性约束过滤，系统根据意图解析得到的 \texttt{max\_price} 字段对所有商品进行价格阈值筛选；其次是进行去重处理，排除掉第一阶段已识别出的推荐项，以保证推荐结果的多样性；最后是动态排序阶段，系统检测目标数据表是否具备用户偏好的排序字段（\texttt{sort\_field}），若存在则执行降序排列，否则退而求其次使用系统默认的维度（\texttt{default\_sort\_field}）。

\begin{algorithm}
\caption{事实数据约束与动态检索过滤（对应文件：base\_agent.py）}
\label{alg:constrained_search}
\begin{algorithmic}[1]
\REQUIRE 数据库会话 database\_session, 意图数据 intent\_data, 排除ID列表 exclude\_ids
\ENSURE 商品列表（最多3条）

\STATE \textit{Model} $\gets$ \texttt{GetModelClass}()
\STATE search\_query $\gets$ database\_session.\texttt{query}(\textit{Model})

\COMMENT{阶段1：硬性预算约束过滤}
\STATE \textit{price\_field} $\gets$ \texttt{GetPriceFieldName}(\textit{Model})
\STATE search\_query $\gets$ search\_query.\texttt{filter}( \textit{Model}.\textit{price\_field} $\le$ intent\_data.max\_price )

\COMMENT{阶段2：排除已命中项（去重）}
\IF{exclude\_ids 不为空}
    \STATE search\_query $\gets$ search\_query.\texttt{filter}( $\sim$\textit{Model}.id.\texttt{in\_}(exclude\_ids) )
\ENDIF

\COMMENT{阶段3：意图驱动的动态排序}
\STATE \textit{sort\_field} $\gets$ intent\_data.sort\_field \OR \textit{Default\_Sort\_Field}
\IF{\textit{Model} 拥有 \textit{sort\_field}}
    \STATE search\_query $\gets$ search\_query.\texttt{order\_by}( \texttt{desc}(\textit{sort\_field}) )
\ENDIF

\RETURN search\_query.\texttt{limit}(3).\texttt{all}()
\end{algorithmic}
\end{algorithm}

ORM 框架会自动将这些链式调用操作转换为底层经过优化的 SQL 查询语句。这种实现方式不仅保证了数据检索的安全性，全面规避了 SQL 注入风险，还使系统能够灵活响应大语言模型提取的动态意图，在多维数据向量空间中精准平衡预算与性能。

在数据层构建阶段，系统对原始数据进行了必要的清洗处理。原始商品数据来源于网络爬虫抓取的 CSV 文件，其中普遍存在字段格式不一、量纲缺失等工程问题。为此，系统设计了自动化的批处理管道：首先去除参数值中的特殊字符与单位前缀（如 “g”、“mAh”），将其转换为可计算的数值类型；其次针对“支持/不支持”等文本描述，统一映射为布尔型标识；最后处理缺失值并完成数据正则化。经过清洗后的规范数据集通过 SQLAlchemy 的 \texttt{SessionLocal} 管道导入数据库，确立了系统的“单一事实来源”，从根源上约束了大模型产生幻觉的空间。

\subsubsection{用户界面与交互式可视化实现}

为了提高推荐系统的可用性，本研究在界面设计中加入了可视化展示功能，使用户能够更加直观地理解产品之间的性能差异。如果仅以文本形式呈现大量参数信息，用户往往难以快速完成比较，因此可视化展示在系统中具有重要作用。

系统前端界面基于 Streamlit 的状态驱动机制实现。当推荐模块返回结果后，界面会根据 \texttt{st.session\_state} 中存储的数据自动刷新显示内容，从而形成类似单页应用的交互体验。

在商品展示快显方面，系统设计了统一的商品卡片组件 \texttt{render\_product\_card()}。该组件通过 Python 的 f-string 模板引擎构建 HTML 片段，并结合外部 CSS 样式表实现玻璃拟态（Glassmorphism）视觉效果：

\begin{algorithm}
\caption{基于状态驱动的视图组件渲染机制（对应文件：app.py）}
\label{alg:render_card}
\begin{algorithmic}[1]
\REQUIRE product\_entity, justification\_text
\ENSURE 无（前端渲染）

\COMMENT{提取核心属性并格式化}
\STATE brand, model $\gets$ \texttt{ExtractInfo}(product\_entity)
\STATE price\_str $\gets$ \texttt{Format}(product\_entity.Price)

\COMMENT{动态生成规格与评分标签}
\STATE specs $\gets$ \texttt{ParseCategorySpecs}(product\_entity)
\STATE top\_scores $\gets$ \texttt{SelectTopThreeScores}(product\_entity)
\STATE score\_html $\gets$ \texttt{JoinTags}(top\_scores, class="score-tag")
\STATE spec\_html $\gets$ \texttt{JoinTags}(specs, class="spec-tag")

\COMMENT{注入大模型生成的推荐理由}
\STATE reason\_html $\gets$ ""
\IF{justification\_text $\neq$ \textsc{Null}}
    \STATE reason\_html $\gets$ \texttt{FormatReason}(justification\_text, class="reason-box")
\ENDIF

\COMMENT{构建并注入 HTML 模板}
\STATE template $\gets$ \texttt{f-string}("<div class='product-card'> \dots </div>")
\STATE \texttt{st.markdown}(template, \texttt{unsafe\_allow\_html}=True)
\end{algorithmic}
\end{algorithm}

使用这种基于 HTML 模板的内嵌组件方案，极大方便在 Streamlit 等响应式数据大屏中动态组合不同类别的特征信息，使页面布局既保留了 Web 开发的灵活性，又兼顾了数据驱动的高效性。

在数据可视化部分，系统通过 Plotly 绘图引擎提供了三类交互式图表，旨在全方位展示产品性能差异并增强结果的可解释性。

第一类为雷达图（Radar Chart），通过 \texttt{go.Scatterpolar} 函数绘制。系统将商品的各维度原始参数进行 Min-Max 归一化处理（映射至 0--100 区间），并以填充面积的形式展示单个产品在多个评分维度上的综合表现。这种呈现方式便于用户直观观察产品各项能力的平衡度，例如观察能力面积的大小以及在特定维度（如便携性或画质）上的偏向程度。图表支持交互式缩放，并在悬停时显示精确的维度分值。

第二类为单指标柱状对比图（Bar Chart），由 \texttt{draw\_comparison} 函数实现。该图表专门用于针对用户最关心的某一特定参数（如价格、综合得分或特定的性能评分）进行横向直观对比。图表采用了系统预设的浅色粉紫主题配色方案（Theme Colors），并自动在柱状顶部标注数值，以便于用户快速捕捉不同产品在单一核心指标上的优劣性。

第三类则是多维能力对比图，通过指定 \texttt{barmode='group'} 参数实现分组柱状对比。相较于雷达图侧重于单产品的“画像”，分组柱状图更强调多款候选产品在同一组核心指标（如便携性、低光画质、视频能力等）上的性能层级对比。用户可以通过交互式的图例切换（Legend Toggle）动态隐藏或显示特定的候选产品，并通过悬停提示框（Hover Tooltip）获取更详尽的规格说明。

这些图表不仅在视觉上采用了玻璃拟态（Glassmorphism）与现代配色风格，更在逻辑上与后端的双层推荐引擎紧密联动，实时反映动态生成的评分数据，帮助用户在复杂的数码产品配置中快速做出理性的购买决策。

\begin{algorithm}
\caption{雷达图渲染算法（draw\_radar）}
\label{alg:draw_radar}
\begin{algorithmic}[1]
\REQUIRE products, dimensions (可选)
\ENSURE 雷达图对象

\STATE 根据 dimensions 确定 labels
\FOR{each product in products}
    \STATE 计算维度得分 values（若 dimensions 为空则使用预定义字段）
    \STATE values $\gets$ values + values[0] \COMMENT{闭合}
    \STATE 添加雷达图轨迹：r = values, theta = labels + labels[0]
\ENDFOR
\STATE 设置极坐标样式、图例等
\RETURN fig
\end{algorithmic}
\end{algorithm}

\begin{algorithm}
\caption{单一指标柱状图对比（draw\_comparison）}
\label{alg:draw_comparison}
\begin{algorithmic}[1]
\REQUIRE products, field\_name
\ENSURE 柱状图对象

\STATE 提取商品名称和得分 scores
\STATE 创建柱状图：x = 名称, y = scores
\STATE 设置标题为 field\_name 格式化后的文本
\RETURN fig
\end{algorithmic}
\end{algorithm}

\begin{algorithm}
\caption{多维能力对比柱状图（draw\_multi\_dimension\_compare）}
\label{alg:multi_dim_compare}
\begin{algorithmic}[1]
\REQUIRE products, dimensions (可选)
\ENSURE 分组柱状图对象

\STATE 确定 labels 和 fields
\FOR{each product in products}
    \STATE 提取该商品在各 field 上的得分
    \STATE 添加分组柱状图轨迹
\ENDFOR
\STATE 设置 barmode = ``group''，图例位置等
\RETURN fig
\end{algorithmic}
\end{algorithm}



\subsection{本章小结}

本章从需求、算法和工程三个层面对系统进行了完整构建。第一节通过问卷调研梳理了用户在电商选购中遇到的核心问题——信息来源分散、评价主观性强、技术参数难以理解，并据此总结出输入理解、数据筛选和结果展示三个功能需求，同时提出了由表象展示层、路由解析层、领域决策层和事实数据层组成的四层架构，为系统整体设计奠定了基础。第二节在此框架下展开算法设计，通过思维链提示与结构化约束，实现了自然语言向结构化参数的稳定转换，构建了结合通用维度与专有维度的商品评分体系，并引入双层推荐架构及跨品类生态推荐机制，在保证推荐效果的同时提升了系统的稳定性和扩展能力。第三节则将前述设计落地为可运行系统，基于Streamlit、SQLite和相关API完成全栈环境搭建，通过面向对象设计实现多Agent推荐模块，并借助商品卡片组件和交互式图表，将推荐结果以图文结合的形式呈现给用户。三节内容环环相扣，共同验证了本文提出方案在理论设计与实际应用两个层面的可行性。

\chapter{系统评估与真实场景化分析}

本章的工作重心落在验证上——用自动化脚本和多维度功能测试来检验系统是否真的好用、逻辑是否真的严密。评估严格围绕项目源代码中内嵌的验证体系展开，重点考察三个方面：意图解析的准确度、推荐逻辑能否形成闭环，以及可视化输出的联动效果是否符合预期。

% ============================================================
\section{评估方案设计与测试基准}
% ============================================================

系统的可靠性需要分层次地去验证。本研究为各核心模块设计了对应的自动化测试脚本，将定性判断转化为可量化的评估结果。

验证工作部署于配备 16GB RAM 的本地工作站（CPU 为 Intel i7 系列），软件栈基于 Python 3.10 构建。推理层接入阿里云 DashScope 平台的 Qwen 系列模型（走 OpenAI 兼容协议），事实数据库层采用 SQLite 3。

验证标准从两个层面展开。组件级单元验证借助 \texttt{verify\_multi\_category.py} 脚本，分别对 \texttt{BaseProductAgent} 基类、\texttt{CameraAgent} 专家代理、\texttt{CategoryDetector} 品类识别器和 \texttt{MultiCategoryAgent} 路由中心进行连通性与逻辑完整性测试；全链路流转验证则由 \texttt{verify\_agent\_flow.py} 等脚本模拟真实对话，追踪从用户输入到系统输出——涵盖推荐理由、Plotly 图表与数据分析文案——的完整链路是否畅通。

在性能基准方面，回归测试设定了两条硬性要求：对于"相机""笔记本"等预设商品品类关键词，识别器的准确率必须达到 100\%；在网络条件正常的情况下，从意图解析到输出三张联动图表（Radar、Bar、Multi-dim）的端到端耗时应控制在 3 至 5 秒以内。

% ============================================================
\section{核心功能逻辑验证}
% ============================================================

为检验系统在极端条件下的健壮性，本节结合项目内的压力测试脚本，重点测试了两类边界场景。

第一类是基于降级检索的边界条件测试，测试用例参考 \texttt{test\_recommendation.py}：输入极低预算（如 2000 元），但要求推荐高性能相机或笔记本。系统的处理路径如下：\texttt{electronics\_agents.py} 中的 \texttt{\_filter\_and\_sort} 方法首先对专家预设集执行硬性过滤，若过滤后符合预算的产品不足三款，则自动触发 \texttt{\_fallback\_search} 兜底逻辑，依据 \texttt{intent} 中的 \texttt{max\_price} 字段重新发起 SQLAlchemy 数据库查询。这一测试直接验证了双层推荐机制的容灾能力——系统在严苛约束下不会因"无匹配数据"而中断对话，也不会凭空捏造型号，而是退而求其次，从全库中找出性价比最高的备选方案。

第二类是动态品类泛化能力验证，测试对象为 \texttt{CategoryDetector} 面对非内置品类（如"扫地机"）时的表现。结果显示，系统能准确识别出 \texttt{is\_new: True} 标识并激活 \texttt{DynamicAgent} 的维度推演逻辑，借助 LLM 的常识推理自发生成了适用于"扫地机"场景的评估维度（如清扫能力、避障系数等）。这说明系统的品类处理能力并非只在预设范围内有效，面对长尾需求同样具备相当的弹性。

% ============================================================
\subsection{系统效能评估分析}
% ============================================================

与传统关键词查询相比，纯粹的参数过滤在面对模糊表达时往往力不从心。\texttt{\_parse\_intent\_generic} 建立的意图映射层能在这类语境下准确抽取 \texttt{usage}（用途）和 \texttt{summary}（诉求摘要），将用户需求更精准地对齐至数据库中的性能向量轴，这是结构化提取相比纯文本检索的核心优势所在。

可视化层面，\texttt{visualizer.py} 生成的动态图表在可解释性与信息密度两个维度上均表现突出。\texttt{\_get\_individual\_reasons} 方法将晦涩的数值对比转化为易懂的自然语言描述，让推荐结果"有据可查"；Plotly 引擎将多维参数归一化至 0--100 区间，生成的雷达图使用户无需深究参数含义，便可快速感知不同产品在便携性、性能、续航等维度上的相对位置。两者配合，使系统的输出结果从"给出答案"进一步迈向"解释答案"。

% ============================================================
\subsection{从输入到图表的终局案例演示}
% ============================================================

本节选取三个典型测试用例，完整呈现系统从意图捕捉到图表反馈的处理过程。

\subsubsection{案例 1：大学生的轻薄办公场景（笔记本电脑）}

\textbf{场景预设}

用户 A 是一名刚入学的大学生，对电脑硬件参数了解不多。其需求较为明确：日常在图书馆写论文、看视频，需要频繁携带外出，对续航和便携性较为敏感，预算在 6000 元以内。他在输入框中写道：

\begin{quote}
``轻薄本，续航一定要好，看视频写论文，6000元以下。''
\end{quote}

\textbf{交互与推理过程}

系统接收请求后，首先通过 \texttt{CategoryDetector} 的语义匹配逻辑，从“写论文”、“图书馆”等关键词中精准识别出 \texttt{laptop}（笔记本电脑）品类。随后，\texttt{MultiCategoryAgent} 将任务路由至 \texttt{LaptopAgent}。

在意图解析阶段，LLM 识别出“续航一定要好”这一强约束偏好，并结合“看视频写论文”的使用描述，将场景锁定为 \texttt{light\_office}（轻薄办公）。系统随即触发预设的权重分配方案：将性能评分（\texttt{Performance\_Score}，权重 0.4）、便携评分（\texttt{Portability\_Score}，权重 0.3）与屏幕评分（\texttt{Display\_Score}，权重 0.3）作为多维评价的核心。由于用户明确提及“6000元以下”的预算限制，系统在意图对象中将 \texttt{max\_price} 硬性锁定为 6000，并将 \texttt{Value\_Score}（性价比评分）设为默认排序字段，以平衡性能表现与资金投入。

\textbf{结果呈现}

系统推荐了三款机型：联想小新 Air 14（¥4,299）、宏碁非凡 Go Pro（¥4,999）与华硕天选 Air（¥5,999）。三款产品的便携性评分均为 92.0，在用户最关注的核心维度上不相上下；差异主要体现在其余维度上——小新 Air 14 性价比评分以 94.0 居首，屏幕评分为 85.0；非凡 Go Pro 性价比同为 94.0，屏幕评分提升至 88.0，但售价高出 700 元；天选 Air 凭借 AMD R9-7940HS 处理器与 RTX 4050 独显将性能评分拉至 86.0，但性价比评分相应降至 92.0，售价也达到预算上限的 5,999 元。

系统为每款产品生成了简短的场景化评语，直接说明产品与用户使用场景的契合点，避免堆砌参数。深度对比分析报告进一步提供了三个层次的横向比较：综合能力雷达图呈现四维能力分布，核心评分分布专项对比便携性数据，多维对比则明确指出各款产品的相对优势——“天选 Air 在性能表现最佳；小新 Air 14 在性价比表现最佳；非凡 Go Pro 在屏幕表现最佳”——将取舍判断的依据清晰交还给用户。

此外，系统在页面末尾附上了生态链延伸建议：“如果轻薄本主要用来处理和剪辑素材，可以搭配一台色彩准确的显示器。”该建议并非用户此次咨询的核心诉求，但体现了系统对用户潜在使用场景扩展的主动预判，在完成即时推荐任务的同时为后续可能的追加咨询提供了自然的引导入口。

\begin{figure}[htbp]
\centering
\includegraphics[width=0.45\textwidth]{laptop.png}
\hfill
\includegraphics[width=0.45\textwidth]{laptop2.png}
\caption{笔记本电脑推荐结果与对比分析}
\end{figure}

\subsubsection{案例 2：极致性能的游戏发烧场景（跨品类生态组合）}

\textbf{场景预设}

用户 B 是一名资深游戏玩家，预算充裕，追求顶级游戏体验。市面上顶级配件太多，组合方式繁复，他不想花几个周末去挨个研究兼容性和最新发布的型号，也不希望花费大量精力逐一核查各配件的兼容性与搭配关系，倾向于获得完整的配置建议。他在输入框中写道：

\begin{quote}
``不差钱，可以玩顶级 3A 游戏的显卡。''
\end{quote}

\textbf{第一轮交互：显卡推荐}

系统通过 \texttt{GPUAgent} 处理该请求。解析过程中，系统识别到用户输入的“不差钱”关键词，触发了针对极端性能词汇的补丁逻辑（Patch Logic）：将 \texttt{max\_price} 阈值上调至 999,999 元，并自动将最优排序字段（\texttt{sort\_field}）映射为该品类最核心的创作性能维度（\texttt{Creative\_Score}），以确保推荐结果聚焦于顶级规格。

最终推荐结果包含三款产品：NVIDIA GeForce RTX 4060 Ti 16GB（¥3,499）、Intel Arc B770（¥2,799）与 NVIDIA GeForce RTX 4060 Ti（¥3,199）。RTX 4060 Ti 16GB 与 Arc B770 在游戏与创作性能评分上完全持平（均为 80.0/82.0），但在功耗散热评分（\texttt{Thermal\_Score}）上，RTX 4060 Ti 16GB 以 92.0 领先。系统在可视化雷达图中完整呈现了 B770 更高的 3DMark 原始跑分，但基于驱动稳定性与生态成熟度的综合博弈，最终将 16GB 版本的 RTX 4060 Ti 列为首选推荐。

完成单品推荐后，系统调用 \texttt{EcosystemConfig} 模块，识别出“游戏”场景下的跨品类关联需求。通过 \texttt{\_generate\_eco\_suggestion\_llm} 函数，系统生成了极具引导性的专业建议：“都上顶级显卡了，画面必须拉满才过瘾！配一台 4K 高刷显示器，能完全发挥这块显卡的潜力。” 这一建议成功激发了用户对显示输出端性能对等性的关注。

\textbf{第二轮交互：显示器推荐}

用户 B 在看到生态链建议后，认识到显示器与显卡的配套关系值得进一步明确，随即追加输入：

\begin{quote}
``配台高分辨率、高刷新率的电竞显示器。''
\end{quote}

系统将品类切换至显示器，由 \texttt{MonitorAgent} 接手检索，推荐结果包含三款产品：LG 27GR95QE（¥9,999，240Hz / 2560×1440）、LG 27GR950（¥4,999，144Hz / 3840×2160）与技嘉 M27Q（¥2,299，170Hz / 2560×1440）。三款产品人体工学评分均为 88.0，差异集中在画质、性能与性价比三个维度：LG 27GR95QE 画质与性能评分均达 98.0，综合素质最高，但性价比评分仅为 68.0，溢价明显；LG 27GR950 画质评分 96.0、性能评分 95.0，以 4K 分辨率提供了更高的画面精细度；技嘉 M27Q 画质评分 94.0、性能评分 94.0，在三款中性价比最为突出。

系统综合分析指出：27GR95QE 在画质与人体工学表现最佳，27GR950 在性能维度最优，并建议用户根据具体需求权衡选择。同时，系统将技嘉 M27Q 作为性价比导向的重点推荐，指出其 27 英寸 2K 分辨率搭配 170Hz 刷新率的组合能够在合理预算内大幅提升游戏体验。

两轮交互完成后，用户 B 从最初模糊的“顶级游戏显卡”需求，经由生态链建议的自然引导，逐步落地为一套显卡与显示器协同考量的完整配置方案。这一过程体现了系统跨品类推荐链路的设计逻辑：用户无需预先规划所有配件，系统在完成当前品类推荐后主动识别关联需求，以生态链建议作为跨品类跳转的触发机制，将单品咨询有序延伸为场景化的整体方案。

\begin{figure}[htbp]
\centering
\includegraphics[width=0.3\textwidth]{image.png}
\hfill
\includegraphics[width=0.3\textwidth]{image-1.png}
\hfill
\includegraphics[width=0.3\textwidth]{image-2.png}
\caption{显卡与显示器推荐结果及生态链建议}
\end{figure}

\subsubsection{案例 3：追求性价比的摄影入门场景（相机）}

\textbf{场景预设}

用户 C 最近迷上了在社交平台上看别人的 Vlog，开始有了自己记录生活的想法，觉得手机拍出来的画面总差点意思，想买一台相机试试。她在网上查了一些攻略，但“机身防抖”“相位对焦”“4K 30fps”这些词让她越看越迷糊，攻略里动辄出现的参数对比表也让她不知道该从哪里入手判断。她的需求其实并不复杂：预算 5000 元左右，拍日常 Vlog，希望拍出来的画面稳、对焦不拉风箱、操作别太复杂。她在系统里写道：

\begin{quote}
``5000元以内拍 Vlog 的相机。''
\end{quote}

\textbf{交互与推理过程}

系统识别出“Vlog”关键词后，将其映射为 \texttt{CameraAgent} 中的 \texttt{vlog} 场景预设。推理引擎自动从 \texttt{SCENARIO\_PRESETS} 中调取热门候选机型（如 ZV-E10, Pocket, Action 等），并针对该特定场景分配权重：视频能力（\texttt{Video\_Score}）占比 0.3，便携性（\texttt{Portability\_Score}）占比 0.3，而将暗光表现（\texttt{LowLight\_Score}）权重设为最高（0.4），旨在筛选出能够胜任全天候记录的器材。由于预算限制在 5000 元以内，系统过滤了昂贵的专业全画幅型号，优先展示轻量化、高集成度的解决方案。

\textbf{结果呈现}

系统推荐了三款产品：大疆 Osmo Pocket 3（¥3,499）、大疆 Osmo Action 4（¥2,199）与佳能 PowerShot G7 X Mark III（¥4,200）。三款产品呈现出明显差异化的能力分布：Osmo Pocket 3 视频能力评分最高（95.0），便携性评分达 98.0，机身仅重 179g，在场景核心维度上综合表现最为突出，系统将其列为首选；Osmo Action 4 便携性评分最优（99.0），机身最轻（145g），ISO 上限更高（12800），更适合户外动态拍摄，且售价仅 2,199 元，性价比突出；PowerShot G7 X Mark III 低光画质评分以 93.86 大幅领先另外两款，但便携性评分仅为 29.2，机身重量达 304g，与另外两款形成鲜明对比，其能力分布与 Vlog 入门场景的核心诉求存在一定错位。

上述差异在雷达图中得到了完整呈现。三款产品在便携性轴线上的分布尤为直观——G7 X Mark III 的便携性短板在图形上一目了然，无需用户逐一比对重量数值即可直观感知。系统进一步在多维对比中明确指出各款产品的相对优势：“Osmo Action 4 在便携性表现最佳；PowerShot G7 X Mark III 在低光画质表现最佳；Osmo Pocket 3 在视频能力表现最佳”，将不同优先级下的取舍逻辑清晰呈现，辅助用户根据自身实际拍摄场景完成最终判断。

系统还在结果页末尾附上了生态链建议：“拍 Vlog 要效率高，配个能随时预览和剪辑的平板电脑会更顺手——相机直连传输素材，大屏剪辑比手机舒服很多。”该建议将推荐视野从单一设备延伸至创作工作流，对有意持续投入内容创作的用户具有实际参考价值，同时也为后续可能的追加咨询提供了引导入口。

\begin{figure}[htbp]
\centering
\includegraphics[width=0.6\textwidth]{camera.png}
\caption{相机推荐结果与多维对比}
\end{figure}


% ============================================================
\subsection{本章小结}
% ============================================================

通过项目内嵌的验证脚本，本章对系统进行了系统性评估。结果表明，Agent 协同架构在应对复杂消费意图、处理极端预算约束以及生成联动可视化反馈三个方面均表现良好。整体技术链路的闭环性符合预期，系统确实能够将繁琐的产品参数转化为直观的视觉决策依据。

% 逐段对比修改，以下是润色后的版本，主要处理了几类问题：
% 排比式铺陈收紧、"这一……"句式替换、过度解释性语言删减、衔接词自然化。



\section{本文贡献总结}

电商平台的商品信息历来以参数为主要载体，用户在下单前往往需要自行消化大量技术指标，与真实购物决策逻辑之间存在明显落差。本文以此为出发点，提出并实现了一套基于大型语言模型与多智能体协同的对话式数码产品推荐系统，主要工作可从以下四个方面加以归纳。

\subsection{提出面向消费场景的对话式推荐范式}

传统电商平台将产品信息拆解为参数条目逐一陈列，对普通消费者形成隐性门槛。本文将大型语言模型的自然语言理解能力与多智能体任务分发机制整合，以多轮对话为主要交互形式构建推荐路径。用户无须预先了解产品规格，只需用日常语言描述使用需求，系统便在对话过程中逐步完成意图识别、方案筛选与结果解释。

\subsection{构建兼顾灵活性与可靠性的双层推荐架构}

大型语言模型擅长理解模糊表达，但若直接用于生成具体商品推荐，"幻觉"问题便难以规避。对此，本文采取双层处理策略：前一层由大模型负责意图解析与结构化查询生成，后一层将查询结果交由本地SQLAlchemy数据库核验执行，确保每条推荐记录均有据可查。当标准查询路径无法覆盖用户意图时，回退检索引擎作为补充介入。这种分工在保留语义理解灵活性的同时，将幻觉风险限定在推荐结果之外。

\subsection{实现跨品类协同推荐}

消费者的实际采购需求往往并不孤立——学生选购笔记本时多半同时需要耳机与移动电源，摄影爱好者升级机身时镜头与存储方案同样在考虑之列。本文为此设计了\texttt{MultiCategoryAgent}路由模块与\texttt{EcosystemConfig}生态配置，使系统具备跨品类识别与联动推荐的能力，将原本分散的选购环节整合为一次连贯的决策过程。

\subsection{以可视化手段增强推荐的可解释性}

推荐结果能否获得用户信任，很大程度上取决于用户是否理解推荐的依据。本文在呈现层面引入基于Plotly的多类型动态可视化组件：雷达图用于多维性能的直观对比，柱状图用于单项指标的横向比较，多维散点图则允许用户从价格、性能、口碑等维度同时审视候选产品。推荐理由以平实语言输出，与图形内容相互补充，让用户不只是拿到一个结论，也能看懂结论从何而来。

\section{局限性与改进方向}

\subsection{商品数据的实时性不足}

当前系统的商品数据存储于本地SQLite数据库，更新依赖离线爬取，难以与电商平台保持同步。促销期间价格往往在数小时内完成调整，库存状态的变化更是持续发生。若用户依据系统呈现的旧价格做出决策，结账时才发现价格已有偏差，推荐的实用性便会大打折扣。接入京东联盟等电商开放平台的官方API是较直接的改进路径，可在实现数据动态更新的同时，减少对自行爬取的依赖。

\subsection{复杂语义场景下的解析局限}

\texttt{CategoryDetector}处理常规模糊表达时运行稳定，但面对隐含前提较强的表达方式时容易出现偏差，例如以反讽传达偏好，或借助多重否定界定需求。这类说法在日常对话中并不少见，通用大模型对消费语境下的情感语义缺乏专项适配，处理时容易遗漏关键信息。引入经过细粒度情感分析微调的模型是一个可行方向，但需结合具体数据集评估，单纯提升模型规模未必奏效。

\subsection{个体偏好的持续建模缺失}

系统当前的推荐权重来自人工预设的场景规则，反映的是群体层面的共性特征，对个体差异无从捕捉。用户对品牌的使用习惯、对外观的审美偏向，乃至对某类参数的特殊敏感，在现有架构中均付之阙如。每次会话从零开始，多次交互之间也不存在任何积累。在隐私保护前提下引入轻量级个人偏好模型，是系统走向实用化绕不开的问题。

\subsection{后续改进方向}

\textbf{其一}，引入RAG以补充软性评价信息。硬件参数之外，握持手感、系统流畅度、售后口碑等使用体验对消费决策的影响同样不可忽视。将评测视频文字稿与用户评价纳入外挂知识库，可使推荐在参数比较之外兼顾真实反馈。

\textbf{其二}，建立用户操作的反馈闭环。重新生成方案、修改预算、反复对比某两款产品——这些操作本身都隐含着对当前推荐的评价信号。若能将其系统性地收集并用于模型校准，系统便有条件从固定规则逐步过渡到数据驱动的个性化策略。

\textbf{其三}，将推荐引擎与Streamlit前端解耦，以标准API形式独立部署，进而分别接入微信小程序与多设备客户端。小程序无需安装、触达成本低；多端流转则允许用户在手机上提问、电脑上深入比较、平板上查看可视化结果，各端衔接而不中断。

本文所构建的系统目前仍处于原型验证阶段，主要意义在于打通一条将大语言模型语义理解与数据库事实约束相结合的技术路径，并在可解释性呈现上做了初步探索。已识别的局限性，自然构成后续工作的起点。
\newpage 

\bibliographystyle{plain}
\bibliography{reference.bib} 


\end{document}
