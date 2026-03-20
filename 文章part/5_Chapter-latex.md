\section{系统架构实现与代码落地}

\subsection{全栈基础架构与运行环境实现}

在第四章完成系统总体架构设计之后，本章进一步从工程实现角度说明系统的具体落地过程。系统实现阶段的核心目标是在保证功能完整性的前提下，使系统具备良好的可维护性与扩展能力。因此，在技术选型和架构实现过程中，本研究重点考虑开发效率、系统稳定性以及后期扩展需求。

在整体技术架构方面，系统采用 Python 作为主要开发语言，并构建了一套轻量化的全栈应用环境。其中，前端界面与交互逻辑基于 Streamlit 框架实现。Streamlit 可以通过 Python 脚本直接构建 Web 界面，从而显著降低前端开发复杂度。同时，该框架提供了基于状态驱动的界面刷新机制，使系统能够在用户输入发生变化时自动更新页面内容。

在系统交互过程中，用户的对话记录以及部分运行状态信息通过 \texttt{st.session\_state} 进行统一管理。如下所示的代码片段展示了应用入口状态的初始化逻辑，该机制能够在多轮交互中保持数据一致性，从而保证推荐流程能够持续依据用户输入进行动态更新：

\begin{lstlisting}[language=Python, caption={系统状态初始化（片段，摘自 app.py）}]
# 初始化状态管理
if 'agent' not in st.session_state:
    st.session_state.agent = MultiCategoryAgent()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'last_results' not in st.session_state:
    st.session_state.last_results = None
\end{lstlisting}

在智能推理部分，系统通过标准 API 接口接入大语言模型服务。该接口遵循 OpenAI API 的调用规范，并通过配置文件统一管理模型参数，例如 \texttt{config.LLM\_MODEL} 和自定义 \texttt{baseUrl} 等。具体实现见如下代码：

\begin{lstlisting}[language=Python, caption={大语言模型客户端初始化（片段，摘自 base\_agent.py）}]
self.client = OpenAI(
    api_key=config.DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
\end{lstlisting}

通过这种方式，模型调用逻辑与系统业务代码保持相对独立，当需要升级或替换底层模型时，只需修改配置文件即可完成迁移。

在数据可视化方面，系统采用 Plotly 作为主要图表绘制工具。Plotly 支持浏览器端交互式图表展示，能够实现动态缩放、悬停提示等功能，使用户可以更加直观地观察产品参数差异和评分结果。

系统的数据存储采用 SQLite 关系型数据库，并通过 SQLAlchemy ORM 框架进行管理。ORM 技术能够将数据库表结构映射为 Python 类对象，从而避免直接编写 SQL 语句所带来的安全隐患，同时也提高了代码的可读性和可维护性。

在项目结构组织方面，系统入口文件为 \texttt{app.py}，主要负责前端界面逻辑与会话控制；商品类别识别模块位于 \texttt{category\_detector.py}；而各类商品推荐逻辑则集中在 \texttt{electronics\_agents.py} 中实现。通过这样的模块划分，系统各组件之间保持较低耦合度，从而提升整体架构的清晰性。

\subsection{多 Agent 推荐模块实现}

在实际电商场景中，不同类别商品的评价标准往往存在较大差异。例如，相机产品通常更加关注画质表现和视频能力，而手机产品则更强调性能与续航能力。如果所有推荐逻辑集中在同一模块中，不仅代码结构复杂，也不利于系统后期扩展。

为了解决这一问题，系统在实现阶段采用了面向对象编程思想，并构建了多 Agent 推荐模块。系统首先在 \texttt{base\_agent.py} 文件中定义了抽象基类 \texttt{BaseProductAgent}，用于描述所有商品推荐代理的共同行为。该基类通过 \texttt{@abstractmethod} 声明若干必须由子类实现的方法，例如获取数据库模型类型以及返回评分维度信息等，其核心结构如下：

\begin{lstlisting}[language=Python, caption={推荐代理基类定义（片段，摘自 base\_agent.py）}]
class BaseProductAgent(ABC):
    @abstractmethod
    def get_category_config(self) -> CategoryConfig:
        """获取品类特有配置"""
        pass
    
    @abstractmethod
    def get_specific_dimensions(self) -> List[ScoringDimension]:
        """获取品类特有评分维度"""
        pass
        
    @abstractmethod
    def get_model_class(self):
        """获取 SQLAlchemy 模型类"""
        pass
\end{lstlisting}

在此基础上，系统在具备数据处理能力的基类 \texttt{BaseDbAgent} 中实现了统一的推荐流程方法 \texttt{handle\_chat\_generic}。该方法采用模板方法模式，将完整推荐流程划分为以下核心代码呈现的多个步骤：

\begin{lstlisting}[language=Python, caption={推荐流程核心代码（片段，摘自 electronics\_agents.py）}]
def handle_chat_generic(self, user_msg, history, model_class, keyword_map):
    # 1. 意图解析：提取预算、评分字段等信息
    intent = self._parse_intent_generic(user_msg, config.name, fields_desc, history)
    
    # ... 匹配需求对应的场景并处理补丁 ...
    
    # 3. 获取预设商品的候选并应用过滤排序规则
    candidates = self._get_preset_products(target_scenario, model_class)
    results = self._filter_and_sort(candidates, intent, model_class)
    
    # 4. 生成单独的推荐理由、以及基于核心维度的可视化图表
    reasons = self._get_individual_reasons(results, user_msg, intent, history)
    charts, analyses = self._generate_visualization(results, intent)
    
    return reasons, charts, results, analyses
\end{lstlisting}

通过这种方式，不同商品类别在实现时只需关注自身特有的评分逻辑，而无需重复编写完整流程代码。

在具体实现中，系统定义了多个继承自基类的子类。例如 \texttt{CameraAgent} 主要负责相机类产品推荐，而 \texttt{PhoneAgent} 则用于手机产品推荐。这些子类通过重写方法定义各自的评价指标与专属设置，例如 \texttt{CameraAgent} 定义的代码如下：

\begin{lstlisting}[language=Python, caption={相机代理实现（片段，摘自 electronics\_agents.py）}]
class CameraAgent(BaseDbAgent):
    """相机推荐代理"""
    def get_specific_dimensions(self) -> List[ScoringDimension]:
        return [
            ScoringDimension("便携性", "Portability_Score", 0.3, "如重量体积"),
            ScoringDimension("低光画质", "LowLight_Score", 0.4, "暗光表现"),
            ScoringDimension("视频能力", "Video_Score", 0.3, "视频拍摄能力")
        ]
\end{lstlisting}

此处定义的维度直接映射到系统对相机进行综合打分的策略中，而手机类则会相似地关注“性能表现”和“续航能力”等指标并为其分配比重。

为了实现不同商品类别之间的统一调度，系统在 \texttt{multi\_agent.py} 中实现了中央调度模块 \texttt{MultiCategoryAgent}。该模块负责接收通用查询并调度子代理，其核心逻辑如下：

\begin{lstlisting}[language=Python, caption={多品类代理调度（片段，摘自 multi\_agent.py）}]
def handle_chat(self, user_msg: str, history=None):
    # 1. 使用识别器识别品类
    category_key, category_name = self.detector.detect_category(user_msg)
    
    # 2. 获取具体代理实例
    agent = self._agents.get(category_key)
    
    # 3. 调用代理处理业务
    reasons, charts, results, analyses = agent.handle_chat(user_msg, history)
    
    return reasons, charts, results, analyses, eco_suggestion
\end{lstlisting}

通过这种方式，系统能够灵活支持多种商品类型，并且在需要扩展新类别时只需新增对应 Agent 并注册，无需修改总体处理链路。

\subsection{商品数据层实现与数据清洗}

推荐系统的有效性在很大程度上依赖于底层数据质量。因此，在系统实现过程中，本研究构建了结构规范的商品数据库，并通过数据清洗保证数据的准确性和一致性。

系统数据库通过 SQLAlchemy ORM 框架进行管理，所有数据模型统一定义在 \texttt{models.py} 文件中。各类商品数据表通过继承 \texttt{Base} 类建立实体对象，例如 \texttt{Camera}、\texttt{Laptop} 等类分别对应数据库中的商品数据表结构。

设计数据模型时，每个类别都包含基础属性如品牌 (Brand)、型号 (Model) 和价格 (Price)。同时针对不同商品类别，系统还设计了专门的参数字段和评分维度，例如通过 SQLAlchemy 定义的 \texttt{Camera} 模型：

\begin{lstlisting}[language=Python, caption={数据模型定义（片段，摘自 models.py）}]
class Camera(Base):
    __tablename__ = "cameras"
    id = Column(Integer, primary_key=True, index=True)
    Brand = Column(String)
    Model = Column(String)
    Price = Column(Float)
    # ...特定参数字段...
    Total_megapixels = Column(Float)
    Weight_g = Column(Float)
    Supports_4K = Column(Boolean)
    # ...预置的评分结果...
    Portability_Score = Column(Float)
    LowLight_Score = Column(Float)
    Video_Score = Column(Float)
\end{lstlisting}

如上所示，手机数据表也会有自己的电池容量（Battery\_mAh）等指标，这些结构化数据与评分参数能够为推荐算法在生成策略时提供更加细粒度的参考信息。

当推荐模块需要获取候选商品时，只需通过 SQLAlchemy 提供的查询接口执行相关方法即可。例如在进行按预算全库兜底搜索时，操作如下：

\begin{lstlisting}[language=Python, caption={数据库查询过滤（片段，摘自 electronics\_agents.py）}]
query = self.db.query(model_class).filter(
    model_class.Price <= intent.get('max_price', 20000)
)
# 基于特定字段进行降序排序
if hasattr(model_class, sort_field):
    query = query.order_by(desc(getattr(model_class, sort_field)))

return query.limit(3).all()
\end{lstlisting}

ORM 框架会自动将这些链式调用操作转换为底层查询 SQL 语句，并返回对象结果，极大地保证了系统的安全性并避免了 SQL 注入风险。

在数据导入阶段，系统对原始数据进行了必要的清洗处理。原始商品数据主要来源于网络爬虫抓取的 CSV 文件，其中部分字段存在格式不统一或缺失情况。因此，系统通过批处理脚本对原始数据进行预处理，包括去除特殊字符、进行数值类型转换以及处理缺失值等操作。同时，对于部分文本属性（如“支持”“不支持”等），系统将其统一转换为布尔类型。经过清洗后的数据最终通过 SQLAlchemy 的 \texttt{SessionLocal} 接口导入数据库，从而形成结构规范的商品数据集。

\subsection{用户界面与交互式可视化实现}

为了提高推荐系统的可用性，本研究在界面设计中加入了可视化展示功能，使用户能够更加直观地理解产品之间的性能差异。如果仅以文本形式呈现大量参数信息，用户往往难以快速完成比较，因此可视化展示在系统中具有重要作用。

系统前端界面基于 Streamlit 的状态驱动机制实现。当推荐模块返回结果后，界面会根据 \texttt{st.session\_state} 中存储的数据自动刷新显示内容，从而形成类似单页应用的交互体验。

在商品展示方面，系统设计了统一的商品卡片组件 \texttt{render\_product\_card()}。该组件通过封装的 HTML 和通过 CSS 实现玻璃拟态效果进行视觉呈现：

\begin{lstlisting}[language=Python, caption={产品卡片前端渲染代码（片段，摘自 app.py）}]
def render_product_card(product, reason=None):
    # ... 解析商品属性与动态加载核心规格 ...
    card_html = f"""
    <div class='product-card'>
        <div class='product-title'>{brand} {model}</div>
        <div class='product-price'><span class='currency'>¥</span> {int(price):,}</div>
        {score_html}    
        {spec_html_str} 
        <div class='reason-box'>{reason_html}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
\end{lstlisting}

使用这样的内嵌组件方案，极大方便在 Streamlit 等响应式数据大屏中输出不同类别的定制化信息，使页面结构更具备视觉区分度和清晰感。

在数据分析部分，系统提供了两类主要图表用于展示产品性能差异。其中第一类为雷达图（Radar Chart），用于展示单个产品在多个评分维度上的综合表现。

\begin{lstlisting}[language=Python, caption={可视化图表呈现（片段，摘自 visualizer.py）}]
# 封装雷达图绘制
fig.add_trace(go.Scatterpolar(
    r=values,
    theta=labels + labels[:1], # 闭合图形线条
    fill='toself',
    name=f"{p.Model}",
    opacity=0.2
))

# 分组柱状图应用
fig.update_layout(
    title=dict(text="核心能力多维对比"),
    barmode='group', # 实现多产品并排横向对比
    yaxis=dict(title='评分', range=[0, 110])
)
\end{lstlisting}

如代码所示，系统利用 \texttt{Scatterpolar} 绘制雷达图帮助直观观察能力面积大小和偏靠程度。第二类图表则是指定 \texttt{barmode='group'} 参数的柱状多维对比，并支持交互式的悬停以显示具体数值进行横评产品比较。

通过商品卡片展示与交互式图表结合，系统不仅能够输出推荐结果，还能够向用户展示推荐依据，从而提升系统的可解释性和用户体验。

\subsection{本章小结}

本章从系统工程实现角度介绍了推荐系统的整体实现过程。首先说明了系统全栈技术环境的搭建，包括 Streamlit 前端框架、SQLite 数据库以及基于 API 的大语言模型接口。随后介绍了多 Agent 推荐模块的实现方式，并说明了面向对象设计在系统扩展性方面的优势。

在数据层方面，本章对数据库建模与数据清洗流程进行了说明，并构建了结构化商品数据库，为推荐模块提供稳定的数据来源。最后，通过商品卡片组件与交互式图表设计，实现了推荐结果的可视化展示。

通过上述实现过程可以看出，本文提出的推荐系统不仅在理论设计层面具有可行性，同时也能够在实际应用环境中得到有效实现。