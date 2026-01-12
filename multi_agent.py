"""
多品类推荐调度器

统一入口，负责：
1. 品类识别
2. 代理调度
3. 动态代理创建
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from openai import OpenAI
import config
from category_detector import CategoryDetector
from base_agent import BaseProductAgent, CategoryConfig, ScoringDimension


class MultiCategoryAgent:
    """
    多品类推荐代理
    
    作为统一入口，根据用户输入自动识别品类，
    并调度到对应的品类代理进行推荐。
    """
    
    def __init__(self):
        self.detector = CategoryDetector()
        self.client = OpenAI(
            api_key=config.DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 已注册的品类代理
        self._agents: Dict[str, BaseProductAgent] = {}
        
        # 动态创建的品类配置缓存
        self._dynamic_configs: Dict[str, CategoryConfig] = {}
        
        # 初始化已知品类代理
        self._init_known_agents()
    
    def _init_known_agents(self):
        """初始化已知品类的代理"""
        # 导入并注册相机代理（主品类）
        try:
            from main_agent import CameraAgent
            self._agents["camera"] = CameraAgent()
        except ImportError:
            print("Warning: CameraAgent not found")
        
        # 后续可以添加更多品类
        # from agents.phone_agent import PhoneAgent
        # self._agents["phone"] = PhoneAgent()
    
    def register_agent(self, category_key: str, agent: BaseProductAgent):
        """
        注册一个品类代理
        
        Args:
            category_key: 品类英文标识
            agent: 代理实例
        """
        self._agents[category_key] = agent
    
    def get_agent(self, category_key: str) -> Optional[BaseProductAgent]:
        """
        获取品类代理
        
        如果不存在则返回 None
        """
        return self._agents.get(category_key)
    
    def handle_chat(
        self, 
        user_msg: str, 
        history: List[Dict] = None
    ) -> Tuple[Any, Any, Any, Any]:
        """
        处理用户对话
        
        自动识别品类并调用对应代理
        
        Args:
            user_msg: 用户消息
            history: 对话历史
            
        Returns:
            与 CameraAgent.handle_chat 相同的返回格式
        """
        # 1. 识别品类
        category_key, category_name = self.detector.detect_category(user_msg)
        print(f"[MultiAgent] 识别品类: {category_key} ({category_name})")
        
        # 2. 获取或创建代理
        agent = self._agents.get(category_key)
        
        if agent is None:
            # 尝试创建动态代理
            agent = self._create_dynamic_agent(category_key, category_name, user_msg)
            if agent:
                self._agents[category_key] = agent
            else:
                # 回退到相机代理
                print(f"[MultiAgent] 无法创建 {category_key} 代理，回退到相机")
                agent = self._agents.get("camera")
                if agent is None:
                    return "抱歉，系统暂时无法处理您的请求。", None, None, None
        
        # 3. 调用代理处理
        return agent.handle_chat(user_msg, history)
    
    def _create_dynamic_agent(
        self, 
        category_key: str, 
        category_name: str,
        user_context: str
    ) -> Optional[BaseProductAgent]:
        """
        动态创建新品类的推荐代理
        
        当用户请求一个未预置的品类时，
        使用 LLM 构建评分体系并创建临时代理。
        
        注意：动态代理目前不支持数据库查询，
        需要后续实现爬虫接口获取数据。
        """
        print(f"[MultiAgent] 尝试动态创建 {category_key} 代理...")
        
        # 构建评分体系
        scoring_system = self.detector.build_scoring_system(
            category_key, 
            category_name, 
            user_context
        )
        
        # 创建配置
        config = self.detector.create_category_config(
            category_key,
            category_name,
            scoring_system
        )
        
        self._dynamic_configs[category_key] = config
        
        # 创建动态代理
        return DynamicAgent(config, self.client)
    
    def get_available_categories(self) -> List[Dict[str, str]]:
        """
        获取所有可用品类列表
        
        Returns:
            品类信息列表 [{"key": "camera", "name": "相机"}, ...]
        """
        categories = []
        
        # 已注册的代理
        for key, agent in self._agents.items():
            try:
                config = agent.get_category_config()
                categories.append({
                    "key": config.name_en,
                    "name": config.name
                })
            except:
                categories.append({"key": key, "name": key})
        
        # 加上已知但未注册的品类
        for key, info in self.detector.KNOWN_CATEGORIES.items():
            if key not in self._agents:
                categories.append({
                    "key": key,
                    "name": info["name"],
                    "available": False
                })
        
        return categories


class DynamicAgent(BaseProductAgent):
    """
    动态品类代理
    
    用于处理所有非相机品类，
    使用实时 API + 内存缓存获取商品数据。
    """
    
    def __init__(self, config: CategoryConfig, client: OpenAI = None):
        self._config = config
        
        if client:
            self.client = client
        else:
            super().__init__()
        
        # 导入数据服务
        from data_service import get_data_service
        self.data_service = get_data_service()
    
    def get_category_config(self) -> CategoryConfig:
        return self._config
    
    def get_specific_dimensions(self) -> List[ScoringDimension]:
        return self._config.scoring_dimensions
    
    def get_db_session(self):
        return None
    
    def get_model_class(self):
        return None
    
    def get_product_info_text(self, product) -> str:
        if isinstance(product, dict):
            brand = product.get('brand', '')
            model = product.get('model', '')
            price = product.get('price', 0)
            scores = product.get('scores', {})
            specs = product.get('specs', {})
            
            info = f"品牌型号: {brand} {model}\n价格: {price}元\n"
            if scores:
                info += "评分: " + ", ".join([f"{k}: {v}" for k, v in scores.items()]) + "\n"
            if specs:
                info += "规格: " + ", ".join([f"{k}: {v}" for k, v in list(specs.items())[:5]])
            return info
        return str(product)
    
    def _parse_intent(self, user_msg: str, history=None) -> dict:
        """解析用户意图"""
        system_prompt = f"""
        你是一个{self._config.name}导购专家，负责从对话中提取结构化信息。
        
        【品类】: {self._config.name}
        
        【待提取信息】：
        1. max_price (预算): 数字，默认 10000
        2. sort_field (排序字段): 根据用户偏好选择，默认 Value_Score
        3. summary (摘要): 用户核心诉求
        4. usage (用途): 使用场景
        
        输出严格 JSON：{{"max_price": 数字, "sort_field": "字段名", "summary": "核心诉求", "usage": "场景"}}
        """
        
        try:
            messages = [{"role": "system", "content": system_prompt}]
            if history:
                messages.extend(history[-4:])
            messages.append({"role": "user", "content": user_msg})
            
            response = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "")
            
            return json.loads(content)
        except Exception as e:
            print(f"[DynamicAgent] 意图解析失败: {e}")
            return {"max_price": 10000, "sort_field": "Value_Score", "summary": "综合选购", "usage": ""}
    
    def handle_chat(self, user_msg: str, history: List[Dict] = None):
        """
        动态代理的对话处理
        
        使用数据服务获取实时商品数据。
        """
        category_key = self._config.name_en
        category_name = self._config.name
        
        # 1. 解析意图
        intent = self._parse_intent(user_msg, history)
        print(f"[DynamicAgent] {category_name} 意图: {intent}")
        
        # 2. 获取商品数据
        products = self.data_service.get_products_by_filter(
            category=category_key,
            max_price=intent.get('max_price', 10000),
            sort_by=intent.get('sort_field', 'Value_Score'),
            limit=50
        )
        
        if not products:
            return f"抱歉，暂时没有找到符合条件的{category_name}商品。请尝试调整预算或需求。", None, None, None
        
        # 3. 选取前3个推荐
        results = products[:3]
        
        # 4. 生成推荐理由
        reasons = self._generate_reasons(results, user_msg, intent)
        
        # 5. 生成可视化（简化版）
        charts = self._generate_charts(results, intent)
        
        # 6. 生成分析
        chart_analyses = self._generate_analyses(results, intent)
        
        return reasons, charts, results, chart_analyses
    
    def _generate_reasons(self, results: List[dict], user_msg: str, intent: dict) -> List[str]:
        """为每个推荐商品生成理由"""
        reasons = []
        category_name = self._config.name
        
        for idx, product in enumerate(results):
            product_info = self.get_product_info_text(product)
            
            system_prompt = f"""
            你是一个{category_name}专业导购。请为这款商品生成一段精炼的推荐理由。
            
            用户需求: {user_msg}
            商品: {product.get('brand')} {product.get('model')}
            信息: {product_info}
            
            要求：
            1. **禁止废话**（如"这是一款不错的选择"）。
            2. 输出格式：
               ✨ **核心优势**：[内容]\n\n🎯 **适用场景**：[内容]
            3. 总字数控制在 150 字以内。
            """
            
            try:
                response = self.client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "生成推荐理由"}
                    ]
                )
                reasons.append(response.choices[0].message.content.strip())
            except:
                reasons.append(f"这款{product.get('brand', '')} {product.get('model', '')}性价比高，适合您的需求。")
        
        return reasons
    
    def _generate_charts(self, results: List[dict], intent: dict):
        """生成三个美观的可视化图表"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            import pandas as pd
            
            charts = []
            names = [f"{r.get('brand', '')} {r.get('model', '')}" for r in results]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']
            
            # --- 图表 1: 综合能力雷达图 ---
            fig_radar = go.Figure()
            
            # 获取所有共有评分字段
            common_scores = set()
            for r in results:
                if r.get('scores'):
                    if not common_scores:
                        common_scores = set(r['scores'].keys())
                    else:
                        common_scores &= set(r['scores'].keys())
            
            score_fields = list(common_scores)[:5]  # 最多5个维度
            
            for i, r in enumerate(results):
                scores = r.get('scores', {})
                values = [scores.get(f, 0) for f in score_fields]
                # 闭合雷达图
                values += values[:1]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=[f.replace('_Score', '').replace('_', ' ') for f in score_fields] + [score_fields[0].replace('_Score', '')],
                    fill='toself',
                    name=names[i],
                    line_color=colors[i % len(colors)],
                    opacity=0.7
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100]),
                    bgcolor='rgba(255,255,255,0.9)'
                ),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                margin=dict(l=40, r=40, t=20, b=40),
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            charts.append(fig_radar)
            
            # --- 图表 2: 核心得分对比 (分组柱状图) ---
            # 选取前3个重要维度
            key_dimensions = score_fields[:3]
            data = []
            for r in results:
                name = f"{r.get('brand', '')} {r.get('model', '')}"
                for dim in key_dimensions:
                    data.append({
                        "Product": name,
                        "Dimension": dim.replace('_Score', '').replace('_', ' '),
                        "Score": r.get('scores', {}).get(dim, 0)
                    })
            
            df_bar = pd.DataFrame(data)
            fig_bar = px.bar(
                df_bar, 
                x="Dimension", 
                y="Score", 
                color="Product", 
                barmode="group",
                color_discrete_sequence=colors,
                height=350
            )
            fig_bar.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis_title="",
                yaxis_title="得分"
            )
            charts.append(fig_bar)
            
            # --- 图表 3: 价格与性价比分布 (散点图) ---
            price_data = []
            for i, r in enumerate(results):
                name = f"{r.get('brand', '')} {r.get('model', '')}"
                price = r.get('price', 0)
                # 使用 Value_Score 或第一个评分作为 Y 轴
                score = r.get('scores', {}).get('Value_Score', 0)
                # 如果没有性价比分，计算平均分
                if score == 0 and r.get('scores'):
                    score = sum(r['scores'].values()) / len(r['scores'])
                
                price_data.append({
                    "Product": name,
                    "Price": price,
                    "Score": score,
                    "Size": 20  # 固定大小圆点
                })
            
            df_scatter = pd.DataFrame(price_data)
            fig_scatter = px.scatter(
                df_scatter,
                x="Price",
                y="Score",
                color="Product",
                size="Size",
                color_discrete_sequence=colors,
                height=350
            )
            fig_scatter.update_layout(
                plot_bgcolor='rgba(240,242,246,0.5)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="价格 (元)",
                yaxis_title="综合评分",
                xaxis=dict(showgrid=True, gridcolor='white'),
                yaxis=dict(showgrid=True, gridcolor='white')
            )
            # 添加标签
            for i, row in df_scatter.iterrows():
                fig_scatter.add_annotation(
                    x=row['Price'],
                    y=row['Score'],
                    text=row['Product'].split(' ')[-1], # 只显示型号
                    yshift=15,
                    showarrow=False
                )
            
            charts.append(fig_scatter)
            
            return tuple(charts)
        except Exception as e:
            print(f"[DynamicAgent] 图表生成失败: {e}")
            import traceback
            traceback.print_exc()
            return (None, None, None)
    
    def _generate_analyses(self, results: List[dict], intent: dict) -> List[str]:
        """为图表生成分析结论"""
        analyses = []
        best = results[0]
        analyses.append(f"{best.get('brand', '')} {best.get('model', '')}综合表现最佳，是首选推荐。")
        
        return analyses

