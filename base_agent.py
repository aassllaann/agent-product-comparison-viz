"""
通用商品推荐代理基类

定义所有品类共享的：
1. 评分维度抽象
2. 推荐流程模板
3. 通用筛选/排序逻辑
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from openai import OpenAI
import config


@dataclass
class ScoringDimension:
    """评分维度定义"""
    name: str           # 维度中文名称 (如 "便携性")
    field: str          # 数据库字段名 (如 "Portability_Score")
    weight: float       # 权重 (0-1)
    description: str    # 维度描述
    
    def __post_init__(self):
        if not 0 <= self.weight <= 1:
            raise ValueError(f"权重必须在 0-1 之间, 当前: {self.weight}")


@dataclass
class CategoryConfig:
    """品类配置"""
    name: str                                       # 品类名称 (如 "相机")
    name_en: str                                    # 英文标识 (如 "camera")
    table_name: str                                 # 数据表名
    scoring_dimensions: List[ScoringDimension]      # 评分维度列表
    scenario_presets: Dict[str, List[str]]          # 场景-预设机型
    scenario_keywords: Dict[str, List[str]]         # 场景-关键词
    default_sort_field: str                         # 默认排序字段
    display_fields: List[str] = field(default_factory=list)  # 展示字段


class BaseProductAgent(ABC):
    """
    通用商品推荐代理基类
    
    所有品类代理都应继承此类，并实现抽象方法。
    """
    
    # ==================== 通用评分维度（所有品类共享） ====================
    UNIVERSAL_DIMENSIONS = [
        ScoringDimension(
            name="性价比",
            field="Value_Score",
            weight=0.20,
            description="价格与性能的综合考量"
        ),
        ScoringDimension(
            name="品牌信誉",
            field="Brand_Score",
            weight=0.15,
            description="品牌口碑和售后服务"
        ),
        ScoringDimension(
            name="用户评价",
            field="User_Rating",
            weight=0.15,
            description="用户真实反馈评分"
        ),
    ]
    
    def __init__(self):
        self.client = OpenAI(
            api_key=config.DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self._config: Optional[CategoryConfig] = None
    
    # ==================== 抽象方法：子类必须实现 ====================
    
    @abstractmethod
    def get_category_config(self) -> CategoryConfig:
        """
        获取品类特有配置
        
        子类实现此方法返回品类的场景预设、关键词等配置
        """
        pass
    
    @abstractmethod
    def get_specific_dimensions(self) -> List[ScoringDimension]:
        """
        获取品类特有评分维度
        
        例如相机有 "低光画质"、"视频能力"，
        手机有 "续航"、"屏幕素质" 等
        """
        pass
    
    @abstractmethod
    def get_db_session(self):
        """获取数据库会话"""
        pass
    
    @abstractmethod
    def get_model_class(self):
        """获取 SQLAlchemy 模型类"""
        pass
    
    @abstractmethod
    def get_product_info_text(self, product) -> str:
        """
        格式化单个商品信息为文本
        
        用于 LLM prompt 中的商品描述
        """
        pass
    
    # ==================== 通用方法：推荐流程 ====================
    
    @property
    def config(self) -> CategoryConfig:
        """懒加载配置"""
        if self._config is None:
            self._config = self.get_category_config()
        return self._config
    
    def get_all_dimensions(self) -> List[ScoringDimension]:
        """合并通用 + 特有评分维度"""
        return self.UNIVERSAL_DIMENSIONS + self.get_specific_dimensions()
    
    def get_dimension_fields(self) -> List[str]:
        """获取所有评分字段名"""
        return [d.field for d in self.get_all_dimensions()]
    
    def detect_scenario(self, usage: str, summary: str) -> Optional[str]:
        """
        根据意图识别场景
        
        Args:
            usage: 解析出的用途
            summary: 解析出的摘要
            
        Returns:
            匹配的场景名称，或 None
        """
        usage_lower = usage.lower()
        summary_lower = summary.lower()
        
        # 通过关键词匹配场景
        for scenario, keywords in self.config.scenario_keywords.items():
            if any(k in usage_lower for k in keywords) or any(k in summary_lower for k in keywords):
                return scenario
        
        # 回退到场景名称直接匹配
        for scenario in self.config.scenario_presets.keys():
            if scenario in usage_lower or scenario in summary_lower:
                return scenario
        
        return None
    
    def get_preset_products(self, scenario: str) -> List[Any]:
        """
        根据场景获取预设商品
        
        Args:
            scenario: 场景名称
            
        Returns:
            匹配的商品列表
        """
        preset_names = self.config.scenario_presets.get(scenario, [])
        if not preset_names:
            return []
        
        Model = self.get_model_class()
        db = self.get_db_session()
        
        candidates = []
        seen_ids = set()
        
        for name in preset_names:
            # 模糊匹配 Model 或 Brand
            products = db.query(Model).filter(
                (Model.Model.ilike(f"%{name}%")) | (Model.Brand.ilike(f"%{name}%"))
            ).all()
            
            for p in products:
                if p.id not in seen_ids:
                    candidates.append(p)
                    seen_ids.add(p.id)
        
        return candidates
    
    def filter_candidates(self, candidates: List[Any], intent: Dict) -> List[Any]:
        """
        对候选集进行筛选和排序
        
        Args:
            candidates: 候选商品列表
            intent: 解析后的用户意图
            
        Returns:
            筛选并排序后的商品列表
        """
        filtered = []
        max_price = intent.get('max_price', 15000)
        
        for product in candidates:
            # 价格过滤
            if hasattr(product, 'Price') and product.Price:
                if product.Price > max_price:
                    continue
            elif hasattr(product, 'price') and product.price:
                if product.price > max_price:
                    continue
            
            filtered.append(product)
        
        # 排序
        sort_field = intent.get('sort_field', self.config.default_sort_field)
        Model = self.get_model_class()
        
        if not hasattr(Model, sort_field):
            sort_field = self.config.default_sort_field
        
        filtered.sort(
            key=lambda x: getattr(x, sort_field, 0) or 0,
            reverse=True
        )
        
        return filtered
    
    def fallback_search(self, intent: Dict, exclude_ids: List[int] = None, limit: int = 3) -> List[Any]:
        """
        全库搜索兜底
        
        当预设不足时，从全库中搜索补充
        """
        from sqlalchemy import desc
        
        if exclude_ids is None:
            exclude_ids = []
        
        Model = self.get_model_class()
        db = self.get_db_session()
        
        # 构建查询
        price_field = 'Price' if hasattr(Model, 'Price') else 'price'
        query = db.query(Model).filter(
            getattr(Model, price_field) <= intent.get('max_price', 15000)
        )
        
        if exclude_ids:
            query = query.filter(~Model.id.in_(exclude_ids))
        
        # 排序
        sort_field = intent.get('sort_field', self.config.default_sort_field)
        if hasattr(Model, sort_field):
            query = query.order_by(desc(getattr(Model, sort_field)))
        
        return query.limit(limit).all()
    
    def build_intent_prompt(self) -> str:
        """
        构建意图解析的 system prompt
        
        子类可覆盖以自定义
        """
        dimensions_desc = "\n".join([
            f"- {d.name}: {d.description}" 
            for d in self.get_all_dimensions()
        ])
        
        scenarios_desc = ", ".join(self.config.scenario_presets.keys())
        
        return f"""
        你是一个{self.config.name}导购专家，负责从对话中提取结构化信息。

        【品类】: {self.config.name}
        
        【可用评分维度】:
        {dimensions_desc}
        
        【典型使用场景】: {scenarios_desc}

        【待提取信息】：
        1. usage (用途): 用户的使用场景
        2. max_price (预算): 数字，默认 15000
        3. sort_field (排序字段): 根据用户偏好选择评分维度的字段名
        4. summary (摘要): 解析出的核心诉求

        输出严格 JSON：{{"max_price": 数字, "sort_field": "字段名", "summary": "核心诉求", "usage": "场景关键词"}}
        """
    
    def generate_recommendation_reasons(
        self, 
        results: List[Any], 
        user_msg: str, 
        intent: Dict,
        history: List[Dict] = None
    ) -> List[str]:
        """
        为每个推荐商品生成推荐理由
        
        Args:
            results: 推荐商品列表
            user_msg: 用户原始消息
            intent: 解析后的意图
            history: 对话历史
            
        Returns:
            推荐理由列表
        """
        if not results:
            return []
        
        reasons = []
        for idx, product in enumerate(results):
            product_info = self.get_product_info_text(product)
            
            system_prompt = f"""
            你是一个专业{self.config.name}导购专家。现在需要为用户推荐的第{idx+1}款商品生成推荐理由。
            
            用户需求: {user_msg}
            核心诉求: {intent.get('summary', '综合选购')}
            预算: {intent.get('max_price', 15000)}元
            关注指标: {intent.get('sort_field', self.config.default_sort_field)}
            
            推荐商品信息:
            {product_info}
            
            要求：
            1. 生成2-3句话的简短推荐理由（80-150字）
            2. 紧密结合用户需求，说明为什么这款商品适合用户
            3. 突出该商品的核心优势
            4. 语言亲切自然，避免空洞的形容词
            5. 直接输出推荐理由文本，不要包含标题
            """
            
            messages = [{"role": "system", "content": system_prompt}]
            if history:
                messages.extend(history[-2:])
            messages.append({"role": "user", "content": f"请生成推荐理由"})
            
            try:
                response = self.client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=messages,
                    stream=False
                )
                reason = response.choices[0].message.content.strip()
                reasons.append(reason)
            except Exception as e:
                print(f"LLM Error for product {idx}: {e}")
                # 备用推荐理由
                reasons.append(f"这款商品在您关注的领域表现出色，非常适合您的使用场景。")
        
        return reasons
