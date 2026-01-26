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
        # 统一从 electronics_agents 导入
        try:
            from electronics_agents import (
                CameraAgent, 
                PhoneAgent, 
                LaptopAgent, 
                HeadphoneAgent, 
                TabletAgent
            )
            self._agents["camera"] = CameraAgent()
            self._agents["phone"] = PhoneAgent()
            self._agents["laptop"] = LaptopAgent()
            self._agents["headphone"] = HeadphoneAgent()
            self._agents["tablet"] = TabletAgent()
        except ImportError as e:
            print(f"Warning: Agents load failed: {e}")
    
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
        """
        # 1. 识别品类
        category_key, category_name = self.detector.detect_category(user_msg)
        print(f"[MultiAgent] 识别品类: {category_key} ({category_name})")
        
        # 2. 获取代理
        agent = self._agents.get(category_key)
        
        if agent is None:
            # 尝试创建动态代理
            agent = self._create_dynamic_agent(category_key, category_name, user_msg)
            if agent:
                self._agents[category_key] = agent
            else:
                # 最后的兜底
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
    动态品类代理 (无数据模式)
    
    用于处理所有非电子产品品类。
    由于去除了 API 爬虫，该代理只负责：
    1. 识别品类并建立评分维度
    2. 解析用户意图
    3. 提供通用的选购建议（无具体商品推荐）
    """
    
    def __init__(self, config: CategoryConfig, client: OpenAI = None):
        self._config = config
        
        if client:
            self.client = client
        else:
            super().__init__()
        
        # 不再初始化数据服务
        self.data_service = None
    
    def get_category_config(self) -> CategoryConfig:
        return self._config
    
    def get_specific_dimensions(self) -> List[ScoringDimension]:
        return self._config.scoring_dimensions
    
    def get_db_session(self):
        return None
    
    def get_model_class(self):
        return None
    
    def get_product_info_text(self, product) -> str:
        return ""
    
    def _parse_intent(self, user_msg: str, history=None) -> dict:
        """解析用户意图"""
        system_prompt = f"""
        你是一个{self._config.name}导购专家，负责从对话中提取结构化信息。
        
        【品类】: {self._config.name}
        
        【待提取信息】：
        1. max_price (预算): 数字
        2. user_needs (核心需求): 用户最关心的点
        3. usage (用途): 使用场景
        
        输出严格 JSON：{{"max_price": 数字, "user_needs": "需求点", "usage": "场景"}}
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
            return {"max_price": 0, "user_needs": "通用", "usage": ""}
    
    def handle_chat(self, user_msg: str, history: List[Dict] = None):
        """
        动态代理的对话处理 (无数据版)
        """
        category_name = self._config.name
        
        # 1. 解析意图
        intent = self._parse_intent(user_msg, history)
        print(f"[DynamicAgent] {category_name} 意图: {intent}")
        
        # 2. 生成通用建议
        advice = self._generate_general_advice(user_msg, intent)
        
        # 3. 构造返回
        # 格式: replies(str), charts(tuple), results(list), chart_analyses(list)
        # charts, results, chart_analyses 均为空（前端应适配处理）
        
        return advice, (None, None, None), [], []
    
    def _generate_general_advice(self, user_msg: str, intent: dict) -> str:
        """生成通用选购建议"""
        dimensions_str = "、".join([f"{d.name}(权重{d.weight})" for d in self._config.scoring_dimensions])
        
        system_prompt = f"""
        你是一个{self._config.name}领域的资深专家。
        系统目前没有该品类的具体商品数据库，但已为你分析出了评分维度：{dimensions_str}。
        
        用户咨询: "{user_msg}"
        用户意图: {intent}
        
        任务：
        请给用户写一段专业的**通用选购指南**。
        1. 告知用户目前系统暂无{self._config.name}的具体实时数据。
        2. 结合定义的评分维度，告诉用户选购此类商品应该关注什么（例如：如果是买{self._config.name}，重点看{self._config.scoring_dimensions[0].name}）。
        3. 给出3-5条具体的选购参数建议（例如：“建议选择 xxx 材质”，“参数 A 最好大于 xxx”），也就是常识性的建议。
        4. 语气亲切、专业。
        """
        
        try:
            response = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "请生成选购指南"}
                ]
            )
            return response.choices[0].message.content.strip()
        except:
            return f"系统目前暂无部分{self._config.name}数据，建议您关注{dimensions_str}等指标进行选购。"
