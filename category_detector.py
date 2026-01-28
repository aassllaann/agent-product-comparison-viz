"""
品类识别器

使用 LLM 从用户输入中：
1. 识别商品品类
2. 动态构建评分体系
"""

import json
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
import config
from base_agent import CategoryConfig, ScoringDimension


class CategoryDetector:
    """
    品类识别器
    
    负责从用户自然语言输入中识别商品品类，
    并能动态为新品类构建评分体系。
    """
    
    # 已知品类映射（三级分类体系）
    # 大类：数码产品 -> 中类 -> 小类
    KNOWN_CATEGORIES = {
        # ========== 中类1: 核心电子设备 (Core Electronics) ==========
        "phone": {
            "name": "手机",
            "category_level_2": "核心电子设备",
            "group": "已有数据",
            "keywords": ["手机", "智能手机", "phone", "iphone", "android"],
            "ecosystem_tags": ["移动娱乐", "音频发烧"]
        },
        "tablet": {
            "name": "平板电脑",
            "category_level_2": "核心电子设备",
            "groups": "已有数据",
            "keywords": ["平板", "ipad", "tablet", "pad", "床上", "追剧", "看电影", "网课"],
            "ecosystem_tags": ["移动娱乐", "摄影创作"]
        },
        "laptop": {
            "name": "笔记本电脑",
            "category_level_2": "核心电子设备",
            "group": "已有数据",
            "keywords": ["笔记本", "电脑", "laptop", "notebook", "macbook", "游戏本"],
            "ecosystem_tags": ["移动办公", "专业游戏", "摄影创作"]
        },
        
        # ========== 中类2: 影音设备 (Audio-Visual) ==========
        "camera": {
            "name": "相机",
            "category_level_2": "影音设备",
            "group": "已有数据",
            "keywords": ["相机", "单反", "微单", "数码相机", "camera"],
            "ecosystem_tags": ["摄影创作"]
        },
        "headphone": {
            "name": "耳机",
            "category_level_2": "影音设备",
            "group": "已有数据",
            "keywords": ["耳机", "headphone", "airpods", "耳麦"],
            "ecosystem_tags": ["移动娱乐", "音频发烧"]
        },
        "bluetooth_speaker": {
            "name": "蓝牙音箱",
            "category_level_2": "影音设备",
            "group": "新增数据",
            "keywords": ["蓝牙音箱", "音箱", "speaker", "便携音箱", "无线音箱"],
            "ecosystem_tags": ["移动办公", "移动娱乐", "音频发烧"]
        },
        
        # ========== 中类3: 数码配件 (Digital Accessories) ==========
        "smartwatch": {
            "name": "智能手表",
            "category_level_2": "数码配件",
            "group": "新增数据",
            "keywords": ["智能手表", "手表", "watch", "apple watch", "运动手表"],
            "ecosystem_tags": ["移动办公", "移动娱乐"]
        },
        "monitor": {
            "name": "显示器",
            "category_level_2": "数码配件",
            "group": "新增数据",
            "keywords": ["显示器", "monitor", "电脑显示器", "4k显示器", "外接屏"],
            "ecosystem_tags": ["移动办公", "专业游戏", "摄影创作"]
        },
        
        # ========== 中类4: 游戏设备 (Gaming) ==========
        "gaming_console": {
            "name": "游戏主机",
            "category_level_2": "游戏设备",
            "group": "新增数据",
            "keywords": ["游戏主机", "主机", "ps5", "xbox", "switch", "掌机"],
            "ecosystem_tags": ["专业游戏"]
        },
        "gpu": {
            "name": "显卡",
            "category_level_2": "游戏设备",
            "group": "新增数据",
            "keywords": ["显卡", "gpu", "显卡", "nvidia", "amd", "rtx", "独显"],
            "ecosystem_tags": ["专业游戏"]
        },
        
        # ========== 其他品类（暂无数据） ==========
        "skincare": {
            "name": "护肤品",
            "category_level_2": "美妆护肤",
            "group": "暂无数据",
            "keywords": ["护肤品", "面霜", "精华", "水乳", "洁面", "爽肤水", "乳液", "面膜"],
            "ecosystem_tags": []
        },
        "cosmetics": {
            "name": "化妆品",
            "category_level_2": "美妆护肤",
            "group": "暂无数据",
            "keywords": ["化妆品", "口红", "粉底", "眼影", "腮红", "彩妆"],
            "ecosystem_tags": []
        },
        "stationery": {
            "name": "文具",
            "category_level_2": "办公文教",
            "group": "暂无数据",
            "keywords": ["文具", "钢笔", "中性笔", "铅笔", "圆珠笔"],
            "ecosystem_tags": []
        },
        "office": {
            "name": "办公用品",
            "category_level_2": "办公文教",
            "group": "暂无数据",
            "keywords": ["打印机", "订书机", "复印机", "投影仪"],
            "ecosystem_tags": []
        },
        "book": {
            "name": "图书",
            "category_level_2": "办公文教",
            "group": "暂无数据",
            "keywords": ["图书", "小说", "教材", "书籍"],
            "ecosystem_tags": []
        },
        "appliance": {
            "name": "小家电",
            "category_level_2": "生活百货",
            "group": "暂无数据",
            "keywords": ["吹风机", "电饭煲", "烤箱", "榨汁机", "咖啡机", "空气炸锅", "冰箱", "洗衣机"],
            "ecosystem_tags": []
        },
        "sports": {
            "name": "运动装备",
            "category_level_2": "生活百货",
            "group": "暂无数据",
            "keywords": ["跑鞋", "球拍", "跑步机", "哑铃", "篮球", "足球"],
            "ecosystem_tags": []
        },
    }
    
    def __init__(self):
        self.client = OpenAI(
            api_key=config.DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    
    def detect_category(self, user_msg: str) -> Tuple[str, str]:
        """
        从用户输入识别品类
        
        Args:
            user_msg: 用户输入的消息
            
        Returns:
            (category_key, category_name) 元组
            如 ("camera", "相机")
        """
        user_msg_lower = user_msg.lower()
        
        # 1. 先尝试关键词匹配（快速路径）
        for category_key, info in self.KNOWN_CATEGORIES.items():
            if any(kw in user_msg_lower for kw in info["keywords"]):
                return (category_key, info["name"])
        
        # 2. 使用 LLM 识别（慢速路径）
        return self._llm_detect(user_msg)
    
    def _llm_detect(self, user_msg: str) -> Tuple[str, str]:
        """使用 LLM 进行品类识别"""
        known_list = ", ".join([
            f"{k}({v['name']})" for k, v in self.KNOWN_CATEGORIES.items()
        ])
        
        system_prompt = f"""
        你是一个商品品类识别专家。请从用户输入中识别他们想要推荐的商品品类。
        
        已知品类: {known_list}
        
        【规则】：
        1. **优先匹配**: 如果用户需求（即使模糊）能对应到已知品类，必须返回对应的已知英文 key。例如“在床上看电影”应匹配到 tablet (平板)，“出差办公装备”应匹配到 laptop (笔记本)。
        2. **复合需求**: 如果用户提到多个品类（如“办公和听歌”），请返回最核心的一个（如 laptop）。
        3. **新品类**: 仅当需求完全不属于已知品类时，才返回 is_new: true 和自定义 key。
        
        输出 JSON: {{"category_key": "英文标识", "category_name": "中文名称", "is_new": true/false}}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "")
            
            result = json.loads(content)
            return (result["category_key"], result["category_name"])
            
        except Exception as e:
            print(f"LLM 品类识别失败: {e}")
            # 默认返回相机
            return ("camera", "相机")
    
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
        
        try:
            response = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请为{category_name}品类设计评分体系。用户上下文：{user_context}"}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "")
            
            return json.loads(content)
            
        except Exception as e:
            print(f"LLM 评分体系构建失败: {e}")
            # 返回通用默认配置
            return {
                "dimensions": [
                    {"name": "性价比", "field": "Value_Score", "weight": 0.4, "description": "价格与性能比"},
                    {"name": "性能", "field": "Performance_Score", "weight": 0.3, "description": "核心性能表现"},
                    {"name": "品质", "field": "Quality_Score", "weight": 0.3, "description": "做工和品质"}
                ],
                "scenarios": {
                    "general": {"keywords": ["推荐", "选购"], "presets": []}
                },
                "default_sort_field": "Value_Score"
            }
    
    def create_category_config(
        self, 
        category_key: str, 
        category_name: str,
        scoring_system: Dict
    ) -> CategoryConfig:
        """
        从评分体系创建 CategoryConfig 对象
        
        Args:
            category_key: 品类英文标识
            category_name: 品类中文名称
            scoring_system: build_scoring_system 的返回值
            
        Returns:
            CategoryConfig 实例
        """
        # 转换评分维度
        dimensions = [
            ScoringDimension(
                name=d["name"],
                field=d["field"],
                weight=d["weight"],
                description=d["description"]
            )
            for d in scoring_system.get("dimensions", [])
        ]
        
        # 转换场景配置
        scenario_presets = {}
        scenario_keywords = {}
        
        for scenario_key, scenario_info in scoring_system.get("scenarios", {}).items():
            scenario_presets[scenario_key] = scenario_info.get("presets", [])
            scenario_keywords[scenario_key] = scenario_info.get("keywords", [])
        
        return CategoryConfig(
            name=category_name,
            name_en=category_key,
            table_name=f"{category_key}s",  # 如 cameras, phones
            scoring_dimensions=dimensions,
            scenario_presets=scenario_presets,
            scenario_keywords=scenario_keywords,
            default_sort_field=scoring_system.get("default_sort_field", "Value_Score")
        )
