import json
from typing import List, Dict, Any, Optional
from sqlalchemy import desc
from models import SessionLocal, Phone, Laptop, Headphone, Tablet, Camera
import visualizer
from openai import OpenAI
import config
from base_agent import BaseProductAgent, CategoryConfig, ScoringDimension

class BaseDbAgent(BaseProductAgent):
    """
    通用数据库代理基类
    
    提取 CameraAgent 中的通用逻辑，用于电子产品代理。
    """
    def __init__(self):
        super().__init__()
        self.db = SessionLocal()
        
    def get_db_session(self):
        return self.db

    def _parse_intent_generic(self, user_msg: str, category_name: str, fields_desc: str, history=None) -> dict:
        """通用意图解析"""
        system_rules = f"""
        你是一个{category_name}导购专家，负责从对话中提取结构化信息。

        【待提取维度（用户均可缺省）】：
        1. usage (用途): 使用场景关键词
        2. budget_level (投入): 预算描述
        3. max_price (预算): 数字，单位元，默认 100000 (无限)
        4. sort_field (排序字段): 基于用户需求选择最合适的评分字段 ({fields_desc})
        5. summary (包含): 解析出的核心诉求摘要
        6. product_type (类型): 显式要求的产品特定类型（如"头戴式"、"入耳式"、"游戏本"等），无明确要求则留空

        【推理规则】：
        - 预算缺省默认为 20000。
        - 若无明显排序偏好，默认使用 Value_Score (性价比) 或 Performance_Score (性能)。
        - 仔细区分用户对产品类型的限制，例如"头戴式耳机"应提取 product_type="头戴式"。

        输出严格 JSON：{{"max_price": 数字, "sort_field": "字段名", "summary": "核心诉求", "usage": "场景关键词", "product_type": "..."}}
        """
        try:
            messages = [{"role": "system", "content": system_rules}]
            if history:
                messages.extend(history[-2:])
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
        except Exception:
            return {"max_price": 20000, "sort_field": "Value_Score", "summary": "综合推荐", "usage": ""}

    def _get_preset_products(self, scenario: str, model_class):
        """获取预设产品"""
        preset_names = self.get_category_config().scenario_presets.get(scenario, [])
        if not preset_names:
            return []
        
        candidates = []
        seen_ids = set()
        seen_models = set()  # 新增：用于记录已添加的型号名称
        
        for name in preset_names:
            # 模糊匹配 Model 或 Brand
            items = self.db.query(model_class).filter(
                (model_class.Model.ilike(f"%{name}%") | model_class.Brand.ilike(f"%{name}%"))
            ).all()
            for item in items:
                # 检查 ID 和 型号名称
                model_name = item.Model.lower().strip()
                if item.id not in seen_ids and model_name not in seen_models:
                    candidates.append(item)
                    seen_ids.add(item.id)
                    seen_models.add(model_name)
        return candidates
        return candidates

    def _filter_and_sort(self, candidates, intent, model_class):
        """筛选和排序"""
        filtered = []
        max_price = intent.get('max_price', 20000)
        
        for item in candidates:
            if item.Price and item.Price > max_price:
                continue
            
            # 类型过滤
            product_type = intent.get('product_type')
            if product_type and product_type.lower() != "null":
                 # 检查 Type 或 Category 字段
                item_type = getattr(item, 'Type', None) or getattr(item, 'Category', None)
                if item_type and product_type not in item_type:
                    continue

            filtered.append(item)
        
        sort_field = intent.get('sort_field', 'Value_Score')
        if not hasattr(model_class, sort_field):
            sort_field = 'Value_Score' if hasattr(model_class, 'Value_Score') else 'Performance_Score'
            
        filtered.sort(key=lambda x: getattr(x, sort_field, 0) or 0, reverse=True)
        return filtered

    def _fallback_search(self, intent, model_class, exclude_ids=None):
        """兜底搜索"""
        if exclude_ids is None:
            exclude_ids = []
            
        query = self.db.query(model_class).filter(
            model_class.Price <= intent.get('max_price', 20000)
        )
        if exclude_ids:
            query = query.filter(~model_class.id.in_(exclude_ids))
            
        # 类型过滤 (同步核心筛选逻辑)
        product_type = intent.get('product_type')
        if product_type and product_type.lower() != "null":
            # 检查 Type (耳机) 或 Category (笔记本) 字段
            if hasattr(model_class, 'Type'):
                query = query.filter(model_class.Type.ilike(f"%{product_type}%"))
            elif hasattr(model_class, 'Category'):
                query = query.filter(model_class.Category.ilike(f"%{product_type}%"))

        sort_field = intent.get('sort_field', 'Value_Score')
        if hasattr(model_class, sort_field):
            query = query.order_by(desc(getattr(model_class, sort_field)))
        else:
            # 默认排序
             if hasattr(model_class, 'Value_Score'):
                 query = query.order_by(desc(model_class.Value_Score))
            
        return query.limit(3).all()

    def _get_individual_reasons(self, results, user_msg, intent, history=None):
        """生成推荐理由"""
        reasons = []
        for idx, item in enumerate(results):
            info = self.get_product_info_text(item)
            prompt = f"""
            你是一个{self.get_category_config().name}导购。请为这款产品生成一段精炼的推荐理由(80字左右)。
            
            用户需求: {user_msg}
            产品信息: {info}
            核心诉求: {intent.get('summary', '综合')}
            
            要求：突出优势，语言自然，不要只有参数堆砌。
            """
            try:
                response = self.client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=[{"role": "system", "content": prompt}]
                )
                reasons.append(response.choices[0].message.content.strip())
            except:
                reasons.append(f"这款{item.Brand} {item.Model}性能出色，非常适合您的需求。")
        return reasons

    def _generate_chart_analyses(self, results, intent):
        """
        生成详细的图表分析文案 (通用版)
        """
        analyses = []
        if not results:
            return analyses
            
        config = self.get_category_config()
        
        # --- 1. 雷达图分析 (首选推荐画像) ---
        cam = results[0]  # 第一款是首选
        radar_dims = config.scoring_dimensions
        
        scores_desc = []
        high_scores = []
        
        for dim in radar_dims:
            score = getattr(cam, dim.field, 0) or 0
            scores_desc.append(f"{dim.name}评分 {score}")
            if score >= 80:
                high_scores.append(dim.name)
        
        radar_detail = (
            f"首选推荐 {cam.Brand} {cam.Model} 的各项指标如下：{', '.join(scores_desc)}。"
            f" 该产品在{', '.join(high_scores) if high_scores else '各项指标'}方面表现{'优异' if high_scores else '均衡'}。"
        )
        analyses.append(radar_detail)
        
        # --- 2. 核心指标对比分析 ---
        sort_field = intent.get('sort_field', config.default_sort_field)
        # 找到对应的中文名
        sort_name = sort_field
        for d in config.scoring_dimensions:
            if d.field == sort_field:
                sort_name = d.name
                break
                
        top3_scores = []
        for c in results:
            val = getattr(c, sort_field, 0) or 0
            top3_scores.append(f"{c.Brand} {c.Model}: {val}")
        
        compare_detail = (
            f"在您最关注的【{sort_name}】维度上，前三款推荐产品的得分分别为：{'；'.join(top3_scores)}。"
            "分值越高代表该项能力越强。"
        )
        analyses.append(compare_detail)
        
        # --- 3. 多维优势分析 ---
        # 取前3个维度进行分析
        compare_dims = config.scoring_dimensions[:3]
        highlights = []
        
        for dim in compare_dims:
            # 找出该维度得分最高的
            best_in_dim = max(results, key=lambda c: getattr(c, dim.field, 0) or 0)
            score = getattr(best_in_dim, dim.field, 0) or 0
            highlights.append(f"{best_in_dim.Model} 在{dim.name}方面表现最佳({score}分)")
            
        multi_dim_detail = (
            f"综合多维能力分析：{'；'.join(highlights)}。"
            "您可以根据具体的使用场景权衡选择。"
        )
        analyses.append(multi_dim_detail)
        
        return analyses

    def handle_chat_generic(self, user_msg, history, model_class, keyword_map):
        """通用的对话处理流程"""
        config = self.get_category_config()
        fields_desc = ", ".join([f"{d.field}({d.name})" for d in config.scoring_dimensions])
        
        # 1. 解析意图
        intent = self._parse_intent_generic(user_msg, config.name, fields_desc, history)
        
        # 2. 场景匹配
        usage = intent.get('usage', '').lower()
        summary = intent.get('summary', '').lower()
        target_scenario = None
        
        for scenario, keywords in keyword_map.items():
            if any(k in usage for k in keywords) or any(k in summary for k in keywords):
                target_scenario = scenario
                break
        
        # 3. 获取候选
        results = []
        if target_scenario:
            candidates = self._get_preset_products(target_scenario, model_class)
            results = self._filter_and_sort(candidates, intent, model_class)
            results = results[:3]
            
        # 4. 兜底
        if len(results) < 3:
            exclude = [i.id for i in results]
            more = self._fallback_search(intent, model_class, exclude)
            results.extend(more)
            results = results[:3]
            
        if not results:
            return f"抱歉，没有找到合适的{config.name}推荐。", None, None, None
            
        # 5. 生成结果
        reasons = self._get_individual_reasons(results, user_msg, intent, history)
        
        # 准备可视化参数
        # Radar: [(Label, Field, MaxVal)] (假设评分为100分制)
        radar_dims = [(d.name, d.field, 100) for d in config.scoring_dimensions]
        
        # Multi-Dim Compare: [(Label, Field)]
        compare_dims = [(d.name, d.field) for d in config.scoring_dimensions[:3]]
        
        charts = (
            visualizer.draw_radar(results, radar_dims),
            visualizer.draw_comparison(results, intent.get('sort_field', config.default_sort_field)),
            visualizer.draw_multi_dimension_compare(results, compare_dims)
        )
        
        analyses = self._generate_chart_analyses(results, intent)
        
        return reasons, charts, results, analyses

    def __del__(self):
        if hasattr(self, 'db'):
            self.db.close()


class CameraAgent(BaseDbAgent):
    """相机推荐代理"""
    
    SCENARIO_PRESETS = {
        "vlog": ["ZV-E10", "G7 X", "Pocket", "Action", "Z30"],
        "travel": ["X100", "GR III", "a6400", "Z fc", "X-T30", "X-S10"],
        "street": ["GR III", "X100", "Leica", "Pen-F", "X-E4"],
        "portrait": ["A7", "R6", "R5", "Z6", "Z5", "5D"],
        "landscape": ["A7 R", "Z7", "D850", "GFX"],
        "beginner": ["R50", "Z30", "M50", "200D", "D3500", "a6000"]
    }

    SCENARIO_KEYWORDS = {
        "vlog": ["vlog", "视频", "拍片", "直播", "up主"],
        "travel": ["travel", "旅行", "旅游"],
        "street": ["street", "街拍", "人文", "扫街"],
        "portrait": ["portrait", "人像", "写真"],
        "landscape": ["landscape", "风光", "风景", "大片"],
        "beginner": ["beginner", "新手", "入门", "小白", "学生"]
    }

    def get_category_config(self) -> CategoryConfig:
        return CategoryConfig(
            name="相机",
            name_en="camera",
            table_name="cameras",
            scoring_dimensions=self.get_specific_dimensions(),
            scenario_presets=self.SCENARIO_PRESETS,
            scenario_keywords=self.SCENARIO_KEYWORDS,
            default_sort_field="LowLight_Score",
            display_fields=["Brand", "Model", "Price", "LowLight_Score", "Video_Score", "Portability_Score"]
        )

    def get_specific_dimensions(self) -> List[ScoringDimension]:
        return [
            ScoringDimension("便携性", "Portability_Score", 0.3, "如重量体积"),
            ScoringDimension("低光画质", "LowLight_Score", 0.4, "暗光表现"),
            ScoringDimension("视频能力", "Video_Score", 0.3, "视频拍摄能力")
        ]

    def get_model_class(self):
        return Camera

    def get_product_info_text(self, p: Camera) -> str:
        return f"型号:{p.Brand} {p.Model}, 价格:{p.Price}, 低光:{p.LowLight_Score}, 视频:{p.Video_Score}, 便携:{p.Portability_Score}"

    def handle_chat(self, user_msg, history=None):
        return self.handle_chat_generic(user_msg, history, Camera, self.SCENARIO_KEYWORDS)


class PhoneAgent(BaseDbAgent):
    """手机推荐代理"""
    
    SCENARIO_PRESETS = {
        "gaming": ["Redmi K70", "iQOO", "Rog", "Red Magic", "Black Shark"],
        "photography": ["Ultra", "Pro", "Pura", "Find X7", "X100", "Pixel"],
        "daily": ["iPhone", "Samsung", "Xiaomi 14", "Honor", "Oppo", "Vivo"],
        "business": ["Fold", "Mate 60", "S24 Ultra", "iPhone 15 Pro Max"]
    }
    
    SCENARIO_KEYWORDS = {
        "gaming": ["游戏", "电竞", "性能", "王者", "吃鸡", "原神"],
        "photography": ["拍照", "摄影", "像", "单反", "长焦", "夜景"],
        "daily": ["日常", "备用", "老人", "学生", "性价比"],
        "business": ["商务", "办公", "折叠", "高端", "面子"]
    }

    def get_category_config(self) -> CategoryConfig:
        return CategoryConfig(
            name="手机",
            name_en="phone",
            table_name="phones",
            scoring_dimensions=self.get_specific_dimensions(),
            scenario_presets=self.SCENARIO_PRESETS,
            scenario_keywords=self.SCENARIO_KEYWORDS,
            default_sort_field="Value_Score",
            display_fields=["Brand", "Model", "Price", "Processor", "RAM_GB", "Storage_GB", "Battery_mAh", "Camera_MP"]
        )

    def get_specific_dimensions(self) -> List[ScoringDimension]:
        return [
            ScoringDimension("性能", "Performance_Score", 0.3, "处理器与游戏性能"),
            ScoringDimension("拍照", "Camera_Score", 0.3, "影像系统能力"),
            ScoringDimension("续航", "Battery_Score", 0.2, "电池与充电"),
            ScoringDimension("性价比", "Value_Score", 0.2, "价格配置比")
        ]

    def get_model_class(self):
        return Phone

    def get_product_info_text(self, p: Phone) -> str:
        return f"型号:{p.Brand} {p.Model}, 价格:{p.Price}, 芯片:{p.Processor}, 内存:{p.RAM_GB}+{p.Storage_GB}, 电池:{p.Battery_mAh}mAh"

    def handle_chat(self, user_msg, history=None):
        return self.handle_chat_generic(user_msg, history, Phone, self.SCENARIO_KEYWORDS)


class LaptopAgent(BaseDbAgent):
    """笔记本推荐代理"""
    
    SCENARIO_PRESETS = {
        "gaming": ["Legion", "Rog", "Omen", "Alienware", "GeForce"],
        "light_office": ["MacBook Air", "XPS", "Surface", "MateBook X", "Gram", "Swift"],
        "programming": ["MacBook Pro", "ThinkPad", "MateBook 16", "MagicBook Pro"],
        "student": ["Xiaoxin", "RedmiBook", "Honor", "Vivobook"]
    }
    
    SCENARIO_KEYWORDS = {
        "gaming": ["游戏", "3a", "独显", "电竞"],
        "light_office": ["轻薄", "办公", "便携", "出差", "女生"],
        "programming": ["代码", "编程", "开发", "程序员", "linux"],
        "student": ["学生", "网课", "论文", "高性价比"]
    }

    def get_category_config(self):
        return CategoryConfig(
            name="笔记本电脑",
            name_en="laptop",
            table_name="laptops",
            scoring_dimensions=[
                ScoringDimension("性能", "Performance_Score", 0.4, "CPU/GPU性能"),
                ScoringDimension("便携", "Portability_Score", 0.3, "重量与厚度"),
                ScoringDimension("屏幕", "Display_Score", 0.3, "屏幕素质"),
                ScoringDimension("性价比", "Value_Score", 0.0, "综合性价比")
            ],
            scenario_presets=self.SCENARIO_PRESETS,
            scenario_keywords=self.SCENARIO_KEYWORDS,
            default_sort_field="Value_Score",
            display_fields=["Brand", "Model", "Price", "CPU", "GPU", "RAM_GB", "Screen_Size_in", "Weight_kg"]
        )

    def get_specific_dimensions(self):
        return self.get_category_config().scoring_dimensions

    def get_model_class(self):
        return Laptop

    def get_product_info_text(self, p: Laptop) -> str:
        return f"型号:{p.Brand} {p.Model}, 价格:{p.Price}, CPU:{p.CPU}, 显卡:{p.GPU}, 内存:{p.RAM_GB}G, 屏幕:{p.Screen_Size_in}寸"

    def handle_chat(self, user_msg, history=None):
        return self.handle_chat_generic(user_msg, history, Laptop, self.SCENARIO_KEYWORDS)


class HeadphoneAgent(BaseDbAgent):
    """耳机推荐代理"""
    
    SCENARIO_PRESETS = {
        "commute": ["Sony", "Bose", "AirPods Pro", "FreeBuds Pro"],
        "hifi": ["Sennheiser", "MDR", "Beyerdynamic", "Hifiman"],
        "sports": ["Beats", "Shokz", "JBL", "Powerbeats"]
    }
    
    SCENARIO_KEYWORDS = {
        "commute": ["通勤", "地铁", "降噪", "飞机", "安静"],
        "hifi": ["音质", "发烧", "高保真", "无损", "听歌"],
        "sports": ["运动", "跑步", "健身", "防汗", "牢固"]
    }

    def get_category_config(self):
        return CategoryConfig(
            name="耳机",
            name_en="headphone",
            table_name="headphones",
            scoring_dimensions=[
                ScoringDimension("音质", "Sound_Score", 0.4, "声音表现"),
                ScoringDimension("舒适度", "Comfort_Score", 0.3, "佩戴体验"),
                ScoringDimension("降噪", "ANC_Score", 0.3, "降噪能力")
            ],
            scenario_presets=self.SCENARIO_PRESETS,
            scenario_keywords=self.SCENARIO_KEYWORDS,
            default_sort_field="Sound_Score",
            display_fields=["Brand", "Model", "Price", "Type", "Wireless", "ANC", "Battery_Hours"]
        )

    def get_specific_dimensions(self):
        return self.get_category_config().scoring_dimensions

    def get_model_class(self):
        return Headphone

    def get_product_info_text(self, p: Headphone) -> str:
        return f"型号:{p.Brand} {p.Model}, 价格:{p.Price}, 类型:{p.Type}, 降噪:{'是' if p.ANC else '否'}, 续航:{p.Battery_Hours}h"

    def handle_chat(self, user_msg, history=None):
        return self.handle_chat_generic(user_msg, history, Headphone, self.SCENARIO_KEYWORDS)


class TabletAgent(BaseDbAgent):
    """平板推荐代理"""
    
    SCENARIO_PRESETS = {
        "video": ["iPad", "MatePad", "Galaxy Tab", "Xiaomi Pad"],
        "drawing": ["iPad Pro", "iPad Air", "Galaxy Tab S", "MatePad Pro"],
        "student": ["iPad 10", "Xiaomi Pad", "Honor Pad", "Redmi Pad"]
    }
    
    SCENARIO_KEYWORDS = {
        "video": ["视频", "电影", "追剧", "游戏", "娱乐"],
        "drawing": ["画画", "设计", "手写笔", "笔记", "绘画"],
        "student": ["学习", "网课", "考研", "阅读", "便宜"]
    }

    def get_category_config(self):
        return CategoryConfig(
            name="平板电脑",
            name_en="tablet",
            table_name="tablets",
            scoring_dimensions=[
                ScoringDimension("性能", "Performance_Score", 0.3, "处理器性能"),
                ScoringDimension("屏幕", "Display_Score", 0.4, "显示效果"),
                ScoringDimension("生产力", "Productivity_Score", 0.3, "办公绘图能力")
            ],
            scenario_presets=self.SCENARIO_PRESETS,
            scenario_keywords=self.SCENARIO_KEYWORDS,
            default_sort_field="Value_Score",
            display_fields=["Brand", "Model", "Price", "Screen_Size_in", "Processor", "Stylus_Support"]
        )

    def get_specific_dimensions(self):
        return self.get_category_config().scoring_dimensions

    def get_model_class(self):
        return Tablet

    def get_product_info_text(self, p: Tablet) -> str:
        return f"型号:{p.Brand} {p.Model}, 价格:{p.Price}, 屏幕:{p.Screen_Size_in}寸, 处理器:{p.Processor}, 手写笔支持:{'是' if p.Stylus_Support else '否'}"

    def handle_chat(self, user_msg, history=None):
        return self.handle_chat_generic(user_msg, history, Tablet, self.SCENARIO_KEYWORDS)
