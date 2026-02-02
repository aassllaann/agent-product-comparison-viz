import json
from typing import List, Dict, Any, Optional
from sqlalchemy import desc
from models import SessionLocal, Phone, Laptop, Headphone, Tablet, Camera, Smartwatch, BluetoothSpeaker, Monitor, GamingConsole, GPU
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
        5. summary (摘要): 解析出的核心诉求
        6. product_type (类型): 显式要求的**目标产品**特定类型（如"头戴式"、"游戏本"等）
        7. brand (品牌): 用户明确指出**想要购买**的品牌名（如"我要英伟达显卡" -> "NVIDIA"）。注意：用户提到的**已拥有**设备的品牌（如"我刚买了PS5"）**不应**填入此字段。
        8. owned_items (已购设备): 用户提到已经拥有的产品品类或型号（如"PS5", "iPhone"）

        【推理规则】：
        - 预算缺省默认为 20000。
        - 如果用户提到"不差钱"、"旗舰"、"顶配"，max_price 设为 999999。
        - **关键**: brand 字段只填用户**想买**的品牌。如果用户说"给我的 Sony 电视配个音箱"，想买的是音箱，brand 字段应为空（除非用户明确说要 Sony 音箱）。

        输出严格 JSON：{{"max_price": 数字, "sort_field": "字段名", "summary": "核心诉求", "usage": "场景关键词", "product_type": "...", "brand": "品牌", "owned_items": []}}
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
            
            # 品牌过滤
            brand_filter = intent.get('brand')
            if brand_filter and brand_filter.lower() not in ["", "null", "none"]:
                # 使用关键词包含关系过滤
                if brand_filter.lower() not in item.Brand.lower() and item.Brand.lower() not in brand_filter.lower():
                    print(f"[Debug] Filtering out {item.Brand} because brand filter is {brand_filter}")
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
        if exclude_ids == None:
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

        # 品牌过滤
        brand_filter = intent.get('brand')
        if brand_filter and brand_filter.lower() not in ["", "null", "none"]:
            query = query.filter(model_class.Brand.ilike(f"%{brand_filter}%"))

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

    def _generate_visualization(self, results, intent):
        """
        统一生成可视化图表和分析
        """
        config = self.get_category_config()
        
        # 1. 生成图表
        radar_dims = [(d.name, d.field, 100) for d in config.scoring_dimensions]
        compare_dims = [(d.name, d.field) for d in config.scoring_dimensions[:3]]
        
        charts = (
            visualizer.draw_radar(results, radar_dims),
            visualizer.draw_comparison(results, intent.get('sort_field', config.default_sort_field)),
            visualizer.draw_multi_dimension_compare(results, compare_dims)
        )
        
        # 2. 生成分析文案
        analyses = []
        if not results:
            return charts, analyses
            
        # --- 雷达图分析 ---
        top_item = results[0]
        radar_dims_conf = config.scoring_dimensions
        scores_desc = []
        high_scores = []
        for dim in radar_dims_conf:
            score = getattr(top_item, dim.field, 0) or 0
            scores_desc.append(f"{dim.name}评分 {score}")
            if score >= 80: high_scores.append(dim.name)
        
        radar_detail = f"首选推荐 {top_item.Brand} {top_item.Model} 指标：{', '.join(scores_desc)}。在该项表现越优异，整体素质越高。"
        analyses.append(radar_detail)
        
        # --- 核心指标对比 ---
        sort_field = intent.get('sort_field', config.default_sort_field)
        sort_name = next((d.name for d in config.scoring_dimensions if d.field == sort_field), sort_field)
        top3_scores = [f"{c.Model}: {getattr(c, sort_field, 0) or 0}" for c in results]
        compare_detail = f"关键指标【{sort_name}】对比：{'；'.join(top3_scores)}。"
        analyses.append(compare_detail)
        
        # --- 多维优势分析 ---
        highlights = []
        for dim in config.scoring_dimensions[:3]:
            best = max(results, key=lambda c: getattr(c, dim.field, 0) or 0)
            highlights.append(f"{best.Model} 在{dim.name}表现最佳")
        multi_dim_detail = f"综合分析：{'；'.join(highlights)}。您可以根据具体需求权衡选择。"
        analyses.append(multi_dim_detail)
        
        return charts, analyses

    def handle_chat_generic(self, user_msg, history, model_class, keyword_map):
        """通用的对话处理流程"""
        config = self.get_category_config()
        fields_desc = ", ".join([f"{d.field}({d.name})" for d in config.scoring_dimensions])
        
        # 1. 解析意图
        intent = self._parse_intent_generic(user_msg, config.name, fields_desc, history)
        
        # 补丁：处理极端性能词汇
        extreme_words = ["最强", "最好", "最高", "不差钱", "旗舰"]
        if any(w in user_msg for w in extreme_words):
            intent['max_price'] = 999999
            # 自动映射到该品类最核心的性能维度
            perf_map = {
                "GPU": "Creative_Score", # 显卡优先保证跑分/创作
                "Phone": "Performance_Score",
                "Laptop": "Performance_Score",
                "Monitor": "Display_Score",
                "GamingConsole": "Performance_Score"
            }
            if config.name_en.capitalize() in perf_map:
                intent['sort_field'] = perf_map[config.name_en.capitalize()]
        
        self.last_intent = intent
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
        
        charts, analyses = self._generate_visualization(results, intent)
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
        "commute": ["通勤", "地铁", "降噪", "飞机", "安静", "出差"],
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
        result = self.handle_chat_generic(user_msg, history, Headphone, self.SCENARIO_KEYWORDS)
        
        # 强力 Fallback: 确保永远有推荐
        reasons, charts, results, analyses = result
        if not results:
            print("[HeadphoneAgent] No results found, using HARD FALLBACK.")
            # 强制推荐 Top 3 (按价格倒序)
            results = self.db.query(Headphone).order_by(desc(Headphone.Price)).limit(3).all()
            if results:
                reasons = ["抱歉，根据具体要求暂未找到匹配，但为您精选了以下几款热门的高端耳机供您参考："]
                charts, analyses = self._generate_visualization(results, {"sort_field": "Sound_Score"})
                return reasons, charts, results, analyses
                
        return result


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


class SmartwatchAgent(BaseDbAgent):
    """智能手表推荐代理"""
    
    SCENARIO_PRESETS = {
        "fitness": ["Garmin", "COROS", "Polar", "Suunto", "Amazfit"],
        "daily": ["Apple Watch", "Galaxy Watch", "Huawei", "小米", "OPPO"],
        "professional": ["Garmin Fenix", "Apple Watch Ultra", "Suunto", "COROS"]
    }
    
    SCENARIO_KEYWORDS = {
        "fitness": ["运动", "跑步", "健身", "马拉松", "训练", "GPS"],
        "daily": ["日常", "通勤", "通知", "支付", "便捷"],
        "professional": ["专业", "户外", "登山", "潜水", "极限"]
    }

    def get_category_config(self):
        return CategoryConfig(
            name="智能手表",
            name_en="smartwatch",
            table_name="smartwatches",
            scoring_dimensions=[
                ScoringDimension("续航", "Battery_Score", 0.25, "电池续航能力"),
                ScoringDimension("健康功能", "Health_Score", 0.25, "健康监测功能"),
                ScoringDimension("智能体验", "Smart_Score", 0.25, "智能功能体验"),
                ScoringDimension("性价比", "Value_Score", 0.25, "综合性价比")
            ],
            scenario_presets=self.SCENARIO_PRESETS,
            scenario_keywords=self.SCENARIO_KEYWORDS,
            default_sort_field="Value_Score",
            display_fields=["Brand", "Model", "Price", "Battery_Days", "OS", "Waterproof_Rating"]
        )

    def get_specific_dimensions(self):
        return self.get_category_config().scoring_dimensions

    def get_model_class(self):
        return Smartwatch

    def get_product_info_text(self, p: Smartwatch) -> str:
        return f"型号:{p.Brand} {p.Model}, 价格:{p.Price}, 续航:{p.Battery_Days}天, 系统:{p.OS}, 防水:{p.Waterproof_Rating}"

    def handle_chat(self, user_msg, history=None):
        return self.handle_chat_generic(user_msg, history, Smartwatch, self.SCENARIO_KEYWORDS)


class BluetoothSpeakerAgent(BaseDbAgent):
    """蓝牙音箱推荐代理"""
    
    SCENARIO_PRESETS = {
        "portable": ["JBL Flip", "JBL Go", "Sony XB", "Bose SoundLink Flex", "小米"],
        "home": ["Bose", "Marshall", "Harman Kardon", "B&O", "Sony"],
        "outdoor": ["JBL Xtreme", "JBL Charge", "Sony XG", "UE BOOM", "Anker"]
    }
    
    SCENARIO_KEYWORDS = {
        "portable": ["便携", "户外", "旅行", "轻便", "随身"],
        "home": ["家用", "桌面", "音质", "发烧", "HiFi"],
        "outdoor": ["户外", "防水", "派对", "大音量", "露营"]
    }

    def get_category_config(self):
        return CategoryConfig(
            name="蓝牙音箱",
            name_en="bluetooth_speaker",
            table_name="bluetooth_speakers",
            scoring_dimensions=[
                ScoringDimension("音质", "Sound_Score", 0.35, "声音表现"),
                ScoringDimension("续航", "Battery_Score", 0.25, "电池续航"),
                ScoringDimension("便携性", "Portability_Score", 0.2, "便携程度"),
                ScoringDimension("性价比", "Value_Score", 0.2, "综合性价比")
            ],
            scenario_presets=self.SCENARIO_PRESETS,
            scenario_keywords=self.SCENARIO_KEYWORDS,
            default_sort_field="Sound_Score",
            display_fields=["Brand", "Model", "Price", "Power_W", "Battery_Hours", "Waterproof_Rating"]
        )

    def get_specific_dimensions(self):
        return self.get_category_config().scoring_dimensions

    def get_model_class(self):
        return BluetoothSpeaker

    def get_product_info_text(self, p: BluetoothSpeaker) -> str:
        return f"型号:{p.Brand} {p.Model}, 价格:{p.Price}, 功率:{p.Power_W}W, 续航:{p.Battery_Hours}h, 防水:{p.Waterproof_Rating}"

    def handle_chat(self, user_msg, history=None):
        return self.handle_chat_generic(user_msg, history, BluetoothSpeaker, self.SCENARIO_KEYWORDS)


class MonitorAgent(BaseDbAgent):
    """显示器推荐代理"""
    
    SCENARIO_PRESETS = {
        "gaming": ["ASUS ROG", "LG", "Samsung Odyssey", "MSI", "Gigabyte"],
        "office": ["Dell U", "LG", "BenQ", "ASUS ProArt", "ViewSonic"],
        "creative": ["Dell U", "BenQ SW", "ASUS ProArt", "LG", "Apple Studio"]
    }
    
    SCENARIO_KEYWORDS = {
        "gaming": ["游戏", "电竞", "144hz", "240hz", "高刷"],
        "office": ["办公", "护眼", "商务", "文档", "编程"],
        "creative": ["设计", "修图", "调色", "摄影", "视频", "剪辑"]
    }

    def get_category_config(self):
        return CategoryConfig(
            name="显示器",
            name_en="monitor",
            table_name="monitors",
            scoring_dimensions=[
                ScoringDimension("画质", "Display_Score", 0.35, "显示效果"),
                ScoringDimension("性能", "Performance_Score", 0.25, "刷新率响应时间"),
                ScoringDimension("人体工学", "Ergonomics_Score", 0.2, "支架调节护眼"),
                ScoringDimension("性价比", "Value_Score", 0.2, "综合性价比")
            ],
            scenario_presets=self.SCENARIO_PRESETS,
            scenario_keywords=self.SCENARIO_KEYWORDS,
            default_sort_field="Display_Score",
            display_fields=["Brand", "Model", "Price", "Screen_Size_in", "Resolution", "Refresh_Rate_Hz", "Panel_Type"]
        )

    def get_specific_dimensions(self):
        return self.get_category_config().scoring_dimensions

    def get_model_class(self):
        return Monitor

    def get_product_info_text(self, p: Monitor) -> str:
        return f"型号:{p.Brand} {p.Model}, 价格:{p.Price}, 尺寸:{p.Screen_Size_in}寸, 分辨率:{p.Resolution}, 刷新率:{p.Refresh_Rate_Hz}Hz"

    def handle_chat(self, user_msg, history=None):
        return self.handle_chat_generic(user_msg, history, Monitor, self.SCENARIO_KEYWORDS)


class GamingConsoleAgent(BaseDbAgent):
    """游戏主机推荐代理"""
    
    SCENARIO_PRESETS = {
        "console": ["PlayStation", "Xbox", "Nintendo Switch"],
        "handheld": ["Steam Deck", "ROG Ally", "Legion Go", "Switch"],
        "retro": ["Anbernic", "Retroid", "Miyoo"]
    }
    
    SCENARIO_KEYWORDS = {
        "console": ["主机", "电视", "客厅", "3A", "大作"],
        "handheld": ["掌机", "便携", "移动", "steam", "pc游戏"],
        "retro": ["复古", "模拟器", "怀旧", "经典"]
    }

    def get_category_config(self):
        return CategoryConfig(
            name="游戏主机",
            name_en="gaming_console",
            table_name="gaming_consoles",
            scoring_dimensions=[
                ScoringDimension("性能", "Performance_Score", 0.35, "游戏性能"),
                ScoringDimension("游戏生态", "Ecosystem_Score", 0.3, "游戏库与独占"),
                ScoringDimension("多媒体", "Media_Score", 0.15, "影音娱乐功能"),
                ScoringDimension("性价比", "Value_Score", 0.2, "综合性价比")
            ],
            scenario_presets=self.SCENARIO_PRESETS,
            scenario_keywords=self.SCENARIO_KEYWORDS,
            default_sort_field="Performance_Score",
            display_fields=["Brand", "Model", "Price", "Storage_GB", "Max_Resolution", "Exclusive_Games_Count"]
        )

    def get_specific_dimensions(self):
        return self.get_category_config().scoring_dimensions

    def get_model_class(self):
        return GamingConsole

    def get_product_info_text(self, p: GamingConsole) -> str:
        return f"型号:{p.Brand} {p.Model}, 价格:{p.Price}, 存储:{p.Storage_GB}GB, 分辨率:{p.Max_Resolution}, 独占游戏:{p.Exclusive_Games_Count}+"

    def handle_chat(self, user_msg, history=None):
        return self.handle_chat_generic(user_msg, history, GamingConsole, self.SCENARIO_KEYWORDS)


class GPUAgent(BaseDbAgent):
    """显卡推荐代理"""
    
    SCENARIO_PRESETS = {
        "gaming_4k": ["RTX 4090", "RTX 4080", "RX 7900 XTX", "RX 7900 XT"],
        "gaming_2k": ["RTX 4070", "RTX 4060 Ti", "RX 7800 XT", "RX 7700 XT"],
        "gaming_1080p": ["RTX 4060", "RX 7600", "Arc"],
        "creator": ["RTX 4090", "RTX 4080", "RTX 4070 Ti"]
    }
    
    SCENARIO_KEYWORDS = {
        "gaming_4k": ["4k", "2160p", "高画质", "光追"],
        "gaming_2k": ["2k", "1440p", "电竞", "高刷"],
        "gaming_1080p": ["1080p", "入门", "性价比", "预算"],
        "creator": ["渲染", "建模", "剪辑", "AI", "创作"]
    }

    def get_category_config(self):
        return CategoryConfig(
            name="显卡",
            name_en="gpu",
            table_name="gpus",
            scoring_dimensions=[
                ScoringDimension("游戏性能", "Gaming_Score", 0.35, "游戏帧率表现"),
                ScoringDimension("创作性能", "Creative_Score", 0.25, "渲染与AI加速"),
                ScoringDimension("功耗散热", "Thermal_Score", 0.2, "温度与噪音"),
                ScoringDimension("性价比", "Value_Score", 0.2, "综合性价比")
            ],
            scenario_presets=self.SCENARIO_PRESETS,
            scenario_keywords=self.SCENARIO_KEYWORDS,
            default_sort_field="Gaming_Score",
            display_fields=["Brand", "Model", "Price", "VRAM_GB", "Chip", "TDP_W"]
        )

    def get_specific_dimensions(self):
        return self.get_category_config().scoring_dimensions

    def get_model_class(self):
        return GPU

    def get_product_info_text(self, p: GPU) -> str:
        return f"型号:{p.Brand} {p.Model}, 价格:{p.Price}, 显存:{p.VRAM_GB}GB, 芯片:{p.Chip}, 功耗:{p.TDP_W}W"

    def handle_chat(self, user_msg, history=None):
        return self.handle_chat_generic(user_msg, history, GPU, self.SCENARIO_KEYWORDS)

