import json
from typing import List
from sqlalchemy import desc
from models import SessionLocal, Camera
import visualizer
from openai import OpenAI
import config
from base_agent import BaseProductAgent, CategoryConfig, ScoringDimension


class CameraAgent(BaseProductAgent):
    """
    相机推荐代理
    
    继承自 BaseProductAgent，实现相机品类特有的评分体系和推荐逻辑。
    """
    
    # 场景-推荐机型集合 (Golden Sets)
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

    def __init__(self):
        super().__init__()
        self.db = SessionLocal()
    
    # ==================== 实现 BaseProductAgent 抽象方法 ====================
    
    def get_category_config(self) -> CategoryConfig:
        """返回相机品类配置"""
        return CategoryConfig(
            name="相机",
            name_en="camera",
            table_name="cameras",
            scoring_dimensions=self.get_specific_dimensions(),
            scenario_presets=self.SCENARIO_PRESETS,
            scenario_keywords=self.SCENARIO_KEYWORDS,
            default_sort_field="LowLight_Score",
            display_fields=["Brand", "Model", "Price", "Weight_g", "Max_ISO", "Supports_4K"]
        )
    
    def get_specific_dimensions(self) -> List[ScoringDimension]:
        """返回相机品类特有评分维度"""
        return [
            ScoringDimension(
                name="便携性",
                field="Portability_Score",
                weight=0.20,
                description="机身重量和体积的综合考量"
            ),
            ScoringDimension(
                name="低光画质",
                field="LowLight_Score",
                weight=0.30,
                description="弱光环境下的成像能力"
            ),
            ScoringDimension(
                name="视频能力",
                field="Video_Score",
                weight=0.25,
                description="视频拍摄性能，包括4K支持等"
            ),
        ]
    
    def get_db_session(self):
        """返回数据库会话"""
        return self.db
    
    def get_model_class(self):
        """返回 Camera 模型类"""
        return Camera
    
    def get_product_info_text(self, cam) -> str:
        """格式化相机信息为文本"""
        return f"""品牌型号: {cam.Brand} {cam.Model}
价格: {cam.Price}元
重量: {cam.Weight_g}g
低光画质评分: {cam.LowLight_Score}
视频能力评分: {cam.Video_Score}
便携性评分: {cam.Portability_Score}
最大ISO: {cam.Max_ISO}
支持4K: {'是' if cam.Supports_4K else '否'}
传感器: {getattr(cam, 'Sensor_type', '未知')}
上市年份: {getattr(cam, 'Year', '未知')}"""
    
    # ==================== 相机特有方法 ====================


    def _parse_intent(self, user_msg, history=None):
        """
        结构化信息提取（全部可缺省）。
        从对话中提取：场景、功能、重量耐受度、经济投入、特殊要求。
        """
        system_rules = """
        你是一个相机导购专家，负责从对话中提取 5 个维度的结构化信息。

        【待提取维度（用户均可缺省）】：
        1. usage (用途): 旅行/校园/体育/风景/人像/天文/家庭/宠物/美食/产品/街拍/Vlog
        2. features (功能): 闪光灯/视频/蓝牙/触屏/GPS/防水/连拍/自拍/无线传输
        3. weight_pref (重量): 3kg没问题/避免负重/必须轻便/单手/口袋/便携
        4. budget_level (投入): 高端(一步到位)/旗舰/普通(实用)/经济(入门)/极致性价比
        5. special (特殊): 新型号/材质好/品牌/外观/续航/操控/二手/老机型

        【推理与缺省规则】：
        - 预算缺省：没提预算则默认为"普通"(15000元)。
        - 预算关键词映射：高端/旗舰/一步到位/不差钱→30000，普通/实用→10000，经济/入门/性价比→5000。
        - 排序字段推理：
            * 强调画质/夜景/天文/风景/人像/高像素→LowLight_Score
            * 强调vlog时，排序字段为Video_Score和Portability_Score的加权（如0.7*Video_Score+0.3*Portability_Score），或优先筛选轻便机型后再按Video_Score排序
            * 强调视频/拍片/4K/直播→Video_Score
            * 强调旅行/轻便/便携/单手/口袋→Portability_Score
            * 强调体育/抓拍/连拍→Video_Score 或 Portability_Score（优先Video_Score）
            * 强调操控/专业/旗舰→LowLight_Score 或 Video_Score（优先LowLight_Score）
        - 若无明显偏好，默认排序字段为LowLight_Score。
        - 价格映射：高端:30000, 普通:10000, 经济:5000。

        输出严格 JSON：{"max_price": 数字, "sort_field": "字段名", "summary": "解析出的核心诉求", "usage": "场景关键词"}
        """
        try:
            messages = [{"role": "system", "content": system_rules}]
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": user_msg})
            response = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            # 清洗 markdown 代码块标记
            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "")
            
            return json.loads(content)
        except Exception:
            # 万能缺省值
            return {"max_price": 15000, "sort_field": "LowLight_Score", "summary": "综合选购", "usage": ""}

    def _get_preset_cameras(self, scenario):
        """根据场景获取预设机型对象"""
        preset_names = self.SCENARIO_PRESETS.get(scenario, [])
        if not preset_names:
            return []
        
        candidates = []
        seen_ids = set()
        for name in preset_names:
            # 模糊匹配 Model 或 Brand
            cams = self.db.query(Camera).filter(
                (Camera.Model.ilike(f"%{name}%") | Camera.Brand.ilike(f"%{name}%"))
            ).all()
            for cam in cams:
                if cam.id not in seen_ids:
                    candidates.append(cam)
                    seen_ids.add(cam.id)
        return candidates

    def _filter_candidates(self, candidates, intent):
        """
        对候选集进行硬/软筛选：
        1. 价格宽松过滤 (严格遵守 max_price)
        2. 重量过滤 (如果提及便携)
        3. 排序
        """
        filtered = []
        max_price = intent.get('max_price', 15000)
        
        # 严格遵守预算，不再使用 1.1 的宽容度
        price_limit = max_price

        for cam in candidates:
            # 价格过滤
            if cam.Price and cam.Price > price_limit:
                continue
            
            # TODO: Add Weight filtering if needed
            # if 'light' in intent.get('weight_pref', '') and cam.Weight_g > 800: continue
            
            filtered.append(cam)
        
        # 排序
        sort_field = intent.get('sort_field', 'LowLight_Score')
        # 确保字段存在，否则默认
        if not hasattr(Camera, sort_field):
            sort_field = 'LowLight_Score'
            
        filtered.sort(key=lambda x: getattr(x, sort_field, 0) or 0, reverse=True)
        return filtered

    def _fallback_search(self, intent, exclude_ids=None):
        """全库搜索兜底"""
        if exclude_ids is None:
            exclude_ids = []
            
        query = self.db.query(Camera).filter(
            Camera.Price <= intent.get('max_price', 15000)
        )
        if exclude_ids:
            query = query.filter(~Camera.id.in_(exclude_ids))
            
        sort_field = intent.get('sort_field', 'LowLight_Score')
        if hasattr(Camera, sort_field):
            query = query.order_by(desc(getattr(Camera, sort_field)))
        else:
            query = query.order_by(desc(Camera.LowLight_Score))
            
        return query.limit(3).all()

    def _get_expert_replies(self, results, user_msg, history=None):
        """生成专家点评"""
        if not results:
            return "没有找到匹配的机型。"
            
        cams_info = "\n".join([f"{i+1}. {c.Brand} {c.Model}: 价格约{c.Price}元, 评分: 低光{c.LowLight_Score}/视频{c.Video_Score}/便携{c.Portability_Score}" for i, c in enumerate(results)])
        
        system_prompt = f"""
        你是一个专业相机的导购专家。根据用户需求和数据库搜索结果，生成一段简短、专业的推荐语。
        
        用户需求: {user_msg}
        搜索结果:
        {cams_info}
        
        要求：
        1. 必须基于搜索结果进行推荐，不要推荐列表也就是搜索结果之外的相机。
        2. 结合用户需求，解释为什么推荐这几款（例如：强调视频能力、性价比或画质）。
        3. 语言亲切自然，条理清晰。
        4. 如果有新旧机型对比，可以简单提及（如：推荐了新款Action 4，相比旧款...）。
        """
        
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": "请给出推荐建议"})
        
        try:
            response = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM Error: {e}")
            return f"根据您的需求，为您精选了以下机型：\n{cams_info}\n它们在您关注的领域表现都很不错！"

    def _get_individual_reasons(self, results, user_msg, intent, history=None):
        """为每台相机生成独立的推荐理由"""
        if not results:
            return []
        
        reasons = []
        for idx, cam in enumerate(results):
            system_prompt = f"""
            你是一个专业相机导购专家。现在需要为用户推荐的第{idx+1}款相机生成推荐理由。
            
            用户需求: {user_msg}
            核心诉求: {intent.get('summary', '综合选购')}
            预算: {intent.get('max_price', 15000)}元
            关注指标: {intent.get('sort_field', 'LowLight_Score')}
            
            推荐机型信息:
            品牌型号: {cam.Brand} {cam.Model}
            价格: {cam.Price}元
            重量: {cam.Weight_g}g
            低光画质评分: {cam.LowLight_Score}
            视频能力评分: {cam.Video_Score}
            便携性评分: {cam.Portability_Score}
            最大ISO: {cam.Max_ISO}
            支持4K: {'是' if cam.Supports_4K else '否'}
            传感器: {getattr(cam, 'Sensor_type', '未知')}
            上市年份: {getattr(cam, 'Year', '未知')}
            
            要求：
            1. 生成2-3句话的简短推荐理由（80-150字）
            2. 紧密结合用户需求，说明为什么这款相机适合用户
            3. 突出该机型的核心优势（如性价比、画质、便携性、视频能力等）
            4. 语言亲切自然，避免空洞的形容词
            5. 如果是第{idx+1}款（非第一款），可以简单对比说明与前面机型的差异
            6. 直接输出推荐理由文本，不要包含"推荐理由:"等标题
            """
            
            messages = [{"role": "system", "content": system_prompt}]
            if history:
                messages.extend(history[-2:])  # 只保留最近2轮对话
            messages.append({"role": "user", "content": f"请为{cam.Brand} {cam.Model}生成推荐理由"})
            
            try:
                response = self.client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=messages,
                    stream=False
                )
                reason = response.choices[0].message.content.strip()
                reasons.append(reason)
            except Exception as e:
                print(f"LLM Error for camera {idx}: {e}")
                # 生成备用推荐理由
                fallback = f"这款{cam.Brand} {cam.Model}售价{cam.Price}元，在{intent.get('sort_field', 'LowLight_Score').replace('_Score', '')}方面表现出色，非常适合您的使用场景。"
                reasons.append(fallback)
        
        return reasons

    def handle_chat(self, user_msg, history=None):
        # 1. 识别并补全意图
        intent = self._parse_intent(user_msg, history)
        results = []
        
        # 2. 决策树逻辑：匹配场景 -> 获取预设 -> 筛选
        usage = intent.get('usage', '').lower()
        summary = intent.get('summary', '').lower()
        
        target_scenario = None
        
        # 优化的关键词匹配逻辑
        for scenario, keywords in self.SCENARIO_KEYWORDS.items():
            # 检查 user inputs 中的关键词
            if any(k in usage for k in keywords) or any(k in summary for k in keywords):
                target_scenario = scenario
                break
        
        # 回退到简单的预设名称匹配（作为保险）
        if not target_scenario:
            for scenario in self.SCENARIO_PRESETS.keys():
                if scenario in usage or scenario in summary:
                    target_scenario = scenario
                    break
        
        # 额外关键词映射
        if not target_scenario:
            if any(w in summary for w in ['视频', '拍片', '直播', 'up主']):
                target_scenario = 'vlog'
            elif any(w in summary for w in ['扫街', '人文']):
                target_scenario = 'street'
            elif any(w in summary for w in ['新手', '入门', '小白']):
                target_scenario = 'beginner'

        if target_scenario:
            # 获取预设
            candidates = self._get_preset_cameras(target_scenario)
            # 筛选 (价格、性能)
            results = self._filter_candidates(candidates, intent)
            # 截取前3
            results = results[:3]

        # 3. 兜底搜索 (如果预设不够3个)
        if len(results) < 3:
            exclude_ids = [c.id for c in results]
            more_cams = self._fallback_search(intent, exclude_ids)
            results.extend(more_cams)
            results = results[:3]

        if not results:
            return "我根据您的描述搜寻了数据库，但目前暂时没有完全匹配的机型。您可以试着稍微提高一点预算，或者减少一些功能要求？", None, None, None

        # 3. 为每台机型生成独立的推荐理由
        replies = self._get_individual_reasons(results, user_msg, intent, history)

        # 4. 绘图
        charts = (
            visualizer.draw_radar(results),
            visualizer.draw_comparison(results, intent['sort_field']),
            visualizer.draw_multi_dimension_compare(results)
        )

        # 5. 生成每个图表的详细分析（基于实际数据）
        chart_analyses = []
        # 1. 推荐机型画像详细分析（主推）
        cam = results[0]
        radar_detail = (
            f"{cam.Brand} {cam.Model} 的便携性评分为 {cam.Portability_Score}，低光画质评分为 {cam.LowLight_Score}，视频能力评分为 {cam.Video_Score}，最大ISO为 {cam.Max_ISO}。"
            f" 这意味着该机型在{'便携性、' if cam.Portability_Score >= 80 else ''}{'低光画质、' if cam.LowLight_Score >= 80 else ''}{'视频能力、' if cam.Video_Score >= 80 else ''}等方面表现{'突出' if any([cam.Portability_Score >= 80, cam.LowLight_Score >= 80, cam.Video_Score >= 80]) else '均衡'}。"
        )
        chart_analyses.append(radar_detail)

        # 2. Top3 性能对比详细分析
        sort_field = intent['sort_field']
        top3_scores = [(c.Brand, c.Model, getattr(c, sort_field)) for c in results]
        top3_scores_str = '，'.join([f"{b}{m}:{s}" for b, m, s in top3_scores])
        best = max(top3_scores, key=lambda x: x[2])
        compare_detail = (
            f"本次推荐的前三款机型在 {sort_field} 指标上的得分分别为：{top3_scores_str}。其中 {best[0]} {best[1]} 表现最佳。"
        )
        chart_analyses.append(compare_detail)

        # 3. 核心能力多维对比详细分析
        dims_map = {
            'Portability_Score': '便携性', 
            'LowLight_Score': '低光画质', 
            'Video_Score': '视频能力'
        }
        highlights = []
        for field, name in dims_map.items():
            best_cam = max(results, key=lambda c: getattr(c, field, 0))
            score = getattr(best_cam, field, 0)
            highlights.append(f"{best_cam.Model} 在{name}方面得分最高({score})")
        
        multi_dim_detail = (
            f"综合多维能力分析：{'；'.join(highlights)}。"
            "您可以根据自己最看重的维度（如追求轻便还是极致画质）进行最终选择。"
        )
        chart_analyses.append(multi_dim_detail)

        return replies, charts, results, chart_analyses

    def __del__(self):
        if hasattr(self, 'db'):
            self.db.close()