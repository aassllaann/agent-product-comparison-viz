import json
from sqlalchemy import desc
from models import SessionLocal, Camera
import visualizer
from openai import OpenAI
import config

class CameraAgent:
     # 场景-推荐机型集合
    scene_presets = {
        "vlog": [
            "ZV-E10",  "PowerShot G7 X Mark III","X-T100"],
         # 可扩展更多场景
     }
    def __init__(self):
        self.db = SessionLocal()
        self.client = OpenAI(
            api_key=config.DASHSCOPE_API_KEY, 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def _parse_intent(self, user_msg, history=None):
        """
        结构化信息提取（全部可缺省）。
        从对话中提取：场景、功能、重量耐受度、经济投入、特殊要求。
        """
        system_rules = """
        你是一个相机导购专家，负责从对话中提取 5 个维度的结构化信息。

        【待提取维度（用户均可缺省）】：
        1. usage (用途): 旅行/校园/体育/风景/人像/天文/家庭/宠物/美食/产品
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

        输出严格 JSON：{"max_price": 数字, "sort_field": "字段名", "summary": "解析出的核心诉求"}
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
            return json.loads(response.choices[0].message.content)
        except:
            # 万能缺省值
            return {"max_price": 15000, "sort_field": "LowLight_Score", "summary": "综合选购"}

    def _get_expert_reply(self, camera, user_msg, history=None):
        """生成人性化的专家回复，支持上下文，且不使用比喻"""
        prompt = (
            f"用户想买相机，说了：'{user_msg}'。你为他推荐了 '{camera.Brand} {camera.Model}'。"
            "请用专业但亲切的口吻进行点评，说明为什么这台相机适合他。"
            "不要使用比喻、不要用‘像xxx一样’等修辞，只给出直接、客观、简明的理由。"
        )
        messages = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages
        )
        return response.choices[0].message.content

    def _get_expert_replies(self, cameras, user_msg, history=None):
        """为多个相机生成推荐理由列表"""
        replies = []
        for cam in cameras:
            replies.append(self._get_expert_reply(cam, user_msg, history))
        return replies

    def handle_chat(self, user_msg, history=None):
        # 1. 识别并补全意图（带历史）
        intent = self._parse_intent(user_msg, history)

        # 2. 场景优先推荐（如vlog）
        preset_results = []
        usage = intent.get('usage', '')
        summary = intent.get('summary', '').lower()
        # 判断是否为vlog场景
        if any(x in summary for x in ['vlog', '视频', '拍片', '直播']) or (isinstance(usage, str) and 'vlog' in usage.lower()):
            preset_names = self.scene_presets.get('vlog', [])
            # 先查找预设机型，满足预算，按集合原顺序返回，支持模糊匹配
            preset_results = []
            seen_ids = set()
            for name in preset_names:
                cam_obj = self.db.query(Camera).filter(
                    (Camera.Model.ilike(f"%{name}%") | Camera.Brand.ilike(f"%{name}%")),
                    Camera.Price <= intent['max_price']
                ).first()
                if cam_obj and cam_obj.id not in seen_ids:
                    preset_results.append(cam_obj)
                    seen_ids.add(cam_obj.id)
                if len(preset_results) == 3:
                    break

        # 3. 若预设机型不足，再补全
        results = list(preset_results)
        if len(results) < 3:
            # 补全剩余推荐
            exclude_ids = [c.id for c in results]
            extra_query = self.db.query(Camera)\
                .filter(Camera.Price <= intent['max_price'])\
                .filter(~Camera.id.in_(exclude_ids))\
                .order_by(desc(getattr(Camera, intent['sort_field'])))\
                .limit(3 - len(results))
            results += extra_query.all()

        if not results:
            return "我根据您的描述搜寻了数据库，但目前暂时没有完全匹配的机型。您可以试着稍微提高一点预算，或者减少一些功能要求？", None, None, None

        # 3. 结果解读（为每台机型生成推荐理由）
        replies = self._get_expert_replies(results, user_msg, history)

        # 4. 绘图（获取全库数据用于性价比分布图）
        all_cams = self.db.query(Camera).limit(100).all()
        charts = (
            visualizer.draw_radar(results[0]),
            visualizer.draw_comparison(results, intent['sort_field']),
            visualizer.draw_price_performance(results, all_cams)
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

        # 3. 性价比分布详细分析
        def price_perf(c):
            return (getattr(c, 'LowLight_Score', 0) + getattr(c, 'Video_Score', 0)) / 2
        all_scores = [(c.Price, price_perf(c)) for c in all_cams]
        rec_scores = [(c.Price, price_perf(c)) for c in results]
        avg_perf = sum([s for _, s in all_scores]) / len(all_scores) if all_scores else 0
        above_avg = sum(1 for _, s in rec_scores if s > avg_perf)
        price_perf_detail = (
            f"推荐机型的性价比分数分别为：{', '.join([f'{p}元/{s:.1f}' for p, s in rec_scores])}。"
            f" 其中 {above_avg} 款机型性价比高于全库平均水平（均值为 {avg_perf:.1f}）。"
        )
        chart_analyses.append(price_perf_detail)

        return replies, charts, results, chart_analyses

    def __del__(self):
        if hasattr(self, 'db'):
            self.db.close()