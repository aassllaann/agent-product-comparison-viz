import json
import re
from sqlalchemy import desc
from models import SessionLocal, Camera
import visualizer
from openai import OpenAI
import config

class CameraAgent:
    def __init__(self):
        self.db = SessionLocal()
        # 使用 OpenAI SDK 访问 DashScope (通义千问)
        self.client = OpenAI(
            api_key=config.DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def _get_llm_intent(self, structured_prompt):
        """
        第一步：高级意图解析。
        LLM 需要根据用户勾选的【场景、功能、重量偏好、预算倾向】来决定 SQL 参数。
        """
        system_msg = """
        你是一个相机导购决策专家。用户提供了一组结构化的需求，请将其转化为数据库查询逻辑。
        
        【逻辑规则】：
        1. "max_price" (预算):
           - 入门/经济：6000
           - 中端/好用：15000
           - 高端/一步到位：100000
        2. "sort_field" (核心权重):
           - 若涉及自然风景、天文、人像 -> "LowLight_Score" (高画质/动态范围)
           - 若涉及旅行、避负重 -> "Portability_Score" (轻量化优先)
           - 若涉及活动、体育、视频 -> "Video_Score" (对焦与帧率优先)
        3. "reason_summary": 综合用户勾选的所有标签，给出一个高度概括的推荐方向。

        请严格返回 JSON 格式：{"max_price": 数字, "sort_field": "字段名", "reason_summary": "字符串"}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="qwen-turbo",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": structured_prompt}
                ],
                response_format={"type": "json_object"}
            )
            intent = json.loads(response.choices[0].message.content)
            return intent
        except Exception as e:
            print(f"意图解析异常: {e}")
            return {"max_price": 10000, "sort_field": "LowLight_Score", "reason_summary": "基于综合平衡为您推荐"}

    def _generate_expert_advice(self, camera, structured_prompt):
        """
        第二步：深度点评生成。
        针对筛选出的 Top 1 相机，结合用户的具体勾选（如：天文、防水、轻便）进行点评。
        """
        system_msg = "你是一位毒舌但专业的器材测评人。请结合用户的多维需求，点评推荐的这款相机，说明它如何满足用户的特定功能需求。字数在100字左右。"
        
        camera_info = f"""
        备选型号：{camera.Brand} {camera.Model}
        核心得分：视频 {camera.Video_Score}, 便携 {camera.Portability_Score}, 画质 {camera.LowLight_Score}
        价格：{camera.Price}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="qwen-turbo",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": f"用户需求：{structured_prompt}\n相机参数：{camera_info}"}
                ]
            )
            return response.choices[0].message.content
        except:
            return "这款相机在您的预算范围内提供了最契合的功能组合，尤其是它在您最关注的拍摄场景中表现优异。"

    def handle_query(self, structured_prompt):
        """
        主入口：
        structured_prompt 是由 app.py 拼接好的多维需求描述字符串。
        """
        # 1. 解析解析多维需求
        intent = self._get_llm_intent(structured_prompt)
        
        max_p = intent.get("max_price", 999999)
        sort_f = intent.get("sort_field", "LowLight_Score")
        
        # 2. 执行数据库检索
        # 即使是复杂的表单需求，最终也落脚在价格上限和核心指标排序上
        results = self.db.query(Camera)\
            .filter(Camera.Price <= max_p)\
            .order_by(desc(getattr(Camera, sort_f)))\
            .limit(3)\
            .all()

        if not results:
            return None, None, intent, "目前库中没有完全满足所有严苛条件的相机，建议放宽预算或功能要求。"

        # 3. 生成基于数据的专家点评
        expert_reason = self._generate_expert_advice(results[0], structured_prompt)

        # 4. 准备可视化组件
        radar_path = visualizer.draw_radar(results[0])
        bar_path = visualizer.draw_comparison(results, sort_f)

        return results, (radar_path, bar_path), intent, expert_reason

    def __del__(self):
        """析构时关闭数据库连接"""
        if hasattr(self, 'db'):
            self.db.close()