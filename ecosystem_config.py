"""
生态系统配置

定义5大生态场景及其关联的商品品类
优化版：增强场景识别和关联推荐精准度
"""

# 生态系统场景定义
ECOSYSTEMS = {
    "移动办公": {
        "name_en": "mobile_office",
        "description": "适合移动办公、远程协作的数码产品组合",
        "core_categories": ["laptop"],  # 核心设备
        "related_categories": ["monitor", "bluetooth_speaker", "smartwatch"],  # 关联设备
        "scenarios": ["办公", "会议", "出差", "远程工作"],
        "keywords": {
            "high": ["办公", "工作", "商务", "出差", "会议", "远程"],  # 高权重关键词
            "medium": ["文档", "表格", "演示", "ppt"]  # 中权重关键词
        }
    },
    
    "移动娱乐": {
        "name_en": "mobile_entertainment",
        "description": "适合日常娱乐、休闲放松的数码产品组合",
        "core_categories": ["phone", "tablet"],
        "related_categories": ["bluetooth_speaker", "headphone", "smartwatch"],
        "scenarios": ["追剧", "听歌", "游戏", "阅读", "运动"],
        "keywords": {
            "high": ["娱乐", "追剧", "看电影", "听歌", "音乐", "阅读"],
            "medium": ["休闲", "放松", "打发时间"]
        }
    },
    
    "专业游戏": {
        "name_en": "professional_gaming",
        "description": "适合游戏爱好者、电竞玩家的高性能设备组合",
        "core_categories": ["laptop", "gaming_console", "gpu"],
        "related_categories": ["monitor", "headphone"],
        "scenarios": ["3A游戏", "电竞", "主机游戏", "PC游戏"],
        "keywords": {
            "high": ["游戏", "电竞", "3a", "fps", "moba", "吃鸡"],
            "medium": ["玩", "打游戏", "主机", "帧率", "显卡"]
        }
    },
    
    "摄影创作": {
        "name_en": "photography_creation",
        "description": "适合摄影师、视频创作者的专业创作工具组合",
        "core_categories": ["camera", "laptop", "gpu"],
        "related_categories": ["monitor", "tablet"],
        "scenarios": ["摄影", "视频剪辑", "后期处理", "修图"],
        "keywords": {
            "high": ["摄影", "拍照", "视频", "剪辑", "修图", "后期", "渲染", "ai"],
            "medium": ["创作", "设计", "建模"]
        }
    },
    
    "音频发烧": {
        "name_en": "audio_enthusiast",
        "description": "适合音乐爱好者、HiFi发烧友的高品质音频设备组合",
        "core_categories": ["headphone", "bluetooth_speaker"],
        "related_categories": ["phone", "tablet"],
        "scenarios": ["听歌", "HiFi", "音乐欣赏"],
        "keywords": {
            "high": ["音质", "hifi", "无损", "发烧", "耳机", "音箱"],
            "medium": ["听", "播放器"]
        }
    },
    
    "健康运动": {
        "name_en": "health_fitness",
        "description": "适合运动爱好者的智能健康设备组合",
        "core_categories": ["smartwatch"],
        "related_categories": ["headphone", "phone"],
        "scenarios": ["跑步", "健身", "运动监测"],
        "keywords": {
            "high": ["跑步", "健身", "运动", "锻炼", "马拉松", "训练"],
            "medium": ["户外", "骑行", "游泳"]
        }
    }
}


def get_ecosystem_by_keywords(user_input: str):
    """
    根据用户输入关键词判断所属生态场景
    优化版：支持关键词权重，返回匹配度最高的场景
    
    Args:
        user_input: 用户输入的消息
        
    Returns:
        匹配的生态场景列表（按匹配度排序）
    """
    matched_ecosystems = {}
    user_input_lower = user_input.lower()
    
    for eco_name, eco_config in ECOSYSTEMS.items():
        score = 0
        
        # 高权重关键词匹配
        for keyword in eco_config["keywords"]["high"]:
            if keyword in user_input_lower:
                score += 10
        
        # 中权重关键词匹配
        for keyword in eco_config["keywords"]["medium"]:
            if keyword in user_input_lower:
                score += 5
        
        if score > 0:
            matched_ecosystems[eco_name] = score
    
    # 按分数排序，返回前2个最相关的场景
    sorted_ecosystems = sorted(matched_ecosystems.items(), key=lambda x: x[1], reverse=True)
    return [name for name, score in sorted_ecosystems[:2]]


def get_related_categories(category_key: str, matched_ecosystems: list = None, max_count: int = 3):
    """
    获取某个品类的关联品类（基于生态系统）
    优化版：基于匹配的场景智能筛选关联品类
    
    Args:
        category_key: 品类标识
        matched_ecosystems: 匹配到的生态场景列表
        max_count: 最多返回数量
        
    Returns:
        关联品类列表
    """
    related = {}  # 使用字典统计关联品类的权重
    
    for eco_name, eco_config in ECOSYSTEMS.items():
        # 如果指定了场景，只考虑这些场景
        if matched_ecosystems and eco_name not in matched_ecosystems:
            continue
            
        # 如果是核心设备，返回该生态的关联设备
        if category_key in eco_config["core_categories"]:
            for cat in eco_config["related_categories"]:
                related[cat] = related.get(cat, 0) + 2
        
        # 如果是关联设备，返回该生态的核心设备
        elif category_key in eco_config["related_categories"]:
            for cat in eco_config["core_categories"]:
                if cat != category_key:  # 排除自己
                    related[cat] = related.get(cat, 0) + 1
    
    # 按权重排序
    sorted_related = sorted(related.items(), key=lambda x: x[1], reverse=True)
    return [cat for cat, weight in sorted_related[:max_count]]


def get_ecosystem_recommendations(user_input: str, current_category: str = None):
    """
    基于用户输入和当前品类，推荐生态系统关联商品
    优化版：更精准的场景识别和关联推荐
    
    Args:
        user_input: 用户输入
        current_category: 当前识别的品类
        
    Returns:
        推荐信息字典
    """
    # 识别场景
    ecosystems = get_ecosystem_by_keywords(user_input)
    
    recommendations = {
        "matched_ecosystems": ecosystems,
        "related_categories": []
    }
    
    if current_category:
        # 基于场景获取关联品类
        related = get_related_categories(current_category, ecosystems)
        recommendations["related_categories"] = related
    
    return recommendations

