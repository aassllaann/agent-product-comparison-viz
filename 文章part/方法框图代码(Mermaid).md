flowchart TD
    A(["👤 用户自然语言输入\n例：'3000块以内拍视频的相机，轻一点'"])

    %% ─────────── 第一层：意图解析 ───────────
    subgraph L1["① 路由与意图解析层（对应贡献一：对话式推荐范式）"]
        direction TB
        B["CategoryDetector\n关键词匹配 + LLM辅助分类\n→ 识别商品品类 (category_key)"]
        C["ParseIntentGeneric\n基于 CoT 思维链 + JSON Schema 约束\n提取结构化槽位：\n• usage（使用场景）\n• max_price（预算上限）\n• sort_field（排序维度）\n• brand（品牌偏好）\n• summary（需求摘要）\n• owned_items（已有设备）"]
        B1{"是否预置品类?"}
        B2["已知品类路径\n直接路由至注册 Agent"]
        B3["未知品类路径\nDynamicAgent：\nLLM 自动推演评分维度"]
    end

    %% ─────────── 第二层：双层推荐引擎 ───────────
    subgraph L2["② 双层场景化推荐引擎 \n（对应贡献二：可靠性 + 灵活性）"]
        direction TB
        D{"第一层：Golden Sets\n专家知识预设\n命中 usage 场景?"}
        E1["场景匹配成功\n从预设精品池中检索\n→ 预算过滤（Price ≤ max_price）\n→ 品牌过滤（brand 字段）\n→ sort_field 排序 → Top-3"]
        E2{"候选数 ≥ 3?"}
        F["第二层：Fallback 降级查询\n• 全量数据库排序\n• 排除已有 ID\n• sort_field 动态排序\n• 补足至 3 款"]
        G["集合并集 + 去重\n确保输出 3~5 款不重复商品"]
    end

    %% ─────────── 第三层：多Agent评分与生态联动 ───────────
    subgraph L3["③ 多 Agent 领域决策层（对应贡献三：跨品类生态推荐）"]
        direction LR
        H["MultiCategoryAgent\n中央调度路由器"]
        subgraph AGENTS["品类专家 Agent 矩阵"]
            I1["CameraAgent\n维度权重：\nPortability×0.3\nLowLight×0.4\nVideo×0.3"]
            I2["LaptopAgent\n维度权重：\nBattery×0.35\nPerformance×0.35\nPortability×0.3"]
            I3["PhoneAgent\n维度权重：\nPerf×0.35\nBattery×0.3\nCamera×0.35"]
            I4["GPUAgent\n维度权重：\nGaming×0.35\nCreative×0.25\nThermal×0.20\nValue×0.20"]
            I5["DynamicAgent\nLLM 即时生成\n3~5个评分维度"]
        end
        subgraph ECO["跨品类生态联动"]
            J1["EcosystemConfig\n生态主题识别：\n移动办公
移动娱乐
专业游戏
摄影创作
音频发烧
健康运动"]
            J2["关键词权重评分\n识别用户所属生态\n过滤已有设备\n生成搭配建议"]
        end
        H --> AGENTS
        H --> ECO
        J1 --> J2
    end

    %% ─────────── 第四层：可视化输出 ───────────
    subgraph L4["④ 可视化输出层（对应贡献四：可解释性增强）"]
        direction TB
        K["LLM 生成推荐理由\n_get_individual_reasons()\n自然语言解释为何推荐此款"]
        L1_v["雷达图（Scatterpolar）\n多维整体性能对比\n• 直观感知各产品强弱项\n• 0~100 归一化坐标轴"]
        L2_v["分组柱状图（barmode=group）\n单项维度横向对比\n• 支持悬停显示精确数值"]
        M["ProductCard 商品卡片\n玻璃拟态 UI 设计\n• 品牌/型号/价格\n• 评分进度条\n• 参数标签"]
    end

    %% ─────────── 数据层 ───────────
    subgraph DB["⑤ 事实数据层（幻觉抑制）"]
        DB1["SQLite + SQLAlchemy ORM\n结构化商品数据库\n覆盖10+数码品类"]
        DB2["离线评分计算\nMin-Max归一化\n→ 0~100标准评分区间"]
        DB3["数据清洗流水线\nCSV爬取 → 类型转换\n→ 缺失值处理 → 入库"]
    end

    N(["📱 前端展示\n推荐理由 + 商品卡片 + 雷达图 + 柱状图 + 跨品类搭配建议"])

    %% ─────────── 连接关系 ───────────
    A --> B --> B1
    B1 -- 已知品类 --> B2 --> C
    B1 -- 未知品类 --> B3 --> C
    C --> D
    D -- "命中场景" --> E1 --> E2
    E2 -- "是" --> G
    E2 -- "否（不足3款）" --> F --> G
    D -- "未命中" --> F
    G --> H
    DB1 --> E1
    DB1 --> F
    DB2 --> DB1
    DB3 --> DB2
    I1 & I2 & I3 & I4 & I5 --> K
    J2 --> K
    K --> L1_v & L2_v & M --> N
