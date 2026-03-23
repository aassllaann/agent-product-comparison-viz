import re

file_path = r"d:\毕业设计\代码\文章part\全文latex存档.md"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Pattern for lstlisting blocks
pattern = re.compile(r"\\begin\{lstlisting\}(.*?)\\end\{lstlisting\}", re.DOTALL)
matches = pattern.findall(content)

print(f"Total lstlisting blocks found: {len(matches)}")

replacements = [
    # 0: intent_prompt
    r"""[language=Pseudocode, caption={通用意图解析算法实现}, label={code:intent_prompt}]
Algorithm 1: Parse User Intent (ParseIntentGeneric)
Input: user_message (String), category_metadata (Object), history_context (Context)
Output: intent_structure (JSON Object)

Procedure:
1. Initialize extraction_schema with fields:
   [usage, budget_level, max_price, sort_field, summary, product_type, brand, owned_items]
2. Define inference_rules:
   a. If user_message mentions "flagship" or "unlimited budget":
         max_price = 999999
   b. Default max_price = 20000 if not specified
3. prompt <- ConstructLLMPrompt(category_metadata, extraction_schema, inference_rules)
4. input_sequence <- Combine(prompt, history_context, user_message)
5. Try:
6.     model_response <- CallLargeLanguageModel(input_sequence, enforce_format="JSON")
7.     intent_structure <- ParseJSON(model_response)
8. Catch Exception:
9.     intent_structure <- DefaultFallbackIntent()
10. Return intent_structure
""",
    # 1: dynamic_scoring
    r"""[language=Pseudocode, caption={基于LLM的动态评分体系构建算法}, label={code:dynamic_scoring}]
Algorithm 2: Build Dynamic Scoring System (BuildScoringSystem)
Input: category_key (String), category_name (String), user_context (String)
Output: scoring_system_config (JSON Object)

Procedure:
1. system_role <- "Domain Expert for " + category_name
2. requirement_def <- [
       "Design 3-5 core scoring dimensions (Name, Field, Weight, Description)",
       "Design 3-5 typical usage scenarios with keyword matching rules",
       "Specify the default sorting dimension"
   ]
3. prompt <- ConstructPrompt(system_role, requirement_def, user_context)
4. Try:
5.     model_response <- CallLargeLanguageModel(prompt, enforce_format="JSON")
6.     scoring_system_config <- ValidateAndParseJSON(model_response)
7.     CacheConfiguration(category_key, scoring_system_config)
8. Catch Exception:
9.     scoring_system_config <- LoadGenericFallbackConfig(category_name)
10. Return scoring_system_config
""",
    # 2: specific_dimensions
    r"""[language=Pseudocode, caption={异构商品专有评分维度定义抽象}, label={code:specific_dimensions}]
Structure ScoringDimension:
    Name: String         // e.g., "Portability"
    Field: String        // e.g., "Portability_Score"
    Weight: Float        // e.g., 0.3
    Description: String  // Dimension explanation

Class CameraExpertAgent Inherits BaseRecommendationAgent:
    Function GetSpecificDimensions() -> List<ScoringDimension>:
        Return [
            ScoringDimension("Portability", "Portability_Score", 0.3, "Weight and Size"),
            ScoringDimension("Low Light", "LowLight_Score", 0.4, "High ISO performance"),
            ScoringDimension("Video Capability", "Video_Score", 0.3, "Video recording specs")
        ]

Class GPUExpertAgent Inherits BaseRecommendationAgent:
    Function GetCategoryConfig() -> CategoryConfig:
        Return CategoryConfig(
            ScoringDimensions=[
                ScoringDimension("Gaming", "Gaming_Score", 0.35, "FPS in games"),
                ScoringDimension("Creative", "Creative_Score", 0.25, "Rendering & AI"),
                ScoringDimension("Thermal", "Thermal_Score", 0.20, "Cooling & Noise"),
                ScoringDimension("Value", "Value_Score", 0.20, "Price-to-performance")
            ]
        )
""",
    # 3: two_layer_search
    r"""[language=Pseudocode, caption={双层场景化推荐检索策略算法}, label={code:two_layer_search}]
Algorithm 3: Two-Layer Scenario Recommendation
Input: user_message (String), category_agent (Object)
Output: candidate_products (List)

Procedure:
1. intent_data <- ParseIntentGeneric(user_message, category_agent.metadata)
2. target_scenario <- MatchScenarioRules(intent_data.usage, intent_data.summary)
3. candidate_products <- EmptyList()

// First Layer: Expert Knowledge Scenario Recall
4. If target_scenario is not Null Then:
5.     preset_pool <- GetPresetProductsFromDatabase(target_scenario)
6.     filtered_pool <- FilterByBudgetAndBrand(preset_pool, intent_data.max_price, intent_data.brand)
7.     candidate_products <- SortByDimension(filtered_pool, intent_data.sort_field)
8.     candidate_products <- Limit(candidate_products, Top=3)

// Second Layer: Fallback Database Retrieval
9. If Length(candidate_products) < 3 Then:
10.    exclude_ids <- GetProductIDs(candidate_products)
11.    fallback_pool <- QueryDatabase(
           Price <= intent_data.max_price,
           Brand matches intent_data.brand,
           ExcludeIDs in exclude_ids,
           SortBy = intent_data.sort_field,
           Limit = 3
       )
12.    candidate_products <- Merge(candidate_products, fallback_pool)
13.    candidate_products <- Limit(candidate_products, 3)

14. Return candidate_products
""",
    # 4: ecosystem_config
    r"""[language=Pseudocode, caption={跨品类生态环境定义与关联结构}, label={code:ecosystem_config}]
Structure EcosystemConfiguration:
    Ecosystems <- Collection of:
        "Mobile_Office_Ecosystem":
            Description: "Digital products for mobile work and remote collaboration"
            Core_Nodes: ["Laptop_Computer"]
            Peripheral_Nodes: ["Portable_Monitor", "Bluetooth_Speaker", "Smartwatch"]
            Trigger_Scenarios: ["Office", "Meeting", "Business Trip", "Remote"]
            Attention_Weights:
                High_Priority_Keywords: ["Business", "Remote Work", "Meeting"]
                Medium_Priority_Keywords: ["Document", "Presentation"]

        "Professional_Gaming_Ecosystem":
            Core_Nodes: ["Gaming_Laptop", "Gaming_Console", "Desktop_GPU"]
            Peripheral_Nodes: ["Gaming_Monitor", "Gaming_Headset"]
            Trigger_Scenarios: ["3A Games", "Esports", "Console Gaming"]
            Attention_Weights:
                High_Priority_Keywords: ["Esports", "Steam", "Cyberpunk"]
                Medium_Priority_Keywords: ["Entertainment", "High Frame Rate"]

Function MatchEcosystem(user_intent):
    Calculate alignment scores for all ecosystems based on intent keywords.
    Return ecosystems with score > threshold.
""",
    # 5: app.py State Init
    r"""[language=Pseudocode, caption={系统状态初始化机制}]
Procedure InitializeSystemSessionState(session)
    // Ensure critical components persist across interactions
    If 'central_router_agent' not exists in session Then:
        session['central_router_agent'] <- Instantiate(MultiCategoryAgent)
        
    If 'conversation_history' not exists in session Then:
        session['conversation_history'] <- EmptyList()
        
    If 'current_recommendation_results' not exists in session Then:
        session['current_recommendation_results'] <- Null
        
    If 'active_visualization_charts' not exists in session Then:
        session['active_visualization_charts'] <- Null
End Procedure
""",
    # 6: base_agent.py LLM init
    r"""[language=Pseudocode, caption={大语言模型客户端初始化架构}]
Procedure InitializeModelClient(self)
    // Setup connection to the Large Language Model service
    self.llm_interface <- Instantiate LLM_Client With Parameters:
        API_Key = RetrieveFromConfig("API_KEY")
        Base_Endpoint = "https://api.model-provider.com/v1"
        Timeout_Seconds = 30
        Max_Retries = 3

    // Verify connection status
    If ConnectionFails(self.llm_interface) Then:
        LogWarning("Model service unavailable. Fallback protocols enabled.")
End Procedure
""",
    # 7: base_agent.py BaseProductAgent (lines 331-347 in source, corresponds to match 7)
    r"""[language=Pseudocode, caption={推荐代理基类抽象接口定义}]
Abstract Class BaseProductAgent:
    // Shared universal dimensions for all products
    Constant UNIVERSAL_DIMENSIONS <- [ Value_Score, Brand_Score, User_Rating ]

    // Abstract methods to be implemented by specific category agents
    Abstract Function GetCategoryConfig() -> CategoryConfig
    Abstract Function GetSpecificDimensions() -> List<ScoringDimension>
    Abstract Function GetDatabaseSchema() -> DatabaseModel
    Abstract Function FormatProductInformation(product) -> String
    
    // Concrete flow pipeline template
    Function ExecuteRecommendationPipeline(user_input, intent) -> Result:
        candidates <- FetchCandidates(intent)
        filtered <- FilterCandidates(candidates)
        sorted <- SortCandidates(filtered, intent.sort_field)
        Return sorted
End Class
""",
    # 8: electronics_agents.py (lines 351-367)
    r"""[language=Pseudocode, caption={核心商品推荐流转控制序列}]
Function ProcessRecommendationRequest(self, user_message, history, db_model) -> Tuple:
    // Step 1: Semantic Intent Extraction
    intent_data <- Execute(parse_intent_generic, user_message)
    
    // Step 2: Extract hard constraints and soft preferences
    budget_limit <- intent_data.max_price
    target_scenario <- MatchScenario(intent_data.usage)
    
    // Step 3: Candidate Generation & Ranking
    candidate_products <- GetPreconfiguredSet(target_scenario)
    ranked_results <- ApplyFilteringAndSorting(candidate_products, intent_data)
    
    // Step 4: Fallback Mechanism (if results < 3)
    If Count(ranked_results) < 3 Then:
        ranked_results <- Execute(fallback_full_database_search, intent_data)
        
    // Step 5: Explainability & Visualization Generation
    justifications <- GenerateExplanationsWithLLM(ranked_results, user_message, intent_data)
    visual_charts, analytical_text <- GenerateVisualizations(ranked_results, intent_data)
    
    Return (justifications, visual_charts, ranked_results, analytical_text)
""",
    # 9: electronics_agents.py CameraAgent
    r"""[language=Pseudocode, caption={具体领域代理（相机）实现示例}]
Class CameraExpertAgent Inherits BaseDatabaseAgent:
    
    Properties:
    Preset_Scenarios: { 
        "Vlog": ["Sony ZV-E10", "Canon G7X"], 
        "Travel": ["Fuji X100", "Nikon Z fc"],
        "Beginner": ["Canon R50", "Nikon Z30"]
    }
    
    Function GetSpecificDimensions() -> List<Dimension>:
        Return [
            Dimension("Portability", "Portability_Score", weight=0.3),
            Dimension("Low Light", "LowLight_Score", weight=0.4),
            Dimension("Video Capability", "Video_Score", weight=0.3)
        ]
        
    Function ExtractProductText(product) -> String:
        Return Concatenate(
            "Model:", product.Brand, product.Model,
            "Price:", product.Price,
            "Parameters: LowLight:", product.LowLight_Score, 
            "Video:", product.Video_Score
        )
End Class
""",
    # 10: multi_agent.py Handle
    r"""[language=Pseudocode, caption={多品类中央路由分发算法}]
Algorithm 4: Central Multi-Category Dispatcher
Input: user_message (String), conversation_history (List)
Output: recommendation_results, visualizations, ecosystem_suggestions

Procedure:
1. Initialize intent_detector <- CategoryDetector()
2. (category_key, category_name) <- intent_detector.DetectCategory(user_message)
   
3. target_agent <- GetRegisteredAgent(category_key)
4. If target_agent is Null Then:
5.     target_agent <- DynamicallyConstructAgent(category_key, category_name)
6.     If target_agent is Null Then:
7.         Return Error("Unable to build handling agent")

8. (reasons, charts, products, analyses) <- target_agent.ProcessRecommendationRequest(user_message, history)

9. ecosystem_suggestions <- AnalyzeEcosystemSynergy(user_message, category_key, products)

10. Return (reasons, charts, products, analyses, ecosystem_suggestions)
""",
    # 11: models.py Camera
    r"""[language=Pseudocode, caption={关系型数据库表的结构化表示}]
Database Schema Definition: Table "Cameras"

PrimaryKey: id (Integer)
Standard_Attributes:
    Brand (String)        // Brand identifier
    Model (String)        // Product model name
    Price (Float)         // Current market price
    Year (Integer)        // Release year
    Image_File (String)   // Path to product image

Technical_Specifications:
    Total_Megapixels (Float) // Sensor resolution
    Sensor_Type (String)     // Full Frame, APS-C, etc.
    Weight_g (Float)         // Device weight
    Supports_4K (Boolean)    // 4K video recording capability

Scoring_Metrics (Calculated offline):
    Portability_Score (Float) [0-100]
    LowLight_Score (Float)    [0-100]
    Video_Score (Float)       [0-100]
""",
    # 12: query = self.db.query...
    r"""[language=Pseudocode, caption={事实数据约束与动态检索过滤}]
Function ExecuteConstrainedSearch(database_session, intent_data, exclude_ids):
    // Construct base query
    search_query <- InitializeQuery(database_session, TargetModel)
    
    // Apply hard constraint: Budget Barrier
    search_query <- search_query.AddCondition( Price <= intent_data.max_price )
    
    // Apply exclusion filter
    If exclude_ids is not Empty Then:
        search_query <- search_query.ExcludeIDs( exclude_ids )
        
    // Apply dynamic sorting based on LLM extracted intention
    If TargetModel possesses intent_data.sort_field Then:
        search_query <- search_query.OrderByDescending( intent_data.sort_field )
    Else:
        search_query <- search_query.OrderByDescending( Default_Sort_Field )
        
    Return search_query.ExecuteAndFetch(Limit=3)
End Function
""",
    # 13: app.py Render Product Card
    r"""[language=Pseudocode, caption={基于状态驱动的视图组件渲染机制}]
Procedure RenderProductCardWidget(product_entity, justification_text)
    // Extract properties
    p_brand <- product_entity.Brand
    p_model <- product_entity.Model
    p_price <- FormatCurrency(product_entity.Price)
    
    // Generate specialized parameter UI
    spec_html <- BuildSpecificationTags(product_entity)
    score_html <- BuildScoreProgressBars(product_entity)
    
    // Construct Interactive Element
    card_container <- CreateUIElement("div", class="product-card glassmorphism-effect")
    card_container.AppendChild( CreateTextNode(p_brand + " " + p_model, class="title") )
    card_container.AppendChild( CreateTextNode("¥" + p_price, class="price-highlight") )
    
    card_container.AppendChild( score_html )
    card_container.AppendChild( spec_html )
    
    // Inject LLM generated reason
    If justification_text is not Null Then:
        card_container.AppendChild( CreateTextBox(justification_text, class="reason-box") )
        
    DisplayToFrontend(card_container)
End Procedure
""",
    # 14: visualizer.py
    r"""[language=Pseudocode, caption={多维性能比较雷达图与柱状图渲染算法}]
Function GenerateVisualizationCharts(candidate_products, dimension_configuration) -> InteractiveCharts:
    radar_chart <- Initialize(PolarChartEngine)
    bar_chart <- Initialize(BarChartEngine)
    
    dimension_labels <- ExtractLabels(dimension_configuration)
    dimension_fields <- ExtractFields(dimension_configuration)
    
    // 1. Render Radar Chart (Holistic Evaluation)
    For Each product In candidate_products:
        dimension_values <- CalculateNormalizedScores(product, dimension_fields)
        
        // Close polygon for radar
        dimension_values.Append(dimension_values[0])
        closed_labels <- Append(dimension_labels, dimension_labels[0])
        
        radar_chart.AddTrace(
            Type="Scatterpolar",
            R_Axis=dimension_values, Theta_Axis=closed_labels,
            Name=product.Model
        )
        
    // 2. Render Bar Chart (Relative Competitiveness)
    bar_chart.AddGroupedBars(
        X_Axis=dimension_labels,
        SeriesData=ExtractAllScores(candidate_products),
        SeriesNames=ExtractModels(candidate_products)
    )
        
    Return (radar_chart, bar_chart)
End Function
"""
]

# Ensure we have the same number of matches and replacements (15 matches for 15 replacements)
if len(matches) == len(replacements):
    new_content = content
    for i, rep in enumerate(replacements):
        old_block = f"\\begin{{lstlisting}}{matches[i]}\\end{{lstlisting}}"
        new_block = f"\\begin{{lstlisting}}{rep}\\end{{lstlisting}}"
        new_content = new_content.replace(old_block, new_block, 1)
        
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Successfully replaced all 15 code blocks with pseudocode.")
else:
    print(f"Mismatch! Found {len(matches)} blocks but we have {len(replacements)} replacements.")
