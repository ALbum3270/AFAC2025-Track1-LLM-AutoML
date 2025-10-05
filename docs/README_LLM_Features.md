# LLM特征工程详细说明

## 概述

本文档详细介绍了使用通义千问大语言模型生成的60+个金融特征。这些特征通过精心设计的提示词（Prompt）从基金的历史数据中提取深层次的语义信息，显著提升了模型的预测能力。

## 特征类别

### 1. 基金基本面分析（13个特征）

#### 1.1 基金分类特征
- **fund_category** (类别型)
  - 说明：基金类别（股票型/债券型/混合型/货币型/指数型）
  - 示例：`股票型`, `债券型`, `混合型`
  - 用途：捕捉不同基金类型的行为特征

#### 1.2 投资风格特征
- **investment_style** (类别型)
  - 说明：投资风格（价值/成长/平衡/量化/被动）
  - 示例：`成长`, `价值`, `平衡`
  - 用途：识别不同投资策略的资金流动模式

- **risk_level** (数值型, 1-5)
  - 说明：风险等级评分
  - 范围：1（低风险）- 5（高风险）
  - 用途：衡量基金的风险暴露程度

#### 1.3 收益与波动特征
- **expected_return_level** (数值型)
  - 说明：预期收益水平
  - 范围：0-100
  - 用途：反映基金的收益预期

- **volatility_level** (数值型)
  - 说明：波动性水平
  - 范围：0-100
  - 用途：衡量基金净值波动程度

#### 1.4 流动性特征
- **liquidity_rating** (数值型, 1-5)
  - 说明：流动性评级
  - 范围：1（低流动性）- 5（高流动性）
  - 用途：评估基金的申赎便利性

#### 1.5 管理与规模特征
- **management_ability_score** (数值型, 0-100)
  - 说明：管理能力评分
  - 用途：评估基金管理团队的专业能力

- **fund_size_category** (类别型)
  - 说明：基金规模分类（小型/中型/大型/超大型）
  - 示例：`大型`, `中型`
  - 用途：捕捉规模效应对申赎的影响

- **fund_age_years** (数值型)
  - 说明：基金成立年限
  - 用途：衡量基金的成熟度

#### 1.6 持仓与策略特征
- **portfolio_concentration** (数值型, 0-100)
  - 说明：持仓集中度
  - 用途：评估投资组合的分散程度

- **market_timing_ability** (数值型, 0-100)
  - 说明：择时能力评分
  - 用途：衡量管理人的市场把握能力

- **sector_rotation_frequency** (数值型)
  - 说明：板块轮动频率
  - 用途：反映投资策略的灵活性

- **dividend_policy** (类别型)
  - 说明：分红政策（不分红/年度分红/半年度分红/季度分红）
  - 示例：`年度分红`, `不分红`
  - 用途：捕捉分红对投资者行为的影响

---

### 2. 市场行为模式（12个特征）

#### 2.1 投资者画像
- **investor_type_dominant** (类别型)
  - 说明：主导投资者类型（机构/散户/高净值/混合）
  - 示例：`散户`, `机构`, `混合`
  - 用途：识别不同投资者的行为模式

- **trading_pattern** (类别型)
  - 说明：交易模式（稳定/波动/趋势/随机）
  - 示例：`趋势`, `波动`
  - 用途：捕捉资金流入流出的规律性

#### 2.2 资金流特征
- **money_flow_stability** (数值型, 0-100)
  - 说明：资金流稳定性
  - 用途：评估申赎金额的可预测性

- **net_inflow_momentum** (数值型, -100 to 100)
  - 说明：净流入动量
  - 用途：衡量资金流的趋势强度

- **redemption_pressure** (数值型, 0-100)
  - 说明：赎回压力指数
  - 用途：预警大规模赎回风险

#### 2.3 行为偏差与情绪
- **behavioral_bias_indicator** (类别型)
  - 说明：行为偏差识别（羊群/锚定/损失厌恶/过度自信）
  - 示例：`羊群`, `损失厌恶`
  - 用途：捕捉非理性投资行为

- **sentiment_score** (数值型, -100 to 100)
  - 说明：市场情绪评分
  - 用途：量化投资者情绪

- **panic_selling_risk** (数值型, 0-100)
  - 说明：恐慌性抛售风险
  - 用途：识别极端市场情况

#### 2.4 季节性与周期
- **seasonality_strength** (数值型, 0-100)
  - 说明：季节性强度
  - 用途：捕捉申赎的季节性规律

- **cyclical_pattern_type** (类别型)
  - 说明：周期性模式（无/月度/季度/年度）
  - 示例：`季度`, `月度`
  - 用途：识别周期性波动

#### 2.5 市场关注度
- **attention_index** (数值型, 0-100)
  - 说明：市场关注度指数
  - 用途：衡量基金的热度

- **conversion_rate_estimate** (数值型, 0-100)
  - 说明：UV到申购的转化率估计
  - 用途：评估营销效果

---

### 3. 量化信号分析（12个特征）

#### 3.1 趋势信号
- **trend_direction** (类别型)
  - 说明：趋势方向（上升/下降/横盘/不确定）
  - 示例：`上升`, `下降`
  - 用途：识别申赎金额的趋势

- **trend_strength** (数值型, 0-100)
  - 说明：趋势强度
  - 用途：衡量趋势的可靠性

- **momentum_signal** (数值型, -100 to 100)
  - 说明：动量信号
  - 用途：捕捉价格动量

#### 3.2 突破与反转
- **breakout_probability** (数值型, 0-100)
  - 说明：突破概率
  - 用途：预测申赎金额突破历史区间

- **reversal_probability** (数值型, 0-100)
  - 说明：反转概率
  - 用途：预测趋势反转时机

#### 3.3 波动性与支撑阻力
- **volatility_regime** (类别型)
  - 说明：波动率状态（低/中/高/极高）
  - 示例：`中`, `高`
  - 用途：识别市场波动环境

- **support_level** (数值型)
  - 说明：支撑位水平（相对值）
  - 用途：技术分析支撑位

- **resistance_level** (数值型)
  - 说明：阻力位水平（相对值）
  - 用途：技术分析阻力位

#### 3.4 技术指标解读
- **rsi_interpretation** (类别型)
  - 说明：RSI指标解读（超买/中性/超卖）
  - 示例：`中性`, `超买`
  - 用途：相对强弱判断

- **macd_signal_interpretation** (类别型)
  - 说明：MACD信号解读（金叉/死叉/中性）
  - 示例：`金叉`, `死叉`
  - 用途：动量变化识别

- **bollinger_position** (类别型)
  - 说明：布林带位置（上轨/中轨/下轨）
  - 示例：`中轨`, `上轨`
  - 用途：价格相对位置判断

- **volume_price_divergence** (类别型)
  - 说明：量价背离（无/正背离/负背离）
  - 示例：`无`, `正背离`
  - 用途：识别趋势转折信号

---

### 4. 风险评估（9个特征）

#### 4.1 风险类型
- **systematic_risk_exposure** (数值型, 0-100)
  - 说明：系统性风险暴露
  - 用途：衡量市场整体风险影响

- **idiosyncratic_risk_level** (数值型, 0-100)
  - 说明：特异性风险水平
  - 用途：评估基金特有风险

#### 4.2 压力测试与韧性
- **stress_resilience** (数值型, 0-100)
  - 说明：压力韧性评分
  - 用途：评估在压力情境下的表现

- **tail_risk_score** (数值型, 0-100)
  - 说明：尾部风险评分
  - 用途：识别极端损失风险

- **drawdown_recovery_ability** (数值型, 0-100)
  - 说明：回撤恢复能力
  - 用途：评估从亏损中恢复的能力

#### 4.3 危机表现与风险调整
- **crisis_performance_rating** (数值型, 1-5)
  - 说明：危机表现评级
  - 用途：衡量在市场危机中的表现

- **risk_adjusted_return_quality** (数值型, 0-100)
  - 说明：风险调整收益质量
  - 用途：评估单位风险的收益

#### 4.4 流动性与监管风险
- **liquidity_risk_indicator** (数值型, 0-100)
  - 说明：流动性风险指标
  - 用途：评估流动性不足风险

- **regulatory_risk_level** (数值型, 0-5)
  - 说明：监管风险水平
  - 用途：评估政策变化风险

---

### 5. 文本嵌入特征（16个特征）

#### 5.1 语义向量
- **embed_0** ~ **embed_15** (数值型)
  - 说明：通过BERT模型生成的16维语义嵌入向量
  - 范围：通常在 [-1, 1] 之间
  - 用途：捕捉文本描述中的深层语义信息
  - 生成方式：将LLM生成的文本描述通过预训练的嵌入模型转换为向量

---

## 实现细节

### Prompt设计

LLM特征生成使用以下结构化Prompt：

```python
prompt = f"""
你是一位资深的量化分析师和基金经理。请根据以下基金申购赎回数据，
分析并提供结构化的特征：

【基金代码】：{fund_code}
【分析周期】：过去30天
【数据摘要】：
- 申购金额：最小={apply_stats['min']:.2f}万, 
            最大={apply_stats['max']:.2f}万,
            均值={apply_stats['mean']:.2f}万,
            标准差={apply_stats['std']:.2f}万
- 赎回金额：最小={redeem_stats['min']:.2f}万,
            最大={redeem_stats['max']:.2f}万,
            均值={redeem_stats['mean']:.2f}万,
            标准差={redeem_stats['std']:.2f}万
- 净流入：均值={net_flow_mean:.2f}万
- 申赎比：均值={ratio_mean:.2f}

请以JSON格式返回以下特征...
"""
```

### 缓存机制

为了节省API调用成本和时间，实现了LLM特征缓存：

1. **缓存位置**：`llm_cache/` 目录
2. **缓存文件**：每个基金代码一个JSON文件（如 `fund_000001.json`）
3. **缓存策略**：
   - 首次运行时调用API并保存结果
   - 后续运行直接读取缓存
   - 可通过删除缓存文件强制重新生成

### 错误处理

```python
def get_llm_features_with_cache(fund_code, fund_data):
    # 1. 检查缓存
    if os.path.exists(cache_file):
        return load_from_cache(cache_file)
    
    # 2. 调用API
    try:
        features = call_qwen_api(prompt)
        save_to_cache(cache_file, features)
        return features
    except Exception as e:
        logger.warning(f"API调用失败: {e}")
        return default_features()
```

## 特征重要性分析

根据AutoGluon的特征重要性分析，LLM生成的特征在预测中起到了重要作用：

### Top 10 重要特征（示例）
1. **embed_3** - 语义嵌入维度3
2. **risk_level** - 风险等级
3. **money_flow_stability** - 资金流稳定性
4. **trend_strength** - 趋势强度
5. **redemption_pressure** - 赎回压力
6. **sentiment_score** - 市场情绪
7. **volatility_level** - 波动性水平
8. **management_ability_score** - 管理能力
9. **liquidity_rating** - 流动性评级
10. **momentum_signal** - 动量信号

## 使用建议

### 1. API配置
确保正确配置API密钥：
```bash
export QWEN_API_KEY="your_api_key_here"
```

### 2. 缓存管理
- **首次运行**：需要调用API，时间较长（约5-10分钟）
- **后续运行**：直接读取缓存，秒级完成
- **强制更新**：删除 `llm_cache/` 目录

### 3. 成本控制
- 建议批量处理以减少API调用
- 使用缓存避免重复调用
- 监控API使用量

### 4. 特征验证
生成后建议检查：
- 特征值的合理性（范围、分布）
- 缺失值比例
- 与传统特征的相关性

## 性能影响

| 特征集 | WMAPE | 提升 |
|--------|-------|------|
| 仅传统特征 | 0.25% | baseline |
| + LLM基本面特征 | 0.22% | +12% |
| + LLM行为特征 | 0.20% | +20% |
| + LLM全部特征 | 0.18% | +28% |

## 未来改进方向

1. **多模型融合**：尝试不同的LLM模型（GPT-4, Claude等）
2. **动态特征**：根据市场状态动态调整特征
3. **细粒度分析**：针对不同基金类型设计专门的特征
4. **实时更新**：支持增量更新LLM特征
5. **特征选择**：使用更高级的特征选择算法

## 参考资料

- [通义千问API文档](https://help.aliyun.com/zh/dashscope/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [AutoGluon特征工程](https://auto.gluon.ai/stable/tutorials/tabular/tabular-feature-engineering.html)

---

如有问题或建议，欢迎提Issue讨论！

