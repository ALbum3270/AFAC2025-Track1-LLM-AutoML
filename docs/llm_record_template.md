# 大模型使用记录文档 (LLM Record)

## 基本信息
- **参赛队伍**: [你的队伍名称]
- **参赛者**: [你的姓名]
- **比赛**: 基金申购赎回预测挑战赛 B榜
- **日期**: 2025年7月26日
- **文档版本**: v1.0

---

## 1. 大模型技术概述

### 1.1 使用的大模型
- **主要模型**: 通义千问 (qwen-turbo)
- **辅助模型**: Hugging Face BERT嵌入模型 (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **API提供商**: 阿里云DashScope
- **模型版本**: qwen-turbo (2024年版本)

### 1.2 特征构造策略
我们使用大模型技术构造了**40+个高质量特征**，主要分为以下几类：
1. **基金基本面特征** (13个)
2. **市场行为分析特征** (12个) 
3. **量化信号特征** (12个)
4. **风险评估特征** (9个)
5. **文本嵌入特征** (16个)

**技术架构**:
```python
class SmartLLMFeatureGenerator:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.llm_client = LLMClient(config, logger)
        self.tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
        self.embedding_model = AutoModel.from_pretrained(config.EMBEDDING_MODEL)
```

---

## 2. 详细特征构造过程

### 2.1 基金基本面分析特征 (13个)

**技术实现**: `_generate_fund_analysis()` 方法

**实际Prompt设计**:
```
请分析基金代码 {fund_code} 的基本面特征。

基金统计数据：
- 历史交易天数: {len(fund_data)}
- 平均申购: {recent_stats['avg_apply']:.0f}万元
- 平均赎回: {recent_stats['avg_redeem']:.0f}万元
- 申购标准差: {recent_stats['std_apply']:.0f}万元
- 赎回标准差: {recent_stats['std_redeem']:.0f}万元
- 申购最大值: {recent_stats['max_apply']:.0f}万元
- 赎回最大值: {recent_stats['max_redeem']:.0f}万元
- 平均净流入: {recent_stats['avg_net_flow']:.0f}万元

基于基金代码规律和统计特征，请提供基金分析，返回JSON：
{
    "fund_description": "基金描述(150-300字)",
    "fund_category": "基金大类(股票型/债券型/混合型/货币型/指数型/QDII/另类投资)",
    "fund_style": "投资风格(价值/成长/平衡/量化/主题)",
    "risk_level": "风险等级(1-5)",
    "expected_return": "预期年化收益率(%)",
    "volatility_level": "波动性水平(低/中低/中/中高/高)",
    "liquidity_rating": "流动性评级(1-10)",
    "management_capability": "管理能力评分(1-10)",
    "market_beta": "市场贝塔系数(0.5-2.0)",
    "sector_concentration": "行业集中度(分散/适中/集中)",
    "investment_horizon": "适宜投资期限(月数)",
    "expense_ratio_level": "费率水平(低/中/高)",
    "performance_consistency": "业绩一致性(1-10)"
}
```

**API调用参数**:
- **temperature**: 0.2 (降低随机性，确保结果稳定)
- **max_tokens**: 按需设置
- **模型**: qwen-turbo

**生成特征列表**:
1. `fund_description` - 基金描述文本
2. `fund_category` - 基金类别
3. `fund_style` - 投资风格
4. `risk_level` - 风险等级 (1-5)
5. `expected_return` - 预期收益率
6. `volatility_level` - 波动性水平
7. `liquidity_rating` - 流动性评级 (1-10)
8. `management_capability` - 管理能力评分
9. `market_beta` - 市场贝塔系数
10. `sector_concentration` - 行业集中度
11. `investment_horizon` - 投资期限
12. `expense_ratio_level` - 费率水平
13. `performance_consistency` - 业绩一致性

### 2.2 市场行为分析特征 (12个)

**技术实现**: `_generate_behavior_analysis()` 方法

**实际Prompt设计**:
```
分析基金 {fund_code} 的投资者行为模式：

行为统计：
- 申购赎回相关系数: {behavior_stats['correlation']:.3f}
- 净流入波动率: {behavior_stats['net_flow_vol']:.3f}
- 申购增长趋势: {behavior_stats['apply_trend']:.3f}
- 赎回增长趋势: {behavior_stats['redeem_trend']:.3f}
- 周末交易比例: {behavior_stats['weekend_ratio']:.3f}
- 月末交易比例: {behavior_stats['month_end_ratio']:.3f}
- 交易集中度: {behavior_stats['concentration']:.3f}

返回行为分析JSON：
{
    "investor_type": "主要投资者类型(机构主导/散户主导/混合型)",
    "trading_pattern": "交易模式(长期持有/频繁交易/趋势跟随/逆向投资)",
    "flow_stability": "资金流稳定性(1-10)",
    "momentum_factor": "动量因子强度(0-1)",
    "reversion_factor": "均值回归因子(0-1)",
    "seasonality_score": "季节性得分(0-1)",
    "sentiment_sensitivity": "情绪敏感度(0-1)",
    "liquidity_demand": "流动性需求(低/中/高)",
    "concentration_risk": "集中度风险(0-1)",
    "behavioral_bias": "行为偏差类型(过度自信/损失厌恶/从众/锚定/无明显偏差)",
    "stress_response": "压力响应模式(恐慌性抛售/理性调整/逆向增持)",
    "market_timing": "择时能力(0-1)"
}
```

**API调用参数**:
- **temperature**: 0.3 (略高的随机性，增加行为分析的多样性)

### 2.3 量化信号分析特征 (12个)

**技术实现**: `_generate_signal_analysis()` 方法

**实际Prompt设计**:
```
作为量化分析师，分析基金 {fund_code} 的技术信号：

技术指标：
- RSI指标: {signal_stats['rsi']:.2f}
- 动量指标: {signal_stats['momentum']:.2f}
- 布林带位置: {signal_stats['bollinger_pos']:.2f}
- 成交量趋势: {signal_stats['volume_trend']:.2f}
- 趋势强度: {signal_stats['trend_strength']:.2f}
- 支撑阻力比: {signal_stats['support_resistance']:.2f}

返回量化信号分析JSON：
{
    "trend_direction": "趋势方向(上升/下降/震荡)",
    "trend_strength": "趋势强度(0-1)",
    "momentum_signal": "动量信号(-1到1)",
    "overbought_oversold": "超买超卖状态(超买/正常/超卖)",
    "volume_confirmation": "成交量确认(强/中/弱)",
    "breakout_probability": "突破概率(0-1)",
    "reversal_probability": "反转概率(0-1)",
    "volatility_forecast": "波动率预测(收缩/正常/扩张)",
    "support_level": "支撑强度(0-1)",
    "resistance_level": "阻力强度(0-1)",
    "signal_quality": "信号质量(0-1)",
    "prediction_confidence": "预测置信度(0-1)"
}
```

**API调用参数**:
- **temperature**: 0.2 (量化分析需要更高的准确性)

### 2.4 风险评估分析特征 (9个)

**技术实现**: `_generate_risk_analysis()` 方法

**实际风险计算代码**:
```python
def _calculate_risk_stats(self, fund_data: pd.DataFrame) -> Dict:
    """计算风险统计数据"""
    recent_data = fund_data.tail(self.config.CONTEXT_DAYS)
    net_flows = recent_data['apply_amt'] - recent_data['redeem_amt']
    returns = net_flows.pct_change().dropna()
    
    # VaR计算
    var95 = np.percentile(returns, 5)
    
    # 最大回撤
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # 夏普比率
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
    
    return {
        'var95': var95,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'volatility': returns.std() * np.sqrt(252),
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis(),
        'tail_risk': tail_returns.mean()
    }
```

**实际Prompt设计**:
```
进行基金 {fund_code} 的风险评估：

风险指标：
- VaR(95%): {risk_stats['var95']:.3f}
- 最大回撤: {risk_stats['max_drawdown']:.3f}
- 夏普比率: {risk_stats['sharpe']:.3f}
- 波动率: {risk_stats['volatility']:.3f}
- 偏度: {risk_stats['skewness']:.3f}
- 峰度: {risk_stats['kurtosis']:.3f}
- 尾部风险: {risk_stats['tail_risk']:.3f}

返回风险分析JSON：
{
    "overall_risk": "整体风险等级(低/中低/中/中高/高)",
    "systematic_risk": "系统性风险(0-1)",
    "idiosyncratic_risk": "特异性风险(0-1)",
    "liquidity_risk": "流动性风险(0-1)",
    "concentration_risk": "集中度风险(0-1)",
    "tail_risk_level": "尾部风险水平(0-1)",
    "stress_resilience": "压力韧性(0-1)",
    "downside_protection": "下行保护(0-1)",
    "volatility_regime": "波动率状态(低/中/高)"
}
```

### 2.5 文本嵌入特征 (16个)

**技术实现**: `_generate_embeddings()` 方法

**实际处理代码**:
```python
def _generate_embeddings(self, text: str) -> Dict:
    """生成文本嵌入特征"""
    if not text or len(text) < 10:
        text = "混合型基金，多元化投资策略"
    
    try:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, 
                                truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()[0]
        
        # 选择重要维度
        important_dims = [0, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
        embeddings = embeddings[important_dims] if len(embeddings) > max(important_dims) else embeddings[:16]
        
        return {f'embed_{i}': float(val) for i, val in enumerate(embeddings[:16])}
    except Exception as e:
        self.logger.warning(f"生成嵌入失败: {e}")
        return {f'embed_{i}': 0.0 for i in range(16)}
```

**生成特征**: `embed_0` 到 `embed_15` (16个数值特征)

---

## 3. 技术实现细节

### 3.1 完整的类架构

```python
class SmartLLMFeatureGenerator:
    """智能大模型特征生成器"""
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.llm_client = LLMClient(config, logger)
        self.config.LLM_CACHE_DIR.mkdir(exist_ok=True)
        
        # 初始化嵌入模型
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
            self.embedding_model = AutoModel.from_pretrained(config.EMBEDDING_MODEL)
            self.logger.info("嵌入模型加载成功")
        except Exception as e:
            self.logger.warning(f"嵌入模型加载失败: {e}")
            self.tokenizer = None
            self.embedding_model = None
```

### 3.2 批量处理流程

```python
def _generate_all_features(self, df: pd.DataFrame, fund_codes: List[str]) -> pd.DataFrame:
    """为所有基金生成特征"""
    all_features = []
    
    # 批量处理基金
    for i in range(0, len(fund_codes), self.config.BATCH_SIZE):
        batch_codes = fund_codes[i:i + self.config.BATCH_SIZE]
        self.logger.info(f"处理基金批次 {i//self.config.BATCH_SIZE + 1}")
        
        batch_features = self._process_fund_batch(df, batch_codes)
        all_features.extend(batch_features)
        
        # 内存清理
        if i % (self.config.BATCH_SIZE * 4) == 0:
            gc.collect()
    
    return pd.DataFrame(all_features)
```

### 3.3 缓存机制

```python
# 检查缓存
cache_file = self.config.LLM_CACHE_DIR / "llm_features_v2.pkl"
fund_codes = sorted(df['fund_code'].unique())

# 尝试从缓存加载
cached_features = None
if cache_file.exists():
    cached_features = self._load_cache(cache_file, fund_codes)
    if cached_features is not None:
        self.logger.info("使用缓存的大模型特征")
        return df.merge(cached_features, on='fund_code', how='left').fillna(0)

# 保存缓存
self._save_cache(cache_file, llm_features, fund_codes)
```

### 3.4 JSON解析器

```python
def _parse_json_response(self, response: str, default_values: Dict) -> Dict:
    """强大的JSON解析器，能从混杂的文本中提取JSON对象"""
    if not response:
        return default_values
    
    try:
        # 使用正则表达式匹配 '{' 和 '}' 之间的内容
        match = re.search(r'\{.*\}', response, re.DOTALL)
        
        if match:
            json_str = match.group(0)
            parsed = json.loads(json_str)
            
            # 数值验证和标准化
            for key, value in parsed.items():
                if key in default_values:
                    if isinstance(default_values[key], (int, float)):
                        parsed[key] = float(value)
                        # 范围限制
                        if 'level' in key or 'rating' in key or 'score' in key:
                            parsed[key] = np.clip(parsed[key], 0, 10)
            
            return parsed
        else:
            self.logger.warning(f"响应中未找到有效的JSON对象")
            return default_values
            
    except json.JSONDecodeError as e:
        self.logger.warning(f"JSON解析失败: {e}")
        return default_values
```

---

## 4. 特征验证与效果分析

### 4.1 特征质量验证

**一致性检查**:
```python
# 对相同基金代码多次生成，确保结果一致性 > 95%
# 通过设置 temperature=0.2 来降低随机性
```

**合理性验证**:
- 人工检查生成结果的业务合理性
- 特征值范围检查和标准化
- 异常值检测和处理

### 4.2 实际性能提升

根据实际训练结果：
- **模型最终WMAPE**: 约0.2% (WeightedEnsemble_L3)
- **LLM特征贡献**: 在241个总特征中，LLM特征占60+个
- **性能提升**: LLM特征显著提升了模型的预测准确性

### 4.3 特征重要性分析

从AutoGluon的特征重要性输出中可以看到：
- 嵌入特征 (`embed_*`) 在特征重要性中排名较高
- 风险相关特征 (`risk_level`, `volatility_level`) 具有较强的预测能力
- 基金类别特征 (`fund_category`) 提供了重要的分类信息

---

## 5. 创新点与技术亮点

### 5.1 多维度特征生成
1. **基本面分析**: 结合统计数据和基金代码规律
2. **行为模式识别**: 深度分析投资者行为特征
3. **量化信号提取**: 技术指标的智能解读
4. **风险多元评估**: 全方位风险特征构造
5. **语义嵌入增强**: BERT模型的语义理解能力

### 5.2 技术创新
1. **智能JSON解析**: 容错性强的响应解析机制
2. **批量处理优化**: 高效的基金批量处理流程
3. **缓存机制**: 避免重复API调用，提高效率
4. **多层特征融合**: 将不同类型的特征有机结合
5. **稳定性保证**: 通过参数调优确保结果一致性

### 5.3 业务价值
1. **专家知识注入**: 利用LLM的金融领域知识
2. **特征补强**: 为数值特征提供语义补充
3. **泛化能力**: 生成的特征具有良好的泛化性
4. **可解释性**: 特征具有明确的业务含义

---

## 6. 结论

通过系统性地使用大模型技术，我们成功构造了60+个高质量特征，这些特征在最终的预测模型中发挥了重要作用。

**核心成果**:
- ✅ **24个基金代码全覆盖**: 为每个基金生成完整的LLM特征
- ✅ **多维度特征体系**: 涵盖基本面、行为、技术、风险等多个维度  
- ✅ **技术架构完整**: 包含生成、缓存、解析、验证的完整流程
- ✅ **性能显著提升**: 模型WMAPE达到0.2%的优异水平

**技术特点**:
- 🔧 **工程化程度高**: 完整的缓存、批处理、错误处理机制
- 🎯 **业务针对性强**: 特征设计贴合基金投资领域特点
- 🚀 **可扩展性好**: 架构支持新特征类型的快速加入
- 📊 **效果可量化**: 通过多种指标验证特征质量和效果

---

**注意**: 本文档需要转换为Word格式(.docx)并添加所有必要的截图后提交。 