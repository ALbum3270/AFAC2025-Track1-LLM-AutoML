# LLM特征存储优化说明

## 💡 改进亮点

将LLM特征直接存储到 `fund_apply_redeem_series.csv` 文件中，而不是单独的缓存文件。

## 🚀 优势

1. **简化数据流程** - AutoGluon可以直接读取包含所有特征的单一CSV文件
2. **提高训练效率** - 避免复杂的DataFrame merge操作
3. **减少内存使用** - 不需要维护多个数据结构
4. **避免数据不同步** - LLM特征与原始数据保持一致
5. **便于调试分析** - 可以直接在Excel/Pandas中查看所有特征

## 📁 文件结构

```
AFAC/
├── fund_apply_redeem_series.csv      # 主数据文件（包含LLM特征）
├── fund_apply_redeem_series.csv.backup  # 原始数据备份
├── llm_cache/                        # LLM缓存目录（避免重复API调用）
│   └── llm_features_v2.pkl          # API调用结果缓存
├── train.py                          # 主训练脚本
└── metrics.py                        # 评估指标
```

## 🔧 使用方法

### 1. 首次运行
```python
# 配置有效的API密钥
config.API_KEY = "your_api_key_here"
config.SAVE_LLM_TO_CSV = True  # 开启CSV保存

# 运行训练
python train.py
```

### 2. 后续运行
- 程序会自动检测CSV中是否已包含LLM特征
- 如果已存在，直接加载使用
- 如果不存在，会生成并保存到CSV

### 3. 配置选项
```python
class Config:
    SAVE_LLM_TO_CSV = True   # 是否保存到CSV（推荐开启）
    LLM_CACHE_DIR = 'llm_cache'  # 缓存目录
```

## 🎯 LLM特征类型

生成的LLM特征包括：

### 基金基本面分析
- `fund_description` - 基金描述
- `fund_category` - 基金类别
- `fund_style` - 投资风格
- `risk_level` - 风险等级
- `expected_return` - 预期收益率

### 市场行为模式
- `investor_type` - 投资者类型
- `trading_pattern` - 交易模式
- `flow_stability` - 资金流稳定性
- `momentum_factor` - 动量因子

### 量化信号分析
- `trend_direction` - 趋势方向
- `momentum_signal` - 动量信号
- `breakout_probability` - 突破概率

### 风险评估
- `overall_risk` - 整体风险
- `systematic_risk` - 系统性风险
- `stress_resilience` - 压力韧性

### 文本嵌入
- `embed_0` 到 `embed_15` - 16维嵌入向量

## 🛡️ 安全机制

1. **自动备份** - 首次修改时自动备份原始CSV文件
2. **智能检测** - 自动检测是否已包含LLM特征，避免重复生成
3. **错误处理** - LLM特征生成失败不影响原始数据
4. **缓存机制** - 相同基金代码集合复用缓存，节省API调用

## 📊 AutoGluon集成

```python
# AutoGluon可以直接使用包含LLM特征的CSV文件
df = pd.read_csv('fund_apply_redeem_series.csv')

# 特征列自动包含所有LLM特征
feature_cols = [col for col in df.columns 
                if col not in ['transaction_date', 'fund_code', 'apply_amt', 'redeem_amt']]

predictor = TabularPredictor(label='apply_amt')
predictor.fit(df[feature_cols + ['apply_amt']])
```

## 🔍 调试建议

1. **查看特征列**：
   ```python
   llm_cols = [col for col in df.columns if 'fund_' in col or 'embed_' in col]
   print(f"LLM特征数量: {len(llm_cols)}")
   ```

2. **检查数据完整性**：
   ```python
   print(df[llm_cols].isnull().sum())
   ```

3. **查看特征样本**：
   ```python
   print(df[['fund_code', 'fund_category', 'fund_style', 'risk_level']].head())
   ```

这种设计大大简化了数据管理流程，特别适合AutoGluon这样的自动机器学习框架！ 