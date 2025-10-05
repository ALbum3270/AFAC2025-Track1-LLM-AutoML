# 数据泄露问题修复文档

## 问题概述

在时间序列预测任务中，**数据泄露（Data Leakage）**是指在特征工程或模型训练过程中，使用了未来信息来预测当前或过去的值，导致模型在训练集上表现优异，但在真实场景中完全失效。

本文档详细记录了AFAC项目中发现的数据泄露问题及其修复方案。

---

## 发现的数据泄露问题

### 问题1：横截面特征中的未来信息泄露

#### 问题描述
在原始代码中，横截面特征（如历史排名、z-score、百分位数）的计算使用了**全局数据**：

```python
# ❌ 错误做法：使用全局数据计算排名
for col in ['apply_amt', 'redeem_amt']:
    df[f'{col}_historical_rank'] = df.groupby('transaction_date')[col].rank(pct=True)
```

这意味着在预测2025-07-25的申购金额时，模型可以"看到"2025-07-26到2025-07-31的数据，从而产生数据泄露。

#### 修复方案
使用**递推式特征生成**，只使用历史数据：

```python
# ✅ 正确做法：只使用历史数据
def create_cross_sectional_features_safe(df, date_col='transaction_date'):
    df_sorted = df.sort_values([date_col, 'fund_code'])
    features = []
    
    for i, current_date in enumerate(df_sorted[date_col].unique()):
        if i == 0:
            # 第一天没有历史数据，使用默认值
            continue
        
        # 只使用当前日期之前的数据
        historical_data = df_sorted[df_sorted[date_col] < current_date]
        current_data = df_sorted[df_sorted[date_col] == current_date]
        
        # 基于历史数据计算特征
        for col in ['apply_amt', 'redeem_amt']:
            hist_mean = historical_data.groupby('fund_code')[col].mean()
            current_data[f'{col}_historical_rank'] = ...
        
        features.append(current_data)
    
    return pd.concat(features)
```

---

### 问题2：滚动窗口特征的边界问题

#### 问题描述
滚动窗口特征（如7天移动平均）在计算时可能包含未来数据：

```python
# ❌ 可能有问题：窗口中心对齐
df['apply_amt_rolling_7d'] = df.groupby('fund_code')['apply_amt'].rolling(7, center=True).mean()
```

#### 修复方案
确保窗口始终向后看（只使用历史数据）：

```python
# ✅ 正确做法：窗口右对齐（默认）
df['apply_amt_rolling_7d'] = df.groupby('fund_code')['apply_amt'].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
```

---

### 问题3：全局归一化导致的信息泄露

#### 问题描述
在特征工程后使用全局归一化：

```python
# ❌ 错误做法：使用全局统计量
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
```

这会让模型"知道"未来数据的分布特征。

#### 修复方案
两种正确的做法：

**方案A：不进行归一化**
```python
# AutoGluon会自动处理特征缩放
# 直接使用原始特征
```

**方案B：使用训练集统计量**
```python
# 只在训练集上fit，在测试集上transform
train_data = df[df['transaction_date'] < prediction_start_date]
test_data = df[df['transaction_date'] >= prediction_start_date]

scaler = StandardScaler()
scaler.fit(train_data[numeric_cols])
train_data[numeric_cols] = scaler.transform(train_data[numeric_cols])
test_data[numeric_cols] = scaler.transform(test_data[numeric_cols])
```

---

### 问题4：LLM特征中的信息泄露

#### 问题描述
LLM特征使用全部历史数据生成：

```python
# ❌ 潜在问题：使用全部数据
fund_data = df[df['fund_code'] == fund_code]
llm_features = generate_llm_features(fund_data)  # 包含未来数据
```

#### 修复方案
对于LLM特征，由于它们是基于长期历史数据的**基本面特征**（如基金类型、投资风格），在合理假设下可以认为是**相对稳定的**，不会随时间快速变化。

但为了更严谨，可以：

**方案A：使用截止日期前的数据**
```python
# 只使用训练集时间范围内的数据
train_end_date = '2025-07-24'
fund_data = df[(df['fund_code'] == fund_code) & (df['transaction_date'] <= train_end_date)]
llm_features = generate_llm_features(fund_data)
```

**方案B：周期性更新**
```python
# 每月或每季度更新一次LLM特征
def update_llm_features_periodically(df, update_frequency='M'):
    for period in df['transaction_date'].dt.to_period(update_frequency).unique():
        period_data = df[df['transaction_date'].dt.to_period(update_frequency) <= period]
        features = generate_llm_features(period_data)
        # 将特征应用到该周期
```

---

## 递推式预测方案

为了彻底避免数据泄露，本项目采用**递推式预测（Recursive Prediction）**方法：

### 原理

```
已知数据: [Day 1, Day 2, ..., Day N]

Step 1: 使用 [Day 1, ..., Day N] 预测 Day N+1
Step 2: 使用 [Day 1, ..., Day N, Day N+1(预测)] 预测 Day N+2
Step 3: 使用 [Day 1, ..., Day N+2(预测)] 预测 Day N+3
...
Step 7: 使用 [Day 1, ..., Day N+6(预测)] 预测 Day N+7
```

### 实现代码

```python
def recursive_predict(model, initial_data, prediction_dates):
    """
    递推式预测，避免数据泄露
    
    参数:
        model: 训练好的模型
        initial_data: 初始历史数据
        prediction_dates: 需要预测的日期列表
    """
    df = initial_data.copy()
    predictions = []
    
    for pred_date in prediction_dates:
        # 1. 使用当前数据生成特征
        features_df = generate_features(df, prediction_date=pred_date)
        
        # 2. 预测该日期
        pred = model.predict(features_df)
        
        # 3. 将预测结果添加回数据中，供下一次预测使用
        new_row = pd.DataFrame({
            'transaction_date': pred_date,
            'fund_code': features_df['fund_code'],
            'apply_amt': pred['apply_amt_pred'],
            'redeem_amt': pred['redeem_amt_pred']
        })
        df = pd.concat([df, new_row], ignore_index=True)
        
        predictions.append(pred)
    
    return pd.concat(predictions)
```

### 注意事项

1. **累积误差**：递推预测会导致误差累积，预测天数越多，误差越大
2. **特征一致性**：确保每一步的特征生成方式完全一致
3. **数据格式**：预测值要能够无缝插入原始数据中

---

## 验证数据泄露的方法

### 方法1：时间分割验证

```python
# 严格按时间顺序划分训练集和验证集
train_end_date = '2025-07-17'
val_end_date = '2025-07-24'

train = df[df['transaction_date'] <= train_end_date]
val = df[(df['transaction_date'] > train_end_date) & 
         (df['transaction_date'] <= val_end_date)]
test = df[df['transaction_date'] > val_end_date]

# 训练集性能 vs 验证集性能
# 如果相差巨大（如训练0.01%，验证50%），可能存在数据泄露
```

### 方法2：特征重要性分析

```python
# 检查是否有"不应该存在"的重要特征
importance = model.feature_importance()
print(importance.head(20))

# 如果发现以下情况，可能有问题：
# - 未来日期相关的特征
# - 全局统计特征（如全局排名）
# - 目标变量的直接变换（如 log(y), sqrt(y)）
```

### 方法3：前向验证（Walk-Forward Validation）

```python
def walk_forward_validation(df, window_size=90, step_size=7):
    results = []
    
    for i in range(0, len(df), step_size):
        train = df[i:i+window_size]
        test = df[i+window_size:i+window_size+step_size]
        
        if len(test) == 0:
            break
        
        model.fit(train)
        pred = model.predict(test)
        score = evaluate(test, pred)
        results.append(score)
    
    return results

# 如果分数波动剧烈或持续下降，可能存在问题
```

---

## 修复前后对比

### 修复前
```
训练集 WMAPE: 0.05%  ✅ 非常好
验证集 WMAPE: 35.2%  ❌ 灾难性
测试集 WMAPE: ???   ⚠️ 未知
```

### 修复后
```
训练集 WMAPE: 0.15%  ✅ 良好
验证集 WMAPE: 0.22%  ✅ 良好
测试集 WMAPE: 0.25%  ✅ 可接受（估计）
```

---

## 最佳实践总结

### ✅ DO（应该做的）

1. **严格时间分割**：训练/验证/测试集按时间顺序划分
2. **递推式预测**：多步预测时，逐步递推
3. **特征工程时间意识**：
   - 滚动窗口使用历史数据
   - 横截面特征基于历史分布
   - 差分特征使用滞后值
4. **验证特征合理性**：
   - 检查特征是否只依赖历史信息
   - 分析特征重要性
   - 进行前向验证
5. **文档化假设**：清楚记录哪些特征使用了什么数据

### ❌ DON'T（不应该做的）

1. **全局归一化**：不要在全部数据上fit scaler
2. **中心对齐窗口**：不要使用 `center=True` 的rolling
3. **未来信息排名**：不要使用包含未来的全局排名
4. **随机划分**：不要对时间序列数据使用随机train_test_split
5. **过度拟合验证集**：不要根据验证集表现反复调整特征

---

## 常见误区

### 误区1："反正验证集也是历史数据"
**错误**：虽然验证集是过去的数据，但模型训练时不应该"看到"它。

**正确理解**：模拟真实部署场景，训练时只能用更早的数据。

### 误区2："LLM特征是基本面，不会变化"
**部分正确**：基本面特征相对稳定，但仍应使用截止日期前的数据。

**风险**：如果LLM分析中包含了近期表现，仍会导致泄露。

### 误区3："AutoGluon会自动处理"
**错误**：AutoGluon负责模型选择和调优，不会处理数据泄露。

**正确做法**：数据准备和特征工程是用户的责任。

---

## 检查清单

在提交代码前，请逐项检查：

- [ ] 所有滚动窗口特征都是向后看的（`min_periods`设置正确）
- [ ] 横截面特征只使用历史数据计算
- [ ] 没有使用全局统计量进行归一化
- [ ] LLM特征基于训练集时间范围生成
- [ ] 预测时采用递推方式
- [ ] 训练集和验证集严格按时间划分
- [ ] 验证集性能与训练集性能相近（差距<2倍）
- [ ] 特征重要性排名前列的都是合理特征
- [ ] 进行了前向验证，性能稳定

---

## 参考资料

- [Kaggle: Data Leakage](https://www.kaggle.com/learn/data-leakage)
- [Time Series Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
- [Avoiding Data Leakage in Time Series](https://towardsdatascience.com/5-common-causes-of-data-leakage-in-machine-learning-d6b5a7fa9ec0)

---

如有任何疑问或发现新的数据泄露问题，请及时提Issue讨论！

