# 🏆 AFAC - 基金申购赎回智能预测系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![AutoGluon](https://img.shields.io/badge/AutoGluon-1.0+-orange.svg)](https://auto.gluon.ai/)

> 基于AutoGluon自动机器学习和大语言模型(LLM)特征增强的基金申购赎回预测解决方案

## 📖 项目简介

本项目是一个完整的时间序列预测系统，用于预测基金的申购和赎回金额。项目结合了传统量化特征工程、自动机器学习框架和大语言模型增强技术，在基金预测竞赛中取得了优异成绩。

### 🌟 核心特性

- 🤖 **LLM特征增强** - 使用通义千问大模型生成60+个高质量金融特征
- 🚀 **AutoGluon自动建模** - 自动模型选择、超参数优化和模型集成
- 📊 **丰富的特征工程** - 200+个时间序列、横截面和技术指标特征
- 🔄 **递推式预测** - 科学的多日递推预测方法，避免累积误差
- 💾 **智能缓存机制** - LLM特征缓存，节省API调用成本
- 🛡️ **数据泄露修复** - 完整的数据泄露检测和修复方案
- ⚡ **内存优化** - 分阶段特征工程，支持大规模数据处理

## 🏗️ 项目架构

```
AFAC/
├── train.py                    # 主训练脚本
├── predict_only.py             # 仅预测脚本（使用已训练模型）
├── metrics.py                  # 评估指标（WMAPE）
├── fund_apply_redeem_series.csv # 主数据文件
├── README.md                   # 项目说明（本文件）
├── requirements.txt            # Python依赖包
├── .gitignore                  # Git忽略文件配置
├── docs/                       # 文档目录
│   ├── README_LLM_Features.md  # LLM特征说明
│   ├── DATA_LEAKAGE_FIX.md    # 数据泄露修复文档
│   └── llm_record_template.md  # LLM使用记录模板
└── llm_cache/                  # LLM特征缓存（自动生成）
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 推荐内存: 8GB+
- 可选: GPU（用于嵌入模型加速）

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/Album/AFAC.git
cd AFAC
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```
> **💡 提示**: 为了保证100%复现项目运行环境，我们同样提供了一个包含了所有依赖包精确版本的 `requirements-frozen.txt` 文件。如果您在使用 `requirements.txt` 安装时遇到依赖冲突或版本问题，可以尝试使用以下命令进行安装：
> ```bash
> pip install -r requirements-frozen.txt
> ```

3. **配置API密钥（可选）**

如果需要使用LLM特征增强，配置通义千问API密钥：

```bash
# Linux/Mac
export QWEN_API_KEY="your_api_key_here"

# Windows
set QWEN_API_KEY=your_api_key_here
```

或创建 `.env` 文件：
```env
QWEN_API_KEY=your_api_key_here
```

### 使用方法

#### 方式一：完整训练和预测

```bash
python train.py
```

**输出文件**：
- `predict_result.csv` - B榜提交文件
- `train_set.csv` - 训练集文件
- `autogluon_models/` - 训练好的模型
- `fund_prediction_enhanced.log` - 详细日志

#### 方式二：仅使用已训练模型预测

```bash
python predict_only.py
```

需要确保 `autogluon_models/` 目录存在已训练的模型。

## 📊 数据格式

### 输入数据格式

CSV文件应包含以下列：

| 列名 | 说明 | 类型 |
|------|------|------|
| transaction_date | 交易日期（格式：YYYYMMDD） | str |
| fund_code | 基金代码 | str |
| apply_amt | 申购金额（万元） | float |
| redeem_amt | 赎回金额（万元） | float |
| uv_key_page_1 | 关键页面1的UV（可选） | int |
| uv_key_page_2 | 关键页面2的UV（可选） | int |
| uv_key_page_3 | 关键页面3的UV（可选） | int |

### 输出预测格式

| 列名 | 说明 |
|------|------|
| fund_code | 基金代码 |
| transaction_date | 预测日期 |
| apply_amt_pred | 预测申购金额 |
| redeem_amt_pred | 预测赎回金额 |

## 🎨 核心特性详解

### 1. LLM特征工程（60+特征）

使用大语言模型生成多维度金融特征：

#### 📌 基金基本面分析（13个特征）
- 基金类别、投资风格、风险等级
- 预期收益率、波动性水平
- 流动性评级、管理能力评分

#### 📌 市场行为模式（12个特征）
- 投资者类型、交易模式
- 资金流稳定性、动量因子
- 行为偏差识别

#### 📌 量化信号分析（12个特征）
- 趋势方向、动量信号
- 突破/反转概率
- 技术指标解读

#### 📌 风险评估（9个特征）
- 系统性/特异性风险
- 压力韧性、尾部风险
- 危机表现评估

#### 📌 文本嵌入（16个特征）
- BERT语义向量特征（embed_0 ~ embed_15）

### 2. 传统特征工程（200+特征）

- **日期特征**：月份、星期、节假日、季节性编码
- **时间序列特征**：
  - 滞后特征（1, 3, 7, 14天）
  - 滚动窗口统计（7, 14, 30天）
  - 差分和百分比变化
- **横截面特征**：历史排名、z-score、百分位数
- **交互特征**：申购赎回比率、净流入、UV转化率
- **技术指标**：RSI、MACD、布林带、EMA

### 3. 模型配置

```python
# 默认配置
MODEL_PRESET = 'best_quality'  # 高质量模式
TIME_LIMIT = 7200              # 2小时训练
PREDICTION_DAYS = 7            # 预测7天
VALIDATION_RATIO = 0.2         # 验证集比例
```

可以在 `train.py` 中的 `Config` 类修改配置。

## 📈 性能评估

项目使用**WMAPE（加权平均绝对百分比误差）**作为评估指标：

```
WMAPE = 0.5 × WMAPE_申购 + 0.5 × WMAPE_赎回
```

典型性能：
- **训练集WMAPE**: ~0.15%
- **验证集WMAPE**: ~0.20%
- **最佳模型**: WeightedEnsemble_L3

## 🔧 高级配置

### 自定义配置

编辑 `train.py` 中的 `Config` 类：

```python
class Config:
    # 数据路径
    DATA_PATH = 'fund_apply_redeem_series.csv'
    
    # LLM配置
    API_KEY = os.getenv("QWEN_API_KEY", "")
    SAVE_LLM_TO_CSV = True  # 是否保存LLM特征到CSV
    
    # 特征工程参数
    LAG_PERIODS = [1, 3, 7, 14]
    ROLLING_WINDOWS = [7, 14, 30]
    
    # 训练参数
    MODEL_PRESET = 'best_quality'
    TIME_LIMIT = 7200  # 训练时间限制（秒）
    
    # 预测参数
    PREDICTION_DAYS = 7
    PREDICTION_START_DATE = '2025-07-25'
```

### 内存优化

如果遇到内存问题，可以调整以下参数：

```python
MAX_MEMORY_MB = 4000           # 最大内存使用
MAX_FEATURE_COUNT = 500        # 最大特征数量
BATCH_SIZE = 5                 # LLM批处理大小
```

## 📚 文档

- [LLM特征详细说明](docs/README_LLM_Features.md)
- [数据泄露修复文档](docs/DATA_LEAKAGE_FIX.md)
- [LLM使用记录模板](docs/llm_record_template.md)

## 🛠️ 技术栈

- **机器学习框架**: [AutoGluon](https://auto.gluon.ai/) 1.0+
- **大语言模型**: 通义千问 (Qwen-turbo)
- **嵌入模型**: Hugging Face Transformers
- **数据处理**: Pandas, NumPy
- **技术指标**: TA-Lib风格指标
- **日志**: Python logging

## 🤝 贡献

欢迎提交Issue和Pull Request！

贡献指南：
1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📝 更新日志

### v1.0.0 (2025-01-XX)
- ✨ 初始版本发布
- 🤖 集成LLM特征生成
- 🚀 AutoGluon自动建模
- 🔄 递推式预测实现
- 🛡️ 数据泄露修复
- ⚡ 内存优化

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

## 👥 作者

- **Album** - [GitHub](https://github.com/Album)

## 🙏 致谢

- 感谢AutoGluon团队提供的优秀自动机器学习框架
- 感谢阿里云通义千问团队提供的大模型API
- 感谢开源社区的各类优秀工具和库

## 📧 联系方式

- Email: album3270@gmail.com
- GitHub Issues: [提交问题](https://github.com/Album/AFAC/issues)

---

⭐ 如果这个项目对你有帮助，请给个Star支持一下！

