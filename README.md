# 🏆 AFAC2025 Track1 — 基金申购赎回智能预测系统

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
 ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
 [![AutoGluon](https://img.shields.io/badge/AutoGluon-1.0+-orange.svg)](https://auto.gluon.ai/)
 [![LLM: Qwen](https://img.shields.io/badge/LLM-Qwen--Turbo-green.svg)](https://tongyi.aliyun.com/qianwen/)

> 🧠 **LLM + AutoML 双引擎驱动的基金申购赎回预测系统**
>  面向 AFAC2025 挑战组·赛题一 —— 基金产品长周期申购与赎回预测

------

## 📖 项目简介

本项目是一个基于 **AutoGluon 自动机器学习框架** 与 **通义千问大模型 (Qwen)** 的智能时间序列预测系统。
 系统面向 **基金产品的长周期申购与赎回预测** 场景，融合传统量化特征、自然语言生成特征与自动建模技术，在 AFAC2025 官方赛题中取得优异表现。

------

## 🌟 核心亮点

| 模块                     | 描述                                     |
| ------------------------ | ---------------------------------------- |
| 🤖 **LLM 特征增强**       | 利用通义千问生成 60+ 高质量金融语义特征  |
| 🚀 **AutoGluon 自动建模** | 自动模型选择、超参优化与集成融合         |
| 📊 **丰富特征工程**       | 构造 200+ 时间序列、横截面与技术指标特征 |
| 🔄 **多日递推预测**       | 递推式时间窗预测，避免误差累积           |
| 💾 **智能缓存机制**       | LLM 特征结果自动缓存，节省 API 成本      |
| 🛡️ **数据泄露检测与修复** | 内置完整防泄露机制                       |
| ⚡ **内存优化设计**       | 分阶段特征工程支持大规模数据集           |

------

## 🏗️ 项目结构

```
AFAC2025-Track1-LLM-AutoML/
├── train.py                    # 主训练入口
├── predict_only.py             # 仅预测模式（复用已训练模型）
├── metrics.py                  # 模型评估指标（WMAPE）
├── fund_apply_redeem_series.csv # 原始数据
├── requirements.txt            # 环境依赖
├── docs/
│   ├── README_LLM_Features.md  # LLM 特征说明
│   ├── DATA_LEAKAGE_FIX.md     # 数据泄露修复方案
│   └── llm_record_template.md  # LLM 使用记录模板
└── llm_cache/                  # LLM 特征缓存（自动生成）
```

------

## 🚀 快速开始

### 🔧 环境准备

- Python ≥ 3.8
- 推荐内存 ≥ 8GB
- 可选 GPU（用于嵌入模型加速）

```bash
git clone https://github.com/ALbum3270/AFAC2025-Track1-LLM-AutoML.git
cd AFAC2025-Track1-LLM-AutoML
pip install -r requirements.txt
```

如遇依赖冲突，可使用：

```bash
pip install -r requirements-frozen.txt
```

------

### 🔑 配置 API 密钥（可选）

```bash
# Linux / Mac
export QWEN_API_KEY="your_api_key_here"

# Windows
set QWEN_API_KEY=your_api_key_here
```

或在项目根目录创建 `.env`：

```
QWEN_API_KEY=your_api_key_here
```

------

### 🧩 使用方法

#### **方式一：完整训练 + 预测**

```bash
python train.py
```

输出结果：

- `predict_result.csv`：B榜提交文件
- `autogluon_models/`：训练好的模型
- `fund_prediction_enhanced.log`：详细日志

#### **方式二：仅使用已训练模型**

```bash
python predict_only.py
```

需保证 `autogluon_models/` 目录存在模型。

------

## 📊 数据格式

| 列名             | 说明                | 类型  |
| ---------------- | ------------------- | ----- |
| transaction_date | 交易日期 (YYYYMMDD) | str   |
| fund_code        | 基金代码            | str   |
| apply_amt        | 申购金额（万元）    | float |
| redeem_amt       | 赎回金额（万元）    | float |
| uv_key_page_1..3 | 核心页面访问 UV     | int   |

### 输出预测格式

| 列名             | 说明         |
| ---------------- | ------------ |
| fund_code        | 基金代码     |
| transaction_date | 日期         |
| apply_amt_pred   | 预测申购金额 |
| redeem_amt_pred  | 预测赎回金额 |

------

## 🎨 特征工程详解

### 🧠 LLM 特征（60+）

包括基金特性、投资行为、量化信号、风险评估与文本嵌入（BERT 特征）。

### 📈 传统特征（200+）

- **时间序列特征**：滞后、滚动统计、差分变化
- **横截面特征**：排名、z-score、百分位
- **交互特征**：申赎比、净流入、转化率
- **技术指标**：RSI、MACD、布林带、EMA

------

## ⚙️ 关键配置（`Config`）

```python
class Config:
    DATA_PATH = 'fund_apply_redeem_series.csv'
    MODEL_PRESET = 'best_quality'
    TIME_LIMIT = 7200  # 秒
    PREDICTION_DAYS = 7
    PREDICTION_START_DATE = '2025-07-25'
```

------

## 📈 模型性能

评估指标：**WMAPE（加权平均绝对百分比误差）**

```
WMAPE = 0.5 × WMAPE_申购 + 0.5 × WMAPE_赎回
```

| 数据集   | WMAPE               |
| -------- | ------------------- |
| 训练集   | ≈ 0.15%             |
| 验证集   | ≈ 0.20%             |
| 最优模型 | WeightedEnsemble_L3 |

------

## 🛠️ 技术栈

| 类别       | 工具                                |
| ---------- | ----------------------------------- |
| AutoML     | [AutoGluon](https://auto.gluon.ai/) |
| 大语言模型 | 通义千问 (Qwen-turbo)               |
| 嵌入       | Hugging Face Transformers           |
| 数据处理   | Pandas, NumPy                       |
| 技术指标   | TA-Lib 风格指标                     |
| 日志       | Python logging                      |

------

## 📚 文档

- LLM 特征说明
- 数据泄露修复方案
- LLM 调用记录模板

------

## 🧩 更新日志

### v1.0.0 (2025-01-XX)

- 初始版本发布
- 集成 LLM 特征生成
- AutoGluon 自动建模
- 多日递推预测
- 数据泄露检测与修复
- 内存优化模块

------

## 👥 作者

**Album**
 📧 Email: [album3270@gmail.com](mailto:album3270@gmail.com)
 🌐 GitHub: [ALbum3270](https://github.com/ALbum3270)

------

## 🙏 致谢

- 感谢 AutoGluon 团队提供卓越的 AutoML 框架
- 感谢通义千问 (Qwen) 团队开放大模型 API
- 感谢开源社区所有贡献者

------

⭐ **如果这个项目对你有帮助，请点亮 Star 支持！**
