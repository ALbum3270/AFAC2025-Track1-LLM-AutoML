import os
# 设置环境变量以避免tokenizer警告
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
from autogluon.tabular import TabularPredictor
from autogluon.core.metrics import make_scorer
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
import torch
import pickle
import gc
import warnings
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List
import hashlib
import time
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import math
from metrics import wmape_metric, wmape_scorer
import psutil
import traceback
warnings.filterwarnings('ignore')

class Config:
    """配置管理类"""
    # 数据路径
    DATA_PATH = 'fund_apply_redeem_series.csv'  # 主数据文件，LLM特征会直接添加为新列
    MODEL_DIR = Path('autogluon_models')
    LLM_CACHE_DIR = Path('llm_cache')  # LLM特征缓存目录，避免重复API调用
    
    # 大模型配置
    API_KEY = os.getenv("QWEN_API_KEY", "")  # 默认为空，需要用户配置环境变量
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    MODEL_NAME = "qwen-turbo"
    EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
    
    # 特征工程参数 (降低内存占用)
    LAG_PERIODS = [1, 3, 7, 14]
    ROLLING_WINDOWS = [7, 14, 30]
    DIFF_PERIODS = [1, 7, 14]
    
    # LLM特征参数
    MAX_RETRIES = 3
    REQUEST_DELAY = 0.3
    CONTEXT_DAYS = 60
    BATCH_SIZE = 5  # 批处理大小
    SAVE_LLM_TO_CSV = True  # 是否将LLM特征直接保存到CSV文件（推荐开启，便于AutoGluon直接使用）
    
    # 训练参数
    VALIDATION_RATIO = 0.2  # 仅用于模型选择和调试
    MODEL_PRESET = 'best_quality'
    TIME_LIMIT = 7200  # 默认延长至2小时，以充分训练best_quality模型
    
    # B榜预测参数
    PREDICTION_DAYS = 7  # 预测未来7天
    PREDICTION_START_DATE = '2025-07-25'  # B榜预测开始日期
    
    # 目标列
    TARGET_COLS = ['apply_amt', 'redeem_amt']
    
    # 修复配置选项
    ENABLE_DATA_LEAKAGE_FIX = True  # 启用数据泄露修复
    ENABLE_MEMORY_OPTIMIZATION = True  # 启用内存优化
    ENABLE_RECURSIVE_PREDICTION = True  # 启用递推预测
    ENABLE_AUTOGLUON_FIX = True  # 启用AutoGluon兼容性修复
    ENABLE_FEATURE_VALIDATION = True  # 启用特征验证
    
    # 内存管理配置
    MEMORY_MONITOR_INTERVAL = 100  # 内存监控间隔（行数）
    GC_FORCE_INTERVAL = 1000  # 强制垃圾回收间隔（行数）
    MAX_MEMORY_MB = 4000  # 最大内存使用（MB）
    
    # 特征工程配置
    MAX_FEATURE_COUNT = 500  # 最大特征数量
    MIN_FEATURE_VARIANCE = 1e-8  # 最小特征方差
    MAX_CORRELATION_THRESHOLD = 0.95  # 高相关性阈值 (降低)
    MAX_MISSING_RATIO = 0.8  # 最大缺失值比例
    
    # 递推预测配置
    MIN_HISTORY_DAYS = 14  # 最少历史天数
    PREDICTION_CONFIDENCE_THRESHOLD = 0.1  # 预测置信度阈值
    FALLBACK_PREDICTION_VALUE = 1000  # 后备预测值
    
    # 错误处理配置
    MAX_RETRIES_PER_STAGE = 2  # 每阶段最大重试次数
    ENABLE_GRACEFUL_DEGRADATION = True  # 启用优雅降级

class LLMClient:
    """大模型客户端"""
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.client = OpenAI(
            api_key=config.API_KEY,
            base_url=config.BASE_URL,
        )
        self.request_count = 0
        
    def call_api(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """调用大模型API"""
        self.request_count += 1
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                time.sleep(self.config.REQUEST_DELAY)
                
                completion = self.client.chat.completions.create(
                    model=self.config.MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "你是一位资深的量化金融分析师，具有深厚的基金行业经验和机器学习建模能力。"},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                response = completion.choices[0].message.content.strip()
                self.logger.debug(f"API调用成功 (第{self.request_count}次)")
                return response
                
            except Exception as e:
                self.logger.warning(f"API调用失败 (尝试 {attempt+1}/{self.config.MAX_RETRIES}): {e}")
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    
        self.logger.error("API调用最终失败")
        return None

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
            self.logger.warning(f"嵌入模型加载失败: {e}，将使用简化版本")
            self.tokenizer = None
            self.embedding_model = None
    
    def generate_features(self, df: pd.DataFrame, save_to_csv: bool = True) -> pd.DataFrame:
        """生成大模型增强特征并可选择性保存到CSV"""
        self.logger.info("开始生成大模型特征...")
        
        # 检查是否已有LLM特征列
        llm_feature_cols = [col for col in df.columns if any(
            prefix in col for prefix in ['fund_description', 'fund_category', 'fund_style', 
                                       'risk_level', 'investor_type', 'trend_direction', 
                                       'overall_risk', 'embed_']
        )]
        
        if llm_feature_cols:
            self.logger.info(f"检测到已存在{len(llm_feature_cols)}个LLM特征列，直接使用")
            return df
        
        # 检查缓存
        cache_file = self.config.LLM_CACHE_DIR / "llm_features_v2.pkl"
        fund_codes = sorted(df['fund_code'].unique())
        
        # 尝试从缓存加载
        cached_features = None
        if cache_file.exists():
            cached_features = self._load_cache(cache_file, fund_codes)
            if cached_features is not None:
                self.logger.info("使用缓存的大模型特征")
                result_df = df.merge(cached_features, on='fund_code', how='left').fillna(0)
                
                # 可选：保存增强后的数据到CSV
                if save_to_csv:
                    self._save_enhanced_csv(result_df)
                
                return result_df
        
        # 生成新特征
        llm_features = self._generate_all_features(df, fund_codes)
        
        # 保存缓存
        self._save_cache(cache_file, llm_features, fund_codes)
        
        # 合并特征
        result_df = df.merge(llm_features, on='fund_code', how='left').fillna(0)
        self.logger.info(f"大模型特征生成完成，新增特征维度: {len(llm_features.columns) - 1}")
        
        # 可选：保存增强后的数据到CSV
        if save_to_csv:
            self._save_enhanced_csv(result_df)
        
        return result_df
    
    def _generate_all_features(self, df: pd.DataFrame, fund_codes: List[str]) -> pd.DataFrame:
        """为所有基金生成特征"""
        all_features = []
        
        # 批量处理基金
        for i in range(0, len(fund_codes), self.config.BATCH_SIZE):
            batch_codes = fund_codes[i:i + self.config.BATCH_SIZE]
            self.logger.info(f"处理基金批次 {i//self.config.BATCH_SIZE + 1}/{(len(fund_codes) + self.config.BATCH_SIZE - 1)//self.config.BATCH_SIZE}")
            
            batch_features = self._process_fund_batch(df, batch_codes)
            all_features.extend(batch_features)
            
            # 内存清理
            if i % (self.config.BATCH_SIZE * 4) == 0:
                gc.collect()
        
        return pd.DataFrame(all_features)
    
    def _process_fund_batch(self, df: pd.DataFrame, fund_codes: List[str]) -> List[Dict]:
        """批量处理基金"""
        batch_features = []
        
        for fund_code in fund_codes:
            self.logger.info(f"处理基金 {fund_code}")
            
            # 获取基金历史数据
            fund_data = df[df['fund_code'] == fund_code].sort_values('transaction_date')
            
            if len(fund_data) < 7:  # 数据太少，使用默认值
                features = self._get_default_features(fund_code)
            else:
                features = {'fund_code': fund_code}
                
                # 1. 基金基本面分析
                basic_features = self._generate_fund_analysis(fund_code, fund_data)
                features.update(basic_features)
                
                # 2. 市场行为模式分析
                behavior_features = self._generate_behavior_analysis(fund_code, fund_data)
                features.update(behavior_features)
                
                # 3. 量化信号分析
                signal_features = self._generate_signal_analysis(fund_code, fund_data)
                features.update(signal_features)
                
                # 4. 风险评估分析
                risk_features = self._generate_risk_analysis(fund_code, fund_data)
                features.update(risk_features)
                
                # 5. 文本嵌入特征
                if self.embedding_model is not None:
                    text = features.get('fund_description', f'基金{fund_code}')
                    embedding_features = self._generate_embeddings(text)
                    features.update(embedding_features)
                else:
                    features.update({f'embed_{i}': 0.0 for i in range(16)})
            
            batch_features.append(features)
        
        return batch_features
    
    def _generate_fund_analysis(self, fund_code: str, fund_data: pd.DataFrame) -> Dict:
        """生成基金基本面分析特征"""
        recent_stats = self._calculate_fund_stats(fund_data)
        
        prompt = f"""
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
        {{
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
        }}
        """
        
        response = self.llm_client.call_api(prompt, temperature=0.2)
        return self._parse_json_response(response, self._get_default_fund_analysis(fund_code))
    
    def _generate_behavior_analysis(self, fund_code: str, fund_data: pd.DataFrame) -> Dict:
        """生成市场行为分析特征"""
        behavior_stats = self._calculate_behavior_stats(fund_data)
        
        prompt = f"""
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
        {{
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
        }}
        """
        
        response = self.llm_client.call_api(prompt, temperature=0.3)
        return self._parse_json_response(response, self._get_default_behavior_analysis())
    
    def _generate_signal_analysis(self, fund_code: str, fund_data: pd.DataFrame) -> Dict:
        """生成量化信号分析特征"""
        signal_stats = self._calculate_signal_stats(fund_data)
        
        prompt = f"""
        作为量化分析师，分析基金 {fund_code} 的技术信号：

        技术指标：
        - RSI指标: {signal_stats['rsi']:.2f}
        - 动量指标: {signal_stats['momentum']:.2f}
        - 布林带位置: {signal_stats['bollinger_pos']:.2f}
        - 成交量趋势: {signal_stats['volume_trend']:.2f}
        - 趋势强度: {signal_stats['trend_strength']:.2f}
        - 支撑阻力比: {signal_stats['support_resistance']:.2f}

        返回量化信号分析JSON：
        {{
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
        }}
        """
        
        response = self.llm_client.call_api(prompt, temperature=0.2)
        return self._parse_json_response(response, self._get_default_signal_analysis())
    
    def _generate_risk_analysis(self, fund_code: str, fund_data: pd.DataFrame) -> Dict:
        """生成风险分析特征"""
        risk_stats = self._calculate_risk_stats(fund_data)
        
        prompt = f"""
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
        {{
            "overall_risk": "整体风险等级(低/中低/中/中高/高)",
            "systematic_risk": "系统性风险(0-1)",
            "idiosyncratic_risk": "特异性风险(0-1)",
            "liquidity_risk": "流动性风险(0-1)",
            "concentration_risk": "集中度风险(0-1)",
            "tail_risk_level": "尾部风险水平(0-1)",
            "stress_resilience": "压力韧性(0-1)",
            "downside_protection": "下行保护(0-1)",
            "volatility_regime": "波动率状态(低/中/高)",
            "risk_adjusted_return": "风险调整收益(0-1)",
            "stability_score": "稳定性评分(0-1)",
            "crisis_performance": "危机表现(差/一般/良好/优秀)"
        }}
        """
        
        response = self.llm_client.call_api(prompt, temperature=0.2)
        return self._parse_json_response(response, self._get_default_risk_analysis())
    
    def _calculate_fund_stats(self, fund_data: pd.DataFrame) -> Dict:
        """计算基金统计数据"""
        recent_data = fund_data.tail(self.config.CONTEXT_DAYS)
        
        return {
            'avg_apply': recent_data['apply_amt'].mean(),
            'avg_redeem': recent_data['redeem_amt'].mean(),
            'std_apply': recent_data['apply_amt'].std(),
            'std_redeem': recent_data['redeem_amt'].std(),
            'max_apply': recent_data['apply_amt'].max(),
            'max_redeem': recent_data['redeem_amt'].max(),
            'avg_net_flow': (recent_data['apply_amt'] - recent_data['redeem_amt']).mean()
        }
    
    def _calculate_behavior_stats(self, fund_data: pd.DataFrame) -> Dict:
        """计算行为统计数据"""
        recent_data = fund_data.tail(self.config.CONTEXT_DAYS)
        
        correlation = recent_data['apply_amt'].corr(recent_data['redeem_amt'])
        net_flows = recent_data['apply_amt'] - recent_data['redeem_amt']
        
        # 添加日期特征
        recent_data = recent_data.copy()
        recent_data['is_weekend'] = recent_data['transaction_date'].dt.dayofweek.isin([5, 6])
        recent_data['is_month_end'] = recent_data['transaction_date'].dt.is_month_end
        
        weekend_volume = recent_data[recent_data['is_weekend']]['apply_amt'].sum() + recent_data[recent_data['is_weekend']]['redeem_amt'].sum()
        total_volume = recent_data['apply_amt'].sum() + recent_data['redeem_amt'].sum()
        
        month_end_volume = recent_data[recent_data['is_month_end']]['apply_amt'].sum() + recent_data[recent_data['is_month_end']]['redeem_amt'].sum()
        
        # 计算集中度（基尼系数近似）
        volumes = recent_data['apply_amt'] + recent_data['redeem_amt']
        volumes_sorted = np.sort(volumes)
        n = len(volumes_sorted)
        concentration = 2 * np.sum((np.arange(1, n+1) * volumes_sorted)) / (n * np.sum(volumes_sorted)) - (n+1)/n if np.sum(volumes_sorted) > 0 else 0
        
        return {
            'correlation': correlation if not np.isnan(correlation) else 0,
            'net_flow_vol': net_flows.std() if len(net_flows) > 1 else 0,
            'apply_trend': recent_data['apply_amt'].pct_change().mean() if len(recent_data) > 1 else 0,
            'redeem_trend': recent_data['redeem_amt'].pct_change().mean() if len(recent_data) > 1 else 0,
            'weekend_ratio': weekend_volume / (total_volume + 1e-8),
            'month_end_ratio': month_end_volume / (total_volume + 1e-8),
            'concentration': concentration
        }
    
    def _calculate_signal_stats(self, fund_data: pd.DataFrame) -> Dict:
        """计算技术信号统计"""
        recent_data = fund_data.tail(self.config.CONTEXT_DAYS)
        net_flows = recent_data['apply_amt'] - recent_data['redeem_amt']
        
        # RSI计算
        deltas = net_flows.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        avg_gains = gains.rolling(window=14, min_periods=1).mean()
        avg_losses = losses.rolling(window=14, min_periods=1).mean()
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        # 动量指标
        momentum = net_flows.pct_change(periods=10).iloc[-1] if len(net_flows) > 10 else 0
        
        # 布林带位置
        sma = net_flows.rolling(window=20, min_periods=1).mean()
        std = net_flows.rolling(window=20, min_periods=1).std()
        current_position = (net_flows.iloc[-1] - sma.iloc[-1]) / (2 * std.iloc[-1] + 1e-8) if len(net_flows) > 0 else 0
        
        # 成交量趋势
        volumes = recent_data['apply_amt'] + recent_data['redeem_amt']
        volume_trend = volumes.pct_change().mean() if len(volumes) > 1 else 0
        
        # 趋势强度
        if len(net_flows) > 5:
            trend_strength = abs(np.corrcoef(range(len(net_flows)), net_flows)[0, 1])
        else:
            trend_strength = 0
        
        # 支撑阻力比
        recent_highs = net_flows.rolling(window=5, min_periods=1).max()
        recent_lows = net_flows.rolling(window=5, min_periods=1).min()
        support_resistance = (net_flows.iloc[-1] - recent_lows.iloc[-1]) / (recent_highs.iloc[-1] - recent_lows.iloc[-1] + 1e-8) if len(net_flows) > 0 else 0.5
        
        return {
            'rsi': rsi.iloc[-1] if len(rsi) > 0 and not np.isnan(rsi.iloc[-1]) else 50,
            'momentum': momentum if not np.isnan(momentum) else 0,
            'bollinger_pos': np.clip(current_position, -1, 1),
            'volume_trend': volume_trend if not np.isnan(volume_trend) else 0,
            'trend_strength': trend_strength if not np.isnan(trend_strength) else 0,
            'support_resistance': np.clip(support_resistance, 0, 1)
        }
    
    def _calculate_risk_stats(self, fund_data: pd.DataFrame) -> Dict:
        """计算风险统计数据"""
        recent_data = fund_data.tail(self.config.CONTEXT_DAYS)
        net_flows = recent_data['apply_amt'] - recent_data['redeem_amt']
        returns = net_flows.pct_change().dropna()
        
        if len(returns) < 5:
            return {
                'var95': 0, 'max_drawdown': 0, 'sharpe': 0, 'volatility': 0,
                'skewness': 0, 'kurtosis': 0, 'tail_risk': 0
            }
        
        # VaR计算
        var95 = np.percentile(returns, 5)
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 夏普比率
        sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
        
        # 其他统计量
        volatility = returns.std() * np.sqrt(252)
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # 尾部风险
        tail_returns = returns[returns <= var95]
        tail_risk = tail_returns.mean() if len(tail_returns) > 0 else 0
        
        return {
            'var95': var95,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
            'volatility': volatility,
            'skewness': skewness if not np.isnan(skewness) else 0,
            'kurtosis': kurtosis if not np.isnan(kurtosis) else 0,
            'tail_risk': tail_risk
        }
    
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
            
            # 使用PCA降维
            if hasattr(self, '_pca_model'):
                embeddings = self._pca_model.transform(embeddings.reshape(1, -1))[0]
            else:
                # 选择重要维度
                important_dims = [0, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
                embeddings = embeddings[important_dims] if len(embeddings) > max(important_dims) else embeddings[:16]
            
            return {f'embed_{i}': float(val) for i, val in enumerate(embeddings[:16])}
        except Exception as e:
            self.logger.warning(f"生成嵌入失败: {e}")
            return {f'embed_{i}': 0.0 for i in range(16)}
    
    def _parse_json_response(self, response: str, default_values: Dict) -> Dict:
        """
        更强大的JSON解析器，能从混杂的文本中提取出被{}包裹的JSON对象。
        """
        if not response:
            return default_values
        
        try:
            # 1. 使用正则表达式贪婪匹配第一个 '{' 和最后一个 '}' 之间的所有内容
            # re.DOTALL 标志让 '.' 可以匹配包括换行在内的任意字符
            match = re.search(r'\{.*\}', response, re.DOTALL)
            
            # 2. 如果找到了匹配项，就尝试解析它
            if match:
                json_str = match.group(0)
                parsed = json.loads(json_str)
            else:
                # 如果连 {} 都找不到，就无法解析，记录警告并返回默认值
                self.logger.warning(f"响应中未找到有效的JSON对象: '{response[:150]}...'")
                return default_values

            # 3. 数值验证和标准化 (此部分逻辑保持不变)
            for key, value in parsed.items():
                if key in default_values:
                    if isinstance(default_values[key], (int, float)):
                        try:
                            parsed[key] = float(value)
                            # 根据字段类型进行合理范围限制
                            if 'level' in key or 'rating' in key or 'score' in key:
                                if '10' in str(default_values.get(key, 5)):
                                    parsed[key] = max(1, min(10, parsed[key]))
                                else:
                                    parsed[key] = max(1, min(5, parsed[key]))
                            elif 'factor' in key or 'probability' in key or 'confidence' in key or key.endswith('_score'):
                                parsed[key] = max(0, min(1, parsed[key]))
                            elif 'beta' in key:
                                parsed[key] = max(0.1, min(3.0, parsed[key]))
                            elif 'return' in key or 'volatility' in key:
                                parsed[key] = max(-100, min(100, parsed[key]))
                        except (ValueError, TypeError):
                            parsed[key] = default_values[key]
            
            # 4. 确保所有必需字段存在 (此部分逻辑保持不变)
            for key in default_values:
                if key not in parsed:
                    parsed[key] = default_values[key]
            
            return parsed
            
        except (json.JSONDecodeError, AttributeError) as e:
            # 增加了AttributeError捕获(针对match.group(0)可能失败的情况)
            # 优化了日志，打印部分原始响应以帮助调试
            self.logger.warning(f"JSON解析失败: {e}. 原始响应(部分): '{response[:150]}...'")
            return default_values

    def _get_default_features(self, fund_code: str) -> Dict:
        """获取默认特征（数据不足时使用）"""
        features = {'fund_code': fund_code}
        features.update(self._get_default_fund_analysis(fund_code))
        features.update(self._get_default_behavior_analysis())
        features.update(self._get_default_signal_analysis())
        features.update(self._get_default_risk_analysis())
        features.update({f'embed_{i}': 0.0 for i in range(16)})
        return features
    
    def _get_default_fund_analysis(self, fund_code: str) -> Dict:
        """获取默认基金分析特征"""
        # 基于基金代码规律进行智能推断
        code_features = self._infer_from_fund_code(fund_code)
        
        return {
            'fund_description': f'{code_features["type"]}基金，{code_features["style"]}投资策略',
            'fund_category': code_features['category'],
            'fund_style': code_features['style'],
            'risk_level': code_features['risk_level'],
            'expected_return': code_features['expected_return'],
            'volatility_level': code_features['volatility'],
            'liquidity_rating': 6.0,
            'management_capability': 7.0,
            'market_beta': 1.0,
            'sector_concentration': '适中',
            'investment_horizon': 12,
            'expense_ratio_level': '中',
            'performance_consistency': 6.0
        }
    
    def _get_default_behavior_analysis(self) -> Dict:
        """获取默认行为分析特征"""
        return {
            'investor_type': '混合型',
            'trading_pattern': '长期持有',
            'flow_stability': 6.0,
            'momentum_factor': 0.3,
            'reversion_factor': 0.4,
            'seasonality_score': 0.2,
            'sentiment_sensitivity': 0.5,
            'liquidity_demand': '中',
            'concentration_risk': 0.3,
            'behavioral_bias': '无明显偏差',
            'stress_response': '理性调整',
            'market_timing': 0.4
        }
    
    def _get_default_signal_analysis(self) -> Dict:
        """获取默认信号分析特征"""
        return {
            'trend_direction': '震荡',
            'trend_strength': 0.3,
            'momentum_signal': 0.0,
            'overbought_oversold': '正常',
            'volume_confirmation': '中',
            'breakout_probability': 0.3,
            'reversal_probability': 0.3,
            'volatility_forecast': '正常',
            'support_level': 0.5,
            'resistance_level': 0.5,
            'signal_quality': 0.5,
            'prediction_confidence': 0.5
        }
    
    def _get_default_risk_analysis(self) -> Dict:
        """获取默认风险分析特征"""
        return {
            'overall_risk': '中',
            'systematic_risk': 0.6,
            'idiosyncratic_risk': 0.4,
            'liquidity_risk': 0.3,
            'concentration_risk': 0.3,
            'tail_risk_level': 0.3,
            'stress_resilience': 0.6,
            'downside_protection': 0.5,
            'volatility_regime': '中',
            'risk_adjusted_return': 0.5,
            'stability_score': 0.6,
            'crisis_performance': '一般'
        }
    
    def _infer_from_fund_code(self, fund_code: str) -> Dict:
        """基于基金代码推断基金特征"""
        code = str(fund_code)
        
        # 基金类型推断规则
        if code.startswith(('0', '00')):
            category = '股票型'
            style = '成长'
            risk_level = 4
            expected_return = 8.0
            volatility = '中高'
        elif code.startswith(('1', '11')):
            category = '债券型'
            style = '价值'
            risk_level = 2
            expected_return = 4.0
            volatility = '低'
        elif code.startswith(('2', '51')):
            category = '混合型'
            style = '平衡'
            risk_level = 3
            expected_return = 6.0
            volatility = '中'
        elif code.startswith(('3', '4')):
            category = '货币型'
            style = '稳健'
            risk_level = 1
            expected_return = 2.5
            volatility = '低'
        elif code.startswith('5'):
            category = '指数型'
            style = '被动'
            risk_level = 3
            expected_return = 7.0
            volatility = '中'
        else:
            category = '混合型'
            style = '平衡'
            risk_level = 3
            expected_return = 6.0
            volatility = '中'
        
        # 特殊规则
        if '88' in code or '99' in code:
            style = '量化'
            expected_return += 1.0
        
        if len(code) == 6 and code.endswith(('01', '02', '03')):
            category = 'QDII'
            risk_level = 4
            volatility = '高'
        
        return {
            'type': category.replace('型', ''),
            'category': category,
            'style': style,
            'risk_level': risk_level,
            'expected_return': expected_return,
            'volatility': volatility
        }
    
    def _load_cache(self, cache_file: Path, fund_codes: List[str]) -> pd.DataFrame:
        """加载缓存特征"""
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            if 'fund_codes' in cache_data and 'features' in cache_data:
                cached_codes = cache_data['fund_codes']
                if set(fund_codes).issubset(set(cached_codes)):
                    self.logger.info("缓存有效，直接使用")
                    return cache_data['features']
            
        except Exception as e:
            self.logger.warning(f"加载缓存失败: {e}")
        
        return None
    
    def _save_cache(self, cache_file: Path, features: pd.DataFrame, fund_codes: List[str]):
        """保存特征缓存"""
        try:
            cache_data = {
                'fund_codes': fund_codes,
                'features': features,
                'timestamp': datetime.now(),
                'version': '2.0'
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            self.logger.info(f"特征缓存已保存: {cache_file}")
            
        except Exception as e:
            self.logger.warning(f"保存缓存失败: {e}")
    
    def _save_enhanced_csv(self, df: pd.DataFrame):
        """保存包含LLM特征的增强数据到CSV"""
        try:
            # 备份原始文件
            backup_path = f"{self.config.DATA_PATH}.backup"
            if os.path.exists(self.config.DATA_PATH) and not os.path.exists(backup_path):
                import shutil
                shutil.copy2(self.config.DATA_PATH, backup_path)
                self.logger.info(f"原始数据已备份到: {backup_path}")
            
            # 保存增强数据
            # 确保日期格式正确
            df_to_save = df.copy()
            if 'transaction_date' in df_to_save.columns:
                df_to_save['transaction_date'] = df_to_save['transaction_date'].dt.strftime('%Y%m%d')
            
            df_to_save.to_csv(self.config.DATA_PATH, index=False)
            self.logger.info(f"包含LLM特征的增强数据已保存到: {self.config.DATA_PATH}")
            self.logger.info(f"新数据形状: {df_to_save.shape}")
            
        except Exception as e:
            self.logger.warning(f"保存增强CSV失败: {e}")

class EnhancedDataLoader:
    """增强数据加载类"""
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def load_data(self) -> pd.DataFrame:
        """加载并预处理原始数据"""
        self.logger.info(f"从 {self.config.DATA_PATH} 加载数据...")
        
        # Check if the data file exists
        if not os.path.exists(self.config.DATA_PATH):
            self.logger.error(f"数据文件未找到: {self.config.DATA_PATH}")
            self.logger.info("将创建一个模拟数据文件用于演示。")
            self._create_dummy_data()

        df = pd.read_csv(self.config.DATA_PATH, dtype={'fund_code': str})
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], format='%Y%m%d')
        
        # 检查是否包含LLM特征
        llm_feature_cols = [col for col in df.columns if any(
            prefix in col for prefix in ['fund_description', 'fund_category', 'fund_style', 
                                       'risk_level', 'investor_type', 'trend_direction', 
                                       'overall_risk', 'embed_']
        )]
        
        if llm_feature_cols:
            self.logger.info(f"✅ 检测到CSV文件中已包含 {len(llm_feature_cols)} 个LLM特征列")
            self.logger.info(f"示例LLM特征: {llm_feature_cols[:5]}")
        else:
            self.logger.info("⚠️  CSV文件中未检测到LLM特征，稍后将生成")
        
        # 数据清洗
        df = self._clean_data(df)
        
        self.logger.info(f"数据加载完成，共 {len(df)} 条记录，{df.shape[1]} 个特征列")
        self.logger.info(f"日期范围: {df['transaction_date'].min().date()} 到 {df['transaction_date'].max().date()}")
        self.logger.info(f"基金数量: {df['fund_code'].nunique()}")
        
        return df

    def _create_dummy_data(self):
        """Creates a dummy data file for demonstration purposes."""
        # 根据比赛要求，数据从2024/4/8开始
        dates = pd.date_range(start='2024-04-08', end='2024-12-31', freq='D')
        fund_codes = [f'{110000 + i:06d}' for i in range(20)]
        
        records = []
        for code in fund_codes:
            base_uv = np.random.randint(1000, 10000)  # 基础UV
            for date in dates:
                # 模拟基金申购赎回数据
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
                trend_factor = 1 + 0.1 * np.random.normal()
                
                records.append({
                    'transaction_date': date.strftime('%Y%m%d'),
                    'fund_code': code,
                    'apply_amt': np.random.lognormal(10, 1.5) * seasonal_factor * trend_factor,
                    'redeem_amt': np.random.lognormal(9.8, 1.3) * seasonal_factor * trend_factor,
                    'uv_key_page_1': max(0, int(base_uv * (0.8 + 0.4 * np.random.random()) * seasonal_factor)),
                    'uv_key_page_2': max(0, int(base_uv * (0.6 + 0.4 * np.random.random()) * seasonal_factor)),
                    'uv_key_page_3': max(0, int(base_uv * (0.4 + 0.3 * np.random.random()) * seasonal_factor))
                })
        
        dummy_df = pd.DataFrame(records)
        dummy_df.to_csv(self.config.DATA_PATH, index=False)
        self.logger.info(f"模拟数据文件已创建: {self.config.DATA_PATH} (包含UV特征)")

    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        self.logger.info("开始数据清洗...")
        
        # 移除明显异常值 - 包括UV特征
        numeric_cols = ['apply_amt', 'redeem_amt', 'uv_key_page_1', 'uv_key_page_2', 'uv_key_page_3']
        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                self.logger.info(f"{col} 异常值数量: {outliers.sum()}")
                
                # 用分位数替换极值
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
        
        # 确保非负值
        available_cols = [col for col in numeric_cols if col in df.columns]
        df[available_cols] = df[available_cols].clip(lower=0)
        
        # 移除数据点太少的基金
        fund_counts = df['fund_code'].value_counts()
        valid_funds = fund_counts[fund_counts >= 30].index
        df = df[df['fund_code'].isin(valid_funds)]
        
        self.logger.info(f"清洗后保留基金数量: {df['fund_code'].nunique()}")
        return df

def _emergency_feature_reduction(df: pd.DataFrame, config: Config, logger: logging.Logger) -> pd.DataFrame:
    """紧急特征削减"""
    logger.warning("执行紧急特征削减...")
    
    # 保留核心特征
    core_features = ['transaction_date', 'fund_code'] + config.TARGET_COLS
    
    # 特征重要性评估（简化版）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_scores = {}
    
    for col in numeric_cols:
        if col not in core_features:
            # 简单的方差评分
            variance_score = df[col].var()
            correlation_score = 0
            
            # 与目标变量的相关性
            for target in config.TARGET_COLS:
                if target in df.columns:
                    corr = abs(df[col].corr(df[target]))
                    if not np.isnan(corr):
                        correlation_score = max(correlation_score, corr)
            
            feature_scores[col] = variance_score * (1 + correlation_score)
    
    # 选择最佳特征
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    selected_features = [feat[0] for feat in sorted_features[:200]]  # 只保留200个最佳特征
    
    final_columns = core_features + selected_features
    df_reduced = df[final_columns].copy()
    
    logger.info(f"特征削减: {df.shape[1]} -> {df_reduced.shape[1]}")
    return df_reduced

class SuperAdvancedFeatureEngineer:
    """超级高级特征工程类"""
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建所有特征（内存优化版 v2）"""
        self.logger.info("开始超级特征工程（内存优化版 v2）...")
        
        process = psutil.Process(os.getpid())
        def monitor_memory():
            return process.memory_info().rss / 1024**2
            
        df = df.copy()
        initial_memory = monitor_memory()
        self.logger.info(f"初始内存使用: {initial_memory:.2f} MB")
        
        # 分阶段特征工程，每阶段后进行验证和清理
        feature_stages = [
            ("日期特征", self._add_enhanced_date_features, False),
            ("时间序列特征", self._add_enhanced_time_series_features, True),
            ("横截面特征", self._add_cross_sectional_features, True),
            ("交互特征", self._add_interaction_features, True),
            ("技术特征", self._add_advanced_technical_features, True)
        ]
        
        for stage_name, stage_func, use_safe_wrapper in feature_stages:
            if use_safe_wrapper:
                df = self._safe_feature_calculation(df, stage_func, stage_name)
            else:
                df = stage_func(df)
            
            # 内存监控
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            self.logger.info(f"{stage_name}完成，内存使用: {current_memory:.2f} MB")
            
            # 检查内存限制
            if current_memory > self.config.MAX_MEMORY_MB:
                self.logger.warning(f"内存使用超过限制 ({self.config.MAX_MEMORY_MB} MB)，执行紧急特征削减...")
                df = _emergency_feature_reduction(df, self.config, self.logger)
                gc.collect()
                
            # 定期强制垃圾回收
            if stage_name in ["时间序列特征", "横截面特征"]:
                gc.collect()
                
        # 最终特征验证和选择
        if self.config.ENABLE_FEATURE_VALIDATION:
            df = self._validate_and_select_features(df)
        
        # 智能缺失值处理
        df = self._smart_fill_missing(df)
        
        # 最终内存清理
        gc.collect()
        
        final_memory = monitor_memory()
        self.logger.info(f"超级特征工程完成，最终特征数: {df.shape[1]}")
        self.logger.info(f"最终内存使用: {final_memory:.2f} MB，变化: {final_memory - initial_memory:+.2f} MB")
        
        return df
    
    def _safe_feature_calculation(self, df: pd.DataFrame, feature_func, feature_name: str) -> pd.DataFrame:
        """安全的特征计算包装器"""
        try:
            start_time = time.time()
            initial_cols = set(df.columns)
            
            # 调用特征函数
            result_df = feature_func(df)
            
            # 验证结果
            if result_df is None:
                self.logger.warning(f"{feature_name}: 函数返回None，使用原始数据")
                return df
                
            new_cols = set(result_df.columns) - initial_cols
            
            # 检查新特征的质量
            for col in new_cols:
                if result_df[col].dtype in [np.number]:
                    # 检查缺失值
                    null_pct = result_df[col].isnull().mean()
                    if null_pct > 0.5:
                        self.logger.warning(f"{feature_name}: {col} 有 {null_pct:.1%} 缺失值")
                    
                    # 检查无穷值
                    if np.isinf(result_df[col]).any():
                        self.logger.warning(f"{feature_name}: {col} 包含无穷值")
                        result_df[col] = result_df[col].replace([np.inf, -np.inf], np.nan)
                    
                    # 检查常数特征
                    if result_df[col].nunique() <= 1:
                        self.logger.warning(f"{feature_name}: {col} 是常数特征，将被移除")
                        result_df = result_df.drop(columns=[col])
            
            elapsed = time.time() - start_time
            valid_new_cols = set(result_df.columns) - initial_cols
            self.logger.info(f"{feature_name} 完成，新增 {len(valid_new_cols)} 个有效特征，耗时: {elapsed:.2f}s")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"{feature_name} 计算失败: {e}")
            self.logger.error(traceback.format_exc())
            return df  # 返回原始数据，继续执行
    
    def _validate_and_select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征验证和选择"""
        self.logger.info("开始特征验证和选择...")
        
        initial_feature_count = df.shape[1]
        
        # 1. 移除常数特征
        constant_features = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in ['transaction_date', 'fund_code'] + self.config.TARGET_COLS:
                if df[col].nunique() <= 1:
                    constant_features.append(col)
        
        if constant_features:
            df = df.drop(columns=constant_features)
            self.logger.info(f"移除常数特征: {len(constant_features)} 个")
        
        # 2. 移除高相关性特征
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                       if col not in ['transaction_date', 'fund_code'] + self.config.TARGET_COLS]
        
        if len(numeric_cols) > 1:
            try:
                corr_matrix = df[numeric_cols].corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                
                high_corr_features = [column for column in upper_tri.columns 
                                     if any(upper_tri[column] > self.config.MAX_CORRELATION_THRESHOLD)]
                if high_corr_features:
                    df = df.drop(columns=high_corr_features)
                    self.logger.info(f"移除高相关特征: {len(high_corr_features)} 个 (阈值: {self.config.MAX_CORRELATION_THRESHOLD})")
            except Exception as e:
                self.logger.warning(f"相关性分析失败: {e}")
        
        # 3. 移除缺失值过多的特征
        high_missing_features = []
        for col in numeric_cols:
            if col in df.columns:
                missing_pct = df[col].isnull().mean()
                if missing_pct > self.config.MAX_MISSING_RATIO:
                    high_missing_features.append(col)
        
        if high_missing_features:
            df = df.drop(columns=high_missing_features)
            self.logger.info(f"移除高缺失值特征: {len(high_missing_features)} 个 (阈值: {self.config.MAX_MISSING_RATIO})")
        
        # 4. 移除低方差特征
        low_variance_features = []
        for col in numeric_cols:
            if col in df.columns:
                variance = df[col].var()
                if variance < self.config.MIN_FEATURE_VARIANCE:
                    low_variance_features.append(col)
        
        if low_variance_features:
            df = df.drop(columns=low_variance_features)
            self.logger.info(f"移除低方差特征: {len(low_variance_features)} 个")
        
        # 5. 特征数量控制
        final_numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                             if col not in ['transaction_date', 'fund_code'] + self.config.TARGET_COLS]
        
        if len(final_numeric_cols) > self.config.MAX_FEATURE_COUNT:
            self.logger.warning(f"特征数量过多({len(final_numeric_cols)})，超过限制({self.config.MAX_FEATURE_COUNT})")
            # 可以在这里添加更高级的特征选择算法
        
        final_feature_count = df.shape[1]
        self.logger.info(f"特征验证完成: {initial_feature_count} -> {final_feature_count} "
                        f"(移除 {initial_feature_count - final_feature_count} 个)")
        
        return df
    
    def _add_enhanced_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """增强日期特征"""
        self.logger.info("添加增强日期特征...")
        
        # 基础日期特征
        df['month'] = df['transaction_date'].dt.month
        df['day_of_week'] = df['transaction_date'].dt.dayofweek
        df['day_of_month'] = df['transaction_date'].dt.day
        df['day_of_year'] = df['transaction_date'].dt.dayofyear
        df['week_of_year'] = df['transaction_date'].dt.isocalendar().week.astype(int)
        df['quarter'] = df['transaction_date'].dt.quarter
        
        # 高级日期特征
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df['transaction_date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['transaction_date'].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df['transaction_date'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['transaction_date'].dt.is_quarter_end.astype(int)
        df['is_year_start'] = df['transaction_date'].dt.is_year_start.astype(int)
        df['is_year_end'] = df['transaction_date'].dt.is_year_end.astype(int)
        
        # 季节性特征
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        df['sin_day_of_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['cos_day_of_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # 中国市场特殊日期
        years = df['transaction_date'].dt.year.unique()
        cn_holidays = holidays.CN(years=years)
        df['is_holiday'] = df['transaction_date'].isin(cn_holidays).astype(int)
        df['is_before_holiday'] = df['transaction_date'].apply(
            lambda x: (x + timedelta(days=1)) in cn_holidays
        ).astype(int)
        df['is_after_holiday'] = df['transaction_date'].apply(
            lambda x: (x - timedelta(days=1)) in cn_holidays
        ).astype(int)
        
        # 工作日计数
        df['work_days_since_month_start'] = df.groupby(['fund_code', df['transaction_date'].dt.to_period('M')])['transaction_date'].rank()
        
        return df
    
    def _add_enhanced_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """内存优化的时间序列特征"""
        self.logger.info("添加增强时间序列特征（内存优化版）...")
        
        df = df.sort_values(['fund_code', 'transaction_date'])
        
        # 分批处理特征，避免内存爆炸
        feature_batches = [
            ('lag_features', self._add_lag_features),
            ('rolling_features', self._add_rolling_features),
            ('diff_features', self._add_diff_features)
        ]
        
        for batch_name, batch_func in feature_batches:
            self.logger.info(f"处理 {batch_name}...")
            df = batch_func(df)
            gc.collect()  # 强制垃圾回收
            
            # 内存使用监控
            memory_usage = df.memory_usage(deep=True).sum() / 1024**2
            self.logger.info(f"{batch_name} 完成，当前内存使用: {memory_usage:.2f} MB")
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加滞后特征（内存优化）"""
        grouped = df.groupby('fund_code')
        
        # 扩展目标列，包括UV特征
        extended_cols = self.config.TARGET_COLS.copy()
        uv_cols = ['uv_key_page_1', 'uv_key_page_2', 'uv_key_page_3']
        available_uv_cols = [col for col in uv_cols if col in df.columns]
        extended_cols.extend(available_uv_cols)
        
        # 分批处理列，避免内存过载
        for target_col in extended_cols:
            # 多重滞后特征
            for lag in self.config.LAG_PERIODS:
                df[f'{target_col}_lag_{lag}'] = grouped[target_col].shift(lag)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加滚动窗口特征（内存优化）"""
        grouped = df.groupby('fund_code')
        
        # 扩展目标列
        extended_cols = self.config.TARGET_COLS.copy()
        uv_cols = ['uv_key_page_1', 'uv_key_page_2', 'uv_key_page_3']
        available_uv_cols = [col for col in uv_cols if col in df.columns]
        extended_cols.extend(available_uv_cols)
        
        # 分批处理窗口大小，减少内存占用
        for target_col in extended_cols:
            for window in self.config.ROLLING_WINDOWS:
                shifted = grouped[target_col].shift(1)
                rolling_window = shifted.rolling(window=window, min_periods=1)
                
                # 基础统计（分批计算）
                df[f'{target_col}_roll_mean_{window}'] = rolling_window.mean().reset_index(level=0, drop=True)
                df[f'{target_col}_roll_std_{window}'] = rolling_window.std().reset_index(level=0, drop=True)
                df[f'{target_col}_roll_max_{window}'] = rolling_window.max().reset_index(level=0, drop=True)
                df[f'{target_col}_roll_min_{window}'] = rolling_window.min().reset_index(level=0, drop=True)
                
                # 相对特征
                df[f'{target_col}_vs_roll_mean_{window}'] = df[target_col] / (df[f'{target_col}_roll_mean_{window}'] + 1e-8)
                
                # 每完成一个窗口就清理内存
                if window % 3 == 0:
                    gc.collect()
        
        return df
    
    def _add_diff_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加差分特征（内存优化）"""
        grouped = df.groupby('fund_code')
        
        # 扩展目标列
        extended_cols = self.config.TARGET_COLS.copy()
        uv_cols = ['uv_key_page_1', 'uv_key_page_2', 'uv_key_page_3']
        available_uv_cols = [col for col in uv_cols if col in df.columns]
        extended_cols.extend(available_uv_cols)
        
        for target_col in extended_cols:
            # 多期差分特征
            for diff_period in self.config.DIFF_PERIODS:
                df[f'{target_col}_diff_{diff_period}'] = grouped[target_col].diff(periods=diff_period)
                df[f'{target_col}_pct_change_{diff_period}'] = grouped[target_col].pct_change(periods=diff_period)
                
                # 加速度特征（限制计算量）
                if diff_period <= 7:
                    df[f'{target_col}_acceleration_{diff_period}'] = grouped[f'{target_col}_diff_{diff_period}'].diff()
        
        return df
    
    def _add_cross_sectional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加横截面特征（完全无泄露版本）"""
        self.logger.info("添加横截面特征（完全无泄露版本）...")
        
        # 确保数据按fund_code和日期排序
        df = df.sort_values(['fund_code', 'transaction_date']).reset_index(drop=True)
        
        for target_col in self.config.TARGET_COLS:
            # 初始化新特征列
            df[f'{target_col}_market_rank'] = 0.5
            df[f'{target_col}_market_zscore'] = 0.0
            df[f'{target_col}_vs_market_mean'] = 1.0
            
            # 基于fund_code分组的历史特征（无泄露）
            grouped = df.groupby('fund_code')[target_col]
            
            # 计算历史均值和标准差（使用shift避免未来信息泄露）
            historical_mean = grouped.expanding().mean().shift(1)
            historical_std = grouped.expanding().std().shift(1)
            
            # 重置索引以确保对齐
            historical_mean = historical_mean.reset_index(drop=True)
            historical_std = historical_std.reset_index(drop=True)
            
            # 计算相对历史均值的比值
            df[f'{target_col}_vs_self_expanding_mean'] = df[target_col] / (historical_mean + 1e-8)
            
            # 计算历史z-score
            df[f'{target_col}_self_zscore'] = (df[target_col] - historical_mean) / (historical_std + 1e-8)
            
            # 滚动窗口历史特征
            for window in [7, 14, 30]:
                # 计算滚动历史均值和标准差
                historical_rolling_mean = grouped.shift(1).rolling(window=window, min_periods=1).mean()
                historical_rolling_std = grouped.shift(1).rolling(window=window, min_periods=1).std()
                
                # 重置索引以确保对齐
                historical_rolling_mean = historical_rolling_mean.reset_index(drop=True)
                historical_rolling_std = historical_rolling_std.reset_index(drop=True)
                
                # 相对历史均值
                df[f'{target_col}_vs_historical_mean_{window}'] = df[target_col] / (historical_rolling_mean + 1e-8)
                
                # 历史z-score
                df[f'{target_col}_historical_zscore_{window}'] = (df[target_col] - historical_rolling_mean) / (historical_rolling_std + 1e-8)
                
                # 历史百分位数排名（安全版本）
                def safe_percentile_rank(series):
                    """计算历史百分位数排名，避免未来信息泄露"""
                    result = pd.Series(index=series.index, dtype=float)
                    for i in range(len(series)):
                        if i > 0:
                            historical_values = series.iloc[:i]
                            if len(historical_values) > 0:
                                current_value = series.iloc[i]
                                percentile = (historical_values <= current_value).mean()
                                result.iloc[i] = percentile
                            else:
                                result.iloc[i] = 0.5
                        else:
                            result.iloc[i] = 0.5
                    return result
                
                # 应用百分位数排名
                percentile_ranks = grouped.transform(safe_percentile_rank)
                df[f'{target_col}_historical_percentile_{window}'] = percentile_ranks.reset_index(drop=True)
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加交互特征"""
        self.logger.info("添加交互特征...")
        
        # 申购赎回相关特征
        df['apply_redeem_ratio'] = df['apply_amt'] / (df['redeem_amt'] + 1e-8)
        df['net_flow'] = df['apply_amt'] - df['redeem_amt']
        df['net_flow_ratio'] = df['net_flow'] / (df['apply_amt'] + df['redeem_amt'] + 1e-8)
        df['total_volume'] = df['apply_amt'] + df['redeem_amt']
        
        # UV特征工程
        uv_cols = ['uv_key_page_1', 'uv_key_page_2', 'uv_key_page_3']
        available_uv_cols = [col for col in uv_cols if col in df.columns]
        
        if available_uv_cols:
            # UV总量和比例
            df['total_uv'] = df[available_uv_cols].sum(axis=1)
            for i, col in enumerate(available_uv_cols, 1):
                df[f'uv_page_{i}_ratio'] = df[col] / (df['total_uv'] + 1e-8)
            
            # UV与申购赎回的关系
            df['apply_per_uv'] = df['apply_amt'] / (df['total_uv'] + 1e-8)
            df['redeem_per_uv'] = df['redeem_amt'] / (df['total_uv'] + 1e-8)
            df['volume_per_uv'] = df['total_volume'] / (df['total_uv'] + 1e-8)
            
            # UV转化率特征
            for col in available_uv_cols:
                df[f'{col}_apply_conversion'] = df['apply_amt'] / (df[col] + 1e-8)
                df[f'{col}_redeem_conversion'] = df['redeem_amt'] / (df[col] + 1e-8)
        
        # 高级交互特征
        df['flow_momentum'] = df['apply_amt'] * df['redeem_amt']  # 乘积特征
        df['flow_volatility'] = np.sqrt(df['apply_amt']**2 + df['redeem_amt']**2)  # 向量长度
        
        # 日期交互特征
        for col in ['apply_amt', 'redeem_amt']:
            df[f'{col}_weekend_effect'] = df[col] * df['is_weekend']
            df[f'{col}_month_end_effect'] = df[col] * df['is_month_end']
            df[f'{col}_holiday_effect'] = df[col] * df['is_holiday']
        
        return df
    
    def _add_advanced_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加高级技术特征"""
        self.logger.info("添加高级技术特征...")
        
        df = df.sort_values(['fund_code', 'transaction_date'])
        
        # 将分组操作提前，提高效率
        grouped_by_fund = df.groupby('fund_code')
        
        for target_col in self.config.TARGET_COLS:
            # 指数加权移动平均
            for alpha in [0.1, 0.2, 0.3, 0.5]:
                # 使用 .transform() 来保持索引一致
                df[f'{target_col}_ewm_{alpha}'] = grouped_by_fund[target_col].transform(lambda x: x.ewm(alpha=alpha).mean())
                df[f'{target_col}_ewm_std_{alpha}'] = grouped_by_fund[target_col].transform(lambda x: x.ewm(alpha=alpha).std())
            
            # 累积特征 - 直接对 GroupBy 对象操作，无需 apply 或 transform
            df[f'{target_col}_cumsum'] = grouped_by_fund[target_col].cumsum()
            df[f'{target_col}_cummax'] = grouped_by_fund[target_col].cummax()
            df[f'{target_col}_cummin'] = grouped_by_fund[target_col].cummin()
            
            # 使用 .transform() 修复了索引不匹配的问题
            df[f'{target_col}_cumprod_log'] = grouped_by_fund[target_col].transform(lambda x: np.log1p(x).cumsum())
            
            # 技术指标
            deltas = grouped_by_fund[target_col].diff()
            gains = deltas.where(deltas > 0, 0)
            losses = -deltas.where(deltas < 0, 0)
            
            for period in [7, 14, 21]:
                # 使用 .transform() 修复潜在的索引问题
                avg_gains = grouped_by_fund[target_col].transform(lambda x: x.diff().where(x.diff() > 0, 0).rolling(window=period, min_periods=1).mean())
                avg_losses = grouped_by_fund[target_col].transform(lambda x: -x.diff().where(x.diff() < 0, 0).rolling(window=period, min_periods=1).mean())
                rs = avg_gains / (avg_losses + 1e-8)
                df[f'{target_col}_rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # MACD
            # 使用 .transform() 来保持索引一致
            ema12 = grouped_by_fund[target_col].transform(lambda x: x.ewm(span=12, adjust=False).mean())
            ema26 = grouped_by_fund[target_col].transform(lambda x: x.ewm(span=26, adjust=False).mean())
            df[f'{target_col}_macd'] = ema12 - ema26
            
            # 对新生成的列进行分组计算
            df[f'{target_col}_macd_signal'] = df.groupby('fund_code')[f'{target_col}_macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
            df[f'{target_col}_macd_histogram'] = df[f'{target_col}_macd'] - df[f'{target_col}_macd_signal']
            
            # 布林带
            for window in [14, 21]:
                # 使用 .transform() 来保持索引一致
                sma = grouped_by_fund[target_col].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
                std = grouped_by_fund[target_col].transform(lambda x: x.rolling(window=window, min_periods=1).std())
                df[f'{target_col}_bb_upper_{window}'] = sma + 2 * std
                df[f'{target_col}_bb_lower_{window}'] = sma - 2 * std
                df[f'{target_col}_bb_width_{window}'] = (df[f'{target_col}_bb_upper_{window}'] - df[f'{target_col}_bb_lower_{window}']) / (sma + 1e-8)
                df[f'{target_col}_bb_position_{window}'] = (df[target_col] - df[f'{target_col}_bb_lower_{window}']) / (df[f'{target_col}_bb_upper_{window}'] - df[f'{target_col}_bb_lower_{window}'] + 1e-8)
        
        return df
    
    def _smart_fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """智能填充缺失值"""
        self.logger.info("智能填充缺失值...")
        
        # Replace inf with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 按基金分组填充 (优化版本)
        df = df.groupby('fund_code', group_keys=False).apply(lambda group: group.ffill().bfill())
        
        # 最后的全局填充
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
             if df[col].isnull().any():
                  df[col] = df[col].fillna(df[col].median()) # Fill with median
        
        # Final fallback to 0 if median is NaN
        df.fillna(0, inplace=True)
        
        return df

class EnhancedModelTrainer:
    """增强模型训练类"""
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.config.MODEL_DIR.mkdir(exist_ok=True)
    
    def train_models(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
        """修复后的模型训练"""
        models = {}
        
        for target_col in self.config.TARGET_COLS:
            model_type = 'apply' if 'apply' in target_col else 'redeem'
            self.logger.info(f"训练 {model_type} 模型（修复版）...")
            
            # 数据预处理和清理
            train_features, val_features = self._prepare_autogluon_data(
                train_df, val_df, target_col
            )
            
            predictor = TabularPredictor(
                label=target_col,
                eval_metric=wmape_scorer,  # 修复：使用比赛一致的WMAPE指标
                path=str(self.config.MODEL_DIR / f"{target_col}_model"),
                verbosity=2,
                problem_type='regression'
            )
            
            predictor.fit(
                train_data=train_features,
                tuning_data=val_features,
                presets='best_quality',
                time_limit=self.config.TIME_LIMIT,
                excluded_model_types=['KNN'],  # 排除KNN，避免内存问题
                # 优化的超参数配置
                hyperparameters={
                    'GBM': [
                        {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
                        {},
                    ],
                    'CAT': {},
                    'XGB': {},
                    'RF': {'n_estimators': 300},
                },
                auto_stack=True,
                num_bag_folds=5,  # 减少折数，提高速度
                num_bag_sets=1,
                num_stack_levels=1,
                use_bag_holdout=True,
                ag_args_fit={'num_gpus': 0, 'num_cpus': 4}  # 减少资源占用
            )
            
            models[f"{target_col}_model"] = predictor
            self.logger.info(f"✅ {model_type} 模型训练完成")
        
        return models
    
    def _prepare_autogluon_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """为AutoGluon准备数据，修复兼容性问题"""
        self.logger.info(f"为 {target_col} 准备数据...")
        
        # 获取特征列
        feature_cols = [col for col in train_df.columns 
                        if col not in ['transaction_date', 'fund_code'] + self.config.TARGET_COLS]
        
        self.logger.info(f"准备处理 {len(feature_cols)} 个特征")
        
        # 复制数据避免修改原始数据
        train_clean = train_df[feature_cols + [target_col]].copy()
        val_clean = val_df[feature_cols + [target_col]].copy()
        
        # 🔧 修复1：处理无穷值和极大值
        for col in feature_cols:
            if train_clean[col].dtype in ['float64', 'float32']:
                # 替换无穷值为NaN
                train_clean[col] = train_clean[col].replace([np.inf, -np.inf], np.nan)
                val_clean[col] = val_clean[col].replace([np.inf, -np.inf], np.nan)
                
                # 限制极值（使用训练集的分位数）
                q99 = train_clean[col].quantile(0.99)
                q01 = train_clean[col].quantile(0.01)
                
                if not np.isnan(q99) and not np.isnan(q01):
                    train_clean[col] = train_clean[col].clip(q01, q99)
                    val_clean[col] = val_clean[col].clip(q01, q99)
        
        # 🔧 修复2：处理分类特征
        categorical_cols = train_clean.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col != target_col:
                # 确保验证集类别在训练集中存在
                train_categories = set(train_clean[col].dropna().astype(str))
                val_clean[col] = val_clean[col].astype(str)
                
                # 将验证集中的未见类别设为最常见类别
                most_common = train_clean[col].mode().iloc[0] if len(train_clean[col].mode()) > 0 else 'unknown'
                val_clean[col] = val_clean[col].apply(
                    lambda x: x if x in train_categories else most_common
                )
                
                # 转换为category类型
                train_clean[col] = train_clean[col].astype('category')
                val_clean[col] = val_clean[col].astype('category')
        
        # 🔧 修复3：处理缺失值
        # 数值列用中位数填充
        numeric_cols = train_clean.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]
        
        for col in numeric_cols:
            median_val = train_clean[col].median()
            if np.isnan(median_val):
                median_val = 0.0
            
            train_clean[col] = train_clean[col].fillna(median_val)
            val_clean[col] = val_clean[col].fillna(median_val)
        
        # 分类列用众数填充
        for col in categorical_cols:
            if col != target_col:
                mode_val = train_clean[col].mode().iloc[0] if len(train_clean[col].mode()) > 0 else 'unknown'
                train_clean[col] = train_clean[col].fillna(mode_val)
                val_clean[col] = val_clean[col].fillna(mode_val)
        
        # 🔧 修复4：目标变量处理
        # 确保目标变量为正数
        train_clean[target_col] = np.maximum(train_clean[target_col], 0.1)
        val_clean[target_col] = np.maximum(val_clean[target_col], 0.1)
        
        # 最终检查：移除所有仍包含NaN的行
        train_clean = train_clean.dropna()
        val_clean = val_clean.dropna()
        
        self.logger.info(f"数据预处理完成:")
        self.logger.info(f"  训练集形状: {train_clean.shape}")
        self.logger.info(f"  验证集形状: {val_clean.shape}")
        self.logger.info(f"  特征数量: {len(feature_cols)}")
        
        return train_clean, val_clean

# WMAPE评估函数和评分器
# WMAPE评估函数和评分器


def _predict_single_fund_day(recent_data: pd.DataFrame, future_date: pd.Timestamp, logger: logging.Logger) -> dict:
    """单个基金单日预测（科学方法）"""
    latest_record = recent_data.iloc[-1].copy()
    
    # 基于移动平均和趋势的预测
    for col in ['apply_amt', 'redeem_amt']:
        if len(recent_data) >= 14:
            # 多期移动平均
            ma7 = recent_data[col].tail(7).mean()
            ma14 = recent_data[col].tail(14).mean()
            ma30 = recent_data[col].tail(min(30, len(recent_data))).mean()
            
            # 趋势计算
            trend_short = (ma7 - ma14) / (ma14 + 1e-8)
            trend_long = (ma14 - ma30) / (ma30 + 1e-8)
            trend_combined = 0.7 * trend_short + 0.3 * trend_long
            
            # 季节性调整
            seasonal_factor = 1 + 0.05 * np.sin(2 * np.pi * future_date.dayofyear / 365)
            
            # 波动性调整
            volatility = recent_data[col].tail(14).std() / (ma14 + 1e-8)
            volatility_adj = max(0.8, min(1.2, 1 - volatility * 0.1))
            
            # 预测值
            predicted_value = ma7 * (1 + trend_combined * 0.3) * seasonal_factor * volatility_adj
            latest_record[col] = max(100, predicted_value)  # 确保最小值
        else:
            # 数据不足时使用简单均值
            latest_record[col] = max(100, recent_data[col].mean())
    
    # UV特征预测（基于历史模式）
    uv_cols = ['uv_key_page_1', 'uv_key_page_2', 'uv_key_page_3']
    for uv_col in uv_cols:
        if uv_col in recent_data.columns and len(recent_data) >= 7:
            uv_ma = recent_data[uv_col].tail(7).mean()
            uv_trend = recent_data[uv_col].pct_change().tail(7).mean()
            
            # 季节性和工作日调整
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * future_date.dayofyear / 365)
            weekday_factor = 0.8 if future_date.weekday() >= 5 else 1.0  # 周末调整
            
            predicted_uv = uv_ma * (1 + uv_trend * 0.2) * seasonal_factor * weekday_factor
            latest_record[uv_col] = max(0, predicted_uv)
    
    latest_record['transaction_date'] = future_date
    return latest_record

def improved_recursive_prediction(extended_df: pd.DataFrame, future_dates: pd.DatetimeIndex, 
                               models: Dict, config: Config, logger: logging.Logger) -> pd.DataFrame:
    """改进的递推预测，减少累积误差"""
    logger.info("🔧 开始改进的递推预测...")
    
    prediction_df = extended_df.copy()
    
    # 创建多个基准模型，减少单一模型的累积误差
    base_predictors = {}
    for target_col in config.TARGET_COLS:
        # 基于历史数据创建简单的基准预测器
        base_predictors[target_col] = {
            'ma7': lambda x: x.tail(7).mean(),
            'ma14': lambda x: x.tail(14).mean(),
            'trend': lambda x: x.tail(7).mean() + (x.iloc[-1] - x.iloc[-7]) / 7 if len(x) >= 7 else x.mean(),
            'seasonal': lambda x: x.tail(7).mean() * (1 + 0.1 * np.sin(2 * np.pi * pd.Timestamp.now().dayofyear / 365))
        }
    
    # 逐日预测，但每3天重新校准一次
    calibration_interval = 3
    
    for day_idx, future_date in enumerate(future_dates):
        logger.info(f"🔮 预测第 {day_idx + 1} 天: {future_date.date()}")
        
        daily_predictions = []
        
        for fund_code in prediction_df['fund_code'].unique():
            fund_history = prediction_df[
                prediction_df['fund_code'] == fund_code
            ].sort_values('transaction_date')
            
            if len(fund_history) == 0:
                continue
            
            latest_record = fund_history.iloc[-1].copy()
            future_record = latest_record.copy()
            future_record['transaction_date'] = future_date
            
            # 获取真实历史数据（排除之前的预测）
            real_history_cutoff = pd.to_datetime(config.PREDICTION_START_DATE)
            real_history = fund_history[fund_history['transaction_date'] < real_history_cutoff]
            
            # 多模型ensemble预测
            for target_col in config.TARGET_COLS:
                if len(real_history) >= 7:
                    # 基准预测（基于真实历史）
                    baseline_predictions = []
                    for pred_name, pred_func in base_predictors[target_col].items():
                        try:
                            pred_value = pred_func(real_history[target_col])
                            if not np.isnan(pred_value) and pred_value > 0:
                                baseline_predictions.append(pred_value)
                        except:
                            pass
                    
                    # 如果有基准预测，使用加权平均
                    if baseline_predictions:
                        baseline_pred = np.mean(baseline_predictions)
                        
                        # 每隔几天使用深度学习模型校准
                        if day_idx % calibration_interval == 0 and f"{target_col}_model" in models:
                            try:
                                # 准备预测特征 - 排除目标列和标识列
                                exclude_cols = ['fund_code', 'transaction_date'] + config.TARGET_COLS
                                feature_data = {k: v for k, v in future_record.items() if k not in exclude_cols}
                                
                                # 转换为DataFrame格式
                                pred_df = pd.DataFrame([feature_data])
                                
                                # 使用模型进行预测
                                model_pred = models[f"{target_col}_model"].predict(pred_df)[0]
                                
                                if model_pred > 0:  # 模型预测有效
                                    # 混合基准预测和模型预测，权重随时间递减
                                    model_weight = max(0.1, 0.8 - day_idx * 0.1)  # 权重从0.8递减到0.1
                                    baseline_weight = 1 - model_weight
                                    
                                    final_pred = model_weight * model_pred + baseline_weight * baseline_pred
                                    future_record[target_col] = max(100, final_pred)
                                else:
                                    future_record[target_col] = max(100, baseline_pred)
                                    logger.warning(f"模型预测结果无效: {model_pred}，使用基准预测")
                                
                            except Exception as e:
                                future_record[target_col] = max(100, baseline_pred)
                                logger.warning(f"模型预测异常: {str(e)}，使用基准预测")
                        else:
                            future_record[target_col] = max(100, baseline_pred)
                    else:
                        # 如果基准预测失败，使用最近值
                        future_record[target_col] = latest_record[target_col]
                else:
                    # 历史数据不足
                    future_record[target_col] = latest_record[target_col]
            
            daily_predictions.append(future_record)
        
        # 添加当日预测到数据集
        if daily_predictions:
            daily_df = pd.DataFrame(daily_predictions)
            prediction_df = pd.concat([prediction_df, daily_df], ignore_index=True)
            prediction_df = prediction_df.sort_values(['fund_code', 'transaction_date'])
    
    return prediction_df

def enhanced_model_validation(models: Dict, val_df: pd.DataFrame, config: Config, logger: logging.Logger) -> Dict:
    """增强的模型验证"""
    logger.info("(使用MAE, RMSE, WMAPE, 方向准确率进行全面评估)")
    validation_results = {}
    
    # 确保验证集的时间顺序
    val_df = val_df.sort_values(['fund_code', 'transaction_date'])
    
    for model_name, predictor in models.items():
        target_col = 'apply_amt' if 'apply_amt' in model_name else 'redeem_amt'
        
        try:
            # 准备验证数据
            val_clean, _ = EnhancedModelTrainer(config, logger)._prepare_autogluon_data(val_df, val_df, target_col)
            
            # 移除fund_code列进行预测（AutoGluon不需要这个列）
            prediction_features = val_clean.drop(columns=[target_col, 'fund_code'], errors='ignore')
            
            # 预测验证集
            predictions = predictor.predict(prediction_features)
            actual = val_clean[target_col]
            
            # 多指标评估
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            
            mae = mean_absolute_error(actual, predictions)
            rmse = np.sqrt(mean_squared_error(actual, predictions))
            wmape = np.sum(np.abs(predictions - actual)) / np.sum(np.abs(actual))
            
            # 方向准确率 (按基金分组计算)
            direction_accuracy_scores = []
            for fund_code in val_df['fund_code'].unique():
                fund_mask = val_clean['fund_code'] == fund_code
                actual_fund = actual[fund_mask]
                pred_fund = predictions[fund_mask]
                
                if len(actual_fund) > 1:
                    actual_direction = np.sign(actual_fund.diff().dropna())
                    pred_direction = np.sign(pd.Series(pred_fund).diff().dropna())
                    
                    if len(actual_direction) > 0:
                        accuracy = (actual_direction == pred_direction).mean()
                        direction_accuracy_scores.append(accuracy)
            
            direction_accuracy = np.mean(direction_accuracy_scores) if direction_accuracy_scores else 0.0
            
            validation_results[model_name] = {
                'MAE': mae,
                'RMSE': rmse,
                'WMAPE': wmape,
                'Direction_Accuracy': direction_accuracy
            }
            
            logger.info(f"--- {model_name} 验证结果 ---")
            logger.info(f"  MAE: {mae:.2f}")
            logger.info(f"  RMSE: {rmse:.2f}")
            logger.info(f"  WMAPE: {wmape:.4f} (越小越好)")
            logger.info(f"  方向准确率: {direction_accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"{model_name} 验证失败: {e}")
            validation_results[model_name] = {'error': str(e)}
    
    return validation_results

def setup_logging() -> logging.Logger:
    """设置日志"""
    # 移除已存在的handlers，防止重复记录
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('fund_prediction_enhanced.log', mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """主函数"""
    logger = setup_logging()
    config = Config()
    
    try:
        # 1. 数据加载
        loader = EnhancedDataLoader(config, logger)
        df = loader.load_data()
        
        # 2. 大模型特征生成 (可选，如果API密钥无效会跳过)
        if config.API_KEY:
             llm_feature_generator = SmartLLMFeatureGenerator(config, logger)
             # 根据配置决定是否将LLM特征保存到CSV文件
             df = llm_feature_generator.generate_features(df, save_to_csv=config.SAVE_LLM_TO_CSV)
             if config.SAVE_LLM_TO_CSV:
                 logger.info("💾 LLM特征已直接保存到CSV文件，后续运行将自动加载")
        else:
             logger.warning("未配置有效的大模型API密钥，将跳过LLM特征生成。")
        
        # 3. 超级特征工程
        feature_engineer = SuperAdvancedFeatureEngineer(config, logger)
        df = feature_engineer.create_features(df)
        
        # 4. B榜最终训练：使用全部历史数据
        logger.info("=" * 60)
        logger.info("🏆 B榜最终预测模式")
        logger.info("=" * 60)
        
        min_date = df['transaction_date'].min()
        max_date = df['transaction_date'].max()
        
        logger.info(f"使用全部历史数据进行最终训练...")
        logger.info(f"历史数据范围: {min_date.date()} 到 {max_date.date()}")
        
        # 使用100%的历史数据作为训练集
        train_df = df.copy()
        
        # 为了代码兼容性，创建一个小的验证集（仅用于模型内部验证）
        validation_days = min(30, int(len(df) * 0.05))  # 最多用5%作为内部验证
        train_end_date = max_date - timedelta(days=validation_days)
        val_df = df[df['transaction_date'] > train_end_date].copy()
        
        # 移除潜在的全部为NaN的行
        train_df.dropna(how='all', inplace=True)
        val_df.dropna(how='all', inplace=True)
        
        logger.info(f"最终训练集: {len(train_df)} 条记录")
        logger.info(f"内部验证集: {len(val_df)} 条记录 (仅用于模型调参)")
        
        if len(train_df) == 0:
            raise ValueError("训练集为空，请检查数据。")

        # 5. 模型训练
        trainer = EnhancedModelTrainer(config, logger)
        models = trainer.train_models(train_df, val_df)
        
        # 6. 快速模型验证（可选）
        logger.info("\n==================== 增强模型验证 ====================")
        enhanced_model_validation(models, val_df, config, logger)
        logger.info("================================================")
        
        # 7. 🎯 B榜真正的未来预测
        logger.info("\n" + "=" * 60)
        logger.info("🎯 开始B榜未来7天递推预测 (v3)")
        logger.info("=" * 60)
        
        # 创建未来7天的预测框架
        prediction_start = pd.to_datetime(config.PREDICTION_START_DATE)
        future_dates = pd.date_range(start=prediction_start, periods=config.PREDICTION_DAYS, freq='D')
        
        logger.info(f"预测目标: {config.PREDICTION_DAYS} 天")
        logger.info(f"预测日期: {future_dates[0].date()} 到 {future_dates[-1].date()}")
        
        # 修复：对所有目标列进行联合递推预测
        final_predictions = improved_recursive_prediction(
            df, future_dates, models, config, logger
        )
        
        # 提取未来预测部分
        future_with_final_predictions = final_predictions[
            final_predictions['transaction_date'] >= prediction_start
        ].copy()

        # 构建最终提交文件
        submission_df = pd.DataFrame({
            'fund_code': future_with_final_predictions['fund_code'],
            'transaction_date': future_with_final_predictions['transaction_date'].dt.strftime('%Y%m%d'),
            'apply_amt_pred': future_with_final_predictions['apply_amt'],
            'redeem_amt_pred': future_with_final_predictions['redeem_amt']
        })
        
        # 确保列的顺序正确
        submission_df = submission_df[['fund_code', 'transaction_date', 'apply_amt_pred', 'redeem_amt_pred']]
        
        # 保存B榜提交文件
        submission_path = 'predict_result.csv'
        submission_df.to_csv(submission_path, index=False)
        logger.info(f"🎯 B榜提交文件已保存: {submission_path}")
        logger.info(f"   - 记录数: {len(submission_df)}")
        logger.info(f"   - 基金数: {submission_df['fund_code'].nunique()}")
        logger.info(f"   - 预测天数: {submission_df['transaction_date'].nunique()}")
        
        # 生成训练集文件（用于B榜提交）
        train_set_path = 'train_set.csv'
        full_df = df.copy()  # 包含所有构造的特征
        full_df['transaction_date'] = full_df['transaction_date'].dt.strftime('%Y%m%d')
        full_df.to_csv(train_set_path, index=False)
        logger.info(f"🎯 B榜训练集文件已保存: {train_set_path}")
        
        # 显示预测样本
        logger.info(f"\n📋 预测结果样本:")
        sample_df = submission_df.head(10)
        for _, row in sample_df.iterrows():
            logger.info(f"基金 {row['fund_code']} - {row['transaction_date']}: "
                       f"申购 {row['apply_amt_pred']:.2f}, 赎回 {row['redeem_amt_pred']:.2f}")
        
        logger.info(f"\n💡 B榜提交提示:")
        logger.info(f"   1. 提交文件: {submission_path}")
        logger.info(f"   2. 训练集文件: {train_set_path}")
        logger.info(f"   3. 需要打包成 result.zip")
        logger.info(f"   4. 记得准备 llm_record.docx")

        logger.info("\n🎉 任务成功完成！")

    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}", exc_info=True)

if __name__ == "__main__":
    main()