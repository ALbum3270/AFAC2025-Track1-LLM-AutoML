#!/usr/bin/env python3
"""
仅预测脚本 - 加载已训练的模型并执行B榜预测
无需重新训练，直接使用保存的模型
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import warnings
from pathlib import Path
import sys
import os
from typing import Dict

# 导入原有的类和函数
from train import Config, EnhancedDataLoader, SuperAdvancedFeatureEngineer, setup_logging, improved_recursive_prediction
from autogluon.tabular import TabularPredictor

warnings.filterwarnings('ignore')

def load_trained_models(config: Config, logger: logging.Logger):
    """加载已训练的模型"""
    models = {}
    
    # 加载apply_amt模型
    apply_model_path = "autogluon_models/apply_amt_model"
    if os.path.exists(apply_model_path):
        try:
            models['apply_amt_model'] = TabularPredictor.load(apply_model_path)
            logger.info("✅ apply_amt 模型加载成功")
        except Exception as e:
            logger.error(f"❌ apply_amt 模型加载失败: {e}")
            return None
    else:
        logger.error(f"❌ apply_amt 模型路径不存在: {apply_model_path}")
        return None
    
    # 加载redeem_amt模型
    redeem_model_path = "autogluon_models/redeem_amt_model"
    if os.path.exists(redeem_model_path):
        try:
            models['redeem_amt_model'] = TabularPredictor.load(redeem_model_path)
            logger.info("✅ redeem_amt 模型加载成功")
        except Exception as e:
            logger.error(f"❌ redeem_amt 模型加载失败: {e}")
            return None
    else:
        logger.error(f"❌ redeem_amt 模型路径不存在: {redeem_model_path}")
        return None
    
    return models

def main():
    # 设置日志
    logger = setup_logging()
    logger.info("🚀 开始仅预测模式 - 使用已训练的模型")
    
    # 加载配置
    config = Config()
    
    # 1. 加载数据和特征工程
    logger.info("\n==================== 数据加载和特征工程 ====================")
    data_loader = EnhancedDataLoader(config, logger)
    df = data_loader.load_data()
    
    # 特征工程
    feature_engineer = SuperAdvancedFeatureEngineer(config, logger)
    df = feature_engineer.create_features(df)
    
    logger.info(f"✅ 数据准备完成，特征数: {df.shape[1]}")
    
    # 2. 加载已训练的模型
    logger.info("\n==================== 加载已训练模型 ====================")
    models = load_trained_models(config, logger)
    
    if models is None:
        logger.error("❌ 模型加载失败，无法继续预测")
        return
    
    # 3. 快速模型验证（可选）- 暂时跳过以避免重复代码
    logger.info("\n==================== 模型验证 ====================")
    logger.info("⏭️ 跳过模型验证，直接进行预测（避免代码重复）")
    
    # 4. B榜递推预测
    logger.info("\n==================== B榜递推预测 ====================")
    logger.info("🎯 开始B榜未来7天递推预测 (v3)")
    
    # 创建未来日期
    prediction_start = pd.to_datetime(config.PREDICTION_START_DATE)
    future_dates = pd.date_range(start=prediction_start, periods=config.PREDICTION_DAYS, freq='D')
    
    logger.info(f"预测目标: {config.PREDICTION_DAYS} 天")
    logger.info(f"预测日期: {future_dates[0].date()} 到 {future_dates[-1].date()}")
    
    # 执行递推预测
    try:
        final_predictions = improved_recursive_prediction(df, future_dates, models, config, logger)
        logger.info("✅ 递推预测完成")
        
        # 5. 保存预测结果
        logger.info("\n==================== 保存预测结果 ====================")
        
        # 提取预测期的数据
        prediction_results = final_predictions[final_predictions['transaction_date'] >= prediction_start]
        
        # 保存详细预测结果
        output_file = f"B榜预测结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        prediction_results.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"✅ 详细预测结果已保存: {output_file}")
        
        # 保存比赛格式结果
        competition_format = prediction_results[['fund_code', 'transaction_date', 'apply_amt', 'redeem_amt']].copy()
        competition_format.columns = ['fund_code', 'date', 'apply_amt_pred', 'redeem_amt_pred']
        
        competition_file = f"比赛提交格式_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        competition_format.to_csv(competition_file, index=False, encoding='utf-8')
        logger.info(f"✅ 比赛提交格式已保存: {competition_file}")
        
        # 打印预测统计
        logger.info("\n==================== 预测统计 ====================")
        logger.info(f"预测基金数量: {prediction_results['fund_code'].nunique()}")
        logger.info(f"预测天数: {prediction_results['transaction_date'].nunique()}")
        logger.info(f"总预测记录数: {len(prediction_results)}")
        
        logger.info(f"申购金额预测范围: {prediction_results['apply_amt'].min():.0f} - {prediction_results['apply_amt'].max():.0f}")
        logger.info(f"赎回金额预测范围: {prediction_results['redeem_amt'].min():.0f} - {prediction_results['redeem_amt'].max():.0f}")
        
        logger.info("\n🎉 预测任务全部完成！")
        
    except Exception as e:
        logger.error(f"❌ 递推预测失败: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 