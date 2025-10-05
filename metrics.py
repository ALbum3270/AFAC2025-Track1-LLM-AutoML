import numpy as np
import pandas as pd
from autogluon.core.metrics import make_scorer

def wmape_single_fund(y_true, y_pred):
    """
    计算单只基金的WMAPE (完全符合比赛规则)
    WMAPE_i = ∑(t=1 to 7) (|pred_t - real_t| / real_t) × (real_t / ∑(k=1 to 7) real_k)
    即：按交易量加权的百分比误差
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 避免除零错误
    total_real = np.sum(y_true)
    if total_real == 0:
        return 0.0
    
    # 按比赛规则计算：加权百分比误差
    wmape = 0.0
    for i in range(len(y_true)):
        if y_true[i] != 0:  # 避免除零
            daily_error = np.abs(y_pred[i] - y_true[i]) / y_true[i]  # 百分比误差
            daily_weight = y_true[i] / total_real  # 交易量权重
            wmape += daily_error * daily_weight
        # 如果y_true[i] == 0，这一天的误差不计入总和
    
    return wmape

def wmape_competition_standard(df_true, df_pred, fund_col='fund_code', apply_col='apply_amt', redeem_col='redeem_amt'):
    """
    完全按照比赛规则计算WMAPE
    
    Args:
        df_true: 真实值DataFrame，包含fund_code, apply_amt, redeem_amt
        df_pred: 预测值DataFrame，包含fund_code, apply_amt_pred, redeem_amt_pred
        fund_col: 基金代码列名
        apply_col: 申购列名
        redeem_col: 赎回列名
    
    Returns:
        申购WMAPE, 赎回WMAPE, 最终WMAPE (申购×0.5 + 赎回×0.5)
    """
    fund_codes = df_true[fund_col].unique()
    
    # 分别计算申购和赎回的WMAPE
    apply_wmapes = []
    redeem_wmapes = []
    apply_weights = []
    redeem_weights = []
    
    # 计算全局总量（用于权重计算）
    total_apply_global = df_true[apply_col].sum()
    total_redeem_global = df_true[redeem_col].sum()
    
    # 按基金分别计算WMAPE
    for fund in fund_codes:
        # 获取该基金的真实值和预测值
        fund_true = df_true[df_true[fund_col] == fund]
        fund_pred = df_pred[df_pred[fund_col] == fund]
        
        # 确保数据行数一致
        if len(fund_true) != len(fund_pred):
            raise ValueError(f"基金 {fund} 的真实值和预测值行数不一致")
        
        # 按行顺序匹配（假设数据已正确排序）
        apply_true = fund_true[apply_col].values
        apply_pred = fund_pred[f'{apply_col}_pred'].values
        redeem_true = fund_true[redeem_col].values  
        redeem_pred = fund_pred[f'{redeem_col}_pred'].values
        
        # 计算单只基金的WMAPE
        apply_wmape_i = wmape_single_fund(apply_true, apply_pred)
        redeem_wmape_i = wmape_single_fund(redeem_true, redeem_pred)
        
        # 计算该基金的权重 = 该基金总量 / 全局总量
        fund_apply_total = np.sum(apply_true)
        fund_redeem_total = np.sum(redeem_true)
        
        apply_weight_i = fund_apply_total / total_apply_global if total_apply_global > 0 else 0
        redeem_weight_i = fund_redeem_total / total_redeem_global if total_redeem_global > 0 else 0
        
        apply_wmapes.append(apply_wmape_i)
        redeem_wmapes.append(redeem_wmape_i)
        apply_weights.append(apply_weight_i)
        redeem_weights.append(redeem_weight_i)
    
    # 按比赛规则计算加权平均WMAPE
    # WMAPE = ∑(n=1 to N) WMAPE_n × weight_n (权重已归一化，无需除以N)
    apply_wmape_final = np.sum([wmape * weight for wmape, weight in zip(apply_wmapes, apply_weights)])
    redeem_wmape_final = np.sum([wmape * weight for wmape, weight in zip(redeem_wmapes, redeem_weights)])
    
    # 最终得分：申购WMAPE × 0.5 + 赎回WMAPE × 0.5
    final_wmape = apply_wmape_final * 0.5 + redeem_wmape_final * 0.5
    
    return apply_wmape_final, redeem_wmape_final, final_wmape

def wmape_metric(y_true, y_pred, fund_weights=None):
    """
    简化版WMAPE（兼容AutoGluon使用）
    注意：这个函数用于AutoGluon内部评估，不是最终比赛评分
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 简化版本：直接计算整体WMAPE
    total_real = np.sum(np.abs(y_true))
    if total_real == 0:
        return 0.0
    
    wmape = np.sum(np.abs(y_pred - y_true)) / total_real
    return wmape

# 别名函数，兼容原有代码
def wmape_by_fund(df_true, df_pred, fund_col='fund_code', apply_col='apply_amt', redeem_col='redeem_amt'):
    """兼容性别名函数"""
    return wmape_competition_standard(df_true, df_pred, fund_col, apply_col, redeem_col)

def wmape_score(y_true, y_pred):
    """
    AutoGluon兼容的WMAPE评分函数
    注意：AutoGluon默认是数值越大越好，所以返回1-WMAPE
    """
    wmape = wmape_metric(y_true, y_pred)
    return 1 - wmape  # 转换为越大越好的指标

# 创建AutoGluon评分器
wmape_scorer = make_scorer(
    name='wmape',
    score_func=wmape_score,
    optimum=1.0,
    greater_is_better=True
)

# 创建比赛专用的评分函数
def competition_score(y_true, y_pred):
    """
    比赛专用评分函数，返回WMAPE值（越小越好）
    """
    return wmape_metric(y_true, y_pred)

competition_scorer = make_scorer(
    name='competition_wmape',
    score_func=lambda y_true, y_pred: 1 - competition_score(y_true, y_pred),
    optimum=1.0,
    greater_is_better=True
) 